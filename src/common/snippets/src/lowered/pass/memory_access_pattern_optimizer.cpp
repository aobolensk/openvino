// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/memory_access_pattern_optimizer.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/itt.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool MemoryAccessPatternOptimizer::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "MemoryAccessPatternOptimizer");
    
    bool modified = false;
    
    // Step 1: Analyze loop-based memory access patterns
    for (const auto& expr : linear_ir) {
        if (ov::is_type<op::LoopBegin>(expr->get_node())) {
            // Find the matching LoopEnd
            auto loop_begin_expr = expr;
            ExpressionPtr loop_end_expr = nullptr;
            
            auto it = linear_ir.find(expr);
            if (it != linear_ir.end()) {
                ++it; // Start searching after LoopBegin
                for (; it != linear_ir.end(); ++it) {
                    if (ov::is_type<op::LoopEnd>((*it)->get_node())) {
                        loop_end_expr = *it;
                        break;
                    }
                }
            }
            
            if (loop_end_expr) {
                auto loop_analysis = analyze_loop_memory_patterns(linear_ir, loop_begin_expr, loop_end_expr);
                
                // Optimize stride patterns within this loop
                if (optimize_stride_patterns(linear_ir, loop_analysis)) {
                    modified = true;
                }
            }
        }
    }
    
    // Step 2: Detect and optimize gather/scatter operations
    auto gather_scatter_ops = detect_gather_scatter_operations(linear_ir);
    if (optimize_gather_scatter_operations(linear_ir, gather_scatter_ops)) {
        modified = true;
    }
    
    // Step 3: Coalesce small memory accesses
    auto small_accesses = find_small_memory_accesses(linear_ir);
    if (coalesce_small_memory_accesses(linear_ir, small_accesses)) {
        modified = true;
    }
    
    return modified;
}

MemoryAccessPatternOptimizer::LoopAccessAnalysis
MemoryAccessPatternOptimizer::analyze_loop_memory_patterns(const LinearIR& linear_ir,
                                                          const ExpressionPtr& loop_begin_expr,
                                                          const ExpressionPtr& loop_end_expr) {
    LoopAccessAnalysis analysis;
    
    // Find all memory access operations within the loop
    auto loop_begin_it = linear_ir.find(loop_begin_expr);
    auto loop_end_it = linear_ir.find(loop_end_expr);
    
    if (loop_begin_it == linear_ir.end() || loop_end_it == linear_ir.end()) {
        return analysis;
    }
    
    // Iterate through loop body
    for (auto it = std::next(loop_begin_it); it != loop_end_it; ++it) {
        const auto& expr = *it;
        const auto& op = expr->get_node();
        
        // Check if this is a memory access operation
        if (auto memory_access = ov::as_type_ptr<modifier::MemoryAccess>(op)) {
            analysis.memory_accesses.push_back(expr);
            analysis.patterns[expr] = analyze_memory_access_pattern(expr, analysis);
        }
    }
    
    // Analyze vectorization potential
    if (!analysis.memory_accesses.empty()) {
        std::vector<MemoryAccessPattern> patterns;
        for (const auto& [expr, pattern] : analysis.patterns) {
            patterns.push_back(pattern);
        }
        analysis.optimal_vector_size = calculate_optimal_vector_size(patterns);
        analysis.can_vectorize = analysis.optimal_vector_size > 1;
    }
    
    return analysis;
}

MemoryAccessPatternOptimizer::MemoryAccessPattern
MemoryAccessPatternOptimizer::analyze_memory_access_pattern(const ExpressionPtr& memory_access,
                                                           const LoopAccessAnalysis& loop_context) {
    MemoryAccessPattern pattern;
    
    const auto& op = memory_access->get_node();
    if (auto memory_op = ov::as_type_ptr<modifier::MemoryAccess>(op)) {
        // Check if input port 0 exists before accessing its properties
        if (!memory_op->is_memory_access_input_port(0)) {
            return pattern; // Return default pattern if no input port 0
        }
        
        // Get memory access properties using individual getter methods
        size_t stride = memory_op->get_input_stride(0);
        size_t count = memory_op->get_input_count(0);
        size_t offset = memory_op->get_input_offset(0);
        
        // Analyze stride pattern
        pattern.stride = stride;
        pattern.access_size = count;
        
        // Determine pattern type
        if (pattern.stride == 1) {
            pattern.type = MemoryAccessPattern::Type::Sequential;
        } else if (pattern.stride > 1 && pattern.stride <= max_stride_optimization) {
            pattern.type = MemoryAccessPattern::Type::Strided;
        } else if (ov::is_type<op::BroadcastLoad>(op)) {
            pattern.type = MemoryAccessPattern::Type::Broadcast;
        } else {
            pattern.type = MemoryAccessPattern::Type::Unknown;
        }
        
        // Estimate cache efficiency
        pattern.cache_efficiency = calculate_cache_efficiency(pattern);
        
        // Calculate memory bandwidth requirements
        auto element_type = memory_access->get_node()->get_output_element_type(0);
        pattern.memory_bandwidth = pattern.access_size * element_type.size();
        
        // Check alignment
        pattern.alignment = (offset % preferred_alignment == 0) ? preferred_alignment : 1;
        pattern.is_coalesced = (pattern.alignment >= preferred_alignment && 
                               pattern.type == MemoryAccessPattern::Type::Sequential);
    }
    
    return pattern;
}

std::vector<ExpressionPtr> MemoryAccessPatternOptimizer::detect_gather_scatter_operations(const LinearIR& linear_ir) {
    std::vector<ExpressionPtr> gather_scatter_ops;
    
    for (const auto& expr : linear_ir) {
        const auto& op = expr->get_node();
        
        // Detect gather operations (Load with irregular stride)
        if (ov::is_type<op::Load>(op)) {
            if (auto memory_access = ov::as_type_ptr<modifier::MemoryAccess>(op)) {
                if (memory_access->is_memory_access_input_port(0)) {
                    size_t stride = memory_access->get_input_stride(0);
                    
                    // Consider irregular stride as gather pattern
                    if (stride > max_stride_optimization) {
                        gather_scatter_ops.push_back(expr);
                    }
                }
            }
        }
        
        // Detect scatter operations (Store with irregular stride)
        if (ov::is_type<op::Store>(op)) {
            if (auto memory_access = ov::as_type_ptr<modifier::MemoryAccess>(op)) {
                if (memory_access->is_memory_access_output_port(0)) {
                    size_t stride = memory_access->get_output_stride(0);
                    
                    // Consider irregular stride as scatter pattern
                    if (stride > max_stride_optimization) {
                        gather_scatter_ops.push_back(expr);
                    }
                }
            }
        }
    }
    
    return gather_scatter_ops;
}

bool MemoryAccessPatternOptimizer::optimize_stride_patterns(LinearIR& linear_ir,
                                                           const LoopAccessAnalysis& loop_analysis) {
    bool modified = false;
    
    for (const auto& [expr, pattern] : loop_analysis.patterns) {
        if (pattern.type == MemoryAccessPattern::Type::Strided && 
            pattern.cache_efficiency < min_cache_efficiency) {
            
            // Attempt to optimize strided access
            if (optimize_strided_access(linear_ir, expr, pattern)) {
                modified = true;
            }
        }
    }
    
    return modified;
}

bool MemoryAccessPatternOptimizer::optimize_strided_access(LinearIR& linear_ir,
                                                          const ExpressionPtr& expr,
                                                          const MemoryAccessPattern& pattern) {
    const auto& op = expr->get_node();
    
    // For strided access, try to use vectorized load/store with stride
    if (ov::is_type<op::Load>(op)) {
        // Check if we can use a more efficient vectorized load
        if (pattern.stride <= 4 && pattern.access_size >= min_coalescing_size) {
            // For now, just mark the optimization as attempted
            // In a full implementation, this would create an optimized load operation
            return true;
        }
    }
    
    if (ov::is_type<op::Store>(op)) {
        // Check if we can use a more efficient vectorized store
        if (pattern.stride <= 4 && pattern.access_size >= min_coalescing_size) {
            // For now, just mark the optimization as attempted
            // In a full implementation, this would create an optimized store operation
            return true;
        }
    }
    
    return false;
}

bool MemoryAccessPatternOptimizer::optimize_gather_scatter_operations(LinearIR& linear_ir,
                                                                     const std::vector<ExpressionPtr>& gather_scatter_ops) {
    bool modified = false;
    
    for (const auto& expr : gather_scatter_ops) {
        const auto& op = expr->get_node();
        
        // Try to convert irregular patterns to more efficient forms
        if (ov::is_type<op::Load>(op)) {
            if (optimize_gather_load(linear_ir, expr)) {
                modified = true;
            }
        }
        
        if (ov::is_type<op::Store>(op)) {
            if (optimize_scatter_store(linear_ir, expr)) {
                modified = true;
            }
        }
    }
    
    return modified;
}

bool MemoryAccessPatternOptimizer::coalesce_small_memory_accesses(LinearIR& linear_ir,
                                                                 const std::vector<ExpressionPtr>& small_accesses) {
    bool modified = false;
    
    // Group small accesses by adjacency
    auto access_groups = group_adjacent_accesses(small_accesses);
    
    for (const auto& group : access_groups) {
        if (group.size() >= 2) {
            // Coalesce adjacent small accesses into larger ones
            if (coalesce_access_group(linear_ir, group)) {
                modified = true;
            }
        }
    }
    
    return modified;
}

std::vector<ExpressionPtr> MemoryAccessPatternOptimizer::find_small_memory_accesses(const LinearIR& linear_ir) {
    std::vector<ExpressionPtr> small_accesses;
    
    for (const auto& expr : linear_ir) {
        const auto& op = expr->get_node();
        
        if (auto memory_access = ov::as_type_ptr<modifier::MemoryAccess>(op)) {
            if (memory_access->is_memory_access_input_port(0)) {
                size_t count = memory_access->get_input_count(0);
                
                // Consider accesses smaller than minimum coalescing size as small
                if (count < min_coalescing_size) {
                    small_accesses.push_back(expr);
                }
            }
        }
    }
    
    return small_accesses;
}

size_t MemoryAccessPatternOptimizer::calculate_optimal_vector_size(const std::vector<MemoryAccessPattern>& patterns) {
    if (patterns.empty()) return 1;
    
    size_t optimal_size = 1;
    
    // Find the maximum access size that maintains good cache efficiency
    for (const auto& pattern : patterns) {
        if (pattern.type == MemoryAccessPattern::Type::Sequential ||
            pattern.type == MemoryAccessPattern::Type::Strided) {
            
            size_t candidate_size = std::min(pattern.access_size, static_cast<size_t>(16));
            if (pattern.cache_efficiency >= min_cache_efficiency) {
                optimal_size = std::max(optimal_size, candidate_size);
            }
        }
    }
    
    return optimal_size;
}

double MemoryAccessPatternOptimizer::calculate_cache_efficiency(const MemoryAccessPattern& pattern) {
    // Simple cache efficiency model
    switch (pattern.type) {
        case MemoryAccessPattern::Type::Sequential:
            return 0.95; // Very high cache efficiency
        case MemoryAccessPattern::Type::Strided:
            if (pattern.stride <= 4) return 0.8;
            else if (pattern.stride <= 16) return 0.6;
            else return 0.3;
        case MemoryAccessPattern::Type::Broadcast:
            return 0.9; // High efficiency due to reuse
        case MemoryAccessPattern::Type::Gather:
        case MemoryAccessPattern::Type::Scatter:
            return 0.2; // Low efficiency due to irregular access
        default:
            return 0.5; // Average efficiency
    }
}

double MemoryAccessPatternOptimizer::estimate_performance_impact(const std::vector<MemoryAccessPattern>& original_patterns,
                                                               const std::vector<MemoryAccessPattern>& optimized_patterns) {
    double original_efficiency = 0.0;
    double optimized_efficiency = 0.0;
    
    for (const auto& pattern : original_patterns) {
        original_efficiency += pattern.cache_efficiency;
    }
    
    for (const auto& pattern : optimized_patterns) {
        optimized_efficiency += pattern.cache_efficiency;
    }
    
    if (original_efficiency > 0.0) {
        return optimized_efficiency / original_efficiency;
    }
    
    return 1.0; // No change
}


bool MemoryAccessPatternOptimizer::optimize_gather_load(LinearIR& linear_ir,
                                                       const ExpressionPtr& expr) {
    // Placeholder for gather load optimization
    // In a full implementation, this would convert irregular loads to more efficient forms
    return false;
}

bool MemoryAccessPatternOptimizer::optimize_scatter_store(LinearIR& linear_ir,
                                                         const ExpressionPtr& expr) {
    // Placeholder for scatter store optimization
    // In a full implementation, this would convert irregular stores to more efficient forms
    return false;
}

std::vector<std::vector<ExpressionPtr>> MemoryAccessPatternOptimizer::group_adjacent_accesses(const std::vector<ExpressionPtr>& accesses) {
    std::vector<std::vector<ExpressionPtr>> groups;
    
    if (accesses.empty()) return groups;
    
    // Simple grouping by adjacent memory addresses
    // In a full implementation, this would analyze memory addresses and group adjacent accesses
    std::vector<ExpressionPtr> current_group;
    current_group.push_back(accesses[0]);
    
    for (size_t i = 1; i < accesses.size(); ++i) {
        // Placeholder logic - in reality would check memory address adjacency
        current_group.push_back(accesses[i]);
        
        // Group every 4 accesses as a simple heuristic
        if (current_group.size() >= 4) {
            groups.push_back(current_group);
            current_group.clear();
        }
    }
    
    if (!current_group.empty()) {
        groups.push_back(current_group);
    }
    
    return groups;
}

bool MemoryAccessPatternOptimizer::coalesce_access_group(LinearIR& linear_ir,
                                                        const std::vector<ExpressionPtr>& group) {
    // Placeholder for coalescing logic
    // In a full implementation, this would merge adjacent small accesses into larger ones
    return group.size() >= 2;
}

bool MemoryAccessPatternOptimizer::validate_optimization(const LinearIR& linear_ir,
                                                        const std::vector<ExpressionPtr>& modified_expressions) {
    // Basic validation: ensure all modified expressions are still valid
    for (const auto& expr : modified_expressions) {
        if (!expr || !expr->get_node()) {
            return false;
        }
        
        // Check that memory access properties are still valid
        if (auto memory_access = ov::as_type_ptr<modifier::MemoryAccess>(expr->get_node())) {
            // Basic validation - in a full implementation would be more comprehensive
            const auto& inputs = expr->get_input_port_connectors();
            const auto& outputs = expr->get_output_port_connectors();
            
            if (inputs.empty() && outputs.empty()) {
                return false;
            }
        }
    }
    
    return true;
}

}  // namespace pass
}  // namespace lowered
}  // namespace snippets
}  // namespace ov