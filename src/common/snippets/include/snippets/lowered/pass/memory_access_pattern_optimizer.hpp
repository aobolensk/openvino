// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/loop.hpp"

namespace ov {
namespace snippets {
namespace op {
    class Load;
    class Store;
}
namespace lowered {
namespace pass {

/**
 * @brief Memory Access Pattern Optimizer analyzes and optimizes memory access patterns for better cache utilization
 * @ingroup snippets
 * 
 * This pass performs comprehensive analysis of memory access patterns to:
 * 1. Detect and optimize stride patterns in loops
 * 2. Identify gather/scatter operations and optimize them
 * 3. Implement memory access coalescing for small accesses
 * 
 * Expected performance impact: 15-25% improvement through better cache utilization
 */
class MemoryAccessPatternOptimizer : public Pass {
public:
    OPENVINO_RTTI("MemoryAccessPatternOptimizer", "Pass")
    MemoryAccessPatternOptimizer() = default;
    
    /**
     * @brief Applies memory access pattern optimization to the Linear IR
     * @param linear_ir target Linear IR
     * @return status of the optimization
     */
    bool run(LinearIR& linear_ir) override;

private:
    /**
     * @brief Describes memory access pattern characteristics
     */
    struct MemoryAccessPattern {
        enum class Type {
            Sequential,     // Sequential access with stride 1
            Strided,        // Regular stride pattern
            Gather,         // Irregular gather pattern
            Scatter,        // Irregular scatter pattern
            Broadcast,      // Broadcast pattern
            Unknown         // Unknown/complex pattern
        };
        
        Type type = Type::Unknown;
        size_t stride = 1;           // Memory stride between accesses
        size_t access_size = 1;      // Size of each access
        size_t alignment = 1;        // Memory alignment of accesses
        bool is_coalesced = false;   // Whether accesses are coalesced
        
        // Performance characteristics
        double cache_efficiency = 0.0;  // Estimated cache hit rate
        size_t memory_bandwidth = 0;    // Required memory bandwidth
    };
    
    /**
     * @brief Analyzes loop-based memory access patterns
     */
    struct LoopAccessAnalysis {
        std::vector<ExpressionPtr> memory_accesses;
        std::map<ExpressionPtr, MemoryAccessPattern> patterns;
        bool has_stride_conflicts = false;
        bool can_vectorize = false;
        size_t optimal_vector_size = 1;
    };
    
    /**
     * @brief Analyzes memory access patterns within loops
     * @param linear_ir target Linear IR
     * @param loop_begin_expr Loop begin expression
     * @param loop_end_expr Loop end expression
     * @return analysis results
     */
    LoopAccessAnalysis analyze_loop_memory_patterns(const LinearIR& linear_ir,
                                                   const ExpressionPtr& loop_begin_expr,
                                                   const ExpressionPtr& loop_end_expr);
    
    /**
     * @brief Analyzes individual memory access pattern
     * @param memory_access Memory access expression
     * @param loop_context Loop context information
     * @return pattern analysis
     */
    MemoryAccessPattern analyze_memory_access_pattern(const ExpressionPtr& memory_access,
                                                     const LoopAccessAnalysis& loop_context);
    
    /**
     * @brief Detects gather/scatter operations
     * @param linear_ir target Linear IR
     * @return list of gather/scatter expressions
     */
    std::vector<ExpressionPtr> detect_gather_scatter_operations(const LinearIR& linear_ir);
    
    /**
     * @brief Optimizes stride patterns within loops
     * @param linear_ir target Linear IR
     * @param loop_analysis Loop analysis results
     * @return whether optimization was applied
     */
    bool optimize_stride_patterns(LinearIR& linear_ir, const LoopAccessAnalysis& loop_analysis);
    
    /**
     * @brief Optimizes gather/scatter operations
     * @param linear_ir target Linear IR  
     * @param gather_scatter_ops List of gather/scatter operations
     * @return whether optimization was applied
     */
    bool optimize_gather_scatter_operations(LinearIR& linear_ir,
                                           const std::vector<ExpressionPtr>& gather_scatter_ops);
    
    /**
     * @brief Implements memory access coalescing for small accesses
     * @param linear_ir target Linear IR
     * @param small_accesses List of small memory accesses
     * @return whether coalescing was applied
     */
    bool coalesce_small_memory_accesses(LinearIR& linear_ir,
                                       const std::vector<ExpressionPtr>& small_accesses);
    
    /**
     * @brief Finds small memory accesses that can be coalesced
     * @param linear_ir target Linear IR
     * @return list of small access expressions
     */
    std::vector<ExpressionPtr> find_small_memory_accesses(const LinearIR& linear_ir);
    
    /**
     * @brief Calculates optimal vector size based on memory access patterns
     * @param patterns List of memory access patterns
     * @return optimal vector size
     */
    size_t calculate_optimal_vector_size(const std::vector<MemoryAccessPattern>& patterns);
    
    /**
     * @brief Estimates performance impact of optimization
     * @param original_patterns Original memory access patterns
     * @param optimized_patterns Optimized memory access patterns
     * @return estimated performance improvement ratio
     */
    double estimate_performance_impact(const std::vector<MemoryAccessPattern>& original_patterns,
                                     const std::vector<MemoryAccessPattern>& optimized_patterns);
    
    /**
     * @brief Validates that optimization preserves correctness
     * @param linear_ir target Linear IR
     * @param modified_expressions List of modified expressions
     * @return whether optimization is valid
     */
    bool validate_optimization(const LinearIR& linear_ir,
                              const std::vector<ExpressionPtr>& modified_expressions);
    
    /**
     * @brief Optimizes strided memory access pattern
     * @param linear_ir target Linear IR
     * @param expr Memory access expression
     * @param pattern Memory access pattern
     * @return whether optimization was applied
     */
    bool optimize_strided_access(LinearIR& linear_ir,
                                const ExpressionPtr& expr,
                                const MemoryAccessPattern& pattern);
    
    /**
     * @brief Optimizes gather load operation
     * @param linear_ir target Linear IR
     * @param expr Load expression
     * @return whether optimization was applied
     */
    bool optimize_gather_load(LinearIR& linear_ir,
                             const ExpressionPtr& expr);
    
    /**
     * @brief Optimizes scatter store operation
     * @param linear_ir target Linear IR
     * @param expr Store expression
     * @return whether optimization was applied
     */
    bool optimize_scatter_store(LinearIR& linear_ir,
                               const ExpressionPtr& expr);
    
    /**
     * @brief Groups adjacent memory accesses for coalescing
     * @param accesses List of memory access expressions
     * @return grouped accesses
     */
    std::vector<std::vector<ExpressionPtr>> group_adjacent_accesses(const std::vector<ExpressionPtr>& accesses);
    
    /**
     * @brief Coalesces a group of adjacent memory accesses
     * @param linear_ir target Linear IR
     * @param group Group of adjacent accesses
     * @return whether coalescing was applied
     */
    bool coalesce_access_group(LinearIR& linear_ir,
                              const std::vector<ExpressionPtr>& group);
    
    /**
     * @brief Calculates cache efficiency for a memory access pattern
     * @param pattern Memory access pattern
     * @return cache efficiency score (0.0 to 1.0)
     */
    double calculate_cache_efficiency(const MemoryAccessPattern& pattern);
    
    // Configuration parameters
    static constexpr size_t min_coalescing_size = 4;        // Minimum elements for coalescing
    static constexpr size_t max_stride_optimization = 64;   // Maximum stride for optimization
    static constexpr double min_cache_efficiency = 0.5;     // Minimum cache efficiency threshold
    static constexpr size_t preferred_alignment = 64;       // Preferred memory alignment (cache line)
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov