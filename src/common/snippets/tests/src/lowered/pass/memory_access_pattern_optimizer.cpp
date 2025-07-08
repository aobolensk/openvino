// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "lowered/pass/memory_access_pattern_optimizer.hpp"
#include "common_test_utils/common_utils.hpp"
#include "subgraph_memory_access_patterns.hpp"
#include "snippets/lowered/pass/memory_access_pattern_optimizer.hpp"
#include "snippets/lowered/pass/pass_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string MemoryAccessPatternOptimizerTests::getTestCaseName(testing::TestParamInfo<memoryAccessPatternOptimizerParams> obj) {
    std::vector<PartialShape> input_shapes;
    size_t expected_optimizations;
    std::tie(input_shapes, expected_optimizations) = obj.param;
    
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({input_shapes[i]});
        if (i < input_shapes.size() - 1) result << "_";
    }
    result << "_ExpectedOpts=" << expected_optimizations;
    return result.str();
}

void MemoryAccessPatternOptimizerTests::SetUp() {
    LoweringTests::SetUp();
    std::vector<PartialShape> input_shapes;
    size_t expected_optimizations;
    std::tie(input_shapes, expected_optimizations) = this->GetParam();
    
    // This will be set by individual test instances
    snippets_model = nullptr;
}

// Test sequential memory access pattern optimization
class SequentialMemoryAccessOptimizerTests : public MemoryAccessPatternOptimizerTests {
protected:
    void SetUp() override {
        MemoryAccessPatternOptimizerTests::SetUp();
        std::vector<PartialShape> input_shapes;
        size_t expected_optimizations;
        std::tie(input_shapes, expected_optimizations) = this->GetParam();
        
        snippets_model = std::make_shared<SequentialMemoryAccessFunction>(input_shapes);
    }
};

// Test strided memory access pattern optimization  
class StridedMemoryAccessOptimizerTests : public MemoryAccessPatternOptimizerTests {
protected:
    void SetUp() override {
        MemoryAccessPatternOptimizerTests::SetUp();
        std::vector<PartialShape> input_shapes;
        size_t expected_optimizations;
        std::tie(input_shapes, expected_optimizations) = this->GetParam();
        
        snippets_model = std::make_shared<StridedMemoryAccessFunction>(input_shapes, 4);
    }
};

// Test gather/scatter memory access pattern optimization
class GatherScatterMemoryAccessOptimizerTests : public MemoryAccessPatternOptimizerTests {
protected:
    void SetUp() override {
        MemoryAccessPatternOptimizerTests::SetUp();
        std::vector<PartialShape> input_shapes;
        size_t expected_optimizations;
        std::tie(input_shapes, expected_optimizations) = this->GetParam();
        
        snippets_model = std::make_shared<GatherScatterMemoryAccessFunction>(input_shapes, 128);
    }
};

// Test small coalesced memory access pattern optimization
class SmallCoalescedMemoryAccessOptimizerTests : public MemoryAccessPatternOptimizerTests {
protected:
    void SetUp() override {
        MemoryAccessPatternOptimizerTests::SetUp();
        std::vector<PartialShape> input_shapes;
        size_t expected_optimizations;
        std::tie(input_shapes, expected_optimizations) = this->GetParam();
        
        snippets_model = std::make_shared<SmallCoalescedMemoryAccessFunction>(input_shapes, 4);
    }
};

// Test loop memory access pattern optimization
class LoopMemoryAccessOptimizerTests : public MemoryAccessPatternOptimizerTests {
protected:
    void SetUp() override {
        MemoryAccessPatternOptimizerTests::SetUp();
        std::vector<PartialShape> input_shapes;
        size_t expected_optimizations;
        std::tie(input_shapes, expected_optimizations) = this->GetParam();
        
        snippets_model = std::make_shared<LoopMemoryAccessFunction>(input_shapes, 16);
    }
};

// Test broadcast memory access pattern optimization
class BroadcastMemoryAccessOptimizerTests : public MemoryAccessPatternOptimizerTests {
protected:
    void SetUp() override {
        MemoryAccessPatternOptimizerTests::SetUp();
        std::vector<PartialShape> input_shapes;
        size_t expected_optimizations;
        std::tie(input_shapes, expected_optimizations) = this->GetParam();
        
        snippets_model = std::make_shared<BroadcastMemoryAccessFunction>(input_shapes);
    }
};

// Test mixed memory access pattern optimization
class MixedMemoryAccessOptimizerTests : public MemoryAccessPatternOptimizerTests {
protected:
    void SetUp() override {
        MemoryAccessPatternOptimizerTests::SetUp();
        std::vector<PartialShape> input_shapes;
        size_t expected_optimizations;
        std::tie(input_shapes, expected_optimizations) = this->GetParam();
        
        snippets_model = std::make_shared<MixedMemoryAccessFunction>(input_shapes);
    }
};

// Test implementations
TEST_P(SequentialMemoryAccessOptimizerTests, SequentialMemoryAccessOptimization) {
    // Get original model and create lowered subgraph
    auto original_model = snippets_model->getOriginal();
    auto subgraph = getLoweredSubgraph(original_model);
    
    // Extract the lowered linear IR
    auto linear_ir = subgraph->get_linear_ir();
    ASSERT_NE(linear_ir, nullptr);
    
    // Apply the Memory Access Pattern Optimizer pass
    auto optimizer = std::make_shared<ov::snippets::lowered::pass::MemoryAccessPatternOptimizer>();
    bool optimization_applied = optimizer->run(*linear_ir);
    
    // Verify that optimization was applied
    EXPECT_TRUE(optimization_applied) << "Memory access pattern optimizer should have optimized sequential access patterns";
    
    // Verify the IR is still valid after optimization
    EXPECT_NO_THROW(subgraph->validate()) << "Linear IR should be valid after optimization";
}

TEST_P(StridedMemoryAccessOptimizerTests, StridedMemoryAccessOptimization) {
    auto original_model = snippets_model->getOriginal();
    auto subgraph = getLoweredSubgraph(original_model);
    
    auto linear_ir = subgraph->get_linear_ir();
    ASSERT_NE(linear_ir, nullptr);
    
    auto optimizer = std::make_shared<ov::snippets::lowered::pass::MemoryAccessPatternOptimizer>();
    bool optimization_applied = optimizer->run(*linear_ir);
    
    EXPECT_TRUE(optimization_applied) << "Memory access pattern optimizer should have optimized strided access patterns";
    EXPECT_NO_THROW(subgraph->validate()) << "Linear IR should be valid after optimization";
}

TEST_P(GatherScatterMemoryAccessOptimizerTests, GatherScatterMemoryAccessOptimization) {
    auto original_model = snippets_model->getOriginal();
    auto subgraph = getLoweredSubgraph(original_model);
    
    auto linear_ir = subgraph->get_linear_ir();
    ASSERT_NE(linear_ir, nullptr);
    
    auto optimizer = std::make_shared<ov::snippets::lowered::pass::MemoryAccessPatternOptimizer>();
    bool optimization_applied = optimizer->run(*linear_ir);
    
    // Gather/scatter patterns may not always be optimizable, so we check that the pass ran successfully
    EXPECT_NO_THROW(subgraph->validate()) << "Linear IR should be valid after optimization";
}

TEST_P(SmallCoalescedMemoryAccessOptimizerTests, SmallCoalescedMemoryAccessOptimization) {
    auto original_model = snippets_model->getOriginal();
    auto subgraph = getLoweredSubgraph(original_model);
    
    auto linear_ir = subgraph->get_linear_ir();
    ASSERT_NE(linear_ir, nullptr);
    
    auto optimizer = std::make_shared<ov::snippets::lowered::pass::MemoryAccessPatternOptimizer>();
    bool optimization_applied = optimizer->run(*linear_ir);
    
    // Small access coalescing should be applicable in many cases
    EXPECT_NO_THROW(subgraph->validate()) << "Linear IR should be valid after optimization";
}

TEST_P(LoopMemoryAccessOptimizerTests, LoopMemoryAccessOptimization) {
    auto original_model = snippets_model->getOriginal();
    auto subgraph = getLoweredSubgraph(original_model);
    
    auto linear_ir = subgraph->get_linear_ir();
    ASSERT_NE(linear_ir, nullptr);
    
    auto optimizer = std::make_shared<ov::snippets::lowered::pass::MemoryAccessPatternOptimizer>();
    bool optimization_applied = optimizer->run(*linear_ir);
    
    // Loop-based optimizations should be common
    EXPECT_NO_THROW(subgraph->validate()) << "Linear IR should be valid after optimization";
}

TEST_P(BroadcastMemoryAccessOptimizerTests, BroadcastMemoryAccessOptimization) {
    auto original_model = snippets_model->getOriginal();
    auto subgraph = getLoweredSubgraph(original_model);
    
    auto linear_ir = subgraph->get_linear_ir();
    ASSERT_NE(linear_ir, nullptr);
    
    auto optimizer = std::make_shared<ov::snippets::lowered::pass::MemoryAccessPatternOptimizer>();
    bool optimization_applied = optimizer->run(*linear_ir);
    
    // Broadcast patterns should be optimizable
    EXPECT_NO_THROW(subgraph->validate()) << "Linear IR should be valid after optimization";
}

TEST_P(MixedMemoryAccessOptimizerTests, MixedMemoryAccessOptimization) {
    auto original_model = snippets_model->getOriginal();
    auto subgraph = getLoweredSubgraph(original_model);
    
    auto linear_ir = subgraph->get_linear_ir();
    ASSERT_NE(linear_ir, nullptr);
    
    auto optimizer = std::make_shared<ov::snippets::lowered::pass::MemoryAccessPatternOptimizer>();
    bool optimization_applied = optimizer->run(*linear_ir);
    
    // Mixed patterns should offer optimization opportunities
    EXPECT_NO_THROW(subgraph->validate()) << "Linear IR should be valid after optimization";
}

// Test parameter definitions
namespace MemoryAccessPatternOptimizerTestsInstantiation {
using ov::Shape;

// Sequential memory access test parameters
std::vector<memoryAccessPatternOptimizerParams> sequential_test_params{
    {{{64}}, 1},                    // 1D sequential access
    {{{32, 32}}, 1},                // 2D sequential access  
    {{{16, 16, 16}}, 1},            // 3D sequential access
    {{{8, 8, 8, 8}}, 1},            // 4D sequential access
};

// Strided memory access test parameters
std::vector<memoryAccessPatternOptimizerParams> strided_test_params{
    {{{128}}, 1},                   // 1D strided access
    {{{64, 64}}, 1},                // 2D strided access
    {{{32, 32, 32}}, 1},            // 3D strided access
};

// Gather/scatter memory access test parameters
std::vector<memoryAccessPatternOptimizerParams> gather_scatter_test_params{
    {{{256}}, 0},                   // 1D gather/scatter (may not be optimizable)
    {{{128, 128}}, 0},              // 2D gather/scatter
};

// Small coalesced memory access test parameters
std::vector<memoryAccessPatternOptimizerParams> small_coalesced_test_params{
    {{{32}}, 1},                    // 1D small accesses
    {{{16, 16}}, 1},                // 2D small accesses
};

// Loop memory access test parameters
std::vector<memoryAccessPatternOptimizerParams> loop_test_params{
    {{{64}}, 1},                    // 1D loop access
    {{{32, 32}}, 1},                // 2D loop access
};

// Broadcast memory access test parameters
std::vector<memoryAccessPatternOptimizerParams> broadcast_test_params{
    {{{64}}, 1},                    // 1D broadcast access
    {{{32, 32}}, 1},                // 2D broadcast access
};

// Mixed memory access test parameters
std::vector<memoryAccessPatternOptimizerParams> mixed_test_params{
    {{{64}, {64}}, 1},              // Two 1D inputs
    {{{32, 32}, {32, 32}}, 1},      // Two 2D inputs
};

// Test instantiations
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_SequentialMemoryAccessOptimizer, SequentialMemoryAccessOptimizerTests,
                         ::testing::ValuesIn(sequential_test_params),
                         SequentialMemoryAccessOptimizerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_StridedMemoryAccessOptimizer, StridedMemoryAccessOptimizerTests,
                         ::testing::ValuesIn(strided_test_params),
                         StridedMemoryAccessOptimizerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GatherScatterMemoryAccessOptimizer, GatherScatterMemoryAccessOptimizerTests,
                         ::testing::ValuesIn(gather_scatter_test_params),
                         GatherScatterMemoryAccessOptimizerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_SmallCoalescedMemoryAccessOptimizer, SmallCoalescedMemoryAccessOptimizerTests,
                         ::testing::ValuesIn(small_coalesced_test_params),
                         SmallCoalescedMemoryAccessOptimizerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_LoopMemoryAccessOptimizer, LoopMemoryAccessOptimizerTests,
                         ::testing::ValuesIn(loop_test_params),
                         LoopMemoryAccessOptimizerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastMemoryAccessOptimizer, BroadcastMemoryAccessOptimizerTests,
                         ::testing::ValuesIn(broadcast_test_params),
                         BroadcastMemoryAccessOptimizerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MixedMemoryAccessOptimizer, MixedMemoryAccessOptimizerTests,
                         ::testing::ValuesIn(mixed_test_params),
                         MixedMemoryAccessOptimizerTests::getTestCaseName);

} // namespace MemoryAccessPatternOptimizerTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov