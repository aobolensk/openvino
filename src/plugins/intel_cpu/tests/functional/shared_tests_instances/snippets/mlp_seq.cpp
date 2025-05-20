// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> inputShape_2D(bool with_dynamic = true) {
    auto shapes = SNIPPETS_TESTS_STATIC_SHAPES(
        {{1, 64}},
        {{2, 64}},
        {{4, 64}},
        {{8, 64}});
    if (with_dynamic) {
        shapes.push_back({{PartialShape{-1, 64}, {{1, 64}, {8, 64}, {8, 64}, {6, 64}}}});
        shapes.push_back({{PartialShape{-1, 64}, {{2, 64}, {2, 64}, {4, 64}, {3, 64}}}});
    }
    return shapes;
}

std::vector<size_t> numHiddenLayers() {
    return {1, 3, 5};
}
std::vector<size_t> hiddenMatmulSizes() {
    return {64, 128, 256};
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D_f32,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::ValuesIn({ov::element::f32, ov::element::bf16}),
                                            ::testing::Values(MLP::default_thread_count),
                                            ::testing::Values(1),  // Subgraph
                                            ::testing::Values(1),  // MLP
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayers()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_Quantized_2D_f32,
                         MLPQuantized,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::ValuesIn({ov::element::f32, ov::element::bf16}),
                                            ::testing::Values(MLPQuantized::default_thread_count),
                                            ::testing::Values(1),  // Subgraph
                                            ::testing::Values(1),  // MLPQuantized
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayers()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLPQuantized::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_Quantized_2D_i8,
                         MLPQuantized,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_i8(1)),
                                            ::testing::ValuesIn({ov::element::f32, ov::element::bf16}),
                                            ::testing::Values(MLPQuantized::default_thread_count),
                                            ::testing::Values(1),  // Subgraph
                                            ::testing::Values(1),  // MLPQuantized
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayers()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLPQuantized::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
