// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/reg_spill.hpp"

#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

std::vector<InputShape> input_shapes{{{}, {{1, 1, 1, 64}}},
                                     {{-1, -1, -1, -1}, {{1, 1, 1, 64}, {1, 1, 1, 17}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_RegSpill,
                         RegSpill,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RegSpill::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
