// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

/* Test Memory Access Pattern Optimizer pass that analyzes and optimizes memory access patterns for better cache utilization */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<PartialShape>, // Input shapes
        size_t                     // Expected optimizations count
> memoryAccessPatternOptimizerParams;

class MemoryAccessPatternOptimizerTests : public LoweringTests, public testing::WithParamInterface<memoryAccessPatternOptimizerParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<memoryAccessPatternOptimizerParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_model;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov