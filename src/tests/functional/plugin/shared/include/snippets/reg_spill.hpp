// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

using RegSpillParams = std::tuple<
    InputShape,  // Input Shape, replicated for all inputs
    size_t,      // Expected num nodes
    size_t,      // Expected num subgraphs
    std::string  // Target Device
    >;

class RegSpill : public testing::WithParamInterface<ov::test::snippets::RegSpillParams>,
                 virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::snippets::RegSpillParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
