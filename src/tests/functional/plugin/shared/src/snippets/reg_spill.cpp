// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/reg_spill.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "snippets/lowered/pass/insert_reg_spills.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

ov::ParameterVector create_parameters(const ov::PartialShape& shape) {
    using namespace ov::opset10;

    ov::ParameterVector parameters;
    for (size_t i = 0; i < 4; ++i) {
        auto parameter = std::make_shared<Parameter>(ov::element::f32, shape);
        parameter->set_friendly_name("input" + std::to_string(i));
        parameters.push_back(parameter);
    }
    return parameters;
}

std::shared_ptr<ov::Model> create_reg_spill_model(const ov::PartialShape& shape) {
    using namespace ov::opset10;

    auto inputs = create_parameters(shape);

    auto body_input0 = std::make_shared<Parameter>(ov::element::f32, shape);
    auto body_input1 = std::make_shared<Parameter>(ov::element::f32, shape);
    auto body_input2 = std::make_shared<Parameter>(ov::element::f32, shape);
    auto body_input3 = std::make_shared<Parameter>(ov::element::f32, shape);

    auto live_across_spill = std::make_shared<Add>(body_input0, body_input1);
    live_across_spill->set_friendly_name("live_across_spill");
    auto forced_spill = std::make_shared<Multiply>(body_input2, body_input3);
    forced_spill->set_friendly_name("forced_reg_spill");
    forced_spill->get_rt_info()[ov::snippets::lowered::pass::force_reg_spill_rt_info] = true;
    auto add = std::make_shared<Add>(live_across_spill, forced_spill);
    auto result = std::make_shared<ov::snippets::op::Result>(add);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{result},
                                            ov::ParameterVector{body_input0, body_input1, body_input2, body_input3});
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
        ov::OutputVector{inputs[0], inputs[1], inputs[2], inputs[3]},
        body);

    return std::make_shared<ov::Model>(ov::OutputVector{subgraph}, inputs);
}

std::shared_ptr<ov::Model> create_reference_model(const ov::PartialShape& shape) {
    using namespace ov::opset10;

    auto inputs = create_parameters(shape);
    auto add = std::make_shared<Add>(inputs[0], inputs[1]);
    auto multiply = std::make_shared<Multiply>(inputs[2], inputs[3]);
    auto result = std::make_shared<Add>(add, multiply);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, inputs);
}

}  // namespace

std::string RegSpill::getTestCaseName(const testing::TestParamInfo<ov::test::snippets::RegSpillParams>& obj) {
    const auto& [input_shape, num_nodes, num_subgraphs, target_device] = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({input_shape.first}) << "_";
    result << "TS[0]=";
    for (const auto& item : input_shape.second) {
        result << ov::test::utils::vec2str(item) << "_";
    }
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void RegSpill::SetUp() {
    const auto& [input_shape, _ref_num_nodes, _ref_num_subgraphs, _target_device] = GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _target_device;
    init_input_shapes(std::vector<InputShape>(4, input_shape));
    function = create_reg_spill_model(inputDynamicShapes.front());
    functionRefs = create_reference_model(inputDynamicShapes.front());
    setIgnoreCallbackMode();
}

TEST_P(RegSpill, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
