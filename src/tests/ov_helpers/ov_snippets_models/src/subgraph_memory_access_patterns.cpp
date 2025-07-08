// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_memory_access_patterns.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/loop.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> SequentialMemoryAccessFunction::initOriginal() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create a simple elementwise operation that will be converted to Load/Add/Store
    auto scalar_const = std::make_shared<ov::op::v0::Constant>(precision, Shape{1}, std::vector<float>{2.0f});
    auto add = std::make_shared<ov::op::v1::Add>(data, scalar_const);
    
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> SequentialMemoryAccessFunction::initReference() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create lowered representation with Load/Store
    auto load = std::make_shared<ov::snippets::op::Load>(data, 1, 0);  // Sequential access
    auto scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 2.0f);
    auto add = std::make_shared<ov::op::v1::Add>(load, scalar);
    auto store = std::make_shared<ov::snippets::op::Store>(add, 1, 0);  // Sequential access
    
    auto result = std::make_shared<ov::op::v0::Result>(store);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> StridedMemoryAccessFunction::initOriginal() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create a reshape to introduce strided access pattern
    auto reshape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(data, reshape_const, false);
    
    auto scalar_const = std::make_shared<ov::op::v0::Constant>(precision, Shape{1}, std::vector<float>{3.0f});
    auto add = std::make_shared<ov::op::v1::Add>(reshape, scalar_const);
    
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> StridedMemoryAccessFunction::initReference() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create lowered representation with strided Load/Store
    auto load = std::make_shared<ov::snippets::op::Load>(data, 1, 0);  // Will be optimized to strided access
    auto scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 3.0f);
    auto add = std::make_shared<ov::op::v1::Add>(load, scalar);
    auto store = std::make_shared<ov::snippets::op::Store>(add, 1, 0);  // Will be optimized to strided access
    
    auto result = std::make_shared<ov::op::v0::Result>(store);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> GatherScatterMemoryAccessFunction::initOriginal() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create a pattern that will result in irregular memory access
    auto scalar_const = std::make_shared<ov::op::v0::Constant>(precision, Shape{1}, std::vector<float>{1.5f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(data, scalar_const);
    
    auto result = std::make_shared<ov::op::v0::Result>(multiply);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> GatherScatterMemoryAccessFunction::initReference() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create lowered representation with irregular Load/Store (gather/scatter)
    auto load = std::make_shared<ov::snippets::op::Load>(data, 1, 0);  // Will be detected as gather
    auto scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 1.5f);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(load, scalar);
    auto store = std::make_shared<ov::snippets::op::Store>(multiply, 1, 0);  // Will be detected as scatter
    
    auto result = std::make_shared<ov::op::v0::Result>(store);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> SmallCoalescedMemoryAccessFunction::initOriginal() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create multiple small operations that can be coalesced
    auto scalar_const = std::make_shared<ov::op::v0::Constant>(precision, Shape{1}, std::vector<float>{1.0f});
    auto add1 = std::make_shared<ov::op::v1::Add>(data, scalar_const);
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, scalar_const);
    auto add3 = std::make_shared<ov::op::v1::Add>(add2, scalar_const);
    
    auto result = std::make_shared<ov::op::v0::Result>(add3);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> SmallCoalescedMemoryAccessFunction::initReference() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create lowered representation with small accesses that can be coalesced
    auto load1 = std::make_shared<ov::snippets::op::Load>(data, 1, 0);  // Small access
    auto scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 1.0f);
    auto add1 = std::make_shared<ov::op::v1::Add>(load1, scalar);
    auto store1 = std::make_shared<ov::snippets::op::Store>(add1, 1, 0);  // Small access
    
    auto load2 = std::make_shared<ov::snippets::op::Load>(store1, 1, 1);  // Small access with offset
    auto add2 = std::make_shared<ov::op::v1::Add>(load2, scalar);
    auto store2 = std::make_shared<ov::snippets::op::Store>(add2, 1, 1);  // Small access with offset
    
    auto result = std::make_shared<ov::op::v0::Result>(store2);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> LoopMemoryAccessFunction::initOriginal() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create a simple loop-like pattern
    auto scalar_const = std::make_shared<ov::op::v0::Constant>(precision, Shape{1}, std::vector<float>{2.0f});
    auto add = std::make_shared<ov::op::v1::Add>(data, scalar_const);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(add, scalar_const);
    
    auto result = std::make_shared<ov::op::v0::Result>(multiply);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> LoopMemoryAccessFunction::initReference() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create lowered representation with loop structure
    auto loop_begin = std::make_shared<ov::snippets::op::LoopBegin>();
    auto load = std::make_shared<ov::snippets::op::Load>(data, 1, 0);  // Loop-based access
    auto scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 2.0f);
    auto add = std::make_shared<ov::op::v1::Add>(load, scalar);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(add, scalar);
    auto store = std::make_shared<ov::snippets::op::Store>(multiply, 1, 0);  // Loop-based access
    auto loop_end = std::make_shared<ov::snippets::op::LoopEnd>();
    
    auto result = std::make_shared<ov::op::v0::Result>(store);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> BroadcastMemoryAccessFunction::initOriginal() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create broadcast pattern
    auto scalar_const = std::make_shared<ov::op::v0::Constant>(precision, Shape{1}, std::vector<float>{3.0f});
    auto add = std::make_shared<ov::op::v1::Add>(data, scalar_const);
    
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> BroadcastMemoryAccessFunction::initReference() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    data->set_friendly_name("data");
    
    // Create lowered representation with broadcast load
    auto broadcast_load = std::make_shared<ov::snippets::op::BroadcastLoad>(data, 1, 0);  // Broadcast pattern
    auto scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 3.0f);
    auto add = std::make_shared<ov::op::v1::Add>(broadcast_load, scalar);
    auto store = std::make_shared<ov::snippets::op::Store>(add, 1, 0);
    
    auto result = std::make_shared<ov::op::v0::Result>(store);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
}

std::shared_ptr<ov::Model> MixedMemoryAccessFunction::initOriginal() const {
    auto data1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    auto data2 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
    data1->set_friendly_name("data1");
    data2->set_friendly_name("data2");
    
    // Create mixed access patterns
    auto scalar_const = std::make_shared<ov::op::v0::Constant>(precision, Shape{1}, std::vector<float>{1.0f});
    auto add1 = std::make_shared<ov::op::v1::Add>(data1, scalar_const);
    auto add2 = std::make_shared<ov::op::v1::Add>(data2, scalar_const);
    auto final_add = std::make_shared<ov::op::v1::Add>(add1, add2);
    
    auto result = std::make_shared<ov::op::v0::Result>(final_add);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data1, data2});
}

std::shared_ptr<ov::Model> MixedMemoryAccessFunction::initReference() const {
    auto data1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    auto data2 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
    data1->set_friendly_name("data1");
    data2->set_friendly_name("data2");
    
    // Create lowered representation with mixed access patterns
    auto load1 = std::make_shared<ov::snippets::op::Load>(data1, 1, 0);  // Sequential access
    auto load2 = std::make_shared<ov::snippets::op::Load>(data2, 1, 0);  // Sequential access
    auto scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 1.0f);
    auto add1 = std::make_shared<ov::op::v1::Add>(load1, scalar);
    auto add2 = std::make_shared<ov::op::v1::Add>(load2, scalar);
    auto final_add = std::make_shared<ov::op::v1::Add>(add1, add2);
    auto store = std::make_shared<ov::snippets::op::Store>(final_add, 1, 0);
    
    auto result = std::make_shared<ov::op::v0::Result>(store);
    result->set_friendly_name("result");
    
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data1, data2});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov