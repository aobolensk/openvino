// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"

/* This file provides test models for Memory Access Pattern Optimizer testing.
 * These models are designed to generate specific memory access patterns that can be optimized:
 * - Stride patterns in loops
 * - Gather/scatter operations
 * - Small memory accesses that can be coalesced
 */

namespace ov {
namespace test {
namespace snippets {

/// Sequential memory access pattern - should be optimized for vectorization
/// Creates a model with sequential Load/Store operations
//   in1
//   Load(stride=1, count=1)
//   Add(scalar)
//   Store(stride=1, count=1)
//   Result
class SequentialMemoryAccessFunction : public SnippetsFunctionBase {
public:
    explicit SequentialMemoryAccessFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

/// Strided memory access pattern - should be optimized for better cache utilization
/// Creates a model with strided Load/Store operations
//   in1
//   Load(stride=4, count=1)
//   Add(scalar)
//   Store(stride=4, count=1)
//   Result
class StridedMemoryAccessFunction : public SnippetsFunctionBase {
public:
    explicit StridedMemoryAccessFunction(const std::vector<PartialShape>& inputShapes, size_t stride = 4) 
        : SnippetsFunctionBase(inputShapes), m_stride(stride) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
    size_t m_stride;
};

/// Gather/scatter memory access pattern - should be detected and optimized
/// Creates a model with irregular memory access patterns
//   in1
//   Load(stride=large_value, count=1)  // Irregular gather
//   Multiply(scalar)
//   Store(stride=large_value, count=1) // Irregular scatter
//   Result
class GatherScatterMemoryAccessFunction : public SnippetsFunctionBase {
public:
    explicit GatherScatterMemoryAccessFunction(const std::vector<PartialShape>& inputShapes, size_t stride = 128) 
        : SnippetsFunctionBase(inputShapes), m_stride(stride) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
    size_t m_stride;
};

/// Small memory accesses that can be coalesced
/// Creates a model with multiple small Load/Store operations
//   in1
//   Load(stride=1, count=1)
//   Load(stride=1, count=1, offset=1)
//   Add
//   Store(stride=1, count=1)
//   Store(stride=1, count=1, offset=1)
//   Result
class SmallCoalescedMemoryAccessFunction : public SnippetsFunctionBase {
public:
    explicit SmallCoalescedMemoryAccessFunction(const std::vector<PartialShape>& inputShapes, size_t num_accesses = 4) 
        : SnippetsFunctionBase(inputShapes), m_num_accesses(num_accesses) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
    size_t m_num_accesses;
};

/// Loop-based memory access pattern - should be optimized for loop vectorization
/// Creates a model with loops containing memory access operations
//   in1
//   LoopBegin
//     Load(stride=1, count=1)
//     Add(scalar)
//     Store(stride=1, count=1)
//   LoopEnd
//   Result
class LoopMemoryAccessFunction : public SnippetsFunctionBase {
public:
    explicit LoopMemoryAccessFunction(const std::vector<PartialShape>& inputShapes, size_t loop_count = 16) 
        : SnippetsFunctionBase(inputShapes), m_loop_count(loop_count) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
    size_t m_loop_count;
};

/// Broadcast memory access pattern - should be optimized for broadcast efficiency
/// Creates a model with broadcast operations
//   in1
//   BroadcastLoad(stride=0, count=1)  // Broadcast pattern
//   Add(scalar)
//   Store(stride=1, count=1)
//   Result
class BroadcastMemoryAccessFunction : public SnippetsFunctionBase {
public:
    explicit BroadcastMemoryAccessFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

/// Complex memory access pattern with mixed access types
/// Creates a model combining different memory access patterns
//   in1   in2
//   Load(sequential)  Load(strided)
//             Add
//   Store(sequential) Store(strided)
//           Result
class MixedMemoryAccessFunction : public SnippetsFunctionBase {
public:
    explicit MixedMemoryAccessFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov