// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/matmul.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

Result MMShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                           [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const VectorDims& shapeA = input_shapes[0].get();
    const VectorDims& shapeB = input_shapes[1].get();
    const size_t rankA = shapeA.size();
    const size_t rankB = shapeB.size();

    // getSupportedDescriptors has done some shape check.
    // 1. Needn't assert the scalar type since the matmul_shape_inference has checked.
    // 2. Needn't check the compatibility of the last two dims
    // 3. 1-D x 1-D is needed
    // 4. transpose is necessary
    // 5. Just support the same rank of matmul
    // 6. simplify the broadcast check
    if (rankA == 1 && rankB == 1 && shapeA[0] == shapeB[0]) {
        return {{m_shapeY}, ShapeInferStatus::success};
    }
    OPENVINO_ASSERT(m_out_rank >= 2, "The output rank should be greater or euqal to 2.");
    const size_t k_lhs = m_transpose_a ? shapeA[rankA - 2] : shapeA[rankA - 1];
    const size_t k_rhs = m_transpose_b ? shapeB[rankB - 1] : shapeB[rankB - 2];
    OPENVINO_ASSERT(k_lhs == k_rhs,
                    "Matmul input shapes are incompatible shape A: ",
                    vec2str(shapeA),
                    m_transpose_a ? "T " : " ",
                    "shape B: ",
                    vec2str(shapeB),
                    m_transpose_b ? "T" : "");

    m_shapeY[m_out_rank - 2] = m_transpose_a ? shapeA[rankA - 1] : shapeA[rankA - 2];
    m_shapeY[m_out_rank - 1] = m_transpose_b ? shapeB[rankB - 2] : shapeB[rankB - 1];

    for (size_t i = 0; i < m_out_rank - 2; ++i) {
        if (shapeA[i] != shapeB[i]) {
            if (shapeB[i] == 1) {
                m_shapeY[i] = shapeA[i];
                continue;
            }
            OPENVINO_ASSERT(shapeA[i] == 1,
                            "Incompatible MatMul batch dimension. Cant merge the first input dimension=",
                            shapeA[i],
                            " with second input dimension=",
                            shapeB[i],
                            " at index=",
                            i);
        }
        m_shapeY[i] = shapeB[i];
    }

    return {{m_shapeY}, ShapeInferStatus::success};
}

ShapeInferPtr MMShapeInferFactory::makeShapeInfer() const {
    if (const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(m_op)) {
        const auto input_rank0 = matmul->get_input_partial_shape(0).rank().get_length();
        const auto input_rank1 = matmul->get_input_partial_shape(1).rank().get_length();

        if (input_rank0 == input_rank1) {
            const auto output_rank = matmul->get_output_partial_shape(0).rank().get_length();
            const bool transpose_a = matmul->get_transpose_a();
            const bool transpose_b = matmul->get_transpose_b();
            return std::make_shared<MMShapeInfer>(output_rank, transpose_a, transpose_b);
        }
        return make_shape_inference(m_op);
    }
    OPENVINO_THROW("Unexpected operation type in the MatMul shape inference factory");
}
}  // namespace ov::intel_cpu::node
