// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lowered/pass/buffer_allocation.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/reduce_decomposition.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/result.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/pass/lowered/gemm_cpu_blocking.hpp"
#include "transformations/snippets/aarch64/pass/lowered/insert_gemm_copy_buffers.hpp"
#include "transformations/snippets/common/shape_inference.hpp"

namespace ov {
namespace test {
namespace snippets {

class BufferAllocationCPUTest : public BufferAllocationTest {
protected:
    std::shared_ptr<ov::snippets::IShapeInferSnippetsFactory> GetShapeInferFactory() const override {
        return std::make_shared<ov::snippets::CPUShapeInferSnippetsFactory>();
    }

    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> getBackendSpecificPasses() override {
        std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> backend_passes;

        backend_passes.emplace_back(
            ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::After,
                                             ov::snippets::lowered::pass::MarkLoops::get_type_info_static()),
            std::make_shared<ov::intel_cpu::pass::GemmCPUBlocking>());

        backend_passes.emplace_back(
            ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::After,
                                             ov::snippets::lowered::pass::SplitLoops::get_type_info_static()),
            std::make_shared<ov::intel_cpu::pass::aarch64::InsertGemmCopyBuffers>());

        return backend_passes;
    }
};

class MHAFP32BufferAllocationTest : public BufferAllocationCPUTest {
protected:
    std::shared_ptr<ov::Model> GetModel(const std::vector<ov::PartialShape>& shapes) const override {
        const auto subtensor_scalar = std::vector<size_t>{1};
        const auto subtensor_power = std::vector<size_t>{1, ov::snippets::utils::get_full_dim_value()};
        const auto subtensor_full = std::vector<size_t>(2, ov::snippets::utils::get_full_dim_value());

        using PortDescriptor = ov::snippets::modifier::MemoryAccess::PortDescriptor;

        // Shapes directly compatible for GEMM without layouts:
        // GEMM0: [M, K] * [K, N] = [M, N], GEMM1: [M, N] * [N, P] = [M, P]
        OPENVINO_ASSERT(shapes.size() == 3, "Incorrect count of input shapes");
        const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[0]);
        const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[1]);
        const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[2]);

        const auto gemm_cpu0 = std::make_shared<ov::intel_cpu::aarch64::GemmCPU>(parameter0,
                                                                                 parameter1,
                                                                                 PortDescriptor{},
                                                                                 PortDescriptor{},
                                                                                 PortDescriptor{});

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(gemm_cpu0);

        // Decomposed Softmax
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(relu1, 1);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, 1);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        const auto gemm_cpu1 = std::make_shared<ov::intel_cpu::aarch64::GemmCPU>(multiply,
                                                                                 parameter2,
                                                                                 PortDescriptor{},
                                                                                 PortDescriptor{},
                                                                                 PortDescriptor{});

        const auto relu2 = std::make_shared<ov::op::v0::Relu>(gemm_cpu1);

        const auto body = std::make_shared<ov::Model>(std::make_shared<ov::snippets::op::Result>(relu2),
                                                      ov::ParameterVector{parameter0, parameter1, parameter2});

        MarkOp(gemm_cpu0, {subtensor_full, subtensor_full}, {subtensor_full});
        MarkOp(gemm_cpu1, {subtensor_full, subtensor_full}, {subtensor_full});
        MarkOp(power, {subtensor_power}, {subtensor_power});

        return body;
    }
};

TEST_P(MHAFP32BufferAllocationTest, BufferAllocationCPU) {
    Validate();
}

namespace BufferAllocationCPUTest_Instances {

// Shapes compatible without layouts: GEMM0 [M,K]*[K,N], GEMM1 [M,N]*[N,P]
std::vector<ov::PartialShape> static_shapes = {
    {1024, 512},
    {512, 1024},
    {1024, 256},
};

std::vector<ov::PartialShape> dynamic_shapes = {
    {-1, -1},
    {-1, -1},
    {-1, -1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWSplit,
                         MHAFP32BufferAllocationTest,
                         ::testing::Combine(::testing::Values(static_shapes),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(278528),
                                            ::testing::Values(5),
                                            ::testing::Values(5)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWSplit,
                         MHAFP32BufferAllocationTest,
                         ::testing::Combine(::testing::Values(static_shapes),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(139264),
                                            ::testing::Values(1),
                                            ::testing::Values(2)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWOSplit,
                         MHAFP32BufferAllocationTest,
                         ::testing::Combine(::testing::Values(static_shapes),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(9445376),
                                            ::testing::Values(5),
                                            ::testing::Values(5)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWOSplit,
                         MHAFP32BufferAllocationTest,
                         ::testing::Combine(::testing::Values(static_shapes),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(5242880),
                                            ::testing::Values(2),
                                            ::testing::Values(2)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWSplit_Dynamic,
                         MHAFP32BufferAllocationTest,
                         ::testing::Combine(::testing::Values(dynamic_shapes),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(0),
                                            ::testing::Values(2),
                                            ::testing::Values(2)),
                         BufferAllocationTest::getTestCaseName);

}  // namespace BufferAllocationCPUTest_Instances
}  // namespace snippets
}  // namespace test
}  // namespace ov
