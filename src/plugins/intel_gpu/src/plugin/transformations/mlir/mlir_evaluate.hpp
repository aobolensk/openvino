// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#if GC_ENABLE_IGC
#    include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#endif
#include "gc/ExecutionEngine/GPURuntime/OclSourceModule.h"
#include "interface/mlir_evaluate_base.hpp"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::intel_gpu::mlir {

using ::mlir::ExecutionEngine;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;

class MLIREvaluateGcGPU : public MLIREvaluateBase {
#if GC_ENABLE_IGC
    std::unique_ptr<const ::mlir::gc::gpu::OclModule> module;
#endif
    std::unique_ptr<const ::mlir::gc::gpu::OclSourceModule> source_module;

public:
    MLIREvaluateGcGPU(OwningOpRef<ModuleOp> _module, const std::shared_ptr<ov::EvaluationContext>& loweringContext);

    [[nodiscard]] bool requires_packed_args() const override {
#if GC_ENABLE_IGC
        return source_module ? !source_module->isStatic() : !module->isStatic();
#else
        return !source_module->isStatic();
#endif
    }
    bool invoke(const ov::TensorVector& inputs, ov::TensorVector& outputs, const ov::EvaluationContext& evaluationContext) override;
    bool invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) override;

private:
    bool invoke_source(const std::vector<cl_mem>& buffers, llvm::ArrayRef<llvm::ArrayRef<int64_t>> shapes, const ov::EvaluationContext& evaluationContext);
#if GC_ENABLE_IGC
    ::mlir::gc::gpu::OclContext build_ocl_context(const ov::EvaluationContext& evaluationContext);
    static void maybe_set_result_events(const ov::EvaluationContext& evaluationContext, ::mlir::gc::gpu::OclContext& ctx);
#endif
};

}  // namespace ov::intel_gpu::mlir
