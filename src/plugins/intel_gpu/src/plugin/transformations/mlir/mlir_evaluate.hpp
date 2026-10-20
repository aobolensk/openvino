// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
#    include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#endif
#ifdef OV_GPU_MLIR_OCL_SOURCE
#    include "gc/ExecutionEngine/GPURuntime/OclSourceModule.h"
#endif
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
#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
    std::unique_ptr<const ::mlir::gc::gpu::OclModule> module;
#endif
#ifdef OV_GPU_MLIR_OCL_SOURCE
    std::unique_ptr<const ::mlir::gc::gpu::OclSourceModule> source_module;
#endif

public:
    MLIREvaluateGcGPU(OwningOpRef<ModuleOp> _module, const std::shared_ptr<ov::EvaluationContext>& loweringContext);

    [[nodiscard]] bool requires_packed_args() const override {
#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
        return module && !module->isStatic();
#else
        return false;
#endif
    }
    bool invoke(const ov::TensorVector& inputs, ov::TensorVector& outputs, const ov::EvaluationContext& evaluationContext) override;
    bool invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) override;

private:
#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
    ::mlir::gc::gpu::OclContext build_ocl_context(const ov::EvaluationContext& evaluationContext);
    static void maybe_set_result_events(const ov::EvaluationContext& evaluationContext, ::mlir::gc::gpu::OclContext& ctx);
#endif
};

}  // namespace ov::intel_gpu::mlir
