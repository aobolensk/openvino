// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_evaluate.hpp"

#include <memory>
#include <vector>

#include "common/convert_common.hpp"
#include "gc/Transforms/Passes.h"
#include "interface/properties.hpp"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"

namespace ov::intel_gpu::mlir {

using namespace ::mlir;

static cl_device_id extract_device_from_context(cl_context context) {
    size_t devices_size = 0;
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size);
    if (err != CL_SUCCESS) {
        OPENVINO_THROW("Error getting context info: ", err);
    }
    if (devices_size / sizeof(cl_device_id) != 1) {
        OPENVINO_THROW("Expected exactly one device in the context, got ", devices_size);
    }

    cl_device_id devices = nullptr;
    // cl_device_id is itself a pointer type and clGetContextInfo takes its output buffer as 'void*',
    // so the extra level of indirection is dictated by the OpenCL API and cannot be avoided here.
    // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, &devices, nullptr);
    if (err != CL_SUCCESS) {
        OPENVINO_THROW("Error getting device IDs: ", err);
    }

    return devices;
}

MLIREvaluateGcGPU::MLIREvaluateGcGPU(OwningOpRef<::mlir::ModuleOp> _module, const std::shared_ptr<ov::EvaluationContext>& loweringContext) {
    if (::ov::intel_gpu::mlir::is_debug()) {
        OPENVINO_MLIR_DEBUG_PRINT("-------------- Source MLIR --------------");
        _module->dump();
        OPENVINO_MLIR_DEBUG_PRINT("-----------------------------------------");
    }

    auto it = loweringContext->find(ov::intel_gpu::ocl_context.name());
    if (it == loweringContext->end()) {
        OPENVINO_THROW("No cl_context provided for OpenCL execution");
    }
    auto* context = reinterpret_cast<cl_context>(it->second.as<ov::intel_gpu::gpu_handle_param>());
    // assuming there's always one device per context
    auto* device = extract_device_from_context(context);

#ifdef OV_GPU_MLIR_OCL_SOURCE
    if (gc::gpu::useOclSourceBackend(device)) {
        auto mod = gc::gpu::OclSourceModule::build(*_module, device, context);
        if (!mod) {
            OPENVINO_THROW(llvm::toString(mod.takeError()));
        }
        source_module = std::make_unique<const gc::gpu::OclSourceModule>(std::move(*mod));
        return;
    }
#endif
#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
    gc::gpu::OclModuleBuilderOpts opts;
    gc::gpu::OclModuleBuilder builder(std::move(_module), opts);
    if (auto mod = builder.build(device, context)) {
        module = std::make_unique<const gc::gpu::OclModule>(*mod);
    } else {
        OPENVINO_THROW("Failed to build gc::gpuOclModule module: ", llvm::toString(mod.takeError()));
    }
#endif
}

bool MLIREvaluateGcGPU::invoke(const ov::TensorVector& inputs, ov::TensorVector& outputs, const ov::EvaluationContext& evaluationContext) {
#ifdef OV_GPU_MLIR_OCL_SOURCE
    if (source_module) {
        const auto& arg_types = evaluationContext.at(ov::internal::mlir_meta::is_kernel_arg_usm.name()).as<std::vector<bool>>();
        OPENVINO_ASSERT(arg_types.size() == inputs.size() + outputs.size(), "MLIR argument count mismatch");
        std::vector<cl_mem> args;
        for (bool is_usm : arg_types) {
            OPENVINO_ASSERT(!is_usm, "OpenCL source MLIR requires device buffers");
        }
        for (const auto& input : inputs) {
            args.push_back(reinterpret_cast<cl_mem>(const_cast<void*>(input.data())));
        }
        for (const auto& output : outputs) {
            args.push_back(reinterpret_cast<cl_mem>(const_cast<void*>(output.data())));
        }
        const auto queue = reinterpret_cast<cl_command_queue>(evaluationContext.at(ov::intel_gpu::ocl_queue.name()).as<void*>());
        std::vector<cl_event> waits;
        const auto wait_it = evaluationContext.find(ov::internal::mlir_meta::wait_list.name());
        if (wait_it != evaluationContext.end()) {
            for (auto event : wait_it->second.as<std::vector<void*>>()) {
                waits.push_back(reinterpret_cast<cl_event>(event));
            }
        }
        cl_event event = nullptr;
        if (auto err = source_module->invoke(queue, args, waits, &event)) {
            OPENVINO_THROW(llvm::toString(std::move(err)));
        }
        const auto results = evaluationContext.find(ov::internal::mlir_meta::result_events.name());
        if (results != evaluationContext.end()) {
            results->second.as<std::vector<void*>*>()->push_back(event);
        } else {
            clReleaseEvent(event);
        }
        return true;
    }
#endif
#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
    gc::gpu::OclContext ctx = build_ocl_context(evaluationContext);
    gc::gpu::StaticExecutor<> exec(*module);

    auto it = evaluationContext.find(ov::internal::mlir_meta::is_kernel_arg_usm.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No is_kernel_arg_usm provided for OpenCL execution");
    }
    std::vector<bool> arg_types = it->second.as<std::vector<bool>>();

    for (size_t i = 0; i < inputs.size(); ++i) {
        exec.arg(inputs[i].data(), arg_types[i]);
    }
    for (size_t i = 0, j = inputs.size(); i < outputs.size(); ++i, ++j) {
        exec.arg(outputs[i].data(), arg_types[j]);
    }

    exec(ctx);

    // Hands the completion events over via the mlir_meta::result_events vector, from which
    // cldnn::mlir_primitive builds the event returned by the primitive
    maybe_set_result_events(evaluationContext, ctx);
    return true;
#else
    OPENVINO_THROW("No MLIR execution backend available");
#endif
}

bool MLIREvaluateGcGPU::invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) {
#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
    gc::gpu::OclContext ctx = build_ocl_context(evaluationContext);
    gc::gpu::DynamicExecutor<> exec(*module);

    // Layout (5 pointers per memref, see MemRefDescriptor::append_to_packed_args
    // in transformations/op/mlir_op.cpp):
    //   [aligned, rank, shape*, strides*, is_usm]
    constexpr size_t kStride = 5;
    OPENVINO_ASSERT(args.size() % kStride == 0, "[GPU] MLIREvaluateGcGPU::invoke_packed: malformed args vector");
    for (size_t i = 0; i < args.size(); i += kStride) {
        exec.arg(
            /*alignedPtr=*/args[i],
            /*rank=*/static_cast<size_t>(reinterpret_cast<uintptr_t>(args[i + 1])),
            /*shape=*/reinterpret_cast<int64_t*>(args[i + 2]),
            /*strides=*/reinterpret_cast<int64_t*>(args[i + 3]),
            /*isUsm=*/reinterpret_cast<uintptr_t>(args[i + 4]) != 0);
    }
    exec(ctx);
    // Hands the completion events over via the mlir_meta::result_events vector, from which
    // cldnn::mlir_primitive builds the event returned by the primitive
    maybe_set_result_events(evaluationContext, ctx);
    return true;
#else
    OPENVINO_THROW("Dynamic shapes are not supported by OpenCL source MLIR");
#endif
}

#if !defined(GC_ENABLE_IGC) || GC_ENABLE_IGC
void MLIREvaluateGcGPU::maybe_set_result_events(const ov::EvaluationContext& evaluationContext, gc::gpu::OclContext& ctx) {
    auto events_it = evaluationContext.find(ov::internal::mlir_meta::result_events.name());
    if (events_it == evaluationContext.end()) {
        return;
    }

    auto retain_event = [](cl_event event) {
        const auto err = clRetainEvent(event);
        if (err != CL_SUCCESS) {
            OPENVINO_THROW("Failed to retain MLIR result event, error: ", err);
        }
    };

    auto* events = events_it->second.as<std::vector<void*>*>();
    events->reserve(events->size() + ctx.events.size());
    for (auto* event : ctx.events) {
        retain_event(event);
        events->push_back(event);
    }
}

// Builds the gc::gpu::OclContext describing a single execution: the queue to enqueue to, the events
// to wait for and whether Graph Compiler has to produce completion events.
//
// The wait-list is stored in the EvaluationContext as a std::vector<void*> owned by an
// ov::Any, while OclContext takes a plain cl_event* + length, so they are copied into a local vector
// first. The OclContext constructor clRetainEvent()s them and copies them into its own storage, so
// that vector is no longer needed once the context is built.
gc::gpu::OclContext MLIREvaluateGcGPU::build_ocl_context(const ov::EvaluationContext& evaluationContext) {
    auto it = evaluationContext.find(ov::intel_gpu::ocl_queue.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No queue provided for OpenCL execution");
    }
    auto* queue = reinterpret_cast<cl_command_queue>(it->second.as<void*>());

    std::vector<void*> waitList;
    uint32_t waitListLen = 0;

    it = evaluationContext.find(ov::internal::mlir_meta::wait_list.name());
    if (it != evaluationContext.end()) {
        waitList = it->second.as<std::vector<void*>>();
        waitListLen = waitList.size();
    }

    const bool createEvents = evaluationContext.count(ov::internal::mlir_meta::result_events.name()) != 0;
    return gc::gpu::OclContext(module->runtime, queue, createEvents, waitListLen, reinterpret_cast<cl_event*>(waitList.data()));
}

#endif
}  // namespace ov::intel_gpu::mlir
