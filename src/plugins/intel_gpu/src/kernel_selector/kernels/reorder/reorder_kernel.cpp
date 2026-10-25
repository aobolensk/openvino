// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BF16);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::UINT16);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::UINT4);
    k.EnableInputDataType(Datatype::INT4);
    k.EnableInputDataType(Datatype::UINT2);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT16);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F4E2M1);
    k.EnableInputDataType(Datatype::F8E4M3);
    k.EnableInputDataType(Datatype::F8E5M2);
    k.EnableInputDataType(Datatype::F8E8M0);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT16);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::UINT16);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::UINT4);
    k.EnableOutputDataType(Datatype::INT4);
    k.EnableOutputDataType(Datatype::UINT2);
    k.EnableOutputDataType(Datatype::BF16);
    k.EnableOutputDataType(Datatype::F4E2M1);
    k.EnableOutputDataType(Datatype::F8E4M3);
    k.EnableOutputDataType(Datatype::F8E5M2);
    k.EnableOutputDataType(Datatype::F8E8M0);
    k.EnableSurfaceInputSupport();
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    k.EnableFP16Emulation();
    return k;
}

JitConstants ReorderKernelRef::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    const bool input_half = params.inputs[0].GetDType() == Datatype::F16;
    const bool output_half = params.outputs[0].GetDType() == Datatype::F16;
    const bool plain_half_copy = input_half && output_half && params.mode == MeanSubtractMode::NONE && params.activations.empty() && params.fused_ops.empty();
    if (!params.engineInfo.supports_fp16 && (input_half || output_half) && !plain_half_copy) {
        auto replace = [&jit](const JitConstants& constants) {
            for (const auto& definition : constants.GetDefinitions()) {
                jit.RemoveConstant(definition.first);
                jit.AddConstant(MakeJitConstant(definition.first, definition.second));
            }
        };
        replace(MakeUnitTypeJitConstants(Datatype::F32));
        replace(MakeTypeJitConstants(Datatype::F32, "CALC"));
        replace({MakeJitConstant("FP16_UNIT_USED", false), MakeJitConstant("FP16_STORAGE", true)});
        if (input_half) {
            replace(MakeTypeJitConstants(Datatype::F32, "INPUT_REORDER"));
            replace({MakeJitConstant("INPUT_REORDER_TYPE", "ushort"),
                     MakeJitConstant("INPUT_REORDER_TYPE_SIZE", 2),
                     MakeJitConstant("DECODE_INPUT_REORDER_COMPUTE_TYPE(v)", "FUNC_CALL(storage_half_to_float)(v)")});
        }
        if (output_half) {
            replace(MakeTypeJitConstants(Datatype::F32, "OUTPUT_REORDER"));
            replace({MakeJitConstant("OUTPUT_REORDER_TYPE", "ushort"),
                     MakeJitConstant("OUTPUT_REORDER_TYPE_SIZE", 2),
                     MakeJitConstant("TO_OUTPUT_REORDER_TYPE(v)", "FUNC_CALL(storage_float_to_half)(convert_float(v))"),
                     MakeJitConstant("TO_OUTPUT_REORDER_TYPE_SAT(v)", "FUNC_CALL(storage_float_to_half)(convert_float(v))")});
        }
    }
    if (params.truncate) {
        jit.AddConstant(MakeJitConstant("CONVERT_TRUNCATE", true));
    }
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    if (params.surface_input) {
        jit.AddConstant(MakeJitConstant("SURFACE_INPUT", true));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            idx_order = {"b", "f", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = {"b", "f", "z", "y", "x"};
        }
        FusedOpsConfiguration conf = {"", idx_order, "res", GetUnitType(params), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    if ( params.inputs[0].GetDType() == Datatype::INT4 ) {
         jit.AddConstant(MakeJitConstant("INT4_INPUT", true));
    }

    if ( params.inputs[0].GetDType() == Datatype::UINT4 ) {
         jit.AddConstant(MakeJitConstant("UINT4_INPUT", true));
    }

    if ( params.inputs[0].GetDType() == Datatype::UINT2 ) {
         jit.AddConstant(MakeJitConstant("UINT2_INPUT", true));
    }

    if ( params.outputs[0].GetDType() == Datatype::UINT4 ) {
         jit.AddConstant(MakeJitConstant("UINT4_OUTPUT", true));
    }

    if ( params.outputs[0].GetDType() == Datatype::INT4 ) {
         jit.AddConstant(MakeJitConstant("INT4_OUTPUT", true));
    }

    if ( params.outputs[0].GetDType() == Datatype::UINT2 ) {
         jit.AddConstant(MakeJitConstant("UINT2_OUTPUT", true));
    }
    
    jit.AddConstant(MakeJitConstant("F4E2M1_INPUT", params.inputs[0].GetDType() == Datatype::F4E2M1 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("F8E5M2_INPUT", params.inputs[0].GetDType() == Datatype::F8E5M2 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("F8E4M3_INPUT", params.inputs[0].GetDType() == Datatype::F8E4M3 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("F8E8M0_INPUT", params.inputs[0].GetDType() == Datatype::F8E8M0 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("F4E2M1_OUTPUT", params.outputs[0].GetDType() == Datatype::F4E2M1 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("F8E5M2_OUTPUT", params.outputs[0].GetDType() == Datatype::F8E5M2 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("F8E4M3_OUTPUT", params.outputs[0].GetDType() == Datatype::F8E4M3 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("F8E8M0_OUTPUT", params.outputs[0].GetDType() == Datatype::F8E8M0 ? 1 : 0));

    return jit;
}

KernelsData ReorderKernelRef::GetKernelsData(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    if (!orgParams.engineInfo.supports_fp16 && !orgParams.fused_ops.empty() &&
        (orgParams.inputs[0].GetDType() == Datatype::F16 || orgParams.outputs[0].GetDType() == Datatype::F16)) {
        return {};
    }
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
