// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shl_convolution.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "csinn/csi_nn.h"
#include "csinn_data_structure.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/shl/shl_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shl.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
namespace {
MemoryPtr prepareWeightMemory(const MemoryPtr& weightsMemory, const ExecutorContext::CPtr& context) {
    DEBUG_LOG("ShlConvExecutor: copy weights");

    const auto& weiDesc = weightsMemory->getDescPtr();
    MemoryPtr packed =
        std::make_shared<Memory>(context->getEngine(),
                                 intel_cpu::CpuBlockedMemoryDesc(weiDesc->getPrecision(), weightsMemory->getShape()));
    cpu_parallel_memcpy(packed->getData(), weightsMemory->getData(), weightsMemory->getSize());
    return packed;
}
}  // namespace

bool ShlConvExecutor::supports(const ConvConfig& config) {
    if (!config.attrs.postOps.empty()) {
        DEBUG_LOG("ShlConvExecutor: PostOps are not supported");
        return false;
    }

    if (config.attrs.isGraphQuantized) {
        DEBUG_LOG("ShlConvExecutor: quantized graphs are not supported");
        return false;
    }

    if (config.attrs.isGrouped) {
        DEBUG_LOG("ShlConvExecutor: Grouped convolutions are not supported");
        return false;
    }

    if (!config.attrs.constantWeights) {
        DEBUG_LOG("ShlConvExecutor: dynamic weights are not supported");
        return false;
    }

    const auto& strides = config.attrs.stride;
    const auto& dilations = config.attrs.dilation;
    const auto& padsL = config.attrs.paddingL;
    const auto& padsR = config.attrs.paddingR;
    if (strides.size() != 2 || dilations.size() != 2 || padsL.size() != 2 || padsR.size() != 2) {
        DEBUG_LOG("ShlConvExecutor: only 2D convolutions are supported");
        return false;
    }

    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);
    if (srcDesc->getShape().getRank() != 4 || dstDesc->getShape().getRank() != 4 ||
        weiDesc->getShape().getRank() != 4) {
        DEBUG_LOG("ShlConvExecutor: supports only 2D convolution tensors");
        return false;
    }

    if (!all_of(ov::element::f32, srcDesc->getPrecision(), weiDesc->getPrecision(), dstDesc->getPrecision())) {
        DEBUG_LOG("ShlConvExecutor: supports only f32 precisions");
        return false;
    }

    if (hasBias(config)) {
        const auto& biaDesc = config.descs.at(ARG_BIAS);
        if (biaDesc->getPrecision() != ov::element::f32) {
            DEBUG_LOG("ShlConvExecutor: bias precision is not supported");
            return false;
        }
        const auto& biasDims = biaDesc->getShape().getStaticDims();
        const auto& outDims = dstDesc->getShape().getDims();
        const size_t channelAxis = dstDesc->hasLayoutType(LayoutType::nspc) ? outDims.size() - 1 : 1;
        if (biasDims.size() != 1 || biasDims[0] != outDims.at(channelAxis)) {
            DEBUG_LOG("ShlConvExecutor: bias must be 1D and match output channels");
            return false;
        }
    }

    const auto srcLayout = getShlDataLayoutByMemoryDesc(srcDesc);
    const auto weiLayout = getShlDataLayoutByMemoryDesc(weiDesc, true);
    const auto dstLayout = getShlDataLayoutByMemoryDesc(dstDesc);
    if (srcLayout == CSINN_LAYOUT_NULL || dstLayout == CSINN_LAYOUT_NULL || weiLayout == CSINN_LAYOUT_NULL) {
        DEBUG_LOG("ShlConvExecutor: unsupported layout");
        return false;
    }

    return true;
}

ShlConvExecutor::ShlConvExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context)
    : params(sess, CSINN_RVV) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto srcLayout = getShlDataLayoutByMemoryDesc(srcDesc);
    const auto weiLayout = getShlDataLayoutByMemoryDesc(weiDesc, true);
    const auto dstLayout = getShlDataLayoutByMemoryDesc(dstDesc);

    initParams(attrs, srcLayout);

    src = ShlTensor(sess,
                    precisionToShlDataType(srcDesc->getPrecision()),
                    srcLayout,
                    srcDesc->getShape().getStaticDims());
    wei = ShlTensor(sess,
                    precisionToShlDataType(weiDesc->getPrecision()),
                    weiLayout,
                    weiDesc->getShape().getStaticDims());
    dst = ShlTensor(sess,
                    precisionToShlDataType(dstDesc->getPrecision()),
                    dstLayout,
                    dstDesc->getShape().getStaticDims());

    with_bias = !memory.at(ARG_BIAS)->getDesc().empty();
    if (with_bias) {
        const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
        bias = ShlTensor(sess,
                         precisionToShlDataType(biasDesc->getPrecision()),
                         getShlDataLayoutByMemoryDesc(biasDesc),
                         biasDesc->getShape().getStaticDims());
    } else {
        bias = ShlTensor(sess);
    }

    packedWeights = prepareWeightMemory(memory.at(ARG_WEI), context);
    wei.setData(packedWeights->getData());

    OPENVINO_ASSERT(csinn_conv2d_init(src.get(),
                                      dst.get(),
                                      wei.get(),
                                      bias.get(),
                                      static_cast<csinn_conv2d_params*>(params.get())) == CSINN_TRUE,
                    "ShlConvExecutor: failed to init convolution");
}

bool ShlConvExecutor::update(const MemoryArgs& memory) {
    src = src.cloneWithNewShape(memory.at(ARG_SRC)->getDescPtr()->getShape().getStaticDims());
    dst = dst.cloneWithNewShape(memory.at(ARG_DST)->getDescPtr()->getShape().getStaticDims());
    if (with_bias) {
        bias = bias.cloneWithNewShape(memory.at(ARG_BIAS)->getDescPtr()->getShape().getStaticDims());
    }

    return true;
}

void ShlConvExecutor::execute(const MemoryArgs& memory) {
    src.setData(memory.at(ARG_SRC)->getData());
    dst.setData(memory.at(ARG_DST)->getData());
    wei.setData(packedWeights->getData());

    if (with_bias) {
        bias.setData(memory.at(ARG_BIAS)->getData());
    }

    OPENVINO_ASSERT(
        csinn_conv2d(src.get(), dst.get(), wei.get(), bias.get(), static_cast<csinn_conv2d_params*>(params.get())) ==
            CSINN_TRUE,
        "ShlConvExecutor: failed to execute");
}

void ShlConvExecutor::initParams(const ConvAttrs& attrs, csinn_layout_enum srcLayout) {
    auto* convParams = static_cast<csinn_conv2d_params*>(params.get());
    convParams->base.layout = srcLayout;
    convParams->group = 1;
    convParams->stride_height = static_cast<int32_t>(attrs.stride[0]);
    convParams->stride_width = static_cast<int32_t>(attrs.stride[1]);
    convParams->pad_top = static_cast<int32_t>(attrs.paddingL[0]);
    convParams->pad_left = static_cast<int32_t>(attrs.paddingL[1]);
    convParams->pad_down = static_cast<int32_t>(attrs.paddingR[0]);
    convParams->pad_right = static_cast<int32_t>(attrs.paddingR[1]);
    convParams->dilation_height = static_cast<int32_t>(attrs.dilation[0] + 1);
    convParams->dilation_width = static_cast<int32_t>(attrs.dilation[1] + 1);
    convParams->out_pad_height = 0;
    convParams->out_pad_width = 0;
}

}  // namespace ov::intel_cpu
