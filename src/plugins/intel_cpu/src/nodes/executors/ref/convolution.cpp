// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(OPENVINO_ARCH_RISCV64)

#    include "nodes/executors/ref/convolution.hpp"

#    include <algorithm>
#    include <cstddef>
#    include <cstdint>
#    include <numeric>
#    include <vector>

#    include "openvino/core/except.hpp"
#    include "openvino/core/type/element_type.hpp"
#    include "openvino/reference/convolution.hpp"
#    include "openvino/reference/group_convolution.hpp"

namespace ov::intel_cpu {

RefConvolutionExecutor::~RefConvolutionExecutor() = default;

RefConvolutionExecutor::RefConvolutionExecutor(ConvAttrs attrs, const MemoryArgs& /*memory*/, ExecutorContext::CPtr /*context*/)
    : m_attrs(std::move(attrs)) {}

bool RefConvolutionExecutor::isNspc(const MemoryDescPtr& desc) {
    return desc && desc->hasLayoutType(LayoutType::nspc);
}

bool RefConvolutionExecutor::update(const MemoryArgs& /*memory*/) {
    return true;
}

impl_desc_type RefConvolutionExecutor::implType() const {
    return impl_desc_type::ref;
}

void RefConvolutionExecutor::anchor() {}

static inline int64_t div_floor(int64_t a, int64_t b) {
    return (a - (a < 0) * (b - 1)) / b;
}

void RefConvolutionExecutor::execute(const MemoryArgs& memory) {
    // Only FP32 reference compute for now
    OPENVINO_ASSERT(memory.at(ARG_SRC)->getPrecision() == ov::element::f32,
                    "RefConvolutionExecutor supports only f32 src");
    OPENVINO_ASSERT(memory.at(ARG_WEI)->getPrecision() == ov::element::f32,
                    "RefConvolutionExecutor supports only f32 weights");
    OPENVINO_ASSERT(memory.at(ARG_DST)->getPrecision() == ov::element::f32,
                    "RefConvolutionExecutor supports only f32 dst");

    const auto srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto& srcDims = srcDesc->getShape().getDims();
    const auto& weiDims = weiDesc->getShape().getDims();
    const auto& dstDims = dstDesc->getShape().getDims();

    const bool nspc = isNspc(srcDesc);
    OPENVINO_ASSERT(!nspc, "RefConvolutionExecutor supports only NCSP/NCDHW layouts");
    const size_t rank = srcDesc->getShape().getRank();
    OPENVINO_ASSERT(rank == 4 || rank == 5, "RefConvolutionExecutor supports only 2D/3D convolutions");

    // Build common N,C,spatial dimensions irrespective of layout
    const size_t N = srcDims[0];
    size_t C = 0;
    std::vector<size_t> in_spatial;
    in_spatial.reserve(rank - 2);
    if (rank == 5) {
        C = srcDims[1];
        in_spatial = {srcDims[2], srcDims[3], srcDims[4]};
    } else {
        C = srcDims[1];
        in_spatial = {srcDims[2], srcDims[3]};
    }

    // Output logical OC and spatial dims (N,C_out,spatial)
    size_t OC = 0;
    std::vector<size_t> out_spatial;
    out_spatial.reserve(rank - 2);
    if (rank == 5) {
        OC = dstDims[1];
        out_spatial = {dstDims[2], dstDims[3], dstDims[4]};
    } else {
        OC = dstDims[1];
        out_spatial = {dstDims[2], dstDims[3]};
    }

    const float* src = memory.at(ARG_SRC)->getDataAs<const float>();
    const float* wei = memory.at(ARG_WEI)->getDataAs<const float>();
    const float* bia = nullptr;
    if (m_attrs.withBias && memory.at(ARG_BIAS) && !memory.at(ARG_BIAS)->getDescPtr()->empty()) {
        bia = memory.at(ARG_BIAS)->getDataAs<const float>();
    }
    float* dst = memory.at(ARG_DST)->getDataAs<float>();

    // Build ov::Shapes (N,C,spatial...)
    ov::Shape in_shape;
    in_shape.reserve(2 + in_spatial.size());
    in_shape.push_back(N);
    in_shape.push_back(C);
    in_shape.insert(in_shape.end(), in_spatial.begin(), in_spatial.end());

    ov::Shape f_shape{weiDims.begin(), weiDims.end()};
    ov::Shape out_shape;
    out_shape.reserve(2 + out_spatial.size());
    out_shape.push_back(N);
    out_shape.push_back(OC);
    out_shape.insert(out_shape.end(), out_spatial.begin(), out_spatial.end());

    // Build strides/dilations/pads for reference API
    ov::Strides strides(m_attrs.stride.begin(), m_attrs.stride.end());
    ov::Strides dilations;
    dilations.reserve(m_attrs.dilation.size());
    for (auto d : m_attrs.dilation)
        dilations.push_back(d + 1);
    ov::CoordinateDiff pads_begin(m_attrs.paddingL.begin(), m_attrs.paddingL.end());
    ov::CoordinateDiff pads_end(m_attrs.paddingR.begin(), m_attrs.paddingR.end());

    // Prepare source and destination buffers in NCSP order expected by ov::reference
    const size_t in_spatial_size =
        std::accumulate(in_spatial.begin(), in_spatial.end(), size_t{1}, std::multiplies<size_t>());
    const size_t out_spatial_size =
        std::accumulate(out_spatial.begin(), out_spatial.end(), size_t{1}, std::multiplies<size_t>());
    const float* src_for_ref = src;
    float* dst_for_ref = dst;

    // Invoke reference convolution (grouped or regular)
    if (m_attrs.isGrouped) {
        ov::reference::group_convolution<float, float, float>(src_for_ref,
                                                              wei,
                                                              dst_for_ref,
                                                              in_shape,
                                                              f_shape,
                                                              out_shape,
                                                              strides,
                                                              dilations,
                                                              pads_begin,
                                                              pads_end);
    } else {
        ov::reference::convolution<float>(src_for_ref,
                                          wei,
                                          dst_for_ref,
                                          in_shape,
                                          f_shape,
                                          out_shape,
                                          strides,
                                          dilations,
                                          pads_begin,
                                          pads_end);
    }

    // Apply bias if present (on NCSP pointer)
    if (bia) {
        float* out_ncsp = dst_for_ref;
        size_t idx = 0;
        for (size_t n = 0; n < N; ++n) {
            for (size_t oc = 0; oc < OC; ++oc) {
                for (size_t s = 0; s < out_spatial_size; ++s, ++idx) {
                    out_ncsp[idx] += bia[oc];
                }
            }
        }
    }

    // No layout reorder back is needed (NCSP only)
}

}  // namespace ov::intel_cpu

#endif  // OPENVINO_ARCH_RISCV64
