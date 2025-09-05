// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/ref/convolution.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

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

    const size_t rank = srcDesc->getShape().getRank();
    OPENVINO_ASSERT(rank == 4 || rank == 5, "RefConvolutionExecutor supports only 2D/3D convolutions");

    // Strides/dilations/pads
    const size_t SD = rank == 5 ? m_attrs.stride[0] : 1;
    const size_t SH = m_attrs.stride[rank == 5 ? 1 : 0];
    const size_t SW = m_attrs.stride[rank == 5 ? 2 : 1];

    const size_t DD = rank == 5 ? (m_attrs.dilation[0] + 1) : 1;
    const size_t DH = m_attrs.dilation[rank == 5 ? 1 : 0] + 1;
    const size_t DW = m_attrs.dilation[rank == 5 ? 2 : 1] + 1;

    const ptrdiff_t padFront = rank == 5 ? m_attrs.paddingL[0] : 0;
    const ptrdiff_t padTop = m_attrs.paddingL[rank == 5 ? 1 : 0];
    const ptrdiff_t padLeft = m_attrs.paddingL[rank == 5 ? 2 : 1];

    // Dimensions
    const size_t N = srcDims[0];
    size_t C, ID = 1, IH, IW;
    if (rank == 5) {
        if (nspc) {
            ID = srcDims[1];
            IH = srcDims[2];
            IW = srcDims[3];
            C = srcDims[4];
        } else {
            C = srcDims[1];
            ID = srcDims[2];
            IH = srcDims[3];
            IW = srcDims[4];
        }
    } else {  // rank 4
        if (nspc) {
            IH = srcDims[1];
            IW = srcDims[2];
            C = srcDims[3];
        } else {
            C = srcDims[1];
            IH = srcDims[2];
            IW = srcDims[3];
        }
    }

    size_t G = m_attrs.isGrouped ? weiDims[0] : 1;
    size_t OC, IC, KD = 1, KH, KW;
    if (rank == 5) {
        if (m_attrs.isGrouped) {
            OC = weiDims[1] * G;
            IC = weiDims[2] * G;
            KD = weiDims[3];
            KH = weiDims[4];
            KW = weiDims[5];
        } else {
            OC = weiDims[0];
            IC = weiDims[1];
            KD = weiDims[2];
            KH = weiDims[3];
            KW = weiDims[4];
        }
    } else {
        if (m_attrs.isGrouped) {
            OC = weiDims[1] * G;
            IC = weiDims[2] * G;
            KH = weiDims[3];
            KW = weiDims[4];
        } else {
            OC = weiDims[0];
            IC = weiDims[1];
            KH = weiDims[2];
            KW = weiDims[3];
        }
    }

    // Output dims
    size_t OD = 1, OH, OW, OC_out;
    if (rank == 5) {
        if (nspc) {
            OD = dstDims[1];
            OH = dstDims[2];
            OW = dstDims[3];
            OC_out = dstDims[4];
        } else {
            OC_out = dstDims[1];
            OD = dstDims[2];
            OH = dstDims[3];
            OW = dstDims[4];
        }
    } else {
        if (nspc) {
            OH = dstDims[1];
            OW = dstDims[2];
            OC_out = dstDims[3];
        } else {
            OC_out = dstDims[1];
            OH = dstDims[2];
            OW = dstDims[3];
        }
    }

    OPENVINO_ASSERT(OC_out == OC, "Mismatch in output channels");
    OPENVINO_ASSERT(IC == C || (m_attrs.isGrouped && (IC == C)), "Mismatch in input channels");

    const float* src = memory.at(ARG_SRC)->getDataAs<const float>();
    const float* wei = memory.at(ARG_WEI)->getDataAs<const float>();
    const float* bia = nullptr;
    if (m_attrs.withBias && memory.at(ARG_BIAS) && !memory.at(ARG_BIAS)->getDescPtr()->empty()) {
        bia = memory.at(ARG_BIAS)->getDataAs<const float>();
    }
    float* dst = memory.at(ARG_DST)->getDataAs<float>();

    auto src_index_ncsp = [&](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        if (rank == 5) {
            return (((n * C + c) * ID + d) * IH + h) * IW + w;
        } else {
            return ((n * C + c) * IH + h) * IW + w;
        }
    };
    auto src_index_nspc = [&](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        if (rank == 5) {
            return ((((n * ID + d) * IH + h) * IW + w) * C + c);
        } else {
            return (((n * IH + h) * IW + w) * C + c);
        }
    };
    auto dst_index_ncsp = [&](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        if (rank == 5) {
            return (((n * OC + c) * OD + d) * OH + h) * OW + w;
        } else {
            return ((n * OC + c) * OH + h) * OW + w;
        }
    };
    auto dst_index_nspc = [&](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        if (rank == 5) {
            return ((((n * OD + d) * OH + h) * OW + w) * OC + c);
        } else {
            return (((n * OH + h) * OW + w) * OC + c);
        }
    };

    auto wei_index = [&](size_t g, size_t ocg, size_t icg, size_t kd, size_t kh, size_t kw) -> size_t {
        if (rank == 5) {
            if (m_attrs.isGrouped) {
                const size_t OCg = weiDims[1];
                const size_t ICg = weiDims[2];
                return (((((g * OCg + ocg) * ICg + icg) * KD + kd) * KH + kh) * KW + kw);
            } else {
                return (((((ocg) * IC + icg) * KD + kd) * KH + kh) * KW + kw);
            }
        } else {
            if (m_attrs.isGrouped) {
                const size_t OCg = weiDims[1];
                const size_t ICg = weiDims[2];
                return ((((g * OCg + ocg) * ICg + icg) * KH + kh) * KW + kw);
            } else {
                return ((((ocg) * IC + icg) * KH + kh) * KW + kw);
            }
        }
    };

    const size_t OCg = m_attrs.isGrouped ? weiDims[1] : OC;
    const size_t ICg = m_attrs.isGrouped ? weiDims[2] : IC;

    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < G; ++g) {
            for (size_t ocg = 0; ocg < OCg; ++ocg) {
                const size_t oc = g * OCg + ocg;
                for (size_t od = 0; od < OD; ++od) {
                    const int64_t base_id = static_cast<int64_t>(od) * static_cast<int64_t>(SD) - padFront;
                    for (size_t oh = 0; oh < OH; ++oh) {
                        const int64_t base_ih = static_cast<int64_t>(oh) * static_cast<int64_t>(SH) - padTop;
                        for (size_t ow = 0; ow < OW; ++ow) {
                            const int64_t base_iw = static_cast<int64_t>(ow) * static_cast<int64_t>(SW) - padLeft;
                            float acc = 0.0f;
                            for (size_t icg = 0; icg < ICg; ++icg) {
                                const size_t ic = g * ICg + icg;
                                for (size_t kd = 0; kd < (rank == 5 ? KD : 1); ++kd) {
                                    const int64_t id = base_id + static_cast<int64_t>(kd) * static_cast<int64_t>(DD);
                                    if (rank == 5 && (id < 0 || id >= static_cast<int64_t>(ID))) continue;
                                    for (size_t kh = 0; kh < KH; ++kh) {
                                        const int64_t ih = base_ih + static_cast<int64_t>(kh) * static_cast<int64_t>(DH);
                                        if (ih < 0 || ih >= static_cast<int64_t>(IH)) continue;
                                        for (size_t kw = 0; kw < KW; ++kw) {
                                            const int64_t iw = base_iw + static_cast<int64_t>(kw) * static_cast<int64_t>(DW);
                                            if (iw < 0 || iw >= static_cast<int64_t>(IW)) continue;

                                            const size_t sidx = nspc ? src_index_nspc(n, ic, (rank == 5 ? static_cast<size_t>(id) : 0),
                                                                                      static_cast<size_t>(ih),
                                                                                      static_cast<size_t>(iw))
                                                                     : src_index_ncsp(n, ic, (rank == 5 ? static_cast<size_t>(id) : 0),
                                                                                      static_cast<size_t>(ih),
                                                                                      static_cast<size_t>(iw));
                                            const size_t widx = wei_index(g, ocg, icg, (rank == 5 ? kd : 0), kh, kw);
                                            acc += src[sidx] * wei[widx];
                                        }
                                    }
                                }
                            }
                            if (bia) {
                                acc += bia[oc];
                            }
                            const size_t didx = nspc ? dst_index_nspc(n, oc, (rank == 5 ? od : 0), oh, ow)
                                                     : dst_index_ncsp(n, oc, (rank == 5 ? od : 0), oh, ow);
                            dst[didx] = acc;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace ov::intel_cpu

