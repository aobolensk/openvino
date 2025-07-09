// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_mm.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/parallel.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/precision_support.h"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov {
namespace intel_cpu {

using namespace executor;
using namespace ov::element;

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, (T)1, std::multiplies<T>()), dims[dims.size() - 1]};
}

static bool useDynamicQuantizationImpl(const FCAttrs& attrs, const MemoryDescPtr& weightDesc) {
    if (attrs.dynamicQuantizationGroupSize != std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    if (!hasIntDotProductSupport()) {
        return false;
    }

    if (weightDesc->getPrecision() != element::i8) {
        return false;
    }

    return true;
}

bool MatMulKleidiAIExecutor::supports(const FCConfig& config) {
    return !static_cast<bool>(config.descs.at(ARG_WEI)->getPrecision() != element::f32 &&
                              !useDynamicQuantizationImpl(config.attrs, config.descs.at(ARG_WEI)));
}

MatMulKleidiAIExecutor::MatMulKleidiAIExecutor(const FCAttrs& attrs,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context)
    : m_attrs(attrs),
      m_memoryArgs(memory) {
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto weiDims = weiMem->getDesc().getShape().getDims();
    auto N = weiDims[0];
    auto K = weiDims[1];
    
    std::cerr << "MatMulKleidiAIExecutor::Constructor - N=" << N << ", K=" << K << 
                 ", src_precision=" << srcMem->getPrecision() << 
                 ", wei_precision=" << weiMem->getPrecision() <<
                 ", weightsNonTransposed=" << attrs.weightsNonTransposed <<
                 ", dynamicQuantizationGroupSize=" << attrs.dynamicQuantizationGroupSize << std::endl;

    if (memory.at(ARG_BIAS)->getDataAs<float>() == nullptr) {
        auto biasDesc = std::make_shared<CpuBlockedMemoryDesc>(f32, Shape({N}));
        biasMem = std::make_shared<Memory>(context->getEngine(), biasDesc);
        biasMem->nullify();
    } else {
        biasMem = memory.at(ARG_BIAS);
    }
    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
    auto originalWeightsDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& wgtDims = originalWeightsDesc->getShape().getStaticDims();
    const VectorDims wgtDims2D = reshapeDownToRank<2>(wgtDims);
    originalWeightsDesc = std::make_shared<CpuBlockedMemoryDesc>(originalWeightsDesc->getPrecision(), Shape{wgtDims2D});
    auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);

    // Whether dynamic quantization is enabled
    useDynamicQuant = useDynamicQuantizationImpl(attrs, originalWeightsDesc);
    std::cerr << "MatMulKleidiAIExecutor::Constructor - useDynamicQuant=" << useDynamicQuant << std::endl;

    if (!useDynamicQuant) {
        auto dstDesc = originalWeightsDesc->cloneWithNewPrecision(memory.at(ARG_SRC)->getDescPtr()->getPrecision());
        auto dnnlDstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc);
        packedWeights = acl_fc_executor::reorderWeights(memory, context, aclfcAttrs, dnnlSrcDesc, dnnlDstDesc);

        // Skip packing if any dimension is zero to avoid KAI kernel crashes
        if (N == 0 || K == 0) {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Skipping f32 packing due to zero dimensions: N=" << N << ", K=" << K << std::endl;
            return;
        }
        const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
        std::cerr << "MatMulKleidiAIExecutor::Constructor - f32 rhsPackedSize=" << rhsPackedSize << " for N=" << N << ", K=" << K << std::endl;
        if (rhsPackedSize == 0) {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Skipping f32 packing due to zero rhsPackedSize" << std::endl;
            return;
        }
        auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(u8, Shape({rhsPackedSize}));
        rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);

        float* bias = biasMem->getDataAs<float>();
        float* rhs_packed = static_cast<float*>(rhsPackedMem->getData());
        float* rhs = static_cast<float*>(packedWeights->getData());
        const size_t rhs_stride = N * sizeof(float);

        const size_t nr = ukernel_f32.get_nr();
        const size_t kr = ukernel_f32.get_kr();
        const size_t sr = ukernel_f32.get_sr();
        std::cerr << "MatMulKleidiAIExecutor::Constructor - f32 ukernel params: nr=" << nr << ", kr=" << kr << ", sr=" << sr << std::endl;

        // Additional safety check before calling KAI packing function
        if (N > 0 && K > 0 && rhs_packed != nullptr && rhs != nullptr && bias != nullptr) {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Calling kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon with:" <<
                         " N=" << N << ", K=" << K << ", nr=" << nr << ", kr=" << kr << ", sr=" << sr << ", rhs_stride=" << rhs_stride <<
                         ", rhs=" << static_cast<void*>(rhs) << ", bias=" << static_cast<void*>(bias) << ", rhs_packed=" << static_cast<void*>(rhs_packed) << std::endl;
            kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
                                                             N,
                                                             K,
                                                             nr,
                                                             kr,
                                                             sr,          // Packing arguments
                                                             rhs_stride,  // RHS stride
                                                             rhs,         // RHS
                                                             bias,        // Bias
                                                             nullptr,     // Scale
                                                             rhs_packed,  // RHS packed
                                                             0,
                                                             nullptr);
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Completed kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon" << std::endl;
        } else {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Skipping f32 packing due to safety check failure:" <<
                         " N=" << N << ", K=" << K << ", rhs_packed=" << static_cast<void*>(rhs_packed) << 
                         ", rhs=" << static_cast<void*>(rhs) << ", bias=" << static_cast<void*>(bias) << std::endl;
        }
    } else {
        MemoryPtr weightsMemory = memory.at(ARG_WEI);
        if (!attrs.weightsNonTransposed) {
            auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);
            auto dnnlDstDesc = acl_fc_executor::makeTransposedWeightDescriptor(dnnlSrcDesc, dnnlSrcDesc);
            weightsMemory = acl_fc_executor::reorderData(dnnlSrcDesc, dnnlDstDesc, memory.at(ARG_WEI), context);
        }

        mr = ukernel_i8.get_mr();
        nr = ukernel_i8.get_nr();
        kr = ukernel_i8.get_kr();
        sr = ukernel_i8.get_sr();
        std::cerr << "MatMulKleidiAIExecutor::Constructor - i8 ukernel params: mr=" << mr << ", nr=" << nr << ", kr=" << kr << ", sr=" << sr << std::endl;

        float* bias = biasMem->getDataAs<float>();
        int8_t* rhs_native_qs8cx = weightsMemory->getDataAs<int8_t>();
        float* rhs_scales = static_cast<float*>(memory.at(ARG_WEI | ARG_ATTR_SCALES)->getData());

        // Skip packing if any dimension is zero to avoid KAI kernel crashes
        if (N == 0 || K == 0) {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Skipping i8 packing due to zero dimensions: N=" << N << ", K=" << K << std::endl;
            return;
        }
        const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
        std::cerr << "MatMulKleidiAIExecutor::Constructor - i8 rhsPackedSize=" << rhsPackedSize << " for N=" << N << ", K=" << K << ", nr=" << nr << ", kr=" << kr << ", sr=" << sr << std::endl;
        if (rhsPackedSize == 0) {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Skipping i8 packing due to zero rhsPackedSize" << std::endl;
            return;
        }
        auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({rhsPackedSize}));
        rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);
        int8_t* rhs_packed_qs8cx = static_cast<int8_t*>(rhsPackedMem->getData());

        kai_rhs_pack_qsi8cx_params params;
        params.lhs_zero_point = 1;
        std::cerr << "MatMulKleidiAIExecutor::Constructor - i8 packing params: lhs_zero_point=" << params.lhs_zero_point << std::endl;

        // Additional safety check before calling KAI packing function
        if (N > 0 && K > 0 && rhs_packed_qs8cx != nullptr && rhs_native_qs8cx != nullptr && bias != nullptr && rhs_scales != nullptr) {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Calling kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon with:" <<
                         " N=" << N << ", K=" << K << ", nr=" << nr << ", kr=" << kr << ", sr=" << sr <<
                         ", rhs_native_qs8cx=" << static_cast<void*>(rhs_native_qs8cx) << ", bias=" << static_cast<void*>(bias) <<
                         ", rhs_scales=" << static_cast<void*>(rhs_scales) << ", rhs_packed_qs8cx=" << static_cast<void*>(rhs_packed_qs8cx) <<
                         ", lhs_zero_point=" << params.lhs_zero_point << std::endl;
            kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1,
                                                     N,
                                                     K,
                                                     nr,
                                                     kr,
                                                     sr,
                                                     rhs_native_qs8cx,
                                                     bias,
                                                     rhs_scales,
                                                     rhs_packed_qs8cx,
                                                     0,
                                                     &params);
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Completed kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon" << std::endl;
        } else {
            std::cerr << "MatMulKleidiAIExecutor::Constructor - Skipping i8 packing due to safety check failure:" <<
                         " N=" << N << ", K=" << K << ", rhs_packed_qs8cx=" << static_cast<void*>(rhs_packed_qs8cx) <<
                         ", rhs_native_qs8cx=" << static_cast<void*>(rhs_native_qs8cx) << ", bias=" << static_cast<void*>(bias) <<
                         ", rhs_scales=" << static_cast<void*>(rhs_scales) << std::endl;
        }

        // Create scratchpad to initialize memory for LHS in update()
        scratchPad = context->getScratchPad();
    }
}

bool MatMulKleidiAIExecutor::update(const MemoryArgs& memory) {
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    const auto& wgtDims = weiDesc->getShape().getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    K = wgtDims[1];
    N = wgtDims[0];
    
    std::cerr << "MatMulKleidiAIExecutor::update - Weight dimensions: K=" << K << ", N=" << N << std::endl;

    const auto& outDims = dstDesc->getShape().getStaticDims();
    if (outDims.size() > 2) {
        M = std::accumulate(outDims.begin(), outDims.end() - 1, 1, std::multiplies<size_t>());
    } else {
        M = outDims[0];
    }
    
    std::cerr << "MatMulKleidiAIExecutor::update - Output dimensions: M=" << M << ", outDims.size()=" << outDims.size() << std::endl;
    // Assign LHS memory
    if (useDynamicQuant) {
        std::cerr << "MatMulKleidiAIExecutor::update - Setting up dynamic quantization LHS memory" << std::endl;
        // Skip allocation if any dimension is zero to avoid KAI kernel crashes
        if (M > 0 && K > 0) {
            const size_t lhsPackedSize = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
            std::cerr << "MatMulKleidiAIExecutor::update - LHS packed size=" << lhsPackedSize << " for M=" << M << ", K=" << K << ", mr=" << mr << ", kr=" << kr << ", sr=" << sr << std::endl;
            auto lhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({lhsPackedSize}));
            lhsPackedMem = scratchPad->createScratchPadMem(lhsPackedDesc);
            std::cerr << "MatMulKleidiAIExecutor::update - Created LHS packed memory: " << static_cast<void*>(lhsPackedMem->getData()) << std::endl;
        } else {
            std::cerr << "MatMulKleidiAIExecutor::update - Skipping LHS allocation due to zero dimensions: M=" << M << ", K=" << K << std::endl;
        }
    } else {
        std::cerr << "MatMulKleidiAIExecutor::update - Not using dynamic quantization" << std::endl;
    }
    return true;
}

void MatMulKleidiAIExecutor::execute(const MemoryArgs& memory) {
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto dstMem = memory.at(ARG_DST);
    auto srcDims = normalizeDimsTo2D(srcMem->getDesc().getShape().getDims());
    auto weiDims = weiMem->getDesc().getShape().getDims();
    auto M = srcDims[0];
    auto K = srcDims[1];
    auto N = weiDims[0];
    
    std::cerr << "MatMulKleidiAIExecutor::execute - Dimensions: M=" << M << ", K=" << K << ", N=" << N <<
                 ", useDynamicQuant=" << useDynamicQuant << std::endl;

    const size_t lhs_stride = K * sizeof(float);
    const size_t dst_stride_row = N * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    float* lhs = srcMem->getDataAs<float>();
    float* dst = dstMem->getDataAs<float>();

    size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    std::cerr << "MatMulKleidiAIExecutor::execute - Memory layout: lhs_stride=" << lhs_stride << ", dst_stride_row=" << dst_stride_row <<
                 ", dst_stride_col=" << dst_stride_col << ", n_blocks=" << n_blocks << ", BLOCK_SIZE=" << BLOCK_SIZE <<
                 ", lhs=" << static_cast<void*>(lhs) << ", dst=" << static_cast<void*>(dst) << std::endl;

    if (!useDynamicQuant) {
        float* rhs_packed = static_cast<float*>(rhsPackedMem->getData());

        parallel_for(n_blocks, [&](size_t n_block) {
            size_t n_start = (n_block * BLOCK_SIZE);
            size_t n_end = std::min(n_start + BLOCK_SIZE, N);
            size_t n_block_size = n_end - n_start;
            // Skip execution if any dimension is zero to avoid KAI kernel crashes
            if (M == 0 || n_block_size == 0 || K == 0) {
                std::cerr << "MatMulKleidiAIExecutor::execute - Skipping f32 matmul block due to zero dimensions: M=" << M << ", n_block_size=" << n_block_size << ", K=" << K << std::endl;
                return;
            }
            
            const size_t rhs_packed_offset = ukernel_f32.get_rhs_packed_offset(n_start, K);
            const size_t dst_offset = ukernel_f32.get_dst_offset(0, n_start, dst_stride_row);
            const float* rhs_ptr = (rhs_packed + rhs_packed_offset / sizeof(float));
            float* dst_ptr = (dst + dst_offset / (sizeof(float)));
            
            std::cerr << "MatMulKleidiAIExecutor::execute - f32 matmul block " << n_block << ": n_start=" << n_start << ", n_end=" << n_end << ", n_block_size=" << n_block_size <<
                         ", rhs_packed_offset=" << rhs_packed_offset << ", dst_offset=" << dst_offset <<
                         ", rhs_ptr=" << static_cast<const void*>(rhs_ptr) << ", dst_ptr=" << static_cast<void*>(dst_ptr) << std::endl;
            
            ukernel_f32.run_matmul(M,
                                   n_block_size,
                                   K,
                                   lhs,
                                   lhs_stride,
                                   rhs_ptr,
                                   dst_ptr,
                                   dst_stride_row,
                                   dst_stride_col,
                                   FLOAT_MIN,
                                   FLOAT_MAX);
        });
    } else {
        // Create packed LHS and RHS
        int8_t* lhs_packed_qa8dx = lhsPackedMem->getDataAs<int8_t>();
        int8_t* rhs_packed_qs8cx = rhsPackedMem->getDataAs<int8_t>();

        // Skip LHS packing if any dimension is zero to avoid KAI kernel crashes
        if (M > 0 && K > 0) {
            std::cerr << "MatMulKleidiAIExecutor::execute - Calling kai_run_lhs_quant_pack_qai8dxp_f32 with:" <<
                         " M=" << M << ", K=" << K << ", mr=" << mr << ", kr=" << kr << ", sr=" << sr <<
                         ", lhs=" << static_cast<void*>(lhs) << ", lhs_stride=" << lhs_stride <<
                         ", lhs_packed_qa8dx=" << static_cast<void*>(lhs_packed_qa8dx) << std::endl;
            kai_run_lhs_quant_pack_qai8dxp_f32(M,
                                               K,  // Dimensions
                                               mr,
                                               kr,
                                               sr,
                                               0,                // Packing dimensions
                                               lhs,              // LHS (F32)
                                               lhs_stride,       // LHS stride
                                               lhs_packed_qa8dx  // LHS packed
            );
            std::cerr << "MatMulKleidiAIExecutor::execute - Completed kai_run_lhs_quant_pack_qai8dxp_f32" << std::endl;
        } else {
            std::cerr << "MatMulKleidiAIExecutor::execute - Skipping LHS packing due to zero dimensions: M=" << M << ", K=" << K << std::endl;
        }

        const size_t lhs_packed_offset = ukernel_i8.get_lhs_packed_offset(0, K);
        const void* lhs_ptr = static_cast<const void*>(lhs_packed_qa8dx + lhs_packed_offset);

        parallel_for(n_blocks, [&](size_t n_block) {
            size_t n_start = (n_block * BLOCK_SIZE);
            size_t n_end = std::min(n_start + BLOCK_SIZE, N);
            size_t n_block_size = n_end - n_start;
            // Skip execution if any dimension is zero to avoid KAI kernel crashes
            if (M == 0 || n_block_size == 0 || K == 0) {
                std::cerr << "MatMulKleidiAIExecutor::execute - Skipping i8 matmul block due to zero dimensions: M=" << M << ", n_block_size=" << n_block_size << ", K=" << K << std::endl;
                return;
            }
            
            const size_t rhs_packed_offset = ukernel_i8.get_rhs_packed_offset(n_start, K);
            const size_t dst_offset = ukernel_i8.get_dst_offset(0, n_start, dst_stride_row);
            const void* rhs_ptr = static_cast<const void*>(rhs_packed_qs8cx + rhs_packed_offset);
            float* dst_ptr = (dst + dst_offset / sizeof(float));
            
            std::cerr << "MatMulKleidiAIExecutor::execute - i8 matmul block " << n_block << ": n_start=" << n_start << ", n_end=" << n_end << ", n_block_size=" << n_block_size <<
                         ", rhs_packed_offset=" << rhs_packed_offset << ", dst_offset=" << dst_offset <<
                         ", lhs_ptr=" << lhs_ptr << ", rhs_ptr=" << rhs_ptr << ", dst_ptr=" << static_cast<void*>(dst_ptr) << std::endl;
            
            ukernel_i8.run_matmul(M,
                                  n_block_size,
                                  K,
                                  lhs_ptr,
                                  rhs_ptr,
                                  dst_ptr,
                                  dst_stride_row,
                                  dst_stride_col,
                                  FLOAT_MIN,
                                  FLOAT_MAX);
        });
    }
}

void MatMulKleidiAIExecutor::moveMemToNumaNode(int numaNodeID) {
    if (curNumaNode == numaNodeID) {
        return;
    }
    curNumaNode = numaNodeID;
    mbind_move(packedWeights, numaNodeID);
    if (m_attrs.withBias) {
        mbind_move(m_memoryArgs.at(ARG_BIAS), numaNodeID);
    }
}

}  // namespace intel_cpu
}  // namespace ov
