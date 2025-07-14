// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_copy_b.hpp"

#include "openvino/core/parallel.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

namespace ov::intel_cpu::aarch64 {

GemmCopyBKernelKaiConfig::GemmCopyBKernelKaiConfig(const size_t n_blk_size)
    : m_static_params(std::make_shared<StaticParams>(n_blk_size)) {
    OPENVINO_ASSERT(n_blk_size != 0, "n_blk_size can not be zero in GemmCopyBKernelKaiConfig.");
    m_hash = compute_hash();
}

bool GemmCopyBKernelKaiConfig::operator==(const GemmCopyBKernelKaiConfig& rhs) const {
    return m_N == rhs.m_N && m_K == rhs.m_K && m_hash == rhs.m_hash &&
           (m_static_params == rhs.m_static_params || *m_static_params == *(rhs.m_static_params));
}

bool GemmCopyBKernelKaiConfig::is_completed() const {
    return !ov::snippets::utils::one_of(0ul, m_N, m_K) || is_empty();
}

bool GemmCopyBKernelKaiConfig::is_empty() const {
    return everyone_is(0ul, m_N, m_K);
}

#ifdef SNIPPETS_DEBUG_CAPS
#    define PRINT(X) ss << #X << " = " << (X) << "\n"
std::string GemmCopyBKernelKaiConfig::to_string() const {
    std::stringstream ss;
    ss << m_static_params->to_string() << "\n";
    PRINT(m_N);
    PRINT(m_K);
    return ss.str();
}
#    undef PRINT
#endif

void GemmCopyBKernelKaiConfig::update(size_t N, size_t K) {
    // If one of the dims is zero, it means that GemmCopyB won't be executed (in Loop with work_amount = 0, for
    // example) To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (ov::snippets::utils::one_of(0ul, N, K)) {
        m_N = 0;
        m_K = 0;
    } else {
        m_N = N;
        m_K = K;
    }
    m_hash = compute_hash();
}

size_t GemmCopyBKernelKaiConfig::compute_hash() const {
    size_t seed = m_static_params->hash;
    seed = dnnl::impl::hash_combine(seed, m_N);
    seed = dnnl::impl::hash_combine(seed, m_K);
    return seed;
}

GemmCopyBKernelKaiConfig::StaticParams::StaticParams(size_t wei_n_blk)
    : wei_N_blk(wei_n_blk),
      hash(init_hash(wei_N_blk)) {}

bool GemmCopyBKernelKaiConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return wei_N_blk == rhs.wei_N_blk && hash == rhs.hash;
}

size_t GemmCopyBKernelKaiConfig::StaticParams::init_hash(size_t wei_n_blk) {
    size_t seed = 0;
    seed = dnnl::impl::hash_combine(seed, wei_n_blk);
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
#    define PRINT(X) ss << #X << " = " << (X) << "\n"
std::string GemmCopyBKernelKaiConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(wei_N_blk);
    return ss.str();
}
#    undef PRINT
#endif

GemmCopyBKaiKernelExecutor::GemmCopyBKaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : snippets::KernelExecutor<GemmCopyBKernelKaiConfig, GemmCopyBCompiledKernel>(std::move(config)) {}


void GemmCopyBKaiKernelExecutor::update_kernel(const GemmCopyBKernelKaiConfig& config,
                                               std::shared_ptr<GemmCopyBCompiledKernel>& kernel) const {
    if (kernel == nullptr) {
        printf("DEBUG: GemmCopyBKaiKernelExecutor::update_kernel - creating new kernel\n");
        kernel = std::make_shared<GemmCopyBCompiledKernel>();
        const auto& n_blk_size = config.get_n_blk_size();
        printf("DEBUG: Allocating bias_buffer of %zu bytes (n_blk_size=%zu)\n",
               n_blk_size * sizeof(float), n_blk_size);
        kernel->bias_buffer->resize(n_blk_size * sizeof(float), 0);
    } else {
        printf("DEBUG: GemmCopyBKaiKernelExecutor::update_kernel - kernel already exists (%p)\n",
               kernel.get());
    }
}

void GemmCopyBKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                               const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                               GemmCopyBKernelKaiConfig& config) const {
    const auto& in0_shape = snippets::utils::get_planar_vdims(expr->get_input_port(0));
    int64_t N = *in0_shape.rbegin();
    int64_t K = *++in0_shape.rbegin();
    printf("DEBUG: GemmCopyBKaiKernelExecutor::update_config - Incoming dims: K=%ld, N=%ld\n",
           static_cast<long>(K), static_cast<long>(N));
    config.update(N, K);
    printf("DEBUG: GemmCopyBKaiKernelExecutor::update_config - Config updated. "
           "is_completed=%d, is_empty=%d\n",
           config.is_completed(), config.is_empty());
}

void GemmCopyBKaiKernelExecutor::execute(const GemmCopyBKaiKernelExecutor* executor,
                                         void* in0,
                                         void* out0) {
    printf("DEBUG: GemmCopyBKaiKernelExecutor::execute - ENTRY\n");
    // return;
    printf("DEBUG: executor=%p, in0=%p, out0=%p\n", executor, in0, out0);

    // EXPERIMENT 1: Pointer aliasing check
    if (in0 == out0) {
        printf("DEBUG: CRITICAL - in0 and out0 are aliased! Packing will overwrite the source.\n");
    }

    // EXPERIMENT 2: Kernel / stack address warning (0xffff upper bits on AArch64)
    auto warn_if_stack = [](uintptr_t addr, const char* name) {
        if ((addr & 0xffff000000000000UL) == 0xffff000000000000UL) {
            printf("DEBUG: WARNING - %s (0x%lx) appears to be in a guard/stack area\n",
                   name, static_cast<unsigned long>(addr));
        }
    };
    warn_if_stack(reinterpret_cast<uintptr_t>(in0), "in0");
    warn_if_stack(reinterpret_cast<uintptr_t>(out0), "out0");

    // Validate executor
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    const auto& config = static_cast<const GemmCopyBKernelKaiConfig&>(executor->get_config());
    const auto& kernel  = executor->get_kernel();
    OPENVINO_ASSERT(kernel, "has nullptr kernel in GemmCopyBKaiKernelExecutor");
    OPENVINO_ASSERT(kernel->copy_b_ukernel, "Invalid ukernel for GemmCopyBKaiKernelExecutor");
    const auto& ukernel = kernel->copy_b_ukernel;

    // Extract parameters
    const size_t K           = config.get_K();
    const size_t N           = config.get_N();
    const size_t n_blk_size  = config.get_n_blk_size();
    const size_t nr          = ukernel->get_nr();
    const size_t kr          = ukernel->get_kr();
    const size_t sr          = ukernel->get_sr();
    const size_t rhsBlkSize  = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n_blk_size, K);
    const size_t n_blocks    = ov::snippets::utils::div_up(N, n_blk_size);
    const size_t total_pack  = rhsBlkSize * n_blocks;

    printf("DEBUG: Config params - K=%zu, N=%zu, n_blk_size=%zu\n", K, N, n_blk_size);
    printf("DEBUG: ukernel params - nr=%zu, kr=%zu, sr=%zu\n", nr, kr, sr);
    printf("DEBUG: Derived params - rhsBlkSize=%zu, n_blocks=%zu, total_pack=%zu bytes\n",
           rhsBlkSize, n_blocks, total_pack);

    // EXPERIMENT 3: Basic sanity checks
    if (config.is_empty()) {
        printf("DEBUG: WARNING - Config marked as empty (K==0 || N==0). Nothing to do.\n");
        return;
    }
    if (rhsBlkSize == 0) {
        printf("DEBUG: ERROR - rhsBlkSize is zero. Aborting.\n");
        return;
    }

    // Static print once for allocation suggestion
    static bool alloc_printed = false;
    if (!alloc_printed) {
        printf("DEBUG: SUGGESTED - Ensure out0 buffer is at least %zu bytes\n", total_pack);
        alloc_printed = true;
    }

    // Main repacking loop
    size_t dst_offset = 0;
    for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
        printf("DEBUG: === BLOCK %zu START ===\n", n_block);

        const size_t n_start = n_block * n_blk_size;
        const size_t n_end   = std::min(n_start + n_blk_size, N);
        const size_t n_step  = n_end - n_start;

        printf("DEBUG: n_start=%zu, n_end=%zu, n_step=%zu\n", n_start, n_end, n_step);

        // Compute pointers
        int8_t* src_ptr = static_cast<int8_t*>(in0) + n_start * sizeof(float);
        int8_t* dst_ptr = static_cast<int8_t*>(out0) + dst_offset;

        printf("DEBUG: src_ptr=%p, dst_ptr=%p, dst_offset=%zu\n",
               src_ptr, dst_ptr, dst_offset);
        
        // INVESTIGATION: Track the specific address that causes aliasing
        if (dst_ptr == reinterpret_cast<void*>(0xffff9043c400)) {
            printf("INVESTIGATION: Found the aliasing address 0xffff9043c400 in GemmCopyB\n");
            printf("INVESTIGATION: This buffer will be reused as GEMM output, causing corruption\n");
        }

        // EXPERIMENT 4: Check overlap between src read and dst write
        uintptr_t src_start = reinterpret_cast<uintptr_t>(src_ptr);
        uintptr_t src_end   = src_start + (n_step * K * sizeof(float));
        uintptr_t dst_start = reinterpret_cast<uintptr_t>(dst_ptr);
        uintptr_t dst_end   = dst_start + rhsBlkSize;

        printf("DEBUG: Memory region overlap check:\n");
        printf("DEBUG:   SRC read: [0x%lx, 0x%lx)\n", src_start, src_end);
        printf("DEBUG:   DST write: [0x%lx, 0x%lx)\n", dst_start, dst_end);

        bool overlap = !(src_end <= dst_start || dst_end <= src_start);
        if (overlap) {
            printf("DEBUG: ERROR - Memory regions OVERLAP! Packed output will overwrite source.\n");
        }

        // EXPERIMENT 5: Bounds check for dst_offset
        if (dst_offset + rhsBlkSize > total_pack) {
            printf("DEBUG: ERROR - dst_offset (%zu) exceeds total_pack (%zu)\n",
                   dst_offset + rhsBlkSize, total_pack);
            assert(false && "dst_offset exceeds allocated out0 size");
        }

        printf("DEBUG: Calling kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon...\n");
        fflush(stdout);

        // Optionally swap to memset for diagnostic
        static int test_mode = 0; // 0 = normal, 1 = memset test
        if (test_mode == 1) {
            printf("DEBUG: MEMSET TEST - zeroing dst_ptr region (%zu bytes)\n", rhsBlkSize);
            // std::memset(dst_ptr, 0xCA, rhsBlkSize);
        } else {
            try {
                kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(/*batch=*/1,
                                                                 n_step,
                                                                 K,
                                                                 nr,
                                                                 kr,
                                                                 sr,
                                                                 N * sizeof(float),          // RHS stride
                                                                 src_ptr,                    // RHS
                                                                 kernel->bias_buffer->data(),// Bias
                                                                 nullptr,                    // Scale
                                                                 dst_ptr,                    // RHS packed
                                                                 0,                          // flags
                                                                 nullptr);
            } catch (const std::exception& e) {
                printf("DEBUG: Exception caught in ukernel: %s\n", e.what());
                throw;
            } catch (...) {
                printf("DEBUG: Unknown exception caught in ukernel\n");
                throw;
            }
        }

        printf("DEBUG: Block %zu repack completed\n", n_block);
        
        // EXPERIMENT 6: Check for stack corruption after kai_run
        volatile uintptr_t stack_canary = 0xDEADBEEFCAFEBABE;
        if (stack_canary != 0xDEADBEEFCAFEBABE) {
            printf("DEBUG: ERROR - Stack canary corrupted! Buffer overflow detected.\n");
        }
        
        // EXPERIMENT 7: Check if function pointers are still valid
        if (executor == nullptr) {
            printf("DEBUG: ERROR - executor pointer corrupted after kai_run\n");
        }
        
        // EXPERIMENT 8: Check if the kernel object is still valid
        if (!kernel || !kernel->copy_b_ukernel) {
            printf("DEBUG: ERROR - kernel object corrupted after kai_run\n");
        }
        
        // EXPERIMENT 9: Check if dst_ptr is still in valid range
        if ((reinterpret_cast<uintptr_t>(dst_ptr) & 0xffff000000000000UL) == 0xffff000000000000UL) {
            printf("DEBUG: ERROR - dst_ptr corrupted to kernel space after kai_run\n");
        }
        
        // EXPERIMENT 10: Test writing to a location near the stack to detect corruption
        volatile int stack_test = 12345;
        if (stack_test != 12345) {
            printf("DEBUG: ERROR - Stack corruption detected near current frame\n");
        }
        
        printf("DEBUG: === BLOCK %zu END ===\n", n_block);

        dst_offset += rhsBlkSize;
    }

    // EXPERIMENT 11: Final corruption check before exit
    printf("DEBUG: Final corruption check before exit\n");
    volatile uintptr_t final_canary = 0xDEADBEEFCAFEBABE;
    if (final_canary != 0xDEADBEEFCAFEBABE) {
        printf("DEBUG: ERROR - Final stack canary corrupted!\n");
    }
    
    if (executor == nullptr) {
        printf("DEBUG: ERROR - executor corrupted before exit\n");
    }
    
    printf("DEBUG: GemmCopyBKaiKernelExecutor::execute - EXIT\n");
}

}  // namespace ov::intel_cpu::aarch64
