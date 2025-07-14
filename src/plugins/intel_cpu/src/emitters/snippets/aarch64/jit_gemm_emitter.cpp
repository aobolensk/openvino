// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_gemm_emitter.hpp"

#include "snippets/utils/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_gemm_emitter::jit_gemm_emitter(jit_generator* h,
                                   cpu_isa_t isa,
                                   const ExpressionPtr& expr,
                                   const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    GemmKernelKaiConfig kernel_config;
    m_kernel_executor_kai = kernel_table->register_kernel<GemmKaiKernelExecutor>(expr, kernel_config);
}

std::set<std::vector<element::Type>> jit_gemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    // Note: currently supports only fp32 on arm
    return {{element::f32, element::f32}};
}

void jit_gemm_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input regs, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
}

void jit_gemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    // todo: use optimized reg spill after CVS-162498
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    Xbyak_aarch64::XReg x0(0);
    Xbyak_aarch64::XReg x1(1);
    Xbyak_aarch64::XReg x2(2);
    Xbyak_aarch64::XReg x3(3);
    // EXPERIMENT: Remove ALL manual stack manipulation - let store_context handle it
    // The manual stack operations were likely corrupting the context saved by store_context
    
    const auto& compiled_kernel = get_compiled_kernel_ptr();
    h->mov(x0, compiled_kernel);
    h->mov(x1, Xbyak_aarch64::XReg(in[0]));
    h->mov(x2, Xbyak_aarch64::XReg(in[1]));
    h->mov(x3, Xbyak_aarch64::XReg(out[0]));

    Xbyak_aarch64::XReg func_reg(9);
    const auto& func_ptr = get_execute_function_ptr();
    h->mov(func_reg, func_ptr);
    std::cout << "get_execute_function_ptr(): " << func_ptr << std::endl;
    
    // EXPERIMENT: Add debugging around the call
    printf("JIT: About to call GemmKaiKernelExecutor::execute\n");
    printf("JIT: compiled_kernel=0x%lx, func_ptr=0x%lx\n", compiled_kernel, func_ptr);
    printf("JIT: in[0]=%lu, in[1]=%lu, out[0]=%lu\n", in[0], in[1], out[0]);
    
    // EXPERIMENT: Validate the function pointer one more time before calling
    if (func_ptr == 0) {
        printf("JIT: ERROR - Function pointer is NULL before call!\n");
    }
    
    // EXPERIMENT: Simplified function call without stack manipulation
    // The issue might be that we're corrupting the JIT context
    
    // Make the function call with original simple approach
    h->blr(func_reg);
    
    printf("JIT: Returned from GemmKaiKernelExecutor::execute\n");

    restore_context(exclude);
}

const uintptr_t jit_gemm_emitter::get_compiled_kernel_ptr() const {
    uintptr_t kernel_ptr = reinterpret_cast<const uintptr_t>(m_kernel_executor_kai.get());
    printf("get_compiled_kernel_ptr(): %lu\n", kernel_ptr);
    return kernel_ptr;
}

const uintptr_t jit_gemm_emitter::get_execute_function_ptr() const {
    uintptr_t func_ptr = reinterpret_cast<const uintptr_t>(GemmKaiKernelExecutor::execute);
    printf("get_execute_function_ptr(): %lu\n", func_ptr);
    
    // EXPERIMENT: Validate function pointer
    if (func_ptr == 0) {
        printf("ERROR: Function pointer is NULL!\n");
    }
    
    // EXPERIMENT: Check if function pointer is in valid code range
    if ((func_ptr & 0xffff000000000000UL) == 0xffff000000000000UL) {
        printf("ERROR: Function pointer in kernel space! Likely corrupted.\n");
    }
    
    // EXPERIMENT: Test if we can safely call the function with NULL args (should segfault predictably)
    // This would help distinguish between corrupted function pointer vs corrupted arguments
    printf("Function pointer validation: GemmKaiKernelExecutor::execute = %p\n", 
           (void*)GemmKaiKernelExecutor::execute);
    
    // EXPERIMENT: Test calling with nullptr to see if it's the function pointer or arguments
    printf("Testing function pointer with nullptr arguments...\n");
    static bool test_done = false;
    if (!test_done) {
        test_done = true;
        printf("About to test call GemmKaiKernelExecutor::execute with nullptr args\n");
        try {
            // This should crash predictably if the function pointer is valid
            // If the function pointer is corrupted, it would crash differently
            GemmKaiKernelExecutor::execute(nullptr, nullptr, nullptr, nullptr);
            printf("ERROR: Function call with nullptr should have crashed!\n");
        } catch (...) {
            printf("Function call with nullptr crashed as expected - function pointer is valid\n");
        }
    }
    
    return func_ptr;
}

}  // namespace ov::intel_cpu::aarch64
