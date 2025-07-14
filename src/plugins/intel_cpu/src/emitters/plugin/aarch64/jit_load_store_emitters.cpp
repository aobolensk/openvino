// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_load_store_emitters.hpp"

#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include <cstdio>
#include <unordered_set>

using namespace Xbyak_aarch64;

static void trace_load_address(const void* address, size_t offset, size_t load_num, const char* precision) {
    fprintf(stderr,
            "JIT LOAD TRACE: addr=0x%p, offset=%zu, load_num=%zu, precision=%s, effective_addr=0x%p\n",
            address,
            offset,
            load_num,
            precision,
            reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(address) + offset));
    fflush(stderr);
}

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;

static const int ARM64_MAX_OFFSET = 4095;

static inline void setup_safe_ptr(jit_generator* h, const XReg& base, int offset, const XReg& temp_reg, XReg& effective_base, int& effective_offset) {
    if (offset >= 0 && offset <= ARM64_MAX_OFFSET) {
        effective_base = base;
        effective_offset = offset;
    } else {
        h->add_imm(temp_reg, base, offset, h->X_DEFAULT_ADDR);
        effective_base = temp_reg;
        effective_offset = 0;
    }
}

jit_load_emitter::jit_load_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   ov::element::Type src_prc,
                                   ov::element::Type dst_prc,
                                   int load_num,
                                   int byte_offset,
                                   ov::element::Type exec_prc,
                                   emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type),
      name_("unknown"),
      load_num_(load_num),
      byte_offset_(byte_offset),
      prc_(src_prc) {
    OV_CPU_JIT_EMITTER_ASSERT(src_prc == dst_prc, "Unsupported precision pair.");
}

void jit_load_emitter::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported isa.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::load_qbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = XReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    auto dst_s = SReg(out_idxs[0]);
    auto dst_d = DReg(out_idxs[0]);

    size_t aux_idx = 0;
    XReg offset_reg = XReg(0);
    bool needs_offset_reg = (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET);
    
    if (needs_offset_reg) {
        offset_reg = XReg(aux_gpr_idxs[aux_idx++]);
    }

    XReg effective_base = src;
    int effective_offset = byte_offset_;
    if (needs_offset_reg) {
        setup_safe_ptr(h, src, byte_offset_, offset_reg, effective_base, effective_offset);
    }

    switch (load_num_) {
    case 0:
        break;
    case 1:
        h->ldr(dst_s, ptr(effective_base, effective_offset));
        break;
    case 2:
        h->ldr(dst_d, ptr(effective_base, effective_offset));
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[aux_idx]);
        h->ldr(dst_d, ptr(effective_base, effective_offset));
        if (needs_offset_reg) {
            h->add_imm(prc, effective_base, 2 * sizeof(float), h->X_DEFAULT_ADDR);
        } else {
            h->add_imm(prc, src, byte_offset_ + 2 * sizeof(float), h->X_DEFAULT_ADDR);
        }
        h->ld1(dst.s[2], ptr(prc));
        break;
    }
    case 4:
        if (needs_offset_reg) {
            h->ldr(QReg(out_idxs[0]), ptr(effective_base, effective_offset));
        } else {
            h->uni_ldr(dst, src, byte_offset_);
        }
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to load.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::load_dbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = XReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    auto dst_h = HReg(out_idxs[0]);
    auto dst_s = SReg(out_idxs[0]);
    auto dst_d = DReg(out_idxs[0]);

    size_t aux_idx = 0;
    XReg offset_reg = XReg(0);
    bool needs_offset_reg = (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET);
    
    if (needs_offset_reg) {
        offset_reg = XReg(aux_gpr_idxs[aux_idx++]);
    }

    XReg effective_base = src;
    int effective_offset = byte_offset_;
    if (needs_offset_reg) {
        setup_safe_ptr(h, src, byte_offset_, offset_reg, effective_base, effective_offset);
    }

    switch (load_num_) {
    case 0:
        break;
    case 1:
        h->ldr(dst_h, ptr(effective_base, effective_offset));
        break;
    case 2:
        h->ldr(dst_s, ptr(effective_base, effective_offset));
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[aux_idx]);
        h->ldr(dst_s, ptr(effective_base, effective_offset));
        if (needs_offset_reg) {
            h->add_imm(prc, effective_base, 2 * sizeof(uint16_t), h->X_DEFAULT_ADDR);
        } else {
            h->add_imm(prc, src, byte_offset_ + 2 * sizeof(uint16_t), h->X_DEFAULT_ADDR);
        }
        h->ld1(dst.h[2], ptr(prc));
        break;
    }
    case 4:
        h->ldr(dst_d, ptr(effective_base, effective_offset));
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to load.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::load_byte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = XReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    auto dst_b = BReg(out_idxs[0]);
    auto dst_h = HReg(out_idxs[0]);
    auto dst_s = SReg(out_idxs[0]);

    size_t aux_idx = 0;
    XReg offset_reg = XReg(0);
    bool needs_offset_reg = (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET);
    
    if (needs_offset_reg) {
        offset_reg = XReg(aux_gpr_idxs[aux_idx++]);
    }

    XReg effective_base = src;
    int effective_offset = byte_offset_;
    if (needs_offset_reg) {
        setup_safe_ptr(h, src, byte_offset_, offset_reg, effective_base, effective_offset);
    }

    switch (load_num_) {
    case 0:
        break;
    case 1:
        h->ldr(dst_b, ptr(effective_base, effective_offset));
        break;
    case 2:
        h->ldr(dst_h, ptr(effective_base, effective_offset));
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[aux_idx]);
        h->ldr(dst_h, ptr(effective_base, effective_offset));
        if (needs_offset_reg) {
            h->add_imm(prc, effective_base, 2 * sizeof(int8_t), h->X_DEFAULT_ADDR);
        } else {
            h->add_imm(prc, src, byte_offset_ + 2 * sizeof(int8_t), h->X_DEFAULT_ADDR);
        }
        h->ld1(dst.b[2], ptr(prc));
        break;
    }
    case 4:
        h->ldr(dst_s, ptr(effective_base, effective_offset));
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to load.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(
        one_of(prc_, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
        "Unsupported precision.");
    OV_CPU_JIT_EMITTER_ASSERT(load_num_ <= 4, "Unexpected number of elements to load.");

    // Add address tracing using intermediate function call approach
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);
    
    // Prepare arguments for trace function
    Xbyak_aarch64::XReg x0(0);  // address parameter
    Xbyak_aarch64::XReg x1(1);  // offset parameter  
    Xbyak_aarch64::XReg x2(2);  // load_num parameter
    Xbyak_aarch64::XReg x3(3);  // precision string parameter
    
    // Load address from input register
    h->mov(x0, Xbyak_aarch64::XReg(in_idxs[0]));
    
    // Load offset as immediate
    h->mov(x1, static_cast<uint64_t>(byte_offset_));
    
    // Load load_num as immediate
    h->mov(x2, static_cast<uint64_t>(load_num_));
    
    // Load precision string pointer
    const char* precision_str = nullptr;
    switch (prc_) {
    case ov::element::f32:
        precision_str = "f32";
        break;
    case ov::element::i32:
        precision_str = "i32";
        break;
    case ov::element::f16:
        precision_str = "f16";
        break;
    case ov::element::i8:
        precision_str = "i8";
        break;
    case ov::element::u8:
        precision_str = "u8";
        break;
    default:
        precision_str = "unknown";
        break;
    }
    
    h->mov(x3, reinterpret_cast<uintptr_t>(precision_str));
    
    // Call trace function
    Xbyak_aarch64::XReg func_reg(9);
    h->mov(func_reg, reinterpret_cast<uintptr_t>(trace_load_address));
    h->blr(func_reg);
    
    restore_context(exclude);

    switch (prc_) {
    case ov::element::f32:
    case ov::element::i32:
        load_qbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::f16:
        load_dbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::i8:
    case ov::element::u8:
        load_byte<isa>(in_idxs, out_idxs);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", prc_.get_type_name());
    }
}

size_t jit_load_emitter::get_aux_gprs_count() const {
    size_t count = 0;
    
    if (load_num_ == 3) {
        count++;
    }
    
    if (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET) {
        count++;
    }

    return count;
}

jit_store_emitter::jit_store_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     ov::element::Type src_prc,
                                     ov::element::Type dst_prc,
                                     int store_num,
                                     int byte_offset,
                                     [[maybe_unused]] arithmetic_mode mode,
                                     ov::element::Type exec_prc,
                                     emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type),
      name_("unknown"),
      store_num_(store_num),
      byte_offset_(byte_offset),
      prc_(dst_prc) {
    OV_CPU_JIT_EMITTER_ASSERT(src_prc == dst_prc, "Unsupported precision pair.");
}

void jit_store_emitter::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported isa.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::store_qbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto src_s = SReg(in_idxs[0]);
    auto src_d = DReg(in_idxs[0]);
    auto src_q = QReg(in_idxs[0]);
    auto dst = XReg(out_idxs[0]);

    size_t aux_idx = 0;
    XReg offset_reg = XReg(0);
    bool needs_offset_reg = (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET);
    
    if (needs_offset_reg) {
        offset_reg = XReg(aux_gpr_idxs[aux_idx++]);
    }

    XReg effective_base = dst;
    int effective_offset = byte_offset_;
    if (needs_offset_reg) {
        setup_safe_ptr(h, dst, byte_offset_, offset_reg, effective_base, effective_offset);
    }

    switch (store_num_) {
    case 0:
        break;
    case 1:
        h->str(src_s, ptr(effective_base, effective_offset));
        break;
    case 2:
        h->str(src_d, ptr(effective_base, effective_offset));
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[aux_idx]);
        h->str(src_d, ptr(effective_base, effective_offset));
        if (needs_offset_reg) {
            h->add_imm(prc, effective_base, 2 * sizeof(float), h->X_DEFAULT_ADDR);
        } else {
            h->add_imm(prc, dst, byte_offset_ + 2 * sizeof(float), h->X_DEFAULT_ADDR);
        }
        h->st1(src.s[2], ptr(prc));
        break;
    }
    case 4:
        h->str(src_q, ptr(effective_base, effective_offset));
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to store.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::store_dbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto src_h = HReg(in_idxs[0]);
    auto src_s = SReg(in_idxs[0]);
    auto src_d = DReg(in_idxs[0]);
    auto dst = XReg(out_idxs[0]);

    size_t aux_idx = 0;
    XReg offset_reg = XReg(0);
    bool needs_offset_reg = (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET);
    
    if (needs_offset_reg) {
        offset_reg = XReg(aux_gpr_idxs[aux_idx++]);
    }

    XReg effective_base = dst;
    int effective_offset = byte_offset_;
    if (needs_offset_reg) {
        setup_safe_ptr(h, dst, byte_offset_, offset_reg, effective_base, effective_offset);
    }

    switch (store_num_) {
    case 0:
        break;
    case 1:
        h->str(src_h, ptr(effective_base, effective_offset));
        break;
    case 2:
        h->str(src_s, ptr(effective_base, effective_offset));
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[aux_idx]);
        h->str(src_s, ptr(effective_base, effective_offset));
        if (needs_offset_reg) {
            h->add_imm(prc, effective_base, 2 * sizeof(uint16_t), h->X_DEFAULT_ADDR);
        } else {
            h->add_imm(prc, dst, byte_offset_ + 2 * sizeof(uint16_t), h->X_DEFAULT_ADDR);
        }
        h->st1(src.h[2], ptr(prc));
        break;
    }
    case 4:
        h->str(src_d, ptr(effective_base, effective_offset));
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to store.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::store_byte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto src_b = BReg(in_idxs[0]);
    auto src_h = HReg(in_idxs[0]);
    auto src_s = SReg(in_idxs[0]);
    auto dst = XReg(out_idxs[0]);

    size_t aux_idx = 0;
    XReg offset_reg = XReg(0);
    bool needs_offset_reg = (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET);
    
    if (needs_offset_reg) {
        offset_reg = XReg(aux_gpr_idxs[aux_idx++]);
    }

    XReg effective_base = dst;
    int effective_offset = byte_offset_;
    if (needs_offset_reg) {
        setup_safe_ptr(h, dst, byte_offset_, offset_reg, effective_base, effective_offset);
    }

    switch (store_num_) {
    case 0:
        break;
    case 1:
        h->str(src_b, ptr(effective_base, effective_offset));
        break;
    case 2:
        h->str(src_h, ptr(effective_base, effective_offset));
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[aux_idx]);
        h->str(src_h, ptr(effective_base, effective_offset));
        if (needs_offset_reg) {
            h->add_imm(prc, effective_base, 2 * sizeof(int8_t), h->X_DEFAULT_ADDR);
        } else {
            h->add_imm(prc, dst, byte_offset_ + 2 * sizeof(int8_t), h->X_DEFAULT_ADDR);
        }
        h->st1(src.b[2], ptr(prc));
        break;
    }
    case 4:
        h->str(src_s, ptr(effective_base, effective_offset));
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to store.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(
        one_of(prc_, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
        "Unsupported precision.");
    OV_CPU_JIT_EMITTER_ASSERT(store_num_ <= 4, "Unexpected number of elements to store.");

    switch (prc_) {
    case ov::element::f32:
    case ov::element::i32:
        store_qbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::f16:
        store_dbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::i8:
    case ov::element::u8:
        store_byte<isa>(in_idxs, out_idxs);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", prc_.get_type_name());
    }
}

size_t jit_store_emitter::get_aux_gprs_count() const {
    size_t count = 0;
    
    if (store_num_ == 3) {
        count++;
    }
    
    if (byte_offset_ < 0 || byte_offset_ > ARM64_MAX_OFFSET) {
        count++;
    }

    return count;
}

}  // namespace ov::intel_cpu::aarch64
