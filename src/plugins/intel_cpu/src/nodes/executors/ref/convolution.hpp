// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(OPENVINO_ARCH_RISCV64)

#    include <cstddef>
#    include <cstdint>
#    include <memory>
#    include <vector>

#    include "cpu_memory.h"
#    include "memory_desc/cpu_memory_desc.h"
#    include "nodes/executors/convolution_config.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

class RefConvolutionExecutor : public Executor {
public:
    RefConvolutionExecutor(ConvAttrs attrs, const MemoryArgs& /*memory*/, ExecutorContext::CPtr /*context*/)
        : m_attrs(std::move(attrs)) {}

    virtual ~RefConvolutionExecutor();

    bool update(const MemoryArgs& memory) override;

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override;

protected:
    // VTable anchor
    virtual void anchor();

private:
    ConvAttrs m_attrs;

    static bool isNspc(const MemoryDescPtr& desc);
};

// Factory function to avoid cross-TU ctor dependency
ExecutorPtr create_ref_convolution_executor(const ConvAttrs& attrs,
                                            const MemoryArgs& memory,
                                            const ExecutorContext::CPtr& context);

}  // namespace ov::intel_cpu

#endif  // OPENVINO_ARCH_RISCV64
