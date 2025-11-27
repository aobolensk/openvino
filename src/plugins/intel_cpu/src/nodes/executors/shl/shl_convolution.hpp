// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "shl.hpp"

namespace ov::intel_cpu {

class ShlConvExecutor : public Executor {
public:
    ShlConvExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    static bool supports(const ConvConfig& config);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::shl;
    }

private:
    void initParams(const ConvAttrs& attrs, csinn_layout_enum srcLayout);

    ShlSession sess;
    ShlTensor src;
    ShlTensor wei;
    ShlTensor dst;
    ShlTensor bias;
    ShlConvParams params = {};
    MemoryCPtr packedWeights;

    bool with_bias = false;
};
using ShlConvExecutorPtr = std::shared_ptr<ShlConvExecutor>;

}  // namespace ov::intel_cpu
