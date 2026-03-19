// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/except.hpp"
#include "snippets/generator.hpp"

namespace ov::intel_cpu {

template <typename DerivedGenerator, typename TargetMachine>
class CPUGeneratorBase : public ov::snippets::Generator {
protected:
    explicit CPUGeneratorBase(const std::shared_ptr<TargetMachine>& target) : ov::snippets::Generator(target) {}

    template <typename ConcreteTargetMachine = TargetMachine>
    bool is_debug_segfault_detector_enabled() const {
#ifdef SNIPPETS_DEBUG_CAPS
        const auto cpu_target_machine = std::dynamic_pointer_cast<ConcreteTargetMachine>(target);
        return cpu_target_machine && cpu_target_machine->debug_config.enable_segfault_detector;
#else
        return false;
#endif
    }

public:
    std::shared_ptr<ov::snippets::Generator> clone() const override {
        const auto cpu_target_machine = std::dynamic_pointer_cast<TargetMachine>(target->clone());
        OPENVINO_ASSERT(cpu_target_machine,
                        "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
        return std::make_shared<DerivedGenerator>(cpu_target_machine);
    }
};

}  // namespace ov::intel_cpu
