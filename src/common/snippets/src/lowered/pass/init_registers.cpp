// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_registers.hpp"

#include <memory>

#include "snippets/generator.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/assign_registers.hpp"
#include "snippets/lowered/pass/init_live_ranges.hpp"
#include "snippets/lowered/pass/insert_reg_spills.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/pass/pass_config.hpp"

namespace ov::snippets::lowered::pass {

InitRegisters::InitRegisters(const std::shared_ptr<const Generator>& generator,
                             const std::shared_ptr<PassConfig>& pass_config)
    : Pass(),
      m_reg_manager(generator),
      m_pass_config(pass_config) {}

bool InitRegisters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitRegisters");
    lowered::pass::PassPipeline reg_pipeline(m_pass_config);
    reg_pipeline.register_pass<lowered::pass::InitLiveRanges>(m_reg_manager);
    reg_pipeline.register_pass<lowered::pass::AssignRegisters>(m_reg_manager);
    std::function<bool(const ExpressionPtr&)> needs_reg_spill = needs_reg_spill_default;
    if (m_pass_config && m_pass_config->has_reg_spill_predicate()) {
        const auto reg_spill_predicate = m_pass_config->get_reg_spill_predicate();
        needs_reg_spill = [reg_spill_predicate](const ExpressionPtr& expr) {
            return needs_reg_spill_default(expr) || reg_spill_predicate(expr);
        };
    }
    reg_pipeline.register_pass<lowered::pass::InsertRegSpills>(m_reg_manager, needs_reg_spill);
    reg_pipeline.run(linear_ir);
    return true;
}

}  // namespace ov::snippets::lowered::pass
