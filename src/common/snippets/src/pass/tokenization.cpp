// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/tokenization.hpp"

#include <cstdint>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/extract_reshapes_from_mha.hpp"
#include "snippets/pass/fc_tokenization.hpp"
#include "snippets/pass/gated_mlp_tokenization.hpp"
#include "snippets/pass/gn_tokenization.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/mlp_seq_tokenization.hpp"
#include "snippets/pass/transform_convert.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/op/convert_saturation.hpp"

namespace ov::snippets::pass {

void SetSnippetsNodeType(const std::shared_ptr<Node>& node, SnippetsNodeType nodeType) {
    auto& rt = node->get_rt_info();
    rt["SnippetsNodeType"] = nodeType;
}

void SetSnippetsSubgraphType(const std::shared_ptr<op::Subgraph>& node, SnippetsSubgraphType nodeType) {
    if (node) {
        auto& rt = node->get_rt_info();
        rt["SnippetsSubgraphType"] = nodeType;
    }
}

SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node>& node) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::GetSnippetsNodeType")
    const auto& rt = node->get_rt_info();
    const auto rinfo = rt.find("SnippetsNodeType");
    if (rinfo == rt.end()) {
        return SnippetsNodeType::NotSet;
    }
    return rinfo->second.as<SnippetsNodeType>();
}

SnippetsSubgraphType GetSnippetsSubgraphType(const std::shared_ptr<const op::Subgraph>& node) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::GetSnippetsSubgraphType")
    if (!node) {
        return SnippetsSubgraphType::NotSet;
    }
    const auto& rt = node->get_rt_info();
    const auto rinfo = rt.find("SnippetsSubgraphType");
    if (rinfo == rt.end()) {
        return SnippetsSubgraphType::NotSet;
    }
    return rinfo->second.as<SnippetsSubgraphType>();
}

void SetTopologicalOrder(const std::shared_ptr<Node>& node, int64_t order) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetTopologicalOrder")
    auto& rt = node->get_rt_info();
    rt["TopologicalOrder"] = order;
}

int64_t GetTopologicalOrder(const std::shared_ptr<const Node>& node) {
    const auto& rt = node->get_rt_info();
    const auto rinfo = rt.find("TopologicalOrder");
    if (rinfo == rt.end()) {
        OPENVINO_THROW("Topological order is required, but not set.");
    }
    return rinfo->second.as<int64_t>();
}

bool EnumerateNodes::run_on_model(const std::shared_ptr<ov::Model>& m) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::EnumerateNodes")
    int64_t order = 0;
    // Todo: We don't really have to set order for every node, just for subgraph parents and children would be enough
    for (auto& node : m->get_ordered_ops()) {
        SetTopologicalOrder(node, order++);
    }
    return true;
}

bool SnippetsTokenization::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(SnippetsTokenization);
    ov::pass::Manager manager(get_pass_config(), "Snippets:Tokenization");
    manager.set_per_pass_validation(false);

    manager.register_pass<EnumerateNodes>();
    manager.register_pass<ExtractReshapesFromMHA>();
    // The following passes mustn't be registered in GraphRewrite with other tokenization passes because of 2 reasons:
    // 1. They have higher priority than other tokenization passes
    // 2. They change the nodes after the matched root node
    manager.register_pass<TokenizeMHASnippets>(m_config);
    manager.register_pass<TokenizeGatedMLPSnippets>(m_config);
    manager.register_pass<TokenizeMLPSeqSnippets>(m_config);

    auto tokenization_passes = manager.register_pass<ov::pass::GraphRewrite>();
    tokenization_passes->add_matcher<TokenizeGNSnippets>();
    tokenization_passes->add_matcher<TokenizeFCSnippets>(m_config);
    tokenization_passes->add_matcher<TokenizeSnippets>(m_config);

    manager.register_pass<CommonOptimizations>(m_config);
    
    std::cerr << "[SnippetsTokenization] About to run passes including CommonOptimizations..." << std::endl;
    manager.run_passes(m);
    std::cerr << "[SnippetsTokenization] Finished running all passes." << std::endl;

    // Safety net: ensure Convert inside subgraph bodies are specialized.
    // In some environments the CommonOptimizations matcher callback may not trigger,
    // leaving plain Convert ops inside Subgraph bodies, which later fail lowering
    // on targets that don't register a generic Convert emitter. Here we explicitly
    // run TransformConvertToConvertTruncation on each Subgraph body.
    {
        ov::pass::Manager body_fix_mgr(get_pass_config(), "Snippets:BodyConvertSpecialization");
        body_fix_mgr.register_pass<ov::snippets::pass::TransformConvertToConvertTruncation>();
        int fixed_bodies = 0;
        size_t total_converts_before = 0;
        size_t total_converts_after = 0;
        for (const auto& node : m->get_ordered_ops()) {
            if (auto sg = ov::as_type_ptr<ov::snippets::op::Subgraph>(node)) {
                size_t converts_before = 0;
                std::vector<std::string> before_names;
                for (const auto& n : sg->body_ptr()->get_ops()) {
                    if (ov::is_type<ov::op::v0::Convert>(n) &&
                        !ov::is_type_any_of<ov::snippets::op::ConvertTruncation, ov::snippets::op::ConvertSaturation>(n)) {
                        ++converts_before;
                        before_names.push_back(n->get_friendly_name());
                    }
                }
                total_converts_before += converts_before;

                body_fix_mgr.run_passes(sg->body_ptr());
                ++fixed_bodies;

                size_t converts_after = 0;
                std::vector<std::string> after_names;
                for (const auto& n : sg->body_ptr()->get_ops()) {
                    if (ov::is_type<ov::op::v0::Convert>(n) &&
                        !ov::is_type_any_of<ov::snippets::op::ConvertTruncation, ov::snippets::op::ConvertSaturation>(n)) {
                        ++converts_after;
                        after_names.push_back(n->get_friendly_name());
                    }
                }
                total_converts_after += converts_after;
                if (converts_before > 0 || converts_after > 0) {
                    std::cerr << "[SnippetsTokenization][Subgraph '" << sg->get_friendly_name() << "'] plain Convert before="
                              << converts_before << ", after=" << converts_after << std::endl;
                    if (!before_names.empty()) {
                        std::cerr << "  before names: ";
                        for (size_t i = 0; i < before_names.size(); ++i) {
                            if (i) std::cerr << ", ";
                            std::cerr << before_names[i];
                        }
                        std::cerr << std::endl;
                    }
                    if (!after_names.empty()) {
                        std::cerr << "  after names: ";
                        for (size_t i = 0; i < after_names.size(); ++i) {
                            if (i) std::cerr << ", ";
                            std::cerr << after_names[i];
                        }
                        std::cerr << std::endl;
                    }
                }
            }
        }
        if (fixed_bodies > 0) {
            std::cerr << "[SnippetsTokenization] Applied Convert->ConvertTruncation specialization to "
                      << fixed_bodies << " subgraph bodies" << std::endl;
            std::cerr << "[SnippetsTokenization] Remaining plain Convert inside subgraph bodies: before="
                      << total_converts_before << ", after=" << total_converts_after << std::endl;
        }
    }
    
    // Debug: count and list subgraphs after tokenization
    int subgraph_count = 0;
    for (const auto& node : m->get_ordered_ops()) {
        if (auto subgraph = ov::as_type_ptr<ov::snippets::op::Subgraph>(node)) {
            subgraph_count++;
            std::cerr << "[SnippetsTokenization] Found subgraph: " << subgraph->get_friendly_name() 
                      << " with " << subgraph->body_ptr()->get_ops().size() << " body ops" << std::endl;
        }
    }
    std::cerr << "[SnippetsTokenization] Total subgraphs created: " << subgraph_count << std::endl;

    // Returning value is false because pass::Manager always apply Validation pass if function was changed.
    // But we don't need to validate the model
    return false;
}

}  // namespace ov::snippets::pass
