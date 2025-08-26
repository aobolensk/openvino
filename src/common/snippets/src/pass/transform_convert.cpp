// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/transform_convert.hpp"

#include <memory>
#include <iostream>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"

ov::snippets::pass::TransformConvertToConvertTruncation::TransformConvertToConvertTruncation() {
    MATCHER_SCOPE(TransformConvertToConvertTruncation);
    auto convert = std::make_shared<ov::pass::pattern::op::Label>(
        ov::pass::pattern::any_input(),
        [](const std::shared_ptr<const Node>& n) {
            // Debug dump for every node checked by this predicate
            try {
                std::cerr << "[TransformConvertToConvertTruncation] Checking node: '"
                          << n->get_friendly_name() << "'"
                          << " type= " << n->get_type_info().name
                          << " inputs=" << n->get_input_size()
                          << " outputs=" << n->get_output_size();
                // Dump basic input/output element types when available
                if (n->get_input_size() > 0) {
                    std::cerr << " in0_et= " << n->get_input_element_type(0);
                }
                if (n->get_output_size() > 0) {
                    std::cerr << " out0_et= " << n->get_output_element_type(0);
                }
                // If it is a Convert op, also dump the destination precision
                if (auto cnv = ov::as_type_ptr<const ov::op::v0::Convert>(n)) {
                    std::cerr << " dst_et= " << cnv->get_destination_type();
                }
                std::cerr << std::endl;
            } catch (...) {
                std::cerr << "[TransformConvertToConvertTruncation] (warn) exception while dumping node info" << std::endl;
            }
            const bool is_convert = ov::is_type<ov::op::v0::Convert>(n);
            const bool is_already_specialized = ov::is_type_any_of<op::ConvertTruncation, op::ConvertSaturation>(n);
            std::cerr << "[TransformConvertToConvertTruncation]  -> predicate result for '"
                      << n->get_friendly_name() << "' : "
                      << (is_convert && !is_already_specialized ? "MATCH" : "SKIP")
                      << std::endl;
            return is_convert && !is_already_specialized;
        });

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::op::v0::Convert>(), matcher_name),
        [](ov::pass::pattern::Matcher& m) {
            OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform,
                               "Snippets::op::TransformConvertToConvertTruncation")
            const auto root = m.get_match_root();
            const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(root);
            OPENVINO_ASSERT(convert, "Convert op is invalid");
            auto convert_truncation = std::make_shared<op::ConvertTruncation>(convert->get_input_source_output(0),
                                                                              convert->get_destination_type());
            convert_truncation->set_friendly_name(convert->get_friendly_name());
            ov::copy_runtime_info(convert, convert_truncation);
            std::cerr << "[TransformConvertToConvertTruncation] Replacing Convert op with friendly name: "
                      << convert->get_friendly_name() << std::endl;
            ov::replace_node(convert, convert_truncation);

            return true;
        });
}
