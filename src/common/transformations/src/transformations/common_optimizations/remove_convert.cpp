// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_convert.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;

ov::pass::RemoveConvert::RemoveConvert() {
    MATCHER_SCOPE(RemoveConvert);

    auto cvt = pattern::wrap_type<ov::op::v0::Convert>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(m.get_match_root());
        if (!cvt) {
            return false;
        }

        fprintf(stderr, "type: %s\n", cvt->get_convert_element_type().c_type_string().c_str());
        fprintf(stderr, "input type: %s\n", cvt->get_input_element_type(0).c_type_string().c_str());
        if (cvt->get_convert_element_type() != cvt->get_input_element_type(0)) {
            return false;
        }

        // cvt->input_value(0).replace(cvt->output(0));
        auto child = cvt->input_value(0).get_node_shared_ptr();

        child->set_friendly_name(cvt->get_friendly_name());
        copy_runtime_info(cvt, child);
        replace_node(cvt, child);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(cvt, matcher_name);
    this->register_matcher(m, callback);
}
