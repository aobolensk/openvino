// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits.h>
#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "zero_types.hpp"

namespace intel_npu {

struct ArgumentDescriptor {
    ze_graph_argument_properties_3_t info;
    uint32_t idx;
    std::string to_string() const {
        std::stringstream sstream;
        sstream << "dims_count: " << info.dims_count << " - [";
        for (uint32_t i = 0; i < std::min<uint32_t>(info.dims_count, ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE); i ++) {
            sstream << info.dims[i] << ",";
        }
        sstream << "]"
                << ", networkLayout: " << std::to_string(static_cast<size_t>(info.networkLayout))
                << ", deviceLayout: " << std::to_string(static_cast<size_t>(info.deviceLayout));
        return sstream.str();
    }
};

namespace zeroUtils {

#define THROW_ON_FAIL_FOR_LEVELZERO_EXT(step, result, graph_ddi_table_ext)                  \
    {                                                                                       \
        ze_result_t ret = (result);                                                         \
        if (ZE_RESULT_SUCCESS != ret) {                                                     \
            OPENVINO_THROW("L0 ",                                                           \
                           step,                                                            \
                           " result: ",                                                     \
                           ze_result_to_string(ret),                                        \
                           ", code 0x",                                                     \
                           std::hex,                                                        \
                           uint64_t(ret),                                                   \
                           " - ",                                                           \
                           ze_result_to_description(ret),                                   \
                           " . ",                                                           \
                           intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext)); \
        }                                                                                   \
    }

#define THROW_ON_FAIL_FOR_LEVELZERO(step, result)          \
    {                                                      \
        ze_result_t ret = (result);                        \
        if (ZE_RESULT_SUCCESS != ret) {                    \
            OPENVINO_THROW("L0 ",                          \
                           step,                           \
                           " result: ",                    \
                           ze_result_to_string(ret),       \
                           ", code 0x",                    \
                           std::hex,                       \
                           uint64_t(ret),                  \
                           " - ",                          \
                           ze_result_to_description(ret)); \
        }                                                  \
    }

static inline ze_command_queue_priority_t toZeQueuePriority(const ov::hint::Priority& val) {
    switch (val) {
    case ov::hint::Priority::LOW:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    case ov::hint::Priority::MEDIUM:
        return ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    case ov::hint::Priority::HIGH:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    default:
        OPENVINO_THROW("Incorrect queue priority.");
    }
}

static inline ov::element::Type_t toOVElementType(const ze_graph_argument_precision_t zeElementType) {
    switch (zeElementType) {
    case ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_ARGUMENT_PRECISION_DYNAMIC:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN:
        return ov::element::Type_t::boolean;
    case ZE_GRAPH_ARGUMENT_PRECISION_NF4:
        return ov::element::Type_t::nf4;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP8_E4M3:
        return ov::element::Type_t::f8e4m3;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP8_E5M2:
        return ov::element::Type_t::f8e5m2;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP8_E8M0:
        return ov::element::Type_t::f8e8m0;
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return ov::element::Type_t::bf16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return ov::element::Type_t::f16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return ov::element::Type_t::f32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
        return ov::element::Type_t::f64;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
        return ov::element::Type_t::i4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return ov::element::Type_t::i8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return ov::element::Type_t::i16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return ov::element::Type_t::i32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
        return ov::element::Type_t::i64;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT2:
        return ov::element::Type_t::u2;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return ov::element::Type_t::u1;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
        return ov::element::Type_t::u4;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return ov::element::Type_t::u8;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return ov::element::Type_t::u16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
        return ov::element::Type_t::u32;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
        return ov::element::Type_t::u64;
    default:
        return ov::element::Type_t::dynamic;
    }
}

static inline uint32_t findCommandQueueGroupOrdinal(
    ze_device_handle_t device_handle,
    const ze_command_queue_group_property_flags_t& command_queue_group_property) {
    auto log = Logger::global().clone("findCommandQueueGroupOrdinal");

    std::vector<ze_command_queue_group_properties_t> command_group_properties;
    uint32_t command_queue_group_count = 0;

    // Discover all command queue groups
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeDeviceGetCommandQueueGroupProperties",
        zeDeviceGetCommandQueueGroupProperties(device_handle, &command_queue_group_count, nullptr));

    log.debug("zero_utils::findCommandQueueGroupOrdinal - resize command_queue_group_count");
    command_group_properties.resize(command_queue_group_count);

    for (auto& prop : command_group_properties) {
        prop.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        prop.pNext = nullptr;
    }

    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetCommandQueueGroupProperties",
                                zeDeviceGetCommandQueueGroupProperties(device_handle,
                                                                       &command_queue_group_count,
                                                                       command_group_properties.data()));

    for (uint32_t index = 0; index < command_group_properties.size(); ++index) {
        const auto& flags = command_group_properties[index].flags;
        if (flags == command_queue_group_property) {
            return index;
        }
    }

    // if we don't find a group where only the proper flag is enabled then search for a group where that flag is
    // enabled
    for (uint32_t index = 0; index < command_group_properties.size(); ++index) {
        const auto& flags = command_group_properties[index].flags;
        if (flags & command_queue_group_property) {
            return index;
        }
    }

    // if still don't find compute flag, return a warning
    log.warning("Fail to find a command queue group that contains compute flag, it will be set to 0.");
    return 0;
}

static inline std::string getLatestBuildError(ze_graph_dditable_ext_curr_t& _graph_ddi_table_ext) {
    Logger _logger("LevelZeroUtils", Logger::global().level());
    _logger.debug("getLatestBuildError start");

    uint32_t graphDdiExtVersion = _graph_ddi_table_ext.version();
    if (graphDdiExtVersion >= ZE_GRAPH_EXT_VERSION_1_4) {
        // Get log size
        uint32_t size = 0;
        // Null graph handle to get error log
        auto result = _graph_ddi_table_ext.pfnBuildLogGetString(nullptr, &size, nullptr);
        if (ZE_RESULT_SUCCESS != result) {
            // The failure will not break normal execution, only warning here
            _logger.warning("getLatestBuildError Failed to get size of latest error log!");
            return "";
        }

        if (size <= 0) {
            // The failure will not break normal execution, only warning here
            _logger.warning("getLatestBuildError No error log stored in driver when error "
                            "detected, may not be compiler issue!");
            return "";
        }

        // Get log content
        std::string logContent{};
        logContent.resize(size);
        result = _graph_ddi_table_ext.pfnBuildLogGetString(nullptr, &size, const_cast<char*>(logContent.data()));
        if (ZE_RESULT_SUCCESS != result) {
            // The failure will not break normal execution, only warning here
            _logger.warning("getLatestBuildError size of latest error log > 0, failed to get "
                            "content of latest error log!");
            return "";
        }
        _logger.debug("getLatestBuildError end");
        return logContent;
    } else {
        return "";
    }
}

static inline bool memory_was_allocated_in_the_same_l0_context(ze_context_handle_t hContext, const void* ptr) {
    ze_memory_allocation_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    auto res = intel_npu::zeMemGetAllocProperties(hContext, ptr, &desc, nullptr);
    if (res == ZE_RESULT_SUCCESS) {
        if (desc.id) {
            if ((desc.type == ZE_MEMORY_TYPE_HOST) || (desc.type == ZE_MEMORY_TYPE_DEVICE) ||
                (desc.type == ZE_MEMORY_TYPE_SHARED)) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace zeroUtils
}  // namespace intel_npu
