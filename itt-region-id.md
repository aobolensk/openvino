# ITT Region ID-Based Event Chaining for Intel CPU Plugin

## Overview

This implementation introduces ITT (Intel Tracing Technology) region ID-based event chaining for the Intel CPU plugin to enable continuous task tracking from model compilation through inference execution in VTune profiler.

## Implementation Details

### Region ID Generation

- **Method**: Atomic counter-based unique ID generation per compiled model
- **Location**: `Plugin::compile_model()` in `src/plugins/intel_cpu/src/plugin.cpp:183`
- **Algorithm**: 
  ```cpp
  static std::atomic<uint64_t> region_counter{0};
  const uint64_t region_id = region_counter.fetch_add(1);
  ```
- **Range**: 0-based incrementing counter for compilation-time models
- **Cached Models**: 1,000,000+ range to distinguish from compilation-time IDs

### Region ID Propagation

The region ID flows through the following components:

1. **Plugin::compile_model()** → generates unique region_id
2. **CompiledModel** → stores region_id as member `m_region_id`
3. **CompiledModelHolder** → provides `region_id()` getter method
4. **SyncInferRequest** → accesses region_id via `m_compiled_model.region_id()`

### ITT Event Chaining

#### Compilation Chain
**File**: `src/plugins/intel_cpu/src/plugin.cpp`

- **Chain Start**: `OV_ITT_TASK_CHAIN(compile_chain, itt::domains::intel_cpu, region_prefix, "compile_model")`
- **Transformations**: `OV_ITT_TASK_NEXT(compile_chain, "transformations")`
- **Finalization**: `OV_ITT_TASK_NEXT(compile_chain, "finalize")`

#### Inference Chain
**File**: `src/plugins/intel_cpu/src/infer_request.cpp`

- **Inference Start**: `OV_ITT_TASK_CHAIN(infer_chain, itt::domains::intel_cpu, region_prefix, "infer")`
- **Graph Execution**: `OV_ITT_TASK_NEXT(infer_chain, "graph_infer")`

### Task Naming Convention

Tasks are named using the pattern: `region_{region_id}_{phase_name}`

Examples:
- `region_0_compile_model`
- `region_0_transformations` 
- `region_0_finalize`
- `region_0_infer`
- `region_0_graph_infer`

## VTune Integration

### Expected VTune Output

When profiling with VTune, users will see:
1. **Contiguous Timeline**: All tasks for a single model appear as connected events
2. **Clear Model Separation**: Different models (region_0, region_1, etc.) are visually distinct
3. **Phase Correlation**: Easy to correlate compilation overhead with inference performance

### Example VTune Timeline

```
Timeline View:
region_0_compile_model    |████████████|
region_0_transformations           |██████|
region_0_finalize                        |███|
region_0_infer                              |█|
region_0_graph_infer                         |█|

region_1_compile_model    |████████████|
region_1_transformations           |██████|
region_1_finalize                        |███|
region_1_infer                              |█|
region_1_graph_infer                         |█|
```

### CLI Profiling Command

To capture ITT events with VTune command line:
```bash
vtune -collect threading -knob enable-stack-collection=false -app-working-dir . ./your_openvino_app
```

## Backward Compatibility

- **Build Compatibility**: ITT macros are conditionally compiled - no impact when ITT is disabled
- **Runtime Compatibility**: Zero overhead when profiling is not active
- **API Compatibility**: No changes to public OpenVINO APIs

## Performance Impact

- **Compilation**: Negligible overhead (~1 atomic increment per model)
- **Inference**: Minimal overhead (string construction only when ITT is enabled)
- **Memory**: +8 bytes per CompiledModel for region_id storage

## Future Enhancements

- **Cross-Plugin Chaining**: Extend region_id concept to other OpenVINO plugins
- **Request-Level Tracking**: Add region_id to individual InferRequest instances
- **Dynamic Batching**: Correlate batch processing phases using region_id
- **Graph Optimization**: Chain individual transformation passes within compilation