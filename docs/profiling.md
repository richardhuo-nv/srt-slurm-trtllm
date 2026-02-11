# Profiling

srtctl supports two profiling backends for performance analysis: **Torch Profiler** and **NVIDIA Nsight Systems (nsys)**.

## Table of Contents

- [Quick Start](#quick-start)
- [Profiling Modes](#profiling-modes)
- [Configuration Options](#configuration-options)
  - [Top-level profiling section](#top-level-profiling-section)
  - [Parameters](#parameters)
- [Constraints](#constraints)
- [How It Works](#how-it-works)
- [Example Configurations](#example-configurations)
- [Output Files](#output-files)
  - [Viewing Results](#viewing-results)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Add a `profiling` section to your job YAML:

```yaml
# must set benchmark type to "manual"
benchmark:
  type: "manual"

# For disaggregated mode (prefill_nodes + decode_nodes)
profiling:
  type: "torch" # or "nsys"
  isl: 1024
  osl: 128
  concurrency: 24
  prefill:
    start_step: 0
    stop_step: 50
  decode:
    start_step: 0
    stop_step: 50
# For aggregated mode (agg_nodes)
# profiling:
#   type: "torch"
#   isl: 1024
#   osl: 128
#   concurrency: 24
#   aggregated:
#     start_step: 0
#     stop_step: 50
```

## Profiling Modes

| Mode    | Description                                                      | Output                                         |
| ------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| `none`  | Default. No profiling, uses `dynamo.sglang` for serving          | -                                              |
| `torch` | PyTorch Profiler. Good for Python-level and CUDA kernel analysis | `/logs/profiles/{mode}/` (Chrome trace format) |
| `nsys`  | NVIDIA Nsight Systems. Low-overhead GPU profiling                | `/logs/profiles/{mode}/` (`*.nsys-rep`)        |

## Configuration Options

### Top-level `profiling` section

```yaml
profiling:
  type: "torch" # Required: "none", "torch", or "nsys"

  # Traffic generator parameters (required when profiling is enabled)
  isl: 1024 # Input sequence length
  osl: 128 # Output sequence length
  concurrency: 24 # Batch size for profiling workload

  # Disaggregated mode: must set both prefill and decode sections
  prefill:
    start_step: 0 # Step to start profiling for prefill workers
    stop_step: 50 # Step to stop profiling for prefill workers
  decode:
    start_step: 0 # Step to start profiling for decode workers
    stop_step: 50 # Step to stop profiling for decode workers


  # Aggregated mode: must set aggregated section (and must NOT set prefill/decode)
  # aggregated:
  #   start_step: 0   # Step to start profiling for aggregated workers
  #   stop_step: 50   # Step to stop profiling for aggregated workers
```

Traffic generator parameters (`isl`, `osl`, `concurrency`) are shared across all phases. Per-phase `start_step`/`stop_step` allow different profiling windows for prefill vs decode workers.

### Parameters

| Parameter               | Description                                   | Default  |
| ----------------------- | --------------------------------------------- | -------- |
| `isl`                   | Input sequence length for profiling requests  | Required |
| `osl`                   | Output sequence length for profiling requests | Required |
| `concurrency`           | Number of concurrent requests (batch size)    | Required |
| `prefill.start_step`    | Step number to begin prefill profiling        | `0`      |
| `prefill.stop_step`     | Step number to end prefill profiling          | `50`     |
| `decode.start_step`     | Step number to begin decode profiling         | `0`      |
| `decode.stop_step`      | Step number to end decode profiling           | `50`     |
| `aggregated.start_step` | Step number to begin aggregated profiling     | `0`      |
| `aggregated.stop_step`  | Step number to end aggregated profiling       | `50`     |

## Constraints

Profiling has specific requirements:

1. **Single worker only**: Profiling requires exactly 1 prefill worker and 1 decode worker (or 1 aggregated worker)

   ```yaml
   resources:
     prefill_workers: 1 # Must be 1
     decode_workers: 1 # Must be 1
   ```

2. **No benchmarking**: Profiling and benchmarking are mutually exclusive

   ```yaml
   benchmark:
     type: "manual" # Required when profiling
   ```

3. **Automatic config dump disabled**: When profiling is enabled, `enable_config_dump` is automatically set to `false`

## How It Works

### Normal Mode (`type: none`)

- Uses `dynamo.sglang` module for serving
- Standard disaggregated inference path

### Profiling Mode (`type: torch` or `nsys`)

- Uses `sglang.launch_server` module instead
- The `--disaggregation-mode` flag is automatically skipped (not supported by launch_server)
- Profiling script (`/scripts/profiling/profile.sh`) runs on leader nodes
- Sends requests via `sglang.bench_serving` to generate profiling workload

### nsys-specific behavior

When using `nsys`, workers are wrapped with:

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  -c cudaProfilerApi --capture-range-end stop \
  -o /logs/profiles/{mode}/{name} \
  python3 -m sglang.launch_server ...
```

## Example Configurations

### Torch Profiler (Recommended for Python analysis)

```yaml
name: "profiling-torch"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 1
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

profiling:
  type: "torch"
  isl: 1024
  osl: 128
  concurrency: 24
  prefill:
    start_step: 0
    stop_step: 50
  decode:
    start_step: 0
    stop_step: 50

benchmark:
  type: "manual"

backend:
  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      tensor-parallel-size: 4
    decode:
      kv-cache-dtype: "fp8_e4m3"
      tensor-parallel-size: 4
```

### Nsight Systems (Recommended for GPU kernel analysis)

```yaml
profiling:
  type: "nsys"
  isl: 2048
  osl: 64
  concurrency: 16
  prefill:
    start_step: 10
    stop_step: 30
  decode:
    start_step: 10
    stop_step: 30
```

## Output Files

After profiling completes, find results in the job's log directory:

```
logs/{job_id}_{workers}_{timestamp}/
├── profile_all.out         # Unified profiling script output
└── profiles/
    ├── prefill/            # Torch profiler traces (if type: torch)
    │   └── *.json
    ├── decode/
    │   └── *.json
    ├── prefill/            # Nsys reports (if type: nsys)
    │   └── *.nsys-rep
    └── decode/
        └── *.nsys-rep
```

### Viewing Results

**Torch Profiler traces:**

- Open in Chrome: `chrome://tracing`
- Or use TensorBoard: `tensorboard --logdir=logs/.../profiles/`

**Nsight Systems reports:**

- Open with NVIDIA Nsight Systems GUI
- Or CLI: `nsys stats logs/.../profiles/decode_0.nsys-rep`

## Troubleshooting

### "Profiling mode requires single worker only"

Reduce your worker counts to 1:

```yaml
resources:
  prefill_workers: 1
  decode_workers: 1
```

### "Cannot enable profiling with benchmark type"

Set benchmark to manual:

```yaml
benchmark:
  type: "manual"
```

### Empty profile output

Ensure `isl`, `osl`, and `concurrency` are set - they're required for the profiling workload.

### Profile too short/long

Adjust `start_step` and `stop_step` to capture the desired range. A typical profiling run uses 30-100 steps.
