# simple_conv_nn_skip4_vitis_unified

U-Net style skip-connection Conv NN partitioned into 4 sub-models for Vitis Unified HLS (KV260 target).

---

## Architecture

```
PART 1 — Encoder-A
  inp (8,8,1)
    → enc_conv1 (8,8,8)  ──────────────────────────── skip1
    → enc_conv2 (8,8,16)
    → enc_pool1 (4,4,16)

PART 2 — Encoder-B + Bottleneck
    → enc_conv3 (4,4,16) ──────────────────────────── skip2
    → enc_pool2 (2,2,16)
    → bottleneck (2,2,16)

PART 3 — Decoder-A
    → dec_up1   (4,4,16)
    → dec_conv1 (4,4,16)
    → skip2_add (4,4,16) ◄─── skip2

PART 4 — Decoder-B + Head
    → dec_up2   (8,8,16)
    → dec_conv2 (8,8,8)
    → skip1_add (8,8,8)  ◄─── skip1
    → gap  (8,)
    → dense1 (16,)
    → dense_out (4,)       ← output (4 classes)
```

### Partition I/O boundaries

| Part | Input shape(s)            | Output shape(s)           |
|------|---------------------------|---------------------------|
| 1    | `(8, 8, 1)`               | `[(4,4,16), (8,8,8)]`     |
| 2    | `(4, 4, 16)`              | `[(2,2,16), (4,4,16)]`    |
| 3    | `[(2,2,16), (4,4,16)]`    | `(4, 4, 16)`              |
| 4    | `[(4,4,16), (8,8,8)]`     | `(4,)`                    |

---

## Requirements

- Python ≥ 3.9
- TensorFlow / Keras
- hls4ml (VitisUnified backend)
- Xilinx Vitis 2023.2 + Vivado 2023.2 (installed at `/tools/Xilinx/`)

---

## Running

### Basic usage

```bash
# Run everything: full model + all 4 partitions + HLS bisect diagnostic
python3 simple_conv_nn_skip4_vitis_unified.py

# Full model only (predict + save results, build stub)
python3 simple_conv_nn_skip4_vitis_unified.py --mode full

# All 4 partitions only — skips full model and diagnostic
python3 simple_conv_nn_skip4_vitis_unified.py --mode parts

# Single partition
python3 simple_conv_nn_skip4_vitis_unified.py --mode part1
python3 simple_conv_nn_skip4_vitis_unified.py --mode part2
python3 simple_conv_nn_skip4_vitis_unified.py --mode part3
python3 simple_conv_nn_skip4_vitis_unified.py --mode part4

# Disable the HLS bisect diagnostic (faster when iterating)
python3 simple_conv_nn_skip4_vitis_unified.py --no-diag

# Combine flags
python3 simple_conv_nn_skip4_vitis_unified.py --mode full --no-diag
python3 simple_conv_nn_skip4_vitis_unified.py --mode parts --no-diag
```

### Mode summary

| `--mode` | full model | part1–4 | diagnostic |
|----------|:---------:|:-------:|:----------:|
| `all`    | ✓         | ✓       | ✓ (if `--diag`) |
| `full`   | ✓         | ✗       | ✓ (if `--diag`) |
| `parts`  | ✗         | ✓       | ✗          |
| `part1`…`part4` | ✗  | one only | ✗        |

---

## Weight & input locking

On the **first run** the script trains the model and generates a random input pool,
then saves both to a lock directory so all subsequent runs are reproducible.

| File | Created when |
|------|-------------|
| `hls4ml_output2/_exp_locked_skip4/full_skip4_weights.h5` | First run (trains 20 epochs) |
| `hls4ml_output2/_exp_locked_skip4/x_input_1000000.npy`   | First run (1 M samples, ~256 MB) |

Delete either file to force re-generation.

### Input value range

The synthetic input pool is drawn from **Uniform[0, 1)** (`np.random.rand`, seed 42).
If your real inputs have a different range (e.g. raw pixels 0–255, or signed sensor data),
change the pool generation block and re-tune `HLS_OUT_PRECISION` / `HLS_GAP_PRECISION`
to avoid fixed-point overflow.

---

## HLS configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `HLS_REUSE_FACTOR` | 8 | DSP sharing across all layers |
| `HLS_PRECISION` | `ap_fixed<16,6>` | Model-level default (weights + accumulators) |
| `HLS_OUT_PRECISION` | `ap_fixed<16,2>` | Conv / pool / upsample outputs |
| `HLS_GAP_PRECISION` | `ap_fixed<32,16>` | GAP + Dense outputs (wider range needed) |
| Backend | `VitisUnified` | |
| Board | `kv260` | `xck26-sfvc784-2LV-c` |
| Clock | 10 ns (100 MHz) | |
| I/O type | `io_stream` + `axi_stream` | |

---

## Outputs

All outputs go to `hls4ml_output2/`:

```
hls4ml_output2/
├── _exp_locked_skip4/          ← locked weights + input pool (never wiped)
│   ├── full_skip4_weights.h5
│   └── x_input_1000000.npy
│
├── hls4mlprj_skip4_full/       ← full model HLS project
│   ├── timing_log.txt
│   ├── hls4ml_model.png
│   ├── x_input.npy
│   ├── y_pred_keras.npy
│   └── y_pred_hls.npy
│
├── hls4mlprj_skip4_part{1-4}/  ← per-partition HLS projects
│   ├── timing_log.txt
│   ├── hls4ml_model.png
│   └── hls_kernel_config.cfg
│
└── diag/                       ← bisect diagnostic (--diag only)
    └── diag_{layer_name}/      ← one HLS project per probed layer
```

---

## HLS bisect diagnostic

When `--diag` is enabled (default), the script probes every layer in the full model
end-to-end through csim and prints a pass/fail table:

```
  [enc_conv1   ]  keras=[+0.0000, +2.3412] mean=+0.4821  hls=[...] mean=...  OK
  [skip2_add   ]  keras=[+0.0000, +3.1200] mean=+0.6100  hls=[...] mean=...  COLLAPSED
  ...
```

Status meanings:

| Status | Meaning |
|--------|---------|
| `OK` | HLS mean within 50 % of Keras mean |
| `COLLAPSED` | Mean diverged — precision overflow likely |
| `COLLAPSED(uniform)` | HLS output is constant — signal lost completely |
| `ERROR(...)` | csim or conversion raised an exception |

---

## Architecture diagram

```
hls4ml_output2/skip4_arch.png
```

Regenerate with:
```bash
python3 hls4ml_output2/_gen_arch_diagram.py
```
