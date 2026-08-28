<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Blog 28 plot scripts

This directory reproduces the three data plots used by the video-generation optimization blog:

- `plot_pipeline_breakdown.py` generates `tech_blog28_bf16_time_breakdown.png`.
- `plot_quality_speed_frontier.py` generates `tech_blog28_quality_speed_frontier.png`.
- `plot_latency_step_down.py` generates `tech_blog28_latency_step_down.png`.

The scripts require Python 3.10 or later and Matplotlib:

```bash
python3 -m pip install matplotlib
```

Run all three scripts from this directory:

```bash
python3 plot_pipeline_breakdown.py
python3 plot_quality_speed_frontier.py
python3 plot_latency_step_down.py
```

By default, the images are written to `output/`. Pass `--output` to write a plot elsewhere:

```bash
python3 plot_quality_speed_frontier.py \
    --output ../docs/source/blogs/media/tech_blog28_quality_speed_frontier.png
```

## Updating the data

`pipeline_breakdown.csv` contains the Figure 1 percentages. Update the three rows when a new
profile is available.

`sweep_metrics4.csv` contains one aggregate row per experiment configuration. The plotting code
uses:

- `gen_sec_mean` as absolute pipeline-forward latency.
- `lpips_mean7` as mean LPIPS against eager BF16.
- compiled dense BF16 (`BF16_s0_t075_d000`) as the common speedup baseline.

The combo name has the form `<GEMM>_s<SAGE>_t<TARGET>_d<DISABLED_UNTIL>`. For example,
`FP8_BLOCK_s1_t075_d086` means FP8 blockwise GEMMs, SAGE enabled, `target_sparsity=0.75`, and
`disabled_until_timestep=0.86`. A `d000` row is the no-Skip-Softmax point for that GEMM/SAGE
family.

The frontier plot uses all configuration rows. Squares mark `d000`; stars mark the conservative
`t075_d086` configuration; triangles mark the aggressive `t075_d100` endpoint. The latency plot
selects three points per GEMM precision: dense attention, SAGE, and SAGE with `t075_d086`.

Each script accepts `--data` so a new CSV can be tested without replacing the checked-in copy:

```bash
python3 plot_quality_speed_frontier.py \
    --data /path/to/new_sweep_metrics.csv \
    --output /tmp/quality_speed_frontier.png
```
