<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Blog 28 plotting and media scripts

These scripts reproduce the B200 figures and visual-comparison assets used by Blog 28. They are
shared on a standalone branch and are not intended for the blog pull request or `main`.

Install the local dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The checked-in `pipeline_breakdown_b200.csv` and `sweep_metrics_b200.csv` files contain the final
blog-ready data. Generate the three plots with:

```bash
.venv/bin/python plot_pipeline_breakdown.py
.venv/bin/python plot_quality_speed_frontier.py
.venv/bin/python plot_latency_step_down.py
```

`to_blog28_csv.py` converts the complete sweep CSV into the six-family, 96-configuration dataset
used by the blog plots:

```bash
.venv/bin/python to_blog28_csv.py /path/to/sweep_metrics_all.csv sweep_metrics_b200.csv
```

Generate the seven P1 GIFs and the seven first-frame comparison strips:

```bash
.venv/bin/python generate_blog_media.py \
    --media-root /path/to/media9 \
    --output-dir /path/to/TensorRT-LLM/docs/source/blogs/media
```

All optimized media selects `target_sparsity=0.75` and
`disabled_until_timestep=0.86`. `generate_blog_media.py` keeps source-directory mapping in one
`CONFIGS` table so a corrected media export can replace the current inputs without changing the
rendering logic.
