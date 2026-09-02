#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the blog28 latency step-down chart."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from plot_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SWEEP_DATA,
    SweepPoint,
    compiled_bf16_latency,
    find_point,
    read_sweep,
)

BAR_COLORS = ("#174F0F", "#438F16", "#76B900")
# Groups whose dense or SAGE family is absent from the dataset are skipped.
GROUPS = (
    ("BF16", "BF16", "BF16 + SAGE"),
    ("FP8 per-tensor", "FP8 per-tensor", "FP8 per-tensor + SAGE"),
    ("NVFP4", "NVFP4", "NVFP4 + SAGE"),
)
BAR_LABELS = ("Dense attention", "+ SAGE", "+ Conservative Skip Softmax")


def _group_points(
    points: list[SweepPoint], dense_family: str, sage_family: str
) -> tuple[SweepPoint, SweepPoint, SweepPoint]:
    return (
        find_point(points, dense_family, 0.75, 0.0),
        find_point(points, sage_family, 0.75, 0.0),
        find_point(points, sage_family, 0.75, 0.86),
    )


def plot(data_path: Path, output_path: Path) -> None:
    """Render absolute latency after successively enabling attention optimizations."""
    points = read_sweep(data_path)
    baseline_latency = compiled_bf16_latency(points)
    available = {point.family for point in points}
    grouped = []
    for title, dense_family, sage_family in GROUPS:
        if dense_family not in available or sage_family not in available:
            print(
                f"[latency] skipping {title}: missing "
                f"{sorted({dense_family, sage_family} - available)}"
            )
            continue
        grouped.append((title, _group_points(points, dense_family, sage_family)))
    if not grouped:
        raise SystemExit("no group has both a dense and a SAGE family in this dataset")
    maximum_latency = max(
        point.latency_seconds for _, group_points in grouped for point in group_points
    )

    fig, axes = plt.subplots(
        len(grouped),
        1,
        figsize=(2160 / 180, 1296 / 180),
        dpi=180,
        sharex=True,
    )
    y_positions = (2, 1, 0)
    for axis, (title, group_points) in zip(axes, grouped):
        latencies = [point.latency_seconds for point in group_points]
        axis.barh(y_positions, latencies, color=BAR_COLORS, height=0.62, zorder=2)
        axis.plot(
            latencies,
            y_positions,
            color="#667085",
            linewidth=1.3,
            marker="o",
            markersize=4.5,
            markerfacecolor="#438F16",
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )
        for y_position, latency in zip(y_positions, latencies):
            speedup = baseline_latency / latency
            axis.text(
                latency + 7,
                y_position,
                f"{latency:.1f} s  ·  {speedup:.2f}×",
                va="center",
                ha="left",
                fontsize=10.5,
                color="#1F2937",
            )
        axis.set_yticks(y_positions, BAR_LABELS)
        axis.tick_params(axis="y", labelsize=10.5, length=0, pad=10, colors="#344054")
        axis.tick_params(axis="x", labelsize=10, colors="#475467")
        axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=8)
        axis.grid(axis="x", color="#DCE6EF", linestyle=(0, (2, 3)), linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#667085")
        axis.xaxis.set_major_locator(MultipleLocator(100))
        axis.set_xlim(0, maximum_latency * 1.17)

    fig.legend(
        handles=[
            Patch(color=BAR_COLORS[0], label="Dense attention"),
            Patch(color=BAR_COLORS[1], label="SAGE"),
            Patch(color=BAR_COLORS[2], label="SAGE + Conservative Skip Softmax"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.65, 0.995),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=1.1,
    )
    fig.supxlabel(
        "Pipeline-forward latency (seconds, lower is better)",
        x=0.62,
        y=0.035,
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.12, top=0.91, hspace=0.38)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_SWEEP_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "tech_blog28_latency_step_down.png",
    )
    args = parser.parse_args()
    plot(args.data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
