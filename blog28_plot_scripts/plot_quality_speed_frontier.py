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

"""Generate the blog28 quality-speed frontier."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from plot_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SWEEP_DATA,
    FAMILY_COLORS,
    FAMILY_ORDER,
    SweepPoint,
    compiled_bf16_latency,
    read_sweep,
)


def _speedup(point: SweepPoint, baseline_latency: float) -> float:
    return baseline_latency / point.latency_seconds


def _pareto_frontier(points: list[SweepPoint], baseline_latency: float) -> list[SweepPoint]:
    """Return points not dominated by higher speedup and lower LPIPS."""
    ordered = sorted(
        points,
        key=lambda point: (-_speedup(point, baseline_latency), point.mean_lpips),
    )
    frontier = []
    best_lpips = math.inf
    for point in ordered:
        if point.mean_lpips < best_lpips - 1e-12:
            frontier.append(point)
            best_lpips = point.mean_lpips
    return sorted(frontier, key=lambda point: _speedup(point, baseline_latency))


def _marker(point: SweepPoint) -> tuple[str, float, float]:
    if not point.skip_softmax_enabled:
        return "s", 58, 1.0
    if math.isclose(point.target_sparsity, 0.75) and math.isclose(
        point.disabled_until_timestep, 0.86
    ):
        return "*", 180, 1.0
    if math.isclose(point.target_sparsity, 0.75) and math.isclose(
        point.disabled_until_timestep, 1.0
    ):
        return "^", 90, 1.0
    return "o", 46, 0.72


def plot(data_path: Path, output_path: Path) -> None:
    """Render speedup versus mean LPIPS for the complete sweep."""
    points = read_sweep(data_path)
    baseline_latency = compiled_bf16_latency(points)
    frontier = _pareto_frontier(points, baseline_latency)

    fig, axis = plt.subplots(figsize=(2160 / 180, 1296 / 180), dpi=180)
    axis.plot(
        [_speedup(point, baseline_latency) for point in frontier],
        [point.mean_lpips for point in frontier],
        color="#374151",
        linewidth=1.8,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    for point in points:
        marker, size, alpha = _marker(point)
        axis.scatter(
            _speedup(point, baseline_latency),
            point.mean_lpips,
            marker=marker,
            s=size,
            color=FAMILY_COLORS[point.family],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.7,
            zorder=4 if marker in {"s", "*", "^"} else 3,
        )

    axis.set_xlabel("Speedup", fontsize=15, fontweight="bold", labelpad=14)
    axis.set_ylabel(
        "Mean LPIPS vs. Eager BF16\n(Lower Is Better)",
        fontsize=15,
        fontweight="bold",
        labelpad=16,
    )
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}×"))
    axis.tick_params(axis="both", labelsize=11, colors="#4B5563")
    axis.grid(True, color="#DCE6EF", linestyle=(0, (2, 3)), linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#667085")
    axis.spines["bottom"].set_color("#667085")

    family_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=7,
            color=FAMILY_COLORS[family],
            label=family,
        )
        # Only families present in this dataset.  FAMILY_ORDER spans every key
        # the loader understands, so listing it wholesale prints legend rows for
        # families that were never run and pushes the second legend off-axis.
        for family in FAMILY_ORDER
        if family in {point.family for point in points}
    ]
    family_legend = axis.legend(
        handles=family_handles,
        title="GEMM/Attention Quantization",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=11,
        title_fontsize=12,
        labelspacing=0.8,
        borderaxespad=0,
    )
    axis.add_artist(family_legend)

    config_handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            markersize=7,
            color="#5B6572",
            label="No Skip Softmax",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markersize=11,
            color="#5B6572",
            label="Conservative config",
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="None",
            markersize=7,
            color="#5B6572",
            label="Aggressive config",
        ),
    ]
    config_legend = axis.legend(
        handles=config_handles,
        title="Skip Softmax Attention Configuration",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.61),
        frameon=False,
        fontsize=11,
        title_fontsize=10,
        labelspacing=0.8,
        borderaxespad=0,
    )
    axis.add_artist(config_legend)
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#374151",
                linewidth=1.8,
                linestyle=(0, (4, 4)),
                label="Pareto frontier",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 0.30),
        frameon=False,
        fontsize=11,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.13, right=0.74, bottom=0.14, top=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_SWEEP_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "tech_blog28_quality_speed_frontier.png",
    )
    args = parser.parse_args()
    plot(args.data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
