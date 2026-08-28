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

"""Generate the blog28 compiled BF16 pipeline-breakdown chart."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_common import DEFAULT_BREAKDOWN_DATA, DEFAULT_OUTPUT_DIR, read_breakdown

COLORS = ("#236B17", "#76B900", "#B5BAC2")


def plot(data_path: Path, output_path: Path) -> None:
    """Render the pipeline breakdown from component percentages."""
    breakdown = read_breakdown(data_path)
    values = [value for _, value in breakdown]

    fig, axis = plt.subplots(figsize=(1860 / 180, 1038 / 180), dpi=180)
    wedges, _, percentages = axis.pie(
        values,
        colors=COLORS,
        startangle=90,
        counterclock=False,
        autopct="%1.1f%%",
        pctdistance=0.68,
        radius=0.88,
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
        textprops={"fontsize": 16, "fontweight": "bold"},
    )
    for index, percentage in enumerate(percentages):
        percentage.set_color("white" if index < 2 else "#202124")

    axis.legend(
        wedges,
        [f"{label}  {value:.1f}%" for label, value in breakdown],
        loc="center left",
        bbox_to_anchor=(0.96, 0.50),
        frameon=False,
        fontsize=16,
        handlelength=1.0,
        labelspacing=1.1,
    )
    axis.set_title("Pipeline Breakdown", fontsize=24, fontweight="bold", pad=24)
    axis.set_aspect("equal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_BREAKDOWN_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "tech_blog28_bf16_time_breakdown.png",
    )
    args = parser.parse_args()
    plot(args.data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
