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

"""Shared data loading for the blog28 plotting scripts."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SWEEP_DATA = SCRIPT_DIR / "sweep_metrics_b200.csv"
DEFAULT_BREAKDOWN_DATA = SCRIPT_DIR / "pipeline_breakdown_b200.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_b200"

FAMILY_INFO = {
    ("BF16", 0): ("BF16", "#F28E2B"),
    ("BF16", 1): ("BF16 + SAGE", "#D62728"),
    ("FP8_STATIC", 0): ("FP8 per-tensor", "#2E7BE5"),
    ("FP8_STATIC", 1): ("FP8 per-tensor + SAGE", "#7B2CBF"),
    ("NVFP4_STATIC", 0): ("NVFP4", "#A6B800"),
    ("NVFP4_STATIC", 1): ("NVFP4 + SAGE", "#2E8B57"),
}
FAMILY_ORDER = (
    "BF16",
    "BF16 + SAGE",
    "FP8 per-tensor",
    "FP8 per-tensor + SAGE",
    "NVFP4",
    "NVFP4 + SAGE",
)
FAMILY_COLORS = {name: color for name, color in FAMILY_INFO.values()}


@dataclass(frozen=True)
class SweepPoint:
    """One aggregate point from the seven-prompt sweep."""

    combo: str
    family: str
    target_sparsity: float
    disabled_until_timestep: float
    mean_lpips: float
    latency_seconds: float

    @property
    def skip_softmax_enabled(self) -> bool:
        return self.disabled_until_timestep > 0.0


def _data_lines(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from (line for line in handle if not line.lstrip().startswith("#"))


def _parse_combo(combo: str) -> tuple[str, int, float, float] | None:
    if combo == "baseline_eager":
        return None
    try:
        quant, sage_token, target_token, disabled_token = combo.rsplit("_", 3)
        sage = int(sage_token.removeprefix("s"))
        target = int(target_token.removeprefix("t")) / 100.0
        disabled = int(disabled_token.removeprefix("d")) / 100.0
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid combo name: {combo}") from error
    if (quant, sage) not in FAMILY_INFO:
        raise ValueError(f"Unsupported quantization/SAGE family: {combo}")
    return quant, sage, target, disabled


def read_sweep(path: Path) -> list[SweepPoint]:
    """Load the aggregate sweep CSV used by the frontier and latency plots."""
    reader = csv.DictReader(_data_lines(path))
    required = {"combo", "lpips_mean7", "gen_sec_mean"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

    points = []
    seen = set()
    for row in reader:
        combo = row["combo"]
        parsed = _parse_combo(combo)
        if parsed is None:
            continue
        if combo in seen:
            raise ValueError(f"Duplicate combo: {combo}")
        seen.add(combo)
        quant, sage, target, disabled = parsed
        family, _ = FAMILY_INFO[(quant, sage)]
        points.append(
            SweepPoint(
                combo=combo,
                family=family,
                target_sparsity=target,
                disabled_until_timestep=disabled,
                mean_lpips=float(row["lpips_mean7"]),
                latency_seconds=float(row["gen_sec_mean"]),
            )
        )
    if not points:
        raise ValueError(f"No sweep points found in {path}")
    return points


def read_breakdown(path: Path) -> list[tuple[str, float]]:
    """Load pipeline components and their percentages."""
    reader = csv.DictReader(_data_lines(path))
    required = {"component", "percent"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError(f"{path} must contain columns: component, percent")
    values = [(row["component"], float(row["percent"])) for row in reader]
    if not math.isclose(sum(value for _, value in values), 100.0, abs_tol=0.05):
        raise ValueError("Pipeline breakdown percentages must sum to 100")
    return values


def find_point(
    points: list[SweepPoint],
    family: str,
    target_sparsity: float,
    disabled_until_timestep: float,
) -> SweepPoint:
    """Find a unique point by family and Skip Softmax settings."""
    matches = [
        point
        for point in points
        if point.family == family
        and math.isclose(point.target_sparsity, target_sparsity)
        and math.isclose(point.disabled_until_timestep, disabled_until_timestep)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {family} point at target_sparsity={target_sparsity}, "
            f"disabled_until_timestep={disabled_until_timestep}; found {len(matches)}"
        )
    return matches[0]


def compiled_bf16_latency(points: list[SweepPoint]) -> float:
    """Return the common compiled dense BF16 speedup baseline."""
    return find_point(points, "BF16", 0.75, 0.0).latency_seconds
