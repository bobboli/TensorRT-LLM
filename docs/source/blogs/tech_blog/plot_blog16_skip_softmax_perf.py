#!/usr/bin/env python3
"""Generate Blog 16 Skip-Softmax performance plots as static PNGs under ../media/.

This docs repo uses Sphinx + MyST (no notebook execution), so we render plots
offline and embed them as images in the Markdown.

Usage (from anywhere):
  python3 docs/source/blogs/tech_blog/plot_blog16_skip_softmax_perf.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required to generate the blog plots.\n"
        f"Import error: {e}\n\n"
        "Install it (example):\n"
        "  python3 -m pip install matplotlib\n"
    ) from e


Phase = Literal["prefill", "decode"]
Gpu = Literal["blackwell", "hopper"]
DType = Literal["bf16", "fp8"]


@dataclass(frozen=True)
class Series:
    seqlen: str  # e.g. "16k"
    dtype: DType
    x_sparsity_pct: List[float]
    y_speedup: List[float]
    baseline_metric: float  # TFLOP/s for prefill; TB/s for decode


def _series(
    seqlen: str,
    dtype: DType,
    rows: Iterable[Tuple[float, float, float]],
    *,
    baseline_idx: int = 0,
) -> Series:
    """rows: (sparsity_pct, metric, speedup)
    - metric is TFLOP/s for prefill, TB/s for decode
    """
    rows_l = list(rows)
    if not rows_l:
        raise ValueError(f"Empty series: seqlen={seqlen}, dtype={dtype}")
    x = [r[0] for r in rows_l]
    y = [r[2] for r in rows_l]
    baseline_metric = rows_l[baseline_idx][1]
    return Series(
        seqlen=seqlen,
        dtype=dtype,
        x_sparsity_pct=x,
        y_speedup=y,
        baseline_metric=baseline_metric,
    )


def get_data() -> Dict[Gpu, Dict[Phase, List[Series]]]:
    """Raw data copied directly from the blog tables.

    Notes:
    - X axis uses achieved sparsity (%).
    - Y axis uses speedup (already normalized vs baseline).
    - We only use the baseline metric (TFLOP/s or TB/s) from the baseline point.
    """
    # Color mapping is keyed only by seqlen so it's consistent across all plots.
    blackwell_prefill = [
        _series(
            "16k",
            "bf16",
            [
                (0.00, 1029.13, 1.00),
                (0.77, 1016.70, 0.99),
                (8.79, 1104.53, 1.07),
                (15.51, 1159.46, 1.13),
                (22.80, 1180.55, 1.15),
                (29.98, 1248.99, 1.21),
                (36.82, 1294.44, 1.26),
                (43.19, 1314.27, 1.28),
                (49.13, 1367.18, 1.33),
                (59.72, 1461.65, 1.42),
                (68.65, 1536.93, 1.49),
                (76.09, 1610.21, 1.56),
                (82.20, 1676.53, 1.63),
                (87.25, 1753.55, 1.70),
                (94.16, 1838.36, 1.79),
                (98.46, 1882.98, 1.83),
            ],
        ),
        _series(
            "16k",
            "fp8",
            [
                (0.00, 1523.57, 1.00),
                (0.77, 1527.17, 1.00),
                (8.81, 1556.14, 1.02),
                (15.50, 1587.11, 1.04),
                (22.78, 1624.80, 1.07),
                (29.96, 1668.90, 1.10),
                (36.82, 1714.68, 1.13),
                (43.13, 1763.96, 1.16),
                (49.05, 1815.13, 1.19),
                (59.65, 1925.32, 1.26),
                (68.59, 2041.63, 1.34),
                (76.01, 2155.09, 1.41),
                (82.17, 2266.71, 1.49),
                (87.14, 2370.11, 1.56),
                (94.12, 2559.36, 1.68),
                (98.46, 2731.16, 1.79),
            ],
        ),
        _series(
            "64k",
            "bf16",
            [
                (0.00, 1038.26, 1.00),
                (0.19, 1036.03, 1.00),
                (38.87, 1302.78, 1.25),
                (49.15, 1376.42, 1.33),
                (56.90, 1436.79, 1.38),
                (62.96, 1489.71, 1.43),
                (67.83, 1535.26, 1.48),
                (71.85, 1575.72, 1.52),
                (75.26, 1612.14, 1.55),
                (80.82, 1673.38, 1.61),
                (85.27, 1723.23, 1.66),
                (88.91, 1771.61, 1.71),
                (91.85, 1811.59, 1.74),
                (94.16, 1842.88, 1.77),
                (97.21, 1889.81, 1.82),
                (99.61, 1925.68, 1.85),
            ],
        ),
        _series(
            "64k",
            "fp8",
            [
                (0.00, 1621.41, 1.00),
                (0.19, 1626.86, 1.00),
                (38.87, 1861.54, 1.15),
                (49.13, 1962.05, 1.21),
                (56.87, 2051.47, 1.27),
                (62.94, 2131.31, 1.31),
                (67.81, 2202.96, 1.36),
                (71.83, 2267.48, 1.40),
                (75.24, 2326.69, 1.43),
                (80.79, 2435.71, 1.50),
                (85.24, 2536.55, 1.56),
                (88.88, 2632.71, 1.62),
                (91.82, 2720.56, 1.68),
                (94.13, 2797.27, 1.73),
                (97.19, 2919.07, 1.80),
                (99.61, 3049.29, 1.88),
            ],
        ),
    ]

    blackwell_decode = [
        _series(
            "16k",
            "fp8",
            [
                (0.00, 5.45670, 1.000000000),
                (1.12915, 5.50139, 1.008189932),
                (3.46680, 5.50559, 1.008959628),
                (8.26111, 5.58582, 1.023662653),
                (15.9973, 5.66580, 1.038319864),
                (25.7050, 5.82381, 1.067276926),
                (36.9904, 6.03396, 1.105789213),
                (48.4955, 6.30147, 1.154813349),
                (58.8501, 6.59167, 1.207995675),
                (68.0176, 6.93543, 1.270993458),
                (74.9176, 7.20972, 1.321260102),
                (79.9042, 7.41881, 1.359578133),
                (83.4595, 7.65325, 1.402541829),
                (98.6420, 8.63264, 1.582025766),
            ],
        ),
        _series(
            "16k",
            "bf16",
            [
                (0.00, 7.08174, 1.000000000),
                (1.12915, 7.11051, 1.004062561),
                (3.46680, 7.14484, 1.008910240),
                (8.26111, 7.27714, 1.027592089),
                (15.9973, 7.49437, 1.058266754),
                (25.7050, 7.80767, 1.102507293),
                (36.9904, 8.20691, 1.158883269),
                (48.4955, 8.67138, 1.224470257),
                (58.8501, 9.17625, 1.295762058),
                (68.0176, 9.69759, 1.369379559),
                (74.9176, 10.1733, 1.436553728),
                (79.9042, 10.5115, 1.484310353),
                (83.4595, 10.7836, 1.522733114),
                (98.6420, 12.1459, 1.715101091),
            ],
        ),
        _series(
            "64k",
            "fp8",
            [
                (0.00, 5.68271, 1.000000000),
                (15.5098, 5.88751, 1.036039143),
                (25.7439, 6.03086, 1.061264784),
                (37.3360, 6.22557, 1.095528366),
                (49.1463, 6.51522, 1.146498766),
                (59.9716, 6.83298, 1.202415749),
                (69.4405, 7.18323, 1.264050075),
                (77.2346, 7.51812, 1.322981465),
                (83.2115, 7.89669, 1.389599328),
                (87.7853, 8.18607, 1.440522216),
                (90.8096, 8.37243, 1.473316428),
                (92.7696, 8.51729, 1.498807787),
                (94.1277, 8.61169, 1.515419580),
                (99.5018, 9.03026, 1.589076339),
            ],
        ),
        _series(
            "64k",
            "bf16",
            [
                (0.00, 7.10208, 1.000000000),
                (15.5098, 7.55759, 1.064137548),
                (25.7439, 7.86696, 1.107698026),
                (37.3360, 8.29362, 1.167773385),
                (49.1463, 8.78393, 1.236810906),
                (59.9716, 9.39590, 1.322978620),
                (69.4405, 9.94081, 1.399704030),
                (77.2346, 10.4971, 1.478031788),
                (83.2115, 11.0106, 1.550334550),
                (87.7853, 11.4227, 1.608359804),
                (90.8096, 11.7000, 1.647404704),
                (92.7696, 11.8966, 1.675086735),
                (94.1277, 12.0313, 1.694053010),
                (99.5018, 12.6231, 1.777380711),
            ],
        ),
    ]

    # Hopper data will be provided later. We still generate placeholder plots
    # so the blog has the final structure (4 plots).
    hopper_prefill: List[Series] = [
        _series(
            "16k",
            "bf16",
            [
                (0.00, 594.05, 1.00),
                (0.25, 583.17, 0.981685043),
                (2.11, 587.57, 0.989091827),
                (13.89, 621.93, 1.046932076),
                (20.68, 642.07, 1.080834947),
                (28.47, 670.81, 1.129214713),
                (36.75, 707.03, 1.190186011),
                (44.49, 740.78, 1.246999411),
                (51.39, 776.40, 1.306960694),
                (62.71, 845.31, 1.422961030),
                (71.23, 907.46, 1.527581853),
                (74.67, 934.99, 1.573924754),
                (80.32, 983.23, 1.655130040),
                (82.64, 1004.08, 1.690228095),
                (86.50, 1038.53, 1.748219847),
                (90.78, 1073.60, 1.807255282),
                (93.69, 1095.82, 1.844659540),
            ],
        ),
        _series(
            "16k",
            "fp8",
            [
                (0.00, 852.81, 1.00),
                (0.00, 825.96, 0.968515848),
                (0.11, 825.94, 0.968492396),
                (3.27, 831.82, 0.975387249),
                (6.63, 839.36, 0.984228609),
                (11.66, 852.60, 0.999753755),
                (18.20, 872.60, 1.023205638),
                (25.56, 898.02, 1.053012981),
                (33.14, 930.08, 1.090606348),
                (47.20, 999.66, 1.172195448),
                (58.88, 1069.88, 1.254535008),
                (63.81, 1103.38, 1.293816911),
                (72.00, 1163.66, 1.364500885),
                (75.42, 1190.04, 1.395433918),
                (81.00, 1235.34, 1.448552433),
                (87.18, 1287.67, 1.509914283),
                (91.30, 1321.73, 1.549852839),
            ],
        ),
        _series(
            "64k",
            "bf16",
            [
                (0.00, 610.30, 1.00),
                (10.23, 611.30, 1.001638538),
                (23.78, 659.31, 1.080304768),
                (49.24, 774.83, 1.269588727),
                (57.27, 822.82, 1.348222186),
                (64.56, 874.52, 1.432934622),
                (70.97, 924.86, 1.515418647),
                (75.80, 966.04, 1.582893659),
                (79.52, 998.75, 1.636490251),
                (84.85, 1048.72, 1.718368016),
                (88.45, 1088.82, 1.784073407),
                (89.84, 1097.54, 1.798361462),
                (92.06, 1123.53, 1.840947075),
                (92.96, 1134.95, 1.859659184),
                (94.47, 1145.16, 1.876388661),
                (96.19, 1173.87, 1.923431099),
                (97.40, 1195.63, 1.959085696),
            ],
        ),
        _series(
            "64k",
            "fp8",
            [
                (0.00, 873.60, 1.00),
                (1.87, 851.87, 0.975125916),
                (8.34, 866.55, 0.991929945),
                (29.72, 937.59, 1.073248626),
                (38.58, 975.00, 1.116071429),
                (47.40, 1025.43, 1.173798077),
                (55.76, 1076.79, 1.232589286),
                (62.57, 1113.24, 1.274313187),
                (68.11, 1151.51, 1.318120421),
                (76.36, 1185.19, 1.356673535),
                (82.11, 1194.77, 1.367639652),
                (84.35, 1200.78, 1.374519231),
                (87.97, 1209.32, 1.384294872),
                (89.44, 1211.58, 1.386881868),
                (91.87, 1213.74, 1.389354396),
                (94.56, 1213.63, 1.389228480),
                (96.36, 1208.45, 1.383298993),
            ],
        ),
    ]
    hopper_decode: List[Series] = [
        _series(
            "16k",
            "bf16",
            [
                # Baseline: 0 (No Skip Kernel)
                (0.00, 4.305, 1.000),
                # Pruned non-zero sparsity points
                (0.25, 4.308, 1.001),
                (2.73, 4.360, 1.013),
                (9.79, 4.501, 1.045),
                (20.27, 4.717, 1.096),
                (31.56, 4.968, 1.154),
                (42.47, 5.223, 1.213),
                (52.01, 5.453, 1.267),
                (66.43, 5.838, 1.356),
                (79.48, 6.191, 1.438),
                (89.58, 6.486, 1.506),
                (98.04, 6.724, 1.562),
            ],
        ),
        _series(
            "16k",
            "fp8",
            [
                # Baseline: 0 (No Skip Kernel)
                (0.00, 4.033, 1.000),
                # Pruned non-zero sparsity points
                (1.66, 3.836, 0.951),
                (6.59, 3.898, 0.967),
                (11.86, 3.987, 0.989),
                (16.71, 4.049, 1.004),
                (21.09, 4.119, 1.021),
                (24.88, 4.181, 1.037),
                (31.01, 4.279, 1.061),
                (36.03, 4.387, 1.088),
                (40.30, 4.471, 1.109),
                (43.97, 4.523, 1.121),
                (46.98, 4.655, 1.154),
                (49.68, 4.620, 1.146),
                (51.95, 4.721, 1.170),
                (55.94, 4.773, 1.183),
                (60.36, 4.968, 1.232),
                (65.78, 5.059, 1.254),
                (79.27, 5.372, 1.332),
            ],
        ),
        _series(
            "64k",
            "bf16",
            [
                # Baseline: 0 (No Skip Kernel)
                (0.00, 4.366, 1.000),
                # Pruned non-zero sparsity points
                (0.05, 4.430, 1.015),
                (0.61, 4.445, 1.018),
                (7.51, 4.590, 1.051),
                (22.86, 4.930, 1.129),
                (39.91, 5.345, 1.224),
                (54.43, 5.730, 1.312),
                (65.59, 6.066, 1.389),
                (73.79, 6.326, 1.449),
                (79.80, 6.526, 1.495),
                (87.53, 6.780, 1.553),
                (93.17, 6.966, 1.595),
                (96.88, 7.090, 1.624),
                (99.49, 7.171, 1.643),
            ],
        ),
        _series(
            "64k",
            "fp8",
            [
                # Baseline: 0 (No Skip Kernel)
                (0.00, 4.101, 1.000),
                # Pruned non-zero sparsity points
                (1.05, 4.040, 0.985),
                (18.14, 4.298, 1.048),
                (33.70, 4.658, 1.136),
                (43.43, 4.863, 1.186),
                (50.29, 5.073, 1.237),
                (55.30, 5.200, 1.268),
                (59.24, 5.151, 1.256),
                (65.09, 5.417, 1.321),
                (69.21, 5.535, 1.350),
                (72.32, 5.539, 1.351),
                (74.75, 5.644, 1.376),
                (76.79, 5.678, 1.385),
                (78.44, 5.768, 1.407),
                (79.84, 5.766, 1.406),
                (82.07, 5.841, 1.424),
                (84.48, 5.876, 1.433),
                (87.19, 5.962, 1.454),
                (93.09, 6.126, 1.494),
            ],
        ),
    ]

    return {
        "blackwell": {"prefill": blackwell_prefill, "decode": blackwell_decode},
        "hopper": {"prefill": hopper_prefill, "decode": hopper_decode},
    }


def _style_for_dtype(dtype: DType) -> Dict:
    if dtype == "bf16":
        return {"linestyle": "-"}
    # Distinguish FP8 without point markers.
    return {"linestyle": "-."}


def plot_phase(
    *,
    gpu: Gpu,
    phase: Phase,
    series_list: List[Series],
    out_path: Path,
    seqlen_colors: Dict[str, str],
) -> None:
    title_gpu = "Blackwell (B200)" if gpu == "blackwell" else "Hopper (H200)"
    title_phase = "Prefill" if phase == "prefill" else "Decode"
    metric_name = "FLOPS" if phase == "prefill" else "Bandwidth"
    metric_unit = "TFLOP/s" if phase == "prefill" else "TB/s"

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=160, constrained_layout=True)

    ax.set_title(f"{title_gpu} - {title_phase}")
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Speedup")
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
    # Keep a fixed y-range for easy visual comparison across all plots.
    ax.set_ylim(0.9, 2.0)

    if not series_list:
        ax.text(
            0.5,
            0.5,
            "Hopper data pending",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xlim(0, 100)
        fig.savefig(out_path)
        plt.close(fig)
        return

    # Stable ordering: group by seqlen, then dtype
    series_list_sorted = sorted(series_list, key=lambda s: (s.seqlen, s.dtype))

    # Plot curves.
    for s in series_list_sorted:
        color = seqlen_colors.setdefault(s.seqlen, None) or seqlen_colors[s.seqlen]
        st = _style_for_dtype(s.dtype)
        label = f"{s.seqlen} {s.dtype.upper()}"
        ax.plot(
            s.x_sparsity_pct,
            s.y_speedup,
            color=color,
            linewidth=2.0,
            label=label,
            **st,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0.9, 2.0)

    ax.legend(
        loc="upper left",
        fontsize=9,
        frameon=True,
        ncol=2,
    )

    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve()
    blog_dir = here.parent
    media_dir = blog_dir.parent / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Ensure consistent seqlen->color mapping across all plots.
    seqlen_colors: Dict[str, str] = {
        "16k": "#1f77b4",  # blue
        "64k": "#ff7f0e",  # orange
        "128k": "#2ca02c",  # green (reserved for future)
        "256k": "#d62728",  # red   (reserved for future)
    }

    data = get_data()

    outputs = [
        ("blackwell", "prefill", "tech_blog16_blackwell_prefill.png"),
        ("blackwell", "decode", "tech_blog16_blackwell_decode.png"),
        ("hopper", "prefill", "tech_blog16_hopper_prefill.png"),
        ("hopper", "decode", "tech_blog16_hopper_decode.png"),
    ]

    for gpu, phase, fname in outputs:
        out_path = media_dir / fname
        plot_phase(
            gpu=gpu,  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
            series_list=data[gpu][phase],  # type: ignore[index]
            out_path=out_path,
            seqlen_colors=seqlen_colors,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
