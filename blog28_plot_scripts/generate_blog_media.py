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

"""Generate the animated and first-frame media used by Blog 28.

The input is the browser-sized ``media9`` tree. Every optimized source selects
``target_sparsity=0.75`` and ``disabled_until_timestep=0.86``. Source-directory
mapping is centralized in ``CONFIGS`` so corrected media can be substituted
without changing the rendering code.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio_v2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROMPTS = (
    "p01_cat_garden",
    "p03_park_kids",
    "p04_drone_coast",
    "p05_neon_sign",
    "p06_woman_smile",
    "p07_horse_gallop",
    "p10_market",
)
P1 = PROMPTS[0]
CELL_SIZE = (384, 216)
LABEL_HEIGHT = 72
GIF_FPS = 16


@dataclass(frozen=True)
class Config:
    slug: str
    source: str
    label: tuple[str, str]


CONFIGS = (
    Config("eager_bf16", "baseline", ("Eager BF16", "reference")),
    Config(
        "bf16_skip_softmax",
        "TRTLLM_WBF16_ABF16/t075_d086",
        ("BF16", "+ Skip Softmax"),
    ),
    Config(
        "bf16_sage_skip_softmax",
        "TRTLLM_WBF16_SAGE_QKINT8_PVFP8/t075_d086",
        ("BF16", "+ SAGE + Skip Softmax"),
    ),
    Config(
        "fp8_skip_softmax",
        "TRTLLM_WFP8STATIC_ABF16/t075_d086",
        ("FP8 per-tensor", "+ Skip Softmax"),
    ),
    Config(
        "fp8_sage_skip_softmax",
        "TRTLLM_WFP8STATIC_SAGE_QKINT8_PVFP8/t075_d086",
        ("FP8 per-tensor", "+ SAGE + Skip Softmax"),
    ),
    Config(
        "nvfp4_skip_softmax",
        "TRTLLM_WNVFP4STATIC_ABF16/t075_d086",
        ("NVFP4", "+ Skip Softmax"),
    ),
    Config(
        "nvfp4_sage_skip_softmax",
        "TRTLLM_WNVFP4STATIC_SAGE_QKINT8_PVFP8/t075_d086",
        ("NVFP4", "+ SAGE + Skip Softmax"),
    ),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _video_path(media_root: Path, config: Config, prompt: str) -> Path:
    path = media_root / config.source / f"{prompt}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _frames(path: Path) -> list[Image.Image]:
    frames = [Image.fromarray(frame).convert("RGB") for frame in iio.imiter(path, plugin="FFMPEG")]
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    if any(frame.size != CELL_SIZE for frame in frames):
        sizes = sorted({frame.size for frame in frames})
        raise ValueError(f"expected {CELL_SIZE} frames in {path}, found {sizes}")
    return frames


def _first_frame(path: Path) -> Image.Image:
    frame = Image.fromarray(iio.imread(path, index=0, plugin="FFMPEG")).convert("RGB")
    if frame.size != CELL_SIZE:
        raise ValueError(f"expected a {CELL_SIZE} frame in {path}, found {frame.size}")
    return frame


def _gif_durations(frame_count: int) -> list[float]:
    durations = []
    elapsed_centiseconds = 0
    for frame_index in range(1, frame_count + 1):
        target_centiseconds = round(frame_index * 100 / GIF_FPS)
        durations.append((target_centiseconds - elapsed_centiseconds) / 100)
        elapsed_centiseconds = target_centiseconds
    return durations


def generate_gifs(media_root: Path, output_dir: Path) -> None:
    for config in CONFIGS:
        frames = _frames(_video_path(media_root, config, P1))
        output = output_dir / f"tech_blog28_video_p01_{config.slug}.gif"
        iio_v2.mimsave(
            output,
            [np.asarray(frame) for frame in frames],
            format="GIF-FI",
            duration=_gif_durations(len(frames)),
            loop=0,
            quantizer="nq",
            palettesize=256,
        )
        print(output)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def generate_comparisons(media_root: Path, output_dir: Path) -> None:
    title_font = _font(22, bold=True)
    subtitle_font = _font(20)
    canvas_size = (CELL_SIZE[0] * len(CONFIGS), LABEL_HEIGHT + CELL_SIZE[1])
    for prompt in PROMPTS:
        canvas = Image.new("RGB", canvas_size, "white")
        draw = ImageDraw.Draw(canvas)
        for index, config in enumerate(CONFIGS):
            x = index * CELL_SIZE[0]
            frame = _first_frame(_video_path(media_root, config, prompt))
            canvas.paste(frame, (x, LABEL_HEIGHT))
            _draw_centered(
                draw, (x, 3, x + CELL_SIZE[0], 36), config.label[0], title_font, "#1F2937"
            )
            _draw_centered(
                draw,
                (x, 34, x + CELL_SIZE[0], LABEL_HEIGHT - 3),
                config.label[1],
                subtitle_font,
                "#4B5563",
            )
            draw.rectangle(
                (x, LABEL_HEIGHT - 4, x + CELL_SIZE[0] - 1, LABEL_HEIGHT - 1), fill="#76B900"
            )
        output = output_dir / f"tech_blog28_visual_comparison_{prompt}.jpg"
        canvas.save(output, quality=92, subsampling=0)
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_gifs(args.media_root, args.output_dir)
    generate_comparisons(args.media_root, args.output_dir)


if __name__ == "__main__":
    main()
