# Sparse Attention (Visual Generation)

> **Status**: Draft. This doc lives outside the published `docs/source/`
> tree; paste it into the visual-generation documentation hub once
> that hub exists.

- [Overview](#overview)
- [Algorithms](#algorithms)
  - [Skip Softmax Attention](#skip-softmax-attention)
  - [Video Sparse Attention (TODO)](#video-sparse-attention-todo)
- [Further Reading](#further-reading)

## Overview

Sparse attention reduces the cost of long-context inference by skipping
work on KV entries that contribute little to the attention output. In
the `VisualGen` pipeline, sparse attention is enabled by setting
`attention.sparse_attention_config` on `VisualGenArgs` (or the
equivalent YAML passed through `--visual_gen_args`). The config object
is a discriminated union — each algorithm has its own subclass of
`BaseSparseAttentionConfig` selected via the `algorithm` field.

This page focuses on the **user-facing API**: how to construct and
pass the config for each supported algorithm, in both Python and YAML
form. For framework design context, see the
[Sparse Attention tech blog](../blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md).

## Algorithms

### Skip Softmax Attention

A kernel-level method (BLASST) that dynamically skips Softmax and BMM2
work for unimportant KV blocks. No prediction step or framework hook —
the kernel decides what to skip from the per-block threshold. For
algorithm details and end-to-end results, see the
[Skip Softmax Attention tech blog][blog16].

[blog16]: ../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md

The value actually consumed by the kernel is
**`threshold_scale_factor`** — the kernel combines it with the
sequence length to compute the final per-block skip threshold at
runtime. Everything else is a way to produce that scalar.

Two configuration paths:

- **Set `threshold_scale_factor` directly.** The value flows straight
  to the kernel. Use this to turn on skip-softmax without a calibrated
  checkpoint ready.

- **Set `target_sparsity` ∈ [0, 1].** The runtime maps it to
  `threshold_scale_factor` via a calibration formula carried in the
  checkpoint's `config.json` under
  `sparse_attention_config.threshold_scale_factor`. If the checkpoint
  has no such block, the runtime raises a clear error. The
  calibration is supported by
  [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

`threshold_scale_factor` and `target_sparsity` are alternatives. If a
config happens to carry both (for example, a user override on top of
a checkpoint default), `threshold_scale_factor` wins and the
calibration formula is ignored.

A reserved third knob — **`warmup`** — accepts a non-negative integer
step count. It has **no runtime effect yet**; it lives in the schema
so future warmup-aware wiring will not break existing configs.

Skip Softmax only works with the **TRTLLM** attention backend (the
default `attention.backend`). Other backends silently bypass
skip-softmax. Cross-attention modules currently fall back to the
`VANILLA` backend (TRT-LLM's C++ attention op requires fused QKV for
non-MLA attention), so skip-softmax does not apply there.

#### Python API

```python
from tensorrt_llm.visual_gen import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig

# Direct threshold:
args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention={
        "backend": "TRTLLM",
        "sparse_attention_config": SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
        ),
    },
)

# Target sparsity (requires the checkpoint to carry a calibration formula):
args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention={
        "backend": "TRTLLM",
        "sparse_attention_config": SkipSoftmaxAttentionConfig(target_sparsity=0.70),
    },
)
```

User-facing fields on `SkipSoftmaxAttentionConfig`:

| Field | Type | Notes |
|---|---|---|
| `algorithm` | `Literal["skip_softmax"]` | Discriminator. |
| `threshold_scale_factor` | `Optional[float]` | Raw scalar; takes precedence over `target_sparsity`. |
| `target_sparsity` | `Optional[float] ∈ [0, 1]` | Resolved via a calibration formula. |
| `warmup` | `Optional[int] ≥ 0` | Reserved; no runtime effect today. |

Calibration state (formula coefficients, per-layer overrides,
per-component sub-configs) lives on private attributes populated by
the loaders. It is **not** part of the user-facing constructor or
YAML schema.

#### YAML configuration

`VisualGen` reads its runtime configuration from a single
`--visual_gen_args <path.yaml>` file. Skip-softmax sits under
`attention.sparse_attention_config:`:

```yaml
# Direct threshold:
attention:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 5000.0
```

```yaml
# Target sparsity (requires a calibrated checkpoint):
attention:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    target_sparsity: 0.70
```

#### Checkpoint calibration format

When `target_sparsity` is used, the calibration formula is read from
the checkpoint. The required `config.json` shape is:

```json
{
  "sparse_attention_config": {
    "threshold_scale_factor": {
      "formula": "a * exp(b * target_sparsity)",
      "coefficients": {"a": 100.0, "b": 5.0}
    }
  }
}
```

Required fields:

- `formula` — an **arbitrary**
  [numexpr](https://numexpr.readthedocs.io/)-evaluable expression of
  `target_sparsity` and one or more named coefficients. Standard math
  functions (`exp`, `log`, `sqrt`, `pow`, `**`, …) are available.
- `coefficients` — coefficient dictionary covering every name
  `formula` references (excluding `target_sparsity`). A single shared
  block is applied across the whole pipeline.

The calibration block will be supported by
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

> **TODO — notify checkpoint owners (migration)**: earlier ModelOpt
> drops shipped calibration in a standalone `sparse.yaml` (or
> per-component `sparse.<name>.yaml`) alongside the checkpoint, and
> used the LLM-style `prefill` key for the coefficient block. Going
> forward, visual-generation checkpoints must:
>
> 1. Carry the calibration inside the model's `config.json` under the
>    `sparse_attention_config` key shown above (the standalone sparse
>    YAML loader is slated for removal).
> 2. Key the coefficient block as **`coefficients`** — not
>    `prefill` — since diffusion has no prefill / decode distinction.
>
> **Notify the ModelOpt team and the owners of any checkpoint that
> currently ships a standalone sparse YAML or uses the `prefill`
> key.** Both changes need to land before re-publishing the affected
> checkpoints.

### Video Sparse Attention (TODO)

*(Placeholder — to be filled in once Video Sparse Attention lands.)*

## Further Reading

- BLASST kernel internals and end-to-end benchmarks:
  [Skip Softmax Attention tech blog][blog16].
