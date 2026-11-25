"""Stream dtype of the Qwen4 fixed-M4 kernels: bfloat16, or float16 on FP16 packs (M1 and M2 Macs).

The fixed-M4 kernels transcribe MLX templates (``qmv``, ``softmax_single_row``,
``row_reduce_small``, ``Sigmoid``, the stage-3 combine) at ``T = bfloat16_t``. MLX instantiates the
same templates at ``T = float16_t`` for FP16 streams, so the FP16 spelling of each kernel is the BF16
source with the Metal scalar type ``bfloat`` replaced by ``half``: the same operation order with
float16 rounding boundaries, i.e. what the stock MLX path computes on an FP16 stream.

The BF16 source is returned unchanged, so BF16 packs run byte-identical kernels.
"""

from __future__ import annotations

import re

import mlx.core as mx

#: Stream dtypes the fixed-M4 kernels serve.
SUPPORTED_STREAM_DTYPES = (mx.bfloat16, mx.float16)

_METAL_TYPE = {mx.bfloat16: "bfloat", mx.float16: "half"}
_SUFFIX = {mx.bfloat16: "", mx.float16: "_f16"}
_BFLOAT_TOKEN = re.compile(r"\bbfloat\b")


def check_stream_dtype(dtype: mx.Dtype, what: str = "Qwen4 fixed-M4 stream") -> mx.Dtype:
    if dtype not in _METAL_TYPE:
        raise ValueError(f"{what} must be bfloat16 or float16; got {dtype}")
    return dtype


def metal_type(dtype: mx.Dtype) -> str:
    """The Metal scalar type of a stream dtype (``bfloat`` or ``half``)."""

    return _METAL_TYPE[check_stream_dtype(dtype)]


def kernel_suffix(dtype: mx.Dtype) -> str:
    """Kernel-name suffix: empty for BF16 (names unchanged), ``_f16`` for FP16."""

    return _SUFFIX[check_stream_dtype(dtype)]


def specialize(source: str, dtype: mx.Dtype) -> str:
    """The kernel source for a stream dtype; BF16 returns ``source`` unchanged."""

    check_stream_dtype(dtype)
    if dtype == mx.bfloat16:
        return source
    return _BFLOAT_TOKEN.sub(_METAL_TYPE[dtype], source)


def storage_name(dtype: mx.Dtype) -> str:
    """Safetensors dtype name of a stream dtype (``BF16`` or ``F16``)."""

    return {mx.bfloat16: "BF16", mx.float16: "F16"}[check_stream_dtype(dtype)]


__all__ = [
    "SUPPORTED_STREAM_DTYPES",
    "check_stream_dtype",
    "kernel_suffix",
    "metal_type",
    "specialize",
    "storage_name",
]
