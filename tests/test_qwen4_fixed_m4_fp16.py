"""FP16 stream parity for the Qwen4 fixed-M4 kernels (M1/M2 FP16 packs).

Every kernel that used to be pinned to BF16 is checked bit for bit against the stock MLX chain it
replaces, at both stream dtypes. BF16 is the control: it must keep passing unchanged. FP16 must match
the stock chain MLX itself runs on an FP16 stream.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from mtplx.kernels import qwen4_m4_route as route_kernel
from mtplx.kernels import qwen4_m4_stage3 as stage3_kernel
from mtplx.kernels.qwen4_m4_routed_down import (
    bind as bind_routed_down_reduce,
    bind_residual_tail,
    bind_residual_tail_two_dispatch,
)
from mtplx.kernels.qwen4_m4_routed_glu import bind as bind_routed_glu
from mtplx.kernels.qwen4_moe_prefill_combine import (
    combine_eligible,
    moe_prefill_combine,
    moe_prefill_combine_reference,
)
from mtplx.kernels.qwen4_qsa_m4_fused_kv_gather import bind_qwen4_qsa_m4_fused_kv_gather
from mtplx.kernels.qwen4_stream_dtype import specialize

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fixed-M4 kernels run on the GPU"
)

DTYPES = [pytest.param(mx.bfloat16, id="bf16"), pytest.param(mx.float16, id="fp16")]
ROWS, TOP_K, HIDDEN, INTER = 4, 10, 2560, 640


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16))


def _same(a: mx.array, b: mx.array) -> bool:
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    return np.array_equal(_bits(a), _bits(b))


def _q4_pack(key, experts, out_dims, in_dims, dtype):
    k = mx.random.split(key, 3)
    weights = mx.random.randint(
        -(2**31), 2**31 - 1, (experts, out_dims, in_dims // 8), dtype=mx.int32, key=k[0]
    ).view(mx.uint32)
    scales = (0.01 + 0.002 * mx.random.normal((experts, out_dims, in_dims // 32), key=k[1])).astype(dtype)
    biases = (-0.08 + 0.002 * mx.random.normal((experts, out_dims, in_dims // 32), key=k[2])).astype(dtype)
    return weights, scales, biases


def _route(key, experts, rows=ROWS, top_k=TOP_K):
    rng = np.random.default_rng(int(np.array(mx.random.randint(0, 2**30, (1,), key=key)).item()))
    ids = np.stack([rng.choice(experts, top_k, replace=False) for _ in range(rows)]).astype(np.uint32)
    scores = rng.random((rows, top_k)).astype(np.float32)
    scores /= scores.sum(axis=1, keepdims=True)
    return mx.array(ids), mx.array(scores)


# --- the substitution itself ------------------------------------------------------------------


def test_specialize_keeps_bf16_source_byte_identical_and_only_renames_the_scalar_type():
    src = "bfloat x = bfloat(1.0f); bfloat16_t keep; mybfloat keep2; const device bfloat* p;"
    assert specialize(src, mx.bfloat16) is src
    assert specialize(src, mx.float16) == (
        "half x = half(1.0f); bfloat16_t keep; mybfloat keep2; const device half* p;"
    )


# --- QSA fused K/V gather -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_kv_gather_matches_two_stock_takes(dtype):
    capacity = 16_384
    rng = np.random.default_rng(20261124)
    keys = mx.array(rng.normal(size=(1, 2, capacity, 256)).astype(np.float32)).astype(dtype)
    values = mx.array(rng.normal(size=(1, 2, capacity, 256)).astype(np.float32)).astype(dtype)
    token_idx = mx.array(rng.integers(0, capacity, size=(4, 2052), dtype=np.int32))
    selected_keys, selected_values = bind_qwen4_qsa_m4_fused_kv_gather(capacity=capacity)(
        keys, values, token_idx
    )
    flat = token_idx.reshape(-1)
    assert _same(selected_keys, mx.take(keys, flat, axis=2).reshape(1, 2, 4, 2052, 256))
    assert _same(selected_values, mx.take(values, flat, axis=2).reshape(1, 2, 4, 2052, 256))


# --- prefill combine --------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,top_k,hidden,scale", [(4096, 10, 2560, 1.0), (2048, 10, 2560, 30.0), (97, 3, 256, 5.0)])
def test_prefill_combine_is_bit_identical_to_the_stock_tail(dtype, rows, top_k, hidden, scale):
    mx.random.seed(rows)
    inds = mx.random.randint(0, 512, (1, rows, top_k)).astype(mx.uint32)
    _, _, inv = _gather_sort(mx.zeros((1, rows, 1, 1, 1), dtype=dtype), inds)
    y_sorted = (mx.random.normal((rows * top_k, 1, hidden)) * scale).astype(dtype)
    gates = mx.softmax((mx.random.normal((1, rows, top_k)) * 3).astype(dtype), axis=-1, precise=True)
    scores = gates / gates.sum(axis=-1, keepdims=True)
    shared = (mx.random.normal((1, rows, hidden)) * scale).astype(dtype)
    y = _scatter_unsort(y_sorted, inv, inds.shape).squeeze(-2)
    stock = ((y * scores[..., None]).sum(axis=-2) + shared).reshape(rows, hidden)
    args = (y_sorted.reshape(rows * top_k, hidden), inv, scores.reshape(rows, top_k), shared.reshape(rows, hidden))
    assert combine_eligible(*args)
    assert _same(moe_prefill_combine(*args), stock)
    assert _same(moe_prefill_combine_reference(*args), stock)


# --- stage-3 combine tails ------------------------------------------------------------------------


def _stock_combine(routed_down, route_scores, shared_down, shared_factor):
    """The SparseMoeBlock tail as the stock path runs it at the stream dtype."""

    routed = (routed_down * route_scores[..., None]).sum(axis=-2)
    return routed + shared_factor[:, None] * shared_down


def _stock_residual(block_out, hyper, inject):
    rows = block_out.shape[0]
    hyper = hyper.reshape(rows, 4, HIDDEN)
    return (hyper + block_out[:, None, :] * inject.reshape(rows, 4)[:, :, None]).reshape(rows, 4 * HIDDEN)


def _tail_inputs(dtype, seed):
    k = mx.random.split(mx.random.key(seed), 6)
    routed_down = (0.3 * mx.random.normal((ROWS, TOP_K, HIDDEN), key=k[0])).astype(dtype)
    scores = mx.softmax(mx.random.normal((ROWS, TOP_K), key=k[1]) * 2, axis=-1).astype(dtype)
    shared_down = (0.2 * mx.random.normal((ROWS, HIDDEN), key=k[2])).astype(dtype)
    shared_factor = mx.random.uniform(0, 1, (ROWS,), key=k[3]).astype(dtype)
    hyper = mx.random.normal((1, ROWS, 4 * HIDDEN), key=k[4]).astype(dtype)
    inject = mx.random.uniform(-2, 2, (1, ROWS, 4), key=k[5]).astype(dtype)
    return routed_down, scores, shared_down, shared_factor, hyper, inject


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_stage3_combine_tails_match_the_stock_chain(dtype, seed):
    routed_down, scores, shared_down, shared_factor, hyper, inject = _tail_inputs(dtype, seed)
    stock = _stock_combine(routed_down, scores, shared_down, shared_factor)
    assert _same(stage3_kernel.bind()(routed_down, shared_down, scores, shared_factor), stock)
    fused = stage3_kernel.bind_residual_tail()(routed_down, shared_down, scores, shared_factor, hyper, inject)
    assert _same(fused.reshape(ROWS, 4 * HIDDEN), _stock_residual(stock, hyper, inject))


# --- routed GLU and routed-down -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_paired_routed_glu_matches_stock_gather_qmm(dtype):
    experts = 16
    k = mx.random.split(mx.random.key(7), 3)
    value = (mx.random.normal((1, ROWS, HIDDEN), key=k[0])).astype(dtype)
    weights, scales, biases = _q4_pack(k[1], experts, 2 * INTER, HIDDEN, dtype)
    expert_ids, _ = _route(k[2], experts)
    gu = mx.gather_qmm(
        mx.expand_dims(value, (-2, -3)), weights, scales, biases,
        rhs_indices=expert_ids.reshape(1, ROWS, TOP_K), transpose=True,
        group_size=32, bits=4, mode="affine", sorted_indices=False,
    )
    gate, up = mx.split(gu, 2, axis=-1)
    expected = (nn.silu(gate) * up).reshape(ROWS, TOP_K, INTER)
    actual = bind_routed_glu()(value.reshape(ROWS, HIDDEN), weights, scales, biases, expert_ids)
    assert _same(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [11, 12])
def test_routed_down_kernels_match_stock_down_plus_combine(dtype, seed):
    experts = 16
    k = mx.random.split(mx.random.key(seed), 6)
    weights, scales, biases = _q4_pack(k[0], experts, HIDDEN, INTER, dtype)
    routed_h = (0.3 * mx.random.normal((ROWS, TOP_K, INTER), key=k[1])).astype(dtype)
    expert_ids, scores_f32 = _route(k[2], experts)
    scores = scores_f32.astype(dtype)
    shared_down = (0.2 * mx.random.normal((ROWS, HIDDEN), key=k[3])).astype(dtype)
    shared_factor = mx.random.uniform(0, 1, (ROWS,), key=k[4]).astype(dtype)
    hyper = mx.random.normal((1, ROWS, 4 * HIDDEN), key=k[5]).astype(dtype)
    inject = mx.random.uniform(-2, 2, (1, ROWS, 4), key=k[5]).astype(dtype)
    stock_down = mx.gather_qmm(
        routed_h.reshape(ROWS, TOP_K, 1, INTER), weights, scales, biases,
        rhs_indices=expert_ids, transpose=True, group_size=32, bits=4,
        mode="affine", sorted_indices=False,
    ).reshape(ROWS, TOP_K, HIDDEN)
    stock = _stock_combine(stock_down, scores, shared_down, shared_factor)
    reduce_out = bind_routed_down_reduce()(
        routed_h, weights, scales, biases, expert_ids, scores, shared_down, shared_factor
    )
    assert _same(reduce_out, stock)
    args = (routed_h, weights, scales, biases, expert_ids, scores, shared_down, shared_factor, hyper, inject)
    two = bind_residual_tail_two_dispatch()(*args)
    one = bind_residual_tail()(*args)
    want = _stock_residual(stock, hyper, inject).reshape(hyper.shape)
    assert _same(two, want)
    assert _same(one, want)


# --- route head ------------------------------------------------------------------------------------


class _ToyBlock:
    def __init__(self, dtype):
        self.num_experts = route_kernel.NUM_EXPERTS
        self.top_k = route_kernel.TOP_K
        self.norm_topk_prob = True
        self.gate = self._q8(route_kernel.NUM_EXPERTS, 11, dtype)
        self.shared_expert_gate = self._q8(1, 12, dtype)

    @staticmethod
    def _q8(out_dims, seed, dtype):
        mx.random.seed(seed)
        layer = nn.QuantizedLinear(HIDDEN, out_dims, bias=False, group_size=64, bits=8)
        layer.scales = layer.scales.astype(dtype)
        layer.biases = layer.biases.astype(dtype)
        return layer


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [7, 8, 9])
def test_route_kernel_matches_the_metal_order_reference(dtype, seed):
    block = _ToyBlock(dtype)
    route_kernel.check_contract(block, index=0)
    mx.random.seed(seed)
    x = mx.random.normal((1, ROWS, HIDDEN)).astype(dtype)
    want = route_kernel.metal_order_reference(block, x)
    got = route_kernel.bind()(
        x.reshape(ROWS, HIDDEN),
        block.gate.weight, block.gate.scales, block.gate.biases,
        block.shared_expert_gate.weight, block.shared_expert_gate.scales,
        block.shared_expert_gate.biases,
    )
    mx.eval(want, got)
    assert np.array_equal(np.array(want[0]), np.array(got[0])), "expert ids"
    assert _same(got[1], want[1]), "route scores"
    assert _same(got[2], want[2]), "shared factor"
