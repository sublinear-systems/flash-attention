"""Use a precomputed softmax reduction without retaining the forward output."""

import pytest
import torch
from cutlass import BFloat16, Float16

from flash_attn.cute.interface import _bwd_preprocess, _flash_attn_bwd, _flash_attn_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="These backward regression cases are validated on SM90",
)


def _reference(q, k, v, q_lengths, k_lengths, scale, causal):
    """Dense FP64 oracle, including unequal causal lengths and fully masked rows."""
    outputs, normalizers = [], []
    q_start = k_start = 0
    group_size = q.shape[-2] // k.shape[-2]
    for q_len, k_len in zip(q_lengths, k_lengths):
        qi = q[q_start : q_start + q_len].transpose(0, 1)
        ki = (
            k[k_start : k_start + k_len]
            .transpose(0, 1)
            .repeat_interleave(group_size, 0)
        )
        vi = (
            v[k_start : k_start + k_len]
            .transpose(0, 1)
            .repeat_interleave(group_size, 0)
        )
        scores = (qi @ ki.transpose(-1, -2)) * scale
        if causal:
            visible = torch.arange(k_len, device=q.device)[None, :] <= (
                torch.arange(q_len, device=q.device)[:, None] + k_len - q_len
            )
            scores = scores.masked_fill(~visible, -torch.inf)
        finite_rows = scores.isfinite().any(-1)
        # Keep logsumexp's derivative finite even on fully masked rows.
        denominator = scores.masked_fill(~finite_rows[..., None], 0).logsumexp(-1)
        probabilities = (scores - denominator[..., None]).exp()
        lse = denominator.masked_fill(~finite_rows, -torch.inf)
        outputs.append((probabilities @ vi).transpose(0, 1))
        normalizers.append(lse)
        q_start += q_len
        k_start += k_len
    return torch.cat(outputs), torch.cat(normalizers, dim=-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "varlen,q_lengths,k_lengths,heads,kv_heads,d,dv,causal,pack_gqa",
    [
        (True, [1, 0, 129, 257], [1, 0, 129, 257], 6, 2, 64, 64, True, True),
        (True, [7, 130], [129, 31], 8, 2, 128, 128, False, True),
        (True, [5, 257], [1, 193], 2, 2, 192, 128, True, False),
        (True, [128, 130], [128, 130], 2, 1, 256, 256, True, True),
        (False, [193, 193], [257, 257], 6, 2, 128, 128, False, True),
        (False, [33, 33], [17, 17], 2, 2, 128, 128, True, False),
    ],
)
def test_precomputed_delta_backward(
    dtype, varlen, q_lengths, k_lengths, heads, kv_heads, d, dv, causal, pack_gqa
):
    torch.manual_seed(321)
    q = torch.randn(sum(q_lengths), heads, d, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lengths), kv_heads, d, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lengths), kv_heads, dv, device="cuda", dtype=dtype)
    refs = [x.double().requires_grad_() for x in (q, k, v)]
    scale = d**-0.5
    ref_out, ref_lse = _reference(*refs, q_lengths, k_lengths, scale, causal)
    if varlen:
        kw = {
            "cu_seqlens_q": torch.tensor(
                [0, *q_lengths], device="cuda", dtype=torch.int32
            ).cumsum(0, dtype=torch.int32),
            "cu_seqlens_k": torch.tensor(
                [0, *k_lengths], device="cuda", dtype=torch.int32
            ).cumsum(0, dtype=torch.int32),
            "max_seqlen_q": max(q_lengths),
            "max_seqlen_k": max(k_lengths),
        }
    else:
        batch = len(q_lengths)
        q, k, v = [x.view(batch, -1, *x.shape[1:]) for x in (q, k, v)]
        ref_out = ref_out.view(batch, q_lengths[0], heads, dv)
        ref_lse = ref_lse.view(heads, batch, q_lengths[0]).transpose(0, 1)
        kw = {}

    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        pack_gqa=pack_gqa,
        return_lse=True,
        **kw,
    )[:2]
    finite_rows = ref_lse.isfinite()
    torch.testing.assert_close(
        lse[finite_rows].double(), ref_lse[finite_rows], rtol=1e-5, atol=1e-5
    )
    assert (out.double() - ref_out).norm() / ref_out.norm() < 3e-3

    go = torch.randn_like(out)
    gl = torch.randn_like(lse).masked_fill(~finite_rows, 0)
    # Masking avoids the undefined 0 * (-inf) of the all-masked LSE rows.
    ref_loss = (ref_out * go.double()).sum() + (
        ref_lse[finite_rows] * gl[finite_rows]
    ).sum()
    ref_grads = torch.autograd.grad(ref_loss, refs)
    delta = (ref_out.detach() * go.double()).sum(-1).transpose(-1, -2).float()
    grads = _flash_attn_bwd(
        q,
        k,
        v,
        None,
        go,
        lse,
        scale,
        causal,
        0.0,
        deterministic=True,
        dlse=gl,
        delta=delta,
        **kw,
    )
    for actual, expected in zip(grads, ref_grads):
        assert torch.isfinite(actual).all()
        assert (
            actual.reshape_as(expected).double() - expected
        ).norm() / expected.norm() < 1e-2

    # Revisit both preprocessing cache variants.
    out_again, lse_again = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        pack_gqa=pack_gqa,
        return_lse=True,
        **kw,
    )[:2]
    torch.testing.assert_close(out_again, out, rtol=0, atol=0)
    torch.testing.assert_close(lse_again, lse, rtol=0, atol=0)
    half_grads = _flash_attn_bwd(
        q, k, v, out, go, lse, scale, causal, 0.0, deterministic=True, dlse=gl, **kw
    )
    # dV does not use the saved output; this also catches accidental cache aliasing.
    torch.testing.assert_close(grads[2], half_grads[2], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_constant_keys_have_zero_query_gradient(dtype):
    q = torch.full((256, 1, 128), 0.125, device="cuda", dtype=dtype)
    k, v = torch.ones_like(q), torch.full_like(q, 32)
    v[0] = 32.25
    cu = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    kw = {
        "cu_seqlens_q": cu,
        "cu_seqlens_k": cu,
        "max_seqlen_q": 256,
        "max_seqlen_k": 256,
    }
    out, lse = _flash_attn_fwd(
        q, k, v, softmax_scale=1.0, causal=True, return_lse=True, **kw
    )[:2]
    go = torch.zeros_like(v)
    go[-1] = 1 / 128
    delta = torch.zeros_like(lse)

    def backward():
        delta[:, -1] = go[-1].float().sum(-1) * (32 + 0.25 / 256)
        return _flash_attn_bwd(
            q,
            k,
            v,
            None,
            go,
            lse,
            1.0,
            True,
            0.0,
            deterministic=True,
            delta=delta,
            **kw,
        )

    rounded_grads = _flash_attn_bwd(q, k, v, out, go, lse, 1.0, True, 0.0, **kw)
    assert rounded_grads[0].abs().max() > 0
    grads = backward()
    # Every key is identical, so the output distribution is independent of Q.
    assert torch.count_nonzero(grads[0]) == 0
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_grads = backward()
    for factor in (1, -2, 4):
        go[-1] = factor / 128
        graph.replay()
        expected = backward()
        for actual, reference in zip(captured_grads, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert torch.count_nonzero(captured_grads[0]) == 0


@pytest.mark.parametrize(
    "dtype,cute_dtype", [(torch.bfloat16, BFloat16), (torch.float16, Float16)]
)
@pytest.mark.parametrize("has_dlse", [False, True])
def test_delta_preprocess_does_not_read_output(dtype, cute_dtype, has_dlse):
    torch.manual_seed(912)
    # NaN in all of O/dO, including padded columns, detects unwanted reads.
    out_storage = torch.full((2, 37, 6, 96), torch.nan, device="cuda", dtype=dtype)
    out = out_storage[..., :80]
    lse = torch.randn((2, 6, 37), device="cuda")
    delta = torch.randn_like(lse)
    dlse = torch.randn_like(lse) if has_dlse else None
    dpsum = torch.full((2, 6, 128), torch.nan, device="cuda")
    lse_log2 = torch.empty_like(dpsum)
    _bwd_preprocess(
        out,
        out,
        dpsum,
        lse,
        lse_log2,
        None,
        None,
        None,
        dlse,
        cute_dtype,
        128,
        80,
        128,
        delta=delta,
    )
    expected = delta if dlse is None else delta - dlse
    torch.testing.assert_close(dpsum[..., :37], expected, rtol=0, atol=0)
    assert torch.count_nonzero(dpsum[..., 37:]) == 0
    torch.testing.assert_close(lse_log2[..., :37], lse * 1.4426950408889634)


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype", "device"])
def test_delta_validation(invalid):
    q = torch.zeros((1, 17, 1, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros((1, 1, 17), device="cuda")
    delta = torch.zeros_like(lse)
    if invalid == "missing":
        delta = None
    elif invalid == "shape":
        delta = delta[..., :-1]
    elif invalid == "dtype":
        delta = delta.to(q.dtype)
    elif invalid == "device":
        delta = delta.cpu()
    with pytest.raises(AssertionError, match="delta"):
        _flash_attn_bwd(q, q, q, None, q, lse, delta=delta)


def test_delta_preprocess_respects_used_lengths_and_clears_accumulator():
    out = torch.full((2, 193, 3, 128), torch.nan, device="cuda", dtype=torch.bfloat16)
    lse = torch.randn((2, 3, 193), device="cuda")
    delta, dlse = torch.randn_like(lse), torch.randn_like(lse)
    used = torch.tensor([13, 129], device="cuda", dtype=torch.int32)
    dpsum = torch.full((2, 3, 256), torch.nan, device="cuda")
    lse_log2 = torch.empty_like(dpsum)
    dq_accum = torch.full((2, 3, 256 * 128), torch.nan, device="cuda")
    _bwd_preprocess(
        out,
        out,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        None,
        used,
        dlse,
        BFloat16,
        128,
        128,
        128,
        delta=delta,
    )
    for batch, length in enumerate((13, 129)):
        rounded = (length + 127) // 128 * 128
        torch.testing.assert_close(
            dpsum[batch, :, :length], (delta - dlse)[batch, :, :length], rtol=0, atol=0
        )
        assert torch.count_nonzero(dpsum[batch, :, length:rounded]) == 0
        assert torch.count_nonzero(dq_accum[batch, :, : rounded * 128]) == 0
