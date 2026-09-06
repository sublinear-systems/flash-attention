"""Preserve the FP32 output accumulator used by the softmax backward reduction."""

import pytest
import torch
from cutlass import BFloat16, Float16

from flash_attn.cute.interface import _bwd_preprocess, _flash_attn_bwd, _flash_attn_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="FP32 forward output currently requires SM90",
)


def _reference(q, k, v, q_lengths, k_lengths, scale, causal):
    """Dense FP64 oracle, including unequal causal lengths and fully masked rows."""
    outputs, normalizers = [], []
    q_start = k_start = 0
    group_size = q.shape[-2] // k.shape[-2]
    for q_len, k_len in zip(q_lengths, k_lengths):
        qi = q[q_start : q_start + q_len].transpose(0, 1)
        ki = k[k_start : k_start + k_len].transpose(0, 1).repeat_interleave(group_size, 0)
        vi = v[k_start : k_start + k_len].transpose(0, 1).repeat_interleave(group_size, 0)
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
        (True, [1, 129, 257], [1, 129, 257], 6, 2, 64, 64, True, True),
        (True, [7, 130], [129, 31], 8, 2, 128, 128, False, True),
        (True, [5, 257], [1, 193], 2, 2, 192, 128, True, False),
        (True, [128, 130], [128, 130], 2, 1, 256, 256, True, True),
        (False, [193, 193], [257, 257], 6, 2, 128, 128, False, True),
        (False, [33, 33], [17, 17], 2, 2, 128, 128, True, False),
    ],
)
def test_fp32_output_and_backward(
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
            "cu_seqlens_q": torch.tensor([0, *q_lengths], device="cuda", dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            ),
            "cu_seqlens_k": torch.tensor([0, *k_lengths], device="cuda", dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            ),
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
        q, k, v, softmax_scale=scale, causal=causal, pack_gqa=pack_gqa, return_lse=True, **kw
    )[:2]
    precise, precise_lse = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        pack_gqa=pack_gqa,
        return_lse=True,
        out=torch.empty_like(out, dtype=torch.float32),
        **kw,
    )[:2]
    assert precise.dtype == torch.float32
    torch.testing.assert_close(precise.to(dtype), out, rtol=0, atol=0)
    torch.testing.assert_close(precise_lse, lse, rtol=0, atol=0)
    finite_rows = ref_lse.isfinite()
    torch.testing.assert_close(
        lse[finite_rows].double(), ref_lse[finite_rows], rtol=1e-5, atol=1e-5
    )
    assert (precise.double() - ref_out).norm() / ref_out.norm() < 3e-3

    go = torch.randn_like(out)
    gl = torch.randn_like(lse).masked_fill(~finite_rows, 0)
    # Masking avoids the undefined 0 * (-inf) of the all-masked LSE rows.
    ref_loss = (ref_out * go.double()).sum() + (ref_lse[finite_rows] * gl[finite_rows]).sum()
    ref_grads = torch.autograd.grad(ref_loss, refs)
    grads = _flash_attn_bwd(
        q, k, v, precise, go, lse, scale, causal, 0.0, deterministic=True, dlse=gl, **kw
    )
    for actual, expected in zip(grads, ref_grads):
        assert torch.isfinite(actual).all()
        assert (actual.reshape_as(expected).double() - expected).norm() / expected.norm() < 1e-2

    # Revisit both cache variants after compiling the FP32 forward and preprocess.
    out_again, lse_again = _flash_attn_fwd(
        q, k, v, softmax_scale=scale, causal=causal, pack_gqa=pack_gqa, return_lse=True, **kw
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
    kw = {"cu_seqlens_q": cu, "cu_seqlens_k": cu, "max_seqlen_q": 256, "max_seqlen_k": 256}
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=1.0,
        causal=True,
        return_lse=True,
        out=torch.empty_like(v, dtype=torch.float32),
        **kw,
    )[:2]
    torch.testing.assert_close(out[-1], torch.full_like(out[-1], 32 + 0.25 / 256), rtol=0, atol=0)
    go = torch.zeros_like(v)
    go[-1] = 1 / 128

    def backward():
        return _flash_attn_bwd(q, k, v, out, go, lse, 1.0, True, 0.0, deterministic=True, **kw)

    grads = backward()
    # Every key is identical, so the output distribution is independent of Q.
    assert torch.count_nonzero(grads[0]) == 0
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out, captured_lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=1.0,
            causal=True,
            return_lse=True,
            out=torch.empty_like(v, dtype=torch.float32),
            **kw,
        )[:2]
        captured_grads = _flash_attn_bwd(
            q, k, v, captured_out, go, captured_lse, 1.0, True, 0.0, deterministic=True, **kw
        )
    for factor in (1, -2, 4):
        go[-1] = factor / 128
        graph.replay()
        expected = backward()
        for actual, reference in zip(captured_grads, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert torch.count_nonzero(captured_grads[0]) == 0


@pytest.mark.parametrize("dtype,cute_dtype", [(torch.bfloat16, BFloat16), (torch.float16, Float16)])
@pytest.mark.parametrize("has_dlse", [False, True])
def test_fp32_preprocess_masks_padded_value_dimension(dtype, cute_dtype, has_dlse):
    torch.manual_seed(912)
    # O/dO have 80 visible columns, with NaN in the physical padding. This
    # exercises multiple rows per thread and a value dimension needing masks.
    out_storage = torch.full((2, 37, 6, 96), torch.nan, device="cuda", dtype=torch.float32)
    dout_storage = torch.full_like(out_storage, torch.nan, dtype=dtype)
    out, dout = out_storage[..., :80], dout_storage[..., :80]
    out.copy_(torch.randn_like(out) + 32)
    dout.copy_(torch.randn_like(dout))
    lse = torch.randn((2, 6, 37), device="cuda")
    dlse = torch.randn_like(lse) if has_dlse else None
    dpsum = torch.full((2, 6, 128), torch.nan, device="cuda")
    lse_log2 = torch.empty_like(dpsum)
    _bwd_preprocess(
        out,
        dout,
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
    )
    expected = (out.double() * dout.double()).sum(-1).transpose(1, 2)
    if dlse is not None:
        expected = expected - dlse
    torch.testing.assert_close(dpsum[..., :37].double(), expected, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(lse_log2[..., :37], lse * 1.4426950408889634)
