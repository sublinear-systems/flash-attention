"""Gates for the SM90 additive-bias backward (rel_bias staging + dbias flush).

The structural path must reproduce the callback path: both add the same bf16
bias values to the recomputed scores in fp32, so dQ/dK/dV agree up to FMA
contraction order and GQA cross-head accumulation order, and the emitted dbias
equals the callback-scattered gradient up to its bf16 rounding.
Also pinned: NaN-canary store coverage (the flush writes EXACTLY the validity
window) and run-to-run bitwise determinism of dbias with the deterministic flag
both off and on.
"""
import pytest
import torch

import cutlass
import cutlass.cute as cute

from flash_attn.cute.interface import DBIAS_TAIL_PAD, _flash_attn_bwd, _flash_attn_fwd

COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]

pytestmark = pytest.mark.skipif(
    COMPUTE_CAPABILITY != 9, reason="rel_bias/dbias backward is SM90-only"
)


def _make_mods(W, S, H):
    """Per-lane callback pair equivalent to the structural path: bias read from a
    flat [B,S,H,W] tensor at (b, q, h, q-kv); grad scattered to the same layout."""

    @cute.jit
    def score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        rel = aux_tensors[0]
        n = cutlass.const_expr(cute.size(scores.shape))
        vf = cute.make_rmem_tensor(n, rel.element_type)
        base = (batch_idx[0] * (S * H) + head_idx[0]) * W
        for i in cutlass.range(n, unroll_full=True):
            qi = q_idx[i]
            d = qi - kv_idx[i]
            dc = cutlass.max(cutlass.min(d, W - 1), 0)
            vf[i] = rel[base + qi * (H * W) + dc]
            if (d < 0) | (d >= W):
                vf[i] = cutlass.BFloat16(0.0)
        return scores + vf.load().to(cutlass.Float32)

    @cute.jit
    def score_mod_bwd(grads, scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        dbias = aux_tensors[1]
        n = cutlass.const_expr(cute.size(grads.shape))
        dust = cute.size(dbias.shape) - 1
        base = (batch_idx[0] * (S * H) + head_idx[0]) * W
        for i in cutlass.range(n, unroll_full=True):
            qi = q_idx[i]
            d = qi - kv_idx[i]
            flat = base + qi * (H * W) + d
            flat = cutlass.Int32(cutlass.select_((d >= 0) & (d < W), flat, dust))
            dbias[flat] = grads[i]
        return grads

    score_mod_bwd.__needs_scores__ = False
    return score_mod, score_mod_bwd


PAD_L, PAD_R = 136, 320


def _coeffs(S, H, W, Wp, device, padded):
    off = PAD_L if padded else 0
    width = Wp if padded else W
    return torch.tensor(
        [S * H * width, width, H * width + 1, -1, off, 1, -1, 0, W],
        dtype=torch.int32, device=device,
    )


def _setup(S=1024, W=256, B=1, H=8, HKV=2, D=128, seed=0,
           full_attention=False):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, S, HKV, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, S, HKV, D, device=dev, dtype=torch.bfloat16)
    rel = 0.5 * torch.randn(B, S, H, W, device=dev, dtype=torch.bfloat16)
    dout = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    kw = (
        dict(softmax_scale=1.0 / D, causal=True)
        if full_attention
        else dict(softmax_scale=1.0 / D, window_size_left=W - 1,
                  window_size_right=0)
    )
    sm, smb = _make_mods(W, S, H)
    out, lse = _flash_attn_fwd(
        q, k, v, pack_gqa=False, score_mod=sm, aux_tensors=[rel.view(-1)],
        return_lse=True, **kw
    )[:2]
    return q, k, v, rel, dout, out, lse, kw, sm, smb


def _run_structural(q, k, v, rel, dout, out, lse, kw, deterministic=False, canary=False):
    B, S, H, W = rel.shape
    rel_padded = torch.nn.functional.pad(rel, (PAD_L, PAD_R)).view(-1)
    Wp = PAD_L + W + PAD_R
    fill = float("nan") if canary else 0.0
    # DBIAS_TAIL_PAD trailing scratch elements: the flush diverts out-of-support
    # lanes there (contents are scratch and NOT deterministic -- always slice)
    dbias = torch.full(
        (B * S * H * W + DBIAS_TAIL_PAD,), fill, device=rel.device, dtype=rel.dtype
    )
    dq, dk, dv = _flash_attn_bwd(
        q, k, v, out, dout, lse, deterministic=deterministic,
        rel_bias=rel_padded, rel_bias_coeffs=_coeffs(S, H, W, Wp, rel.device, True),
        dbias=dbias, dbias_coeffs=_coeffs(S, H, W, Wp, rel.device, False), **kw
    )
    return dq, dk, dv, dbias


def test_structural_matches_callbacks():
    q, k, v, rel, dout, out, lse, kw, sm, smb = _setup()
    B, S, H, W = rel.shape
    dbias_ref = torch.zeros(B * S * H * W + 1, device=rel.device, dtype=torch.float32)
    dq_a, dk_a, dv_a = _flash_attn_bwd(
        q, k, v, out, dout, lse, score_mod=sm, score_mod_bwd=smb,
        aux_tensors=[rel.view(-1), dbias_ref], **kw
    )
    dq_b, dk_b, dv_b, dbias_b = _run_structural(q, k, v, rel, dout, out, lse, kw)

    # same bf16 bias values added in fp32 -> same math up to FMA contraction order
    # (the two paths fuse the scale-multiply and bias-add differently), plus the
    # GQA dK/dV cross-head accumulation order
    torch.testing.assert_close(dk_a, dk_b, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dv_a, dv_b, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dq_a, dq_b, rtol=1e-2, atol=1e-2)
    # the structural dbias is the f16-converted dS the GEMMs consume; the callback
    # path scattered fp32 dS -- equal up to that one rounding
    N = B * S * H * W
    torch.testing.assert_close(
        dbias_b[:N].float(), dbias_ref[:N], rtol=1e-2, atol=1e-2
    )


def test_structural_full_attention_zeros_distances_beyond_bias_support():
    values = _setup(S=2048, W=1024, H=2, HKV=2, full_attention=True)
    q, k, v, rel, dout, out, lse, kw, sm, smb = values
    B, S, H, W = rel.shape
    dbias_ref = torch.zeros(B * S * H * W + 1, device=rel.device,
                            dtype=torch.float32)
    reference = _flash_attn_bwd(
        q, k, v, out, dout, lse, score_mod=sm, score_mod_bwd=smb,
        aux_tensors=[rel.view(-1), dbias_ref], **kw
    )
    structural = _run_structural(q, k, v, rel, dout, out, lse, kw)

    for actual, expected in zip(structural[:3], reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    N = B * S * H * W
    torch.testing.assert_close(
        structural[3][:N].float(), dbias_ref[:N], rtol=1e-2, atol=1e-2
    )


def test_structural_store_coverage():
    q, k, v, rel, dout, out, lse, kw, _, _ = _setup()
    B, S, H, W = rel.shape
    _, _, _, dbias = _run_structural(q, k, v, rel, dout, out, lse, kw, canary=True)
    written = ~torch.isnan(dbias[: B * S * H * W].view(B, S, H, W).float())
    valid = (torch.arange(W, device=rel.device)[None, :]
             <= torch.arange(S, device=rel.device)[:, None])[None, :, None, :]
    assert bool((written == valid).all()), (
        f"holes {(valid & ~written).sum().item()}, strays {(written & ~valid).sum().item()}"
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_structural_bitwise_dbias(deterministic):
    q, k, v, rel, dout, out, lse, kw, _, _ = _setup(seed=3)
    B, S, H, W = rel.shape
    r1 = _run_structural(q, k, v, rel, dout, out, lse, kw, deterministic=deterministic)
    r2 = _run_structural(q, k, v, rel, dout, out, lse, kw, deterministic=deterministic)
    N = B * S * H * W
    assert torch.equal(r1[3][:N], r2[3][:N]), "dbias must be run-to-run bitwise"
    if deterministic:
        # with the flag, the whole backward is bitwise (dK/dV semaphore-ordered
        # across q heads, dQ accumulation ordered)
        assert torch.equal(r1[0], r2[0]), "dQ must be bitwise under deterministic=True"
        assert torch.equal(r1[1], r2[1]) and torch.equal(r1[2], r2[2])
