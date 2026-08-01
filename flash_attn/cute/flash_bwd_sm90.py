import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute import FastDivmodDivisor
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from quack.sm90_utils import gemm_zero_init, gemm_w_idx

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,
    SingleTileVarlenScheduler,
)
from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwd
from flash_attn.cute.softmax import apply_score_mod_inner, apply_score_mod_bwd_inner
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.utils import AuxData
from flash_attn.cute.block_sparse_utils import (
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
    consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90,
)


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        deterministic: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        q_subtile_factor: cutlass.Constexpr[int] = 1,
        dQ_single_wg: bool = False,
        has_dbias: cutlass.Constexpr = False,
        has_rel_bias: cutlass.Constexpr = False,
    ):
        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size, or 64 when
        # dKV_swapAB=True (needed for WGMMA M=64 atom to partition the dK/dV
        # accumulator — otherwise head_dim in {144, 160, 176} crash the CuTe
        # tiled-mma partition).
        hdim_multiple_of = 64 if dKV_swapAB else 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.deterministic = deterministic
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage]
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_wg_mma = (self.num_threads // 128) - 1
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_wg_mma
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        # May be overridden in __call__ for varlen inputs.
        if qhead_per_kvhead > 1:
            assert self.same_hdim_kv, "GQA backward requires head_dim == head_dim_v"
            assert self.num_wg_mma == 2, "GQA backward assumes 2 warp groups"
        # These are tuned for speed
        # Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share
        # them and then shuffle to get the value whenever we need? This can reduce register
        # pressure when SdP_swapAB, where each thread needs to keep statistics for (kBlockM / 4)
        # rows. If !SdP_swapAB, each thread only needs to keep statistics for 2 rows.
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

        self.buffer_align_bytes = 1024

        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        # A score_mod_bwd whose joint graph does not use the pre-mod score (any additive
        # bias, and every mod whose derivative depends only on the indices) can set
        # `__needs_scores__ = False` to skip staging it: the kernel then keeps neither the
        # fp32 copy of the S tile nor the per-element reads of it.
        self.score_mod_bwd_needs_scores: cutlass.Constexpr = getattr(
            score_mod_bwd, "__needs_scores__", True
        )
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        # Additive-bias gradient sink: dS w.r.t. an additive attention bias IS the
        # bias's gradient, and the dS tile already sits in smem for the dQ/dK GEMMs
        # -- so the gradient is emitted as a plain smem->gmem copy after those GEMMs
        # drain, outside the register-critical softmax/GEMM region (no per-element
        # gmem scatter in the hot loop, no atomics, single writer per element).
        self.has_dbias = has_dbias
        # Additive bias in relative (distance) layout, staged through smem: each
        # m-iteration cooperatively vector-loads the tile's bias rows (contiguous
        # per q row) into sBias while GEMM1 runs, and the softmax recompute reads
        # bias from smem -- no per-element gmem gathers inside the register-critical
        # region. Requires score_mod=None; the affine row addressing contract is
        # documented on load_rel_bias/apply_rel_bias.
        self.has_rel_bias = has_rel_bias
        if has_rel_bias:
            assert score_mod is None and score_mod_bwd is None, (
                "rel_bias replaces score_mod in the backward"
            )
        # bias row window: the tile needs tile_n contiguous elements per q row plus
        # 4 elements of alignment slack (the 8B cp.async chunks start at the
        # 4-element floor of each row's minimum index). Two effects vs the old
        # +32/16B scheme: (1) the double-buffered sBias fits the SM90 smem budget
        # next to a 2-stage PdS, and (2) the smem row stride becomes 66 words, so
        # the four q-rows (spaced 2) that one WGMMA-fragment apply instruction
        # touches land 4 banks apart instead of all on one bank -- the old 80-word
        # stride made every bias read of the apply loop a 4-way conflict.
        self.rel_row_elems = self.tile_n + 4
        # bias tiles are produced by the otherwise-idle producer warps 1..3 through a
        # cp.async pipeline (double-buffered), so the consumers never run a CTA-wide
        # staging barrier; PdS drops to a single stage to fund the second bias buffer
        self.rel_bias_stage = 2
        self.q_subtile_factor = q_subtile_factor
        # SSA batching width for the score-mod calls on the SdP accumulator. Aux-reading
        # mods default to 1 because the backward cannot promise what `__vec_size__` promises
        # on the forward: with SdP_swapAB the accumulator is transposed, so a thread's
        # consecutive fragment elements walk q, not kv, and a mod that assumes "vec_size
        # adjacent kv indices for one q row" would read the wrong elements.
        #
        # `__bwd_vec_size__` is the opt-in for mods that index strictly per lane (one
        # scalar access per element, using the q_idx/kv_idx SSA lanes as given, no
        # adjacency assumption). Those mods are correct at any width and batching their
        # SSA ops cuts the per-element index and arithmetic overhead. It is deliberately
        # a separate attribute from `__vec_size__` so a mod written for the forward's
        # vectorized contract never silently changes backward behaviour.
        default_vec_size: cutlass.Constexpr = 1 if cutlass.const_expr(has_aux_tensors) else 4
        self.vec_size: cutlass.Constexpr = min(
            getattr(score_mod, "__bwd_vec_size__", default_vec_size),
            getattr(score_mod_bwd, "__bwd_vec_size__", default_vec_size),
        )
        if self.vec_size < 1:
            raise ValueError(f"__bwd_vec_size__ must be >= 1, got {self.vec_size}")
        self.qk_acc_dtype = Float32
        # dQ_single_wg: WG0 computes the full dQ GEMM, WG1 skips it.
        # Only valid for 2 MMA warp groups.
        # Credit: Ben Spector
        if dQ_single_wg:
            assert self.num_wg_mma == 2, "dQ_single_wg only supports 2 warp groups"
        self.num_wg_dQ = 1 if dQ_single_wg else self.num_wg_mma

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        Q_stage,
        num_threads,
        V_in_regs=False,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        else:
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
        # Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
        # The M dimension (tile_m) doesn't matter for the layout, only the K dimension
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
        self.sQ_layout, self.sdO_layout = [
            # Need to set major_mode_size (mms) to accommodate Q and Q.T
            sm90_utils.make_smem_layout(
                self.dtype,
                LayoutEnum.ROW_MAJOR,
                shape,
                stage,
                major_mode_size=mms,
            )
            for shape, stage, mms in [
                ((self.tile_m, self.tile_hdim), self.Q_stage, self.tile_hdim // wg_d_dKV),
                ((self.tile_m, self.tile_hdimv), self.dO_stage, self.tile_hdim // wg_d_dKV),
            ]
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ
        # Accomodate both K and K.T
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=None,
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        # There's only V, no V.T, so layout is normal
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), None
        )
        # Accomodate both S and S.T
        wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
        )
        # dQaccum R->S
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            # thr_layout
            cute.make_layout((self.num_threads_per_warp_group, self.num_wg_dQ)),
            cute.make_layout(128 // Float32.width),  # val_layout
        )
        # dKVaccum for GQA epilogue - reuses sV+sK memory recast as f32
        # TODO: assert that sVaccum and sKaccum don't overflow smem

    def _get_tiled_mma(self):
        maybe_swap_mn = lambda shape, swap: (shape[1], shape[0], *shape[2:]) if swap else shape
        # S = Q @ K.T, dP = dO @ V.T
        atom_layout_SdP = (self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1)
        tiler_mn_SdP = (self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1])
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_SdP, self.SdP_swapAB),
            tiler_mn=(64, tiler_mn_SdP[1] if not self.SdP_swapAB else tiler_mn_SdP[0]),
        )
        # dV = P.T @ dO, dK = dS.T @ Q
        atom_layout_dKV = (self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1)
        tiler_mn_dK = (self.tile_n // atom_layout_dKV[0], self.tile_hdim // atom_layout_dKV[1])
        tiler_mn_dV = (self.tile_n // atom_layout_dKV[0], self.tile_hdimv // atom_layout_dKV[1])
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=maybe_swap_mn(atom_layout_dKV, self.dKV_swapAB),
                tiler_mn=(64, tiler_mn_d[1] if not self.dKV_swapAB else tiler_mn_d[0]),
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        # dQ = dS @ K
        assert self.num_wg_dQ % self.AtomLayoutMdQ == 0
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_wg_dQ // self.AtomLayoutMdQ, 1)
        tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K if not self.dQ_swapAB else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN if not self.dQ_swapAB else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_dQ, self.dQ_swapAB),
            tiler_mn=(64, tiler_mn_dQ[1] if not self.dQ_swapAB else tiler_mn_dQ[0]),
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQaccum_layout, Float32),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        cosize_sBias = (
            self.rel_bias_stage * self.tile_m * self.rel_row_elems
            if const_expr(self.has_rel_bias)
            else 0
        )
        n_bias_mbar = 2 * self.rel_bias_stage if const_expr(self.has_rel_bias) else 0
        n_ds_mbar = 2 if const_expr(self.has_dbias and self.has_rel_bias) else 0
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.Q_stage], 128
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            mbar_ptr_Bias: cute.struct.MemRange[cutlass.Int64, n_bias_mbar]
            mbar_ptr_dS: cute.struct.MemRange[cutlass.Int64, n_ds_mbar]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sBias: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sBias], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        mdBias: Optional[cute.Tensor] = None,
        mdBiasParams: Optional[cute.Tensor] = None,
        mRelBias: Optional[cute.Tensor] = None,
        mRelBiasParams: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # For GQA (qhead_per_kvhead > 1), multiple Q heads accumulate into the same dK/dV,
        # so we need the float32 accum path + postprocess.
        # For varlen_k with qhead_per_kvhead == 1, we use ragged TMA tensors.
        self.varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None

        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )

        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        # Non-varlen inputs are (b, s, n, h), varlen inputs are (s, n, h).
        # We convert both to a seqlen-major view with head-dim second.
        # Each tensor may have different rank when Q is padded (seqused_q) but K/V are unpadded (cu_seqlens_k).
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        if const_expr(self.qhead_per_kvhead == 1):
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            # Accum tensors are (b, n, s*h) for non-varlen and (n, s*h) for varlen.
            accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        # Non-varlen stats are (b, n, s), varlen stats are (n, s).
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        # (batch, num_head, num_m_blocks, cluster_size) -> (num_m_blocks, cluster_size, num_head, batch)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])
        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=[2, 3, 1, 0]) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32

        REG_LIMIT = 504 if self.num_wg_mma == 2 else 512
        if const_expr(self.num_wg_mma == 2):
            if const_expr(self.num_wg_dQ == 1):
                self.num_mma_regs_wg0 = 256
                self.num_mma_regs_wg1 = 224
            else:
                self.num_mma_regs_wg0 = 240
                self.num_mma_regs_wg1 = 240
            self.num_mma_regs = self.num_mma_regs_wg0  # for backward compat
            self.num_producer_regs = 24
            assert (
                self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs <= REG_LIMIT
            )
        else:  # 3 warp groups
            self.num_mma_regs_wg0 = 160
            self.num_mma_regs_wg1 = 160
            self.num_mma_regs = 160
            self.num_producer_regs = 32
            assert self.num_mma_regs_wg0 * self.num_wg_mma + self.num_producer_regs <= REG_LIMIT

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        if const_expr(self.qhead_per_kvhead == 1):
            mdK_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdK
            )
            mdV_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdV
            )
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdK_tma,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdV_tma,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        if const_expr(mCuSeqlensK is not None or mSeqUsedK is not None):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        self.spt = (self.is_causal or self.is_local) and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mQ.shape[2]),
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batch
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.spt,
            head_swizzle=self.deterministic,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None and not self.has_rel_bias):
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            softmax_scale_log2 = LOG2_E

        fastdiv_mods = None
        if const_expr(aux_data.tensors is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1) else mdK,
            tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            mdBias,
            mdBiasParams,
            mRelBias,
            mRelBiasParams,
            window_size_left,
            window_size_right,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        mdBias: Optional[cute.Tensor] = None,
        mdBiasParams: Optional[cute.Tensor] = None,
        mRelBias: Optional[cute.Tensor] = None,
        mRelBiasParams: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_Bias = None
        if const_expr(self.has_rel_bias):
            # producer = the 64 threads of producer warps 2..3 (each issues its own
            # cp.asyncs and arrives at commit); consumers release via per-warp
            # elected arrives
            bias_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, 64
            )
            bias_consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                self.num_mma_threads // cute.arch.WARP_SIZE,
            )
            pipeline_Bias = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_Bias.data_ptr(),
                num_stages=self.rel_bias_stage,
                producer_group=bias_producer_group,
                consumer_group=bias_consumer_group,
                # per-warp elected release arrives (same shape as the Q/dO
                # pipelines): 256 per-thread arrives per iteration serialize on
                # the mbarrier and sit on the consumer critical path
                elect_one_release=True,
                defer_sync=True,
            )
        # A dedicated flusher warp (via this dS handoff pipeline) was measured at
        # 2.5x WORSE than the drained-bottom flush on the MMA threads: one warp
        # cannot move a full dS tile per iteration inside the producer warp group's
        # register budget (24 regs spills massively, and even a 232/232/40 split
        # leaves it issue-bound). Kept behind const_expr(False) as the measured
        # negative result; see flush_dbias_loop.
        pipeline_dS = None
        if const_expr(False):
            # dS-tile handoff to the flusher warp: the MMA threads produce (commit
            # after the sdS publish barrier), warp 3 consumes (flushes the tile to
            # the bias-gradient tensor, then releases so the next r2s may overwrite)
            pipeline_dS = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_dS.data_ptr(),
                num_stages=1,
                producer_group=cutlass.pipeline.CooperativeGroup(
                    cutlass.pipeline.Agent.Thread, self.num_mma_threads
                ),
                consumer_group=cutlass.pipeline.CooperativeGroup(
                    cutlass.pipeline.Agent.Thread, 32
                ),
                defer_sync=True,
            )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sBias = None
        if const_expr(self.has_rel_bias):
            sBias = storage.sBias.get_tensor(
                cute.make_layout(
                    (self.rel_bias_stage, self.tile_m, self.rel_row_elems),
                    stride=(
                        self.tile_m * self.rel_row_elems,
                        self.rel_row_elems,
                        1,
                    ),
                )
            )
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if const_expr(self.has_rel_bias):
                if warp_idx > 1:
                    # otherwise-idle producer warps 2..3 (warp 0 = TMA loads,
                    # warp 1 = dQaccum store): produce the bias tiles, running
                    # ahead through the double-buffered pipeline
                    self.load_rel_bias(
                        mRelBias,
                        mRelBiasParams,
                        sBias,
                        pipeline_Bias,
                        block_info,
                        SeqlenInfoCls,
                        TileSchedulerCls,
                    )


            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                    qhead_per_kvhead_divmod,
                )
            if warp_idx == 1:
                self.dQaccum_store(
                    mdQaccum,
                    sdQaccum,
                    block_info,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    mdQ_semaphore,
                )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdK_semaphore,
                mdV_semaphore,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sBias,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_data,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
                mdBias,
                mdBiasParams,
                mRelBias,
                mRelBiasParams,
                pipeline_Bias,
                pipeline_dS,
            )
            if const_expr(self.num_wg_dQ == self.num_wg_mma):
                # Both WGs compute dQ
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                # WG0 computes dQ, WG1 skips it
                warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
                if warp_idx_in_mma < 4:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                    self.mma(*mma_args, is_dQ_wg=True)
                else:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                    self.mma(*mma_args, is_dQ_wg=False)

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx
                    if const_expr(self.qhead_per_kvhead == 1)
                    else head_idx // qhead_per_kvhead_divmod
                )
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))

                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

                if const_expr(not self.use_block_sparsity):
                    total_m_block_cnt = m_block_max - m_block_min
                    process_tile = (
                        const_expr(not self.is_local and not self.is_varlen_q)
                        or m_block_min < m_block_max
                    )
                else:
                    total_m_block_cnt = get_total_q_block_count_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                    process_tile = total_m_block_cnt > Int32(0)

                if process_tile:
                    if const_expr(not self.use_block_sparsity):
                        first_m_block = m_block_min
                        pipeline_Q.producer_acquire(
                            producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"]
                        )
                        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                        load_Q(first_m_block, producer_state=producer_state_Q)
                        # Wait for bwd preprocess to finish writing LSE and dPsum
                        cute.arch.griddepcontrol_wait()
                        load_LSE(first_m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(
                            producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"]
                        )
                        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                        load_dO(first_m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            pipeline_Q.producer_acquire(producer_state_Q)
                            load_Q(m_block, producer_state=producer_state_Q)
                            load_LSE(m_block, producer_state=producer_state_Q)
                            producer_state_dO_cur = (
                                producer_state_dO
                                if const_expr(self.Q_stage != self.dO_stage)
                                else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(producer_state_dO_cur)
                            load_dO(m_block, producer_state=producer_state_dO_cur)
                            load_dPsum(m_block, producer_state=producer_state_dO_cur)
                            producer_state_Q.advance()
                            producer_state_dO.advance()
                    else:
                        producer_state_Q, producer_state_dO = produce_block_sparse_q_loads_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            producer_state_Q,
                            producer_state_dO,
                            pipeline_Q,
                            pipeline_dO,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            load_LSE,
                            load_dPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            Q_stage_eq_dO_stage=(self.Q_stage == self.dO_stage),
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def apply_score_mod(
        self,
        acc_S: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        # [NOTE] SdP_swapAB: swapAB transposes the tile, so use (n, m) indexing
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            # The backward never runs Pack-GQA (the entry point forces pack_gqa=False),
            # so q_idx is a plain query index and head_idx is a plain query-head index.
            # Passing qhead_per_kvhead here would make the inner helper apply the
            # Pack-GQA unpacking transform to unpacked indices, handing score mods
            # q_idx // qhead_per_kvhead and head_idx * qhead_per_kvhead + q_idx %
            # qhead_per_kvhead instead of the real coordinates.
            qhead_per_kvhead=1,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor: cute.Tensor,
        score_tensor: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            tScS,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            # The backward never runs Pack-GQA (the entry point forces pack_gqa=False),
            # so q_idx is a plain query index and head_idx is a plain query-head index.
            # Passing qhead_per_kvhead here would make the inner helper apply the
            # Pack-GQA unpacking transform to unpacked indices, handing score mods
            # q_idx // qhead_per_kvhead and head_idx * qhead_per_kvhead + q_idx %
            # qhead_per_kvhead instead of the real coordinates.
            qhead_per_kvhead=1,
            transpose_indices=self.SdP_swapAB,
            needs_scores=self.score_mod_bwd_needs_scores,
        )

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sBias: Optional[cute.Tensor],
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        mdBias: Optional[cute.Tensor] = None,
        mdBiasParams: Optional[cute.Tensor] = None,
        mRelBias: Optional[cute.Tensor] = None,
        mRelBiasParams: Optional[cute.Tensor] = None,
        pipeline_Bias: Optional[cutlass.pipeline.PipelineAsync] = None,
        pipeline_dS: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_idx_dQ = warp_group_idx if const_expr(self.num_wg_dQ > 1) else 0
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(wg_idx_dQ))
        # S = Q @ K.T
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB
        )
        mma_qk_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=self.SdP_swapAB
        )
        # dP = dO @ V.T
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB
        )
        mma_dov_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tdPrV, swap_AB=self.SdP_swapAB
        )
        # dV += P.T @ dO
        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(
                gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=self.dKV_swapAB
            )
        else:
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        # dK += dS.T @ Q
        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn = partial(
                gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=self.dKV_swapAB
            )
        else:
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        # dQ = dS @ K
        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        if const_expr(is_dQ_wg):
            _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
            )
            mma_dsk_fn = partial(
                gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=self.dQ_swapAB,
            )

        # Smem copy atom tiling for P/dS R2S
        copy_P_r2s = None
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        if const_expr(sP is not None):
            sP_cpy = sP if const_expr(not self.SdP_swapAB) else sPt
            copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sP_cpy,
                tidx,
                transpose=self.SdP_swapAB,
                position_independent=True,
                major_mode_size=mms_PdS,
            )
        sdS_cpy = sdS if const_expr(not self.SdP_swapAB) else sdSt
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=mms_PdS,
        )

        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        # When shuffle=True, rows are distributed across 8 quads (4 threads each) within a warp.
        # Each thread loads only ceil(num_rows/8) values;
        shfl_copy = copy_utils.tiled_copy_1d(sLSE.element_type, num_threads=8, num_copy_elems=2)
        if const_expr(self.shuffle_LSE):
            tLSEsLSE = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsLSE)
            # ((2, 1), 1, 2) -> (((2, 1), 1), 2)
            tLSEsLSE = cute.group_modes(tLSEsLSE, 0, 2)
        if const_expr(self.shuffle_dPsum):
            tLSEsdPsum = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsdPsum)
            tLSEsdPsum = cute.group_modes(tLSEsdPsum, 0, 2)

        tdQsdQaccum = None
        if const_expr(is_dQ_wg):
            smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_P_r2s=copy_P_r2s,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQaccum=tdQsdQaccum,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            # acc_dV=acc_dV,
            # acc_dK=acc_dK,
            is_dQ_wg=is_dQ_wg,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        consumer_state_Bias = None
        if const_expr(self.has_rel_bias):
            consumer_state_Bias = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.rel_bias_stage
            )
        producer_state_dS = None
        if const_expr(self.has_dbias and self.has_rel_bias):
            producer_state_dS = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, 1
            )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            recompute_fastdiv_mods_q = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if const_expr(fastdiv_mods is not None and fastdiv_mods[0] is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask = AttentionMaskCls(seqlen)
            score_mod_fn = partial(
                self.apply_score_mod,
                thr_mma_SdP=thr_mma_SdP,
                softmax_scale=softmax_scale,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_bwd_fn = partial(
                self.apply_score_mod_bwd,
                thr_mma_SdP=thr_mma_SdP,
                softmax_scale=softmax_scale,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn_cur = partial(
                score_mod_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            score_mod_bwd_fn_cur = partial(
                score_mod_bwd_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            rel_bias_apply_fn_cur = None
            if const_expr(self.has_rel_bias):
                rel_bias_apply_fn_cur = partial(
                    self.apply_rel_bias,
                    thr_mma_SdP=thr_mma_SdP,
                    sBias=sBias,
                    mRelBiasParams=mRelBiasParams,
                    softmax_scale=softmax_scale,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    n_block=n_block,
                )
            dbias_flush_fn_cur = None
            if const_expr(self.has_dbias):
                dbias_flush_fn_cur = partial(
                    self.flush_dbias,
                    sdS=sdS,
                    mdBias=mdBias,
                    mdBiasParams=mdBiasParams,
                    tidx=tidx,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    n_block=n_block,
                    seqlen_info=seqlen,
                )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
            else:
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        mask_mod=self.mask_mod,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                    )
                    dKV_accumulate = False
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        if const_expr(self.has_rel_bias):
                            (
                                consumer_state_Q,
                                consumer_state_dO,
                                consumer_state_Bias,
                                producer_state_dS,
                            ) = mma_one_m_block_all(
                                m_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mask_fn=mask_fn,
                                score_mod_fn=score_mod_fn_cur,
                                score_mod_bwd_fn=score_mod_bwd_fn_cur,
                                dbias_flush_fn=dbias_flush_fn_cur,
                                rel_bias_apply_fn=rel_bias_apply_fn_cur,
                                pipeline_Bias=pipeline_Bias,
                                consumer_state_Bias=consumer_state_Bias,
                                pipeline_dS=pipeline_dS,
                                producer_state_dS=producer_state_dS,
                                dKV_accumulate=dKV_accumulate,
                                is_last_m=m_block == m_block_max - 1,
                            )
                        else:
                            consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                                m_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mask_fn=mask_fn,
                                score_mod_fn=score_mod_fn_cur,
                                score_mod_bwd_fn=score_mod_bwd_fn_cur,
                                dbias_flush_fn=dbias_flush_fn_cur,
                                dKV_accumulate=dKV_accumulate,
                                is_last_m=m_block == m_block_max - 1,
                            )
                        dKV_accumulate = True
                else:
                    consumer_state_Q, consumer_state_dO = consume_block_sparse_mma_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mma_one_m_block_all,
                        mask,
                        self.mask_mod,
                        is_causal=self.is_causal,
                        is_local=self.is_local,
                        thr_mma_SdP=thr_mma_SdP,
                        score_mod_fn=score_mod_fn_cur,
                        score_mod_bwd_fn=score_mod_bwd_fn_cur,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                    )

                if const_expr(self.qhead_per_kvhead == 1):
                    acc_dK.store(acc_dK.load() * softmax_scale)
                self.epilogue_dKV(
                    acc_dV,
                    mdV,
                    sV,
                    acc_dK,
                    mdK,
                    sK,
                    seqlen,
                    tma_atom_dK,
                    tma_atom_dV,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                    qhead_per_kvhead_divmod,
                    mdK_semaphore,
                    mdV_semaphore,
                )
            else:
                # KV tile with zero Q blocks produces no dK/dV; write zeros.
                if const_expr(self.use_block_sparsity or self.is_local or self.is_varlen_q):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        """Retrieve the statistic for a given accumulator row.

        When shuffle=False, direct register indexing.
        When shuffle=True, warp shuffle from the thread group that holds the value.
        """
        if const_expr(not shuffle):
            return tSrS[row]
        # tSrS: (((2, 1), 1), 1)), distributed across 8 threads in the warp
        vecsize = cute.size(tSrS, mode=[0, 0])  # 2
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        # register index: 0, 1, 0, 1, ..., 2, 3, 2, 3, ...
        return utils.shuffle_sync(tSrS[idx0 + idx1 * vecsize], offset=off * 4 + (lane % 4))

    @cute.jit
    def load_rel_bias(
        self,
        mRelBias: cute.Tensor,
        mRelBiasParams: cute.Tensor,
        sBias: cute.Tensor,
        pipeline_Bias: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """Bias-tile producer, run on the otherwise-idle producer warps 2..3
        (warp 0 issues the TMA loads, warp 1 stores dQaccum).

        Walks the same (tile, m-iteration) sequence as the consumers and stages each
        iteration's bias rows into the double-buffered sBias through a cp.async
        pipeline: producer_acquire -> 8B cp.asyncs -> producer_commit (per-thread
        cp.async mbarrier arrive). Consumers only mbarrier-wait -- no CTA barrier,
        so the MMA warp groups keep their skew.

        The bias is additive in relative (distance) layout: flat = P0*b + P1*h +
        P2*q + P3*kv + P4 with P3 == -1, so for a fixed q row the tile's tile_n
        elements are CONTIGUOUS (descending in kv). Each row window is fetched from
        the 4-element-aligned floor of its minimum index; the reader reconstructs
        the alignment offset arithmetically. The caller pads the bias tensor so
        every window read is in bounds.

        Thread mapping: 32 threads walk row-major (row, chunk) pairs -- consecutive
        threads fetch consecutive 8B chunks, coalesced within each row.
        """
        chunks_per_row = cutlass.const_expr(self.rel_row_elems // 4)
        n_chunks = cutlass.const_expr(self.tile_m * chunks_per_row)
        passes = cutlass.const_expr((n_chunks + 63) // 64)
        copy_atom_64 = cute.make_copy_atom(
            cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=64
        )
        tidx = cute.arch.thread_idx()[0] - 64  # producer warps 2..3 -> [0, 64)
        P0 = mRelBiasParams[0]
        P1 = mRelBiasParams[1]
        P2 = mRelBiasParams[2]
        P4 = mRelBiasParams[4]
        # bias tensors are kernel inputs: under programmatic dependent launch they
        # may still be written by the previous kernel at this point
        cute.arch.griddepcontrol_wait()
        producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.rel_bias_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )
            if process_tile:
                kv_hi = n_block * self.tile_n + self.tile_n - 1
                c1 = P0 * batch_idx + P1 * head_idx - kv_hi + P4
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    pipeline_Bias.producer_acquire(producer_state)
                    buf = producer_state.index
                    sbase = sBias.iterator + buf * (self.tile_m * self.rel_row_elems)
                    for k in cutlass.range(passes, unroll=1):
                        g = tidx + k * 64
                        if g < n_chunks:
                            r = g // chunks_per_row
                            ch = g - r * chunks_per_row
                            q = cutlass.min(
                                m_block * self.tile_m + r, seqlen.seqlen_q - 1
                            )
                            fmin = c1 + P2 * q
                            a0 = fmin - (fmin & 3)
                            gsrc_ptr = cute.make_ptr(
                                self.dtype,
                                (mRelBias.iterator + a0 + ch * 4).toint(),
                                mRelBias.memspace,
                                assumed_align=8,
                            )
                            sdst_ptr = cute.make_ptr(
                                self.dtype,
                                (sbase + r * self.rel_row_elems + ch * 4).toint(),
                                sBias.memspace,
                                assumed_align=8,
                            )
                            cute.copy(
                                copy_atom_64,
                                cute.make_tensor(gsrc_ptr, cute.make_layout(4)),
                                cute.make_tensor(sdst_ptr, cute.make_layout(4)),
                            )
                    # cp.async completion arrive: orders the async-proxy smem
                    # writes for the consumers' acquire, which a plain arrive after
                    # a wait_group does NOT (async->generic proxy visibility)
                    pipeline_Bias.producer_commit(producer_state)
                    producer_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def apply_rel_bias(
        self,
        acc_S: cute.Tensor,
        m_block,
        buf,
        thr_mma_SdP: cute.ThrMma,
        sBias: cute.Tensor,
        mRelBiasParams: cute.Tensor,
        softmax_scale,
        batch_idx,
        head_idx,
        n_block,
    ):
        """acc_S = acc_S * softmax_scale + bias, bias read from the smem-staged tile.

        Index math mirrors load_rel_bias: smem col = (fmin(q) mod 4) + (kv_hi - kv).
        Reads out of the bias's logical support hit the caller's padding and are
        masked to -inf downstream (application is premask), exactly like a premask
        score modification.
        """
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)
        if cutlass.const_expr(self.SdP_swapAB):
            q_pos = cutlass.const_expr(1)
            kv_pos = cutlass.const_expr(0)
        else:
            q_pos = cutlass.const_expr(0)
            kv_pos = cutlass.const_expr(1)
        P0 = mRelBiasParams[0]
        P1 = mRelBiasParams[1]
        P2 = mRelBiasParams[2]
        P4 = mRelBiasParams[4]
        kv_hi = n_block * self.tile_n + self.tile_n - 1
        c1 = P0 * batch_idx + P1 * head_idx - kv_hi + P4
        m0 = m_block * self.tile_m
        n_vals = cutlass.const_expr(cute.size(acc_S.shape))
        for i in cutlass.range(n_vals, unroll_full=True):
            q = tScS[i][q_pos]
            kv = tScS[i][kv_pos]
            fmin = c1 + P2 * q
            col = (fmin & 3) + (kv_hi - kv)
            val = sBias[buf, q - m0, col]
            acc_S[i] = acc_S[i] * softmax_scale + cutlass.Float32(val)

    @cute.jit
    def flush_dbias_loop(
        self,
        mdBias: cute.Tensor,
        mdBiasParams: cute.Tensor,
        sdS: cute.Tensor,
        pipeline_dS: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """Bias-gradient flusher, run on the otherwise-idle producer warp 3.

        Trails the MMA warp groups through the same (tile, m-iteration) walk: waits
        for each iteration's converted dS tile (committed by the MMA threads after
        the sdS publish barrier), copies it to the bias-gradient tensor with the
        affine index/validity convention of flush_dbias, and releases the stage so
        the next r2s may overwrite it. This takes the flush entirely off the MMA
        critical path -- with a single PdS stage the bottom-of-iteration flush
        otherwise serializes against the next iteration's r2s.

        Thread mapping: 32 threads walk kv columns in two coalesced 64-wide
        stripes per q row.
        """
        P0 = mdBiasParams[0]
        P1 = mdBiasParams[1]
        P2 = mdBiasParams[2]
        P3 = mdBiasParams[3]
        P4 = mdBiasParams[4]
        P5 = mdBiasParams[5]
        P6 = mdBiasParams[6]
        P7 = mdBiasParams[7]
        P8 = mdBiasParams[8]
        dust = cute.size(mdBias.shape) - 1
        stripes = cutlass.const_expr(self.tile_n // 32)
        tidx = cute.arch.thread_idx()[0] - 96  # producer warp 3 -> [0, 32)
        consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )
            if process_tile:
                base0 = P0 * batch_idx + P1 * head_idx + P4
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    pipeline_dS.consumer_wait(
                        consumer_state, pipeline_dS.consumer_try_wait(consumer_state)
                    )
                    for st in cutlass.range_constexpr(stripes):
                        c = tidx + st * 32
                        kv = n_block * self.tile_n + c
                        base = base0 + P3 * kv
                        dcol = P6 * kv + P7
                        kv_ok = kv < seqlen.seqlen_k
                        for r in cutlass.range(self.tile_m, unroll=4):
                            q = m_block * self.tile_m + r
                            d = P5 * q + dcol
                            ok = kv_ok & (d >= 0) & (d < P8) & (q < seqlen.seqlen_q)
                            tgt = cutlass.Int32(
                                cutlass.select_(ok, base + P2 * q, dust)
                            )
                            mdBias[tgt] = sdS[r, c, 0]
                    pipeline_dS.consumer_release(consumer_state)
                    consumer_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def flush_dbias(
        self,
        m_block,
        smem_idx,
        sdS: cute.Tensor,
        mdBias: cute.Tensor,
        mdBiasParams: cute.Tensor,
        tidx: Int32,
        batch_idx,
        head_idx,
        n_block,
        seqlen_info: SeqlenInfoQK,
    ):
        """Emit the dS tile from smem as an additive-bias gradient.

        mdBiasParams (Int32[9]) describes an affine flat index and a validity window:
            flat = P0*b + P1*h + P2*q + P3*kv + P4, valid iff 0 <= P5*q + P6*kv + P7 < P8
        Invalid lanes (outside the bias's support, or rows/cols beyond seqlen) are
        diverted branchlessly to the dustbin slot -- mdBias's LAST element, which the
        caller allocates and ignores. Each valid element has exactly one writer
        (tiles are disjoint, the grid is per (batch, q-head)), so the store is
        deterministic with no atomics. Values are the same converted dS the dQ/dK
        GEMMs consume. Consecutive threads walk consecutive kv columns, so every
        store instruction is a coalesced row stripe.
        """
        P0 = mdBiasParams[0]
        P1 = mdBiasParams[1]
        P2 = mdBiasParams[2]
        P3 = mdBiasParams[3]
        P4 = mdBiasParams[4]
        P5 = mdBiasParams[5]
        P6 = mdBiasParams[6]
        P7 = mdBiasParams[7]
        P8 = mdBiasParams[8]
        dust = cute.size(mdBias.shape) - 1
        rows_per_pass = cutlass.const_expr(self.num_mma_threads // self.tile_n)
        c = tidx % self.tile_n
        r0 = tidx // self.tile_n
        kv = n_block * self.tile_n + c
        base = P0 * batch_idx + P1 * head_idx + P3 * kv + P4
        dcol = P6 * kv + P7
        kv_ok = kv < seqlen_info.seqlen_k
        for k in cutlass.range(self.tile_m // rows_per_pass, unroll=4):
            r = r0 + k * rows_per_pass
            q = m_block * self.tile_m + r
            d = P5 * q + dcol
            ok = kv_ok & (d >= 0) & (d < P8) & (q < seqlen_info.seqlen_q)
            val = sdS[r, c, smem_idx]
            # predicated store, no dustbin: out-of-band lanes previously stored to
            # a single shared dust address, and same-address stores from many warps
            # serialize on one L2 slice (measured: an all-dust flush is ~2.5x
            # slower than the real scatter). The dust slot stays in the contract
            # for the callback path; the structural flush simply skips the store.
            if ok:
                mdBias[cutlass.Int32(base + P2 * q)] = val

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        copy_P_r2s: Optional[Callable],
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tdQsdQaccum: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        mask_fn: Optional[Callable] = None,
        score_mod_fn: Optional[Callable] = None,
        score_mod_bwd_fn: Optional[Callable] = None,
        dbias_flush_fn: Optional[Callable] = None,
        rel_bias_apply_fn: Optional[Callable] = None,
        pipeline_Bias: Optional[cutlass.pipeline.PipelineAsync] = None,
        consumer_state_Bias=None,
        pipeline_dS: Optional[cutlass.pipeline.PipelineAsync] = None,
        producer_state_dS=None,
        dKV_accumulate: Boolean = True,
        is_last_m: Boolean = True,
    ):
        consumer_state_dO_cur = (
            consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        # (1) [GEMM 1] S = Q @ K^T
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        # If shuffle_LSE, OOB reads are OK since sLSE is already padded
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        # (2) [GEMM 2] dP = dO @ V.T
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)

        acc_S_pre = None
        if const_expr(self.score_mod_bwd is not None and self.score_mod_bwd_needs_scores):
            acc_S_pre = cute.make_fragment_like(acc_S)
            cute.autovec_copy(acc_S, acc_S_pre)

        if const_expr(self.score_mod is not None):
            score_mod_fn(acc_S, m_block=m_block)

        if const_expr(rel_bias_apply_fn is not None):
            # per-thread mbarrier wait on the producer warps' staged tile (release
            # semantics make their cp.asyncs visible) -- NO CTA barrier, the MMA
            # warp groups keep their skew; release lets the producers refill the
            # buffer two iterations out
            pipeline_Bias.consumer_wait(
                consumer_state_Bias, pipeline_Bias.consumer_try_wait(consumer_state_Bias)
            )
            rel_bias_apply_fn(acc_S, m_block=m_block, buf=consumer_state_Bias.index)
            pipeline_Bias.consumer_release(consumer_state_Bias)
            consumer_state_Bias.advance()



        # (3) [Pointwise 1] P = exp(S - LSE)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])

        # Convert P from f32 -> f16
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        # R2S for P
        if const_expr(not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                PdS_barrier.arrive_and_wait()
            copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

        if const_expr(self.score_mod_bwd is not None):
            score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)

        # Convert dS from f32 -> f16
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)

        # If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure (1) P is written in case of !mma_dkv_is_rs and
        # (2) dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)):
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()

        if const_expr(pipeline_dS is not None):
            # wait until the flusher warp released the (single) dS stage
            pipeline_dS.producer_acquire(producer_state_dS)
        # R2S for dS
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)

        # (5) [GEMM 3] dV += P.T @ dO
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )
        else:
            mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        if const_expr(pipeline_dS is not None):
            # tile-wide visible: hand the dS stage to the flusher warp
            pipeline_dS.producer_commit(producer_state_dS)
            producer_state_dS.advance()

        if const_expr(is_dQ_wg):
            # (6) [GEMM 4] dQ = dS @ K
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)  # release dO as dV mma is done

            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)

            # dQ R2S: wait for dQaccum_store to free the smem buffer, then write dQ to smem
            # When dQ_single_wg, only WG0 enters here so warp_group_idx == 0
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            tdQrdQaccum_flat = cute.make_tensor(
                acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
            )
            cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )

            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
        else:
            # dQ_single_wg: WG1 skips dQ, only does dV wait + dK
            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)

        if const_expr(dbias_flush_fn is not None and pipeline_dS is None):
            # no flusher warp: drained-bottom flush, the only spill-safe MMA-side home
            dbias_flush_fn(m_block=m_block, smem_idx=smem_idx_PdS)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        if const_expr(rel_bias_apply_fn is not None):
            # pipeline states have value semantics across cute.jit calls: hand the
            # advanced states back to the caller like the Q/dO states
            return (
                consumer_state_Q,
                consumer_state_dO,
                consumer_state_Bias,
                producer_state_dS,
            )
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if const_expr(self.qhead_per_kvhead == 1):
            mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen_k)[
                None, None, head_idx
            ]
            mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen_k)[
                None, None, head_idx
            ]
            gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
            gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
            store_dK, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
            )
            store_dV, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
            )
            sdV = sV if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sV)
            sdK = sK if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sK)
            copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dV,
                sdV,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dK,
                sdK,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dV_r2s(acc_dV, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dK_r2s(acc_dK, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
        else:
            deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
            sdKaccum_shape0 = self.tile_n * self.tile_hdim // self.num_wg_mma
            sdVaccum_shape0 = self.tile_n * self.tile_hdimv // self.num_wg_mma
            sdKaccum_layout = cute.make_layout((sdKaccum_shape0, self.num_wg_mma))
            sdVaccum_layout = cute.make_layout((sdVaccum_shape0, self.num_wg_mma))
            head_idx_kv = head_idx // qhead_per_kvhead_divmod
            if const_expr(deterministic_KV):
                assert mdK_semaphore is not None
                assert mdV_semaphore is not None
                mdK_semaphore_cur = mdK_semaphore[n_block, None, head_idx_kv, batch_idx]
                mdV_semaphore_cur = mdV_semaphore[n_block, None, head_idx_kv, batch_idx]
                lock_value = head_idx % self.qhead_per_kvhead
            mdKaccum_cur = seqlen.offset_batch_K(
                mdK, batch_idx, dim=2, padded=True, multiple=self.tile_hdim
            )[None, head_idx_kv]
            mdVaccum_cur = seqlen.offset_batch_K(
                mdV, batch_idx, dim=2, padded=True, multiple=self.tile_hdimv
            )[None, head_idx_kv]
            gdKaccum_ = cute.local_tile(mdKaccum_cur, (self.tile_n * self.tile_hdim,), (n_block,))
            gdKaccum = cute.flat_divide(gdKaccum_, (sdKaccum_shape0,))
            gdVaccum_ = cute.local_tile(mdVaccum_cur, (self.tile_n * self.tile_hdimv,), (n_block,))
            gdVaccum = cute.flat_divide(gdVaccum_, (sdVaccum_shape0,))
            # These two overlap each other
            sVaccum_ptr = cute.recast_ptr(sV.iterator, dtype=Float32)
            sdKaccum = cute.make_tensor(sVaccum_ptr, sdKaccum_layout)
            sdVaccum = cute.make_tensor(sVaccum_ptr, sdVaccum_layout)
            tiled_copy_dKVaccum_r2s = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                cute.make_layout((self.num_threads_per_warp_group, self.num_wg_mma)),
                cute.make_layout(128 // Float32.width),
            )
            thr_copy_dKVaccum_r2s = tiled_copy_dKVaccum_r2s.get_slice(tidx)
            tdKsdKaccum = thr_copy_dKVaccum_r2s.partition_D(sdKaccum)
            tdVsdVaccum = thr_copy_dKVaccum_r2s.partition_D(sdVaccum)

            read_flag = const_expr(not deterministic_KV)
            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.wait_eq(mdK_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdKrdKaccum_flat = cute.make_tensor(acc_dK.iterator, tdKsdKaccum.shape)
            cute.autovec_copy(tdKrdKaccum_flat, tdKsdKaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKaccum[None, wg_idx].iterator,
                            gdKaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dKacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()

            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.arrive_inc(mdK_semaphore_cur.iterator, tidx, 0, 1)
                barrier.wait_eq(mdV_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdVrdVaccum_flat = cute.make_tensor(acc_dV.iterator, tdVsdVaccum.shape)
            cute.autovec_copy(tdVrdVaccum_flat, tdVsdVaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdVaccum[None, wg_idx].iterator,
                            gdVaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dVacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()
            if const_expr(deterministic_KV):
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                barrier.arrive_inc(mdV_semaphore_cur.iterator, tidx, 0, 1)

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # warp-local thread index (dQaccum_store runs on warp 1, global tidx 32-63)
        warp_local_tidx = tidx % cute.arch.WARP_SIZE
        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            else:
                mdQaccum_cur = cute.domain_offset(
                    (seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, head_idx]
                )
            # ((M * K / num_wg_dQ, num_wg_dQ), num_m_blocks)
            gdQaccum = cute.local_tile(
                mdQaccum_cur,
                (
                    cute.make_layout(
                        (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
                    ),
                ),
                (None,),
            )

            if const_expr(mdQ_semaphore is not None):
                # mdQ_semaphore is (num_m_blocks, cluster_size, num_head, batch) after transpose
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min
            else:
                total_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    for iter_idx in cutlass.range(loop_count, unroll=1):
                        m_block = m_block_min + iter_idx
                        m_block_safe = m_block

                        num_dQ_chunks = self.num_wg_dQ
                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            if const_expr(not self.deterministic):
                                # If deterministic, we already waited at the end of the prev iter
                                cute.arch.cp_async_bulk_wait_group(
                                    num_dQ_chunks - 1 - warp_group_idx, read=read_flag
                                )
                            cute.arch.barrier_arrive(
                                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )

                        # Semaphore acquire: wait for prior n_blocks to finish writing this m_block
                        if const_expr(self.deterministic):
                            if const_expr(self.spt):
                                _, n_block_max_for_m_block = block_info.get_n_block_min_max(
                                    seqlen, m_block_safe
                                )
                                lock_value = n_block_max_for_m_block - 1 - n_block
                            else:
                                lock_value = n_block
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,  # flag_offset
                                lock_value,
                            )

                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )
                            with cute.arch.elect_one():
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQaccum[None, warp_group_idx].iterator,
                                    gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
                                    self.tma_copy_bytes["dQ"],
                                )
                            cute.arch.cp_async_bulk_commit_group()

                        # Semaphore release: signal that this n_block is done with this m_block
                        if const_expr(self.deterministic):
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            barrier.arrive_inc(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,  # flag_offset
                                1,
                            )
                else:
                    if const_expr(self.deterministic):
                        # Ordered by dq_write_order ranks (each n_block's position in the
                        # target m_block's contributor list); no skip-signaling needed --
                        # ranks only count actual contributors.
                        dQaccum_store_block_sparse_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            sdQaccum,
                            gdQaccum,
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                            num_dQ_warp_groups=self.num_wg_dQ,
                            num_threads_per_warp_group=self.num_threads_per_warp_group,
                            tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                            deterministic=True,
                            mdQ_semaphore_cur=mdQ_semaphore_cur,
                            warp_local_tidx=warp_local_tidx,
                        )
                    else:
                        dQaccum_store_block_sparse_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            sdQaccum,
                            gdQaccum,
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                            num_dQ_warp_groups=self.num_wg_dQ,
                            num_threads_per_warp_group=self.num_threads_per_warp_group,
                            tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                        )

            # For local masking + deterministic (non-spt): signal remaining m_blocks
            # that this n_block won't visit, so they don't deadlock waiting.
            if const_expr(
                self.deterministic and not self.spt and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        warp_local_tidx,
                        0,  # flag_offset
                        1,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)
