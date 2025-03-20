#!/usr/bin/env python
# tcast/tritoncast.py: triton methods for casting
# SPDX-License-Identifier: MIT
# ruff: noqa: D103

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch
import triton
import triton.language as tl

from .common import FP8_DTYPES, STD_DTYPES, CastMode, Modes, RoundMode, ScaleMode
from .snippets import get_exponent, get_shared_exponent, modify_exponent, quantize_float
from .tensor import Tensor
from .utils import is_triton_available

#####
##### NOTE: This is a train wreck. Still deciding what to put here and what to put
##### in the triton snippets. The snippets are more general and can be used for
##### triton kernels. This just needs to implement cast, upcast, and vcast.
#####


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

 # fmt: off
@triton.jit
def make_block_ptr(pidm, pidn, tensor, shape0, shape1, stride0, stride1, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    """Create a block pointer for the given tensor."""
    block_row = pidm * TILE_M
    block_col = pidn * TILE_N
    offset = block_row * stride0 + block_col * stride1
    return tl.make_block_ptr(
        base=tensor + offset,
        shape=(shape0, shape1),
        strides=(stride0, stride1),
        offsets=(0, 0),
        block_shape=(TILE_M, TILE_N),
        order=(0, 1),
    )


@triton.jit
def make_all_block_ptrs(
    pid: int,
    i_ten: tl.tensor, i_shape0: int, i_shape1: int, i_stride0: int, i_stride1: int,
    o_ten: tl.tensor, o_shape0: int, o_shape1: int, o_stride0: int, o_stride1: int,
    s_ten: tl.tensor, s_shape0: int, s_shape1: int, s_stride0: int, s_stride1: int,
    # z_ten: tl.tensor, z_shape0: int, z_shape1: int, z_stride0: int, z_stride1: int,
    # m_ten: tl.tensor, m_shape0: int, m_shape1: int, m_stride0: int, m_stride1: int,
    TILE0: tl.constexpr, TILE1: tl.constexpr,
) -> tuple[tl.tensor]:
    """Create the block pointers for all but the metadata."""
    i_block = make_block_ptr(pid, i_ten, i_shape0, i_shape1, i_stride0, i_stride1, TILE0, TILE1)
    o_block = make_block_ptr(pid, o_ten, o_shape0, o_shape1, o_stride0, o_stride1, TILE0, TILE1)
    s_block = make_block_ptr(pid, s_ten, s_shape0, s_shape1, s_stride0, s_stride1, TILE0, TILE1)
    # z_block = make_block_ptr(pid, z_ten, z_shape0, z_shape1, z_stride0, z_stride1, TILE0, ZTILE)
    # m_block = make_block_ptr(pid, m_ten, m_shape0, m_shape1, m_stride0, m_stride1, TILE0, MTILE)
    # return i_block, o_block, s_block, z_block, m_block
    return i_block, o_block, s_block


#fmt: on
@triton.jit
def store_tensors(
    i_ten, o_ten, optr, o_block, obits,
    s_ten, sptr, s_block, sbits,
    z_ten, zptr, z_block, zbits,
    m_ten, mptr, m_block, mratio: tl.constexpr, castmode
): # fmt: on
    """Store the tensors, omitting the i and x tensors."""
    if castmode == CastMode.VIRTUAL:
        tl.store(o_block, optr.to(o_ten.dtype))
    elif castmode == CastMode.ACTUAL:
        if mptr is not None:
            tl.store(m_block, mptr.cast(tl.uint8))
        if o_ten.dtype.is_floating() == optr.is_floating():
            tl.store(o_block, optr.to(o_ten.dtype))
        else:
            tl.store(o_block, optr.cast(o_ten.dtype))
        if s_block is not None:
            if s_ten.dtype.is_floating() == sptr.is_floating():
                tl.store(s_block, sptr.to(s_ten.dtype))
            else:
                tl.store(s_block, sptr.cast(s_ten.dtype))
        if z_block is not None:
            if z_ten.dtype.is_floating() == zptr.is_floating():
                tl.store(z_block, zptr.to(z_ten.dtype))
            else:
                tl.store(z_block, zptr.cast(z_ten.dtype))
    else:
        if mptr:
            tl.store(m_block, mptr)
            coptr = tl.zeros(optr.shape[0], optr.shape[1] // mratio, dtype=optr.dtype)
            for i in range(optr.shape[0]):
                cj = 0
                for j in range(optr.shape[1]):
                    if mptr[i, j]:
                        coptr[i, cj] = optr[i, j]
                        cj += 1
            optr = coptr
        if ratio := i_ten.shape[1] // optr.shape[1] > 1:
            coptr = tl.zeros(optr.shape[0], optr.shape[1] // ratio, dtype=o_ten.dtype)
            for i in range(optr.shape[0]):
                for j in range(optr.shape[1]):
                    cj, r = j // ratio, j % ratio
                    coptr[i, cj] = (coptr[i, cj] << (r * obits)) | optr[i, j]
            tl.store(o_block, coptr)
        else:
            tl.store(o_block, optr.to(o_ten.dtype))

        if ratio := s_ten.shape[1] // sptr.shape[1] > 1:
            csptr = tl.zeros(sptr.shape[0], sptr.shape[1] // ratio, dtype=s_ten.dtype)
            for i in range(sptr.shape[0]):
                for j in range(sptr.shape[1]):
                    cj, r = j // ratio, j % ratio
                    csptr[i, cj] = (csptr[i, cj] << (r * sbits)) | sptr[i, j]
            tl.store(s_block, csptr)
        else:
            tl.store(s_block, sptr.to(s_ten.dtype))

        if zptr is not None:
            if ratio := z_ten.shape[1] // zptr.shape[1] > 1:
                czptr = tl.zeros(zptr.shape[0], zptr.shape[1] // ratio, type=z_ten.dtype)
                for i in range(zptr.shape[0]):
                    for j in range(zptr.shape[1]):
                        cj, r = j // ratio, j % ratio
                        czptr[i, cj] = (czptr[i, cj] << r) | zptr[i, j]
                tl.store(z_block, czptr)
            else:
                tl.store(z_block, zptr.to(z_ten.dtype))


@triton.jit
def cast_unscaled(x: tl.tensor, emin: int, mbits: int, maxfloat: float, roundmode: RoundMode, pid: int) -> tl.tensor:
    """Cast tensor to a new dtype without scaling."""
    tl.device_assert(x.dtype.is_floating(), "Only floating point types supported.")
    ix = x.to(tl.uint32, bitcast=True)
    scale = mbits - get_exponent(ix, bias=False).clamp(min=emin)
    x = round(modify_exponent(ix, scale, add=True).to(tl.float32, bitcast=True), roundmode, pid)
    x = modify_exponent(x.to(tl.uint32, bitcast=True), scale, subtract=True)
    return x.to(tl.float32, bitcast=True).clamp(-maxfloat, maxfloat)


@triton.jit
def sparsify(
    iptr: tl.tensor, TILE0: tl.constexpr, TILE1: tl.constexpr, SPARSE_M: tl.constexpr, SPARSE_N: tl.constexpr,
) -> tuple[tl.tensor]:
    """Sparsify the input tensor, which should be a vector of length TILE."""
    if SPARSE_N == 1:
        return iptr, None
    optr = tl.zeros_like(iptr)
    mptr = tl.zeros(optr.shape, dtype=tl.int1)
    for m in range(TILE0):
        for i in range(TILE1, SPARSE_N):
            for _ in range(SPARSE_M):
                argmax = i + tl.argmax(tl.abs(iptr[m, i : i + SPARSE_N]))
                optr[m, argmax] = iptr[m, argmax]
                iptr[m, argmax] = 0.0
                mptr[m, argmax] = True
    return optr, mptr


#fmt: off
@triton.jit
def exponent_cast_kernel(
    i_ten: tl.tensor, i_shape0: int, i_shape1: int, i_stride0: int, i_stride1: int,
    o_ten: tl.tensor, o_shape0: int, o_shape1: int, o_stride0: int, o_stride1: int,
    s_ten: tl.tensor, s_shape0: int, s_shape1: int, s_stride0: int, s_stride1: int,
    m_ten: tl.tensor, m_shape0: int, m_shape1: int, m_stride0: int, m_stride1: int,
    roundmode: RoundMode, scalemode: ScaleMode, castmode: CastMode,
    emax: int, emin: int, mbits: int, maxfloat: float, midmax: float, obits: int, sbits: int,
    TILE0: tl.constexpr, TILE1: tl.constexpr, ITILE: tl.constexpr, OTILE: tl.constexpr, STILE: tl.constexpr,
    MTILE: tl.constexpr,
    SPARSE_M: tl.constexpr, SPARSE_N: tl.constexpr,
):
    """Power of two scaled float or int."""
    pid = tl.program_id(axis=0)
    i_block, o_block, s_block, _, m_block = make_all_block_ptrs(
        pid,
        i_ten, i_shape0, i_shape1, i_stride0, i_stride1, ITILE,
        o_ten, o_shape0, o_shape1, o_stride0, o_stride1, OTILE,
        s_ten, s_shape0, s_shape1, s_stride0, s_stride1, STILE,
        None, 0, 0, 0, 0, 1,
        m_ten, m_shape0, m_shape1, m_stride0, m_stride1, MTILE,
        TILE0, TILE1
    )
    iptr = tl.load(i_block).to(tl.float32)
    # iptr, mptr = sparsify(pid, iptr, TILE0, TILE1, SPARSE_M, SPARSE_N)
    sptr = get_shared_exponent(tl.abs(iptr), emax, emin, mbits, maxfloat, midmax, scalemode)
    mptr = None
    optr = quantize_float(iptr, sptr, emax, emin, mbits, maxfloat, roundmode, pid, clip=False)
    if castmode == CastMode.VIRTUAL:
        scale = (sptr - 127).to(tl.int8) # scale the outputs from target dtype range to unscaled range
        optr = modify_exponent(optr.to(tl.uint32, bitcast=True), scale, add=True).to(tl.float32, bitcast=True)
    m_ratio = tl.cdiv(SPARSE_N, SPARSE_M)
    store_tensors(o_ten, optr, o_block, obits, s_ten, sptr, s_block, sbits, None, None, None, mptr, m_block, m_ratio, castmode)


# fmt: on
class TritonCast:
    """Triton: accelerated methods for casting."""

    @staticmethod
    def supports(tensor: Tensor) -> bool:
        """Check if the cast operation is supported by in the triton code."""
        if not is_triton_available():
            return False
        if Modes.cast == CastMode.UPCAST:
            return False # TODO(ericd) implement upcast, then change this
        else:
            return (
                Modes.cast != CastMode.COMPRESS
                and tensor.input.dim() == 2
                and tensor.input.dtype in STD_DTYPES
                and tensor.dtype.torch_dtype in FP8_DTYPES
                and not tensor.needs_pad()
                and tensor.dtype.is_square
            )

    @staticmethod
    def cast(tensor: Tensor, transpose_scale: bool = False) -> bool:
        """Cast data to the specified dtype."""

        def get_tensor_info(tensor: Tensor, name: str):
            if (t := getattr(tensor, name, None)) is not None:
                return (t, t.size(0), t.size(1), t.stride(0), t.stride(1))
            else:
                return (None, 0, 0, 0, 0)

        nspec, sspec = tensor.dtype.nspec, tensor.dtype.sspec
        t = tensor.input
        input_info = (t, t.size(0), t.size(1), t.stride(0), t.stride(1))
        output_info = get_tensor_info(tensor, "tensor")
        scale_info = get_tensor_info(tensor, "scale")
        # zero_info = get_tensor_info(tensor, "zero")
        # meta_info = get_tensor_info(tensor, "meta")
        mask_info = get_tensor_info(tensor, "mask")
        nspec_info = (nspec.emax, nspec.emin, nspec.mbits, nspec.finfo.maxfloat, nspec.finfo.midmax)
        M, N = input_info[1:3]
        sspec_info = sspec.get_tile_info(M, N)[2:]
        if not nspec.is_codebook:
            sspec_info = (sspec_info[0], sspec_info[1], sspec_info[4], sspec_info[5]) # no subtiles needed

        grid = (tl.cdiv(M, sspec_info[0]) * tl.cdiv(N, sspec_info[1]),)
        if sspec.scale.is_exponent:
            exponent_cast_kernel[grid](
                *input_info, *output_info, *scale_info, *mask_info, Modes.round, Modes.scale, Modes.cast,
                *nspec_info, nspec.bits, sspec.scale.bits, *sspec_info,
            )
        else:
            raise ValueError("Unsupported cast operation.")
        tensor.postcast(transpose_scale)
        # fmt: on


    @staticmethod
    def upcast(tensor: Tensor, torch_dtype: torch.dtype = None):
        """Upcast data to the specified dtype."""
        #TODO(ericd): must implement
        if torch_dtype is None:
            torch_dtype = tensor.original_dtype
        if tensor.original_dtype.is_floating() != torch_dtype.is_floating():
            raise ValueError("Upcast requires the same dtype family (both float or both int.")
        if not tensor.quantized:
            raise ValueError("Upcast requires a quantized tensor.")
        raise NotImplementedError("Upcast not implemented for TritonCast.")
