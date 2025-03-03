#!/usr/bin/env python
# tcast/tricast.py: triton methods for casting
# SPDX-License-Identifier: MIT
# ruff: noqa: D103

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch
import triton
import triton.language as tl

from .common import CastMode, Modes, RoundMode, ScaleMode
from .tensor import Tensor
from .utils import is_triton_available


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"



# fmt: off
@triton.jit
def as_float(x): return tl.cast(x, tl.float32, bitcast=True)
@triton.jit
def as_int(x): return tl.cast(x, tl.int32, bitcast=True)
@triton.jit
def as_uint(x): return tl.cast(x, tl.uint32, bitcast=True)
@triton.jit
def as_type(x, dtype): return tl.cast(x, dtype, bitcast=True)
@triton.jit
def get_exponent(x): return (as_uint(x) >> 23) & 0xFF
# fmt: off
@triton.jit
def lp_enabled(LPCODE: tl.constexpr) -> tl.constexpr: return LPCODE != 0
@triton.jit
def lp_forward_type(LPCODE: tl.constexpr) -> tl.constexpr: return LPCODE & 3
@triton.jit
def lp_backward_type(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 2) & 3
@triton.jit
def lp_output_type(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 4) & 3
@triton.jit
def lp_scale_type(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 6) & 3
@triton.jit
def lp_scale_scope(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 8) & 3
@triton.jit
def lp_block(LPCODE: tl.constexpr) -> tl.constexpr: return 1 << (((LPCODE >> 10) & 3) + 1)
@triton.jit
def lp_icp_qkds(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 13 & 1) != 0
@triton.jit
def lp_icp_pvdo(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 14 & 1) != 0
@triton.jit
def lp_fake(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 15 & 1) != 0
@triton.jit
def lp_nanoo(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 16 & 1) != 0
@triton.jit
def lp_backward(LPCODE: tl.constexpr) -> tl.constexpr: return lp_backward_type(LPCODE) != 0
@triton.jit
def lp_output(LPCODE: tl.constexpr) -> tl.constexpr: return lp_output_type(LPCODE) != 0
@triton.jit
def lp_icp(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> 13 & 3) != 0
@triton.jit
def lp_tensor(LPCODE: tl.constexpr) -> tl.constexpr: return lp_scale_scope(LPCODE) == 0
@triton.jit
def lp_scale_is_fp32(CODE: tl.constexpr) -> tl.constexpr: return CODE == 0
@triton.jit
def lp_scale_is_fp16(CODE: tl.constexpr) -> tl.constexpr: return CODE == 1
@triton.jit
def lp_scale_is_bf16(CODE: tl.constexpr) -> tl.constexpr: return CODE == 2
@triton.jit
def lp_scale_is_exponent(CODE: tl.constexpr) -> tl.constexpr: return CODE == 3
# @triton.jit
# def lp_get_mbits(LPCODE: tl.constexpr, select: tl.constexpr) -> tl.constexpr:
#     return MBITS[((LPCODE >> (2 * select)) & 3) - 1][lp_nanoo(LPCODE)]
# @triton.jit
# def lp_get_emax(LPCODE: tl.constexpr, select: tl.constexpr) -> tl.constexpr:
#     return EMAX[((LPCODE >> (2 * select)) & 3) - 1][lp_nanoo(LPCODE)]
# @triton.jit
# def lp_get_emin(LPCODE: tl.constexpr, select: tl.constexpr) -> tl.constexpr:
#     return EMIN[((LPCODE >> (2 * select)) & 3) - 1][lp_nanoo(LPCODE)]
# @triton.jit
# def lp_get_maxval(LPCODE: tl.constexpr, select: tl.constexpr) -> tl.constexpr:
#     return MAXVALS[((LPCODE >> (2 * select)) & 3) - 1][lp_nanoo(LPCODE)]
# @triton.jit
# def lp_get_dtype(LPCODE: tl.constexpr, select: tl.constexpr) -> tl.constexpr:
#     return DTYPES[((LPCODE >> (2 * select)) & 3) - 1][lp_nanoo(LPCODE)]
# fmt: on

# @triton.jit
# def scaled_quantize(x, scale, LPCODE: tl.constexpr, select: tl.constexpr, descale=True):
#     """Returns the quantized x and the scale used."""
#     tl.static_assert(lp_enabled(LPCODE))
#     FAKE: tl.constexpr = lp_fake(LPCODE)
#     EPS: tl.constexpr = 6.10352e-05
#     MAXVAL: tl.constexpr = lp_get_maxval(LPCODE, select)
#     if descale:
#         out = x / scale
#     else:
#         out = x * scale
#     if FAKE:
#         valexp = tl.floor(tl.log2(tl.maximum(EPS, tl.abs(out))))
#         rscale = tl.exp2(lp_get_mbits(LPCODE, select) - tl.maximum(valexp, lp_get_emin(LPCODE, select)))
#         # WTF is there no tl.math.round()?
#         out *= rscale
#         out = tl.where(out < 0.0, tl.ceil(out - 0.5), tl.floor(out + 0.5))
#         out = tl.clamp(out / rscale, -MAXVAL, MAXVAL)
#         out = tl.cast(out, x.type.element_ty)
#         # out = tl.clamp(tl.extra.tl.libdevice.rint(out * rscale) / rscale, -MAXVAL, MAXVAL)
#     else:
#         out = tl.cast(out, lp_get_dtype(LPCODE, select))
#     return out

@triton.jit
def modify_exponent(x, y, direction: int = 0):
    """Modify the exponent of a float. direction is 1 for add, -1 for subtract, 0 for replace."""
    x = as_uint(x)
    if direction == 0:
        x = tl.where(get_exponent(x) == 0, 0, x & 0x807FFFFF | (y << 23))
    elif direction == 1:
        if y >= 127:
            x = tl.where(get_exponent(x) == 0, 0, x + ((y - 127) << 23))
        else:
            x = tl.where(get_exponent(x) == 0, 0, x - ((127 - y) << 23))
    else:
        if y >= 127:
            x = tl.where(get_exponent(x) == 0, 0, x - ((y - 127) << 23))
        else:
            x = tl.where(get_exponent(x) == 0, 0, x + ((127 - y) << 23))
    return as_float(x)


@triton.jit
def quantize_float(values, shared_exp, emin, mbits, maxfloat, roundmode, SEED: tl.constexpr, clip=True):
    """Quantize the float values to an exponent (OCP MX biased scale)."""
    ZERO: tl.constexpr = 0
    AWAY: tl.constexpr = 1
    EVEN: tl.constexpr = 2
    STOCHASTIC: tl.constexpr = 3
    emin += 127
    values = modify_exponent(values, shared_exp, direction=-1)
    valexp = get_exponent(values)
    offset = tl.where(emin <= valexp, 0, tl.minimum(mbits, emin - valexp))
    roundbit = 1 << (23 - 1 - mbits + offset)
    mask = (roundbit << 1) - 1
    values = as_uint(values)
    if roundmode == AWAY:  # ties away from zero
        values += roundbit
    elif roundmode == STOCHASTIC:
        r = tl.randint(SEED, values)
        r = r & mask
        values += r
    else:
        tie = (roundbit - 1) & values == 0
        if roundmode == ZERO:  # ties towards zero, not trunc
            values += tl.where(tie, 0, roundbit)
        if roundmode == EVEN:  # ties to nearest even
            values += tl.where(tie & (values & (roundbit << 1) == 0), 0, roundbit)
    valexp = get_exponent(values)
    offset = tl.where(emin <= valexp, 0, emin - valexp)
    mask = (1 << (23 - mbits + offset)) - 1
    values = as_float(tl.where(offset > mbits or valexp == 255, 0, values & ~mask))
    if clip:
        values = tl.clamp(values, min=-maxfloat, max=maxfloat)
    # values remain target dtype values, will need to be unscaled for virtual cast
    return values


@triton.jit
def get_shared_scale(values, maxfloat):
    """Find the shared float scale for a block of values."""
    return maxfloat / tl.max(tl.abs(values))


@triton.jit
def get_shared_exponent(values, emax, emin, mbits, maxfloat, midmax, scalemode):
    """Find the shared exponent for a block of values."""
    CEIL: tl.constexpr = 1
    MIDMAX: tl.constexpr = 2
    OPTION3: tl.constexpr = 3
    TOPBINADE: tl.constexpr = 4
    absmax = tl.max(tl.abs(values))
    shared_exp = get_exponent(absmax)  # shared_exp is a biased exponent
    # set the exponent of absmax to emax comparing with midmax, maxfloat, etc.
    absmax = modify_exponent(absmax, 127 + emax)
    increment = tl.zeros_like(shared_exp)
    if scalemode == CEIL:
        increment = as_uint(tl.where(as_uint(absmax) & 0x007FFFFF != 0, 1, 0))
    elif scalemode == MIDMAX:
        increment = as_uint(tl.where(absmax >= midmax, 1, 0))
    if scalemode == TOPBINADE:
        increment = as_uint(tl.where(absmax > maxfloat, 1, 0))
    if scalemode == OPTION3:
        rounded = quantize_float(absmax, 127, emin, mbits, maxfloat, 2, 19, clip=False)
        increment = as_uint(tl.where(get_exponent(as_uint(rounded)) != 127 + emax, 1, 0))
    # returns the biased exponent, adjusted by the dtype's emax
    shared_exp = shared_exp + increment - emax
    return shared_exp


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
def make_block_ptr(
    pid: int, tensor: tl.tensor, shape0: int, shape1: int, stride0: int, stride1: int,
    TILE0: tl.constexpr, TILE1: tl.constexpr
) -> tl.tensor:
    """Create a block pointer for the given tensor."""
    if tensor is None:
        return None
    block_row = (pid // (shape1 // TILE1)) * TILE0
    block_col = (pid % (shape1 // TILE1)) * TILE1
    offset = block_row * stride0 + block_col * stride1
    return tl.make_block_ptr(
        base=tensor + offset,
        shape=(shape0, shape1),
        strides=(stride0, stride1),
        offsets=(0, 0),
        block_shape=(TILE0, TILE1),
        order=(0, 1),
    )


@triton.jit
def make_all_block_ptrs( # fmt: off
    pid: int,
    i_ten: tl.tensor, i_shape0: int, i_shape1: int, i_stride0: int, i_stride1: int,
    o_ten: tl.tensor, o_shape0: int, o_shape1: int, o_stride0: int, o_stride1: int,
    s_ten: tl.tensor, s_shape0: int, s_shape1: int, s_stride0: int, s_stride1: int,
    # z_ten: tl.tensor, z_shape0: int, z_shape1: int, z_stride0: int, z_stride1: int,
    # m_ten: tl.tensor, m_shape0: int, m_shape1: int, m_stride0: int, m_stride1: int,
    TILE0: tl.constexpr, TILE1: tl.constexpr,
) -> tuple[tl.tensor]: # fmt: on
    """Create the block pointers for all but the metadata."""
    i_block = make_block_ptr(pid, i_ten, i_shape0, i_shape1, i_stride0, i_stride1, TILE0, TILE1)
    o_block = make_block_ptr(pid, o_ten, o_shape0, o_shape1, o_stride0, o_stride1, TILE0, TILE1)
    s_block = make_block_ptr(pid, s_ten, s_shape0, s_shape1, s_stride0, s_stride1, TILE0, TILE1)
    # z_block = make_block_ptr(pid, z_ten, z_shape0, z_shape1, z_stride0, z_stride1, TILE0, ZTILE)
    # m_block = make_block_ptr(pid, m_ten, m_shape0, m_shape1, m_stride0, m_stride1, TILE0, MTILE)
    # return i_block, o_block, s_block, z_block, m_block
    return i_block, o_block, s_block


@triton.jit
def store_tensors( # fmt: off
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
def modify_exponent(x, num, add: bool = False, subtract: bool = False, bias: bool = False):
    """Modify the exponent of a floating point number passed in as bits."""
    num = (num + (127 if bias else 0)) << 23
    return x + num if add else x - num if subtract else (x & 0x8007FFFF) + num


@triton.jit
def get_exponent(x, bias: bool = False):
    """Get the mantissa of a floating point number passed in as bits."""
    return ((x >> 23) & 0xFF) - (0 if bias else 127)


@triton.jit
def round(x: tl.tensor, roundmode: RoundMode, seed: int = 42) -> tl.tensor:
    """Round floating point."""
    if roundmode == RoundMode.STOCHASTIC:
        rand_ptr = tl.randn(seed, x.shape, nrounds=1)
        return tl.libdevice.trunc(x + tl.where(x < 0, -rand_ptr), rand_ptr)
    if roundmode == RoundMode.EVEN:
        return tl.libdevice.rint(x)
    if roundmode == RoundMode.AWAY:
        return tl.libdevice.round(x)
    if roundmode == RoundMode.ZERO:
        trunc = tl.libdevice.trunc(x)
        return tl.where(x - trunc == 0.5, trunc, tl.libdevice.round(x))


# @triton.jit
# def cast_unscaled(x: tl.tensor, emin: int, mbits: int, maxfloat: float, roundmode: RoundMode, pid: int) -> tl.tensor:
#     """Cast tensor to a new dtype without scaling."""
#     tl.device_assert(x.dtype.is_floating(), "Only floating point types supported.")
#     ix = x.to(tl.uint32, bitcast=True)
#     scale = mbits - get_exponent(ix, bias=False).clamp(min=emin)
#     x = round(modify_exponent(ix, scale, add=True).to(tl.float32, bitcast=True), roundmode, pid)
#     x = modify_exponent(x.to(tl.uint32, bitcast=True), scale, subtract=True)
#     return x.to(tl.float32, bitcast=True).clamp(-maxfloat, maxfloat)


@triton.jit
def quantize_float( # fmt: off
    values: tl.tensor, scale: tl.tensor, emax: int, emin: int, mbits: int, maxfloat: float,
    roundmode: RoundMode, seed: int, clip: bool = True,
) -> tl.tensor: # fmt: on
    """Quantize the float values to an int scale (unbiased power of 2), exponent (OCP MX biased scale), or float scale."""
    # get scale as unbiased scale exponent
    if scale.dtype.is_floating():
        values /= scale
        shared_exp = tl.zeros(scale.shape, dtype=tl.uint32)
        roundmode = RoundMode.FLOOR
    elif scale.dtype.is_int_signed():
        shared_exp = (scale - emax).to(tl.uint32)
    else:
        shared_exp = scale.to(tl.uint32) - 127
    values = values.to(tl.uint32, bitcast=True)
    values = modify_exponent(values, shared_exp, subtract=True)
    rscale = mbits - get_exponent(values, bias=False).clamp(min=emin)
    values = modify_exponent(values, rscale, add=True, bias=False)
    values = round(values.to(tl.float32, bitcast=True), roundmode, seed).to(tl.uint32, bitcast=True)
    values = modify_exponent(values, rscale, subtract=True)
    if clip:
        values = values.to(tl.float32, bitcast=True).clamp(min=-maxfloat, max=maxfloat)
    return values


@triton.jit
def get_shared_exponent(
    vals: tl.tensor, emax: int, emin: int, mbits: int, maxfloat: float, midmax: float, scalemode: ScaleMode
) -> tl.tensor:
    """Find the shared exponent for a block of values. vals is cast to float and absolute value."""
    absmax = tl.abs(vals).max().to(dtype=tl.uint32, bitcast=True)
    has_mant = absmax & 0x7FFFFF != 0
    shared_exp = get_exponent(absmax, bias=True)
    absmax = modify_exponent(absmax, emax, replace=True, bias=True).to(tl.float32, bitcast=True)
    if scalemode == ScaleMode.MIDMAX:
        shared_exp[absmax >= midmax] += 1
    elif scalemode == ScaleMode.CEIL:
        shared_exp[has_mant] += 1
    elif scalemode == ScaleMode.TOPBINADE:
        shared_exp[absmax > maxfloat] += 1
    elif scalemode == ScaleMode.OPTION3:
        rounded = quantize_float(absmax, shared_exp - emax, emax, emin, mbits, maxfloat, RoundMode.EVEN, None, clip=False)
        shared_exp[get_exponent(rounded.to(dtype=tl.uint32, bitcast=True), bias=True) != shared_exp] += 1
    return (shared_exp - emax).to(tl.uint8)


# @triton.jit
# def sparsify(
#     iptr: tl.tensor, TILE0: tl.constexpr, TILE1: tl.constexpr, SPARSE_M: tl.constexpr, SPARSE_N: tl.constexpr,
# ) -> tuple[tl.tensor]:
#     """Sparsify the input tensor, which should be a vector of length TILE."""
#     if SPARSE_N == 1:
#         return iptr, None
#     optr = tl.zeros_like(iptr)
#     mptr = tl.zeros(optr.shape, dtype=tl.int1)
#     for m in range(TILE0):
#         for i in range(TILE1, SPARSE_N):
#             for _ in range(SPARSE_M):
#                 argmax = i + tl.argmax(tl.abs(iptr[m, i : i + SPARSE_N]))
#                 optr[m, argmax] = iptr[m, argmax]
#                 iptr[m, argmax] = 0.0
#                 mptr[m, argmax] = True
#     return optr, mptr


# @triton.jit
# def asymmetric_cast_kernel( # fmt: off
#     i_ten: tl.tensor, i_shape0: int, i_shape1: int, i_stride0: int, i_stride1: int, i_ratio,
#     o_ten: tl.tensor, o_shape0: int, o_shape1: int, o_stride0: int, o_stride1: int, o_ratio,
#     s_ten: tl.tensor, s_shape0: int, s_shape1: int, s_stride0: int, s_stride1: int, s_ratio,
#     z_ten: tl.tensor, z_shape0: int, z_shape1: int, z_stride0: int, z_stride1: int, z_ratio,
#     m_ten: tl.tensor, m_shape0: int, m_shape1: int, m_stride0: int, m_stride1: int, m_ratio,
#     roundmode: RoundMode, castmode: CastMode,
#     bits: int, semin: int, smbits: int, smaxfloat: float, zemin: int, zmbits: int, zmaxfloat: float,
#     TILE0: tl.constexpr, TILE1: tl.constexpr, SPARSE_M: tl.constexpr, SPARSE_N: tl.constexpr,
# ):
#     """Unsigned int, float scale, float or int zero point."""
#     pid = tl.program_id(axis=0)
#     i_block, o_block, s_block, z_block, m_block = make_all_block_ptrs(
#         pid,
#         i_ten, i_shape0, i_shape1, i_stride0, i_stride1, i_ratio,
#         o_ten, o_shape0, o_shape1, o_stride0, o_stride1, o_ratio,
#         s_ten, s_shape0, s_shape1, s_stride0, s_stride1, s_ratio,
#         z_ten, z_shape0, z_shape1, z_stride0, z_stride1, z_ratio,
#         m_ten, m_shape0, m_shape1, m_stride0, m_stride1, m_ratio,
#         TILE0, TILE1
#     )
#     # fmt: on
#     iptr = tl.load(i_block).to(tl.float32)
#     iptr, mptr = sparsify(pid, iptr, TILE0, TILE1, SPARSE_M, SPARSE_N)
#     tmin, tmax = tl.min(iptr).clamp(max=0.0), tl.max(iptr).clamp(min=0.0)
#     maxint = (1 << bits) - 1
#     sptr = cast_unscaled((tmax - tmin) / maxint, semin, smbits, smaxfloat, roundmode, pid).clamp(min=EPS)
#     # zero point can be float or int
#     if zemin is None:
#         zdtype = tl.int8 if zmbits <= 8 else tl.int16
#         zptr = round(-tmin / sptr).to(zdtype)
#     else:
#         zptr = cast_unscaled(-tmin, zemin, zmbits, zmaxfloat, roundmode, pid)
#     optr = (round(iptr / sptr) + zptr).clamp(0, maxint) if zemin is None else round((iptr + zptr) / sptr).clamp(0, maxint)
#     if castmode == CastMode.VIRTUAL:
#         optr = sptr * optr - zptr if zemin is None else sptr * (optr - zptr)
#     store_tensors(o_ten, optr, o_block, s_ten, sptr, s_block, z_ten, zptr, z_block, m_ten, mptr, m_block, castmode)


# @triton.jit
# def symmetric_cast_kernel( # fmt: off
#     i_ten: tl.tensor, i_shape0: int, i_shape1: int, i_stride0: int, i_stride1: int, i_ratio,
#     o_ten: tl.tensor, o_shape0: int, o_shape1: int, o_stride0: int, o_stride1: int, o_ratio,
#     s_ten: tl.tensor, s_shape0: int, s_shape1: int, s_stride0: int, s_stride1: int, s_ratio,
#     m_ten: tl.tensor, m_shape0: int, m_shape1: int, m_stride0: int, m_stride1: int, m_ratio,
#     roundmode: RoundMode, castmode: CastMode,
#     bits: int, emin: int, mbits: int, maxfloat: float,
#     TILE0: tl.constexpr, TILE1: tl.constexpr, SPARSE_M: tl.constexpr, SPARSE_N: tl.constexpr,
# ):
#     """Signed int, float scale."""
#     pid = tl.program_id(axis=0)
#     i_block, o_block, s_block, _, _, m_block = make_all_block_ptrs(
#         pid,
#         i_ten, i_shape0, i_shape1, i_stride0, i_stride1, i_ratio,
#         o_ten, o_shape0, o_shape1, o_stride0, o_stride1, o_ratio,
#         s_ten, s_shape0, s_shape1, s_stride0, s_stride1, s_ratio,
#         None, 0, 0, 0, 0, 1,
#         m_ten, m_shape0, m_shape1, m_stride0, m_stride1, m_ratio,
#         TILE0, TILE1
#     )
#     # fmt: on
#     iptr = tl.load(i_block).to(tl.float32)
#     iptr, mptr = sparsify(pid, iptr, TILE0, TILE1, SPARSE_M, SPARSE_N)
#     maxint = (1 << (bits - 1)) - 1
#     sptr = cast_unscaled(tl.max(tl.abs(iptr)) / maxint, emin, mbits, maxfloat, roundmode, pid).clamp(min=EPS)
#     optr = round(iptr / sptr).clamp(-maxint, maxint)
#     if castmode == CastMode.VIRTUAL:
#         optr *= sptr
#     store_tensors(o_ten, optr, o_block, s_ten, sptr, s_block, None, None, None, m_ten, mptr, m_block, castmode)


# @triton.jit
# def codebook_cast_kernel( # fmt: off
#     i_ten: tl.tensor, i_shape0: int, i_shape1: int, i_stride0: int, i_stride1: int, i_ratio,
#     o_ten: tl.tensor, o_shape0: int, o_shape1: int, o_stride0: int, o_stride1: int, o_ratio,
#     s_ten: tl.tensor, s_shape0: int, s_shape1: int, s_stride0: int, s_stride1: int, s_ratio,
#     x_ten: tl.tensor, x_shape0: int, x_shape1: int, x_stride0: int, x_stride1: int, x_ratio,
#     m_ten: tl.tensor, m_shape0: int, m_shape1: int, m_stride0: int, m_stride1: int, m_ratio,
#     cb_name: str, cb_mappings: tl.tensor, cb_midpoints: tl.tensor,
#     roundmode: RoundMode, scalemode: ScaleMode, castmode: CastMode,
#     emax: int, emin: int, mbits: int, maxfloat: float, midmax: float,
#     TILE0: tl.constexpr, TILE1: tl.constexpr,
#     SUBTILE0: tl.constexpr, SUBTILE1: tl.constexpr,
#     SPARSE_M: tl.constexpr, SPARSE_N: tl.constexpr,
# ):
#     """Implicit codebook."""
#     pid = tl.program_id(axis=0)
#     i_block, _, s_block, _, m_block = make_all_block_ptrs(
#         pid,
#         i_ten, i_shape0, i_shape1, i_stride0, i_stride1, i_ratio,
#         o_ten, o_shape0, o_shape1, o_stride0, o_stride1, o_ratio,
#         s_ten, s_shape0, s_shape1, s_stride0, s_stride1, s_ratio,
#         x_ten, x_shape0, x_shape1, x_stride0, x_stride1, x_ratio,
#         m_ten, m_shape0, m_shape1, m_stride0, m_stride1, m_ratio,
#         TILE0, TILE1
#     )
#     # fmt: on
#     iptr = tl.load(i_block).to(tl.float32)
#     iptr, mptr = sparsify(pid, iptr, TILE0, TILE1, SPARSE_M, SPARSE_N)
#     sptr = get_shared_exponent(tl.abs(iptr), emax, emin, mbits, maxfloat, midmax, scalemode)
#     scale = (sptr - 127).to(tl.int8) # scale the inputs into target dtype range
#     iptr = modify_exponent(iptr.to(tl.uint32, bitcast=True), scale, subtract=True).to(tl.float32, bitcast=True)

#     # get the constants for the implicit codebook or the tensors for the explicit codebooks
#     if cb_name in IMPLICIT_CODEBOOKS:
#         cb = IMPLICIT_CODEBOOKS[cb_name]
#         cbmap, cbmid = cb["mappings"], cb["midpoints"]
#         shared_mappings = tl.shared_memory((len(cbmap), len(cbmap[0])), dtype=tl.float32)
#         shared_midpoints = tl.shared_memory((len(cbmid), len(cbmid[0])), dtype=tl.float32)
#         if tl.threadid(0) == 0:
#             shared_mappings = tl.array(cbmap)
#             shared_midpoints = tl.array(cbmid)
#         tl.barrier()
#         mappings = shared_mappings[tl.arange(len(cbmap)), tl.arange(len(cbmap[0]))]
#         midpoints = shared_midpoints[tl.arange(len(cbmid)), tl.arange(len(cbmid[0]))]
#     else:
#         mappings = tl.load(cb_mappings)
#         midpoints = tl.load(cb_midpoints)

#     for m in tl.static_range(TILE0 // SUBTILE0):
#         for n in tl.static_range(TILE1 // SUBTILE1):
#             siptr = iptr[m * SUBTILE0:(m + 1) * SUBTILE0, n * SUBTILE1:(n + 1) * SUBTILE1]
#             sxptr = tl.zeros(siptr.shape, dtype=tl.uint8)
#             soptr = tl.zeros(siptr.shape, dtype=tl.uint8)
#             best_score = None
#             for meta in range(len(mappings)):
#                 idxptr = tl.zeros(siptr.shape, dtype=tl.uint8)
#                 for index in range(len(mappings[meta])):
#                     mid = midpoints[meta][index]
#                     if (roundmode == RoundMode.ZERO and mid < 0.0) or (roundmode != RoundMode.ZERO and mid >= 0.0):
#                         idxptr[siptr >= mid] = index + 1
#                     else:
#                         idxptr[siptr > mid] = index + 1
#                 score = tl.sum(tl.abs(sxptr - mappings[meta][idxptr])) / tl.sum(siptr)
#                 if best_score is None or score < best_score:
#                     best_score, sxptr, best_index = score, meta, idxptr
#             # output is the values of the mappings if virtual, else the indices
#             if castmode == CastMode.VIRTUAL:
#                 soptr = mappings[sxptr][best_index].to(o_ten.dtype)
#                 soptr = modify_exponent(soptr.to(tl.uint32, bitcast=True), scale, add=True).to(tl.float32, bitcast=True)
#             else:
#                 soptr = best_index
#             o_block.advance(m, n)
#             tl.store(o_block, soptr.to(o_ten.dtype))
#             if x_block is not None:
#                 x_block.advance(m, n)
#                 tl.store(x_block, sxptr)
#     store_tensors(None, None, None, s_ten, sptr, s_block, None, None, None, m_ten, mptr, m_block, castmode)


@triton.jit
def exponent_cast_kernel( # fmt: off
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
    # fmt: on
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


class TritonCast:
    """Triton: accelerated methods for casting."""

    @staticmethod
    def supports(tensor: Tensor, prescaled: bool = False, premapped: bool = False, magnitude: bool = False) -> bool:
        """Check if the cast operation is supported by in the triton code."""
        return (
            is_triton_available()
            and Modes.cast != CastMode.COMPRESS
            and not magnitude
            and not prescaled
            and not premapped
            and tensor.dtype.is_tile
            and tensor.dtype.nspec.is_float
        )

    @staticmethod
    def cast(tensor: Tensor, transpose_scale: bool = False) -> bool:
        """Cast data to the specified dtype."""

        def get_tensor_info(tensor: Tensor, name: str):
            if (t := getattr(tensor, name, None)) is not None:
                return (t, t.size(0), t.size(1), t.stride(0), t.stride(1))
            else:
                return (None, 0, 0, 0, 0)

        if not TritonCast.supports(tensor):
            return False

        tensor.precast(transpose_scale)
        nspec, sspec = tensor.dtype.nspec, tensor.dtype.sspec
        t = tensor.original
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
        # fmt: off
        # if tensor.dtype.is_codebook:
        #     codebook_info = (nspec.name, None, None) if nspec.is_implicit else (None, *nspec.get_codebooks())
        #     codebook_cast_kernel[grid](
        #         *input_info, *output_info, *scale_info, *meta_info, *mask_info, *codebook_info,
        #         Modes.round, Modes.scale, Modes.cast, *nspec_info, *sspec_info
        #     )
        # elif sspec.zero:
        #     scalef, zerof = sspec.scale, sspec.zero
        #     s_spec = (scalef.emin, scalef.mbits, scalef.finfo.maxfloat)
        #     z_spec = (zerof.emin, zerof.mbits, zerof.finfo.maxfloat)
        #     asymmetric_cast_kernel[grid](
        #         *input_info, *output_info, *scale_info, *zero_info, *mask_info, Modes.round, Modes.scale, Modes.cast,
        #         nspec.bits, *s_spec, *z_spec, *sspec_info
        #     )
        # elif sspec.scale.is_float and nspec.is_int:
        #     s_spec = (sspec.scale.emin, sspec.scale.mbits, sspec.scale.finfo.maxfloat)
        #     symmetric_cast_kernel[grid](
        #         *input_info, *output_info, *scale_info, *mask_info, Modes.round, Modes.scale, Modes.cast,
        #         nspec.bits, *s_spec, *sspec_info
        #     )
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
    def upcast(tensor: Tensor, torch_dtype: torch.dtype):
        """Upcast data to the specified dtype."""
        raise NotImplementedError("Upcast not implemented for TritonCast.")

    @staticmethod
    def compress(tensor: Tensor):
        """Compress data to the specified dtype."""
        raise NotImplementedError("Compress not implemented for TritonCast.")

    @staticmethod
    def decompress(tensor: Tensor):
        """Decompress data to the specified dtype."""
        raise NotImplementedError("Decompress not implemented for TritonCast.")
