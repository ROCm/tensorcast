#!/usr/bin/env python
# tcast/kernels.py: triton kernels for TensorCast
# SPDX-License-Identifier: MIT
# ruff: noqa: D103

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import triton
import triton.language as tl
from triton.language.extra import libdevice

# mode constants
RMODE_ZERO = tl.constexpr(0)
RMODE_AWAY = tl.constexpr(1)
RMODE_EVEN = tl.constexpr(2)
RMODE_STOCHASTIC = tl.constexpr(3)
SMODE_FLOOR = tl.constexpr(0)
SMODE_CEIL = tl.constexpr(1)
SMODE_MIDMAX = tl.constexpr(2)
SMODE_OPTION3 = tl.constexpr(3)
SMODE_TOPBINADE = tl.constexpr(4)
CMODE_VIRTUAL = tl.constexpr(0)
CMODE_ACTUAL = tl.constexpr(1)
CMODE_COMPRESSED = tl.constexpr(2)


# utilities
# fmt: off
@triton.jit
def get_triton_dtype(name: str) -> tl.core.dtype: return tl.str_to_ty(name)
@triton.jit
def as_float(x): return tl.cast(x, tl.float32, bitcast=True)
@triton.jit
def as_int(x): return tl.cast(x, tl.int32, bitcast=True)
@triton.jit
def as_uint(x): return tl.cast(x, tl.uint32, bitcast=True)
@triton.jit
def as_uint8(x): return tl.cast(x, tl.uint8, bitcast=True)
@triton.jit
def as_type(x, dtype): return tl.cast(x, dtype, bitcast=True)
@triton.jit
def bias_scale(scale, emax): return (scale - emax + 127).to(tl.uint8)
@triton.jit
def unbias_scale(scale, emax): return (scale.to(tl.int32) + emax - 127)
# fmt: on


@triton.jit
def get_exponent(x, unbiased: bool = True):
    """Get the exponent of a float, returned as int32."""
    if x.dtype != tl.uint32:
        tl.device_assert(x.dtype == tl.float32, "get_exponent input must be fp32 or uint32")
        x = as_uint(x)
    return ((x >> 23) & 0xFF).to(tl.int32) - (127 * int(unbiased))


@triton.jit
def modify_exponent(x, y, replace: bool = False):
    """Modify the exponent of a float. dir is 1 for add, -1 for subtract, 0 for replace."""
    # y is unbiased exponent or exponent offest, type int32
    tl.device_assert(x.dtype == tl.float32 or x.dtype == tl.uint32, "modify_exponent input must be fp32 or uint32")
    xexp = get_exponent(x, unbiased=False) # get biased exponents to check for special case of 0
    mask = xexp == 0
    if replace:
        xexp = y + 127
    else:
        xexp += y
    return as_float(tl.where(mask, 0, x & 0x807FFFFF | (xexp << 23)))


@triton.jit
def _round_device(x, roundmode, seed, offset): # TODO(ericd): get libdevice to work
    """Round floating point using libdevice."""
    if roundmode == RMODE_STOCHASTIC:
        rand_ptr = tl.randn(seed, offset)
        return libdevice.trunc(x + tl.where(x < 0, -rand_ptr), rand_ptr)
    if roundmode == RMODE_EVEN:
        return libdevice.rint(x)
    if roundmode == RMODE_AWAY:
        return libdevice.round(x)
    trunc = libdevice.trunc(x) # ROUND_ZERO
    return tl.where(x - trunc == 0.5, trunc, libdevice.round(x))


@triton.jit
def _round_bitcast(x, denorm, mbits, roundmode, seed, offset):
    """Round bitcasted float values."""
    shift = 23 - mbits + denorm
    roundbit = 1 << (shift - 1)
    mask = (roundbit << 1) - 1
    x = as_uint(x)
    if roundmode == RMODE_AWAY:  # ties away from zero
        x += roundbit
    elif roundmode == RMODE_STOCHASTIC: # TODO(ericd): get this to work
        x += tl.randint(seed, x) & mask
    else:
        tie = ((roundbit - 1) & x) == 0
        if roundmode == RMODE_ZERO:  # ties towards zero, not trunc
            x += tl.where(tie, 0, roundbit)
        elif roundmode == RMODE_EVEN:  # ties to nearest even
            x += tl.where(tie & (x & (roundbit << 1) == 0), 0, roundbit)
    return as_float(x >> shift << shift)


@triton.jit
def round(x, denorm, emax, mbits, roundmode, use_bitcast, seed, offset):
    """Round the float values."""
    if use_bitcast:
        x = _round_bitcast(x, denorm, mbits, roundmode, seed, offset)
    else:
        x = modify_exponent(x, (mbits - denorm) - emax)  # scale to exponent 0, then up to account for mbits and denorm
        x = _round_device(x, roundmode, seed, offset)
        x = modify_exponent(x, emax - (mbits - denorm)) # descale the mbits and scale up to EMAX for clamp
    return x


@triton.jit
def quantize_float(
    x, scale, emax, emin, mbits, maxfloat, roundmode, castmode, is_exp, otype, use_bitcast=True, clip=True, seed=19, offset=0
):
    """
    Quantize the values with a float or exponent scale (OCP MX biased).

    x is the float input tensor (fp32, fp16, or bf16)
    scale is the scale factor (float or exponent)
    emax, emin, mbits, maxfloat are the target dtype parameters
    roundmode is 0: ROUND_ZERO, 1: ROUND_AWAY, 2: ROUND_EVEN, 3: ROUND_STOCHASTIC
    castmode is 0: CMODE_VIRTUAL, 1: CMODE_ACTUAL, 2: CMODE_COMPRESSED
    is_exp is True if the scale is exponent (OCP MX biased E8M0), otherwise it is float
    otype is the output datatype type
    use_bitcast is True if the rounding should be done with the integer method vs the libdevice method
    clip is True if the output should be clamped to the target dtype range
    seed and offset are for stochastic rounding only
    """
    if is_exp:
        scale = tl.exp2(emax - scale)  # scale is exponent scale (OCP MX biased)
    values = as_uint(x.to(tl.float32) * scale)
    denorm = as_int(tl.clamp(as_float(emin - get_exponent(values)), 0.0, mbits))
    values = round(values, denorm, emax, mbits, roundmode, use_bitcast, seed, offset)
    if clip:
        values = tl.clamp(values, min=-maxfloat, max=maxfloat)
    if castmode == CMODE_VIRTUAL:
        values /= scale
        otype = x.type.element_ty
    return values.to(otype)


@triton.jit
def get_descale(scale_or_exponent, emax, is_exp):
    """Convert the scale into a descale float."""
    # An exponent scale is stored as a biased uint8 that is offset by EMAX,
    # i.e. exp_scale = shared_exp - EMAX + 127, so descale = (exp_scale - 127 + EMAX).exp2()
    if is_exp:
        return 1.0 / tl.exp2(scale_or_exponent.to(tl.int32) - 127 + emax)
    else:
        return 1.0 / scale_or_exponent


@triton.jit
def get_shared_exponent(x, scalemode, emax, maxfloat, midmax, biased=False):
    """Find the shared exponent for a block of values."""
    # shared_exp is unbiased exponent (maxexp or maxexp + 1), stored as int32
    absmax = tl.max(tl.abs(x.to(tl.float32)))
    shared_exp = get_exponent(absmax)
    absmax = modify_exponent(absmax, emax, replace=True) # scale absmax to top target dtype emax
    increment = tl.zeros_like(shared_exp)
    if scalemode == SMODE_CEIL:
        increment[as_uint(absmax) & 0x007FFFFF != 0] = 1
    elif scalemode == SMODE_MIDMAX:
        increment[absmax >= midmax] = 1
    if scalemode == SMODE_TOPBINADE:
        increment[absmax > maxfloat] = 1
    if scalemode == SMODE_OPTION3:
        rounded = quantize_float(absmax, 1.0, 0, 0, 0, RMODE_EVEN, clip=False)
        increment[get_exponent(rounded) != emax] = 1
    shared_exp += increment
    if biased:
        shared_exp = (shared_exp + 127 - emax).to(tl.uint8)
    return shared_exp


@triton.jit
def get_scale(x, is_exp, emax, biased=False, scalemode=SMODE_FLOOR, maxfloat=0.0, midmax=0.0):
    """Find the scale factor, returning the exponent."""
    # An exponent scale is stored as a biased uint8 that is offset by EMAX,
    # i.e. exp_scale = shared_exp - EMAX + 127, so descale = (exp_scale - 127 + EMAX).exp2()
    # or an unbiased int32 with no offset
    if is_exp:
        return get_shared_exponent(x, scalemode, emax, maxfloat, midmax, biased)
    else:
        return maxfloat / tl.max(tl.abs(x))


@triton.jit
def apply_incoherence(x, imatrix, trans=False, use_fp32=False):
    """Applies incoherence processing to the given tensor."""
    # trans means that the input matrix is the second input matrix to tl.dot
    # use_fp32 means cast both the input and the incoherence matrix to fp32 before dot product
    if use_fp32:
        if trans:
            out = tl.dot(imatrix.to(tl.float32), x.to(tl.float32))
        else:
            out = tl.dot(x.to(tl.float32), imatrix.to(tl.float32))
    else:
        cast_im = imatrix.to(x.type.element_ty)
        if trans:
            out = tl.dot(cast_im, x)
        else:
            out = tl.dot(x, cast_im)
    # cast the result to the same type as x
    return out.to(x.type.element_ty)
