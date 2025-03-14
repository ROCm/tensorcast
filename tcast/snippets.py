#!/usr/bin/env python
# tcast/snippets.py: triton methods for cconfiguration and casting
# SPDX-License-Identifier: MIT
# ruff: noqa: D103

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import triton
import triton.language as tl

# these encode the configuration in a 32-bit integer
# first 8 bits are the block size and axes
LP_SIZE_POS: tl.constexpr = 0
LP_SQUARE_POS: tl.constexpr = 3
LP_AXIS0_POS: tl.constexpr = 4
LP_AXIS1_POS: tl.constexpr = 6
# next 4 bits are the scale type
LP_SCALE_POS: tl.constexpr = 8
# next 18 bits are the datatypes for q (act), k (weight), v, p, do, ds
LP_Q_POS: tl.constexpr = 11
LP_K_POS: tl.constexpr = 14
LP_V_POS: tl.constexpr = 17
LP_P_POS: tl.constexpr = 20
LP_DO_POS: tl.constexpr = 23
LP_DS_POS: tl.constexpr = 26
# next 2 bits are the icp flags
LP_ICP_QK_POS: tl.constexpr = 29
LP_ICP_PV_POS: tl.constexpr = 30
LP_ICP_FP32_POS: tl.constexpr = 31
# now we are headed for 64 bits...
LP_SCALEMODE_POS: tl.constexpr = 32  # 3 bits, FLOOR, CEIL, MIDMAX, OPTION3, TOPBINADE
LP_ROUNDMODE_POS: tl.constexpr = 35  # 2 bits, ZERO, AWAY, EVEN, STOCHASTIC
LP_CASTMODE_POS: tl.constexpr = 37  # 2 bits, VIRTUAL, ACTUAL, COMPRESSED
# these are the masks for the configuration
LP_SIZE_MASK: tl.constexpr = 0x7
LP_AXIS_MASK: tl.constexpr = 0x3
LP_SQUARE_MASK: tl.constexpr = 0x1
LP_SCALE_MASK: tl.constexpr = 0x7
LP_TENSOR_MASK: tl.constexpr = 0x7
LP_ICP_MASK: tl.constexpr = 0x1
LP_ROUNDMODE_MASK: tl.constexpr = 0x3
LP_SCALEMODE_MASK: tl.constexpr = 0x7
LP_CASTMODE_MASK: tl.constexpr = 0x3
# these are the indices for the tensors to be quantized
LP_Q_INDEX: tl.constexpr = 0
LP_K_INDEX: tl.constexpr = 1
LP_V_INDEX: tl.constexpr = 2
LP_P_INDEX: tl.constexpr = 3
LP_DO_INDEX: tl.constexpr = 4
LP_DS_INDEX: tl.constexpr = 5
LP_ACT_INDEX: tl.constexpr = 0
LP_WEIGHT_INDEX: tl.constexpr = 1
# roundmode
LP_RMODE_ZERO: tl.constexpr = 0
LP_RMODE_AWAY: tl.constexpr = 1
LP_RMODE_EVEN: tl.constexpr = 2
LP_RMODE_STOCHASTIC: tl.constexpr = 3
# scalemode
LP_SMODE_FLOOR: tl.constexpr = 0
LP_SMODE_CEIL: tl.constexpr = 1
LP_SMODE_MIDMAX: tl.constexpr = 2
LP_SMODE_OPTION3: tl.constexpr = 3
LP_SMODE_TOPBINADE: tl.constexpr = 4
# castmode
LP_CMODE_VIRTUAL: tl.constexpr = 0
LP_CMODE_ACTUAL: tl.constexpr = 1
LP_CMODE_COMPRESSED: tl.constexpr = 2
# scale types: these are the dtypes containing the scales; None means we use the input tensor type as the scale type
LP_STYPE_MATCH: tl.constexpr = 0
LP_STYPE_FP32: tl.constexpr = 1
LP_STYPE_FP16: tl.constexpr = 2
LP_STYPE_BF16: tl.constexpr = 3
LP_STYPE_E8M0: tl.constexpr = 4
LP_STYPE_E5M3: tl.constexpr = 5
LP_STYPE_TLTYPE: tl.constexpr = [tl.float32, tl.float16, tl.bfloat16, tl.uint8, tl.uint8]
# quant types: these are the actual dtypes we are casting to; fp6 uses fp8e4, fp4 uses uint8 (2 packed values per uint8)
# None means we do not quantize at all
LP_QUANT_NONE: tl.constexpr = 0
LP_QUANT_E5M2: tl.constexpr = 1
LP_QUANT_E5M2B16: tl.constexpr = 2
LP_QUANT_E4M3FN: tl.constexpr = 3
LP_QUANT_E4M3FNUZ: tl.constexpr = 4
LP_QUANT_E3M2FNUZ: tl.constexpr = 5
LP_QUANT_E2M3FNUZ: tl.constexpr = 6
LP_QUANT_E2M1FNUZ: tl.constexpr = 7
LP_QUANT_TLTYPE: tl.constexpr = [
    tl.float8e5, tl.float8e5b16, tl.float8e4nv, tl.float8e4b8, tl.float8e4nv, tl.float8e4nv, tl.float8e4nv
]


# these functions use the constants above to extract the configuration
# fmt: off
@triton.jit
def lp_enabled(LPCODE: tl.constexpr) -> tl.constexpr: return LPCODE != 0
@triton.jit
def lp_get_size(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_SIZE_POS) & LP_SIZE_MASK
@triton.jit
def lp_is_square(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_SQUARE_POS) & LP_SQUARE_MASK
@triton.jit
def lp_quant_code(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    tl.static_assert(TCODE < 6)
    return (LPCODE >> (LP_Q_POS + TCODE * 3)) & LP_TENSOR_MASK
@triton.jit # do we quantize this tensor?
def lp_do_quant(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, TCODE) != 0
@triton.jit
def lp_number_mxfp(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, TCODE) > 4
@triton.jit
def lp_q_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, LP_Q_INDEX)
@triton.jit
def lp_k_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, LP_K_INDEX)
@triton.jit
def lp_v_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, LP_V_INDEX)
@triton.jit
def lp_p_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, LP_P_INDEX)
@triton.jit
def lp_do_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, LP_DO_INDEX)
@triton.jit
def lp_ds_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_quant_code(LPCODE, LP_DS_INDEX)
# incoherence processing flags
@triton.jit
def lp_icp_qk(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_ICP_QK_POS) & LP_ICP_MASK
@triton.jit
def lp_icp_pv(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_ICP_PV_POS) & LP_ICP_MASK
@triton.jit
def lp_icp_fp32(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_ICP_FP32_POS) & LP_ICP_MASK
@triton.jit
def lp_needs_icp(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    is_qkds = TCODE in (LP_Q_INDEX, LP_K_INDEX, LP_DS_INDEX)
    return lp_icp_qk(LPCODE) and is_qkds or lp_icp_pv(LPCODE) and not is_qkds
@triton.jit
def lp_act_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_q_code(LPCODE)
@triton.jit
def lp_weight_code(LPCODE: tl.constexpr) -> tl.constexpr: return lp_k_code(LPCODE)
@triton.jit
def lp_roundmode(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_ROUNDMODE_POS) & LP_ROUNDMODE_MASK
@triton.jit
def lp_scalemode(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_SCALEMODE_POS) & LP_SCALEMODE_MASK
@triton.jit
def lp_castmode(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_CASTMODE_POS) & LP_CASTMODE_MASK

# scale type functions
@triton.jit
def lp_scale_code(LPCODE: tl.constexpr) -> tl.constexpr: return (LPCODE >> LP_SCALE_POS) & LP_SCALE_MASK
@triton.jit
def lp_match_scale_to_input(LPCODE: tl.constexpr, ) -> tl.constexpr: return lp_scale_code(LPCODE) == 0
@triton.jit
def lp_scale_is_fp32(LPCODE: tl.constexpr) -> tl.constexpr: return lp_scale_code(LPCODE) == 1
@triton.jit
def lp_scale_is_fp16(LPCODE: tl.constexpr) -> tl.constexpr: return lp_scale_code(LPCODE) == 2
@triton.jit
def lp_scale_is_bf16(LPCODE: tl.constexpr) -> tl.constexpr: return lp_scale_code(LPCODE) == 3
@triton.jit
def lp_scale_is_exponent(LPCODE: tl.constexpr) -> tl.constexpr: return lp_scale_code(LPCODE) == 4
# fmt: on


@triton.jit
def lp_scale_dtype(LPCODE: tl.constexpr, dtype=None) -> tl.constexpr:
    t = lp_scale_code(LPCODE)
    tl.static_assert(t < 5)
    tl.device_assert((dtype is None) != (t == 0))
    if t == LP_STYPE_MATCH:
        return dtype
    return LP_STYPE_TLTYPE[t-1]

@triton.jit
def lp_quant_dtype(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    t = lp_quant_code(LPCODE, TCODE)
    if t == LP_QUANT_NONE:
        return None
    return LP_QUANT_TLTYPE[t-1]


# bit cast functions
# fmt: off
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

# number format information
@triton.jit
def lp_number_mbits(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    return (2., 2., 3., 3., 2., 3., 1.)[lp_quant_code(LPCODE, TCODE-1)]
@triton.jit
def lp_number_ebits(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    return (5., 5., 4., 4., 3., 2., 2.)[lp_quant_code(LPCODE, TCODE-1)]
@triton.jit
def lp_number_emax(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    return (15., 15., 8., 7., 4., 2., 2.)[lp_quant_code(LPCODE, TCODE-1)]
@triton.jit
def lp_number_emin(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    return (-14., -15., -6., -7., -2., 0., 0.)[lp_quant_code(LPCODE, TCODE-1)]
@triton.jit
def lp_number_maxval(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    return (57344., 57344., 448., 240., 28., 7.5, 6.)[lp_quant_code(LPCODE, TCODE-1)]
@triton.jit
def lp_number_midmax(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    return (61440., 61440., 480., 248., 28., 7.75, 7.)[lp_quant_code(LPCODE, TCODE-1)]
# fmt: on


@triton.jit
def get_exponent(x):
    """Get the exponent of a float."""
    tl.device_assert(x.dtype == tl.float32)
    return (as_uint(x) >> 23) & 0xFF


@triton.jit
def modify_exponent(x, y, direction: int = 0):
    """Modify the exponent of a float. direction is 1 for add, -1 for subtract, 0 for replace."""
    tl.device_assert(x.dtype == tl.float32)
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
def round(x: tl.tensor, roundmode, seed=19, offset=0) -> tl.tensor:
    """Round floating point."""
    if roundmode == LP_RMODE_STOCHASTIC:
        rand_ptr = tl.randn(seed, offset)
        return tl.libdevice.trunc(x + tl.where(x < 0, -rand_ptr), rand_ptr)
    if roundmode == LP_RMODE_EVEN:
        return tl.libdevice.rint(x)
    if roundmode == LP_RMODE_AWAY:
        return tl.libdevice.round(x)
    if roundmode == LP_RMODE_ZERO:
        trunc = tl.libdevice.trunc(x)
        return tl.where(x - trunc == 0.5, trunc, tl.libdevice.round(x))


USE_ROUND: tl.constexpr = False

# mapping fp4 in fp32 to int4 (keep for later)
# also, quant to e4m3fn: fake quant fp4 to fp32, cast to fp8, bitcast to uint8
# then... ((x>>4) & c) | ((x >> 2) & 3) packs it into 4 bits
# but the denorm is 0 or -0, so we need to handle that
# where x != 0 and  (q & 7) == 0, q | 1 : q
# then pack with 1 0 3 2 5 4 7 6...
# for fp32 direct, we need the sign bit, then max(exponent - 126, 0)
# ((x >> 28) & 0x4) | (max(126, ((x >> 23) & 0x7f) - 126) << 1) | (x >> 23) & 1
# we want bits 30:22 259
# sign is f32 >> 30 << 3
# 6.0 exp 129 mant 1 uint4 0 11 1 in e4m3fn it is   0x4c -> 0x7
# 4.0 exp 129 mant 0 uint4 0 11 0                   0x48 -> 0x6
# 3.0 exp 128 mant 1 uint4 0 10 1                   0x44
# 2.0 exp 128 mant 0 uint4 0 10 0                   0x40
# 1.5 exp 127 mant 1 uint4 0 01 1                   0x3c
# 1.0 exp 127 mant 0 uint4 0 01 0                   0x38
# 0.5 exp 126 mant 0 uint4 0 00 1                   0x34
# 0.0 exp 0 mant 0   uint4 0 00 0                   0x30

@triton.jit
def quantize_float(
    x, scale, shared_exp, LPCODE: tl.constexpr, TCODE: tl.constexpr, seed=19, offset=0, roundmode=None, clip=True
):
    """Quantize the float values with a float or exponent scale (OCP MX biased)."""
    tl.static_assert(lp_enabled(LPCODE) and lp_do_quant(LPCODE, TCODE))
    if roundmode is None:
        roundmode = lp_roundmode(LPCODE)

    # There is the input tensor triton dtype, the target dtype, and the output triton dtype.
    # For castmode == virtual, we quantize to the target dtype, but the output dtype is the input dtype.
    # For castmode == actual, we quantize to the target dtype, and the output dtype is the target dtype if
    # it exists in triton, otherwise we upcast to the output triton dtype.
    # For castmode == compress, we pack two e2m1 values into tl.uint8
    target_code = lp_quant_code(LPCODE, TCODE)
    input_dtype = x.type.element_ty
    VIRTUAL = lp_castmode(LPCODE) == LP_CMODE_VIRTUAL
    COMPRESSED = lp_castmode(LPCODE) == LP_CMODE_COMPRESSED
    if VIRTUAL:
        output_dtype = input_dtype
    elif lp_castmode(LPCODE) == LP_CMODE_ACTUAL:
        output_dtype = lp_quant_dtype(LPCODE, TCODE)
    elif COMPRESSED and target_code == LP_QUANT_E2M1FNUZ:
        output_dtype = tl.uint8

    # whether the scale is an exponent or a float, this will get us into the range of the target dtype
    tmpscale = tl.exp2(scale) if scale.dtype.is_uint8() else scale
    out = x * tmpscale

    # If roundmode is even and the target dtype is a tl fp8, we can just cast directly using tl.cast.
    # Otherwise we will have to do fake quantization in fp32 and then cast to the target dtype.

    if roundmode == LP_RMODE_EVEN and target_code < LP_QUANT_E3M2FNUZ:
        return tl.cast(out, lp_quant_dtype(LPCODE, TCODE)).to(output_dtype)

    # do fake quantization in fp32, then go from there
    values = tl.cast(out, tl.float32)
    valexp = get_exponent(out)
    rscale = tl.exp2(lp_number_mbits(LPCODE, TCODE) - tl.maximum(valexp, lp_number_emin(LPCODE, TCODE)))
    values *= rscale # scale based on mbits to allow rounding on the rescaled values
    if USE_ROUND:
        values = round(values, LPCODE, roundmode, seed, offset)
    else:
        # clunky integer method
        EMIN = lp_number_emax(LPCODE, TCODE) + 127
        MAXVAL: tl.constexpr = lp_number_maxval(LPCODE, TCODE)
        MBITS: tl.constexpr = lp_number_mbits(LPCODE, TCODE)
        values = modify_exponent(values, shared_exp, direction=-1)
        valexp = get_exponent(values)
        shift = tl.where(EMIN <= valexp, 0, tl.minimum(MBITS, EMIN - valexp))
        roundbit = 1 << (23 - 1 - MBITS + shift)
        mask = (roundbit << 1) - 1
        values = as_uint(values)
        if roundmode == LP_RMODE_AWAY:  # ties away from zero
            values += roundbit
        elif roundmode == LP_RMODE_STOCHASTIC:
            r = tl.randint(seed, values)
            r = r & mask
            values += r
        else:
            tie = ((roundbit - 1) & values) == 0
            if roundmode == LP_RMODE_ZERO:  # ties towards zero, not trunc
                values += tl.where(tie, 0, roundbit)
            if roundmode == LP_RMODE_EVEN:  # ties to nearest even
                values += tl.where(tie & (values & (roundbit << 1) == 0), 0, roundbit)
        valexp = get_exponent(values)
        shift = tl.where(EMIN <= valexp, 0, EMIN - valexp)
        mask = (1 << (23 - MBITS + shift)) - 1
        values = as_float(tl.where(shift > MBITS or valexp == 255, 0, values & ~mask))
    values /= rscale
    if clip:
        values = tl.clamp(values, min=-MAXVAL, max=MAXVAL)
    values /= rscale
    # values remain target dtype values, will need to be unscaled for virtual cast
    if VIRTUAL:
        values /= scale
    if COMPRESSED and target_code == LP_QUANT_E2M1FNUZ:
        values = as_uint(values.to(tl.float8e4nv))
        ((values >> 28) & 0x4) | (max(126, ((x >> 23) & 0x7f) - 126) << 1) | (x >> 23) & 1
        values = ((values >> 4) & 0x7) | ((values >> 2) & 3)
    else:
        values = tl.cast(values, output_dtype)
    return values


@triton.jit
def get_shared_exponent(values, LPCODE: tl.constexpr, TCODE: tl.constexpr):
    """Find the shared exponent for a block of values."""
    absmax = tl.max(tl.abs(values))
    shared_exp = get_exponent(absmax)  # shared_exp is a biased exponent
    # set the exponent of absmax to emax comparing with midmax, maxfloat, etc.
    EMAX = lp_number_emax(LPCODE, TCODE)
    MAXFLOAT = lp_number_maxval(LPCODE, TCODE)
    absmax = modify_exponent(absmax, 127 + EMAX)
    increment = tl.zeros_like(shared_exp)
    scalemode = lp_scalemode(LPCODE)
    if scalemode == LP_SMODE_CEIL:
        increment = as_uint(tl.where(as_uint(absmax) & 0x007FFFFF != 0, 1, 0))
    elif scalemode == LP_SMODE_MIDMAX:
        increment = as_uint(tl.where(absmax >= lp_number_midmax(LPCODE, TCODE), 1, 0))
    if scalemode == LP_SMODE_TOPBINADE:
        increment = as_uint(tl.where(absmax > MAXFLOAT, 1, 0))
    if scalemode == LP_SMODE_OPTION3:
        EMIN = lp_number_emin(LPCODE, TCODE)
        rounded = quantize_float(absmax, 127, EMIN, LPCODE, TCODE, 0, 0, LP_RMODE_EVEN, clip=False)
        increment = as_uint(tl.where(get_exponent(as_uint(rounded)) != 127 + EMAX, 1, 0))
    # returns the biased exponent, adjusted by the dtype's emax
    shared_exp = shared_exp + increment - EMAX
    return shared_exp


@triton.jit
def apply_incoherence(x, imatrix, trans=False, use_fp32=False):
    """Applies incoherence processing to the given tensor."""
    if use_fp32:
        if trans:
            x = tl.dot(tl.cast(imatrix, tl.float32), tl.cast(x, tl.float32)).to(x.type.element_ty)
        else:
            x = tl.dot(tl.cast(x, tl.float32), tl.cast(imatrix, tl.float32)).to(x.type.element_ty)
    else:
        if trans:
            x = tl.dot(tl.cast(imatrix, x.type.element_ty), x).to(x.type.element_ty)
        else:
            x = tl.dot(x, tl.cast(imatrix, x.type.element_ty)).to(x.type.element_ty)
    return x


@triton.jit
def scale_and_quantize(x, imatrix, LPCODE: tl.constexpr, TCODE: tl.constexpr, seed=19, offset=0, trans=False):
    """Returns the quantized x and the scale used."""
    if not (lp_enabled(LPCODE) and lp_do_quant(LPCODE, TCODE)):
        return x, 1.0
    tl.static_assert(lp_enabled(LPCODE))
    if lp_match_scale_to_input(LPCODE):
        stype = x.type.element_ty
    else:
        stype = lp_scale_dtype(LPCODE)
    MAXVAL: tl.constexpr = lp_number_maxval(LPCODE, TCODE)
    if lp_needs_icp(LPCODE, TCODE):
        x = apply_incoherence(x, imatrix, trans)
    absx = tl.abs(x)
    absmax = tl.max(absx)
    if stype == tl.uint8:
        shared_exp = get_shared_exponent(absx, LPCODE, TCODE)
        scale = tl.exp2(shared_exp - 127)
    else:
        shared_exp = lp_number_emax(LPCODE, TCODE)
        scale = MAXVAL / absmax
    out = quantize_float(x, scale, shared_exp, LPCODE, TCODE, seed=seed, offset=offset)
    return out, scale.to(stype)
