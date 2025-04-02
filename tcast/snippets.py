#!/usr/bin/env python
# tcast/snippets.py: triton methods for configuration and casting
# SPDX-License-Identifier: MIT
# ruff: noqa: D103

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import triton
import triton.language as tl
from triton.language.extra import libdevice

# these encode the configuration in a 64-bit integer
SIZE_POS = tl.constexpr(0)
SQUARE_POS = tl.constexpr(3)
FP8_TYPE_POS = tl.constexpr(4)
# next 4 bits are the scale type
SCALE_POS = tl.constexpr(8)
# next 18 bits are the datatypes for q (act), k (weight), v, p, do, ds
Q_POS = tl.constexpr(11)
K_POS = tl.constexpr(14)
V_POS = tl.constexpr(17)
P_POS = tl.constexpr(20)
DO_POS = tl.constexpr(23)
DS_POS = tl.constexpr(26)
# next 2 bits are the icp flags
ICP_QK_POS = tl.constexpr(29)
ICP_PV_POS = tl.constexpr(30)
ICP_FP32_POS = tl.constexpr(31)
# now we are headed for 64 bits...
SCALEMODE_POS = tl.constexpr(32)  # 3 bits, FLOOR, CEIL, MIDMAX, OPTION3, TOPBINADE
ROUNDMODE_POS = tl.constexpr(35)  # 2 bits, ZERO, AWAY, EVEN, STOCHASTIC
CASTMODE_POS = tl.constexpr(37)  # 2 bits, VIRTUAL, ACTUAL, COMPRESSED
AXIS0_POS = tl.constexpr(39)
AXIS1_POS = tl.constexpr(41)
# these are the masks for the configuration
SIZE_MASK = tl.constexpr(7)
FP8_TYPE_MASK = tl.constexpr(15)
AXIS_MASK = tl.constexpr(3)
SQUARE_MASK = tl.constexpr(1)
SCALE_MASK = tl.constexpr(7)
TENSOR_MASK = tl.constexpr(7)
ICP_MASK = tl.constexpr(1)
ROUNDMODE_MASK = tl.constexpr(3)
SCALEMODE_MASK = tl.constexpr(7)
CASTMODE_MASK = tl.constexpr(3)
# these are the indices for the tensors to be quantized
Q_INDEX = tl.constexpr(0)
K_INDEX = tl.constexpr(1)
V_INDEX = tl.constexpr(2)
P_INDEX = tl.constexpr(3)
DO_INDEX = tl.constexpr(4)
DS_INDEX = tl.constexpr(5)
ACT_INDEX = tl.constexpr(0)
WEIGHT_INDEX = tl.constexpr(1)
# roundmode
RMODE_ZERO = tl.constexpr(0)
RMODE_AWAY = tl.constexpr(1)
RMODE_EVEN = tl.constexpr(2)
RMODE_STOCHASTIC = tl.constexpr(3)
# scalemode
SMODE_FLOOR = tl.constexpr(0)
SMODE_CEIL = tl.constexpr(1)
SMODE_MIDMAX = tl.constexpr(2)
SMODE_OPTION3 = tl.constexpr(3)
SMODE_TOPBINADE = tl.constexpr(4)
# castmode
CMODE_VIRTUAL = tl.constexpr(0)
CMODE_ACTUAL = tl.constexpr(1)
CMODE_COMPRESSED = tl.constexpr(2)
# scale types: these are the dtypes containing the scales; None means we use the input tensor type as the scale type
STYPE_MATCH = tl.constexpr(0)
STYPE_FP32 = tl.constexpr(1)
STYPE_FP16 = tl.constexpr(2)
STYPE_BF16 = tl.constexpr(3)
STYPE_E8M0 = tl.constexpr(4)
STYPE_E5M3 = tl.constexpr(5)
STYPE_TLTYPE = ("fp32", "fp16", "bf16", "u8", "u8")
# quant types: these are the actual dtypes we are casting to; fp6 uses fp8e4, fp4 uses uint8 (2 packed values per uint8)
# None means we do not quantize at all
QUANT_NONE = tl.constexpr(0)
QUANT_E5M2 = tl.constexpr(1)
QUANT_E5M2B16 = tl.constexpr(2)
QUANT_E4M3FN = tl.constexpr(3)
QUANT_E4M3FNUZ = tl.constexpr(4)
QUANT_E3M2FNUZ = tl.constexpr(5)
QUANT_E2M3FNUZ = tl.constexpr(6)
QUANT_E2M1FNUZ = tl.constexpr(7)
QUANT_TLTYPE = ("fp8e5", "fp8e5b16", "fp8e4nv", "fp8e4b8")


# these functions use the constants above to extract the configuration
# fmt: off
@triton.jit
def enabled(LPCODE: tl.constexpr) -> bool: return LPCODE.value != 0
@triton.jit
def shift_mask(LPCODE: tl.constexpr, POS: tl.constexpr, MASK: tl.constexpr): return LPCODE.value >> POS.value & MASK.value
@triton.jit
def fp8_code(LPCODE: tl.constexpr) -> int: return shift_mask(LPCODE, FP8_TYPE_POS, FP8_TYPE_MASK)
@triton.jit
def fp8_supported(LPCODE: tl.constexpr, NCODE) -> bool: return (fp8_code(LPCODE) & (1 << NCODE - 1 != 0)) != 0
@triton.jit
def get_size(LPCODE: tl.constexpr) -> int: return (LPCODE >> SIZE_POS) & SIZE_MASK
@triton.jit
def is_square(LPCODE: tl.constexpr) -> int: return (LPCODE >> SQUARE_POS) & SQUARE_MASK
@triton.jit
def quant_pos(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> int: return (Q_POS, K_POS, V_POS, P_POS, DO_POS, DS_POS)[TCODE]
@triton.jit
def quant_code(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> int: return shift_mask(LPCODE, quant_pos(LPCODE, TCODE), TENSOR_MASK)
@triton.jit
def needs_quant(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> bool: return quant_code(LPCODE, TCODE) != 0
@triton.jit
def number_mxfp(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> bool: return quant_code(LPCODE, TCODE) > 4
@triton.jit
def q_code(LPCODE: tl.constexpr) -> int: return quant_code(LPCODE, Q_INDEX)
@triton.jit
def k_code(LPCODE: tl.constexpr) -> int: return quant_code(LPCODE, K_INDEX)
@triton.jit
def v_code(LPCODE: tl.constexpr) -> int: return quant_code(LPCODE, V_INDEX)
@triton.jit
def p_code(LPCODE: tl.constexpr) -> int: return quant_code(LPCODE, P_INDEX)
@triton.jit
def do_code(LPCODE: tl.constexpr) -> int: return quant_code(LPCODE, DO_INDEX)
@triton.jit
def ds_code(LPCODE: tl.constexpr) -> int: return quant_code(LPCODE, DS_INDEX)
@triton.jit
def icp_qk(LPCODE: tl.constexpr) -> bool: return shift_mask(LPCODE, ICP_QK_POS, ICP_MASK) != 0
@triton.jit
def icp_pv(LPCODE: tl.constexpr) -> bool: return shift_mask(LPCODE, ICP_PV_POS, ICP_MASK) != 0
@triton.jit
def icp_fp32(LPCODE: tl.constexpr) -> bool: return shift_mask(LPCODE, ICP_FP32_POS, ICP_MASK) == 0
@triton.jit
def act_code(LPCODE: tl.constexpr) -> int: return q_code(LPCODE)
@triton.jit
def weight_code(LPCODE: tl.constexpr) -> int: return k_code(LPCODE)
@triton.jit
def roundmode(LPCODE: tl.constexpr) -> int: return shift_mask(LPCODE, ROUNDMODE_POS, ROUNDMODE_MASK)
@triton.jit
def roundeven(LPCODE: tl.constexpr) -> int: return roundmode(LPCODE) == int(RMODE_EVEN)
@triton.jit
def scalemode(LPCODE: tl.constexpr) -> int: return shift_mask(LPCODE, SCALEMODE_POS, SCALEMODE_MASK)
@triton.jit
def castmode(LPCODE: tl.constexpr) -> int: return shift_mask(LPCODE, CASTMODE_POS, CASTMODE_MASK)
@triton.jit
def virtual(LPCODE: tl.constexpr) -> int: return castmode(LPCODE) == int(CMODE_VIRTUAL)
@triton.jit
def scale_code(LPCODE: tl.constexpr): return shift_mask(LPCODE, SCALE_POS, SCALE_MASK)
@triton.jit
def scale_is_exponent(LPCODE: tl.constexpr) -> bool: return scale_code(LPCODE) == int(STYPE_E8M0)
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
def number_mbits(NCODE) -> int:
    return (2, 2, 3, 3, 2, 3, 1)[NCODE-1]
@triton.jit
def number_ebits(NCODE) -> int:
    return (5, 5, 4, 4, 3, 2, 2)[NCODE-1]
@triton.jit
def number_emax(NCODE) -> int:
    return (15, 15, 8, 7, 4, 2, 2)[NCODE-1]
@triton.jit
def number_emin(NCODE) -> int:
    return (-14, -15, -6, -7, -2, 0, 0)[NCODE-1]
@triton.jit
def number_maxfloat(NCODE) -> float:
    return (57344., 57344., 448., 240., 28., 7.5, 6.)[NCODE-1]
@triton.jit
def number_midmax(NCODE) -> float:
    return (61440., 61440., 480., 248., 28., 7.75, 7.)[NCODE-1]

# fmt: on
@triton.jit
def needs_icp(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> bool:
    if (((TCODE == Q_INDEX) or (TCODE == K_INDEX)) or (TCODE == DS_INDEX)) and icp_qk(LPCODE):
        return True
    if (((TCODE == P_INDEX) or (TCODE == V_INDEX)) or (TCODE == DO_INDEX)) and icp_pv(LPCODE):
        return True
    return False


@triton.jit
def needs_quant_or_icp(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tuple[bool, bool]:
    dq = needs_quant(LPCODE, TCODE)
    di = needs_icp(LPCODE, TCODE)
    return dq or di, dq, di


@triton.jit
def scale_dtype(LPCODE: tl.constexpr, xtype: tl.core.dtype) -> tl.core.dtype:
    NCODE = scale_code(LPCODE)
    if NCODE == STYPE_MATCH:
        return xtype
    return get_triton_dtype(STYPE_TLTYPE[NCODE - 1])


@triton.jit
def quant_dtype(LPCODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    # compute dtype, must be supported on hardware backend
    NCODE = quant_code(LPCODE, TCODE)
    return get_triton_dtype(QUANT_TLTYPE[NCODE - 1])


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
    x = as_uint(x)
    # get biased exponents to check for special case of 0
    xexp = get_exponent(x, unbiased=False)
    mask = xexp == 0
    if replace:
        xexp = y + 127
    else:
        xexp += y
    return as_float(tl.where(mask, 0, x & 0x807FFFFF | (xexp << 23)))


@triton.jit
def round(x: tl.tensor, rmode, seed=19, offset=0) -> tl.tensor:
    """Round floating point."""
    if rmode == RMODE_STOCHASTIC:
        rand_ptr = tl.randn(seed, offset)
        return libdevice.trunc(x + tl.where(x < 0, -rand_ptr), rand_ptr)
    if rmode == RMODE_EVEN:
        return libdevice.rint(x)
    if rmode == RMODE_AWAY:
        return libdevice.round(x)
    if rmode == RMODE_ZERO:
        trunc = libdevice.trunc(x)
        return tl.where(x - trunc == 0.5, trunc, libdevice.round(x))


USE_ROUND: tl.constexpr = False


@triton.jit
def quantize_float(x, scale, LPCODE: tl.constexpr, TCODE: tl.constexpr, seed=19, offset=0, rmode=None, clip=True):
    """Quantize the float values with a float or exponent scale (OCP MX biased)."""
    # If float-scaled, the shared_exponent will be EMAX of the target dtype.
    tl.static_assert(enabled(LPCODE) and needs_quant(LPCODE, TCODE))
    if rmode is None:
        rmode = roundmode(LPCODE)

    # There is the input tensor triton dtype, the target dtype, and the output triton dtype.
    # For castmode == virtual, we quantize to the target dtype, but the output dtype is the input dtype.
    # For castmode == actual, we quantize to the target dtype, and the output dtype is the target dtype if
    # it exists in triton, otherwise we upcast to the output triton dtype.
    # For castmode == compress, we pack two e2m1 values into tl.uint8
    NCODE = quant_code(LPCODE, TCODE)  # NCODE != 0 because of the assert above
    CMODE = castmode(LPCODE)
    PACKED = CMODE == CMODE_COMPRESSED and NCODE == QUANT_E2M1FNUZ
    if PACKED:
        output_dtype = tl.uint8
    elif CMODE == CMODE_VIRTUAL:
        output_dtype = x.type.element_ty
    else:
        output_dtype = quant_dtype(LPCODE, TCODE)

    # quantize

    MAXFLOAT = number_maxfloat(NCODE)
    EMAX = number_emax(NCODE)
    EMIN = number_emin(NCODE)
    MBITS = number_mbits(NCODE)

    if scale_is_exponent(LPCODE):
        shared_exp = scale
        scale = tl.exp2(scale)
    else:
        shared_exp = EMAX
    # scale to EMAX
    values = x.to(tl.float32)
    if EMAX != shared_exp:
        values = modify_exponent(as_uint(values), EMAX - shared_exp)
    else:
        values = x * scale
    values = as_uint(values)
    valexp = get_exponent(values)
    denorm = EMIN - valexp
    # clamp is not supported for int32?!
    denorm = tl.where(denorm < 0, 0, tl.where(denorm > MBITS, MBITS, denorm))
    # denorm[denorm < 0] = 0
    # denorm[denorm > MBITS] = MBITS
    if USE_ROUND:
        # scale to exponent 0 (range (-2, 2)), then up to account for mbits and denorm
        values = modify_exponent(values, (MBITS - denorm) - EMAX)
        values = round(values, rmode, seed, offset)
        # descale the mbits and scale up to EMAX for clamp
        values = modify_exponent(values, EMAX - (MBITS - denorm))
    else:
        # bit method
        shift = 23 - MBITS + denorm
        roundbit = 1 << (shift - 1)
        mask = (roundbit << 1) - 1
        values = as_uint(values)
        if rmode == RMODE_AWAY:  # ties away from zero
            values += roundbit
        elif rmode == RMODE_STOCHASTIC:
            values += tl.randint(seed, values) & mask
        else:
            tie = ((roundbit - 1) & values) == 0
            if rmode == RMODE_ZERO:  # ties towards zero, not trunc
                values += tl.where(tie, 0, roundbit)
            elif rmode == RMODE_EVEN:  # ties to nearest even
                values += tl.where(tie & (values & (roundbit << 1) == 0), 0, roundbit)
        values = values >> shift << shift
        values = as_float(values)
    if clip:  # always clip unless OPTION3 scale factor selection
        values = tl.clamp(values, min=-MAXFLOAT, max=MAXFLOAT)

    # values remain target dtype values, will need to be unscaled for virtual cast
    if virtual(LPCODE):
        values /= scale
    if PACKED:
        tl.device_assert(False, "Packing not implemented yet")
        # values = as_uint(values.to(tl.float8e5))
        # values = ((values >> 28) & 0x4) | (max(126, ((x >> 23) & 0x7f) - 126) << 1) | (x >> 23) & 1
        # values = ((values >> 4) & 0x7) | ((values >> 2) & 3)
    else:
        values = tl.cast(values, output_dtype)
    return values


@triton.jit
def get_shared_exponent(values, LPCODE: tl.constexpr, TCODE: tl.constexpr):
    """Find the shared exponent for a block of values."""
    # shared_exp is unbiased exponent (maxexp), stored as int32
    absmax = tl.max(tl.abs(values))
    shared_exp = get_exponent(absmax)  # will be unbiased tl.int32
    # set the exponent of absmax to emax comparing with midmax, maxfloat, etc.
    NCODE = quant_code(LPCODE, TCODE)
    EMAX = number_emax(NCODE)
    MAXFLOAT = number_maxfloat(NCODE)
    absmax = modify_exponent(absmax, EMAX, replace=True)
    increment = tl.zeros_like(shared_exp)
    smode = scalemode(LPCODE)
    if smode == SMODE_CEIL:
        increment[as_uint(absmax) & 0x007FFFFF != 0] = 1
        # increment = as_uint(tl.where(as_uint(absmax) & 0x007FFFFF != 0, 1, 0))
    elif smode == SMODE_MIDMAX:
        increment[absmax >= number_midmax(NCODE)] = 1
        # increment = as_uint(tl.where(absmax >= number_midmax(NCODE), 1, 0))
    if smode == SMODE_TOPBINADE:
        increment[absmax > MAXFLOAT] = 1
        # increment = as_uint(tl.where(absmax > MAXFLOAT, 1, 0))
    if smode == SMODE_OPTION3:
        rounded = quantize_float(absmax, 1.0, 0, LPCODE, TCODE, 0, 0, RMODE_EVEN, clip=False)
        increment[get_exponent(rounded) != EMAX] = 1
        # increment = as_uint(tl.where(get_exponent(as_uint(rounded)) != EMAX, 1, 0))
    # returns the unbiased max (or max+1) exponent
    shared_exp = shared_exp + increment
    return shared_exp


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


@triton.jit
def get_descale(scale_or_exponent, LPCODE: tl.constexpr, TCODE: tl.constexpr):
    """Convert the scale into a descale float."""
    # An exponent scale is stored as a biased uint8 that is offset by EMAX,
    # i.e. exp_scale = shared_exp - EMAX + 127, so descale = (exp_scale - 127 + EMAX).exp2()
    if scale_is_exponent(LPCODE):
        return tl.exp2(scale_or_exponent + number_emax(quant_code(LPCODE, TCODE)) - 127)
    else:
        return 1.0 / scale_or_exponent


@triton.jit
def scale_and_quantize(x, imatrix, LPCODE: tl.constexpr, TCODE: tl.constexpr, seed=19, offset=0, trans=False):
    """Returns the quantized x and the scale used."""
    # tl.static_assert(enabled(LPCODE))
    if needs_icp(LPCODE, TCODE):
        tl.device_assert(imatrix is not None)
        x = apply_incoherence(x, imatrix, trans, icp_fp32(LPCODE))
    stype = scale_dtype(LPCODE, x.type.element_ty)
    if not needs_quant(LPCODE, TCODE):
        return x, tl.cast(1.0, stype)
    NCODE = quant_code(LPCODE, TCODE)
    MAXFLOAT = number_maxfloat(NCODE)
    IS_EXP = scale_is_exponent(LPCODE)
    absx = tl.abs(x)
    absmax = tl.max(absx)
    if IS_EXP:
        # The shared exponent is typically the exponent of the max value
        # on one higher.  That exponent is returned here unbiased tl.int32.
        # quantize_float will have to convert it to a float scale.
        scale = get_shared_exponent(absx, LPCODE, TCODE)
    else:
        # The shared_exp is EMAX because the scale is computed to ensure that.
        scale = MAXFLOAT / absmax
    out = quantize_float(x, scale, LPCODE, TCODE, seed=seed, offset=offset)
    # If the scale is an exponent, we need to convert the int32 shared exponent to a
    # biased exponent, offset by EMAX, and convert to the scale type.  The descale
    # can be acquired by calling get_descale.
    if IS_EXP:
        shared_exp = scale - number_emax(NCODE) + 127
        scale = shared_exp.to(stype)
    else:
        scale = scale.to(stype)
    return out, scale
