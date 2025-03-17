#!/usr/bin/env python
# tests/utils.py: unit test utilities
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import importlib
import struct

import torch

import tcast

SHAPES_2D = [(1024, 1024)]
SHAPES_3D = [(8, 128, 64)]
SHAPES_4D = [(8, 128, 64, 64)]
SHAPES_23D = SHAPES_2D + SHAPES_3D
SHAPES_234D = SHAPES_2D + SHAPES_3D + SHAPES_4D

TORCH_DTYPES_16 = [torch.float16, torch.bfloat16]
TORCH_DTYPES_32 = [torch.float32]
TORCH_DTYPES_8 = [torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
TORCH_DTYPES_32_16 = TORCH_DTYPES_32 + TORCH_DTYPES_16
TORCH_DTYPES_16_8 = TORCH_DTYPES_16 + TORCH_DTYPES_8
TORCH_DTYPES_32_16_8 = TORCH_DTYPES_32_16 + TORCH_DTYPES_8

V2_AVAILABLE = importlib.util.find_spec("tcastv2") is not None
MX_AVAILABLE = importlib.util.find_spec("mx") is not None
SQT_AVAILABLE = importlib.util.find_spec("sqt") is not None
logger = tcast.get_logger("tcast_unittest")


def dependency_assert(lib: str = "v2"):
    """Assert with message for v2/mx/sqt tests."""
    if lib == "v2" and not V2_AVAILABLE:
        logger.error(
            "tcastv2 package not found:\n"
            "> mkdir tmp; cd tmp\n"
            "> git clone https://github.com/ROCm/tensorcast.git\n"
            "> git switch v2\n"
            "> cp -r tensorcast/tcast $TCAST/tcastv2\n"
        )
        raise AssertionError("tcastv2 package not found")
    if lib == "mx" and not MX_AVAILABLE:
        logger.error(
            "mx package not found:\n"
            "> mkdir tmp; cd tmp\n"
            "> git clone github.com/microsoft/microxcaling\n"
            "> cp -r microxcaling/mx $TCAST/mx\n"
        )
        raise AssertionError("mx package not found")
    if lib == "sqt" and not SQT_AVAILABLE:
        logger.error(
            "sqt package not found; need to get access from @alirezak:\n"
            "> mkdir tmp; cd tmp\n"
            "> git clone https://github.com/Xilinx/sqt.git\n"
            "> cp -r sqt/sqt $TCAST/sqt\n"
        )
        raise AssertionError("sqt package not found")


def convert_triton_dtype_to_v2(dtype: tcast.DataType):
    """Convert a triton DataType to a v2 DataType."""
    dependency_assert("v2")
    import tcastv2

    if dtype.is_codebook:
        raise NotImplementedError("Codebook not supported in v2/triton conversion")
    if dtype.nspec.torch_dtype:
        ncode = dtype.nspec.torch_dtype
    else:
        n = dtype.nspec
        ncode = f"e{n.ebits}m{n.mbits}b{n.bias}"
        infnan = n.infnan.name.lower()
        if infnan != "ieee":
            ncode += infnan
    if dtype.is_unscaled:
        scode = None
    else:
        if dtype.sspec.is_2d:
            raise NotImplementedError("2D sspec not supported in v2/triton conversion")
        s = dtype.sspec
        if s.scale.is_exponent:
            scode = f"e{s.scale.ebits}m{s.scale.mbits}"
        else:
            scode = f"e{s.scale.ebits}m{s.scale.mbits}b{s.scale.bias}{s.scale.infnan}"
        if s.zero:
            if s.zero.is_int:
                scode += f"_int{s.zero.bits}"
            elif s.zero.is_float:
                scode += f"_e{s.zero.ebits}m{s.zero.mbits}b{s.zero.bias}{s.zero.infnan}"
            else:
                raise ValueError(f"Unknown zero type {s.zero}")
        if not s.is_tensor:
            scode += f"_t{0 if s.is_channel else s.tile1}"
    return tcastv2.DataType(nspec=ncode, sspec=scode)


def compare_2(tensor1, tensor2):
    torch.testing.allclose(tensor1, tensor2)


def float_to_bits(value_in_float):
    s = struct.pack("@f", value_in_float)
    return struct.unpack("@I", s)[0]


def bits_to_float(value_in_uint):
    s = struct.pack("@I", value_in_uint)
    return struct.unpack("@f", s)[0]


def get_leading_zeros(uival):
    andv = 0x80000000
    rval = 0
    for _ in range(31):
        if uival & andv == 0:
            rval += 1
            andv = andv >> 1
        else:
            return rval
    return rval


def round_func(sign, mantisa, rmode, mbits, mantisaNotScaled, scale):
    masks = [0x0, 0x400000, 0x600000, 0x700000, 0x780000, 0x7C0000, 0x7E0000, 0x7F0000, 0x7F8000, 0x7FC000, 0x7FE000]
    adds = [0x0, 0x400000, 0x200000, 0x100000, 0x80000, 0x40000, 0x20000, 0x10000, 0x8000, 0x4000, 0x2000, 0x0]
    opmasks = [0x00000000, 0xFFFFFFFF]
    mask = masks[mbits]
    add = adds[mbits]
    if rmode == "zero":
        return mantisa & mask
    elif rmode == "even":
        if (mantisa & (~mask) == adds[mbits + 1]) and not (mantisa & add):
            if (
                (pow(2, (23 - (mbits - scale))) - 1) & mantisaNotScaled != 0
            ):  # consider all relevant mantissa bits before scaling (including implicit one scaling) as in SQT
                add = adds[mbits + 1]
                mantisa += add & opmasks[((mantisa & mask) != mask)]
                return mantisa & mask
            else:
                return mantisa & mask
        else:
            add = adds[mbits + 1]
            mantisa += add & opmasks[((mantisa & mask) != mask)]
            return mantisa & mask
    elif rmode == "away":
        add = adds[mbits + 1]
        mantisa += add & opmasks[((mantisa & mask) != mask)]
        return mantisa & mask
    else:
        raise NotImplementedError(f"RoundMode {rmode.name} is not currently implemented in round_func")


def float_to_bfp(fval, max_exp, rmode, mbits):
    bits = float_to_bits(fval)  # bits in form of uint32
    sign = bits & 0x80000000  # sign bit
    exp = (bits & 0x7F800000) >> 23  # exponent
    if (exp == 0) and (max_exp != 0):
        correctedExponent = 1
    else:
        correctedExponent = exp
    scale = max(0, max_exp - correctedExponent)  # scale required to go to maxexp
    mant = bits & 0x7FFFFF  # mantisa bits
    if exp == 0:  # subnormal
        mantScaled = mant >> scale  # scale to max exponent
        mant = round_func(sign, mantScaled, rmode, mbits, mant, scale)  # rounding
        if mant == 0:
            qbits = sign | mant
        else:
            if max_exp == 0:
                qbits = sign | mant
            else:
                lziro = get_leading_zeros(mant << 9)  # 9: 1bit sign and 8 bits exponent are not considered
                if lziro == 0:  # this is only possible when maxExp equals 1 and we scaled with 0
                    qbits = sign | mant
                elif lziro > max_exp:
                    qbits = sign
                elif lziro == max_exp:
                    qbits = sign | mant
                else:
                    mant = (mant << (lziro + 1)) & 0x7FFFFF  # scale back so implicit 1 is bit23 and remove it
                    qbits = sign | ((max_exp - lziro) << 23) | mant
    else:
        # insert implicit 1. Since we are quantizing, and bit0 is being eliminated, we don't care about
        # bit0 - might be important for rounding. We will see in the tests ..
        mantScaled = (mant >> 1) | 0x400000
        mantScaled = mantScaled >> scale  # scale to max exponent
        mant = round_func(sign, mantScaled, rmode, mbits, mant, scale)  # rounding
        if mant == 0:
            qbits = sign | mant
        else:
            lziro = get_leading_zeros(mant << 9)  # 9: 1bit sign and 8 bits exponent are not considered
            if lziro >= max_exp:
                mant = (mant << max_exp) & 0x7FFFFF
                qbits = sign | mant  # make subnorm
            else:
                mant = (mant << (lziro + 1)) & 0x7FFFFF  # scale back so implicit 1 is bit23 and remove it
                qbits = sign | ((max_exp - lziro) << 23) | mant
    return bits_to_float(qbits)


def block_to_bfp(tensor, dtype, rmode):
    # input tensor is 1D and implicitly, the tensor size is our blocksize
    assert tensor.ndim == 1
    # shared exponent is largest number's exponent
    max_idx = tensor.abs().argmax()
    bits = float_to_bits(tensor[max_idx].item())
    max_exp = (bits >> 23) & 0xFF
    tq = torch.zeros_like(tensor)

    for i in range(tensor.shape[0]):
        tq[i] = float_to_bfp(tensor[i].item(), max_exp, rmode, dtype.nspec.mbits + 1)

    return tq


def tensor_to_bfp(tensor, axis, dtype, rmode):
    blocksize = dtype.sspec.tile1
    ftensor = tensor.nan_to_num(nan=0.0).float()  # d0, d1, ... d_(axis), ..., dn
    ftensor = ftensor.permute(
        *[i for i in range(axis)], *[i for i in range(axis + 1, ftensor.ndim)], axis
    )  # d0, d1, ..., dn, d_(axis)
    ftensor = ftensor.unsqueeze(ftensor.ndim)  # d0, d1, ..., dn, d_(axis), 1
    ftensor = ftensor.reshape(
        *ftensor.shape[:-2], ftensor.shape[ftensor.ndim - 2] // blocksize, blocksize
    )  # d0, d1, ..., dn, d_(axis)/blocksize, blocksize
    fshape = ftensor.shape
    ftensor = ftensor.reshape(-1, blocksize)
    for i in range(ftensor.shape[0]):
        ftensor[i] = block_to_bfp(ftensor[i], dtype, rmode)
    ftensor = ftensor.reshape(*fshape)  # d0, d1, ..., dn, d_(axis)/blocksize, blocksize
    ftensor = ftensor.reshape(*ftensor.shape[:-2], ftensor.shape[ftensor.ndim - 2] * blocksize)  # d0, d1, ..., dn, d_(axis)
    ftensor = ftensor.permute(
        *[i for i in range(axis)], ftensor.ndim - 1, *[i for i in range(axis, ftensor.ndim - 1)]
    )  # d0, d1, ..., d_(axis), ..., dn
    return ftensor
