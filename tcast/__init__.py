#!/usr/bin/env python
# tcast/__init__.py: tcast package
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

from .common import CastMode, ComputeMode, Modes, RoundMode, ScaleMode, get_enum
from .datatype import DataType
from .injector import TorchInjector, MixedPrecisionInjector
from .number import Codebook, NumberLine, NumberSpec
from .scale import ScaleSpec
from .tensor import Tensor
from .torchcast import TorchCast
from .tritoncast import TritonCast
from .utils import (
    cdiv,
    hadamard_transform,
    is_float8_available,
    is_float8_fnuz_available,
    is_triton_available,
    next_power_of_2,
    printoptions,
)

__version__ = "0.3.0"


def initialize(
    roundmode: RoundMode = None, scalemode: ScaleMode = None, computemode: ComputeMode = None, castmode: CastMode = None
) -> dict:
    """For overriding default modes in Cast.  Optional."""
    return Modes.set_modes(roundmode, scalemode, computemode, castmode)


#####
##### Predefined datatypes accessible as tcast.float32, tcast.mxfp8e4, etc
##### NOTE: bias defaults to 2^(ebits-1) - 1 unless overridden by Z in the eXmYbZ descriptor.
##### Exceptions are external in torch.float8_e5m2fnuz and torch.float8_e4m3fnuz.
#####

### unscaled

# torch tensor dtypes
float32 = DataType(name="float32")
float16 = DataType(tname="float16")
bfloat16 = DataType(name="bfloat16")
float8_e5m2 = DataType(name="float8_e5m2")
float8_e4m3fn = DataType(name="float8_e4m3fn")
float8_e5m2fnuz = DataType(name="float8_e5m2fnuz")
float8_e4m3fnuz = DataType(name="float8_e4m3fnuz")

# OCP 8-bit unscaled datatypes
bf8 = e5m2 = float8_e5m2
fp8 = e4m3fn = float8_e4m3fn

# MI300 8-bit unscaled datatypes
e5m2fnuz = float8_e5m2fnuz
e4m3fnuz = float8_e4m3fnuz

# IEEE P3109 8-bit unscaled datatypes
binary8p1 = DataType("e7m0b63inuz", name="binary8p1")
binary8p2 = DataType("e6m1b32inuz", name="binary8p2")
binary8p3 = DataType("e5m2b16inuz", name="binary8p3")
binary8p4 = DataType("e4m3b8inuz", name="binary8p4")
binary8p5 = DataType("e3m4b4inuz", name="binary8p5")
binary8p6 = DataType("e2m5b2inuz", name="binary8p6")

### tensor scaled

# scale codes f=float16 (for uintK and int16 only), b=float32, f=e5m3, e=e8m0, i=int8
# uint has two (scale and zero point)

uint16_ff = DataType("uint16", "float16_float16", "uint16_ff")
uint16_bb = DataType("uint16", "bfloat16_bfloat16", "uint16_bb")
uint16_fi = DataType("uint16", "float16_int16", "uint16_fi")
uint16_bi = DataType("uint16", "bfloat16_int16", "uint16_bi")

uint8_ff = DataType("uint16", "float16_float16", "uint8_ff")
uint8_bb = DataType("uint16", "bfloat16_bfloat16", "uint8_bb")
uint8_fi = DataType("uint16", "float16_int8", "uint8_fi")
uint8_bi = DataType("uint8", "bfloat16_int8", "uint8_bi")

int16_f = DataType("int8", "float16", "int16_f")
int16_b = DataType("int8", "bfloat16", "int16_b")
int16_e = DataType("int8", "e8m0", "int16_e")

int8_f = DataType("int8", "float16", "int8_f")
int8_b = DataType("int8", "bfloat16", "int8_b")
int8_e = DataType("int8", "e8m0", "int8_e")

e5m2_e = DataType("e5m2", "e8m0", "e5m2_e")
e4m3_e = DataType("e4m3fn", "e8m0", "e4m3_e")

### channel scaled

# t0 indicates channel, default dim is -1, explicit dim (such as 0) encoded with t0d0

uint16_ff = DataType("uint16", "float16_float16_t0", "uint16_ffc")
uint16_bb = DataType("uint16", "bfloat16_bfloat16_t0", "uint16_bbc")
uint16_fi = DataType("uint16", "float16_int16_t0", "uint16_fic")
uint16_bi = DataType("uint16", "bfloat16_int16_t0", "uint16_bic")

uint8_ffc = DataType("uint8", "float16_float16_t0", "uint8_ffc")
uint8_bbc = DataType("uint8", "bfloat16_bfloat16_t0", "uint8_bbc")
uint8_fic = DataType("uint8", "float16_int8_t0", "uint8_fic")
uint8_bic = DataType("uint8", "bfloat16_int8_t0", "uint8_bic")

int16_fc = DataType("int8", "float16_t0", "int16_fc")
int16_bc = DataType("int8", "bfloat16_t0", "int16_bc")
int16_ec = DataType("int8", "e8m0_t0", "int16_ec")

int8_fc = DataType("int8", "float16_t0", "int8_fc")
int8_bc = DataType("int8", "bfloat16_t0", "int8_bc")
int8_ec = DataType("int8", "e8m0_t0", "int8_ec")

e5m2_ec = DataType("e5m2", "e8m0_t0", "e5m2_ec")
e4m3_ec = DataType("e4m3fn", "e8m0_t0", "e4m3_ec")

### tile scaled

# OCP MXFP and MXINT, tile size 32

mxbf8 = mxfp8e5 = DataType("e5m2", "e8m0_t32", "mxfp8e5")
mxfp8 = mxfp8e4 = DataType("e4m3fn", "e8m0_t32", "mxfp8e4")
mxbf6 = mxfp6e3 = DataType("e3m2fnuz", "e8m0_t32", "mxfp6e3")
mxfp6 = mxfp6e2 = DataType("e2m3fnuz", "e8m0_t32", "mxfp6e2")
mxfp4 = mxfp4e2 = DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")
mxint8 = DataType("int8", "e8m0_t32", "mxint8")
mxint4 = DataType("int4", "e8m0_t32", "mxint4")

# OCP MXFP and MXINT, tile size 16

mxbf8t16 = DataType("e5m2", "e8m0_t16", "mxbf8t16")
mxfp8t16 = DataType("e4m3fn", "e8m0_t16", "mxfp8t16")
mxbf6t16 = DataType("e3m2fnuz", "e8m0_t16", "mxbf6t16")
mxfp6t16 = DataType("e2m3fnuz", "e8m0_t16", "mxfp6t16")
mxfp4t16 = DataType("e2m1fnuz", "e8m0_t16", "mxfp4t16")
mxint8t16 = DataType("int8", "e8m0_t16", "mxint8t16")
mxint4t16 = DataType("int4", "e8m0_t16", "mxint4t16")

# NVF4, tile size 16

nvf4 = DataType("e2m1fnuz", "e4m3_t16", "nvf4")


# MSFP (old school Microsoft), tile size 16, exponent scale, subtile size 2, implemented as codebooks
# BFP16 is essentially OCP MXINT8, but with a tile size of 8

mx9 = DataType("cb81ie_int9", "e8m0_t16s2", "mx9")
mx6 = DataType("cb51ie_int7", "e8m0_t16s2", "mx6")
mx4 = DataType("cb31ie_int5", "e8m0_t16s2", "mx4")
bfp16 = DataType("int8", "e8m0_t8", "bfp16")

### square tile scaled

mxfp4s = DataType("e2m1fnuz", "e8m0_t32_t32", "mxfp4s")
mxbf6s = DataType("e3m2fnuz", "e8m0_t32_t32", "mxbf6s")
mxfp6s = DataType("e2m3fnuz", "e8m0_t32_t32", "mxfp6s")
mxfp8s = DataType("e4m3fn", "e8m0_t32_t32", "mxfp8s")
mxfp4s16 = DataType("e2m1fnuz", "e8m0_t16_t16", "mxfp4s16")
mxbf6s16 = DataType("e3m2fnuz", "e8m0_t16_t16", "mxbf6s16")
mxfp6s16 = DataType("e2m3fnuz", "e8m0_t16_t16", "mxfp6s16")
mxfp8s16 = DataType("e4m3fn", "e8m0_t16_t16", "mxfp8s16")

###
### example implicit codebook datatypes for tiles and subtiles
###

# mxfp4 or mxint4, 8 subtiles, 4.5 bpv
mxfi4 = DataType("cb41fi_e2m2fnuz", "e8m0_t32s4")
# mxfp4 with optional trailing mantissa bit, 8 subtiles, 4.5 bpv
mxfp4m = DataType("cb41f01_e2m2fnuz", "e8m0_t32s4", "mxfp4m")
# mxfp4 with four different starting exponents, 4 subtiles, 4.5 bpv
mxfp4e = DataType("cb42fe0123_e3m2fnuz", "e8m0_t32s8", "mxfp4e")
# mxfp4 shifted 4 times up or down the number line, 4 subtiles, 4.5 bpv
mxfp4f4 = DataType("cb42f1346_e2m3fnuz", "e8m0_t32s8", "mxfp4f4")


def number(code: str | torch.dtype) -> NumberSpec | Codebook:
    """Create a number spec from a string code."""
    return Codebook(str(code)) if str(code).lower().startswith("cb") else NumberSpec(code)


def scale(code: str) -> ScaleSpec:
    """Create a scale spec from a string code."""
    return ScaleSpec(code)


def datatype(nspec: str | NumberSpec = None, sspec: str | ScaleSpec = None, name: str = None) -> DataType:
    """Create an implicitly scaled or unscaled datatype from a number spec code."""
    return DataType(nspec, sspec, name)


def cast(
    tensor: torch.Tensor,
    dtype: DataType | torch.dtype,
    roundmode: RoundMode | str = None,
    scalemode: ScaleMode | str = None,
    computemode: ComputeMode | str = None,
    castmode: CastMode | str = None,
    transpose_scale: bool = False,
) -> torch.Tensor | Tensor:
    """
    Virtual, actual or compressed cast of torch.Tensor to dtype.

    Returns torch.Tensor if castmode is "virtual", otherwise tcast.Tensor.
    transpose_scale is used for channel and tile scaled datatypes, where the scale is applied to the channel or tile.
    In back propagation, the scale spec can be transposed without having to create a new scale spec.
    """
    Modes.set_modes(roundmode, scalemode, computemode, castmode)
    if not isinstance(tensor, torch.Tensor) or not isinstance(dtype, DataType | torch.dtype):
        raise ValueError("tcast.cast: tensor and dtype must be torch.Tensor and DataType or torch.dtype")
    tensor = Tensor(tensor, DataType(str(dtype)) if isinstance(dtype, torch.dtype) else dtype, transpose_scale=transpose_scale)
    out_dtype = dtype if isinstance(dtype, torch.dtype) else tensor.original_dtype
    if (
        Modes.compute in (ComputeMode.ANY, ComputeMode.TRITON)
        and TritonCast.cast(tensor)
        or Modes.compute in (ComputeMode.ANY, ComputeMode.TORCH)
        and TorchCast.cast(tensor)
    ):
        if Modes.castmode == CastMode.VIRTUAL:
            torch_tensor = tensor.tensor.to(out_dtype)
            del tensor
            tensor = torch_tensor
        Modes.restore_modes()
        return tensor
    raise NotImplementedError("tcast.cast: datatype conversion not yet implemented in Triton or Torch")


def upcast(
    tensor: Tensor,
    torch_dtype: torch.dtype,
    roundmode: RoundMode | str = None,
    scalemode: ScaleMode | str = None,
    computemode: ComputeMode | str = None,
) -> torch.Tensor | Tensor:
    """
    Convert a tcast.Tensor to a new datatype.

    Typically used for an upcast, the scale data in the tcast.Tensor is used.
    """
    Modes.set_modes(roundmode, scalemode, computemode, "virtual")
    if not isinstance(tensor, Tensor) or not isinstance(torch_dtype, torch.dtype):
        raise ValueError("tcast.upcast: tensor and torch_dtype must be tcast.Tensor and torch.dtype")
    if (
        Modes.compute in (ComputeMode.ANY, ComputeMode.TRITON)
        and TritonCast.upcast(tensor, torch_dtype)
        or Modes.compute in (ComputeMode.ANY, ComputeMode.TORCH)
        and TorchCast.cast(tensor, torch_dtype)
    ):
        torch_tensor = tensor.tensor
        assert torch_tensor.dtype == torch_dtype
        del tensor
        Modes.restore_modes()
        return torch_tensor
    raise NotImplementedError("tcast.cast: datatype conversion not yet implemented in Triton or Torch")
