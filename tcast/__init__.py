#!/usr/bin/env python
# tcast/__init__.py: tcast package
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from pathlib import Path
from typing import overload

import torch

from .common import CastMode, ComputeMode, Modes, RoundMode, ScaleMode, get_enum
from .config import LPConfig
from .datatype import DataType
from .injector import mixed_precision_injector, torch_injector
from .number import Codebook, NumberLine, NumberSpec
from .scale import ScaleSpec
from .tensor import Tensor
from .torchcast import TorchCast
from .tritoncast import TritonCast
from .utils import (
    cdiv,
    get_logger,
    is_float8_available,
    is_float8_fnuz_available,
    is_triton_available,
    kurtosis,
    make_outliers,
    next_power_of_2,
    printoptions,
    set_seed,
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
float16 = DataType(name="float16")
bfloat16 = DataType(name="bfloat16")
float8_e5m2 = DataType("float8_e5m2")
float8_e4m3fn = DataType("float8_e4m3fn")
float8_e5m2fnuz = DataType("float8_e5m2fnuz")
float8_e4m3fnuz = DataType("float8_e4m3fnuz")

# OCP 8-bit unscaled datatypes
bf8 = e5m2 = float8_e5m2
fp8 = e4m3fn = float8_e4m3fn

# MI300 8-bit unscaled nanoo datatypes
bf8n = e5m2fnuz = float8_e5m2fnuz
fp4n = e4m3fnuz = float8_e4m3fnuz

# IEEE P3109 8-bit unscaled datatypes
binary8p1 = DataType("e7m0b63inuz", name="binary8p1")
binary8p2 = DataType("e6m1b32inuz", name="binary8p2")
binary8p3 = DataType("e5m2b16inuz", name="binary8p3")
binary8p4 = DataType("e4m3b8inuz", name="binary8p4")
binary8p5 = DataType("e3m4b4inuz", name="binary8p5")
binary8p6 = DataType("e2m5b2inuz", name="binary8p6")

### tensor scaled

# scale codes F=float32, f=float16, b=bfloat16, e=e8m0, n=e4m3
# uint has two (scale and zero point), int has just the scale

uint8_FF = DataType("uint8", "float32_float32", "uint8_FF")
uint8_ff = DataType("uint8", "float16_float16", "uint8_ff")
uint8_bb = DataType("uint8", "bfloat16_bfloat16", "uint8_bb")
uint8_Fi = DataType("uint8", "float32_int8", "uint8_Fi")
uint8_fi = DataType("uint8", "float16_int8", "uint8_fi")
uint8_bi = DataType("uint8", "bfloat16_int8", "uint8_bi")
uint8_ni = DataType("uint8", "e4m3_int8", "uint8_ni")

int8_F = DataType("int8", "float32", "int8_F")
int8_f = DataType("int8", "float16", "int8_f")
int8_b = DataType("int8", "bfloat16", "int8_b")
int8_n = DataType("int8", "e4m3", "int8_n")
int8_e = DataType("int8", "e8m0", "int8_e")

bf8_F = DataType("e5m2", "float32", "bf8_F")
bf8_f = DataType("e5m2", "float16", "bf8_f")
bf8_b = DataType("e5m2", "bfloat16", "bf8_b")
bf8_n = DataType("e5m2", "e4m3", "bf8_n")
bf8_e = DataType("e5m2", "e8m0", "bf8_e")
bf8n_F = DataType("e5m2fnuz", "float32", "bf8n_F")
bf8n_f = DataType("e5m2fnuz", "float16", "bf8n_f")
bf8n_b = DataType("e5m2fnuz", "bfloat16", "bf8n_b")
bf8n_n = DataType("e5m2fnuz", "e4m3", "bf8n_n")
bf8n_e = DataType("e5m2fnuz", "e8m0", "bf8n_e")

fp8_F = DataType("e4m3fn", "float32", "fp8_F")
fp8_f = DataType("e4m3fn", "float16", "fp8_f")
fp8_b = DataType("e4m3fn", "bfloat16", "fp8_b")
fp8_n = DataType("e4m3fn", "e4m3", "fp8_n")
fp8_e = DataType("e4m3fn", "e8m0", "fp8_e")
fp8n_F = DataType("e4m3fnuz", "float32", "fp8n_F")
fp8n_f = DataType("e4m3fnuz", "float16", "fp8n_f")
fp8n_b = DataType("e4m3fnuz", "bfloat16", "fp8n_b")
fp8n_n = DataType("e4m3fnuz", "e4m3", "fp8n_n")
fp8n_e = DataType("e4m3fnuz", "e8m0", "fp8n_e")

### channel scaled
uint8_FFc = DataType("uint8", "float32_float32", "uint8_FFc")
uint8_ffc = DataType("uint8", "float16_float16", "uint8_ffc")
uint8_bbc = DataType("uint8", "bfloat16_bfloat16", "uint8_bbc")
uint8_Fic = DataType("uint8", "float32_int8", "uint8_Fic")
uint8_fic = DataType("uint8", "float16_int8", "uint8_fic")
uint8_bic = DataType("uint8", "bfloat16_int8", "uint8_bic")
uint8_nic = DataType("uint8", "e4m3_int8", "uint8_nic")

uint4_FFc = DataType("uint4", "float32_float32_t0", "uint84_FFc")
uint4_ffc = DataType("uint4", "float16_float16_t0", "uint4_ffc")
uint4_bbc = DataType("uint4", "bfloat16_bfloat16_t0", "uint4_bbc")
uint4_Fic = DataType("uint4", "float32_int8_t0", "uint4_Fic")
uint4_fic = DataType("uint4", "float16_int8_t0", "uint4_fic")
uint4_bic = DataType("uint4", "bfloat16_int8_t0", "uint4_bic")
uint4_nic = DataType("uint4", "e4m3_int8_t0", "uint4_nic")

int8_Fc = DataType("int8", "float32_t0", "int8_Fc")
int8_fc = DataType("int8", "float16_t0", "int8_fc")
int8_bc = DataType("int8", "bfloat16_t0", "int8_bc")
int8_nc = DataType("int8", "e4m3_t0", "int8_nc")
int8_ec = DataType("int8", "e8m0_t0", "int8_ec")

int4_Fc = DataType("int4", "float32_t0", "int4_Fc")
int4_fc = DataType("int4", "float16_t0", "int4_fc")
int4_bc = DataType("int4", "bfloat16_t0", "int4_bc")
int4_nc = DataType("int4", "e4m3_t0", "int4_nc")
int4_ec = DataType("int4", "e8m0_t0", "int4_ec")

bf8_F = DataType("e5m2", "float32_t0", "bf8_F")
bf8_f = DataType("e5m2", "float16_t0", "bf8_f")
bf8_b = DataType("e5m2", "bfloat16_t0", "bf8_b")
bf8_n = DataType("e5m2", "e4m3_t0", "bf8_n")
bf8_e = DataType("e5m2", "e8m0_t0", "bf8_e")
bf8n_F = DataType("e5m2fnuz", "float32_t0", "bf8n_F")
bf8n_f = DataType("e5m2fnuz", "float16_t0", "bf8n_f")
bf8n_b = DataType("e5m2fnuz", "bfloat16_t0", "bf8n_b")
bf8n_n = DataType("e5m2fnuz", "e4m3_t0", "bf8n_n")
bf8n_e = DataType("e5m2fnuz", "e8m0_t0", "bf8n_e")

fp8_Fc = DataType("e4m3fn", "float32_t0", "fp8_Fc")
fp8_fc = DataType("e4m3fn", "float16_t0", "fp8_fc")
fp8_bc = DataType("e4m3fn", "bfloat16_t0", "fp8_bc")
fp8_nc = DataType("e4m3fn", "e4m3_t0", "fp8_nc")
fp8_ec = DataType("e4m3fn", "e8m0_t0", "fp8_ec")
fp8n_Fc = DataType("e4m3fnuz", "float32_t0", "fp8n_Fc")
fp8n_fc = DataType("e4m3fnuz", "float16_t0", "fp8n_fc")
fp8n_bc = DataType("e4m3fnuz", "bfloat16_t0", "fp8n_bc")
fp8n_nc = DataType("e4m3fnuz", "e4m3_t0", "fp8n_nc")
fp8n_ec = DataType("e4m3fnuz", "e8m0_t0", "fp8n_ec")

### tile scaled

# FP8 with square tiles

fp8nF32s = DataType("e4m3fnuz", "float32_t32_t32", "fp8nF32s")
fp8nf32s = DataType("e4m3fnuz", "float16_t32_t32", "fp8nf32s")
bf8nbt32s = DataType("e5m2fnuz", "bfloat16_t16_t16", "bf8nbt32s")

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
bfp16t16 = DataType("int8", "e8m0_t16", "bfp16t16")

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


@overload
def configuration(method: int) -> LPConfig: ...


@overload
def configuration(method: Path) -> LPConfig: ...


@overload
def configuration(method: str) -> LPConfig: ...


@overload
def configuration(**method) -> LPConfig: ...


def configuration(method) -> LPConfig:
    """Create an LP configuration with a code, json path, shortcut, or params."""
    if isinstance(method, int):
        return LPConfig(code=method)
    if isinstance(method, Path):
        return LPConfig(json_path=method)
    if isinstance(method, str):
        return LPConfig(shortcut=method)
    return LPConfig(**method)


def randomize_imatrix(cls, imatrix: torch.Tensor) -> torch.Tensor:
    """Randomize a Walsh-Hadamard matrix while preserving orthogonality."""
    return LPConfig.randomize_imatrix(imatrix)


def get_imatrix(size: int, dtype: torch.dtype = torch.float32, walsh: bool = True, randomize: bool = True) -> torch.Tensor:
    """Create an identity matrix with optionally randomized Walsh-Hadamard rows."""
    return LPConfig.get_imatrix(size, dtype, walsh, randomize)


logger = get_logger("tcast")


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
    tensor = Tensor(
        tensor, DataType(str(dtype)) if isinstance(dtype, torch.dtype) else dtype, transpose_scale=transpose_scale, precast=True
    )
    out_dtype = dtype if isinstance(dtype, torch.dtype) else tensor.original_dtype
    tri_supports, tri_requested = TritonCast.supports(tensor), Modes.compute == ComputeMode.TRITON
    tor_supports, tor_requested = TorchCast.supports(tensor), Modes.compute == ComputeMode.TORCH
    assert tri_supports or tor_supports, f"tri_supports {tri_supports} tor_supports {tor_supports}"
    if not (tri_supports or tor_supports):
        raise NotImplementedError("tcast.cast: datatype conversion not yet implemented in Triton or Torch")
    tri_success = tor_success = None
    if tri_supports and tri_requested:
        tri_success = TritonCast.cast(tensor)
    logger.info(f"tcast.cast: TritonCast requested: {tri_requested} supports: {tri_supports} success: {tri_success}")
    if tor_supports and tor_requested and not tri_success:
        tor_success = TorchCast.cast(tensor)
    logger.info(f"tcast.cast: TorchCast requested: {tor_requested} supports: {tor_supports} success: {tor_success}")
    if tri_success == False:  # noqa: E712
        raise AssertionError("tcast.cast: datatype conversion FAILED in Triton")
    if not tor_success:
        raise AssertionError("tcast.cast: datatype conversion FAILED in Torch")
    tensor.postcast()
    if Modes.cast == CastMode.VIRTUAL:
        torch_tensor = tensor.output.to(out_dtype)
        del tensor
        tensor = torch_tensor
    Modes.restore_modes()
    return tensor
