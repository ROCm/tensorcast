"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/__init__.py: package

from pathlib import Path

import torch

from .cast import Cast, RoundMode, ScaleMode
from .datatype import DataType
from .number import NumberSpec
from .scale import ScaleSpec
from .utils import TensorCastInternalError, check_literal, is_float8_available, is_gpu_available, is_installed, printoptions

__version__ = Path(__file__).with_name("version.txt").open().read().strip()


def initialize(roundmode: RoundMode = None, scalemode: ScaleMode = None):
    """Only needed if overriding system defaults for rounding."""
    # later, will be required to initialize torch extension
    if roundmode is not None:
        Cast.roundmode = roundmode
    if scalemode is not None:
        Cast.scalemode = scalemode


def make_number(code: str) -> NumberSpec:
    """Create a number spec from a string code."""
    return NumberSpec(code)


def make_scale(code: str) -> ScaleSpec:
    """Create a scale spec from a string code."""
    return ScaleSpec(code)


def make_datatype(ncode: str, scode: str = None, name: str = None) -> DataType:
    """Create an implicitly scaled or unscaled datatype from a number spec code."""
    return DataType(ncode, scode, name)


def cast(x: torch.Tensor, dtype: DataType, roundmode: RoundMode = None, scalemode: ScaleMode = None) -> torch.Tensor:
    """Virtual cast a tensor to a scaled or unscaled datatype."""
    return Cast.cast(x, dtype, roundmode, scalemode)


#####
##### predefined datatypes accessible as tensorcast.float32, tensorcast.mxfp8e4, etc
#####

# unscaled, which can be used in conjunction with a separate scale
float32 = DataType(torch.float32)
float16 = DataType(torch.float16)
bfloat16 = DataType(torch.bfloat16)
if is_float8_available():
    e5m2 = DataType(torch.float8_e5m2)
    e5m2fnuz = DataType(torch.float8_e5m2fnuz)
    e4m3fn = DataType(torch.float8_e4m3fn)
    e4m3fnuz = DataType(torch.float8_e4m3fnuz)
e3m2fnuz = DataType("e3m2fnuz")
e2m3fnuz = DataType("e2m3fnuz")
e2m1fnuz = DataType("e2m1fnuz")

# tensor scaled
uint16_ff = DataType("uint16", "float16_float16", "uint16_ff")
uint16_bb = DataType("uint16", "bfloat16_bfloat16", "uint16_bb")
int16_f = DataType("int16", "float16", "int16_f")
int16_b = DataType("int16", "bfloat16", "int16_b")
int16_e = DataType("int16", "e8m0", "int16_e")
uint8_ff = DataType("uint16", "float16_float16", "uint16_ff")
uint8_bb = DataType("uint16", "bfloat16_bfloat16", "uint16_bb")
uint8_fi = DataType("uint16", "float16_int8", "uint16_fi")
uint8_bi = DataType("uint16", "bfloat16_int8", "uint16_bi")
int8_f = DataType("int8", "float16", "int8_f")
int8_b = DataType("int8", "bfloat16", "int8_b")
int8_e = DataType("int8", "e8m0", "int8_e")
e5m2_e = DataType("e5m2", "e8m0", "e5m2_e")
e5m2z_e = DataType("e5m2fnuz", "e8m0", "e5m2z_e")
e4m3_e = DataType("e4m3fn", "e8m0", "e4m3_e")
e4m3z_e = DataType("e4m3fnuz", "e8m0", "e4m3z_e")
e3m2_e = DataType("e3m2fnuz", "e8m0", "e3m2_e")
e2m3_e = DataType("e2m3fnuz", "e8m0", "e2m3_e")

# MX
mxfp8e5 = DataType("e5m2", "e8m0_t32", "mxfp8e5")
mxfp8e4 = DataType("e4m3fn", "e8m0_t32", "mxfp8e4")
mxfp6e3 = DataType("e3m2fnuz", "e8m0_t32", "mxfp6e3")
mxfp6e2 = DataType("e2m3fnuz", "e8m0_t32", "mxfp6e2")
mxfp4e2 = DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")
mxint8 = DataType("int8", "e8m0_t32", "mxint8")
mxint4 = DataType("int4", "e8m0_t32", "mxint4")
bfp16 = DataType("int8", "e8m0_t8", "bfp16")

# other tile scaled
uint4_ff32 = DataType("uint4", "float16_float16_t32", "uint4_ff32")
uint4_bb32 = DataType("uint4", "bfloat16_bfloat16_t32", "uint4_bb32")
uint4_fi32 = DataType("uint4", "float16_int8_t32", "uint4_fi32")
uint4_bi32 = DataType("uint4", "bfloat16_int8_t32", "uint4_bi32")
int4_f32 = DataType("int4", "float16_t32", "int4_f32")
int4_b32 = DataType("int4", "bfloat16_t32", "int4_b32")
