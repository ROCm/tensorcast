"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/__init__.py: package

from collections.abc import Callable
from pathlib import Path

import torch

from .cast import Cast, CastMode, ComputeMode, RoundMode, ScaleMode
from .datatype import DataType
from .extension import Extension
from .number import NumberSpec
from .scale import ScaleData, ScaledTensor, ScaleSpec
from .utils import (
    check_literal,
    is_float8_available,
    is_float8_fnuz_available,
    is_gpu_available,
    is_installed,
    is_power_of_2,
    next_power_of_2,
    printoptions,
)

__version__ = Path(__file__).with_name("version.txt").open().read().strip()


def initialize(
    roundmode: RoundMode = None,
    scalemode: ScaleMode = None,
    compmode: ComputeMode = None,
    ext_create: bool = False,
    ext_name: str = None,
    ext_path: Path = None,
    ext_exec: bool = False,
    ext_cpu_only: bool = False,
    ext_verbose: bool = False,
):
    """For overriding default modes and/or customizing torch cpp_extension."""
    if roundmode is not None:
        check_literal(roundmode, RoundMode)
        Cast.roundmode = roundmode
    if scalemode is not None:
        check_literal(scalemode, ScaleMode)
        Cast.scalemode = scalemode
    if compmode is not None or ext_create:
        if compmode:
            check_literal(compmode, ComputeMode)
            Cast.compmode = compmode
        if ext_create or (compmode is not None and compmode != "torch"):
            if Cast.extension is not None:
                if ext_create:
                    raise RuntimeError("tcast extension has already been created.")
            else:
                Cast.extension = Extension(ext_name, ext_path, ext_exec, ext_cpu_only, ext_verbose)


def number(code: str) -> NumberSpec:
    """Create a number spec from a string code."""
    return NumberSpec(code)


def scale(code: str) -> ScaleSpec:
    """Create a scale spec from a string code."""
    return ScaleSpec(code)


def datatype(nspec: str | NumberSpec, sspec: str | ScaleSpec = None, name: str = None) -> DataType:
    """Create an implicitly scaled or unscaled datatype from a number spec code."""
    return DataType(nspec, sspec, name)


def cast(
    x: torch.Tensor,
    dtype: DataType,
    castmode: CastMode = "virtual",
    scaledata: ScaleData = None,
    roundmode: RoundMode = None,
    scalemode: ScaleMode = None,
    compmode: ComputeMode = None,
    better: Callable = None,
    select: int = None,
    noreshape: bool = False,
) -> ScaledTensor:
    """Virtual cast a tensor to a scaled or unscaled datatype, possibly prescaled, or just find the scales."""
    return Cast.cast(
        x, dtype, castmode, roundmode, scalemode, compmode, scaledata, better=better, select=select, noreshape=noreshape
    )


def get_scales(
    x: torch.Tensor,
    dtype: DataType,
    roundmode: RoundMode = None,
    scalemode: ScaleMode = None,
    compmode: ComputeMode = None,
    noreshape: bool = False,
) -> ScaleData:
    """Find the scales for this tensor and dtype."""
    return cast(x, dtype, "scale", None, roundmode, scalemode, compmode, noreshape=noreshape).scaledata


def vcast(
    x: torch.Tensor,
    dtype: DataType,
    scaledata: ScaleData | None = None,
    roundmode: RoundMode = None,
    scalemode: ScaleMode = None,
    compmode: ComputeMode = None,
) -> torch.Tensor:
    """Virtual cast a tensor to a scaled or unscaled datatype."""
    if scaledata is not None:
        return cast(x, dtype, "prescaled", scaledata, roundmode, scalemode, compmode).tensor
    else:
        return cast(x, dtype, "virtual", None, roundmode, scalemode, compmode).tensor


def sparse(x: torch.Tensor, stile: int, dense: int, dim: int = -1) -> torch.Tensor:
    """Virtual cast a tensor to a scaled or unscaled datatype."""
    return Cast.sparse(x, stile, dense, dim)


#####
##### Predefined datatypes accessible as tcast.float32, tcast.mxfp8e4, etc
##### NOTE: bias defaults to 2^(ebits-1) - 1 unless overridden in the eXmYbZ descriptor.
##### Exceptions are external in torch.float8_e5m2fnuz and torch.float8_e4m3fnuz.
#####

### unscaled

# torch tensor dtypes
float32 = DataType(torch.float32)
float16 = DataType(torch.float16)
bfloat16 = DataType(torch.bfloat16)
if is_float8_available():
    float8_e5m2 = DataType(torch.float8_e5m2)
    float8_e4m3fn = DataType(torch.float8_e4m3fn)
if is_float8_fnuz_available():
    float8_e5m2fnuz = DataType(torch.float8_e5m2fnuz)  # bias is 16, nonstandard, matches MI300
    float8_e4m3fnuz = DataType(torch.float8_e4m3fnuz)  # bias is 8, nonstandard, matches MI300

# 5-bit exponent
e5m2 = DataType("e5m2")
e5m2fnuz = DataType("e5m2fnuz")  # bias is 15, DOES NOT MATCH torch.float8_e5m2fnuz
e5m2b16fnuz = DataType("e5m2b16fnuz")  # bias of 16, DOES MATCH torch.float8_e5m2fnuz and MI300
binary8p3 = DataType("e5m2b16fnuz")  # IEEE P3109 bias 16 matches torch.float8_e5m2fnuz and MI300
# 4-bit exponent
e4m3fnuz = DataType("e4m3fnuz")  # bias is 7, DOES NOT MATCH torch.float8_e5m2fnuz
e4m3fn = DataType("e4m3fn")
e4m3b8fnuz = DataType("e4m3b8fnuz")  # bias is 8, DOES MATCH torch.float8_e4m3fnuz and MI300
binary8p4 = DataType("e4m3b8fnuz")  # IEEE P3109 bias 8 matches torch.float8_e4m3fnuz and MI300
# 3-bit exponent
binary8p5 = DataType("e3m4b4fnuz")  # IEEE P3109 bias 4 consistent with other P3109 float8 types
e3m3fnuz = DataType("e3m3fnuz")  # bias 3
e3m2fnuz = DataType("e3m2fnuz")  # bias 3
# 2-bit exponent
e2m3fnuz = DataType("e2m3fnuz")  # bias 1
e2m1fnuz = DataType("e2m1fnuz")  # bias 1

### tensor scaled

# scale codes f=float16 (for uintK and int16 only), b=float32, f=e5m3, e=e8m0, i=int8
# uint has two (scale and zero point)

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


### tile scaled

# MX, tile size 32, exponent scale (BFP tile size 8)

mxfp8e5 = DataType("e5m2", "e8m0_t32", "mxfp8e5")
mxfp8e4 = DataType("e4m3fn", "e8m0_t32", "mxfp8e4")
mxfp6e3 = DataType("e3m2fnuz", "e8m0_t32", "mxfp6e3")
mxfp6e2 = DataType("e2m3fnuz", "e8m0_t32", "mxfp6e2")
mxfp4e2 = DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")
mxfp5e2 = DataType("e2m2fnuz", "e8m0_t32", "mxfp5e2")
mxint8 = DataType("int8", "e8m0_t32", "mxint8")
mxint4 = DataType("int4", "e8m0_t32", "mxint4")
bfp16 = DataType("int8", "e8m0_t8", "bfp16")

# Float-scaled integer, tile size 32

uint4_ff32 = DataType("uint4", "float16_float16_t32", "uint4_ff32")
uint4_fi32 = DataType("uint4", "float16_int8_t32", "uint4_fi32")
uint4_bb32 = DataType("uint4", "bfloat16_bfloat16_t32", "uint4_bb32")
uint4_bi32 = DataType("uint4", "bfloat16_int8_t32", "uint4_bi32")
uint4_f12i432 = DataType("uint4", "e5m6_int4_t32", "uint4_f12i432")
int4_b32 = DataType("int4", "bfloat16_t32", "int4_b32")
int4_f32 = DataType("int4", "e5m3_t32", "int4_f32")

# 1-bit exponent (integer), bias overriden to 1, equivalent to int8 or int4
e1m6b1_e32 = DataType("e1m6b1fnuz", "e8m0_t32", "e1m6_e32")
e1m2b1_e32 = DataType("e1m2b1fnuz", "e8m0_t32", "e1m2_e32")
e1m2b1_f32 = DataType("e1m2b1fnuz", "e5m3_t32", "e1m2_e32")

# e5m3 scaled fp4, fp6,  and fp8
e2m1_f32 = DataType("e2m1fnuz", "e5m3_t32", "e2m1_f32")
e3m2_f32 = DataType("e3m2fnuz", "e5m3_t32", "e3m2_f32")
e2m3_f32 = DataType("e2m3fnuz", "e5m3_t32", "e2m3_f32")
e4m3_f32 = DataType("e4m3fn", "e5m3_t32", "e4m3_f32")

# Original MX, tile size 16, exponent scale, prime bit shared between adjacent values
mx9 = DataType("int8", "e8m0_t16s2o1", "mx9")
mx6 = DataType("int5", "e8m0_t16s2o1", "mx6")
mx4 = DataType("int3", "e8m0_t16s2o1", "mx4")

# Subtile offsets, tile size 32, exponent scale
e2m1_e32s8o1 = DataType("e2m1fnuz", "e8m0_t32s8o1", "e2m1_e32s8o1")
e2m1_e32s4o1 = DataType("e2m1fnuz", "e8m0_t32s4o1", "e2m1_e32s4o1")
e2m1_e32s8o2 = DataType("e2m1fnuz", "e8m0_t32s8o2", "e2m1_e32s8o2")
e2m1_e32s4o2 = DataType("e2m1fnuz", "e8m0_t32s4o2", "e2m1_e32s4o2")

###
### implicit codebook datatypes for tiles and subtiles; offsets (if any) are built in to lookups
###

icb44fi4_e32 = DataType("icb44fi_e4m3fn", "e8m0_t32", "icb44fi4_e32")
icb44fi4_f32 = DataType("icb44fi_e4m3fn", "e5m0_t32", "icb44fi4_f32")
icb44fi4_e32s16 = DataType("icb44fi_e4m3fn", "e8m0_t32s16", "icb44fi4_e32s16")
icb44fi4_f32s16 = DataType("icb44fi_e4m3fn", "e5m3_t32s16", "icb44fi4_f32s16")
icb44fi4_e32s8 = DataType("icb44fi_e4m3fn", "e8m0_t32s8", "icb44fi4_e32s8")
icb44fi4_f32s8 = DataType("icb44fi_e4m3fn", "e5m3_t32s8", "icb44fi4_f32s8")
icb44fi4_e32s4 = DataType("icb44fi_e4m3fn", "e8m0_t32s4", "icb44fi4_e32s4")
icb44fi4_f32s4 = DataType("icb44fi_e4m3fn", "e5m3_t32s4", "icb44fi4_f32s4")

icb44fi3_e32 = DataType("icb44fi_e3m2fnuz", "e8m0_t32", "icb44fi3_e32")
icb44fi3_f32 = DataType("icb44fi_e3m2fnuz", "e5m0_t32", "icb44fi3_f32")
icb44fi3_e32s16 = DataType("icb44fi_e3m2fnuz", "e8m0_t32s16", "icb44fi3_e32s16")
icb44fi3_f32s16 = DataType("icb44fi_e3m2fnuz", "e5m3_t32s16", "icb44fi3_f32s16")
icb44fi3_e32s8 = DataType("icb44fi_e3m2fnuz", "e8m0_t32s8", "icb44fi3_e32s8")
icb44fi3_f32s8 = DataType("icb44fi_e3m2fnuz", "e5m3_t32s8", "icb44fi3_f32s8")
icb44fi3_e32s4 = DataType("icb44fi_e3m2fnuz", "e8m0_t32s4", "icb44fi3_e32s4")
icb44fi3_f32s4 = DataType("icb44fi_e3m2fnuz", "e5m3_t32s4", "icb44fi3_f32s4")

icb44fi2_e32 = DataType("icb44fi_e2m3fnuz", "e8m0_t32", "icb44fi2_e32")
icb44fi2_f32 = DataType("icb44fi_e2m3fnuz", "e5m0_t32", "icb44fi2_f32")
icb44fi2_e32s16 = DataType("icb44fi_e2m3fnuz", "e8m0_t32s16", "icb44fi2_e32s16")
icb44fi2_f32s16 = DataType("icb44fi_e2m3fnuz", "e5m3_t32s16", "icb44fi2_f32s16")
icb44fi2_e32s8 = DataType("icb44fi_e2m3fnuz", "e8m0_t32s8", "icb44fi2_e32s8")
icb44fi2_f32s8 = DataType("icb44fi_e2m3fnuz", "e5m3_t32s8", "icb44fi2_f32s8")
icb44fi2_e32s4 = DataType("icb44fi_e2m3fnuz", "e8m0_t32s4", "icb44fi2_e32s4")
icb44fi2_f32s4 = DataType("icb44fi_e2m3fnuz", "e5m3_t32s4", "icb44fi2_f32s4")
