"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/__init__.py: package

from collections.abc import Callable
from pathlib import Path

import torch

from .cast import Cast, CastMode, ComputeMode, RoundMode, ScaleMode
from .datatype import DataType
from .embedded_codebook import find_codebook
from .extension import Extension
from .lookup import LookupBuilder
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
    scaledata: ScaleData = None,
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

# MX, tile size 32, exponent scale (BFP tile size 8)

mxfp8e5 = DataType("e5m2", "e8m0_t32", "mxfp8e5")
mxfp8e4 = DataType("e4m3fn", "e8m0_t32", "mxfp8e4")
mxfp6e3 = DataType("e3m2fnuz", "e8m0_t32", "mxfp6e3")
mxfp6e2 = DataType("e2m3fnuz", "e8m0_t32", "mxfp6e2")
mxfp4e2 = DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")
mxint8 = DataType("int8", "e8m0_t32", "mxint8")
mxint4 = DataType("int4", "e8m0_t32", "mxint4")
bfp16 = DataType("int8", "e8m0_t8", "bfp16")

# Float-scaled integer, tile size 32

uint8_ff32 = DataType("uint8", "float16_float16_t32", "uint8_ff32")
uint4_ff32 = DataType("uint4", "float16_float16_t32", "uint4_ff32")
uint4_bb32 = DataType("uint4", "bfloat16_bfloat16_t32", "uint4_bb32")
uint4_fi32 = DataType("uint4", "float16_int8_t32", "uint4_fi32")
uint4_bi32 = DataType("uint4", "bfloat16_int8_t32", "uint4_bi32")
uint4_f12i432 = DataType("uint4", "e5m6_int4_t32", "uint4_f12i4")
int4_f32 = DataType("int4", "float16_t32", "int4_f32")
int4_b32 = DataType("int4", "bfloat16_t32", "int4_b32")
# 1-bit exponent (integer)
e1m6b1fnuz = DataType("e1m6b1fnuz", "e8m0_t8", "e1m6_e32")  # bias overriden to 1, equivalent to int8
e1m2b1fnuz = DataType("e1m2b1fnuz", "e8m0_t8", "e1m2_e32")  # bias overriden to 1, equivalent to int4

# Original MX, tile size 16, exponent scale, prime bit shared between adjacent values

mx9 = DataType("int8", "e8m0_t16s2o1", "mx9")
mx6 = DataType("int5", "e8m0_t16s2o1", "mx6")
mx4 = DataType("int3", "e8m0_t16s2o1", "mx4")

# Subtile offsets, tile size 32, exponent scale
e2m1_e32s8o1 = DataType("e2m1fnuz", "e8m0_t32s8o1", "e2m1_e32s8o1")
e2m1_e32s4o1 = DataType("e2m1fnuz", "e8m0_t32s4o1", "e2m1_e32s4o1")
e2m1_e32s8o2 = DataType("e2m1fnuz", "e8m0_t32s8o2", "e2m1_e32s8o2")
e2m1_e32s4o2 = DataType("e2m1fnuz", "e8m0_t32s4o2", "e2m1_e32s4o2")

# lookup table datatypes for tiles and subtiles; offsets are built in to lookups
builder = LookupBuilder("e4m3fn", 4)

ilt_fi432_e32s8 = builder.make_datatype(NumberSpec("ilt_fi43_e2m3fnuz"), "e32s8")
ilt_fi433_e32s8 = builder.make_datatype(NumberSpec("ilt_fi43_e3m2fnuz"), "e32s8")
ilt_fi434_e32s8 = builder.make_datatype(NumberSpec("ilt_fi43_e4m3fnuz"), "e32s8")
ilt_fis434_e32s8 = builder.make_datatype(NumberSpec("ilt_fis43_e4m3fnuz"), "e32s8")
ilt_fis434_e32s4 = builder.make_datatype(NumberSpec("ilt_fis43_e4m3fnuz"), "e32s4")

ilt_fi443_e32s8 = builder.make_datatype(NumberSpec("ilt_fi44_e3m2fnuz"), "e32s8")
ilt_fi444_e32s8 = builder.make_datatype(NumberSpec("ilt_fi44_e4m3fnuz"), "e32s8")
ilt_fi432_e32s4 = builder.make_datatype(NumberSpec("ilt_fi43_e2m3fnuz"), "e32s4")
ilt_fi433_e32s4 = builder.make_datatype(NumberSpec("ilt_fi43_e3m2fnuz"), "e32s4")
ilt_fi434_e32s4 = builder.make_datatype(NumberSpec("ilt_fi43_e4m3fnuz"), "e32s4")
ilt_fi443_e32s4 = builder.make_datatype(NumberSpec("ilt_fi44_e3m2fnuz"), "e32s4")
ilt_fi444_e32s4 = builder.make_datatype(NumberSpec("ilt_fi44_e4m3fnuz"), "e32s4")

ilt_f422_e32s8 = builder.make_datatype(NumberSpec("ilt_f42_e2m3fnuz"), "e32s8")
ilt_f432_e32s8 = builder.make_datatype(NumberSpec("ilt_f43_e2m3fnuz"), "e32s8")
ilt_f433_e32s8 = builder.make_datatype(NumberSpec("ilt_f43_e3m2fnuz"), "e32s8")
ilt_f434_e32s8 = builder.make_datatype(NumberSpec("ilt_f43_e4m3fnuz"), "e32s8")
ilt_f443_e32s8 = builder.make_datatype(NumberSpec("ilt_f44_e3m2fnuz"), "e32s8")
ilt_f444_e32s8 = builder.make_datatype(NumberSpec("ilt_f44_e4m3fnuz"), "e32s8")
ilt_f422_e32s4 = builder.make_datatype(NumberSpec("ilt_f42_e2m3fnuz"), "e32s4")
ilt_f432_e32s4 = builder.make_datatype(NumberSpec("ilt_f43_e2m3fnuz"), "e32s4")
ilt_f423_e32s4 = builder.make_datatype(NumberSpec("ilt_f42_e3m2fnuz"), "e32s4")

ilt_f433_e32s4 = builder.make_datatype(NumberSpec("ilt_f43_e3m2fnuz"), "e32s4")
ilt_f434_e32s4 = builder.make_datatype(NumberSpec("ilt_f43_e4m3fnuz"), "e32s4")
ilt_f443_e32s4 = builder.make_datatype(NumberSpec("ilt_f44_e3m2fnuz"), "e32s4")
ilt_f444_e32s4 = builder.make_datatype(NumberSpec("ilt_f44_e4m3fnuz"), "e32s4")

ilt_fi422_e32 = builder.make_datatype(NumberSpec("ilt_fi42_e2m3fnuz"), "e32")
ilt_fi432_e32 = builder.make_datatype(NumberSpec("ilt_fi43_e2m3fnuz"), "e32")
ilt_f422_e32 = builder.make_datatype(NumberSpec("ilt_f43_e2m3fnuz"), "e32")
ilt_f432_e32 = builder.make_datatype(NumberSpec("ilt_f43_e2m3fnuz"), "e32")

ilt_f42_e32 = builder.make_datatype(NumberSpec("ilt_f42_e4m3fnuz"), "e32")
ilt_i42_e32 = builder.make_datatype(NumberSpec("ilt_i42_e4m3fnuz"), "e32")
ilt_fi42_e32 = builder.make_datatype(NumberSpec("ilt_fi42_e4m3fnuz"), "e32")
ilt_f43_e32 = builder.make_datatype(NumberSpec("ilt_f43_e4m3fnuz"), "e32")
ilt_i43_e32 = builder.make_datatype(NumberSpec("ilt_i43_e4m3fnuz"), "e32")
ilt_fi43_e32 = builder.make_datatype(NumberSpec("ilt_fi43_e4m3fnuz"), "e32")

ilt_f42_e32s8 = builder.make_datatype(NumberSpec("ilt_f42_e4m3fnuz"), "e32s8")
ilt_i42_e32s8 = builder.make_datatype(NumberSpec("ilt_i42_e4m3fnuz"), "e32s8")
ilt_fi42_e32s8 = builder.make_datatype(NumberSpec("ilt_fi42_e4m3fnuz"), "e32s8")
ilt_f43_e32s8 = builder.make_datatype(NumberSpec("ilt_f43_e4m3fnuz"), "e32s8")
ilt_i43_e32s8 = builder.make_datatype(NumberSpec("ilt_i43_e4m3fnuz"), "e32s8")
ilt_fi43_e32s8 = builder.make_datatype(NumberSpec("ilt_fi43_e4m3fnuz"), "e32s8")
ilt_f44_e32s8 = builder.make_datatype(NumberSpec("ilt_f44_e4m3fnuz"), "e32s8")
ilt_i44_e32s8 = builder.make_datatype(NumberSpec("ilt_i44_e4m3fnuz"), "e32s8")
ilt_fi44_e32s8 = builder.make_datatype(NumberSpec("ilt_fi44_e4m3fnuz"), "e32s8")

ilt_f42_e32s4 = builder.make_datatype(NumberSpec("ilt_f42_e4m3fnuz"), "e32s4")
ilt_i42_e32s4 = builder.make_datatype(NumberSpec("ilt_i42_e4m3fnuz"), "e32s4")
ilt_fi42_e32s4 = builder.make_datatype(NumberSpec("ilt_fi42_e4m3fnuz"), "e32s4")
ilt_f43_e32s4 = builder.make_datatype(NumberSpec("ilt_f43_e4m3fnuz"), "e32s4")
ilt_i43_e32s4 = builder.make_datatype(NumberSpec("ilt_i43_e4m3fnuz"), "e32s4")
ilt_fi43_e32s4 = builder.make_datatype(NumberSpec("ilt_fi43_e4m3fnuz"), "e32s4")
ilt_f44_e32s4 = builder.make_datatype(NumberSpec("ilt_f44_e4m3fnuz"), "e32s4")
ilt_i44_e32s4 = builder.make_datatype(NumberSpec("ilt_i44_e4m3fnuz"), "e32s4")
ilt_fi44_e32s4 = builder.make_datatype(NumberSpec("ilt_fi44_e4m3fnuz"), "e32s4")

# i420_e32 = builder.make_datatype(NumberSpec("i420_e2m1fnuz_e4m3fn"), "e32")
# i420_e32s8 = builder.make_datatype(NumberSpec("i420_e2m1fnuz_e4m3fn"), "e32s8")
# i420_e32s4 = builder.make_datatype(NumberSpec("i420_e2m1fnuz_e4m3fn"), "e32s4")
# i421_e32s8 = builder.make_datatype(NumberSpec("i421_e2m1fnuz_e4m3fn"), "e32s8")
# i421_e32s4 = builder.make_datatype(NumberSpec("i421_e2m1fnuz_e4m3fn"), "e32s4")
# i422_e32s8 = builder.make_datatype(NumberSpec("i422_e2m1fnuz_e4m3fn"), "e32s8")
# i422_e32s4 = builder.make_datatype(NumberSpec("i422_e2m1fnuz_e4m3fn"), "e32s4")
# i430_e32 = builder.make_datatype(NumberSpec("i430_e2m1fnuz_e4m3fn"), "e32")
# i430_e32s8 = builder.make_datatype(NumberSpec("i430_e2m1fnuz_e4m3fn"), "e32s8")
# i430_e32s4 = builder.make_datatype(NumberSpec("i430_e2m1fnuz_e4m3fn"), "e32s4")
# i431_e32s8 = builder.make_datatype(NumberSpec("i431_e2m1fnuz_e4m3fn"), "e32s8")
# i431_e32s4 = builder.make_datatype(NumberSpec("i431_e2m1fnuz_e4m3fn"), "e32s4")

# standard fp4 and int4, 4 bit index and 1 bit select
l41f4f4_e32 = builder.make_datatype(builder.make_lspec("f4-f4"), "e32")  # dummy to compare lookup mxfp4 with non-lookup mxfp4
l41f6f6_e32 = builder.make_datatype(builder.make_lspec("f6-f6"), "e32")  # dummy to compare lookup mxfp4 with non-lookup mxfp4
l41f6i6_e32 = builder.make_datatype(builder.make_lspec("f6-i6"), "e32")
l41f4i6_e32 = builder.make_datatype(builder.make_lspec("f4-i6"), "e32")
l41f4i6_e32s8 = builder.make_datatype(builder.make_lspec("f4-i6"), "e32s8")
l41f4i6_e32s4 = builder.make_datatype(builder.make_lspec("f4-i6"), "e32s4")
l41f63_e32 = builder.make_datatype(builder.make_lspec("f6-f3"), "e32")
l41f63_e32s8 = builder.make_datatype(builder.make_lspec("f6-f3"), "e32s8")
l41f63_e32s4 = builder.make_datatype(builder.make_lspec("f6-f3"), "e32s4")
l41f64_e32 = builder.make_datatype(builder.make_lspec("f6-f4"), "e32")
l41f64_e32s8 = builder.make_datatype(builder.make_lspec("f6-f4"), "e32s8")
l41f64_e32s4 = builder.make_datatype(builder.make_lspec("f6-f4"), "e32s4")
# standard fp4 and/or int4, shifted, 4 bit index and 2 bit select
l42f6431_e32 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1"), "e32")
l42f6431_e32s8 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1"), "e32s8")
l42f6431_e32s4 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1"), "e32s4")
l42f6420_e32 = builder.make_datatype(builder.make_lspec("f6-f4-f2-f0"), "e32")
l42f6420_e32s8 = builder.make_datatype(builder.make_lspec("f6-f4-f2-f0"), "e32s8")
l42f6420_e32s4 = builder.make_datatype(builder.make_lspec("f6-f4-f2-f0"), "e32s4")
l42f7531_e32 = builder.make_datatype(builder.make_lspec("f7-f5-f3-f1"), "e32")
l42f7531_e32s8 = builder.make_datatype(builder.make_lspec("f7-f5-f3-f1"), "e32s8")
l42f7531_e32s4 = builder.make_datatype(builder.make_lspec("f7-f5-f3-f1"), "e32s4")
# standard fp4 and/or int4, shifted, 4 bit index and 3 bit select
l43f_e32 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f3-f2-f1-f0"), "e32")
l43f_e32s8 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f3-f2-f1-f0"), "e32s8")
l43f_e32s4 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f3-f2-f1-f0"), "e32s4")

l43fi4_e32 = builder.make_datatype(builder.make_lspec("i7-f6-i5-f4-i3-f2-i1-f0"), "e32")
l43fi4_e32s8 = builder.make_datatype(builder.make_lspec("i7-f6-i5-f4-i3-f2-i1-f0"), "e32s8")
l43fi4_e32s4 = builder.make_datatype(builder.make_lspec("i7-f6-i5-f4-i3-f2-i1-f0"), "e32s4")
# l43fi3_e32 = builder.make_datatype(builder.make_lspec("i7-f6-i5-f4-i3-f2-i1-f0"), "e32")
l43fi3_e32s8 = builder.make_datatype(builder.make_lspec("i6-f4-i2-f0-i6s1-f4s1-i2s1-f0s1"), "e32s8")
l43fi3_e32s4 = builder.make_datatype(builder.make_lspec("i6-f4-i2-f0-i6s1-f4s1-i2s1-f0s1"), "e32s4")
l43fsi3_e32s4 = builder.make_datatype(builder.make_lspec("f6-f4-f2-f0-i6s1-i4s1-i2s1-i0s1"), "e32s4")

l42fis_e32s8 = builder.make_datatype(builder.make_lspec("f4-i6-f4s1-i6s1"), "e32s8")
l42fis_e32s4 = builder.make_datatype(builder.make_lspec("f4-i6-f4s1-i6s1"), "e32s4")
l43fis_e32s8 = builder.make_datatype(builder.make_lspec("f4-i6-f4s1-i6s1-f4s2-i6s2-f4s3-i6s3"), "e32s8")
l43fis_e32s4 = builder.make_datatype(builder.make_lspec("f4-i6-f4s1-i6s1-f4s2-i6s2-f4s3-i6s3"), "e32s4")

# standard fp4 and/or int4, shifted and scaled, 4 bit index and 3 bit select
l43f7654f64s12_e32s8 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f6s1-f4s1-f6s2-f4s2"), "e32s8")
l43f7654f64s12_e32s4 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f6s1-f4s1-f6s2-f4s2"), "e32s4")
l43f7654f64s24_e32s8 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f6s2-f4s2-f6s4-f4s4"), "e32s8")
l43f7654f6s24_e32s4 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f6s2-f4s2-f6s4-f4s4"), "e32s4")
l43f6431f63s12_e32s8 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1-f6s1-f4s1-f6s2-f4s2"), "e32s8")
l43f6431f63s12_e32s4 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1-f6s1-f3s1-f6s2-f3s2"), "e32s4")
l43f6431f63s24_e32s8 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1-f6s2-f3s2-f6s4-f3s4"), "e32s8")
l43f6431f63s24_e32s4 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1-f6s2-f3s2-f6s4-f3s4"), "e32s4")
l43f6431f64s24_e32s8 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1-f6s2-f4s2-f6s4-f4s4"), "e32s8")
l43f6431f64s24_e32s4 = builder.make_datatype(builder.make_lspec("f6-f4-f3-f1-f6s2-f4s2-f6s4-f4s4"), "e32s4")
# standard fp4 and int4, shifted, 4 bit index and 4 bit select
l44fi_e32 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f3-f2-f1-f0-i7-i6-i5-i4-i3-i2-i1-i0"), "e32")
l44fi_e32s8 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f3-f2-f1-f0-i7-i6-i5-i4-i3-i2-i1-i0"), "e32s8")
l44fi_e32s4 = builder.make_datatype(builder.make_lspec("f7-f6-f5-f4-f3-f2-f1-f0-i7-i6-i5-i4-i3-i2-i1-i0"), "e32s4")

del builder
