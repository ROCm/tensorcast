#!/usr/bin/env python
# tcast/attention.py: configuration and kernels for low precision attention interface
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import json
import math
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from . import kernels as K
from .common import CastMode, RoundMode, ScaleMode
from .datatype import DataType
from .incoherence import ICP
from .number import NumberSpec
from .utils import convert_enum, get_logger, getenv, is_power_of_2, triton_fp8_support

logger = get_logger("tcast")


# def is_class(obj, class_type):
#     """Check if the object is a subclass, even if it is not a class."""
#     return inspect.isclass(obj) and issubclass(obj, class_type)

#####
##### The Triton attention interface starts here
#####
##### The configuration is passed as a 64-bit integer called ACODE.  By convention, the
##### individual tensors (Q, K, ..) are identified by TCODE, and the predefined datatypes
##### are identified by NCODE.
#####
##### The generic Triton kernels are imported as K, and the Triton language is imported as tl.


# These encode the configuration in a 64-bit integer, indicating the bit positions and masks for each field.

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
# used to be axes here
# these are the masks for the configuration
SIZE_MASK = tl.constexpr(7)
FP8_TYPE_MASK = tl.constexpr(15)
# used to be axis mask here
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


# These functions use the constants above to extract the configuration.


# fmt: off
@triton.jit
def enabled(ACODE: tl.constexpr) -> bool: return ACODE.value != 0
@triton.jit
def shift_mask(ACODE: tl.constexpr, POS: tl.constexpr, MASK: tl.constexpr): return ACODE.value >> POS.value & MASK.value
@triton.jit
def fp8_code(ACODE: tl.constexpr) -> int: return shift_mask(ACODE, FP8_TYPE_POS, FP8_TYPE_MASK)
@triton.jit
def fp8_supported(ACODE: tl.constexpr, NCODE) -> bool: return (fp8_code(ACODE) & (1 << NCODE - 1 != 0)) != 0
@triton.jit
def get_size(ACODE: tl.constexpr) -> int: return (ACODE >> SIZE_POS) & SIZE_MASK
@triton.jit
def is_square(ACODE: tl.constexpr) -> int: return (ACODE >> SQUARE_POS) & SQUARE_MASK
@triton.jit
def quant_pos(TCODE: tl.constexpr) -> int: return (Q_POS, K_POS, V_POS, P_POS, DO_POS, DS_POS)[TCODE]
@triton.jit
def quant_code(ACODE: tl.constexpr, TCODE: tl.constexpr) -> int: return shift_mask(ACODE, quant_pos(TCODE), TENSOR_MASK)
@triton.jit
def needs_quant(ACODE: tl.constexpr, TCODE: tl.constexpr) -> bool: return quant_code(ACODE, TCODE) != 0
@triton.jit
def number_mxfp(ACODE: tl.constexpr, TCODE: tl.constexpr) -> bool: return quant_code(ACODE, TCODE) > 4
@triton.jit
def q_code(ACODE: tl.constexpr) -> int: return quant_code(ACODE, Q_INDEX)
@triton.jit
def k_code(ACODE: tl.constexpr) -> int: return quant_code(ACODE, K_INDEX)
@triton.jit
def v_code(ACODE: tl.constexpr) -> int: return quant_code(ACODE, V_INDEX)
@triton.jit
def p_code(ACODE: tl.constexpr) -> int: return quant_code(ACODE, P_INDEX)
@triton.jit
def do_code(ACODE: tl.constexpr) -> int: return quant_code(ACODE, DO_INDEX)
@triton.jit
def ds_code(ACODE: tl.constexpr) -> int: return quant_code(ACODE, DS_INDEX)
@triton.jit
def icp_qk(ACODE: tl.constexpr) -> bool: return shift_mask(ACODE, ICP_QK_POS, ICP_MASK) != 0
@triton.jit
def icp_pv(ACODE: tl.constexpr) -> bool: return shift_mask(ACODE, ICP_PV_POS, ICP_MASK) != 0
@triton.jit
def icp_fp32(ACODE: tl.constexpr) -> bool: return shift_mask(ACODE, ICP_FP32_POS, ICP_MASK) == 0
@triton.jit
def act_code(ACODE: tl.constexpr) -> int: return q_code(ACODE)
@triton.jit
def weight_code(ACODE: tl.constexpr) -> int: return k_code(ACODE)
@triton.jit
def roundmode(ACODE: tl.constexpr) -> int: return shift_mask(ACODE, ROUNDMODE_POS, ROUNDMODE_MASK)
@triton.jit
def roundeven(ACODE: tl.constexpr) -> int: return roundmode(ACODE) == int(K.RMODE_EVEN)
@triton.jit
def scalemode(ACODE: tl.constexpr) -> int: return shift_mask(ACODE, SCALEMODE_POS, SCALEMODE_MASK)
@triton.jit
def castmode(ACODE: tl.constexpr) -> int: return shift_mask(ACODE, CASTMODE_POS, CASTMODE_MASK)
@triton.jit
def virtual(ACODE: tl.constexpr) -> int: return castmode(ACODE) == int(K.CMODE_VIRTUAL)
@triton.jit
def scale_code(ACODE: tl.constexpr): return shift_mask(ACODE, SCALE_POS, SCALE_MASK)
@triton.jit
def scale_is_exponent(ACODE: tl.constexpr) -> bool: return scale_code(ACODE) == int(STYPE_E8M0)

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

# Some functions in the kernel namespace should be called in the attention namespace.
get_descale = K.get_descale
get_triton_dtype = K.get_triton_dtype

# fmt: on
@triton.jit
def needs_icp(ACODE: tl.constexpr, TCODE: tl.constexpr) -> bool:
    if (((TCODE == Q_INDEX) or (TCODE == K_INDEX)) or (TCODE == DS_INDEX)) and icp_qk(ACODE):
        return True
    if (((TCODE == P_INDEX) or (TCODE == V_INDEX)) or (TCODE == DO_INDEX)) and icp_pv(ACODE):
        return True
    return False


@triton.jit
def needs_quant_or_icp(ACODE: tl.constexpr, TCODE: tl.constexpr) -> tuple[bool, bool]:
    dq = needs_quant(ACODE, TCODE)
    di = needs_icp(ACODE, TCODE)
    return dq or di, dq, di


@triton.jit
def scale_dtype(ACODE: tl.constexpr, xtype: tl.core.dtype) -> tl.core.dtype:
    NCODE = scale_code(ACODE)
    if NCODE == STYPE_MATCH:
        return xtype
    return get_triton_dtype(STYPE_TLTYPE[NCODE - 1])


@triton.jit
def quant_dtype(ACODE: tl.constexpr, TCODE: tl.constexpr) -> tl.constexpr:
    # compute dtype, must be supported on hardware backend
    NCODE = quant_code(ACODE, TCODE)
    return get_triton_dtype(QUANT_TLTYPE[NCODE - 1])


# This is a debug variable to choose between rounding methods.  The first scales the tensor such a standard round
# function will complete the quantization. However, if the backends are not properly linked into the
# triton.language.extra.libdevice, the round function will not be available.  In that case, we need to
# use the second method, which uses bit manipulation to accomplish the same thing.  This is controlled in
# the call to K.quantize_float in scale_and_quantize below.

USE_ROUND: tl.constexpr = False


@triton.jit
def scale_and_quantize(x, imatrix, ACODE: tl.constexpr, TCODE: tl.constexpr, seed=19, offset=0, trans=False):
    """Returns the quantized x and the scale used."""
    # incoherence first
    if needs_icp(ACODE, TCODE):
        tl.device_assert(imatrix is not None)
        x = K.apply_incoherence(x, imatrix, trans, icp_fp32(ACODE))

    # if no quantization, the scale is 1
    stype = scale_dtype(ACODE, x.type.element_ty)
    if not needs_quant(ACODE, TCODE):
        return x, tl.cast(1.0, stype)

    # for quantization, we need the number specs and the output type
    NCODE = quant_code(ACODE, TCODE)
    EMAX = number_emax(NCODE)
    MAXFLOAT = number_maxfloat(NCODE)
    CMODE = castmode(ACODE)
    IS_EXP = scale_is_exponent(ACODE)
    tl.device_assert(CMODE != K.CMODE_COMPRESSED or NCODE != QUANT_E2M1FNUZ, "Packing not implemented yet")

    # get the scale (unbiased if exponent)
    scale = K.get_scale(x, IS_EXP, EMAX, scalemode(ACODE), MAXFLOAT, number_midmax(NCODE))

    # quantize
    out = K.quantize_float(
        x,
        scale,
        number_emax(NCODE),
        number_emin(NCODE),
        MAXFLOAT,
        roundmode(ACODE),
        CMODE,
        IS_EXP,
        quant_dtype(ACODE, TCODE),
        use_bitcast=not USE_ROUND,
        seed=seed,
        offset=offset,
    )

    # If the scale is an exponent, we need to convert the int32 shared exponent to a
    # biased exponent, offset by EMAX, and convert to the scale type.  The descale
    # can be acquired by calling get_descale.
    if IS_EXP and CMODE != K.CMODE_VIRTUAL:
        scale = K.bias_scale(scale, EMAX, number_emin(NCODE))
    else:
        scale = scale.to(stype)
    return out, scale


#####
##### The PyTorch interface starts here.  AttentionCode maps attrs to encoding and back; Attention owns
##### the attributes and is mostly reached though the API in the tcast namespace.
#####

# fmt: off
def _get_shortcuts():
    """Return the LP shortcuts, which will be accessed through the SHORTCUT environment variable."""
    shortcuts = {}

    def _set_shortcut_attrs(block, square, stype, q, p, qk, pv, fp32, smode, rmode, cmode) -> dict:
        return dict(
            block_size=(block, block) if square else (1, block), scale_dtype=stype,
            q_dtype=q, k_dtype=q, v_dtype=q, p_dtype=p, do_dtype=p, ds_dtype=p,
            icp_qk=qk, icp_pv=pv, icp_fp32=fp32, scalemode=smode, roundmode=rmode, castmode=cmode, fp8_types=triton_fp8_support()
        )

    def _get_shortcut_name(block, square, q, p, qk, pv, fp32, smode, rmode, cmode):
        all_or_split = "split" if q != ptype or pv != qk else "all"
        name = f"{all_or_split}_{'match' if stype is None else stype}_"
        name += f"{q}_{p}_" if q != p else f"{q}_"
        icp = "icp" if pv and qk else "icpqk" if qk else "icppv" if pv else ""
        if fp32:
            icp += "32"
        if icp:
            name += icp + "_"
        name += f"{block}x{block}" if square else f"1x{block}"
        name += f"_{smode.upper()[0]}{rmode.upper()[0]}{cmode.upper()[0]}"
        return name

    for icp_fp32 in (False, True):
        for icp_qk in (False, True):
            for icp_pv in (False, True):
                if not (icp_qk or icp_pv) and icp_fp32:
                    continue
                for qtype in (("e4m3fn", "e4m3fnuz"),):
                    for ptype in (("e4m3fn", "e4m3fnuz"), ("e5m2", "e5m2fnuz")):
                        for qpidx in (0, 1):
                            for block in (8, 16, 32, 64, 128):
                                for square in (True,):
                                    for stype in (None, "float32"):
                                        for roundmode in ("even", "away"):
                                            for scalemode in ("floor",):
                                                for castmode in ("virtual", "actual"):
                                                    name = _get_shortcut_name(
                                                        block, square, qtype[qpidx], ptype[qpidx], icp_qk, icp_pv, icp_fp32,
                                                        scalemode, roundmode, castmode
                                                    )
                                                    if name in shortcuts:
                                                        raise ValueError(f"Duplicate shortcut: {name}")
                                                    attrs = _set_shortcut_attrs(
                                                        block, square, stype, qtype[qpidx], ptype[qpidx],
                                                        icp_qk, icp_pv, icp_fp32, scalemode, roundmode, castmode
                                                    )
                                                    shortcuts[name] = attrs
    return shortcuts
    # fmt: on


class AttentionCode:
    """Represents encoding/decoding of Attention codes."""

    # fmt: off
    SCALE_TYPES = [None, "e8m23", "e5m10", "e8m7", "e8m0", "e5m3"]
    CODED_TYPES = [None, "e5m2", "e5m2fnuz", "e4m3fn", "e4m3fnuz", "mxfp6e3", "mxfp6e2", "mxfp4e2"]
    # these tuples are position, mask, type or result, and value if code == 0 (None if ignored)
    CODE_POSITIONS = {
        "fp8_types":    (int(FP8_TYPE_POS),  int(FP8_TYPE_MASK),  int,          None),
        "block_size":   (int(SIZE_POS),      int(SIZE_MASK),      int,          0),
        "block_square": (int(SQUARE_POS),    int(SQUARE_MASK),    bool,         True),
        "scale_dtype":  (int(SCALE_POS),     int(SCALE_MASK),     SCALE_TYPES,  None),
        "q_dtype":      (int(Q_POS),         int(TENSOR_MASK),    CODED_TYPES,  None),
        "k_dtype":      (int(K_POS),         int(TENSOR_MASK),    CODED_TYPES,  None),
        "v_dtype":      (int(V_POS),         int(TENSOR_MASK),    CODED_TYPES,  None),
        "p_dtype":      (int(P_POS),         int(TENSOR_MASK),    CODED_TYPES,  None),
        "do_dtype":     (int(DO_POS),        int(TENSOR_MASK),    CODED_TYPES,  None),
        "ds_dtype":     (int(DS_POS),        int(TENSOR_MASK),    CODED_TYPES,  None),
        "icp_qk":       (int(ICP_QK_POS),    int(ICP_MASK),       bool,         False),
        "icp_pv":       (int(ICP_PV_POS),    int(ICP_MASK),       bool,         False),
        "icp_fp32":     (int(ICP_FP32_POS),  int(ICP_MASK),       bool,         False),
        "roundmode":    (int(ROUNDMODE_POS), int(ROUNDMODE_MASK), RoundMode,    0),
        "scalemode":    (int(SCALEMODE_POS), int(SCALEMODE_MASK), ScaleMode,    0),
        "castmode":     (int(CASTMODE_POS),  int(CASTMODE_MASK),  CastMode,     0),
    }
    # fmt: on

    @classmethod
    def _lookup_info(cls, value, lookup):
        """Returns the common string, lookup index, the DataType and torch.dtype."""
        # value is input as a string or datatype
        common = index = dtype = torch_dtype = nspec = None
        if value is not None:
            if isinstance(value, DataType):
                dtype = value
            elif DataType.valid(name=value):
                dtype = DataType(name=value)
            elif DataType.valid(ncode=value):
                dtype = DataType(nspec=value)
            if dtype:
                common = dtype.nspec.name
                nspec = dtype.nspec
            elif NumberSpec.valid(value):
                nspec = NumberSpec(value)
            common = nspec.name
        lut_value = dtype.name if dtype else common if value is not None else None
        index = lookup.index(lut_value) if lut_value in lookup else None
        return common, index, dtype, torch_dtype

    @classmethod
    def _get_value(cls, code: np.int64, key: str):
        """From a code and a key, decode the value."""
        assert key in cls.CODE_POSITIONS, "Unknown key {key} for decoding"
        start, mask, type_or_lookup, _ = cls.CODE_POSITIONS.get(key)
        value = int((code >> start) & mask)
        if isinstance(type_or_lookup, tuple | list):
            # lists are strings representing datatypes, possibly including None
            value = type_or_lookup[value]
        elif (e := convert_enum(type_or_lookup, value, int)) is not None:
            value = e
        # elif is_class(type_or_lookup, Enum):
        #     value = type_or_lookup(value).name.lower()
        elif isinstance(type_or_lookup, type):
            if key == "block_size":
                assert value <= SIZE_MASK
                value = 2 ** (1 + value)
                value = (value, value) if cls._get_value(code, "block_square") else (1, value)
            else:
                value = type_or_lookup(value)
        elif type_or_lookup is None:  # ignored w/r/t encoding
            value = None
        return value

    @classmethod
    def _set_value(cls, key: str, value) -> np.int64:
        """Fron an attribute {key: value} return a code to add to the overall code."""
        if key not in cls.CODE_POSITIONS:  # this includes code, shortcut, and json_path
            return np.int64(0)
        shift, mask, type_or_lookup, _ = cls.CODE_POSITIONS.get(key)
        if type_or_lookup is None:
            return np.int64(0)  # do not encode
        if e := (convert_enum(type_or_lookup, value, int)) is not None:
            value = e
        # if is_class(type_or_lookup, Enum):
        #     if isinstance(value, str):
        #         value = get_enum(type_or_lookup, value)
        #     elif isinstance(value, int):
        #         value = type_or_lookup(value)
        #     assert isinstance(value, type_or_lookup)
        #     value = type_or_lookup(value).value
        elif isinstance(type_or_lookup, tuple | list):
            common, index, dtype, torch_dtype = cls._lookup_info(value, type_or_lookup)
            if index is None:
                raise KeyError(f"Invalid {key} value: {value}")
            value = index
        elif key == "block_size":
            if not is_power_of_2(value[0]) or not is_power_of_2(value[1]):
                raise ValueError(f"Invalid block size: {value[0]}, {value[1]}, must be powers of 2")
            s0, s1 = int(math.log2(value[0])) - 1, int(math.log2(value[1]) - 1)
            if max(s0, s1) > mask:
                raise ValueError(f"Invalid block size: {value[0]}, {value[1]}, must be less than {mask+1}")
            combined = (max(s0, s1) & mask) << shift
            if s0 == s1:
                sq_shift, sq_mask, _, _ = cls.CODE_POSITIONS.get("block_square")
                combined |= (1 & sq_mask) << sq_shift
            elif min(s0, s1) != 1:
                raise ValueError(f"Invalid block size: {value[0]}, {value[1]}, must be square or vector")
            value = combined
            mask = mask | (1 << sq_shift)
        return np.int64((value & mask) << shift)

    def __init__(self, **kwargs):
        self.code = self.encode(kwargs)

    @classmethod
    def encode(cls, attrs: dict) -> int:
        """Encode the attributes into a 64-bit integer."""
        code = np.int64(0)
        for key, value in attrs.items():
            code |= cls._set_value(key, value)
        return code

    @classmethod
    def infer_zero_code(cls, attrs: dict) -> bool:
        """Infer if the code is zero from the attributes."""
        for key in attrs:
            if key in cls.CODE_POSITIONS:
                shift, mask, type_or_lookup, zero_val = cls.CODE_POSITIONS.get(key)
                if isinstance(attrs[key], list | tuple):
                    val_to_test = max(attrs[key])
                elif (e := convert_enum(type_or_lookup, attrs[key], int)) is not None:
                    val_to_test = e
                else:
                    val_to_test = attrs[key]
                if zero_val not in (None, val_to_test):
                    return False
        return True
        # return all(cls.CODE_POSITIONS.get(key)[-1] in (None, attrs[key]) for key in attrs if key in cls.CODE_POSITIONS)

    @classmethod
    def decode(cls, code: int) -> dict:
        """Decode the code into a dictionary of attributes."""
        return {key: cls._get_value(code, key) for key in cls.CODE_POSITIONS} if code > 0 else {}


class Attention:
    """This describes what quantization options are to be used."""

    SHORTCUTS = _get_shortcuts()

    def __init__(
        self,
        code: np.int64 = None,
        json_path: Path | str = None,
        shortcut: str = None,
        block_size: tuple[int, int] = (0, 0),
        scale_dtype: DataType = None,
        q_dtype: DataType = None,
        k_dtype: DataType = None,
        v_dtype: DataType = None,
        p_dtype: DataType = None,
        do_dtype: DataType = None,
        ds_dtype: DataType = None,
        icp_qk: bool = False,
        icp_pv: bool = False,
        icp_fp32: bool = False,
        scalemode: ScaleMode | str | int = 0,
        roundmode: RoundMode | str | int = 0,
        castmode: CastMode | str | int = 0,
        fp8_types: int = triton_fp8_support(),
    ):
        self.code = code
        self.json_path = json_path
        self.shortcut = shortcut
        self.block_size = block_size
        self.scale_dtype = scale_dtype
        self.q_dtype = q_dtype
        self.k_dtype = k_dtype
        self.v_dtype = v_dtype
        self.p_dtype = p_dtype
        self.do_dtype = do_dtype
        self.ds_dtype = ds_dtype
        self.icp_qk = icp_qk
        self.icp_pv = icp_pv
        self.icp_fp32 = icp_fp32
        self.scalemode = ScaleMode(scalemode)
        self.roundmode = RoundMode(roundmode)
        self.castmode = CastMode(castmode)
        self.fp8_types = fp8_types

        # 1) pass a code to the constructor, defining the instance from the code
        # 2) pass a json file to the constructor, defining the instance from the json file
        # 3) pass a shortcut to the constructor, which, if defined, will define the instance
        # 4) pass all attributes other than the code/json_path, and shortcut
        # 5) pass nothing to the constructor, but set a code in the environment
        # 6) pass nothing to the constructor, but set a json file in the environment
        # 7) pass nothing to the constructor, but set a shortcut in the environment
        # 8) pass nothing to the constructor, but set attributes in the environment

        def json_attrs(json_path):
            if not json_path.exists():
                raise ValueError(f"json_path {json_path} does not exist.")
            with open(json_path) as f:
                attrs = json.load(f)
                # json files can have lists, but we want tuples
                for key, value in attrs.copy().items():
                    if isinstance(value, list):
                        attrs[key] = tuple(value)
            return attrs

        def shortcut_attrs(shortcut):
            if shortcut not in self.SHORTCUTS:
                raise ValueError(f"Invalid shortcut: {shortcut}")
            return self.SHORTCUTS[shortcut]

        def env_attrs():
            attrs = {}
            for attr in ("scale", "q", "k", "v", "p", "do", "ds", "act", "weight"):
                attr = attr + "_dtype"
                if attr in ("act_dtype", "weight_dtype"):
                    value = getenv(attr, None, str, "TC")
                    if value:
                        actual_attr = "q_dtype" if attr == "act_dtype" else "k_dtype"
                        actual_value = attrs[actual_attr]
                        if value == actual_value:
                            continue
                        elif actual_value:
                            raise ValueError(f"Conflict between {attr}: {value} and {actual_attr}: {actual_value}")
                        attrs[actual_attr] = value
                else:
                    attrs[attr] = getenv(attr, None, str, "TC")
            for attr in ("icp_qk", "icp_pv", "icp_fp32"):
                attrs[attr] = getenv(attr, False, bool, "TC")
            for attr in ("scalemode", "roundmode", "castmode"):
                attrs[attr] = getenv(attr, 0, str, "TC")
            attrs["block_size"] = getenv("block_size", "0, 0", tuple, "TC")
            return attrs

        if self.code is not None:
            attrs = AttentionCode.decode(self.code)
        elif self.json_path is not None:
            attrs = json_attrs(self.json_path)
        elif self.shortcut is not None:
            attrs = shortcut_attrs(self.shortcut)
        elif (code := getenv("code", None, np.int64, "TC")) is not None:
            attrs = AttentionCode.decode(code)
        elif (json_path := getenv("json_path", None, Path, "TC")) is not None:
            attrs = json_attrs(json_path)
        elif (shortcut := getenv("shortcut", None, str, "TC")) is not None:
            attrs = shortcut_attrs(shortcut)
        elif AttentionCode.infer_zero_code(self.__dict__):
            attrs = env_attrs()
        else:
            attrs = self.__dict__.copy()

        # convert dtypes if necessary
        for key, value in attrs.items():
            if key in AttentionCode.CODE_POSITIONS:
                type_or_lookup = AttentionCode.CODE_POSITIONS.get(key)[2]
                if (e := convert_enum(type_or_lookup, value)) is not None:
                    value = e
                # if is_class(type_or_lookup, Enum):
                #     if isinstance(value, str):
                #         value = get_enum(type_or_lookup, value)
                #     elif isinstance(value, int):
                #         value = type_or_lookup(value)
                if isinstance(type_or_lookup, tuple | list):
                    common, index, dtype, torch_dtype = AttentionCode._lookup_info(value, type_or_lookup)
                    if index is None:
                        raise KeyError(f"Invalid {key} value: {value}; no match in {type_or_lookup}")
                    value = dtype
                if hasattr(self, key) and key in AttentionCode.CODE_POSITIONS:
                    setattr(self, key, value)
        self.json_path = self.shortcut = None
        self.code = self.to_code()

    # fmt: off
    @property
    def unscaled(self) -> bool: return self.block_size == (0, 0)
    @property
    def tensor_scaled(self) -> bool: return self.block_size == (1, 1)
    @property
    def channel_scaled(self) -> bool: return self.block_size == (0, 1)
    @property
    def square_block(self) -> bool: return not self.tensor_scaled and self.block_size[0] == self.block_size[1]
    @property
    def weight_dtype(self) -> DataType: return self.k_dtype
    @property
    def act_dtype(self) -> DataType: return self.q_dtype
    @property
    def need_imatrix(self) -> bool: return self.icp_qk or self.icp_pv
    @property
    def enabled(self) -> bool: return bool(self.code)
    # fmt: on

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Attention(",
                f"  code=0x{self.code:010x}, json_path={self.json_path}, shortcut={self.shortcut}, ",
                f"  block_size={self.block_size}, square={self.square_block}, ",
                f"  scale_dtype={self.scale_dtype}, q_dtype={self.q_dtype}, k_dtype={self.k_dtype}, ",
                f"  v_dtype={self.v_dtype}, p_dtype={self.p_dtype}, do_dtype={self.do_dtype}, ds_dtype={self.ds_dtype}, ",
                f"  icp_qk={self.icp_qk}, icp_pv={self.icp_pv}, icp_fp32={self.icp_fp32}, ",
                f"  scalemode={self.scalemode}, roundmode={self.roundmode}, castmode={self.castmode}",
                ")",
            ]
        )

    def short_repr(self, tensors="qkvpos") -> str:
        """Return a short representation of the Attention."""
        tensors = tensors.lower()
        icpstr = scalestr = typestr = ""
        if self.need_imatrix:
            icpstr = "icp:"
            if "q" in tensors or "k" in tensors or "ds" in tensors:
                icpstr += f"{['', 'QK'][self.icp_qk]}"
            if "p" in tensors or "v" in tensors or "do" in tensors:
                icpstr += f"{['', 'PV'][self.icp_pv]}"
            icpstr += f"{['', '32'][self.icp_fp32]}"
        if any(getattr(self, f"{t}_dtype") is not None for t in tensors):
            scalestr = f"types:scale:{str(self.scale_dtype)},"
            for t in tensors:
                if dtype := getattr(self, f"{t}_dtype"):
                    comma = "," if typestr else ""
                    typestr += f"{comma}{t.upper()}:{dtype.name}"
            if icpstr:
                typestr += "  "
        return f"{scalestr}{typestr}{icpstr}"

    def __str__(self) -> str:
        return repr(self)

    def to_code(self) -> np.int64:
        """Generate a code from the Attention attributes."""
        attrs = self.__dict__.copy()
        attrs.pop("code", None)
        attrs.pop("json_path", None)
        attrs.pop("shortcut", None)
        encoder = AttentionCode(**attrs)
        return encoder.code

    def dtype_from_index(self, index: int) -> DataType:
        """Return the datatype for the given index."""
        return (self.q_dtype, self.k_dtype, self.v_dtype, self.p_dtype, self.ds_dtype, self.do_dtype)[index]

    def make_quant_and_scale(self, tensor: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and scale the tensor."""
        # This should not be called if configuration is not enabled
        if not self.enabled:
            raise ValueError("Attention is not enabled.")
        # If the scale type is None, use the type of the tensor.
        sdtype = self.scale_dtype.nspec.torch_dtype if self.scale_dtype is not None else tensor.dtype
        assert sdtype is not None and sdtype.is_floating_point
        # Next, the quantization.  If the type is None, return the input tensor and a scale of 1
        qdtype = self.dtype_from_index(index)
        needs_icp = self.icp_qk and index in (Q_INDEX, K_INDEX, DS_INDEX) or self.icp_pv and index in (V_INDEX, P_INDEX, DO_INDEX)
        if qdtype is None:
            if needs_icp:
                quant, scale = torch.empty_like(tensor), torch.tensor([1.0], dtype=sdtype, device=tensor.device)
            else:
                quant, scale = tensor, torch.tensor([1.0], dtype=sdtype, device=tensor.device)
        else:
            shape = tuple([triton.cdiv(tensor.size(s), self.block_size[s]) for s in (0, 1)])
            quant, scale = (
                torch.empty_like(tensor, dtype=qdtype.nspec.torch_dtype),
                torch.empty(shape, dtype=sdtype, device=tensor.device),
            )
        return quant, scale

    def get_incoherence_matrix(
        self, size: int, torch_dtype: str | torch.dtype = torch.float32, walsh: bool = True, randomize: bool = True
    ) -> torch.Tensor:
        """Return the incoherence matrix for the given datatype."""
        if not self.need_imatrix:
            return None
        dtype = (
            DataType(name="float32")
            if self.icp_fp32
            else DataType(name=torch_dtype)
            if isinstance(torch_dtype, str)
            else torch_dtype
        )
        return ICP.get_imatrix(size, dtype, walsh=walsh, randomize=randomize)

    def print_shortcuts(self):
        """Print the shortcuts for the Attention class."""
        logger.info(f"Attention has {len(self.SHORTCUTS)} shortcuts.")
        for shortcut, attrs in self.SHORTCUTS.items():
            tmp_cfg = Attention(**attrs)
            logger.debug(f"Shortcut {shortcut} has code {hex(tmp_cfg.code)}")

    def __eq__(self, other):
        if not isinstance(other, Attention):
            return False
        for attr in self.__dict__:
            if attr not in other.__dict__:
                return False
            if attr.endswith("_dtype"):
                if str(getattr(self, attr)) != str(getattr(other, attr)):
                    return False
            elif not attr.endswith("mode"):
                if getattr(self, attr) != getattr(other, attr):
                    return False
        return True
