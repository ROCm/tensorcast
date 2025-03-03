#!/usr/bin/env python
# tcast/config.py: configuration for low precision attention and linear layers
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from typing import ClassVar, NamedTuple

import torch

from .datatype import DataType
from .utils import cdiv, getenv, is_triton_available

logger = logging.getLogger(f"tcast.{__name__}")

EPS = torch.finfo(torch.float32).smallest_normal


class _LPCode:
    """Represents encoding/decoding of LP Config codes."""
    SCALE_TYPES = [
        "none", "float32", "float16", "bfloat16", "e8m0", "e5m3",
        "fp32_fp32", "fp16_fp16", "bf16_bf16", "fp32_int8, fp16_int8", "bf16_int8"
    ]
    CODED_LP_TYPES = [
        "none", "float8_e5m2", "float8_e5m2fnuz", "float8_e4m3fn", "float8_e4m3fnuz",
        "mxfp6e3", "mxfp6e2", "mxfp4e2"
    ]
    CODE_POSITIONS = {
        # block info is in the first 8 bits
        "block_size": (0, 3), # 1, 4, 8, 16, 32, 64, 128, 256 are valid block sizes (note that 2 is not)
        "block_square": (3, 1), # indicates that the block is square
        "block_axis1": (2, 2), # 2 bit index for the reduction dumension that is possibly of size 1
        "block_axis2": (6, 2), # 2 bit index for the reduction dimension that is definitely not of size 1
        # scale type is 4 bits
        "scale_type": (8, 4),
        # datatypes are 3 bits each, 18 total bits
        "act_q_dtype": (12, 3),
        "act_dtype": (12, 3),
        "q_dtype": (12, 3),
        "weight_k_dtype": (15, 3),
        "weight_dtype": (15, 3),
        "k_dtype": (15, 3),
        "v_dtype": (18, 3),
        "p_dtype": (21, 3),
        "do_dtype": (24, 3),
        "ds_dtype": (27, 3),
        "icp": (30, 1),
        "icp_qk": (30, 1),
        "icp_pv": (31, 1),
    }
    def __init__(self, **kwargs):
        if "code" in kwargs:
            self.code = kwargs["code"]
        else:
            self.code = 0
            reverse_block_axes = None
            for key, value in kwargs.items():
                if key == "block_size":
                    # block size works like this: it can be either square (both dimensions 2, 4, 8, 16, 32, 64, 128, or 256)
                    # or vector (one dimension is 1 and the other is 2, 4, 8, 16, 32, 64, 128, or 256).
                    # The encoded block size is x such that 2**(x+1) is the block size.
                    # The block axes are the dims of the block that are reduced for the scale. If the block is not square,
                    # then the order of the axes (dims in PyTorch) is important, and we swap them so that the first axis is the one
                    # that has length 1.
                    assert isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value)
                    if not all(v in (1, 2, 4, 8, 16, 32, 64, 128, 256) for v in value):
                        raise ValueError(f"Invalid block size: {value}")
                    self.code |= ((max(value).bit_length() - 2) << self.CODE_POSITIONS["block_size"][0])
                    self.code |= ((min(value) != 1) << self.CODE_POSITIONS["block_square"][0])
                    if value[0] != value[1]:
                        if (value[0] == 1) != (value[1] == 1):
                            raise ValueError("Block size axes must both be 1 or neither.")
                    reverse_block_axes = value[1] == 1
                elif key == "block_axes":
                    assert reverse_block_axes is not None
                    assert isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value)
                    value = tuple(reversed(value)) if reverse_block_axes else value
                    self.code |= (value[0] << self.CODE_POSITIONS["block_axis1"][0])
                    self.code |= (value[1] << self.CODE_POSITIONS["block_axis2"][0])
                else:
                    if key.count("_") == 2:  # account for LPConfig attrs that do double duty for linear and attention
                        key = "_".join(key.split("_")[1:])
                    if key not in self.CODE_POSITIONS:
                        raise ValueError(f"Invalid key: {key}")
                    type_table = self.SCALE_TYPES if key == "scale_type" else self.CODED_LP_TYPES
                    if value.lower() not in type_table:
                        raise ValueError(f"Invalid datatype: {value}")
                    self.code |= (type_table.index(value.lower()) << self.CODE_POSITIONS[key][0])

    @staticmethod
    def decode(code: int) -> dict:
        """Decode the code into a dictionary of attributes."""

        def _get_value(code: int, key: str) -> int:
            start, length = _LPCode.CODE_POSITIONS[key]
            return (code >> start) & ((1 << length) - 1)

        attrs = {}
        if code > 0:
            attrs["block_axes"] = (_get_value(code, "block_axes1"), _get_value(code, "block_axes2"))
            block_size = 2**(2 + _get_value(code, "block_size"))
            attrs["block_size"] = (block_size, block_size) if _get_value(code, "block_square") else (1, block_size)
            for key, (start, length) in _LPCode.CODE_POSITIONS.items():
                if key == "scale_type":
                    attrs[key] = _LPCode.SCALE_TYPES[(code >> start) & ((1 << length) - 1)]
                elif key.startswith("icp"):
                    attrs[key] = bool((code >> start) & ((1 << length) - 1))
                elif not key.startswith("block"):
                    attrs[key] = _LPCode.CODED_LP_TYPES[(code >> start) & ((1 << length) - 1)]
        return attrs


@dataclass
class LPConfig:
    """This describes what quantization options are to be used."""

    code: int = 0                           # 32-bit code for the quantization option
    json_path: Path | str = None            # path to a JSON file with the quantization options
    block_size: tuple[int, int] = (0, 0)    # block size for scaling
    block_axes: tuple[int, int] = (0, 0)    # axes for scaling
    scale_dtype: str | int = None           # scale type can be a string or an int; "none" matches the scale type to the input
    act_q_dtype: str | int = None           # some native datatypes can be expressed as integers, the rest can be with strings
    weight_k_dtype: str | int = None        # a string of "none" means do not quantize
    v_dtype: DataType = None
    p_dtype: DataType = None
    do_dtype: DataType = None
    ds_dtype: DataType = None
    icp: bool = False
    icp_pv: bool = False

    def __setattr__(self, name, value):
        if "dtype" in name and isinstance(value, str):
            value = DataType(name=value)
        if name in ("act_dtype", "act_q_dtype", "q_dtype"):
            self.act_q_dtype = value
        elif name in ("weight_dtype", "weight_k_dtype", "k_dtype"):
            self.weight_k_dtype = value
        elif name in ("icp", "icp_qk"):
            self.icp = value
        elif hasattr(self, name):
            super().__setattr__(name, value)

    # fmt: off
    @property
    def square_block(self) -> bool: return self.block_size[0] == self.block_size[1]
    @property
    def weight_dtype(self) -> DataType: return self.weight_k_dtype
    @property
    def act_dtype(self) -> tuple[int, int]: return self.act_q_dtype
    @property
    def q_dtype(self) -> tuple[int, int]: return self.act_q_dtype
    @property
    def k_dtype(self) -> tuple[int, int]: return self.weight_k_dtype
    @property
    def icp_qk(self) -> bool: return self.icp
    @property
    def enabled(self) -> bool: return bool(self.code)
    # fmt: on

    def _get_attrs_from_code(self, code):
        """Decode the code into the attributes. Called from _get_attrs."""
        self.code, attrs = code, _LPCode.decode(code)
        for key, value in attrs.items():
            setattr(self, key, value)

    def _get_attrs_from_json(self, jpath: Path | str = None) -> bool:
        """Load the attributes from a JSON file. Called from _get_attrs."""
        if jpath is None:
            jpath = getenv("json", "", str)
            if jpath:
                jpath = Path(jpath)
            else:
                jpath = Path(__file__).parent / "config.json"
        if jpath.exists():
            with open(jpath) as f:
                jdata = json.load(f)
            for key, value in jdata.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            return True
        return False

    def _get_attrs(self, jname: str = None, code: int = None) -> bool:
        """The FA interface does not include quantization parameters, so we get them from the env, a json filem or a code."""
        # if code is 0, disable
        # if code > 0, use the code
        # if code is None
        # get enablement from the environment,
        #  else use the code
        if code is not None:
            self.enable, self.code = code != 0, code
            self._get_attrs_from_code(self)
        else:
            self.enable = getenv("enable", dtype=bool)
            done = (self.enable and isinstance(jname, Path | str) or getenv("json", "", str)) and self._get_attrs_from_json(jname)
            if not done:
                for attr in (
                    "scale_dtype", "act_dtype", "weight_dtype", "q_dtype", "k_dtype", "v_dtype", "p_dtype", "do_dtype", "ds_dtype"
                ):
                    setattr(self, attr, getenv(attr, "none", str))
                for attr in ("icp_qk", "icp_pv"):
                    setattr(self, attr, getenv(attr, False, bool))
                setattr(self, "block_size", getenv("block_size", "0, 0", tuple))
                setattr(self, "block_axes", getenv("block_axes", "0, 0", tuple))


    def __post_init__(self):
        # When creating the LPConfig with no params, it is disabled, because code defaults to 0
        # When creating the LPConfig with non-zero code it is enabled and the code is used to configure the attributes
        # When creating the LPConfig with a json file, it is enabled and the json file is used to configure the attributes
        # When creating the LPConfig with other attributes, the code is defined by the other attributes
        if self.json_path:
            self.enable, self.code = False, 0
        elif self.enable is None and self.code is None and self.json_path is None:
            self.code = 0
        elif self.enable is None:
            if self.code is None
                self._get_attrs()
            if 


    def to_code(self) -> int:
        pass






    def _get_attrs_from_env(self):
        """The FA interface does not include quantization parameters, so we get them from the environment."""
        self.enable = getenv("enable", dtype=bool)
        if not self.enable:
            return
        # get some values
        if getenv("fake", False, bool):
            self.fake = True
        self.forward_type = getenv("forward_type", FP8Type.E4, FP8Type)
        self.backward_type = getenv("backward_type", FP8Type.NONE, FP8Type)
        self.output_type = getenv("output_type", FP8Type.NONE, FP8Type)
        self.scale_type = getenv("scale_type", ScaleType.FP32, ScaleType)
        self.scope = getenv("scope", ScaleScope.BLOCK, ScaleScope)
        self.icp_qkds = getenv("icp_qkds", False, bool)
        self.icp_pvdo = getenv("icp_pvdo", False, bool)
        self.block = getenv("block", 32, int)

    def __post_init__(self):
        """Encode the AttentionConfig into a 32-bit code.

        The bits are as follows:
        2 for fwd type
        2 for bwd type
        2 for output type
        2 for scale type
        2 for scale scope
        3 for block size
        1 for qkds ICP
        1 for pvdo ICP
        1 for fake quantization
        1 for NANOO
        """
        if self.enable is None:
            self._get_attrs_from_env()
        if not self.enable:
            self.code = 0
            return
        if not self.is_block_scaled:
            raise NotImplementedError("Row and tensor scaling are not yet implemented.")
        if self.is_tensor_scaled and self.icp:
            raise ValueError("ICP is not supported with tensor scaling.")
        self.code = self.forward_type.value
        self.code |= self.backward_type.value << 2
        self.code |= self.output_type.value << 4
        self.code |= self.scale_type.value << 6
        self.code |= self.scale_scope.value << 8
        if self.scale_scope != ScaleScope.TENSOR:
            block = int(math.log2(self.block)) - 1
            self.code |= block << 10
        self.code |= int(self.icp_qkds) << 13
        self.code |= int(self.icp_pvdo) << 14
        self.code |= int(self.fake) << 15
        self.code |= int(self.is_hip) << 16
        assert self.code != 0


class RoundMode(Enum):
    """Round modes, must match extension."""

    ZERO = 0  # nearest, ties toward zero
    AWAY = 1  # nearest, ties away from zero
    EVEN = 2  # nearest, ties to even
    STOCHASTIC = 3


class ScaleMode(Enum):
    """Scale modes, must match extension."""

    FLOOR = 0  # OCP MX standard
    CEIL = 1  # proposed by Intel
    MIDMAX = 2  # AMD's finest
    OPTION3 = 3  # proposed by Meta
    TOPBINADE = 4  # proposed by Microsoft


class ComputeMode(Enum):
    """Compute mode (if available)."""

    TORCH = 0  # torch operations
    TRITON = 1  # triton operations
    GPU = 2  # cuda/hip extension
    CPU = 3  # cpp host extension
    ANY = 4  # best available extension


class CastMode(Enum):
    """Virtual (fake), actual (with scales), compressed (packed sparsity and multiple values packed into single elements."""

    VIRTUAL = 0  # fake quantization, just returns with the same torch.dtype and shape
    ACTUAL = 1  # data, scale, zero, meta, and mask in the appropriate datatypes
    COMPRESS = 2  # packs multiple values into single elements, e.g. 2 4-bit codebook indices, sparsity mask 8x compression


class InfNaN(Enum):
    """Inf and NaN handling."""

    IEEE = 0  # standard IEEE 754 methods
    FN = 1  # all non-sign bits on = NaN, all non-sign bits off = Inf
    FNUZ = 2  # neg zero = Nan, no inf
    INUZ = 3  # proposed by IEEE working group P3109


def get_enum(etype: Enum, s: str, silent: bool = False) -> int | None:
    """Create the enum from a string matching the name."""
    if isinstance(s, etype):
        return s
    if isinstance(s, str):
        if s.upper() in dir(etype):
            return etype[s.upper()]
        if not silent:
            raise ValueError(f"'{s}' is not a valid {etype.__name__} name.")
    return None


class ScaleData(NamedTuple):
    """ScaleData: scales, zero points, and meta data for a datatype."""

    scale: torch.Tensor
    zero: torch.Tensor
    tensor: torch.Tensor
    meta: torch.Tensor
    mask: torch.Tensor


IMPLICIT_CODEBOOKS = dict(
    cb42fe0123_e3m2fnuz=dict(
        index_bits=4,
        meta_bits=1,
        midmax=14.0,
        mappings=[
            [0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 6.0000, 8.0000, 12.0000],
            [0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 3.0000, 4.0000, 6.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.5000, 2.0000, 3.0000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.7500, 1.0000, 1.5000],
        ],
        midpoints=[
            [0.5000, 1.5000, 2.5000, 3.5000, 5.0000, 7.0000, 10.0000],
            [0.2500, 0.7500, 1.2500, 1.7500, 2.5000, 3.5000, 5.0000],
            [0.1250, 0.3750, 0.6250, 0.8750, 1.2500, 1.7500, 2.5000],
            [0.0625, 0.1875, 0.3125, 0.4375, 0.6250, 0.8750, 1.2500],
        ],
    ),
    cb42f1346_e2m3fnuz=dict(
        index_bits=4,
        meta_bits=1,
        midmax=7.5,
        mappings=[
            [0.0000, 0.7500, 1.2500, 1.7500, 2.5000, 3.5000, 5.0000, 7.0000],
            [0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 3.0000, 4.0000, 6.0000],
            [0.0000, 0.3750, 0.8750, 1.3750, 1.8750, 2.7500, 3.7500, 5.5000],
            [0.0000, 0.1250, 0.6250, 1.1250, 1.6250, 2.2500, 3.2500, 4.5000],
        ],
        midpoints=[
            [0.3750, 1.0000, 1.5000, 2.0000, 3.0000, 4.0000, 6.0000],
            [0.2500, 0.7500, 1.2500, 1.7500, 2.5000, 3.5000, 5.0000],
            [0.1875, 0.6250, 1.1250, 1.6250, 2.2500, 3.2500, 4.5000],
            [0.0625, 0.3750, 0.8750, 1.3750, 1.8750, 2.7500, 3.7500],
        ],
    ),
)


MX2NUMSPEC = dict(
    mxfp8e5='DataType("e5m2", "e8m0_t32", "mxfp8e5")',
    mxfp8e4='DataType("e4m3fn", "e8m0_t32", "mxfp8e4")',
    mxfp6e3='DataType("e3m2fnuz", "e8m0_t32", "mxfp6e3")',
    mxfp6e2='DataType("e2m3fnuz", "e8m0_t32", "mxfp6e2")',
    mxfp4e2='DataType("e2m1fnuz", "e8m0_t32", "mxfp6e2")',
    mxint8='DataType("int8", "e8m0_t32", "mxint8")',
    mxint4='DataType("int4", "e8m0_t32", "mxint4")',
    bfp16='DataType("int8", "e8m0_t8", "bfp16")',
    mx9='DataType("cb81e01", "e8m0_t16s2", "mx9")',
    mx6='DataType("cb51e01", "e8m0_t16s2", "mx6")',
    mx4='DataType("cb31e01", "e8m0_t16s2", "mx4")',
)


class Modes:
    """Static class to manage modes used in casting."""

    round: ClassVar[RoundMode] = RoundMode.EVEN
    scale: ClassVar[ScaleMode] = ScaleMode.FLOOR
    compute: ClassVar[ComputeMode] = ComputeMode.ANY
    cast: ClassVar[CastMode] = CastMode.ACTUAL
    saved_round: ClassVar[RoundMode] = RoundMode.EVEN
    saved_scale: ClassVar[ScaleMode] = ScaleMode.FLOOR
    saved_compute: ClassVar[ComputeMode] = ComputeMode.ANY
    saved_cast: ClassVar[CastMode] = CastMode.ACTUAL
    warn_cpu: ClassVar[bool] = True
    warn_gpu: ClassVar[bool] = True
    warn_triton: ClassVar[bool] = True

    @classmethod
    def restore_modes(cls):
        """Restore the saved modes."""
        cls.round, cls.scale, cls.compute, cls.cast = cls.saved_round, cls.saved_scale, cls.saved_compute, cls.saved_cast

    @classmethod
    def set_modes(
        cls, roundmode: str | RoundMode, scalemode: str | ScaleMode, computemode: str | ComputeMode, castmode: str | CastMode
    ) -> dict:
        """Check and set the rounding and scaling modes.  If compmode requires extension, load it."""
        cls.saved_round, cls.saved_scale, cls.saved_compute, cls.saved_cast = cls.round, cls.scale, cls.compute, cls.cast
        roundmode, scalemode, computemode, castmode = (
            get_enum(RoundMode, roundmode),
            get_enum(ScaleMode, scalemode),
            get_enum(ComputeMode, computemode),
            get_enum(CastMode, castmode),
        )
        if roundmode is not None:
            cls.round = roundmode
        if scalemode is not None:
            cls.scale = scalemode
        if castmode is not None:
            cls.cast = castmode
        if computemode is not None:
            if computemode == ComputeMode.GPU and cls.warn_gpu:
                logger.warning(f"Extension {computemode} not implemented, using torch")
                cls.warn_gpu = False
                cls.compute = ComputeMode.TORCH
            elif computemode == ComputeMode.CPU and cls.warn_cpu:
                logger.warning(f"Extension {computemode} not implemented, using torch")
                cls.warn_cpu = False
                cls.compute = ComputeMode.TORCH
            elif computemode == ComputeMode.TRITON and cls.warn_triton and not is_triton_available():
                logger.warning("triton not installed, using torch")
                cls.warn_triton = False
                cls.compute = ComputeMode.TORCH
            else:
                cls.compute = computemode
        return dict(roundmode=cls.round, scalemode=cls.scale, computemode=cls.compute, castmode=cls.cast)
