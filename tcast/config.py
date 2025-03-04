#!/usr/bin/env python
# tcast/config.py: configuration for low precision attention and linear layers
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import torch

from .datatype import DataType
from .utils import getenv

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
                    # then the order of the axes (dims in PyTorch) is important, and we swap them so that the first axis is the
                    # one that has length 1.
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

    def __post_init__(self):
        # 1) pass a code to the constructor
        # 2) pass a json file to the constructor
        # 3) pass nothing to the constructor, but set a code in the environment
        # 4) pass nothing to the constructor, but set a json file in the environment
        # 5) pass nothing to the constructor, but set attributes in the environment
        env_code, env_json = getenv("code", None, int), getenv("json", None, str)
        code = self.code if self.code is not None else env_code if self.json_path is None else None
        json_path = Path(self.json_path) if self.json_path else Path(env_json) if env_json else None
        if code is not None:
            self.code, attrs = code, _LPCode.decode(code)
        elif json_path and json_path.exists():
            with open(json_path) as f:
                attrs = json.load(f)
        else:
            attrs = {}
            for attr in (
                "scale_dtype", "act_dtype", "weight_dtype", "q_dtype", "k_dtype", "v_dtype", "p_dtype", "do_dtype", "ds_dtype"
            ):
                attrs[attr] = getenv(attr, "", str)
            for attr in ("icp_qk", "icp_pv"):
                attrs[attr] = getenv(attr, False, bool)
            attrs["block_size"] = getenv("block_size", "0, 0", tuple)
            attrs["block_axes"] = getenv("block_axes", "0, 0", tuple)
        for key, value in attrs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def to_code(self) -> int:
        pass

