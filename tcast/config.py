#!/usr/bin/env python
# tcast/config.py: configuration for low precision attention and linear layers
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path

import torch

from .datatype import DataType
from .snippets import (
    LP_AXIS0_POS,
    LP_AXIS1_POS,
    LP_AXIS_MASK,
    LP_DO_POS,
    LP_DS_POS,
    LP_ICP_FP32_POS,
    LP_ICP_MASK,
    LP_ICP_PV_POS,
    LP_ICP_QK_POS,
    LP_K_POS,
    LP_P_POS,
    LP_Q_POS,
    LP_SCALE_MASK,
    LP_SCALE_POS,
    LP_SIZE_MASK,
    LP_SIZE_POS,
    LP_SQUARE_MASK,
    LP_SQUARE_POS,
    LP_TENSOR_MASK,
    LP_V_POS,
)
from .utils import get_logger

logger = get_logger("tcast")

EPS = torch.finfo(torch.float32).smallest_normal


def _get_shortcuts():
    """Return the LP shortcuts, which will be accessed through the SHORTCUT environment variable."""
    lp_shortcuts = {}
    for icp_fp32 in (False, True):
        for icp_qk in (False, True):
            for icp_pv in (False, True):
                if not (icp_qk or icp_pv) and icp_fp32:
                    continue
                for qtype in (("float8_e4m3fn", "float8_e4m3fnuz"),):
                    for ptype in  (("float8_e4m3fn", "float8_e4m3fnuz"), ("float8_e5m2", "float8_e5m2fnuz")):
                        for block in (8, 16, 32, 64, 128):
                            for square in (True,):
                                for stype in ("none", "float32"):
                                    sdict = {}
                                    all_or_split = "split" if qtype != ptype or icp_pv != icp_qk else "all"
                                    name = f"{all_or_split}_"
                                    qt, pt = qtype.removeprefix("float8_"), ptype[0].removeprefix("float8_")
                                    name += f"{qt}_{pt}_" if qtype != ptype else f"{qt}_"
                                    sdict["icp_pv"] = icp_pv
                                    sdict["icp_qk"] = icp_qk
                                    sdict["icp_fp32"] = icp_fp32
                                    icp = "icp"
                                    if icp_pv != icp_qk:
                                        icp += "qk" if icp_qk else "pv"
                                    if icp_fp32:
                                        icp += "32"
                                    name += icp + "_"
                                    name += f"{block}x{block}" if square else f"1,{block}"
                                    sdict["block_size"] = (block, block) if square else (1, block)
                                    sdict["block_axes"] = (0, 1)
                                    sdict["scale_type"] = stype
                                    for tensor in ("q", "k", "v"):
                                        sdict[f"{tensor}_dtype"] = qtype
                                    for tensor in ("p", "ds", "do"):
                                        sdict[f"{tensor}_dtype"] = ptype
                                    lp_shortcuts[name] = sdict
    return lp_shortcuts


LP_SHORTCUTS: dict[str, dict] = _get_shortcuts()


class _LPCode:
    """Represents encoding/decoding of LP Config codes."""

    SCALE_TYPES = ["none", "float32", "float16", "bfloat16", "e8m0", "e5m3"]
    CODED_LP_TYPES = [
        "none",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "mxfp6e3",
        "mxfp6e2",
        "mxfp4e2",
    ]
    CODE_POSITIONS = {
        "block_size": (LP_SIZE_POS, LP_SIZE_MASK),  # 2, 4, 8, 16, 32, 64, 128, 256 are valid block sizes
        "block_square": (LP_SQUARE_POS, LP_SQUARE_MASK),  # indicates that the block is square
        "block_axis0": (LP_AXIS0_POS, LP_AXIS_MASK),  # 2 bit index for the reduction dumension that is possibly of size 1
        "block_axis1": (LP_AXIS1_POS, LP_AXIS_MASK),  # 2 bit index for the reduction dimension that is definitely not of size 1
        # scale type is 3 bits
        "scale_type": (LP_SCALE_POS, LP_SCALE_MASK),
        # datatypes are 3 bits each, 18 total bits
        "q_dtype": (LP_Q_POS, LP_TENSOR_MASK),
        "k_dtype": (LP_K_POS, LP_TENSOR_MASK),
        "v_dtype": (LP_V_POS, LP_TENSOR_MASK),
        "p_dtype": (LP_P_POS, LP_TENSOR_MASK),
        "do_dtype": (LP_DO_POS, LP_TENSOR_MASK),
        "ds_dtype": (LP_DS_POS, LP_TENSOR_MASK),
        "icp_qk": (LP_ICP_QK_POS, LP_ICP_MASK),
        "icp_pv": (LP_ICP_PV_POS, LP_ICP_MASK),
        "icp_fp32": (LP_ICP_FP32_POS, LP_ICP_MASK),  # use fp32 for incoherence processing tl.dot
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
                    self.code |= (max(value).bit_length() - 2) << self.CODE_POSITIONS["block_size"][0]
                    self.code |= (min(value) != 1) << self.CODE_POSITIONS["block_square"][0]
                    if value[0] != value[1]:
                        if (value[0] == 1) != (value[1] == 1):
                            raise ValueError("Block size axes must both be 1 or neither.")
                    reverse_block_axes = value[1] == 1
                elif key == "block_axes":
                    assert reverse_block_axes is not None
                    assert isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value)
                    value = tuple(reversed(value)) if reverse_block_axes else value
                    self.code |= (value[0] & self.CODE_POSITIONS["block_axis0"][1]) << self.CODE_POSITIONS["block_axis0"][0]
                    self.code |= (value[1] & self.CODE_POSITIONS["block_axis0"][1]) << self.CODE_POSITIONS["block_axis1"][0]
                else:
                    if key.count("_") == 2:  # account for LPConfig attrs that do double duty for linear and attention
                        key = "_".join(key.split("_")[1:])
                    if key not in self.CODE_POSITIONS:
                        raise ValueError(f"Invalid key: {key}")
                    type_table = self.SCALE_TYPES if key == "scale_type" else self.CODED_LP_TYPES
                    if value.lower() not in type_table:
                        raise ValueError(f"Invalid datatype: {value}")
                    self.code |= type_table.index(value.lower()) << self.CODE_POSITIONS[key][0]

    @staticmethod
    def decode(code: int) -> dict:
        """Decode the code into a dictionary of attributes."""

        def _get_value(code: int, key: str) -> int:
            start, mask = _LPCode.CODE_POSITIONS[key]
            return (code >> start) & mask

        attrs = {}
        if code > 0:
            attrs["block_axes"] = (_get_value(code, "block_axes1"), _get_value(code, "block_axes2"))
            block_size = 2 ** (2 + _get_value(code, "block_size"))
            attrs["block_size"] = (block_size, block_size) if _get_value(code, "block_square") else (1, block_size)
            for key, (start, mask) in _LPCode.CODE_POSITIONS.items():
                if key == "scale_type":
                    attrs[key] = _LPCode.SCALE_TYPES[(code >> start) & mask]
                elif key.startswith("icp"):
                    attrs[key] = bool((code >> start) & mask)
                elif not key.startswith("block"):
                    attrs[key] = _LPCode.CODED_LP_TYPES[(code >> start) & mask]
        return attrs


@dataclass
class LPConfig:
    """This describes what quantization options are to be used."""

    # the first three attributes, independently, can define the instance of the LPConfig
    code: int = 0  # 32-bit code for the quantization option
    json_path: Path | str = None  # path to a JSON file with the quantization options
    shortcut: str = None  # shortcut for predefined quantization option sets
    # the remaining attributes will be ignored if any of the previous three are passed as parameters
    block_size: tuple[int, int] = (0, 0)  # block size for scaling
    block_axes: tuple[int, int] = (0, 0)  # axes for scaling
    scale_dtype: str | int = None  # scale type can be a string or an int; "none" matches the scale type to the input
    q_dtype: str | int = None  # some native datatypes can be expressed as integers, the rest can be with strings
    k_dtype: str | int = None  # a string of "none" means do not quantize
    v_dtype: DataType = None
    p_dtype: DataType = None
    do_dtype: DataType = None
    ds_dtype: DataType = None
    icp_qk: bool = False
    icp_pv: bool = False
    icp_fp32: bool = False

    def __setattr__(self, name, value):
        if "dtype" in name and isinstance(value, str):
            value = DataType(name=value)
        if name in ("act_dtype", "q_dtype"):
            self.q_dtype = value
        elif name in ("weight_dtype", "k_dtype"):
            self.k_dtype = value
        elif hasattr(self, name):
            super().__setattr__(name, value)

    # fmt: off
    @property
    def square_block(self) -> bool: return self.block_size[0] == self.block_size[1]
    @property
    def weight_dtype(self) -> DataType: return self.k_dtype
    @property
    def act_dtype(self) -> tuple[int, int]: return self.q_dtype
    @property
    def need_imatrix(self) -> bool: return self.icp_qk or self.icp_pv
    @property
    def enabled(self) -> bool: return bool(self.code)
    # fmt: on

    def __str__(self) -> str:
        return f"LPConfig(code=0x{self.code:08x}, json_path={self.json_path}, shortcut={self.shortcut}, " \
               f"block_size={self.block_size}, block_axes={self.block_axes}, square={self.square_block}, " \
               f"scale_dtype={self.scale_dtype}, q_dtype={self.q_dtype}, k_dtype={self.k_dtype}, " \
               f"v_dtype={self.v_dtype}, p_dtype={self.p_dtype}, do_dtype={self.do_dtype}, ds_dtype={self.ds_dtype}, " \
               f"icp_qk={self.icp_qk}, icp_pv={self.icp_pv}, icp_fp32={self.icp_fp32})"

    def __post_init__(self):
        # 1) pass a code to the constructor, defining the instance from the code
        # 2) pass a json file to the constructor, defining the instance from the json file
        # 3) pass a shortcut to the constructor, which, if defined, will define the instance
        # 4) pass all attributes other than the code/json_path, and shortcut
        # 5) pass nothing to the constructor, but set a code in the environment
        # 6) pass nothing to the constructor, but set a json file in the environment
        # 7) pass nothing to the constructor, but set a shortcut in the environment
        # 8) pass nothing to the constructor, but set attributes in the environment
        env_code, env_json = self.getenv("code", None, int), self.getenv("json", None, str)
        code = self.code if self.code is not None else env_code if self.json_path is None else None
        json_path = Path(self.json_path) if self.json_path else Path(env_json) if env_json else None
        shortcut = self.getenv("shortcut", None, str)
        if code is not None:
            # methods 1 and 5
            attrs = _LPCode.decode(code)
        elif json_path:
            # methods 2 and 6
            if not json_path.exists():
                raise ValueError(f"json_path {json_path} does not exist.")
            with open(json_path) as f:
                attrs = json.load(f)
        elif shortcut is not None:
            # methods 3 and 7
            if shortcut not in LP_SHORTCUTS:
                raise ValueError(f"Invalid shortcut: {shortcut}")
            attrs = LP_SHORTCUTS[shortcut]
        else:
            # method 8
            attrs = {}
            for attr in (
                "scale_dtype",
                "act_dtype",
                "weight_dtype",
                "q_dtype",
                "k_dtype",
                "v_dtype",
                "p_dtype",
                "do_dtype",
                "ds_dtype",
            ):
                attrs[attr] = self.getenv(attr, "", str)
            for attr in ("icp_qk", "icp_pv"):
                attrs[attr] = self.getenv(attr, False, bool)
            attrs["block_size"] = self.getenv("block_size", "0, 0", tuple)
            attrs["block_axes"] = self.getenv("block_axes", "0, 0", tuple)
        # attrs exist here for all but method 4 (in which all params are params to __init__)
        for key, value in attrs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.code = self.to_code()

    def to_code(self) -> int:
        """Generate a code from the LPConfig attributes."""
        attrs = self.__dict__.copy()
        attrs.pop("code", None)
        attrs.pop("json_path", None)
        return _LPCode(**attrs).code

    def get_incoherence_matrix(self, size: int, dtype: DataType | str = "float32") -> torch.Tensor:
        """Return the incoherence matrix for the given datatype."""
        if not self.need_imatrix:
            raise ValueError("No incoherence matrix needed.")
        dtype = DataType(name="float32") if self.icp_fp32 else DataType(name=dtype) if isinstance(dtype, str) else dtype
        return self.get_imatrix(size, dtype, walsh=True, randomize=True)

    @classmethod
    def getenv(cls, x: str, dflt: str = "", dtype=str, lowprec=False):
        """Get an environment variable with optional default and type conversion."""
        x = x.upper()
        if lowprec and not x.startswith("LP_"):
            x = f"LP_{x}"
        elif not x.startswith("FLASH_ATTENTION_TRITON_AMD_"):
            x = f"FLASH_ATTENTION_TRITON_AMD_{x}"
        setting = os.getenv(x, dflt).lower()
        if dtype is bool:
            return setting in ("1", "true", "yes")
        if dtype is int and setting.replace("-", "").isdigit():
            return int(setting)
        if dtype is float and setting.replace(".", "").replace("-", "").isdigit():
            return float(setting)
        if dtype is tuple:
            return (int(i) for i in setting.split(","))
        if issubclass(dtype, Enum):
            return dtype(int(setting)) if setting.isdigit else dtype[setting.upper()]
        return setting

    @classmethod
    def randomize_imatrix(cls, imatrix: torch.Tensor) -> torch.Tensor:
        """Randomize a Walsh-Hadamard matrix while preserving orthogonality."""
        diag = torch.diag(torch.randint(0, 2, (imatrix.shape[0],), dtype=imatrix.dtype, device=imatrix.device) * 2 - 1)
        return diag @ imatrix

    @classmethod
    def get_imatrix(
        cls, size: int, dtype: torch.dtype = torch.float32, walsh: bool = True, randomize: bool = True
    ) -> torch.Tensor:
        def sign_changes(matrix):
            return [sum(int(matrix[j, i] != matrix[j, i + 1]) for i in range(size - 1)) for j in range(size)]

        imatrix = torch.tensor([[1, 1], [1, -1]], dtype=dtype).cuda()
        while imatrix.size(0) < size:
            imatrix = torch.kron(imatrix, torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=imatrix.device))
        imatrix /= torch.tensor(size, dtype=dtype, device=imatrix.device).sqrt()
        if walsh:
            changes = sign_changes(imatrix)
            order = torch.tensor(changes, dtype=imatrix.dtype, device=imatrix.device).argsort()
            imatrix = imatrix[order, :]
        if randomize:
            imatrix = cls.randomize_imatrix(imatrix)
        return imatrix


def test_lpconfig(logger: logging.Logger) -> int:
    """Test the LPConfig class."""
    logger.info(f"LPConfig has {len(LP_SHORTCUTS)} shortcuts.")
    # clear out the LP_ env vars
    for key in os.environ.copy():
        if key.startswith("LP_"):
            os.environ.pop(key)
    # 1) pass a code to the constructor
    cfg_param_code = LPConfig(code=0x00000000)

    # 2) pass a json file to the constructor
    cfg_param_json = LPConfig(json_path=Path(__file__).with_name("config.json"))

    # 3) pass a shortcut to the constructor
    cfg_param_shortcut = LPConfig(shortcut="split_e4m3fnuz_e5m2fnuz_icpqk32_32x32")

    # 4) pass everythomg to the constructor except for code, json_path, shortcut
    cfg_param_attrs = LPConfig(
        block_size=(32, 32), block_axes=(0, 1), scale_dtype="none",
        q_dtype="e4m3fnuz", k_dtype="e4m3fnuz", v_dtype="e4m3fnuz",
        p_dtype="e5m2fnuz", ds_dtype="e5m2fnuz", do_dtype="e5m2fnuz",
        icp_qk=True, icp_pv=False, icp_fp32=True
    )

    # 5) set a code env var and pass no parameters
    os.environ["LP_CODE"] = "0x00000000"
    cfg_env_code = LPConfig()
    os.environ.pop("LP_CODE")

    # 6) set a json path env var and pass no parameters
    os.environ["LP_JSON_PATH"] = str(Path(__file__).with_name("config.json"))
    cfg_env_json = LPConfig()
    os.environ.pop("LP_JSON_PATH")

    # 7) set a shortcut env var and pass no parameters
    os.environ["LP_SHORTCUT"] = "split_e4m3fnuz_e5m2fnuz_icpqk32_32x32"
    cfg_env_shortcut = LPConfig()
    os.environ.pop("LP_SHORTCUT")

    # 6) pass nothing to the constructor, but set attributes in the environment
    import subprocess
    command = f"source {str(Path(__file__).with_name('config.env'))} && env"
    process = subprocess.Popen(command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error(f"Error executing script: {stderr.decode()}")
        cfg_env_attrs = None
    else:
        env_vars = {}
        for line in stdout.decode().splitlines():
            if line.startswith("LP_") and "=" in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
        for key, value in env_vars.items():
            os.environ[key] = value
        cfg_env_attrs = LPConfig()

    comparisons, errors = 0, 0
    config_pairs = (
        ("cfg_param_code", cfg_param_code), # method 1
        ("cfg_param_json", cfg_param_json), # method 2
        ("cfg_param_shortcut", cfg_param_shortcut), # method 3
        ("cfg_param_attrs", cfg_param_attrs), # method 4
        ("cfg_env_code", cfg_env_code), # method 5
        ("cfg_env_json", cfg_env_json), # method 6
        ("cfg_env_shortcut", cfg_env_shortcut), # method 7
        ("cfg_env_attrs", cfg_env_attrs), # method 8
    )
    for lhs_name, lhs in config_pairs:
        for rhs_name, rhs in config_pairs:
            if lhs is not None and rhs is not None and lhs_name != rhs_name:
                comparisons += 1
                if lhs != rhs:
                    errors += 1
    logger.info(f"Compared {comparisons} configurations with {errors} errors.")
    return errors
