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

import numpy as np
import torch
import triton
import triton.language as tl

from . import snippets as lp
from .common import STD_DTYPES, CastMode, RoundMode, ScaleMode, get_enum
from .datatype import DataType
from .number import NumberSpec
from .utils import get_logger, kurtosis, make_outliers, triton_fp8_support

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
                for qtype in (("e4m3fn", "e4m3fnuz"),):
                    for ptype in (("e4m3fn", "e4m3fnuz"), ("e5m2", "e5m2fnuz")):
                        for qpidx in (0, 1):
                            for block in (8, 16, 32, 64, 128):
                                for square in (True,):
                                    for stype in (None, "float32"):
                                        sdict = {}
                                        all_or_split = "split" if qtype != ptype or icp_pv != icp_qk else "all"
                                        name = f"{all_or_split}_{'match' if stype is None else stype}_"
                                        qt, pt = qtype[qpidx], ptype[qpidx]
                                        name += f"{qt}_{pt}_" if qtype[qpidx] != ptype[qpidx] else f"{qt}_"
                                        sdict["icp_pv"] = icp_pv
                                        sdict["icp_qk"] = icp_qk
                                        sdict["icp_fp32"] = icp_fp32
                                        icp = "icp" if icp_pv and icp_qk else "icpqk" if icp_qk else "icppv" if icp_pv else ""
                                        if icp_fp32:
                                            icp += "32"
                                        if icp:
                                            name += icp + "_"
                                        name += f"{block}x{block}" if square else f"1x{block}"
                                        sdict["block_size"] = (block, block) if square else (1, block)
                                        sdict["block_axes"] = (0, 1)
                                        sdict["scale_dtype"] = stype
                                        for tensor in ("q", "k", "v"):
                                            sdict[f"{tensor}_dtype"] = qtype[qpidx]
                                        for tensor in ("p", "ds", "do"):
                                            sdict[f"{tensor}_dtype"] = ptype[qpidx]
                                        if name in lp_shortcuts:
                                            raise ValueError(f"Duplicate shortcut: {name}")
                                        lp_shortcuts[name] = sdict
    return lp_shortcuts


LP_SHORTCUTS: dict[str, dict] = _get_shortcuts()


class _LPCode:
    """Represents encoding/decoding of LP Config codes."""

    SCALE_TYPES = [None, "e8m23", "e5m10", "e8m7", "e8m0", "e5m3"]
    CODED_LP_TYPES = [None, "e5m2", "e5m2fnuz", "e4m3fn", "e4m3fnuz", "mxfp6e3", "mxfp6e2", "mxfp4e2"]
    CODE_POSITIONS = {
        "fp8_types": (int(lp.FP8_TYPE_POS), int(lp.FP8_TYPE_MASK)),
        "block_size": (int(lp.SIZE_POS), int(lp.SIZE_MASK)),
        "block_square": (int(lp.SQUARE_POS), int(lp.SQUARE_MASK)),
        "block_axis0": (int(lp.AXIS0_POS), int(lp.AXIS_MASK)),
        "block_axis1": (int(lp.AXIS1_POS), int(lp.AXIS_MASK)),
        # scale type is 3 bits
        "scale_dtype": (int(lp.SCALE_POS), int(lp.SCALE_MASK)),
        # datatypes are 3 bits each, 18 total bits
        "q_dtype": (int(lp.Q_POS), int(lp.TENSOR_MASK)),
        "k_dtype": (int(lp.K_POS), int(lp.TENSOR_MASK)),
        "v_dtype": (int(lp.V_POS), int(lp.TENSOR_MASK)),
        "p_dtype": (int(lp.P_POS), int(lp.TENSOR_MASK)),
        "do_dtype": (int(lp.DO_POS), int(lp.TENSOR_MASK)),
        "ds_dtype": (int(lp.DS_POS), int(lp.TENSOR_MASK)),
        "icp_qk": (int(lp.ICP_QK_POS), int(lp.ICP_MASK)),
        "icp_pv": (int(lp.ICP_PV_POS), int(lp.ICP_MASK)),
        "icp_fp32": (int(lp.ICP_PV_POS), int(lp.ICP_MASK)),  # use fp32 for incoherence processing tl.dot
        "roundmode": (int(lp.ROUNDMODE_POS), int(lp.ROUNDMODE_MASK)),
        "scalemode": (int(lp.SCALEMODE_POS), int(lp.SCALEMODE_MASK)),
        "castmode": (int(lp.CASTMODE_POS), int(lp.CASTMODE_MASK)),
    }

    def __init__(self, **kwargs):
        self.code = np.int64(0)
        if "code" in kwargs:
            self.code = kwargs["code"]
        elif not self.infer_zero_code(kwargs):
            reverse_block_axes = None
            for key, value in kwargs.items():
                if key == "fp8_types":
                    self.code |= triton_fp8_support() << self.CODE_POSITIONS[key][0]
                elif key == "block_axes":
                    pass
                elif key == "block_size":
                    # block size works like this: it can be either square (both dimensions 2, 4, 8, 16, 32, 64, 128, or 256)
                    # or vector (one dimension is 1 and the other is 2, 4, 8, 16, 32, 64, 128, or 256).
                    # The encoded block size is x such that 2**(x+1) is the block size.
                    # The block axes are the dims of the block that are reduced for the scale. If the block is not square,
                    # then the order of the axes (dims in PyTorch) is important, and we swap them so that the first axis is
                    # the one that has length 1.
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
                elif key.startswith("icp"):
                    self.code |= int(value) << self.CODE_POSITIONS[key][0]
                elif key.endswith("dtype"):
                    type_table = self.SCALE_TYPES if key == "scale_dtype" else self.CODED_LP_TYPES
                    value = (
                        NumberSpec.standard(value.name)
                        if isinstance(value, DataType)
                        else NumberSpec.standard(value)
                        if isinstance(value, str)
                        else value
                    )
                    if key not in self.CODE_POSITIONS:
                        raise ValueError(f"Invalid datatype key: {key}")
                    if value not in type_table:
                        raise ValueError(f"Invalid datatype value: {value}")
                    self.code |= type_table.index(value) << self.CODE_POSITIONS[key][0]
                elif key.endswith("mode"):
                    if isinstance(value, str):
                        value = get_enum({"roundmode": RoundMode, "scalemode": ScaleMode, "castmode": CastMode}[key], value)
                    if not isinstance(value, Enum):
                        raise ValueError(f"Invalid {key} value: {value}")
                    self.code |= value.value << self.CODE_POSITIONS[key][0]
                else:
                    raise ValueError(f"Invalid LPConfig attribute: {key}")

    @staticmethod
    def infer_zero_code(attrs: dict) -> bool:
        """Infer if the code is zero from the attributes."""
        for key, value in attrs.items():
            # don't check for modes
            if (
                key.startswith("icp")
                and value
                or key.startswith("block")
                and value != (0, 0)
                or key.endswith("dtype")
                and value is not None
            ):
                return False
        return True

    @staticmethod
    def decode(code: int) -> dict:
        """Decode the code into a dictionary of attributes."""

        def _get_value(code: int, key: str) -> int:
            start, mask = _LPCode.CODE_POSITIONS[key]
            return (code >> start) & mask

        attrs = {}
        if code > 0:
            # block_axes is disabled in the code
            # attrs["block_axes"] = (_get_value(code, "block_axis0"), _get_value(code, "block_axis1"))
            # instead, we use those bits to determine fp8 support, so we can't reconstruct axes from the code
            block_size = 2 ** (1 + _get_value(code, "block_size"))
            attrs["block_size"] = (block_size, block_size) if _get_value(code, "block_square") else (1, block_size)
            for key, (start, mask) in _LPCode.CODE_POSITIONS.items():
                if key == "scale_dtype":
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
    code: np.int64 = None  # 64-bit code for the quantization option
    json_path: Path | str = None  # path to a JSON file with the quantization options
    shortcut: str = None  # shortcut for predefined quantization option sets
    # the remaining attributes will be ignored if any of the previous three are passed as parameters
    block_size: tuple[int, int] = (0, 0)  # block size for scaling (00 unscaled, 01 channel scaled, 11 tensor scaled)
    block_axes: tuple[int, int] = (0, 0)  # axes for scaling
    scale_dtype: DataType = None
    q_dtype: DataType = None
    k_dtype: DataType = None
    v_dtype: DataType = None
    p_dtype: DataType = None
    do_dtype: DataType = None
    ds_dtype: DataType = None
    icp_qk: bool = False
    icp_pv: bool = False
    icp_fp32: bool = False
    scalemode: ScaleMode | str = "floor"
    roundmode: RoundMode | str = "even"
    castmode: CastMode | str = "virtual"

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
    @property
    def xcode(self) -> bool: return f"0x{self.code:010x}"
    # fmt: on

    def __repr__(self) -> str:
        return "\n".join(
            [
                "LPConfig(",
                f"  code=0x{self.code:010x}, json_path={self.json_path}, shortcut={self.shortcut}, ",
                f"  block_size={self.block_size}, block_axes={self.block_axes}, square={self.square_block}, ",
                f"  scale_dtype={self.scale_dtype}, q_dtype={self.q_dtype}, k_dtype={self.k_dtype}, ",
                f"  v_dtype={self.v_dtype}, p_dtype={self.p_dtype}, do_dtype={self.do_dtype}, ds_dtype={self.ds_dtype}, ",
                f"  icp_qk={self.icp_qk}, icp_pv={self.icp_pv}, icp_fp32={self.icp_fp32}, ",
                f"  scalemode={self.scalemode}, roundmode={self.roundmode}, castmode={self.castmode}",
                ")",
            ]
        )

    def short_repr(self, tensors="qkvpos") -> str:
        """Return a short representation of the LPConfig."""
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

    def __post_init__(self):
        # 1) pass a code to the constructor, defining the instance from the code
        # 2) pass a json file to the constructor, defining the instance from the json file
        # 3) pass a shortcut to the constructor, which, if defined, will define the instance
        # 4) pass all attributes other than the code/json_path, and shortcut
        # 5) pass nothing to the constructor, but set a code in the environment
        # 6) pass nothing to the constructor, but set a json file in the environment
        # 7) pass nothing to the constructor, but set a shortcut in the environment
        # 8) pass nothing to the constructor, but set attributes in the environment
        env_code, env_json = self.getenv("code", None, int), self.getenv("json_path", None, str)
        code = self.code if self.code is not None else env_code if self.json_path is None else None
        json_path = Path(self.json_path) if self.json_path else Path(env_json) if env_json else None
        shortcut = self.shortcut if self.shortcut is not None else self.getenv("shortcut", None, str)
        if code is not None:
            # methods 1 and 5
            attrs = _LPCode.decode(code)
        elif json_path:
            # methods 2 and 6
            if not json_path.exists():
                raise ValueError(f"json_path {json_path} does not exist.")
            with open(json_path) as f:
                attrs = json.load(f)
                # json files can have lists, but we want tuples
                for key, value in attrs.copy().items():
                    if isinstance(value, list):
                        attrs[key] = tuple(value)
        elif shortcut is not None:
            # methods 3 and 7
            if shortcut not in LP_SHORTCUTS:
                raise ValueError(f"Invalid shortcut: {shortcut}")
            attrs = LP_SHORTCUTS[shortcut]
            self.shortcut = None
        elif _LPCode.infer_zero_code(self.__dict__):
            # method 8
            attrs = {}
            for attr in (
                "scale_dtype",
                "q_dtype",
                "k_dtype",
                "v_dtype",
                "p_dtype",
                "do_dtype",
                "ds_dtype",
                "act_dtype",
                "weight_dtype",
            ):
                if attr in ("act_dtype", "weight_dtype"):
                    value = self.getenv(attr, None, str)
                    if value:
                        actual_attr = "q_dtype" if attr == "act_dtype" else "k_dtype"
                        actual_value = attrs[actual_attr]
                        if value == actual_value:
                            continue
                        elif actual_value:
                            raise ValueError(f"Conflict between {attr}: {value} and {actual_attr}: {actual_value}")
                        attrs[actual_attr] = value
                else:
                    attrs[attr] = self.getenv(attr, None, str)
            for attr in ("icp_qk", "icp_pv", "icp_fp32"):
                attrs[attr] = self.getenv(attr, False, bool)
            for attrs in ("scalemode", "roundmode", "castmode"):
                attrs[attr] = self.getenv(attr, 0, str)
            attrs["block_size"] = self.getenv("block_size", "0, 0", tuple)
            attrs["block_axes"] = self.getenv("block_axes", "0, 0", tuple)
        else:
            # method 4
            for key, value in self.__dict__.copy().items():
                if key.endswith("_dtype") and isinstance(value, str):
                    setattr(self, key, DataType(name=value))
                if key.endswith("mode"):
                    setattr(self, key,
                        get_enum({"roundmode": RoundMode, "scalemode": ScaleMode, "castmode": CastMode}[key], value)
                    )
            self.code = self.to_code()
            return
        # attrs exist here for all but method 4 (in which all params are params to __init__)
        for key, value in attrs.items():
            if "_dtype" in key:
                if isinstance(value, str):
                    setattr(self, key, DataType(name=value))
                else:
                    setattr(self, key, None)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid LPConfig attribute: {key}")
        self.code = self.to_code()

    def to_code(self) -> np.int64:
        """Generate a code from the LPConfig attributes."""
        attrs = self.__dict__.copy()
        attrs.pop("code", None)
        attrs.pop("json_path", None)
        attrs.pop("shortcut", None)
        encoder = _LPCode(**attrs)
        return encoder.code

    def lp_code(self) -> triton.language.constexpr:
        return tl.constexpr(self.code)

    def dtype_from_index(self, index: int) -> DataType:
        """Return the datatype for the given index."""
        return (self.q_dtype, self.k_dtype, self.v_dtype, self.p_dtype, self.ds_dtype, self.do_dtype)[index]

    def make_quant_and_scale(self, tensor: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and scale the tensor."""
        # This should not be called if configuration is not enabled
        if not self.enabled:
            raise ValueError("LPConfig is not enabled.")
        # If the scale type is None, use the type of the tensor.
        sdtype = self.scale_dtype.nspec.torch_dtype if self.scale_dtype is not None else tensor.dtype
        assert sdtype is not None and sdtype.is_floating_point
        # Next, the quantization.  If the type is None, return the input tensor and a scale of 1
        qdtype = self.dtype_from_index(index)
        needs_icp = (
            self.icp_qk
            and index in (lp.Q_INDEX, lp.K_INDEX, lp.DS_INDEX)
            or self.icp_pv
            and index in (lp.V_INDEX, lp.P_INDEX, lp.DO_INDEX)
        )
        if qdtype is None:
            if needs_icp:
                quant, scale = torch.empty_like(tensor), torch.tensor([1.0], dtype=sdtype, device=tensor.device)
            else:
                quant, scale = tensor, torch.tensor([1.0], dtype=sdtype, device=tensor.device)
        else:
            shape = tuple([triton.cdiv(tensor.size(self.block_axes[s]), self.block_size[s]) for s in (0, 1)])
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
        return self.get_imatrix(size, dtype, walsh=walsh, randomize=randomize)

    @classmethod
    def getenv(cls, x: str, dflt: str = "", dtype=str, lowprec=True):
        """Get an environment variable with optional default and type conversion."""
        x = x.upper()
        if lowprec and not x.startswith("LP_"):
            x = f"LP_{x}"
        elif lowprec is False and not x.startswith("FLASH_ATTENTION_TRITON_AMD_"):
            x = f"FLASH_ATTENTION_TRITON_AMD_{x}"
        setting = os.getenv(x, dflt)
        if setting is None:
            return None
        if isinstance(setting, str):
            setting = setting.lower()
        if dtype is bool:
            return setting in ("1", "true", "yes")
        if dtype is int:
            try:
                if setting.startswith("0x"):
                    return int(setting, 16)
                elif setting.replace("-", "").isdigit():
                    return int(setting)
            except ValueError as err:
                raise ValueError(f"Invalid integer value for {x}: {setting}") from err
        if dtype is float and setting.replace(".", "").replace("-", "").isdigit():
            return float(setting)
        if dtype is tuple:
            return tuple(int(i) for i in setting.split(","))
        if issubclass(dtype, Enum):
            return dtype(int(setting)) if setting.isdigit() else dtype[setting.upper()]
        return setting

    @classmethod
    def randomize_imatrix(cls, imatrix: torch.Tensor) -> torch.Tensor:
        """Randomize a Walsh-Hadamard matrix while preserving orthogonality."""
        diag = torch.diag(torch.randint(0, 2, (imatrix.shape[0],), dtype=imatrix.dtype, device=imatrix.device) * 2 - 1)
        return diag @ imatrix

    @classmethod
    def create_imatrix(
        cls,
        size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        walsh: bool = True,
        randomize: bool = True,
    ) -> torch.Tensor:
        def sign_changes(matrix):
            """Count the number of changes in sign across a row."""
            return [sum(int(matrix[j, i] != matrix[j, i + 1]) for i in range(size - 1)) for j in range(size)]

        # create the Hadamard matrix
        # the Hadamard matrix is defined recursively as H(2n) = H(n) ⊗ [[1, 1], [1, -1]]
        # where ⊗ is the Kronecker product

        imatrix = torch.tensor([[1, 1], [1, -1]], dtype=dtype).cuda()
        device = imatrix.device
        while imatrix.size(0) < size:
            imatrix = torch.kron(imatrix, torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device))

        # scale the Hadamard matrix
        imatrix /= torch.tensor(size, dtype=dtype, device=device).sqrt()

        if walsh:
            # convert to Walsh-Hadamard matrix, ordering the rows ascending by the number of
            # sign changes per row
            changes = sign_changes(imatrix)
            order = torch.tensor(changes, dtype=dtype, device=device).argsort()
            imatrix = imatrix[order, :]

        if randomize:
            # randomize the Hadamard or Walsh-Hadamard matrix, by multiplying
            # a random diagonal matrix of 1 and -1 by the matrix
            diag = torch.diag(torch.randint(0, 2, (imatrix.shape[0],), dtype=dtype, device=device) * 2 - 1)
            imatrix = diag @ imatrix
        return imatrix.requires_grad_(False)

        # imatrix = cls.randomize_imatrix(imatrix) <--- put this in the fcn above

    ICP_MATRICES = {}

    @classmethod
    def get_imatrix(
        cls,
        size: int,
        torch_dtype: str | torch.dtype = torch.float32,
        device: torch.device = None,
        walsh: bool = True,
        randomize: bool = False,
    ) -> torch.Tensor:
        """Get, create, or randomize the incoherence matrix for the given size and dtype."""

        def get_key(size: int, torch_dtype: str | torch.dtype, walsh: bool, randomize: bool) -> str:
            key = f"{str(torch_dtype)[6:]}_{size}_"
            if randomize:
                key += "R"
            if walsh:
                key += "W"
            key += "H"
            return key

        if not isinstance(torch_dtype, torch.dtype):
            torch_dtype = DataType(name=torch_dtype).torch_dtype
        if torch_dtype not in STD_DTYPES:
            raise ValueError(f"Invalid imatrix dtype (must be float32, float16, or bfloat16): {torch_dtype}")
        if not any(size == 2**x for x in range(2, 8)):
            raise ValueError(f"Invalid size for incoherence matrix: {size} (must power of 2 in [8, 128]")
        key = get_key(size, torch_dtype, walsh, randomize)
        if randomize:
            if key in cls.ICP_MATRICES:
                # re-randomize but don't replace (might have implications on testing assumptions)
                return cls.randomize_imatrix(cls.ICP_MATRICES[key])
            nr_key = get_key(size, torch_dtype, walsh, False)
            if nr_key in cls.ICP_MATRICES:
                # there is a non-randomized version in the cache so randomize it and store it under the new key
                cls.ICP_MATRICES[key] = cls.randomize_imatrix(cls.ICP_MATRICES[nr_key])
            else:
                # it doesn't, so create a new matrix
                cls.ICP_MATRICES[key] = cls.create_imatrix(size, torch_dtype, device, walsh=walsh, randomize=randomize)
        # non-randomized case: create one if it doesn't exist
        elif key not in cls.ICP_MATRICES:
            cls.ICP_MATRICES[key] = cls.create_imatrix(size, torch_dtype, device, walsh=walsh, randomize=randomize)
        if walsh:
            cls.check_walsh(cls.ICP_MATRICES[key])
        else:
            cls.check_hadamard(cls.ICP_MATRICES[key])
        return cls.ICP_MATRICES[key]

    @classmethod
    def check_hadamard(cls, matrix):
        """Check if a matrix is a Hadamard matrix."""
        # Ensure the matrix is scaled correctly, i.e. +- a single scaled value
        assert (matrix.abs() == matrix[0, 0].abs()).all(), "Matrix absolute values are not all the same"
        matrix = matrix.sign()
        # Check that all entries are either +1 or -1
        assert torch.all((matrix == 1) | (matrix == -1)), "Matrix contains elements other than +1 and -1"
        # Check orthogonality: H * H^T = n * I
        size = matrix.size(0)
        identity_matrix = size * torch.eye(size, dtype=matrix.dtype, device=matrix.device)
        torch.testing.assert_close(matrix @ matrix.t(), identity_matrix)

    @classmethod
    def check_walsh(cls, matrix):
        """Check if a matrix is a Walsh matrix."""
        cls.check_hadamard(matrix)
        size = matrix.size(0)
        changes = [sum(int(matrix[j, i] != matrix[j, i + 1]) for i in range(size - 1)) for j in range(size)]
        # check if the number of sign changes is in increasing order by row
        order = torch.tensor(changes, dtype=matrix.dtype, device=matrix.device).argsort()
        reorder = torch.argsort(order)
        torch.testing.assert_close(order, reorder)

    @classmethod
    def check_icp(
        cls,
        logger: logging.Logger,
        torch_dtype,
        size,
        odim,
        walsh,
        randomize,
        float32,
        outlier_scale,
        outlier_range,
        outlier_prob,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check effects of transforms."""
        M: torch.Tensor = LPConfig.get_imatrix(size, torch_dtype, walsh=walsh, randomize=randomize)
        A = torch.randn(size * odim, size, dtype=torch_dtype, device=M.device)
        B = torch.randn(size * odim, size, dtype=torch_dtype, device=M.device)
        if outlier_prob > 0.0:
            A = make_outliers(A, scale=outlier_scale, range=outlier_range, prob=outlier_prob)
            B = make_outliers(B, scale=outlier_scale, range=outlier_range, prob=outlier_prob)
        if float32:
            M, A, B = M.float(), A.float(), B.float()
        A_kurtosis = kurtosis(A)
        B_kurtosis = kurtosis(B)
        AB_ref = (A @ B.t()).to(torch_dtype)
        AB_ref_norm = torch.norm(AB_ref).item()
        AM = (A @ M).to(torch_dtype)
        BM = (B @ M).to(torch_dtype)
        AM_kurtosis = kurtosis(AM)
        BM_kurtosis = kurtosis(BM)
        AB_icp = (AM @ BM.t()).to(torch_dtype)
        AB_icp_norm = torch.norm(AB_icp).item()
        diff_norm = torch.norm(AB_ref - AB_icp).item() / AB_ref_norm
        args = f"{size*odim}x{size} {str(torch_dtype)[6:]} "
        args += "RWH" if walsh and randomize else "WH" if walsh else "H"
        args += " F32MM" if float32 else ""
        if outlier_prob > 0.0:
            args += f" O{outlier_scale}R{outlier_range}P{outlier_prob}"
        logger.info(f"test_icp({args}):")
        logger.info(f"\tnorms:    diff_norm={diff_norm} ref_norm={AB_ref_norm} icp_norm={AB_icp_norm}")
        logger.info(f"\tkurtosis: A={A_kurtosis} AM={AM_kurtosis} B={B_kurtosis} BM={BM_kurtosis}")
        return AB_ref, AB_icp

    @classmethod
    def _method_param_code(cls):
        return LPConfig(code=0xA929204C)

    @classmethod
    def _method_param_json(cls):
        json_path = Path(__file__).parent / "tests" / "config_attrs.json"
        return LPConfig(json_path=json_path)

    @classmethod
    def _method_param_shortcut(cls):
        return LPConfig(shortcut="split_match_e4m3fnuz_e5m2fnuz_icpqk32_32x32")

    @classmethod
    def _method_param_attrs(cls):
        return LPConfig(
            block_size=(32, 32),
            block_axes=(0, 1),
            scale_dtype=None,
            q_dtype="e4m3fnuz",
            k_dtype="e4m3fnuz",
            v_dtype="e4m3fnuz",
            p_dtype="e5m2fnuz",
            ds_dtype="e5m2fnuz",
            do_dtype="e5m2fnuz",
            icp_qk=True,
            icp_pv=False,
            icp_fp32=True,
        )

    @classmethod
    def _method_env_code(cls):
        os.environ["LP_CODE"] = "0xa929204c"
        return LPConfig()

    @classmethod
    def _method_env_json(cls):
        json_path = Path(__file__).parent / "tests" / "config_attrs.json"
        os.environ["LP_JSON_PATH"] = str(json_path)
        return LPConfig()

    @classmethod
    def _method_env_shortcut(cls):
        os.environ["LP_SHORTCUT"] = "split_match_e4m3fnuz_e5m2fnuz_icpqk32_32x32"
        return LPConfig()

    @classmethod
    def _method_env_attrs(cls):
        import subprocess

        bash_path = Path(__file__).parent / "tests" / "config_attrs.sh"
        command = f"source {str(bash_path)} && env"
        process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Error executing script: {stderr.decode()}")
            return None
        env_vars = {}
        for line in stdout.decode().splitlines():
            if line.startswith("LP_") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value
        for key, value in env_vars.items():
            os.environ[key] = value
        return LPConfig()

    @classmethod
    def methods(cls):
        """Return the methods for configuring the LPConfig class."""
        return [
            LPConfig._method_param_code,
            LPConfig._method_param_json,
            LPConfig._method_param_shortcut,
            LPConfig._method_param_attrs,
            LPConfig._method_env_code,
            LPConfig._method_env_json,
            LPConfig._method_env_shortcut,
            LPConfig._method_env_attrs,
        ]

    @classmethod
    def check_config(cls, logger, index1: int, index2: int, print_shortcuts: bool = False) -> bool:
        """Test the LPConfig class for a pair of methods for configuring the class."""

        def _clear_env():
            for key in os.environ.copy():
                if key.startswith("LP_"):
                    os.environ.pop(key)

        # check for valid indices
        all_methods = cls.methods()
        num_methods = len(all_methods)
        if index1 == index2 or index1 < 0 or index2 < 0 or index1 >= num_methods or index2 >= num_methods:
            logger.error(f"Invalid indices: {index1}, {index2}")
            return False
        # sort the indices
        index1, index2 = min(index1, index2), max(index1, index2)
        # shortcut info
        if print_shortcuts:
            logger.info(f"LPConfig has {len(LP_SHORTCUTS)} shortcuts.")
            for shortcut, attrs in LP_SHORTCUTS.items():
                tmp_cfg = LPConfig(**attrs)
                logger.debug(f"Shortcut {shortcut} has code {tmp_cfg.xcode}")
        # run the methods
        cfgs = []
        for method in [all_methods[index1], all_methods[index2]]:
            _clear_env()
            cfgs.append(method())
        cfg1, cfg2 = cfgs[:2]
        matches = cfg1 == cfg2
        logger.info(f"code {index1} {cfg1.xcode} code ({index2}) {cfg1.xcode} {'PASS' if matches else 'FAIL'}")
        return matches
