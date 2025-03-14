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
from .utils import get_logger, kurtosis, make_outliers

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
                    for ptype in (("float8_e4m3fn", "float8_e4m3fnuz"), ("float8_e5m2", "float8_e5m2fnuz")):
                        for qpidx in (0, 1):
                            for block in (8, 16, 32, 64, 128):
                                for square in (True,):
                                    for stype in (None, "float32"):
                                        sdict = {}
                                        all_or_split = "split" if qtype != ptype or icp_pv != icp_qk else "all"
                                        name = f"{all_or_split}_{'match' if stype is None else stype}_"
                                        qt, pt = qtype[qpidx].removeprefix("float8_"), ptype[qpidx].removeprefix("float8_")
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

    SCALE_TYPES = [None, "float32", "float16", "bfloat16", "e8m0", "e5m3"]
    CODED_LP_TYPES = [
        None,
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
        "scale_dtype": (LP_SCALE_POS, LP_SCALE_MASK),
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
        self.code = 0
        if "code" in kwargs:
            self.code = kwargs["code"]
        elif not self.infer_zero_code(kwargs):
            reverse_block_axes = None
            for key, value in kwargs.items():
                if key == "block_size":
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
                    self.code |= int(value)<< self.CODE_POSITIONS[key][0]
                elif key.endswith("dtype"):
                    type_table = self.SCALE_TYPES if key == "scale_dtype" else self.CODED_LP_TYPES
                    value = value.name if isinstance(value, DataType) else value
                    if key not in self.CODE_POSITIONS:
                        raise ValueError(f"Invalid datatype key: {key}")
                    if value not in type_table:
                        raise ValueError(f"Invalid datatype value: {value}")
                    self.code |= type_table.index(value) << self.CODE_POSITIONS[key][0]
                else:
                    raise ValueError(f"Invalid LPConfig attribute: {key}")

    @staticmethod
    def infer_zero_code(attrs: dict) -> bool:
        """Infer if the code is zero from the attributes."""
        for key, value in attrs.items():
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
            attrs["block_axes"] = (_get_value(code, "block_axis0"), _get_value(code, "block_axis1"))
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
    code: int = None  # 32-bit code for the quantization option
    json_path: Path | str = None  # path to a JSON file with the quantization options
    shortcut: str = None  # shortcut for predefined quantization option sets
    # the remaining attributes will be ignored if any of the previous three are passed as parameters
    block_size: tuple[int, int] = (0, 0)  # block size for scaling
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

    # fmt: off
    @property
    def square_block(self) -> bool: return self.block_size[0] == self.block_size[1]
    @property
    def weight_dtype(self) -> DataType: return self.k_dtype
    @property
    def act_dtype(self) -> DataType: return self.q_dtype
    @property
    def need_imatrix(self) -> bool: return self.icp_qk or self.icp_pv
    @property
    def enabled(self) -> bool: return bool(self.code)
    @property
    def xcode(self) -> bool: return f"0x{self.code:08x}"
    # fmt: on

    def __repr__(self) -> str:
        return "\n".join([
            "LPConfig(",
            f"  code=0x{self.code:08x}, json_path={self.json_path}, shortcut={self.shortcut}, ",
            f"  block_size={self.block_size}, block_axes={self.block_axes}, square={self.square_block}, ",
            f"  scale_dtype={self.scale_dtype}, q_dtype={self.q_dtype}, k_dtype={self.k_dtype}, ",
            f"  v_dtype={self.v_dtype}, p_dtype={self.p_dtype}, do_dtype={self.do_dtype}, ds_dtype={self.ds_dtype}, ",
            f"  icp_qk={self.icp_qk}, icp_pv={self.icp_pv}, icp_fp32={self.icp_fp32}",
            ")"
        ])

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
            attrs["block_size"] = self.getenv("block_size", "0, 0", tuple)
            attrs["block_axes"] = self.getenv("block_axes", "0, 0", tuple)
        else:
            # method 4
            for key, value in self.__dict__.copy().items():
                if key.endswith("_dtype") and isinstance(value, str):
                    setattr(self, key, DataType(name=value))
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

    def to_code(self) -> int:
        """Generate a code from the LPConfig attributes."""
        attrs = self.__dict__.copy()
        attrs.pop("code", None)
        attrs.pop("json_path", None)
        attrs.pop("shortcut", None)
        encoder = _LPCode(**attrs)
        return encoder.code

    def get_incoherence_matrix(self, size: int, dtype: DataType | str = "float32") -> torch.Tensor:
        """Return the incoherence matrix for the given datatype."""
        if not self.need_imatrix:
            raise ValueError("No incoherence matrix needed.")
        dtype = DataType(name="float32") if self.icp_fp32 else DataType(name=dtype) if isinstance(dtype, str) else dtype
        return self.get_imatrix(size, dtype, walsh=True, randomize=True)

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


def test_config(logger: logging.Logger) -> int:
    """Test the LPConfig class."""
    logger.info(f"LPConfig has {len(LP_SHORTCUTS)} shortcuts.")
    for shortcut, attrs in LP_SHORTCUTS.items():
        tmp_cfg = LPConfig(**attrs)
        logger.info(f"Shortcut {shortcut} has code {tmp_cfg.xcode}")
    # clear out the LP_ env vars
    for key in os.environ.copy():
        if key.startswith("LP_"):
            os.environ.pop(key)
    # 1) pass a code to the constructor
    cfg_param_code = LPConfig(code=0xa929204c) # 0xa929214c
    logger.warning(f"cfg_param_code.code = {hex(cfg_param_code.code)}")
    # 2) pass a json file to the constructor
    json_path = Path(__file__).parent / "tests" / "config_attrs.json"
    cfg_param_json = LPConfig(json_path=json_path)
    logger.info(f"cfg_param_json.code = {hex(cfg_param_json.code)}")
    # 3) pass a shortcut to the constructor
    cfg_param_shortcut = LPConfig(shortcut="split_match_e4m3fnuz_e5m2fnuz_icpqk32_32x32")
    logger.info(f"cfg_param_shortcut.code = {hex(cfg_param_shortcut.code)}")
    # 4) pass everythomg to the constructor except for code, json_path, shortcut
    cfg_param_attrs = LPConfig(
        block_size=(32, 32),
        block_axes=(0, 1),
        scale_dtype=None,
        q_dtype="float8_e4m3fnuz",
        k_dtype="float8_e4m3fnuz",
        v_dtype="float8_e4m3fnuz",
        p_dtype="float8_e5m2fnuz",
        ds_dtype="float8_e5m2fnuz",
        do_dtype="float8_e5m2fnuz",
        icp_qk=True,
        icp_pv=False,
        icp_fp32=True,
    )
    # 5) set a code env var and pass no parameters
    os.environ["LP_CODE"] = "0xa929204c"
    cfg_env_code = LPConfig()
    logger.info(f"cfg_env_code.code = {hex(cfg_env_code.code)}")
    os.environ.pop("LP_CODE")
    # 6) set a json path env var and pass no parameters
    json_path = Path(__file__).parent / "tests" / "config_attrs.json"
    assert json_path.exists(), f"No test json file {str(json_path)} exists."
    os.environ["LP_JSON_PATH"] = str(json_path)
    cfg_env_json = LPConfig()
    logger.info(f"cfg_env_json.code = {hex(cfg_env_json.code)}")
    os.environ.pop("LP_JSON_PATH")
    # 7) set a shortcut env var and pass no parameters
    os.environ["LP_SHORTCUT"] = "split_match_e4m3fnuz_e5m2fnuz_icpqk32_32x32"
    cfg_env_shortcut = LPConfig()
    logger.info(f"cfg_env_shortcut.code = {hex(cfg_env_shortcut.code)}")
    os.environ.pop("LP_SHORTCUT")
    # 8) pass nothing to the constructor, but set attributes in the environment
    import subprocess

    bash_path = Path(__file__).parent / "tests" / "config_attrs.sh"
    assert bash_path.exists(), f"No test bash file {str(bash_path)} exists."
    command = f"source {str(bash_path)} && env"
    process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error(f"Error executing script: {stderr.decode()}")
        cfg_env_attrs = None
    else:
        env_vars = {}
        for line in stdout.decode().splitlines():
            if line.startswith("LP_") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value
        for key, value in env_vars.items():
            os.environ[key] = value
        cfg_env_attrs = LPConfig()
        logger.info(f"cfg_env_attrs.code = {hex(cfg_env_attrs.code)}")

    errors = 0
    config_pairs = (
        ("cfg_param_code", cfg_param_code),  # method 1
        ("cfg_param_json", cfg_param_json),  # method 2
        ("cfg_param_shortcut", cfg_param_shortcut),  # method 3
        ("cfg_param_attrs", cfg_param_attrs),  # method 4
        ("cfg_env_code", cfg_env_code),  # method 5
        ("cfg_env_json", cfg_env_json),  # method 6
        ("cfg_env_shortcut", cfg_env_shortcut),  # method 7
        ("cfg_env_attrs", cfg_env_attrs),  # method 8
    )
    for lhs_name, lhs in config_pairs:
        for rhs_name, rhs in config_pairs:
            if lhs is not None and rhs is not None and lhs_name != rhs_name:
                if lhs != rhs:
                    for attr in lhs.__dict__:
                        lhs_attr, rhs_attr = getattr(lhs, attr), getattr(rhs, attr)
                        if lhs_attr != rhs_attr:
                            logger.error(f"{lhs_name}.{attr} ({lhs_attr}) != {rhs_name}.{attr} ({rhs_attr})")
                    errors += 1
    logger.info(f"Compared {len(config_pairs)} configurations with {errors} errors.")
    return errors


def test_icp(
    logger: logging.Logger, torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob
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
