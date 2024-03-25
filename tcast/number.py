"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/number.py: number format specification

from dataclasses import dataclass
import re
from typing import Literal

import torch

from .utils import is_float8_available, is_float8_fnuz_available

InfNan = Literal["ieee", "fn", "fnuz"]

MX2NUMSPEC = dict(
    mxfp8e5="e5m2",
    mxfp8e4="e4m3fn",
    mxfp6e3="e3m2fnuz",
    mxfp6e2="e2m3fnuz",
    mxfp4e2="e2m1fnuz",
    mxint8="int8 or e1m6fnuz",
    mxint4="int4 or e1m2fnuz",
    bfp16="int8 or e1m6fnuz",
)

MXUNSUPPORTED = ("mx9", "mx6", "mx3")


@dataclass
class NumberSpec:
    """Specification for an unscaled number format."""

    _name: str = None
    bits: int = None
    ebits: int = None
    mbits: int = None
    bias: int = None
    signed: bool = True
    infnan: InfNan = "ieee"
    is_float: bool = False
    is_int: bool = False
    is_uint: bool = False
    is_exponent: bool = False
    emax: int = None
    emin: int = None
    maxfloat: float = None
    smallest_normal: float = None
    smallest_subnormal: float = None
    eps: float = None
    midmax: float = None
    maxint: int = None
    minint: int = None
    torch_dtype: torch.dtype = None

    lookup_table: list[list[float]] = None
    lookup_name: str = None
    midpoints: list[list[float]] = None
    mapnames: list[str] = None
    index_bits: int = None
    lookup_bits: int = None
    _number_line: list[float] = None

    def __init__(self, code: str | torch.dtype):
        self._decode(code)
        self._check()

    @property
    def name(self):
        """NumberSpec name or both lookup name and number name."""
        if self.is_lookup:
            return self.lookup_name + "_" + self._name
        return self._name

    @property
    def max(self) -> int | float:
        """Returns maxfloat for floats, maxint for integers."""
        return self.maxfloat if self.is_float else self.maxint

    @property
    def min(self) -> int | float:
        """Returns -maxfloat for floats, minint for integers."""
        return -self.maxfloat if self.is_float else self.minint

    @property
    def tiny(self) -> int | float:
        """Returns smallest_normal, as in torch.finfo."""
        return self.smallest_normal

    @property
    def number_line(self) -> list[float] | None:
        return self.get_number_line()

    @property
    def lookup(self) -> list[list[float]]:
        return self.lookup_table

    @property
    def is_lookup(self) -> bool:
        return self.lookup_table is not None

    @property
    def num_mappings(self) -> int:
        return 2**self.lookup_bits if self.is_lookup else 0

    @property
    def current_mappings(self) -> int:
        return len(self.lookup_table) if self.is_lookup else 0

    @property
    def num_values(self) -> int:
        return 2**self.index_bits if self.is_lookup else 0

    def get_number_line(self) -> list[float]:
        """All possible values for this number specification."""
        if not self._number_line:
            if self.bits > 8:
                raise ValueError(f"NumberSpec: too many number line values for {self.bits} bits")
            if not (self.is_float or self.ebits == 1):
                raise ValueError("NumberSpec: number line must be for float or float-like numbers.")
            # get the non-negative numbers then mirror for negatives, giving all 2^bits values, including 2 zeros
            line = [i * self.smallest_subnormal for i in range(2**self.mbits)]  # subnormals
            for e in range(self.emax - self.emin + 1):
                line += [(self.smallest_normal + i * self.smallest_subnormal) * 2**e for i in range(2**self.mbits)]
            self._number_line = [-v for v in reversed(line)] + line
        return self._number_line

    def add_mapping(self, mapping: list[float], mapname: str):
        """Add a new lookup."""
        if not self.is_lookup:
            raise RuntimeError("NumberSpec.add_mapping called for non-lookup number spec.")
        if self.current_mappings == self.num_mappings:
            raise RuntimeError(f"NumberSpec.add_mapping exceeds number of mappings specified ({self.num_mappings}).")
        for v in mapping:
            if v not in self.number_line:
                raise ValueError(f"NumberSpec.add_mapping: mapping value {v} is not representable in {self.name}")
        self.lookup_table.append(mapping)
        self.midpoints.append([(mapping[i] + mapping[i + 1]) / 2.0 for i in range(len(mapping) - 1)])
        self.mapnames.append(mapname)

    def get_mapping(
        self, index: int = None, torch_dtype: torch.dtype = torch.float32, device: torch.device = "cuda"
    ) -> torch.Tensor:
        """Return one or all of the lookups as a tensor."""
        if not self.is_lookup:
            raise RuntimeError("NumberSpec.get_mapping called for non-lookup number spec.")
        if index is not None:
            if index >= len(self.lookup_table):
                raise ValueError(f"Lookup: getting mapping {index} when there are only {len(self.lookup_table)}.")
            vals = self.lookup_table[index]
        else:
            vals = self.lookup_table
        return torch.tensor(vals, dtype=torch_dtype, device=device)

    def get_midpoints(
        self, index: int = None, torch_dtype: torch.dtype = torch.float32, device: torch.device = "cuda"
    ) -> torch.Tensor:
        """Return one or all of the midpoint vectors as a tensor."""
        if not self.is_lookup:
            raise RuntimeError("NumberSpec.get_midpoints called for non-lookup number spec.")
        if index is not None:
            if index >= len(self.midpoints):
                raise ValueError(f"Lookup: getting lookup {index} when there are only {len(self.midpoints)}.")
            vals = self.midpoints[index]
        else:
            vals = self.midpoints
        return torch.tensor(vals, dtype=torch_dtype, device=device)

    def mapname(self, index: int) -> str:
        if not self.is_lookup:
            raise RuntimeError("NumberSpec.mapname called for non-lookup number spec.")
        if index is None or index >= len(self.mapnames):
            raise ValueError(f"Lookup: getting lookup {index} when there are only {len(self.mapnames)}.")
        return self.mapnames[index]

    def indices_from_vals(self, vals: list[float] | float, line: list[float] = None) -> list[int]:
        """Reverse search the number line to return indices."""
        if isinstance(vals, float):
            vals = [vals]
        if line is None:
            line = self.number_line
        return [line.index(v) for v in vals]

    def vals_from_indices(self, indices: list[int] | int, line: list[float] = None) -> list[float]:
        """Return values given a list of indices into number line."""
        if isinstance(indices, int):
            indices = [indices]
        if line is None:
            line = self.number_line
        return [line[i] for i in indices]

    def _decode(self, code: str | torch.dtype) -> None:
        """Sets fields based on input code string."""
        # 1.  Handle the case of the spec defined by a torch.dtype
        if isinstance(code, torch.dtype):
            self.torch_dtype = code
            code = str(code)
        name = code.lower().removeprefix("torch.")
        # 2.  Check for common datatype names that are not number formats
        if name in MX2NUMSPEC:
            tilesize = 8 if name.startswith("bfp") else 32
            raise ValueError(
                f"\tNumberSpec: code '{name}' is a scaled datatype rather than a number format.\n"
                f"\tThe equivalent NumberSpec name is '{MX2NUMSPEC[name]}', to be used in conjunction\n"
                f"\twith a ScaleSpec name of 'e8m0-{tilesize}' when creating the DataType."
            )
        elif name in MXUNSUPPORTED:
            raise NotImplementedError(
                f"\tNumberSpec: code '{name}' is a scaled datatype rather than a number format.\n"
                f"\tMX types (a/k/a bfp prime) are not yet supported."
            )
        # 3.  Check for instrisic non-standard bias
        bias_hack = int(name.startswith("float8") and name.endswith("fnuz"))  # implicit non-standard bias for torch fnuz
        name = name.removeprefix("float8_").removeprefix("float8")
        # 4.  Handle lookup table specs, which are separated from the compute spec by an underscore.
        if name.count("_") == 1:
            lcode, ccode = name.split("_")
            self._decode(ccode)
            if m := re.fullmatch(r"l(\d)(\d)(.*)", lcode):
                self.index_bits, self.lookup_bits, self.lookup_name = int(m.group(1)), int(m.group(2)), m.group(3)
                self.lookup_table, self.midpoints, self.mapnames = [], [], []
                return
            raise ValueError(f"NumberSpec lookup code {code} is invalid.")
        # 5.  Handle P3109-style string codes
        if m := re.fullmatch(r"binary(\d+)(p\d)", name):
            bits = int(m.group(1))
            prec = int(m.group(2)[1:]) if m.group(2) else 0
            if bits not in (8, 16, 32, 64):
                raise ValueError(f"NumberSpec: code '{name}': binary formats must be 8, 16, 32, or 64 bits.")
            if bits != 8:
                if bits == 64:
                    raise NotImplementedError(f"NumberSpec: code '{name}': 64-bit binary formats are not yet supported.")
                if prec:
                    raise ValueError(f"NumberSpec: code '{name}': precision is only supported for 8-bit binary formats.")
                name = f"e8m{bits - 9}"
            else:
                if prec not in range(2, 7):
                    raise ValueError(f"NumberSpec: code '{name}': precision must be in range [2, 6].")
                ebits = 8 - prec
                name = f"e{ebits}m{prec-1}b{2**(ebits-1)}fnuz"
        # 6.  Handle float/bfloat/int/uint style string codes for widths > 8
        if m := re.fullmatch(r"(float|bfloat|int|uint)(\d+)", name):
            prefix, bits = m.group(1), int(m.group(2))
            if prefix == "bfloat":
                self.ebits, self.mbits, self.bias = 8, bits - 9, 127
            elif prefix == "float":
                if bits > 16:
                    self.ebits, self.mbits, self.bias = 8, bits - 9, 127
                elif bits > 8:
                    self.ebits, self.mbits, self.bias = 5, bits - 6, 15
                else:
                    raise ValueError(f"NumberSpec: code '{name}': float8 and smaller formats require EMB format.")
            elif prefix[0] == "u":
                self.ebits, self.mbits, self.bias, self.signed, self.infnan = 0, bits, None, False, None
            else:
                self.ebits, self.mbits, self.bias, self.infnan = 1, bits - 2, 1, "fnuz"
        # 7.  Handle EMB stype string codes
        if self.mbits is None:
            if m := re.fullmatch(r"e(\d+)m(\d+)(b\d+)?(fn|fnuz)?", name):
                self.ebits, self.mbits, self.bias = int(m.group(1)), int(m.group(2)), m.group(3)
                self.infnan = m.group(4) or "ieee"
                self.signed = not (self.infnan == "ieee" and self.mbits == 0)
                if self.bias is None:
                    self.bias = 2 ** (self.ebits - 1) - 1 + bias_hack
                else:
                    self.bias = int(self.bias[1:])
        if self.ebits is None:
            raise ValueError(f"NumberSpec: code {code} is not a valid format.")
        self._name = name
        # 8.  Fill in the remaining fields in the spec from ebits/mbits/signed/infnan
        self.is_int = self.ebits == 1 and self.bias == 1 and self.signed and self.infnan == "fnuz"
        self.is_float = not self.is_int and self.signed and self.infnan is not None
        self.is_uint = self.bias is None and not self.signed and self.infnan is None
        self.is_exponent = self.ebits > 0 and self.mbits == 0 and not self.signed and self.infnan == "ieee"
        assert self.is_float or self.is_exponent or self.is_int or self.is_uint
        self.bits = self.ebits + self.mbits + int(self.signed)
        self.maxint = 2 ** (self.bits - int(self.signed)) - 1
        self.minint = -self.maxint if self.signed else 0
        if self.is_float or self.is_int:
            self.emax = 2**self.ebits - 1 - self.bias - int(self.infnan == "ieee")
            self.emin = 1 - self.bias
            self.maxfloat = 2**self.emax * (2.0 - (1 + int(self.infnan == "fn")) * 2 ** (-self.mbits))
            self.midmax = (2 ** (self.emax + 1) - self.maxfloat) / 2.0 + self.maxfloat
            self.eps = 2**-self.mbits
            self.smallest_normal = 2**self.emin
            self.smallest_subnormal = self.smallest_normal * self.eps
        # 9.  See if what we have matches a torch.dtype
        if self.torch_dtype is None:
            self.torch_dtype = self._find_torch_dtype()

    def _find_torch_dtype(self) -> torch.dtype | None:
        if self.is_uint:
            return torch.uint8 if self.bits == 8 else torch.uint16 if self.bits == 16 else torch.uint32
        if self.is_int:
            return torch.int8 if self.bits == 8 else torch.int16 if self.bits == 16 else torch.int32
        if self.is_exponent and self.bits == 8:
            return torch.uint8
        if self.bits == 32 and self.ebits == 8 and self.mbits == 23 and self.bias == 127 and self.infnan == "ieee":
            return torch.float32
        if self.bits == 16 and self.ebits == 5 and self.mbits == 10 and self.bias == 15 and self.infnan == "ieee":
            return torch.float16
        if self.bits == 16 and self.ebits == 8 and self.mbits == 7 and self.bias == 127 and self.infnan == "ieee":
            return torch.bfloat16
        if self.bits == 8 and is_float8_available():
            if self.ebits == 5 and self.mbits == 2 and self.bias == 15 and self.infnan == "ieee":
                return torch.float8_e5m2
            if self.ebits == 4 and self.mbits == 3 and self.bias == 7 and self.infnan == "fn":
                return torch.float8_e4m3fn
        if self.bits == 8 and is_float8_fnuz_available():
            if self.ebits == 5 and self.mbits == 2 and self.bias == 16 and self.infnan == "fnuz":
                return torch.float8_e5m2fnuz
            if self.ebits == 4 and self.mbits == 3 and self.bias == 8 and self.infnan == "fnuz":
                return torch.float8_e4m3fnuz
        return None

    def _check(self) -> None:
        # TODO(ericd): additional checks for bad/unsupported combinations of values that parsed correctly
        if self.bits > 32:
            raise NotImplementedError(f"NumberSpec: ({self.name}) bit widths > 32 are unsupported")
        if not self.is_uint:
            if self.ebits < 1 or self.ebits > 8:
                raise ValueError(f"NumberSpec: ({self.name}) ebits '{self.ebits}' needs to be in [1, 8]")

    @classmethod
    def valid(cls, code: str | torch.dtype) -> bool:
        """Checks validity without raising an exception."""
        try:
            cls(code)
            return True
        except (ValueError, NotImplementedError):
            return False
