"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/number.py: number format specification

from dataclasses import dataclass
import re
from typing import Literal

import torch

from .utils import is_float8_available

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

    def __init__(self, code: str | torch.dtype):
        self._decode(code)
        self._check()

    @property
    def name(self) -> str:
        """Returns the name.  May be overloaded in a subclass."""
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

    def get_number_line(self) -> list[float]:
        """All possible values for this number specification."""
        if self.bits > 8:
            raise ValueError(f"NumberSpec: too many number line values for {self.bits} bits")
        if not (self.is_float or self.ebits == 1):
            raise ValueError("NumberSpec: number line must be for float or float-like numbers.")
        # get the non-negative numbers then mirror for negatives, giving all 2^bits values, including 2 zeros
        line = [i * self.smallest_subnormal for i in range(2**self.mbits)]  # subnormals
        for e in range(self.emax - self.emin + 1):
            line += [(self.smallest_normal + i * self.smallest_subnormal) * 2 ** e for i in range(2**self.mbits)]
        return [-v for v in reversed(line)] + line

    def _decode(self, code: str | torch.dtype) -> None:
        """Sets fields based on input code string."""
        # 1.  Handle the case of the spec defined by a torch.dtype
        if isinstance(code, torch.dtype):
            self.torch_dtype = code
            code = str(code)
        code = code.lower().removeprefix("torch.")
        if ttype := getattr(torch, code, False):
            if self.torch_dtype is None and isinstance(ttype, torch.dtype):
                self.torch_dtype = ttype
        bias_hack = int(code.startswith("float8") and code.endswith("fnuz"))  # implicit non-standard bias for torch fnuz types
        name = code = code.removeprefix("float8_")
        # 2.  Check for implicitly scaled datatypes
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
        # 3.  Handle float/bfloat/int/uint style string codes for widths > 8
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
        # 4.  Handle EMB stype string codes
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

        # 5.  Fill in the remaining fields in the spec from ebits/mbits/signed/infnan
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

        # 6.  See if what we have matches a torch.dtype
        if self.torch_dtype is None:
            self.torch_dtype = self._find_torch_dtype()

    def _find_torch_dtype(self) -> torch.dtype | None:
        if self.bits == 32 and self.ebits == 8 and self.mbits == 23 and self.bias == 127 and self.infnan == "ieee":
            return torch.float32
        if self.bits == 16 and self.ebits == 5 and self.mbits == 10 and self.bias == 15 and self.infnan == "ieee":
            return torch.float16
        if self.bits == 16 and self.ebits == 8 and self.mbits == 7 and self.bias == 127 and self.infnan == "ieee":
            return torch.bfloat16
        if self.bits == 8 and is_float8_available():
            if self.ebits == 5 and self.mbits == 2 and self.bias == 15 and self.infnan == "ieee":
                return torch.float8_e5m2
            if self.ebits == 5 and self.mbits == 2 and self.bias == 16 and self.infnan == "fnuz":
                return torch.float8_e5m2fnuz
            if self.ebits == 4 and self.mbits == 3 and self.bias == 7 and self.infnan == "fn":
                return torch.float8_e4m3fn
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
