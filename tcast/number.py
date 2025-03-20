#!/usr/bin/env python
# tcast/number.py: number format specification
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from collections import namedtuple
from dataclasses import dataclass
import re
from typing import NamedTuple

import torch

from .common import FP8_DTYPES, IMPLICIT_CODEBOOKS, MX2NUMSPEC, InfNaN, get_enum
from .utils import is_float8_available, is_float8_fnuz_available


class Finfo(NamedTuple):
    """Similar to torch.finfo."""

    minint: int
    maxint: int
    minfloat: float
    maxfloat: float
    midmax: float
    eps: float
    smallest_normal: float
    smallest_subnormal: float


@dataclass
class NumberSpec:
    """Specification for an unscaled number format."""

    _name: str = None
    _torch_dtype: torch.dtype = None
    ebits: int = None
    mbits: int = None
    bias: int = None
    signed: bool = True
    infnan: InfNaN = InfNaN.IEEE
    finfo: Finfo = None
    emax: int = None
    emin: int = None
    _bias_hack: int = 0

    def __init__(self, code: str | torch.dtype):
        """Sets fields based on input code string, transforming the string to a canonical form."""
        # 1.  Handle the case of the spec defined by a torch.dtype
        if isinstance(code, torch.dtype):
            self._torch_dtype = code
            code = str(code)
        name = code.lower().removeprefix("torch.")

        # 2.  Check for common datatype names that are not number formats because they have an implicit scale
        if name in MX2NUMSPEC:
            raise ValueError(
                f"\tNumberSpec: code '{name}' is a scaled datatype rather than a number format.\n"
                f"\tThe creation call is '{MX2NUMSPEC[name]}', or just use the predefined type 'tcast.{name}"
            )

        # 3.  Check for instrisic non-standard bias
        self._bias_hack = int(name.startswith("float8") and name.endswith("fnuz"))  # implicit non-standard bias for torch fnuz
        name = name.removeprefix("float8").removeprefix("_")

        # 4.  Handle P3109-style string codes; these are a hybrid of "fn" and "fnuz" that we call "inuz"
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
                if prec not in range(1, 7):
                    raise ValueError(f"NumberSpec: code '{name}': precision must be in range [1, 7].")
                ebits = 8 - prec
                if prec == 0:
                    name = f"e{ebits}m{prec-1}b{2**(ebits-1)-1}inuz"
                else:
                    name = f"e{ebits}m{prec-1}b{2**(ebits-1)}inuz"

        # 5.  Handle float/bfloat/int/uint style string codes for widths > 8
        if m := re.fullmatch(r"(float|bfloat|int|uint)(\d+)", name):
            prefix, bits = m.group(1), int(m.group(2))
            if prefix == "bfloat" or prefix == "float" and bits > 16:
                name = f"e8m{bits - 9}"
            elif prefix == "float":
                if bits > 8:
                    name = f"e5m{bits - 6}"
                else:
                    raise ValueError(f"NumberSpec: code '{name}': float8 and smaller formats require EMB format.")
            elif prefix[0] == "u":
                self.ebits, self.mbits, self.bias, self.signed, self.infnan = 0, bits, None, False, None
            else:
                name = f"e1m{bits - 2}b1fnuz"

        # 6.  Handle EMB stype string codes (everything but uint gets here in EMB format)
        if self.signed:
            if m := re.fullmatch(r"e(\d+)m(\d+)(b\d+)?(ieee|fn|fnuz|inuz)?", name):
                self.ebits, self.mbits, self.bias = int(m.group(1)), int(m.group(2)), m.group(3)
                if m.group(4):
                    self.infnan = get_enum(InfNaN, m.group(4))
                # the following kludge was brought to you by OCP unsigned floating point "exponent" specs
                self.signed = not (self.infnan == InfNaN.IEEE and self.mbits == 0)
                self.bias = 2 ** (self.ebits - 1) - 1 + self._bias_hack if self.bias is None else int(self.bias[1:])
            else:
                raise ValueError(f"NumberSpec: code {name} is not a valid format.")
        if self.bits > 32:
            raise NotImplementedError(f"NumberSpec: ({self.name}) bit widths > 32 are unsupported")

        # All done!  Now we can populate the remaining spec fields.
        self._name = name
        assert int(self.is_float) + int(self.is_exponent) + int(self.is_int) + int(self.is_uint) == 1
        maxint = 2 ** (self.bits - int(self.signed)) - 1
        minint = -maxint if self.signed else 0
        maxfloat = minfloat = midmax = smallest_normal = smallest_subnormal = eps = 0.0
        if self.ebits:
            self.emax = 2**self.ebits - 1 - self.bias - int(self.infnan == InfNaN.IEEE)
            self.emin = 1 - self.bias
            maxfloat = 2**self.emax * (2.0 - (1 + int(self.infnan in (InfNaN.FN, InfNaN.INUZ))) * 2 ** (-self.mbits))
            minfloat = -maxfloat
            midmax = (2 ** (self.emax + 1) - maxfloat) / 2.0 + maxfloat
            if self.infnan in (InfNaN.FN, InfNaN.INUZ):
                midmax -= (midmax - maxfloat) / 2.0
            eps = 2**-self.mbits
            smallest_normal = 2**self.emin
            smallest_subnormal = smallest_normal * eps
        self.finfo = Finfo(minint, maxint, minfloat, maxfloat, midmax, eps, smallest_normal, smallest_subnormal)

        # 9.  See if what we have matches a torch.dtype
        if self._torch_dtype is None:
            self._torch_dtype = self._find_torch_dtype()

    # fmt: off
    @property
    def name(self) -> str: return self._name
    @property
    def std_bias(self) -> bool: return self.bias == 2**(self.ebits - 1) - 1 - self._bias_hack
    @property
    def bits(self) -> int: return self.ebits + self.mbits + int(self.signed)
    @property
    def is_float(self) -> bool: return self.ebits > 1 and self.signed
    @property
    def is_int(self) -> bool: return self.ebits == 1
    @property
    def is_uint(self) -> bool: return self.ebits == 0
    @property
    def is_exponent(self) -> bool: return self.ebits > 1 and not self.signed
    @property
    def is_fp8(self) -> bool: return self.torch_dtype in FP8_DTYPES
    @property
    def is_codebook(self): return False
    @property
    def torch_dtype(self) -> torch.dtype | None: return self._torch_dtype
    # fmt: on

    def _find_torch_dtype(self) -> torch.dtype | None:
        """Find the corresponding torch dtype for specs that exactly match a torch.dtype."""
        if self._torch_dtype:
            return self._torch_dtype
        if (self.is_uint or self.is_exponent) and self.bits == 8:
            return torch.uint8
        if self.is_int:
            return torch.int8 if self.bits == 8 else torch.int16 if self.bits == 16 else torch.int32 if self.bits == 32 else None
        if self.bits == 32 and self.ebits == 8 and self.mbits == 23 and self.bias == 127 and self.infnan == InfNaN.IEEE:
            return torch.float32
        if self.bits == 16 and self.ebits == 5 and self.mbits == 10 and self.bias == 15 and self.infnan == InfNaN.IEEE:
            return torch.float16
        if self.bits == 16 and self.ebits == 8 and self.mbits == 7 and self.bias == 127 and self.infnan == InfNaN.IEEE:
            return torch.bfloat16
        if self.bits == 8 and is_float8_available():
            if self.ebits == 5 and self.mbits == 2 and self.bias == 15 and self.infnan == InfNaN.IEEE:
                return torch.float8_e5m2
            if self.ebits == 4 and self.mbits == 3 and self.bias == 7 and self.infnan == InfNaN.FN:
                return torch.float8_e4m3fn
        if self.bits == 8 and is_float8_fnuz_available():
            if self.ebits == 5 and self.mbits == 2 and self.bias == 16 and self.infnan == InfNaN.FNUZ:
                return torch.float8_e5m2fnuz
            if self.ebits == 4 and self.mbits == 3 and self.bias == 8 and self.infnan == InfNaN.FNUZ:
                return torch.float8_e4m3fnuz
        return None

    @classmethod
    def valid(cls, code: str | torch.dtype) -> bool:
        """Checks validity without raising an exception."""
        try:
            cls(code)
            return True
        except (ValueError, NotImplementedError):
            return False


class NumberLine:
    """All possible values for a number specification."""

    def __init__(self, nspec: NumberSpec):
        if nspec.bits > 9:
            raise ValueError(f"NumberSpec: too many number line values for {nspec.bits} bits")
        if not (nspec.is_float or nspec.is_int):
            raise ValueError("NumberSpec: number line must be for float or float-like numbers.")
        # get the non-negative numbers then mirror for negatives, giving all 2^bits values, including 2 zeros
        line = [i * nspec.finfo.smallest_subnormal for i in range(2**nspec.mbits)]  # subnormals
        for e in range(nspec.emax - nspec.emin + 1):
            line += [(nspec.finfo.smallest_normal + i * nspec.finfo.smallest_subnormal) * 2**e for i in range(2**nspec.mbits)]
        if nspec.infnan in ("fn", "inuz"):
            line = [0.0] + line[:-1]
        self.line = [-v for v in reversed(line)] + line
        self.nspec = nspec

    def vals_in_line(self, vals: float | list[float]) -> bool:
        """Check if all values are in the number line."""
        return vals in self.line if isinstance(vals, float) else all(v in self.line for v in vals)

    def indices_have_vals(self, indices: list[int]) -> bool:
        """Check if all values are in the number line."""
        return all(i in range(len(self.line)) for i in indices)

    def indices_from_vals(self, vals: float | list[float]) -> int | list[int]:
        """Reverse search the number line to return indices."""
        assert self.vals_in_line(vals)
        return self.line.index(vals) if isinstance(vals, float) else [self.line.index(v) for v in vals]

    def vals_from_indices(self, indices: list[int]) -> list[float]:
        """Return values given a list of indices into number line."""
        assert self.indices_have_vals(indices)
        return [self.line[i] for i in indices]


class Codebook(NumberSpec):
    """Methods for codebook operations."""

    def __init__(self, code: str):
        assert code.startswith("cb")
        code = code.lower()
        self.mappings, self.midpoints, self.mapnames = [], [], []
        self.symmetric = True
        self.index_bits = self.meta_bits = 0
        self.implicit = False
        self._number_line = None
        cbname = label = None
        subcodes = code.split("_")
        if len(subcodes) == 2:
            cbname, compname = subcodes
        elif len(subcodes) == 3:
            cbname, compname, label = subcodes
        if not (isinstance(code, str) and code.startswith("cb") and cbname is not None and compname is not None):
            raise ValueError(
                f"NumberSpec codebook code {code} has one of two forms: <cbname>_<compspec> or <cbname>_<compspec>_<label>."
                f"Codebook cbname is 'cb<index_bits><meta_bits><implicit_code>' and compspec is a valid NumberSpec."
            )
        super().__init__(compname)
        if m := re.fullmatch(r"cb(\d)(\d)(.*)", cbname):
            self.index_bits, self.meta_bits, icode = (int(m.group(1)), int(m.group(2)), m.group(3))
            if self.index_bits not in range(1, 9):
                raise ValueError(f"NumberSpec: codebook index bits must be in range [1, 8], not {self.index_bits}.")
            if self.meta_bits not in range(1, 9):
                raise ValueError(f"NumberSpec: codebook meta bits must be in range [1, 8], not {self.meta_bits}.")
            if icode:
                self._parse_code(icode)
                self.implicit = True
        else:
            raise ValueError(f"NumberSpec codebook code {code} is invalid.")
        self.cbname, self.label = cbname, label

    # fmt: off
    @property
    def name(self) -> str: return f"{self.label}" if self.label else f"{self.cbname}_{self._name}"
    @property
    def is_codebook(self): return True
    @property
    def is_implicit(self) -> bool: return self.implicit and self.name in IMPLICIT_CODEBOOKS
    @property
    def num_mappings(self) -> int: return 2**self.meta_bits
    @property
    def current_mappings(self) -> int: return len(self.mappings)
    @property
    def full_mappings(self) -> bool: return self.current_mappings == self.num_mappings
    @property
    def num_values(self) -> int: return 2**self.index_bits if self.is_codebook else 0
    # fmt: on

    @property
    def number_line(self) -> list[float]:
        if self._number_line is None:
            self._number_line = NumberLine(self)
        return self._number_line

    def add_mapping(self, mapping: list[float], mapname: str = None):
        """Add a new codebook table."""
        if self.current_mappings == self.num_mappings:
            raise RuntimeError(f"NumberSpec.add_mapping exceeds number of mappings specified ({self.num_mappings}).")
        if not self.number_line.vals_in_line(mapping):
            raise ValueError(f"NumberSpec.add_mapping: some values are not representable in {self.name}")
        self.symmetric = self.symmetric and mapping[: len(mapping) // 2] == [-v for v in reversed(mapping[len(mapping) // 2 :])]
        self.mappings.append(mapping)
        self.midpoints.append([(mapping[i] + mapping[i + 1]) / 2.0 for i in range(len(mapping) - 1)].append(self.finfo.midmax))
        self.mapnames.append(mapname if mapname else str(len(self.mappings) - 1))

    def add_mappings(self, mappings: list[list[float]], mapnames: list[str] = None):
        """Add multiple mappings to the codebook."""
        for i in range(len(mappings)):
            self.add_mapping(mappings[i], mapnames[i] if mapnames else None)

    def get_codebook(self, torch_dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the codebook as a tensor."""
        vals = [i[len(self.mappings[i]) // 2 :] for i in self.mappings] if self.symmetric else self.mappings
        mids = [i[len(self.midpoints[i]) // 2 :] for i in self.midpoints] if self.symmetric else self.midpoints
        vals, mids = torch.tensor(vals, dtype=torch_dtype, device=device), torch.tensor(mids, dtype=torch_dtype, device=device)
        return vals, mids

    def mapname(self, index: int) -> str:
        if index is None or index >= len(self.mapnames):
            raise ValueError(f"NumberSpec: getting codebook mapname {index} when there are only {len(self.mapnames)}.")
        return self.mapnames[index]

    def _parse_code(self, icode: str) -> None:
        ImplicitSpec = namedtuple("ImplicitSpec", ["pattern", "offsets", "eoffsets", "decr"])

        # get the numberspecs and default offsets for f and i codes
        fspec = fline = foffset = ispec = iline = ioffset = None
        if "f" in icode:
            if self.index_bits not in (3, 4, 5, 6):
                raise ValueError(f"NumberSpec: code {icode} with a float subcode requires 3-6 index bits.")
            fspec = NumberSpec(["e2m0fnuz", "e2m1fnuz", "e2m2fnuz", "e2m3fnuz"][self.index_bits - 3])
            fline = NumberLine(fspec)
            foffset = len(self.number_line.line) - 1 - self.number_line.indices_from_vals(fline.line[-1])
        if "i" in icode:
            # if self.index_bits not in range(2, 3, 4, 5, 6):
            #     raise ValueError(f"NumberSpec: code {icode} with an integer subcode requires 2-6 index bits.")
            ispec = NumberSpec(
                ["e1m0b1fnuz", "e1m1b1fnuz", "e1m2b1fnuz", "e1m3b1fnuz", "e1m4b1fnuz", "e1m5b1fnuz", "e1m6b1fnuz"][
                    self.index_bits - 2
                ]
            )
            iline = NumberLine(ispec)
            ioffset = len(self.number_line.line) - 1 - self.number_line.indices_from_vals(iline.line[-1])
        fi_specinfo = dict(f=(fspec, fline, foffset), i=(ispec, iline, ioffset))

        # one or more pattern codes are specified, create a list of ImplicitSpecs to build the mappigs
        if matches := list(re.finditer(r"([fips])([^fips]*)", icode)):
            ispecs: list[ImplicitSpec] = []
            subcodes = tuple(m.group(1, 2) for m in matches)
            for subcode in subcodes:
                eoffsets, decr = [0], 1
                if subcode[0] in "fi":
                    if m := re.fullmatch(r"(\d*)(e)?(\d+)?", subcode[1]):
                        o, e, eo = m.groups()
                        offsets = [int(s) for s in o] if o else [fi_specinfo[subcode[0]][2]]
                        eoffsets = [int(s) for s in eo] if eo else [0, 1] if e else [0]
                    else:
                        raise ValueError(f"NumberSpec: code {icode} has invalid implicit codebook specifier.")
                else:
                    if m := re.fullmatch(r"(\d)?(\d*)", subcode[1]):
                        d, o = m.groups()
                        decr = int(d) if d else 1
                        offsets = [int(s) for s in o] if o else [0]
                    else:
                        raise ValueError(f"NumberSpec: code {icode} has invalid implicit codebook specifier.")
                ispecs.append(ImplicitSpec(subcode[0], offsets, eoffsets, decr))
            mapcount = sum(max(1, len(s.offsets)) * len(s.eoffsets) for s in ispecs)
            if mapcount != self.num_mappings:
                raise ValueError(f"NumberSpec: code {icode} has {mapcount} mappings, expected {self.num_mappings}.")
        else:
            raise ValueError(f"NumberSpec: {icode} is not a valid implicit codebook specifier.")

        # now we can populate the codebook
        num_positive = 2 ** (self.index_bits - 1) - 1
        for ispec in ispecs:
            if ispec.pattern in "fi":
                s, line = fi_specinfo[ispec.pattern][:2]
                for eoffset in ispec.eoffsets:
                    scaled = [i * 2 ** (self.emax - s.emax - eoffset) for i in line.line if i > 0.0]
                    if len(scaled) == num_positive and self.number_line.vals_in_line(scaled):
                        indices = self.number_line.indices_from_vals(scaled)
                        for offset in ispec.offsets:
                            oindices = [i + fi_specinfo[ispec.pattern][2] - offset for i in indices]
                            if self.number_line.indices_have_vals(oindices):
                                vals = self.number_line.vals_from_indices(oindices)
                                while len(vals) < num_positive + 1:
                                    vals = [0.0] + vals
                                self.add_mapping([-v for v in reversed(vals)] + vals, f"{ispec.pattern}{offset}")
                            else:
                                raise ValueError(
                                    f"NumberSpec: code {icode} has invalid indices for {ispec.pattern} offset {offset}."
                                )
                    else:
                        raise ValueError(f"NumberSpec: code {icode} has invalid values for {ispec.pattern} eoffset {eoffset}.")
            else:
                for offset in ispec.offsets:
                    decr, indices, start = ispec.decr, [], len(self.number_line.line) - 1 - offset
                    for i in range(num_positive):
                        indices.append(start - i * decr)
                        decr = int(ispec.pattern == "p")
                    indices.reverse()
                    if self.number_line.indices_have_vals(indices):
                        vals = self.number_line.vals_from_indices(indices)
                        while len(vals) < num_positive + 1:
                            vals = [0.0] + vals
                        self.add_mapping([-v for v in reversed(vals)] + vals, f"{ispec.pattern}{ispec.decr}_{offset}")
                    else:
                        raise ValueError(f"NumberSpec: code {icode} has invalid indices for {ispec.pattern} offset {offset}.")
