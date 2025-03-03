"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/datatype.py: combines number spec and scale spec to define a datatype

import torch

from .number import Codebook, NumberSpec
from .scale import ScaleSpec


class DataType:
    """Everything needed to define a scaled or unscaled datatype."""

    registry = {}

    def __init__(self, nspec: str | NumberSpec = None, sspec: str | ScaleSpec = None, name: str = None):
        if nspec is None:
            if sspec is not None:
                raise ValueError("DataType must be initialized with either a name or a number spec.")
            if name in self.registry:
                return self.registry[name]
            if not isinstance(name, str):
                raise ValueError("DataType must be initialized with either a name or a number spec.")
            segments = name.split("_")
            if len(segments) == 1:
                nspec = segments[0]
                sspec = None
            elif segments[0].lower().startswith("cb"):
                nspec = "_".join(segments[:2])
                if len(segments) > 2:
                    sspec = "_".join(segments[2:])
            else:
                nspec = segments[0]
                sspec = "_".join(segments[1:])
        self.nspec = (
            nspec
            if isinstance(nspec, NumberSpec)
            else Codebook(nspec)
            if str(nspec).lower().startswith("cb")
            else NumberSpec(nspec)
        )
        self.sspec = sspec if isinstance(sspec, ScaleSpec) else ScaleSpec(sspec) if sspec else None
        self._name = name
        for attr in ("channel", "tile", "subtile", "sparse", "tensor", "multiscale", "2d"):
            setattr(self, f"is_{attr}", self.sspec and getattr(self.sspec, f"is_{attr}"))
        self.is_unscaled, self.is_codebook = self.sspec is None, self.nspec.is_codebook
        self._check()
        self.registry[self.name] = self

    def _check(self):
        prefix = f"DataType: '{self.name}'"
        if self.is_unscaled:
            if not self.nspec.is_float:
                raise ValueError(f"{prefix} only float numbers can be cast unscaled.")
        else:
            if self.is_subtile and not self.is_codebook:
                raise ValueError(f"{prefix} subtile scaling is only permitted with codebook data.")
            if self.nspec.is_exponent:
                raise ValueError(f"{prefix} exponent number spec is only permitted as a scale.")
            if self.sspec.zero and not self.nspec.is_uint:
                raise ValueError(f"{prefix} zero spec in scale is incompatible with float or signed int data spec.")
            if self.nspec.is_uint and not self.sspec.zero:
                raise ValueError(f"{prefix} uint data requires a zero point.")
            if self.nspec.is_int and not (self.sspec.scale.is_exponent or self.sspec.scale.is_float):
                raise ValueError(f"{prefix} int data requires either a float or exponent scale.")

    @property
    def name(self):
        if self._name:
            return self._name
        return str(self)

    def bits_per_value(self, tensor: torch.Tensor | None) -> float:
        """Given scaling metadata and codebook metadata, how many bits per value?  Does not include codebook tables themselves."""
        if self.is_unscaled or (tensor is None and (self.is_tensor or self.is_channel)):
            return float(self.nspec.bits)
        if self.is_2d:
            raise NotImplementedError("2D scaling not yet implemented.")
        scale_bits = float(self.sspec.scale.bits + (self.sspec.zero.bits if self.sspec.zero else 0))
        if self.is_tensor:
            return (scale_bits + self.nspec.bits * tensor.numel()) / tensor.numel()
        if self.is_channel:
            assert self.sspec.tile == 0
            csize = tensor.size(self.sspec.dim)
            return (scale_bits * tensor.numel() / csize + self.nspec.bits * tensor.numel()) / tensor.numel()
        assert self.is_tile
        tsize = self.sspec.tile
        ssize = self.sspec.subtile if self.is_subtile else tsize
        meta_bits = (self.sspec.offset if self.is_offset else 0) + (
            self.nspec.meta_bits if self.is_codebook else 0
        ) * tsize // ssize
        value_bits = tsize * (self.nspec.index_bits if self.is_codebook else self.nspec.bits)
        return (scale_bits + meta_bits + value_bits) / tsize

    def __str__(self):
        s = self.nspec.name
        if self.sspec:
            s += "_" + self.sspec.name
        return s

    @classmethod
    def valid(cls, ncode: str = None, scode: str = None, name: str = None) -> bool:
        """Returns True if the code generates a valid datatype."""
        try:
            cls(ncode, scode, name)
            return True
        except ValueError:
            return False
