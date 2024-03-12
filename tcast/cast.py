"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/cast.py: casting methods

from typing import ClassVar, Literal

import torch

from .datatype import DataType
from .number import NumberSpec
from .utils import check_literal

RoundMode = Literal["even", "nearest", "zero", "stochastic"]
ScaleMode = Literal["standard", "auto"]


class Cast:
    """Static class with implementations in PyTorch."""

    roundmode: ClassVar[RoundMode] = "even"  # the current rounding mode
    scalemode: ClassVar[ScaleMode] = "standard"  # the current scale selection mode

    @classmethod
    def _round(cls, x: torch.Tensor) -> torch.Tensor:
        if cls.roundmode == "stochastic":
            return torch.sign(x) * torch.trunc(torch.abs(x) + torch.rand_like(x))
        if cls.roundmode == "even":
            return torch.round(x)
        if cls.roundmode == "nearest":
            # return torch.trunc(x + x.sign() * 0.5) torch thinks 0.4999999701976776123046875 + 0.5 is 1.0
            return torch.where(x.abs().frac() == 0.5, torch.trunc(x + x.sign() * 0.5), x.round())
        return torch.trunc(x)

    @classmethod
    def _safe_frexp(cls, x: torch.Tensor) -> torch.Tensor:
        return x.float().add(torch.finfo(torch.float32).eps * (x == 0.0)).frexp().exponent

    @classmethod
    def _cast_unscaled(cls, x: torch.Tensor, nspec: NumberSpec) -> torch.Tensor:
        assert x.is_floating_point and nspec.is_float
        valexp = (cls._safe_frexp(x) - 1).clamp_min(nspec.emin)
        rscale = (nspec.mbits - valexp).exp2()
        x = cls._round(x * rscale).div(rscale).clamp(-nspec.maxfloat, nspec.maxfloat)
        return x

    @classmethod
    def vcast(cls, x: torch.Tensor, dtype: DataType) -> torch.Tensor:
        """Virtual cast, atomic."""
        xtype = x.dtype
        x = x.clone() #.float()
        if dtype.is_unscaled:
            return cls._cast_unscaled(x, dtype.nspec).to(xtype)
        assert dtype and dtype.sspec
        dim = None if dtype.sspec.is_tensor else -1
        x = dtype.sspec.reshape_tensor(x)
        eps = torch.finfo(x.dtype).eps
        if dtype.nspec.is_uint:
            assert dtype.sspec.scale.is_float
            tmin, tmax = torch.aminmax(x, dim=dim, keepdim=True)
            tmin, tmax = tmin.clamp_max(0.0), tmax.clamp_min(0.0)
            scale = cls._cast_unscaled((tmax - tmin) / dtype.nspec.maxint, dtype.sspec.scale)
            if dtype.sspec.zero.is_float:
                zero = cls._cast_unscaled(-tmin, dtype.sspec.zero)
                x = scale * cls._round((x + zero) / scale.clamp_min(eps)).clamp(0, dtype.nspec.maxint) - zero
            else:
                zero = cls._round(-tmin / scale)
                x = scale * ((cls._round(x / scale.clamp_min(eps)) + zero).clamp(0, dtype.nspec.maxint) - zero)
        elif dtype.sspec.scale.is_float:
            scale = (
                cls._cast_unscaled(1.0 / x.abs().amax(dim=dim, keepdim=True), dtype.sspec.scale)
                if dtype.nspec.is_float
                else cls._cast_unscaled(x.abs().amax(dim=dim, keepdim=True) / dtype.nspec.maxint, dtype.sspec.scale)
            )
            x = cls._round(x / scale.clamp_min(eps)).clamp(dtype.nspec.minint, dtype.nspec.maxint) * scale
        else:
            # get po2 scale (mx style) and scale the tensor into dtype-representable range
            maxexp = cls._safe_frexp(x).amax(dim=dim, keepdim=True) - 1 - dtype.nspec.emax
            if dtype.nspec.ebits > 1 and cls.scalemode == "auto":
                maxexp[(x * (-maxexp).exp2()).abs().amax(dim=dim) > dtype.nspec.midmax] += 1
            nscale = (-maxexp).exp2()
            x *= nscale                                           # scale x into range of the target dtype
            valexp = (cls._safe_frexp(x) - 1).clamp_min(dtype.nspec.emin)   # get the independent exponents, clipped to emin
            rscale = (dtype.nspec.mbits - valexp).exp2()
            x = cls._round(x * rscale).div(rscale).clamp(-dtype.nspec.maxfloat, dtype.nspec.maxfloat)
            x /= nscale
        x = dtype.sspec.revert_tensor(x)
        return x.to(xtype)

    @classmethod
    def cast(cls, x: torch.Tensor, dtype: DataType, roundmode: RoundMode = None, scalemode: ScaleMode = None) -> torch.Tensor:
        """Generic cast interface."""
        # currently not so generic as we are only doing virtual cast
        # roundmode and scalemode are optional overrides
        check_literal(roundmode, RoundMode, True)
        check_literal(scalemode, ScaleMode, True)
        saveround, savescale = cls.roundmode, cls.scalemode
        cls.roundmode = roundmode if roundmode else saveround
        cls.scalemode = scalemode if scalemode else savescale
        x = cls.vcast(x, dtype)
        cls.roundmode = saveround
        cls.scalemode = savescale
        return x
