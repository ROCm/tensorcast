"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/cast.py: casting methods

from typing import ClassVar, Literal

import torch

from .datatype import DataType
from .extension import Extension
from .number import NumberSpec
from .utils import check_literal

RoundMode = Literal["even", "nearest", "zero", "stochastic"]
ScaleMode = Literal["max", "auto"]
ComputeMode = Literal["cpu", "gpu", "torch"]


class Cast:
    """Static class with implementations in PyTorch."""

    roundmode: ClassVar[RoundMode] = "even"  # the current rounding mode
    scalemode: ClassVar[ScaleMode] = "max"  # the current scale selection mode
    compmode: ClassVar[ComputeMode] = "torch"  # use PyTorch operators for both CPU and GPU
    extension: ClassVar[Extension] = None

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
    def _vcast(cls, x: torch.Tensor, dtype: DataType) -> torch.Tensor:
        """Virtual cast, atomic."""
        if cls.compmode != "torch":
            if cls.extension is None:
                cls.extension = Extension()
            if cls.extension.has_operation("vcast", cls.compmode):
                return cls.extension.exec_operation(x, dtype, "vcast", cls.compmode)
        xtype = x.dtype
        x = x.clone()
        if dtype.is_unscaled:
            return cls._cast_unscaled(x, dtype.nspec).to(xtype)
        assert dtype and dtype.sspec
        dim = None if dtype.sspec.is_tensor else -1
        x = dtype.sspec.reshape_tensor(x)
        eps = torch.finfo(torch.float32).eps
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
            x *= nscale  # scale x into range of the target dtype
            valexp = (cls._safe_frexp(x) - 1).clamp_min(dtype.nspec.emin)  # get the independent exponents, clipped to emin
            rscale = (dtype.nspec.mbits - valexp).exp2()
            x = cls._round(x * rscale).div(rscale).clamp(-dtype.nspec.maxfloat, dtype.nspec.maxfloat)
            x /= nscale
        x = dtype.sspec.revert_tensor(x)
        return x.to(xtype)

    @classmethod
    def sparse(cls, tensor: torch.Tensor, stile: int, dense: int, dim: int = -1) -> torch.Tensor:
        """Simple structured sparsity, M of N, where M is dense values retained out of N."""
        if tensor.shape[-1] % stile != 0:
            raise NotImplementedError(
                f"Last tensor dim ({tensor.shape[-1]}) must be evenly divisible by sparse tile size ({stile})"
            )
        assert dense > 0 and dense < stile
        tshape = tensor.shape
        t = tensor.clone().transpose(dim, -1).reshape(-1, stile).abs()
        idx = t.argsort(dim=-1, descending=True)
        premask = torch.full(t.shape, True, dtype=torch.bool, device=tensor.device)
        mask = torch.empty_like(premask)
        premask[..., :dense] = False
        mask.scatter_(-1, idx, premask)
        return t.masked_fill_(mask, 0.0).reshape(tshape).transpose(dim, -1)

    @classmethod
    def cast(
        cls,
        x: torch.Tensor,
        dtype: DataType,
        roundmode: RoundMode = None,
        scalemode: ScaleMode = None,
        compmode: ComputeMode = None,
    ) -> torch.Tensor:
        """Generic cast interface."""
        # currently not so generic as we are only doing virtual cast
        # roundmode and scalemode are optional overrides
        check_literal(roundmode, RoundMode, True)
        check_literal(scalemode, ScaleMode, True)
        check_literal(compmode, ComputeMode, True)
        saveround, savescale, savecomp = cls.roundmode, cls.scalemode, cls.compmode
        cls.roundmode = roundmode if roundmode else saveround
        cls.scalemode = scalemode if scalemode else savescale
        cls.compmode = compmode if compmode else savecomp
        x = cls._vcast(x, dtype)
        cls.roundmode = saveround
        cls.scalemode = savescale
        cls.compmode = savecomp
        return x
