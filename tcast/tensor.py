#!/usr/bin/env python
# tcast/tensor.py: cast tensor types and utilities
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from typing import NamedTuple

from einops import rearrange
import torch

from .common import CastMode, ComputeMode, Modes, ScaleData
from .datatype import DataType
from .utils import cdiv, get_logger, is_multiple

logger = get_logger()


class ShapeInfo(NamedTuple):
    """Shape and type for original, actual, and compressed tensors."""

    oshape: torch.Size
    odtype: torch.dtype
    ashape: torch.Size
    adtype: torch.dtype
    cshape: torch.Size
    cdtype: torch.dtype
    pratio: int
    sratio: int

    def get_info(self, castmode: CastMode) -> tuple[torch.Size, torch.dtype]:
        """Get the shape and dtype for the tensor."""
        if castmode == CastMode.VIRTUAL:
            return self.oshape, self.odtype
        elif castmode == CastMode.ACTUAL:
            return self.ashape, self.adtype
        return self.cshape, self.cdtype


class Tensor:
    """A quantized and compressed tensor with scaling and descriptive information."""

    def __init__(self, tensor: torch.Tensor, dtype: DataType, transpose_scale: bool = False, precast: bool = True):
        self.dtype, self.input = dtype, tensor
        self.original_shape, self.original_device, self.original_dtype = tensor.size(), tensor.device, tensor.dtype
        self.output = self.tenscale = self.scale = self.zero = self.meta = self.mask = None
        self.transpose_scale = transpose_scale
        self.is_compressed = self.quantized = False
        self.shape_transforms = []
        self.shape_info = {}
        if precast:
            self.precast()

    @property
    def needs_pad(self) -> bool:
        """See if the tensor dims are not multiples of the tile scales."""
        if self.input.dim() != 2:
            raise NotImplementedError("More than 2D shapes are not supported yet.")
        if not self.dtype.is_tile or self.dtype.is_2d or self.dtype.is_sparse:
            raise NotImplementedError("Only square tiles are supported.")
        tile0, tile1 = self.dtype.sspec.tile0, self.dtype.sspec.tile1
        size0, size1 = self.input.size() if not self.transpose_scale else self.input.size()[::-1]
        return not (is_multiple(size0, tile0) and is_multiple(size1, tile1))

    def get_scaledata(self) -> ScaleData:
        """Get the scale data for the tensor."""
        if self.dtype.is_unscaled:
            return ScaleData()
        return ScaleData(self.scale, self.zero, self.tenscale, self.meta, self.mask)

    def get_torch_dtypes(self, name: str) -> tuple[torch.dtype, torch.dtype, int, int]:
        """Get the actual and compressed torch types and compressed packing factor for a tensor."""
        nspec, sspec = self.dtype.nspec, self.dtype.sspec
        if sspec is None:
            assert self.dtype.is_unscaled
            adtype = nspec.torch_dtype if name == "output" else None
            return adtype, adtype, 1, 1
        if name == "mask":
            return torch.bool, torch.uint8, 8, 1
        sratio = sspec.sparse_ratio if name == "output" else 1
        if nspec.is_codebook:
            if name == "output":
                return torch.uint8, torch.uint8, cdiv(8, nspec.index_bits), sratio
            if name == "meta":
                assert nspec.meta_bits <= 8
                num_subtiles = sspec.tile0 // sspec.subtile0 * sspec.tile1 // sspec.subtile1
                bits_per_tile = nspec.meta_bits * num_subtiles
                cdtype = torch.uint8 if bits_per_tile <= 8 else torch.uint16 if bits_per_tile <= 16 else torch.uint32
                return torch.uint8, cdtype, cdiv(8 * torch.dtype.itemsize, nspec.meta_bits), 1
        nspec = sspec.scale if name == "scale" else sspec.zero if name == "zero" else sspec.tensor if name == "tensor" else nspec
        if nspec.torch_dtype is not None:
            adtype = nspec.torch_dtype
        else:
            adtype = (
                torch.uint8
                if nspec.bits <= 8
                else torch.float32
                if nspec.is_float
                else torch.uint32
                if nspec.bits > 16
                else torch.uint16
            )
        if sspec.is_tensor and name == "output":
            return adtype, adtype, 1, 1
        cdtype = torch.uint8 if nspec.bits < 8 else adtype
        pratio = 1 if cdtype.is_floating_point else cdiv(8 * cdtype.itemsize, nspec.bits)
        return adtype, cdtype, pratio, sratio

    def get_shapeinfo(self, name: str, tile0: int = 1, tile1: int = 1, subtile0: int = 1, subtile1: int = 1) -> ShapeInfo:
        """Get the shapes, dtypes, and packing ratio for a tensor."""
        # oshape, odtype = reshaped input tensor shape and dtype
        # 1D scales have sparsity and packing in dim1
        if name not in self.shape_info:
            oshape, odtype = self.input.size(), self.input.dtype
            adtype, cdtype, pratio, sratio = self.get_torch_dtypes(name)
            ashape = cshape = oshape
            if name == "output":
                cshape = torch.Size((ashape[0], cdiv(ashape[1], pratio * sratio)))
            if name == "mask":
                cshape = torch.Size((ashape[0], cdiv(ashape[1], 8)))
            if name == "tensor" or name in ("scale", "zero") and self.dtype.is_tensor:
                ashape = cshape = torch.Size((1,))
            if name in ("scale", "zero") and not self.dtype.is_tensor:
                ashape = torch.Size((cdiv(ashape[0], tile0), cdiv(ashape[1], tile1)))
                cshape = torch.Size((ashape[0], cdiv(ashape[1], pratio)))
            if name == "meta":
                ashape = torch.Size((cdiv(ashape[0], subtile0), cdiv(ashape[1], subtile1)))
                cshape = torch.Size((ashape[0], cdiv(ashape[1], pratio)))
            self.shape_info[name] = ShapeInfo(oshape, odtype, ashape, adtype, cshape, cdtype, pratio, sratio)
        return self.shape_info[name]

    def precast(self):
        """Prior to cast, set up tensors."""
        if self.dtype.is_unscaled:
            self.output = torch.zeros_like(self.input, dtype=self.get_shapeinfo("output").get_info(Modes.cast)[1])
            return
        if self.input.ndim > 4:
            raise NotImplementedError("More than 4D shapes are not supported.")
        if self.dtype.is_tensor and Modes.cast != CastMode.COMPRESS:
            # no need for reshape, just create the tensors
            self.output = torch.zeros_like(self.input, dtype=self.get_shapeinfo("output").get_info(Modes.cast)[1])
            if Modes.cast != CastMode.VIRTUAL:
                _, dtype = self.get_shapeinfo("scale").get_info(Modes.cast)
                self.scale = torch.zeros(torch.Size((1,)), dtype=dtype, device=self.original_device)
                if self.dtype.nspec.is_uint:
                    _, dtype = self.get_shapeinfo("zero").get_info(Modes.cast)
                    self.zero = torch.zeros(torch.Size((1,)), dtype=dtype, device=self.original_device)
        else:
            # reshape the input tensor to 2D if it is not already
            if self.input.ndim == 1:
                self.input = rearrange(self.input, "n -> 1 n")
                self.shape_transforms.append(("output", "rearrange", "1 n -> n", {}))
            elif self.input.ndim == 3:
                self.input = rearrange(self.input, "b s h -> (b s) h")
                self.shape_transforms.append(("output", "rearrange", "(b s) n -> b s n", {"b": self.original_shape[0]}))
            elif self.input.ndim == 4:
                if self.transpose_scale:
                    self.input = rearrange(self.input, "b c h w -> b (c h w)", {})
                    self.shape_transforms.append(
                        (
                            "output",
                            "rearrange",
                            "b (c h w) -> b c h w",
                            {"c": self.original_shape[1], "h": self.original_shape[2], "w": self.original_shape[3]},
                        )
                    )
                else:
                    self.input = rearrange(self.input, "b c h w -> (b h w) c")
                    self.shape_transforms.append(
                        (
                            "output",
                            "rearrange",
                            "(b h w) c -> b c h w",
                            {"b": self.original_shape[0], "h": self.original_shape[2], "w": self.original_shape[3]},
                        )
                    )
            if self.transpose_scale:
                self.input = rearrange(self.input, "n c -> c n")
            size0, size1 = self.input.size()
            sspec = self.dtype.sspec
            tile0, tile1, subtile0, subtile1, _, _ = sspec.get_tile_info(size0, size1, Modes.cast == CastMode.VIRTUAL)

            if Modes.compute == ComputeMode.TORCH:
                if size0 % tile0 != 0 or size1 % tile1 != 0:
                    # need to pad
                    pad = (tile0 - size0 % tile0, tile1 - size1 % tile1)
                    self.input = torch.nn.functional.pad(self.input, (0, pad[1], 0, pad[0]), mode="constant", value=0)
                if sspec.is_2d:
                    self.input = rearrange(self.input, "(n nt) (c ct) -> (n c) (nt ct)", nt=tile0, ct=tile1)
                else:
                    self.input = rearrange(self.input, "n (c ct) -> (n c) ct", ct=tile1)

            # get the shape information for the tensor, scale, zero, meta, and mask, and create them
            shape, dtype = self.get_shapeinfo("output", tile0, tile1, subtile0, subtile1).get_info(Modes.cast)
            self.output = torch.zeros(shape, dtype=dtype, device=self.original_device)
            if Modes.cast != CastMode.VIRTUAL:
                shape, dtype = self.get_shapeinfo("scale", tile0, tile1, subtile0, subtile1).get_info(Modes.cast)
                self.scale = torch.empty(shape, dtype=dtype, device=self.original_device)
                if self.dtype.nspec.is_uint:
                    shape, dtype = self.get_shapeinfo("zero", tile0, tile1, subtile0, subtile1).get_info(Modes.cast)
                    self.zero = torch.empty(shape, dtype=dtype, device=self.original_device)
                if self.dtype.is_sparse:
                    shape, dtype = self.get_shapeinfo("mask", tile0, tile1, subtile0, subtile1).get_info(Modes.cast)
                    self.mask = torch.zeros(shape, dtype=dtype, device=self.original_device)
                if self.dtype.is_codebook:
                    shape, dtype = self.get_shapeinfo("meta", tile0, tile1, subtile0, subtile1).get_info(Modes.cast)
                    self.meta = torch.zeros(shape, dtype=dtype, device=self.original_device)

    def postcast(self):
        """Reshape the tensors to match the original shape."""
        if self.dtype.is_unscaled or self.dtype.is_tensor and Modes.cast != CastMode.COMPRESS:
            return
        if Modes.cast == CastMode.VIRTUAL:
            if self.transpose_scale:
                self.output = self.output.transpose(0, 1)
            if len(self.original_shape) == 4 and not self.transpose_scale:
                # input channels are in the last dimension, so we need to transpose back
                shape = [self.original_shape[0], self.original_shape[3], self.original_shape[2], self.original_shape[1]]
                self.output = self.output.reshape(shape).transpose(1, -1).contiguous()
            else:
                self.output = self.output.reshape(self.original_shape)
        elif self.transpose_scale:
            self.output = self.output.transpose(0, 1)
            if self.scale is not None:
                self.scale = self.scale.transpose(0, 1)
                if self.zero is not None:
                    self.zero = self.zero.transpose(0, 1)
            if self.meta is not None:
                self.meta = self.meta.transpose(0, 1)
            if self.mask is not None:
                self.mask = self.mask.transpose(0, 1)

    def update(self, **kwargs):
        """Update the tensors during cast."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                if Modes.cast == CastMode.VIRTUAL:
                    setattr(self, key, value)
                else:
                    assert hasattr(self, key) and isinstance(getattr(self, key), torch.Tensor)
                    getattr(self, key).copy_(value)
        self.quantized = True
