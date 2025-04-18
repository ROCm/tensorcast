<!-- markdownlint-disable MD033 MD041 -->

# Scaling

Scaling is specified by a [ScaleSpec](../tcast/scale.py) called *sspec*.  Scale types are specified by
[NumberSpec](../tcast/number.py) called *nspec*.  The two are combined to define a
[DataType](../tcast/datatype.py).

Scaling specifications differentiate between tensor scales, channel scales, tile scales, and individual scales
(i.e. value exponents).  A *tile* in TensorCast is also known as a "block" or "group", but here the term "tile"
is used, matching the Microxcaling
([paper](https://arxiv.org/pdf/2310.10537.pdf),
[github](https://github.com/microsoft/microxcaling.git))
terminology from Microsoft, who co-developed the
[OCP MX formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
and published earlier work with
[MSFP](https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf),
a block floating point format.

Microsoft also introduced *subtile* in
[With Shared Microexponents, A Little Shifting Goes a Long Way](https://arxiv.org/abs/2302.08007),
where the initial version of MX (without individual exponents) shared one or more scale offset bits to preserve
precision.  These datatypes include MX9, MX6, and MX4 (also known as BFP Prime).  This scaling is supported
in TensorCast in the form of [implicit codebooks](./codebook.md), but only as a storage datatype that is upcast
to MXFP compute datatypes.  The subtile in the scale is specific to the codebook.

In TensorCast, tensor, channel, tile, and individual scales are supported, but not all combinations are supported.

<br></br>

## Types of Scales and Data

The data that we are quantizing (defined in part by a `NumberSpec` (nspec) can be unscaled (such as `bfloat16`),
in which
case there is no scale data.  Only floating point data is allowed to be unscaled. Otherwise the nspec can be
float, int, or uint.  The scale(s) can be float, exponent, or int.  The output and scale tensors are stored in a
[tcast.Tensor](./tensor.md) for both **actual** and **compressed** cast modes.  For **virtual** ("fake") cast mode,
the scale data is not stored; only an unscaled float `torch.Tensor` with the torch.dtype of the input tensor is
returned from the cast.

### Storage

A scaled and quantized tensor has a datatype, and up to 5 scaling tensors, all stored in `tcast.Tensor` along with
the unquantized input data and the quantized output data.  These scaling tensors are ***scale***, ***zero***,
***tenscale***, ***meta***, and ***mask***.  The ***meta*** tensor identifies codebooks used for a tile or subtile,
and is described in the [codebook](./codebook.md) documentation.  The ***mask*** tensor is for sparsity, described
in the [sparse](./sparse.md) documentation.  The ***scale*** tensor contains the scale for tensor-scaled or tile-scaled
data.  The ***zero*** tensor is only used for unsigned integer quantized output, where it stores an integer or
floating point offset, also called the "zero point".  The ***tenscale*** tensor is used for the tensor scale only
when it is combined with another scale scope.  In that case, the scale spec property `is_multiscale` is `True`.

### Tensor scaling

A tensor scale, uncombined with a channel or tile scale, is typically a scalar float32.  In this case,
it is stored as a single-element tensor as the ***scale*** attribute of `tcast.Tensor`.

When the tensor scale is a float type with float data, it is generally computed such that the highest magnitude
value maps to the maximum (or minimum) value representable in the target *nspec*:

``` python
def get_float_tensor_scale(x: torch.Tensor, dtype: DataType) -> torch.Tensor:
    nspec, sspec, snspec = dtype.nspec, dtype.sspec, dtype.sspec.scale
    assert nspec.is_float and (snspec.is_float or snspec.is_exponent)
    if snspec.is_float: # scale
        # unquantized scale factor in float32
        scale = nspec.finfo.maxfloat / x.abs().max().float()
        # unscaled cast on an unscaled dtype
        scale = tcast.cast(scale, DataType(nspec=snspec))
    else:
        assert snspec.is_exponent, "Only float or exponent scales for tensor scaling"
        maxexp = tcast.TorchCast.get_exponents(x).max()
        # e8m0 is actual exponent + scale bias - nspec emax
        scale = maxexp - nspec.emax + snspec.bias).to(torch.uint8)
    return scale
```

### Tile scaling

A tile is a one or two dimensional slice of a tensor, where each slice has a shared scale factor.

Unsigned integer data is asymmetric  around zero, meaning that there is a zero point in addition to the scale factor.
The scale factor is some form of float, and the zero point can be float or int (the latter guarantees precise 0.0
representation for reduced precision scales, with a potential increase in quantization noise).

Signed integer is symmetric around zero, dropping the highest magnitude negative value to avoid bias in the
quantization. The scale number spec can be either a float or an exponent.

Floating point data has an inherent individual scale, but the tensor/channel/tile is in addition to that.  The scale
can be float or exponent, just as with integer data.

A tile spec describes the size of the tile (T) and optional subtile (S) for one dimension.  It also has the
optional sparsity description N of M, with N being the post-sparsity size and M being the pre-sparsity size
(sparsity block size or SBS in ancient Microsoft scrolls).  There can be zero, one or two tile specs in a scale
spec, for tensor scaling, 1D tile scaling, and 2D tile scaling respectively.

A channel scale is a special case of a tile.  Instead of the tile size T being a power of two, it can be 0 or
or omitted altogether, which denotes channel scaling in that dimension.  A 2D tile description can contain up to one
channel scale.

### Sparsity

Although sparsity is not usually thought of as being a part of scaling, it is in TensorCast.  One reason is that
when structured sparsity is applied, 2:4 for example, the 50% reduction in weights effectively doubles the tile size
if the sparsfied dimension. Another reason is that the scaling and sparsity are interdependent w/r/t bits per value
and compression, and it is easier to have that information in one place.

Sparsity is encoded as a string by adding "**n***N***m***M*" to a scale segment to define "N" as the dense values
and "M" as the size of the sparsity block.  More about sparsity is [here](./sparse.md).

## Scale Specification String Encoding

`ScaleSpec` can be defined via parameters to the class initializer, but it is perhaps more convenient to encode
the information in a string (as is done with `NumberSpec`).

The `NumberSpec` components of a scale spec are the ***scale***, optional ***zero***, and optional ***tenscale***.
Since hierarchical tensor scaling is not supported for asymmetric (uint) tile scaling, we need either one or two
number specs.  By design, number specs have names that have no underscore ("float8_" prefix is redundant, thus
removed), we can use underscores to partition the string descriptor into number specs and tile specs.

Tile descriptors (1 for 1D scaling, 2 for 2D scaling) describe the tile and subtile sizes (T and S) and the sparsity
N and M.  There are 0, 1, or 2 tile descriptors following 1 or 2 number specs.  Overall, we can have anywhere from 2
to 4 underscore-separated segments of nspec and tile info.  If there is one tile segment, it is for the last dimension,
if there are two they are for the next to last and last dimension.  The cast function allows a transpose to be applied
to the scale, swapping the tile specs.  Both "t" and "t0" can be used to specify channel scaled dimension.

Examples in tables show various combinations:

| scale spec             | description                                                                                    |
|:----------------------:|:-----------------------------------------------------------------------------------------------|
|float32                 |tensor scaled, fp32 scale, either fp or int data (not uint)                                     |
|float32_t0              |channel scaled in last dim, fp32 scale, fp or int data                                          |
|e8m0_t32                |with nspec e3m2fnuz, makes MXFP6E3 OCP, tile 32 in last dim                                     |
|float16_int8_t16n2m4    |asymmetric (uint) data, fp16 scale, int8 zero point, 2:4 sparsity and 1x16 tile                 |
|e8m0_t16m4_t16s4        |implicit codebook, exponent scale on 16x16 tile, codebook selection on 4x4 subtile              |
|bfloat16_bfloat16_t0_t32|asymmetric (uint) data, bf16 scale and zero point, channelx32 tile                              |
|e4m3_float32_t16_t16    |with e4m3fn data, this is similar to NVDA MXFP8E4 (non-OCP), fp32 tensor scale on fp7 16x16 tile|

This table shows the segments and the types of the scale tensors in the `tcast.Tensor` after actual or compressed cast.

|seg 1   |seg 2   |seg 3   |seg 4   |data   |scale   |zero  |tenscale |dim -2 |dim -1  | notes                                 |
|:------:|:------:|:------:|:------:|------:|:-----:|:----:|:--------:|:-----:|:------:|:--------------------------------------|
|float32 |        |        |        |fp/int |fp32   |      |          |       |        |fp data fp tensor scale                |
|float32 |t0      |        |        |fp/int |       |      |          |       |channel |int8 data dim -1 fp channel            |
|e8m0    |t32     |        |        |fp/int |       |      |          |       |        |fp data, fp16 scale 1x32 tile          |
|float16 |int8    |t16n2m4 |        |uint   |fp16   | int8 |          |       |t16 2:4 |sparse uint4 tile 16                   |
|e8m0    |t16s4   |t16s4   |        |icb    |e8m0   |      |          |t16 s4 |t16 s4  |codebook 16x16 tile, 4x4 subtile       |
|bfloat16|bfloat16|t0      |t32     |uint   |       |      |          |channel|t32     |uint data, bf16 scale, zero, channelx32|
|e4m3    |float32 |t16     |t16     |e4m3fn |e4m3fn |      |fp32      |t16    |t16     |NVDA MXFP8 16x16 w/ fp32 tensor        |

---

[Documentation](./README.md)
</br>
[Testing](../tests/README.md)
</br>
[Examples](../examples/README.md)
</br>
[TensorCast Home](../README.md)
