<!-- markdownlint-disable MD033 MD041 -->

# Scaling

Scaling is specified by a [ScaleSpec](../tcast/scale.py).
Scaling specifications differentiate between tensor scales, channel scales, tile scales, and individual scales
(i.e. value exponents).  A *tile* in TensorCast is also known as a "block" or "group", but here the term "tile" is used,
matching the Microxcaling ([paper](https://arxiv.org/pdf/2310.10537.pdf),
[github](https://github.com/microsoft/microxcaling.git)) terminology from Microsoft, who co-developed the
[OCP MX formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
and published earlier work with
[MSFP](https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf),
a block floating point format.

Microsoft also introduced *subtile* in
[With Shared Microexponents, A Little Shifting Goes a Long Way](https://arxiv.org/abs/2302.08007),
where the initial version of MX (without individual exponents) shared one or more scale offset bits to preserve
precision.  These datatypes include MX9, MX6, and MX4 (also known as BFP Prime).  This scaling is supported
in TensorCast in the form of [implicit codebooks](./codebook.md), but only as a storage datatype that is upcast
to MXFP compute datatypes.  The subtile in the scale is specific to the codebook

In TensorCast, tensor, channel, tile, and individual scales are supported, but tensor, channel, and tile scales cannot
currently be mixed.
<br></br>

## Types of Scales and Data

Scales can be either float or exponent, the latter being a biased unsigned int.  The OCP MX standard specifies
`E8M0`, in which the bias is 127.  TensorCast supports a generic `E<x>M0`, for x of 4, 5, 6, 7, 8, in which the
bias is calculated as 2<sup>n-1</sup> - 1.  Since a scale factor is unsigned, it should be possible to store a
floating point scale factor such as `E5M3` as an 8-bit value.

Unsigned integer data is generally asymmetric, meaning that there is a zero point in addition to the scale factor.
The scale factor is some form of float, and the zero point can be float or int (the latter guarantees precise 0.0
representation for reduced precision scales, with a potential increase in quantization noise).

Signed integer is generally symmetric around zero, dropping the highest magnitude negative value to avoid bias in the
quantization. The scale number spec can be either a float or an exponent. Support for unbalanced (asymmetric) scaling
is not planned.

Floating point data has an inherent individual scale, but the tensor/channel/tile is in addition to that.  The scale
can be float or exponent, just as with integer data.

Unscaled data, such as bfloat16, does not have a scale spec in the [datatype](./datatype.md), so the scale spec
is either a tensor, channel, or tile scale.  A tensor scale is simply a scalar (or two scalars for unsigned data),
and is specified by defining the number spec(s) for the scale with no tile specification.  A channel scale is a tile
scale, in which tile is the size of the tensor in the dimension of the scale, and is specified with a tile size of zero.
A tile scale has a tile size and the dimension of the tile.  The dimension defaults to -1 (the last dimension of the tensor).

A limitation in V1 as of now is that padding of tensors is not implemented, so the tensor size in the specified dimension must be a
multiple of the tile size.

### Scale Specification String Encoding

The components of a scale being scale number spec, optional zero point number spec, and optional tile spec, the string encoding
of a scale specification is the concatenation of the string encodings of the constituents, joined by underscores.

The number specs are defined above.  The tile scale is of the form "tXdY", where X is the size of the tile, a power of two between 2
and 1024 (or 0 for channel scaling) and Y is the dimension of the tile. If the dimension is -1, "dY" is omitted.  However, for channel
scaling the tile spec must be included, even if it is only "t0".

### ScaleSpec Implementation Notes

Until 2D tiles, subtiles, hierarchical scaling, and compression are implemented in V2, ScaleSpec is pretty simple.  There are methods
for reshaping the tensor to make PyTorch-based scale discovery a bit more straightforward.

---

[TensorCast Home](../README.md)
