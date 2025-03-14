<!-- markdownlint-disable MD033 MD041 -->

# tcast.Tensor: Data, Scales, and Metadata

The [tcast.Tensor](./tcast/tensor.py) is a place to store scales, metadata, sparsity masks, and
most importantly, *hide as much of the padding/reshaping/broadcasting ugliness as possible.*

Tensor shapes, and their accompanying scale, metadata, and sparsity mask shapes, are somewhat
complicated. There are compressed (packed) shapes; shapes that are temporary to aid in scale
discovery and sparsity, shapes targeted to specific checkpoint storage or backend-specific
ordering; and probably more.  This document spells out at least some of the details.

Complexity comes into play in the dimension along which the scale tile or sparsity tile exists,
with codebook subtiles, and particularly with two dimensional tiles.  As seen in the
[scaling documentation](./scale.md), a vector tile may be along any dimension in the tensor
(defaulting to the last dimension).  Tensors with matrix tiles need to be padded, permuted and
reshaped in order to use torch operators to find and apply scales.

## Compression

In order to benefit from the storage and data movement advantages of small storage datatypes,
hey need to be compressed (say, into 4 bits instead of fake quantized to 16 or 32 bits).
TensorCast manages this through the `tcast.Tensor` wrapper than include tensor data, the datatype,
scales, and other metadata.  The tensors within this wrapper are:

* input (unquantized data that gets deleted once the cast is completed)
* scale (float or exponent scale factors)
* zero (float or int zero point offset for asymetrical int-quantized data)
* meta (metadata, currently limited to use in codebook datatypes)
* mask (1D sparsity mask for tile-scaled structural sparsity)

Each of these tensors has a shape and a torch.dtype for uncompressed and compressed versions.
In some cases, the compressed and uncompressed versions have the same shape but a different
torch.dtype.

## Input

The tensor that holds the actual weight/activation/gradient maintains its original shape an
torch dtype in uncompressed, unquantized form. It can be reshaped prior to the cast call,
however, to make it easier for the datatype's scale dimension(s) to be defined and applied.
If the scale dimension has a stride of 1, the process should be faster.

Compression is another thing.  A 4 bit datatype can be packed into 8 bit elements;
an MXFP8/6/4 can be stored in `torch.float8_e4m3fn` (or `torch.float8_e5m2`) with an exponent
scale.  The shape remains the same unless packed.  The torch dtype may change.  Sparsity introduces
the opportunity for compression while increasing the scale tile size proportionally to the sparsity
level.  TensorCast aims for good compression while maintaining alignments that are conducive to fast
compression and decompression kernels.

## Output

This is where the quantized and possibly sparse output is stored.  For virtual castmode, the output is
the same datatype as the input, and it is not scaled.  For actual castmode, the output is in the target
datatype if possible (standard torch.dtype), otherwise it is upcast to the next smallest torch.dtype.  For
example, e4m3fnuz would cast to `torch.float8_e4m3fnuz`.  The fp6 and fp4 types, e3m2fnuz, e2m3fnuz,
and e2m1fnuz would also cast to `torch.float8_e4m3fnuz`.  All of the representable values in those types
can be represented in fp8 without further scaling.  Even though "actual" mode is not compressed, just the
datatype change compresses by 2x from bfloat16 and 4x from float32.

With compressed mode, the fp4 would be packed into uint8, two values per byte. Unsigned and signed int
datatypes would pack in a similar matter.  For two bit values (probably weights), we get 4 values per byte.
Codebook datatypes with 16 mappings would be stored as uint8 in actual mode or compressed mode, but the
latter mode would pack two uint4 values into each byte for 2x further compression over the fp8 compute
type.  Packing and unpacking 6 and 3 bit values is not planned, but if a feasible performant method is
available, it will go into the schedule.

With 25% and 50% sparsity, the pruned values would be stored as zero for virtual or actual cast; the
difference would be the tensor datatype.  With compression, we get 4x or 2x compression by omitting the
pruned values in addition to reduction due to the input and output tensor torch.dtypes.   The downside
is that the mask is retained and must be used to reinstate the pruned zeros unless the sparsity is
supported in the GEMM hardware.

## Scale and Zero Point

The uncompressed scale tensor has a shape based on the data tensor, but with the scale dimension(s)
being divided by the tile size in the dimension(s).  The compressed shape is the same unless multiple
values are packed into a single tensor element. In that case, the scale dimension is further reduced.
For matrix scale tiles, only one of the two dimensions is reduced in size for the packing.

The zero tensor has the same uncompressed shape as the scale tensor, but may have a different
torch dtype.  The compressed zero point may contain more than one value per `torch.uint8`,
thus may differ from the compressed scale shape if it does not have the same packing ratio.

## Tenscale

This is the tensor scale in the case of multiscale, which combines a tensor scale with tile scaling.
This is a scalar value that is stored as a 16 or 32 bit float.  If the scale is *only* a tensor scale,
it will be store in the scale tensor and the tenscale will be `None`.

## Meta

Metadata is currently specific to codebooks, and can be per tile or per subtile.  The uncompressed
shape for tiles is the same as the scale shape.  For subtile codebooks, the scaled dimension is
multiplied by the number of subtiles in each tile in the appropriate dimension(s).

Compression of metadata is a bit more complex.  Since PyTorch does not support unsigned int tensors
larger than uint8, metadata is always `torch.uint8` (who wants two's complement getting involved).
The limit on meta bits per subtile is 8, so in the worst case the compressed shape would be the
same as the uncompressed shape.  Otherwise the bits will be packed into
`(meta_bits * num_subtiles + 7) // 8` bytes, and the scale dimension will be that number times
the number of tiles.

## Mask

The uncompressed sparsity mask is the same shape as the uncompressed data tensor, but it's type
is `torch.bool`.  Since BoolTensor is 8 bits per value, the compressed sparse dimension will be the
uncompressed size divided by 8.

> *Matrix (2D) sparsity is not supported. There are two challenges here.  One is that a 2D mask has
to have the same number of dense values in both axes, which a limiting constraint.  The other is
that you can't compress in two axes.*

---

[Documentation](./README.md)
</br>
[TensorCast Home](../README.md)
