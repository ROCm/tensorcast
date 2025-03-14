<!-- markdownlint-disable MD033 MD041 -->

# TensorCast Overview

> TODO(ericd): *this needs to be finished*

## Key Abstractions

TensorCast is based on the [datatype](./datatype.md), which comprises a [number specification](./number.md)
and a [scale specification](./scale.md) (which includes [structured sparsity](./sparse.md)).

There is a distinction between compute datatypes and storage datatypes.  The latter exist to reduce storage required
and data movement bandwidth and energy through reducing precision and dynamic range of the stored numbers.
These values are upcast to a compute format in the GEMM accelerator (formats that are supported in multipliers) at
the last possible moment for a given platform.  Storage formats are commonly used in inference, and are necessary
to mitigate memory-bound GEMMs and to emulate future hardware.

The primary operation is the `cast`, which is the conversion operator.  "Cast" is used in the sense of a
datatype conversion in programming languages such as C++, or in torch via `tensor.bfloat16()` or
`tensor.to(torch.float8_e4m3fn)`. The compute datatype for the current platform is always a `torch.Tensor` and
the appropriate `torch.dtype` or `triton.language.dtype`  The storage datatype (or future compute datatype) is
stored as a [tcast.Tensor](./shapes.md), which contains the scales, metadata, sparsity mask, and compressed values.

Tiled scaling (also *block* or *group*) is key to accomodate dynamic range using very low bit datatypes.  Tiles are
typically contiguous values along the inner dimension of a matrix multiplication, and compute hardware that supports
such datatypes efficiently makes use of the shared scale.  *Subtiles* are subdivisions of tiles, and while they must
all use the scale share across the tile, they can provide additional information shared by the values within the
subtile that may give more context to the upscaling of those values.  Additional information stored at the tile or
subtile level is called *metadata*. An example of what subtile metadata can do comes from Microsoft's Microxcaling
([paper](https://arxiv.org/pdf/2302.08007), [github](https://github.com/microsoft/microxcaling.git)), in which a
metadata bit shared by values in a subtile can indicate that the values' exponent is actually one less than the shared
exponent.  (This paper is the source of the TensorCast nomenclature of *tile* and *subtile*.)

Tiles and subtiles can be two dimensional (typically square).  This is an approach to mitigating the
*transpose problem* in training, which arises from the fact that the GEMMs in the backward pass are quantized along
a different dimension. With 1D scaling, this requires the weight and activation to be requantized for the backward
pass, and for the loss gradient to be quantized twice.  The overhead caused by this leads to 2D tiles, where the
scale factor does not need to change, and no requantization is needed, and the *actual* or *compress* storage
can be used to minimize data movement.  However, this increases the number of values sharing an exponent from 32
to 32x32=1024 (for OCP MX) which exacerbates denomralization and underflow for low precision formats.

---

[Documentation](./README.md)
</br>
[TensorCast Home](../README.md)
