<!-- markdownlint-disable MD033 MD041 -->

# Sparsity

A simple sparsity function is provided that preserves the N highest magnitude values from M values in a tile along
the specified dimension.  In practical hardware terms, the dimension would be the inner dimension of a GEMM
and M and N would be mandated by the hardware platform.  Clearly, sparsity has many variations, and magnitude
may not be the best qualifier, but this is a start.

> Matrix (2D) sparsity is not supported. There are two challenges here.  One is that a 2D mask has to have the same number of dense values in both axes, which a limiting constraint.  The other is that you can't compress in two axes.

Sparsity is specified as part of ScaleSpec, through the parameters `sparse_n` and `sparse_m`.  In the ScaleSpec
definition string, these are specified in the tile spec, which defines the tile size, subtile size, n and m.
An example of MXFP4 with OCP scaling would be `e2m1fnuz` for the NumberSpec and "e8m0_t32" for the ScaleSpec.
If we wanted 50% sparse along the tile dimension in subtiles of 4 values, the code would be "e8m0_t32n2m4".

```python
import tcast

def cast_sparse_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = tcast.datatype("e2m1fnuz", "e8m0_t32n2m4")
    tensor = tcast.cast(x, dtype, castmode="compress") # tcast.Tensor is returned
    return tensor.output, tensor.mask
```

By using the "compress" mode, we can reduce the size of the sparsity mask by a factor of 8, packing eight
one-bit values into a `torch.uint8` tensor. We can also reduce the size of the quantized and sparsified
output tensor by a factor of 2 just storing the dense values, and by a factor of the size of the unquantized
datatype divided by the size of the quantized datatype.  For MXFP4, the quantized datatype would also
be stored as `torch.uint8` with two fp4 values packed into each byte, which amounts to a 4x reduction from
unquantized `torch.bfloat16`.

Using the "virtual" mode (a/k/a fake quantization), there would be no reduction for the data (zeros would
replace the pruned values, and the tensor would remain `torch.bfloat16` and the sparsity mask would be the
same shape as the input tensor, with each bit stored as `torch.bool`, which is 8 bits per value.

---

[Documentation](./README.md)
</br>
[Testing](../tests/README.md)
</br>
[Examples](../examples/README.md)
</br>
[TensorCast Home](../README.md)
