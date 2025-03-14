<!-- markdownlint-disable MD033 MD041 -->

# TensorCast Documentation

The documentation for TensorCast, such that it is, is in the following files in this directory.
Some of these documents are complete, others barely started, the rest somewhere in between.

- [Overview](./overview.md): General information about the TensorCast package
- [API](./api.md): Top level interface
- [Configuration Modes](./modes.md): How rounding, casting, scale selection, and operation implementations are managed
- [tcast.Tensor](./shapes.md): Where the scales, masks, and metadata live
- [Casting Engines](./cast.md): TritonCast and TorchCast
- [Number Specification](./number.md): How number formats are represented
- [Scaling Specification](./scale.md): How scaling is represented
- [Sparsity](./sparse.md): Specifying structured sparsity
- [Data Types](./datatype.md): DataType = NumberSpec + ScaleSpec
- [Codebooks](./codebook.md): Lookup table quantization, a form of number specification
- [Attention](./attention.md): The attention kernel low precision API and configuration

---
s
[Documentation](./README.md)
</br>
[TensorCast Home](../README.md)
