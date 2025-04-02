<!-- markdownlint-disable MD033 MD041 -->

# TensorCast Documentation

> *The codebase is changing as fixes are made and testing continues, which makes the documentation lag
behind a bit.*

The documentation for TensorCast is in the following files in this directory.
Some of these documents are complete, others barely started, the rest somewhere in between.

While the attention document is not even started yet, the slide deck for the low precision
attention project in TensorCast is there right below.

|document|topic|
|:------:|:----------|
|[Overview](./overview.md)| General information about the TensorCast package|
|[API](./api.md)| Top level interface|
|[Configuration Modes](./modes.md)| How rounding, casting, scale selection, and operation implementations are managed |
|[tcast.Tensor](./shapes.md)| Where the scales, masks, and metadata live|
|[Casting Engines](./cast.md)| TritonCast and TorchCast|
|[Number Specification](./number.md)| How number formats are represented|
|[Scaling Specification](./scale.md)| How scaling is represented|
|[Sparsity](./sparse.md)| Specifying structured sparsity|
|[Data Types](./datatype.md)| DataType = NumberSpec + ScaleSpec|
|[Codebooks](./codebook.md)| Lookup table quantization, a form of number specification|
|[Attention](./attention.md)| The attention kernel low precision API and configuration|
|[Attention Example](./attn_example.md)| Small example of attn interface and use|
|[Low Precision Attention](./slides.md)| Presentation on Low Precision Attention Quantization Package|

---

[Documentation](./README.md)
</br>
[TensorCast Home](../README.md)
