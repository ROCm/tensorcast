<!-- markdownlint-disable MD033 MD041 -->

# TorchCast and TritonCast

> TODO(ericd): *needs proofreading*

There are currently two separate implementations ("engines") of casting,
TorchCast](../tcast/torchcast.py) and [TritonCast](../tcast/tritoncast.py).  They implement
functionality in PyTorch and Triton, respectively.
The medium term intent is to have most features implemented in both places, which provides unit test
advatages and gives Triton a chance to gain more acceptance.  Longer term, there may be some promise
in making TensorCast more of a fine-grained package of functions that can minimize both functionality
and dependencies based on workflows.  If that happens, there could be a PyTorch-free version, a
Triton-free version, even a Python-free version, with base libraries for C++/CPU, C++/GPU, Triton,
PyTorch, NPU, etc. with datatype "typesets" that isolate platform-specific datatypes and sparsity to
make the lightest-weight quantization "snippets" with the least friction to adoption.

## Common Interfaces

Both TritonCast and TorchCast engines will have a straightforward interface, generally called from
the API cast and upcast functions.  `Cast` will receive a `torch.Tensor` and a `tcast.DataType`,
while `upcast` will receive a `tcast.Tensor` and a `torch.dtype`.  Before the cast or upcast,
thre will be a call to the engine's `supports` function. Then there wiil be a call to
the `tcast.Tensor.precast` function, which will create properly padded/permuted/shaped/reshaped
input, output, and scale tensors.  The `cast` call for each engine will be run, with the `tcast.Tensor`
tensors being updated, then `postcast` will get the newly created tensors into their final shapes.
For *virtual* cast, the output tensor will be upcast and rescaled with all other tensors deleted.
For other cast methods, the input will be deleted and all other tensors kept.

## Focus

- ### TorchCast Focus

  - **completeness**
    - features (most new functionality will appear here first)
  - **flexibility**
    - test bed for interfaces, so that TritonCast can adopt the best practices only
    - fewer limitations (on tile sizes, for example) so TritonCast can focus on the most common
  - **simplicity**
    - be as straight forward as possible and isolate more complex code in other functions
    - focus on PTQ, LoRA, QAT (but not pretraining)
    - focus on *virtual* cast, except for final casting of PTQ and QAT weights

- ### TritonCast Focus

  - **performance**
    - fused kernels
    - *actual* and *compress* cast modes
    - performance tuning
    - simplified integration into kernels outside of TensorCast
    - coarse grained performance (e.g. total pretraining cost leveraging low precision)
  - **convergence**
    - augmentation of kernels with advanced quantization technique
    - focus on pretraining workflows (which need bith convergence and performance)
    - variations in tile sizing and other tweaks to existing datatypes
    - novel datatypes such as implicit codebooks
  - **configurability**
    - There are many combination of options such as datatype, scale type, scale shape, and
    incoherence processing that will need to be evaluated and pruned from the recipe search space,
    so making that as controllable and invisible as possible is a focus
  - **stability**
    - this will be closest to product development
    - unit testing will be prioritized for use cases in TritonCast

## TorchCast Engine

Because PyTorch ops are not well suited for tiles that are not complete channels or tensors,
more reshaping and permutation is required (especially for 2D tiles and 4D convolutional tensors)
than for custom kernels.  Compression requires even more kernel fusion and shape manipulation.
We can expect `torch.compile` to pick up some of that, but that is beyond the scope of TensorCast,
so `TorchCast` will focus entirely on *virtual* casting.

## TritonCast Engine

Initial supported functionality will be for square 2D and 1D scales with float scale factors and float
data (with FP8 leading the way).  The main emphasis is the Triton kernels that make up the interface
between `tcast` and the configuration capability that will begin with the
[attention kernel](./attention.md) support, then move to fused GEMM kernels with comprehensive
embedded quantization, casting, and processing the tensors to reduce the quantization
error that comes from outlier values for very low precision datatypes.

The new Triton emphasis in TensorCast is expected to grow to where *actual* and *compress* modes
are exclusively supported in TritonCast, at least for 2025.  Although *virtual* ("fake") quantization
is not the immediate need (and can be supported by a fused cast+upcast kernel), it may be desirable
to have the same level of support for *virtual* in TritonCast as in TorchCast.

---

[Documentation](./README.md)
</br>
[Testing](../tests/README.md)
</br>
[Examples](../examples/README.md)
</br>
[TensorCast Home](../README.md)
