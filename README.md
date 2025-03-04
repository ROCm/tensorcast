<!-- markdownlint-disable MD033 MD041 -->

# TensorCast (tcast)

> This is the *triton* branch of TensorCast which is just now starting to be populated (3/3/25).
> For working code, use the v2 branch for now.  This branch will begin to be functional beginning
> 3/21/25.

TensorCast is an open source PyTorch casting/quantization library. It is based on PyTorch 2.4+.

TensorCast exists for the purpose of exploring alternative scaling strategies and adaptive datatypes,
to enable researchers to provide evidence-based proposals for enhancing GEMM accelerator hardware.
With this new branch, development of performant Triton quantization methods and some additional
scaling methods is underway.

The scope of TensorCast is defining datatypes and converting tensors from one datatype to another.
While unscaled, tensor-scaled, and channel-scaled datatypes are supported, the emphasis is on
tile-scaled datatypes such as the OCP MX datatypes ([paper](https://arxiv.org/pdf/2310.10537.pdf),
[spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf))
and storage datatypes.

We attempt to keep things simple, but the high degree of customization can introduce complexity
at times.  We welcome feedback (and direct contribution) that will help us improve this package.

Contributors:

- Eric Dellinger [@ericd](mailto:eric.dellinger@amd.com)
- Alireza Khodamoradi [@alirezak](mailto:alireza.khodamoradi@amd.com)
