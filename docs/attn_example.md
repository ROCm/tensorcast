<!-- markdownlint-disable MD033 MD041 -->

# Low Precision Example

> TODO(ericd): *in progress*

This is a description of the [low precision example](../examples/low_precision_interface.py).

This example shows a matrix multiply operation using the TensorCast low precision attention interface,
both configuration (PyTorch) and procedural (Triton).

## Options available through configuration

- Each of the tensors that are inputs to GEMMs (just Q and K in this example, but Q, K, V, P, DS, DO in attention)
can have any of the standard FP8 datatypes, or None if we are not to quantize that tensor.
- The scale factor can be float32, float16, bfloat16, or, if set to None, the same dtype as the tensor being scaled.
- The block size can be set to 8x8, 16x16, 32x32, 64x64, or 128x128, such that the size cannot exceed the BLOCK_M,
BLOCK_N, or BLOCK_D that defined the input and output shapes of each program (Q is MxD, K<sup>T</sup> is DxN, and Out
is MxN).
- Incoherence processing can be applied either to both input tensors or neither.  If one of the tensors is unquantized
we still must apply ICP to it.  The imatrix is created through the configuration interface, with a dtype of float32,
float16, pr bfloat16.  ICP is modeled using matrix multiplication (there are musch more performant methods), and the
inputs to those transforms can be cast to float32 or the imatrix can be cast to the input dtype (the former is expected
to get better resuts).  The imatrix dtype is set during creation using the PyTorch interface, while the size is the
block size (also BLOCK_Q), and the float32 cast is accessed inside the Triton kernel interface,
- Stochastic rounding
- Outlier insertion, for determining how extreme the outliers need to be to require ICP

### Means of configuration

The tcast API overloads `tcast.configuration` method.  If the type of a single parameter is `int`, the configuration
is created from the uint32 code that is passed as LPCODE through the Triton kernels.  This is something use if you
already know what it is (after creating a confuguration through some other means).  If it is `pathlib.Path`, then
a configuration is built from a JSON file (an example can be found [here](../tcast/tests/config_attrs.json)).  If
it is `str`, the configuration is taken from a "shortcut", one of a few hundred predefined configurations defined
in [LPConfig](./tcast/config.py).

These three methods are ustized in the example by using one of the `--code`, `--json`, or `--shortcut` command line
args.  The other method is to use the remaining config args (and others in a full attention configuration):

- `--block_size`
- `--scale_dtype`
- `--q_dtype`
- `--k_dtype`
- `--icp_qk`
- `--icp_fp32`

### Overall limitations

- The size of the `tl.dot`inputs and output must not exceed BLOCK_Q.  This is due to how descaled values are
accumulated.

### Limitations in this example

- Block size is square.  Someday soon, 1D blocks will be allowed, but the descale operation then becomes a GEMM itself,
and the backwards pass needs to requantize along the other axis, so 16-bit values must be saved for backward. as well.
- Other block dimensions (BLOCK_M, BLOCK_N, BLOCK_D) must be multiples of BLOCK_Q.  This sould be easy to fix with
masking.
- No performance tuning has been done.

## Options available through procedural interface

The main interface for incoherence processing and quantization is `lp.scale_and_quantize()`, where *lp* is
`import tcast.snippets as lp`.

There are many small functions that are based on the `LPCODE` (the complete configuration encoded as uint32) and
in some cases the index of the tensor (such as `lp.Q_INDEX`).

### Used in the example kernel

- `lp.enabled`: may be used for a `tl.static_assert()`; here it is used to run the code in baseline mode
- `lp.needs_quant(LPCODE, lp.K_INDEX)`: k has a dtype other than None (in this case some FP8 dtype)
- `lp.needs_quant_or_icp(LPCODE, lp.Q_INDEX)`: returns needs_sq, needs_quant, needs_icp, where *needs_icp*
also takes into account `K`, and *needs_sq* means that `scale_and_quantize` must be called for Q
- `lp.Q_INDEX` and `lp.K_INDEX` to inquire about independent quantization dtypes
- `lp.scale_and_quantize`: the primary interface to `tcast` through the attention interface

### A subset of other functions in lp

#### Using LPCODE as the only parameter

- `lp.roundmode(LPCODE)`: one of `lp.RMODE_ZERO`, `lp.RMODE_AWAY`, `lp.RMODE_EVEN`, `lp.RMODE_STOCHASTIC`
- `lp_scalemode(LPCODE)`: one of `lp.SMODE_FLOOR`, `lp.SMODE_CEIL`, `lp.SMODE_MIDMAX`, `lp.SMODE_OPTION3`,
`lp.SMODE_TOPBINADE`
  - only used when the scale is an exponent, such as OCP MX
- `lp_castmode(LPCODE)`: one of `lp.CMODE_VIRTUAL`, `lp.CMODE_ACTUAL`, `lp.CMODE_COMPRESS`
- `lp.scale_type(LPCODE)`: get the `tl.dtype` of the scale factors

#### Using both LPCODE and a tensor index code

- `lp.get_quant_type(LPCODE, TCODE)`: get the `tl.dtype` to quantize a specific tensor to
- `lp.number_mbits`, `lp.number_ebits`, `lp.number_emax`, `lp.number_emin`, `lp_number_maxval`: number specification
values, enabling quantization to FP8 when FP8 is not supported on a platform
  - currently works for standard fp8, but will be extended for MXFP and arbitrary datatypes for generic quantization

#### Quantization, scale determination, and utilities

- `lp.get_exponent`
- `lp.modify_exponent`
- `lp.round`
- `lp.quantize_float`
- `lp.get_shared_exponent`
- `lp.apply_incoherence`

---

[Documentation](./README.md)
</br>
[TensorCast Home](../README.md)
