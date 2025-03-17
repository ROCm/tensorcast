<!-- markdownlint-disable MD033 MD041 -->

# TensorCast (tcast)

> This is the *triton* branch of TensorCast which recently started to be populated (3/3/25).
> For working code, use the v2 branch for now.  This branch will ***begin*** to be functional
> no later than 3/21/25.

TensorCast is an open source PyTorch casting/quantization library. It is based on PyTorch 2.5+
and Triton 3.3+.

TensorCast exists for the purpose of exploring alternative scaling strategies and datatypes, including
adaptive datatypes, to enable researchers to provide evidence-based proposals and tooling for:

- identifying the best methods for leveraging low precision scaled datatypes in training and inference
- enhancing GEMM accelerator hardware
- developing performant Triton kernels

With this new branch, development of performant Triton quantization methods and some additional
scaling methods is underway.

The scope of TensorCast is defining datatypes and converting tensors from one datatype to another.
While unscaled, tensor-scaled, and channel-scaled datatypes are supported, the emphasis is on
tile-scaled datatypes such as the OCP MX datatypes ([paper](https://arxiv.org/pdf/2310.10537.pdf),
[spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf))
and storage datatypes.

We attempt to keep things simple, but the high degree of customization can introduce complexity
at times.  We welcome feedback (and direct contribution) that will help us improve this package.

## Status Updates

### 3/17/25

- added tests, including comparisons with MX (microxcaling) and tcast v2 branch
- restructured tests to make it easier to debug everything
- fixed a few bugs

### 3/14/25

The v2 branch is entirely in PyTorch, and has many of the features in development on the triton branch,
with the notable exceptions of Triton kernels, the attention configuration and Triton interface,
two dimensional scaling, and *actual* and *compress* casting modes.  What it does have works.

The triton branch is another matter.  It is almost all new, and the Big Test starts up next week.
Everything is checked in, current as of today.  The status is this:

### Documentation

Most of the documentation is written and pushed.  What is remaining is the final proofreading for some
of the files, more detail in the overview, and the attention doc needs to be written.   These are
in the tensorcast/docs directory.  :red_circle: indicates not done, :green_circle: indicates done,
:yellow_circle: indicates done, but needs proofreading, editing, or more content.

:green_circle: README.md: an index of the documents in the docs directory

:yellow_circle: api.md: top level interface functions, predefined datatypes, initialization

:red_circle: attention.md: probably the most important, but not written yet (next week for sure)

:orange_circle: overview.md: needs to be finished

:yellow_circle: cast.md: `tcast.TritonCast` and `tcast.TorchCast` cast mode/engine descriptions

:yellow_circle: number.md: `tcast.NumberSpec` number specifications

:green_circle: codebook.md: `tcast.Codebook` codebooks, subclass of number specs

:green_circle: scale.md: scale specifications

:yellow_circle: datatype.md: datatype specifications

:green_circle: sparse.md: sparsity support

:yellow_circle: modes.md: settings for rounding, scale selection, compute, and casting

:green_circle: shapes.md: describes the `tcast.Tensor` class

### `tcast` Code

The first colored circle indicates the state of the code being *finished*.  The second one indicates how close to *working* they are.  A question mark means it looks okay, but I just wrote/ported it, so I don't know.

:yellow_circle::yellow_circle: \_\_init\_\_.py ***refactored*** usually last to be finished

:green_circle::question: modules.py ***ported*** from the main branch

:green_circle::question: injector.py ***ported*** from the main branch

:green_circle::green_circle: tests/config_attrs.sh ***new*** example of configuration through environment variables

:green_circle::green_circle: tests/config_attrs.json ***new*** example of configuration through json configuration file

:green_circle::yellow_circle: config.py ***new*** configuration for fused Triton kernel quantization and incoherence which has had significant but incomplete testing

:green_circle::question: common.py ***new*** contains enumerations and global `Modes`

:green_circle::question: utils.py ***refactored** most of them probably work

:green_circle::yellow_circle: number.py ***refactored*** needs better testing

:green_circle::yellow_circle: scale.py ***refactored*** needs better testing

:green_circle::yellow_circle: datatype.py ***refactored*** very close, mostly depends on number and scale

:yellow_circle::red_circle: snippets.py ***new*** Triton interface casting and low level functions and constants

:red_circle::red_circle: tritoncast.py ***new*** Triton cast engine

:green_circle::yellow_circle: torchcast.py ***refactored*** PyTorch cast engine

:yellow_circle::red_circle: tensor.py ***new*** mostly need thorough work on shaping

### Unit Tests

The circles = written and RYG = pass<80%, 80%<=pass<100%, 100% pass

:green_circle::question: test_bfp_export.py ***ported*** not entirely sure what this does

:green_circle::question: test_bfp.py ***ported*** block floating point

:green_circle::yellow_circle: test_config.py ***new*** looks good, need to run some tests again

:green_circle::red_circle: test_incoherence.py ***new** mostly fails due to tolerances for GEMM outputs need adjustments

:green_circle::yellow_circle: test_torch.py ***ported*** compares direct torch cast with quantized cast

:green_circle::red_circle: test_mx.py ***ported*** compares Microxcaling repo with tcast for MXFP

### Examples

The examples from the main branch have been ported over, but not tested.

:green_circle::question: export_i8.py ***ported***

:green_circle::question: linear_bfp16.py ***ported***

:green_circle::question: model_custom.py ***ported***

:green_circle::question: model.inject.py ***ported***

## Contributors

- Eric Dellinger [@ericd](mailto:eric.dellinger@amd.com)
- Alireza Khodamoradi [@alirezak](mailto:alireza.khodamoradi@amd.com)

## Detailed TensorCast Documentation

There is more detail on TensorCast usage and underlying concepts, with links in an
[index here](./docs/README.md)  Not all of them are complete yet, bu some are, and can
help you get started.

## High Priority Features

Many of the features of TensorCast are present and working in the v2 branch of TensorCast.  The triton
branch involves considerable refactoring due to the nature of some key features (e.g. 2D scaling, actual
and compressed casting modes, Triton compute mode, and a new API for configuration and quantization
from within fused Triton kernels such as attention).

The priorities are aimed at the Flash Attention project, for MI300/325 FP8 datatypes with float32, float16,
of bfloat16 scales, and both vector and 2D matrix scale tiles.  This functionality will be delivered
incrementally, with initial functionality in late March.  The primary features should all work with
reasonable performance in early April.

### Phase 1A Features (March-April 2025)

- Cast modes: virtual (fake), actual (quantized and scaled, but not compressed)
- Round modes: even and ***stochastic***
- Scale types: floating point (float32, float16, bfloat16)
- Compute modes: PyTorch operations and Triton kernels, but perhaps not both, depending on the operation
- Oultlier mitigation: Randomized Walsh-Hadamard transforms for incoherence processing
- Attention low precision compute configuration
  - independent fp8 datatypes for q, k, v, p, ds, do (tl.fp8e5, tl.fp8e5b16, tl.fp8e4nv, tl.fp8e4b8)
  - scale types (tl.float32, tl.float16, tl.bfloat16, or match scale type to input tensor type)
  - independent incoherence processing for q/k/ds and p/v/do
  - scaling and incoherence processing to match tl.dot input sizes
  - incoherence processing matches input dtype or casts to float32
  - broad configuration flexibility
    - from environment variables
    - from json files
    - from an integer code
    - from predefined shortcuts
    - via parameters passed to an LPConfig dataclass.
  - strict checking of numerically consistent configuration combinations
- Attention Triton interfaces
  - configuration passed through kernels as a single uint32, extracting:
    - dtype for scale and each of q, k, v, p, ds, do
    - block size and dimension(s) to reduce for scale
  - single function call per tensor (coarse grained)
  - **qq, qscale = scale_and_quantize(q_ptr, imatrix, LPCODE, LP_Q_INDEX:, seed, offset, transpose)**
  - fine grained interfaces
    - round, quantize_float, get_shared_exponent, apply_incoherence
- Attention PyTorch interfaces
  - LPConfig (configure low precision with many options for experimentation)
  - LPConfig.get_imatrix (create Hadamard, Walsh-Hadamard, or randomized Walsh-Hadamard)
  - LPConfig.randomize_imatrix (re-randomize existing incoherence matrices)
- Broad datatype support (beyond just fp8 variants)

## Phase 1B Features (May 2025)

*Italicized* items are strech goals (lower priority)

- Compute modes: more coverage in Triton
- Unscaled floating point types
- Attention interface and low precision compute configuration
  - scaling can be a superset of tl.dot inputs
  - tl.dot can be an accumulated subblock of BLOCK_M, BLOCK_N, BLOCK_DMODEL
- *Float hierarchical scaling with e4m3 (per Nvidia, implemented as e5m3)*
- *Structured (2:4) sparsity*

## Phase 2A Features (June 2025)

Phase 2 is focused on OCP MXPF support in MI355.  The objective is to find recipes for using low precision
to accelerate pretraining, and find software/kernel enhancements that improve upon the hardware
implementations of OCP datatypes in performant ways to improve convergence.

- Cast modes: compress mode
  - Sparse outputs @ 50% size (pruned zeros are omitted)
  - Sparse masks @ 12.5% size (pack 8 bools into uint8)
- Scale modes (exponent selection): floor (OCP MX), ceil, midmax, option3 (Meta), and topbinade (Microsoft)
- Round modes: +nearest, +zero
- Compute modes: Triton fused GEMM and attention kernels

## Phase 2B Features (Q3 2025)

- Signed and unsigned integer specifications uint**K** and int**K** for K in [3, 16]
- Floating point e**X**m**Y***infnan* for **X** in [1, 8], **Y** in [0, 16], *infnan* "fn", "fnuz", or none
- Exponent scale type e8m0 (per OCP MX spec)
- Tensor scaled floating point types with a floating point scale or e8m0
- Tensor scaled unsigned integers with float scales and either float or int zero points
- Tensor scaled signed integers with float or exponent scales
- Single channel scaled types, as decribed above in tensor scaling
- Single dimension tile scaled types, as described above; tile sizes are powers of two with exponents in [2, 8]
- Two dimensional scale tiles (to address the transpose problem in training)
- Codebook number specs
- Implicit codebooks
- *Structured sparsity, single dimension*
- *Float hierarchical scaling with e4m3 (per Nvidia, implemented as e5m3)*
- *MSFP MX9/MX6/MX4 datatype support*
- *hierarchical scaling (tensor + tile + subtile + individual exponents)*

## Phase 3

Beyond Q3 we will focus on MI400 and later.  We will also explore the benefits of creating more
restricted subsets of TensorCast functionality ("snippets").  This is intended to remove dependencies
on PyTorch, or Triton, or even Python so that the basic quantization, compression, and incoherence
processing can be delivered in a customized way while reusing the base technology as much as possible.
Whether we follow through on this depends on whether there is a need for it internally or externally.

---

[TensorCast Documentation](./docs/README.md)
