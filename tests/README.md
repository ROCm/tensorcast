<!-- markdownlint-disable MD033 MD041 -->

# TensorCast Testing

TensorCast uses pytest in the tensorcast/tests directory.  The way it is organized is that
the pytest decorators are in the test_\* files.  Those tests call the base tests in the base\_*
files.  Some of the test code resides in the classes that are being tested, such as LPConfig.py
in the tcast directory.

Access to the tests can be through pytorch, ot through tensorcast/test_harness.p, the latter of
which is a bit earlier to use for debugging underlying code that going through the pytest
framework.

In some cases, other libraries are used to sanity check tcast triton.  In test_mx.py, the
tests compare the results of the MX (microxcaling) library from Microsoft to the results of
tcast.  In test_torch_virtual.py, all virtual castmode tests compare vs tcast branch v2. There
might be some tests that use SQT (an earlier verion of TensorCast).  All of these cases require
another library.  MX is easy enough to clone and use.  TensorCast v2 is harder in that two
distinct branches are being used at the same time.  The method I used was to clone tensorcast
elsewhere (outside my python path), then rename the tcast subdirectory to tcastv2 and copy it to
my tensorcast clone, parallel to tcast.  SQT is problematic because it is a private repo, and
you have to get permission, so I am waiting a bit on that.

This document is organized as follows:

- [Test Coverage](#test-coverage)
  - [Unscaled](#unscaled)
  - [Unsigned Integer](#unsigned-integer)
  - [Signed Integer](#signed-integer)
  - [Float Scaled Float](#float-scaled-float)
  - [Exponent Scaled Float](#exponent-scaled-float)
  - [Exponent Scaled Integer](#exponent-scaled-integer)
  - [Scalling Options](#scaling-options)
- [Additional Coverage](#additional-coverage)
  - [Modes](#modes)
  - [Shapes](#shapes)
  - [Subtiles](#subtiles-and-implicit-codebooks)
  - [Miscellaneous](#miscellaneous)
- [Priorities](#priorities)
  - [Essentials](#essentials)
  - [Low Precision Attention](#low-precision-attention)
  - [OCP MXFP and MXINT](#mxfp-and-mxint)
  - [Everything Else](#everything-else)

## Test Coverage

If both TorchCast and TritonCast supported everything (all modes and cast methods), there
would be a lot of functionality to cover.  First, the basic casting support:

### Unscaled

Tcast supports unscaled floats only.  These are datatypes with a number spec, but no scale spec.
At this time, that includes only the torch dtypes for 32, 16, and 8 bit floats, which is
**7 datatypes**.

### Unsigned Integer

Unsigned integer has a floating point scale, and either a float or int8 zero point.  Taking
a subset of the possibilities (e.g. scale and float zero point ar the same type, and int zero
is always int8) we get six variants for each bitwidth (e.g. uint2 through uint8 which is seven.
This is **42 variations** just for tensor scaled unit.  See the [scaling options](#scaling-options)
below.

### Signed Integer

For int data, we have only a float scale, and e8m0 scale (OCP), or eventually an e4m3 unsigned
float scale (hybrid with a tensor scale, Nvidia's creation, but we will ignore this for now).
A subset of possible widths (e.g. 3 to 8, plus 16) gives us seven widths and four scale types
(fp32, bf16, fp16, e8m0, e4m3), resulting in **28 variations**.

### Float Scaled Float

Generally, the float data (unquantized) is one of fp32, fp16, or bf16.  Let the data/scale
combinations be fp32/fp32, fp16/fp32, fp16/fp16, bf16/fp32, and bf16/bf16.  Let's also consider
the many types of float number formats we want to quantize *to*: 4 standard fp8, plus e3m4;
1 fp7 (e3m3), 2 fp6, 2 fp4 (incl. e3m0), plus a few random things, such as bfloat19, and maybe a
couple more.  Let's say 14. So, 5 * 14 = **70 variations**.

### Exponent Scaled Float

Again using fp32/fp16/bf16 as the possible unquantized datatypes, with only one scale (e8m0),
we have only 3 input/scale pairs and the same 14 target dtypes, **42 variations**.

### Exponent Scaled Integer

This is probably an uninteresting combo, but for int3 to int8, then int16 it just adds
**7 variations**.

### Scaling Options

Besides tensor scaling, there is 1D channel scaling and various sizes of 1D tile scale.  Then
we have square 2D and asymmetric 2D tiles.  For each 1D tile scale, there is a 2D scale with
the other dimension being a channel scale.  The, for all of these, there is a possible tensor
tensor scale, hierarchically applied.

Suppose we test 1D tiles of size 4 to 256 (power of two, 7 sizes), plus channel scales,
which is 8 variations.   Next, we will add 2D tiles for each of the 1D non-channel tiles,
addind 7 more, to 15 variations. Now consider square tiles of 4x4, 8x8, 16x16, 32x32, 64x64,
and 128x128, bringing us to 23.  Asymmetric 2D scales are not the highest priority because they
do not solve the transpose problem, but we will do 16x32 and 32x64 just to see if it works,
so two more, which is 25.  Next, all of the 1D and asymmetric 2D scales can be transposed,
giving us 50 variations.  Every one of those can have a one of four (fp32/fp16/bf16/e8m0)
hierarchical tensor scales, so we are up to 200.  We are not done.

> There are datatypes such as codebooks and Microsoft's MX4/6/9 micoxcaling ("prime") formats
> (which are implemented as codebooks) that have subtiles. We will put this off for a later time.

Finally, there is sparsity, which is part of the scaling. Sparsity is 1, 2, 4, or 8 retained
values out of a 1D tile that is 4, 8, 16, or 32. (Structured sparsity in blocks larger than 32
is not feasible. Leaving out sparsity levels of less than 12.5% of the values being retained,
we have three sparsity (12.5%, 25%, or 50% dense), so 11 sparsity variations (tile size 4 has only
2 options).  Sparsity is only 1D, but with the transpose it is 22.  With a hierarchical tensor scale
that is doubled to **44 variations**.

Once we add in the single tensor scale, we have 1 + 200 + 44 = 245 scaling options.

### Altogether, Now

Altogether, 7 unscaled datatypes, 189 scaled datatypes with 245 scale shape + sparsity options,
we have ***46,312 variations***.

## Additional Coverage

Besides datatypes, scales, and quantization, there are other things to test.

### Modes

We have four round modes, five scale selection modes (only for e8m0), two compute modes (torch
and triton), and three cast modes (virtual, actual, and compress).  There are 120 combinations.

Initially, the plan is to cover *virtual* for torch, and all three cast modes for triton, which
brings us back down to 80.

### Shapes

Coverage needs to include 2D, 3D, and 4D tensors in the context of reshaping to handle 2D
scale factors, which is a bit more work in PyTorch than in Triton.  These will need to cover
edge cases (dimensions not a multiple of the tile size, and so forth).  Figure on a couple
of dozen shapes.

### Subtiles and Implicit Codebooks

Codebooks benefit greatly from subtiles, and other datatypes might as well.  If the hardware
implementations of tiles scaling are used, then the entire tile will need to share a single scale.
With subtiles, we will not have hardware support, but will rely on fused kernel operations
(as we would for float scaling), but subtiles can help mitigate outliers, or choose a different
codebook entry per subtile based on metadata stored in another tensor.

The description of implicit codebooks through strings is pretty extensive, and we need tests
for that.  Subtiles can also be used for exponent offsets (from the shared e8m0 scale, as long
as we are upcasting to a compute datatype that has at least one more exponent bit than the
storage datatype.

### Miscellaneous

Up to this point, we have covered quantization and sparsification.  What is left is the
new configuration interface, incoherence processing, the triton snippets, and at least a proof
of concept with a triton kernel using the triton snippets.  There is also the upcast API call,
which takes a quantized scaled tensor (actual or compressed) and converts it to a single
float tensor that matches what a virtual cast would have returned.

There are also optimzation hacks.  For one, if we are doing a float-tile-scaled float quantization
to a datatype that happens to have a direct cast mode implementation (to a standard `torch.dtype`)
we could expect a speedup over a generic quantization algorithm.  

## Priorities

This is a lot of testing, and writing the tests and getting them to pass is a considerable amount
of work, so this section is about the order.

### Essentials

The structure of the code has to be functional for some subset of features.  The NumberSpecs,
ScaleSpecs, DataTypes, Tensor (which includes shaping, packing, and tensor type), the API and
underlying Triton and Torch implementations all have to work.  Generally this is tested through
higher level tests for quantization (like `import tcast`).

### Low Precision Attention

This is the top priority project for this version of TensorCast. The primary datatypes are the four
FP8 types with 1D and 2D tile scales and float scale factors.  In order to get new things running,
the *virtual* cast helps, but for performance we need *actual* and *compress* cast modes, and the
TritonCast engine is where the focus is.  This is where the `LPConfig`, incoherence processing,
and the high level Triton interfaces such as `scale_and_quantize` are essential, and some examples
of the interface being used in an attention kernel.

### MXFP And MXINT

For MI355 we will need more exploration into attention and scaling, and probably more work with
incoherence processing due to the smaller exponent bit fields in fp6 and fp4.  Not only should
all of the OCP MX types be implemented, but variations on scaling and ICP will be needed. For
fp6 and fp4 we will definitely need stochastic rounding.  Sparsity may also be of interest.

This might be the time to get Nvidia's MX variant (nv4, block size 16, e4m3 and e5m3 scales)
fully supported.

### Everything Else

After the needs of MI355 are met, it is time to test out everything else: all float, uint, int
data with all of the supported scale types and shapes; all round modes and scale modes; and
codebooks.

---

[Documentation](.docs/README.md)
</br>
[TensorCast Home](../README.md)
