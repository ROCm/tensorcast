<!-- markdownlint-disable MD033 MD041 -->

# Modes

> TODO(ericd): *needs proofreading and code snippets need to be verified.*

This file describes the modes for rounding, exponent scale selection, casting, and compute.

TensorCast has modes that are kept globally, initialized via `tcast.initialize()`, and overridden
on calls to `tcast.cast()` and `tcast.upcast()`.  The initialization and overrides may be specified
by the enumeration classes [RoundMode](#rounding-mode), [ScaleMode](#scale-selection-mode),
[CastMode](#cast-mode), and [ComputeMode](#compute-mode), which can be found
in the [source](../tcast/common.py), or by the strings that describe the modes.

## Rounding Mode

There are four round modes supported.  The rounding occurs after the tensor is scaled such that rounding
to an integer preserves only the mantissa bits that are allowed by the quantization dtype.  Post
rounding and clamping to (-maxval, maxval), the values are descaled.

### *Even*

The default mode, "even" performs round to nearest, ties to nearest even.  This is the only round mode
available for direct cast in PyTorch and Triton, so choosing another mode prevents taking advantage of
the performance that direct cast can provide.

```python
qx = tl.cast(x, tl.float8e4m3b8) # Triton
qx = x.to(torch.float8_e4m3fnuz) # PyTorch
```

### *Away*

This mode is round nearest, with ties away from zero.  This method is easier to compute than "even"", but is
the IEEE standard, and it does introduce bias by increasing the magnitude for ties.

### *Zero*

This mode is round nearest, with ties toward zero.  Like "away" it introduces bias by decreasing the magnitude
for ties.

### *Stochastic*

This is a random method where the probability of rounding up is based on the distance between the unrounded number
and the next highest integer.  This is considered useful for low precision floating point at worst, and essential
at best, particularly for training.

## Scale Selection Mode

Exponent scaling (power of two scales) is not as simple as it may appear.  No fewer than five methods have been
proposed within the MXFP community.  TensorCast supports these five methods, described below.  These methods
are all based on the maximum absolute value in the block sharing the scale.  In the descriptions below, we will
call this ***absmax***.

Regardless of the method, the scale is modified based on the maximum exponent in the datatype and the bias of the
e8m0 exponent (which is 127).  `scale = (adjusted maxexp) - (max exponent in cast dtype) + (e8m0 bias)` The last
line of this code is the same in the individual snippets, but omitted there.

```python
# xblock is highest magnitude in a block of X
# mode is ScaleMode
# nspec is the NumberSpec of the cast dtype
# integer values are represented with 1 ebit, and always use floor method
# scale is the biased and adjusted uint8 that goes into the scale tensor
absmax = xblock.abs().max()
log2exp = absmax.log2()
maxexp = maxexp_floor = log2exp.floor()
maxexp_ceil  = log2exp.ceil()
absmax_scaled = absmax / maxexp_floor.exp2()
if nspec.ebits > 1 and mode != ScaleMode.FLOOR:
    if mode == ScaleMode.CEIL:
        maxexp = maxexp_ceil
    elif mode == ScaleMode.MIDMAX:
        # scale midmax to match xblock
        midmax = nspec.finfo.midmax / (2**nspec.emax)
        maxexp = maxexp_ceil if absmax_scaled > midmax else maxexp_floor
    elif mode == ScaleMode.OPTION3:
        # scale absmax and round nearest even and unscale it and get floor
        ascale = (maxexp_floor - nspec.mbits).exp2()
        maxexp = (absmax / ascale).round() * ascale).log2().floor()
    elif mode == ScaleMode.TOPBINADE:
        maxfloat = nspec.finfo.maxfloat / (2**nspec.emax)
        maxexp = maxexp_ceil if absmax_scaled > maxfloat else maxexp_floor
# maxexp is unbiased, and sspec.scale.bias is 127 for e8m0,
# and nspec.emax is the adjustment in the OCP spec
scale = (maxexp - nspec.emax + sspec.scale.bias).to(torch.uint8)
```

### *Floor*

The primary advantages for *floor* is that MI355 and AIE4 implement it in hardware and
that it is the OCP "recommended" method.  This is, for a block of X:

```python
# nspec is the data tensor number and sspec.scale is the scale tensor number
maxexp = xblock.abs().max().log2().floor() 
```

### *Ceil*

This was proposed by Intel (along with `E3M0` for fp4 gradients).  It is just what you would think,
and it is demonstrably worse than *floor* overall.

```python
# nspec is the data tensor number and sspec.scale is the scale tensor number
maxexp = xblock.abs().max().log2().ceil() 
```

### *Midmax*

This was developed by Eric Dellinger and Stuart Biles.  It strives to avoid clipping by adding 1 to the floor
method if ***absmax*** is closer to the next higher integer than the max representable value in the datatype.
This is particularly effective with e4m3fn due to the fact that the max value *representable* is reserved for NaN,
and we are stuck with the max value *available*.  This makes the *floor* option suboptimal (and I'm being
nice here) for e4m3fn.  *Midmax* is also better for mxfp4e2.

```python
# scale midmax and absmax into (0, 2] so they can be compared
absmax = xblock.abs().max()
maxexp = absmax.log2().floor()
midmax_scaled = nspec.finfo.midmax / (2**nspec.emax)
absmax_scaled = absmax / maxexp.exp2()
maxexp += int(absmax_scaled > midmax_scaled)
```

### *Option3*

Proposed by Meta (and they did call it "option 3"), it is similar to *midmax*.  

```python
absmax = xblock.abs().max()
maxexp = absmax.log2().floor()
# scale absmax to rounding range
ascale = (maxexp - nspec.mbits).exp2()
# note that the round mode is *even*, so we can use torch.round()
maxexp = ((absmax / ascale).round() * ascale).log2().floor()
```

### *TopBinade*

Proposed by Microsoft, with the desire to avoid clipping when doing stochastic rounding.
This accomplishes that goal, but it is very nearly as conservative as the *ceil* approach
(which was also proposed in conjunction with stochastic rounding). AMD does not have any
data regarding efficacy versus these other methods.

```python
# scale maxfloat and absmax into (0, 2] so they can be compared
absmax = xblock.abs().max()
maxexp = absmax.log2().floor()
maxfloat_scaled = nspec.finfo.maxfloat / (2**nspec.emax)
absmax_scaled = absmax / maxexp.exp2()
maxexp += int(absmax_scaled > mmaxfloat_scaled)
```

## Cast Mode

### *Virtual*

This mode is better known as "fake quantization", in that the output of the cast operation is
unscaled and in the same torch.dtype as the input (unquantized) tensor. If sparsity is done,
the pruned values will be represented as zero.  No scales are retained.  No benefit is realized
for data storage or data movement, and at best 16-bit GEMM computation will be done.

When performance and memory are not an issue, such as in post training quantization, *virtual*
mode can be used until final export of the model weights.

### *Actual*

The *actual* mode means that the output quantized tensor has the same shape as the input tensor,
but has the smallest torch.dtype that can represent the scaled, quantized tensors. Generally these
will be 8-bit floats or integers (signed or unsigned). Scales are retained, which will be 8-bit
unsigned integers for e8m0 scaling, or 8, 16, or 32 bits for float scaling.  The sparsity mask,
if any, will be `torch.bool`, which is 8 bits per boolean value and the output tensor will maintain
the input tensor's shape (with zeros), just as with *virtual*.

### *Compress*

The *compress* mode is like *actual*, but some of the easier compression will be taken advantage of.
Perhaps it would be better to name this mode *packed* to avoid confusion with general compression
algorithms.

- 4-bit (fp4/it4/uint4) and 2-bit values will be stored in uint8 tensors, packed 2 and 4 per byte respectively
- 1-bit values (binary weights or sparsity masks) will be stored as uint8, with 8 packed values per
byte
- metadata (from [codebooks](./codebook.md) currently) will be packed as efficiently as possible
- sparse output tensors will have pruned values removed, resulting in a decrease of the sparse dimension
size by a factor of 2, 4, or 8
- 3- and 6-bit values will be stored as uint8, but packed as if they were 4 and 8 bit values.  This
will be revisited if and when a performant method for packing/unpacking/moving these with more efficient
packing is found

## Compute Mode

There are currently two compute methods, *torch*, where the operations are implemented
using PyTorch ops, and *triton*, where the operations are implemented as Triton kernels.

Some operations are not implemented in either at this time, a situation which will result in
`NotImplementedError` being raised. Otherwise, if the selected compute mode supports the operation
but fails in some way, the exception will be caught, reported as a warning if the other compute
mode supports it, otherwise that exception will be raised. If the unselected mode fails, then *that*
exception will be raised.

`TorchCast` will be more comprehensive in its feature coverage, but in some cases it may be slower
due to padding, permuting, and reshaping that may not be needed in `TritonCast`, as well as multiple
GPU kernel launches (which may be mitigated by `torch.compile`).  As features come online (they get
written and debugged), it will be better to use the `triton` mode, with the fallback on `torch`.

### *Torch*

This mode will benefit from the fact that the features available in the v2 branch of TensorCast,
despite the refactoring in the triton branch, are probably close to functioning correctly.

### *Triton*

The initial focus of this mode is to support the configuration and interface to the Triton attention
kernels.  As with `torch` mode, when the conditions are right (casting to a supported torch.dtype,
with round mode *even*) a direct cast of the scaled tensor will be used.

However, in training experiments with very low precision (<= 6 bits, as in MXFP6 and MXFP4), stochastic
rounding is expected to be used extensively.  Fp6 is not yet supported in either PyTorch or Triton,
and fp4 is only supported in Triton, so more general quantization methods will be needed.  Datatypes
that are not supported in `tl.scaled_dot` may need a performant fused GEMM kernel.

---

[Documentation](./README.md)
</br>
[Testing](../tests/README.md)
</br>
[Examples](../examples/README.md)
</br>
[TensorCast Home](../README.md)
