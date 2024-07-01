<!-- markdownlint-disable MD033 MD041 -->

# TensorCast (tcast)

TensorCast is a casting/quantization library in development based on PyTorch 2.2+.

The scope of TensorCast is defining datatypes and converting tensors between datatypes.  A "datatype" is a number format
specification combined with an optional scaling specification.  A "cast" is the conversion of a tensor from one datatype
to another.  A conversion can include compressed tensors that pack values and scaling information (*actual* cast) or
regular torch tensors (*virtual* cast, or "fake quantization").  In version 1 of TensorCast, only virtual casting is supported.

The focus of TensorCast is on OCP MX datatypes described in
[OCP MX Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
as well as additional datatypes pertinent to AMD and such other types as needed to support research in the area of low precision
for machine learning.  This focus includes everything needed to describe and convert the datatypes, but also reference
code in various forms that can be used to verify implementations elsewhere.

Contributors:

- Eric Dellinger [@ericd](mailto:eric.dellinger@amd.com)
- Alireza Khodamoradi [@alirezak](mailto:alireza.khodamoradi@amd.com)

## Structure

The primary data structures are defined in the classes [NumberSpec](#numberspec), [ScaleSpec](#scalespec), and [DataType](#datatype).
The conversion operators are in the static class [Cast](#cast).

### NumberSpec

Number format specifications are wholly independent of scaling, and simply define the characteristics of floating point, integer,
and unsigned integer formats.  Some additional support to ease the conversion process is included.

#### Inherent Types

Four number categories are represented: floating point, signed integer, unsigned integer, and exponent.

A floating point format includes the normal attributes: exponent width, mantissa width, bias or maximum unbiased exponent, and
handling of infinite numbers and NaN.  All floating point numbers are signed, with an implicit bit and subnormal support.  Three
modes of inf/NaN handling are supported: *ieee*, *fn*, and *fnuz*.  The *ieee* mode is the default, and follows the standard IEEE
model of reserving the highest biased exponent value for infinite numbers and NaN.  The *fn* mode (**f**inite + **n**an) does not
represent inf, and uses the highest representable value (all bits excluding sign are ones) as the NaN value.  The *fnuz* mode
(**f**inite + **n**an represented as **u**nsigned **z**ero), which is LLVM/MLIR standard, where the meaning is that negative
zero indicates NaN, and positive zero is zero.

Athough unsigned floats and disabling subnormals are potential future features, they are not planned.  There is, however,
a special case for describing the power of two scale factors defined by OCP (which are technically unsigned integers),
but sematically it is a biased exponent.  Therefore, a mantissa of width zero and an *ieee* mode indicates an unsigned,
biased power of two OCP scale.  This is the exponent type mentioned above.

Integers are defined simply by the number of bits and the presence or absence of a sign bit.

#### NumberSpec String Encoding

A `NumberSpec` is created using a string encoding.  In TensorCast, string encodings are used to define numbers, scales and datatypes
as an alternative to args and kwargs, although support for construction via parameters is a potential addition.  The encoding is
generally EMB format for floats, [u]intK for integers.  Exceptions are made for common types (e.g. *float32*, *bfloat16*).  The EMB
format is of the form "eXmY[bZ]", where X, Y, and Z are the exponent width, mantissa width, and bias.  If the bias is not specified,
the default is (2**(X-1) - 1).  **A notable exception occurs with torch dtypes** `torch.float8_e5m2fnuz` and `torch.float8_e4m3fnuz`, in
which the biases are 16 and 8 respectively. These correspond to the Graphcore/Nanoo representations, and in the TensorCast EMB format
are defined as `e5m2b16fnuz` and `e4m3b8fnuz`.

> Note: flexibility is built in, but testing is limited so far.
>
> - *uintK* and *intK* implemented for 2 <= K <= 32, tested for K in [4, 8, 16]
> - exponent *eXm0* implemented for 4 <= X <= 8, tested for X = 8
> - *eXmYbZ* implemented for 1 <= X <= 8 and 0 <= Y <= 23, testing limited to standard and minifloats

A `NumberSpec` can alternatively be created using a `torch.dtype` or the string representation thereof.
Since there are different existing naming conventions, the string decoder accepts but strips away any leading "torch." or "float8_".

#### Auxiliary Data

During construction, the number spec calculates commonly used information such as *emax*, *emin*, as well as *bits*, *max*, *min*,
*smallest_normal*, and *eps*, in the manner of `torch.finfo` and `torch.iinfo`. Another value, *midmax* is midway between *max* and
2\*\*(*emax* + 1), which can be used for alternative power of two scale selection.

#### NumberSpec Implementation Notes

Signed integer values with a power of two scale are typically implemented as fixed point, with a sign, a single integer bit, and
a fractional component that is *bits* - 2 wide.  This can be (and is) represented as a normally biased float with a single exponent
bit and *bits* - 2 mantissa bits.  Arithmetically the exponent bit acts as the integer bit. This facilitates casting, while leaving
the actual storage format (floating point, 2's complement, or sign magnitude) as a platform-specific implementation detail. As a
result, integer number specs have both `torch.finfo` and `torch.iinfo` values.

If the number specification is an exact match to a torch.dtype (regardless of whether a torch dtype or name was used to create the
spec), that dtype will be accessible through the NumberSpec's torch_dtype attribute.

### ScaleSpec

Scaling specifications must differentiate between tensor scales, channel scales, tile scales, subtile scales, and individual scales
(i.e. value exponents).  A *tile* in TensorCast is also known as a "block" or "group", but here the term "tile" is used, matching the
Microxcaling ([paper](https://arxiv.org/pdf/2310.10537.pdf), [github](https://github.com/microsoft/microxcaling.git)) terminology
from Microsoft, who developed the OCP MX formats, and did earlier work with
[MSFP](https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf), a block floating point format
that is implemented in the next AIE.

Microsoft also introduced *subtile* in
[With Shared Microexponents, A Little Shifting Goes a Long Way](https://arxiv.org/abs/2302.08007),
where the initial version of MX (without individual exponents) shared one or more scale offset bits to preserve precision.  These
datatypes include MX9, MX6, and MX4 (also known as BFP Prime), which are implemented in an upcoming AIE device.  Those MX types are
planned for version 2 of TensorCast.

In TensorCast V1, tensor, channel, tile, and individual scales are supported, but the first three are mutually exclusive, and the tile
is one dimensional.  Two dimensional tiles and hierarchical tensor/tile/subtile scaling are scheduled for V2.

#### Types of Data and Scales

Unsigned integer data is generally asymmetric, meaning that there is a zero point in addition to the scale factor. The scale factor
is some form of float, and the zero point can be float or int (the latter guarantees precise 0.0 representation for reduced precision
scales, with a slight loss of SQNR).  Integer and exponent number specs are not supported for unsigned int scales.

Signed integer is generally symmetric around zero, dropping the highest magnitude negative value to avoid bias in the quantization.
The scale numberspec can be either a float or an exponent.  Allowing an integer or unsigned bias adjustment in addition to exponent
types
is being considered for V2.  Support for unbalanced (asymmetric) scaling is not planned.

Floating point data has an inherent individual scale, but the tensor/channel/tile scale is restricted to exponent numspecs in V1.
A floating point scale is planned for V2.

Unscaled data, such as bfloat16, does not have a scale spec in the datatype, so in the scale spec we currently have either a tensor,
channel, or tile scale.  A tensor scale is simply a scalar (or two scalars for unsigned data), and is specified by defining the
number spec(s) for the scale with no tile specification.  A channel scale is a tile scale, in which tile is the size of the tensor
in the dimension of the scale, and is specified with a tile size of zero.  A tile scale has a tile size and the dimension of the
tile.  The dimension defaults to -1 (the last dimension of the tensor).

A limitation in V1 as of now is that padding of tensors is not implemented, so the tensor size in the specified dimension must be a
multiple of the tile size.

#### ScaleSpec String Encoding

The components of a scale being scale number spec, optional zero point number spec, and optional tile spec, the string encoding
of a scale specification is the concatenation of the string encodings of the constituents, joined by underscores.

The number specs are defined above.  The tile scale is of the form "tXdY", where X is the size of the tile, a power of two between 2
and 1024 (or 0 for channel scaling) and Y is the dimension of the tile. If the dimension is -1, "dY" is omitted.  However, for channel
scaling the tile spec must be included, even if it is only "t0".

#### ScaleSpec Implementation Notes

Until 2D tiles, subtiles, hierarchical scaling, and compression are implemented in V2, ScaleSpec is pretty simple.  There are methods
for reshaping the tensor to make PyTorch-based scale discovery a bit more straightforward.

### DataType

The datatype is simply a number specification and an optional scale specification.  If no scale spec is provided, the datatype is
unscaled.  Support for unscaled integer types is unsupported, although it may make sense for int16.

#### Predefined DataTypes

PyTorch has dtypes in the torch namespace; TensorCast has predefined dtypes in the tcast namespace.

These are standard datatypes that are expected to be commonly used.  Unscaled dtypes include the standard torch floating point types,
including float8 types if supported in your Pytorch installation.  Also included are unscaled versions of the MXFP types: `e3m2fnuz`,
`e2m3fnuz`, and `e2m1fnuz`.

Tensor scaled types include uint16, int16, uint8, and int8 as well as the MXFP 8 and 6-bit numberspecs. The naming convention for
the dtype names is the numberspec and a scale indicator, encoded as "f" for float16, "b" for bfloat16, "e" for exponent scales.
The uint types have two such indicators, the second being for the zero point, and the zero point has an "i" to indicate an int8
zero point number spec instead of the disallowed "e".  Floating point dtypes all have the "e" designation.

Tile scaled predefined types are the MXFP and MXINT types: mxfp8e5, mxfp8e4, mxfp6e3, mxfp6e2, mxfp4e2, mxint8, and mxint4, all of
which have a tile size of 32.  Also included is `bfp16`, which is like mxint8 but with a block size of 8.  Other tile scaled dtypes
are the uint8 and uint4 variants of ff, bb, fi, bi with tile size 32 and int8/int4 with float16 and bfloat16 scales.

#### DataType String Encoding

A datatype string is the contactenation of the number spec and the scale spec, but construction is done not with an overall
string, but by passing the number spec, the optional scale spec, and an optional concise name (e.g. *mxfp4e2*) to the DataType
constructor.

### Cast

Cast is a static class that contains the PyTorch code to perform rounding, scaling, and quantization.  When the torch extension is
implemented, the cast class will be able to route the cast call to the appropriate implementation (e.g. python, cpu C++, gpu C++)
based on a CastMode, tensor characteristics, and available kernels.

Public methods generally correspond to the API methods in the tcast namespace.  Private methods include \_vcast, \_round, \_cast_unscaled,
and \_safe_frexp.

### Package Level API

The classes in tcast need not be used directly.  An API wraps essential functionality.

#### initialize

The initialize function currently just sets default roundmode and/or scalemode so that overrides in the cast
calls are not necessary.  This is optional.  Soon, there will also be a default for ComputeMode, which will
select between PyTorch ops, C++/CPU extension, or C++/HIP-CUDA extension.

```python
import tcast
tcast.initialize(roundmode="even", scalemode="max")
```

#### number

This function, given a valid code string, returns a NumberSpec, which can then be used to create a DataType.

```python
import tcast
nspec = tcast.number("e5m6") # fp12, an abbreviated version of fp16
```

#### scale

This function, given a valid code string, returns a ScaleSpec, which can then be used to create a DataType.

```python
import tcast
sspec = tcast.scale("e8m0_t32") # power of 2 scaling on the last dimension with tile size 32
```

#### datatype

This function, given a number spec (NumberSpec or valid numspec code string), an optional scale (ScaleSpec or valid
scale spec code string), and an optional name for the datatype, returns a DataType, which can be passed to a cast function.
If the name is omitted, one is manufactured.

```python
import tcast
nspec, sspec = number("e5m6"), scale("e8m0_t32")
dtype = tcast.datatype(nspec, sspec, name="e5m6_e32")
# or
dtype = tcast.datatype("e5m6", "e8m0_t32", name="e5m6_e32")
```

#### cast

This is intended to be a universal interface to the Cast class, but will be supplemented by task-specific cast methods,
suct as `sparse`.  For the current virtual cast limitation, so scale data needs to be returned, and the only parameters
needed are the input `torch.Tensor` and `DataType`, with optional overrides for roundmode and scalemode.

```python
import tcast
x = tcast.cast(
        torch.randn(1024, 1024, device="cuda", dtype=torch.float16),
        tcast.datatype("e5m6", "e8m0_t32", name="e5m6_e32"),
        roundmode="nearest",
        scalemode="auto"
    )
```

Many common datatypes are predefined, which simplifies the calls:

```python
import tcast
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = tcast.cast(x, tcast.mxfp6e2)
```

#### sparse

A simple sparsity function is provided that preserves the M highest magnitude values from N values in a tile along
the specified dimension.  In practical hardware terms, the dimension would be the inner dimension of a GEMM and M and N
would be mandated by the hardware platform.  Clearly, sparsity has many variations, and magnitude may not be the best
qualifier, but this is a start.

```python
import tcast
s = tcast.sparse(x, 8, 4)  # 4 of 8 dense values from each tile of 8
```

## Development Plan

The feature set planned for version 1 is:

- Virtual (“fake”) casting (torch.float32, torch.float16, torch.bfloat16 in/out)
- Signed and unsigned integer specifications uint**K** and int**K** for K in [3, 16]
- Floating point e**X**m**Y***infnan* for **X** in [1, 8], **Y** in [0, 16], *infnan* "fn", "fnuz", or none
- Exponent types e**X**m0 for **X** in [4, 8] (biased power of two scale factors)
- Unscaled floating point types
- Tensor scaled floating point types with exponent scale
- Tensor scaled unsigned integers with float scales and either float or int zero points
- Tensor scaled signed integers with float or exponent scales
- Single channel scaled types, as decribed above in tensor scaling
- Single dimension tile scaled types, as described above; tile sizes are powers of two with exponents in [2, 10]
- M of N sparsity within tiles or subtiles
- round modes: nearest, even, zero, and stochastic
- scale modes (exponent selection): max and midmax
- PyTorch python operations for casting
- *C++ (CPU) casting in PyTorch extension*
- *C++ (HIP/CUDA) casting in PyTorch extension*

The feature set planned for version 2 is:

- Actual (compressed) casting
- 2D tile specifications
- 1D and 2D subtile specifications with scale offsets from tile scale
- tile and subtile-specific number specifications with selection metadata ("multicast")
- lookup table number specs
- MSFP MX9/MX6/MX4 datatype support
- hierarchical scaling (tensor + tile + subtile + individual exponents)
