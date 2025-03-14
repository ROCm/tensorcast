<!-- markdownlint-disable MD033 MD041 -->

# Numbers

> TODO(ericd) *this needs proofreading*

Number formats in TensorCast are defined as [NumberSpec](../tcast/number.py) instances.

## Number Specifications

Number format specifications are wholly independent of scaling and simply define the characteristics of a number
representation. Five number categories are represented: floating point, signed integer, unsigned integer, exponent,
and codebook.

A floating point format includes the normal attributes: exponent width, mantissa width, bias or maximum unbiased
exponent, and handling of infinite numbers and NaN.  All floating point numbers are signed, with an implicit bit and
subnormal support.  Three modes of inf/NaN handling are supported: *ieee*, *fn*, and *fnuz*.  The *ieee* mode is the
default, and follows the standard IEEE model of reserving the highest biased exponent value for infinite numbers
and NaN. The *fn* mode (**f**inite + **n**an) does not represent inf, and uses the highest representable value (all
bits excluding sign are ones) as the NaN value.  The *fnuz* mode (**f**inite + **n**an represented as **u**nsigned
**z**ero), which is LLVM/MLIR standard, where the meaning is that negative zero indicates NaN, and positive zero
is zero.

Athough unsigned floats and disabling subnormals are potential future features, they are not planned.  There is,
however, a special case for describing the power of two scale factors defined by OCP (which are technically unsigned
integers), but sematically it is a biased exponent.  Therefore, a mantissa of width zero and an *ieee* mode
indicates an unsigned, biased power of two OCP scale.  This is the exponent type mentioned above.

Integers are defined simply by the number of bits and the presence or absence of a sign bit.

Codebooks are a storage format in which a 2D array of compute datatype values, an index from tile or subtile metadata
selects the row, and the individual value within the tile or subtile is an index to select the column.  The compute
value is then scaled appropriately.  Codebooks are able to store more dynamic range and/or precision than a number
format the size of the index value, but the compute values available to a tile will be a small subset of the possible
values of that compute datatype.

Generally, codebooks are created through clustering scaled but unquantized values to minimize quantization error
across the tiles that share a codebook.  This can be compute intensive, so it is not necessarily useful for dynamic
casting, but has potential for 2, 3, or 4 bit weights, optimized post-training.

TensorCast provides two codebook mechanisms: in the first, the codebooks are explicitly defined by the user and added
to the number specification after it is instantiated.  In the second, a string encoding can define patterns of codebook
entries that are generated automatically during instantiation.

### Number Specification String Encoding

A `NumberSpec` is created using a string encoding.  The encoding is generally EMB (exponent-mantissa-bias) format
for floats and [u]intK for integers.  Additional encodings correspond to commonly used unscaled datatype names and
`torch.dtype` names.  

Exceptions are made for common types (e.g. *float32*, *bfloat16*).  The EMB format is of the form "eXmY[bZ]", where
X, Y, and Z are the exponent width, mantissa width, and bias.  If the bias is not specified the default is
(2**(X-1) - 1).  **A notable exception occurs with torch dtypes** `torch.float8_e5m2fnuz` and `torch.float8_e4m3fnuz`,
in which the biases are 16 and 8 respectively. These correspond to the Graphcore/Nanoo representations, and in the
TensorCast EMB format are defined as `e5m2b16fnuz` and `e4m3b8fnuz`.

> Note: flexibility is built in, but testing is limited so far.
>
> - *uintK* and *intK* implemented for 2 <= K <= 32, tested for K in [4, 8, 16]
> - exponent *eXm0* implemented for 4 <= X <= 8, tested for X = 8
> - *eXmYbZ* implemented for 1 <= X <= 8 and 0 <= Y <= 23, testing limited to standard and minifloats

A `NumberSpec` can alternatively be created using a `torch.dtype` or the string representation thereof.
Since there are different existing naming conventions, the string decoder accepts but strips away any leading "torch."
or "float8_".

Lookup types have additional data, including the number of lookup tables, the size of each lookup table, and the
compute datatype, which is itself a NumberSpec.  The string codes therefore are a lookup code joined to the compute
numberspec code by an underscore, such as "l42f6431_e4m3fn".  The lookup code is "lXY*name*", where X is the number of
bits in the index values, and Y is the number of bits to select a table.  Thus, a single table is 2<sup>X</sup> long,
and there are 2<sup>Y</sup> tables.  The *name* is simply a unique identifier.

#### Auxiliary Data

During construction, the number spec calculates commonly used information such as *emax*, *emin*, as well as *bits*,
*max*, *min*, *smallest_normal*, and *eps*, in the manner of `torch.finfo` and `torch.iinfo`. Another value, *midmax*
is midway between *max* and 2\*\*(*emax* + 1), which can be used for alternative power of two scale selection.

#### NumberSpec Implementation Notes

Signed integer values with a power of two scale are typically implemented as fixed point, with a sign, a single integer
bit, and a fractional component that is *bits* - 2 wide.  This can be (and is) represented as a normally biased float
with a single exponent bit and *bits* - 2 mantissa bits.  Arithmetically the exponent bit acts as the integer bit. This
facilitates casting, while leaving the actual storage format (floating point, 2's complement, or sign magnitude) as a
platform-specific implementation detail. As a result, integer number specs have both `torch.finfo` and `torch.iinfo`
values.

If the number specification is an exact match to a torch.dtype (regardless of whether a torch dtype or name was used
to create the spec), that dtype will be accessible through the NumberSpec's torch_dtype attribute.

---

[Documentation](./README.md)
</br>
[TensorCast Home](../README.md)
