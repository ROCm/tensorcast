<!-- markdownlint-disable MD033 MD041 -->

# Codebooks

Codebooks in TensorCast are of two types: a generic codebook in which the codebook mappings (lookup tables)
are added to the NumberSpec instance after instantiation; and "implicit" codebooks in which the mappings are
generated during instantiation from the string code passed to the NumberSpec constructor.  Beyond that
difference, they have the same cast/compression/upcast behavior.

General codebook creation is outside the scope of TensorCast; there is no support for clustering, centroid
optimization or selection of weights that share a codebook. Dynamic cast is supported via a binary search
through each mapping in the codebook, with the mapping selection determined by a metric supplied through
the cast call.  This is inherently slower than a standard number spec, and illustrates the challenges for
dynamic quantization of tensors during inference (activations) and training (weights, activations, gradients).

## Codebook Creation

The tcast api includes a creation function called `number` that takes a string code or a `torch.dtype` and
returns either a `NumberSpec` or a `Codebook` (which is a subclass of `NumberSpec`).  A `Codebook` is returned
if the input is a string beginning with "cb" (case insensitive).

```python
    codebook = tcast.number("cb41fi_e2m3fnuz")
```

Codebooks can also be created via contstructor:

```python
    codebook = tcast.Codebook("cb42_e4m3fn_mycodebook")
```

## Codebook String Codes

A codebook is defined with a string that has either two or three substrings separated by underscores.  The first
describes the codebook and the second is the number specification code for the compute type which which the codebook
will be populated.  A third, optional substring is a concise name by which that codebook will be distinguished from
other codebooks, as the initial description may be too vague ("cb42") or too long and cryptic ("cb42f6431").

The generic codebook description is of the form:

```text
    cb<vbits><mbits>[<desc>]_<cspec>[_<label>]
```

`vbits` is a single digit that is the bit width of the values that are indices into a specific mapping lookup table.
`mbits` is a single digit that is the number of bits of metadata that selects the mapping lookup table. `desc` is a
code specific to implicit codebook mapping creation (described below) and is not used for generic codebooks.  `cspec`
is the `NumberSpec` code for the compute type contained in the codebook lookup tables. `label` is the optional
alternative codebook name.

The codebook dimensions, then, are 2<sup>mbits</sup> and 2<sup>vbits</sup>, and the codebook ebtries are the size of the
compute number spec (`NumberSpec(cspec).bits`).

## Implicit Codebook Description Encoding

Implicit codebooks are fixed codebooks that are able to be upcast from encodings to compute values directly
(or nearly so) in hardware, without the need to store or transfer actual codebook tables.  In TensorCast they
can also be though of as a shortcut to populate codebooks algorithmically to simplify experimentation.

Implicit codebook mappings are ordered subsets of values representable in the compute type. TensorCast considers
the compute type as a signed integer that corresponds to the bits that represent each value, so -15 to 15 for
fp5 (e2m2fnuz).  Patterns created and shifted up and down this number line.

### FP4 and INT4 Patterns

These patterns can be based on a specific number type, e.g. fp4 or int4.  Implicit codebooks are currently *symmetric only*, so the non-negative value mappings for both fp4 and int4 look like this:

<div align="center">
    <img src="./CB41FI_dark.png#gh-dark-mode-only", alt="CB41FI", width=128>
    <img src="./CB41FI_light.png#gh-light-mode-only", alt="CB41FI", width=128>
</div>
</br>

This example would be encoded as `cb41fi`, with "f" indicating a standard fp4 (e2m1) pattern, and "i" indicating a
standard int4 pattern.

The "4" in the decription indicates the number of bits in each encoded value.  This tells the codebook that the pattern
for "f" is fp4 (as opposed to fp6).  Currently, TensorCast will infer exponent width for the "f" code as 2 bits, so that
fp4 is e2m1 and fp5 would be e2m2.

### Positional Modifiers for FP4 and INT4

The "f" or "i" patterns can have a modifier that specifies the number of positions a pattern is shifted down the compute
type number line from the top of the number line.  One or more positions can be specified.  For example, if we want
an fp4 pattern that is shifted 1, 3, 4, and 6 positions from the top of e2m3fnuz, we encode it as `cb42f1346`.

<div align="center">
    <img src="./CB42F1346_dark.png#gh-dark-mode-only", alt="CB42F1346", width=128>
    <img src="./CB42F1346_light.png#gh-light-mode-only", alt="CB42F1346", width=128>
</div>
</br>

The blue pattern is actual fp4, which is in position 3 relative to the top of e2m3.

This mechanism may be used after either the "f" or "i" specifier, or both.  Since the "2" in "cb42" indicates
four mappings, the specifier notation should specify four mappings, otherwise TensorCast will error out.  This
notation can lead to long and illegible strings, which is why the optional label can be added to the overall
specification.

### Implicit Trailing Mantissa Bits

Since the purpose of codebooks is to impart additional precision and/or dynamic range that cannot otherwise be
represented in the limited bits available, an option is provided to create mappings that have one or more implied
trailing mantissa bits.  Suppose the base pattern is e2m1, and an additional mantissa bit was desired. Two patterns
would be generated for compute type e2m2: one pattern of fp4 with trailing 0, the other with trailing 1.  This
notation is `m[<n>]`, where "n" defaults to 1, and is the number of mantissa bits.

Thus, `cb41fm` or `cb41m1` would produce the following:

<div align="center">
    <img src="./CB41FM_dark.png#gh-dark-mode-only", alt="CB41FM", width=128>
    <img src="./CB41FM_light.png#gh-light-mode-only", alt="CB41FM", width=128>
</div>
</br>

This pair of mappings represents all but one of the values of fp5, and could be expected to provide precision at
a level between fp4 and fp5.  When applied to an int4 pattern, the same concept is applied as a fixed point trailing
fractional bit.

In order to have a lossless upcast to the compute type, the compute type must have sufficient mantissa bits available.

### Implicit Exponent Offset

Dynamic range may be addressed by having a pattern shifted such that the highest value has a lower exponent than
the scale would indicate. Such a pattern would never be selected for a scaled tile, but could be useful for a
subtile.  An example of this is with the original MicroXcaling types MX4, MX6, and MX9.  These integer (fixed point)
types had a scale tile size of 16, with subtiles of size 2. In these types, the integer bit is the equivalent of the
implicit bit in floating point, but explicitly stored.  A zero integer bit indicates subnormals.  For low-width
values, a bit of metadata for each subtile (called the "prime" bit) indicates that the values in the subtile are
subnormals, but the leading (integer) bit is implied, so that the trailing bit can be used for additional precision.
In one hardware implementation (not codebooks), a shift of the product is performed based on the presence or absence
of prime bits in the two terms.

In implicit codebooks, the equivalent is to divide the values in the pattern by 2.

<div align="center">
    <img src="./CB51IE_dark.png#gh-dark-mode-only", alt="CB51IE", width=128>
    <img src="./CB51IE_light.png#gh-light-mode-only", alt="CB51IE", width=128>
</div>
</br>

This option does require subtiles, as well as sufficient room in the compute type's dynamic range.  As with the "m"
option, the notation is `e[<n>]`, where "n" defaults to 1, and is the number of offset bits.

### Progressive Patterns

A progressive pattern starts at or near the top of the compute distribution, and moves down the number line in
increasing decrements.  A progressive pattern with an initial decrement of one targeting e2m3fnuz would begin
at 31, decrement by 1 to 30, decrement by 2 to 28, etc.  The benefit is to have more resolution at the highest
magnitudes, yet extend the dynamic range.  The notation is `p<dec>[<offset>]`, where "dec" is the initial decrement
and "offset" (default 0) is the distance from the top of the number line to the start of the progression.

A pair of progressive mappings, `cb41p1p2`:

<div align="center">
    <img src="./CB41P1P2_dark.png#gh-dark-mode-only", alt="CB41P1P2", width=128>
    <img src="./CB41P1P2_light.png#gh-light-mode-only", alt="CB41P1P2", width=128>
</div>
</br>

### Static Patterns

A static pattern starts at an "offset" from the top of the compute distribution and decrements by a fixed number.
The notation is `s<dec>[<offset>]`.

---

[TensorCast Home](../README.md)
