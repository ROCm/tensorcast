<!-- markdownlint-disable MD033 MD041 -->

# Codebooks

A codebook is a mechanism by which the values in a tile (or block, or group) of a tensor can be represented by an
index that selects a compute value from a lookup table.  Because each individual tile may best quantize to a
different subset of compute values, a codebook has multiple lookup tables, which are indexed by a value shared by
the tile, similar to the sharing of a scale factor.  Unlike the scale factor, the metadata that selects the
lookup table can optionally be at the subtile level.

Each lookup table contains 2<sup>index_bits</sup> compute values, where *index_bits* is the width of each encoded
value in the tile, and in practice should be between 2 and 6 bits. The values in the lookup table are a (possibly
very small) subset of the values representable in the compute datatype.  Selection of the lookup table itself
is done via metadata at the tile or subtile level.  There are 2<sup>metadata_bits</sup> lookup tables in the codebook.

The lookup tables in the codebook and the mappings from index to compute value in each lookup table are typically
determined through clustering methods.  A codebook is a shared resource, which makes its creation and population
a challenging optimization problem.

Codebooks in TensorCast are of two types: a generic codebook in which the codebook mappings (lookup tables)
are manually added to the NumberSpec instance after instantiation; and "implicit" codebooks in which the mappings are
generated during instantiation from the string code passed to the Codebook constructor.  Beyond that
difference, they have the same cast/compression/upcast behavior.

A codebook consists of:

* an unscaled compute type such as e4m3fn, e2m3fnuz, etc.
* the number of metadata bits needed to select a mapping
* the number of index bits needed to select a compute value within a mapping
* the matrix of size 2<sup>metadata_bits</sup> x 2<sup>index_bits</sup> containing the compute values to look up

General codebook definition is outside the scope of TensorCast; there is no current support for clustering,
centroid optimization or selection of weights that share a codebook. Dynamic cast is supported via a binary search
through each mapping in the codebook, with the mapping selection determined by a metric supplied through
the cast call.  This is inherently slower than a standard number spec, and illustrates the challenges for
dynamic quantization of tensors during inference (activations) and training (weights, activations, gradients).
<br></br>

## Codebook String Codes

A codebook is defined with a string that has either two or three substrings separated by underscores.  The first
describes the codebook and the second is the number specification code for the compute type with which the codebook
will be populated.  A third, optional substring is a concise name by which that codebook will be distinguished from
other codebooks, as the initial description may be too vague ("cb42") or too long and cryptic ("cb42f1346").

The generic codebook description is of the form:

```text
    cb<index_bits><metadata_bits>[<desc>]_<cspec>[_<label>]
```

`index_bits` is a single digit that is the bit width of the values that are indices into a specific
mapping lookup table. `metadata_bits` is a single digit that is the number of bits of metadata that select the
mapping lookup table. `desc` is a code specific to implicit codebook mapping creation (described
[below](#implicit-codebook-description-encoding)) and is not used for generic codebooks.  `cspec` is the
`NumberSpec` code for the compute type contained in the codebook lookup tables. `label` is the optional
alternative codebook name.

The codebook dimensions, then, are 2<sup>metadata_bits</sup> and 2<sup>index_bits</sup>, and the codebook entries
are the size of the compute values (`NumberSpec(cspec).bits`).
<br></br>

## Codebook Creation

The tcast API includes a creation function called `number` that takes a string code or a `torch.dtype` and
returns either a `NumberSpec` or a `Codebook` (which is a subclass of `NumberSpec`).  A `Codebook` is returned
if the input is a string beginning with "cb" (case insensitive).

```python
    # generic codebook, 16x16 fp8
    codebook = tcast.number("cb44_e4m3fn_mycodebook")
    # populate with codebook matrix
    codebook.add_mappings(maplists)
    # implicit codebook                
    implicit_codebook = tcast.number("cb41fi_e2m3fnuz")
```

Codebooks can also be created via constructor:

```python
    codebook = tcast.Codebook("cb44_e4m3fn_mycodebook")
```

<br></br>

## Implicit Codebook Description Encoding

Implicit codebooks are fixed codebooks that are able to be upcast from encodings to compute values directly
(or nearly so) in hardware, without the need to store or transfer actual codebook tables.  In TensorCast they
can also be thought of as a shortcut to populate codebooks algorithmically to simplify experimentation.

The compute type is viewed as a number line, with all representable values having a corresponding unsigned
integer index. For e2m3fnuz, for example, there are 64 values, with indices 0-63.  Implicit codebooks are
*symmetric*, in that each positive value also has a negative, and both 0 and -0 are represented (although the
latter is `NaN`).  Thus a codebook mapping with 16 entries has 7 unique magnitudes.

The implicit mappings are pattern-based.  A pattern is a set of indices in the positive value space of the
compute type that can be shifted as a group up and down the number line.  The description establishes patterns
and the topmost index of the pattern.  Each shifted pattern is then prefixed with the index for 0.0, then mirrored
to include negatives, and converted from indices to the actual values and added to the codebook as a mapping.
The number of mappings described in the encoding must match the number of mappings specified by `metadata_bits`.

There are four types of patterns used:

* "f": floating point distribution
* "i": integer distribution
* "p": progressive distribution (distance between value indices into the compute number line increase with each value)
* "s": static distribution (distance between value indices into the compute number line is constant)

The codebook mappings are variations of these patterns.  These mappings are specified through one or more clauses
beginning with "f", "i", "p", or "s", and containing modifiers with further information.

>Note: syntactically legal encodings will raise an exception if any resulting mappings cannot be represented
in the compute type.

### F pattern (floating point)

This pattern is the set of positive values in fp\<index_bits\>, upcast to the compute type.  It is only valid for
3, 4, or 5 index bits. The number format is one of e2m0, e2m1, e2m2.  **Scaled to (-2, 2)**, the positive
values are:

format| values
-------|--------
e2m0   | 0.25, 0.5, 1.0
e2m1   | 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5
e2m2   | 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75

These values are then mapped to the unscaled compute number line (in this case, e2m3fnuz):

format | e2m3 values | e2m3 indices
-------|--------|------
e2m1   | 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 | 36, 40, 44, 48, 52, 56, 60

From here, the pattern is shifted to the top of the distribution:

format | e2m3 indices, shifted to maxval
-------|--------------
e2m1   | 39, 43, 47, 51, 55, 59, 63

These are the indices that are shifted down by a number specified by the offset indicated by the modifier.  The default
offset for e2m1 is depends on the compute datatype (which results in the values before the shift to the top, i.e.
pure e2m1fnuz).  Otherwise, the offset is in [0, 9], and each index is decremented by that offset.

>Note that adjacent fp4 values will not have the same ratio (75% or 50%) after being shifted by index.

The simple syntax for the modifiers of the "f" clause is `f[<offsets>]`, where `offsets` is optional, and when
present is
one or more digits indicating the decrement from the top of the compute number line.  If `offsets` is omitted, the
default for that fp3/fp4/fp5 is used, and one mapping is generated, otherwise the number of offsets is the number of
mappings generated.

#### E modifier

An alternative syntax adds the "e" modifier, with the syntax `f[<offsets>][e[<eoffsets>]]`.  Here, `eoffsets` is one or
more of 0, 1, 2, and 3, and modifies the pattern(s) generated by `f[<offsets>]` by an additional decrement that reduces
the exponent in the compute value by the number(s) in `eoffsets`.  This is only useful when using subtiles because
it serves to target a pattern to a subtile that has no values with the same exponent as the tile exponent, and is one
option for dealing with outliers by reducing underflow.

If the "e" is present with no trailing digits, behavior is equivalent to "e01".  The number of mappings generated by
each "f" clause is the number of offsets times the number of eoffsets (if any, otherwise 1).

### I pattern (integer)

This pattern is the set of positive values in int\<index_bits\>, upcast to the compute type.  It is only valid for
index_bits i3, 4, or 5. The number format is one of int3, int4, int5.  Scaled to (-2, 2), the positive values are:

format| values
-------|--------
int3   | 0.25, 0.5, 1.0
int4   | 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75
int5   | 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875

From here, the syntax and behavior are identical to the "f" pattern, including the "e" modifier.  An example
with the "e" modifier (`cb61ie_e2m3fnuz_mx6`) can be seen [here](#implicit-exponent-offset).  An example of
combining the "f" and "i" clauses (`cb41fi_e2m2fnuz`) can be seen [here](#fp4-and-int4-combined-pattern).

### P pattern (progressive)

This pattern is not based on an fp or int distribution, but instead is entirely index based.  The syntax
of the clause is `p<interval>[<offsets>]`, where `interval` is the initial decrement (from the first/highest
index in the pattern to the second).  Each subsequent pattern index is decremented by `interval + 1`, `interval + 2`,
etc.  The `offsets` are as in the other pattern types; the default is 0.  A "p" with no modifiers is permitted,
and will be equivalent to `p10`.

```python
# produce positive value indices for progressive pattern
pclause = "p10246"
interval = 1 if len(pclause) == 1 else int(pclause[1])
offsets = [0] if len(pclause) == 2 else [int(c) for c in pclause[2:]]
maxidx = 2 ** compspec.bits - 1
numpos = 2 ** (index_bits - 1) - 1
patterns = []
for o in offsets:
    pattern = []
    incr = 0
    for i in range(numpos):
        pattern.append(maxidx - o - incr)
        incr += interval * (i + 1)
    patterns.append(list(reversed(pattern)))
```

Zero or more "p" clauses can be combined with any other pattern type clause.

### S pattern (static interval)

This pattern is like the progressive pattern in every way except that the interval remains the same throughout
the pattern.  This method can concentrate the values in the upper exponents for more precision, or spread them out
for more dynamic range.
<br></br>

## Implicit Codebook Examples

>*Some examples in this section use compute types that are smaller than what might be implemented for a
given platform. This is to minimize the diagram size and complexity.*

### FP4 and INT4 Combined Pattern

Both fp4 and int4 upcast losslessly to fp5.  Whereas int4 has more than half its positive values in the top exponent,
fp4 have 2, 2, 2, and 1 positive values across the top four exponents.  This will lead to fp4 being the preferred
mapping roughly 70% of the time for weights.

<div align="center">
    <img src="./images/CB41FI_dark.png#gh-dark-mode-only", alt="CB41FI", width=200>
    <img src="./images/CB41FI_light.png#gh-light-mode-only", alt="CB41FI", width=200>
</div>

This example would be encoded as `cb41fi_e2m2fnuz`, with "f" indicating a standard fp4 (e2m1) pattern, and
"i" indicating a standard int4 pattern.

The "4" in the decription indicates the number of bits in each encoded value.  This tells the codebook that the pattern
for "f" is fp4 (as opposed to fp6).  Currently, TensorCast will infer exponent width for the "f" code as 2 bits,
so that fp4 is e2m1 and fp5 would be e2m2.

### Positional Modifiers for FP4 and INT4

The "f" or "i" patterns can have a modifier that specifies the number of positions a pattern is shifted down
the compute
type number line from the top of the number line.  One or more positions can be specified.  For example, if we want
an fp4 pattern that is shifted 1, 3, 4, and 6 positions from the top of e2m3fnuz, we encode it as `cb42f1346_e2m3fnuz`.

<div align="center">
    <img src="./images/CB42F1346_dark.png#gh-dark-mode-only", alt="CB42F1346", width=200>
    <img src="./images/CB42F1346_light.png#gh-light-mode-only", alt="CB42F1346", width=200>
</div>

The blue pattern is actual fp4, which is in position 3 relative to the top of e2m3.

This mechanism may be used after either the "f" or "i" specifier, or both.  Since the "2" in "cb42" indicates
four mappings, the specifier notation should specify four mappings, otherwise TensorCast will error out.  This
notation can lead to long and illegible strings, which is why the optional label can be added to the overall
specification.

### Implicit Trailing Mantissa Bits

Since the purpose of codebooks is to impart additional precision and/or dynamic range that cannot otherwise be
represented in the limited bits available, an option *could* provided to create mappings that have one or more implied
trailing mantissa bits.  Suppose the base pattern is e2m1, and an additional mantissa bit was desired. Two patterns
would be generated for compute type e2m2: one pattern of fp4 with trailing 0, the other with trailing 1.  However,
this can be accomplished with the previoulsy descibed syntax: `cb41f01_e2m2fnuz`:

<div align="center">
    <img src="./images/CB41FM_dark.png#gh-dark-mode-only", alt="CB41FM", width=200>
    <img src="./images/CB41FM_light.png#gh-light-mode-only", alt="CB41FM", width=200>
</div>

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

The specifier for MX6 using fp6 compute would be `cb51ie_e2m3_mx6` or `cb51ie01_e2m3_mx6`.

In implicit codebooks, the  equivalent is to divide the values in the pattern by 2.

<div align="center">
    <img src="./images/CB51IE_dark.png#gh-dark-mode-only", alt="CB51IE", width=200>
    <img src="./images/CB51IE_light.png#gh-light-mode-only", alt="CB51IE", width=200>
</div>

### Progressive Patterns

This example shows two progressive patterns, both with offset 0, with 1 and 2 as the initial interval, mapped to
compute fp6e2: `cb41p1p2_e2m3fnuz`:

<div align="center">
    <img src="./images/CB41P1P2_dark.png#gh-dark-mode-only", alt="CB41P1P2", width=200>
    <img src="./images/CB41P1P2_light.png#gh-light-mode-only", alt="CB41P1P2", width=200>
</div>

---

[Documentation](./README.md)
</br>
[Testing](../tests/README.md)
</br>
[Examples](../examples/README.md)
</br>
[TensorCast Home](../README.md)
