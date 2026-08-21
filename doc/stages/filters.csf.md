(filters.csf)=

# filters.csf

The **Cloth Simulation Filter (CSF)** classifies ground points based on the
approach outlined in {cite:p}`zhang2016easy`.

```{eval-rst}
.. embed::
```

## Example

The sample pipeline below uses CSF to segment ground and non-ground returns,
using default options, and writing only the ground returns to the output file.

```json
[
    "input.las",
    {
        "type":"filters.csf"
    },
    {
        "type":"filters.expression",
        "expression":"Classification == 2"
    },
    "output.laz"
]
```

## PDG execution status

PDG has a byte-exact bounded CUDA implementation for a serial,
non-OpenMP subset of `filters.csf`: `smooth=false`, finite logical-double XYZ,
unsigned-byte Classification, a cloth with at least two rows and columns and at
most 4,096 cells, no more than 64 iterations, nonnegative `rigidness`, positive
`resolution`, and the supported return/class options. It preserves the pinned
implementation's coordinate transform, cloth graph order, rasterization and
void-fill order, in-place constraint/collision schedule, strict interpolation,
class writes, point order, and tested diagnostics. It does not build a private
spatial index.

This CUDA lane is available only through forced/experimental or standalone
planner-selected execution. It is not selected automatically: the B0023
same-machine exact benchmark measured 0.289x pinned PDAL at 1,000,000 points.
`smooth=true`, an OpenMP-enabled build, `ignore`, `debug`, `dir`, `where`,
invalid options, nonfinite coordinates, larger cloths, and adjacent Grid
composition retain the original PDAL stage or the exact host wrapper. The
current cloth is neither tiled nor reusable across stages, so this status does
not claim complete CSF or P3 grid coverage.

## Options

resolution

: Cloth resolution. \[Default: **1.0**\]

ignore

: A {ref}`range <ranges>` of values of a dimension to ignore.

returns

: Return types to include in output.  Valid values are "first", "last",
  "intermediate" and "only". \[Default: **"last, only"**\]

threshold

: Classification threshold. \[Default: **0.5**\]

hdiff

: Height difference threshold. \[Default: **0.3**\]

smooth

: Perform slope post-processing? \[Default: **true**\]

step

: Time step. \[Default: **0.65**\]

rigidness

: Rigidness. \[Default: **3**\]

iterations

: Maximum number of iterations. \[Default: **500**\]

```{include} ground_cls_opts.md
```

```{include} filter_opts.md
```
