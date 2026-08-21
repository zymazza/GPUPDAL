(filters.skewnessbalancing)=

# filters.skewnessbalancing

**Skewness Balancing** classifies ground points based on the approach outlined
in {cite:p}`bartels2010threshold`.

```{eval-rst}
.. embed::
```

```{note}
For Skewness Balancing to work well, the scene being processed needs to be
quite flat, otherwise many above ground features will begin to be included
in the ground surface.
```

## Example

The sample pipeline below uses the Skewness Balancing filter to segment ground
and non-ground returns, using default options, and writing only the ground
returns to the output file.

```json
[
    "input.las",
    {
        "type":"filters.skewnessbalancing"
    },
    {
        "type":"filters.expression",
        "expression":"Classification == 2"
    },
    "output.laz"
]
```

## Options

```{include} ground_cls_opts.md
```

```{include} filter_opts.md
```

```{note}
The Skewness Balancing method is touted as being threshold-free. We may
still in the future add convenience parameters that are common to other
ground segmentation filters, such as `returns` or `ignore` to limit the
points under consideration for filtering.
```

## PDG exact CUDA coverage

PDG preserves the upstream implementation by default. An internal forced test
mode can accelerate only the whole-view Z-ordering step when Z is stored as
binary64, every logical value is finite, and no two values compare equal
(including `-0.0` and `+0.0`). CUDA produces the exact permutation; PDG then
applies that permutation to every point field and runs the upstream sequential
moment and sign-crossing recurrence on the host in its original arithmetic
order.

Equal or nonfinite Z values, other physical Z types, `where`, multi-view or
order-unsafe graphs, and unsupported options retain the unchanged upstream
path. The current lane is not selected automatically, does not retain a
resident device product, and is not a claim that the complete filter is
GPU-native. B0128 performance-qualifies only the exact forced
comparator-unique 1,000,000-point format-7 fixture on the recorded SM89
machine at 1.192843x pinned PDAL. Its device kernels are only about 0.023555%
of complete-process wall time; broader data and automatic selection remain
unqualified.
