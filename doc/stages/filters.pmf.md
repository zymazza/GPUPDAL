(filters.pmf)=

# filters.pmf

The **Progressive Morphological Filter (PMF)** is a method of
segmenting ground and non-ground returns. This filter is an implementation
of the method described in
{cite:p}`zhang2003progressive`.

```{eval-rst}
.. embed::
```

```{warning}
{ref}`filters.smrf` performs much better, with fewer degenerate behaviors. You should
prefer this over `filters.pmf` in most cases.
```

## Example

```json
[
    "input.las",
    {
        "type":"filters.pmf"
    },
    "output.las"
]
```

## Notes

- [slope] controls the height threshold at each iteration. A slope of 1.0
  represents a 1:1 or 45º.
- [initial_distance] is \_intended\_ to be set to account for z noise, so for a
  flat surface if you have an uncertainty of around 15 cm, you set
  [initial_distance] large enough to not exclude these points from the ground.
- For a given iteration, the height threshold is determined by multiplying
  slope by [cell_size] by the difference in window size between the
  current and last iteration, plus the [initial_distance]. This height
  threshold is constant across all cells and is maxed out at the
  [max_distance] value. If the difference in elevation between a point and its
  “opened” value (from the morphological operator) exceeds the height threshold,
  it is treated as non-ground.  So, bigger slope leads to bigger height
  thresholds, and these grow with each iteration (not to exceed the max).  With
  flat terrain, keep this low, the thresholds are small, and stuff is more
  aggressively dumped into non-ground class.  In rugged terrain, open things up
  a little, but then you can start missing buildings, veg, etc.
- Very large [max_window_size] values will result in a lot of potentially
  extra iteration. This parameter can have a strongly negative impact on
  computation performance.
- [exponential] is used to control the rate of growth of morphological window
  sizes toward [max_window_size]. Linear growth preserves gradually changing
  topographic features well, but demands considerable compute time. The default
  behavior is to grow the window sizes exponentially, thus reducing the number
  of iterations.
- This filter will mark all returns deemed to be ground returns with a
  classification value of 2 (per the LAS specification). To extract only these
  returns, users can add a {ref}`expression filter<filters.expression>` to the pipeline.

```json
{
  "type":"filters.expression",
  "expression":"Classification == 2"
}
```

```{note}
{cite:p}`zhang2003progressive` describes the consequences and relationships of the parameters
in more detail and is the canonical resource on the topic.
```

## PDG execution status

PDG has a byte-exact bounded CUDA implementation for finite logical-double XYZ,
unsigned-byte Classification, a global raster of at most 4,096 cells,
morphology radius at most 64, and at most 64 progressive schedule passes. A
required planner-resident lane can also execute larger budget-fitting frames
with deterministic one-cell halo tiles. When the complete proof/phase pair
fits, the raster is constructed and proved in the provisional device allocation
that becomes the morphology backings; larger admitted frames retain the exact
host-build/tiled phase path.

Both lanes preserve the pinned implementation's distinct initial and final
cell-binning expressions, first-source minimum bits, return filtering, strict
height comparison, class writes, point order, and tested diagnostics. The
planner-owned raster is constructed once per PMF stage. When the complete
16-byte-per-cell proof/phase pair fits, CUDA selects the first source for equal numeric minima,
compacts original populated cells, and proves every target against every
compact source using precise binary64 arithmetic. Identical-bit
equal-distance sources are accepted; distinct bits discard the provisional
workspace before device Classification materialization/H2D or any
Classification mutation. The wrapper catches only that named ambiguity and
runs pinned upstream on a private copy of the untouched view. Success promotes
that same allocation into the two morphology backings, so the completed raster
is never transferred. One byte
below the complete proof boundary, the exact host path
uses a bounded literal nearest scan through 255 populated cells and an
occupancy hierarchy above it. Nonfinite represented centers, unsupported
options such as `ignore` and `where`, and budget failures retain the original
PDAL stage or fail closed when CUDA is explicitly required.

Compatible contiguous PMF stages can share one planner product and the same
promoted device phase allocation when their cell/Grid contract and original
JSON `returns` value are identical. Each stage still reconstructs and consumes
its own exact raster generation because morphology overwrites both phase
backings; no raster or surface contents are reused. Different return sources,
different cell frames, non-PMF Grid consumers, and cross-kind Grid/cloth
composition remain fail-closed boundaries.

CUDA execution remains forced/experimental or standalone planner-selected and
is not selected automatically. B0030 includes planner-product setup and all
transfers, records zero raster H2D/D2H bytes, and measures controlled exact
results from 3.156x to 38.950x pinned PDAL across dense and sparse
65x65-513x513 frames. The exact device tie proof still scans every compact
source per target cell, and the host occupancy hierarchy can still degenerate;
neither has a proved worst-case subquadratic bound. The Grid also cannot yet be
reused semantically by an adjacent stage. Corrected B0031 includes one shared
product, two complete wrapper generations, resident boundaries, return
selection, and a complete six-column readback. It records zero raster
transfers, byte-exact output, and 1.039838x pinned PDAL on its controlled 65x65
pair with symmetric worker teardown and comparison outside timing; the product
unit separately proves one backing allocation. The earlier 4.590078x,
1.056201x, and comparison-inclusive 1.016953x measurements are superseded.
These results do not
claim general PMF scalability, automatic placement, complete PMF coverage,
cross-kind reuse, or P3 completion.

## Options

cell_size

: Cell Size. \[Default: 1\]

exponential

: Use exponential growth for window sizes? \[Default: true\]

ignore

: Range of values to ignore. \[Optional\]

initial_distance

: Initial distance. \[Default: 0.15\]

returns

: Comma-separated list of return types into which data should be segmented.
  Valid groups are "last", "first", "intermediate" and "only". \[Default:
  "last, only"\]

max_distance

: Maximum distance. \[Default: 2.5\]

max_window_size

: Maximum window size. \[Default: 33\]

slope

: Slope. \[Default: 1.0\]

```{include} ground_cls_opts.md
```

```{include} filter_opts.md
```
