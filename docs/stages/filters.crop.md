# filters.crop

`filters.crop` selects points inside or outside spatial bounds. The first
native envelope targets the common bounding-box operation and lowers it to the
same stable predicate/compaction primitive used by expression and range
filters.

## Exact compatibility status

Native execution accepts exactly one parenthesized PDAL bounds string with two
or three intervals:

```json
{"type":"filters.crop","bounds":"([0,10],[20,30],[-5,5])"}
```

The `outside` option may be omitted or supplied as a JSON Boolean. Faces are
inclusive. A NaN coordinate is outside the box, and therefore survives an
`outside: true` crop, matching `BOX2D`/`BOX3D`. Survivor order is unchanged.
Adjacent assign, ferry, expression, range, and crop operations stay in one
ordered region, including custom intermediate dimensions.

Multiple bounds, `a_srs`, center/distance crops, polygons, OGR geometry,
tagged or branching graphs, and option-rich forms remain on unchanged PDAL.
These forms are fully functional through the product shell but are not counted
as GPU-native. Eligibility and rewritten-pipeline preparation complete before
point data or output is emitted.

## CUDA status

Both the specialized uncompressed-LAS path and the in-process packed bridge
can evaluate the exact envelope on CUDA. Device execution uses the existing
bounded expression VM and stable compaction, then recomputes output count,
bounds, and logical return summaries before publication. The ordinary
in-process host VM remains the exact path when CUDA is unavailable or not
selected.

No crop-only automatic-CUDA performance claim is made yet. Automatic selection
expands only after a clean same-machine break-even matrix shows a win; force and
require switches are retained solely for differential and tuning gates.

## Verification

Unit tests cover 2D/3D parsing, whitespace, inclusive minimum/maximum faces,
inside/outside inversion, and NaNs. Exact process differentials cover direct
and hybrid Host/CUDA execution, two-point CUDA chunks, empty standard-mode
input, stable selective output, same-SRS input, and fallback for multi-bounds,
reprojection, and invalid bounds. Reader matrices exercise the fused crop
through LAS, LAZ, BPF, PLY, PCD, text, and sorted COPC. The crop CUDA primitive
passes memcheck, initcheck, synccheck, and racecheck with zero findings.
