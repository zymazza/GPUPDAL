# `filters.radialdensity`

Status: exact host replacement and shared-radius CUDA implementation, including
deterministic capacity-driven tiling. Representative real-corpus diagnostics
are exact and faster, but automatic CUDA selection is not yet qualified from a
clean tree.

The stage preserves upstream PDAL's default radius of `1`, registration and
publication of the binary64 `RadialDensity` dimension, debug message, and
literal density factor:

```text
1.0 / ((4.0 / 3.0) * 3.14159 * (radius * radius * radius))
```

For every point, the exact path counts the 3D neighbors satisfying PDAL's
strict `distance < radius²` predicate, including the query point and duplicate
coordinates, converts the count to binary64, and performs one round-to-nearest
multiplication by the factor. The host and CUDA implementations keep that
operation order; they do not substitute a different value of pi or a sphere
volume routine.

The CUDA envelope requires a finite positive radius, finite logical-double XYZ,
and a point count representable by each compact index tile. Empty input is an
exact no-op. Nonpositive or nonfinite radii, nonfinite coordinates,
`where`/`where_merge`, string-valued or unknown options, and unproven layouts
retain the unchanged upstream KD3 stage. These are compatibility boundaries,
not claims that the stage is complete merely because fallback exists.

## Capacity-driven execution

The shared radius executor first checks the live device budget: the lesser of
20 GiB, 75% of currently free memory, and 75% of total memory, using a
conservative 128 bytes per point. A fitting view uses one shared uniform-grid
query. A larger view is partitioned into deterministic half-open XY core tiles
with a closed halo equal to the radius; the neighbor query remains 3D. Every
source point owns exactly one core tile, tile members retain source order, and
ghosts participate in queries but are never published. Only owner rows scatter
back into the original PointView order, and execution fails rather than
publishing a partial mosaic if a tile including ghosts exceeds capacity.

Two-dimensional ownership is conservative for a 3D query because it may carry
extra Z-separated candidates but cannot omit an in-radius neighbor. The
reverse—3D partitioning for a 2D query—is rejected. Exact distance testing in
the spatial index remains authoritative at halo faces.

`PDG_SPATIAL_TILE_EDGE` and `PDG_REQUIRE_SPATIAL_TILING` are internal
differential/tuning gates, not public behavior. The current implementation
uses two reusable pinned/device allocation lanes whenever there is more than
one tile. A lane is scattered and recycled only after its completion event;
the other stream can continue transfers, index construction, query work, or
copy-back. Pageable staging uses one reusable lane. Plan-wide tiled residency
across adjacent stages remains P2 performance work; no speedup is claimed for
this stage yet.

## Verification

Three host tiling units pin negative and exact-face ownership, one owner per
point, sorted source membership, closed ghosts, ghost-mask transfer,
owner-only scatter, empty input, invalid frames, nonfinite coordinates, and
capacity rejection. Whole-view and tiled radius counts and scaled values are
compared bit for bit across three different tile edges.

On the physical RTX 4090, an 8,323-point seam-heavy property compares every
tiled count and scaled binary64 value with the whole-view host result. Separate
forced whole-view and forced multi-tile process differentials produce the same
LAS artifact, streams, status, and diagnostics as upstream. A ten-case process
matrix covers the default, exact radius boundary, expanded/tiny/large radii,
negative and zero values, empty input, conditional fallback, and string-option
fallback. The complete forced-tiled pipeline and primitive properties are
clean under Compute Sanitizer memcheck, initcheck, racecheck, and synccheck,
including repeated two-lane recycling, exceptional teardown with another lane
in flight, and caller-owned pool recovery. The units and ten-case matrix are
clean under ASan/UBSan with leak detection.

Complete-process dirty-tree diagnostics are exact on independent ALS and TLS
corpora in both LAS and LAZ. A five-sample, 21,970,934-point ADKLR run forced
through 16 tiles has an 8.164x median speedup over pinned PDAL; one-shot local
RGB/NIR LAS and VEIL LAZ trials at 5,114,283 and 36,772,046 points are
7.335x and 20.228x. A ten-sample 250,000-point ALS prefix is 2.104x, while
smaller prefix probes preserve the expected CUDA-startup crossover. Every
reported artifact is byte-identical to the oracle, including every warm-up and
measured output in repeated runs.

Those retained reports are diagnostic rather than accepted release benchmarks:
the working tree was dirty and the independent corpus checks were one-shot.
Automatic selection remains disabled until clean-tree, same-machine,
density-stratified break-even trials pass and tiled outputs can remain resident
across compatible stages.
