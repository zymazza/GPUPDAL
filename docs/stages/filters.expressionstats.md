# `filters.expressionstats`

Status: exact native host implementation and automatically selected finite-
target CUDA histogram path inside the measured work envelope.

The host implementation preserves pinned PDAL expression preparation,
canonical expression strings, lexicographic statistic order, numeric bin
order, overlapping expressions, duplicate counts, empty statistic entries,
custom dimensions, NaN map behavior, stream/standard execution, metadata, and
errors.

The CUDA envelope gathers only the target and predicate-read dimensions as
binary64 columns, evaluates the existing exact predicate VM, performs stable
device selection, radix-sorts selected values with original indices, and uses
reduce-by-key to emit only compact bins. Each bin retains the first matching
global index, so equivalent `+0` and `-0` keys use the same bit representation
as ordered `std::map<double>` insertion. Nonfinite targets and predicates
outside the exact VM use the host implementation.

Automatic replacement is count- and work-aware. For a linear local LAS/LAZ
reader, exact LAS writer, and cardinality that remains known through any
preceding assign, ferry, or transformation stages, one expression selects CUDA
from 4,000,000 points, two from 2,000,000, and three or more from 1,000,000.
The planner reads only the fixed LAS header, honors the reader `count` limit,
and declines selection across `start`, filtering, partitioning, or any
unproved cardinality change. Unknown counts, smaller inputs, zero expressions,
unsupported values/programs, explicit disable, device absence, and recoverable
CUDA failure retain the exact host or unchanged-PDAL path. Internal
`PDG_EXPERIMENTAL_CUDA_HYBRID` and `PDG_REQUIRE_CUDA_HYBRID` gates remain for
differential validation.

The host fallback avoids device-program compilation and point-id staging when
CUDA is not requested. The CUDA path reuses a stage-local pinned/device
workspace across its batches. A later resident-plan path must still remove the
current host gather and H2D boundary when adjacent native stages already own
the required columns.

Current evidence:

- 19 exact host/fallback complete-process cases in Debug, Release, and
  ASan/UBSan;
- ten forced-CUDA complete-process cases covering standard and streaming
  execution, overlapping predicates, custom intermediates, no-match bins,
  multi-view composition, and exact metadata order;
- host units for map order, counts, signed zero, empty matches, nonfinite
  rejection, and predicate eligibility;
- a 21,970,934-point host differential with exact 790,955,889-byte LAS and
  15,242-byte metadata artifacts;
- a passing physical 131,103-point host/device property on the RTX 4090;
- zero memcheck/initcheck/synccheck errors and zero racecheck hazards for the
  direct property; the ten-case process matrix is also clean under memcheck
  and racecheck, with successful initcheck and synccheck runs;
- clean option-free B0008 medians of 1.126x pinned PDAL at 1,000,000 points
  for three cheap predicates, 1.212x at 2,000,000 for two, and 1.105x at
  4,000,000 for one, with all output bytes identical;
- exact just-below-boundary host controls at 0.957x for 500,000/three,
  0.973x for 1,000,000/two, and 0.979x for 2,000,000/one;
- an option-free one-million-point Nsight Systems trace with 1.337595 ms of
  kernels and 0.563131 ms of device memory operations; its one pinned
  allocation consumes 0.869249 ms and confirms stage-local workspace reuse,
  while first-use stream creation still costs 97.046651 ms;
- zero memcheck errors on that final automatic threshold path; and
- final aggregate gates of 246 passes plus one opt-in skip in 247 Host Debug
  and ASan/UBSan registrations, 347 plus the skip in 348 CUDA Release
  registrations, and 142/142 in PDAL's published suite.

B0008 and D0044 record the accepted selector. The earlier B0007/D0043 work
curve and multi-corpus trials remain useful historical diagnostics, but the
release claim rests on the clean persistent-workspace cheapest-work curve
rather than favorable or single-sample runs.
`filters.stats` and option-free `filters.info` remain host-selected because
their standalone forced-CUDA paths do not beat pinned PDAL.
