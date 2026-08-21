# filters.hag_delaunay

Status: exact planner-owned masked-2D-index CUDA implementation for explicit
count three, with pinned host repair/fallback. B0237/D0236 automatically
selects only one separately calibrated count-three direct-LAS composition.

## Exact compatibility status

The native lane separates ground and non-ground rows with the unsigned-byte
Classification field and queries the planner-owned 2D kNN index. Ground rows
receive positive zero. Non-ground rows preserve the pinned three-point
Delaunator seed, circumradius, winding, and barycentric operation order,
including the upstream outside sentinel and extrapolation bounds.

Candidate-boundary ties, incomplete bounded-grid searches, unsupported or
runtime-incompatible coordinates, nonfinite arithmetic, and
missing/insufficient ground references retain the existing exact pinned-host
behavior before publication. The stage never builds a private spatial index.
Default compatibility mode preserves artifact bytes, metadata, point order,
stdout, stderr, and exit status.

Default count and count four or wider remain host-owned. Explicit count-three
option variants remain available only inside their documented exact
envelopes; that functional GPU-native coverage is not a general automatic
selection claim.

## Automatic count-three direct composition

B0237 corrects and qualifies exactly this graph:

    readers.las
      -> filters.hag_delaunay(count=3)
      -> writers.las(extra_dims=all)

The JSON root, stages, and option sets must be literal. Both filenames must
end in lowercase .las. The source must be mapped, uncompressed LAS 1.4 point
format 7 with 40-byte records and exactly one unsigned-32 OffsetTime Extra
Bytes descriptor; the output record must be 48 bytes after appending binary64
HeightAboveGround. Point count must be 500,001 through 16,000,002, and the
device must match the qualified RTX 4090/SM89/CUDA 13.3/driver 610.43.03
profile.

The selected plan contains one HAG-Delaunay region, one lane, and one
planner-owned 2D kNN index. Its measured facts are 25 upload bytes and 8 spill
bytes per point with zero packing, plus the full 112-byte-per-point shared
index. Stage-local device scratch is 39 bytes per point; placement and
resident preflight reserve the complete 184-byte-per-point high-water.

Grammar, semantic Extra Bytes layout, cardinality, budget, source, profile,
rewrite, preflight, successful CUDA execution proof, and rejected destinations
remain before commitment. A selected device decline restores the unchanged
public host pipeline; the required form exits 124 without output. Tie and
incomplete rows use the exact established repair. Existing, aliased, and
symlink destinations retain pinned status, diagnostics, and filesystem state.
The no-ground public row preserves the pinned 2.10.0 diagnostic and SIGSEGV
status. Default/count-four, other option sets, compression, other Extra Bytes
layouts, and neighboring devices/profiles remain exact but explicit or
host-selected.

## Performance and verification

The exact direct ladder covers 50,001 through 16,000,002 points. The 400,002
row reaches only 1.068195x pinned PDAL. A fresh nine-pair review run at 450K
reaches only 1.093356x, below the predeclared 10% margin, so automatic
selection begins at 500,001. Final-code nine-pair public medians are 1.245213x
at 500,001 (0.426662 versus 0.342642 seconds) and 2.049354x at 1,000,002
points (0.838204 versus 0.409009 seconds), with all candidate samples faster.

The public matrix proves exact floor, main, and cap selection; 400,002,
450,000, 500,000, and hash-pinned cap-plus-one fallback; exact and one-byte-
below VRAM boundaries;
neighboring grammar/semantic-layout/source/proof and injected-device-decline
refusals; one-query tie and incomplete repair; no-ground diagnostics/status;
and oracle-identical existing/alias/symlink behavior. Same-final-binary stats
prove calibrated placement, the direct resident executor, actual successful
CUDA use, boundary agreement, one region/index/lane, and no host XYZ mirror.
The compiled placement audit matches 214/218 directions, and raw provenance
verifies all 224 unique reports.
