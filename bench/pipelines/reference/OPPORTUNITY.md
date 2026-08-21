# Expanded reference-suite opportunity ledger

This ledger applies D0239's speed-first rule to the exact r1-r14 baseline and
B0241--B0256's measured follow-ups.  It ranks complete workflows, not stage
count.  A row may advance
only after its pinned-PDAL baseline, complete-process profile, exactness gate,
and refusal boundary are reproducible.  Supporting r10/r14 variants have zero
headline weight.

Reference machine: Ryzen 9 7900, RTX 4090/SM89, driver 610.43.03, CUDA 13.3,
Release `-O3 -DNDEBUG`, NVMe/ext4, frozen UTC epoch 1704067200.  Each headline
row has one warmup and three alternating measured pairs.  Cold rows use
file-scoped `POSIX_FADV_DONTNEED` before every process and no warmup.

| Priority | Workload | Warm PDAL / PDG | Cold PDAL / PDG | Current direction |
| ---: | --- | ---: | ---: | --- |
| closed | `r14-convert-compress` | B0269 final 9-pair 0.442823 / 0.166724 s (**2.656026x**) | B0269 final 9-pair 0.452086 / 0.172929 s (**2.614287x**) | B0251 selects exact two-worker fixed-50K chunk compression only for the literal 1M format-7 reference layout; B0252 requalifies the fail-closed final launcher. The production primitive is 1.735238x; 13 cases/21 executions prove bytes, activation, refusals, and determinism. Six variants remain serial host-selected. B0268 packs the writer's records in parallel in the streaming batch hook (from 1.52x). B0269 parallel record unpacking (from 2.08x). |
| closed | `r10-decimate` | B0269 final 9-pair 0.390266 / 0.269736 s (**1.446846x**) | B0269 final 9-pair 0.393065 / 0.276087 s (**1.423698x**) | B0249 retains the literal direct-host route because it resolves 1.041226x/1.034230x warm/cold against the same-final engine path. B0254's four-worker prototype resolves warm but is unresolved cold at paired 1.002292x +/- 0.009367, so r10 retains the default reader; format-7 decode, not voxel selection, dominates. B0269 parallel record unpacking (from 1.14x). |
| closed | `r7-dsm` | B0273 final 9-pair 0.219415 / 0.169673 s (**1.293160x**) | B0273 final 9-pair 0.223505 / 0.177176 s (**1.261488x**) | B0254 retains four ordered reader workers only for the literal 1M compressed format-7/36-byte r7 route. Complete TIFF metadata/bytes, streams, status, and refusals are exact; every neighboring shape keeps its prior path. B0269 parallel record unpacking (from 1.03x). B0269 parallel record unpacking took it from parity to 1.32x; B0273 banded raster accumulation is neutral for bin-mode max (writer about 0.02 s). |
| closed | `r9-polygon-clip` | B0269 final 9-pair 0.357511 / 0.188027 s (**1.901385x**) | B0269 final 9-pair 0.359444 / 0.188024 s (**1.911693x**) | B0247 fixed the axis-swapped empty workload and rejected direct-delegation prototypes; the exact host route stayed a 0.94x/0.95x loss until B0259 made four-worker LAZ chunk compression the writer default, which turns r9 positive without a route. B0269 parallel record unpacking (from 1.35x). |
| closed | `r12-tile` | B0269 final 9-pair 0.480788 / 0.246977 s (**1.946695x**) | B0269 final 9-pair 0.483505 / 0.254122 s (**1.902648x**) | B0246 rejected direct delegation and 1/2/4/6/8 reader-worker tuning; the frozen engine route was a 0.97x loss until B0259 defaulted the seven LAZ tile writers to four compression workers. B0268 parallel record packing for all seven tiles (from 1.20x). B0269 parallel record unpacking (from 1.46x). |
| closed | `r13-merge` | B0269 final 9-pair 0.498510 / 0.185848 s (**2.682353x**) | B0269 final 9-pair 0.501952 / 0.185886 s (**2.700326x**) | B0245 rejected the one-worker route (warm-positive, cold-unresolved). B0259 makes four-worker LAZ chunk compression the writer default; r13 becomes a resolved 9/9 win in both states without reviving the prototype. B0268 parallel record packing (from 1.44x). B0269 parallel record unpacking (from 1.95x). |
| closed | `r8-colorize` | B0266 final 9-pair 1.659275 / 0.788347 s (**2.104750x**) | B0266 final 9-pair 1.665351 / 0.795523 s (**2.093405x**) | B0243 rejected the exact direct-delegation prototype. B0266 attributes the wall to two per-point PROJ reprojections and makes `filters.reprojection` exact and parallel in streaming and standard modes (fixed-slot pool, cloned transformations, scratch-then-commit with serial fallback on any failure); colorization and LAZ I/O are unchanged. |
| closed | `r5-copc-query` | B0270 final 9-pair 0.267451 / 0.146114 s (**1.830419x**) | B0270 final 9-pair 0.287872 / 0.148106 s (**1.943686x**) | Parity since D0239 because the reference pins `requests=1` for deterministic tile order. B0270 keeps the one request thread but decompresses on a decode pool and emits tiles in pinned fetch order; `filters.stats` (0.03 s) is now the exposed serial remainder. |
| closed | `r4-denoise-thin` | B0274 final 9-pair 4.625770 / 0.552174 s (**8.377378x**) | B0274 final 9-pair 4.648073 / 0.546866 s (**8.499474x**) | B0227 selected a literal 1M CUDA statistical-outlier route (3.69x). B0258's worker kNN passes, B0267's hashed sample table, and B0268/B0269's parallel LAS I/O made the exact host path faster at 1M and 4M, so B0272 retires the selector; r4 now runs the general host path (from 5.31x). B0274 concurrent KD builds (from 7.42x). |
| active | `r11-classify-refine` | B0274 final 9-pair 7.095794 / 0.550370 s (**12.892762x**) | B0274 final 9-pair 7.381037 / 0.618992 s (**11.924286x**) | B0258 fixed-chunk host worker passes; B0259 default four-worker LAZ compression; B0261 pinned private-index outlier on cached backing; B0263 cached-coordinate KD3 as the published default with a mutation epoch (classifier tree/pass 3--4x cheaper). Forcing the CUDA statistical outlier alone is exact but slower; a shared-device SMRF -> outlier -> classifier composition remains the recorded device candidate. B0274 concurrent KD builds (from 8.87x). |

All eight added headlines now have a complete measured slice. B0254 completes
the LAZ reader screen, rejects the forced-inline hot-loop prototype, selects
four workers only for the bounded r7 headline, and rejects r10 promotion after
its cold interval spans parity. B0251/B0252 close the writer hypothesis with a
bounded exact selector; B0250's earlier r14 reader-worker sweep also remains
rejected. B0249 closes r10
after exact direct-host dispatch removes unused engine startup in both cache
states; its profile shows voxel selection itself is not the opportunity.
B0248 closes r7 after the same bounded startup removal. B0247 closes corrected r9
after direct-exec and private embedded-CLI prototypes resolve slower. B0246
closes r12 after both direct delegation and reader-worker tuning fail
their gates. B0245 closes r13's warm-positive but cold-unresolved one-reader-
worker prototype. B0243 rejected r8's exact but unproved direct-host prototype,
and B0241 closed r11's regression without selecting a new native backend. B0256
supersedes B0255's unsafe cache adoption, removes r11's redundant second KD3
build only after an explicit fresh-tree invalidation, and resolves
1.029892x/1.037311x warm/cold without changing its host route; the explicit
large-LOF cache lane
remains exact and measured 20.295840x pinned PDAL on its 4M proof. B0257
rejected the neighbor-vote seam by attribution (about 0.1% of wall) and B0258
took the measured limiter instead: the two serial per-point kNN passes now run
on exact fixed-chunk host workers, moving r11 to 4.309167x/4.345954x
warm/cold. The largest remaining absolute candidate walls are r6 (about
2.0 s), r8 (1.66 s), r11 (1.62 s), r4 (1.26 s), and r2 (1.21 s).

B0259/D0258 then made the exact four-worker LAZ chunk compressor the
`writers.las` default after a same-binary engine probe over all fourteen
headlines: r6 6.316372x/6.387622x, r13 1.437716x/1.434382x, r1
1.081819x/1.101200x warm/cold over nine pairs; r9, r12, r8, r2, and r11 also
improve; non-LAZ workloads unchanged; all exact. Remaining candidate walls
after B0259 (warm medians): r8 1.48 s, r11 1.47 s, r6 1.42 s, r4 1.21 s,
r2 1.02 s, r3 0.66 s.

B0261 restores pinned outlier semantics on the cached backing after B0260's
defect finding: r11 5.207843x/5.267489x warm/cold; other rows within noise.

B0262 makes r6's exact tie repair keep a private cached tree when the planner
proves the writer alone follows the region: r6 8.648309x/8.664582x warm/cold.
Remaining warm candidate walls: r8 1.52 s, r11 1.37 s, r4 1.23 s, r2 1.06 s,
r6 1.05 s, r3 0.66 s.

B0263 makes cached-coordinate KD3 the published default with refresh-at-reuse:
r11 7.372457x/7.286586x warm/cold. Remaining warm candidate walls: r8 1.50 s,
r4 1.21 s, r2 1.05 s, r6 1.04 s, r11 0.97 s, r3 0.67 s.

B0264 runs the exact host SMRF's morphology, void fills, and point passes on
fixed-chunk workers: r3 1.352415x/1.338430x (from parity), r11
8.867727x/8.863616x. Remaining warm candidate walls: r8 1.49 s, r4 1.22 s,
r6 1.04 s, r2 1.05 s (fork SMRF port still serial), r11 0.79 s, r3 0.49 s.

B0265 pools the fork SMRF port's morphology (r2 1.624190x/1.604373x).
Remaining warm candidate walls: r8 1.50 s, r4 1.22 s, r6 1.05 s, r2 0.95 s,
r11 0.79 s, r3 0.49 s.

B0266 makes reprojection exact and parallel (r8 2.104750x/2.093405x, r1
1.487025x/1.458954x). Remaining warm candidate walls: r4 1.22 s, r6 1.05 s,
r2 0.95 s, r11 0.80 s, r8 0.79 s, r3 0.50 s.

B0267 hashes the sample filter's voxel table (r4 5.306974x/5.271122x).
Remaining warm candidate walls: r6 1.08 s, r2 0.95 s, r4 0.89 s, r11 0.84 s,
r8 0.81 s, r3 0.49 s.

B0268 packs `writers.las` records in parallel in both modes (r14
2.084120x/2.048271x, r13 1.949116x/1.963703x, r12 1.456797x/1.459860x, r6
9.136681x/9.095218x, r1 1.573685x/1.580186x; every LAZ-writing row gains).
Remaining warm candidate walls (B0268 suite medians): r6 1.02 s, r2 0.88 s,
r4 0.85 s, r11 0.76 s, r8 0.69 s, r3 0.49 s, r10 0.35 s, r12 0.33 s, r1
0.30 s.

B0269 unpacks `readers.las` records in parallel in both modes (r14
2.656026x/2.614287x, r13 2.682353x/2.700326x, r7 1.320508x/1.307258x, r9
1.901385x/1.911693x, r1 2.114222x/2.107619x, r12 1.946695x/1.902648x, r10
1.446846x/1.423698x, r3 1.586718x/1.579299x, r6 10.122291x/9.872648x; every
`readers.las` row gains). Remaining warm candidate walls (B0269 suite
medians): r6 0.97 s, r2 0.84 s, r4 0.78 s, r11 0.68 s, r8 0.63 s, r3 0.41 s,
r10 0.28 s, r12 0.26 s, r5 0.25 s.

B0270 decodes COPC tiles in parallel under `requests=1` with pinned
ordering (r5 1.830419x/1.943686x, from parity). Remaining warm candidate
walls (B0270 warm repeat medians): r6 0.98 s, r2 0.86 s, r4 0.81 s, r11
0.68 s, r8 0.62 s, r3 0.41 s, r10 0.28 s, r12 0.26 s, r1 0.23 s.

B0271 widens `--fast` (tie order; fast aggregates 2.711573x/2.712707x and
4.923939x/4.927839x, recorded separately). B0272 retires the automatic r4
CUDA outlier selector (r4 7.422330x/7.440248x on the exact host path).
Remaining warm candidate walls (B0272 warm suite medians): r6 0.97 s, r2
0.84 s, r11 0.68 s, r8 0.63 s, r4 0.62 s, r3 0.43 s, r10 0.28 s, r12 0.27 s,
r1 0.23 s.

B0273 accumulates `writers.gdal` rasters on row bands (r3
2.077188x/2.037489x from 1.59x; r7 unchanged). Remaining warm candidate
walls (B0273 warm suite medians): r6 0.95 s, r2 0.82 s, r11 0.67 s, r4
0.63 s, r8 0.62 s, r3 0.32 s, r10 0.28 s, r12 0.25 s, r1 0.23 s.

B0274 builds every KD2/KD3 nanoflann tree concurrently (r6
10.689432x/10.523255x, r11 12.892762x/11.924286x, r4 8.377378x/8.499474x;
r2 and r3 also gain). Remaining warm candidate walls (B0274 warm suite
medians): r6 0.84 s, r2 0.72 s, r8 0.61 s, r11 0.56 s, r4 0.56 s, r3 0.28 s,
r10 0.26 s, r12 0.25 s, r1 0.24 s.

Current aggregate results (B0274):

- warm equal-workload geometric mean: **2.900213x**;
- warm total-wall speedup: **5.404502x** (27.862333 / 5.155393 s);
- cold equal-workload geometric mean: **2.888163x**;
- cold total-wall speedup: **5.360525x** (27.823962 / 5.190530 s).

Superseded aggregate results (B0273):

- warm equal-workload geometric mean: **2.772537x**;
- warm total-wall speedup: **5.002094x** (28.129230 / 5.623491 s);
- cold equal-workload geometric mean: **2.727890x**;
- cold total-wall speedup: **4.851449x** (28.104653 / 5.793043 s).

Superseded aggregate results (B0272):

- warm equal-workload geometric mean: **2.700250x**;
- warm total-wall speedup: **4.853772x** (28.290238 / 5.828505 s);
- cold equal-workload geometric mean: **2.632576x**;
- cold total-wall speedup: **4.644633x** (28.747381 / 6.189377 s).

Superseded aggregate results (B0270; first pair was the claim, warm repeat
in parentheses):

- warm equal-workload geometric mean: **2.613780x** (2.674064x);
- warm total-wall speedup: **4.559263x** (28.911513 / 6.341269 s; repeat
  4.727176x);
- cold equal-workload geometric mean: **2.624146x**;
- cold total-wall speedup: **4.574280x** (28.742968 / 6.283605 s).

The large current wins are r1, r2, r3, r4, r6, r8, r11, r13, and r14. No aggregate may
omit or downweight a loss; a missing, failed, malformed, or inexact report
makes the suite incomplete. The three-pair aggregate's r11 rows are 4.297126x
warm and 4.279659x cold; B0258's independent nine-pair final-binary warm/cold
qualifications are the r11 selection evidence.

The fresh aggregate uses one final launcher hash and one pinned-oracle hash for
all fourteen rows; B0251 separately records the selected shared-library hash,
and aggregation fails closed on executable or r9-cardinality drift. Warm/cold
aggregate reports hash to
`a6df24912563e2720b61511f74f85b74afe2ff210a054a9e23a27c34f90bbd68`
and `62fb48d21f24da656d52d982bc88d1bf6c32b7bfbc265c0fb7d6d4e91bab1da7`.
It also uses the harness-owned clock namespace so thin routes are
actually exercised. B0247's corrected substantive r9 row supersedes B0243's empty-r9
aggregate. B0247's r9,
B0246's r12, B0245's r13, and B0243's r8 prototype reports are stronger
rejection checks; they are not mixed with suite rounds. B0244's aggregate
describes its removed pre-correction prototype and is historical only.

Complete-process profile payload SHA-256 values (hash of the sorted per-file
hash manifest) are:

- r7 `73d9fdf221a19e819f8e9ce1012250a26d71020d395ffbae676b117423f1b3ce`;
- r7 after B0248 direct dispatch: human-readable child profile report
  `550fc97aed7c4c69a9aaef2643ebf26a79475431e1caf6487193e348263a942f`;
- r7 after B0254 bounded four-worker selection: final public profile payload
  `fd575c7a6cdb452edbacd5d23a3ca5aad5f63b77863cad3e9df5821dca970b12`;
  its descendant timer-reset warning permits limiter attribution only;
- r8 before B0242
  `dd5ed76ad621e29d97d80b1bd87b479e3456dcaa97497b5ca479427a71d671dd`;
- r8 rejected direct prototype after B0242
  `e6bb518b6aa5bd25fdee88abe6b501ff899eb78d4efa669254dc1e1a81023cb9`;
- r8 retained engine path after B0243
  `43b09de62e373631688b99c59b313cc81688a5ab9192686efd138041bad02e42`;
- r9 pre-correction empty workload (historical)
  `1151b194a02b08ce32695ba56ec5d8018ddd2ba5b80f1112f836b116e0050ac1`;
- r9 corrected retained candidate
  `58ac1db7d52db0191c3fabc1241040dea35d4d69b09a8f8b536d4ec4a8220bc3`;
- r9 corrected pinned oracle
  `fd980bde1a47a6143919987c4353ac1b696a5d950385aea29a9bd76cd446374a`;
- r10 `1c3e5f2fd5d9e6f140cfbafeb44bc2f66de9163e13d498240192dedae9f07a01`;
- r10 after B0249 direct dispatch: human-readable child profile report
  `c5645b7ab5b5e1b1fc1621804dc1b29e1a74fef30b742cdce6c8fd7929515ffe`;
- r11 before B0241
  `88622496099877cc03a501e6be3dcea963f88302ef2db38e4e6bc3224cf2b942`;
- r11 after B0241
  `36b5c918cd4e90538ddeef92dbe783bf7781df435c81c9492a3ff7e068ffe26b`;
- r11 after B0255 unsafe first host-index reuse (historical)
  `92ae565ad4a06f3f15e4ce7fe2c2f33a7ecae990dcae09297b005bb461dd789e`;
- r11 after B0256 corrected fresh host-index reuse
  `3ea0b3b34d09b124d03da70cb85b8520e8b5893d2666dad85c846f4e42e94254`;
- r11 B0257 in-process neighborhood attribution report (harness, not a
  sampled profile; `gprofng` totals on this machine are undersampled 7--10x)
  `d07a8910e2167b5d9158c44998f555b915507d1657b5bf73a46eb5ec3d1a70b4`;
- r11 B0258 final warm/cold host-worker reports
  `a248c80f31c497fcc68dc7b16bc2512655c08df3bd71896720087aa48730788d` and
  `c209461448a19c0f2bd92033db3657b94e2041fd0049eea644be7e792b7bd42c`;
- r12 `c9d2702460bc20d7d7c601285f225d2dcca95ea8d0a304703bd7602841aabaaa`;
- r13 before B0244
  `151576d0de3745a2cb88566305b14bd7fbb8af705e15a3026be472c402fc8730`;
- r13 rejected prototype after B0244
  `870441a80afc549619a16b23bd44f747d9e9d20a54347081de652429c7f6a65d`;
- r14 `aa72843127b28b3942268a22e7af2a1a484cdd7667817701cb5eb30ee83ce908`.
- r14 after B0250 all-direction matrix and reader-worker rejection: public
  profile payload
  `76403e4a678bf4d7f7c96e74230e69f691b05ceb848172ee38f669a76fa3413b`;
  its descendant timer-reset warning forbids a function-level CPU share.
- r14 after B0251 selected two-worker compression: final public profile
  payload `d0ef22606d24d46f8427978edc3d09f96e9abb73378161303579174c2193adad`;
  the same descendant timer warning is retained and disclosed.
- r14 after B0252's fail-closed launcher correction: corrected-final public
  profile payload
  `042a4f0546e4e436e609963f6df5911b7671c24513f957e01c19688330f8d23c`;
  the same descendant timer warning is retained and disclosed.
