# Functional and GPU-native stage coverage

GPUPDAL is the PDAL fork itself, not a separate reduced-functionality point
processor. The `gpupdal` executable follows a three-level execution policy:

1. use a specialized exact CUDA or mapped-host path when its full pipeline is
   proven compatible;
2. otherwise replace compatible point-local regions inside the original PDAL
   pipeline and execute every surrounding stage through the same
   `PipelineManager`;
3. otherwise execute the sibling pinned `pdal` binary with the original argv.

This policy separates four questions that must not be conflated:

- **Functionally supported:** can a valid configured PDAL command or pipeline
  run through an exact CUDA path or exact upstream/host fallback?
- **GPU-native:** does the documented work actually execute on the device?
- **Performance-qualified:** does an accepted same-machine complete-process
  measurement establish the end-to-end value of that exact envelope? A
  negative result still qualifies the decision to stay on the host.
- **Automatically selected:** does the option-free public planner select that
  envelope without a force, require, experimental, or `gpupdal resident` request?

These categories are independent. The answer to the first is yes for every
driver and application in the configured fork. A stage may be GPU-native but
unmeasured, performance-qualified on a forced fixture but host-selected, or
automatically selected only as part of a measured fused region. Host fallback
is never called GPU-native.

Host fallback is a valid final performance backend when it is exact and wins
end to end. D0208 supersedes D0019's catalog-wide CUDA and zero-fallback
completion criteria: full functional compatibility remains catalog-wide, but
CUDA-native coverage is reported separately and never earns credit merely for
existing. An exact-but-slower native lane stays host-selected. The §7 catalog
still records functional/native breadth; measured reference wall chooses
optimization work.

## Configured catalog snapshot

On the 2026-08-21 controlled Debian 12 CUDA release build, `gpupdal --drivers`
is byte-identical to the sibling PDAL executable and reports 124 drivers:

| Driver family | Configured | Functionally supported | GPU-native | Performance-qualified | Automatically selected |
| --- | ---: | ---: | --- | --- | --- |
| Filters | 84 | 84 | See the option-envelope audit below | See the audit below | 23 filters |
| Readers | 25 | 25 | 1 direct LAS format family | Direct LAS only inside the accepted fused pipeline envelope | Only as an endpoint of an automatically selected fused region |
| Writers | 15 | 15 | 1 direct LAS format family | Direct LAS only inside the accepted fused pipeline envelope | Only as an endpoint of an automatically selected fused region |
| Total | 124 | 124 | Never inferred from functional support | Never inferred from a kernel timing | Never inferred from device code existing |

Optional plugins disabled at CMake configuration time are not counted in the
124. They are also absent from the sibling PDAL build. The full optional-plugin
dependency matrix remains an F0/P5 release task.

## Implemented-stage audit at D0099

The table is intentionally envelope-specific. “Diagnostic only” means a
complete process may have been timed, but the run is dirty, one-shot, lacks the
required profile/record, or is otherwise not accepted as a performance
qualification. “Unmeasured” means no end-to-end measurement is recorded; it
does not estimate whether the path is fast or slow. The twenty-three automatically
selected filters are the five point-program filters plus Morton ordering,
expression statistics, LOF inside B0043's exact measured resident
LOF/default-LAS pipeline, nndistance inside B0045/B0050's exact measured
resident nndistance/default-LAS pipeline, and normal/eigenvalues/
covariancefeatures only inside B0075's exact measured composition, plus
estimaterank/optimalneighborhood only inside B0080's exact measured
composition, plus outlier only inside B0092's exact measured composition with
NNDistance, plus radiusassign only inside B0093's exact direct-LAS envelope,
plus approximatecoplanar only inside B0097's exact direct-output composition,
and radialdensity only inside B0127's exact measured same-radius composition
with outlier, plus neighborclassifier only inside B0231's exact direct
Classification composition, and sort only inside B0232's exact direct
permutation-publisher composition, plus label_duplicates only inside B0233's
exact measured label/NNDistance/assignment hybrid composition, and
skewnessbalancing only inside B0234's exact direct permutation-publisher
composition, plus HAG-NN only inside B0235's exact count-one direct-output
composition.
Other planner-resident force/experimental execution
and the production-default-off P1.5 placement model do not add to that count.

## Reachability: what a user following this table actually gets (B0176/B0177)

A fifth property sits behind the four columns and is not implied by any of
them: whether an envelope can be *reached* by reconstructing it from its
recorded description. B0176 and B0177 measured seven automatic envelopes on the
1,000,000-point fixture through the public `gpupdal pipeline`:

| Route | Measured | Recorded | Reachable |
| --- | ---: | --- | :-: |
| `assign` + `ferry` (point program) | 7.492x | fused point program | yes |
| `lof(minpts=10)` + assign | 7.901x | 9.597x at 4M | yes |
| `approximatecoplanar(knn=8)` + ferry | 4.075x | 4.231x at 1M | yes |
| `nndistance(kth,k=10)` + assign | 0.997x | 5.229x at 4M | no — selection |
| `estimaterank` + `optimalneighborhood` | **7.821x** | 8.157x | yes — requires `estimaterank.knn=14` (B0184) |
| eigen family, three consumers | **10.134x** | 10.257x | yes — requires `always_up=false` (B0183) |
| `radiusassign` (direct) | **2.728x** | 5.503x | yes — requires `is3d=true`, `update_expression` (B0184) |
| `outlier` + `nndistance` | **23.785x** | 21.390x | yes (B0185) |
| radius `outlier` + `radialdensity` | **17.846x** | 15.897x | yes — `assign.value` must be a string, not an array (B0185) |
| `mortonorder` | 1.029x | listed automatic | no — missing model |

Reachable does not mean the others are wrong: they were qualified on exact
shapes whose full detail may not survive prose. Under D0208, where the
criterion is measured speed on real pipelines, **qualified but unreachable is
its own defect class** and should be recorded as one rather than counted as
coverage.

**B0183/B0184 substantially revise this section, and the revision is a
correction to method, not to the engine.** The seven B0177 rows were
reconstructed by hand-transcribing predicates out of `RuntimePlacement.cpp`.
`test/data/pdg/placement-calibration-sm89.json` stores under `pipelines` the
*exact pipeline every model was calibrated on*, and reading it flips four
verdicts: the eigen family, rank-optimal, radiusassign, and outlier+nndistance
are all reachable and fast. **Reconstruct envelopes from the calibration file,
never from the matcher source.**

**B0185 closes this out: all six compose envelopes reach**, byte-exact, at
10.021x, 7.821x, 4.153x, 2.728x, 23.785x and 17.846x. No envelope in this
family was ever unreachable. B0184's two remaining "refusals" were both errors
of measurement:

- `runResidentPipelineImpl` takes an `automaticAdmission` parameter that is
  **false** for an explicit `gpupdal resident` invocation and **true** when routed
  from `gpupdal pipeline`, and two envelopes are gated behind facts set only in the
  latter. Reading `--stats` from `gpupdal resident` to explain a `gpupdal pipeline`
  timing disables the route being measured. **Never mix the two.**
- Radius-outlier + radialdensity failed only because its `assign.value` was
  written as a one-element array; the envelope tests `is_string()`.

So the entire reachability defect reduces to this: envelopes are matched on
**exact JSON document text** — key counts, scalar-versus-array shape, and
options whose defaults are the opposite of what is required. Copy the
calibrated pipeline out of the calibration file verbatim; do not retype it.

`--stats` now carries an unconditional `plan` section with the planner-derived
attributes the matchers test, so a future mismatch can be read off directly
instead of inferred (B0185).

## Reading this table: coverage work versus fast-path work (D0204, D0208)

D0208 restates the goal: speed on the reference pipelines in
`bench/pipelines/reference/`, not catalog-wide CUDA coverage. A row in this
table matters to the extent its stage appears in one of those workflows and
costs time there. Catalog-wide *functional* coverage and byte-exact output are
unchanged; catalog-wide *CUDA* coverage is no longer a release criterion, and
the remaining native count is no longer tracked as a completion metric.

The four columns are independent and none implies another. D0204 adds one more
distinction that this table must make obvious, because conflating the two has
been the main way these records get misread:

- **On the measured fast path.** The route is GPU-native, positively
  performance-qualified, and either automatically selected or a named
  automatic-selection candidate. These entries exist because they make real
  pipelines faster.
- **Catalog-coverage obligation.** The route is GPU-native and exact, but is
  measured *slower* than stock or has no positive end-to-end qualification, so
  it stays host-selected. These entries exist because D0019 requires
  catalog-wide native coverage, not because they are fast. `filters.pmf`
  (0.583x), `filters.csf` (0.289x), `filters.elm`
  (0.866x), `filters.transformation` (0.357x--1.008x), `filters.iqr` (0.951x),
  `filters.mad` (0.949x), and `filters.colorinterp` (0.345x--0.873x) are
  coverage entries. A negative qualification is still a qualification: it is
  recorded honestly in the Performance-qualified column and must never be read
  as a speedup or used to argue for automatic selection.

A "Yes" in Performance-qualified therefore means *measured*, not *faster* —
always read the cited direction. D0204's admission rule follows from this: a
new standalone per-stage kernel is scheduled only when a same-machine profile
shows that stage dominating a real end-to-end pipeline.

| Implemented stage or exact envelope | Functionally supported | GPU-native | Performance-qualified | Automatically selected |
| --- | --- | --- | --- | --- |
| Pure `readers.las` / `writers.las` format 0–3 and 6–8 translation | Yes | Device translation exists, but the option-free pure-translation path is mapped host | Yes for the mapped-host endpoints in B0001–B0002; no standalone CUDA qualification | No |
| Direct LAS endpoints inside an eligible compute-heavy fused point-program region | Yes | Yes | Yes, B0005: 18.835x for the accepted fused envelope | Yes, only when the complete compute/work/format selector envelope matches |
| Direct LAS publication after one resident output region | Yes in the bounded default LAS/order/cardinality-preserving envelope | The resident filters are GPU-native; canonical translation and final record overlays are host work | Yes for the named 4M `LOF -> assign` envelope: B0238 supersedes B0041--B0043 with an 18.301x final proof and 18.551x option-free confirmation. B0050 updates the named automatic `nndistance(k=10) -> assign` endpoint from B0045's 5.229x to 12.601x pinned PDAL. B0085 qualifies the exact explicit 1M direct-radiusassign endpoint at 5.588x, and B0093 qualifies its option-free automatic route at 5.366x. B0092 qualifies the automatic 1M outlier/NNDistance composition at 20.301x. B0096 qualifies the exact approximate-coplanar/ferry route from 250K through 16M at 2.477x--5.049x (4.248x at 1M), and B0097 qualifies its option-free automatic 1M command at 4.231x. B0124 qualifies the named explicit standalone 1M statistical-outlier/direct-Classification route at 10.740x; B0125 qualifies the named explicit standalone 1M radius-outlier/direct-Classification route at 8.107x; B0126 qualifies the explicit 1M same-radius outlier/radialdensity/UserData composition at 15.897x pinned PDAL and 1.757x current hybrid; B0127 qualifies its automatic 250K--4M envelope at 4.718x--45.178x pinned PDAL. B0044's normal result remains a positive one-shot diagnostic only | Yes only in B0043's unchanged LOF/default-LAS envelope, B0045/B0050's nndistance/default-LAS, B0092's 50K--16M exact outlier/NNDistance default-LAS shape, B0093's 250K--16M exact direct-radiusassign shape, B0097's 250K--16M exact approximate-coplanar/ferry shape, and B0127's 250K--4M exact same-radius outlier/radialdensity shape when their measured placement, input/output, device/profile, and preflight gates accept; B0124/B0125/B0126's other direct-publication shapes remain unselected |
| `filters.assign`, `filters.ferry`, `filters.expression`, `filters.range`, `filters.crop` | Yes | Yes, in their exact fused point-program envelopes | Yes as measured fused regions (B0005 and B0012–B0014), not as arbitrary standalone GPU launches | Yes, only in the measured fused/direct region classes |
| `filters.decimation`, `filters.head`, `filters.tail` | Yes | Yes for the exact bounded standard/streaming ordinal program envelopes; B0102 physically proves global sequence and chunk-boundary behavior | No standalone ordinal qualification; B0102's exact 1M/4M `decimation(step=2) -> assign` rows are negative at 0.450x/0.763x. B0119 separately qualifies the named exact 2M forced-hybrid `mortonorder -> head(100)` composition at 1.395x; its reverted selective-permutation prototype improves only 4.327% over the clean composition | No; retained force-only, and B0119 adds no composition selector |
| `filters.locate` | Yes | Yes for the exact typed first-tie reduction envelope; B0103 physically proves ties, sentinels, NaN/infinity handling, coordinate decode, global indices, and chunk merging | No; B0103's exact 1M/4M `assign -> locate(Z,min)` rows are negative one-shot diagnostics at 0.388x/0.706x pinned PDAL | No; retained force-only |
| `filters.transformation` | Yes | Yes for the exact affine binary64 XYZ envelope | Yes, negative standalone gate: 0.357x at 250K, 0.851x at 4M, and a one-shot 1.008x at 21.97M. B0115's exact affine-plus-assignment fused-region direction is also negative at 0.583x and adds no qualification | No |
| `filters.iqr` | Yes | Yes in its exact bounded selection envelope | Yes, negative standalone gate through 21.97M (0.951x) | No |
| `filters.mad` | Yes | Yes in its exact bounded selection envelope | Yes, negative standalone gate through 21.97M (0.949x) | No |
| `filters.sort` | Yes | Yes in its stable/tie-free exact envelope; B0131 additionally composes the strict mapped LAS source and exact full-record permutation publisher | Yes for B0130's named forced-CUDA comparator-unique 1M Z fixture at 1.398097x pinned PDAL and B0232's strict direct/public form at 3.242887x/3.267173x pinned PDAL; the older 21.97M row is dirty diagnostic evidence | Yes only for B0232's exact 600K--16M uncompressed format-7/36-byte `sort(Z,ASC,NORMAL)` mapped-source/permutation-publisher envelope; every other form remains host-selected |
| `filters.mortonorder` | Yes | Yes | Yes, B0006: 1.248x–1.528x ordinary and 1.487x reverse in the accepted envelope; B0118 separately qualifies the exact 2M forced-hybrid ordinary-Morton-plus-assignment graph at 1.253x | Yes from 2M finite, nondegenerate points for Morton itself; B0118 adds no automatic composition selector |
| `filters.groupby` | Yes | Yes in its exact terminal partition envelope | No; diagnostic complete-process result is 0.913x and is not an accepted release benchmark/profile | No |
| `filters.returns` | Yes | Yes in its exact return-partition envelope | No; diagnostic `returns` + `merge` result is 0.982x | No |
| `filters.merge` | Yes | No; it is an optimized exact host view-composition bridge | No; only the diagnostic 0.982x `returns` + `merge` composition is recorded | No |
| `filters.divider` | Yes | Yes in its exact count-partition envelope | No; diagnostic complete-process result is 0.952x | No |
| `filters.splitter` | Yes | Yes for its exact nonpositive-buffer primary-cell envelope; positive-buffer work stays host | No; diagnostic complete-process result is 0.932x | No |
| `filters.colorinterp` | Yes | Yes for finite binary64 exact map envelopes | Yes, negative standalone gate: 0.345x–0.873x | No |
| `filters.stats` | Yes | Physically observed only for B0120's explicit finite-basic six-dimension route; broader runtime/sanitizer qualification remains pending | No; B0120's exact current-clean 2M result is a negative 0.429x diagnostic | No; option-free execution remains pinned-host selected |
| `filters.info` | Yes | Yes for the option-free exact bounds/count envelope; point/query forms retain the exact host path | No; B0104's current exact 1M/4M diagnostics are 0.526x/0.745x pinned PDAL, and record-summary reuse has only a 1.040% measured ceiling | No; retained force-only |
| `filters.expressionstats` | Yes | Yes | Yes, B0008: 1.105x–1.212x at its work-aware thresholds | Yes, at the measured one/two/three-expression thresholds |
| `filters.outlier` | Yes | Yes for the bounded radius and statistical envelopes; B0068 preserves its statistical Classification finale through an exact direct-LAS resident composition. B0087 reuses one planner-owned max-k rowset with an adjacent NNDistance projection; B0090 repairs at most 16 incomplete bounded mean rows in parallel from resident coordinates. B0124 composes the strict mapped source and direct Classification publisher with the standalone statistical shape; incomplete rows retain exact host repair. B0125 does the same for the bounded standalone radius shape, while retaining the exact host count comparison and all-outlier finale. B0126 shares one planner-owned radius index and direct source/publication lifetime with an adjacent radialdensity consumer | Yes in the exact statistical-outlier -> NNDistance direct composition: B0088 qualifies explicit 1M at 21.389523x, B0091 qualifies improved explicit 4M at 48.758796x, and B0092 qualifies automatic 1M at 20.300698x. B0124 separately qualifies only the named explicit standalone 1M statistical forced-hybrid/direct routes at 6.521497x/10.739669x; B0125 qualifies only the named explicit standalone 1M radius forced-hybrid/direct routes at 4.946130x/8.107334x. B0126 qualifies only the named explicit same-radius forced-hybrid/direct composition at 9.156744x/15.896697x; B0127 qualifies only its exact automatic 250K--4M form at 4.717949x--45.178325x. B0227 separately qualifies only the full-input-fingerprinted and device-calibrated 1M r4 statistical-outlier route at 3.686747x complete-process median. Broader standalone and composition shapes remain unqualified | Yes inside B0092's exact measured `mean_k=8,multiplier=2,class=7 -> nndistance(kth,k=10)` default-LAS composition from 50K through 16M, B0127's exact `radius=1.01,min_k=2,class=7 -> radialdensity(radius=1.01) -> UserData` default-LAS composition from 250K through 4M on the pinned SM89 profile, and B0227's literal r4 `mean_k=8,multiplier=2` grammar at the measured 1M input summary and full-file fingerprint on RTX 4090/SM89/CUDA 13.3/driver 610.43.03; B0124/B0125's standalone shapes remain unselected |
| `filters.radialdensity` | Yes | Yes for finite positive-radius whole-view/tiled envelopes; B0101 keeps the exact binary64 result device-resident for its adjacent assignment consumer, and B0126/B0127 share one planner-owned radius index/source/publication lifetime with a preceding radius outlier | Yes for B0100's explicit exact 1M format-7 `radius=1.01` forced-CUDA pipeline at 4.720x pinned PDAL, for B0101's explicit exact 1M VLR-free format-7 `radius=1.01 -> UserData` resident composition at 8.966x pinned PDAL/1.770x the same-binary hybrid control, and for B0126/B0127's exact same-radius outlier/radialdensity/UserData composition at 15.897x explicit 1M and 4.718x--45.178x automatic 250K--4M; the older 2.104x–20.228x rows remain diagnostic | Yes only inside B0127's exact measured 250K--4M format-7/36-byte same-radius composition on the pinned SM89 profile; B0101 and broader radialdensity shapes remain unselected |
| `filters.approximatecoplanar` | Yes | Yes in the bounded shared-eigensystem envelope; tied/incomplete rows repair on host | Yes in resident composition through B0017's 3.962x `ferry` envelope. B0096 current-binary pairs performance-qualify only the exact direct-output `knn=8 -> Coplanar=>UserData` composition from 250K through 16M at 2.477x--5.049x (4.248x at 1M), and B0097 qualifies its option-free automatic 1M command at 4.231x. B0070 rejects sparse repair transfer at only 0.49% shell-wall improvement, while B0095/B0097 close further endpoint tuning below the 5--10% gate | Yes only inside B0097's exact measured 250K--16M format-7/default-threshold/direct-output composition after placement, preflight, and one-index proofs; the qualification gate additionally requires positive repair on its named fixture. The older standalone selector remains compiled off and broader shapes fall back |
| `filters.lof` | Yes | Yes in the bounded shared-kNN envelope; ambiguous/incomplete closures repair exactly on host. B0238 keeps the single pinned KD3 index but gives large repairs a contiguous immutable XYZ backing and deterministic parallel passes | Yes in resident composition, B0017: 8.459x with `assign`; B0238 supersedes B0043's 9.597x result for the same exact option-free 4M public pipeline with an 18.301x final proof and 18.551x option-free confirmation | Yes only in B0043's existing exact explicit `minpts=10`/`UserData`-assign/default-LAS shape after runtime placement/preflight acceptance; B0238 changes repair performance, not selection |
| `filters.normal` | Yes | Yes in the bounded shared-eigensystem envelope | Yes in resident composition, B0018: 4.037x with `assign`; B0075's exact 1M three-consumer public pipeline is 10.257x | Yes only inside B0075's exact measured normal/eigenvalues/covariancefeatures/three-assignment/default-LAS shape |
| `filters.eigenvalues` | Yes | Yes in the bounded shared-eigensystem envelope | Yes only as part of B0075's exact 1M three-consumer public pipeline at 10.257x; no standalone qualification | Yes only inside B0075's exact measured composition |
| `filters.covariancefeatures` | Yes | Partial: algebraic features execute on device; `Omnivariance` and `Eigenentropy` retain the exact host transcendental bridge | Yes only as part of B0075's exact 1M raw/dimensionality three-consumer public pipeline at 10.257x; no standalone qualification | Yes only inside B0075's exact measured composition |
| `filters.nndistance` | Yes | Yes in the bounded shared-kNN envelope; at most 16 incomplete kth rows for public `k=1..15` repair on device with exact host fallback outside the bound. B0091 makes the exact partial-select/ordered-merge repair the default and proves it independently from an adjacent outlier repair. B0110 explicitly composes `kth,k=10` with the strict one-binary64 direct publisher; B0111 additionally reuses the mapped LAS source in that same explicit envelope. B0116 measures an exact forced per-stage hybrid chain after duplicate labeling; B0117's reverted prototype separately proves actual one-region execution | Yes for the exact 4M `k=10`/`UserData`-assign/default-LAS pipeline (B0050: 12.601x), B0091's explicit 4M outlier composition (48.758796x), B0092's automatic 1M composition (20.300698x), and B0233's exact 250K--16M duplicate-label hybrid composition (2.846x public at the floor and 6.392x at 1M). B0117's resident direction remains slower than the hybrid and adds no qualification | Yes in B0045/B0050's exact `k=10`/`UserData`-assign/default-LAS shape, B0092's exact 50K--16M outlier composition, and B0233's literal 250K--16M label/NNDistance/assignment hybrid on the pinned SM89 profile; every neighboring B0233 graph/layout remains host-selected |
| `filters.estimaterank` | Yes | Partial: neighborhood/covariance work is device-resident; the exact Jacobi SVD finale is host | Yes only as part of B0080's exact 1M public rank/optimal composition at 8.157x; no standalone qualification | Yes only inside B0080's exact measured composition |
| `filters.optimalneighborhood` | Yes | Partial: shared covariance work is device-resident; the exact entropy/log finale is host | Yes only as part of B0080's exact 1M public rank/optimal composition at 8.157x; no standalone qualification | Yes only inside B0080's exact measured composition |
| `filters.neighborclassifier` | Yes | Partial: the bounded shared-neighborhood query and vote execute on device; conservative boundary-tie rows still use the exact pinned KD3 repair. B0099 attributes 1,947 ties/zero incomplete rows and rejects a no-value invariant-vote prototype, so this host ownership is intentional | Yes for B0231's exact direct mapped-LAS source/Classification publisher over the measured 250K--16M uncompressed format-7/36-byte `k=7` envelope. The final 1M public route is 4.033570x pinned PDAL; 50K is a measured 0.685x loss and remains host-selected. B0098's 3.960x explicit 1M row and the older 22M row remain historical | Yes, only for B0231's strict 250K--16M direct-boundary envelope on the exact qualified SM89 profile; all neighboring shapes fail closed |
| `filters.radiusassign` | Yes | Partial: shared radius selection is CUDA; ordered assignment-expression evaluation is host. B0082 can additionally hydrate mapped LAS coordinates and ReturnNumber directly and publish UserData without a full PointView endpoint | Yes for B0085's exact explicit 1M direct-LAS envelope at 5.588x and B0093's option-free automatic route at 5.366x pinned PDAL; B0019's ordinary lane remains dirty diagnostic evidence and broader shapes are unqualified | Yes only for B0093's exact `radius=2,is3d=true,ReturnNumber[1:1] -> ReturnNumber[2:15],UserData=9` direct-LAS envelope from 250K through 16M on the pinned SM89 profile |
| `filters.label_duplicates` | Yes | Yes in the exact adjacent-predecessor envelope. B0116 proves the standalone CUDA wrapper inside a forced per-stage hybrid chain, including exact host-column restoration; B0117's reverted prototype proves actual one-region execution | Yes for B0233's exact 250K--16M forced/public `Classification` duplicate-label -> `nndistance(k=10)` -> `UserData=Duplicate` chain at 3.261x--12.190x forced and 2.846x/6.392x public at 250K/1M. The standalone lane remains negative, and the current resident probe remains 4.3% slower than the hybrid | Yes only inside B0233's literal five-stage, uncompressed format-7/36-byte, zero-VLR/EVLR, 250K--16M hybrid composition on the exact SM89 profile; standalone, resident, and neighboring forms remain host-selected |
| `filters.smrf` | Yes through the exact KD2Index host wrapper or pinned fallback | No qualified device lane; the compiled bounded canvas uses a known-inexact eighth-neighbor tie rule and fails closed | No; B0021's 0.599x result predates discovery of the device fill defect and is not a valid qualification | No |
| `filters.pmf` | Yes | Yes in the bounded and required planner-resident envelopes | No; dirty-snapshot B0022 (0.583x) and B0031 (1.039838x) are complete-process diagnostics; larger primitive figures are not pipeline claims | No |
| `filters.csf` | Yes | Yes in the bounded serial-oracle, `smooth=false` envelope | No; dirty-snapshot B0023 is a negative complete-process diagnostic (0.289x) | No |
| `filters.elm` | Yes | Yes in the bounded stable-cell envelope | No; dirty-snapshot B0024 is a negative complete-process diagnostic (0.866x) | No |
| `filters.skewnessbalancing` | Yes | Partial: CUDA produces the exact unique-Z order; the pinned recurrence and Classification update remain host. B0129 can reuse a strict mapped LAS source and publish the resulting full-record permutation without a PointView writer round trip | Yes for B0128's exact forced comparator-unique 1M format-7 fixture at 1.192843x, B0129's strict explicit direct form at 2.887507x, and B0234's exact 450K--16M automatic direct envelope at 1.200177x public at the floor and 3.091007x at 1M; B0025 is historical dirty evidence | Yes only inside B0234's literal 450K--16M uncompressed format-7/36-byte direct composition on the exact SM89 profile; ordinary and neighboring forms remain host-selected |
| `filters.hag_nn`, `count=1..64` | Yes | Yes inside the proved masked-2D-index envelopes; rejected rows repair on host before publication. The strict mapped source and one-binary64 publisher are available explicitly for counts one through 64, the shared index's own `maximumNeighbors` cap. B0239 additionally repairs only rejected rows through one planner-owned compatibility ground KD2 product inside the fingerprinted r2 hybrid | Yes for B0105's named count-four forced fixture, B0121/B0122/B0123's named count-two/count-three/count-one forced-hybrid/direct routes, B0135/B0138/B0141's named count-five/six/seven routes, and B0144's representative counts 16 and 64. B0235 qualifies the exact count-one direct composition from 450K through 16,000,002 points; corrected B0239 qualifies the complete fingerprinted 1M r2 reference at 1.270989x with host SMRF and named LAZ publication | Yes for B0235's literal count-one mapped-LAS/direct-double-output envelope and separately for B0239's exact fingerprinted 1M r2 fixture/grammar/named-layout/SM89 hybrid. Counts 2--64 outside those proofs and all neighboring layouts/options/cardinalities/fixtures/profiles remain explicit or host-selected |
| `filters.hag_nn`, `count>=65` | Yes through unchanged pinned-PDAL fallback | No | **Unmeasured** as PDG acceleration | No |
| `filters.hag_delaunay`, explicit `count=3` | Yes | Yes inside the proved masked-2D-index envelope; rejected rows repair on host before publication. The strict mapped source and one-binary64 publisher remain explicitly available | Yes for B0237's corrected exact 500,001--16,000,002 automatic direct envelope, with final public medians of 1.245213x at 500,001 and 2.049354x at 1,000,002. B0132's 2.204694x explicit 1M row remains the implementation/profile foundation | Yes only for B0237/D0236's literal count-three mapped-LAS/direct-double-output envelope: uncompressed format-7 40-byte input with one unsigned-32 `OffsetTime` descriptor, 48-byte output, `extra_dims=all`, one planner-owned 2D index/region/lane, 184-byte/point peak, successful CUDA proof, and the exact SM89 profile. Default/count-four and every neighboring layout/option/cardinality/profile remain explicit or host-selected |
| `filters.randomize` | Yes | No; this is an audited exact host ordering bridge | Yes only as a bridge inside the B0012–B0014 mixed resident class, not as GPU work | No |
| Every other configured stage/option | Yes through unchanged PDAL | No current device claim | **Unmeasured** as PDG acceleration | No |

The terrain diagnostics cover complete processes rather than inferred kernel
speed, but their dirty snapshots are not performance qualifications under
D0099. SMRF and CSF lose standalone at 0.599x and 0.289x; neither exposes a
reusable planner product today, and residency alone is only a hypothesis until
a clean mixed-pipeline profile shows a material spill/re-upload cost. PMF loses
its bounded standalone lane at 0.583x, while compatible adjacent PMFs reach
only 1.039838x after allocation reuse; semantic surface reuse remains unproved.
B0037 tested the PMF-to-HAG hypothesis on a controlled 1,003,520-point exact
all-native graph. Forced Morton-BVH execution is only 0.846x pinned PDAL, and
the masked HAG gather consumes 3.77 seconds of the 4.69-second candidate while
normal's gather is 72.5 milliseconds. The Grid-to-index boundaries are not the
dominant cost on this fixture, the 2D HAG product is semantically incompatible
with normal's 3D all-point index, and this cross-kind composition is deferred.
The one-shot result is diagnostic, not performance qualification or automatic
selection (D0100/B0037).

## Execution paths

In this older path-oriented table, “qualified” in a path name means physically
validated for exact execution; it does not mean performance-qualified. The
D0099 audit above is authoritative for performance and automatic-selection
status.

| Path | Current coverage | Compatibility behavior | Validation |
| --- | --- | --- | --- |
| Direct CUDA LAS | Uncompressed LAS formats 0–3 and 6–8 through the exact default format-7 writer; ordered `filters.assign`, `filters.ferry`, `filters.expression`, `filters.range`, and single-bounds `filters.crop` regions | Stable point order after filtering, exact header/bounds/return summaries, atomic no-overwrite publication | Generated all-format/chunk matrices, exact process differentials, large all-pass and selective differentials, four Compute Sanitizer tools |
| In-process CUDA region | Contiguous assign/ferry/expression/range/single-bounds-crop regions within otherwise ordinary linear PDAL pipelines | Original reader, host filters, writer, options, and stage order remain intact; only a proven region is replaced | Host/CUDA differentials through LAS/LAZ, BPF, PLY, PCD, and text readers; sorted COPC standard-mode differential |
| In-process host region | The same point-program region when CUDA is unsupported or below its performance gate | Uses the exact host VM and stable host compaction without changing the rest of the PDAL graph | Host reader matrix and ASan/UBSan |
| Qualified forced ordinal CUDA region; host-selected standalone | Exact `filters.decimation`, `filters.head`, and `filters.tail` programs in direct LAS and in-process regions | Global selection state is retained across chunks; standard and streaming semantics are distinct; automatic selection remains disabled | Eight focused physical device/process gates pass across modes, kinds, chunks, and split graphs; memcheck, initcheck, racecheck, and synccheck are clean. B0102's exact 1M/4M directional rows reach only 0.450x/0.763x pinned PDAL and the chunk-amortization control loses, so the lane is GPU-native coverage but not performance-qualified or selected (D0163) |
| Qualified forced global-reduction CUDA region; host-selected standalone | Exact first-tie `filters.locate` minima/maxima in linear, order-proven in-process pipelines | Typed values, NaNs, infinities, PDAL sentinels, empty/invalid-kind views, and chunk-global source indices are preserved; automatic selection remains disabled | The 15-case host matrix remains exact; five focused physical CUDA units/process graphs and all four Compute Sanitizer tools pass. B0103's exact 1M/4M assignment-to-locate diagnostics reach only 0.388x/0.706x pinned PDAL, and the measured direct-boundary hypothesis does not yet establish a 5--10% win. The lane is GPU-native coverage but not performance-qualified or selected (D0164) |
| Qualified forced coordinate-map CUDA region; host-selected standalone | Inline, non-inverted `filters.transformation`; full projective host execution and affine double-XYZ CUDA execution | Original XYZ values feed all three row-major outputs; homogeneous division, operation order, fallback matrix files/inversion/SRS semantics, and coordinate-product invalidation are preserved | Nine-case host/ASan matrix, physical 131,103-point affine property, forced process differentials, and all four Compute Sanitizer tools pass. Clean complete-process CUDA is 0.357x PDAL at 250,000 points, 0.851x at 4,000,000, and only 1.008x in a one-shot 21,970,934-point trial. B0115's current exact affine-plus-assignment hybrid direction is 0.583x; its 32 device kernels total just 0.849310 milliseconds/0.141552% of wall, so device fusion cannot meet the 5--10% gate. Automatic selection remains rejected and the endpoint is sufficiently optimized (D0176) |
| Qualified forced robust-statistics CUDA region; host-selected standalone | Exact `filters.iqr` and `filters.mad` global selections in linear, order-proven pipelines | Upstream percentile indices, upper medians, independent `nth_element` copies, strict fences/deviation division, physical conversion, NaNs, degenerate multipliers, and stable survivors are preserved | 16-case host/ASan matrix, physical 131,103-point property, forced IQR/MAD process differentials, all four Compute Sanitizer tools, and two 21,970,934-point exact local differentials pass. Clean CUDA reaches only 0.951x PDAL for IQR and 0.949x for MAD at full scale, so standalone host selection remains correct |
| Qualified forced ordering CUDA region; bounded automatic direct composition; ordinary boundary host-selected | Exact `filters.sort` typed permutations in linear, order-proven pipelines; the automatic direct form is limited to canonical uncompressed LAS 1.4 format-7/36-byte `sort(Z,ASC,NORMAL)` with finite comparator-unique physical Z and 600K--16M points on the exact SM89 profile | PDAL's normal/stable pass sequence, directions, last-dimension priority, duplicate/NaN/signed-zero behavior, custom keys, and original point-view behavior are preserved. The direct route proves mapped-source identity, one 8-byte/point key upload, one 8-byte/point permutation download, zero indexes/packing, exact nonidentity full-record publication, terminal-spill elision, a measured conservative 64-byte/point peak, and atomic fallback before data-dependent exactness or side-effect-free publication refusal | The existing host/ASan, physical properties, forced differential, and all-four-sanitizer evidence remains. B0232 adds a 50K--16M exact direct ladder, separate conservative model, 550K public refusal, 600K/1M automatic positives, exact/one-byte-below memory gates, duplicate/non-finite and existing/alias/symlink oracle fallback, and memcheck/racecheck. The final public route is 1.842475x PDAL at 600K and 3.267173x at 1M; ordinary sort, other layouts/options/cardinalities/profiles, and comparator ties remain host-selected (D0231) |
| Qualified automatic adjacent-label CUDA hybrid; host-selected standalone and resident prototype | Exact unconditioned `filters.label_duplicates` over the current point order, including an index-free resident producer that can feed later point programs or precede a planner-owned neighborhood index. B0233 automatically composes only the literal `Classification` label -> `nndistance(k=10)` -> `UserData=Duplicate` graph on uncompressed format-7/36-byte input from 250K through 16M on the pinned SM89 profile | Every selected field is converted to binary64 and compared only with the immediate predecessor; the first `Duplicate` byte is untouched, empty dimensions are vacuously equal, and self-referential `Duplicate` input remains sequential host work. B0233 requires actual CUDA use by all three replacement regions and falls back before writer execution on grammar, layout, count, profile, or recoverable-device drift | The earlier 15-case host/10-case CUDA process matrix, physical all-type property, wrapper units, resident composition, and all-four primitive sanitizer evidence remain. B0233 adds the exact 50K--16M hybrid ladder, separate complete-process affine selector, 250K/1M automatic positives, fallback/refusal and existing/alias/symlink differentials, aggregate Host/CUDA suites, and actual-process memcheck/racecheck. Final public results are 2.846286x at 250K and 6.391572x at 1M. The current resident probe is still 4.3% slower and remains rejected (D0082/D0177-D0178/D0232) |
| GPU-native count-one-through-64 HAG-NN; bounded automatic count-one direct composition; wider routes host-selected | `filters.hag_nn` is functionally supported for every option through exact CUDA or pinned host fallback. Whole-view counts one through 64 are GPU-native for finite binary64 XY (and finite Z for wider counts), unsigned-byte Classification, enough ground references, and a proved nearest-ground candidate order. They consume the planner-owned masked 2D kNN index and may retain the HAG column across an adjacent point-program bridge. The strict mapped-source/one-binary64 publisher remains explicitly available for counts one through 64. B0235 automatically selects only the literal count-one graph on uncompressed LAS 1.4 format-7/40-byte input with one exact unsigned-32 `OffsetTime` Extra Bytes descriptor, 48-byte output, `extra_dims=all`, 450K--16,000,002 points, one index/region/lane, and the exact SM89 profile | Ground rows receive positive zero. Count one preserves pinned subtraction and ignored-option behavior; wider counts preserve ordered inverse-squared-distance interpolation, strict `max_distance`, and inclusive no-extrapolation bounds. Candidate ties or bounded-grid incompleteness repair with the original host filter before publication. Nonfinite XY, no ground, nonfinite Z or too few grounds for wider counts, `count >= 65`, unsupported options/layouts, or unavailable CUDA retain pinned behavior. B0235 additionally requires mapped-source identity, exact 25-byte upload/8-byte spill/112-byte index facts, a conservative 160-byte/point peak, executable rewrite, actual device execution, and successful atomic direct publication. A following 3D neighborhood stage rebuilds the index; Grid/non-Grid products use distinct regions and explicit boundaries | The expanded 200-case host/192-case CUDA matrix and lifecycle coverage remain exact. B0105 and B0121--B0144 qualify representative explicit counts one through 64, reaching 2.518270x for count one and 4.374517x for count 64 on their named fixtures. B0235 adds the exact 50,001--16,000,002 direct ladder, separate conservative model, 450K/1,000,002/cap automatic positives, hash-pinned cap-plus-one refusal, exact/one-byte-below memory gates, grammar/layout/source/profile/proof refusals, existing/alias/symlink oracle fallback, aggregate host/CUDA gates, and actual-route memcheck/racecheck. Final public medians are 1.216823x at 450K and 2.461791x at 1,000,002; 400,002's marginal 1.104286x direct result remains below the safer floor. Counts 2--64 and all neighboring forms remain explicit or host-selected (D0094-D0095, D0097-D0099, D0166-D0169, D0175, D0182-D0184, D0195-D0201, D0234) |
| GPU-native data-dependent count-three HAG Delaunay; bounded automatic direct composition; wider routes host-selected | `filters.hag_delaunay` is functionally supported through exact CUDA or pinned fallback; explicit `count=3`, boolean `allow_extrapolation`, unsigned-byte ground class, runtime-compatible finite binary64 XYZ, enough grounds, and a proved three-candidate order are GPU-native. The lane consumes the planner-owned masked 2D kNN index and may retain its exact HAG column across an adjacent point-program bridge. B0237 automatically selects only the literal count-three graph on uncompressed LAS 1.4 format-7/40-byte input with one unsigned-32 `OffsetTime` Extra Bytes descriptor, 48-byte output, `extra_dims=all`, 500,001--16,000,002 points, one index/region/lane, and the exact SM89 profile | The kernel reproduces the three-point Delaunator seed/circumradius/winding order and pinned barycentric arithmetic. Ties and incomplete search use the original exact host repair before publication. Nonfinite/runtime-incompatible coordinates, insufficient grounds, default/count-four-or-wider triangulations, unsupported options/layouts, or unavailable CUDA retain pinned behavior. B0237 additionally proves mapped-source identity, semantic Extra Bytes identity, 25-byte upload/8-byte spill/112-byte index facts, a conservative 184-byte/point peak, executable rewrite, calibrated placement/executor agreement, successful device execution, and atomic direct publication | The 22-case host/15-case CUDA matrix, repeated exactness, arithmetic comparisons, lifecycle, and native/repaired bridges remain. B0237 retains B0236's exact 50,001--16,000,002 direct ladder and conservative model, raises the automatic floor to 500,001 after a fresh 450K row reaches only 1.093356x, and adds semantic-layout/device-decline refusals plus same-route stats/profile proof. The process matrix covers floor/main/cap positives, 500,000/cap-plus-one and one-byte memory refusals, grammar/layout/source/profile/proof and one-query tie/incomplete/no-ground coverage, and publication-collision fallback. The placement audit is 214/218 with 224/224 raw reports verified. Final public medians are 1.245213x at 500,001 and 2.049354x at 1,000,002. Default/count-four and neighboring forms remain explicit or host-selected (D0096-D0099, D0173-D0174, D0235-D0236) |
| Unqualified bounded SMRF CUDA prototype; exact host wrapper retained | Exact `filters.smrf` host execution uses the pinned oracle's own KD2Index for eight-neighbor void filling. The compiled CUDA prototype remains bounded to a 4,096-cell raster and morphology radius 64 but is not selectable as an exact lane | B0214--B0216 proved that the device kernel's integer-distance/index cutoff-tie rule differs from the oracle's KD2Index traversal and can flip Classification. `smrfSupportsExactDevice` therefore fails closed for every input; required resident/CUDA execution errors, while ordinary execution retains the exact host wrapper or pinned fallback | The expanded 18-case host process matrix is the relevant exactness gate. The former seven-case CUDA matrix and B0021's 0.599x row did not exercise the contested tie and no longer qualify a device lane. Re-enable only after the device path uses an oracle-equivalent fill and a committed contested-tie differential passes (D0221) |
| Qualified bounded PMF CUDA plus provisional planner-owned tiled/device-source/phase region and compatible adjacent-PMF allocation reuse; host-selected | The D0084 global lane retains its at-most-4,096-cell envelope. A required planner-resident D0088-D0093 lane also accepts larger finite logical-double XYZ/unsigned-byte Classification frames with finite represented centers and explicit device/pinned-host budgets, morphology radius at most 64, at most 64 schedule passes, supported returns/classes, and `only_ground`. When the complete 16-byte-per-cell proof/phase pair fits, source construction, tie proof, and phase storage are device-resident; larger or under-budget frames retain the exact host-build/tiled path. Contiguous PMF stages with the same Grid/cell contract and exact original JSON `returns` value may share one product/allocation | Upstream's distinct initial/final fractional-cell binning, first-source equal minima, progressive windows, column-major morphology, strict height comparison, return diagnostics, and selected-record preservation are reproduced. The device path chooses the lowest selected source index for equal numeric minima, compacts only original populated cells, and checks every equal-distance source bit pattern before publishing. Distinct-bit ties discard the provisional workspace before device Classification materialization/H2D or mutation; the wrapper then runs pinned upstream on a private copy of the untouched view. Success promotes that same allocation into the two phase backings and records zero raster H2D/D2H. Adjacent PMFs reuse only the allocation and rebuild one exact raster generation per stage. The host fallback retains D0091's literal/occupancy-hierarchy builder. Default and broad experimental execution delegate directly to PDAL; `ignore`, `where`, nonfinite centers/input, budget failures, different return/cell sources, non-PMF Grid consumers, and cross-kind composition retain PDAL or fail closed when CUDA is explicitly required | The 16-case host/nine-case CUDA matrix remains exact. RasterGrid/build units and physical PMF tests cover all seams, exact and one-byte-below budgets, provisional lifetime/promotion/retry, signed-zero/multiway tie bits, heterogeneous large-origin/fractional raw-raster parity, finite-center rejection, bounded/full-device/host-tiled execution, >4,096-cell differentials, determinism, primitive rejection before mutation, exact wrapper fallback, and two-/three-stage lifecycle success, tie-fallback, and exception cleanup. Host Debug and ASan/UBSan each pass 415 plus one skip in 416 registrations; CUDA Release passes 603 plus nine documented skips in 612 registrations. All four focused Compute Sanitizer tools are clean. B0030 records zero raster-transfer bytes and exact 3.156x-38.950x controlled dense/sparse results; final B0031 v4 records one product, a separately proved reused allocation, two raster generations, zero raster transfers, exact output, and 1.039838x for the symmetric-scope controlled PMF pair with comparison outside timing. The earlier 4.590078x, 1.056201x, and comparison-inclusive 1.016953x measurements are superseded. There is no automatic model, worst-case scalability proof, semantic surface reuse, cross-kind Grid/cloth reuse, catalog closure, or P3 exit (D0088-D0093) |
| Qualified bounded CSF CUDA region; host-selected standalone | Exact `filters.csf` global-canvas execution for finite logical-double XYZ, unsigned-byte Classification, at least a 2x2 and at most 4,096-cell cloth, supported returns/classes/`only_ground`, bounded iterations (`0 <= iterations <= 64`), `rigidness >= 0`, and `resolution > 0` | The pinned serial, non-OpenMP source order is reproduced for the coordinate transform, cloth graph, rasterization/fill, in-place constraints, collision, strict interpolation, and diagnostics; no private spatial index is built. Native admission requires `smooth=false` and the serial-oracle build capability. `dir`, `debug`, `ignore`, `where`, smoothing, OpenMP-enabled builds, invalid options, nonfinite coordinates, oversized cloths, and adjacent Grid composition retain unchanged PDAL or the exact host wrapper | A 22-case host/14-case CUDA process matrix, eight focused host tests, 12 focused physical CUDA tests, five repeated CUDA matrices, and all four Compute Sanitizer tools pass. B0023 is byte-exact at 1M but only 0.289x PDAL, so no automatic model is accepted. There is no tiled/scalable cloth, reusable planner Grid product, composition speed claim, or P3 exit (D0085) |
| Qualified bounded ELM CUDA region; host-selected standalone | Exact `filters.elm` whole-view cell execution for finite logical-double XYZ, unsigned-byte Classification, finite positive `cell`, finite threshold, valid output class, at most 4,096 cells, and no `where` | Upstream frame truncation and `floor(coordinate - minimum) / cell` binning, column-major cell keys, numeric-Z order with source-stable equal values and signed zeros, strict `fabs(current-next) < threshold`, classification writes, point order, empty behavior, and diagnostics are reproduced. No private spatial index is built; invalid options, nonfinite/oversized runtime input, `where`, and adjacent Grid composition retain unchanged PDAL or the exact host wrapper | A 15-case host/10-case CUDA process matrix, 11 focused host tests, 17 focused physical CUDA tests, five repeated CUDA matrices, an exact 1M scratch-budget boundary, and all four Compute Sanitizer tools pass. B0024 is byte-exact at 1M but only 0.866x PDAL, so no automatic model is accepted. There is no tiled/scalable raster, reusable planner Grid product, composition speed claim, or P3 exit (D0086) |
| Qualified bounded skewness CUDA-ordering hybrid plus strict automatic direct global-order composition; ordinary boundary host-selected | Exact `filters.skewnessbalancing` ordering substep for a nonempty whole view with physical binary64, finite, comparator-unique logical Z and valid class options. B0234's automatic direct route additionally requires canonical uncompressed LAS 1.4 format-7/36-byte mapped input/output, `extra_dims=all`, the literal option-free three-stage graph, 450K--16M points, and the exact SM89 profile | CUDA returns an exact full-record Z permutation and proves the absence of equal keys; the wrapper preserves the pinned sequential moment/sign-crossing recurrence and Classification writes on the host. The direct route reads Z from the mapped source and publishes the complete permutation canonically. Placement and preflight reserve 65 bytes/point. Ties including signed-zero equivalents, nonfinite or non-binary64 Z, options, graph/layout/profile/budget drift, and side-effect-free publication refusal run unchanged PDAL before commitment | The existing host/CUDA matrix and sanitizer evidence remain. B0234 adds an exact 50K--16M direct ladder, a separate conservative model, a public 400K rejection, 450K/1M/16M automatic positives, exact and one-byte-below memory gates, grammar/layout/source/profile/proof and tie/nonfinite refusals, existing/alias/symlink oracle fallback, aggregate Host/CUDA suites, and actual-route memcheck/racecheck. Final public results are 1.200177x pinned PDAL at 450K and 3.091007x at 1M; ordinary and neighboring skewness forms remain host-selected (D0087/D0189/D0190/D0233) |
| Qualified automatic Morton CUDA region | Exact ordinary and reverse `filters.mortonorder` permutations in linear, order-proven pipelines; finite, nondegenerate views with at least 2,000,000 points select CUDA automatically | The upstream most-significant-bit comparator, reverse grid/interleave/reversal, stable duplicate-code order, distinct empty behavior, and complete point views are preserved; smaller, unsupported, disabled, or device-less runs use host execution | 15-case host/ASan matrix, physical 131,103-point dual-mode property, ordinary/reverse forced process differentials, and all four direct Compute Sanitizer tools pass. Clean commit `7b27cb1fb` records option-free ten-sample medians of 1.248x PDAL for ordinary and 1.487x for reverse at 2,000,000 points, and 1.528x ordinary at 4,000,000. A clean option-free Nsight trace proves automatic kernel launch; automatic-path memcheck reports zero errors. B0006 and D0041 record the accepted gate |
| Qualified forced categorical-partition CUDA region; host-selected standalone | Exact terminal `filters.groupby` chains over one or more input views | PDAL `int64` conversion, source-first output-view ids/file numbering, stable order within each group, empty input, and runtime conversion failures are preserved | 16-case host/ASan matrix, physical ordering property, forced groupby differential, and all four direct primitive sanitizer tools pass. The exact artifact-set benchmark covers four 21,970,934-point outputs but CUDA is 0.913x PDAL, so automatic standalone selection is rejected and resident partition publication remains open |
| Qualified forced return-partition CUDA region plus native view merge; host-selected standalone | Exact `filters.returns` over one or more views and exact no-option `filters.merge` composition | Fixed first/intermediate/last/only identities, stable source order, malformed return semantics, empty warnings, persistent merge view, SRS warning, and downstream single-view reopening are preserved | 29-case host/ASan matrix, physical 131,103-point property, forced process differential, all four direct primitive sanitizer tools, and 21,970,934-point composition pass. Complete forced CUDA is exact but 0.982x PDAL at full scale, so standalone host selection remains correct |
| Qualified forced count/cell partition CUDA region; host-selected standalone | Exact count-based `filters.divider` and finite positive-length `filters.splitter` across one or more input views; positive splitter buffers use the native host backend | Divider empty-view identities and stable partition/round-robin order; splitter persistent origins/map, first-encounter view ids, negative-boundary quirk, strict buffered membership, and up-to-four-view duplication are preserved | 35-case Debug/Release/ASan matrix, physical 131,103-point properties, forced process differentials, and all four direct primitive sanitizer tools pass. Exact artifact manifests cover 17 divider and 49 splitter outputs at 21,970,934 points; CUDA is 0.952x and 0.932x PDAL, so standalone host selection remains correct |
| Qualified forced color-map CUDA region; host-selected standalone | Exact `filters.colorinterp` with embedded or arbitrary GDAL ramps, explicit/partial/automatic bounds, clamp/invert, sample-deviation, and MAD forms in stream or standard mode | Existing RGB outside unclamped ranges, upper medians, stateful per-view bounds, empty/one-point inputs, skipped stream points, and exact ramp bytes/binning are preserved | 39-case Debug/Release/ASan matrix, physical 131,103-point RGB property, three forced process differentials, all four Compute Sanitizer tools, and the 21,970,934-point exact host differential pass. Clean explicit-range CUDA is 0.345x PDAL at 250,000 points and 0.787x at 21,970,934; auto-range is 0.756x at 4,000,000 and 0.873x at full scale, so standalone host selection remains correct |
| Force-only ordered-summary CUDA region; host-selected | Exact `filters.stats` metadata over streaming batches or standard-mode views; option-free execution retains upstream | Dimension order, online moment bits, advanced/enum/count/global metadata, warnings, bbox/SRS metadata, point order, and view topology are preserved | The prior 22-case Debug/Release/ASan matrix, direct summary-state comparison, compiled CUDA property, and 21,970,934-point exact host run remain. B0120 adds one physical exact 2M six-dimension CUDA metadata differential and fresh profile, but the complete process is only 0.429x PDAL and the 498.966048-millisecond serial recurrence is 8.701x the incremental CPU stats cost. No performance or automatic claim follows; broader physical matrix and four Compute Sanitizer tools remain unpromoted because the route is retained force-only |
| Qualified forced information-reduction CUDA region; host-selected standalone | Exact option-free `filters.info` schema/bounds/count metadata in stream or standard mode; point-list and nearest-query forms retain the exact host implementation | One fused device reduction preserves BOX3D sentinels, NaNs, infinities, signed-zero bits, first ties, global indices, coordinate decode, and ordered chunk merges; automatic selection remains disabled | The prior 20-case Debug/Release/ASan matrix and host evidence remain exact. B0104 adds the physical 131,103-point property, a forced metadata differential, all four Compute Sanitizer tools, and exact 1M/4M negative diagnostics at 0.526x/0.745x. Sixteen profiled kernels total only 0.26863 ms; the exact host-vs-bare control leaves a 1.040% reuse ceiling, so the lane is GPU-native coverage but not performance-qualified or selected (D0165) |
| Qualified automatic expression-histogram CUDA region | Exact `filters.expressionstats` for finite binary64 targets and predicates in the exact device VM; known-cardinality linear LAS/LAZ pipelines select CUDA at 4M points for one expression, 2M for two, and 1M for three or more | Stable predicate selection, numeric map order, duplicate counts, overlapping expressions, empty statistics, canonical labels, and the first signed-zero representation are preserved with device sort/reduce-by-key; unknown cardinality, unsupported values/programs, smaller jobs, explicit disable, or missing devices retain host execution | 19-case host/ASan matrix, ten-case physical forced-CUDA matrix, 131,103-point property, all four direct Compute Sanitizer tools, and automatic-path memcheck pass. Clean option-free B0008 medians are 1.126x PDAL at 1M/three, 1.212x at 2M/two, and 1.105x at 4M/one; the three below-threshold controls remain host-selected and exact. A clean option-free trace verifies stage-local pinned/device workspace reuse; cross-stage residency remains the next target |
| Proposed standalone automatic approximate-coplanar CUDA region; compiled off and not qualified | Default or explicit `knn=8` in a direct regular-file `readers.las` → filter → `writers.las` pipeline at 262,144 points or more, with uncompressed `.las` output and `extra_dims=all`; qualification-build candidate is the first visible `NVIDIA GeForce RTX 4090` at SM 89 built with CUDA 13.3; production artifacts compile the proposal off | The fixed-header count probe honors the reader count cap and rejects COPC, unknown/non-regular inputs, unsupported options and graphs, disabled/unavailable CUDA, and unqualified devices. The replacement is a standalone terminal region with no residency-reuse claim. Every failed gate delegates unchanged or uses the exact host wrapper. `PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA=1` proves selection plus execution | Revision `e0486c78e` clean forced two-warmup/ten-sample reports at 262,144 points are hash-exact and measure 2.158750x for LAS and 1.974880x for point-format-8 LAS. A 2.200810x LAZ report is dirty-tree diagnostic evidence only. These do not prove default selection. A clean LAZ report, clean option-free processes, proof-guard/fallback results, and an option-free profile remain absent, so this standalone proposal adds nothing to the current count of twenty. B0097's separately measured resident/direct-output composition is automatic under its own stricter envelope. Other compiler-supported NVIDIA architectures remain forced portability targets pending Vast physical qualification |
| Pending shared spatial-index CUDA region | Planner-owned compact 2D/3D Morton cell tables, exact bulk radius counts and scaling, deterministic capacity-driven radius tiles, bounded-grid and exact adaptive-BVH kNN, ordered mean and nearest-neighbor distances, fused covariance/eigensystems, typed device output projection, both modes of `filters.outlier` and `filters.nndistance`, exact `filters.radialdensity`, and bounded kNN envelopes of `filters.normal`, `filters.eigenvalues`, `filters.covariancefeatures`, and `filters.approximatecoplanar` — with statistical outlier able to consume and retain a planner-resident 3D kNN product before an adjacent compatible consumer and radius outlier sharing one planner-owned 3D radius region with an adjacent radialdensity consumer in B0126/B0127; `filters.normal`, `filters.eigenvalues`, `filters.covariancefeatures`, `filters.nndistance`, `filters.estimaterank` (D0078: 4.155x), `filters.optimalneighborhood` (D0079: 5.062x), `filters.neighborclassifier` (D0080: 3.743x), and the exact shared-radius selection of `filters.radiusassign` (D0081/B0019: 5.503x) are planner-resident with measured models | Stable CUB ordering and compact cells; strict PDAL `distance < radius²`; exact stage arithmetic; explicit query/status handling; adaptive Morton-BVH selection; exact covariance/Eigen behavior; direct self-inclusive neighbor counts for approximate-coplanar; statistical outlier preserves the pinned per-row mean, bounded incomplete-row resident repair or exact KD3 fallback, global mean/variance recurrence, warning behavior, and host Classification finale while keeping a downstream resident Classification column canonical; radius outlier stays outside the kNN adapter but uses the shared radius adapter and preserves the pinned strict count comparison/all-outlier finale; radiusassign OR-domain masks, optional 2D vertical caps, and source-order selection followed by the original ordered host assignment finale; resident XYZ/index/eigensystem/output columns, including unsigned-byte `Coplanar`, across compatible whole-view runs; deterministic half-open XY core ownership with closed radius ghosts, source-order gather, owner-only scatter, and complete-mosaic/capacity checks; planner-budgeted event-governed radius lanes with an `N=2..6` proof sweep and fixed two-lane default; 28-byte grid/76-byte adaptive-kNN accounting; B0087 adds an explicitly budgeted `12*k + 1`-byte ordered max-k query product for only a planner-proved adjacent outlier/NNDistance pair | B0087 retains exact max-k reuse after an 11.315433% same-binary direction gate and B0088 qualifies the explicit 1M direct composition at 21.389523x. B0090/B0091 remove the two bounded incomplete-row cliffs; the clean exact 4M route reaches 48.758796x and its final profile leaves 95.545682% of kernel work in the one shared gather. B0092 fits the separate direct-composition model and automatically selects only the exact 50K--16M measured shape; standalone claims do not follow. B0126 qualifies the exact explicit 1M same-radius one-index/direct-lifetime composition at 15.897x. B0127 then fits its distinct direct-composition model, advances the placement audit to 152/152 over 35 models, and automatically selects only the exact 250K--4M format-7/36-byte shape at 4.718x--45.178x; the exact 50K row remains unselected. Both query results remain stage-local. Branches, bridges, incompatible regions/dimensions, and wider/small-view shapes decline to the prior exact path. The prior shared suite retains its physical property, differential, and all-four-sanitizer evidence. Radiusassign adds exact host and forced-CUDA process matrices, host/device shared-index properties, upstream-vs-wrapper units, a resident assignment-column reuse gate, memcheck/racecheck, a seven-row resident-executor calibration, and the byte-identical 22M B0019 lane. B0079 adds the separately fitted exact rank/optimal composition; B0080 qualifies and automatically selects only its exact measured public shape at 8.157x. Radiusassign's traversal is native CUDA, but its expression finale is an exact host bridge and is not described as wholly device-native. B0010/D0053 record the exact class-specific lane sweeps and fail-closed active-width proof guard. The serialized loop uses the uniform grid for standalone execution, the Morton BVH otherwise, and an explicit tie-repair proof guard. Natural over-budget shared-index tiling, plan-wide tiled residency/reuse, adaptive exact kNN tiling, remaining clients/options, clean-tree break-even for earlier candidates, and physical runtime qualification beyond SM 89 remain pending |
| Audited host ordering bridge | Original `filters.randomize` with no `where`, retained in-place before later native regions | Exact libstdc++ seeded shuffle or `std::random_device` behavior; the stage preserves view count and does not falsely make unstable input order stable | 14-case Debug/Release/ASan composition matrix and a byte-exact 21,970,934-point randomize/native-assignment differential; not counted as a device implementation |
| Mapped-host LAS | Exact default translation and exact assign/ferry programs where the specialized host path wins | Byte-identical canonical writer output | Generated LAS matrix and large process differentials |
| Unchanged PDAL | Every other configured stage, option, application, plugin, graph shape, I/O format, and failure path | Original PDAL implementation receives the original command or remains an untouched stage in the hybrid graph | 142/142 published upstream PDAL tests plus fallback process differentials |

Resident multi-bridge neighborhood composition is additionally implemented in a
force/experimental envelope: coordinate-preserving bridges retain the index and
eigensystem cache. Approximate-coplanar can reuse an equal-`knn` eigensystem,
publish a typed unsigned-byte `Coplanar` column, and let an adjacent point
program consume that column without a host re-upload. Tied/incomplete
eigensystem rows use exact host KD3 repair before later device projections;
this is hybrid exact work and is not counted as wholly GPU-native. Autzen/RMIT
diagnostic speedups (2.42x/3.44x) are not a promotion checkpoint.
The approximate-coplanar tie-repair and resident-composition fixtures each
retain exact output across five consecutive physical SM-89 runs.

The current aggregate gate records 382 executed passes plus one expected
opt-in corpus skip in 383 Host Debug registrations and the same 382 plus one
skip in the ASan/UBSan preset. Exact-differential CMake injects
`detect_leaks=0` itself while the separate candidate-focused leak lane remains.
Three initial sanitizer failures compared process-specific LeakSanitizer
diagnostics and reran clean under that prescribed mode. Physical CUDA Release
records 553 passes plus nine documented corpus/full-fixture skips in 562
registrations in 140.51 seconds. That CUDA aggregate is an explicitly SM-89-only
runtime checkpoint. Separately, a
serialized CUDA 13.3 `all`-architecture build with tests off and the portable
guard on compiles `pdg_core` 51/51 at `--parallel 1`, including
`NeighborhoodKernels.cu`; that object contains SASS for every architecture
supported by that installed compiler and newest-target SM-120 PTX. This is
compile portability, not physical exactness evidence for other SMs. The
published upstream PDAL procedure passes
142/142: 140 local executables plus two remote-fixture tests on their
network-enabled rerun. These totals measure regression coverage, not catalog
completion or automatic performance qualification.

The public `gpupdal` executable is a thin dispatcher. Commands and parsed pipeline
graphs with no current candidate stage execute sibling `pdal` directly,
without loading the PDAL/GDAL/CUDA-linked `pdg-engine`; candidate, ambiguous,
or internally forced requests enter the engine and retain the same conservative
selection logic. When a LAS writer requests `extra_dims=all`, the hybrid
rewriter also compares ordered input and rewritten dimension layouts. A custom
layout-order change delegates before point or output side effects.

An exact uncompressed `writers.las` sink whose only non-routing option is the
literal `extra_dims=all` is planner-native under D0222. Its prepared PDAL
layout supplies the physical record width used by placement; ordinary
producers publish their declared columns, while a reorder-only terminal region
publishes its declared output-position-to-input-position permutation and uses
the mapped source for the complete records. B0223 extends the physical CUDA
selection matrix to seventeen cases and proves the measured compressed
format-7, 1M-point, 36 -> 100-byte normal/covariancefeatures output
byte-for-byte at 6.267794x median. Only that row is automatically device
selected; other cardinalities and carried source Extra Bytes retain exact host
selection. Named extra dimensions, compressed `.laz` output, and any
additional writer option remain host/delegated and are not covered by this
claim.

B0224/D0223 supersede only the compressed part of that statement. A `.laz`
sink with literal `extra_dims=all`, no additional writer option, and omitted
compression or compression true as a boolean/string is functionally native.
Its only automatic performance-qualified use is the checked-in r6
normal/covariancefeatures shape at exactly 1M points with compressed format-7
input and 36 -> 100-byte records; that public route is byte-exact at 4.529005x
median. A central runtime guard refuses every other compressed `all` graph,
including the older eigen-family and rank/optimal compositions. The physical
selection matrix has twenty cases covering all three admitted spellings,
executable rewrite/preflight/executor evidence, and wider-layout refusals.
Named extra dimensions, `compression=false`, and additional writer options
remain delegated; r2 is not unlocked.

B0225/D0224 measures rather than merely lists that r2 deferral. The complete
checked-in r2 reference remains exact at 0.988063x median, and resident stats
show its SMRF and HAG-NN regions both lack placement models behind the named
writer. Existing exact substitution evidence is slower. Therefore
`HeightAboveGround=float32` LAZ publication remains pinned-host functional
coverage, not native or performance-qualified coverage; it will be reconsidered
only with a complete profitable SMRF/HAG route.

B0226/D0225 performance-qualifies a host-dispatch route for r1 without adding
native stage coverage. `filters.reprojection` remains pinned-host; for the
literal measured 1M reference JSON/bounds and compressed format-7/36-byte
header facts, the thin launcher execs pinned PDAL before loading `pdg-engine`.
The public result is at parity with pinned PDAL and the paired direct-versus-
engine control is 1.075614x +/- 0.011503. Other counts, layouts, bounds, SRS
pairs, option forms, and CLI modifiers remain in-engine. This is functionally
supported and performance-qualified host selection, never GPU-native
reprojection.

B0227/D0226 performance-qualifies automatic `filters.outlier` CUDA selection
only inside the literal measured r4 route: 1M compressed format-7/36-byte input,
statistical `mean_k=8`/`multiplier=2`, the exact adjacent range and radius-1
sample grammar, and RTX 4090/SM89/CUDA 13.3/driver 610.43.03. The complete
public route is byte/diagnostic/status/order exact at 3.686747x median including
its full-input fingerprint. Range uses the existing exact point program;
`filters.sample` and LAZ I/O remain upstream. Other counts, layouts, option
forms, CLI modes, device profiles, or input fingerprints retain host/delegated
selection. This is neither generalized outlier placement nor GPU-native sample
coverage. **Retired from automatic selection by B0272/D0272**: after B0258's
host worker kNN passes and B0267's hashed sample table the exact host path
measures faster than this route at 1M (0.58 s vs 0.72 s) and at 4M (2.55 s
vs 2.78 s forced), because the route pays the fixed CUDA startup and a serial
host coordinate gather. The route and its lane stay reachable behind
`PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA`; the default r4 route is the
exact host path.

B0228/D0227 performance-qualifies a host-dispatch route for r5 without adding
native stage coverage. The engine cannot consume `readers.copc`; for the
literal measured COPC -> option-free stats -> uncompressed LAS grammar, the
thin launcher execs pinned PDAL before loading `pdg-engine`. The public result
is at parity with pinned PDAL and the paired direct-versus-engine control is
1.047266x +/- 0.016446 with 9/9 wins. Root, stage order, bounds, numeric
representations, options, lowercase endpoints, and the plain CLI invocation
must match; neighboring forms remain in-engine. `readers.copc`,
`filters.stats`, and `writers.las` all remain pinned-host in this route; this is
not GPU-native COPC pushdown or stats coverage.

B0229/D0228 performance-qualifies a host-dispatch route for r3 without adding
native stage coverage. Exact SMRF replacement is disabled, unchanged SMRF
makes the adjacent range rewrite order-unstable, and `writers.gdal` is outside
the native/resident endpoint; for the literal measured LAZ -> option-free SMRF
-> ground range -> GDAL IDW grammar, the thin launcher therefore execs pinned
PDAL before loading `pdg-engine`. The public result is at parity with pinned
PDAL and the paired direct-versus-engine control is 1.027780x +/- 0.005196 with
9/9 wins. Root, stage order, numeric representation, options, lowercase
endpoints, and the plain CLI invocation must match; neighboring forms remain
in-engine. `filters.smrf`, `filters.range`, and `writers.gdal` all remain
pinned-host in this route; this is not GPU-native terrain or raster coverage.

B0230/D0229 performance-qualifies a host-dispatch route for r2 without adding
native stage coverage. Its literal named writer makes plan-structure admission
refuse before placement or point processing; for the checked-in LAZ ->
option-free SMRF -> option-free HAG-NN -> named LAZ grammar, the thin launcher
therefore execs pinned PDAL before loading `pdg-engine`. The public result is at
parity with pinned PDAL and the paired direct-versus-engine control is
1.014563x +/- 0.005922 with 9/9 wins. Root, stage order, option type/value,
case-insensitive COPC suffix, lowercase endpoints, and the plain CLI invocation
must match; neighboring forms remain in-engine. D0224 remains in force:
`filters.smrf`, `filters.hag_nn`, named `HeightAboveGround=float32`
publication, and LAZ I/O all remain pinned-host in this route.

B0239/D0238 supersedes that performance selection only for the immutable 1M
r2 reference endpoint. Exact host SMRF feeds the planner-owned CUDA HAG-NN
lane; 347 equal-distance rows repair through the pinned compatibility ground
KD2 index before the unchanged named LAZ writer. The complete public pipeline
is byte/metadata/order/diagnostic/status exact at 1.270989x pinned PDAL with
9/9 wins. Automatic admission freezes the complete source fingerprint,
header/extent, literal grammar, named layout, and qualified SM89 profile.
Every neighboring fixture, count, option, layout, spelling, or profile retains
the B0230 direct-host route. This adds bounded GPU-native HAG-NN work inside r2,
not native SMRF, LAZ I/O, named-writer, or generalized terrain coverage.

D0239/B0240 expands the measured product target to fourteen equal-weight
headlines. r7 DSM, r8 raster colorization with CRS interaction, r9 standalone
polygon clipping with geometry reprojection, r10 decimation, r11 built-in
classification/refinement, r12 deterministic tiling, r13 heterogeneous merge,
and r14 conversion/compression are all functionally exact through unchanged
PDAL/host execution. Their three-pair warm/cold baselines are now performance
qualified as current host selections, not GPU-native coverage: warm speedups
are 0.940675x, 0.990701x, 0.897125x, 0.981252x, 0.902831x, 0.950555x,
0.961789x, and 0.979110x respectively. No new headline is automatically
accelerated. The suite's equal-workload geometric mean is 1.211602x warm and
1.206003x cold; total-wall speedup is 1.563923x warm and 1.554365x cold.

The zero-weight r10/r14 companions separately cover ordinal, radius, grid,
voxel-center, density-aware, LAS/LAZ, recompression, and supported COPC
directions. All are byte/metadata/order/stream/status exact under the
deterministic reference options and none has a stable positive automatic gate.
COPC writers require a fixed seed and one writer thread, and COPC readers one
request, because pinned PDAL's defaults do not produce repeatable bytes for
this fixture. B0240 chooses r11 attribution/composition next because it is the
largest remaining reference wall and loss; the existing standalone
neighborclassifier lane does not qualify that mixed compressed graph.

D0240/B0241 closes that r11 attribution without changing coverage. A literal
direct-delegation prototype was exact but measured 0.901714x and was removed.
The actual regression was the optional KD3 coordinate-cache branch in ordinary
uncached nanoflann callbacks; cached and uncached construction-time
specializations restore r11 to unresolved oracle parity at 1.000990x warm and
1.003846x cold while retaining the qualified 4M cached-LOF path at 20.295840x.
r11 remains exact host execution: its domain-qualified neighbor-classifier is
not GPU-native, performance-qualified, or automatically selected. The updated
headline aggregates are 1.220209x/1.215129x warm/cold geometric mean and
1.632359x/1.627712x total-wall speedup. r8 colorization attribution is next.

B0242/B0243 closes r8 attribution without changing stage coverage. A literal
direct-host prototype preserved every artifact and process observable, but its
corrected-final 21-pair route control did not resolve and its cold public row
resolved slower at 0.988765x pinned PDAL. The prototype was removed: r8 still
executes exact host reprojection, GDAL colorization, LAZ decode, and LAZ encode
through the engine fallback. It adds no GPU-native, performance-qualified, or
automatic-selection entry. The final aggregate is 1.220823x/1.214832x
warm/cold geometric mean and 1.633139x/1.627446x total-wall speedup; r13 merge
is next.

B0244/B0245 and D0243/D0244 close r13 attribution without changing coverage.
The one-reader-worker direct prototype was exact and resolved faster warm, but
its corrected-final 41-pair cold gate remained unresolved. The matcher,
fingerprint, proof, and override were removed. `readers.las`, `filters.merge`,
and `writers.las` continue unchanged in the engine/PDAL host path and gain no
host-native, GPU-native, performance-qualified, or automatic-selection entry.
The current aggregate remains B0243's 1.220823x/1.214832x warm/cold geometric
mean and 1.633139x/1.627446x total-wall speedup. r12 tiling is next.

B0246/D0245 closes r12 attribution without changing coverage. The exact
seven-output graph remains unchanged PDAL host splitter/writer execution. A
literal direct-delegation prototype is slower than pinned PDAL and unresolved
against the same-final engine route in both cache states; a bounded reader
worker sweep also rejects 1/2/4/6/8 in favor of the default seven. The matcher
was removed and the literal graph is frozen engine-owned. No `readers.las`,
`filters.splitter`, `writers.las`, host-native, GPU-native,
performance-qualified, or automatic-selection entry is added. Current
aggregates remain 1.220823x/1.214832x warm/cold geometric mean and
1.633139x/1.627446x total-wall speedup; r9 polygon clipping is next.

B0247/D0246 corrects r9's axis-swapped empty geometry and closes its
attribution without widening coverage. The substantive reference emits
473,825 points and exercises multipolygon, hole, disjoint-member, boundary,
and geometry-CRS behavior through unchanged `filters.crop`/PDAL host
execution. Direct-exec and private embedded-CLI prototypes are exact but
resolve slower than pinned PDAL and are removed; the literal graph is frozen
engine-owned. No `filters.crop`, host-native, GPU-native,
performance-qualified, or automatic-selection entry is added. Corrected
fresh same-binary aggregates are 1.205506x/1.206910x warm/cold geometric mean
and 1.617787x/1.615619x total-wall speedup; r7 DSM is next.

B0248/D0247 performance-qualifies a bounded direct-host dispatch for the
literal r7 DSM headline without changing stage coverage. The complete
one-metre Float64 maximum-Z, first/only-return graph executes unchanged sibling
PDAL directly and resolves 1.058781x warm / 1.049022x cold against the
same-final engine path. Public comparison remains a reported 0.982017x warm
loss and unresolved 0.990449x cold versus pinned PDAL; this is a faster exact
PDG backend, not a native raster algorithm or claim that r7 beats PDAL.
Root/stage/option/type/extension/SRS/bounds, CLI-modifier, malformed-input, and
`PDG_*` control drift remains engine-owned. A bounded generated compressed-LAZ
matrix exercises the selected route, full raster bytes/metadata, refusals, and
determinism. r7 gains performance-qualified automatic direct-host execution,
but no `filters.returns`, `writers.gdal`, host-native, or GPU-native stage.
Fresh corrected-clock aggregates are 1.223460x/1.227305x warm/cold geometric
mean and 1.600366x/1.621301x total-wall speedup; r10 decimation is next.

B0249/D0248 performance-qualifies a bounded direct-host dispatch for the
literal r10 headline without changing stage coverage. The complete lowercase
LAZ -> `filters.voxelcentroidnearestneighbor(cell=2.5)` -> lowercase LAZ graph
with string `compression:"true"` executes unchanged sibling PDAL directly and
resolves 1.041226x warm / 1.034230x cold against the same-final engine path.
Its public pinned-PDAL comparisons remain unresolved at 1.002615x warm and
0.993073x cold median speedup. Root/stage/option/type/value/extension,
CLI-modifier, malformed-graph, and `PDG_*` control drift remains engine-owned;
all six zero-weight r10 variants remain engine/host selected. A bounded
nine-case generated-fixture matrix proves selected dense/sparse/empty/
malformed/deterministic behavior, legal and invalid drift, exact LAZ
bytes/order, streams, and status. r10 gains performance-qualified automatic
direct-host execution but no `filters.voxelcentroidnearestneighbor`, LAZ,
host-native, or GPU-native stage. Fresh same-binary aggregates are
1.235323x/1.237401x warm/cold geometric mean and 1.631522x/1.633568x
total-wall speedup; r14 conversion/compression is next.

B0250/D0249 closes the r14 conversion/compression slice without changing
coverage or selection. The LAS -> LAZ headline already uses B0197's generic
two-stage LAS-family direct sibling-PDAL delegation; all six zero-weight
LAS/LAZ/COPC directions likewise execute unchanged PDAL. A bounded
12-case/19-execution matrix adds all-direction positives, repeat byte
determinism, LAS count/compression and COPC identity checks,
truncated/malformed sources, unsupported writer refusal, and complete
artifact/stream/status comparison. A 1/2/4/6/8 reader-worker sweep finds one
worker 1.017714x warm but cold unresolved at 1.001577x +/- 0.014533, so no
tuning is selected. r14 remains functionally exact host execution through the
existing generic direct-host route and gains no new performance-qualified
selector, host-native LAS/LAZ/COPC stage, or GPU-native stage. The unchanged
product retains the B0249 aggregate: 1.235323x/1.237401x warm/cold geometric
mean and 1.631522x/1.633568x total-wall speedup. Exact parallel LAZ chunk
compression is the next feasibility target.

B0251/D0250 adds one bounded exact host-native codec backend without widening
standalone stage or GPU-native coverage. For the literal r14 1M LAS -> LAZ
headline and exact LAS 1.4 format-7/36-byte reference layout, `writers.las`
uses two independently compressed fixed-50K chunks at a time and publishes
them in source order. The production primitive is byte-exact and 1.735238x
faster. B0252/D0251 correct the fail-closed environment boundary and keep the
activation hook test-only; corrected final public qualification is 1.515827x
warm and 1.504457x cold with 9/9 wins. Grammar, layout, count, bounds,
modifier, and control drift and the six r14 companions remain serial
host-selected. The 13-case/21-execution matrix proves full artifacts/process
observables and actual two-worker writer activation. Fresh aggregates are
1.262824x/1.260608x warm/cold geometric mean and 1.641092x/1.634768x
total-wall speedup. This is functional and
performance-qualified host-native compression coverage inside the existing
PDAL writer, not a new stage, GPU-native stage, or CUDA selection. Ordered
B0253/D0252 then reject wider 10/12-worker decode pools and a Point14
mode-gated setup cleanup: all wider-pool r7/r10 warm screens are slower, and
the setup cleanup's 1.014837x primitive gain leaves public r10 unresolved at
1.012374x median. The production reader/codec remains unchanged. A maintained
formats-6--8/Extra-Bytes decoder unit and primitive are coverage and
measurement infrastructure, not host-native stage coverage or a selected
backend.

B0254/D0253 complete the reader-worker screen and add one bounded workload
tuning without changing stage-native coverage. Four workers resolve on the
final literal r7 DSM route at 1.031858x warm and 1.020398x cold; the launcher
appends the existing upstream reader option only for the exact 1M compressed
format-7/36-byte r7 grammar and facts. The integrated r10 cold interval remains
unresolved, so r10 keeps its default reader and no general LAZ schedule is
selected. Count, layout, grammar, option, CLI-modifier, environment, and
neighboring-workload drift retain their prior exact path. This is performance-
qualified direct-host workload scheduling, not a new host-native LAS reader,
stage, GPU-native stage, or CUDA path. B0254 aggregates were 1.267318x/
1.262102x warm/cold geometric mean and 1.642633x/1.632910x total wall.

B0256/D0255 select one semantics-preserving host product reuse without
changing stage-native coverage. B0255's first implementation was superseded
after review found that generic XYZ assignment could leave a stale incoming
PointView product. Corrected statistical `filters.outlier` explicitly
invalidates incoming products, builds its unchanged fresh nanoflann KD3 tree,
and publishes that tree for an adjacent neighborhood consumer. Radius mode is
unchanged. The literal r11 process remains upstream host execution and is
exact at 1.029892x warm / 1.037311x cold over nine pairs in each state; its
domain-qualified neighbor classifier remains outside automatic CUDA
selection. This is qualified host index reuse, not a new host-native stage,
GPU-native stage, dispatcher route, grammar, or placement model. B0256
aggregates were 1.267705x/1.278088x warm/cold geometric mean and 1.656538x/
1.665729x total wall.

B0258/D0257 make the exact host statistical `filters.outlier` and
`filters.neighborclassifier` per-point kNN passes run on fixed-chunk host
workers over the read-only PointView-owned nanoflann index. Per-row results
are bit-identical at any worker count; the outlier's serial online-moment
reduction, the classifier's vote/merge/application, options, diagnostics,
output order, and both stages' native/GPU-native/automatic envelopes are
unchanged. The literal r11 process is exact at **4.309167x warm / 4.345954x
cold** over nine pairs per state (paired 4.321977x +/- 0.034672 and 4.352975x
+/- 0.038573, 9/9 wins), 3.050913x at a 50K control and 4.297817x at 250K,
with the same-final-binary serial control at 1.035408x. This is an exact
host-native performance qualification of general stage behavior — any
pipeline using either stage above the 4,096-rows-per-worker threshold benefits
— not a new dispatcher route, grammar, placement model, or GPU-native
coverage. Forcing the experimental CUDA statistical outlier alone on r11 stays
exact but is slower (4.069126x) than the host-worker path.

B0274/D0274 build every KD2/KD3 nanoflann tree concurrently
(structure-identical `divideTreeConcurrent`, at most eight threads, snapshot
copy on fixed chunks): the r6 repair tree 0.104 s -> 0.058 s and every host
neighborhood consumer's build likewise; final numbers are recorded in B0274.
Exact host-native behavior; coverage labels do not change.

B0273/D0273 accumulate `writers.gdal` rasters on row bands in parallel
(each band replays the full point order and updates only its own rows, so
every cell sees pinned PDAL's exact update sequence; streaming dynamic
radius grids and percentiles stay serial): final numbers are recorded in
B0273. This is exact host-native writer behavior; coverage labels do not
change.

B0271/D0271 widen the `--fast` contract to resolve kNN distance ties in
device order (`pdg::knnStatusMask()` clears `KnnDistanceTie` at the index,
so no identity-observing consumer runs its host tie repair under `--fast`):
r6 **11.416957x / 11.336293x** (25 of 1,000,000 records differ from stock in
attributes only) and r2 **2.109707x / 2.115076x** (125 records) under the fast
contract; every other headline is record-identical to its exact run; the
default contract is unchanged and re-proven byte-identical. Fast-contract
aggregates are recorded separately from the exact ones in B0271. Coverage
labels do not change; the fast contract never satisfies exact product
coverage.

B0270/D0270 decode COPC tiles in parallel under `requests=1` while
emitting them in pinned fetch order (one request thread as asked; a decode
pool of at most eight threads; sequence-keyed emission; keep-alive
backpressure on fetched-but-unconsumed tiles): r5 **1.830419x /
1.943686x** (from parity); final numbers are recorded in B0270. This is exact
host-native reader behavior; coverage labels do not change.

B0269/D0269 unpack `readers.las` records in parallel in both execution
modes: a new `Streamable::readStreamBatch` reader hook and per-tile runs in
the standard `read()` consume tiles in the pinned order and run `loadPoint`
on disjoint rows through a fixed-slot pool (at most eight slots); small
runs, streaming `start`, and the read callback stay serial. Every
LAS/LAZ-reading headline gains (three-pair probe: r14 2.56x, r13 2.68x, r7
1.35x from parity, r9 1.96x, r1 2.14x, r12 1.93x, r10 1.42x, r3 1.59x);
final numbers are recorded in B0269. This is exact host-native reader
behavior; coverage labels do not change.

B0268/D0268 pack `writers.las` point records in parallel in both execution
modes: the standard block path and a new `processStreamBatch` override run
`fillPointBuf` on a fixed-slot pool (at most eight slots), each slot packing
a contiguous run of records at their fixed offsets into its own
`las::Summary`, merged in slot order (identical strict-comparison folds). A
run in which any slot would have logged a per-point warning (PDRF < 6
Overlap/Classification > 31) or threw is discarded before anything is
committed and repeated with the serial loop, so stderr, diagnostics, and
partial output are pinned; `discard_high_return_numbers`, `where`, the
streaming first-point setup, and small runs stay serial. Every LAZ/LAS
writing headline gains (r14 2.08x from 1.47x, r13 1.92x from 1.44x, r12
1.47x from 1.23x, r6 9.35x from 8.65x, r1 1.57x from 1.49x on five-pair
probes); final numbers are recorded in B0268. This is exact host-native
writer behavior; coverage labels do not change.

B0267/D0267 replace `filters.sample`'s ordered voxel map with a hashed table,
read candidate lists in place, and prune neighbor voxels that cannot contain
a violating coordinate; every greedy decision is unchanged and proven by a
new pinned-oracle sample matrix. r4 is **5.306974x / 5.271122x** (from
3.80x); aggregates are recorded in B0267. Coverage labels do not change.

B0266/D0266 make `filters.reprojection` exact and parallel in both streaming
and standard modes: a new `Streamable::processStreamBatch` hook lets the
filter reproject each 10,000-row streaming batch on a fixed-slot pool (at most
eight slots, one lazily cloned GDAL transformation per slot), with
slot-ordered appends, post-join skip marking, and first-failure diagnostics
identical to pinned PDAL; `where`, tiny batches, and clone failures keep the
serial loop. r8 is **2.093332x / 2.140982x** (from 1.10x) and r1
**1.489033x / 1.485207x** (from 1.08x); aggregates are recorded in B0266.
This is exact host-native performance of general stage behavior; coverage
labels do not change.

B0265/D0265 give the fork's SMRF port (r2's automatic route) the same pooled
diamond morphology through an internal pass pool: r2 **1.624190x /
1.604373x** (from 1.52x/1.48x); aggregates **1.750757x/1.737266x** geometric
mean and **3.268150x/3.192511x** total wall. Coverage labels do not change.

B0264/D0264 run the exact host `filters.smrf` progressive-filter diamond
morphology (pooled, eight workers), void-fill queries, classify-ground pass,
and minimum-Z raster (per-worker partials merged under the serial rule) on
fixed-chunk host workers with bit-identical results, proven by forced-worker
repeats of the whole SMRF matrix and three pinned-oracle 240 x 240 lattice
cases. SMRF at 1M falls from about 0.34 s to 0.17 s: r3 **1.352415x /
1.338430x** (from parity) and r11 **8.867727x / 8.863616x**; aggregates
**1.731986x/1.729459x** geometric mean and **3.185933x/3.147773x** total
wall. r2's automatic route uses the fork's SMRF port and is unchanged.
Coverage labels do not change.

B0263/D0263 make D0237's exact cached-coordinate KD3 backing the
`PointView::build3dIndex()` default for every published product, with a view
coordinate epoch that refreshes a reused snapshot from live coordinates so
pinned nanoflann's stale-tree/live-coordinate reuse semantics hold
bit-for-bit (proven by `Kd3Refresh` units, the r11 pinned-oracle
producer/mutator/consumer matrix, and full Host/CUDA aggregates with the
snapshot verifier armed). Every host KD3 consumer builds about 4x and
queries 3--4x cheaper at 24 bytes per point. r11 is **7.372457x warm /
7.286586x cold**; aggregates are **1.672490x/1.663704x** geometric mean and
**3.073590x/3.042106x** total wall. This is exact host-native performance of
general library behavior; coverage labels do not change.

B0262/D0262 cut r6's exact eigen tie repair from 0.525 s to 0.142 s: when
the planner proves the terminal writer alone follows a neighborhood region
(`pdg_region_terminal_sink`), the repair builds a private cached-coordinate
nanoflann tree instead of publishing an uncached one nobody reads. r6 is
**8.648309x warm / 8.664582x cold** (from 6.32x/6.39x); aggregates are
1.633482x/1.624540x geometric mean and 2.924049x/2.894853x total wall. Every
other graph keeps the published uncached tree; coverage labels do not change.
The same entry fixes launcher r2-marker arming on the engine route (a latent
CUDA test failure since B0243) and records the first full physical CUDA
aggregate of the session at 827/827.

B0260/B0261 and D0259/D0260 withdraw B0256's reuse claim: the published
statistical-outlier tree was byte-inexact against pinned PDAL once a
coordinate mutator preceded a later KD3 consumer (found and pinned by a new
pinned-oracle differential lane; the forked-host-CLI oracle and
`PDG_ORACLE_PDAL` delegation had hidden it). The outlier is again private and
fresh, now on the exact cached-coordinate backing; r11 is **5.207843x warm /
5.267489x cold** and the aggregates are 1.592353x/1.595639x geometric mean
and 2.794157x/2.794157x total wall. Coverage labels do not change.

B0259/D0258 supersede D0250's serial-default LAZ writer. `writers.las` now
compresses independent fixed-50K lazperf chunks on `min(4, hardware threads)`
workers by default; the internal channel still lets the launcher keep its
measured two-worker r14 selection. A same-binary engine probe over all
fourteen headlines showed every LAZ-writing workload faster at two and four
workers and every non-LAZ workload unchanged, all exact. Final public
aggregates are **1.597555x warm / 1.589241x cold** equal-workload geometric
mean and **2.792319x / 2.772361x** total-wall speedup; nine-pair r6 is
6.316372x/6.387622x, r13 1.437716x/1.434382x, r1 1.081819x/1.101200x. This is
general exact host-native writer behavior, not a route, automatic envelope,
or GPU-native coverage.

B0231/D0230 performance-qualifies the existing exact
`filters.neighborclassifier(k=7)` CUDA path for automatic selection without
widening its stage semantics. The public selector requires a literal
three-stage LAS/filter/LAS grammar, uncompressed format-7 36-byte input and
output records, 250K--16M points, mapped-source and direct-Classification
boundary facts, the 112-byte/point planner-owned index, and the exact SM89
profile. The 1M public process is byte-, stdout-, stderr-, order-, and
status-exact at 4.033570x pinned PDAL. The measured 50K loss and tested option,
compression, format, `k`, source, and preflight drift retain the exact host
path before commitment; runtime-placement units separately reject topology,
layout, and index drift. An injected post-execution proof failure exits 1
without publication or host retry because the selected route has already run.

B0232/D0231 performance-qualifies the existing exact direct
`filters.sort(Z,ASC,NORMAL)` composition without changing the broader stage
contract. Automatic selection requires the literal three-stage grammar,
uncompressed format-7 36-byte input/output records, `extra_dims=all`, finite
comparator-unique Z, 600K--16M points, 8-byte upload/download facts, zero
packing/indexes, a measured conservative 64-byte/point reservation, and the
exact SM89 profile. Final public runs are exact at 1.842475x pinned PDAL at
600K and 3.267173x at 1M. Duplicate/non-finite data, grammar/layout/profile/
budget drift, and side-effect-free publication refusals return to the
unchanged oracle before commitment.

B0233/D0232 performance-qualifies the existing exact per-stage
`filters.label_duplicates(Classification) -> filters.nndistance(k=10) ->
filters.assign(UserData = Duplicate)` hybrid without qualifying standalone
labeling or the slower resident prototype. Automatic selection requires the
literal five-stage graph, uncompressed LAS 1.4 format-7/36-byte input with a
375-byte header, zero VLRs/EVLRs and no trailing bytes, 250K--16M points, and
the exact SM89 profile. Final public runs are exact at 2.846286x pinned PDAL at
250K and 6.391572x at 1M. The measured 50K loss, grammar/layout/profile drift,
retain the original host path. Recoverable label CUDA failure uses the exact
wrapper host operation inside the rewritten graph before writer execution;
ordinary writer filesystem behavior is pinned by existing, input/output-alias,
and symlink differentials.

B0234/D0233 performance-qualifies the existing exact direct
`filters.skewnessbalancing` composition without changing the ordinary stage
boundary. Automatic selection requires the literal option-free three-stage
LAS/skewness/LAS graph, `extra_dims=all`, uncompressed format-7 36-byte
input/output records, mapped source, finite comparator-unique Z, 450K--16M
points, a conservative 65-byte/point reservation, and the exact SM89 profile.
Final public runs are exact at 1.200177x pinned PDAL at 450K and 3.091007x at
1M; the public 400K control is only 1.065118x and stays host-selected. Ties,
nonfinite data, grammar/layout/profile/budget/proof drift, and side-effect-free
publication refusals return to the unchanged oracle before commitment.

The generic packed bridge exposes a `FixedPointTable` batch as one packed AoS
transfer, transposes only touched dimensions on device, keeps custom
intermediates in device SoA storage, and repacks only written dimensions. This
prevents a CUDA region from degenerating into a per-field/per-point host copy.
The direct LAS path removes the generic PointView boundary entirely when the
whole I/O envelope is proven.

## Exact point-program coverage

The following stages are GPU-native in their documented exact option envelope:

- `filters.assign`: ordered statements, conditions, custom intermediates, and
  the bounded arithmetic/logical expression VM;
- `filters.ferry`: mappings that preserve PDAL layout resolution and source
  semantics;
- `filters.expression`: stable filtering with the exact conditional VM;
- `filters.range`: grouped bounds, negation, inclusive/exclusive endpoints,
  and NaN behavior lowered to the same predicate primitive;
- `filters.crop`: one 2D or 3D bounds string, optional Boolean `outside`,
  inclusive faces, stable compaction, and PDAL-compatible NaN behavior.
- `filters.sort`: exact typed normal/stable pass semantics, directions, and
  multi-dimension priority. The ordinary boundary remains host-selected; only
  B0232's strict mapped-source/permutation-publisher form is automatic from
  600K through 16M.
- `filters.mortonorder`: exact ordinary most-significant-bit ordering and
  reverse Morton traversal, with stable duplicate codes and a conservative
  finite/nondegenerate CUDA envelope selected automatically from 2,000,000
  points.
- `filters.expressionstats`: exact canonical-expression metadata and ordered
  binary64 maps, with finite-target predicate/select/radix histograms selected
  automatically at the measured persistent-workspace 4M/2M/1M
  one/two/three-expression work curve.
- `filters.lof`: exact shared-kNN LOF with host closure repair, selected only
  inside B0043's explicit `minpts=10`/`UserData`-assign/default-LAS public
  shape after the measured resident placement and no-side-effect preflight
  accept. B0238 accelerates only large exact repair closures with the single
  cached compatibility index and deterministic host workers; the selector is
  unchanged.
- `filters.nndistance`: exact shared-kNN kth distance, selected only inside
  B0045's explicit `k=10`/`UserData`-assign/default-LAS public shape after the
  measured resident placement and no-side-effect preflight accept.

Twenty-two additional filters have complete exact host implementations and
CUDA implementations in the documented option envelopes, but are not yet
counted in the ten qualified filters:

- `filters.decimation`: finite `step >= 1`, nonnegative `offset` and `limit`,
  with PDAL's distinct standard and streaming candidate sequences;
- `filters.head`: nonnegative `count` and Boolean `invert`, with global state
  across stream batches;
- `filters.tail`: nonnegative `count` and Boolean `invert` in standard mode,
  including the exact oversized-count warning.
- `filters.locate`: case-insensitive minimum/maximum selection with first-tie,
  NaN/infinity, sentinel, empty-view, and invalid-kind behavior preserved.
- `filters.transformation`: full row-major homogeneous host maps and an exact
  affine double-XYZ device envelope, with matrix files, inversion, and SRS
  override delegated.
- `filters.iqr`: exact 25th/75th order statistics, multiplier fences, and
  strict stable selection;
- `filters.mad`: exact upper medians, absolute deviations, configurable scale,
  division predicate, and stable selection.
- `filters.groupby`: exact signed-64 categorical grouping across multiple input
  views, with source-first view creation and stable per-group order.
- `filters.returns`: exact first/intermediate/last/only partitioning across
  multiple input views, with fixed output identities, stable source order, and
  warning behavior preserved.
- `filters.divider`: exact count-based sequential and round-robin partitioning
  across multiple input views, including requested empty views.
- `filters.splitter`: exact persistent XY cell partitioning on host, with an
  exact primary-cell CUDA envelope for nonpositive buffers.
- `filters.colorinterp`: exact GDAL ramp loading, stream/standard range state,
  clamp/invert and stddev/MAD forms, with a finite binary64 CUDA map envelope.
- `filters.stats`: exact ordered online moments, metadata, warnings, bbox/SRS,
  enumeration/count/global forms on host, with a finite basic-summary CUDA
  envelope that preserves the serial recurrence within each dimension.
- `filters.info`: exact schema, bounds, count, selected-point, nearest-query,
  and historical parser behavior on host, with a fused XYZ bounds/count CUDA
  envelope for the option-free form.
- `filters.outlier`: exact radius and statistical behavior on host; positive,
  finite radius mode has a shared bulk-count CUDA envelope with exact
  capacity-driven core/ghost tiling, while statistical mode has a bounded
  shared-grid kNN and exact ordered-mean CUDA envelope for
  `0 <= mean_k < 64`. Both retain pinned nanoflann outside that envelope; the
  host statistical kNN pass runs on exact fixed-chunk workers (B0258).
- `filters.radialdensity`: exact upstream count-to-volume scaling on host;
  finite positive radius has the shared whole-view or deterministic tiled CUDA
  envelope and publishes only core-owner values in original source order.
  Nonpositive/nonfinite radii, conditional forms, and invalid option types
  retain unchanged PDAL.
- `filters.normal`: exact default kNN covariance, Eigen decomposition,
  curvature, and `always_up` behavior on host; `2 <= knn < 64` has a shared
  kNN/covariance/eigensystem CUDA envelope whose four outputs are projected
  into resident device SoA. Radius, viewpoint, refinement, and conditional
  forms remain on unchanged PDAL.
- `filters.eigenvalues`: exact ascending eigenvalues and optional
  normalization on host; `2 <= knn < 64` with unit stride has the shared
  kNN/covariance/eigensystem CUDA envelope whose three outputs are projected
  into resident device SoA. Radius, strided, and conditional forms remain on
  unchanged PDAL.
- `filters.covariancefeatures`: exact non-density feature sets and all three
  eigenvalue modes on host; the bounded single-thread/unit-stride kNN envelope
  uses the shared CUDA eigensystem and device projection for every algebraic
  feature. Exact `Omnivariance` and `Eigenentropy` use a narrow host
  transcendental bridge and are therefore not fully GPU-native. Density,
  optimal/radius neighborhoods, parallel/strided, and conditional forms remain
  on unchanged PDAL. Adjacent exact assign/ferry stages can consume all
  retained result columns without re-uploading them.
- `filters.approximatecoplanar`: exact defaults, direct self-inclusive neighbor
  requests, strict threshold comparisons, zero-covariance diagnostic and prior
  value preservation, and solver-failure behavior on host; `3 <= knn <= 64`
  reuses the shared CUDA eigensystem and writes a resident unsigned-byte
  `Coplanar` column. Automatic selection remains off. Too-small views,
  out-of-envelope neighbor counts, conditional forms, invalid options, and
  recoverable device failures retain exact host or unchanged-PDAL execution.
- `filters.smrf`: exact selected-return host execution uses the pinned
  KD2Index for eight-nearest inpainting. The bounded global-raster CUDA
  prototype is compiled but unqualified and fails closed because its
  eighth-neighbor cutoff-tie rule is known to differ from the oracle.
  Automatic selection, tiled/scalable canvases, and composable grid residency
  remain disabled (B0214--B0216, D0221).
- `filters.pmf`: exact selected-return host execution and the bounded D0084
  global CUDA envelope retain the pinned initial/final fractional-cell
  binning, first-source minima, deterministic one-nearest fill,
  exponential/linear progressive morphology, strict thresholds, and
  classification. D0088 adds a required planner-resident tiled correctness
  lane with one-cell halos, owner-only mosaics, full-pass barriers, cumulative
  budget checks, and exact >4,096-cell seam fixtures. D0089 initially
  constructed the source raster once per execution in the two already-budgeted
  host backings, preserving exact source-bit/tie semantics and guaranteeing
  region completion after failure. It rejects
  nanoflann-order-dependent distinct-valued void-fill ties before publication.
  D0090 declares two full-frame device phase backings and materializes them
  only after the single raster build passes tie proof. When the pair fits the
  runtime budget, the lane performs one raster H2D, retains every morphology
  and filter surface on device, and performs zero raster D2H; otherwise it
  retains the exact host-tiled path. The existing tile scratch provides a
  column-major populated-source list without extra allocation. B0028's exact
  dense/sparse fixtures reach 9.009503x/4.764988x, but automatic selection,
  adjacent Grid composition, cross-stage product reuse, device-native source
  construction, and worst-case scalable sparse/void fill remained open after
  D0090. D0091 replaces unbounded literal source scans above 255 populated
  cells with an exact occupancy hierarchy in the second planned host backing.
  Raw-bit heterogeneous, large-origin/fractional, nonfinite-center, and
  multiway tie/reattempt tests are exact. B0029 observes near-linear work for
  its named 256-source 257x257/513x513 distributions, but branch-and-bound can
  still degenerate. D0092 adds the separately budgeted fitting device-source
  path: exact minimum-bit selection and compact-source tie proof occur in the
  provisional phase allocation, which is promoted only after success. B0030
  records zero raster transfers and exact controlled 3.156x-38.950x results.
  Its proof remains target-cells-times-sources work, so no general scalability,
  automatic, or broad composition claim follows. D0093 admits the narrower
  compatible adjacent-PMF case: one resident product and one promoted phase
  allocation carry sequential, fully consumed raster generations across stages
  with the same Grid/cell and exact JSON return source. Every PMF stage rebuilds
  its raster because morphology overwrites both backings. B0031 observes one
  `GridBuild`, two `RasterBuild` generations, zero raster transfers, exact
  output, and 1.039838x on the final symmetric-scope controlled pair. The
  earlier 4.590078x primitive/full-pipeline, 1.056201x asymmetric-teardown, and
  comparison-inclusive 1.016953x measurements are superseded. Different
  return/cell sources,
  semantic surface reuse, and cross-kind Grid/cloth composition remain
  rejected.

These remaining CUDA paths are force/require-only until each path's outstanding
device differential, Compute Sanitizer, exactness, residency, and same-machine
break-even gates pass on the reference RTX 4090. Some have already passed
their physical and sanitizer gates but remain host-selected after a negative
standalone performance result. Compilation or host exactness alone does not
promote the 22-filter automatic qualification count.

`filters.merge` is also native in its exact no-option envelope. It is a host
view-composition operation rather than a useful standalone GPU kernel: the
implementation retains PDAL's persistent output view and SRS warning, clears
the multi-view planner boundary, and lets a following qualified region choose
its fastest host or CUDA path without delegating the full pipeline.

`filters.randomize` remains the original exact PDAL implementation and is not
included in either device count. Its audited no-`where` form is a single-view
ordering bridge, so it no longer closes later native regions. Exact
Fisher-Yates permutation generation is serial and library-version dependent;
the future device-resident form will upload the host mapping once and retain
device indirection instead of pretending that a standalone round trip is GPU
acceleration (D0025).

The P1.5 placement model remains production-default-off under D0054–D0056. Its
frozen SM-89 calibration audit predicts 52/52 measured winners in the pre-P1.5
complete-process matrix and explicitly accounts for non-column result spills
such as masks and permutations. The calibrated profile now lives in the core,
is keyed exactly by device/SM/driver/toolkit, and is checked against the
manifest before the audit runs. A bounded region rule can select profitable
linear resident regions across an unsupported host boundary while charging one
shared startup/synchronization toll and aggregating only selected-region
physical terms. Unknown profiles, models, memory envelopes, or branch
topologies fail closed. This does not add to the automatic or native stage
counts. The explicit `gpupdal resident` observer now constructs bounded runtime
facts and reports planned boundaries plus transfer-inferred crossings. Its
physical small V7 process selects host and is exact. Its two-region mixed probe
is also exact, but executes one-stream PDAL point-program wrappers rather than
the B0005-calibrated direct fused-LAS two-lane executor; it therefore does not
complete V4/E2, validate executable D4 placement, or establish B1/B2. Runtime
stats expose that mismatch. D0057 separately connects one terminal assign/ferry
region to the calibrated B0005 executor, feeds it the planner's D1 VRAM budget,
reports its D3 two-lane schedule and exact transfers, and passes the hash-pinned
complete-process differential. This adds no stage or automatic/native coverage
count, does not qualify the mixed wrapper, and does not complete D4 or V4/E2.
Normal `gpupdal pipeline` behavior remains unchanged. The next
production-relevant slice is a resumable planner-owned device batch with a real
materialized unsupported-stage spill/upload boundary.

D0058 supplies that private, planner-scoped resumable boundary executor for
selected assign/ferry regions. It executes planned first/last-use column
lifetimes, preflights the bounded two-lane schedule, allocates its packed lanes,
and probes their column high-water sequence before writer side effects. It
materializes explicit full-record spill/upload around an unsupported seeded
`filters.randomize` stage. The hash-pinned
21,970,934-point V4 load is byte-identical to pinned PDAL and all compact CUDA
sanitizer lanes are clean. This is architecture coverage, not a new public
stage port. D0059 gives the generic executor an explicit physical boundary fact
table. Its four V4 predictions now equal the observed full-record transfer and
pack/repack totals, and its predicted two-lane peak equals the executor
schedule, satisfying E2 without changing coverage. The fact table is admitted
only after every candidate region passes the PointView rewrite envelope; an
unsupported region retains calibration-default accounting. The executor still
reports that its compute calibration differs from the frozen direct-LAS
residual, so D4 and catalog-wide native coverage remain open.

D0060 closes V5/E3 for that generic resident point-program machinery without
adding stage coverage. On the hash-pinned 21,970,934-point load, the reported
1,757,674,720-byte whole-view lane exceeds the 268,435,456-byte planner budget,
while 168 fixed tiles execute in two lanes with an exact 20,971,520-byte
planned/observed peak and reproduce pinned PDAL byte-for-byte. The lower-only
budget control is internal, generic-only, and applied after direct-path and
rewrite-envelope checks. This result does not close the table's pending
shared-index neighborhood or adaptive-kNN tiling work.

Unsupported expression functions, type-widening ferry mappings, unproven
coordinate/layout semantics, multi-bounds or geometry/SRS crop variants,
the unsafe standard-decimation offset domain after an unknown-size stage,
option-rich or order-unstable locate variants, transformation matrix files,
inversion, SRS overrides, or projective CUDA forms,
option-rich robust filters or nonfinite/negative-zero robust CUDA keys,
statistical outlier requests with `mean_k` outside 0..63, too few points, or an
incomplete bounded kNN search, nonpositive/nonfinite radius grids, nonfinite
coordinates, or a radius tile beyond the current 32-bit compact-index envelope,
radialdensity conditional/invalid-option forms or radii outside the finite
positive device envelope,
nndistance requests with `k` outside 1..63, too few points, invalid modes,
conditional forms, or incomplete device searches,
normal-filter radius/viewpoint/refinement/conditional forms, eigenvalue radius
or strided forms, covariance-feature density/optimal/radius/parallel/strided
forms, their conditional forms, `knn` outside 2..63, unrepaired tied or
incomplete device neighborhoods, approximate-coplanar `knn` outside 3..64,
too-small views, conditional forms, or invalid option types, sort `where`
variants, NaN
CUDA keys, or sort normal/multi-key CUDA ties,
returns `where` variants, divider capacity/expression modes, splitter `where`
or unsafe numeric domains, colorinterp `where`/invalid-type forms or nonfinite
CUDA values, advanced/enumerate/count/global stats CUDA forms, stats `where`
or invalid-type forms, info point/query CUDA forms, nonfinite expressionstats
targets or predicates outside the exact VM, randomize `where`/invalid-type
forms, merge SRS overrides,
unstable reader order,
option-rich variants, or non-linear/tagged graphs remain on PDAL. Eligibility
is checked before point data or output is emitted.
Failures are part of compatibility: for example, a value-dependent PDAL ferry
conversion failure must not become a successful CUDA result.

## Active acceleration waves

D0099 pauses the former port-first queue. Work now follows complete-process
evidence in this order:

1. profile representative mixed pipelines; B0037 rejected the first
   terrain-to-feature sharing hypothesis, while B0038 finds exact host closure
   repair—not the resident endpoint—as the largest measured LOF cost;
2. prototype planner-owned selective device repair for measured
   ambiguous/incomplete neighborhood closures; B0039 rejects the current
   grid's tie order, so retain exact host repair and defer a new index until it
   is planner-owned and reusable. B0238 instead optimizes the already-pinned
   repair: one contiguous KD3 coordinate backing plus deterministic parallel
   passes reduce exact repair from 2.184 to 0.505 seconds without adding an
   index or changing result bits;
3. retain B0043's no-side-effect automatic resident LAS publication only for
   its exact measured LOF/default-LAS envelope. B0238 improves that unchanged
   public route to 18.301x pinned PDAL (18.551x option-free); the surviving
   spill remains too small to justify a device-column sink;
4. B0044's cheap reusable-endpoint sweep selected only nndistance from positive
   coplanar/normal/nndistance diagnostics. B0045 qualifies its repeated direct
   endpoint and exact public admission boundary. B0046 localizes more than 92%
   of both exact endpoint captures to rewritten manager execution and rejects
   the 51–52-millisecond publisher as the next target. B0047 places only
   0.50–0.51 seconds in reader/table work. B0048 rejects index and adjacent
   bridge work as dominant and localizes 2.234 of 2.507 resident seconds to
   nndistance's broad query/projection call. B0049 finds one 2.061-second exact
   host repair behind that aggregate, rejects pageable-status work, and rejects
   a slower full Morton-BVH substitution. B0050 replaces that exceptional
   repair with a bounded exact resident-coordinate scan and improves the clean
   automatic median from 3.390 to 1.413 seconds. B0051's disposable exact 4M
   probe hydrates mapped LAS into resident binary64 XYZ at a 0.00840-second
   median, plus 0.09870 seconds for fresh allocation, versus the measured
   0.58237-second reader/table-plus-upload ceiling. B0052 integrates that
   source only into the already-qualified endpoint and measures a clean exact
   automatic median of 0.91103 seconds versus 17.81584 seconds pinned PDAL
   (19.555713x), 35.54% below B0050 candidate wall. The source is therefore
   selected automatically inside the existing envelope; this improves its
   performance qualification without adding functional or GPU-native stage
   coverage. B0053 then profiles 218.521 milliseconds of total kernel work and
   rejects a 64/128/256-thread reusable-gather sweep because its best valid
   variant saves only 4.947 milliseconds without a material complete-process
   gain. B0054 then partitions 0.175985 seconds of
   validation/placement/preflight and finds 0.173186 seconds in necessary
   CUDA device/profile and initial memory-budget startup, not redundant PDAL
   validation; no prototype or coverage/selection change follows. B0055 then
   removes a measured redundant complete-coordinate kNN envelope scan, reducing
   the exact automatic median from 0.91103 to 0.82147 seconds; functional and
   GPU-native coverage and automatic selection remain unchanged while the
   existing endpoint's performance qualification improves. B0056 rejects a
   further exact extrema-reuse prototype because its 0.834/0.825/0.844-second
   process samples do not beat that qualified median. B0057 then attributes
   0.058527 seconds to direct-source hydration. B0058 finds only 0.003948
   seconds in its final wait and rejects D2H-only tuning. B0059's disposable
   record-summary scan reproduces the selected frame/backend bit-for-bit in
   0.00426759 seconds at 4M and 0.0148655 seconds at 16M, clearing a bounded
   production gate without changing coverage, qualification, or selection.
   B0060 then removes the direct endpoint's resident host XYZ mirror and host
   configuration scan. Its clean exact automatic median is 0.755523193 seconds
   versus 17.891433883 seconds pinned PDAL (23.680853x), improving the existing
   endpoint's performance qualification by 8.02744% from B0055 without
   changing functional support, GPU-native coverage, or automatic selection.
   B0061's PointView-only NNDistance publication elision is exact but reduces
   median wall by only 0.697339%, so it is reverted without changing any of the
   independent categories: functional support, GPU-native coverage,
   performance qualification, or automatic selection. B0062 then retains
   NNDistance only on device through the already-qualified direct assignment
   endpoint, removes 32,000,000 bytes H2D and 32,000,008 bytes D2H, and lowers
   the clean exact median from 0.755523193 to 0.712510475 seconds. This improves
   that endpoint's performance qualification and internal automatic route; it
   adds no functional or GPU-native stage coverage. B0063 then attributes the
   apparent status-call wall to the queued kNN gather and rejects pinned status
   storage without changing any of the four coverage categories. B0064
   replaces the measured one-row repair underfill only inside the same bounded
   automatic endpoint, improving its performance qualification to a
   28.812312x exact complete-pipeline speedup without changing functional
   support, GPU-native stage coverage, or the automatic-selection envelope.
   Generic resident NNDistance callers retain the serial repair. The fresh
   promoted profile leaves 97.3487% of device-kernel time in the reusable
   gather already tested by B0053, so the endpoint is sufficiently optimized
   for now;
5. extend planner-owned index, eigensystem, Grid, and output-column reuse only
   where a mixed profile proves that the avoided boundary is material;
6. qualify ordinary-data placement for already-positive forced/resident lanes;
7. retain negative standalone stages on host unless composition changes their
   complete-pipeline result; and
8. resume point, feature, terrain, clustering, geometry, raster, and I/O ports
   for required catalog coverage after the higher-value measured surfaces are
   exhausted.

Each new native entry requires option/error parity, an exact process
differential, schedule perturbation, sanitizer coverage, and a same-machine
break-even result before automatic selection. Functional availability never
waits for that port: unchanged PDAL remains the compatibility implementation.

## Reproducing the coverage gates

```sh
build/pdg-cuda-release/bin/gpupdal --drivers

ctest --test-dir build/pdg-host-debug --output-on-failure
ctest --test-dir build/pdg-cuda-debug --output-on-failure

cmake --build --preset pdal-upstream-tests
ctest --preset pdal-upstream-tests
```

The upstream preset is sequential because some official tests share fixture
paths. Its STAC/EPT/COPC cases require access to the public network fixtures.
See `docs/testing-strategy.md` for the complete differential and sanitizer
matrix.
