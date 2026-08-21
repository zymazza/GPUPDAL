# Vast NVIDIA qualification protocol

Status: approved read-only protocol; not yet executed.

No Vast instance has been rented and no Vast spend has occurred. Credentials
remain only in the repository's ignored `.env` file with mode `0600`. Never
print that file, log a credential value, put a secret on a command line, enable
shell tracing while credentials are loaded, or copy credentials into a build,
artifact, ledger, issue, or CI log.

This protocol qualifies compiled NVIDIA targets; it does not assume that a
successful build is bit-exact on hardware. The first wave is deliberately
small, serialized, and independently stoppable.

## Authorization and hard limits

The user has authorized at most **$100** in total. The first wave has a hard
cap of **$23**, including compute and attached storage. The unused portion of
the $100 authorization is not permission to exceed the first-wave cap. Stop
and obtain an explicit second-wave decision before spending more than $23.

Only one rented instance may exist at a time, and only one heavy lane may run
at a time across the local and rented machines. CUDA compilation, sanitizer
runs, corpus hashing, and benchmarks are all heavy lanes. Every rented row has
both a two-hour wall-clock deadline and its dollar cap; reaching either limit
ends the row immediately.

| Row | Exact target for this wave | Toolkit | Hard cap |
| --- | --- | --- | ---: |
| Local control | Local RTX 4090, SM 89 | Current CUDA 13.3 | $0 |
| Legacy current-toolkit floor | NVIDIA T4, SM 75 | CUDA 13.3 | $3 |
| Ampere datacenter | NVIDIA A100, SM 80 | CUDA 13.3 | $5 |
| Ampere consumer or inference | Exactly one of RTX 3090 or A10, SM 86 | CUDA 13.3 | $4 |
| Hopper | NVIDIA H100, SM 90 | CUDA 13.3 | $8 |
| Optional legacy artifact | NVIDIA P40, SM 61 | CUDA 12.x | $3 |

The maximum is $20 without the optional P40 row and $23 with it. The SM-86 row
qualifies the exact model actually rented—RTX 3090 **or** A10—not both. Never
infer an A10 result from an RTX 3090 result, or the reverse. Record exact model
and memory variant for A100 and H100 as well.

## Profile identity

Qualification and automatic-selection profiles are keyed by all of:

- exact normalized NVIDIA model name;
- compute capability major and minor; and
- CUDA toolkit major and minor used to build the artifact.

Driver, VBIOS, VRAM size, OS image, compiler, CCCL version, and whether native
SASS or PTX JIT executed are evidence fields even when they are not selector
keys. Evidence for one key does not qualify another model, SM, toolkit, or JIT
path. CUDA 13.3 compile coverage for SM 75–121 plus newest-target PTX remains
compile portability only. The P40 row must use the separately maintained CUDA
12.x artifact because CUDA 13 does not provide an SM-61 offline target.

## Preflight and resource gates

Complete these checks before starting the two-hour clock for a row where the
provider permits, and immediately after login otherwise:

1. Confirm no other Vast instance exists and no local heavy lane is running.
2. Resolve an offer whose displayed exact model, SM, VRAM, hourly price,
   storage price, and image satisfy the row. Do not substitute a nearby model.
3. Calculate the maximum possible charge through the two-hour deadline,
   including storage. Reject an offer that can exceed the row cap.
4. Record the repository revision, pinned PDAL oracle revision, toolchain, and
   public-fixture hashes before work begins.
5. Confirm `nvidia-smi`, driver/toolkit compatibility, writable scratch space,
   and adequate host RAM and VRAM. Do not start a heavy lane below 8 GiB
   `MemAvailable`, while swap use is growing materially, or when the bounded
   fixture plus build cannot fit in available disk/VRAM.
6. Use one CUDA compile job (`--parallel 1`), sequential tests, bounded inputs,
   and no background benchmark, sanitizer, hash, or compile work.
7. Set a hard teardown alarm for two hours after instance creation and a cost
   alarm below the row cap. A failed prerequisite ends the row; do not consume
   the remaining budget trying unrelated images or models.

Public fixture inputs are read-only. Derived outputs, build trees, and logs
belong on instance-local scratch. Do not upload private corpora. The default
public pair is the named Autzen and RMIT fixtures after their redistributable
source, license, size, and SHA-256 values are entered in the ledger; any
substitution must be named and justified before the lease begins.

## Mandatory gates for every row

A row is `pass` only if every applicable gate below passes. Otherwise record
`fail` or `incomplete`; never convert missing evidence into an inferred pass.

### 1. Build gate

Build the pinned revision with the intended CUDA artifact and exact-mode
compiler flags. The target architecture must compile in the product
translation units, including `NeighborhoodKernels.cu`. Record the configuration
and artifact hashes. A compile success alone does not qualify runtime support.

### 2. Native-SASS gate

Inspect the resulting neighborhood-kernel object or linked artifact and prove
that it contains a native cubin for the row's exact SM. Record the inspection
result without embedding a binary dump in the ledger. A PTX-only artifact does
not pass this gate.

### 3. PTX gate

Run the compact exact lane once through native SASS and once with compatible
PTX JIT forced. Record which PTX target and driver JIT path were used. Both
paths must produce the same oracle-visible bits. If the row cannot legally JIT
the embedded PTX, mark the PTX gate `not-applicable` with the compiler/driver
reason; do not call it passed.

### 4. Exactness gate

Run the compact fixed-bit architecture lane and the complete
approximate-coplanar process differential. Compare output bytes, metadata,
point order, stdout, stderr, and exit status with the pinned PDAL oracle. A
failure must record the first differing byte and decoded LAS field. Do not
relax tolerances or regenerate goldens.

### 5. Repeatability gate

Repeat the exact automatic-path fixture at least five consecutive times in
fresh processes. Every candidate artifact and diagnostic must match the oracle
and the other repeats. Any intermittent difference fails the row.

### 6. Sanitizer gate

Run the bounded architecture and approximate-coplanar lanes under Compute
Sanitizer memcheck, initcheck, racecheck, and synccheck, one tool at a time.
Zero errors and zero race hazards are required. Preserve the existing
test-specific proof guards; do not disable device security settings to obtain
hardware counters.

### 7. Default-selection benchmark gate

The primary benchmark must be an ordinary option-free `gpupdal pipeline`
invocation: no force, require, experimental, backend, or selector environment
variable may choose CUDA. Use the pinned PDAL process as the same-machine
baseline, frozen observable time, two warmups, and at least ten alternating
samples. Compare the complete output and diagnostics on every invocation.
Configure the clean qualification artifact with
`PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR=ON`; ordinary production artifacts
leave it off until this protocol passes. The provisional D0049 benchmark shape
is exactly `readers.las` → `filters.approximatecoplanar` → `writers.las`, with
uncompressed `.las` output and `extra_dims=all`.

After—not during—the timed option-free samples, use
`PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA=1` in a separate proof process
to confirm that the default rewrite and CUDA execution occurred. Run the
262,143-point control with the exact-host-fallback proof and the 262,144-point
boundary in LAS, LAZ, and LAS point format 8. A profile is not qualified if a
forced run is fast but the ordinary process remains host-selected.

## Performance acceptance rule

Automatic selection is a performance promise. For an exact model+SM+toolkit
profile, retain results from at least two physical hosts and two named public
fixtures. In every host-by-fixture cell:

- the worst candidate/baseline median speedup must be at least **1.25x**; and
- the candidate elapsed-time p75 must be strictly less than the pinned-PDAL
  elapsed-time p25.

Both conditions are mandatory. Pooling samples across hosts, hiding a slow
fixture in an aggregate, substituting a different GPU model, or using a forced
CUDA timing is prohibited. If the first wave supplies only one host for a
profile, it can reject that profile or establish provisional evidence, but it
cannot promote it automatically.

## Secret-free append-only ledger

Keep an append-only JSONL or equivalent ledger outside tracked source until it
has been reviewed for secrets. Each row contains only non-secret evidence:

- protocol version, row id, status, UTC start/stop times, and stop reason;
- authorized row cap, quoted compute/storage rates, final billed amount, and
  remaining first-wave budget;
- provider offer id and a non-secret host label—never an API key, token,
  authorization header, cookie, SSH private key, or `.env` value;
- repository revision, oracle revision, dirty/clean status, build preset and
  configuration, compiler, toolkit, CCCL, driver, OS image, and artifact hash;
- exact GPU model, PCI/device identifier if non-secret, SM, VRAM, native-cubin
  target, PTX target, and observed native-versus-JIT path;
- fixture name, public source/license reference, size, input SHA-256, point
  format, and point count;
- gate status for build, SASS, PTX, exactness, repeatability, each sanitizer,
  default selection, fallback controls, and teardown;
- output hashes, first difference details when failing, sample counts, raw
  elapsed times, medians, p25/p75, and worst-cell speedup; and
- destroy request time, destroy confirmation time, storage enumeration after
  destroy, remaining resource count, and cleanup verification status.

Commands may be recorded only after removing credentials and ephemeral secret
paths. Never run or capture `env`, `printenv`, shell xtrace, provider debug HTTP
logging, or `.env` contents. Redaction after logging is not an acceptable
credential-control strategy.

## Mandatory teardown and storage verification

Teardown is part of every row, including failed, timed-out, and manually
aborted rows:

1. Stop benchmark/test processes and collect only the secret-free ledger and
   intended evidence.
2. Issue the provider destroy operation for the instance; stopping it is not
   sufficient.
3. Poll the provider control plane until the instance is absent or explicitly
   destroyed.
4. Enumerate attached and detached storage, delete row-created volumes and
   snapshots, and verify that no billable storage remains.
5. Verify that there are zero active instances, zero row-created volumes, and
   no continuing hourly charge before starting another row.
6. Record destroy and cleanup confirmation plus the final billed amount in the
   ledger. If verification is unavailable, treat the row as still active,
   start no new row, and escalate immediately.

Ephemeral instance SSH material may be removed after evidence transfer;
long-lived user keys and the local ignored `.env` are not uploaded or modified
by this protocol. Completion means both technical gates and billing cleanup
are verified—not merely that the remote shell disconnected.

## First-wave exit

Stop the first wave when all selected rows have a terminal status, the $23 cap
would be reached, or any cleanup cannot be verified. Summarize passes,
failures, incomplete two-host/two-fixture requirements, exact model-specific
limits, spend, and verified zero remaining resources. Only then may a separate
decision propose selector profiles or a second wave within the outer $100 user
authorization.
