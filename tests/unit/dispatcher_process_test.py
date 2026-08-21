#!/usr/bin/env python3
"""Linux process-boundary checks for the thin sibling dispatcher."""
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile


PDG, ENGINE, ORACLE = map(pathlib.Path, sys.argv[1:4])


def run(path, pipeline, env=None):
    file = path.parent / "pipeline.json"
    file.write_text(pipeline)
    return subprocess.run([str(path), "pipeline", str(file)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env)


def wrapper(path, log, tag):
    path.write_text(
        "#!/bin/sh\n"
        "suffix=''\n"
        "if [ -n \"${PDG_LAZ_COMPRESSION_THREADS:-}\" ]; then\n"
        "  suffix=\"$suffix|threads=$PDG_LAZ_COMPRESSION_THREADS\"\n"
        "fi\n"
        "if [ -n \"${PDG_INTERNAL_FAST_MODE:-}\" ]; then\n"
        "  suffix=\"$suffix|fast=$PDG_INTERNAL_FAST_MODE\"\n"
        "fi\n"
        "printf '%s|%s%s\\n' '" + tag + "' \"$*\" \"$suffix\" >> '" +
        str(log) + "'\n"
        "exit 0\n")
    path.chmod(0o755)


with tempfile.TemporaryDirectory(prefix="pdg-dispatch-") as temp:
    root = pathlib.Path(temp)
    pdg = root / "pdg"
    shutil.copy2(PDG, pdg)
    log = root / "route.log"
    wrapper(root / "pdal", log, "oracle")
    wrapper(root / "pdg-engine", log, "engine")

    # `gpupdal verify` is a first-class public command.  It invokes the packaged
    # helper directly, before ambient PDG_* routing, and pins the candidate and
    # oracle arguments after all user-supplied options.
    verify_log = root / "verify.log"
    (root / "pdg-verify.py").write_text(
        "import os, pathlib, sys\n"
        "pathlib.Path(os.environ['PDG_VERIFY_TEST_LOG']).write_text("
        "'\\n'.join(sys.argv[1:]))\n"
        "raise SystemExit(37)\n")
    verify_environment = os.environ.copy()
    verify_environment["PDG_VERIFY_TEST_LOG"] = str(verify_log)
    verify_environment["PDG_FUTURE_UNLISTED_PROOF_CONTROL"] = "1"
    result = subprocess.run([str(pdg), "verify", "--output-dir", "proof"],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=verify_environment)
    assert result.returncode == 37, result
    verify_arguments = verify_log.read_text().splitlines()
    assert verify_arguments == ["--output-dir", "proof", "--candidate",
                                str(pdg), "--oracle", str(root / "pdal"),
                                "--oracle-source", "sibling-pdal"], \
        verify_arguments
    redirected = verify_environment.copy()
    redirected["PDG_ORACLE_PDAL"] = str(root / "configured-pdal")
    result = subprocess.run([str(pdg), "verify"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=redirected)
    assert result.returncode == 2 and "silently attest" in result.stderr
    result = subprocess.run([str(pdg), "--fast", "verify"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 2 and "default exact mode" in result.stderr

    opaque = '["input.e57", {"type":"filters.opaque_plugin"}, "output.las"]'
    result = run(pdg, opaque)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()

    log.unlink()
    candidate = ('{"pipeline":[{"type":"readers.las","filename":"in.las"},'
                 '{"type":"filters.assign"},'
                 '{"type":"writers.las","filename":"out.las"}]}')
    result = run(pdg, candidate)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0226: the measured four-stage r1 grammar goes directly to the oracle.
    log.unlink()
    r1_input = root / "in.laz"
    r1_header = bytearray(6_747_641)
    r1_header[0:4] = b"LASF"
    r1_header[24:26] = bytes((1, 4))
    struct.pack_into("<H", r1_header, 94, 375)
    struct.pack_into("<I", r1_header, 96, 475)
    r1_header[104] = 0x87
    struct.pack_into("<H", r1_header, 105, 36)
    struct.pack_into("<Q", r1_header, 247, 1_000_000)
    for offset in (131, 139, 147):
        struct.pack_into("<d", r1_header, offset, 0.01)
    for offset in (155, 163, 171):
        struct.pack_into("<d", r1_header, offset, 0.0)
    for offset, value in ((179, 185999.99), (187, 184500.0),
                          (195, 494999.99), (203, 494923.21),
                          (211, 500.41), (219, 367.44)):
        struct.pack_into("<d", r1_header, offset, value)
    r1_input.write_bytes(r1_header)
    r1 = ('{"pipeline":[{"type":"readers.las","filename":"' +
          str(r1_input) + '"},'
          '{"type":"filters.crop","bounds":'
          '"([184874.9975,185624.9925],[494942.405,494980.795])"},'
          '{"type":"filters.reprojection","in_srs":"EPSG:28992",'
          '"out_srs":"EPSG:3857"},'
          '{"type":"writers.las","filename":"out.laz",'
          '"compression":"true"}]}')
    result = run(pdg, r1)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()

    # The same grammar outside its calibrated 1M fact stays in-engine.
    log.unlink()
    struct.pack_into("<Q", r1_header, 247, 999_999)
    r1_input.write_bytes(r1_header)
    result = run(pdg, r1)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    struct.pack_into("<Q", r1_header, 247, 1_000_000)
    r1_input.write_bytes(r1_header)

    # A 1M header without the measured compressed format-7/36-byte layout
    # also fails closed.
    log.unlink()
    r1_header[104] = 0x07
    r1_input.write_bytes(r1_header)
    result = run(pdg, r1)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    r1_header[104] = 0x87
    r1_input.write_bytes(r1_header)

    # Every present PDG_* control except PDG_ORACLE_PDAL fails closed to the
    # engine, including current controls and names added in the future. Use an
    # otherwise direct r1 route so this proves environment precedence.
    log.unlink()
    skewness_requirement = os.environ.copy()
    skewness_requirement["PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT"] = "1"
    result = run(pdg, r1, skewness_requirement)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    future_requirement = os.environ.copy()
    future_requirement["PDG_FUTURE_UNLISTED_PROOF_CONTROL"] = "1"
    result = run(pdg, r1, future_requirement)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0243 rejects the exact-but-unproved r8 direct-delegation prototype.
    # Even the measured literal graph and layout remain engine-owned.
    log.unlink()
    r8 = ('{"pipeline":[{"type":"readers.las","filename":"' +
          str(r1_input) + '"},'
          '{"type":"filters.reprojection","in_srs":"EPSG:28992",'
          '"out_srs":"EPSG:3857"},'
          '{"type":"filters.colorization","raster":"orthophoto.tif",'
          '"dimensions":"Red:1:1.0, Green:2:1.0, Blue:3:1.0"},'
          '{"type":"filters.reprojection","in_srs":"EPSG:3857",'
          '"out_srs":"EPSG:28992"},'
          '{"type":"writers.las","filename":"r8-output.laz",'
          '"compression":"true"}]}')
    result = run(pdg, r8)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0245 rejects the warm-positive but cold-unresolved r13 reader override.
    # The literal graph remains engine-owned without probing local inputs.
    log.unlink()
    r13 = ('{"pipeline":[{"type":"readers.las",'
           '"filename":"merge-a.laz","tag":"left"},'
           '{"type":"readers.las","filename":"merge-b.laz",'
           '"tag":"right"},{"type":"filters.merge",'
           '"inputs":["left","right"]},{"type":"writers.las",'
           '"filename":"output.laz","compression":"true"}]}')
    result = run(pdg, r13)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0246 rejects the exact-but-unresolved r12 direct-delegation prototype.
    # The literal measured graph remains engine-owned.
    log.unlink()
    r12 = ('{"pipeline":[{"type":"readers.las","filename":"' +
           str(r1_input) + '"},{"type":"filters.splitter",'
           '"length":256.0,"origin_x":"184320.0",'
           '"origin_y":"494848.0","buffer":0.0},'
           '{"type":"writers.las",'
           '"filename":"output-tile-#.laz",'
           '"compression":"true"}]}')
    result = run(pdg, r12)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0247: corrected r9 is substantive, but both direct-exec and in-process
    # fallback prototypes resolved slower. The literal graph remains engine-owned.
    log.unlink()
    r9 = ('{"pipeline":[{"type":"readers.las","filename":"input.laz",'
          '"override_srs":"EPSG:28992"},{"type":"filters.crop",'
          '"polygon":"MULTIPOLYGON(((5.823268532898 52.441354258557,'
          '5.834298426905 52.441313140400,5.834304006406 52.441865170849,'
          '5.823273974811 52.441906289668,5.823268532898 52.441354258557),'
          '(5.826578874353 52.441480038643,5.829887853234 52.441467718545,'
          '5.829889924905 52.441674730066,5.826580930545 52.441687050237,'
          '5.826578874353 52.441480038643)),((5.836505103731 '
          '52.441373797364,5.840917059642 52.441356980476,5.840921306182 '
          '52.441771003009,5.836509308994 52.441787820100,5.836505103731 '
          '52.441373797364)))","a_srs":"EPSG:4326"},'
          '{"type":"writers.las","filename":"output.laz",'
          '"compression":"true"}]}')
    result = run(pdg, r9)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # CLI modifiers stay engine-owned even for the otherwise measured graph.
    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--stream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # Direct delegation preserves the configured oracle's exit and streams.
    log.unlink()
    configured_oracle = root / "configured-pdal"
    configured_oracle.write_text(
        "#!/bin/sh\n"
        "printf '%s|%s\\n' configured \"$*\" >> '" + str(log) + "'\n"
        "printf 'configured stdout\\n'\n"
        "printf 'configured stderr\\n' >&2\n"
        "exit 23\n")
    configured_oracle.chmod(0o755)
    configured = os.environ.copy()
    configured["PDG_ORACLE_PDAL"] = str(configured_oracle)
    result = run(pdg, r1, configured)
    assert result.returncode == 23, result
    assert result.stdout == "configured stdout\n", result.stdout
    assert result.stderr == "configured stderr\n", result.stderr
    assert log.read_text().startswith("configured|pipeline "), log.read_text()

    # Supported PDAL CLI modifiers are outside the measured route.
    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--stream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # Any engine-specific override still owns routing and diagnostics.
    log.unlink()
    override = os.environ.copy()
    override["PDG_DISABLE_HYBRID"] = "1"
    result = run(pdg, r1, override)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # Automatic neighborclassifier proof requirements are engine-owned even
    # when the pipeline text also matches a thin direct-oracle grammar.
    log.unlink()
    neighborclassifier_requirement = os.environ.copy()
    neighborclassifier_requirement[
        "PDG_REQUIRE_AUTOMATIC_NEIGHBORCLASSIFIER_RESIDENT"] = "1"
    result = run(pdg, r1, neighborclassifier_requirement)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # Automatic sort placement and post-execution exactness proof are also
    # engine-owned; the thin oracle fast path must never consume the route
    # requirement before the engine can prove it.
    log.unlink()
    sort_requirement = os.environ.copy()
    sort_requirement["PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT"] = "1"
    result = run(pdg, r1, sort_requirement)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # Automatic HAG-NN placement and post-execution exactness proof are also
    # engine-owned; the thin oracle fast path must never consume the route
    # requirement before the engine can prove it.
    log.unlink()
    hag_nn_requirement = os.environ.copy()
    hag_nn_requirement["PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT"] = "1"
    result = run(pdg, r1, hag_nn_requirement)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # Automatic HAG Delaunay placement and post-execution exactness proof are
    # also engine-owned; the thin oracle fast path must never consume the
    # route requirement before the engine can prove it.
    log.unlink()
    hag_delaunay_requirement = os.environ.copy()
    hag_delaunay_requirement[
        "PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT"] = "1"
    result = run(pdg, r1, hag_delaunay_requirement)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0228: the literal measured r5 COPC query bypasses an engine which would
    # only delegate it. The calibrated route covers only the plain invocation.
    log.unlink()
    r5 = ('{"pipeline":[{"type":"readers.copc",'
          '"filename":"input.copc.laz",'
          '"bounds":"([184874.9975,185624.9925],'
          '[494942.405,494980.795])","resolution":1.0,"requests":1},'
          '{"type":"filters.stats"},'
          '{"type":"writers.las","filename":"output.las"}]}')
    result = run(pdg, r5)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()

    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--stream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    drifted_r5 = r5.replace('"requests":1', '"requests":2')
    result = run(pdg, drifted_r5)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0229: the literal r3 DTM route discards every engine attempt and can
    # delegate directly only for the plain measured invocation.
    log.unlink()
    r3 = ('{"pipeline":[{"type":"readers.las","filename":"input.laz"},'
          '{"type":"filters.smrf"},'
          '{"type":"filters.range","limits":"Classification[2:2]"},'
          '{"type":"writers.gdal","filename":"output.tif",'
          '"resolution":1.0,"output_type":"idw"}]}')
    result = run(pdg, r3)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()

    log.unlink()
    r3_override = os.environ.copy()
    r3_override["PDG_DISABLE_HYBRID"] = "1"
    result = run(pdg, r3, r3_override)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--nostream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    drifted_r3 = r3.replace("Classification[2:2]", "Classification[1:1]")
    result = run(pdg, drifted_r3)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # B0248/B0254 contract: only the complete measured r7 DSM grammar bypasses
    # the engine, and only the measured 1M format-7 layout receives four
    # reader workers; raster/return policy and invocation drift remain
    # engine-owned.
    log.unlink()
    r7 = ('{"pipeline":[{"type":"readers.las","filename":"' +
          str(r1_input) + '",'
          '"override_srs":"EPSG:28992"},{"type":"filters.returns",'
          '"groups":"first,only"},{"type":"writers.gdal",'
          '"filename":"output.tif","resolution":1.0,'
          '"output_type":"max","dimension":"Z","binmode":true,'
          '"data_type":"float64","nodata":-9999.0,'
          '"bounds":"([184500,185999.99],[494923.21,494999.99])"}]}')
    result = run(pdg, r7)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" in log.read_text(), log.read_text()

    # Admission facts are checked at the real process boundary, not inferred
    # from the literal graph. Each neighboring header/file shape retains the
    # pre-existing direct-oracle route but must not receive the tuned reader.
    log.unlink()
    struct.pack_into("<Q", r1_header, 247, 999_999)
    r1_input.write_bytes(r1_header)
    result = run(pdg, r7)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()
    struct.pack_into("<Q", r1_header, 247, 1_000_000)

    log.unlink()
    r1_header[104] = 0x86
    r1_input.write_bytes(r1_header)
    result = run(pdg, r7)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()
    r1_header[104] = 0x87

    log.unlink()
    struct.pack_into("<H", r1_header, 105, 35)
    r1_input.write_bytes(r1_header)
    result = run(pdg, r7)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()
    struct.pack_into("<H", r1_header, 105, 36)

    log.unlink()
    r1_input.write_bytes(r1_header[:-1])
    result = run(pdg, r7)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()
    r1_input.write_bytes(r1_header)

    # The preload-only deterministic clock control is intentionally outside
    # the product PDG_* namespace and must not disable measured thin routes.
    log.unlink()
    r7_frozen_clock = os.environ.copy()
    r7_frozen_clock["PDAL_TEST_FROZEN_EPOCH"] = "1704067200"
    result = run(pdg, r7, r7_frozen_clock)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" in log.read_text(), log.read_text()

    log.unlink()
    r7_override = os.environ.copy()
    r7_override["PDG_DISABLE_HYBRID"] = "1"
    result = run(pdg, r7, r7_override)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()

    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--nostream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()

    log.unlink()
    drifted_r7 = r7.replace('"groups":"first,only"', '"groups":"last"')
    result = run(pdg, drifted_r7)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()

    # B0249/B0254 contract: only the literal r10 headline skips the engine,
    # but B0254's four-worker schedule remains unselected after its final cold
    # interval spans parity. Cell/writer policy, invocation, and
    # product-control drift remain engine-owned.
    log.unlink()
    r10 = ('{"pipeline":[{"type":"readers.las",'
           '"filename":"' + str(r1_input) + '"},'
           '{"type":"filters.voxelcentroidnearestneighbor","cell":2.5},'
           '{"type":"writers.las","filename":"output.laz",'
           '"compression":"true"}]}')
    result = run(pdg, r10)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()

    log.unlink()
    struct.pack_into("<Q", r1_header, 247, 999_999)
    r1_input.write_bytes(r1_header)
    result = run(pdg, r10)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()
    struct.pack_into("<Q", r1_header, 247, 1_000_000)
    r1_input.write_bytes(r1_header)

    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--nostream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()

    log.unlink()
    r10_override = os.environ.copy()
    r10_override["PDG_DISABLE_HYBRID"] = "1"
    result = run(pdg, r10, r10_override)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()

    log.unlink()
    drifted_r10 = r10.replace('"cell":2.5', '"cell":1.0')
    result = run(pdg, drifted_r10)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "--readers.las.threads=4" not in log.read_text(), log.read_text()

    # B0251/D0250, corrected by B0252/D0251: the exact 1M uncompressed
    # format-7 r14 headline arms two-worker lazperf in sibling PDAL.
    # Cardinality/layout/grammar drift refuses the
    # tuning while retaining the older generic direct-PDAL route. Modifiers
    # and externally supplied product controls stay engine-owned.
    log.unlink()
    r14_input = root / "r14-input.las"
    r14_header = bytearray(r1_header[:375])
    struct.pack_into("<I", r14_header, 96, 375)
    r14_header[104] = 0x07
    with r14_input.open("wb") as stream:
        stream.write(r14_header)
        stream.truncate(36_000_375)
    r14 = ('{"pipeline":[{"type":"readers.las",'
           '"filename":"' + str(r14_input) + '"},'
           '{"type":"writers.las","filename":"output.laz",'
           '"compression":"true"}]}')
    result = run(pdg, r14)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "|threads=2\n" in log.read_text(), log.read_text()

    log.unlink()
    struct.pack_into("<Q", r14_header, 247, 999_999)
    with r14_input.open("r+b") as stream:
        stream.write(r14_header)
    result = run(pdg, r14)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "|threads=" not in log.read_text(), log.read_text()
    struct.pack_into("<Q", r14_header, 247, 1_000_000)
    with r14_input.open("r+b") as stream:
        stream.write(r14_header)

    log.unlink()
    drifted_r14 = r14.replace('"compression":"true"',
                              '"compression":true')
    result = run(pdg, drifted_r14)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    assert "|threads=" not in log.read_text(), log.read_text()

    log.unlink()
    run(pdg, r14)
    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--nostream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    r14_override = os.environ.copy()
    r14_override["PDG_DISABLE_HYBRID"] = "1"
    result = run(pdg, r14, r14_override)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "|threads=" not in log.read_text(), log.read_text()

    log.unlink()
    r14_thread_override = os.environ.copy()
    r14_thread_override["PDG_LAZ_COMPRESSION_THREADS"] = "8"
    result = run(pdg, r14, r14_thread_override)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "|threads=" not in log.read_text(), log.read_text()

    # D0261: a leading --fast is consumed by the launcher, arms the internal
    # marker for whichever route the stripped command takes, and changes no
    # routing. An externally injected marker is removed before the engine.
    log.unlink()
    fast_file = root / "fast.json"
    fast_file.write_text(opaque)
    result = subprocess.run([str(pdg), "--fast", "pipeline", str(fast_file)],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text() == f"oracle|pipeline {fast_file}|fast=1\n", \
        log.read_text()
    log.unlink()
    fast_file.write_text(candidate)
    result = subprocess.run([str(pdg), "--fast", "pipeline", str(fast_file)],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text() == f"engine|pipeline {fast_file}|fast=1\n", \
        log.read_text()
    log.unlink()
    injected = os.environ.copy()
    injected["PDG_INTERNAL_FAST_MODE"] = "1"
    result = run(pdg, opaque, injected)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "|fast=" not in log.read_text(), log.read_text()
    # D0271: the flag also arms the marker on the environment-selected engine
    # route (an external PDG_* variable), where the injected-marker guard
    # must strip only the ambient value, not the consumed flag.
    log.unlink()
    fast_file.write_text(opaque)
    routed = os.environ.copy()
    routed["PDG_REQUIRE_HYBRID"] = "1"
    result = subprocess.run([str(pdg), "--fast", "pipeline", str(fast_file)],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=routed)
    assert result.returncode == 0, result.stderr
    assert log.read_text() == f"engine|pipeline {fast_file}|fast=1\n", \
        log.read_text()
    log.unlink()
    routed["PDG_INTERNAL_FAST_MODE"] = "1"
    result = run(pdg, opaque, routed)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()
    assert "|fast=" not in log.read_text(), log.read_text()

    # B0239: the literal r2 ground-normalization route selects the engine only
    # for the exact measured 1M reference layout.
    log.unlink()
    r2 = ('{"pipeline":[{"type":"readers.las","filename":"' +
          str(r1_input) + '"},'
          '{"type":"filters.smrf"},{"type":"filters.hag_nn"},'
          '{"type":"writers.las","filename":"output.laz",'
          '"compression":"true",'
          '"extra_dims":"HeightAboveGround=float32"}]}')
    result = run(pdg, r2)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    struct.pack_into("<Q", r1_header, 247, 999_999)
    r1_input.write_bytes(r1_header)
    result = run(pdg, r2)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("oracle|pipeline "), log.read_text()
    struct.pack_into("<Q", r1_header, 247, 1_000_000)
    r1_input.write_bytes(r1_header)

    log.unlink()
    r2_override = os.environ.copy()
    r2_override["PDG_DISABLE_HYBRID"] = "1"
    result = run(pdg, r2, r2_override)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--nostream"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    log.unlink()
    drifted_r2 = r2.replace("HeightAboveGround=float32",
                            "HeightAboveGround=float64")
    result = run(pdg, drifted_r2)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    # The explicit PDG-only resident command always belongs to the engine,
    # even when its pipeline would otherwise be opaque to the thin launcher.
    log.unlink()
    resident = root / "resident.json"
    resident.write_text(opaque)
    result = subprocess.run([str(pdg), "resident", str(resident), "--stats",
                             str(root / "stats.json")], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|resident "), log.read_text()

    # Ambiguous CLI options must remain in-engine rather than direct-oracle.
    log.unlink()
    result = subprocess.run([str(pdg), "pipeline", str(root / "pipeline.json"),
                             "--unknown"], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text().startswith("engine|pipeline "), log.read_text()

    (root / "pdg-engine").unlink()
    result = run(pdg, candidate)
    assert result.returncode == 127 and "unable to execute PDG engine" in result.stderr

    wrapper(root / "pdg-engine", log, "engine")
    (root / "pdal").chmod(0o644)
    result = run(pdg, opaque)
    assert result.returncode == 126 and "unable to execute pinned PDAL fallback" in result.stderr

with tempfile.TemporaryDirectory(prefix="pdg-engine-fallback-") as temp:
    root = pathlib.Path(temp)
    pdg = root / "pdg"
    engine = root / "pdg-engine"
    shutil.copy2(PDG, pdg)
    shutil.copy2(ENGINE, engine)
    log = root / "oracle.log"
    wrapper(root / "pdal", log, "oracle")
    pipeline = root / "unsupported.json"
    pipeline.write_text(
        '{"pipeline":[{"type":"readers.las","filename":"in.las"},'
        '{"type":"filters.assign"},'
        '{"type":"writers.las","filename":"out.las"}]}'
    )
    result = subprocess.run([str(pdg), "pipeline", str(pipeline), "--unknown"],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    assert log.read_text() == f"oracle|pipeline {pipeline} --unknown\n"
