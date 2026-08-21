# Project origin and attribution

GPUPAL (GPU Pointcloud Abstraction Library) is an independent project derived
from the PDAL source tree. Its private-release bootstrap snapshot was based on
the local development branch at:

- source commit: `ff6f9346290d730bc4ff4ea962260c6d12e3038e`
- pinned compatibility oracle: the upstream PDAL revision recorded in
  `cmake/pdg-oracle.cmake`
- upstream project: <https://github.com/PDAL/PDAL>

The GPUPAL repository intentionally begins with a new root commit rather
than copying upstream Git history. This changes repository lineage, not source
provenance or licensing obligations.

PDAL-derived source remains subject to the copyright notices, BSD 3-Clause
conditions, disclaimer, and non-endorsement clause in `LICENSE.txt`. Detailed
compatibility-implementation attribution is retained in `NOTICE`. Bundled and
vendored components retain their own notices and license texts in their source
files and directories.

No affiliation with or endorsement by PDAL, Hobu, Inc., Flaxen Consulting LLC,
or PDAL contributors is implied.
