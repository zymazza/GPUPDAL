#!/usr/bin/env python3
"""Exact process matrix for the bounded filters.hag_nn count-one through 64 lanes.

Counts one through seven are covered by explicit hand-written cases. The wider
range is covered by cases generated over count (D0203): the lane is a proof
ladder rather than a cost or implementation ladder, so the same obligations are
generated per count instead of transcribed per count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    stage: dict[str, object]
    hybrid: bool = True
    cuda: bool = True
    verbose: bool = False
    backend: str | None = None
    cuda_required: bool = True
    proof: str | None = None


CASES = (
    Case("defaults", "unique", {"type": "filters.hag_nn"}),
    Case("forced-uniform-grid", "unique", {"type": "filters.hag_nn"},
         backend="grid"),
    Case("forced-morton-bvh", "unique", {"type": "filters.hag_nn"},
         backend="bvh"),
    Case("supported-empty", "empty", {"type": "filters.hag_nn"}),
    Case("custom-ground-class", "custom-class", {
        "type": "filters.hag_nn", "class": 9,
    }),
    Case("same-xy-ground", "same-xy", {"type": "filters.hag_nn"}),
    Case("one-ground", "one-ground", {"type": "filters.hag_nn"}),
    Case("all-ground", "all-ground", {"type": "filters.hag_nn"}),
    Case("ignored-count-one-options", "unique", {
        "type": "filters.hag_nn", "count": 1, "max_distance": -17.5,
        "allow_extrapolation": False,
    }),
    Case("equal-distance-tie-host-repair", "tie",
         {"type": "filters.hag_nn"}, cuda=False),
    Case("equal-distance-tie-uniform-grid", "tie",
         {"type": "filters.hag_nn"}, backend="grid", cuda_required=False,
         proof="tie"),
    Case("equal-distance-tie-morton-bvh", "tie",
         {"type": "filters.hag_nn"}, backend="bvh", cuda_required=False,
         proof="tie"),
    Case("no-ground-diagnostic", "no-ground",
         {"type": "filters.hag_nn"}, cuda=False, verbose=True),
    Case("nonfinite-xy-host-fallback", "nonfinite-xy",
         {"type": "filters.hag_nn"}, cuda=False),
    Case("count-two-interpolation", "interpolation", {
        "type": "filters.hag_nn", "count": 2, "max_distance": 100.0,
        "allow_extrapolation": True,
    }),
    Case("count-two-no-extrapolation", "outside", {
        "type": "filters.hag_nn", "count": 2,
        "allow_extrapolation": False,
    }),
    Case("count-two-one-ground", "one-ground", {
        "type": "filters.hag_nn", "count": 2,
    }, cuda_required=False, proof="insufficient"),
    Case("count-two-same-xy", "same-xy", {
        "type": "filters.hag_nn", "count": 2,
    }),
    Case("count-two-max-distance-none", "interpolation", {
        "type": "filters.hag_nn", "count": 2, "max_distance": 0.5,
    }),
    Case("count-two-negative-max-distance", "interpolation", {
        "type": "filters.hag_nn", "count": 2, "max_distance": -2.0,
    }),
    Case("count-two-max-distance-equality", "max-boundary", {
        "type": "filters.hag_nn", "count": 2, "max_distance": 2.0,
    }),
    Case("count-two-inclusive-ground-bounds", "bounds-edge", {
        "type": "filters.hag_nn", "count": 2,
        "allow_extrapolation": False,
    }),
    Case("count-three-interpolation", "count-three-interpolation", {
        "type": "filters.hag_nn", "count": 3, "max_distance": 100.0,
        "allow_extrapolation": True,
    }),
    Case("count-three-uniform-grid-unique", "count-three-unique", {
        "type": "filters.hag_nn", "count": 3,
    },
         backend="grid", cuda_required=True),
    Case("count-three-bvh-unique", "count-three-unique", {
        "type": "filters.hag_nn", "count": 3,
    }, backend="bvh", cuda_required=True),
    Case("count-three-same-xy", "same-xy-count-three", {
        "type": "filters.hag_nn", "count": 3,
    }),
    Case("count-three-max-distance-equality", "count-three-boundary", {
        "type": "filters.hag_nn", "count": 3,
        "max_distance": 2.0,
    }),
    Case("count-three-max-distance-partial-cutoff", "count-three-boundary", {
        "type": "filters.hag_nn", "count": 3,
        "max_distance": 1.25,
    }),
    Case("count-three-negative-max-distance", "count-three-boundary", {
        "type": "filters.hag_nn", "count": 3,
        "max_distance": -2.0,
    }),
    Case("count-three-no-extrapolation", "count-three-outside", {
        "type": "filters.hag_nn", "count": 3,
        "allow_extrapolation": False,
    }),
    Case("count-three-large-finite-coordinates", "count-three-large-finite", {
        "type": "filters.hag_nn", "count": 3,
    }),
    Case("count-three-underflow", "count-three-underflow", {
        "type": "filters.hag_nn", "count": 3,
    }),
    Case("count-three-signed-zero", "count-three-signed-zero", {
        "type": "filters.hag_nn", "count": 3,
    }),
    Case("count-three-inclusive-ground-bounds", "count-three-bounds-edge", {
        "type": "filters.hag_nn", "count": 3,
        "allow_extrapolation": False,
    }),
    Case("count-three-insufficient-one-ground", "one-ground",
         {"type": "filters.hag_nn", "count": 3,
          }, cuda_required=False,
         proof="insufficient"),
    Case("count-three-insufficient-two-grounds", "two-grounds",
         {"type": "filters.hag_nn", "count": 3,
          }, cuda_required=False,
         proof="insufficient"),
    Case("count-three-no-ground-diagnostic", "no-ground", {
        "type": "filters.hag_nn", "count": 3,
    }, verbose=True, cuda_required=False),
    Case("count-four-interpolation", "count-four-interpolation", {
        "type": "filters.hag_nn", "count": 4, "max_distance": 100.0,
        "allow_extrapolation": True,
    }),
    Case("count-four-uniform-grid-unique", "count-four-unique", {
        "type": "filters.hag_nn", "count": 4,
    }, backend="grid", cuda_required=True),
    Case("count-four-bvh-unique", "count-four-unique", {
        "type": "filters.hag_nn", "count": 4,
    }, backend="bvh", cuda_required=True),
    Case("count-four-same-xy", "same-xy-count-four", {
        "type": "filters.hag_nn", "count": 4,
    }),
    Case("count-four-max-distance-equality", "count-four-boundary", {
        "type": "filters.hag_nn", "count": 4,
        "max_distance": 4.0,
    }),
    Case("count-four-max-distance-partial-cutoff", "count-four-boundary", {
        "type": "filters.hag_nn", "count": 4,
        "max_distance": 2.5,
    }),
    Case("count-four-negative-max-distance", "count-four-boundary", {
        "type": "filters.hag_nn", "count": 4,
        "max_distance": -4.0,
    }),
    Case("count-four-no-extrapolation", "count-four-outside", {
        "type": "filters.hag_nn", "count": 4,
        "allow_extrapolation": False,
    }),
    Case("count-four-large-finite-coordinates", "count-four-large-finite", {
        "type": "filters.hag_nn", "count": 4,
    }),
    Case("count-four-underflow", "count-four-underflow", {
        "type": "filters.hag_nn", "count": 4,
    }),
    Case("count-four-signed-zero", "count-four-signed-zero", {
        "type": "filters.hag_nn", "count": 4,
    }),
    Case("count-four-inclusive-ground-bounds", "count-four-bounds-edge", {
        "type": "filters.hag_nn", "count": 4,
        "allow_extrapolation": False,
    }),
    Case("count-four-insufficient-one-ground", "one-ground", {
        "type": "filters.hag_nn", "count": 4,
    }, cuda_required=False, proof="insufficient"),
    Case("count-four-insufficient-three-grounds", "three-grounds", {
        "type": "filters.hag_nn", "count": 4,
    }, cuda_required=False, proof="insufficient"),
    Case("count-four-no-ground-diagnostic", "no-ground", {
        "type": "filters.hag_nn", "count": 4,
    }, verbose=True, cuda_required=False),
    Case("count-two-nonfinite-z-host-repair", "nonfinite-z", {
        "type": "filters.hag_nn", "count": 2,
    }, cuda_required=False, proof="nonfinite-z"),
    Case("count-three-nonfinite-z-host-repair", "count-three-nonfinite-z", {
        "type": "filters.hag_nn", "count": 3,
    }, cuda_required=False, proof="nonfinite-z"),
    Case("count-three-nonfinite-xy-host-repair", "nonfinite-xy", {
        "type": "filters.hag_nn", "count": 3,
    }, cuda_required=False),
    Case("count-four-nonfinite-z-host-repair", "count-four-nonfinite-z", {
        "type": "filters.hag_nn", "count": 4,
    }, cuda_required=False, proof="nonfinite-z"),
    Case("count-four-nonfinite-xy-host-repair", "nonfinite-xy", {
        "type": "filters.hag_nn", "count": 4,
    }, cuda_required=False),
    Case("count-two-tie-host-repair", "tie", {
        "type": "filters.hag_nn", "count": 2,
    }, cuda_required=False, proof="tie"),
    Case("count-three-third-fourth-boundary-tie-uniform-grid", "third-fourth-boundary-tie", {
        "type": "filters.hag_nn", "count": 3,
    }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-three-third-fourth-boundary-tie-morton-bvh", "third-fourth-boundary-tie", {
        "type": "filters.hag_nn", "count": 3,
    }, backend="bvh", cuda_required=False, proof="tie"),
    Case("count-three-distance-overflow-uniform-grid", "distance-overflow", {
        "type": "filters.hag_nn", "count": 3,
    }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-four-fourth-fifth-boundary-tie-uniform-grid",
         "fourth-fifth-boundary-tie", {
             "type": "filters.hag_nn", "count": 4,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-four-fourth-fifth-boundary-tie-morton-bvh",
         "fourth-fifth-boundary-tie", {
             "type": "filters.hag_nn", "count": 4,
         }, backend="bvh", cuda_required=False, proof="tie"),
    Case("count-four-distance-overflow-uniform-grid",
         "count-four-distance-overflow", {
             "type": "filters.hag_nn", "count": 4,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-five-uniform-grid-unique", "count-five-unique", {
        "type": "filters.hag_nn", "count": 5,
    }, backend="grid", cuda_required=True),
    Case("count-five-bvh-unique", "count-five-unique", {
        "type": "filters.hag_nn", "count": 5,
    }, backend="bvh", cuda_required=True),
    Case("count-five-interpolation", "count-five-interpolation", {
        "type": "filters.hag_nn", "count": 5, "max_distance": 100.0,
        "allow_extrapolation": True,
    }),
    Case("count-five-max-distance-equality", "count-five-boundary", {
        "type": "filters.hag_nn", "count": 5,
        "max_distance": 5.0,
    }),
    Case("count-five-max-distance-partial-cutoff", "count-five-boundary", {
        "type": "filters.hag_nn", "count": 5,
        "max_distance": 2.5,
    }),
    Case("count-five-negative-max-distance", "count-five-boundary", {
        "type": "filters.hag_nn", "count": 5,
        "max_distance": -5.0,
    }),
    Case("count-five-no-extrapolation", "count-five-outside", {
        "type": "filters.hag_nn", "count": 5,
        "allow_extrapolation": False,
    }),
    Case("count-five-large-finite-coordinates", "count-five-large-finite", {
        "type": "filters.hag_nn", "count": 5,
    }),
    Case("count-five-underflow", "count-five-underflow", {
        "type": "filters.hag_nn", "count": 5,
    }),
    Case("count-five-signed-zero", "count-five-signed-zero", {
        "type": "filters.hag_nn", "count": 5,
    }),
    Case("count-five-inclusive-ground-bounds", "count-five-bounds-edge", {
        "type": "filters.hag_nn", "count": 5,
        "allow_extrapolation": False,
    }),
    Case("count-five-insufficient-one-ground", "one-ground", {
        "type": "filters.hag_nn", "count": 5,
    }, cuda_required=False, proof="insufficient"),
    Case("count-five-nonfinite-z-host-repair", "count-five-nonfinite-z", {
        "type": "filters.hag_nn", "count": 5,
    }, cuda_required=False, proof="nonfinite-z"),
    Case("count-five-nonfinite-xy-host-repair", "count-five-nonfinite-xy", {
        "type": "filters.hag_nn", "count": 5,
    }, cuda_required=False),
    Case("count-five-fifth-sixth-boundary-tie-uniform-grid",
         "fifth-sixth-boundary-tie", {
             "type": "filters.hag_nn", "count": 5,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-five-fifth-sixth-boundary-tie-morton-bvh",
         "fifth-sixth-boundary-tie", {
             "type": "filters.hag_nn", "count": 5,
         }, backend="bvh", cuda_required=False, proof="tie"),
    Case("count-five-distance-overflow-uniform-grid",
         "count-five-distance-overflow", {
             "type": "filters.hag_nn", "count": 5,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-five-incomplete-grid", "count-five-incomplete", {
        "type": "filters.hag_nn", "count": 5,
    }, backend="grid", cuda_required=False, proof="incomplete"),
    Case("count-six-uniform-grid-unique", "count-six-unique", {
        "type": "filters.hag_nn", "count": 6,
    }, backend="grid", cuda_required=True),
    Case("count-six-bvh-unique", "count-six-unique", {
        "type": "filters.hag_nn", "count": 6,
    }, backend="bvh", cuda_required=True),
    Case("count-six-interpolation", "count-six-interpolation", {
        "type": "filters.hag_nn", "count": 6, "max_distance": 100.0,
        "allow_extrapolation": True,
    }),
    Case("count-six-max-distance-equality", "count-six-boundary", {
        "type": "filters.hag_nn", "count": 6,
        "max_distance": 6.0,
    }),
    Case("count-six-max-distance-partial-cutoff", "count-six-boundary", {
        "type": "filters.hag_nn", "count": 6,
        "max_distance": 3.0,
    }),
    Case("count-six-negative-max-distance", "count-six-boundary", {
        "type": "filters.hag_nn", "count": 6,
        "max_distance": -6.0,
    }),
    Case("count-six-no-extrapolation", "count-six-outside", {
        "type": "filters.hag_nn", "count": 6,
        "allow_extrapolation": False,
    }),
    Case("count-six-large-finite-coordinates", "count-six-large-finite", {
        "type": "filters.hag_nn", "count": 6,
    }),
    Case("count-six-underflow", "count-six-underflow", {
        "type": "filters.hag_nn", "count": 6,
    }),
    Case("count-six-signed-zero", "count-six-signed-zero", {
        "type": "filters.hag_nn", "count": 6,
    }),
    Case("count-six-inclusive-ground-bounds", "count-six-bounds-edge", {
        "type": "filters.hag_nn", "count": 6,
        "allow_extrapolation": False,
    }),
    Case("count-six-insufficient-one-ground", "one-ground", {
        "type": "filters.hag_nn", "count": 6,
    }, cuda_required=False, proof="insufficient"),
    Case("count-six-nonfinite-z-host-repair", "count-six-nonfinite-z", {
        "type": "filters.hag_nn", "count": 6,
    }, cuda_required=False, proof="nonfinite-z"),
    Case("count-six-nonfinite-xy-fallback", "count-six-nonfinite-xy", {
        "type": "filters.hag_nn", "count": 6,
    }, cuda_required=False),
    Case("count-six-sixth-seventh-boundary-tie-uniform-grid",
         "sixth-seventh-boundary-tie", {
             "type": "filters.hag_nn", "count": 6,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-six-sixth-seventh-boundary-tie-morton-bvh",
         "sixth-seventh-boundary-tie", {
             "type": "filters.hag_nn", "count": 6,
         }, backend="bvh", cuda_required=False, proof="tie"),
    Case("count-six-distance-overflow-uniform-grid",
         "count-six-distance-overflow", {
             "type": "filters.hag_nn", "count": 6,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-six-incomplete-grid", "count-six-incomplete", {
        "type": "filters.hag_nn", "count": 6,
    }, backend="grid", cuda_required=False, proof="incomplete"),
    Case("count-seven-uniform-grid-unique", "count-seven-unique", {
        "type": "filters.hag_nn", "count": 7,
    }, backend="grid", cuda_required=True),
    Case("count-seven-bvh-unique", "count-seven-unique", {
        "type": "filters.hag_nn", "count": 7,
    }, backend="bvh", cuda_required=True),
    Case("count-seven-interpolation", "count-seven-interpolation", {
        "type": "filters.hag_nn", "count": 7, "max_distance": 100.0,
        "allow_extrapolation": True,
    }),
    Case("count-seven-max-distance-equality", "count-seven-boundary", {
        "type": "filters.hag_nn", "count": 7,
        "max_distance": 7.0,
    }),
    Case("count-seven-max-distance-partial-cutoff", "count-seven-boundary", {
        "type": "filters.hag_nn", "count": 7,
        "max_distance": 3.0,
    }),
    Case("count-seven-negative-max-distance", "count-seven-boundary", {
        "type": "filters.hag_nn", "count": 7,
        "max_distance": -7.0,
    }),
    Case("count-seven-no-extrapolation", "count-seven-outside", {
        "type": "filters.hag_nn", "count": 7,
        "allow_extrapolation": False,
    }),
    Case("count-seven-large-finite-coordinates", "count-seven-large-finite", {
        "type": "filters.hag_nn", "count": 7,
    }),
    Case("count-seven-underflow", "count-seven-underflow", {
        "type": "filters.hag_nn", "count": 7,
    }),
    Case("count-seven-signed-zero", "count-seven-signed-zero", {
        "type": "filters.hag_nn", "count": 7,
    }),
    Case("count-seven-inclusive-ground-bounds", "count-seven-bounds-edge", {
        "type": "filters.hag_nn", "count": 7,
        "allow_extrapolation": False,
    }),
    Case("count-seven-insufficient-one-ground", "one-ground", {
        "type": "filters.hag_nn", "count": 7,
    }, cuda_required=False, proof="insufficient"),
    Case("count-seven-nonfinite-z-host-repair", "count-seven-nonfinite-z", {
        "type": "filters.hag_nn", "count": 7,
    }, cuda_required=False, proof="nonfinite-z"),
    Case("count-seven-nonfinite-xy-fallback", "count-seven-nonfinite-xy", {
        "type": "filters.hag_nn", "count": 7,
    }, cuda_required=False),
    Case("count-seven-seventh-eighth-boundary-tie-uniform-grid",
         "seventh-eighth-boundary-tie", {
             "type": "filters.hag_nn", "count": 7,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-seven-seventh-eighth-boundary-tie-morton-bvh",
         "seventh-eighth-boundary-tie", {
             "type": "filters.hag_nn", "count": 7,
         }, backend="bvh", cuda_required=False, proof="tie"),
    Case("count-seven-distance-overflow-uniform-grid",
         "count-seven-distance-overflow", {
             "type": "filters.hag_nn", "count": 7,
         }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-seven-incomplete-grid", "count-seven-incomplete", {
        "type": "filters.hag_nn", "count": 7,
    }, backend="grid", cuda_required=False, proof="incomplete"),
    Case("count-two-boundary-tie-uniform-grid", "boundary-tie", {
        "type": "filters.hag_nn", "count": 2,
    }, backend="grid", cuda_required=False, proof="tie"),
    Case("count-two-boundary-tie-morton-bvh", "boundary-tie", {
        "type": "filters.hag_nn", "count": 2,
    }, backend="bvh", cuda_required=False, proof="tie"),
    Case("count-two-incomplete-grid", "incomplete-two", {
        "type": "filters.hag_nn", "count": 2,
    }, backend="grid", cuda_required=False, proof="incomplete"),
    Case("count-three-incomplete-grid", "count-three-incomplete", {
        "type": "filters.hag_nn", "count": 3,
    }, backend="grid", cuda_required=False, proof="incomplete"),
    Case("count-four-incomplete-grid", "count-four-incomplete", {
        "type": "filters.hag_nn", "count": 4,
    }, backend="grid", cuda_required=False, proof="incomplete"),
    Case("count-sixty-five-host-fallback", "count-sixty-five-interpolation", {
        "type": "filters.hag_nn", "count": 65,
    }, hybrid=False, cuda=False),
    Case("where-fallback", "unique", {
        "type": "filters.hag_nn", "where": "Classification != 2",
    }, hybrid=False, cuda=False),
    Case("count-zero-error", "unique", {
        "type": "filters.hag_nn", "count": 0,
    }, hybrid=False, cuda=False),
    Case("invalid-class-error", "unique", {
        "type": "filters.hag_nn", "class": 256,
    }, hybrid=False, cuda=False),
    Case("missing-classification-error", "missing-classification",
         {"type": "filters.hag_nn"}, hybrid=False, cuda=False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--oracle-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--candidate-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


# --- Count-generated coverage (D0203) -------------------------------------
#
# `GENERATED_COUNTS` spans the admissible range up to the shared index's own
# `maximumNeighbors` cap of 64. Each shape below states the property it proves
# and constructs rows so that property holds for any count in the range; the
# fixture kind carries the count so `rows()` can regenerate it.

GENERATED_COUNTS = (8, 16, 32, 64)


def generated_rows(shape: str, k: int) -> list[tuple[float, float, float, int]]:
    """Rows proving `shape` for exactly this count.

    Ground rows are class 2 and query rows class 1. Every shape keeps all
    retained candidate distances distinct unless it is deliberately proving a
    tie, so a native-required case never silently exercises host repair.
    """
    if shape == "unique":
        # k + 1 grounds at even X, query at 0.5: distances 0.5, 1.5, 3.5, ...
        # are strictly increasing, so the (k + 1)-th candidate is distinct.
        rows = [(2.0 * i, 0.0, 10.0 + i, 2) for i in range(k + 1)]
        return rows + [(0.5, 0.0, 500.0, 1)]
    if shape == "interpolation":
        # Exactly k grounds, so every retained candidate is used.
        rows = [(2.0 * i, 0.0, 10.0 + i, 2) for i in range(k)]
        return rows + [(0.5, 0.0, 500.0, 1)]
    if shape == "boundary":
        # Distances are exactly 1..k from the query, so max_distance == k is an
        # equality boundary and k / 2 is a partial cutoff.
        rows = [(float(i), 0.0, 10.0 + i, 2) for i in range(1, k + 1)]
        return rows + [(0.0, 0.0, 500.0, 1)]
    if shape == "bounds-edge":
        # Query lies on the inclusive X edge of the ground bounding box.
        rows = [(2.0 * i, 0.0, 10.0 + i, 2) for i in range(k - 1)]
        rows.append((0.0, 4.0, 300.0, 2))
        return rows + [(0.0, 1.0, 500.0, 1)]
    if shape == "outside":
        # Query lies outside the ground bounding box for no-extrapolation.
        rows = [(2.0 * i, 0.0, 10.0 + i, 2) for i in range(k)]
        return rows + [(100000.0, 0.0, 500.0, 1)]
    if shape == "nonfinite-z":
        rows = [(2.0 * i, 0.0, 10.0 + i, 2) for i in range(k)]
        rows[-1] = (rows[-1][0], 0.0, float("nan"), 2)
        return rows + [(0.5, 0.0, 500.0, 1)]
    if shape == "nonfinite-xy":
        rows = [(2.0 * i, 0.0, 10.0 + i, 2) for i in range(k)]
        rows[-1] = (float("nan"), 0.0, rows[-1][2], 2)
        return rows + [(0.5, 0.0, 500.0, 1)]
    if shape == "incomplete":
        # Grounds sit far outside the query's shell budget.
        rows = [(5001.0 + i, 0.0, 10.0 + i, 2) for i in range(k)]
        return [(0.0, 0.0, 500.0, 1)] + rows
    if shape == "tie":
        # A mirrored ground makes the k-th and (k + 1)-th distances equal.
        rows = [(float(i), 0.0, 10.0 + i, 2) for i in range(1, k + 1)]
        rows.append((float(-k), 0.0, 300.0, 2))
        return rows + [(0.0, 0.0, 500.0, 1)]
    if shape == "underflow":
        # Squared distances are subnormal but strictly increasing.
        rows = [(1.0e-160 * (2.0 ** i), 0.0, 10.0 + i, 2) for i in range(k)]
        return rows + [(0.0, 0.0, 500.0, 1)]
    if shape == "large-finite":
        # Squared distances stay finite binary64 and strictly increasing.
        rows = [(1.0e150 * (1.0 + 0.001 * i), 0.0, 10.0 + i, 2)
                for i in range(k)]
        return rows + [(0.0, 0.0, 500.0, 1)]
    if shape == "distance-overflow":
        # Every squared distance overflows to infinity, so all candidates tie.
        rows = [(1.4e154 + 1.0e152 * i, 0.0, 10.0 + i, 2) for i in range(k)]
        return rows + [(0.0, 0.0, 500.0, 1)]
    if shape == "signed-zero":
        rows = [(2.0 * i, -0.0 if i % 2 else 0.0, 10.0 + i, 2)
                for i in range(k)]
        return rows + [(0.5, 0.0, 500.0, 1)]
    raise ValueError(f"unknown generated shape {shape}")


def generated_cases() -> tuple["Case", ...]:
    cases: list[Case] = []
    for k in GENERATED_COUNTS:
        def kind(shape: str) -> str:
            return f"gen-{shape}-{k}"

        cases.extend((
            Case(f"gen-count{k}-uniform-grid-unique", kind("unique"), {
                "type": "filters.hag_nn", "count": k,
            }, backend="grid", cuda_required=True),
            Case(f"gen-count{k}-bvh-unique", kind("unique"), {
                "type": "filters.hag_nn", "count": k,
            }, backend="bvh", cuda_required=True),
            Case(f"gen-count{k}-interpolation", kind("interpolation"), {
                "type": "filters.hag_nn", "count": k,
                "max_distance": 1000.0, "allow_extrapolation": True,
            }),
            Case(f"gen-count{k}-max-distance-equality", kind("boundary"), {
                "type": "filters.hag_nn", "count": k,
                "max_distance": float(k),
            }),
            Case(f"gen-count{k}-max-distance-partial-cutoff", kind("boundary"),
                 {"type": "filters.hag_nn", "count": k,
                  "max_distance": float(k) / 2.0}),
            Case(f"gen-count{k}-negative-max-distance", kind("boundary"), {
                "type": "filters.hag_nn", "count": k,
                "max_distance": -float(k),
            }),
            Case(f"gen-count{k}-no-extrapolation", kind("outside"), {
                "type": "filters.hag_nn", "count": k,
                "allow_extrapolation": False,
            }),
            Case(f"gen-count{k}-inclusive-ground-bounds", kind("bounds-edge"), {
                "type": "filters.hag_nn", "count": k,
                "allow_extrapolation": False,
            }),
            Case(f"gen-count{k}-large-finite-coordinates",
                 kind("large-finite"), {
                     "type": "filters.hag_nn", "count": k,
                 }),
            Case(f"gen-count{k}-underflow", kind("underflow"), {
                "type": "filters.hag_nn", "count": k,
            }),
            Case(f"gen-count{k}-signed-zero", kind("signed-zero"), {
                "type": "filters.hag_nn", "count": k,
            }),
            Case(f"gen-count{k}-insufficient-one-ground", "one-ground", {
                "type": "filters.hag_nn", "count": k,
            }, cuda_required=False, proof="insufficient"),
            Case(f"gen-count{k}-nonfinite-z-host-repair", kind("nonfinite-z"), {
                "type": "filters.hag_nn", "count": k,
            }, cuda_required=False, proof="nonfinite-z"),
            Case(f"gen-count{k}-nonfinite-xy-fallback", kind("nonfinite-xy"), {
                "type": "filters.hag_nn", "count": k,
            }, cuda_required=False),
            Case(f"gen-count{k}-boundary-tie-uniform-grid", kind("tie"), {
                "type": "filters.hag_nn", "count": k,
            }, backend="grid", cuda_required=False, proof="tie"),
            Case(f"gen-count{k}-boundary-tie-morton-bvh", kind("tie"), {
                "type": "filters.hag_nn", "count": k,
            }, backend="bvh", cuda_required=False, proof="tie"),
            Case(f"gen-count{k}-distance-overflow-uniform-grid",
                 kind("distance-overflow"), {
                     "type": "filters.hag_nn", "count": k,
                 }, backend="grid", cuda_required=False, proof="tie"),
            Case(f"gen-count{k}-incomplete-grid", kind("incomplete"), {
                "type": "filters.hag_nn", "count": k,
            }, backend="grid", cuda_required=False, proof="incomplete"),
        ))
    return tuple(cases)


CASES = CASES + generated_cases()


def rows(kind: str) -> list[tuple[float, float, float, int]]:
    if kind.startswith("gen-"):
        shape, _, count = kind[len("gen-"):].rpartition("-")
        return generated_rows(shape, int(count))
    if kind in ("empty", "missing-classification"):
        return []
    if kind == "custom-class":
        return [(0.0, 0.0, 10.0, 9), (1.0, 0.0, 17.0, 1),
                (4.0, 0.0, 14.0, 9)]
    if kind == "same-xy":
        return [(0.0, 0.0, 10.0, 2), (0.0, 0.0, 17.0, 1),
                (4.0, 0.0, 14.0, 2)]
    if kind == "same-xy-count-three":
        return [(0.0, 0.0, 10.0, 2), (0.0, 0.0, 14.0, 1),
                (2.0, 0.0, 18.0, 2), (4.0, 0.0, 20.0, 2)]
    if kind == "same-xy-count-four":
        return [(0.0, 0.0, 10.0, 2), (0.0, 0.0, 14.0, 1),
                (2.0, 0.0, 18.0, 2), (4.0, 0.0, 20.0, 2),
                (7.0, 0.0, 24.0, 2)]
    if kind == "one-ground":
        return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 17.0, 1),
                (100.0, 100.0, 21.0, 1)]
    if kind == "two-grounds":
        return [(0.0, 0.0, 10.0, 2), (2.0, 0.0, 12.0, 2),
                (10.0, 0.0, 17.0, 1)]
    if kind == "three-grounds":
        return [(0.0, 0.0, 10.0, 2), (2.0, 0.0, 12.0, 2),
                (5.0, 0.0, 14.0, 2), (10.0, 0.0, 17.0, 1)]
    if kind == "all-ground":
        return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 17.0, 2)]
    if kind == "count-three-unique":
        return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (1.4, 0.0, 30.0, 1)]
    if kind == "count-three-interpolation":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (3.0, 0.0, 100.0, 1)]
    if kind == "count-four-unique":
        return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (7.0, 0.0, 22.0, 2),
                (11.0, 0.0, 26.0, 2),
                (1.4, 0.0, 30.0, 1)]
    if kind == "count-five-unique":
        return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (7.0, 0.0, 22.0, 2),
                (11.0, 0.0, 26.0, 2), (15.0, 0.0, 28.0, 2),
                (1.4, 0.0, 30.0, 1)]
    if kind == "count-six-unique":
        return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (7.0, 0.0, 22.0, 2),
                (11.0, 0.0, 26.0, 2), (15.0, 0.0, 28.0, 2),
                (19.0, 0.0, 30.0, 2), (1.4, 0.0, 34.0, 1)]
    if kind == "count-seven-unique":
        return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (7.0, 0.0, 22.0, 2),
                (11.0, 0.0, 26.0, 2), (15.0, 0.0, 28.0, 2),
                (19.0, 0.0, 30.0, 2), (23.0, 0.0, 32.0, 2),
                (1.4, 0.0, 36.0, 1)]
    if kind == "count-four-interpolation":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (3.0, 0.0, 100.0, 1)]
    if kind == "count-five-interpolation":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 24.0, 2), (3.0, 0.0, 100.0, 1)]
    if kind == "count-six-interpolation":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 24.0, 2), (20.0, 0.0, 26.0, 2),
                (3.0, 0.0, 100.0, 1)]
    if kind == "count-seven-interpolation":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 24.0, 2), (20.0, 0.0, 26.0, 2),
                (24.0, 0.0, 28.0, 2), (3.0, 0.0, 100.0, 1)]
    if kind == "count-three-boundary":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (0.0, 0.0, 30.0, 1)]
    if kind == "count-four-boundary":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (4.0, 0.0, 22.0, 2),
                (0.0, 0.0, 30.0, 1)]
    if kind == "count-five-boundary":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (4.0, 0.0, 22.0, 2),
                (5.0, 0.0, 26.0, 2), (0.0, 0.0, 30.0, 1)]
    if kind == "count-six-boundary":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (4.0, 0.0, 22.0, 2),
                (5.0, 0.0, 26.0, 2), (6.0, 0.0, 30.0, 2),
                (0.0, 0.0, 34.0, 1)]
    if kind == "count-seven-boundary":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 18.0, 2),
                (3.0, 0.0, 14.0, 2), (4.0, 0.0, 22.0, 2),
                (5.0, 0.0, 26.0, 2), (6.0, 0.0, 30.0, 2),
                (7.0, 0.0, 32.0, 2), (0.0, 0.0, 36.0, 1)]
    if kind == "count-three-bounds-edge":
        return [(0.0, 0.0, 10.0, 2), (5.0, 4.0, 30.0, 2),
                (0.0, 2.0, 100.0, 1), (2.0, 0.0, 20.0, 2)]
    if kind == "count-four-bounds-edge":
        return [(0.0, 0.0, 10.0, 2), (5.0, 4.0, 30.0, 2),
                (0.0, 4.0, 18.0, 2), (2.0, 0.0, 20.0, 2),
                (0.0, 1.0, 100.0, 1)]
    if kind == "count-five-bounds-edge":
        return [(0.0, 0.0, 10.0, 2), (5.0, 4.0, 30.0, 2),
                (0.0, 4.0, 18.0, 2), (2.0, 0.0, 20.0, 2),
                (6.0, 1.0, 24.0, 2), (0.0, 1.0, 100.0, 1)]
    if kind == "count-six-bounds-edge":
        return [(0.0, 0.0, 10.0, 2), (5.0, 4.0, 30.0, 2),
                (0.0, 4.0, 18.0, 2), (2.0, 0.0, 20.0, 2),
                (6.0, 1.0, 24.0, 2), (8.0, 0.0, 28.0, 2),
                (0.0, 1.0, 100.0, 1)]
    if kind == "count-seven-bounds-edge":
        return [(0.0, 0.0, 10.0, 2), (5.0, 4.0, 30.0, 2),
                (0.0, 4.0, 18.0, 2), (2.0, 0.0, 20.0, 2),
                (6.0, 1.0, 24.0, 2), (8.0, 0.0, 28.0, 2),
                (4.0, 3.0, 26.0, 2), (0.0, 1.0, 100.0, 1)]
    if kind == "count-three-outside":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (100.0, 0.0, 120.0, 1)]
    if kind == "count-four-outside":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (100.0, 0.0, 120.0, 1)]
    if kind == "count-five-outside":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 24.0, 2), (100.0, 0.0, 120.0, 1)]
    if kind == "count-six-outside":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 24.0, 2), (20.0, 0.0, 26.0, 2),
                (100.0, 0.0, 120.0, 1)]
    if kind == "count-seven-outside":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (8.0, 0.0, 14.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 24.0, 2), (20.0, 0.0, 26.0, 2),
                (24.0, 0.0, 28.0, 2), (100.0, 0.0, 120.0, 1)]
    if kind == "count-three-nonfinite-z":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, float("nan"), 2), (6.0, 0.0, 30.0, 1)]
    if kind == "count-four-nonfinite-z":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, 18.0, 2),
                (12.0, 0.0, float("nan"), 2), (6.0, 0.0, 30.0, 1)]
    if kind == "count-five-nonfinite-z":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, 18.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, float("nan"), 2),
                (6.0, 0.0, 30.0, 1)]
    if kind == "count-six-nonfinite-z":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, 18.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 26.0, 2), (20.0, 0.0, float("nan"), 2),
                (6.0, 0.0, 30.0, 1)]
    if kind == "count-seven-nonfinite-z":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, 18.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 26.0, 2), (20.0, 0.0, 28.0, 2),
                (24.0, 0.0, float("nan"), 2), (6.0, 0.0, 30.0, 1)]
    if kind == "third-fourth-boundary-tie":
        return [(1.0, 0.0, 10.0, 2), (0.0, 1.0, 14.0, 2),
                (-1.0, 0.0, 18.0, 2), (0.0, -1.0, 22.0, 2),
                (0.0, 0.0, 30.0, 1)]
    if kind == "fourth-fifth-boundary-tie":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 14.0, 2),
                (3.0, 0.0, 18.0, 2), (4.0, 0.0, 22.0, 2),
                (-4.0, 0.0, 26.0, 2), (0.0, 0.0, 30.0, 1)]
    if kind == "fifth-sixth-boundary-tie":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 14.0, 2),
                (3.0, 0.0, 18.0, 2), (4.0, 0.0, 22.0, 2),
                (5.0, 0.0, 26.0, 2), (-5.0, 0.0, 28.0, 2),
                (0.0, 0.0, 30.0, 1)]
    if kind == "sixth-seventh-boundary-tie":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 14.0, 2),
                (3.0, 0.0, 18.0, 2), (4.0, 0.0, 22.0, 2),
                (5.0, 0.0, 26.0, 2), (6.0, 0.0, 30.0, 2),
                (-6.0, 0.0, 32.0, 2), (0.0, 0.0, 34.0, 1)]
    if kind == "seventh-eighth-boundary-tie":
        return [(1.0, 0.0, 10.0, 2), (2.0, 0.0, 14.0, 2),
                (3.0, 0.0, 18.0, 2), (4.0, 0.0, 22.0, 2),
                (5.0, 0.0, 26.0, 2), (6.0, 0.0, 30.0, 2),
                (7.0, 0.0, 32.0, 2), (-7.0, 0.0, 34.0, 2),
                (0.0, 0.0, 36.0, 1)]
    if kind == "tie":
        return [(0.0, 0.0, 10.0, 2), (2.0, 0.0, 20.0, 2),
                (1.0, 0.0, 100.0, 1)]
    if kind == "boundary-tie":
        return [(0.0, 0.0, 10.0, 2), (-1.0, 0.0, 20.0, 2),
                (3.0, 0.0, 30.0, 2), (1.0, 0.0, 100.0, 1)]
    if kind == "incomplete-two":
        return [(0.0, 0.0, 20.0, 1), (5001.0, 0.0, 10.0, 2),
                (5002.0, 0.0, 11.0, 2)]
    if kind == "count-three-incomplete":
        return [(0.0, 0.0, 20.0, 1), (5001.0, 0.0, 10.0, 2),
                (5002.0, 0.0, 11.0, 2), (5003.0, 0.0, 12.0, 2)]
    if kind == "count-four-incomplete":
        return [(0.0, 0.0, 20.0, 1), (5001.0, 0.0, 10.0, 2),
                (5002.0, 0.0, 11.0, 2), (5003.0, 0.0, 12.0, 2),
                (5004.0, 0.0, 13.0, 2)]
    if kind == "count-five-incomplete":
        return [(0.0, 0.0, 20.0, 1), (5001.0, 0.0, 10.0, 2),
                (5002.0, 0.0, 11.0, 2), (5003.0, 0.0, 12.0, 2),
                (5004.0, 0.0, 13.0, 2), (5005.0, 0.0, 14.0, 2)]
    if kind == "count-six-incomplete":
        return [(0.0, 0.0, 20.0, 1), (5001.0, 0.0, 10.0, 2),
                (5002.0, 0.0, 11.0, 2), (5003.0, 0.0, 12.0, 2),
                (5004.0, 0.0, 13.0, 2), (5005.0, 0.0, 14.0, 2),
                (5006.0, 0.0, 15.0, 2)]
    if kind == "count-seven-incomplete":
        return [(0.0, 0.0, 20.0, 1), (5001.0, 0.0, 10.0, 2),
                (5002.0, 0.0, 11.0, 2), (5003.0, 0.0, 12.0, 2),
                (5004.0, 0.0, 13.0, 2), (5005.0, 0.0, 14.0, 2),
                (5006.0, 0.0, 15.0, 2), (5007.0, 0.0, 16.0, 2)]
    if kind == "count-three-large-finite":
        return [(8.0e153, 8.0e153, 10.0, 2), (1.0e154, 8.0e153, 14.0, 2),
                (8.0e153, 1.0e154, 18.0, 2),
                (8.4e153, 8.6e153, 30.0, 1)]
    if kind == "count-four-large-finite":
        return [(8.0e153, 8.0e153, 10.0, 2),
                (1.0e154, 8.0e153, 14.0, 2),
                (8.0e153, 1.0e154, 18.0, 2),
                (1.0e154, 1.0e154, 22.0, 2),
                (8.4e153, 8.6e153, 30.0, 1)]
    if kind == "count-five-large-finite":
        return [(8.0e153, 8.0e153, 10.0, 2),
                (1.0e154, 8.0e153, 14.0, 2),
                (8.0e153, 1.0e154, 18.0, 2),
                (1.0e154, 1.0e154, 22.0, 2),
                (1.1e154, 1.2e154, 26.0, 2),
                (8.4e153, 8.6e153, 30.0, 1)]
    if kind == "count-six-large-finite":
        return [(8.0e153, 8.0e153, 10.0, 2),
                (1.0e154, 8.0e153, 14.0, 2),
                (8.0e153, 1.0e154, 18.0, 2),
                (1.0e154, 1.0e154, 22.0, 2),
                (1.1e154, 1.2e154, 26.0, 2),
                (1.15e154, 1.3e154, 28.0, 2),
                (8.4e153, 8.6e153, 34.0, 1)]
    if kind == "count-seven-large-finite":
        return [(8.0e153, 8.0e153, 10.0, 2),
                (1.0e154, 8.0e153, 14.0, 2),
                (8.0e153, 1.0e154, 18.0, 2),
                (1.0e154, 1.0e154, 22.0, 2),
                (1.1e154, 1.2e154, 26.0, 2),
                (1.15e154, 1.3e154, 28.0, 2),
                (1.2e154, 1.35e154, 30.0, 2),
                (8.4e153, 8.6e153, 36.0, 1)]
    if kind == "distance-overflow":
        return [(1.4e154, 0.0, 10.0, 2), (1.5e154, 0.0, 14.0, 2),
                (1.6e154, 0.0, 18.0, 2), (1.7e154, 0.0, 22.0, 2),
                (0.0, 0.0, 30.0, 1)]
    if kind == "count-four-distance-overflow":
        return [(1.4e154, 0.0, 10.0, 2), (1.5e154, 0.0, 14.0, 2),
                (1.6e154, 0.0, 18.0, 2), (1.7e154, 0.0, 22.0, 2),
                (1.8e154, 0.0, 26.0, 2), (0.0, 0.0, 30.0, 1)]
    if kind == "count-five-distance-overflow":
        return [(1.4e154, 0.0, 10.0, 2), (1.5e154, 0.0, 14.0, 2),
                (1.6e154, 0.0, 18.0, 2), (1.7e154, 0.0, 22.0, 2),
                (1.8e154, 0.0, 26.0, 2), (1.9e154, 0.0, 28.0, 2),
                (0.0, 0.0, 30.0, 1)]
    if kind == "count-six-distance-overflow":
        return [(1.4e154, 0.0, 10.0, 2), (1.5e154, 0.0, 14.0, 2),
                (1.6e154, 0.0, 18.0, 2), (1.7e154, 0.0, 22.0, 2),
                (1.8e154, 0.0, 26.0, 2), (1.9e154, 0.0, 28.0, 2),
                (2.0e154, 0.0, 32.0, 2), (0.0, 0.0, 34.0, 1)]
    if kind == "count-seven-distance-overflow":
        return [(1.4e154, 0.0, 10.0, 2), (1.5e154, 0.0, 14.0, 2),
                (1.6e154, 0.0, 18.0, 2), (1.7e154, 0.0, 22.0, 2),
                (1.8e154, 0.0, 26.0, 2), (1.9e154, 0.0, 28.0, 2),
                (2.0e154, 0.0, 32.0, 2), (2.1e154, 0.0, 34.0, 2),
                (0.0, 0.0, 36.0, 1)]
    if kind == "count-three-underflow":
        return [(1.0e-160, 0.0, 10.0, 2), (2.0e-160, 0.0, 14.0, 2),
                (4.0e-160, 0.0, 18.0, 2), (0.0, 0.0, 30.0, 1)]
    if kind == "count-four-underflow":
        return [(1.0e-160, 0.0, 10.0, 2), (2.0e-160, 0.0, 14.0, 2),
                (4.0e-160, 0.0, 18.0, 2), (8.0e-160, 0.0, 22.0, 2),
                (0.0, 0.0, 30.0, 1)]
    if kind == "count-five-underflow":
        return [(1.0e-160, 0.0, 10.0, 2), (2.0e-160, 0.0, 14.0, 2),
                (4.0e-160, 0.0, 18.0, 2), (8.0e-160, 0.0, 22.0, 2),
                (1.6e-159, 0.0, 26.0, 2), (0.0, 0.0, 30.0, 1)]
    if kind == "count-six-underflow":
        return [(1.0e-160, 0.0, 10.0, 2), (2.0e-160, 0.0, 14.0, 2),
                (4.0e-160, 0.0, 18.0, 2), (8.0e-160, 0.0, 22.0, 2),
                (1.6e-159, 0.0, 26.0, 2), (3.2e-159, 0.0, 30.0, 2),
                (0.0, 0.0, 34.0, 1)]
    if kind == "count-seven-underflow":
        return [(1.0e-160, 0.0, 10.0, 2), (2.0e-160, 0.0, 14.0, 2),
                (4.0e-160, 0.0, 18.0, 2), (8.0e-160, 0.0, 22.0, 2),
                (1.6e-159, 0.0, 26.0, 2), (3.2e-159, 0.0, 30.0, 2),
                (6.4e-159, 0.0, 32.0, 2), (0.0, 0.0, 36.0, 1)]
    if kind == "count-three-signed-zero":
        return [(-0.0, 0.0, 10.0, 2), (5.0, -0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (1.0, 0.5, 30.0, 1)]
    if kind == "count-four-signed-zero":
        return [(-0.0, 0.0, 10.0, 2), (5.0, -0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (7.0, 2.0, 22.0, 2),
                (1.0, 0.5, 30.0, 1)]
    if kind == "count-five-signed-zero":
        return [(-0.0, 0.0, 10.0, 2), (5.0, -0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (7.0, 2.0, 22.0, 2),
                (9.0, 0.0, 26.0, 2), (1.0, 0.5, 30.0, 1)]
    if kind == "count-six-signed-zero":
        return [(-0.0, 0.0, 10.0, 2), (5.0, -0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (7.0, 2.0, 22.0, 2),
                (9.0, 0.0, 26.0, 2), (11.0, -0.0, 30.0, 2),
                (1.0, 0.5, 34.0, 1)]
    if kind == "count-seven-signed-zero":
        return [(-0.0, 0.0, 10.0, 2), (5.0, -0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (7.0, 2.0, 22.0, 2),
                (9.0, 0.0, 26.0, 2), (11.0, -0.0, 30.0, 2),
                (13.0, -0.0, 32.0, 2), (1.0, 0.5, 36.0, 1)]
    if kind == "max-boundary":
        return [(0.0, 0.0, 10.0, 2), (5.0, 0.0, 30.0, 2),
                (2.0, 0.0, 100.0, 1)]
    if kind == "bounds-edge":
        return [(0.0, 0.0, 10.0, 2), (5.0, 4.0, 30.0, 2),
                (0.0, 2.0, 100.0, 1)]
    if kind == "nonfinite-z":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (1.0, 0.0, float("nan"), 1)]
    if kind == "count-six-nonfinite-xy":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, 18.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 26.0, 2), (float("nan"), 0.0, 30.0, 2),
                (1.0, 0.0, 34.0, 1)]
    if kind == "count-seven-nonfinite-xy":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, 18.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 26.0, 2), (20.0, 0.0, 30.0, 2),
                (float("nan"), 0.0, 32.0, 2), (1.0, 0.0, 36.0, 1)]
    if kind == "count-five-nonfinite-xy":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (8.0, 0.0, 18.0, 2), (12.0, 0.0, 22.0, 2),
                (16.0, 0.0, 26.0, 2),
                (float("nan"), 0.0, 30.0, 1)]
    if kind == "no-ground":
        return [(0.0, 0.0, 10.0, 1), (1.0, 0.0, 17.0, 1)]
    if kind == "nonfinite-xy":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (float("nan"), 0.0, 20.0, 1)]
    if kind == "interpolation":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (1.0, 0.0, 100.0, 1), (3.0, 0.0, 200.0, 1)]
    if kind == "outside":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 18.0, 2),
                (9.0, 0.0, 100.0, 1)]
    return [(0.0, 0.0, 10.0, 2), (1.0, 0.0, 100.0, 1),
            (4.0, 0.0, 14.0, 2), (8.0, 0.0, 200.0, 1),
            (10.0, 0.0, 20.0, 2)]


def write_fixture(path: Path, kind: str) -> None:
    if kind == "missing-classification":
        path.write_text("X,Y,Z\n0,0,10\n", encoding="utf-8")
        return
    lines = ["X,Y,Z,Classification\n"]
    for x, y, z, classification in rows(kind):
        lines.append(f"{x},{y},{z},{classification}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-hag-nn-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.cuda:
                continue
            executed += 1
            fixture = generated / f"{case.fixture}.csv"
            write_fixture(fixture, case.fixture)
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({"pipeline": [
                    {"type": "readers.text", "filename": "input.csv"},
                    case.stage,
                    {"type": "writers.las", "filename": "out.las",
                     "extra_dims": "all"},
                ]}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

            environment = os.environ.copy()
            if args.require_cuda:
                environment.update({"PDG_DISABLE_NATIVE": "1",
                                    "PDG_REQUIRE_HYBRID": "1"})
                environment.pop("PDG_DISABLE_CUDA_HYBRID", None)
                if case.cuda_required:
                    environment["PDG_REQUIRE_CUDA_HYBRID"] = "1"
                    environment.pop("PDG_EXPERIMENTAL_CUDA_HYBRID", None)
                else:
                    environment["PDG_EXPERIMENTAL_CUDA_HYBRID"] = "1"
                    environment.pop("PDG_REQUIRE_CUDA_HYBRID", None)
            elif case.hybrid:
                environment.update({
                    "PDG_DISABLE_NATIVE": "1",
                    "PDG_REQUIRE_HYBRID": "1",
                    "PDG_DISABLE_CUDA_HYBRID": "1",
                })
                environment.pop("PDG_REQUIRE_CUDA_HYBRID", None)
            else:
                for name in (
                    "PDG_DISABLE_NATIVE", "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID", "PDG_REQUIRE_CUDA_HYBRID",
                    "PDG_EXPERIMENTAL_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            if case.backend == "grid":
                environment["PDG_FORCE_UNIFORM_GRID"] = "1"
                environment.pop("PDG_FORCE_MORTON_BVH", None)
            elif case.backend == "bvh":
                environment["PDG_FORCE_MORTON_BVH"] = "1"
                environment.pop("PDG_FORCE_UNIFORM_GRID", None)
            else:
                environment.pop("PDG_FORCE_UNIFORM_GRID", None)
                environment.pop("PDG_FORCE_MORTON_BVH", None)
            if case.proof == "tie":
                environment["PDG_REQUIRE_HAG_NN_TIE_FALLBACK"] = "1"
            else:
                environment.pop("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", None)
            if case.proof == "incomplete":
                environment["PDG_KNN_DEVICE_SHELL_BUDGET"] = "1"
                environment["PDG_REQUIRE_HAG_NN_HOST_FALLBACK"] = "1"
            else:
                environment.pop("PDG_KNN_DEVICE_SHELL_BUDGET", None)
                environment.pop("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", None)
            if case.proof == "insufficient":
                environment[
                    "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK"
                ] = "1"
            else:
                environment.pop(
                    "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", None)
            if case.proof == "nonfinite-z":
                environment["PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK"] = "1"
            else:
                environment.pop("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK", None)

            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"hag-nn-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"input.csv={fixture.resolve()}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            for preload in args.oracle_preload:
                command.extend(("--oracle-preload", str(preload.resolve())))
            for preload in args.candidate_preload:
                command.extend(("--candidate-preload",
                                str(preload.resolve())))
            command.extend(("--", "pipeline", "pipeline.json"))
            if case.verbose:
                command.extend(("--verbose", "8"))
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact hag_nn process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
