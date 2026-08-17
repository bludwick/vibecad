# SPDX-License-Identifier: LGPL-2.1-or-later

"""Contracts for the deterministic geometry-worker tuning benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_PATH = ROOT / "tools" / "geometry_worker_benchmark.py"


def _benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "geometry_worker_benchmark",
        BENCHMARK_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_benchmark_scores_exact_and_tolerant_truth() -> None:
    benchmark = _benchmark_module()
    response = {
        "ok": True,
        "geometry": {
            "solids": 1,
            "volume_mm3": 6000.0000002,
            "bounds_mm": {"size": [10.0, 20.0, 30.0]},
            "query_results": [
                {"name": "top_face", "matched_count": 1, "cardinality_ok": True}
            ],
        },
    }
    truth = (
        benchmark.Expectation("geometry.solids", 1),
        benchmark.Expectation("geometry.volume_mm3", 6000.0, absolute_tolerance=1e-5),
        benchmark.Expectation("geometry.bounds_mm.size", [10.0, 20.0, 30.0]),
        benchmark.Expectation("geometry.query_results.0.matched_count", 1),
    )

    assert benchmark.score_response(response, truth) == []

    failures = benchmark.score_response(
        response,
        (benchmark.Expectation("geometry.query_results.0.matched_count", 2),),
    )
    assert failures == [
        {
            "path": "geometry.query_results.0.matched_count",
            "expected": 2,
            "actual": 1,
            "absolute_tolerance": 0.0,
        }
    ]


def test_geometry_benchmark_stability_projection_excludes_runtime_noise() -> None:
    benchmark = _benchmark_module()
    first = {
        "ok": True,
        "elapsed_seconds": 0.02,
        "execution": {"elapsed_ms": 15, "mode": "isolated_geometry_worker"},
        "geometry": {"faces": 6, "query_results": [{"matched_count": 1}]},
    }
    second = {
        "ok": True,
        "elapsed_seconds": 0.25,
        "execution": {"elapsed_ms": 240, "mode": "isolated_geometry_worker"},
        "geometry": {"faces": 6, "query_results": [{"matched_count": 1}]},
    }

    assert benchmark.stability_projection(first) == benchmark.stability_projection(
        second
    )

    second["geometry"]["faces"] = 7
    assert benchmark.stability_projection(first) != benchmark.stability_projection(
        second
    )


def test_repository_truth_uses_exact_brep_bounds_not_cached_triangulation() -> None:
    benchmark = _benchmark_module()
    calls = []
    shape = SimpleNamespace(
        isValid=lambda: True,
        Solids=[object()],
        Shells=[object()],
        Faces=[object()],
        Wires=[object()],
        Edges=[object()],
        Vertexes=[object()],
        optimalBoundingBox=lambda *arguments: (
            calls.append(arguments)
            or SimpleNamespace(XLength=10.0, YLength=20.0, ZLength=30.0)
        ),
    )

    truth = benchmark._derived_shape_truth(shape)

    assert calls == [(False, False)]
    assert truth[-1] == benchmark.Expectation(
        "geometry.bounds_mm.size",
        [10.0, 20.0, 30.0],
        1.0e-6,
    )
