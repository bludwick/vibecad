# SPDX-License-Identifier: LGPL-2.1-or-later

"""Deterministic correctness and stability benchmark for geometry inspection."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, NamedTuple


BENCHMARK_SCHEMA = "vibecad-geometry-worker-benchmark-v1"
DEFAULT_REPEATS = 3
_BENCHMARK_ARGUMENTS_ENV = "VIBECAD_GEOMETRY_BENCHMARK_ARGUMENTS"
_BENCHMARK_SCRIPT_ENV = "VIBECAD_GEOMETRY_BENCHMARK_SCRIPT"
_RUNTIME_NOISE_FIELDS = frozenset(
    {"elapsed_ms", "elapsed_seconds", "worker_elapsed_ms"}
)


class Expectation(NamedTuple):
    path: str
    expected: Any
    absolute_tolerance: float = 0.0


class BenchmarkCase(NamedTuple):
    name: str
    build_shape: Callable[[Any, Any], Any]
    queries: tuple[dict[str, Any], ...]
    truth: tuple[Expectation, ...]
    source: str = "analytic"
    derive_shape_truth: bool = False


def _value_at_path(payload: Any, path: str) -> Any:
    value = payload
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value[part]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = value[int(part)]
        else:
            raise KeyError(path)
    return value


def _matches(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=float(tolerance),
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _matches(actual_item, expected_item, tolerance)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def score_response(
    response: Mapping[str, Any],
    truth: Sequence[Expectation],
) -> list[dict[str, Any]]:
    """Return exact, compact mismatches between one worker response and truth."""

    failures: list[dict[str, Any]] = []
    for expectation in truth:
        try:
            actual = _value_at_path(response, expectation.path)
        except (KeyError, IndexError, TypeError, ValueError):
            actual = None
        if _matches(actual, expectation.expected, expectation.absolute_tolerance):
            continue
        failures.append(
            {
                "path": expectation.path,
                "expected": expectation.expected,
                "actual": actual,
                "absolute_tolerance": expectation.absolute_tolerance,
            }
        )
    return failures


def stability_projection(value: Any) -> Any:
    """Remove timing noise while retaining every geometry-visible result."""

    if isinstance(value, Mapping):
        return {
            str(key): stability_projection(item)
            for key, item in value.items()
            if str(key) not in _RUNTIME_NOISE_FIELDS
        }
    if isinstance(value, list):
        return [stability_projection(item) for item in value]
    return value


def _query_truth(name: str, expected_count: int) -> Expectation:
    return Expectation(
        f"geometry.query_results.{name}.matched_count",
        expected_count,
    )


def _benchmark_cases() -> tuple[BenchmarkCase, ...]:
    def query(
        name: str,
        element_type: str,
        geometry_type: str,
        expected_count: int | None = None,
        **filters: Any,
    ) -> dict[str, Any]:
        result = {
            "name": name,
            "element_type": element_type,
            "geometry_type": geometry_type,
            **filters,
        }
        if expected_count is not None:
            result["expected_count"] = expected_count
        return result

    return (
        BenchmarkCase(
            "box",
            lambda Part, App: Part.makeBox(10.0, 20.0, 30.0),
            (
                query(
                    "top_plane",
                    "face",
                    "Plane",
                    1,
                    normal=[0.0, 0.0, 1.0],
                    min_area_mm2=199.999,
                    max_area_mm2=200.001,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation("geometry.faces", 6),
                Expectation("geometry.edges", 12),
                Expectation("geometry.volume_mm3", 6000.0, 1.0e-6),
                Expectation("geometry.bounds_mm.size", [10.0, 20.0, 30.0], 1.0e-6),
                _query_truth("top_plane", 1),
            ),
        ),
        BenchmarkCase(
            "cylinder",
            lambda Part, App: Part.makeCylinder(4.0, 12.0),
            (
                query(
                    "shaft_surface",
                    "face",
                    "Cylinder",
                    1,
                    axis_direction=[0.0, 0.0, 1.0],
                    radius_mm=4.0,
                ),
                query(
                    "rim_edges",
                    "edge",
                    "Circle",
                    2,
                    axis_direction=[0.0, 0.0, 1.0],
                    radius_mm=4.0,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation("geometry.volume_mm3", math.pi * 4.0**2 * 12.0, 1.0e-6),
                _query_truth("shaft_surface", 1),
                _query_truth("rim_edges", 2),
            ),
        ),
        BenchmarkCase(
            "tube",
            lambda Part, App: Part.makeCylinder(5.0, 20.0).cut(
                Part.makeCylinder(3.0, 20.0)
            ),
            (
                query(
                    "outer_wall",
                    "face",
                    "Cylinder",
                    1,
                    axis_direction=[0.0, 0.0, 1.0],
                    radius_mm=5.0,
                ),
                query(
                    "inner_wall",
                    "face",
                    "Cylinder",
                    1,
                    axis_direction=[0.0, 0.0, 1.0],
                    radius_mm=3.0,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation(
                    "geometry.volume_mm3",
                    math.pi * (5.0**2 - 3.0**2) * 20.0,
                    1.0e-6,
                ),
                _query_truth("outer_wall", 1),
                _query_truth("inner_wall", 1),
            ),
        ),
        BenchmarkCase(
            "plate_with_hole",
            lambda Part, App: Part.makeBox(40.0, 30.0, 6.0).cut(
                Part.makeCylinder(3.0, 6.0, App.Vector(10.0, 15.0, 0.0))
            ),
            (
                query(
                    "hole_wall",
                    "face",
                    "Cylinder",
                    1,
                    axis_direction=[0.0, 0.0, 1.0],
                    radius_mm=3.0,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation(
                    "geometry.volume_mm3",
                    40.0 * 30.0 * 6.0 - math.pi * 3.0**2 * 6.0,
                    1.0e-6,
                ),
                _query_truth("hole_wall", 1),
            ),
        ),
        BenchmarkCase(
            "frustum",
            lambda Part, App: Part.makeCone(8.0, 3.0, 15.0),
            (
                query(
                    "conical_wall",
                    "face",
                    "Cone",
                    1,
                    axis_direction=[0.0, 0.0, 1.0],
                    reference_radius_mm=8.0,
                    semi_angle_degrees=math.degrees(math.atan2(-5.0, 15.0)),
                ),
                query(
                    "different_cone",
                    "face",
                    "Cone",
                    reference_radius_mm=7.0,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation(
                    "geometry.volume_mm3",
                    math.pi * 15.0 * (8.0**2 + 8.0 * 3.0 + 3.0**2) / 3.0,
                    1.0e-6,
                ),
                _query_truth("conical_wall", 1),
                _query_truth("different_cone", 0),
            ),
        ),
        BenchmarkCase(
            "sphere",
            lambda Part, App: Part.makeSphere(7.0),
            (
                query(
                    "spherical_surface",
                    "face",
                    "Sphere",
                    1,
                    radius_mm=7.0,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation(
                    "geometry.volume_mm3", 4.0 * math.pi * 7.0**3 / 3.0, 1.0e-6
                ),
                _query_truth("spherical_surface", 1),
            ),
        ),
        BenchmarkCase(
            "torus",
            lambda Part, App: Part.makeTorus(12.0, 3.0),
            (
                query(
                    "toroidal_surface",
                    "face",
                    "Torus",
                    1,
                    axis_direction=[0.0, 0.0, 1.0],
                    major_radius_mm=12.0,
                    minor_radius_mm=3.0,
                ),
                query(
                    "different_torus",
                    "face",
                    "Torus",
                    major_radius_mm=13.0,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation(
                    "geometry.volume_mm3", 2.0 * math.pi**2 * 12.0 * 3.0**2, 1.0e-6
                ),
                Expectation("geometry.bounds_mm.size", [30.0, 30.0, 6.0], 1.0e-6),
                _query_truth("toroidal_surface", 1),
                _query_truth("different_torus", 0),
            ),
        ),
        BenchmarkCase(
            "ellipse",
            lambda Part, App: Part.Ellipse(App.Vector(), 10.0, 4.0).toShape(),
            (
                query(
                    "elliptic_edge",
                    "edge",
                    "Ellipse",
                    1,
                    major_radius_mm=10.0,
                    minor_radius_mm=4.0,
                ),
                query(
                    "different_ellipse",
                    "edge",
                    "Ellipse",
                    minor_radius_mm=5.0,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.edges", 1),
                Expectation("geometry.bounds_mm.size", [20.0, 8.0, 0.0], 1.0e-6),
                _query_truth("elliptic_edge", 1),
                _query_truth("different_ellipse", 0),
            ),
        ),
        BenchmarkCase(
            "rotated_cylinder",
            lambda Part, App: Part.makeCylinder(
                5.0,
                20.0,
                App.Vector(1.0, 2.0, 3.0),
                App.Vector(1.0, 1.0, 1.0),
            ),
            (
                query(
                    "rotated_shaft",
                    "face",
                    "Cylinder",
                    1,
                    axis_direction=[1.0, 1.0, 1.0],
                    radius_mm=5.0,
                    angle_tolerance_degrees=1.0e-6,
                ),
            ),
            (
                Expectation("ok", True),
                Expectation("geometry.valid", True),
                Expectation("geometry.solids", 1),
                Expectation("geometry.volume_mm3", math.pi * 5.0**2 * 20.0, 1.0e-6),
                _query_truth("rotated_shaft", 1),
            ),
        ),
    )


def _repository_brep_cases() -> tuple[BenchmarkCase, ...]:
    repository = Path(__file__).resolve().parents[1]
    paths = sorted(
        (repository / "data" / "tests" / "ModelRefineTests").glob("*.brep")
    ) + sorted(
        (repository / "tests" / "src" / "Mod" / "Part" / "App" / "brepfiles").glob(
            "*.brep"
        )
    )
    return tuple(
        BenchmarkCase(
            f"repository/{path.relative_to(repository).as_posix()}",
            lambda Part, App, source=path: Part.read(str(source)),
            (),
            (Expectation("ok", True),),
            source="repository_brep",
            derive_shape_truth=True,
        )
        for path in paths
    )


def _derived_shape_truth(shape: Any) -> tuple[Expectation, ...]:
    bounds = shape.optimalBoundingBox(False, False)
    return (
        Expectation("geometry.valid", bool(shape.isValid())),
        Expectation("geometry.solids", len(shape.Solids)),
        Expectation("geometry.shells", len(shape.Shells)),
        Expectation("geometry.faces", len(shape.Faces)),
        Expectation("geometry.wires", len(shape.Wires)),
        Expectation("geometry.edges", len(shape.Edges)),
        Expectation("geometry.vertices", len(shape.Vertexes)),
        Expectation(
            "geometry.bounds_mm.size",
            [bounds.XLength, bounds.YLength, bounds.ZLength],
            1.0e-6,
        ),
    )


def _named_query_results(response: Mapping[str, Any]) -> dict[str, Any]:
    geometry = response.get("geometry")
    if not isinstance(geometry, Mapping):
        return {}
    query_results = geometry.get("query_results")
    if not isinstance(query_results, list):
        return {}
    return {
        str(item.get("name") or index): item
        for index, item in enumerate(query_results)
        if isinstance(item, Mapping)
    }


def _scoreable_response(response: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(response)
    geometry = result.get("geometry")
    if isinstance(geometry, Mapping):
        clean_geometry = dict(geometry)
        clean_geometry["query_results"] = _named_query_results(response)
        result["geometry"] = clean_geometry
    return result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 6)


class _DocumentService:
    def __init__(self, document: Any) -> None:
        self._document = document

    def _active_document(self) -> Any:
        return self._document


def run_benchmark(
    *,
    repeats: int = DEFAULT_REPEATS,
    selected_cases: set[str] | None = None,
    include_repository_breps: bool = False,
) -> dict[str, Any]:
    """Execute the public geometry-read path against deterministic truth."""

    import FreeCAD as App
    import Part

    from VibeCADDocumentReferences import reference_for_target
    from VibeCADGeometry import worker_executable
    from VibeCADGeometryInspection import (
        capture_geometry_read,
        complete_geometry_read,
    )

    repeat_count = max(1, int(repeats))
    available_cases = _benchmark_cases() + (
        _repository_brep_cases() if include_repository_breps else ()
    )
    cases = tuple(
        case
        for case in available_cases
        if not selected_cases or case.name in selected_cases
    )
    document = App.newDocument("GeometryWorkerBenchmark")
    service = _DocumentService(document)
    case_results: list[dict[str, Any]] = []
    all_wall_times: list[float] = []
    all_worker_times: list[float] = []
    all_response_sizes: list[int] = []
    try:
        for case_index, case in enumerate(cases, start=1):
            feature = document.addObject("Part::Feature", f"Benchmark_{case_index}")
            feature.Label = case.name.replace("_", " ").title()
            feature.Shape = case.build_shape(Part, App)
            document.recompute()
            reference = reference_for_target(document, feature)
            truth = case.truth + (
                _derived_shape_truth(feature.Shape) if case.derive_shape_truth else ()
            )
            responses: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            wall_times: list[float] = []
            worker_times: list[float] = []
            response_sizes: list[int] = []
            for run_index in range(repeat_count):
                captured = capture_geometry_read(
                    service,
                    {
                        "reference": reference,
                        "analysis_level": "full",
                        "queries": [dict(item) for item in case.queries],
                    },
                )
                started = time.monotonic()
                response = complete_geometry_read(captured)
                wall_time = round(time.monotonic() - started, 6)
                responses.append(response)
                wall_times.append(wall_time)
                worker_time = response.get("execution", {}).get("elapsed_seconds")
                if isinstance(worker_time, (int, float)):
                    worker_times.append(float(worker_time))
                response_sizes.append(
                    len(
                        json.dumps(
                            response,
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                )
                for failure in score_response(_scoreable_response(response), truth):
                    failures.append({"run": run_index + 1, **failure})
            first_projection = stability_projection(responses[0])
            unstable_runs = [
                index + 1
                for index, response in enumerate(responses[1:], start=1)
                if stability_projection(response) != first_projection
            ]
            query_counts = {
                name: int(result.get("matched_count") or 0)
                for name, result in _named_query_results(responses[0]).items()
            }
            geometry = responses[0].get("geometry")
            observed = (
                {
                    key: geometry.get(key)
                    for key in (
                        "shape_type",
                        "valid",
                        "solids",
                        "shells",
                        "faces",
                        "wires",
                        "edges",
                        "vertices",
                        "volume_mm3",
                        "area_mm2",
                        "bounds_mm",
                    )
                }
                if isinstance(geometry, Mapping)
                else {}
            )
            case_result: dict[str, Any] = {
                "name": case.name,
                "source": case.source,
                "passed": not failures and not unstable_runs,
                "failures": failures,
                "unstable_runs": unstable_runs,
                "query_match_counts": query_counts,
                "observed": observed,
                "metrics": {
                    "wall_seconds": wall_times,
                    "worker_seconds": worker_times,
                    "response_bytes": response_sizes,
                },
            }
            if failures or unstable_runs:
                case_result["first_response"] = responses[0]
            case_results.append(case_result)
            all_wall_times.extend(wall_times)
            all_worker_times.extend(worker_times)
            all_response_sizes.extend(response_sizes)
            document.removeObject(feature.Name)
            document.recompute()
    finally:
        App.closeDocument(document.Name)

    passed_count = sum(1 for result in case_results if result["passed"])
    slowest_case = max(
        case_results,
        key=lambda result: max(result["metrics"]["worker_seconds"] or [0.0]),
        default=None,
    )
    return {
        "schema": BENCHMARK_SCHEMA,
        "worker": str(worker_executable()),
        "python": sys.version.split()[0],
        "repeats": repeat_count,
        "summary": {
            "case_count": len(case_results),
            "passed": passed_count,
            "failed": len(case_results) - passed_count,
            "wall_seconds_p50": (
                round(statistics.median(all_wall_times), 6) if all_wall_times else None
            ),
            "wall_seconds_p95": _percentile(all_wall_times, 0.95),
            "worker_seconds_p50": (
                round(statistics.median(all_worker_times), 6)
                if all_worker_times
                else None
            ),
            "worker_seconds_p95": _percentile(all_worker_times, 0.95),
            "worker_seconds_max": (
                round(max(all_worker_times), 6) if all_worker_times else None
            ),
            "slowest_case": slowest_case["name"] if slowest_case else None,
            "response_bytes_p50": (
                int(statistics.median(all_response_sizes))
                if all_response_sizes
                else None
            ),
            "response_bytes_p95": _percentile(all_response_sizes, 0.95),
        },
        "cases": case_results,
    }


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VibeCAD's installed geometry worker with exact solids."
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        choices=[case.name for case in _benchmark_cases()],
        help="Run one named case; repeat the option to select several.",
    )
    parser.add_argument(
        "--repository-breps",
        action="store_true",
        help="Also stress the worker with repository BREP regression fixtures.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _freecadcmd_executable() -> Path:
    override = str(os.environ.get("VIBECAD_FREECADCMD") or "").strip()
    repository = Path(__file__).resolve().parents[1]
    candidates = (
        Path(override).expanduser() if override else None,
        repository / "build" / "release" / "bin" / "FreeCADCmd",
        repository / "build" / "release" / "bin" / "FreeCADCmd.exe",
        repository / "build" / "release" / "bin" / "Release" / "FreeCADCmd.exe",
        *(
            Path(found)
            for name in ("FreeCADCmd", "freecadcmd")
            if (found := shutil.which(name))
        ),
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "No FreeCADCmd runtime was found. Build VibeCAD or set VIBECAD_FREECADCMD."
    )


def _launch_in_freecad(argv: Sequence[str] | None) -> int:
    environment = os.environ.copy()
    environment[_BENCHMARK_ARGUMENTS_ENV] = json.dumps(
        list(sys.argv[1:] if argv is None else argv),
        ensure_ascii=True,
    )
    environment[_BENCHMARK_SCRIPT_ENV] = str(Path(__file__).resolve())
    bootstrap = (
        "import json, os, runpy, sys; "
        f"script = os.environ[{_BENCHMARK_SCRIPT_ENV!r}]; "
        f"sys.argv = [script] + json.loads(os.environ[{_BENCHMARK_ARGUMENTS_ENV!r}]); "
        "runpy.run_path(script, run_name='__main__')"
    )
    completed = subprocess.run(
        [str(_freecadcmd_executable()), "--safe-mode", "-c", bootstrap],
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        import FreeCAD  # noqa: F401
    except ImportError:
        return _launch_in_freecad(argv)

    arguments = _arguments(argv)
    result = run_benchmark(
        repeats=arguments.repeat,
        selected_cases=set(arguments.cases or []),
        include_repository_breps=arguments.repository_breps,
    )
    rendered = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.write_text(
            rendered + "\n",
            encoding="utf-8",
        )
        rendered = json.dumps(
            {
                "schema": BENCHMARK_SCHEMA,
                "output": str(output),
                "summary": result["summary"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    print(rendered, flush=True)
    return 0 if result["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
