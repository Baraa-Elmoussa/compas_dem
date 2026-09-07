"""Shared helpers for the compas_dem solver benchmark suite.

Every case script builds a :class:`~compas_dem.problem.Problem` on top of a
:class:`~compas_dem.models.BlockModel` and hands it to :func:`run_benchmark`,
which solves it with ``MasonryDEM`` (in-process, always attempted) and
``ThreeDEC`` (only if ``COMPAS_3DEC_EXECUTABLE`` is set), timing each solve
and reporting whether the structure stood or collapsed under gravity.

To run the real 3DEC backend, set two environment variables before running a
case (PowerShell shown; also works with ``$env:COMPAS_3DEC_VERSION``):

    $env:COMPAS_3DEC_EXECUTABLE = "C:/path/to/3dec.exe"
    $env:COMPAS_3DEC_VERSION = "7.0"

Without them, every case still runs MasonryDEM and reports 3DEC as "skipped".
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from compas_dem.material import Stone
from compas_dem.models import BlockModel
from compas_dem.problem import Problem
from compas_dem.problem import Solver

HERE = Path(__file__).parent
RUNS = HERE / "runs"

DEFAULT_MU = 0.6
DEFAULT_COHESION = 0.0
DEFAULT_KN = 1e9
DEFAULT_KT = 5e8
DEFAULT_DENSITY = 2400.0
DEFAULT_DISPLACEMENT_THRESHOLD = 0.03  # [m] below this, a block counts as "settled" not "collapsed"


def assign_stone(model: BlockModel, density: float = DEFAULT_DENSITY) -> None:
    """Assign a uniform stone material to every block, so mass/inertia are defined."""
    stone = Stone(density=density)
    model.add_material(stone)
    model.assign_material(stone, elements=list(model.elements()))


def build_problem(
    model: BlockModel,
    name: str,
    mu: float = DEFAULT_MU,
    c: float = DEFAULT_COHESION,
    kn: float = DEFAULT_KN,
    kt: float = DEFAULT_KT,
    g: float = 9.81,
) -> Problem:
    """Build a Problem with a Mohr-Coulomb contact model, a joint stiffness
    (used by 3DEC and, if present, by masonry_dem) and gravity-only loading.
    """
    problem = Problem(model, name=name)
    problem.set_contact_model("MohrCoulomb", mu=mu, c=c)
    problem.set_joint_model(kn=kn, kt=kt)
    problem.add_boundary_condition("gravity").add_gravity(g=g)
    return problem


@dataclass
class BenchmarkResult:
    solver: str
    status: str  # "ok" | "skipped" | "error"
    elapsed: Optional[float] = None
    max_displacement: Optional[float] = None
    mean_displacement: Optional[float] = None
    verdict: Optional[str] = None  # "stood" | "collapsed" | None
    note: str = ""
    results: Optional[object] = None  # the raw compas_dem Results, for optional visualization


def _stability(model: BlockModel, results) -> dict:
    """Max/mean rigid-body displacement magnitude over the free (non-support) blocks."""
    displacements = []
    for element in model.blocks():
        d = results.displacement(element.graphnode)
        if d is None:
            continue
        displacements.append(sum(v * v for v in d) ** 0.5)
    if not displacements:
        return {"max_displacement": None, "mean_displacement": None}
    return {
        "max_displacement": max(displacements),
        "mean_displacement": sum(displacements) / len(displacements),
    }


def _threedec_solver(**kwargs) -> Optional[Solver]:
    """Build a Solver.ThreeDEC(), same as the other compas_dem 3DEC examples.

    ``executable`` is passed straight from ``COMPAS_3DEC_EXECUTABLE`` (or
    ``None``) -- compas_3dec resolves ``None`` through its own discovery, so
    this does not gate on the env var itself. Returns ``None`` only if
    ``compas_3dec`` is not installed at all; any other failure (no
    executable found, no license, ...) surfaces from ``problem.solve()``
    and is caught by ``_run_one`` as an "error" row.
    """
    try:
        import compas_3dec  # noqa: F401
    except ImportError:
        return None
    RUNS.mkdir(exist_ok=True)
    kwargs.setdefault("executable", os.getenv("COMPAS_3DEC_EXECUTABLE"))
    kwargs.setdefault("version", os.getenv("COMPAS_3DEC_VERSION", "7.0"))
    kwargs.setdefault("workspace", RUNS)
    kwargs.setdefault("suppress_output", True)
    kwargs.setdefault("timeout", 300)
    return Solver.ThreeDEC(**kwargs)


def _run_one(problem: Problem, model: BlockModel, label: str, solver: Solver, threshold: float) -> BenchmarkResult:
    problem.set_solver(solver)
    t0 = time.perf_counter()
    try:
        results = problem.solve()
    except Exception as exc:  # noqa: BLE001 - a failed backend must not kill the benchmark
        return BenchmarkResult(label, status="error", elapsed=time.perf_counter() - t0, note=f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - t0
    stability = _stability(model, results)
    max_d = stability["max_displacement"]
    verdict = "stood" if (max_d is not None and max_d < threshold) else ("collapsed" if max_d is not None else None)
    return BenchmarkResult(label, status="ok", elapsed=elapsed, verdict=verdict, results=results, **stability)


def run_benchmark(
    problem: Problem,
    displacement_threshold: float = DEFAULT_DISPLACEMENT_THRESHOLD,
    masonry_dem_kwargs: Optional[dict] = None,
    threedec_kwargs: Optional[dict] = None,
) -> list[BenchmarkResult]:
    """Solve ``problem`` with each available backend, timing every solve.

    A backend whose max free-block displacement stays below
    ``displacement_threshold`` [m] is reported as "stood"; anything larger
    (including a diverged solve) is "collapsed". A backend that isn't
    installed or configured (masonry_dem missing, no 3DEC executable) is
    reported as "skipped" rather than failing the whole run.
    """
    rows: list[BenchmarkResult] = []
    model = problem.model

    if masonry_dem_kwargs is not None:
        try:
            import masonry_dem  # noqa: F401
        except ImportError:
            rows.append(BenchmarkResult("MasonryDEM", status="skipped", note="masonry_dem is not installed in this environment"))
        else:
            rows.append(_run_one(problem, model, "MasonryDEM", Solver.MasonryDEM(**masonry_dem_kwargs), displacement_threshold))

    solver = _threedec_solver(**(threedec_kwargs or {}))
    if solver is None:
        rows.append(BenchmarkResult("3DEC", status="skipped", note="compas_3dec is not installed in this environment"))
    else:
        rows.append(_run_one(problem, model, "3DEC", solver, displacement_threshold))

    return rows


def print_report(case_name: str, rows: list[BenchmarkResult]) -> None:
    print(f"\n=== {case_name} ===")
    header = f"{'Solver':<12}{'Status':<10}{'Time [s]':>10}{'Max disp [m]':>15}{'Verdict':>12}  Note"
    print(header)
    print("-" * len(header))
    for row in rows:
        time_str = f"{row.elapsed:.3f}" if row.elapsed is not None else "-"
        disp_str = f"{row.max_displacement:.4f}" if row.max_displacement is not None else "-"
        print(f"{row.solver:<12}{row.status:<10}{time_str:>10}{disp_str:>15}{(row.verdict or '-'):>12}  {row.note}")
