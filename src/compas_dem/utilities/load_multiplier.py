from typing import Optional

from compas_dem.models import BlockModel
from compas_dem.problem.problem import Problem
from compas_dem.problem.solvers import Solver


def find_load_multiplier(
    problem: Problem,
    model: BlockModel,
    solver: Solver,
    live_loads: list,
    lam_start: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 30,
    verbose: bool = True,
):
    """Find the maximum load multiplier λ via bisection.

    Dead loads already defined on the problem (gravity, point loads, etc.) are
    held constant throughout.  The live loads are scaled by λ at each iteration
    and added on top before solving, then removed so the problem is unchanged
    between calls.

    Parameters
    ----------
    problem : :class:`~compas_dem.problem.Problem`
        Problem with dead loads, supports, and contact model already defined.
    model : :class:`~compas_dem.models.BlockModel`
        The block model to solve against.
    solver : :class:`~compas_dem.problem.solvers.Solver`
        Configured solver instance (``Solver.PRD(...)`` or ``Solver.DPRD(...)``).
    live_loads : list
        Live load definitions as ``[[block_index, [fx, fy, fz]], ...]``.
        Each entry is scaled by λ before being passed to the solver.
    lam_start : float, optional
        Initial upper bracket for λ.  Doubled repeatedly until infeasible.
        Default ``1.0``.
    tol : float, optional
        Bisection convergence tolerance on λ.  Default ``1e-3``.
    max_iter : int, optional
        Maximum number of bisection iterations.  Default ``30``.
    verbose : bool, optional
        Print progress.  Default ``True``.

    Returns
    -------
    lam_max : float
        Last feasible load multiplier.
    result : :class:`~compas_dem.problem.Results` or ``None``
        The Results object at ``lam_max``.

    Examples
    --------
    >>> lam, result = find_load_multiplier(
    ...     problem,
    ...     model,
    ...     Solver.DPRD(),
    ...     live_loads=[[10, [0, 0, -1000]]],
    ... )
    >>> print(f"λ_max = {lam:.4f}")
    """
    if solver.name == "LMGC90":
        raise ValueError(
            "find_load_multiplier does not support LMGC90. "
            "With LMGC90, increase the live load incrementally and monitor the unbalanced residual force (URF) — "
            "the load at which the URF stops converging is your effective capacity."
        )
    if solver.name in ("CRA", "RBE"):
        raise ValueError(
            "find_load_multiplier does not support CRA/RBE. These are penalty-based solvers with no feasibility status — use PRD or DPRD for load multiplier analysis."
        )

    _FEASIBLE_STATUSES = {"optimal", "optimal_inaccurate"}

    def _apply_live_loads(lam: float) -> int:
        """Inject λ-scaled live loads; return the snapshot index to restore from."""
        bc = problem._boundary_conditions
        n_before = len(bc._point_loads)
        for block_idx, force in live_loads:
            scaled = [f * lam for f in force]
            problem.add_point_load(block_idx, scaled)
        return n_before

    def _restore_loads(n_before: int) -> None:
        problem._boundary_conditions._point_loads = (
            problem._boundary_conditions._point_loads[:n_before]
        )

    def _solve_at(lam: float):
        """Returns (feasible, result). Cleans up live loads regardless of outcome."""
        n_before = _apply_live_loads(lam)
        problem.solver(solver)
        try:
            result = model.solve(problem)
        except Exception as exc:
            _restore_loads(n_before)
            if verbose:
                print(f"    solver raised: {exc}")
            return False, None
        _restore_loads(n_before)

        if result is None:
            return False, None
        feasible = result.metadata.get("solver_status") in _FEASIBLE_STATUSES
        return feasible, result if feasible else None

    # ------------------------------------------------------------------
    # Phase 1 — find an upper bracket that is definitely infeasible
    # ------------------------------------------------------------------
    lam_lo = 0.0
    lam_hi = lam_start

    if verbose:
        print("Finding upper bracket...")

    while True:
        feasible, _ = _solve_at(lam_hi)
        if not feasible:
            break
        lam_hi *= 2.0
        if verbose:
            print(f"  Extending bracket → lam_hi = {lam_hi:.4f}")

    if verbose:
        print(f"Bracket confirmed: [{lam_lo:.4f}, {lam_hi:.4f}]\n")

    # ------------------------------------------------------------------
    # Phase 2 — bisect
    # ------------------------------------------------------------------
    last_feasible_result = None

    for i in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        feasible, result = _solve_at(lam_mid)

        if feasible:
            lam_lo = lam_mid
            last_feasible_result = result
        else:
            lam_hi = lam_mid

        if verbose:
            status_str = "FEASIBLE  " if feasible else "INFEASIBLE"
            print(
                f"  iter {i + 1:2d}: λ = {lam_mid:.5f}  →  {status_str}   bracket = [{lam_lo:.5f}, {lam_hi:.5f}]"
            )

        if (lam_hi - lam_lo) < tol:
            break

    lam_max = lam_lo

    if verbose:
        print(f"\nλ_max ≈ {lam_max:.5f}")

    return lam_max, last_feasible_result
