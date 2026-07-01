from compas_dem.analysis.resolve import resolve_centroidal_displacements
from compas_dem.analysis.resolve import resolve_centroidal_loads
from compas_dem.interactions import FrictionContact
from compas_dem.models import BlockModel
from compas_dem.problem.problem import Problem
from compas_dem.problem.results import Results

try:
    from compas_dprd.prd import DPRDModel
except ImportError:
    raise ImportError("compas_dprd is not installed. Install it locally to use the DPRD solver.")


def dprd_solve(
    problem: Problem,
    model: BlockModel,
    linear: bool = True,
    associative: bool = True,
    non_associative_params: dict = None,
    non_linear_params: dict = None,
    mu: float = None,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> DPRDModel:
    """Translate a Problem into a DPRDModel, run the analysis, and post-process
    results back into the BlockModel in-place.

    Parameters
    ----------
    problem : :class:`~compas_dem.problem.Problem`
    model : :class:`~compas_dem.models.BlockModel`
    linear : bool, optional
        If ``True`` (default), run the one-shot linear solve.
    associative : bool, optional
        If ``True`` (default), use associative Coulomb friction.
    non_associative_params : dict, optional
        Parameters for non-associative friction.
        Keys: ``mu``, ``betta``, ``xi``, ``gamma``, ``c_0k``, ``tol``, ``max_iter``.
    non_linear_params : dict, optional
        Parameters for the incremental nonlinear solve (``linear=False``).
        ``{nsteps: 80, open_tol: 1e-3}``
    mu : float, optional
        Friction coefficient. Falls back to the contact model's ``mu``.
    solver : str, optional
        CVXPY back-end solver. Default ``"CLARABEL"``.
    verbose : bool, optional
        Print solver output. Default ``False``.

    Returns
    -------
    :class:`compas_dprd.prd.DPRDModel`
    """
    if mu is None:
        if problem.contact_properties.contact_model:
            mu = problem.contact_properties.contact_model.mu
        else:
            raise ValueError("No friction coefficient defined. Add a contact model via problem.add_contact_model('MohrCoulomb', mu=...) before solving.")

    dprd = DPRDModel(model)

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------
    centroidal_loads = resolve_centroidal_loads(problem, model)
    loads = []
    for idx, entry in centroidal_loads.items():
        f = entry["force"]
        m = entry["moment"]
        if abs(f.x) > 1e-12:
            loads.append(["fx", idx, float(f.x)])
        if abs(f.y) > 1e-12:
            loads.append(["fy", idx, float(f.y)])
        if abs(f.z) > 1e-12:
            loads.append(["fz", idx, float(f.z)])
        if abs(m.x) > 1e-12:
            loads.append(["mx", idx, float(m.x)])
        if abs(m.y) > 1e-12:
            loads.append(["my", idx, float(m.y)])
        if abs(m.z) > 1e-12:
            loads.append(["mz", idx, float(m.z)])
    if loads:
        dprd.set_force(loads)

    # ------------------------------------------------------------------
    # Displacement BCs (nonlinear only)
    # ------------------------------------------------------------------
    if not linear:
        centroidal_displacements = resolve_centroidal_displacements(problem)
        disps = []
        for idx, entry in centroidal_displacements.items():
            t = entry.get("translation") or [0.0, 0.0, 0.0]
            r = entry.get("rotation") or [0.0, 0.0, 0.0]
            t = [v if v is not None else 0.0 for v in t]
            r = [v if v is not None else 0.0 for v in r]
            disps.append((idx, t + r))
        if disps:
            dprd.set_displacement_bc(disps)

    # ------------------------------------------------------------------
    # Non-associative parameters
    # ------------------------------------------------------------------
    if not associative:
        params = non_associative_params or {}
        dprd.set_non_associative_params(**params)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    nl = non_linear_params or {}
    _nsteps = nl.get("nsteps", 80)
    _open_tol = nl.get("open_tol", 1e-3)

    if linear and associative:
        cvx_result = dprd.solve_static_associative(solver=solver, mu=mu, verbose=verbose)
    elif linear and not associative:
        cvx_result = dprd.solve_static_non_associative(solver=solver, verbose=verbose)
    elif not linear and associative:
        cvx_result = dprd.solve_static_associative_non_linear(nsteps=_nsteps, mu=mu, open_tol=_open_tol, solver=solver, verbose=verbose)
    else:
        cvx_result = dprd.solve_static_non_associative_non_linear(nsteps=_nsteps, open_tol=_open_tol, solver=solver, verbose=verbose)

    dprd.post_process_results()

    dprd.name = "DPRD"
    results = _post_processing_dprd(dprd, problem, model)
    results.metadata["mu"] = mu
    results.metadata["solver_status"] = cvx_result.status
    return results


# def _post_processing_dprd_graph(dprd: DPRDModel, model: BlockModel) -> None:
#     """Write DPRDModel results directly onto the BlockModel graph (in-place). Kept for reference."""
#     graph = model.graph
#     for block in model.elements():
#         T = graph.node_attribute(block.graphnode, "transform")
#         if T is not None:
#             graph.node_attribute(block.graphnode, "transformation", T)
#     for edge in graph.edges():
#         contacts = graph.edge_attribute(edge, "contacts")
#         if not contacts:
#             continue
#         fc: FrictionContact = contacts[0]
#         total_fn = float(sum(f["c_np"] for f in fc.forces))
#         n_pts = len(fc.points)
#         resultant = fc.resultantforce
#         force_vec = list(resultant) if resultant is not None else [0.0, 0.0, 0.0]
#         graph.edge_attribute(edge, "contact_polygon", fc.polygon)
#         graph.edge_attribute(edge, "contact_data", fc)
#         graph.edge_attribute(edge, "force_magnitude", total_fn)
#         graph.edge_attribute(edge, "force", force_vec)
#         graph.edge_attribute(edge, "face_contact", n_pts >= 3)
#         graph.edge_attribute(edge, "edge_contact", n_pts == 2)
#         graph.edge_attribute(edge, "point_contact", n_pts == 1)


def _post_processing_dprd(dprd: DPRDModel, problem: Problem, model: BlockModel) -> Results:
    """Build a standalone :class:`~compas_dem.problem.Results` from DPRD solver output.

    Does **not** mutate the model or its graph.

    Returns
    -------
    :class:`~compas_dem.problem.Results`
    """
    results = Results(model_id=str(model.guid), problem_id=str(problem.guid))
    graph = model.graph

    for block in model.elements():
        T = graph.node_attribute(block.graphnode, "transform")
        if T is not None:
            results.set_node(block.graphnode, "transformation", T)

    for edge in graph.edges():
        contacts = graph.edge_attribute(edge, "contacts")
        if not contacts:
            continue

        fc: FrictionContact = contacts[0]
        n_pts = len(fc.points)

        resultant_lines = fc.resultantforce
        if resultant_lines:
            force_vec = list(resultant_lines[0].vector)
            force_mag = float(resultant_lines[0].vector.length)
        else:
            force_vec = [0.0, 0.0, 0.0]
            force_mag = 0.0

        results.set_edge(edge, "contact_polygon", fc.polygon)
        results.set_edge(edge, "contact_data", fc)
        results.set_edge(edge, "force_magnitude", force_mag)
        results.set_edge(edge, "force", force_vec)
        results.set_edge(edge, "face_contact", n_pts >= 3)
        results.set_edge(edge, "edge_contact", n_pts == 2)
        results.set_edge(edge, "point_contact", n_pts == 1)
        results.set_edge(edge, "contact_point", [list(p) for p in fc.points])
        results.set_edge(edge, "force_normal", [f["c_np"] - f["c_nn"] for f in fc.forces])
        results.set_edge(edge, "force_tangent1", [f["c_u"] for f in fc.forces])
        results.set_edge(edge, "force_tangent2", [f["c_v"] for f in fc.forces])
        if fc.frame is not None:
            results.set_edge(edge, "contact_frame", fc.frame)

    return results
