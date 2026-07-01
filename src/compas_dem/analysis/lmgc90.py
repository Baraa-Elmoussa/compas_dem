from collections import defaultdict

import numpy as np

import compas.geometry as cg
from compas_dem.analysis.resolve import _element_mass
from compas_dem.analysis.resolve import resolve_centroidal_displacements
from compas_dem.analysis.resolve import resolve_centroidal_loads
from compas_dem.interactions import EdgeContact
from compas_dem.interactions import FrictionContact
from compas_dem.models import BlockModel
from compas_dem.problem.problem import Problem
from compas_dem.problem.results import Results

try:
    from compas_lmgc90.solver import Solver
except (ImportError, FileNotFoundError):
    raise ImportError("compas_lmgc90 is not installed. Install it to use the LMGC90 solver.")


# ---------------------------------------------------------------------------
# UFR – Unbalanced Force Ratio
# ---------------------------------------------------------------------------


def compute_urf(solver: Solver, problem: Problem, model: BlockModel) -> float:
    """Return the Unbalanced Force Ratio for the current simulation step.

    UFR = Σ_i |F_i_net| / ( Σ_i |F_i_applied| + Σ_ij |F_ij_contact| )
    """
    result = solver.last_result
    blocks = {b.graphnode: b for b in model.elements()}
    lmgc_to_graphnode = {i: el.graphnode for i, el in enumerate(model.elements())}

    g_vec = np.array([0.0, 0.0, -9.81])
    centroidal_loads = resolve_centroidal_loads(problem, model)

    applied_forces = {}
    for idx, block in blocks.items():
        external = np.asarray(list(centroidal_loads[idx]["force"]), dtype=float)
        gravity = _element_mass(block) * g_vec
        applied_forces[idx] = external + gravity

    contact_net: dict = defaultdict(lambda: np.zeros(3))
    for i in range(len(result.interaction_bodies)):
        cd_id, an_id = result.interaction_bodies[i]
        f = np.asarray(result.interaction_force_global[i], dtype=float)
        contact_net[lmgc_to_graphnode[cd_id - 1]] += f
        contact_net[lmgc_to_graphnode[an_id - 1]] -= f

    numerator = sum(np.linalg.norm(applied_forces.get(idx, np.zeros(3)) + cf) for idx, cf in contact_net.items())
    for idx, af in applied_forces.items():
        if idx not in contact_net:
            numerator += np.linalg.norm(af)

    total_applied = sum(np.linalg.norm(af) for af in applied_forces.values())
    total_contact = sum(result.interaction_force_magnitude[i] for i in range(len(result.interaction_bodies)))
    denominator = total_applied + total_contact

    return 0.0 if denominator == 0.0 else numerator / denominator


def lmgc90_solve(
    problem: Problem,
    model: BlockModel,
    contact_law: str = "IQS_CLB",
    duration: float = None,
    n_steps: int = None,
    dt: float = None,
    theta: float = 0.7,
    urf_threshold: float = None,
    track_block: int = None,
) -> Solver:
    """Translate a Problem into a configured LMGC90 Solver. Run the simulation and
    post-process results back into the BlockModel in-place.

    Parameters
    ----------
    problem : :class:`~compas_dem.problem.Problem`
        The problem containing forces, BCs, and contact properties.
    model : :class:`~compas_dem.models.BlockModel`
        The block model to solve.
    contact_law : str, optional
        LMGC90 contact law identifier. Default ``"IQS_CLB"``.
    duration : float, optional
        Total simulation time [s].
    n_steps : int, optional
        Number of time steps.
    dt : float, optional
        Time step size [s].
    theta : float, optional
        Time-integration parameter. Default ``0.7``.
    urf_threshold : float, optional
        Unbalanced Force Ratio convergence threshold.

    Returns
    -------
    :class:`compas_lmgc90.solver.Solver`
    """
    given = sum(x is not None for x in [duration, n_steps, dt])
    if given == 3:
        raise ValueError("Provide exactly two of duration, n_steps, dt — the third is computed automatically.")
    elif given == 1:
        raise ValueError("Provide exactly two of duration, n_steps, dt.")
    elif given == 0:
        print("No time parameters provided; defaulting to duration=0.5s, n_steps=50.")
        duration, n_steps = 0.5, 50
        dt = duration / n_steps
    else:
        if duration is None:
            duration = dt * n_steps
        elif n_steps is None:
            n_steps = round(duration / dt)
        else:
            dt = duration / n_steps

    # ------------------------------------------------------------------
    # Density: first block with material, or fallback
    # ------------------------------------------------------------------
    density = None
    for block in model.blocks():
        if block.material and block.material.density:
            density = block.material.density
            break

    # ------------------------------------------------------------------
    # Contact friction
    # ------------------------------------------------------------------
    if problem.contact_properties.contact_model:
        mu = problem.contact_properties.contact_model.mu
    else:
        raise Warning("No contact properties with a contact model found in the problem; defaulting to mu=0.6.")

    # ------------------------------------------------------------------
    # Resolve BCs
    # ------------------------------------------------------------------
    centroidal_displacements = resolve_centroidal_displacements(problem)
    centroidal_loads = resolve_centroidal_loads(problem, model)

    # Mark fully-fixed blocks as supports on the model
    for block in model.elements():
        idx = block.graphnode
        disp = centroidal_displacements.get(idx)
        if disp is not None:
            t = disp["translation"] or [0.0, 0.0, 0.0]
            r = disp["rotation"] or [0.0, 0.0, 0.0]
            if all(v == 0.0 for v in t) and all(v == 0.0 for v in r):
                block.is_support = True

    solver = Solver(model, density=density, dt=dt, theta=theta)

    # ------------------------------------------------------------------
    # Displacement BCs → apply_velocity
    # ------------------------------------------------------------------
    for i, block in enumerate(model.elements()):
        idx = block.graphnode
        disp = centroidal_displacements.get(idx)
        if disp is None:
            continue

        translation = disp["translation"] or [None, None, None]
        rotation = disp["rotation"] or [None, None, None]

        for component, value in zip(["Vx", "Vy", "Vz"], translation):
            if value is not None:
                solver.apply_velocity(
                    block_index=i,
                    component=component,
                    value=np.array([[0.0, duration], [value / duration, 0.0]]),
                )
        for component, value in zip(["Rx", "Ry", "Rz"], rotation):
            if value is not None:
                solver.apply_velocity(
                    block_index=i,
                    component=component,
                    value=np.array([[0.0, duration], [value / duration, 0.0]]),
                )

    # ------------------------------------------------------------------
    # Applied forces → per-axis time series
    # ------------------------------------------------------------------
    t_series = np.array([0.0, duration * 0.98, duration])

    for idx, entry in centroidal_loads.items():
        f = entry["force"]
        m = entry["moment"]
        ramp = entry.get("loading_type", "ramp") == "ramp"

        def _vals(v):
            return [0, v, v] if ramp else [v, v, 0]

        if abs(f.x) > 1e-12:
            solver.apply_force(block_index=idx, component="Fx", value=np.array([t_series, _vals(f.x)]))
        if abs(f.y) > 1e-12:
            solver.apply_force(block_index=idx, component="Fy", value=np.array([t_series, _vals(f.y)]))
        if abs(f.z) > 1e-12:
            solver.apply_force(block_index=idx, component="Fz", value=np.array([t_series, _vals(f.z)]))
        if abs(m.x) > 1e-12:
            solver.apply_force(block_index=idx, component="Mx", value=np.array([t_series, _vals(m.x)]))
        if abs(m.y) > 1e-12:
            solver.apply_force(block_index=idx, component="My", value=np.array([t_series, _vals(m.y)]))
        if abs(m.z) > 1e-12:
            solver.apply_force(block_index=idx, component="Mz", value=np.array([t_series, _vals(m.z)]))

    # ------------------------------------------------------------------
    # Contact law
    # ------------------------------------------------------------------
    solver.contact_law(contact_law, mu)

    solver.preprocess()

    force_time = []
    urf_history = []
    displacement_history = []
    initial_pos = np.array(solver.trimeshes[track_block].centroid()) if track_block is not None else None
    print("Starting LMGC90 solver analysis...")
    for step in range(n_steps):
        if step == 0:
            result = solver.lmgc90.compute_one_step()

            for i, block in enumerate(model.elements()):
                pos = np.array(result.bodies[i])
                rot = np.array(result.body_frames[i]).reshape(3, 3)
                block.init_frame = cg.Frame(pos, rot[0, :], rot[1, :])

            solver._update_meshes(result)
            solver.last_result = result
        else:
            solver.run(nb_steps=1)

        if track_block is not None:
            current_pos = np.array(solver.trimeshes[track_block].centroid())
            displacement_history.append(current_pos - initial_pos)

        if urf_threshold is not None:
            if step % 10 == 0:
                urf = compute_urf(solver, problem, model)
                urf_history.append(urf)
                print(f"Completed step {step}/{n_steps}...  UFR = {urf:.2e}")
                if urf >= 1.0:
                    print(f"Diverged at step {step} (UFR = {urf:.2e} >= 1.0). Stopping.")
                    break

                _jump_window = 200
                _Max_URF_JUMP_FACTOR = 3.5

                if len(urf_history) > _jump_window:
                    baseline = np.mean(urf_history[-_jump_window - 1 : -1])
                    if urf > baseline * _Max_URF_JUMP_FACTOR:
                        print(f"Failure detected at step {step} (UFR jumped from ~{baseline:.2e} to {urf:.2e}). Stopping.")
                        break
                if urf < urf_threshold:
                    print(f"Converged at step {step} (UFR = {urf:.2e} < {urf_threshold:.2e}). Stopping early.")
                    break

        elif step % 10 == 0:
            print(f"Completed step {step}/{n_steps}...")

        if step % 10 == 0:
            result = solver.last_result
            force_time.append([result.interaction_force_magnitude[i] for i in range(len(result.interaction_bodies))])

    solver.force_time = force_time
    solver.urf_history = urf_history
    solver.displacement_history = displacement_history

    print("LMGC90 solver run complete.")
    results = _post_processing_lmgc90(solver, problem, model)
    results.metadata["mu"] = mu
    results.metadata["force_time"] = force_time
    results.metadata["urf_history"] = urf_history
    results.metadata["displacement_history"] = [d.tolist() for d in displacement_history]

    solver.name = "LMGC90"
    solver.finalize()

    return results


# def _post_processing_lmgc90(solver: "Solver", model: BlockModel) -> None:
#     """Write LMGC90 results directly onto the BlockModel graph (in-place).

#     This function mutates the model. It is kept for viewer compatibility.
#     For a standalone serializable result use :func:`_post_processing_lmgc90_results`.

#     Block attributes set
#     --------------------
#     transformation : :class:`compas.geometry.Transformation`

#     Edge attributes set
#     -------------------
#     contact_point, force_magnitude, force_vector, contact_polygon, gap, status,
#     contact_data, face_contact, edge_contact, point_contact
#     """
#     elements = list(model.elements())
#     contact_data = solver.get_contacts()
#     result = solver.last_result
#     graph = model.graph

#     for i, block in enumerate(elements):
#         pos = np.array(result.bodies[i])
#         rot = np.array(result.body_frames[i]).reshape(3, 3)
#         new_frame = cg.Frame(pos, rot[0, :], rot[1, :])
#         T = cg.Transformation.from_frame_to_frame(block.init_frame, new_frame)
#         graph.node_attribute(block.graphnode, "transformation", T)

#     _per_point_keys = [
#         "contact_points",
#         "force_normal",
#         "force_tangent1",
#         "force_tangent2",
#         "gaps",
#         "status",
#     ]
#     _new_key_name = [
#         "contact_point",
#         "force_normal",
#         "force_tangent1",
#         "force_tangent2",
#         "gap",
#         "status",
#     ]

#     contact_groups = {}
#     for i in range(len(solver.last_result.interaction_coords)):
#         body_pair = tuple(sorted(b - 1 for b in solver.last_result.interaction_bodies[i]))
#         if body_pair not in contact_groups:
#             contact_groups[body_pair] = []
#         contact_groups[body_pair].append(i)

#     Added_Edges = 0
#     for pair, points in contact_groups.items():
#         u, v = pair

#         if not points:
#             continue

#         if graph.has_edge((u, v)):
#             edge = (u, v)
#         elif graph.has_edge((v, u)):
#             edge = (v, u)
#         else:
#             if not graph.has_node(u) or not graph.has_node(v):
#                 continue
#             graph.add_edge(u, v)
#             edge = (u, v)
#             Added_Edges += 1

#         graph.edge_attribute(edge, "face_contact", False)
#         graph.edge_attribute(edge, "point_contact", False)
#         graph.edge_attribute(edge, "edge_contact", False)

#         for k, name in zip(_per_point_keys, _new_key_name):
#             graph.edge_attribute(edge, name, [contact_data[k][i] for i in points])
#         graph.edge_attribute(
#             edge,
#             "force_magnitude",
#             float(np.linalg.norm(np.sum([result.interaction_force_magnitude[p] for p in points], axis=0))),
#         )
#         graph.edge_attribute(edge, "force_vector", [list(result.interaction_force_global[p]) for p in points])
#         graph.edge_attribute(edge, "force", np.sum([result.interaction_force_global[p] for p in points], axis=0).tolist())

#         contact_frames = [
#             cg.Frame(
#                 point=cg.Point(*result.interaction_coords[p]),
#                 xaxis=cg.Vector(*result.interaction_tangent1[p]),
#                 yaxis=cg.Vector(*result.interaction_normals[p]),
#             )
#             for p in points
#         ]

#         graph.edge_attribute(edge, "contact_frame", contact_frames[0])

#         polygon_pts = [result.interaction_coords[p] for p in points]
#         contact_pts = graph.edge_attribute(edge, "contact_point")
#         contact_frames = graph.edge_attribute(edge, "contact_frame")

#         if len(polygon_pts) >= 3:
#             graph.edge_attribute(edge, "contact_polygon", cg.Polygon(polygon_pts))
#             graph.edge_attribute(edge, "face_contact", True)
#             fc = FrictionContact(points=[cg.Point(*p) for p in contact_pts])
#             lmgc_tangent = cg.Vector(*result.interaction_tangent1[points[0]])
#             lmgc_normal = cg.Vector(*result.interaction_normals[points[0]])
#             lmgc_tangent2 = lmgc_normal.cross(lmgc_tangent).unitized()
#             fc._frame = cg.Frame(contact_frames.point, lmgc_tangent, lmgc_tangent2)

#             for p in points:
#                 Ft, Fn, Fs = result.interaction_rloc[p]
#                 fc.forces.append({"c_np": max(Fn, 0), "c_nn": max(-Fn, 0), "c_u": -Ft, "c_v": -Fs})
#             graph.edge_attribute(edge, "contact_data", fc)

#         elif len(polygon_pts) == 2:
#             print(f"Edge contact between bodies {u} and {v} with contact points {polygon_pts}. Setting edge_contact=True.")
#             graph.edge_attribute(edge, "edge_contact", True)

#             lmgc_tangent = cg.Vector(*result.interaction_tangent1[points[0]])
#             lmgc_tangent2 = cg.Vector(*result.interaction_tangent2[points[0]])

#             ec = EdgeContact(
#                 points=[cg.Point(*p) for p in polygon_pts],
#                 frame=cg.Frame(
#                     cg.Line(cg.Point(*polygon_pts[0]), cg.Point(*polygon_pts[1])).midpoint,
#                     lmgc_tangent,
#                     lmgc_tangent2,
#                 ),
#             )
#             for p in points:
#                 Ft, Fn, Fs = result.interaction_rloc[p]
#                 ec.forces.append({"c_np": max(Fn, 0), "c_nn": max(-Fn, 0), "c_u": -Ft, "c_v": -Fs})
#             graph.edge_attribute(edge, "edge_contact", True)
#             graph.edge_attribute(edge, "contact_data", ec)

#         elif len(polygon_pts) == 1:
#             graph.edge_attribute(edge, "point_contact", True)

#         else:
#             print(f"Warning: contact between bodies {u} and {v} has no contact points.")

#         contact_frames = [
#             cg.Frame(
#                 result.interaction_coords[p],
#                 result.interaction_tangent1[p],
#                 result.interaction_normals[p],
#             )
#             for p in points
#         ]
#         graph.edge_attribute(edge, "contact_frames", contact_frames)

#     if Added_Edges > 0:
#         print(f"Added {Added_Edges} edges to the model graph to account for contacts without existing edges.")


def _post_processing_lmgc90(solver: "Solver", problem: Problem, model: BlockModel) -> Results:
    """Build a standalone :class:`~compas_dem.problem.Results` from LMGC90 solver output.

    Does **not** mutate the model or its graph. All data is written only to the
    returned :class:`Results` object, which can be serialized independently via
    ``compas.json_dump``.

    Parameters
    ----------
    solver : :class:`compas_lmgc90.solver.Solver`
    problem : :class:`~compas_dem.problem.Problem`
    model : :class:`~compas_dem.models.BlockModel`

    Returns
    -------
    :class:`~compas_dem.problem.Results`
    """
    results = Results(model_id=str(model.guid), problem_id=str(problem.guid))

    elements = list(model.elements())
    contact_data = solver.get_contacts()
    result = solver.last_result

    # ------------------------------------------------------------------
    # Node data — block transformations
    # ------------------------------------------------------------------
    for i, block in enumerate(elements):
        pos = np.array(result.bodies[i])
        rot = np.array(result.body_frames[i]).reshape(3, 3)
        new_frame = cg.Frame(pos, rot[0, :], rot[1, :])
        T = cg.Transformation.from_frame_to_frame(block.init_frame, new_frame)
        results.set_node(block.graphnode, "transformation", T)

    _per_point_keys = [
        "contact_points",
        "force_normal",
        "force_tangent1",
        "force_tangent2",
        "gaps",
        "status",
    ]
    _new_key_name = [
        "contact_point",
        "force_normal",
        "force_tangent1",
        "force_tangent2",
        "gap",
        "status",
    ]

    # Group solver contact indices by body pair
    contact_groups: dict[tuple, list] = {}
    for i in range(len(result.interaction_coords)):
        body_pair = tuple(sorted(b - 1 for b in result.interaction_bodies[i]))
        if body_pair not in contact_groups:
            contact_groups[body_pair] = []
        contact_groups[body_pair].append(i)

    # ------------------------------------------------------------------
    # Edge data — contacts
    # ------------------------------------------------------------------
    graph = model.graph
    for pair, points in contact_groups.items():
        u, v = pair

        if not points:
            continue

        # Determine canonical edge direction from the graph (read-only).
        # If neither direction exists, record it so model.attach() can add it.
        if graph.has_edge((u, v)):
            edge = (u, v)
        elif graph.has_edge((v, u)):
            edge = (v, u)
        else:
            edge = (u, v)
            results.metadata.setdefault("added_edges", []).append(list(edge))

        results.set_edge(edge, "face_contact", False)
        results.set_edge(edge, "point_contact", False)
        results.set_edge(edge, "edge_contact", False)

        for k, name in zip(_per_point_keys, _new_key_name):
            results.set_edge(edge, name, [contact_data[k][i] for i in points])

        results.set_edge(
            edge,
            "force_magnitude",
            float(np.linalg.norm(np.sum([result.interaction_force_magnitude[p] for p in points], axis=0))),
        )
        results.set_edge(edge, "force_vector", [list(result.interaction_force_global[p]) for p in points])
        results.set_edge(edge, "force", np.sum([result.interaction_force_global[p] for p in points], axis=0).tolist())

        contact_frames = [
            cg.Frame(
                point=cg.Point(*result.interaction_coords[p]),
                xaxis=cg.Vector(*result.interaction_tangent1[p]),
                yaxis=cg.Vector(*result.interaction_normals[p]),
            )
            for p in points
        ]
        results.set_edge(edge, "contact_frame", contact_frames[0])

        polygon_pts = [result.interaction_coords[p] for p in points]
        # Read from results (not graph) — these were just written above
        contact_frame_0 = results._edge_data[f"{edge[0]},{edge[1]}"]["contact_frame"]

        if len(polygon_pts) >= 3:
            results.set_edge(edge, "contact_polygon", cg.Polygon(polygon_pts))
            results.set_edge(edge, "face_contact", True)
            fc = FrictionContact(points=[cg.Point(*p) for p in polygon_pts])
            lmgc_tangent = cg.Vector(*result.interaction_tangent1[points[0]])
            lmgc_normal = cg.Vector(*result.interaction_normals[points[0]])
            lmgc_tangent2 = lmgc_normal.cross(lmgc_tangent).unitized()
            fc._frame = cg.Frame(contact_frame_0.point, lmgc_tangent, lmgc_tangent2)
            for p in points:
                Ft, Fn, Fs = result.interaction_rloc[p]
                fc.forces.append({"c_np": max(Fn, 0), "c_nn": max(-Fn, 0), "c_u": -Ft, "c_v": -Fs})
            results.set_edge(edge, "contact_data", fc)

        elif len(polygon_pts) == 2:
            print(f"Edge contact between bodies {u} and {v} with contact points {polygon_pts}. Setting edge_contact=True.")
            lmgc_tangent = cg.Vector(*result.interaction_tangent1[points[0]])
            lmgc_tangent2 = cg.Vector(*result.interaction_tangent2[points[0]])
            ec = EdgeContact(
                points=[cg.Point(*p) for p in polygon_pts],
                frame=cg.Frame(
                    cg.Line(cg.Point(*polygon_pts[0]), cg.Point(*polygon_pts[1])).midpoint,
                    lmgc_tangent,
                    lmgc_tangent2,
                ),
            )
            for p in points:
                Ft, Fn, Fs = result.interaction_rloc[p]
                ec.forces.append({"c_np": max(Fn, 0), "c_nn": max(-Fn, 0), "c_u": -Ft, "c_v": -Fs})
            results.set_edge(edge, "edge_contact", True)
            results.set_edge(edge, "contact_data", ec)

        elif len(polygon_pts) == 1:
            results.set_edge(edge, "point_contact", True)

        else:
            print(f"Warning: contact between bodies {u} and {v} has no contact points.")


    return results
