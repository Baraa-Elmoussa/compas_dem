import numpy as np
from compas_viewer.config import Config
from compas_viewer.scene import ViewerSceneObject
from compas_viewer.viewer import Viewer

import compas.geometry as cg
from compas.colors import Color
from compas.scene import Group
from compas_dem.models import BlockModel
from compas_dem.problem.results import Results

config = Config()


def show_blocks():
    from compas_viewer import Viewer

    viewer: DEMViewer = Viewer()  # type: ignore

    viewer.groups["supports"].show = True
    viewer.groups["blocks"].show = True
    viewer.groups["contacts"].show = False
    viewer.groups["interactions"].show = False

    obj: ViewerSceneObject

    for obj in viewer.groups["supports"].children:
        obj.show_faces = True
        obj.update()

    for obj in viewer.groups["blocks"].children:
        obj.show_faces = True
        obj.update()

    viewer.ui.sidebar.update()
    viewer.renderer.update()


def show_contacts():
    from compas_viewer import Viewer

    viewer: DEMViewer = Viewer()  # type: ignore

    viewer.groups["supports"].show = True
    viewer.groups["blocks"].show = True
    viewer.groups["contacts"].show = True
    viewer.groups["interactions"].show = False

    obj: ViewerSceneObject

    for obj in viewer.groups["supports"].children:
        obj.show_faces = False
        obj.update()

    for obj in viewer.groups["blocks"].children:
        obj.show_faces = False
        obj.update()

    viewer.ui.sidebar.update()
    viewer.renderer.update()


def show_interactions():
    from compas_viewer import Viewer

    viewer: DEMViewer = Viewer()  # type: ignore

    viewer.groups["supports"].show = True
    viewer.groups["blocks"].show = True
    viewer.groups["contacts"].show = False
    viewer.groups["interactions"].show = True

    obj: ViewerSceneObject

    for obj in viewer.groups["supports"].children:
        obj.show_faces = False
        obj.update()

    for obj in viewer.groups["blocks"].children:
        obj.show_faces = False
        obj.update()

    viewer.ui.sidebar.update()
    viewer.renderer.update()


config.ui.menubar.items.append(
    {
        "title": "COMPAS DEM",
        "items": [
            {
                "title": "Show Blocks",
                "action": show_blocks,
            },
            {
                "title": "Show Contacts",
                "action": show_contacts,
            },
            {
                "title": "Show Interactions",
                "action": show_interactions,
            },
        ],
    }
)


class DEMViewer(Viewer):
    """Viewer for COMPAS DEM models.

    Parameters
    ----------
    model : :class:`compas_dem.models.BlockModel`
        The DEM model to visualize.
    config : :class:`compas_viewer.config.Config`, optional
        The viewer configuration. If not provided, a default configuration is used.

    Methods
    -------
    setup()
        Sets up the viewer by creating groups and adding geometry for blocks, supports, contacts, and interactions.
    add_solution(scale=1.0)
        Adds the solution to the viewer, including updated block positions, resultant forces at contacts, and support reactions.
        The `scale` parameter can be adjusted to make force vectors visible based on their magnitude relative to the block geometry.

    """

    blockcolor: Color = Color.grey().lightened(85)
    supportcolor: Color = Color.red().lightened(50)
    interfacecolor: Color = Color.cyan().lightened(50)
    graphnodecolor: Color = Color.blue()
    graphedgecolor: Color = Color.cyan().lightened(50)

    def __init__(self, model: BlockModel, config=config):
        super().__init__(config=config)
        self.model = model
        self.groups = {}

    # def add_formdiagram(self, formdiagram: FormDiagram, maxradius=50, minradius=10):
    #     formgroup = self.scene.add_group(name="FormDiagram")
    #     formgroup.add(formdiagram.viewmesh, facecolor=Color.magenta(), name="Diagram")  # type: ignore

    #     group = self.scene.add_group(name="Supports", parent=formgroup)
    #     for vertex in formdiagram.vertices_where(is_support=True):
    #         group.add(formdiagram.vertex_point(vertex), pointsize=10, pointcolor=Color.red())  # type: ignore

    #     fmax = max(formdiagram.edges_attribute("_f"))  # type: ignore
    #     pipes = []
    #     for edge in formdiagram.edges():
    #         force = formdiagram.edge_attribute(edge, "_f")
    #         radius = maxradius * force / fmax  # type: ignore
    #         if radius > minradius:
    #             cylinder = Cylinder.from_line_and_radius(formdiagram.edge_line(edge), radius)
    #             pipes.append(cylinder)

    #     group = self.scene.add_group(name="Pipes", parent=formgroup)
    #     group.add_from_list(pipes, surfacecolor=Color.blue())  # type: ignore

    # def add_forcediagram(self, forcediagram):
    #     self.scene.add(forcediagram, show_faces=False, show_lines=True)

    def setup(self):
        self._setup_groups()

        # add stuff
        self._add_supports()
        self._add_blocks()
        self._add_contacts()
        self._add_graph()

    # =============================================================================
    # Groups
    # =============================================================================

    def _setup_groups(self):
        self.groups["model"] = self.scene.add_group(name="Model")
        self.groups["supports"] = self.scene.add_group(
            name="Supports", parent=self.groups["model"]
        )
        self.groups["blocks"] = self.scene.add_group(
            name="Blocks", parent=self.groups["model"]
        )
        self.groups["contacts"] = self.scene.add_group(
            name="Contacts", parent=self.groups["model"], show=False
        )
        self.groups["interactions"] = self.scene.add_group(
            name="Interactions", parent=self.groups["model"], show=False
        )

    # =============================================================================
    # Blocks and Contacts
    # =============================================================================

    def _add_supports(self):
        parent: Group = self.groups["supports"]

        for block in self.model.supports():
            parent.add(
                block.modelgeometry,
                facecolor=self.supportcolor,  # type: ignore
                edgecolor=self.supportcolor.contrast,
                linewidth=0.5,  # type: ignore
                name=f"Block {block.graphnode}",  # type: ignore
            )

    def _add_blocks(self):
        parent: Group = self.groups["blocks"]

        for block in self.model.blocks():
            parent.add(
                block.modelgeometry,
                facecolor=self.blockcolor,  # type: ignore
                edgecolor=self.blockcolor.contrast,
                linewidth=0.5,  # type: ignore
                name=f"Block {block.graphnode}",  # type: ignore
            )

    def _add_contacts(self):
        parent: Group = self.groups["contacts"]

        for contact in self.model.contacts():
            geometry = contact.polygon
            color = self.interfacecolor
            parent.add(
                geometry, linewidth=1, surfacecolor=color, linecolor=color.contrast
            )  # type: ignore

    # =============================================================================
    # Graph
    # =============================================================================

    def _add_graph(self):
        parent: Group = self.groups["interactions"]

        node_point = {
            node: self.model.graph.node_element(node).point
            for node in self.model.graph.nodes()
        }  # type: ignore
        points = list(node_point.values())
        lines = [
            cg.Line(node_point[u], node_point[v]) for u, v in self.model.graph.edges()
        ]

        nodegroup = self.scene.add_group(name="Nodes", parent=parent)  # type: ignore
        edgegroup = self.scene.add_group(name="Edges", parent=parent)  # type: ignore

        nodegroup.add_from_list(points, pointsize=10, pointcolor=self.graphnodecolor)  # type: ignore
        edgegroup.add_from_list(lines, linewidth=1, linecolor=self.graphedgecolor)  # type: ignore

    def add_solution(
        self, results: Results, name: str = "Solution", scale: float = 1.0
    ):
        """Add a solved :class:`~compas_dem.problem.Results` to the viewer.

        Parameters
        ----------
        results : :class:`~compas_dem.problem.Results`
            The results object returned by ``model.solve(problem)``.
        name : str, optional
            Label for this solution's group in the scene tree. Default ``"Solution"``.
        scale : float, optional
            Scaling factor for force vectors. Adjust to match force magnitudes to geometry.
        """
        try:
            from compas_viewer.components import Treeform

            self.ui.sidebar.show_objectsetting = False
            info_treeform = Treeform()
            self.ui.sidebar.add(info_treeform)

            def on_scene_selected(_, node):
                data = node.attributes.get("data")
                info_treeform.update_from_dict(data or {})

            self.ui.sidebar.sceneform.action = on_scene_selected
        except Exception:
            print(
                "Warning: compas_viewer.components.Treeform not available. Sidebar info panel will not be displayed."
            )
            pass

        moved_blocks = []

        solution_group = self.scene.add_group(name=name)
        updated_blocks = self.scene.add_group(
            name="Updated_Blocks", parent=solution_group
        )
        resultant_forces = self.scene.add_group(name="Forces", parent=solution_group)
        face_contacts = self.scene.add_group(
            name="Contact_Polygons", parent=solution_group
        )
        edge_contacts = self.scene.add_group(
            name="Contact_Edges", parent=solution_group
        )
        supports_group = self.scene.add_group(name="Supports", parent=solution_group)
        reactions = self.scene.add_group(name="Reactions", parent=supports_group)
        support_contacts = self.scene.add_group(
            name="Support_Contacts", parent=supports_group
        )
        point_results = self.scene.add_group(
            name="Point Results : [Fn, Ft1, Ft2]", parent=solution_group
        )
        degenerate_contacts = self.scene.add_group(
            name="Degenerate_Contacts", parent=solution_group
        )
        interface_contacts = self.scene.add_group(
            name="Interface_Contacts", parent=solution_group
        )
        thrust_points = self.scene.add_group(
            name="Thrust_Points", parent=solution_group
        )

        block_ln = []
        for block in self.model.elements():
            T = results.transformation(block.graphnode) or cg.Transformation()
            new_block = block.modelgeometry.transformed(T)
            moved_blocks.append(new_block)
            tx, ty, tz = T.translation_vector

            facecolor = self.supportcolor if block.is_support else self.blockcolor
            obj = updated_blocks.add(
                new_block,
                name=f"block_{block.graphnode}",
                facecolor=facecolor,
                opacity=0.25,
            )
            obj.attributes["data"] = {
                "Block": {
                    "graphnode": block.graphnode,
                    "is_support": block.is_support,
                },
                "Position": {
                    "x": round(block.point.x, 4),
                    "y": round(block.point.y, 4),
                    "z": round(block.point.z, 4),
                },
                "Displacement": {
                    "dx": round(tx, 6),
                    "dy": round(ty, 6),
                    "dz": round(tz, 6),
                },
            }

            try:
                block_ln.append(block.modelgeometry.edge_length([0, 1]))
            except Exception:
                pass

        # Joined-blocks aggregate interfaces (masonry_dem model.save_joined):
        # one resultant per body-body interface replaces the per-member-contact
        # arrows; member polygons/lines are still drawn individually.  Only a
        # masonry_dem extract_results with save_joined=True writes this
        # metadata key — for every other solver the list is empty, the member
        # set is empty, and every code path below is byte-identical to before.
        body_interfaces = results.metadata.get("body_interfaces") or []
        interface_member_edges = set()
        for itf in body_interfaces:
            for u, v in itf["edges"]:
                interface_member_edges.add((u, v))
                interface_member_edges.add((v, u))

        forces = [
            np.array(results.force(edge) or [0, 0, 0]) for edge in results.edges()
        ]
        forces += [np.array(itf["force"]) for itf in body_interfaces]
        max_force = max(np.linalg.norm(f) for f in forces) if forces else 0.0
        block_scale = scale * max(block_ln) / max_force if max_force > 0 else 1.0

        face_contact_edges = results.face_contact_edges()
        edge_contact_edges = results.edge_contact_edges()

        # =============================================================================
        # Supports and reactions
        # =============================================================================
        support_nodes = set(support.graphnode for support in self.model.supports())
        support_edges = {s: [] for s in support_nodes}

        for edge in results.edges():
            u, v = edge
            if u in support_nodes:
                support_edges[u].append(edge)
            if v in support_nodes:
                support_edges[v].append(edge)

        for support_node, edges in support_edges.items():
            point_forces: list[tuple[cg.Point, cg.Vector]] = []
            support_block = self.model.graph.node_element(support_node)
            support_point = support_block.point

            for edge in edges:
                if edge in face_contact_edges:
                    fc = results.contact_data(edge)

                    if fc is None:
                        continue

                    resultant = fc.resultantline()
                    if resultant is None:
                        continue

                    from_support = cg.Vector.from_start_end(
                        support_point, resultant.midpoint
                    )
                    if resultant.vector.dot(from_support) < 0:
                        resultant.vector.flip()

                    point_forces.append((fc.resultantpoint, resultant.vector))

                    contact_polygon = results.contact_polygon(edge)

                    if contact_polygon is None:
                        continue

                    if contact_polygon.area < 1e-6:
                        print(
                            f"WARNING:\nContact polygon for support edge {edge} has very small area ({contact_polygon.area:.2e}), skipping visualization. \n"
                        )
                        continue

                    try:
                        support_brep = contact_polygon.to_brep()
                    except Exception as exc:
                        print(
                            f"WARNING:\nSupport contact polygon for edge {edge} could not be "
                            f"meshed ({type(exc).__name__}), skipping visualization. \n"
                        )
                        continue

                    support_contacts.add(
                        support_brep,
                        name=f"contact_polygon_{edge}",
                        color=Color.brown(),
                        opacity=0.5,
                    )

                elif edge in edge_contact_edges:
                    ec = results.contact_data(edge)
                    support_contacts.add(
                        cg.Line(ec.points[0], ec.points[1]),
                        name=f"contact_line_{edge}",
                        linewidth=2,
                        linecolor=Color.brown(),
                    )

                    if ec.resultantline() is None:
                        continue

                    resultant = ec.resultantline()
                    from_support = cg.Vector.from_start_end(
                        support_point, resultant.midpoint
                    )

                    if resultant.vector.dot(from_support) < 0:
                        resultant.vector.flip()
                    point_forces.append((ec.resultantpoint, ec.resultantline().vector))

            if point_forces:
                weights = [f.length for _, f in point_forces]
                total_weight = sum(weights)

                if total_weight > 0:
                    position = cg.Point(
                        sum(p.x * w for (p, _), w in zip(point_forces, weights))
                        / total_weight,
                        sum(p.y * w for (p, _), w in zip(point_forces, weights))
                        / total_weight,
                        sum(p.z * w for (p, _), w in zip(point_forces, weights))
                        / total_weight,
                    )
                    resultant = cg.Vector(0, 0, 0)
                    for _, f in point_forces:
                        if f:
                            resultant += f

                    forcevector = resultant * 0.5
                    p1 = position + forcevector * block_scale
                    p2 = position - forcevector * block_scale
                    obj_reac = reactions.add(
                        cg.Line(p1, p2),
                        name=f"F=({resultant.x:.1f}, {resultant.y:.1f}, {resultant.z:.1f}) \n|F|={resultant.length:.1f}",
                        linewidth=2.5,
                        color=Color.red(),
                    )
                    obj_reac.attributes["data"] = {
                        "Support node": support_node,
                        "Reaction force components": {
                            "Fx": round(resultant.x, 3),
                            "Fy": round(resultant.y, 3),
                            "Fz": round(resultant.z, 3),
                        },
                        "Resultant reaction magnitude": round(resultant.length, 3),
                    }

        # =============================================================================
        # Visualize forces at contacts
        # =============================================================================

        # Face contacts
        # --------------
        for edge in face_contact_edges:
            fc = results.contact_data(edge)
            contact_polygon = results.contact_polygon(edge)

            if fc is None:
                continue
            # A degenerate (sliver) contact can carry no force points, which makes
            # resultantforce divide by zero. Skip such contacts.
            if not getattr(fc, "points", None):
                print(
                    f"WARNING:\nContact for edge {edge} has no force points, skipping visualization. \n"
                )
                continue
            if contact_polygon is None:
                print(
                    f"WARNING:\nFace contact for edge {edge} has no polygon, skipping visualization. \n"
                )
                continue
            resultant = fc.resultantforce[0].vector
            resultant_line = fc.resultantline(scale=block_scale)
            # save_joined: member contacts of an aggregated interface get ONE
            # arrow per interface (drawn below), not one per member contact.
            if edge in interface_member_edges:
                resultant_line = None
            if resultant_line is not None:
                obj = resultant_forces.add(
                    resultant_line,
                    name=f"F=({resultant.x:.1f}, {resultant.y:.1f}, {resultant.z:.1f}) \n|F|={resultant.length:.1f}",
                    linewidth=2.5,
                    linecolor=Color.blue(),
                )
                obj.attributes["data"] = {
                    "Edge": str(edge),
                    "Resultant force": {
                        "Fx": round(resultant.x, 3),
                        "Fy": round(resultant.y, 3),
                        "Fz": round(resultant.z, 3),
                    },
                    "Resultant magnitude": round(resultant.length, 3),
                }

            if contact_polygon.area < 1e-5:
                if len(contact_polygon.points) < 3:
                    print(
                        f"WARNING:\nContact polygon for edge {edge} has less than 3 points, skipping visualization. \n"
                    )
                    continue
                lines_polyg = [line.length for line in contact_polygon.lines]
                line_min = min(lines_polyg)
                line_max = max(lines_polyg)

                if line_max > 0.001 and line_max / line_min > 10:
                    longest_line = contact_polygon.lines[np.argmax(lines_polyg)]
                    degenerate_contacts.add(
                        longest_line,
                        name=f"contact_line_{edge}",
                        linewidth=2,
                        linecolor=Color.red(),
                    )
                elif line_max / line_min <= 10:
                    print(f"Contact_polygon {contact_polygon}")
                    degenerate_contacts.add(
                        contact_polygon.centroid,
                        name=f"contact_point_{edge}",
                        pointsize=5,
                        pointcolor=Color.red(),
                    )
                continue

            # A sliver contact polygon can have area just above the 1e-6 guard
            # yet be too near-collinear for OCC to build a face from (it raises
            # Standard_ConstructionError). Fall back to drawing it as a line
            # rather than aborting the whole visualization.
            try:
                contact_brep = contact_polygon.to_brep()
            except Exception as exc:
                lines_polyg = [line.length for line in contact_polygon.lines]
                longest_line = contact_polygon.lines[int(np.argmax(lines_polyg))]
                print(
                    f"WARNING:\nContact polygon for edge {edge} could not be meshed "
                    f"({type(exc).__name__}); drawing as a line instead. \n"
                )
                degenerate_contacts.add(
                    longest_line,
                    name=f"contact_line_{edge}",
                    linewidth=2,
                    linecolor=Color.red(),
                )
                continue

            obj = face_contacts.add(
                contact_brep,
                name=f"contact_polygon_{edge}",
                color=Color.green(),
                opacity=0.5,
            )
            n_pts = len(fc.forces)
            trib = contact_polygon.area / n_pts if n_pts > 0 else 1.0
            fn_v = np.array([f["c_np"] for f in fc.forces])
            ft1_v = np.array([f["c_u"] for f in fc.forces])
            ft2_v = np.array([f["c_v"] for f in fc.forces])
            obj.attributes["data"] = {
                "Edge": str(edge),
                "Contact polygon": {
                    "area": round(contact_polygon.area, 6),
                    "n_points": n_pts,
                },
                "Max stress [MPa]": {
                    "s_n": round(float(fn_v.max() / trib) / 1e6, 4),
                    "s_t1": round(float(np.abs(ft1_v).max() / trib) / 1e6, 4),
                    "s_t2": round(float(np.abs(ft2_v).max() / trib) / 1e6, 4),
                },
            }

            for point, force in zip(fc.points, fc.forces):
                c_np, c_u, c_v = force["c_np"], force["c_u"], force["c_v"]
                t1, t2, n = fc.frame.xaxis, fc.frame.yaxis, fc.frame.zaxis
                forcevector_unsc = n * c_np + t1 * c_u + t2 * c_v

                if forcevector_unsc and point is not None:
                    forcevector = forcevector_unsc * block_scale
                    p1 = point.translated(forcevector)
                    p2 = point.translated(-forcevector)
                    point_obj = point_results.add(
                        cg.Line(p1, p2),
                        name=f"[{c_np:.1f}, {c_u:.1f}, {c_v:.1f}] \n|F|={forcevector_unsc.length:.1f}",
                        linewidth=2.5,
                        linecolor=Color.magenta(),
                    )
                    point_obj.attributes["data"] = {
                        "Edge": str(edge),
                        "Contact point": {
                            "x": round(point.x, 4),
                            "y": round(point.y, 4),
                            "z": round(point.z, 4),
                        },
                        "Force components": {
                            "c_np": round(c_np, 3),
                            "c_u": round(c_u, 3),
                            "c_v": round(c_v, 3),
                        },
                        "Resultant force magnitude": round(forcevector_unsc.length, 3),
                    }

        # Edge contacts
        # --------------
        for edge in edge_contact_edges:
            ec = results.contact_data(edge)

            if ec is None or ec.resultantforce is None:
                continue
            resultant = ec.resultantforce.vector
            line = ec.resultantline(scale=block_scale) if ec else None

            if line is None:
                continue
            # save_joined: aggregated interfaces get one arrow (drawn below)
            if edge not in interface_member_edges:
                resultant_forces.add(
                    line,
                    name=f"F=({resultant.x:.1f}, {resultant.y:.1f}, {resultant.z:.1f}) \n|F|={resultant.length:.1f}",
                    linewidth=2.5,
                    linecolor=Color.blue(),
                )

            edge_contacts.add(
                cg.Line(ec.points[0], ec.points[1]),
                name=f"contact_line_{edge}",
                linewidth=2,
                linecolor=Color.red(),
            )

            t1, t2, n = ec.frame.xaxis, ec.frame.yaxis, ec.frame.zaxis
            for point, force in zip(ec.points, ec.forces):
                c_np, c_u, c_v = force["c_np"], force["c_u"], force["c_v"]
                forcevector_unsc = n * c_np + t1 * c_u + t2 * c_v

                if forcevector_unsc and point is not None:
                    forcevector = forcevector_unsc * block_scale
                    p1 = point.translated(forcevector)
                    p2 = point.translated(-forcevector)
                    point_obj = point_results.add(
                        cg.Line(p1, p2),
                        name=f"[{c_np:.1f}, {c_u:.1f}, {c_v:.1f}] \n|F|={forcevector_unsc.length:.1f}",
                        linewidth=2.5,
                        linecolor=Color.magenta(),
                    )
                    point_obj.attributes["data"] = {
                        "Edge": str(edge),
                        "Contact point": {
                            "x": round(point.x, 4),
                            "y": round(point.y, 4),
                            "z": round(point.z, 4),
                        },
                        "Force components": {
                            "c_np": round(c_np, 3),
                            "c_u": round(c_u, 3),
                            "c_v": round(c_v, 3),
                        },
                        "Resultant force magnitude": round(forcevector_unsc.length, 3),
                    }

        # Point (vertex) contacts
        # -----------------------
        # A single-point contact carries a VertexContact (contact_data) and a
        # Point (contact_geometry). Draw the point marker and its force glyph.
        for edge in results.point_contact_edges():
            vc = results.contact_data(edge)
            geom = results.contact_geometry(edge)
            point = cg.Point(*geom) if geom is not None else getattr(vc, "point", None)
            if vc is None or point is None or vc.frame is None:
                continue

            edge_contacts.add(
                point,
                name=f"contact_point_{edge}",
                pointsize=6,
                pointcolor=Color.red(),
            )

            t1, t2, n = vc.frame.xaxis, vc.frame.yaxis, vc.frame.zaxis
            for force in vc.forces:
                if edge in interface_member_edges:
                    break  # save_joined: one arrow per interface (drawn below)
                c_np, c_u, c_v = force["c_np"], force["c_u"], force["c_v"]
                forcevector_unsc = n * c_np + t1 * c_u + t2 * c_v
                if not forcevector_unsc:
                    continue
                forcevector = forcevector_unsc * block_scale
                obj = resultant_forces.add(
                    cg.Line(
                        point.translated(forcevector), point.translated(-forcevector)
                    ),
                    name=f"F=({forcevector_unsc.x:.1f}, {forcevector_unsc.y:.1f}, {forcevector_unsc.z:.1f}) \n|F|={forcevector_unsc.length:.1f}",
                    linewidth=2.5,
                    linecolor=Color.blue(),
                )
                obj.attributes["data"] = {
                    "Edge": str(edge),
                    "Contact point": {
                        "x": round(point.x, 4),
                        "y": round(point.y, 4),
                        "z": round(point.z, 4),
                    },
                    "Force components": {
                        "c_np": round(c_np, 3),
                        "c_u": round(c_u, 3),
                        "c_v": round(c_v, 3),
                    },
                    "Resultant force magnitude": round(forcevector_unsc.length, 3),
                }

        # Joined-body interfaces (masonry_dem model.save_joined)
        # ------------------------------------------------------
        # One aggregate resultant per body-body interface: total force applied
        # at the |F|-weighted centre of pressure. Member polygons stay
        # individual. Empty for every other solver (metadata key absent).
        for itf in body_interfaces:
            # The interface SURFACE: one welded mesh, member contact polygons
            # kept as separate faces in their own vertex order. Drawn as a mesh
            # rather than a single ring on purpose — an interface spanning a
            # concave corner is not planar, and a non-convex one has no valid
            # ring ordering. `None` when every member is an edge/point contact,
            # and absent entirely on results saved before meshes were written.
            interface_mesh = itf.get("mesh")
            if interface_mesh is not None:
                obj = interface_contacts.add(
                    interface_mesh,
                    name=f"interface_{tuple(itf['bodies'])}",
                    facecolor=Color.brown(),
                    opacity=0.6,
                    show_lines=True,
                )
                obj.attributes["data"] = {
                    "Bodies": itf["bodies"],
                    "Member contacts": len(itf["edges"]),
                    "Faces": interface_mesh.number_of_faces(),
                    "Contact area": round(itf["area"], 6),
                }

            F = cg.Vector(*itf["force"])
            if F.length == 0:
                continue

            # Draw the resultant through the THRUST point — the point the
            # normal resultant actually passes through (the normal tractions
            # exert no moment about it), so stringing these together across a
            # series of joints traces the thrust line. Falls back to the
            # section centroid on an open joint carrying no compression.
            thrust = itf.get("thrust")
            centroid = itf.get("centroid") or itf.get("point")
            point = cg.Point(*(thrust if thrust is not None else centroid))

            if thrust is not None:
                thrust_points.add(
                    cg.Point(*thrust),
                    name=f"thrust_{tuple(itf['bodies'])}",
                    pointcolor=Color.red(),
                    pointsize=8,
                )

            half = F * 0.5 * block_scale
            obj = resultant_forces.add(
                cg.Line(point - half, point + half),
                name=f"Interface {tuple(itf['bodies'])}: F=({F.x:.1f}, {F.y:.1f}, {F.z:.1f}) \n|F|={F.length:.1f}",
                linewidth=3,
                linecolor=Color.blue(),
            )
            ecc = itf.get("eccentricity")
            obj.attributes["data"] = {
                "Bodies": itf["bodies"],
                "Member contacts": len(itf["edges"]),
                "Force components": {
                    "Fx": round(F.x, 3),
                    "Fy": round(F.y, 3),
                    "Fz": round(F.z, 3),
                },
                "Resultant magnitude": round(F.length, 3),
                "Normal force N": round(itf.get("N", 0.0), 3),
                "Moment about section": [round(m, 3) for m in itf["moment"]],
                "Eccentricity": (
                    None if ecc is None else [round(e, 6) for e in ecc]
                ),
                "Contact area": round(itf["area"], 6),
            }
