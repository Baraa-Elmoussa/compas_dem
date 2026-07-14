from compas.data import Data


class Results(Data):
    """Standalone container for DEM solver results.

    Decoupled from both the model and the problem — identified by
    ``model_id`` and ``problem_id`` so it can be validated before attaching.

    Stores per-node and per-edge graph attributes, plus solver-level metadata
    (time-series, history data). All data is serializable via ``compas.json_dump``.

    Edge keys are stored internally as ``"u,v"`` strings so they survive JSON
    round-trips. All accessor methods accept ``(u, v)`` tuples and automatically
    try the reverse direction.

    Parameters
    ----------
    model_id : str
        GUID of the model this result belongs to.
    problem_id : str
        GUID of the problem that produced this result.
    """

    def __init__(self, model_id: str, problem_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.problem_id = problem_id
        self._node_data: dict[int, dict] = {}
        self._edge_data: dict[str, dict] = {}
        self.metadata: dict = {}

    @property
    def __data__(self) -> dict:
        return {
            "model_id": self.model_id,
            "problem_id": self.problem_id,
            "node_data": self._node_data,
            "edge_data": self._edge_data,
            "metadata": self.metadata,
        }

    @classmethod
    def __from_data__(cls, data: dict) -> "Results":
        obj = cls(model_id=data["model_id"], problem_id=data["problem_id"])
        obj._node_data = {int(k): v for k, v in data["node_data"].items()}
        obj._edge_data = data["edge_data"]
        obj.metadata = data.get("metadata", {})
        return obj

    # =========================================================================
    # Low-level write
    # =========================================================================

    def set_node(self, node: int, attr: str, value) -> None:
        """Store a node attribute value."""
        if node not in self._node_data:
            self._node_data[node] = {}
        self._node_data[node][attr] = value

    def set_edge(self, edge: tuple, attr: str, value) -> None:
        """Store an edge attribute value. Edge is stored as a ``'u,v'`` string key."""
        key = f"{edge[0]},{edge[1]}"
        if key not in self._edge_data:
            self._edge_data[key] = {}
        self._edge_data[key][attr] = value

    # =========================================================================
    # Generic accessors
    # =========================================================================

    def node_attribute(self, node: int, attr: str):
        """Return a node attribute, or ``None`` if not set."""
        return self._node_data.get(node, {}).get(attr)

    def edge_attribute(self, edge: tuple, attr: str):
        """Return an edge attribute, trying both ``(u, v)`` and ``(v, u)``."""
        key = f"{edge[0]},{edge[1]}"
        if key in self._edge_data:
            return self._edge_data[key].get(attr)
        key_rev = f"{edge[1]},{edge[0]}"
        if key_rev in self._edge_data:
            return self._edge_data[key_rev].get(attr)
        return None

    # =========================================================================
    # Iteration
    # =========================================================================

    def nodes(self):
        """Iterate over node indices."""
        return iter(self._node_data)

    def edges(self):
        """Iterate over edges as ``(u, v)`` tuples."""
        for key in self._edge_data:
            u, v = key.split(",")
            yield int(u), int(v)

    def face_contact_edges(self):
        """Yield edges where ``face_contact`` is ``True``."""
        return [e for e in self.edges() if self.face_contact(e)]

    def edge_contact_edges(self):
        """Yield edges where ``edge_contact`` is ``True``."""
        return [e for e in self.edges() if self.edge_contact(e)]

    def point_contact_edges(self):
        """Yield edges where ``point_contact`` is ``True``."""
        return [e for e in self.edges() if self.point_contact(e)]

    # =========================================================================
    # Named node accessors
    # =========================================================================

    def transformation(self, node: int):
        """Return the :class:`compas.geometry.Transformation` for a block."""
        return self.node_attribute(node, "transformation")

    def displacement(self, node: int):
        """Return the translation ``[dx, dy, dz]`` derived from the transformation."""
        T = self.transformation(node)
        if T is None:
            return None
        return [T.matrix[0][3], T.matrix[1][3], T.matrix[2][3]]

    def rotation(self, node: int):
        """Return the 3×3 rotation sub-matrix derived from the transformation."""
        T = self.transformation(node)
        if T is None:
            return None
        return [[T.matrix[i][j] for j in range(3)] for i in range(3)]

    # =========================================================================
    # Named edge accessors — contact topology
    # =========================================================================

    def face_contact(self, edge: tuple) -> bool:
        """Return ``True`` if the contact is a face (polygon) contact."""
        return self.edge_attribute(edge, "face_contact") or False

    def edge_contact(self, edge: tuple) -> bool:
        """Return ``True`` if the contact is an edge (line) contact."""
        return self.edge_attribute(edge, "edge_contact") or False

    def point_contact(self, edge: tuple) -> bool:
        """Return ``True`` if the contact is a point contact."""
        return self.edge_attribute(edge, "point_contact") or False

    def contact_point(self, edge: tuple):
        """Return the list of contact points for the edge."""
        return self.edge_attribute(edge, "contact_point")

    def contact_polygon(self, edge: tuple):
        """Return the contact :class:`compas.geometry.Polygon` for the edge."""
        return self.edge_attribute(edge, "contact_polygon")

    def contact_geometry(self, edge: tuple):
        """Return the geometry matching the contact class of the edge.

        A :class:`compas.geometry.Polygon` for a face contact, a
        :class:`compas.geometry.Line` (between the two contact points) for an
        edge contact, or a :class:`compas.geometry.Point` for a vertex/point
        contact. Dispatch on :meth:`face_contact` / :meth:`edge_contact` /
        :meth:`point_contact` to know which type to expect.
        """
        return self.edge_attribute(edge, "contact_geometry")

    def contact_frame(self, edge: tuple):
        """Return the contact :class:`compas.geometry.Frame` for the edge."""
        return self.edge_attribute(edge, "contact_frame")

    def contact_data(self, edge: tuple):
        """Return the :class:`~compas_dem.interactions.FrictionContact` object for the edge."""
        return self.edge_attribute(edge, "contact_data")

    def status(self, edge: tuple):
        """Return the per-point contact status list (LMGC90 only)."""
        return self.edge_attribute(edge, "status")

    def gap(self, edge: tuple):
        """Return the per-point gap list for the edge."""
        return self.edge_attribute(edge, "gap")

    # =========================================================================
    # Named edge accessors — forces
    # =========================================================================

    def force(self, edge: tuple):
        """Return the resultant force vector ``[fx, fy, fz]`` for the edge."""
        return self.edge_attribute(edge, "force")

    def force_magnitude(self, edge: tuple) -> float:
        """Return the scalar resultant force magnitude for the edge."""
        return self.edge_attribute(edge, "force_magnitude")

    def force_vector(self, edge: tuple):
        """Return the per-point force vectors (LMGC90 only)."""
        return self.edge_attribute(edge, "force_vector")

    def force_normal(self, edge: tuple):
        """Return the per-point normal force list for the edge."""
        return self.edge_attribute(edge, "force_normal")

    def force_tangent1(self, edge: tuple):
        """Return the per-point first tangential force list for the edge."""
        return self.edge_attribute(edge, "force_tangent1")

    def force_tangent2(self, edge: tuple):
        """Return the per-point second tangential force list for the edge."""
        return self.edge_attribute(edge, "force_tangent2")
