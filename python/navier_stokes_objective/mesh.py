from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy import sparse


_DIM_RE = re.compile(r"^\s*(num_dim|num_nodes|num_elem|num_side_sets)\s*=\s*(\d+)\s*;")
_ARRAY_RE_TEMPLATE = r"\b{name}\b\s*=\s*(.*?);"
_QUAD_SIDE_NODES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _numbers_for_array(text: str, name: str, dtype: type) -> np.ndarray:
    match = re.search(_ARRAY_RE_TEMPLATE.format(name=re.escape(name)), text, flags=re.S)
    if match is None:
        raise ValueError(f"Could not find array {name!r} in mesh file")
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?", match.group(1))
    if dtype is int:
        return np.asarray([int(v) for v in values], dtype=np.int64)
    return np.asarray([float(v.replace("D", "E").replace("d", "e")) for v in values], dtype=float)


@dataclass(frozen=True)
class ChannelMesh:
    path: Path
    nodes: np.ndarray
    cells: np.ndarray
    side_sets: list[list[list[int]]]
    side_edges: list[list[tuple[int, int]]]
    cell_areas: np.ndarray
    node_mass: np.ndarray
    stiffness: sparse.csr_matrix

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_cells(self) -> int:
        return int(self.cells.shape[0])

    def sideset_nodes(self, sideset: int) -> np.ndarray:
        nodes: set[int] = set()
        for a, b in self.side_edges[sideset]:
            nodes.add(a)
            nodes.add(b)
        return np.asarray(sorted(nodes), dtype=np.int64)

    def sideset_node_weights(self, sideset: int) -> np.ndarray:
        weights = np.zeros(self.num_nodes)
        for a, b in self.side_edges[sideset]:
            length = float(np.linalg.norm(self.nodes[a] - self.nodes[b]))
            weights[a] += 0.5 * length
            weights[b] += 0.5 * length
        return weights


def read_channel_mesh(path: str | Path) -> ChannelMesh:
    mesh_path = Path(path).expanduser().resolve()
    text = mesh_path.read_text()
    dims: dict[str, int] = {}
    for line in text.splitlines():
        match = _DIM_RE.match(line)
        if match:
            dims[match.group(1)] = int(match.group(2))
    try:
        space_dim = dims["num_dim"]
        num_nodes = dims["num_nodes"]
        num_cells = dims["num_elem"]
        num_side_sets = dims["num_side_sets"]
    except KeyError as exc:
        raise ValueError(f"Missing mesh dimension {exc.args[0]!r} in {mesh_path}") from exc
    if space_dim != 2:
        raise NotImplementedError("Only the 2D channel mesh is supported")

    coordx = _numbers_for_array(text, "coordx", float)
    coordy = _numbers_for_array(text, "coordy", float)
    if coordx.size != num_nodes or coordy.size != num_nodes:
        raise ValueError("Coordinate array length does not match num_nodes")
    nodes = np.column_stack([coordx, coordy])

    conn = _numbers_for_array(text, "connect1", int)
    cells = conn.reshape(num_cells, 4) - 1
    if np.any(cells < 0) or np.any(cells >= num_nodes):
        raise ValueError("Cell connectivity contains node ids outside the mesh")

    side_sets: list[list[list[int]]] = [[[] for _ in range(4)] for _ in range(num_side_sets)]
    side_edges: list[list[tuple[int, int]]] = [[] for _ in range(num_side_sets)]
    for ss in range(num_side_sets):
        elems = _numbers_for_array(text, f"elem_ss{ss + 1}", int) - 1
        sides = _numbers_for_array(text, f"side_ss{ss + 1}", int) - 1
        if elems.size != sides.size:
            raise ValueError(f"Side set {ss + 1} has mismatched elem/side counts")
        for elem, side in zip(elems, sides, strict=True):
            side_sets[ss][int(side)].append(int(elem))
            loc_a, loc_b = _QUAD_SIDE_NODES[int(side)]
            a = int(cells[int(elem), loc_a])
            b = int(cells[int(elem), loc_b])
            side_edges[ss].append((a, b))

    cell_areas = _cell_areas(nodes, cells)
    node_mass = np.zeros(num_nodes)
    for area, cell in zip(cell_areas, cells, strict=True):
        node_mass[cell] += area / 4.0
    stiffness = _graph_stiffness(nodes, cells)

    return ChannelMesh(
        path=mesh_path,
        nodes=nodes,
        cells=cells,
        side_sets=side_sets,
        side_edges=side_edges,
        cell_areas=cell_areas,
        node_mass=node_mass,
        stiffness=stiffness,
    )


def _cell_areas(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    pts = nodes[cells]
    x = pts[:, :, 0]
    y = pts[:, :, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))


def _graph_stiffness(nodes: np.ndarray, cells: np.ndarray) -> sparse.csr_matrix:
    edge_weights: dict[tuple[int, int], float] = {}
    for cell in cells:
        for i, j in _QUAD_SIDE_NODES:
            a = int(cell[i])
            b = int(cell[j])
            if a > b:
                a, b = b, a
            length = float(np.linalg.norm(nodes[a] - nodes[b]))
            if length <= 0.0:
                continue
            edge_weights[(a, b)] = edge_weights.get((a, b), 0.0) + 1.0 / length

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(nodes.shape[0])
    for (a, b), w in edge_weights.items():
        diag[a] += w
        diag[b] += w
        rows.extend([a, b])
        cols.extend([b, a])
        data.extend([-w, -w])
    rows.extend(range(nodes.shape[0]))
    cols.extend(range(nodes.shape[0]))
    data.extend(diag.tolist())
    return sparse.coo_matrix((data, (rows, cols)), shape=(nodes.shape[0], nodes.shape[0])).tocsr()
