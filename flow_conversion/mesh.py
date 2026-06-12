from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_DIM_RE = re.compile(r"^\s*(num_dim|num_nodes|num_elem|num_side_sets)\s*=\s*(\d+)\s*;")
_ARRAY_RE_TEMPLATE = r"\b{name}\b\s*=\s*(.*?);"
_ELEM_TYPE_RE = re.compile(r"connect1:elem_type\s*=\s*\"?([^\";\n]+)\"?")
_TRI_SIDE_NODES = ((0, 1), (1, 2), (2, 0))
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
class DarcyMesh:
    path: Path
    nodes: np.ndarray
    cells: np.ndarray
    element_type: str
    side_sets: list[list[list[int]]]
    side_edges: list[list[tuple[int, int, int]]]

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def local_sides(self) -> tuple[tuple[int, int], ...]:
        return _TRI_SIDE_NODES if self.element_type == "TRI" else _QUAD_SIDE_NODES

    @property
    def nodes_per_cell(self) -> int:
        return int(self.cells.shape[1])

    def cells_on_sideset(self, sideset: int) -> set[int]:
        if sideset < 0 or sideset >= len(self.side_sets):
            return set()
        return {cell for side in self.side_sets[sideset] for cell in side}


def read_darcy_mesh(path: str | Path) -> DarcyMesh:
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
        raise NotImplementedError("Only 2D axisymmetric meshes are supported")

    elem_match = _ELEM_TYPE_RE.search(text)
    if elem_match is None:
        raise ValueError("Could not find connect1:elem_type in mesh file")
    elem_type_text = elem_match.group(1).upper()
    if "TRI" in elem_type_text:
        element_type = "TRI"
        nodes_per_cell = 3
        local_sides = _TRI_SIDE_NODES
    elif "QUAD" in elem_type_text:
        element_type = "QUAD"
        nodes_per_cell = 4
        local_sides = _QUAD_SIDE_NODES
    else:
        raise NotImplementedError(f"Unsupported element type {elem_type_text!r}")

    coordx = _numbers_for_array(text, "coordx", float)
    coordy = _numbers_for_array(text, "coordy", float)
    if coordx.size != num_nodes or coordy.size != num_nodes:
        raise ValueError("Coordinate array length does not match num_nodes")
    nodes = np.column_stack([coordx, coordy])

    conn = _numbers_for_array(text, "connect1", int)
    if conn.size != num_cells * nodes_per_cell:
        raise ValueError("Connectivity length does not match num_elem and element type")
    cells = conn.reshape(num_cells, nodes_per_cell) - 1
    if np.any(cells < 0) or np.any(cells >= num_nodes):
        raise ValueError("Cell connectivity contains node ids outside the mesh")

    side_sets: list[list[list[int]]] = [[[] for _ in local_sides] for _ in range(num_side_sets)]
    side_edges: list[list[tuple[int, int, int]]] = [[] for _ in range(num_side_sets)]
    for ss in range(num_side_sets):
        elems = _numbers_for_array(text, f"elem_ss{ss + 1}", int) - 1
        sides = _numbers_for_array(text, f"side_ss{ss + 1}", int) - 1
        if elems.size != sides.size:
            raise ValueError(f"Side set {ss + 1} has mismatched elem/side counts")
        for elem, side in zip(elems, sides, strict=True):
            elem_i = int(elem)
            side_i = int(side)
            if side_i < 0 or side_i >= len(local_sides):
                raise ValueError(f"Side set {ss + 1} references invalid local side {side_i + 1}")
            side_sets[ss][side_i].append(elem_i)
            loc_a, loc_b = local_sides[side_i]
            side_edges[ss].append((elem_i, int(cells[elem_i, loc_a]), int(cells[elem_i, loc_b])))

    return DarcyMesh(
        path=mesh_path,
        nodes=nodes,
        cells=cells,
        element_type=element_type,
        side_sets=side_sets,
        side_edges=side_edges,
    )
