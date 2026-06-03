from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _convert(value: str, typ: str) -> Any:
    if typ == "bool":
        return value.strip().lower() == "true"
    if typ == "int":
        return int(value)
    if typ == "double":
        return float(value)
    return value


def _parse_parameter_list(elem: ET.Element) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for child in elem:
        if child.tag == "ParameterList":
            name = child.attrib["name"]
            out[name] = _parse_parameter_list(child)
        elif child.tag == "Parameter":
            name = child.attrib["name"]
            out[name] = _convert(child.attrib["value"], child.attrib.get("type", "string"))
    return out


def _get(data: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass(frozen=True)
class NavierStokesConfig:
    xml_path: Path
    mesh_file: str
    nt: int
    end_time: float
    theta: float
    reynolds_number: float
    use_parametric_control: bool
    use_parabolic_inflow: bool
    use_nonpenetrating_walls: bool
    cylinder_center_x: float
    cylinder_center_y: float
    cylinder_radius: float
    integrated_objective_type: str
    final_time_objective_type: str
    state_cost: float
    state_boundary_cost: float
    control_cost: float
    final_time_state_cost: float
    raw: dict[str, Any]

    @property
    def dt(self) -> float:
        return self.end_time / float(self.nt)

    @property
    def viscosity(self) -> float:
        return 1.0 / self.reynolds_number


def read_config(
    xml_path: str | Path,
    *,
    time_steps: int | None = None,
    end_time: float | None = None,
    theta: float | None = None,
) -> NavierStokesConfig:
    path = Path(xml_path).expanduser().resolve()
    root = ET.parse(path).getroot()
    raw = _parse_parameter_list(root)

    nt = int(time_steps if time_steps is not None else _get(raw, ("Time Discretization", "Number of Time Steps"), 100))
    T = float(end_time if end_time is not None else _get(raw, ("Time Discretization", "End Time"), 1.0))
    th = float(theta if theta is not None else _get(raw, ("Time Discretization", "Theta"), 1.0))
    problem = raw.get("Problem", {})

    # The XML in this example uses lowercase "type"; objective.cpp asks for
    # uppercase "Type".  Match the effective defaults while accepting both.
    int_type = problem.get("Integrated Objective Type", problem.get("Integrated Objective type", "Dissipation"))
    ft_type = problem.get("Final Time Objective Type", problem.get("Final Time Objective type", "Tracking"))

    return NavierStokesConfig(
        xml_path=path,
        mesh_file=str(_get(raw, ("Mesh", "File Name"), "channel.txt")),
        nt=nt,
        end_time=T,
        theta=th,
        reynolds_number=float(problem.get("Reynolds Number", 200.0)),
        use_parametric_control=bool(problem.get("Use Parametric Control", False)),
        use_parabolic_inflow=bool(problem.get("Use Parabolic Inflow", False)),
        use_nonpenetrating_walls=bool(problem.get("Use Non-Penetrating Walls", False)),
        cylinder_center_x=float(problem.get("Cylinder Center X", 0.0)),
        cylinder_center_y=float(problem.get("Cylinder Center Y", 0.0)),
        cylinder_radius=float(problem.get("Cylinder Radius", 0.5)),
        integrated_objective_type=str(int_type),
        final_time_objective_type=str(ft_type),
        state_cost=float(problem.get("State Cost", 1.0)),
        state_boundary_cost=float(problem.get("State Boundary Cost", 1.0)),
        control_cost=float(problem.get("Control Cost", 0.0)),
        final_time_state_cost=float(problem.get("Final Time State Cost", 1.0)),
        raw=raw,
    )
