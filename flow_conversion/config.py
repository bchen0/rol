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
            out[child.attrib["name"]] = _parse_parameter_list(child)
        elif child.tag == "Parameter":
            out[child.attrib["name"]] = _convert(child.attrib["value"], child.attrib.get("type", "string"))
    return out


def _get(data: dict[str, Any], section: str, key: str, default: Any) -> Any:
    sub = data.get(section, {})
    if not isinstance(sub, dict):
        return default
    return sub.get(key, default)


@dataclass(frozen=True)
class FilteredDarcyConfig:
    xml_path: Path
    mesh_file: str
    use_param_var: bool
    parametrization_type: int
    use_darcy_flow: bool
    dynamic_viscosity: float
    fluid_density: float
    reynolds_number: float
    minimum_permeability: float
    maximum_permeability: float
    diffuser_radius: float
    diffuser_top: float
    inlet_radius: float
    target_axial_velocity: float
    target_radial_velocity: float
    target_type: int
    only_axial: bool
    optimize_domain_fraction: bool
    invert_domain_fraction: bool
    integration_domain_fraction: float
    polynomial_weight: bool
    target_velocity_weight: bool
    target_weighting_power: float
    normalized_misfit: bool
    radial_tracking_scale: float
    axial_tracking_scale: float
    filter_radius: float
    cubature_degree: int
    boundary_cubature_degree: int
    filter_cubature_degree: int
    density_basis_degree: int
    filter_basis_degree: int
    pressure_basis_degree: int
    element_type: str
    rvec: tuple[float, ...]
    zvec: tuple[float, ...]
    raw: dict[str, Any]

    @property
    def inlet_velocity(self) -> float:
        return self.reynolds_number * self.dynamic_viscosity / (
            self.fluid_density * 2.0 * self.inlet_radius
        )


def read_config(
    xml_path: str | Path,
    *,
    use_param_var: bool | None = None,
) -> FilteredDarcyConfig:
    path = Path(xml_path).expanduser().resolve()
    root = ET.parse(path).getroot()
    raw = _parse_parameter_list(root)
    problem = raw.get("Problem", {})
    if not isinstance(problem, dict):
        problem = {}

    use_param = bool(
        _get(raw, "Problem", "Use Optimal Constant Velocity", False)
        if use_param_var is None
        else use_param_var
    )
    r_defaults = (0.6875, 3.0, 10.0, 10.0, 3.0, 0.6875)
    z_defaults = (-14.001, -9.0, -3.0, 3.0, 9.0, 14.001)
    rvec = tuple(float(problem.get(f"r{i}", r_defaults[i])) for i in range(6))
    zvec = tuple(float(problem.get(f"z{i}", z_defaults[i])) for i in range(6))

    return FilteredDarcyConfig(
        xml_path=path,
        mesh_file=str(_get(raw, "Mesh", "File Name", "mesh.txt")),
        use_param_var=use_param,
        parametrization_type=int(problem.get("Parametrization Type", 0)),
        use_darcy_flow=bool(problem.get("Use Darcy Flow", True)),
        dynamic_viscosity=float(problem.get("Dynamic Viscosity", 0.84e-8)),
        fluid_density=float(problem.get("Fluid Density", 8.988e-11)),
        reynolds_number=float(problem.get("Reynolds Number", 50.0)),
        minimum_permeability=float(problem.get("Minimum Permeability", 3e-7)),
        maximum_permeability=float(problem.get("Maximum Permeability", 3e-6)),
        diffuser_radius=float(problem.get("Diffuser Radius", 10.0)),
        diffuser_top=float(problem.get("Diffuser Top", 9.9763392)),
        inlet_radius=float(problem.get("Inlet Radius", 0.6875)),
        target_axial_velocity=float(problem.get("Target Axial Velocity", 15.0)),
        target_radial_velocity=float(problem.get("Target Radial Velocity", 0.0)),
        target_type=int(problem.get("Target Type", 1)),
        only_axial=bool(problem.get("Only Use Axial Velocity", False)),
        optimize_domain_fraction=bool(problem.get("Optimize Domain Fraction", False)),
        invert_domain_fraction=bool(problem.get("Invert Domain Fraction", False)),
        integration_domain_fraction=float(problem.get("Integration Domain Fraction", 1.0)),
        polynomial_weight=bool(problem.get("Use Polynomial Weight", False)),
        target_velocity_weight=bool(problem.get("Use Target Velocity Weight", False)),
        target_weighting_power=float(problem.get("Target Weighting Power", 0.0)),
        normalized_misfit=bool(problem.get("Use Normalized Misfit", False)),
        radial_tracking_scale=float(problem.get("Radial Tracking Scale", 1.0)),
        axial_tracking_scale=float(problem.get("Axial Tracking Scale", 1.0)),
        filter_radius=float(problem.get("Filter Radius", 1e-2)),
        cubature_degree=int(problem.get("Cubature Degree", 4)),
        boundary_cubature_degree=int(problem.get("Boundary Cubature Degree", 4)),
        filter_cubature_degree=int(problem.get("Filter Cubature Degree", 4)),
        density_basis_degree=int(problem.get("Density Basis Degree", 1)),
        filter_basis_degree=int(problem.get("Filter Basis Degree", 1)),
        pressure_basis_degree=int(problem.get("Pressure Basis Degree", 1)),
        element_type=str(problem.get("Element Type", "QUAD")).upper(),
        rvec=rvec,
        zvec=zvec,
        raw=raw,
    )
