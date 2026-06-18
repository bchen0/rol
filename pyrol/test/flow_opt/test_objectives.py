import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

import pyrol.flow_opt as flow_opt_module
from pyrol.flow_opt import (
    BrinkmanObjective,
    DarcyObjective,
    FilteredDarcyObjective,
)


HERE = Path(__file__).resolve().parent


def sublist(node, name):
    for child in node.findall("ParameterList"):
        if child.get("name") == name:
            return child
    raise KeyError(name)


def set_parameter(node, name, value):
    for child in node.findall("Parameter"):
        if child.get("name") == name:
            child.set("value", str(value))
            return
    raise KeyError(name)


def get_parameter(node, name):
    for child in node.findall("Parameter"):
        if child.get("name") == name:
            return child.get("value")
    raise KeyError(name)


def write_smoke_xml(source, destination):
    tree = ElementTree.parse(source)
    root = tree.getroot()

    problem = sublist(root, "Problem")
    set_parameter(problem, "Check derivatives", "false")
    set_parameter(problem, "Solve Optimization Problem", "false")

    simopt = sublist(root, "SimOpt")
    solve = sublist(simopt, "Solve")
    set_parameter(solve, "Output Iteration History", "false")

    mesh = sublist(root, "Mesh")
    set_parameter(mesh, "File Name", "../../mesh/small-axisymmetric-tri.txt")

    general = sublist(root, "General")
    set_parameter(general, "Print Verbosity", "0")
    set_parameter(general, "Output Level", "0")

    tree.write(destination)


def mesh_path_for_xml(xml_path):
    root = ElementTree.parse(xml_path).getroot()
    mesh = sublist(root, "Mesh")
    mesh_name = get_parameter(mesh, "File Name")
    return (xml_path.parent / mesh_name).resolve()


def centered_difference(values_plus, values_minus, eps):
    return (values_plus - values_minus) / (2.0 * eps)


def reference_executable(model_name):
    env_name = f"PYROL_FLOW_OPT_{model_name.upper()}_REFERENCE"
    env_path = os.environ.get(env_name)
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None

    ref_dir = os.environ.get("PYROL_FLOW_OPT_REFERENCE_DIR")
    candidates = []
    if ref_dir:
        candidates.append(Path(ref_dir))
    candidates.append(Path(flow_opt_module.__file__).resolve().parent)
    candidates.append(HERE)

    exe_name = f"flow_opt_{model_name}_reference"
    for directory in candidates:
        path = directory / exe_name
        if path.exists():
            return path
    return None


def write_vector(path, values):
    path.write_text("\n".join(f"{x:.17e}" for x in values) + "\n")


def parse_reference_output(output):
    result = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        key = parts[0]
        if key in {"VALUE", "GRADIENT_DOT", "HESS_VEC_DOT"}:
            result[key.lower()] = float(parts[1])
        elif key in {"GRADIENT", "HESS_VEC"}:
            count = int(parts[1])
            values = np.array([float(x) for x in parts[2:]], dtype=float)
            if values.shape != (count,):
                raise AssertionError(f"bad {key} length in reference output")
            result[key.lower()] = values
        elif key == "LOCAL_SIZE":
            result["local_size"] = int(parts[1])
    return result


def run_reference(executable, xml_path, z, v):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        z_path = tmp_path / "z.txt"
        v_path = tmp_path / "v.txt"
        write_vector(z_path, z)
        write_vector(v_path, v)
        completed = subprocess.run(
            [str(executable), str(xml_path), str(z_path), str(v_path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return parse_reference_output(completed.stdout)


class TestFlowOptObjectives(unittest.TestCase):
    MODELS = (
        ("darcy", "darcy", DarcyObjective),
        ("brinkman", "brinkman", BrinkmanObjective),
        ("filteredDarcy", "filtered_darcy", FilteredDarcyObjective),
    )

    def test_value_gradient_and_hess_vec_line_up(self):
        for model_name, reference_name, objective_type in self.MODELS:
            with self.subTest(model=model_name):
                source_xml = HERE / "models" / model_name / "input.xml"
                xml_path = HERE / "models" / model_name / "input-smoke.xml"
                write_smoke_xml(source_xml, xml_path)

                mesh_path = mesh_path_for_xml(xml_path)
                if not mesh_path.exists():
                    self.skipTest(
                        f"missing generated axisymmetric mesh file: {mesh_path}"
                    )

                objective = objective_type(xml_path)
                z = np.full(objective.local_size, 0.5)
                if objective.parameter_size:
                    z[-objective.parameter_size :] = np.array([0.0, 15.0])

                rng = np.random.default_rng(20240618 + objective.comm_rank)
                v = rng.uniform(-0.25, 0.25, objective.local_size)
                if objective.local_field_size:
                    norm = np.linalg.norm(v[: objective.local_field_size])
                    if norm > 0.0:
                        v[: objective.local_field_size] /= 4.0 * norm
                if objective.parameter_size:
                    v[-objective.parameter_size :] = np.array([0.1, -0.1])

                value = objective.value(z)
                gradient = objective.gradient(z)
                hess_vec = objective.hess_vec(v, z)
                value2, gradient2 = objective.value_and_gradient(z)

                self.assertTrue(math.isfinite(value))
                self.assertTrue(math.isfinite(value2))
                self.assertEqual(gradient.shape, (objective.local_size,))
                self.assertEqual(gradient2.shape, (objective.local_size,))
                self.assertEqual(hess_vec.shape, (objective.local_size,))
                self.assertTrue(np.all(np.isfinite(gradient)))
                self.assertTrue(np.all(np.isfinite(gradient2)))
                self.assertTrue(np.all(np.isfinite(hess_vec)))
                np.testing.assert_allclose(value, value2, rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(gradient, gradient2)

                eps = 1e-5
                fd_grad_dot = centered_difference(
                    objective.value(z + eps * v),
                    objective.value(z - eps * v),
                    eps,
                )
                grad_dot = objective.gradient_dot(z, v)
                self.assertAlmostEqual(
                    grad_dot,
                    fd_grad_dot,
                    delta=max(1e-4, 1e-3 * abs(fd_grad_dot)),
                )

                fd_hess_dot = centered_difference(
                    objective.gradient_dot(z + eps * v, v),
                    objective.gradient_dot(z - eps * v, v),
                    eps,
                )
                hess_dot = objective.hess_vec_dot(v, z)
                self.assertAlmostEqual(
                    hess_dot,
                    fd_hess_dot,
                    delta=max(1e-3, 1e-2 * abs(fd_hess_dot)),
                )

                if objective.comm_size == 1:
                    executable = reference_executable(reference_name)
                    if executable is None:
                        self.skipTest(f"missing C++ reference executable for {model_name}")
                    reference = run_reference(executable, xml_path, z, v)
                    self.assertEqual(reference["local_size"], objective.local_size)
                    np.testing.assert_allclose(
                        reference["value"], value, rtol=1e-12, atol=1e-8
                    )
                    np.testing.assert_allclose(
                        reference["gradient"], gradient, rtol=1e-10, atol=1e-8
                    )
                    np.testing.assert_allclose(
                        reference["hess_vec"], hess_vec, rtol=1e-9, atol=1e-7
                    )
                    np.testing.assert_allclose(
                        reference["gradient_dot"], grad_dot, rtol=1e-10, atol=1e-8
                    )
                    np.testing.assert_allclose(
                        reference["hess_vec_dot"], hess_dot, rtol=1e-9, atol=1e-7
                    )


if __name__ == "__main__":
    unittest.main()
