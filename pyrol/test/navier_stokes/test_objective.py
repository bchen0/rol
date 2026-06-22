import math
import shutil
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from pyrol.navier_stokes import L1DynObjective, NavierStokesObjective


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


def write_smoke_xml(path):
    tree = ElementTree.parse(HERE / "input.xml")
    root = tree.getroot()

    problem = sublist(root, "Problem")
    set_parameter(problem, "Print Uncontrolled State", "false")
    set_parameter(problem, "Use Parametric Control", "true")

    reduced = sublist(root, "Reduced Dynamic Objective")
    set_parameter(reduced, "Use Sketching", "false")
    set_parameter(reduced, "Use Hessian", "true")

    time = sublist(root, "Time Discretization")
    set_parameter(time, "End Time", "0.05")
    set_parameter(time, "Number of Time Steps", "2")

    general = sublist(root, "General")
    set_parameter(general, "Print Verbosity", "0")
    set_parameter(general, "Output Level", "0")

    tree.write(path)


class TestNavierStokesObjective(unittest.TestCase):
    def test_l1_dynamic_objective_value_and_prox(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            xml_path = work / "input.xml"
            write_smoke_xml(xml_path)

            objective = L1DynObjective(xml_path)
            z = np.array([-0.3, 0.4])
            default_control = objective.default_control()
            expected_default = np.array([
                0.0,
                -6.0 * math.sin(2.0 * math.pi * 0.74 * (0.05 / 2.0)),
            ])

            dt = 0.05 / 2.0
            expected_value = objective.l1_control_cost * dt * abs(z[1])
            self.assertEqual(objective.num_steps, 2)
            self.assertEqual(objective.num_controls, 2)
            self.assertAlmostEqual(objective.theta, 1.0)
            np.testing.assert_allclose(default_control, expected_default)
            self.assertAlmostEqual(objective.value(z), expected_value)

            step = 2.0
            expected_prox = np.array([
                z[0],
                z[1] - step * dt * objective.l1_control_cost,
            ])
            np.testing.assert_allclose(objective.prox(z, step), expected_prox)

            outside_bounds = np.array([objective.upper_bound + 1.0, 0.0])
            self.assertGreater(objective.value(outside_bounds), 1e300)

    def test_value_gradient_and_hess_vec_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shutil.copy2(HERE / "channel.txt", work / "channel.txt")
            xml_path = work / "input.xml"
            cache_dir = work / "cache"
            mesh_output_dir = work / "mesh_output"
            cache_dir.mkdir()
            write_smoke_xml(xml_path)

            cwd = Path.cwd()
            objective = NavierStokesObjective(
                xml_path,
                cache_dir=cache_dir,
                spinup_time=0,
                mesh_output_dir=mesh_output_dir,
            )
            default_control = objective.default_control()
            z = np.zeros(objective.num_controls)
            v = np.ones(objective.num_controls)

            value = objective.value(z)
            gradient = objective.gradient(z)
            hess_vec = objective.hess_vec(v, z)
            value2, gradient2 = objective.value_and_gradient(z)

            self.assertEqual(objective.num_steps, 2)
            self.assertEqual(objective.num_controls, 2)
            self.assertEqual(Path.cwd(), cwd)
            if objective.comm_rank == 0:
                self.assertTrue((mesh_output_dir / "cell_to_node_quad.txt").is_file())
                self.assertTrue((mesh_output_dir / "nodes.txt").is_file())
            self.assertEqual(default_control.shape, (2,))
            self.assertAlmostEqual(default_control[0], 0.0)
            self.assertTrue(math.isfinite(value))
            self.assertTrue(math.isfinite(value2))
            self.assertEqual(gradient.shape, (2,))
            self.assertEqual(gradient2.shape, (2,))
            self.assertEqual(hess_vec.shape, (2,))
            self.assertTrue(np.all(np.isfinite(gradient)))
            self.assertTrue(np.all(np.isfinite(gradient2)))
            self.assertTrue(np.all(np.isfinite(hess_vec)))
            self.assertAlmostEqual(value, value2)


if __name__ == "__main__":
    unittest.main()
