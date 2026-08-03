"""Dynamic Navier-Stokes objective facade for PyROL."""

from contextlib import contextmanager
import math
import os
import threading
from xml.etree import ElementTree
from pyrol.getTypeName import *

import numpy as np
import torch

from ._navier_stokes import _L1DynObjective, _NavierStokesObjective

_CWD_LOCK = threading.Lock()


@contextmanager
def _temporary_working_directory(path):
    if path is None:
        yield
        return

    path = os.path.abspath(os.fspath(path))
    os.makedirs(path, exist_ok=True)
    with _CWD_LOCK:
        previous = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)


def _sublist(node, name):
    if node is None:
        return None
    for child in node.findall("ParameterList"):
        if child.get("name") == name:
            return child
    return None


def _parameter(node, name, default, cast):
    if node is None:
        return default
    for child in node.findall("Parameter"):
        if child.get("name") == name:
            return cast(child.get("value"))
    return default


def _bool_parameter(node, name, default):
    return _parameter(node, name, default, lambda value: value.lower() == "true")


def _navier_stokes_default_control(xml_path, num_controls):
    root = ElementTree.parse(xml_path).getroot()
    problem = _sublist(root, "Problem")
    initial_guess = _sublist(problem, "Initial Guess")
    top_initial_guess = _sublist(root, "Initial Guess")

    if _bool_parameter(top_initial_guess, "Read From File", False):
        raise ValueError(
            "The XML requests Initial Guess/Read From File; construct z from "
            "the initial_control.*.txt files instead."
        )

    time = _sublist(root, "Time Discretization")
    end_time = _parameter(time, "End Time", 1.0, float)
    num_steps = _parameter(time, "Number of Time Steps", num_controls, int)
    dt = end_time / float(num_steps)

    reynolds = _parameter(problem, "Reynolds Number", 200.0, float)
    amp0 = 6.0 - (reynolds - 200.0) / 1600.0
    strouhal0 = 0.74 - (reynolds - 200.0) * (0.115 / 800.0)
    amplitude = _parameter(initial_guess, "Amplitude", amp0, float)
    strouhal = _parameter(initial_guess, "Strouhal Number", strouhal0, float)
    phase = _parameter(initial_guess, "Phase Shift", 0.0, float)

    steps = np.arange(num_controls, dtype=float)
    times = steps * dt
    return -amplitude * np.sin(2.0 * math.pi * strouhal * times + phase)


class NavierStokesObjective:
    """Reduced dynamic Navier-Stokes objective backed by Trilinos."""

    def __init__(
        self,
        xml_path,
        cache_dir=None,
        spinup_time=None,
        mesh_output_dir=None,
    ):
        xml_path = os.path.abspath(os.fspath(xml_path))
        self._xml_path = xml_path
        self._mesh_output_dir = (
            None if mesh_output_dir is None
            else os.path.abspath(os.fspath(mesh_output_dir))
        )
        cache_arg = "" if cache_dir is None else os.path.abspath(os.fspath(cache_dir))
        spinup_arg = -1.0 if spinup_time is None else float(spinup_time)
        with _temporary_working_directory(self._mesh_output_dir):
            self._impl = _NavierStokesObjective(xml_path, cache_arg, spinup_arg)

    @property
    def num_steps(self):
        return self._impl.num_steps

    @property
    def num_controls(self):
        return self._impl.num_controls

    @property
    def comm_rank(self):
        return self._impl.comm_rank

    @property
    def comm_size(self):
        return self._impl.comm_size

    def value(self, z, tol=1e-8):
        return self._impl.value(z, tol)

    def gradient(self, z, tol=1e-8):
        return self._impl.gradient(z, tol)

    def hess_vec(self, v, z, tol=1e-8):
        return self._impl.hess_vec(v, z, tol)

    def value_and_gradient(self, z, tol=1e-8):
        return self._impl.value_and_gradient(z, tol)

    def default_control(self):
        return _navier_stokes_default_control(self._xml_path, self.num_controls)

    initial_control = default_control


class L1DynObjective(getTypeName('Objective')):
    """Nonsmooth dynamic L1 control penalty from the Navier-Stokes example."""

    def __init__(self, xml_path):
        self._xml_path = os.fspath(xml_path)
        self._impl = _L1DynObjective(self._xml_path)
        super().__init__()

    @property
    def num_steps(self):
        return self._impl.num_steps

    @property
    def num_controls(self):
        return self._impl.num_controls

    @property
    def l1_control_cost(self):
        return self._impl.l1_control_cost

    @property
    def lower_bound(self):
        return self._impl.lower_bound

    @property
    def upper_bound(self):
        return self._impl.upper_bound

    @property
    def theta(self):
        return self._impl.theta

    def value(self, z, tol=1e-8):
        # Todo: Change this to check that z is of type TensorVector. Needs a little rearranging though
        #  since TensorVector is defined outside this package (but doesn't have to be)
        if hasattr(z, "torch_object"):
            z = z.torch_object.numpy()
        if isinstance(z, torch.Tensor):
            z = z.numpy()
        if len(z.shape) > 1:
            z = z.squeeze()
        return self._impl.value(z, tol)

    def prox(self, z, step, tol=1e-8):
        # Todo: Change this to check that z is of type TensorVector. Needs a little rearranging though
        #  since TensorVector is defined outside this package (but doesn't have to be)
        if hasattr(z, "torch_object"):
            z = z.torch_object.numpy()
        if isinstance(z, torch.Tensor):
            z = z.numpy()
        if len(z.shape) > 1:
            z = z.squeeze()
        return self._impl.prox(z, step, tol)

    def default_control(self):
        return _navier_stokes_default_control(self._xml_path, self.num_controls)

    initial_control = default_control
