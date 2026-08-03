"""Dynamic Navier-Stokes objective facade for PyROL."""

from contextlib import contextmanager
import math
import os
import sys
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


def _torch_tensor_data(value):
    flat = getattr(value, "flat", None)
    if isinstance(flat, torch.Tensor):
        return flat

    tensor = getattr(value, "tensor", None)
    if isinstance(tensor, torch.Tensor):
        return tensor

    torch_object = getattr(value, "torch_object", None)
    if isinstance(torch_object, torch.Tensor):
        return torch_object

    if isinstance(value, torch.Tensor):
        return value

    return None


def _set_torch_tensor_data(target, source):
    flat = getattr(target, "flat", None)
    if isinstance(flat, torch.Tensor):
        flat.view(-1).copy_(source.view(-1))
        return

    tensor = getattr(target, "tensor", None)
    if isinstance(tensor, torch.Tensor):
        tensor.copy_(source.reshape_as(tensor))
        return

    torch_object = getattr(target, "torch_object", None)
    if isinstance(torch_object, torch.Tensor):
        try:
            target.torch_object = source.reshape_as(torch_object)
        except AttributeError:
            torch_object.copy_(source.reshape_as(torch_object))
        return

    if isinstance(target, torch.Tensor):
        target.copy_(source.reshape_as(target))
        return

    raise TypeError("target is not backed by a torch.Tensor")


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
        root = ElementTree.parse(self._xml_path).getroot()
        time = _sublist(root, "Time Discretization")
        end_time = _parameter(time, "End Time", 1.0, float)
        self._dt = end_time / float(self._impl.num_steps)
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
        z_tensor = _torch_tensor_data(z)
        if z_tensor is not None:
            return self._torch_value(z_tensor)

        if len(z.shape) > 1:
            z = z.squeeze()
        return self._impl.value(z, tol)

    def prox(self, *args):
        if len(args) == 2:
            z, step = args
            tol = 1e-8
            return self._prox_direct(z, step, tol)
        if len(args) == 3:
            z, step, tol = args
            return self._prox_direct(z, step, tol)
        if len(args) == 4:
            out, z, step, tol = args
            return self._prox_in_place(out, z, step, tol)

        raise TypeError("prox expects (z, step[, tol]) or (out, z, step, tol)")

    def _prox_direct(self, z, step, tol):
        z_tensor = _torch_tensor_data(z)
        if z_tensor is not None:
            return self._torch_prox(z_tensor, step)

        if len(z.shape) > 1:
            z = z.squeeze()
        return self._impl.prox(z, step, tol)

    def _prox_in_place(self, out, z, step, tol):
        del tol
        z_tensor = _torch_tensor_data(z)
        if z_tensor is None:
            raise TypeError("in-place L1DynObjective.prox requires torch input")

        result = self._torch_prox(z_tensor, step)
        _set_torch_tensor_data(out, result)

    def _torch_value(self, z):
        with torch.no_grad():
            z = z.squeeze()
            self._check_torch_controls(z)
            if torch.any((z < self.lower_bound) | (z > self.upper_bound)):
                return self.l1_control_cost * 0.1 * sys.float_info.max

            abs_z = torch.abs(z)
            if abs_z.numel() < 2:
                return 0.0

            theta = self.theta
            value = self._dt * (
                (1.0 - theta) * abs_z[:-1] + theta * abs_z[1:]
            ).sum()
            return (self.l1_control_cost * value).item()

    def _torch_prox(self, z, step):
        if step < 0:
            raise ValueError("step must be nonnegative")

        with torch.no_grad():
            z = z.squeeze()
            self._check_torch_controls(z)
            threshold = self._torch_prox_threshold(z, step)
            result = torch.sign(z) * torch.clamp(torch.abs(z) - threshold, min=0.0)
            return torch.clamp(result, self.lower_bound, self.upper_bound)

    def _torch_prox_threshold(self, z, step):
        threshold = torch.full_like(z, float(step) * self._dt * self.l1_control_cost)
        theta = self.theta
        threshold[0] = float(step) * self._dt * self.l1_control_cost * (1.0 - theta)
        threshold[-1] = float(step) * self._dt * self.l1_control_cost * theta
        return threshold

    def _check_torch_controls(self, z):
        if not torch.is_floating_point(z):
            raise TypeError("z must be backed by a floating-point torch.Tensor")
        if z.ndim != 1:
            raise ValueError("z must be one-dimensional after squeeze")
        if z.numel() != self.num_controls:
            raise ValueError(
                f"z must have length {self.num_controls}; got {z.numel()}"
            )

    def default_control(self):
        return _navier_stokes_default_control(self._xml_path, self.num_controls)

    initial_control = default_control
