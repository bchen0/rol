from TorchBoundConstraint import TorchBoundConstraint
from TorchVectors import TensorVector

from pyrol import Objective, Problem, Solver, getCout
from pyrol.pyrol.Teuchos import ParameterList

import torch


class ShiftedQuadratic(Objective):
    def __init__(self, target):
        self.target = target
        super().__init__()

    def value(self, x, tol):
        del tol
        residual = x.torch_object - self.target
        return (0.5 * torch.sum(residual * residual)).item()

    def gradient(self, g, x, tol):
        del tol
        g.torch_object = x.torch_object - self.target

    def hessVec(self, hv, v, x, tol):
        del x, tol
        hv.torch_object = v.torch_object


def build_parameter_list():
    params = ParameterList()
    params["General"] = ParameterList()
    params["General"]["Output Level"] = 1
    params["Step"] = ParameterList()
    params["Step"]["Type"] = "Trust Region"
    params["Step"]["Trust Region"] = ParameterList()
    params["Step"]["Trust Region"]["Subproblem Model"] = "SPG"
    params["Status Test"] = ParameterList()
    params["Status Test"]["Gradient Tolerance"] = 1e-10
    params["Status Test"]["Iteration Limit"] = 100
    return params


def main():
    torch.set_default_dtype(torch.float64)

    target = torch.tensor([2.0, -1.0, 0.25])
    x = TensorVector(torch.tensor([-0.5, 1.5, 0.0]))
    g = x.clone()

    lower = TensorVector(torch.zeros_like(x.torch_object))
    upper = TensorVector(torch.ones_like(x.torch_object))
    bounds = TorchBoundConstraint(lower, upper)

    problem = Problem(ShiftedQuadratic(target), x, g)
    problem.addBoundConstraint(bounds)

    solver = Solver(problem, build_parameter_list())
    solver.solve(getCout())

    print(f"x = {x.torch_object}")


if __name__ == "__main__":
    main()
