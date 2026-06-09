"""Python objective wrapper for the dynamic Navier-Stokes ROL example.

The public entry point mirrors ``objective.cpp``:

    obj = NavierStokesReducedObjective.from_xml(".../input.xml")
    value = obj.value(z)
    grad = obj.gradient(z)
    hv = obj.hess_vec(z, v)

The default backend is a self-contained graph spatial model built from the
Exodus/ncdump channel mesh.  The optional ``backend="fe"`` path uses the
Taylor-Hood finite-element pieces in this package while preserving the same
reduced objective API.
"""

from .objective import NavierStokesReducedObjective
from .fe import TaylorHoodSpace
from .fem_assembler import TaylorHoodAssembler
from .fem_dynamic import DynamicNavierStokesFEConstraint
from .fem_pde import NavierStokesLocalResidual
from .fem_qoi import NavierStokesFEObjective
from .linear_solvers import KLU2SparseLinearSolver, ScipySparseLinearSolver

__all__ = [
    "NavierStokesReducedObjective",
    "TaylorHoodSpace",
    "TaylorHoodAssembler",
    "DynamicNavierStokesFEConstraint",
    "NavierStokesLocalResidual",
    "NavierStokesFEObjective",
    "KLU2SparseLinearSolver",
    "ScipySparseLinearSolver",
]
