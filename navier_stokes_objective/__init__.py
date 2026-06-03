"""Convenience import shim for the source package under ``python/``."""

from pathlib import Path

_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "python" / "navier_stokes_objective"
__path__ = [str(_SOURCE_PACKAGE)] + __path__

from .objective import NavierStokesReducedObjective
from .fe import TaylorHoodSpace
from .fem_assembler import TaylorHoodAssembler
from .fem_dynamic import DynamicNavierStokesFEConstraint
from .fem_pde import NavierStokesLocalResidual
from .fem_qoi import NavierStokesFEObjective

__all__ = [
    "NavierStokesReducedObjective",
    "TaylorHoodSpace",
    "TaylorHoodAssembler",
    "DynamicNavierStokesFEConstraint",
    "NavierStokesLocalResidual",
    "NavierStokesFEObjective",
]
