from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_navier_stokes_objective import compare_results, compute_python_result


from navier_stokes_paths import XML


def test_compare_harness_accepts_matching_reference_and_rejects_perturbation() -> None:
    result = compute_python_result(
        xml_path=XML,
        backend="graph",
        time_steps=3,
        end_time=0.03,
        control=np.array([0.0, 0.01, -0.02]),
        direction=np.array([0.0, 0.03, -0.04]),
    )

    ok, errors = compare_results(result, result, rtol=0.0, atol=0.0)
    assert ok
    assert errors["value_abs"] == 0.0

    perturbed = dict(result)
    perturbed["gradient"] = list(result["gradient"])
    perturbed["gradient"][1] += 1.0e-3
    ok, errors = compare_results(result, perturbed, rtol=1.0e-12, atol=1.0e-12)
    assert not ok
    assert errors["gradient_linf_abs"] > 0.0
