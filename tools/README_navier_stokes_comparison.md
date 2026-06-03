# Navier-Stokes Objective Comparison Harness

`compare_navier_stokes_objective.py` evaluates the Python reduced objective and
compares it with a C++/ROL reference result when one is available.

The reference side must provide JSON with this shape:

```json
{
  "value": 1.23,
  "gradient": [0.0, 0.1, -0.2],
  "hess_vec": [0.0, 0.01, -0.03]
}
```

The Python side uses the same explicit control and Hessian-vector direction
passed on the command line:

```bash
conda run -n nn_surrogate_opt python tools/compare_navier_stokes_objective.py \
  --backend fe \
  --time-steps 3 \
  --end-time 0.03 \
  --fe-cell-ids 2,3,4 \
  --control 0,0.01,-0.02 \
  --direction 0,0.03,-0.04 \
  --reference-json rol_reference.json
```

If a ROL probe executable prints the JSON object to stdout, call it directly:

```bash
conda run -n nn_surrogate_opt python tools/compare_navier_stokes_objective.py \
  --backend fe \
  --time-steps 3 \
  --end-time 0.03 \
  --control 0,0.01,-0.02 \
  --direction 0,0.03,-0.04 \
  --reference-command /path/to/rol_probe --time-steps 3 --end-time 0.03
```

This workspace currently does not include a built Trilinos/ROL executable or
PyROL bindings, so the harness can generate the Python result here but cannot
run the C++ reference until such a build is supplied.
