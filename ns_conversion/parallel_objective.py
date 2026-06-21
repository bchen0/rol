# parallel_objective.py

import importlib
import sys
import traceback

from mpi4py import MPI


COMM = MPI.COMM_WORLD
ROOT = 0

_next_id = 0
_shutdown_sent = False


def _load_type(module_name, qualname):
    module = importlib.import_module(module_name)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _type_spec(objective_type):
    if isinstance(objective_type, str):
        if ":" in objective_type:
            return tuple(objective_type.split(":", 1))
        module_name, _, qualname = objective_type.rpartition(".")
        if not module_name:
            raise ValueError("Use 'module:ClassName' or 'module.ClassName'.")
        return module_name, qualname

    qualname = objective_type.__qualname__
    if "<locals>" in qualname:
        raise ValueError("Objective type must be importable, not locally defined.")
    return objective_type.__module__, qualname


def _worker_loop(root=ROOT):
    objects = {}

    while True:
        try:
            msg = COMM.bcast(None, root=root)
            op = msg[0]

            if op == "stop":
                break

            if op == "init":
                _, object_id, module_name, qualname, args, kwargs = msg
                objective_type = _load_type(module_name, qualname)
                objects[object_id] = objective_type(*args, **kwargs)

            elif op == "call":
                _, object_id, method_name, args, kwargs = msg
                getattr(objects[object_id], method_name)(*args, **kwargs)

            else:
                raise RuntimeError(f"Unknown parallel objective command: {op}")

        except Exception:
            traceback.print_exc()
            COMM.Abort(1)


def park_worker_ranks(root=ROOT):
    if COMM.Get_size() == 1:
        return

    if COMM.Get_rank() != root:
        _worker_loop(root=root)
        sys.exit(0)


class ParallelObjectiveProxy:
    def __init__(self, local_objective, object_id, root=ROOT):
        self._local_objective = local_objective
        self._object_id = object_id
        self._root = root

    def _call(self, method_name, *args, **kwargs):
        if COMM.Get_size() > 1:
            COMM.bcast(
                ("call", self._object_id, method_name, args, kwargs),
                root=self._root,
            )
        return getattr(self._local_objective, method_name)(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self._local_objective, name)
        if callable(attr):
            def wrapped(*args, **kwargs):
                return self._call(name, *args, **kwargs)
            return wrapped
        return attr

    def close(self):
        shutdown_workers(root=self._root)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def make_parallel_objective(objective_type, *args, root=ROOT, **kwargs):
    global _next_id

    if COMM.Get_size() == 1:
        if isinstance(objective_type, str):
            module_name, qualname = _type_spec(objective_type)
            objective_type = _load_type(module_name, qualname)
        return objective_type(*args, **kwargs)

    if COMM.Get_rank() != root:
        _worker_loop(root=root)
        sys.exit(0)

    object_id = _next_id
    _next_id += 1

    module_name, qualname = _type_spec(objective_type)

    COMM.bcast(
        ("init", object_id, module_name, qualname, args, kwargs),
        root=root,
    )

    local_type = _load_type(module_name, qualname)
    local_objective = local_type(*args, **kwargs)

    return ParallelObjectiveProxy(local_objective, object_id, root=root)


def shutdown_workers(root=ROOT):
    global _shutdown_sent

    if COMM.Get_size() == 1:
        return
    if COMM.Get_rank() != root:
        return
    if _shutdown_sent:
        return

    COMM.bcast(("stop",), root=root)
    _shutdown_sent = True


### Usage example
# from parallel_objective import (
#     park_worker_ranks,
#     make_parallel_objective,
#     shutdown_workers,
# )

# park_worker_ranks()

# import pyrol.navier_stokes as ns

# obj = make_parallel_objective(
#     ns.NavierStokesObjective,
#     xml_path="/path/to/input_02.xml",
#     cache_dir="/path/to/cache",
#     spinup_time=0.0,
# )

# try:
#     # Your old serial code mostly goes here.
#     z = obj.default_control()
#     J = obj.value(z)
#     g = obj.gradient(z)

# finally:
#     shutdown_workers()
    
### SLURM example
    
# #SBATCH --ntasks=8
# #SBATCH --cpus-per-task=1

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# srun python your_script.py


# Seria run: python your_script.py
    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
    
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# python your_script.py
    
# parallel run: 
# #SBATCH --ntasks=8
# #SBATCH --cpus-per-task=1

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# srun python your_script.py
