from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import socket
import subprocess
from typing import Optional


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _detect_cluster_name() -> Optional[str]:
    name = os.environ.get("SLURM_CLUSTER_NAME")
    if name:
        return name.strip().lower()

    if shutil.which("scontrol"):
        try:
            result = subprocess.run(
                ["scontrol", "show", "config"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            match = re.search(r"(?m)^\s*ClusterName\s*=\s*(\S+)", result.stdout)
            if match:
                return match.group(1).strip().lower()
        except Exception:
            pass

    return None


def _int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _cpu_affinity_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class CpuRuntime:
    hostname: str
    cluster_name: Optional[str]
    is_unity: bool
    in_slurm: bool
    is_unity_login_node: bool
    slurm_job_id: Optional[str]
    slurm_cpus_per_task: int
    slurm_ntasks: int
    cpu_affinity_count: int
    available_cpus: int
    threads_per_run: int
    parallel_runs: int
    force_cpu: bool

    def apply_torch_thread_limits(self, torch_module) -> None:
        """Apply the per-process thread limit after torch is imported."""
        torch_module.set_num_threads(self.threads_per_run)
        try:
            torch_module.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only allows changing inter-op threads before parallel work starts.
            pass

    def print_summary(self, total_runs: Optional[int] = None) -> None:
        print("=" * 88)
        print("EXECUTION ENVIRONMENT")
        print("-" * 88)
        print(f"Hostname:                  {self.hostname}")
        print(f"Cluster:                   {self.cluster_name}")
        print(f"Unity detected:            {self.is_unity}")
        print(f"Inside Slurm allocation:   {self.in_slurm}")
        print(f"Slurm job ID:              {self.slurm_job_id}")
        print(f"SLURM_NTASKS:              {self.slurm_ntasks}")
        print(f"SLURM_CPUS_PER_TASK:       {self.slurm_cpus_per_task}")
        print(f"CPU affinity count:        {self.cpu_affinity_count}")
        print(f"Usable CPUs:               {self.available_cpus}")
        print(f"Threads per training run:  {self.threads_per_run}")
        print(f"Parallel training runs:    {self.parallel_runs}")
        print(f"Force CPU on Unity:        {self.force_cpu}")
        if total_runs is not None:
            print(f"Experiment runs:           {total_runs}")
        print("=" * 88)


def configure_cpu_runtime(
    *,
    total_runs: int,
    threads_per_run: int = 1,
    prefer_cpu_on_unity: bool = True,
    require_slurm_on_unity: bool = True,
) -> CpuRuntime:
    """Detect Unity/Slurm resources and configure one-thread-per-run execution.

    This function does not create a new Slurm allocation. A running Jupyter kernel
    cannot safely acquire extra CPU cores after launch and retroactively attach them
    to the current process. It therefore reads the allocation already granted to the
    kernel and derives the number of independent training processes that may run.

    Call this function before importing NumPy/PyTorch so BLAS/OpenMP thread limits
    are inherited by worker processes.
    """
    if total_runs <= 0:
        raise ValueError("total_runs must be positive")
    if threads_per_run <= 0:
        raise ValueError("threads_per_run must be positive")

    for name in _THREAD_ENV_VARS:
        os.environ[name] = str(threads_per_run)

    hostname = socket.gethostname().strip().lower()
    cluster_name = _detect_cluster_name()
    in_slurm = "SLURM_JOB_ID" in os.environ
    is_unity = cluster_name == "unity" or hostname.startswith("unity")
    is_unity_login_node = is_unity and not in_slurm and hostname.startswith("login")

    affinity = _cpu_affinity_count()
    slurm_cpus_per_task = _int_env("SLURM_CPUS_PER_TASK", 0)
    slurm_ntasks = max(1, _int_env("SLURM_NTASKS", 1))

    if in_slurm and slurm_cpus_per_task > 0:
        available_cpus = min(affinity, slurm_cpus_per_task)
    else:
        available_cpus = affinity

    if require_slurm_on_unity and is_unity_login_node:
        raise RuntimeError(
            "Unity login node detected without a Slurm allocation. Start Jupyter "
            "inside a compute allocation (for example, one task with multiple "
            "CPUs per task) before running the training sweep."
        )

    parallel_runs = max(
        1,
        min(total_runs, available_cpus // threads_per_run),
    )

    return CpuRuntime(
        hostname=hostname,
        cluster_name=cluster_name,
        is_unity=is_unity,
        in_slurm=in_slurm,
        is_unity_login_node=is_unity_login_node,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_ntasks=slurm_ntasks,
        cpu_affinity_count=affinity,
        available_cpus=available_cpus,
        threads_per_run=threads_per_run,
        parallel_runs=parallel_runs,
        force_cpu=bool(prefer_cpu_on_unity and is_unity),
    )
