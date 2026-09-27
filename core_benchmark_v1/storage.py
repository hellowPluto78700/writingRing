"""Atomic artifacts, content identity, and per-run advisory locks."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator
from uuid import uuid4
import numpy as np
import torch
from .protocol import Protocol, aliases, digest, runs


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, value: Any) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with atomic_path(path) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def save_torch(path: Path, value: Any) -> None:
    with atomic_path(path) as temporary:
        torch.save(value, temporary)


def load_torch(path: Path) -> dict[str, Any]:
    # Load only checkpoints produced locally by this benchmark.
    return torch.load(path, map_location="cpu", weights_only=False)


def source_identity(repo: Path) -> dict[str, str]:
    paths = sorted((repo / "core_benchmark_v1").glob("*.py"))
    paths += [repo / p for p in ("snn/accel_reconstruction_eval/datasets.py", "snn/action0_dataset.py", "src/writingring/segment_padding.py")]
    return {str(p.relative_to(repo)): file_hash(p) for p in paths if p.is_file()}


def environment(repo: Path) -> dict[str, Any]:
    versions = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    return {"packages": versions, "git_commit": commit.stdout.strip(), "python": __import__("sys").version}


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_lock(root: Path, repo: Path | None = None) -> tuple[Protocol, dict[str, Any]]:
    lock = read_json(root / "protocol.lock.json")
    p = Protocol.from_dict(lock["protocol"])
    if p.fingerprint != lock["protocol_hash"]:
        raise ValueError("Protocol lock was modified")
    if repo is not None and digest(source_identity(repo)) != lock["source_hash"]:
        raise ValueError("Source changed after prepare; use a new results directory")
    if digest(lock["data_metadata"]) != lock["metadata_hash"] or digest(lock["source_files"]) != lock["source_hash"]:
        raise ValueError("Provenance metadata changed")
    if digest(lock["runs"]) != digest([asdict(r) for r in runs(p)]) or lock["aliases"] != aliases():
        raise ValueError("Run manifest/aliases differ from the locked protocol")
    if lock["identity"] != digest([lock["protocol_hash"], lock["dataset_hash"], lock["metadata_hash"], lock["source_hash"]]):
        raise ValueError("Invalid benchmark identity")
    return p, lock


def checkpoint_metadata(run: Any, lock: dict[str, Any]) -> dict[str, Any]:
    return {"identity": lock["identity"], "run": asdict(run)}


def validate_checkpoint(payload: dict[str, Any], run: Any, lock: dict[str, Any]) -> None:
    if payload.get("identity") != lock["identity"] or digest(payload.get("run")) != digest(asdict(run)):
        raise ValueError(f"Checkpoint identity mismatch: {run.key}")


def mark_complete(directory: Path, run: Any, lock: dict[str, Any], required: list[str]) -> None:
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run.key}: missing artifacts {missing}")
    save_json(directory / "complete.json", {**checkpoint_metadata(run, lock), "status": "PASS",
              "files": {name: file_hash(directory / name) for name in required}})


def validate_complete(directory: Path, run: Any, lock: dict[str, Any]) -> None:
    complete = read_json(directory / "complete.json")
    validate_checkpoint(complete, run, lock)
    if complete.get("status") != "PASS" or not complete.get("files"):
        raise ValueError(f"Incomplete run {run.key}")
    for name, expected in complete["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Artifact changed: {directory / name}")
