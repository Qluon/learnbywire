"""Run identity and artifacts; contains no optimizer-control implementation."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np
import torch

VERSION = "1.1.1"


@dataclass(frozen=True)
class TrainResult:
    status: str
    committed_steps: int
    run_dir: Path
    metrics: dict
    model_path: Path | None
    checkpoint: Path | None


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def clock():
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW) if hasattr(time, "CLOCK_MONOTONIC_RAW") else time.perf_counter()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def json_value(value):
    def convert(item):
        if isinstance(item, set):
            return sorted(item)
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, (Path, torch.dtype)):
            return str(item)
        raise TypeError(f"Unsupported configuration value: {type(item).__name__}")
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def model_identity(model):
    hashes = {"trainable": hashlib.sha256(), "frozen": hashlib.sha256()}
    signature = []
    for name, parameter in model.named_parameters():
        description = [name, list(parameter.shape), str(parameter.dtype), parameter.requires_grad]
        signature.append(description)
        digest = hashes["trainable" if parameter.requires_grad else "frozen"]
        digest.update(json.dumps(description).encode())
        value = parameter.detach().cpu().contiguous()
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    config = getattr(model, "config", None)
    config = config.to_dict() if callable(getattr(config, "to_dict", None)) else {}
    for key in ("_name_or_path", "transformers_version"):
        config.pop(key, None)
    adapters = {name: json_value(value.to_dict()) for name, value in getattr(model, "peft_config", {}).items()}
    return dict(model_class=f"{type(model).__module__}.{type(model).__qualname__}",
                parameter_signature=signature, model_config=json_value(config), adapters=adapters,
                attention_implementation=getattr(getattr(model, "config", None), "_attn_implementation", None),
                initial_trainable_sha256=hashes["trainable"].hexdigest(),
                frozen_parameters_sha256=hashes["frozen"].hexdigest())


def versions():
    values = {"trainerplus": VERSION}
    for name in ("torch", "transformers", "accelerate", "tokenizers", "peft"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return values


def new_run(output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:10])
    run.mkdir(exist_ok=False)
    source = run / "source" / "trainerplus"
    shutil.copytree(Path(__file__).parent, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return run


@contextmanager
def isolated_caches(run):
    names = {"TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton"}
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name, child in names.items():
            path = (Path(run)/"cache"/child).resolve()
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def preserve_random_state(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def checkpoint_record(path, *, contract, step, control):
    path = Path(path)
    files = {p.relative_to(path).as_posix():sha_file(p) for p in path.rglob("*")
             if p.is_file() and p.name != "trainerplus_checkpoint.json"}
    save(path / "trainerplus_checkpoint.json", dict(schema=1, workload_sha256=identity(contract),
        contract=contract, committed_steps=step, control=control, files=files))


def verify_checkpoint(path, contract):
    path = Path(path).resolve()
    saved = json.loads((path / "trainerplus_checkpoint.json").read_text())
    if saved.get("schema") != 1 or saved.get("workload_sha256") != identity(contract):
        raise ValueError("Checkpoint workload mismatch: data, model, settings, control or software changed")
    for relative, expected in saved["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path) or not file.is_file() or sha_file(file) != expected:
            raise ValueError("Checkpoint file integrity mismatch")
    step = saved.get("committed_steps")
    if type(step) is not int or not 0 <= step < contract["total_steps"]:
        raise ValueError("Checkpoint must be before the final update of the same budget")
    return path, step
