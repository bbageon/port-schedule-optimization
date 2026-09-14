"""Content-addressed v3 evaluation contracts (YR-317-a).

An old arm file without a contract is evidence to preserve, never a resumable run.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import yaml


def digest(value) -> str:
    # JSON converts integer day keys to strings; normalize before sorting so
    # days 1, 2, 10 hash identically before and after saving a 30-day result.
    normalized = json.loads(json.dumps(value, allow_nan=False))
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def network_identity(net) -> dict:
    tensors = {}
    for name, tensor in sorted(net.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        tensors[name] = {"shape": list(t.shape), "dtype": str(t.dtype),
                         "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    return {"class": f"{type(net).__module__}.{type(net).__qualname__}",
            "training": net.training, "state_sha256": digest(tensors)}


def runtime_identity() -> dict:
    source = Path(__file__).resolve().parents[1]
    paths = sorted(source.rglob("*.py"))
    # Simulator profiles are read relative to the execution directory.
    configs = Path("configs")
    paths += sorted(p for p in configs.rglob("*") if p.is_file())
    hashes = {}
    for path in paths:
        label = ("v3/" + path.relative_to(source).as_posix()
                 if path.is_relative_to(source) else path.as_posix())
        raw = path.read_bytes()
        if path.suffix in {".py", ".yaml", ".yml", ".json", ".csv", ".md"}:
            raw = raw.replace(b"\r\n", b"\n")
        hashes[label] = hashlib.sha256(raw).hexdigest()
    return {"source_and_config_sha256": digest(hashes), "files": hashes,
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "yaml": str(yaml.__version__),
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def arm_contract(job: dict, runtime: dict, runner) -> dict:
    params = inspect.signature(runner).bind(**{k: v for k, v in job.items()
                                              if not k.startswith("_")})
    params.apply_defaults()
    settings = dict(params.arguments)
    settings["days"] = [asdict(day) for day in settings["days"]]
    settings["n_days"] = len(settings["days"])
    for key in ("seller_net", "buyer_net"):
        settings[key] = network_identity(settings[key])
    return {"schema": "yard_rl.v3.arm-contract.v1", "label": job["_label"],
            "settings": settings, "runtime": runtime,
            "observation_unit": "one continuous month; days are dependent"}


def validate_label(label: str) -> None:
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe or empty arm label: {label!r}")


def write_json(path: Path, payload: dict) -> None:
    """Replace only after serialization and writing finish successfully."""
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=1) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(raw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_arm(path: Path, contract: dict) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    stored = data.pop("contract", None)
    key = data.pop("contract_sha256", None)
    result_key = data.pop("result_sha256", None)
    if stored != contract or key != digest(contract):
        raise ValueError(f"Incompatible/legacy evaluation cache: {path}. Use a new output directory.")
    if result_key != digest(data) or data.get("arm") != contract["label"]:
        raise ValueError(f"Corrupt evaluation cache: {path}")
    for field in ("rollout_calls", "policy_exceptions", "txn_failed"):
        if type(data.get(field)) is not int or data[field] < 0:
            raise ValueError(f"Missing/invalid measured guard {field}: {path}")
    return data


def save_arm(path: Path, result, contract: dict) -> None:
    data = asdict(result)
    write_json(path, {**data, "contract": contract, "contract_sha256": digest(contract),
                      "result_sha256": digest(data)})
