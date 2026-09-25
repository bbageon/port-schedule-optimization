"""Versioned weights/optimizer checkpoints; NOT physical world snapshots."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .. import V4_BASE_COMMIT
from .model import BlockPolicy, INPUT_DIM

FORMAT = "yard-v5-shared-block-ppo-1"


def save_checkpoint(path, runtime):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "input_dim": INPUT_DIM,
                "hidden": runtime.policy.hidden, "v4_base_commit": V4_BASE_COMMIT,
                "policy": runtime.policy.state_dict(), "optimizer": runtime.optimizer.state_dict(),
                "config": asdict(runtime.config), "updates": len(runtime.updates),
                "action_rng": runtime.action_rng.get_state(),
                "scope": "weights-and-optimizer-only; no physical world resume"}, path)


def load_policy(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != FORMAT or data.get("input_dim") != INPUT_DIM:
        raise ValueError("Not a compatible v5 unified PPO checkpoint")
    policy = BlockPolicy(hidden=int(data["hidden"]))
    policy.load_state_dict(data["policy"], strict=True)
    return policy
