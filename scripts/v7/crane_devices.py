"""Select a neural compute device without moving the terminal or its random sampler."""
import os


def device_for(phase, requested=None):
    import torch
    device = requested or os.environ.get(f"YARD_{phase.upper()}_DEVICE", "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported {phase} device: {device}")
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this Python environment")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device
