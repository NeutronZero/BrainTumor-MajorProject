"""Version-aware AMP layer — §37 correction. CPU-safe for CI (§60)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext


def _torch_amp_autocast(device_type: str = "cuda"):
    """Return autocast context: torch.amp (Torch>=2) else torch.cuda.amp."""
    try:
        import torch

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type)
        return torch.cuda.amp.autocast()
    except Exception:
        return nullcontext()


def autocast_if_cuda(enabled: bool = True):
    try:
        import torch

        if enabled and torch.cuda.is_available():
            return _torch_amp_autocast("cuda")
    except Exception:
        pass
    return nullcontext()


def make_grad_scaler(enabled: bool = True):
    """Return GradScaler or None (CPU / disabled). Never crashes on CPU CI."""
    try:
        import torch

        if not (enabled and torch.cuda.is_available()):
            return None
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=True)
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        return None


def require_cuda_arch(min_major: int = 7, min_minor: int = 0) -> str:
    """Fail-fast CUDA capability gate (prior F4: P100 sm_60 vs torch>=2.10).

    Returns the device string. Raises RuntimeError with a clear message when
    CUDA is present but below sm_70. Call after torch import in train scripts;
    CPU-only runs are unaffected (returns "cpu").
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if (major, minor) < (min_major, min_minor):
        raise RuntimeError(
            f"GPU {name} is sm_{major}{minor}; this project requires "
            f"sm_{min_major}{min_minor}+ (T4 sm_75 target). Aborting instead of "
            f"silently falling back.")
    return "cuda"


@contextmanager
def inference_no_grad():
    """no_grad on CUDA-or-CPU torch runs; hard error when torch is absent
    (inference without torch is meaningless — never fall through unguarded,
    and never double-yield, which corrupted caller error handling)."""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("inference_no_grad requires torch") from e
    with torch.no_grad():
        yield
