import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.utils.amp import (  # noqa: E402
    autocast_if_cuda,
    make_grad_scaler,
    require_cuda_arch,
)


def test_cpu_paths_never_crash():
    import torch
    assert make_grad_scaler(True) is None or torch.cuda.is_available()
    with autocast_if_cuda(True):
        pass
    assert require_cuda_arch() in ("cpu", "cuda")
