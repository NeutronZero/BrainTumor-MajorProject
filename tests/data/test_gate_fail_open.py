import subprocess, sys
from pathlib import Path


def test_gate_fails_honestly_without_dataset():
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "scripts/data_gate/run_gate.py"],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr
    assert (root / "outputs" / "data_gate_0" / "gate_decision.json").exists()
