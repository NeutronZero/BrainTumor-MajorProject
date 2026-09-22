"""Regression: M12 bootstrap producer must be committed, deterministic, and
reproduce the documented headline intervals exactly (6 dp).

Accuracy: exact Clopper-Pearson [0.988371, 0.998375] (k=995, n=1000).
Macro-F1: 10,000-replicate percentile bootstrap [0.990400, 0.999101]
(fixed seed RandomState(42)).
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "evaluate" / "bootstrap_ci.py"
CLAIMED_ACC = [0.988371, 0.998375]
CLAIMED_F1 = [0.9904, 0.999101]  # 0.990400 at 6 dp


def _run() -> str:
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = ROOT / "outputs" / "PBA-001" / "bootstrap_ci.json"
    assert out.is_file()
    return out.read_text(encoding="utf-8")


def test_bootstrap_reproduces_claimed_intervals():
    payload = json.loads(_run())
    assert payload["accuracy"]["k_correct"] == 995
    assert payload["n"] == 1000
    assert [round(v, 6) for v in payload["accuracy"]["ci95"]] == CLAIMED_ACC
    m = payload["macro_f1"]
    assert m["B"] == 10000 and m["seed"] == 42
    assert [round(v, 6) for v in m["ci95"]] == CLAIMED_F1
    assert round(m["point"], 6) == 0.995167
    # Producer must remain committed (SAS §24 reference).
    assert SCRIPT.is_file()


def test_bootstrap_deterministic_across_runs():
    assert _run() == _run()
