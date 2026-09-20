"""Verify PBA-003 outputs: control==locked eval, no aggregate score, per-arm completeness."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "outputs" / "PBA-003"
s = json.loads((out / "pba003.json").read_text())
print("keys:", list(s.keys()))
print("control:", s["control_values"])
blob = json.dumps(s).lower()
assert "no aggregate robustness score" in blob
assert "robustness_score" not in blob.replace("no aggregate robustness score", "")
print("no aggregate score: OK")
for k, v in s["per_perturbation"].items():
    c, g = v["cls"], v["seg"]
    print(k, "| cls acc", c["accuracy"], "d", c["d_acc"], "f1", c["macro_f1"],
          "ece", c["ece_frozen_T"], "unc", c["uncertain_n"], "err", c["errors_n"],
          "cbw", c["cbw_n"], "new", len(c["new_errors_vs_control"]),
          "recov", len(c["recovered_vs_control"]),
          "| seg dice", g["mean_dice"], "d", g["d_dice"],
          "empty", g["empty_pred"], "multi", g["multi_component"],
          "bycls", g["by_class_mean_dice"])
ev = json.loads((root / "outputs" / "test_evaluation_7b860dca72ea.json").read_text())
assert s["control_values"]["cls_acc"] == ev["cls_primary"]["accuracy"] == 0.995
assert abs(s["control_values"]["seg_dice"] - ev["seg_primary"]["mean_dice"]) < 1e-4
print("control arm == locked eval: OK")
# failure-mode spot checks
for pid in ("gauss_noise_s0p05", "jpeg_q50", "downscale_0p75x", "intensity_shift_p0p05"):
    arm = json.loads((out / ("pert_" + pid + ".json")).read_text())
    assert arm["cls"]["n"] == arm["cls"]["evaluable"] == 1000
    assert arm["seg"]["n"] == arm["seg"]["evaluable"] == 860
print("populations intact on all arms: OK")
