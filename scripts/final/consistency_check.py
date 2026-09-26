"""Final consistency check: pinned doc claims vs frozen artifacts.

Each entry: (doc, claim, artifact path, extractor). Fails loudly on any
drift. Writes outputs/FINAL/consistency.json. No test contact, no model
execution (evidence reads + hash checks only).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "FINAL"
    out.mkdir(parents=True, exist_ok=True)
    J = lambda p: json.loads((root / p).read_text())  # noqa: E731
    ev = J("outputs/test_evaluation_7b860dca72ea.json")
    checks: list[dict] = []

    def check(doc, claim, actual, expected, tol=0.0):
        ok = abs(actual - expected) <= tol if isinstance(expected, float) else actual == expected
        checks.append({"doc": doc, "claim": str(claim), "actual": actual, "pass": bool(ok)})

    c, s = ev["cls_primary"], ev["seg_primary"]
    check("viva§4/pack", "CLS acc 0.9950", c["accuracy"], 0.995)
    check("viva§4/pack", "CLS F1 0.9952", round(c["macro_f1"], 4), 0.9952)
    check("viva§4/pack", "CLS ECE 0.0027", round(c["ece_post_T"], 4), 0.0027)
    check("viva§4/pack", "CLS uncertain 4", c["uncertain"], 4)
    check("viva§4/pack", "SEG Dice 0.8617", round(s["mean_dice"], 4), 0.8617)
    check("viva§4/pack", "SEG median 0.9385", round(s["median_dice"], 4), 0.9385)
    check("viva§4/pack", "SEG IoU 0.7934", round(s["mean_iou"], 4), 0.7934)
    check(
        "viva§4/pack",
        "disagreement 10/856",
        (
            ev["disagreement_confident_tumor_empty_seg"]["count"],
            ev["disagreement_confident_tumor_empty_seg"]["n_confident_tumor"],
        ),
        (10, 856),
    )
    check(
        "viva§4/pack",
        "N993 acc delta 3.5e-05",
        round(c["accuracy"] - ev["cls_sensitivity"]["accuracy"], 8),
        3.5e-05,
        tol=1e-9,
    )
    check("viva§4/pack", "N853 dice 0.8610", round(ev["seg_sensitivity"]["mean_dice"], 4), 0.861)
    add = J("outputs/test_evaluation_7b860dca72ea_addendum_auc.json")
    check("viva§4/pack", "AUC 0.999936", add["roc_auc_ovr_primary_N1000"], 0.999936)
    cal = J("outputs/CLS-001/calibration_frozen.json")
    check(
        "viva§10/card",
        "T/taus",
        (cal["temperature"], cal["tau1"], cal["tau2"]),
        (0.5115954875946045, 0.95, 0.05),
    )
    rob = J("outputs/ROB-001/locked_eval.json")
    check("viva§16/card", "ROB Dice 0.8207", round(rob["seg_primary"]["mean_dice"], 4), 0.8207)
    check(
        "viva§16/card",
        "ROB disagreement 9",
        rob["disagreement_confident_tumor_empty_seg"]["count"],
        9,
    )
    u = J("outputs/UNC-001/unc001_locked.json")
    check("viva§12/card", "UNC detect 0.926", round(u["detection"]["detection_rate"], 3), 0.926)
    check("viva§12/card", "UNC burden 0.033", round(u["burden"]["false_alert_rate"], 3), 0.033)
    uv = J("outputs/UNC-001/unc001.json")
    check(
        "viva§12/card",
        "UNC val 0.75/0.25/0.027",
        (
            uv["detection"]["detection_rate"],
            uv["detection"]["baseline_detection_rate"],
            round(uv["burden"]["false_alert_rate"], 3),
        ),
        (0.75, 0.25, 0.027),
    )
    p3 = J("outputs/PBA-003/pba003.json")["per_perturbation"]["gauss_noise_s0p05"]
    check("viva§17/card", "noise CLS d_acc -0.019", p3["cls"]["d_acc"], -0.019)
    check("viva§17/card", "noise SEG d_dice -0.16873", p3["seg"]["d_dice"], -0.16873, tol=1e-6)
    f16 = J("outputs/SYSINT/fp16_check.json")
    check(
        "viva§18/card",
        "FP16 floors pass",
        (f16["floors"]["macro_f1_pass"], f16["floors"]["mean_dice_pass"]),
        (True, True),
    )
    check(
        "viva§18/card",
        "FP16 analyze median 0.0301",
        f16["latency_fp16"]["analyze"]["median"],
        0.0301,
    )
    q = J("outputs/QUALITY/gate_validation.json")
    check(
        "viva§13/ws1",
        "quality 0/1000 + 8/9",
        (
            q["false_rejects"],
            sum(1 for v in q["corruption_detection"].values() if v["verdict"] == "reject"),
        ),
        (0, 8),
    )
    for p, pre in (
        ("checkpoints/CLS-001/best.pt", "451e4fc4"),
        ("checkpoints/SEG-001/best.pt", "ce29df5e"),
        ("checkpoints/ROB-001/best.pt", "36091d63"),
    ):
        check("manifest/card", f"{p} hash", sha(root / p)[:8], pre)
    man = (root / "docs" / "release_manifest.md").read_text(encoding="utf-8")
    check(
        "manifest",
        "UTF-8 readable + 0 MISSING",
        ("MISSING" not in man) and ("451e4fc4" in man),
        True,
    )

    # ---- Evidence-freshness guards (post-audit): VV / UNC / EXPL artifacts
    # must reference the CURRENT checkpoint + code hashes, not pre-release
    # derivatives. Catches the stale-evidence drift class permanently.
    vv = J("outputs/VV/gate45_deploy_evidence.json")
    cls_now = sha(root / "checkpoints/CLS-001/best.pt")[:16]
    seg_now = sha(root / "checkpoints/SEG-001/best.pt")[:16]
    check(
        "vv_gate45",
        "VV gate45 hashes current",
        (
            vv["failed"] == []
            and vv["checks"]["hash:checkpoints/CLS-001/best.pt"]["detail"].startswith(cls_now)
            and vv["checks"]["hash:checkpoints/SEG-001/best.pt"]["detail"].startswith(seg_now)
        ),
        True,
    )
    unc = J("outputs/UNC-001/test_freeze.json")
    check(
        "unc_test_freeze",
        "UNC-001 freeze hash current",
        unc["frozen"]["checkpoint_sha256"][:16] == cls_now,
        True,
    )
    exp = J("outputs/EXPL-001/expl001.json")
    check(
        "expl001",
        "EXPL-001 module hashes current",
        (
            exp["module_hashes"]["inference/service.py"]
            == sha(root / "src/brain_tumor/inference/service.py")[:16]
            and exp["module_hashes"]["explain/gradcam.py"]
            == sha(root / "src/brain_tumor/explain/gradcam.py")[:16]
            and exp["module_hashes"]["explain/report.py"]
            == sha(root / "src/brain_tumor/explain/report.py")[:16]
        ),
        True,
    )
    g3 = J("outputs/VV/gate3_cross_interface.json")
    check(
        "vv_gate3",
        "cross-interface equivalence holds",
        (g3["service_vs_api"] is True and g3["service_vs_streamlit"] is True),
        True,
    )

    failed = [c for c in checks if not c["pass"]]
    (out / "consistency.json").write_text(
        json.dumps({"n": len(checks), "failed": failed}, indent=1)
    )
    print(
        f"consistency: {len(checks) - len(failed)}/{len(checks)} pass; "
        f"failed={[f['claim'] for f in failed] if failed else 'none'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
