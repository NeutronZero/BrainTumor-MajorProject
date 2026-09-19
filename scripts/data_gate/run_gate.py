"""run_gate.py — Data Gate 0 orchestrator, Revision A (BRISC dual-tree aware).

Returns FAIL (exit 2) on any violation. Fail-closed: every check must pass,
including nested split-overlap. Never overrides evidence, never touches data.

Outputs (outputs/data_gate_0/):
  dataset_manifest.csv, dataset_summary.json, integrity_report.json,
  source_report.json, duplicate_report.json, near_duplicate_report.json,
  mask_report.json, split_overlap_report.json, gate_decision.json, gate_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate import (  # noqa: E402
    audit_duplicates,
    audit_near_duplicates,
    audit_split_overlap,
    characterize_dataset,
    inspect_masks,
    verify_hashes,
    verify_source,
)
from data_gate.common import outputs_dir  # noqa: E402


def main() -> int:
    out = outputs_dir()
    results: dict[str, dict] = {}
    # acquisition + characterization + integrity (hard requirements)
    results["source"] = verify_source.main()
    results["hashes"] = verify_hashes.main()
    results["characterize"] = characterize_dataset.main()
    results["exact_duplicates"] = audit_duplicates.main()
    results["near_duplicates"] = audit_near_duplicates.main()
    results["masks"] = inspect_masks.main()
    results["split_overlap"] = audit_split_overlap.main()

    hard = ["source", "hashes", "characterize", "exact_duplicates", "split_overlap"]
    hard_fail = [k for k in hard if not results[k].get("pass")]
    masks_ok = results["masks"].get("pass")
    # §12 decision: PASS => classification+seg+local core; FAIL => re-scope
    if not hard_fail and masks_ok:
        decision = "PASS"
        core = ["classification", "segmentation", "localization"]
    else:
        decision = "FAIL"
        core = ["classification", "calibration", "uncertainty", "explainability",
                "optimization", "application"]
    payload = {"gate": "Data Gate 0", "gate_tooling_rev": "A",
               "decision": decision,
               "hard_failures": hard_fail,
               "masks_pass": bool(masks_ok),
               "core_if_proceed": core,
               "limitation": ("Complete patient-level independence between train and test "
                               "cannot be claimed."),
               "checks": {k: v.get("pass") for k, v in results.items()}}
    (out / "gate_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Data Gate 0 — decision: " + decision, ""]
    for k, v in results.items():
        lines.append(f"- {k}: {'PASS' if v.get('pass') else 'FAIL'}"
                     + (f" ({v.get('reason')})" if v.get("reason") else ""))
    lines += ["", f"Core if proceed: {', '.join(core)}",
              "Limitation: complete patient-level independence cannot be claimed."]
    (out / "gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"DATA GATE 0: {decision}")
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
