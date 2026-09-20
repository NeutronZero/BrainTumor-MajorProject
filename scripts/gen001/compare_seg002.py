"""compare_seg002.py — GEN-001 engineering comparison (evidence, not training).

Reads outputs/GEN-001/seed{42,43,44}/{metrics,val_analysis}.json plus the
frozen SEG-001 comparator (contract §2 + outputs/SEG-001-analysis record).
Writes outputs/GEN-001/aggregate.json (mean±std, per-seed retained) and
outputs/GEN-001/decision.md (ACCEPTED/REJECTED/INCONCLUSIVE on the GEN-001
engineering objective only, evaluated deterministically against the amended
numeric criteria). No test contact. No model execution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SEEDS = (42, 43, 44)
PRIMARY_BASE = 0.8518   # SEG-001 per-image val mean Dice (amendment A1)
LATENCY_REJECT = 0.0356  # 2x SEG-001 median under identical §L1 (amendment A2)


def _mean(xs):
    import statistics
    return (sum(xs) / len(xs), statistics.pstdev(xs))


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "GEN-001"
    per = {}
    for s in SEEDS:
        d = out / f"seed{s}"
        m = json.loads((d / "metrics.json").read_text())
        v = json.loads((d / "val_analysis.json").read_text())
        assert m["seed"] == s == v["seed"], "seed mismatch"
        per[s] = {"global_train_metric": m["val_dice_global"],
                  "epochs_run": m["epochs_run"], "params": v["params"],
                  "mean": v["val_dice"]["mean"], "median": v["val_dice"]["median"],
                  "p10": v["val_dice"]["p10"], "by_class": v["by_class"],
                  "empty": v["localization_val"]["empty_null"],
                  "multi": v["localization_val"]["multi"],
                  "l1_median": (v["l1_latency"] or {}).get("median"),
                  "peak_mem": v["peak_memory"]}
    means = [per[s]["mean"] for s in SEEDS]
    avg, sd = _mean(means)
    lat = [per[s]["l1_median"] for s in SEEDS if per[s]["l1_median"] is not None]

    reasons = []
    status = "INCONCLUSIVE"
    if any(per[s]["l1_median"] is not None and per[s]["l1_median"] > LATENCY_REJECT
           for s in SEEDS):
        status, reasons = "REJECTED", [f"latency median > {LATENCY_REJECT}s (§L1)"]
    elif avg - sd > PRIMARY_BASE and all(
            per[s]["by_class"][c]["p10"] >= {"glioma": 0.0852, "meningioma": 0.8676,
                                             "pituitary": 0.7664}[c] - 0.02
            for s in SEEDS for c in ("glioma", "meningioma", "pituitary")):
        status = "ACCEPTED"
        reasons = [f"mean {avg:.4f}±{sd:.4f} clears {PRIMARY_BASE} with no tail breach"]
    else:
        reasons = [f"mean {avg:.4f}±{sd:.4f} vs comparator {PRIMARY_BASE}: "
                   "within noise or tail/latency guard not fully cleared"]

    agg = {"experiment": "GEN-001", "seeds": list(SEEDS), "per_seed": per,
           "aggregate": {"mean_of_means": avg, "sd_of_means": sd,
                         "primary_comparator": PRIMARY_BASE,
                         "latency_reject_s": LATENCY_REJECT,
                         "latencies_measured": lat},
           "decision": {"status": status, "reasons": reasons,
                        "scope": "GEN-001 engineering objective ONLY"}}
    (out / "aggregate.json").write_text(json.dumps(agg, indent=1))
    (out / "decision.md").write_text(
        f"# GEN-001 decision\n\n**{status}**\n\n" +
        "\n".join(f"- {r}" for r in reasons) +
        f"\n\nMean±SD {avg:.4f}±{sd:.4f} vs SEG-001 {PRIMARY_BASE}.\n"
        "Scope: GEN-001 engineering objective only. Production unchanged.\n")
    print(json.dumps({"decision": status, "mean": round(avg, 4),
                      "sd": round(sd, 4)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
