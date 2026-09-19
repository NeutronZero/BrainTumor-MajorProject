"""REL-001 reliability engine (Workstream 3): descriptive evidence summary.

Pure function of already-computed SB-1 signals. No model calls, no
retraining, no threshold/contract changes, no prediction or system_state
override. No single numerical reliability score: a structured summary with
an explicit basis list.

Fixed rule (pre-specified, never tuned):
  summary = "stable" iff ALL hold: classification confident,
  consistency agreement == 1.0, quality verdict == accept, and
  (predicted notumor OR segmentation nonempty+localized).
  Otherwise "review" with the failing conditions listed in basis.
"clinical_meaning" is always False.
"""

from __future__ import annotations


def reliability_report(classification: dict, consistency: dict,
                       quality: dict | None, segmentation_state: str,
                       localization: dict, system_state: str) -> dict:
    basis: list[str] = []
    ok = True

    if classification.get("classification_state") == "confident":
        basis.append(f"calibrated confidence {classification.get('confidence'):.4f} "
                     f"({classification.get('predicted_class')})")
    else:
        ok = False
        basis.append("classification uncertain — review")

    if consistency.get("agreement_fraction") == 1.0:
        basis.append(f"{consistency.get('k', 8)}/{consistency.get('k', 8)} "
                     "perturbation agreement (stable)")
    else:
        ok = False
        basis.append(f"perturbation agreement {consistency.get('agreement_fraction')} "
                     "— review")

    if quality is None:
        basis.append("input quality not assessed")
    elif quality.get("verdict") == "accept":
        basis.append("acceptable input quality")
    else:
        ok = False
        basis.append(f"input quality reject {quality.get('failed')} — review")

    if classification.get("predicted_class") == "notumor":
        basis.append("no tumor predicted — nothing to localize")
    elif segmentation_state == "nonempty" and localization.get("area_pixels", 0) > 0:
        basis.append(f"nonempty segmentation, area={localization['area_pixels']} px, localized")
    else:
        ok = False
        basis.append(f"segmentation {segmentation_state} with tumor prediction — review")

    return {
        "classification": {"class": classification.get("predicted_class"),
                           "confidence": classification.get("confidence"),
                           "state": classification.get("classification_state")},
        "consistency": {"k": consistency.get("k"),
                        "agreement": consistency.get("agreement_fraction"),
                        "state": "stable" if consistency.get("agreement_fraction") == 1.0
                        else "unstable"},
        "quality": {"state": "acceptable" if (quality or {}).get("verdict") == "accept"
                    else ("rejected" if quality else "not_assessed")},
        "segmentation": {"nonempty": segmentation_state == "nonempty",
                         "area_px": localization.get("area_pixels", 0)},
        "localization": {"state": "localized"
                         if (segmentation_state == "nonempty"
                             and localization.get("area_pixels", 0) > 0)
                         else ("empty" if segmentation_state == "empty"
                               else segmentation_state)},
        "reliability": {"summary": "stable" if ok else "review",
                        "basis": basis,
                        "clinical_meaning": False},
        "system_state_unchanged": system_state,
    }
