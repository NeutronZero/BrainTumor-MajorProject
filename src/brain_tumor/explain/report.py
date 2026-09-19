"""EXPL-001 report generator: machine JSON + human-readable text.

The text report always carries the engineering/non-clinical disclaimer and
the Grad-CAM boundary (contribution visualization, not proof of attention,
no clinical meaning claimed).
"""

from __future__ import annotations

DISCLAIMER = ("Automated engineering prototype output — not a clinical "
              "diagnosis. Grad-CAM shows regions contributing to the model's "
              "classification output; it does not prove the model attends to "
              "the tumor and has no validated clinical meaning.")


def build_report(payload: dict) -> dict:
    c = {"predicted_class": payload["predicted_class"],
         "confidence": payload["confidence"],
         "classification_state": payload["classification_state"]}
    loc = payload["localization"]
    con = payload.get("consistency", {})
    lines = [
        "BRAIN TUMOR ANALYSIS (engineering prototype — not clinical)",
        "",
        f"Predicted class: {c['predicted_class']} ({c['confidence']:.4f})",
        f"Classification state: {c['classification_state']}",
        f"System state: {payload['system_state']}",
        f"Consistency: {con.get('agreement_fraction')} "
        f"({'FLAGGED' if con.get('flagged') else 'stable'})",
    ]
    if payload["segmentation_state"] == "nonempty":
        lines += [f"Segmentation: nonempty, area={loc['area_pixels']} px",
                  f"Bbox (xyxy): {loc['bbox']}", f"Centroid: {loc['centroid']}"]
    elif payload["segmentation_state"] == "empty":
        lines += ["Segmentation: empty — no tumor region localized."]
    else:
        lines += ["Segmentation unavailable — system degraded."]
    if payload.get("warnings"):
        lines += [f"Warnings: {', '.join(payload['warnings'])}"]
    lines += ["", f"IMPORTANT LIMITATION: {DISCLAIMER}"]
    payload["report_text"] = "\n".join(lines)
    payload["disclaimer"] = DISCLAIMER
    return payload
