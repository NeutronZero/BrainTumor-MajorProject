"""Streamlit direct-call app — does NOT call FastAPI over HTTP (§19).

Thin UI over the shared InferenceService. Uploaded bytes always reach the
service. UNC-001 consistency renders as a descriptive signal; it never
alters system_state (contract amendment deferred as a PI decision).
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brain_tumor.inference.service import InferenceService  # noqa: E402

LIMIT_BYTES = 10 * 1024 * 1024


def build_payload(svc: InferenceService, data: bytes) -> dict:
    """Bytes -> full result payload. Error envelope mirrors the API contract."""
    from PIL import Image
    if len(data) > LIMIT_BYTES:
        return {"error": "file_too_large", "detail": "limit 10MB"}
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:  # noqa: BLE001
        return {"error": "undecodable_image", "detail": "cannot decode upload"}
    try:
        payload = svc.analyze(img).model_dump()
        payload["consistency"] = svc.consistency(img)
        return payload
    except Exception as e:  # noqa: BLE001 — never leak stack/paths
        return {"error": "inference_failed", "detail": str(type(e).__name__)}


def render_result(st, payload: dict) -> None:
    """Explicit field rendering with empty-segmentation and error branches."""
    if "error" in payload:
        st.error(f"{payload['error']}: {payload.get('detail', '')}")
        return
    st.subheader(f"Predicted: {payload['predicted_class']}")
    st.metric("Calibrated confidence", f"{payload['confidence']:.4f}")
    st.write(f"Classification state: `{payload['classification_state']}`")
    st.write(f"System state: `{payload['system_state']}`")
    with st.expander("Class probabilities"):
        st.json(payload["probabilities"])
    con = payload.get("consistency", {})
    st.write(f"Consistency: agreement {con.get('agreement_fraction')} "
             f"({'FLAGGED' if con.get('flagged') else 'stable'}, "
             f"k={con.get('k')}, σ={con.get('sigma')})")
    st.caption("Consistency is a descriptive failure-detection signal; "
               "it does not alter system_state.")
    if payload["segmentation_state"] == "nonempty":
        st.write(f"Segmentation: nonempty, area={payload['localization']['area_pixels']} px")
        st.write(f"Bbox (xyxy, original px): {payload['localization']['bbox']}")
        st.write(f"Centroid: {payload['localization']['centroid']}")
    elif payload["segmentation_state"] == "empty":
        st.info("Segmentation: empty — no tumor region localized.")
    else:
        st.warning("Segmentation unavailable — system degraded.")
    if payload.get("warnings"):
        st.write(f"Warnings: {', '.join(payload['warnings'])}")


def render_explanation(st, payload: dict) -> None:
    """EXPL-001 presentation: report text + visual overlays (observer output)."""
    import base64
    if "error" in payload:
        st.error(f"{payload['error']}: {payload.get('detail', '')}")
        return
    render_result(st, payload)
    st.subheader("Explanation (observer — predictions unchanged)")
    st.text(payload.get("report_text", ""))
    g = payload.get("gradcam", {})
    if g.get("overlay_png_b64"):
        st.image(base64.b64decode(g["overlay_png_b64"]),
                 caption=f"Grad-CAM ({g.get('target_layer')}) — contribution "
                         "visualization, no clinical meaning claimed")
    sv = payload.get("segmentation_vis")
    if sv and sv.get("overlay_png_b64"):
        st.image(base64.b64decode(sv["overlay_png_b64"]),
                 caption="Segmentation overlay (mask + bbox + centroid)")
        if sv.get("focus_in_bbox") is not None:
            st.write(f"CAM mass inside bbox: {sv['focus_in_bbox']:.2f} (descriptive)")


if __name__ == "__main__":
    try:
        import streamlit as st

        @st.cache_resource
        def get_inference_service() -> InferenceService:
            return InferenceService.from_registry()

        st.title("BrainTumor-MajorProject (research prototype — not clinical)")
        svc = get_inference_service()
        up = st.file_uploader("Upload MRI slice",
                              type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        if st.button("Analyze") and up is not None:
            render_result(st, build_payload(svc, up.getvalue()))
        if st.button("Explain") and up is not None:
            from PIL import Image
            try:
                render_explanation(st, svc.explain(Image.open(io.BytesIO(up.getvalue())),
                                                  raw=up.getvalue()))
            except Exception as e:  # noqa: BLE001 — never leak stack/paths
                st.error(f"inference_failed: {type(e).__name__}")
    except ImportError:
        print("streamlit not installed")
