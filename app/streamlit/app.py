"""Streamlit direct-call app — does NOT call FastAPI over HTTP (§19).

Thin UI over the shared InferenceService. Uploaded bytes always reach the
service. UNC-001 consistency renders as a descriptive signal; it never
alters system_state (contract amendment deferred as a PI decision).

REL-002 UI pass v2 ("clinical console"): status color language applied
consistently across badge/chips/tiles, verdict-first result card, observer
metric tiles, polished sidebar rail and tab bar. Frozen contracts preserved:
direct in-process calls, provenance-labeled sections, empty/unavailable
segmentation branches, error envelopes, standing non-diagnosis footer.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brain_tumor.inference.service import InferenceService  # noqa: E402

LIMIT_BYTES = 10 * 1024 * 1024
DISCLAIMER = (
    "Engineering prototype output — not a medical diagnosis. "
    "Not a certified medical device; never use for clinical "
    "diagnosis, triage, or therapy planning. Research use only."
)

# ---- status color language (one semantic color per system state) ------------
_STATE_STYLE = {
    "healthy": ("#4FB3BF", "●", "HEALTHY"),
    "tumor_localized": ("#F2CC8F", "◆", "TUMOR LOCALIZED"),
    "tumor_unlocalized": ("#E07A5F", "▲", "TUMOR UNLOCALIZED"),
    "uncertain": ("#C9A7EB", "◐", "UNCERTAIN"),
    "degraded": ("#E06C75", "✕", "DEGRADED"),
}
_ACCENT = "#4FB3BF"
_CLASS_ACCENT = {
    "glioma": "#E07A5F",
    "meningioma": "#F2CC8F",
    "pituitary": "#81B29A",
    "notumor": "#4FB3BF",
}


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
        # Same threading as API /analyze: compute once, pass down. Outputs
        # are bit-identical to the pre-threading implementation (every stage
        # is deterministic); this only removes redundant forward passes.
        result = svc.analyze(img)
        payload = result.model_dump()
        qual = svc.quality(data)
        payload["quality"] = qual
        con = svc.consistency(img, clean_pred=result.predicted_class)
        payload["consistency"] = con
        payload["reliability"] = svc.reliability(img, raw=data, result=result, con=con, qual=qual)
        return payload
    except Exception as e:  # noqa: BLE001 — never leak stack/paths
        return {"error": "inference_failed", "detail": str(type(e).__name__)}


def render_result(st, payload: dict, image_bytes: bytes | None = None) -> None:
    """Explicit field rendering with empty-segmentation and error branches.

    Sections are labeled by provenance: MODEL OUTPUT vs engineering observers.
    """
    if "error" in payload:
        st.error(f"**{payload['error']}** — {payload.get('detail', '')}")
        return
    if image_bytes is not None:
        st.image(image_bytes, caption="Input image (as uploaded)")
    st.subheader("Model output")
    st.write(f"Predicted: {payload['predicted_class']}")
    st.metric("Calibrated confidence", f"{payload['confidence']:.4f}")
    st.write(f"Classification state: `{payload['classification_state']}`")
    st.write(f"System state: `{payload['system_state']}`")
    with st.expander("Class probabilities"):
        st.json(payload["probabilities"])
    st.subheader("Engineering observers (descriptive; do not alter the above)")
    q = payload.get("quality", {})
    if q:
        st.write(
            f"Input quality: `{q.get('verdict')}`"
            + (f" — {', '.join(q.get('failed', []))}" if q.get("failed") else "")
        )
    con = payload.get("consistency", {})
    st.write(
        f"Consistency: agreement {con.get('agreement_fraction')} "
        f"({'FLAGGED' if con.get('flagged') else 'stable'}, "
        f"k={con.get('k')}, σ={con.get('sigma')})"
    )
    st.caption(
        "Consistency is a descriptive failure-detection signal; it does not alter system_state."
    )
    rel = payload.get("reliability", {}).get("reliability", {})
    if rel:
        st.write(
            f"Reliability: `{rel.get('summary')}` (clinical meaning: {rel.get('clinical_meaning')})"
        )
        with st.expander("Reliability basis"):
            for b in rel.get("basis", []):
                st.write(f"- {b}")
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
    # Standing non-diagnosis footer on every result render (usage.md promise:
    # non-clinical status travels with every result, not just the title).
    st.caption(DISCLAIMER)


def _state_badge_html(system_state: str) -> str:
    color, glyph, label = _STATE_STYLE.get(system_state, ("#8B93A7", "●", system_state.upper()))
    return (
        f"<span style='display:inline-flex;align-items:center;gap:7px;"
        f"padding:5px 14px;border-radius:999px;border:1px solid {color}55;"
        f"background:{color}14;color:{color};font-size:0.78rem;font-weight:700;"
        f"letter-spacing:0.08em'>{glyph} {label}</span>"
    )


def _observer_tile(label: str, value: str, detail: str, color: str) -> str:
    dot = (
        f"<span style='width:8px;height:8px;border-radius:50%;"
        f"background:{color};display:inline-block'></span>"
    )
    return (
        f"<div style='border:1px solid #2A3140;border-radius:10px;padding:10px 14px;"
        f"background:#161B26'>"
        f"<div style='display:flex;align-items:center;gap:8px'>"
        f"{dot}<span style='font-size:0.68rem;letter-spacing:0.12em;"
        f"color:#8B93A7;font-weight:700'>{label}</span></div>"
        f"<div style='margin-top:4px;font-size:0.95rem;font-weight:600;color:#E8EDF4'>{value}</div>"
        f"<div style='font-size:0.72rem;color:#8B93A7;margin-top:2px'>{detail}</div>"
        f"</div>"
    )


def _css() -> str:
    return """
    <style>
        section[data-testid="stSidebar"] {
            background: #0B0F16;
            border-right: 1px solid #1E2530;
        }
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%; border-radius: 8px; font-weight: 600;
        }
        .block-container { padding-top: 1.6rem; max-width: 1200px; }
        #MainMenu, header[data-testid="stHeader"] { visibility: hidden; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1E2530; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 18px; border-radius: 8px 8px 0 0;
            font-size: 0.82rem; letter-spacing: 0.08em; font-weight: 600;
            color: #8B93A7; background: transparent;
        }
        .stTabs [aria-selected="true"] {
            color: #E8EDF4; background: #161B26;
            border-bottom: 2px solid #4FB3BF;
        }
        [data-testid="stMetric"] {
            background: #161B26; border: 1px solid #2A3140;
            border-radius: 10px; padding: 12px 16px;
        }
        [data-testid="stMetricLabel"] p { font-size: 0.72rem !important;
            letter-spacing: 0.1em; color: #8B93A7 !important; }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        .stImage img { border-radius: 10px; border: 1px solid #1E2530; }
        .stExpander { border: 1px solid #1E2530 !important; border-radius: 10px; }
        hr { border-color: #1E2530; }
    </style>
    """


def _render_result_card(st, payload: dict, image_bytes: bytes) -> None:
    """Verdict-first styled card: state badge, image | metrics, class chips.

    Provenance labeling and every frozen branch preserved (see render_result).
    """
    if "error" in payload:
        st.error(f"**{payload['error']}** — {payload.get('detail', '')}")
        st.caption(DISCLAIMER)
        return

    # verdict-first: the resolved system state leads the card
    st.markdown(_state_badge_html(payload["system_state"]), unsafe_allow_html=True)

    img_col, verdict_col = st.columns([2, 3])
    with img_col:
        st.image(image_bytes, caption="Input image (as uploaded)", use_container_width=True)
    with verdict_col:
        st.metric("Calibrated confidence", f"{payload['confidence']:.4f}")
        st.progress(min(max(payload["confidence"], 0.0), 1.0), text="confidence")
        st.markdown(
            f"<div style='margin-top:10px'>"
            f"<span style='font-size:0.68rem;letter-spacing:0.12em;color:#8B93A7;"
            f"font-weight:700'>PREDICTED CLASS</span><br>"
            f"<span style='font-size:1.35rem;font-weight:700;"
            f"color:{_CLASS_ACCENT.get(payload['predicted_class'], '#E8EDF4')}'>"
            f"{payload['predicted_class']}</span>"
            f"<span style='color:#8B93A7;font-size:0.85rem'> · "
            f"{payload['classification_state']}</span></div>",
            unsafe_allow_html=True,
        )
        # class chips: accent bar + share per class (descriptive ordering)
        for cls, p in sorted(payload["probabilities"].items(), key=lambda kv: kv[1], reverse=True):
            accent = _CLASS_ACCENT.get(cls, "#8B93A7")
            pct = p * 100
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
                f"<span style='flex:0 0 84px;font-size:0.78rem;color:#8B93A7'>{cls}</span>"
                f"<span style='flex:1;background:#232936;border-radius:3px;height:8px'>"
                f"<span style='display:block;height:8px;width:{pct:.1f}%;"
                f"background:{accent};border-radius:3px'></span></span>"
                f"<span style='flex:0 0 52px;text-align:right;font-size:0.78rem;"
                f"color:#E8EDF4;font-variant-numeric:tabular-nums'>{p:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # observer tiles: quality / consistency / reliability at a glance
    q = payload.get("quality", {})
    q_ok = (q.get("verdict") == "accept") if q else None
    con = payload.get("consistency", {})
    rel = payload.get("reliability", {}).get("reliability", {})
    tiles = st.columns(3)
    with tiles[0]:
        st.markdown(
            _observer_tile(
                "QUALITY",
                f"{q.get('verdict', 'n/a')}".upper(),
                (", ".join(q.get("failed", [])) if q.get("failed") else "Q01–Q09 clear")
                if q
                else "not assessed",
                "#81B29A" if q_ok else ("#E07A5F" if q_ok is False else "#8B93A7"),
            ),
            unsafe_allow_html=True,
        )
    with tiles[1]:
        agreement = con.get("agreement_fraction")
        st.markdown(
            _observer_tile(
                "CONSISTENCY",
                f"{agreement * 100:.0f}%" if isinstance(agreement, (int, float)) else "n/a",
                f"K={con.get('k')} · σ={con.get('sigma')}"
                + (" · FLAGGED" if con.get("flagged") else ""),
                "#E07A5F" if con.get("flagged") else "#81B29A",
            ),
            unsafe_allow_html=True,
        )
    with tiles[2]:
        st.markdown(
            _observer_tile(
                "RELIABILITY",
                str(rel.get("summary", "n/a")).upper(),
                str(rel.get("clinical_meaning", "—")),
                "#81B29A" if rel.get("summary") == "stable" else "#F2CC8F",
            ),
            unsafe_allow_html=True,
        )

    st.subheader("Model output")
    render_result(st, payload, image_bytes=None)


def render_explanation(st, payload: dict, image_bytes: bytes | None = None) -> None:
    """EXPL-001 presentation: report text + visual overlays (observer output)."""
    if "error" in payload:
        st.error(f"**{payload['error']}** — {payload.get('detail', '')}")
        return
    render_result(st, payload, image_bytes=image_bytes)
    st.subheader("Explanation (observer — predictions unchanged)")
    st.text(payload.get("report_text", ""))
    g = payload.get("gradcam", {})
    if g.get("overlay_png_b64"):
        st.image(
            base64.b64decode(g["overlay_png_b64"]),
            caption=f"Grad-CAM ({g.get('target_layer')}) — contribution "
            "visualization, no clinical meaning claimed",
            use_container_width=True,
        )
    sv = payload.get("segmentation_vis")
    if sv and sv.get("overlay_png_b64"):
        st.image(
            base64.b64decode(sv["overlay_png_b64"]),
            caption="Segmentation overlay (mask + bbox + centroid)",
            use_container_width=True,
        )
        if sv.get("focus_in_bbox") is not None:
            st.write(f"CAM mass inside bbox: {sv['focus_in_bbox']:.4f} (descriptive)")


def _run_analysis(st, svc, data: bytes, image_bytes: bytes) -> None:
    """Shared analyze path: spinner, cache into session, render card + observers."""
    with st.spinner("Running classification, segmentation, and observers…"):
        payload = build_payload(svc, data)
    st.session_state["bt_payload"] = payload
    st.session_state["bt_image_bytes"] = image_bytes
    if "error" in payload:
        st.error(f"**{payload['error']}** — {payload.get('detail', '')}")
        st.caption(DISCLAIMER)
        return
    _render_result_card(st, payload, image_bytes)


def _run_explain(st, svc, data: bytes, image_bytes: bytes) -> None:
    """Explain path: mirrors /explain semantics (explain handles its own guards)."""
    with st.spinner("Building explanation (Grad-CAM + overlays)…"):
        from PIL import Image

        try:
            payload = svc.explain(Image.open(io.BytesIO(data)), raw=data)
        except Exception as e:  # noqa: BLE001 — never leak stack/paths
            st.error(f"**inference_failed** — {type(e).__name__}")
            st.caption(DISCLAIMER)
            return
    st.session_state["bt_explain"] = payload
    st.session_state["bt_image_bytes"] = image_bytes
    render_explanation(st, payload, image_bytes=image_bytes)


if __name__ == "__main__":
    try:
        import streamlit as st

        st.set_page_config(
            page_title="BrainTumor-MajorProject",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        @st.cache_resource
        def get_inference_service() -> InferenceService:
            return InferenceService.from_registry()

        st.markdown(_css(), unsafe_allow_html=True)

        # ---- sidebar: upload + run controls --------------------------------
        with st.sidebar:
            st.markdown(
                "<div style='margin:4px 0 14px 0'>"
                "<span style='font-size:1.25rem;font-weight:700;color:#E8EDF4'>"
                "🧠 BrainTumor Console</span><br>"
                "<span style='font-size:0.74rem;letter-spacing:0.1em;color:#8B93A7'>"
                "RESEARCH PROTOTYPE — NOT CLINICAL</span></div>",
                unsafe_allow_html=True,
            )
            svc = get_inference_service()
            clf_loaded = svc.clf is not None
            seg_available = bool(svc.segmentation_available)
            led = lambda ok: (  # noqa: E731 — tiny inline status dot helper
                f"<span style='display:inline-block;width:8px;height:8px;"
                f"border-radius:50%;background:{'#81B29A' if ok else '#E06C75'};"
                f"margin-right:6px'></span>"
            )
            st.markdown(
                f"<div style='font-size:0.8rem;color:#8B93A7;line-height:1.9'>"
                f"{led(clf_loaded)}classifier {'loaded' if clf_loaded else 'unavailable'}<br>"
                f"{led(seg_available)}segmenter {'loaded' if seg_available else 'unavailable'}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            up = st.file_uploader(
                "Upload MRI slice",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                key="bt_uploader",
            )
            has_file = up is not None
            data = up.getvalue() if has_file else b""
            run_analyze = st.button("Analyze", type="primary", disabled=not has_file)
            run_explain = st.button("Explain", disabled=not has_file)
            if has_file:
                st.caption(f"{up.name} · {len(data) / (1024 * 1024):.2f} MB")

        # ---- main: tabs (Analyze | Explanation) ----------------------------
        analyze_tab, explain_tab = st.tabs(["Analyze", "Explanation"])
        with analyze_tab:
            if run_analyze and has_file:
                _run_analysis(st, svc, data, data)
            elif "bt_payload" in st.session_state and "bt_image_bytes" in st.session_state:
                cached = st.session_state["bt_payload"]
                if "error" not in cached:
                    _render_result_card(st, cached, st.session_state["bt_image_bytes"])
                else:
                    st.caption(
                        "Previous run ended with an error — upload a new image and press Analyze."
                    )
            else:
                st.markdown(
                    "<div style='text-align:center;padding:56px 0;color:#8B93A7'>"
                    "<div style='font-size:2.4rem'>🧠</div>"
                    "<div style='margin-top:8px;font-size:0.92rem'>"
                    "Upload an MRI slice in the sidebar, then press "
                    "<b style='color:#4FB3BF'>Analyze</b>.</div></div>",
                    unsafe_allow_html=True,
                )
        with explain_tab:
            if run_explain and has_file:
                _run_explain(st, svc, data, data)
            elif "bt_explain" in st.session_state and "bt_image_bytes" in st.session_state:
                render_explanation(
                    st, st.session_state["bt_explain"], st.session_state["bt_image_bytes"]
                )
            else:
                st.markdown(
                    "<div style='text-align:center;padding:56px 0;color:#8B93A7'>"
                    "<div style='font-size:2.4rem'>🔍</div>"
                    "<div style='margin-top:8px;font-size:0.92rem'>"
                    "Press <b style='color:#4FB3BF'>Explain</b> to build Grad-CAM "
                    "and segmentation overlays for the current image.</div></div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption(DISCLAIMER)
    except ImportError:
        print("streamlit not installed")
