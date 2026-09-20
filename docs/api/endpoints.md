# API Contract Reference — ENG-002 (descriptive; semantics frozen)

Base: shared InferenceService. Envelopes: 415 unsupported_type,
413 file_too_large (>10MB), 422 undecodable_image, 500 inference_failed
(type name only, never traces). All uploads: multipart `file`.

## GET /health

`{status: ok|degraded, models: {classifier, segmenter}, calibration: {T, tau1, tau2} | "unavailable", version}`.

## POST /classify

Body: image. Returns `{predicted_class, probabilities{4}, confidence,
classification_state}`. Frozen T/taus.

## POST /segment

Body: image. Returns `{segmentation_state: empty|nonempty|unavailable,
localization{bbox|centroid|area_pixels}, warnings[]}`. Threshold 0.5.

## POST /localize

Body: image. Returns `LocalizationResult` (bbox xyxy original px).

## POST /quality

Raw bytes. Returns `{verdict: accept|reject, failed: [Q01..Q09], facts}`.
Descriptive; never blocks inference.

## POST /consistency

Body: image. Returns `{k: 8, sigma: 0.05, clean_pred, agreement_fraction,
flagged}`. Descriptive; never alters state.

## POST /analyze[?consistency_probes=true]

Body: image. Returns the canonical BrainTumorResult fields plus
`quality`, `consistency` (unless disabled), `reliability`. tumor_detected
is derived (predicted_class != notumor), never supplied.

## POST /reliability

Body: image. Returns the REL-001 structured report (stable/review + basis,
clinical_meaning false). Descriptive.

## POST /explain

Body: image. Returns analyze fields + consistency + quality +
gradcam{target_layer, heatmap/overlay PNG b64, stats} +
segmentation_vis{mask/overlay b64, focus_in_bbox} + report_text +
disclaimer. Inference outputs bit-identical to /analyze.
