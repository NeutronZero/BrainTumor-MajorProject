"""Canonical contracts — BrainTumor-MajorProject v2.3 §21-25 (FROZEN).

Five freezes applied:
 1. class vocabulary: glioma | meningioma | pituitary | notumor
 2. localization: bbox xyxy in ORIGINAL input pixels, centroid (cx,cy),
    area_pixels = sum of qualifying components
 3. calibration: T + tau1/tau2 selected on validation only, then locked
 4. split: official test 1000 LOCKED; train pool 5000 -> 4000/1000
 5. single InferenceService shared by FastAPI + Streamlit

tumor_detected is DERIVED (predicted_class != "notumor"), never supplied.
system_state precedence: degraded > uncertain > tumor_unlocalized
  > tumor_localized > healthy. Contradictions raise ValidationError.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ClassLabel = Literal["glioma", "meningioma", "pituitary", "notumor"]
ClassificationState = Literal["confident", "uncertain"]
SegmentationState = Literal["available", "unavailable", "empty", "nonempty"]
SystemState = Literal[
    "degraded", "uncertain", "tumor_unlocalized", "tumor_localized", "healthy"
]
WarningCode = Literal[
    "segmentation_unavailable",
    "segmentation_empty",
    "low_confidence",
    "multiple_components",
    # "oversized_component_filtered" is a RESERVED code: the frozen extractor
    # (localization/extract.py) implements no oversized-component filtering and
    # never emits it. Adding such filtering would alter the frozen localization
    # semantics (largest_plus_total at threshold 0.5, min_area 10), so it stays
    # dead vocabulary here — accepted by the contract, produced by nothing.
    "oversized_component_filtered",
]

CANONICAL_CLASSES: tuple[str, ...] = ("glioma", "meningioma", "pituitary", "notumor")

# Localization frozen schema (§48-49 + freeze #3)
LOCALIZATION_SCHEMA = {
    "bbox_format": "xyxy",  # [x_min, y_min, x_max, y_max], ints
    "coordinate_space": "original_input_pixels",
    "centroid": "center of LARGEST qualifying component, (cx, cy) float in original pixels",
    "area_pixels": "sum of ALL qualifying components, int >= 0",
    "threshold": 0.5,
    "component_policy": "largest_plus_total",
    "min_area_pixels": 10,
    "empty_result": {"bbox": None, "centroid": None, "area_pixels": 0},
}


class ClassificationResult(BaseModel):
    predicted_class: ClassLabel
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    classification_state: ClassificationState

    @field_validator("probabilities")
    @classmethod
    def _check_probs(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v.keys()) != set(CANONICAL_CLASSES):
            raise ValueError(f"probabilities keys must be exactly {CANONICAL_CLASSES}")
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"probabilities must sum to 1.0, got {total}")
        for k, p in v.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probability {k}={p} out of range")
        return v

    @model_validator(mode="after")
    def _check_confidence_consistent(self) -> "ClassificationResult":
        top = max(self.probabilities.values())
        if abs(top - self.confidence) > 1e-6:
            raise ValueError("confidence must equal max(probabilities)")
        top_cls = max(self.probabilities, key=lambda k: self.probabilities[k])
        if top_cls != self.predicted_class:
            raise ValueError("predicted_class must be argmax(probabilities)")
        return self


class LocalizationResult(BaseModel):
    """Frozen: bbox xyxy ints in original input pixels; None when empty."""

    bbox: Optional[tuple[int, int, int, int]] = None
    centroid: Optional[tuple[float, float]] = None
    area_pixels: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_empty_consistent(self) -> "LocalizationResult":
        if self.area_pixels == 0:
            if self.bbox is not None or self.centroid is not None:
                raise ValueError("empty localization must have bbox=None, centroid=None")
        else:
            if self.bbox is None or self.centroid is None:
                raise ValueError("non-empty localization requires bbox + centroid")
            x0, y0, x1, y1 = self.bbox
            if not (x1 > x0 and y1 > y0):
                raise ValueError(f"invalid bbox xyxy={self.bbox}")
        return self


class BrainTumorResult(BaseModel):
    """Canonical inference contract. tumor_detected is derived, not supplied."""

    predicted_class: ClassLabel
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    classification_state: ClassificationState
    segmentation_state: SegmentationState
    localization: LocalizationResult = Field(default_factory=LocalizationResult)
    warnings: list[WarningCode] = Field(default_factory=list)
    system_state: SystemState

    @property
    def tumor_detected(self) -> bool:
        return self.predicted_class != "notumor"

    @field_validator("probabilities")
    @classmethod
    def _check_probs(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v.keys()) != set(CANONICAL_CLASSES):
            raise ValueError(f"probabilities keys must be exactly {CANONICAL_CLASSES}")
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"probabilities must sum to 1.0, got {total}")
        return v

    @model_validator(mode="after")
    def _enforce_invariants(self) -> "BrainTumorResult":
        pc = self.predicted_class
        ss = self.system_state
        seg = self.segmentation_state
        cs = self.classification_state
        w = self.warnings
        area = self.localization.area_pixels

        # Precedence + invariants (§24)
        if ss == "healthy" and pc != "notumor":
            raise ValueError("healthy requires predicted_class == notumor")
        if ss == "healthy" and cs != "confident":
            raise ValueError("healthy requires confident classification")
        if ss != "degraded" and (seg == "unavailable"
                                 or "segmentation_unavailable" in w):
            raise ValueError(
                "segmentation unavailable must derive system_state == degraded")
        if ss == "tumor_localized":
            if pc == "notumor":
                raise ValueError("tumor_localized requires tumor class")
            if seg != "nonempty":
                raise ValueError("tumor_localized requires nonempty mask")
            if area <= 0:
                raise ValueError("tumor_localized requires area_pixels > 0")
        if ss == "tumor_unlocalized":
            if pc == "notumor":
                raise ValueError("tumor_unlocalized requires tumor class")
            if seg != "empty":
                raise ValueError("tumor_unlocalized requires empty mask")
        if ss == "degraded" and "segmentation_unavailable" not in w:
            raise ValueError("degraded requires segmentation_unavailable warning")
        if ss == "uncertain" and cs != "uncertain":
            raise ValueError("uncertain system_state requires uncertain classification")
        if seg == "unavailable" and "segmentation_unavailable" not in w:
            raise ValueError("unavailable segmentation requires warning")
        if seg == "empty" and area != 0:
            raise ValueError("empty segmentation requires area_pixels == 0")
        if seg == "nonempty" and area <= 0:
            raise ValueError("nonempty segmentation requires area_pixels > 0")
        # argmax/confidence invariant — same guarantee ClassificationResult
        # carries (M1): 6-dp wire rounding stays within the 1e-6 tolerance.
        if abs(max(self.probabilities.values()) - self.confidence) > 1e-6:
            raise ValueError("confidence must equal max(probabilities)")
        if max(self.probabilities, key=lambda k: self.probabilities[k]) != pc:
            raise ValueError("predicted_class must be argmax(probabilities)")
        return self


def derive_system_state(
    predicted_class: str,
    classification_state: str,
    segmentation_state: str,
    warnings: list[str],
) -> str:
    """Precedence: degraded > uncertain > tumor_unlocalized > tumor_localized > healthy."""
    if segmentation_state == "unavailable" or "segmentation_unavailable" in warnings:
        return "degraded"
    if classification_state == "uncertain":
        return "uncertain"
    if predicted_class != "notumor" and segmentation_state == "empty":
        return "tumor_unlocalized"
    # "nonempty" only: seg=="available" (mask not computed) has no
    # contract-constructible tumor state — deriving tumor_localized here would
    # contradict the BrainTumorResult validator. Such inputs now derive
    # "uncertain" (fall-through), which IS constructible.
    if predicted_class != "notumor" and segmentation_state == "nonempty":
        return "tumor_localized"
    if predicted_class == "notumor":
        return "healthy"
    return "uncertain"
