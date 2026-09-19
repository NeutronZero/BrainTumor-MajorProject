"""Classifier factories — evidence candidates, fresh weights only."""

from __future__ import annotations

HEADS = {"convnext_tiny": "classifier", "densenet121": "classifier",
         "efficientnet_b0": "classifier", "resnet50": "fc"}

_ENUMS = {"convnext_tiny": "ConvNeXt_Tiny_Weights", "densenet121": "DenseNet121_Weights",
          "efficientnet_b0": "EfficientNet_B0_Weights", "resnet50": "ResNet50_Weights"}


def _enum_name(name: str) -> str:
    return _ENUMS[name]


def state_for_ckpt(model):
    """Strip DataParallel `.module` prefix on save (prior B8 lesson)."""
    from brain_tumor.utils.engine import _state_for_ckpt
    return _state_for_ckpt(model)


def build_classifier(name: str, n_classes: int = 4, pretrained: bool = False):
    """Weights API hardened per prior audit (B8): enum first, string alias,
    legacy pretrained= fallback. Never passes bare "DEFAULT" string."""
    import torchvision.models as M
    if name not in HEADS:
        raise ValueError(f"unknown architecture {name!r}")
    try:
        # Preferred: get_model with standard alias (torchvision >= 0.13).
        model = M.get_model(name, weights="IMAGENET1K_V1" if pretrained else None)
    except Exception:
        fn = {"convnext_tiny": M.convnext_tiny, "densenet121": M.densenet121,
              "efficientnet_b0": M.efficientnet_b0, "resnet50": M.resnet50}[name]
        try:
            enum = getattr(getattr(M, _enum_name(name)), "DEFAULT", None)
            model = fn(weights=enum if pretrained else None)
        except TypeError:  # older torchvision
            model = fn(pretrained=pretrained)
    import torch.nn as nn
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    elif name == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    else:  # convnext_tiny: classifier = [LayerNorm, Flatten, Linear]
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    return model
