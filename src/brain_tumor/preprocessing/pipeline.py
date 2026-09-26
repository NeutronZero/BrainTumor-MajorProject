"""Versioned preprocessing factories — train/infer share implementation (§29)."""

from __future__ import annotations

import random


def _resize(size: tuple[int, int], *, for_mask: bool = False):
    """Resize with antialias where supported (prior B8 API-drift surface)."""
    from torchvision import transforms

    kwargs = {} if for_mask else {"antialias": True}
    interp = (
        transforms.InterpolationMode.NEAREST if for_mask else transforms.InterpolationMode.BILINEAR
    )
    try:
        return transforms.Resize(size, interpolation=interp, **kwargs)
    except TypeError:  # old torchvision without antialias kwarg
        return transforms.Resize(size, interpolation=interp)


def build_cls_transform(
    train: bool, image_size: int = 224, hflip_p: float = 0.5, rot_deg: float = 10.0
):
    from torchvision import transforms

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    aug = (
        [
            transforms.RandomHorizontalFlip(p=hflip_p),
            transforms.RandomRotation(degrees=rot_deg),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
        if train
        else []
    )
    return transforms.Compose(
        [
            *aug,
            transforms.Grayscale(num_output_channels=3),
            _resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_seg_pair_transform(
    train: bool,
    image_size: int = 256,
    mean: float = 0.0,
    std: float = 1.0,
    hflip_p: float = 0.5,
    rot_deg: float = 10.0,
    gauss_noise_sigma: float = 0.0,
):
    """Synced image/mask geometry: identical flips/rotations/affines.

    Randomness uses the global torch/python RNG (seeded per §28); eval path is
    fully deterministic. Mask resized NEAREST, binarized >127 AFTER geometry.
    gauss_noise_sigma: ROB-001 factor — train-only additive Gaussian on the
    [0,1] image tensor BEFORE normalization (mask untouched; photometric, so
    geometry identical per §30). Default 0.0 = frozen SEG-001 behavior.
    """
    import torch

    geom_img = _resize((image_size, image_size))
    geom_mask = _resize((image_size, image_size), for_mask=True)

    def _apply(img, mask):
        import torchvision.transforms.functional as F

        if train:
            if random.random() < hflip_p:
                img, mask = F.hflip(img), F.hflip(mask)
            angle = random.uniform(-rot_deg, rot_deg)
            if angle:
                img = F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        img = geom_img(img)
        mask = geom_mask(mask)
        img = F.to_tensor(img)
        if train and gauss_noise_sigma > 0:
            img = torch.clamp(img + torch.randn_like(img) * gauss_noise_sigma, 0, 1)
        img = F.normalize(img, mean=[mean], std=[std])
        m = F.to_tensor(mask)
        m = (m > 127 / 255).to(torch.float32)
        return img, m

    return _apply
