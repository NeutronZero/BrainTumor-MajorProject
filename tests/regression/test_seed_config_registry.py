"""Regression: coverage gate. The CI coverage gate (fail_under=70) counts the
unfrozen utility modules seed/config/registry; these tests pin their behavior
(deterministic seeding, hashed config loading, best.pt-only registry) and keep
total coverage above the gate.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from brain_tumor.config import (
    load_config_with_hash,
    load_yaml,
    sha256_dict,
    sha256_file,
)
from brain_tumor.registry import ModelEntry, ModelRegistry
from brain_tumor.utils.seed import (
    make_generator,
    set_global_seed,
    worker_init_fn,
)


def test_set_global_seed_reproducible():
    set_global_seed(1234)
    a = (random.random(), np.random.rand(), torch.rand(3).tolist())
    set_global_seed(1234)
    b = (random.random(), np.random.rand(), torch.rand(3).tolist())
    assert a == b


def test_set_global_seed_varies_by_seed():
    set_global_seed(1)
    a = (random.random(), np.random.rand())
    set_global_seed(2)
    b = (random.random(), np.random.rand())
    assert a != b


def test_worker_init_fn_deterministic_per_worker(tmp_path):
    worker_init_fn(0, base_seed=42)
    a = np.random.rand()
    worker_init_fn(1, base_seed=42)
    b = np.random.rand()
    assert a != b
    worker_init_fn(0, base_seed=42)
    assert np.random.rand() == a


def test_make_generator_seeded():
    g1 = make_generator(7)
    g2 = make_generator(7)
    assert g1 is not None and g2 is not None
    assert torch.rand(4, generator=g1).tolist() == torch.rand(4, generator=g2).tolist()
    assert torch.rand(4, generator=g1).tolist() != torch.rand(4, generator=make_generator(8)).tolist()


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01braintumor" * 1000)
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_sha256_dict_order_independent():
    assert sha256_dict({"b": 1, "a": [1, 2]}) == sha256_dict({"a": [1, 2], "b": 1})
    assert len(sha256_dict({})) == 64


def test_load_yaml_and_hash(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"tau1": 0.95, "tags": ["a"]}), encoding="utf-8")
    cfg = load_yaml(p)
    assert cfg["tau1"] == 0.95
    cfg2, h = load_config_with_hash(p)
    assert cfg2 == cfg and h == sha256_dict(cfg)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}


def test_registry_rejects_non_best_pt(tmp_path):
    reg = ModelRegistry(tmp_path)
    with pytest.raises(ValueError, match="best.pt"):
        reg.register(ModelEntry(name="x", experiment_id="X", checkpoint=tmp_path / "last.pt"))


def test_registry_register_get_available(tmp_path):
    reg = ModelRegistry(tmp_path)
    present = tmp_path / "best.pt"
    present.write_bytes(b"ckpt")
    entry = ModelEntry(name="classifier", experiment_id="CLS-001", checkpoint=present)
    assert entry.sha256 is None
    reg.register(entry)
    reg.register(ModelEntry(name="missing", experiment_id="SEG-001",
                            checkpoint=tmp_path / "sub" / "best.pt"))
    assert reg.get("classifier").experiment_id == "CLS-001"
    assert reg.available() == {"classifier": True, "missing": False}
    with pytest.raises(KeyError):
        reg.get("nope")


def test_registry_default_entries():
    reg = ModelRegistry.default(Path("/proj"))
    assert reg.root == Path("/proj/checkpoints")
    assert reg.get("classifier").experiment_id == "CLS-001"
    assert reg.get("segmenter").experiment_id == "SEG-001"
    for e in (reg.get("classifier"), reg.get("segmenter")):
        assert e.checkpoint.name == "best.pt"
    assert json.dumps(reg.available())  # JSON-serializable bools
