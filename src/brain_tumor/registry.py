"""Model registry — only best.pt is production (§56)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEntry:
    name: str
    experiment_id: str
    checkpoint: Path  # must be best.pt
    sha256: str | None = None


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._entries: dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> None:
        if entry.checkpoint.name != "best.pt":
            raise ValueError("registry only references best.pt (§56)")
        self._entries[entry.name] = entry

    def get(self, name: str) -> ModelEntry:
        return self._entries[name]

    def available(self) -> dict[str, bool]:
        return {k: v.checkpoint.exists() for k, v in self._entries.items()}

    @classmethod
    def default(cls, project_root: Path) -> ModelRegistry:
        reg = cls(project_root / "checkpoints")
        reg.register(
            ModelEntry(
                name="classifier",
                experiment_id="CLS-001",
                checkpoint=project_root / "checkpoints" / "CLS-001" / "best.pt",
            )
        )
        reg.register(
            ModelEntry(
                name="segmenter",
                experiment_id="SEG-001",
                checkpoint=project_root / "checkpoints" / "SEG-001" / "best.pt",
            )
        )
        return reg
