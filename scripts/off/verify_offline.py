"""OFF-001 offline verification: socket-blocked inference path.

Establishes offline operation by disabling network access at the socket
layer AFTER imports, then runs startup + checkpoint load + synthetic
inference + /health + API path with the guard active. Any outbound
connection attempt raises and is counted (expect zero during inference).
A control connection attempt proves the guard itself works.

No downloads, no dependency/model/config changes, no test data.
Writes outputs/OFF/*.json. Deterministic synthetic input (seed 31) with
already-established expectations (pituitary / tumor_unlocalized).
"""

from __future__ import annotations

import io
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

BLOCKED = {"count": 0}

_REAL_SOCKET = socket.socket


class _GuardedSocket(_REAL_SOCKET):
    """Block non-loopback connects (external network); loopback stays up so
    asyncio/httpx teardown works. Counts + raises on external attempts."""

    @staticmethod
    def _external(address) -> bool:
        try:
            host = address[0] if isinstance(address, tuple) else str(address)
        except Exception:  # noqa: BLE001
            return True
        h = str(host).lower()
        return not (h.startswith("127.") or h in ("::1", "localhost"))

    def connect(self, *a, **k):
        if self._external(a[0] if a else ""):
            BLOCKED["count"] += 1
            raise OSError("OFF-001: network disabled")
        return super().connect(*a, **k)

    def connect_ex(self, *a, **k):
        if self._external(a[0] if a else ""):
            BLOCKED["count"] += 1
            return 1
        return super().connect_ex(*a, **k)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "app" / "api"))
    out = root / "outputs" / "OFF"
    out.mkdir(parents=True, exist_ok=True)

    # Imports first (no network needed for local imports by construction).
    from brain_tumor.inference.service import InferenceService  # noqa: E402

    # Engage socket block for everything below.
    socket.socket = _GuardedSocket  # type: ignore
    env = {"guard": "non-loopback socket connects raise; loopback exempt "
                    "(asyncio teardown); external attempts counted",
           "control": None, "inference_blocked_calls": None}

    # Control: outbound attempt through the PATCHED class must fail.
    try:
        socket.socket().connect(("8.8.8.8", 53))
        env["control"] = "GUARD-BROKEN: connect succeeded"
    except OSError:
        env["control"] = "guard effective (control connect raised)"
    env["control_probe_blocks"] = BLOCKED["count"]
    (out / "offline_environment.json").write_text(json.dumps(env, indent=1))

    # Startup + checkpoint load under guard.
    t0 = time.perf_counter()
    svc = InferenceService.from_registry()
    startup = {"classifier": svc.clf is not None, "segmenter": svc.seg is not None,
               "load_s": round(time.perf_counter() - t0, 2),
               "T": svc.T, "tau1": svc.tau1, "tau2": svc.tau2}
    (out / "startup.json").write_text(json.dumps(startup, indent=1))

    # Deterministic synthetic inference (established expectations).
    from PIL import Image
    rng = np.random.RandomState(31)
    buf = io.BytesIO()
    Image.fromarray((rng.rand(256, 256) * 255).astype("uint8")).save(buf, format="PNG")
    raw, img = buf.getvalue(), Image.open(io.BytesIO(buf.getvalue()))
    r = svc.analyze(img).model_dump()
    inference = {"predicted_class": r["predicted_class"],
                 "system_state": r["system_state"],
                 "matches_established": (r["predicted_class"] == "pituitary"
                                         and r["system_state"] == "tumor_unlocalized"),
                 "consistency_k": svc.consistency(img)["k"],
                 "quality": svc.quality(raw)["verdict"]}
    (out / "inference.json").write_text(json.dumps(inference, indent=1))

    # API path under guard (in-process transport, no sockets by design).
    import main as api_main
    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    h = client.get("/health").json()
    a = client.post("/analyze", files={"file": ("t.png", raw, "image/png")}).json()
    api = {"health": h["status"], "analyze_state": a["system_state"],
           "cross_match": a["system_state"] == r["system_state"]}
    (out / "network_check.json").write_text(json.dumps({
        "blocked_calls_during_inference": BLOCKED["count"] - env["control_probe_blocks"],
        "api_path_works_offline": api["health"] == "ok"}, indent=1))

    verdict = {
        "guard_effective": env["control"].startswith("guard effective"),
        "startup_ok": startup["classifier"] and startup["segmenter"],
        "inference_matches": inference["matches_established"],
        "api_ok": api["health"] == "ok" and api["cross_match"],
        "zero_blocked_calls": (BLOCKED["count"] - env["control_probe_blocks"]) == 0,
    }
    verdict["PASS"] = all(verdict.values())
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    (out / "api.json").write_text(json.dumps(api, indent=1))
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
