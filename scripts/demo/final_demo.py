"""Render the final deterministic demonstration record.

Composes outputs/DEMO/demo_matrix.json + walkthrough.json + acceptance_api.json
(all committed acceptance evidence) into a narrated outputs/DEMO/final_demo.md.
No model execution, no test contact — pure documentation rendering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "DEMO"
    m = json.loads((out / "demo_matrix.json").read_text())
    w = json.loads((out / "walkthrough.json").read_text())
    a = json.loads((out / "acceptance_api.json").read_text())
    assert not [c for c in m["cases"] if not c["contract_holds"]], "fixture drift"
    assert a["failed"] == [], a["failed"]
    L = [
        "# Final Demonstration Record (deterministic, synthetic inputs only)",
        "",
        "Narrated walkthrough of the frozen system. Every step below is",
        "pinned by committed acceptance evidence; re-running the demo suite",
        "reproduces these outcomes exactly.",
        "",
    ]
    for c in m["cases"]:
        o = c["observed"]
        L += [
            f"## {c['case']}",
            f"- Expected contract: `{c['expected']}`",
            f"- Observed: class={o.get('predicted_class')}, "
            f"state={o.get('classification_state')}, "
            f"system={o.get('system_state')}, seg={o.get('segmentation_state')}, "
            f"quality={o.get('quality')}",
            "- Contract holds: yes",
            "",
        ]
    L += [
        "## Operator walkthrough",
        "Recorded steps: " + " → ".join(s["step"] for s in w["steps"]),
        "",
        "## API acceptance",
        "Schema regression: no failures. Service/API equivalence holds.",
        "",
        "> Research prototype — not clinical. Demonstrations show system",
        "> contracts, never diagnostic accuracy.",
        "",
    ]
    (out / "final_demo.md").write_text("\n".join(L), encoding="utf-8")
    print(f"final_demo.md written ({len(m['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
