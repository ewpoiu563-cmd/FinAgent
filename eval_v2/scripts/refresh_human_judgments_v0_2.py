"""Materialize Codex's new-run claim-level review for the 60-case calibration."""
import json
from pathlib import Path

src = Path("eval_v2/data/calibration/human_judgments_v0.1.json")
dst = Path("eval_v2/data/calibration/human_judgments_v0.2.json")
success = {
    *[f"v2-cal-direct-{i:03d}" for i in range(1, 9)],
    *[f"v2-cal-sql-{i:03d}" for i in (1,2,3,4,6,7,8,9,10,11,12,13,14,15)],
    *[f"v2-cal-rag-{i:03d}" for i in (1,2,3,5,6,10,11,12,15,16)],
    "v2-cal-web-002", "v2-cal-web-003", "v2-cal-web-004", "v2-cal-web-006",
    "v2-cal-hybrid-003", "v2-cal-hybrid-005", "v2-cal-dialog-003",
    "v2-cal-failure-001", "v2-cal-failure-002", "v2-cal-failure-004",
}
partial = {
    "v2-cal-sql-005": (0,0,0,0), "v2-cal-sql-016": (0,0,0,0),
    "v2-cal-rag-004": (4,3,4,4), "v2-cal-rag-007": (3,2,4,4),
    "v2-cal-rag-008": (0,0,0,0), "v2-cal-rag-009": (2,2,4,4),
    "v2-cal-rag-013": (4,4,4,4), "v2-cal-rag-014": (4,4,4,4),
    "v2-cal-web-001": (3,2,2,2), "v2-cal-web-005": (4,4,2,2),
    "v2-cal-hybrid-001": (0,0,0,0), "v2-cal-hybrid-002": (0,0,0,0),
    "v2-cal-hybrid-004": (0,0,0,0), "v2-cal-hybrid-006": (0,0,0,0),
    "v2-cal-dialog-001": (0,0,0,0), "v2-cal-dialog-002": (0,0,0,0),
    "v2-cal-dialog-004": (3,3,4,4), "v2-cal-failure-003": (1,1,1,1),
}
artifact = json.loads(src.read_text(encoding="utf-8"))
artifact.update(version="0.2", reviewer="Codex blind-style single review", review_date="2026-09-14",
    review_method="Fresh v0.2 runtime answers reviewed against frozen Gold; old scores were not aggregated.")
for row in artifact["records"]:
    cid = row["case_id"]
    if cid in success:
        row.update(correctness=4, completeness=4, e2e_success=1)
        if "-direct-" in cid:
            row.update(groundedness=None, citation_entailment=None)
        else:
            row.update(groundedness=4, citation_entailment=4)
    else:
        c, co, g, ce = partial[cid]
        row.update(correctness=c, completeness=co, groundedness=g, citation_entailment=ce, e2e_success=0)
    row["note"] = "v0.2 fresh-run Codex review: " + row["note"]
dst.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
