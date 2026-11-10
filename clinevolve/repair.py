"""Single-target repair of a fresh task agent (A0) by one repairer program.

Every candidate starts from A0 and receives the identical feedback package and objective. Candidates are generated in
waves (1..5, 6..10, ...), evaluated on the validation cohort and gated against A0 (G6 against an A0 replicate run). For
each budget k the released version is the gate-passing candidate with the largest validation gain on the target among the
first k candidates (ties: fewer mean tokens, then candidate index), or A0 if none passed. The improvement of that version
over A0 on held-out patients is CI@k.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from clinevolve.config import REPO_ROOT, SHORT
from clinevolve.gates import evaluate_gates, objective_text
from clinevolve.runtime import Runtime, log_to, save_json, sha256


def common_inputs(rt: Runtime, target: str, root: Path, partition: int, cases: Optional[Path] = None) -> Dict[str, str]:
    """A0 repair-set feedback package and A0 validation evaluations (run twice: A0 and its replicate A0rep)."""
    common = Path(root) / "common"
    fb = common / f"feedback_{SHORT[target]}.json"
    if not fb.exists():
        rep = rt.evaluate(REPO_ROOT, "A0", "repair", common, partition, cases)
        rt.build_feedback(rep, target, fb, "A0", partition, cases=cases)
    return {"feedback": str(fb),
            "a0_val": str(rt.evaluate(REPO_ROOT, "A0", "validation", common, partition, cases)),
            "a0rep_val": str(rt.evaluate(REPO_ROOT, "A0rep", "validation", common, partition, cases))}


def repair(rt: Runtime, *, meta_path: Path, target: str, out: Path, partition: int = 1, tag: str = "M",
           feedback: Optional[Path] = None, cases: Optional[Path] = None) -> Dict[str, Any]:
    cfg = rt.cfg["repair"]
    out = Path(out)
    br = out / f"branches/{tag}_{SHORT[target]}"
    br.mkdir(parents=True, exist_ok=True)
    state_p, log = br / "state.json", br / "driver.log"
    st = json.loads(state_p.read_text()) if state_p.exists() else {
        "meta_id": tag, "target": target, "partition": partition, "meta_program_hash": sha256(meta_path),
        "candidates": {}}
    if st["meta_program_hash"] != sha256(meta_path):
        raise RuntimeError("repairer program changed after this repair run started")
    ci = common_inputs(rt, target, out, partition, cases)
    fb = Path(feedback) if feedback else Path(ci["feedback"])
    st["feedback_sha256"], st["feedback_path"] = sha256(fb), str(fb)
    save_json(state_p, st)
    a0rec = rt.a0_record(partition, target)
    objective = objective_text(target)
    G = rt.cfg["gates"]
    log_to(log, f"start repairer={tag} hash={st['meta_program_hash'][:12]} feedback={st['feedback_sha256'][:12]}")

    def one(i: int) -> str:
        cid = f"p{partition}_{tag}_{SHORT[target]}_c{i:02d}"
        c = st["candidates"].get(cid)
        if c is None:
            rec = rt.generate(meta_path=meta_path, parent_repo=REPO_ROOT, parent_record=a0rec, ancestors=[], cid=cid,
                              target=target, objective=objective, feedback=fb, root=br, partition=partition)
            c = {"index": i, "status": rec.get("generation_status"), "record": rec, "repo": rec.get("repo_path"),
                 "lines": (rec.get("lines_added") or 0) + (rec.get("lines_deleted") or 0), "usage": rec.get("_usage")}
            st["candidates"][cid] = c
            save_json(state_p, st)
        if c["status"] == "generated" and "gates" not in c:
            c["validation"] = str(rt.evaluate(Path(c["repo"]), cid, "validation", br, partition, cases))
            c["gates"] = evaluate_gates(G, Path(ci["a0_val"]), Path(c["validation"]), Path(ci["a0_val"]), target,
                                        a0_replicate_val=Path(ci["a0rep_val"]), reference_is_a0=True)
            save_json(state_p, st)
        return cid

    lo = 0
    for hi in cfg["generation_waves"]:
        with ThreadPoolExecutor(max(1, hi - lo)) as ex:
            list(ex.map(one, range(lo + 1, hi + 1)))
        lo = hi
        log_to(log, f"wave <= {hi}: " + ", ".join(
            f"{k}:{v['status'][:3]}:{'P' if v.get('gates', {}).get('all_pass') else '-'}"
            for k, v in sorted(st["candidates"].items())))
    st["selection"] = select(st, cfg["budgets_k"])
    st["done"] = True
    save_json(state_p, st)
    log_to(log, f"done selection={st['selection']}")
    return st


def select(st: Dict[str, Any], budgets) -> Dict[str, Any]:
    cands = sorted(st["candidates"].items(), key=lambda kv: kv[1]["index"])
    safe = [(k, v) for k, v in cands if v.get("gates", {}).get("all_pass")]
    sel = {}
    for k in budgets:
        pool = [(cid, v) for cid, v in safe if v["index"] <= k]
        best = sorted(pool, key=lambda kv: (-kv[1]["gates"]["target"]["gain_pp"], kv[1]["gates"]["mean_tokens"],
                                            kv[1]["index"]))
        sel[str(k)] = best[0][0] if best else "A0"
    return {"selected": sel, "selected_repo": {k: (st["candidates"][v]["repo"] if v != "A0" else str(REPO_ROOT))
                                               for k, v in sel.items()},
            "first_safe_index": min([v["index"] for _, v in safe], default=None),
            "n_generated": sum(v["status"] == "generated" for _, v in cands), "n_safe": len(safe)}
