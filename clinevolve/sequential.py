"""Sequential repair of several targets in a fixed order (default PD -> PL -> PT) with an archive.

At each stage the repairer generates ``candidates_per_stage`` candidates, split evenly across the archive members (rank
order). Each parent receives feedback from its own repair-set evaluation, extended by the failure rates of the targets
already repaired, which must be preserved. Candidates are scored on the repair set (success = target failure rate strictly
below the parent's); the best ``validation_candidates_per_stage`` successful candidates are validated against all gates
with their parent as reference (retention gate G2 for every earlier repaired target; G6 against A0). Gate-passing
candidates enter the archive (capacity K), ranked by target gain, then smallest worst retention change, then fewer
tokens; rank 1 is the main-path version for the next stage. If no candidate passes, the archive carries forward and the
target is not counted as repaired.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from clinevolve.config import REPO_ROOT, SHORT
from clinevolve.gates import evaluate_gates, load, objective_text, paired_rate
from clinevolve.runtime import Runtime, log_to, save_json


def run_sequence(rt: Runtime, *, meta_path: Path, out: Path, partition: int = 1, tag: str = "seq",
                 cases: Optional[Path] = None) -> Dict[str, Any]:
    cfg = rt.cfg["sequential"]
    G = rt.cfg["gates"]
    K, budget, nval, use_gates = (int(cfg["archive_capacity"]), int(cfg["candidates_per_stage"]),
                                  int(cfg["validation_candidates_per_stage"]), bool(cfg["use_gates"]))
    root = Path(out) / f"{tag}/p{partition}"
    common = Path(out) / f"common/p{partition}"
    root.mkdir(parents=True, exist_ok=True)
    state_p, log = root / "state.json", root / "driver.log"
    st = json.loads(state_p.read_text()) if state_p.exists() else {"partition": partition, "stages": [], "repaired": []}
    a0 = {"cid": "A0", "repo": str(REPO_ROOT), "record": rt.a0_record(partition, cfg["stage_order"][0]),
          "repair": str(rt.evaluate(REPO_ROOT, "A0", "repair", common, partition, cases)),
          "validation": str(rt.evaluate(REPO_ROOT, "A0", "validation", common, partition, cases))}
    a0rep_val = str(rt.evaluate(REPO_ROOT, "A0rep", "validation", common, partition, cases))
    if not st["stages"]:
        st["archive"], st["main"] = [a0], a0
        save_json(state_p, st)
    records: Dict[str, Dict] = {"A0": a0["record"]}
    for S in st["stages"]:
        for cid, c in S["candidates"].items():
            if c.get("record"):
                records[cid] = c["record"]

    def ancestors(rec: Dict) -> List[Dict]:
        chain, pid = [], rec.get("parent_candidate_id")
        while pid:
            chain.append(records[pid])
            pid = records[pid].get("parent_candidate_id")
        return chain[::-1]

    for si, target in enumerate(cfg["stage_order"]):
        if si < len(st["stages"]) and st["stages"][si].get("done"):
            continue
        if si >= len(st["stages"]):
            st["stages"].append({"target": target, "candidates": {}})
        S = st["stages"][si]
        retained = list(st["repaired"])
        parents = st["archive"]
        log_to(log, f"stage {si} {target} parents={[p['cid'] for p in parents]} retained={retained}")
        fbs = {}
        for p in parents:
            rep = load(Path(p["repair"]))
            extra = ([("PREVIOUSLY REPAIRED CAPABILITIES THAT MUST BE PRESERVED (current parent failure rates on the "
                       "repair set): " + "; ".join(f"{t}: {float(rep[t].mean()):.3f}" for t in retained)
                       + ". Do not undo or weaken the earlier repairs while improving the current target.")]
                     if retained else None)
            fbs[p["cid"]] = str(rt.build_feedback(Path(p["repair"]), target, root / f"feedback/s{si}_{p['cid']}.json",
                                                  p["cid"], partition, extra, cases))
        jobs = [(f"p{partition}_{tag}_s{si + 1}_c{i + 1}", parents[i % len(parents)]) for i in range(budget)]
        todo = [(c, p) for c, p in jobs if c not in S["candidates"]]

        def gen(job):
            cid, p = job
            rec = rt.generate(meta_path=meta_path, parent_repo=Path(p["repo"]), parent_record=p["record"],
                              ancestors=ancestors(p["record"]) if p["cid"] != "A0" else [], cid=cid, target=target,
                              objective=objective_text(target, retained), feedback=Path(fbs[p["cid"]]), root=root,
                              partition=partition)
            return cid, p, rec

        with ThreadPoolExecutor(max(1, len(todo))) as ex:
            for cid, p, rec in ex.map(gen, todo):
                S["candidates"][cid] = {"parent": p["cid"], "status": rec.get("generation_status"), "record": rec,
                                        "repo": rec.get("repo_path")}
                records[cid] = rec
                save_json(state_p, st)
        pmap = {p["cid"]: p for p in parents}
        for cid, c in S["candidates"].items():                      # repair-set scoring
            if c["status"] != "generated" or "repair_rate" in c:
                continue
            c["repair"] = str(rt.evaluate(Path(c["repo"]), cid, "repair", root, partition, cases))
            pr, cr, _ = paired_rate(load(Path(pmap[c["parent"]]["repair"])), load(Path(c["repair"])), target)
            c["repair_parent_rate"], c["repair_rate"], c["repair_success"] = pr, cr, int(cr < pr)
            save_json(state_p, st)
        succ = sorted([(c["repair_rate"], k) for k, c in S["candidates"].items() if c.get("repair_success")])
        S["validated"] = [k for _, k in succ[:nval]]
        passers = []
        for cid in S["validated"]:                                   # validation + gates
            c = S["candidates"][cid]
            if "gates" not in c:
                c["validation"] = str(rt.evaluate(Path(c["repo"]), cid, "validation", root, partition, cases))
                c["gates"] = evaluate_gates(G, Path(pmap[c["parent"]]["validation"]), Path(c["validation"]),
                                            Path(a0["validation"]), target, retained, use_gates=use_gates,
                                            a0_replicate_val=Path(a0rep_val), reference_is_a0=(c["parent"] == "A0"))
                save_json(state_p, st)
            if c["gates"]["all_pass"]:
                passers.append(cid)
        rank = sorted(passers, key=lambda k: (-S["candidates"][k]["gates"]["target"]["gain_pp"],
                                              S["candidates"][k]["gates"]["worst_retention_pp"],
                                              S["candidates"][k]["gates"]["mean_tokens"]))
        S["passers_ranked"] = rank
        if rank:
            st["archive"] = [{"cid": k, "repo": S["candidates"][k]["repo"], "record": S["candidates"][k]["record"],
                              "repair": S["candidates"][k]["repair"], "validation": S["candidates"][k]["validation"]}
                             for k in rank[:K]]
            st["main"] = st["archive"][0]
            st["repaired"].append(target)
        S["main_after"], S["archive_after"], S["done"] = st["main"]["cid"], [a["cid"] for a in st["archive"]], True
        save_json(state_p, st)
        log_to(log, f"stage {si} done: passers={rank} main={S['main_after']}")
    st["versions"] = {"A0": "A0", **{f"A{i + 1}": s["main_after"] for i, s in enumerate(st["stages"])}}
    st["final_repo"] = st["main"]["repo"]
    st["done"] = True
    save_json(state_p, st)
    return st
