"""ClinEvolve-Lineage: tree search over task-agent versions with a frozen repairer.

Starting from A0, each step expands one parent with ``children_per_step`` children (each child = one repair session on
the parent, with feedback from the parent's own repair-set evaluation). Every child is scored on the validation cohort
against A0 (gain on the target and gates G1-G6); only safe versions (all gates passed) or the root are expandable.

Parent selection policies
  greedy   the safe version with the largest validation gain (ties: earlier birth)
  uniform  uniformly among safe versions (seeded)
  lineage  clade-metaproductivity selection (Huxley-Goedel-Machine style): the safe version v maximizing
               CMP(v) = mean_{u in clade(v)} g(u) + c * sqrt( ln(N + 1) / (|clade(v)| + 1) )
           where clade(v) is v and all its descendants (A0 excluded), g(u) the validation gain of u (0 for failed
           generations), N the number of versions in the tree and c = ``ucb_c_pp``.

The output is the best safe version of the tree by validation gain (ties: earlier birth).
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from clinevolve.config import REPO_ROOT, TARGETS
from clinevolve.gates import evaluate_gates, objective_text
from clinevolve.runtime import Runtime, log_to, save_json, sha256

POLICIES = ("greedy", "uniform", "lineage")


def descendants(st: Dict, cid: str) -> List[Dict]:
    kids = [n for n in st["nodes"].values() if n.get("parent") == cid]
    out = []
    for k in kids:
        out += [k] + descendants(st, k["cid"])
    return out


def pick_parent(st: Dict, step: int, policy: str, ucb_c: float, seed: int) -> Dict:
    nodes = list(st["nodes"].values())
    eligible = [n for n in nodes if n["cid"] == "A0" or n.get("safe")]
    if policy == "greedy":
        return max(eligible, key=lambda n: (n.get("gain") or 0.0, -n["birth"]))
    if policy == "uniform":
        rng = np.random.default_rng(seed + step)
        return eligible[int(rng.integers(len(eligible)))]
    N = max(1, sum(1 for n in nodes if n["cid"] != "A0"))
    best, best_v = None, -1e9
    for n in eligible:
        clade = [n] + descendants(st, n["cid"])
        gains = [c.get("gain") or 0.0 for c in clade if c["cid"] != "A0"]
        if gains:
            cmp_ = float(np.mean(gains))
        else:   # an unexpanded node inherits its parent's gain as prior
            cmp_ = (st["nodes"].get(n.get("parent"), {}).get("gain") or 0.0) if n.get("parent") else 0.0
        v = cmp_ + ucb_c * math.sqrt(math.log(N + 1) / (len(gains) + 1))
        if v > best_v:
            best, best_v = n, v
    return best


def search(rt: Runtime, *, meta_path: Path, out: Path, policy: Optional[str] = None, partition: int = 1,
           cases: Optional[Path] = None) -> Dict[str, Any]:
    cfg, G = rt.cfg["lineage"], rt.cfg["gates"]
    policy = policy or cfg["policy"]
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    target = TARGETS.get(cfg["target"], cfg["target"])
    steps, children, ucb_c, seed = int(cfg["steps"]), int(cfg["children_per_step"]), float(cfg["ucb_c_pp"]), int(cfg["seed"])
    root = Path(out) / f"lineage/p{partition}/{policy}"
    common = Path(out) / f"lineage/p{partition}/common"
    root.mkdir(parents=True, exist_ok=True)
    state_p, log = root / "tree.json", root / "search.log"
    a0_val = rt.evaluate(REPO_ROOT, "A0", "validation", common, partition, cases)
    a0rep_val = rt.evaluate(REPO_ROOT, "A0rep", "validation", common, partition, cases)
    st = json.loads(state_p.read_text()) if state_p.exists() else None
    if st is None:
        a0 = {"cid": "A0", "repo": str(REPO_ROOT), "record": rt.a0_record(partition, target), "parent": None,
              "depth": 0, "birth": 0, "gain": 0.0, "safe": True}
        st = {"partition": partition, "policy": policy, "target": target, "meta_hash": sha256(meta_path),
              "nodes": {"A0": a0}, "steps": []}
        save_json(state_p, st)

    def ancestors(n: Dict) -> List[Dict]:
        chain, pid = [], n.get("parent")
        while pid:
            chain.append(st["nodes"][pid]["record"])
            pid = st["nodes"][pid].get("parent")
        return chain[::-1]

    def feedback_for(n: Dict) -> str:
        if n.get("feedback"):
            return n["feedback"]
        rep = (rt.evaluate(REPO_ROOT, "A0", "repair", common, partition, cases) if n["cid"] == "A0"
               else rt.evaluate(Path(n["repo"]), n["cid"], "repair", root, partition, cases))
        fb = root / f"feedback/{n['cid']}.json"
        if not fb.exists():
            rt.build_feedback(rep, target, fb, n["cid"], partition, cases=cases)
        n["feedback"] = str(fb)
        return n["feedback"]

    while len(st["steps"]) < steps:
        si = len(st["steps"])
        parent = pick_parent(st, si, policy, ucb_c, seed)
        fb = feedback_for(parent)
        save_json(state_p, st)
        log_to(log, f"step {si}: expand {parent['cid']} (gain {parent.get('gain', 0):+.1f})")
        cids = [f"p{partition}_{policy[:2]}_s{si + 1}_c{j + 1}" for j in range(children)]

        def child(cid: str) -> Dict:
            rec = rt.generate(meta_path=meta_path, parent_repo=Path(parent["repo"]), parent_record=parent["record"],
                              ancestors=ancestors(parent) if parent["cid"] != "A0" else [], cid=cid, target=target,
                              objective=objective_text(target), feedback=Path(fb), root=root, partition=partition)
            n = {"cid": cid, "parent": parent["cid"], "depth": parent["depth"] + 1,
                 "birth": 1 + si * children + int(cid.rsplit("_c", 1)[1]) - 1,
                 "status": rec.get("generation_status"), "record": rec, "repo": rec.get("repo_path")}
            if n["status"] == "generated":
                val = rt.evaluate(Path(n["repo"]), cid, "validation", root, partition, cases)
                g = evaluate_gates(G, Path(a0_val), Path(val), Path(a0_val), target, a0_replicate_val=Path(a0rep_val),
                                   reference_is_a0=True)
                n.update({"validation": str(val), "gates": g, "gain": g["target"]["gain_pp"], "safe": bool(g["all_pass"])})
            else:
                n.update({"gain": None, "safe": False})
            return n

        with ThreadPoolExecutor(children) as ex:
            for n in ex.map(child, cids):
                st["nodes"][n["cid"]] = n
        best = max([n for n in st["nodes"].values() if n.get("safe")], key=lambda n: n.get("gain", 0.0))
        st["steps"].append({"step": si, "parent": parent["cid"], "children": cids, "best_safe": best["cid"],
                            "best_safe_gain": best.get("gain", 0.0),
                            "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save_json(state_p, st)
        log_to(log, f"step {si} done: " + str([(c, st['nodes'][c].get('gain'), st['nodes'][c].get('safe')) for c in cids]))
    final = max([n for n in st["nodes"].values() if n.get("safe")], key=lambda n: (n.get("gain", 0.0), -n["birth"]))
    st["final"], st["final_repo"], st["done"] = final["cid"], final["repo"], True
    save_json(state_p, st)
    return st
