"""Self-improvement of the repairer (meta_agent.py) by a two-round tournament.

Round 0  the base repairer M0 generates S candidates per target (PD, PT) from A0 with the common feedback; each is gated
         on the validation cohort against A0. Their gate diagnostics are the evidence for round 1.
Round 1  M0 edits its own program V times (sandboxed self-edit sessions, ``clinevolve.generation_worker`` mode "meta")
         from the evidence report; variants that fail ``clinevolve.meta_check`` are discarded. Each valid variant generates
         S candidates per target from A0, gated as above.
Round 2  the round-1 champion edits itself V times from the gate diagnostics of its own candidates, with an objective
         that additionally asks for endpoint protection; variants are scored the same way.

Variants are scored by the expected safe gain
    ESG(M) = 1/|T| sum_t 1/S sum_s gain_t(c_s) * 1{c_s passes G1-G6}
(failed generations count as 0). Two selection rules are provided:

* ``incumbent_competes = False`` (default; target-side self-improvement): the champion of each round is the best
  self-edited program; round 2 compares the round-1 champion with its own variants (ties favour the round-1 champion).
* ``incumbent_competes = True``: M0 (and in round 2 the round-1 champion) compete with the variants and win ties
  (M0 first, then the parent); ``round1_score = "best_safe_gain"`` scores round 1 by the mean over targets of the best
  gate-passing validation gain (ties: more gate-passing candidates), as an alternative to ESG.

All scoring uses repair-feedback and validation patients only.
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from clinevolve.config import FD, PD, PL, PT, REPO_ROOT, SHORT
from clinevolve.gates import evaluate_gates, objective_text
from clinevolve.meta_check import check as meta_check
from clinevolve.repair import common_inputs
from clinevolve.runtime import Runtime, log_to, save_json

OBJ_R1 = (
    "You are improving the META-LEVEL REPAIRER (meta_agent.py), not the task agent. meta_agent.py defines how you are "
    "instructed when you repair the clinical task agent. The feedback reports repair attempts this meta_agent.py produced on "
    "source data (patches, repair-set effects, safety-gate outcomes). Modify meta_agent.py so that future repairs of a FRESH "
    "task agent achieve the LARGEST real reduction of the target failure (diagnostic process, laboratory interpretation, or "
    "treatment planning) while keeping the other workflow endpoints, final diagnosis, tool-call validity and resource use "
    "from getting worse. Task-agent patches MAY and often SHOULD contain concrete, disease-specific clinical guidance (for "
    "example guideline-concordant management for the suspected condition, which examinations, laboratory tests and imaging "
    "to obtain for which presentation, how to interpret specific laboratory values); do not discourage this and do not "
    "impose artificial limits on patch size or expected gain. Do not copy patient cases into meta_agent.py. Keep the "
    "MetaAgent class name, its constructor, the forward() signature and the call to chat_with_agent with the bash and edit "
    "tools, and keep passing the objective, feedback, allowed/forbidden paths and extra context into the instruction. "
    "Modify only meta_agent.py.")
OBJ_R2 = OBJ_R1 + (
    " Pay particular attention to the evidence below on WHY strong patches were rejected: a patch must deliver its large "
    "target gain WITHOUT increasing the other endpoints. For example, if a patch makes the agent request more "
    "examinations or laboratory tests, it should also tell the agent how to interpret those results correctly (reference "
    "ranges, direction of abnormality) so lab-interpretation errors do not rise; changes to treatment reasoning must not "
    "alter the diagnosis or the work-up.")
NAMES = {"PD": "diagnostic process", "PL": "lab interpretation", "PT": "treatment", "FD": "final diagnosis"}
EPS = {PD: "PD", PL: "PL", PT: "PT"}
GATE_KEYS = ("G1_target", "G3_nontarget_dx", "G4_tools", "G5_resources", "G6_new_regression")


def self_improve(rt: Runtime, *, base_meta: Path, out: Path, partition: int = 1, cases: Optional[Path] = None,
                 incumbent_competes: bool = False, round1_score: str = "esg",
                 evidence_report: Optional[Path] = None) -> Dict[str, Any]:
    cfg, G = rt.cfg["self_improvement"], rt.cfg["gates"]
    V, S = int(cfg["variants_per_round"]), int(cfg["candidates_per_target"])
    targets = {SHORT[t]: t for t in cfg["targets"]}
    root = Path(out) / f"self_improve/p{partition}"
    root.mkdir(parents=True, exist_ok=True)
    log = root / "tournament.log"
    ci = {ts: common_inputs(rt, t, root, partition, cases) for ts, t in targets.items()}
    a0rec = {ts: rt.a0_record(partition, t) for ts, t in targets.items()}

    def gate(val: str, ts: str) -> Dict:
        return evaluate_gates(G, Path(ci[ts]["a0_val"]), Path(val), Path(ci[ts]["a0_val"]), targets[ts],
                              a0_replicate_val=Path(ci[ts]["a0rep_val"]), reference_is_a0=True)

    def score_one(job):
        rnd, name, mpath, ts, j = job
        sroot = root / f"{rnd}/score/{name}_{ts}"
        cid = f"p{partition}_{rnd}_{name}_{ts}_c{j}"
        rec = rt.generate(meta_path=Path(mpath), parent_repo=REPO_ROOT, parent_record=a0rec[ts], ancestors=[], cid=cid,
                          target=targets[ts], objective=objective_text(targets[ts]), feedback=Path(ci[ts]["feedback"]),
                          root=sroot, partition=partition)
        if rec.get("generation_status") != "generated":
            return name, ts, {"cid": cid, "gen": False, "why": str(rec.get("generation_error"))[:200]}
        val = str(rt.evaluate(Path(rec["repo_path"]), cid, "validation", sroot, partition, cases))
        g = gate(val, ts)
        return name, ts, {"cid": cid, "gen": True, "gain": g["target"]["gain_pp"], "pass": bool(g["all_pass"]),
                          "repo": rec["repo_path"], "val": val}

    def score_round(rnd: str, metas: Dict[str, str]) -> Dict:
        tp = root / f"{rnd}/table.json"
        if tp.exists():
            return json.loads(tp.read_text())
        jobs = [(rnd, n, p, ts, j) for n, p in metas.items() for ts in targets for j in range(1, S + 1)]
        table: Dict = {}
        with ThreadPoolExecutor(max(1, len(jobs))) as ex:
            for n, ts, r in ex.map(score_one, jobs):
                table.setdefault(n, {}).setdefault(ts, []).append(r)
        save_json(tp, table)
        return table

    def esg(table: Dict) -> Dict[str, float]:
        return {n: sum(sum(r["gain"] for r in d.get(ts, []) if r.get("pass")) / S for ts in targets) / len(targets)
                for n, d in table.items()}

    def best_safe_gain(table: Dict) -> Dict[str, tuple]:
        out_ = {}
        for n, d in table.items():
            best = [max([r["gain"] for r in d.get(ts, []) if r.get("pass")], default=0.0) for ts in targets]
            out_[n] = (sum(best) / len(best), sum(1 for ts in targets for r in d.get(ts, []) if r.get("pass")))
        return out_

    def report(table: Dict, meta: str) -> str:
        lines = []
        for ts in targets:
            for r in table.get(meta, {}).get(ts, []):
                if not r["gen"]:
                    lines.append(f"- {ts} candidate {r['cid']}: NO PATCH PRODUCED ({r['why']}) -> failed repair")
                    continue
                g = gate(r["val"], ts)
                nt = {EPS[k.replace("nontarget_", "")]: v for k, v in g.items() if k.startswith("nontarget_")}
                failed = [k for k in GATE_KEYS if not g[k]]
                lines.append(
                    f"- {ts} candidate {r['cid']}: target failure reduction {g['target']['gain_pp']:+.1f} pp; other "
                    "endpoints " + ", ".join(f"{NAMES[k]} {v:+.1f} pp" for k, v in nt.items())
                    + f", final diagnosis {g['final_dx_pp']:+.1f} pp; previously-correct patients newly wrong: excess "
                    f"over noise floor {(g['new_regression_cand'] - g['new_regression_reference']) * 100:+.1f} pp; "
                    + ("PASSED all gates" if g["all_pass"] else "REJECTED by " + ", ".join(failed)))
        return ("EVIDENCE (validation patients). These are repair patches that your current program (meta_agent.py) "
                "produced, each from an unmodified task agent and the failure feedback of these patients, and how the "
                "pre-registered safety gates judged them. Gates: target gain >= 1 pp; each other workflow endpoint <= "
                "+3 pp; final diagnosis <= +1 pp; tool validity; resources; newly wrong patients <= 5 pp above the noise "
                "floor.\n" + "\n".join(lines))

    def variants(rnd: str, parent_path: Path, rep_text: str, objective: str) -> Dict[str, str]:
        d = root / rnd
        (d / "evidence").mkdir(parents=True, exist_ok=True)
        (d / "evidence/report.txt").write_text(rep_text)
        pp = d / "parent/meta_agent.py"
        pp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(parent_path, pp)
        import clinevolve.generate as gc

        def one(v: int):
            outp = d / f"metas/v{v}/meta_agent.py"
            if outp.exists():
                return f"{rnd}v{v}", str(outp)
            ws = d / f"variants/v{v}/repo"
            if ws.parent.exists():
                shutil.rmtree(ws.parent)
            ws.parent.mkdir(parents=True)
            gc.copy_workspace(REPO_ROOT, ws)
            shutil.copy(pp, ws / "meta_agent.py")
            rt.meta_edit(meta_path=pp, workspace=ws, report=d / "evidence/report.txt", objective=objective,
                         tag=f"p{partition}_{rnd}_v{v}", log_dir=d / "meta_logs")
            new = (ws / "meta_agent.py").read_text()
            chk = meta_check(ws / "meta_agent.py")
            log_to(log, f"{rnd} variant v{v}: changed={new != pp.read_text()} check={chk}")
            if new == pp.read_text() or not chk.get("ok"):
                return None
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(new)
            return f"{rnd}v{v}", str(outp)

        with ThreadPoolExecutor(V) as ex:
            return dict(x for x in ex.map(one, range(1, V + 1)) if x)

    m0 = Path(base_meta)
    t0 = score_round("r0", {"M0": str(m0)})
    log_to(log, f"R0 esg={esg(t0)}")
    rep1 = Path(evidence_report).read_text() if evidence_report else report(t0, "M0")
    v1 = variants("r1", m0, rep1, OBJ_R1)
    t1 = score_round("r1", v1)
    if incumbent_competes:
        t1_all = {**t0, **t1}
        if round1_score == "best_safe_gain":
            s1 = best_safe_gain(t1_all)
            c1 = sorted(s1, key=lambda n: (-s1[n][0], -s1[n][1], n != "M0", n))[0]
        else:
            s1 = esg(t1_all)
            c1 = sorted(s1, key=lambda n: (-round(s1[n], 6), n != "M0", n))[0]
    else:
        s1 = esg(t1)
        c1 = sorted(s1, key=lambda n: (-round(s1[n], 6), n))[0] if s1 else "M0"
    p1 = str(m0) if c1 == "M0" else v1[c1]
    log_to(log, f"R1 scores={s1} champion {c1}")
    v2 = variants("r2", Path(p1), report({**t0, **t1}, c1), OBJ_R2)
    t2 = score_round("r2", v2)
    paths = {**v2}
    if c1 != "M0":
        t2["parent"] = t1[c1]
        paths["parent"] = p1
    if incumbent_competes:
        t2["M0"] = t0["M0"]
        paths["M0"] = str(m0)
    s2 = esg(t2)
    rank = {"M0": 0, "parent": 1} if incumbent_competes else {"parent": 0}
    c2 = sorted(s2, key=lambda n: (-round(s2[n], 6), rank.get(n, 2), n))[0] if s2 else c1
    champion = paths.get(c2, p1) if s2 else p1
    ck = root / "champion/meta_agent.py"
    ck.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(champion, ck)
    info = {"partition": partition, "r1_champion": c1, "r2_champion": c2, "scores_r0": esg(t0), "scores_r1": s1,
            "scores_r2": s2, "incumbent_competes": incumbent_competes, "round1_score": round1_score,
            "champion_path": str(ck)}
    save_json(root / "selection.json", info)
    log_to(log, f"champion = {c2} (round 1: {c1}); round-2 scores {s2}")
    return info
