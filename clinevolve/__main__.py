"""ClinEvolve command line.

    python -m clinevolve evaluate     --program DIR --role validation --id NAME
    python -m clinevolve repair       --target PT [--repairer meta_agent.py]
    python -m clinevolve sequential   [--repairer meta_agent.py]
    python -m clinevolve self-improve [--incumbent-competes] [--round1-score best_safe_gain]
    python -m clinevolve lineage      [--policy lineage|greedy|uniform] [--repairer meta_agent.py]
    python -m clinevolve check-repairer PATH

Every command accepts --config (default configs/clinevolve.yaml) and --partition (default 1). Results are written under
paths.work_dir; every command is resumable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinevolve.config import REPO_ROOT, TARGETS, load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="clinevolve", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/clinevolve.yaml"))
    ap.add_argument("--partition", type=int, default=1, help="outer_fold value of the case manifest to use")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="run one task-agent program on one cohort")
    e.add_argument("--program", default=str(REPO_ROOT))
    e.add_argument("--role", required=True, choices=["repair", "validation", "locked_test"])
    e.add_argument("--id", required=True)
    r = sub.add_parser("repair", help="single-target repair of A0 (k candidates, gated selection)")
    r.add_argument("--target", required=True, choices=sorted(TARGETS))
    r.add_argument("--repairer", default=str(REPO_ROOT / "meta_agent.py"))
    r.add_argument("--tag", default="M")
    s = sub.add_parser("sequential", help="sequential repair PD -> PL -> PT with an archive")
    s.add_argument("--repairer", default=str(REPO_ROOT / "meta_agent.py"))
    s.add_argument("--tag", default="seq")
    si = sub.add_parser("self-improve", help="two-round self-improvement tournament of the repairer")
    si.add_argument("--repairer", default=str(REPO_ROOT / "meta_agent.py"))
    si.add_argument("--incumbent-competes", action="store_true")
    si.add_argument("--round1-score", default="esg", choices=["esg", "best_safe_gain"])
    si.add_argument("--evidence-report", default=None, help="optional text file used as round-1 evidence")
    li = sub.add_parser("lineage", help="ClinEvolve-Lineage tree search with a frozen repairer")
    li.add_argument("--repairer", default=str(REPO_ROOT / "meta_agent.py"))
    li.add_argument("--policy", default=None, choices=["lineage", "greedy", "uniform"])
    c = sub.add_parser("check-repairer", help="static/stubbed validity check of a repairer program")
    c.add_argument("path")
    a = ap.parse_args()

    if a.cmd == "check-repairer":
        from clinevolve.meta_check import check
        print(json.dumps(check(Path(a.path))))
        return
    cfg = load_config(a.config if Path(a.config).exists() else None)
    from clinevolve.runtime import Runtime
    rt = Runtime(cfg)
    out = Path(cfg["paths"]["work_dir"])
    if a.cmd == "evaluate":
        p = rt.evaluate(Path(a.program), a.id, a.role, out / "evaluations", a.partition)
        print(p)
    elif a.cmd == "repair":
        from clinevolve.repair import repair
        st = repair(rt, meta_path=Path(a.repairer), target=TARGETS[a.target], out=out / "repair", partition=a.partition,
                    tag=a.tag)
        print(json.dumps(st["selection"], indent=1))
    elif a.cmd == "sequential":
        from clinevolve.sequential import run_sequence
        st = run_sequence(rt, meta_path=Path(a.repairer), out=out / "sequential", partition=a.partition, tag=a.tag)
        print(json.dumps({"versions": st["versions"], "final_repo": st["final_repo"]}, indent=1))
    elif a.cmd == "self-improve":
        from clinevolve.self_improve import self_improve
        info = self_improve(rt, base_meta=Path(a.repairer), out=out, partition=a.partition,
                            incumbent_competes=a.incumbent_competes, round1_score=a.round1_score,
                            evidence_report=Path(a.evidence_report) if a.evidence_report else None)
        print(json.dumps(info, indent=1))
    elif a.cmd == "lineage":
        from clinevolve.lineage import search
        st = search(rt, meta_path=Path(a.repairer), out=out, policy=a.policy, partition=a.partition)
        print(json.dumps({"final": st["final"], "final_repo": st["final_repo"]}, indent=1))


if __name__ == "__main__":
    main()
