"""One meta-agent session, run as a subprocess by ``clinevolve.runtime`` (python -m clinevolve.generation_worker SPEC).

mode "task": repairer ``meta_path`` writes one candidate version of the task agent (edits task_agent.py in a fresh copy
             of the parent program); the candidate is recorded with its code difference and scope check.
mode "meta": repairer ``meta_path`` edits its own program (meta_agent.py in a workspace copy) from an evidence report.

Unless the sandbox is disabled, the session re-executes itself inside a bubblewrap sandbox: the code base and Python
environment are read-only, patient data, the case manifest and every evaluation output are hidden, and only this
session's own workspace, patch and log directories are writable.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from clinevolve.config import REPO_ROOT

SANDBOXED = os.environ.get("CLINEVOLVE_SANDBOXED") == "1"
META_FORBIDDEN = ["task_agent.py", "agent", "clinevolve", "configs", "utils", "outputs", "runs", "data"]


def _real(p: str) -> str:
    return str(Path(p).resolve())


def bwrap_args(sandbox: dict, keep: list[tuple[str, bool]]) -> list[str]:
    args = ["bwrap", "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    for ro in sandbox["readonly"]:
        if Path(ro).exists():
            args += ["--ro-bind", _real(ro), _real(ro)]
    for h in dict.fromkeys(_real(x) for x in sandbox["hidden"] if Path(x).exists()):
        args += ["--tmpfs", h] if Path(h).is_dir() else ["--ro-bind", "/dev/null", h]
    for path, ro in keep:
        for p in dict.fromkeys([path, _real(path)]):
            args += ["--ro-bind" if ro else "--bind", p, p]
    return args


def reexec_sandboxed(spec_path: str, sandbox: dict, keep: list[tuple[str, bool]]) -> None:
    env = dict(os.environ, CLINEVOLVE_SANDBOXED="1")
    if not sandbox.get("enabled", True):
        os.environ["CLINEVOLVE_SANDBOXED"] = "1"
        return
    cmd = bwrap_args(sandbox, keep) + ["--chdir", str(REPO_ROOT), "--", sys.executable, "-m",
                                       "clinevolve.generation_worker", spec_path]
    os.execvpe("bwrap", cmd, env)


def count_usage() -> dict:
    """Count meta-model calls and tokens (wraps litellm.completion)."""
    import agent.llm as al
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    orig = al.litellm.completion

    def counted(*a, **k):
        r = orig(*a, **k)
        try:
            u = r.get("usage") if hasattr(r, "get") else getattr(r, "usage", None)
            usage["calls"] += 1
            usage["prompt_tokens"] += int((u.get("prompt_tokens") if hasattr(u, "get") else u.prompt_tokens) or 0)
            usage["completion_tokens"] += int((u.get("completion_tokens") if hasattr(u, "get") else u.completion_tokens) or 0)
        except Exception:
            pass
        return r

    al.litellm.completion = counted
    return usage


def load_meta(meta_path: str, mm: dict, chat_file: Path):
    ms = importlib.util.spec_from_file_location("clinevolve_repairer_program", meta_path)
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    meta = mod.MetaAgent(model=mm["name"], chat_history_file=str(chat_file), temperature=mm["temperature"], max_tokens=4096)
    meta.max_tokens = mm["max_tokens"]
    return meta


# ---------------------------------------------------------------------------------------------- task candidate
def run_task(spec_path: str, spec: dict) -> None:
    root, cid = Path(spec["root"]), spec["cid"]
    gen = root / "gen"
    reg_p = gen / f"{cid}_registry.json"
    ws, patches = root / "workspaces" / cid, root / "patches" / cid
    if not SANDBOXED:
        partition = {"experiment": "clinevolve", "outer_fold": spec["partition"], "repair_target": spec["target"]}
        records = []
        for r in list(spec["ancestors"]) + [spec["parent_record"]]:
            r = {k: v for k, v in dict(r).items() if not k.startswith("_")}
            r.update({"target_id": spec["target"], "partition": partition, "outer_fold": spec["partition"]})
            records.append(r)
        records[-1]["repo_path"] = spec["parent_repo"]
        gen.mkdir(parents=True, exist_ok=True)
        reg_p.write_text(json.dumps({"schema_version": "exp2_candidate_registry_v1",
                                     "updated_at_utc": spec["parent_record"].get("updated_at_utc"),
                                     "n_candidates": len(records), "candidates": records}, indent=1))
        for d in (ws, patches):                     # an interrupted attempt restarts clean
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        keep = [(str(ws), False), (str(patches), False), (str(gen), False),
                (str(Path(spec["feedback"]).parent), True), (str(Path(spec["meta_path"]).parent), True),
                (spec["run_manifest"], True)]
        if _real(spec["parent_repo"]) != _real(str(REPO_ROOT)):
            keep.append((spec["parent_repo"], True))
        reexec_sandboxed(spec_path, spec["sandbox"], keep)
    import clinevolve.generate as gc
    from clinevolve.registry import CandidateRegistry
    usage = count_usage()
    meta = load_meta(spec["meta_path"], spec["meta_model"], gen / f"{cid}_meta_chat.md")
    try:
        gc.generate_candidate(registry=CandidateRegistry(reg_p), run_manifest_path=Path(spec["run_manifest"]),
                              parent_candidate_id=spec["parent_record"]["candidate_id"], candidate_id=cid, node_id=cid,
                              feedback_path=Path(spec["feedback"]), workspace_root=ws, patch_root=patches,
                              meta_agent=meta, objective=spec["objective"])
    except Exception as e:  # noqa: BLE001 - recorded in the registry / log; the caller treats it as a failed generation
        print("GENERATION_EXCEPTION", type(e).__name__, str(e)[:300])
    (gen / f"{cid}_usage.json").write_text(json.dumps(usage))


# ---------------------------------------------------------------------------------------------- repairer self-edit
def run_meta(spec_path: str, spec: dict) -> None:
    log_dir = Path(spec["log_dir"])
    if not SANDBOXED:
        keep = [(spec["workspace"], False), (str(Path(spec["report"]).parent), True),
                (str(Path(spec["meta_path"]).parent), True), (str(log_dir), False)]
        reexec_sandboxed(spec_path, spec["sandbox"], keep)
    usage = count_usage()
    meta = load_meta(spec["meta_path"], spec["meta_model"], log_dir / f"{spec['tag']}_chat.md")
    try:
        meta.forward(repo_path=spec["workspace"], objective=spec["objective"], feedback=Path(spec["report"]).read_text(),
                     allowed_paths=["meta_agent.py"], forbidden_paths=META_FORBIDDEN,
                     extra_context={"procedure": "repairer_self_improvement", "update": spec["tag"]})
    except Exception as e:  # noqa: BLE001
        print("META_UPDATE_EXCEPTION", type(e).__name__, str(e)[:300])
    (log_dir / f"{spec['tag']}_usage.json").write_text(json.dumps(usage))


def main() -> None:
    spec_path = sys.argv[1]
    spec = json.loads(Path(spec_path).read_text())
    {"task": run_task, "meta": run_meta}[spec["mode"]](spec_path, spec)


if __name__ == "__main__":
    main()
