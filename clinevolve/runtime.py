"""Execution runtime shared by every ClinEvolve procedure.

- program evaluation: a task-agent program is run on the cases of one partition/role, sharded over concurrent episode
  workers against a pool of vLLM task servers; per-case clinical endpoints are merged into one parquet file
- repair feedback packages built from a parent's repair-set evaluation
- candidate generation: one meta-agent session per candidate, in a subprocess and (by default) a bubblewrap sandbox
  (see ``clinevolve.generation_worker``)
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from clinevolve.config import PL, REPO_ROOT

PY = sys.executable
EVAL_LABEL = "treatment_guideline_failure"   # directory label used by clinevolve.evaluate; all endpoints are computed
_SAVE_LOCK = threading.Lock()


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_json(p: Path, obj: Any) -> None:
    """Atomic JSON save, safe across threads."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _SAVE_LOCK:
        tmp = Path(f"{p}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        tmp.replace(p)


def log_to(p: Path, msg: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")


def served_name(model: str) -> str:
    return model.split("/", 1)[-1]


class ServerPool:
    """Bounded concurrency per model server. ``acquire()`` yields the URL of a server with a free slot."""

    def __init__(self, servers: List[str], per_server: int):
        if not servers:
            raise ValueError("at least one server URL is required")
        self.servers = list(servers)
        self.sem = {s: threading.BoundedSemaphore(per_server) for s in self.servers}
        self._rr = 0
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self) -> Iterator[str]:
        while True:
            with self._lock:
                order = self.servers[self._rr:] + self.servers[:self._rr]
                self._rr = (self._rr + 1) % len(self.servers)
            for s in order:
                if self.sem[s].acquire(blocking=False):
                    try:
                        yield s
                    finally:
                        self.sem[s].release()
                    return
            time.sleep(1.0)


class Runtime:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.repo = REPO_ROOT
        self.cases = Path(cfg["paths"]["cases"])
        self.work = Path(cfg["paths"]["work_dir"])
        self.work.mkdir(parents=True, exist_ok=True)
        ev = cfg["evaluation"]
        self.shards = int(ev["shards_per_evaluation"])
        self.task_pool = ServerPool(cfg["models"]["task"]["servers"], int(ev["evaluations_per_server"]))
        self.meta_pool = ServerPool(cfg["models"]["meta"]["servers"], int(ev["generations_per_server"]))
        self.run_manifest = self.work / "_run_config.json"
        self._write_run_manifest()

    # ------------------------------------------------------------------ configuration handed to the generator
    def _write_run_manifest(self) -> None:
        m = self.cfg["models"]
        save_json(self.run_manifest, {
            "run_manifest_version": "clinevolve_run_config_v1",
            "selection_status": "frozen",
            "a0": {"model": m["task"]["name"], "temperature": 0.0, "max_tokens": 4096},
            "search": {"candidate_parent_rule": "tree_search"},
            "candidate_modification_scope": {"allowed_paths": self.cfg["scope"]["allowed_paths"],
                                             "forbidden_paths": self.cfg["scope"]["forbidden_paths"]},
        })

    # ------------------------------------------------------------------ cases
    def case_table(self) -> pd.DataFrame:
        return pd.read_csv(self.cases, dtype={"case_id": str, "patient_id": str, "subject_id": str})

    # ------------------------------------------------------------------ evaluation
    @staticmethod
    def eval_path(root: Path, role: str, cid: str) -> Path:
        return Path(root) / f"evals/{role}/{cid}.parquet"

    def evaluate(self, repo: Path, cid: str, role: str, root: Path, partition: int = 1,
                 cases: Optional[Path] = None) -> Path:
        """Run program ``repo`` on every case of (partition, role); returns the merged per-case endpoint table.

        Results are cached by (root, role, cid); a per-program file lock prevents duplicate concurrent evaluation."""
        out = self.eval_path(root, role, cid)
        if out.exists():
            return out
        lock = Path(root) / "_locks" / f"{role}__{cid}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            if out.exists():
                return out
            with self.task_pool.acquire() as server:
                return self._evaluate(Path(repo), cid, role, Path(root), partition, server, Path(cases or self.cases))

    def _evaluate(self, repo: Path, cid: str, role: str, root: Path, partition: int, server: str, cases: Path) -> Path:
        out = self.eval_path(root, role, cid)
        m = pd.read_csv(cases, dtype={"case_id": str, "patient_id": str, "subject_id": str})
        m = m[(m.outer_fold == partition) & (m.role == role)]
        if m.empty:
            raise ValueError(f"no cases for partition {partition}, role {role} in {cases}")
        shard_dir = root / f"_shards/{role}/{cid}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        parts = [p for p in np.array_split(m, min(self.shards, len(m))) if len(p)]
        env = dict(os.environ, VLLM_API_BASE=server, OPENAI_API_BASE=server,
                   LOCAL_MODEL_ENDPOINTS=json.dumps({served_name(self.cfg["models"]["task"]["name"]): server}))
        hpi = Path(self.cfg["paths"]["mimic_cdm_dir"]) / "history_of_present_illness.csv"

        def run(i: int) -> int:
            sm = shard_dir / f"manifest_{i}.csv"
            parts[i].to_csv(sm, index=False)
            cmd = [PY, "-m", "clinevolve.evaluate", "--candidate-repo", str(repo), "--candidate-id", cid,
                   "--model", self.cfg["models"]["task"]["name"], "--target-id", EVAL_LABEL,
                   "--outer-fold", str(partition), "--role", role, "--fold-manifest", str(sm), "--hpi-path", str(hpi),
                   "--output-root", str(shard_dir / f"s{i}")]
            if role == "locked_test":
                cmd.append("--allow-locked-test")
            with open(shard_dir / f"s{i}.log", "a") as log:
                return subprocess.run(cmd, cwd=self.repo, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

        with ThreadPoolExecutor(len(parts)) as ex:
            rcs = list(ex.map(run, range(len(parts))))
        if any(rcs):
            raise RuntimeError(f"evaluation shards failed for {cid} {role} partition {partition}: {rcs}")
        frames = [pd.read_parquet(p) for p in
                  shard_dir.glob(f"s*/fold{partition}/{EVAL_LABEL}/{role}/{cid}/evaluations.parquet")]
        df = pd.concat(frames, ignore_index=True)
        df["case_id"] = df.case_id.astype(str)
        # rows of errored episodes come back without split metadata; the split is known from the shard
        df["role"] = df["role"].fillna(role) if "role" in df else role
        df["outer_fold"] = df["outer_fold"].fillna(partition).astype(int) if "outer_fold" in df else partition
        if "agent_id" in df:
            df["agent_id"] = df["agent_id"].fillna(cid)
        if df.case_id.nunique() != len(m):
            raise RuntimeError(f"merged {df.case_id.nunique()} of {len(m)} cases for {cid} {role}")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        return out

    # ------------------------------------------------------------------ feedback
    def build_feedback(self, evaluation: Path, target: str, out: Path, agent_id: str, partition: int = 1,
                       extra_text: Optional[List[str]] = None, cases: Optional[Path] = None) -> Path:
        """Repair feedback package from a parent's repair-set evaluation: aggregate endpoint failure rates, failing case
        identifiers and up to ``feedback_max_examples`` failure examples (trajectory + failed criterion)."""
        import clinevolve.feedback as fb
        fb.VALID_REPAIR_TARGETS = set(fb.VALID_REPAIR_TARGETS) | {PL}
        out = Path(out)
        m = pd.read_csv(cases or self.cases, dtype={"case_id": str, "patient_id": str, "subject_id": str})
        fm = out.parent / f"{out.stem}_manifest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        m[m.outer_fold == partition].to_csv(fm, index=False)
        fb.build_from_files(fold_manifest_path=fm, evaluation_path=Path(evaluation), output_path=out,
                            outer_fold=partition, target_id=target, agent_id=agent_id,
                            max_examples=int(self.cfg["evaluation"]["feedback_max_examples"]))
        if extra_text:
            d = json.loads(out.read_text())
            d["text_feedback"] += "\n\n" + "\n\n".join(extra_text)
            out.write_text(json.dumps(d))
        return out

    # ------------------------------------------------------------------ candidate records
    def a0_record(self, partition: int, target: str) -> Dict[str, Any]:
        from clinevolve.registry import make_a0_record
        rec = make_a0_record(outer_fold=partition, target_id=target, node_id=f"p{partition}_A0", repo_path=self.repo,
                             base_commit=None, agent_model=self.cfg["models"]["task"]["name"],
                             metadata={"role": "root"})
        return rec.to_dict()

    # ------------------------------------------------------------------ generation
    def sandbox_spec(self) -> Dict[str, Any]:
        sb = self.cfg["sandbox"]
        # directories are replaced by an empty tmpfs, files by /dev/null
        hidden = [self.cfg["paths"]["mimic_cdm_dir"], str(self.work), self.cfg["paths"]["cases"],
                  str(self.repo / "data")] + list(sb.get("extra_hidden_paths", []))
        return {"enabled": bool(sb["enabled"]), "hidden": hidden,
                "readonly": [str(self.repo), sys.prefix] + list(sb.get("extra_readonly_paths", []))}

    def _meta_env(self, server: str) -> Dict[str, str]:
        return dict(os.environ, LOCAL_MODEL_ENDPOINTS=json.dumps({served_name(self.cfg["models"]["meta"]["name"]): server}))

    def generate(self, *, meta_path: Path, parent_repo: Path, parent_record: Dict[str, Any], ancestors: List[Dict[str, Any]],
                 cid: str, target: str, objective: str, feedback: Path, root: Path, partition: int = 1) -> Dict[str, Any]:
        """One task-agent candidate from repairer program ``meta_path`` (one meta-agent session). Returns the candidate
        registry record (``generation_status`` == 'generated' on success) plus ``_usage`` and ``_wall_seconds``."""
        root = Path(root)
        spec = {"mode": "task", "meta_path": str(meta_path), "parent_repo": str(parent_repo),
                "parent_record": parent_record, "ancestors": ancestors, "cid": cid, "target": target,
                "partition": partition, "objective": objective, "feedback": str(feedback), "root": str(root),
                "run_manifest": str(self.run_manifest), "meta_model": self.cfg["models"]["meta"],
                "sandbox": self.sandbox_spec()}
        return self._run_worker(spec, root / f"gen/{cid}_spec.json", root / f"gen/{cid}.log",
                                root / f"gen/{cid}_registry.json", cid)

    def meta_edit(self, *, meta_path: Path, workspace: Path, report: Path, objective: str, tag: str,
                  log_dir: Path) -> Dict[str, Any]:
        """One self-edit session: repairer ``meta_path`` edits ``workspace/meta_agent.py`` from ``report``."""
        spec = {"mode": "meta", "meta_path": str(meta_path), "workspace": str(workspace), "report": str(report),
                "objective": objective, "tag": tag, "log_dir": str(log_dir), "meta_model": self.cfg["models"]["meta"],
                "sandbox": self.sandbox_spec()}
        log_dir.mkdir(parents=True, exist_ok=True)
        sp = log_dir / f"{tag}_spec.json"
        sp.write_text(json.dumps(spec))
        with self.meta_pool.acquire() as server, open(log_dir / f"{tag}.log", "a") as lf:
            subprocess.run([PY, "-m", "clinevolve.generation_worker", str(sp)], cwd=self.repo, stdout=lf,
                           stderr=subprocess.STDOUT, env=self._meta_env(server))
        u = log_dir / f"{tag}_usage.json"
        return json.loads(u.read_text()) if u.exists() else {}

    def _run_worker(self, spec: Dict[str, Any], spec_p: Path, log: Path, rec_p: Path, cid: str) -> Dict[str, Any]:
        spec_p.parent.mkdir(parents=True, exist_ok=True)
        spec_p.write_text(json.dumps(spec))
        with self.meta_pool.acquire() as server:
            t0 = time.time()
            with open(log, "a") as lf:
                subprocess.run([PY, "-m", "clinevolve.generation_worker", str(spec_p)], cwd=self.repo, stdout=lf,
                               stderr=subprocess.STDOUT, env=self._meta_env(server))
            wall = time.time() - t0
        recs = json.loads(rec_p.read_text())["candidates"] if rec_p.exists() else []
        rec = next((c for c in recs if c["candidate_id"] == cid),
                   {"candidate_id": cid, "generation_status": "generation_failed"})
        use_p = spec_p.parent / f"{cid}_usage.json"
        rec["_usage"] = json.loads(use_p.read_text()) if use_p.exists() else {}
        rec["_wall_seconds"] = wall
        return rec
