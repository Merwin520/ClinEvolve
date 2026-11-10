"""Unit tests for the ClinEvolve method components (no LLM, no patient data). Run: python -m unittest discover tests"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinevolve.config import DEFAULTS, FD, IT, PD, PL, PT  # noqa: E402
from clinevolve.gates import evaluate_gates, new_regression  # noqa: E402
from clinevolve.lineage import pick_parent  # noqa: E402
from clinevolve.repair import select  # noqa: E402

G = DEFAULTS["gates"]
N = 200


def table(tmp: Path, name: str, rates: dict, seed: int, tool_calls=10.0, tokens=1000, flip_from=None, flip=0):
    """Synthetic per-case endpoint table. ``flip_from``: copy that table and turn ``flip`` correct PD cases into failures."""
    rng = np.random.default_rng(seed)
    if flip_from is not None:
        d = pd.read_parquet(flip_from).copy()
        ok = d.index[d[PD] == 0][:flip]
        d.loc[ok, PD] = 1
    else:
        d = pd.DataFrame({"case_id": [str(i) for i in range(N)]})
        for e in (PD, PL, PT, FD, IT):
            d[e] = (rng.random(N) < rates.get(e, 0.1)).astype(float)
        d["trajectory_status"] = "success"
        d["error"] = ""
        d["n_invalid_calls"] = 0
        d["n_tool_execution_errors"] = 0
    d["n_tool_calls"] = tool_calls
    traj = tmp / f"{name}_traj.json"
    traj.write_text(json.dumps({"usage": {"prompt_tokens": tokens, "completion_tokens": 0}}))
    d["trajectory_path"] = str(traj)
    p = tmp / f"{name}.parquet"
    d.to_parquet(p)
    return p


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.a0 = table(self.tmp, "a0", {PT: 0.4, PD: 0.3, PL: 0.2, FD: 0.1, IT: 0.0}, 0)
        self.rep = table(self.tmp, "a0rep", {}, 0, flip_from=self.a0, flip=0)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def improved(self, name, **kw):
        d = pd.read_parquet(self.a0)
        fail = d.index[d[PT] == 1][:20]        # 20 / 200 = 10 pp fewer PT failures
        d.loc[fail, PT] = 0
        p = self.tmp / f"{name}.parquet"
        d.to_parquet(p)
        return table(self.tmp, name, {}, 0, flip_from=p, **kw)

    def test_pass(self):
        c = self.improved("c_ok")
        g = evaluate_gates(G, self.a0, c, self.a0, PT, a0_replicate_val=self.rep)
        self.assertAlmostEqual(g["target"]["gain_pp"], 10.0)
        self.assertTrue(g["all_pass"])

    def test_nontarget_and_new_regression_fail(self):
        c = self.improved("c_bad", flip=16)    # +8 pp PD failures among A0-correct cases
        g = evaluate_gates(G, self.a0, c, self.a0, PT, a0_replicate_val=self.rep)
        self.assertFalse(g["G3_nontarget_dx"])
        self.assertFalse(g["G6_new_regression"])
        self.assertFalse(g["all_pass"])

    def test_retention_gate(self):
        c = self.improved("c_ret", flip=6)     # +3 pp on PD, a previously repaired target (margin 2 pp)
        g = evaluate_gates(G, self.a0, c, self.a0, PT, retained=[PD], a0_replicate_val=self.rep)
        self.assertFalse(g["G2_retention"])
        g2 = evaluate_gates(G, self.a0, c, self.a0, PT, retained=[], a0_replicate_val=self.rep)
        self.assertTrue(g2["G2_retention"])

    def test_resources(self):
        c = self.improved("c_res", tool_calls=13.0)   # 1.3 x tool calls > 1.25
        g = evaluate_gates(G, self.a0, c, self.a0, PT, a0_replicate_val=self.rep)
        self.assertFalse(g["G5_resources"])

    def test_gate_ablation(self):
        c = self.improved("c_abl", flip=16)
        g = evaluate_gates(G, self.a0, c, self.a0, PT, use_gates=False, a0_replicate_val=self.rep)
        self.assertTrue(g["all_pass"])

    def test_new_regression_rate(self):
        a = pd.DataFrame({PD: [0, 0, 1, 0], PL: [0, 0, 0, 0], PT: [0, 1, 0, 0], FD: [0, 0, 0, 0]})
        v = pd.DataFrame({PD: [1, 0, 1, 0], PL: [0, 0, 0, 0], PT: [0, 1, 0, 0], FD: [0, 0, 0, 0]})
        self.assertAlmostEqual(new_regression(a, v), 0.25)


class SelectionTests(unittest.TestCase):
    def test_ci_at_k(self):
        def c(i, gain, ok, tok=100.0):
            return {"index": i, "status": "generated", "repo": f"/r{i}",
                    "gates": {"all_pass": ok, "target": {"gain_pp": gain}, "mean_tokens": tok}}
        st = {"candidates": {"c1": c(1, 3.0, False), "c2": c(2, 2.0, True), "c3": c(3, 6.0, True, 90),
                             "c4": c(4, 6.0, True, 80), "c6": c(6, 9.0, True)}}
        s = select(st, [1, 3, 5, 10])
        self.assertEqual(s["selected"], {"1": "A0", "3": "c3", "5": "c4", "10": "c6"})
        self.assertEqual(s["first_safe_index"], 2)


class LineageTests(unittest.TestCase):
    def tree(self):
        n = lambda cid, parent, gain, safe, birth: {"cid": cid, "parent": parent, "gain": gain, "safe": safe, "birth": birth}
        return {"nodes": {"A0": n("A0", None, 0.0, True, 0), "a": n("a", "A0", 5.0, True, 1),
                          "b": n("b", "A0", 4.0, True, 2), "a1": n("a1", "a", -2.0, True, 3),
                          "a2": n("a2", "a", -1.0, False, 4), "b1": n("b1", "b", 6.0, True, 5)}}

    def test_greedy(self):
        self.assertEqual(pick_parent(self.tree(), 0, "greedy", 2.0, 1)["cid"], "b1")

    def test_lineage_prefers_productive_clade(self):
        st = self.tree()
        # clade(a) mean = (5 - 2 - 1)/3 = 0.67; clade(b) mean = (4 + 6)/2 = 5; clade(b1) = 6 with larger bonus
        best = pick_parent(st, 0, "lineage", 2.0, 1)
        self.assertIn(best["cid"], {"b", "b1"})
        self.assertNotEqual(pick_parent(st, 0, "lineage", 0.0, 1)["cid"], "a")

    def test_uniform_seeded(self):
        st = self.tree()
        self.assertEqual(pick_parent(st, 3, "uniform", 2.0, 7)["cid"], pick_parent(st, 3, "uniform", 2.0, 7)["cid"])


class RepairerCheckTests(unittest.TestCase):
    def test_base_repairer_valid(self):
        from clinevolve.meta_check import check
        self.assertTrue(check(ROOT / "meta_agent.py")["ok"])

    def test_broken_repairer_rejected(self):
        from clinevolve.meta_check import check
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "meta_agent.py"
            p.write_text((ROOT / "meta_agent.py").read_text().replace("feedback", "fb_removed"))
            self.assertFalse(check(p)["ok"])


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap not installed")
class SandboxTests(unittest.TestCase):
    def test_hidden_readonly_and_writable(self):
        from clinevolve.generation_worker import bwrap_args
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "data").mkdir(); (d / "data/secret.csv").write_text("patient")
            (d / "cases.csv").write_text("case")
            (d / "code").mkdir(); (d / "ws").mkdir()
            args = bwrap_args({"readonly": [str(d / "code")], "hidden": [str(d / "data"), str(d / "cases.csv")]},
                              [(str(d / "ws"), False)])
            sh = (f"test ! -e {d}/data/secret.csv && test ! -s {d}/cases.csv && ! touch {d}/code/x 2>/dev/null "
                  f"&& touch {d}/ws/ok")
            r = subprocess.run(args + ["--", "bash", "-c", sh])
            self.assertEqual(r.returncode, 0)
            self.assertTrue((d / "ws/ok").exists())


if __name__ == "__main__":
    unittest.main()
