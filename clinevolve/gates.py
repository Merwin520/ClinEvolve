"""Safety gates G1-G6.

For candidate C, reference R and validation cohort V, with f_e(X) the failure rate of version X on endpoint e over the
paired cases evaluable for both versions, a candidate is released only if

  G1 target       f_t(R) - f_t(C) >= target_minimum_improvement_pp
  G2 retention    f_r(C) - f_r(R) <= retention_margin_pp                  for every previously repaired target r
  G3 non-target   f_e(C) - f_e(R) <= nontarget_workflow_margin_pp        other workflow endpoints (PD, PL, PT)
                  f_FD(C) - f_FD(R) <= final_diagnosis_margin_pp
  G4 tools        increase <= tool_safety_margin_pp for each tool-safety endpoint (invalid call, tool execution error,
                  context overflow, trajectory failure, instruction/tool failure)
  G5 resources    mean tool calls <= ratio x reference; mean total tokens <= ratio x reference
  G6 new errors   NR(C) - NR_ref <= new_regression_margin_pp

NR(C) is the share of validation patients correct under A0 on any of PD, PL, PT or FD who fail that endpoint under C.
Batched inference is not bitwise deterministic, so NR_ref is the new-regression rate of a replicate run of A0 against A0
when the parent is A0, and the parent's own value otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from clinevolve.config import FD, IT, PD, PL, PT, WORKFLOW


def load(p: Path) -> pd.DataFrame:
    d = pd.read_parquet(p)
    d["case_id"] = d.case_id.astype(str)
    return d.set_index("case_id")


def paired_rate(a: pd.DataFrame, b: pd.DataFrame, ep: str):
    m = pd.concat([a[ep].rename("a"), b[ep].rename("b")], axis=1, join="inner").dropna()
    return float(m.a.mean()), float(m.b.mean()), len(m)


def tool_flags(d: pd.DataFrame) -> pd.DataFrame:
    st = d.trajectory_status.astype(str) if "trajectory_status" in d else pd.Series("success", index=d.index)
    err = d.error.astype(str) if "error" in d else pd.Series("", index=d.index)
    return pd.DataFrame({
        "invalid": (d.get("n_invalid_calls", 0) > 0).astype(float),
        "tool_err": (d.get("n_tool_execution_errors", 0) > 0).astype(float),
        "ctx": err.str.contains("ContextWindow", na=False).astype(float),
        "traj_fail": (st != "success").astype(float),
        "instr": d[IT].astype(float)}, index=d.index)


def mean_tokens(d: pd.DataFrame) -> float:
    vals = []
    for tp in d.trajectory_path.astype(str):
        try:
            u = json.loads(Path(tp).read_text()).get("usage") or {}
            vals.append((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def new_regression(a0: pd.DataFrame, v: pd.DataFrame) -> float:
    eps = [PD, PL, PT, FD]
    m = a0[eps].join(v[eps], lsuffix="_a", rsuffix="_v", how="inner")
    bad = np.zeros(len(m), bool)
    denom = np.zeros(len(m), bool)
    for e in eps:
        a, b = m[e + "_a"], m[e + "_v"]
        valid = a.notna() & b.notna()
        denom |= (valid & (a == 0)).to_numpy()
        bad |= (valid & (a == 0) & (b == 1)).to_numpy()
    return float(bad.sum() / max(denom.sum(), 1))


def evaluate_gates(G: Dict, reference_val: Path, cand_val: Path, a0_val: Path, target: str,
                   retained: Iterable[str] = (), use_gates: bool = True, a0_replicate_val: Optional[Path] = None,
                   reference_is_a0: bool = True) -> Dict:
    """Gate decision for one candidate. ``G`` is the ``gates`` section of the config.

    Returns per-gate booleans, the deltas behind them, ``target.gain_pp`` and ``all_pass``. With ``use_gates=False``
    (gate ablation) a candidate passes whenever its validation target gain is positive."""
    retained = list(retained)
    P_, C_, A_ = load(reference_val), load(cand_val), load(a0_val)
    res: Dict = {}
    pr, cr, n = paired_rate(P_, C_, target)
    res["target"] = {"parent": pr, "cand": cr, "n": n, "gain_pp": (pr - cr) * 100}
    g1 = res["target"]["gain_pp"] >= G["target_minimum_improvement_pp"]
    res["G1_target"] = g1
    worst_ret, g2 = -np.inf, True
    for t in retained:
        p, c, _ = paired_rate(P_, C_, t)
        res[f"retention_{t}"] = (c - p) * 100
        worst_ret = max(worst_ret, (c - p) * 100)
        g2 &= (c - p) * 100 <= G["retention_margin_pp"]
    res["G2_retention"] = g2
    g3 = True
    for t in [w for w in WORKFLOW if w != target and w not in retained]:
        p, c, _ = paired_rate(P_, C_, t)
        res[f"nontarget_{t}"] = (c - p) * 100
        g3 &= (c - p) * 100 <= G["nontarget_workflow_margin_pp"]
    p, c, _ = paired_rate(P_, C_, FD)
    res["final_dx_pp"] = (c - p) * 100
    g3 &= res["final_dx_pp"] <= G["final_diagnosis_margin_pp"]
    res["G3_nontarget_dx"] = g3
    tp, tc = tool_flags(P_), tool_flags(C_)
    common = tp.index.intersection(tc.index)
    d_tool = ((tc.loc[common].mean() - tp.loc[common].mean()) * 100).to_dict()
    res["tool_delta_pp"] = d_tool
    g4 = all(v <= G["tool_safety_margin_pp"] for v in d_tool.values())
    res["G4_tools"] = g4
    rt = C_.n_tool_calls.mean() / max(P_.n_tool_calls.mean(), 1e-9)
    tp_, tc_ = mean_tokens(P_), mean_tokens(C_)
    rk = (tc_ / tp_) if (tp_ and not np.isnan(tp_)) else 1.0
    res["tool_ratio"], res["token_ratio"] = rt, rk
    R = G["resource_inflation_margin"]
    g5 = rt <= R["mean_tool_calls_ratio"] and rk <= R["mean_total_tokens_ratio"]
    res["G5_resources"] = g5
    nr_c = new_regression(A_, C_)
    nr_p = (new_regression(A_, load(a0_replicate_val)) if (reference_is_a0 and a0_replicate_val is not None)
            else new_regression(A_, P_))
    res["new_regression_reference"], res["new_regression_cand"] = nr_p, nr_c
    g6 = (nr_c - nr_p) * 100 <= G["new_regression_margin_pp"]
    res["G6_new_regression"] = g6
    res["worst_retention_pp"] = worst_ret if retained else 0.0
    res["mean_tokens"] = tc_
    res["all_pass"] = bool(g1 and g2 and g3 and g4 and g5 and g6) if use_gates else bool(res["target"]["gain_pp"] > 0)
    return res


def objective_text(target: str, retained: Iterable[str] = ()) -> str:
    """Repair objective handed to the repairer for one target (with retention of earlier repairs, if any)."""
    objectives = {
        PD: "Improve the agent's diagnostic process (physical examination first, complete disease-relevant laboratory "
            "work-up, appropriate imaging)",
        PL: "Improve the agent's laboratory-result interpretation (correct direction and clinical meaning of abnormal "
            "values and their integration into the diagnosis)",
        PT: "Improve the agent's treatment-guideline reasoning (guideline-concordant management for the diagnosed "
            "condition)",
    }
    retained = list(retained)
    keep = (" while preserving the previously repaired " + " and ".join(retained)) if retained else ""
    return (f"{objectives[target]} on the provided repair-set evidence{keep}, and without degrading other workflow "
            "behavior, final diagnosis quality, tool safety or resource discipline. Make one coherent modification "
            "within the allowed paths only.")
