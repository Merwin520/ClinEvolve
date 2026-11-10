"""Validate a repairer program (meta_agent.py) without calling the LLM.

python -m clinevolve.meta_check PATH_TO_META_AGENT.py  -> prints JSON {"ok": bool, ...}

The program must import, construct a MetaAgent, and its forward() (called through the standard generation wrapper with
chat_with_agent stubbed) must send an instruction to the LLM that contains the objective and the feedback, with the bash
and edit tools available. Self-edited repairers that fail this check are discarded.
"""
import importlib.util
import json
import sys
from pathlib import Path

from clinevolve.config import REPO_ROOT


def main() -> None:
    import agent.llm_withtools_meta as lw
    import clinevolve.generate as gc

    captured = {}

    def stub(instruction, **k):
        captured["instruction"] = instruction
        captured["tools"] = k.get("tools_available")
        return [{"role": "assistant", "content": "ok"}]

    res = {"ok": False}
    try:
        lw.chat_with_agent = stub
        ms = importlib.util.spec_from_file_location("clinevolve_meta_check", sys.argv[1])
        mod = importlib.util.module_from_spec(ms)
        ms.loader.exec_module(mod)
        if hasattr(mod, "chat_with_agent"):
            mod.chat_with_agent = stub
        m = mod.MetaAgent(model="openai/check", chat_history_file="/dev/null", temperature=0.7, max_tokens=2048)
        gc.call_meta_agent(meta_agent=m, repo_path=REPO_ROOT, objective="Improve X.", feedback="Failures: Y.",
                           allowed_paths=["task_agent.py"], forbidden_paths=["meta_agent.py"], iterations_left=None,
                           extra_context={"procedure": "meta_check"})
        ins = captured.get("instruction")
        assert isinstance(ins, str) and len(ins) > 200, "forward() did not call chat_with_agent with an instruction"
        assert "Improve X." in ins and "Failures: Y." in ins, "objective/feedback not passed to the LLM"
        assert set(captured.get("tools") or []) >= {"bash", "edit"}, f"tools changed: {captured.get('tools')}"
        res = {"ok": True, "instruction_chars": len(ins)}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    print(json.dumps(res))


def check(path: Path) -> dict:
    import subprocess
    out = subprocess.run([sys.executable, "-m", "clinevolve.meta_check", str(path)], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip().splitlines()
    try:
        return json.loads(out[-1])
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "meta_check produced no result"}


if __name__ == "__main__":
    main()
