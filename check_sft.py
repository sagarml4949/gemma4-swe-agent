"""Verify system prompts and tool-call integrity in the generated SFT files.

An empty system prompt once slipped through silently, so this checks the
property directly across every row rather than eyeballing one sample.
"""
import json
from pathlib import Path

PROMPTS = Path("C:/Users/nerel/Downloads/submission/prompts")
EXPECT = {
    "tool_lora": (PROMPTS / "analyzer.md").read_text(encoding="utf-8").strip(),
    "main_lora": (PROMPTS / "system.md").read_text(encoding="utf-8").strip(),
}

out: list[str] = []
bad = 0

for adapter, expected in EXPECT.items():
    for split in ("train", "val"):
        p = Path(f"out/sft/{adapter}_{split}.jsonl")
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_empty_sys = sum(1 for r in rows
                          if not r["messages"][0]["content"].strip())
        n_wrong_sys = sum(1 for r in rows
                          if r["messages"][0]["content"] != expected)
        n_no_sysrole = sum(1 for r in rows if r["messages"][0]["role"] != "system")
        # Every tool message must answer a tool_call from the immediately
        # preceding assistant turn, else the conversation is malformed.
        # A terminal submit_patch legitimately has no succeeding tool message
        # because it ends the task.
        n_orphan_tool = 0
        n_dangling_call = 0
        n_last_not_assistant = 0
        for r in rows:
            msgs = r["messages"]
            if msgs[-1]["role"] != "assistant":
                n_last_not_assistant += 1
            for i, m in enumerate(msgs):
                if m["role"] == "tool":
                    prev = msgs[i - 1]
                    if not prev.get("tool_calls"):
                        n_orphan_tool += 1
                if m["role"] == "assistant" and m.get("tool_calls"):
                    is_terminal_submit = (
                        i == len(msgs) - 1
                        and len(m["tool_calls"]) == 1
                        and m["tool_calls"][0]["name"] == "submit_patch"
                    )
                    nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                    if not is_terminal_submit and (nxt is None or nxt["role"] != "tool"):
                        n_dangling_call += 1
        ok = (not any([n_empty_sys, n_wrong_sys, n_no_sysrole, n_orphan_tool,
                       n_dangling_call, n_last_not_assistant]))
        bad += 0 if ok else 1
        out.append(f"{'PASS' if ok else 'FAIL'}  {adapter}_{split}  rows={len(rows)}")
        out.append(f"      sys prompt: empty={n_empty_sys} mismatched={n_wrong_sys} "
                   f"wrong-role={n_no_sysrole}")
        out.append(f"      structure:  orphan_tool={n_orphan_tool} "
                   f"dangling_call={n_dangling_call} "
                   f"not_ending_on_assistant={n_last_not_assistant}")
        out.append(f"      sys prompt length: {len(expected)} chars")

out.append("")
out.append("SFT PROMPT/STRUCTURE CHECK FAILED" if bad else "ALL SFT CHECKS PASS")
Path("sftcheck.txt").write_text("\n".join(out), encoding="utf-8")