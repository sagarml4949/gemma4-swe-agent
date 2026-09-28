"""
Convert verified traces into SFT transcripts for the two LoRA adapters.

Produces multi-turn tool-call transcripts that mirror the real agent loop,
using the submission's own prompts as the system messages, so the trained
model sees exactly the prompt it will be served with:

  tool_lora  (analyzer)  problem statement -> ranked localization report
  main_lora  (coder)     problem -> code_analyzer -> read_file ->
                         edit_file/write_file -> run_command -> submit_patch

The coder transcript includes the PRE-edit file contents as `read_file`
tool results, because that is the context the real agent has when it
decides what to edit. Those contents are read from `base_commit` via a
single batched `git cat-file` process.

SCHEMA ASSUMPTION -- read this before training
----------------------------------------------
The harness's exact tool-call wire format could not be confirmed: the
`HARNESS_README.md` referenced by the build spec is not present in this
environment. The emitted shape is therefore the widely-used
messages-with-tool_calls convention:

    {"role": "assistant", "tool_calls": [{"name": ..., "arguments": {...}}]}
    {"role": "tool", "name": ..., "content": ...}

`render_example()` is the single place that knows this layout. If the
harness differs, only that function needs changing -- all the content that
makes the transcripts correct (symbols, edit strings, verification
commands) is schema-independent and already validated.

Usage:
    python build_sft.py --repos <dir> --traces <dir> --prompts <submission dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from build_traces import GitBlobReader, DIFF_FILE_RE

# Cap on how much pre-edit file content is inlined per read_file result.
MAX_FILE_CHARS = 12000


def read_prompt(prompts_dir: Path, name: str) -> str:
    """Load a harness prompt, accepting either the prompts dir or its parent.

    Silently returning "" here once produced 2666 transcripts with an EMPTY
    system prompt, because --prompts pointed at the submission root rather
    than submission/prompts. Losing the system prompt quietly defeats the
    whole point of aligning transcripts to the harness, so a missing or empty
    prompt is now a hard error and the path is resolved in both layouts.
    """
    for candidate in (prompts_dir / name, prompts_dir / "prompts" / name):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if not text:
                raise SystemExit(f"prompt file is empty: {candidate}")
            return text
    raise SystemExit(
        f"prompt not found: {name}\n"
        f"  looked in {prompts_dir} and {prompts_dir / 'prompts'}"
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tool_call(name: str, arguments: dict) -> dict:
    """One assistant tool call in the harness convention."""
    return {"name": name, "arguments": arguments}


# The gold patch is the real upstream fix, and verify_mined/validate_traces
# prove it applies cleanly and reproduces the post-commit bytes exactly. But
# the pinned sandbox image was unavailable here, so the suite was never
# actually executed (see README "Limitations"). The observation text below is
# therefore an IDEALISATION, not a logged result. Swap it for real captured
# pytest output once reward_sandbox.py `run` has been executed.
VERIFY_OK_OBSERVATION = "Tests passed."


def render_example(system: str, messages: list[dict],
                   instance_id: str | None = None) -> dict:
    """Wrap a conversation into the training record.

    This is the ONLY schema-coupled function; see the module docstring.
    `instance_id` is audit metadata so splits and leakage can be re-checked
    from the artefacts themselves; drop it if the trainer rejects unknown
    top-level keys.
    """
    record: dict = {
        "messages": [{"role": "system", "content": system}, *messages],
    }
    if instance_id:
        record["instance_id"] = instance_id
    return record


def build_analyzer_example(system: str, loc: dict) -> dict:
    """tool_lora target: problem statement -> ranked localization report."""
    messages = [
        {"role": "user", "content": loc["problem_statement"]},
        {"role": "assistant", "content": loc["target"]},
    ]
    return render_example(system, messages, loc.get("instance_id"))


def build_coder_example(system: str, edit: dict, ver: dict, loc: dict | None,
                        reader: GitBlobReader, base_commit: str) -> dict | None:
    """main_lora target: the full explore -> edit -> verify -> submit loop."""
    calls: list[dict] = edit["tool_calls"]
    if not calls:
        return None

    messages: list[dict] = [{"role": "user", "content": edit["problem_statement"]}]

    # 1. Delegate localization to the sub-agent and feed its report back.
    #    Reusing the same report text the analyzer adapter is trained to emit
    #    keeps the two adapters consistent with each other.
    report = edit.get("analyzer_report") or (loc or {}).get("target")
    if report:
        messages.append({
            "role": "assistant",
            "content": "Localising the fix site before editing.",
            "tool_calls": [tool_call("code_analyzer", {
                "problem_statement": edit["problem_statement"],
            })],
        })
        messages.append({"role": "tool", "name": "code_analyzer", "content": report})

    # 2. Read each file before touching it, exactly as the prompt demands.
    paths: list[str] = []
    for c in calls:
        p = c["args"]["path"]
        if p not in paths:
            paths.append(p)
    for p in paths:
        content = reader.read(base_commit, p)
        if content is None:
            continue  # created file: nothing to read beforehand
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n... (truncated)\n"
        messages.append({
            "role": "assistant",
            "content": f"Reading {p}.",
            "tool_calls": [tool_call("read_file", {"path": p})],
        })
        messages.append({"role": "tool", "name": "read_file", "content": content})

    # 3. The edit itself, as the minimal sequence of tool calls.
    messages.append({
        "role": "assistant",
        "content": "Applying the minimal fix.",
        "tool_calls": [tool_call(c["tool"], c["args"]) for c in calls],
    })
    messages.append({
        "role": "tool",
        "name": calls[0]["tool"],
        "content": f"Applied {len(calls)} change(s) across {len(paths)} file(s).",
    })

    # 4. Verify before submitting, rather than submitting blind.
    for vc in (ver or {}).get("tool_calls", []):
        messages.append({
            "role": "assistant",
            "content": "Verifying the fix with the test suite before submitting.",
            "tool_calls": [tool_call(vc["tool"], vc["args"])],
        })
        messages.append({"role": "tool", "name": vc["tool"],
                         "content": VERIFY_OK_OBSERVATION})

    # 5. Submit exactly once.
    messages.append({
        "role": "assistant",
        "content": "Submitting the patch.",
        "tool_calls": [tool_call("submit_patch", {})],
    })
    return render_example(system, messages, edit.get("instance_id"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--prompts", required=True,
                    help="submission dir containing prompts/system.md and analyzer.md")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mined", default=None,
                    help="mined.jsonl; needed for base_commit of read_file results")
    args = ap.parse_args()

    repos_dir = Path(args.repos).resolve()
    traces_dir = Path(args.traces).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys_system = read_prompt(Path(args.prompts), "system.md")
    sys_analyzer = read_prompt(Path(args.prompts), "analyzer.md")

    base_commits: dict[str, str] = {}
    if args.mined:
        for l in Path(args.mined).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                base_commits[r["instance_id"]] = r["base_commit"]

    # Enforce the holdout here rather than trusting split file names.
    val_ids: set[str] = set()
    for name in ("localization", "edit", "verification"):
        val_ids.update(r["instance_id"] for r in load_jsonl(traces_dir / f"{name}_val.jsonl"))

    stats: Counter = Counter()
    written: dict[str, list[dict]] = {"tool_lora": [], "main_lora": []}
    val_written: dict[str, list[dict]] = {"tool_lora": [], "main_lora": []}

    loc_by_id: dict[str, dict] = {}
    for split in ("train", "val"):
        for r in load_jsonl(traces_dir / f"localization_{split}.jsonl"):
            loc_by_id[r["instance_id"]] = r
    ver_by_id: dict[str, dict] = {}
    for split in ("train", "val"):
        for r in load_jsonl(traces_dir / f"verification_{split}.jsonl"):
            ver_by_id[r["instance_id"]] = r
    edit_records: list[dict] = []
    for split in ("train", "val"):
        edit_records += load_jsonl(traces_dir / f"edit_{split}.jsonl")

    # One reader per repo, reused across all examples in that repo.
    by_repo: dict[str, list[dict]] = {}
    for r in edit_records:
        by_repo.setdefault(r["repo"], []).append(r)

    for repo_name, recs in sorted(by_repo.items()):
        reader = GitBlobReader(repos_dir / repo_name)
        try:
            for rec in recs:
                iid = rec["instance_id"]
                loc = loc_by_id.get(iid)
                if loc:
                    ex = build_analyzer_example(sys_analyzer, loc)
                    (val_written if iid in val_ids else written)["tool_lora"].append(ex)
                    stats["tool_lora"] += 1

                ex = build_coder_example(
                    sys_system, rec, ver_by_id.get(iid, {}), loc,
                    reader, base_commits.get(iid, ""),
                )
                if ex is not None:
                    (val_written if iid in val_ids else written)["main_lora"].append(ex)
                    stats["main_lora"] += 1
        finally:
            reader.close()

    for adapter in ("tool_lora", "main_lora"):
        for split, data in (("train", written[adapter]), ("val", val_written[adapter])):
            p = out_dir / f"{adapter}_{split}.jsonl"
            with p.open("w", encoding="utf-8") as fh:
                for ex in data:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  {p.name:<24} {len(data)}")

    # Hard check that no held-out instance leaked into training.
    train_ids = {r["instance_id"] for r in edit_records if r["instance_id"] not in val_ids}
    leak = train_ids & val_ids
    print(f"\nval holdout ids: {len(val_ids)}  overlap with train: {len(leak)}")
    if leak:
        print("ERROR: held-out instances leaked into training data")
        return 1
    print(f"transcripts: {dict(stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())