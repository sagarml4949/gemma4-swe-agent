"""
Validate that the edit traces faithfully reproduce the real patch.

The strongest check available: replay each trace's edit_file(old_string,
new_string) calls against the PRE-edit file content, then compare the
reconstructed result to the actual POST-commit file content for that path.
If they match byte-for-byte, the edit trace is a faithful supervision
signal for the exact tool calls the agent must learn to emit.

Also verifies each old_string actually occurs in the pre-edit file, since a
trace whose old_string is absent would teach the model to hallucinate
context.

Usage:
    python validate_traces.py --repos <dir> --traces <dir> [--sample N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from build_traces import GitBlobReader


def replay(base_text: str | None, calls: list[dict]) -> tuple[str | None, str]:
    """Apply edit_file/write_file calls in order; return (final_text, error)."""
    current = base_text
    for call in calls:
        tool = call.get("tool")
        args = call["args"]
        path = args["path"]
        if tool == "write_file":
            # A created file has no pre-image; the trace supplies full content.
            current = args["content"]
            continue
        if tool != "edit_file":
            continue
        if current is None:
            return None, f"edit_file on missing pre-image for {path}"
        old, new = args["old_string"], args["new_string"]
        if old not in current:
            return None, f"old_string absent in {path}"
        if current.count(old) > 1:
            return None, f"old_string ambiguous ({current.count(old)}x) in {path}"
        current = current.replace(old, new, 1)
    return current, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--mined", required=True, help="mined.jsonl for base/commit shas")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.repos).resolve()
    traces_dir = Path(args.traces).resolve()
    mined = {json.loads(l)["instance_id"]: json.loads(l)
             for l in Path(args.mined).read_text(encoding="utf-8").splitlines() if l.strip()}

    records: list[dict] = []
    for split in ("train", "val"):
        p = traces_dir / f"edit_{split}.jsonl"
        if p.is_file():
            records += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    rng = random.Random(args.seed)
    if args.sample:
        records = rng.sample(records, min(args.sample, len(records)))

    by_repo: dict[str, list[dict]] = {}
    for r in records:
        by_repo.setdefault(r["repo"], []).append(r)

    reasons: Counter = Counter()
    exact = calls_total = 0
    checked = 0

    for repo_name, recs in sorted(by_repo.items()):
        reader = GitBlobReader(root / repo_name)
        try:
            for rec in recs:
                inst = mined.get(rec["instance_id"])
                if not inst:
                    reasons["no mined record"] += 1
                    continue
                base_sha, post_sha = inst["base_commit"], inst["commit"]
                # Group calls per path, preserving order.
                paths: dict[str, list[dict]] = {}
                for call in rec["tool_calls"]:
                    paths.setdefault(call["args"]["path"], []).append(call)
                ok_all = True
                for path, calls in paths.items():
                    calls_total += 1
                    pre = reader.read(base_sha, path)
                    if pre is not None and not pre.endswith("\n"):
                        pre += "\n"
                    got, err = replay(pre, calls)
                    if got is None:
                        reasons[err.split(" in ")[0]] += 1
                        ok_all = False
                        continue
                    want = reader.read(post_sha, path)
                    if want is None:
                        reasons["path missing at commit"] += 1
                        ok_all = False
                        continue
                    if want and not want.endswith("\n"):
                        want += "\n"
                    if got == want:
                        exact += 1
                    else:
                        reasons["reconstruction mismatch"] += 1
                        ok_all = False
                checked += 1
        finally:
            reader.close()

    print(f"checked instances: {checked}  (file-edits replayed: {calls_total})")
    print(f"exact reconstructions: {exact}/{calls_total}"
          + (f" = {exact/calls_total:.1%}" if calls_total else ""))
    if reasons:
        print("failure reasons:")
        for r, n in reasons.most_common(10):
            print(f"  {n:>4}  {r}")
    return 0 if calls_total and exact == calls_total else 1


if __name__ == "__main__":
    sys.exit(main())