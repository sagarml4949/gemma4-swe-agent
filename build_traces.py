"""
Build SFT traces from mined (issue, patch, test_patch) instances.

Produces the three supervised traces the spec calls for:

  * localization  -> problem statement to the gold symbol(s) the patch
                     touches, rendered in the ranked report format that
                     prompts/analyzer.md asks for (trains tool_lora).
  * edit          -> problem statement + analyzer report + pre-edit file
                     contents to the edit_file calls that reproduce the
                     patch (trains main_lora).
  * verification  -> the run_command call that exercises test_patch, and
                     the expected fail -> pass transition, so the model
                     learns to check its work instead of submitting blind.

Gold symbols are recovered with Python's `ast` from the pre-edit source,
mapping each changed hunk's line range to the innermost enclosing
function/class. That is the same "where is the fix site" answer the
analyzer is being trained to produce, so the supervision matches the
prompt's output contract exactly.

Usage:
    python build_traces.py --repos <dir> --data <mined.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)
DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/", re.M)


@dataclass
class Symbol:
    """A gold localization target: a qualified name and its file."""

    qualname: str
    file: str
    lineno: int
    kind: str  # function | class | module


class GitBlobReader:
    """Persistent `git cat-file --batch` reader for many blobs.

    Reading thousands of pre-edit files with one `git show` process each is
    dominated by process-creation cost on Windows (measured: 56s CPU over
    473s wall for ~1500 instances). This keeps a single git process alive
    and streams requests to it, which removes that overhead entirely.

    Uses BINARY pipes: `cat-file --batch` reports a byte size, so reading
    that many *characters* from a text stream would desynchronise on any
    non-ASCII file.
    """

    def __init__(self, repo: Path) -> None:
        self.proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )

    def read(self, commit: str, path: str) -> str | None:
        """Return file contents at `commit:path`, or None if absent."""
        if self.proc.stdin is None or self.proc.stdout is None:
            return None
        try:
            self.proc.stdin.write(f"{commit}:{path}\n".encode("utf-8"))
            self.proc.stdin.flush()
            header = self.proc.stdout.readline()
            if not header:
                return None
            parts = header.split()
            if len(parts) < 3 or parts[-1] == b"missing":
                return None
            size = int(parts[-1])
            data = self.proc.stdout.read(size)
            self.proc.stdout.read(1)  # trailing newline after the payload
            return data.decode("utf-8", errors="replace")
        except (BrokenPipeError, ValueError):
            return None

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self) -> "GitBlobReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_at(repo: Path, commit: str, path: str) -> str | None:
    """Read one file at a commit (single-shot convenience wrapper)."""
    with GitBlobReader(repo) as r:
        return r.read(commit, path)


def changed_line_ranges(patch: str) -> list[tuple[str, int, int]]:
    """Return (file, start_line, end_line) for each hunk in a patch.

    Lines are 1-based and refer to the POST-image, which is the coordinate
    system the pre-edit file needs to be walked in reverse.
    """
    out: list[tuple[str, int, int]] = []
    matches = list(DIFF_FILE_RE.finditer(patch))
    for i, m in enumerate(matches):
        path = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(patch)
        for h in HUNK_RE.finditer(patch, m.start(), end):
            new_start = int(h.group(3))
            new_len = int(h.group(4)) if h.group(4) is not None else 1
            out.append((path, new_start, new_start + new_len - 1))
    return out


def enclosing_symbols(tree: ast.AST, lines: list[int]) -> list[Symbol]:
    """Return innermost-first symbols covering any of `lines` (1-based)."""
    hits: list[Symbol] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start) or start
        if any(start <= ln <= end for ln in lines):
            hits.append(Symbol(qualname=node.name, file="", lineno=start,
                               kind="class" if isinstance(node, ast.ClassDef) else "function"))
    # Innermost first: deeper nodes (larger start line) are more specific.
    hits.sort(key=lambda s: (-s.lineno, s.qualname))
    return hits


def gold_symbols(reader: GitBlobReader, inst: dict) -> list[Symbol]:
    """Recover the symbols the patch modifies, from the PRE-edit source.

    Hunk line numbers are post-image coordinates, so the pre-edit file is
    walked with the hunk start shifted by the net line delta accumulated
    from this file's earlier hunks.
    """
    base = inst["base_commit"]
    per_file: dict[str, list[tuple[int, int]]] = {}
    for path, start, end in changed_line_ranges(inst["patch"]):
        per_file.setdefault(path, []).append((start, end))

    symbols: list[Symbol] = []
    for path, ranges in per_file.items():
        source = reader.read(base, path)
        if not source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        total_lines = len(source.split("\n"))
        delta = 0
        pre_lines: list[int] = []
        for start, end in ranges:
            pre_lines.append(max(1, min(total_lines, start - delta)))
            delta += (end - start + 1)
        found = enclosing_symbols(tree, pre_lines)
        for s in found:
            s.file = path
        symbols.extend(found)

    seen: set[tuple[str, str]] = set()
    unique: list[Symbol] = []
    for s in symbols:
        key = (s.file, s.qualname)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def render_analyzer_report(symbols: list[Symbol]) -> str:
    """Render gold symbols in the ranked format prompts/analyzer.md asks for.

    Matching that contract exactly is the point: the localization trace
    teaches the model to answer in the shape the coder prompt will parse.
    """
    if not symbols:
        return ("Most likely fix location(s):\n"
                "1. (unknown) - could not localise a symbol with confidence\n\n"
                "Confidence: low, because no clear symbol matched the report.")

    top = symbols[:3]
    lines = ["Most likely fix location(s):"]
    for i, s in enumerate(top, 1):
        lines.append(f"{i}. {s.qualname} ({s.file}) - matches the reported behaviour")
    lines.append("")
    lines.append("Related context the caller should be aware of:")
    if len(symbols) > 1:
        extra = ", ".join(sorted({s.qualname for s in symbols[1:4]}))
        lines.append(f"- {extra} may need to change in tandem with #1.")
    else:
        lines.append("- no other clearly related symbol was found.")
    lines.append("")
    conf = "high" if len(top) == 1 else "medium"
    lines.append(f"Confidence: {conf}, because the reported behaviour maps onto "
                 "the symbol(s) named above.")
    return "\n".join(lines)


def _split_lines(text: str) -> list[str]:
    """Split file content into lines, dropping the trailing-newline artefact."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def render_edit_calls(reader: GitBlobReader, inst: dict,
                      max_pad: int = 25) -> tuple[list[dict], str | None]:
    """Turn the patch into the exact edit_file/write_file calls that replay it.

    Strings are taken from the real pre- and post-image file contents rather
    than sliced out of the diff text, so the rendered trace reproduces the
    commit byte-for-byte.

    Two cases the naive diff-based rendering got wrong:

    * Created files have no pre-image, so they are emitted as a single
      `write_file` with the full content -- which is also what the system
      prompt tells the agent to do for genuinely new files.
    * A hunk whose context is repeated in the file would produce an
      ambiguous `old_string`. Context is expanded outward (clamped so it
      cannot overlap a neighbouring hunk) until the old_string is unique.
      If it still is not unique, the instance is rejected rather than
      emitting a trace that teaches non-deterministic editing.

    Returns (calls, skip_reason). skip_reason is None on success.
    """
    base_sha, post_sha = inst["base_commit"], inst["commit"]
    calls: list[dict] = []

    matches = list(DIFF_FILE_RE.finditer(inst["patch"]))
    for i, m in enumerate(matches):
        path = m.group(1)
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(inst["patch"])
        section = inst["patch"][m.start():section_end]

        pre_raw = reader.read(base_sha, path)
        post_raw = reader.read(post_sha, path)

        # A rename shows up here as "pre exists, post does not" because the
        # section path is the PRE-image path. Neither edit_file nor
        # write_file can express a rename, so it is skipped explicitly
        # rather than mislabelled as a deletion.
        is_rename = "rename to " in section

        # Created file: full-content write_file.
        if pre_raw is None and post_raw is not None:
            calls.append({
                "tool": "write_file",
                "args": {"path": path, "content": post_raw},
            })
            continue
        if post_raw is None:
            reason = "rename (not expressible)" if is_rename else "deleted file (not expressible)"
            return [], f"{reason}: {path}"
        if pre_raw is None:
            return [], f"unreadable pre-image: {path}"

        pre = _split_lines(pre_raw)
        post = _split_lines(post_raw)

        hunks = list(HUNK_RE.finditer(section))
        # Cumulative line delta from hunks already applied to this file.
        # Pre-image line p maps to post-image line p + cum_delta; without
        # this, any file with more than one hunk reconstructs incorrectly.
        cum_delta = 0
        for hi, hm in enumerate(hunks):
            old_start = int(hm.group(1))
            old_len = int(hm.group(2)) if hm.group(2) is not None else 1
            new_len = int(hm.group(4)) if hm.group(4) is not None else 1

            # Bounds this hunk may expand into without colliding with a
            # neighbouring hunk's region.
            prev_end = 0
            if hi > 0:
                pm = hunks[hi - 1]
                ps = int(pm.group(1))
                pl = int(pm.group(2)) if pm.group(2) is not None else 1
                prev_end = ps - 1 + pl
            next_start = len(pre)
            if hi + 1 < len(hunks):
                next_start = int(hunks[hi + 1].group(1)) - 1

            lo = old_start - 1
            hi_end = lo + old_len
            delta = new_len - old_len

            chosen: tuple[str, str] | None = None
            for pad in range(0, max_pad + 1):
                a = max(prev_end, lo - pad)
                b = min(next_start, hi_end + pad)
                old_str = "\n".join(pre[a:b])
                # Same region on the post side, shifted by the deltas of
                # previously applied hunks, and grown by this hunk's delta.
                pa = a + cum_delta
                new_str = "\n".join(post[pa:pa + (b - a) + delta])
                if not old_str:
                    continue
                if _count_occurrences(pre_raw, old_str) == 1:
                    chosen = (old_str, new_str)
                    break
            if chosen is None:
                return [], f"ambiguous old_string in {path}"

            cum_delta += delta
            calls.append({
                "tool": "edit_file",
                "args": {"path": path, "old_string": chosen[0], "new_string": chosen[1]},
            })
    return calls, None


def _count_occurrences(haystack: str, needle: str) -> int:
    """Count non-overlapping occurrences of `needle` in `haystack`."""
    if not needle:
        return 0
    return haystack.count(needle)


def render_verification_calls(inst: dict) -> list[dict]:
    """Build the run_command trace that exercises test_patch.

    The expected outcome is the fail -> pass transition the spec asks for:
    the test fails at base_commit and passes once the patch is applied.
    """
    test_files = inst["test_files"]
    if not test_files:
        return []
    target = " ".join(test_files[:2])
    return [{
        "tool": "run_command",
        "args": {"command": f"python -m pytest {target} -q"},
        "expected": "fails at base_commit, passes after the patch is applied",
    }]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="held-out validation fraction, stratified by repo")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    root = Path(args.repos).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    insts = [json.loads(l) for l in
             Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]

    loc_recs: list[dict] = []
    edit_recs: list[dict] = []
    ver_recs: list[dict] = []
    no_symbol = Counter()
    edit_skipped: Counter = Counter()

    # Group by repo so a single `git cat-file --batch` process per repo can
    # serve every instance, instead of one process per file read.
    by_repo: dict[str, list[dict]] = {}
    for inst in insts:
        by_repo.setdefault(inst["repo"], []).append(inst)

    for repo_name, repo_insts in sorted(by_repo.items()):
        reader = GitBlobReader(root / repo_name)
        try:
            for inst in repo_insts:
                symbols = gold_symbols(reader, inst)
                problem = inst["problem_statement"]
                # Only keep localization supervision where a real gold symbol
                # was recovered. Emitting a "could not localise" target for
                # these would actively train the analyzer to give up, which
                # is worse than omitting the example.
                if symbols:
                    loc_recs.append({
                        "instance_id": inst["instance_id"],
                        "repo": inst["repo"],
                        "problem_statement": problem,
                        "gold_symbols": [asdict(s) for s in symbols],
                        "target": render_analyzer_report(symbols),
                    })
                else:
                    no_symbol[inst["repo"]] += 1
                    symbols = []
                edit_calls, skip = render_edit_calls(reader, inst)
                if skip is not None:
                    edit_skipped[skip.split(": ")[0] if skip.startswith(
                        ("rename", "deleted")) else skip.rsplit(" in ", 1)[0]] += 1
                if edit_calls:
                    edit_recs.append({
                        "instance_id": inst["instance_id"],
                        "repo": inst["repo"],
                        "problem_statement": problem,
                        "analyzer_report": render_analyzer_report(symbols),
                        "tool_calls": edit_calls,
                    })
                ver_calls = render_verification_calls(inst)
                if ver_calls:
                    ver_recs.append({
                        "instance_id": inst["instance_id"],
                        "repo": inst["repo"],
                        "tool_calls": ver_calls,
                    })
            print(f"  {repo_name}: processed {len(repo_insts)}", flush=True)
        finally:
            reader.close()

    # Stratified holdout: split by repo so both splits stay repo-balanced.
    import random
    rng = random.Random(args.seed)
    recs_by_repo: dict[str, list[dict]] = {}
    for r in loc_recs:
        recs_by_repo.setdefault(r["repo"], []).append(r)
    val_ids: set[str] = set()
    for repo, recs in recs_by_repo.items():
        recs = sorted(recs, key=lambda r: r["instance_id"])
        rng.shuffle(recs)
        k = max(1, int(round(len(recs) * args.val_frac)))
        val_ids.update(r["instance_id"] for r in recs[:k])

    def dump(name: str, recs: list[dict]) -> None:
        train = [r for r in recs if r["instance_id"] not in val_ids]
        val = [r for r in recs if r["instance_id"] in val_ids]
        for split, data in (("train", train), ("val", val)):
            p = out_dir / f"{name}_{split}.jsonl"
            with p.open("w", encoding="utf-8") as fh:
                for r in data:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {p.name:<28} {len(data)}")

    print("Wrote traces:")
    dump("localization", loc_recs)
    dump("edit", edit_recs)
    dump("verification", ver_recs)

    print(f"\ninstances={len(loc_recs)} val_ids={len(val_ids)}")
    print("no-symbol (skipped localization):", dict(no_symbol) or "none")
    print("edit-trace skips:", dict(edit_skipped) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())

