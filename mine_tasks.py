"""
Mine SWE-bench-style (issue, patch, test_patch) instances from git history.

Applies the competition's mining criteria:
  * the commit jointly modifies Python source AND a test_*.py file,
  * a linkable issue reference (#N) is captured when present,
  * the change is real code (not docs/lockfiles/vendor/type-only churn).

Mining runs against locally cloned repos rather than the GitHub REST API,
because unauthenticated REST is capped at 60 req/hour, which is far too few
to walk four repositories' history.

Each repo is walked with a SINGLE `git log -p` process that emits commit
metadata and unified diffs together, rather than one `git diff` per commit;
that turns an hours-long crawl into a seconds-long one.

Usage:
    python mine_tasks.py --repos <dir> --out <file.jsonl> [--repo NAME ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Field/unit separators that cannot occur in commit text.
RS = "\x1e"
RS_S = "\x1f"


# Paths whose changes never constitute a source fix.
EXCLUDE_PATH_RE = re.compile(
    r"(^|/)(docs?|documentation|examples?|benchmarks?|scripts?|tools?)/"
    r"|\.md$|\.rst$|\.txt$|\.lock$|\.cfg$|\.toml$|\.ini$|\.yml$|\.yaml$"
    r"|(^|/)(\.github|node_modules|vendor|_vendor|venv|\.venv)/"
    r"|\.pyc$|\.so$|\.pyd$|\.png$|\.svg$|\.json$"
)


# Pure-mechanical edits: no behavioural change worth learning from.
MECHANICAL_RE = re.compile(
    r"^\s*(#|//|\"\"\"|'''|\*|from __future__|import |$)"
)


ISSUE_RE = re.compile(r"#(\d+)")
PR_RE = re.compile(r"\(#(\d+)\)\s*$")


TEST_FILE_RE = re.compile(r"(^|/)tests?/.*\.py$|(^|/)test_[^/]*\.py$|[^/]*_test\.py$")


# Matches the "diff --git a/<path> b/<path>" header of a per-file hunk.
DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)


MAX_SRC_FILES = 10
MAX_TEST_FILES = 10
MAX_PATCH_CHARS = 20000
MAX_COMMIT_SUBJECT = 200


def is_test_path(path: str) -> bool:
    return bool(TEST_FILE_RE.search(path))


def is_source_path(path: str) -> bool:
    return path.endswith(".py") and not is_test_path(path) and not EXCLUDE_PATH_RE.search(path)


@dataclass
class Instance:
    """One mined (issue, patch, test_patch) triple."""

    repo: str
    instance_id: str
    base_commit: str
    commit: str
    problem_statement: str
    patch: str
    test_patch: str
    src_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    issue_number: int | None = None
    commit_subject: str = ""
    patch_hash: str = ""


def run_git(repo: Path, *args: str) -> str:
    """Run a git command in `repo`, returning stdout as text."""
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr[:400]}")
    return res.stdout


def content_hash(text: str) -> str:
    """Stable content hash for deduplication.

    Python's built-in hash() is randomized per process for strings
    (PYTHONHASHSEED), so it does not deduplicate reproducibly across runs.
    A cryptographic digest is stable, and also avoids the astronomically
    unlikely collisions a truncated hash could produce.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_commits(repo: Path, limit: int | None = None) -> list[dict]:
    """Return commit metadata + unified diffs for the repo's history.

    Runs ONE `git log -p` process for the whole history. Spawning a separate
    `git diff` per commit is orders of magnitude slower on Windows, where
    process creation dominates; streaming the patches with the metadata
    keeps the full crawl to a few seconds per repository.

    Records are delimited by RS, which git only ever emits here, so it
    cannot collide with diff content.
    """
    args = ["log", "-p", "--no-color", "-U3", "--no-merges",
            f"--format={RS}%H{RS_S}%P{RS_S}%an{RS_S}%aI{RS_S}%s"]
    if limit:
        args.append(f"-n{limit}")
    out = run_git(repo, *args)

    commits = []
    for chunk in out.split(RS):
        lines = chunk.splitlines()
        if not lines:
            continue
        parts = lines[0].split(RS_S)
        if len(parts) < 5 or len(parts[0]) < 7:
            continue
        patch = "\n".join(lines[1:])
        files = [m.group(1) for m in DIFF_FILE_RE.finditer(patch)]
        commits.append(
            {
                "sha": parts[0],
                "parents": parts[1].split(),
                "author": parts[2],
                "date": parts[3],
                "subject": parts[4],
                "patch": patch,
                "files": files,
            }
        )
    return commits


def split_patch_by_file(patch: str) -> dict[str, str]:
    """Split a multi-file unified diff into {path: file_section}.

    Each section keeps its "diff --git" header so the result can be
    rejoined into a patch that still applies with `git apply`.
    """
    sections: dict[str, str] = {}
    matches = list(DIFF_FILE_RE.finditer(patch))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(patch)
        sections[m.group(1)] = patch[start:end]
    return sections


def build_instance(repo_name: str, c: dict) -> Instance | None:
    """Turn one commit into an Instance, or None if it fails the criteria."""
    files = c["files"]
    src = [f for f in files if is_source_path(f)]
    tests = [f for f in files if is_test_path(f) and f.endswith(".py")]

    if not src or not tests:
        return None
    if len(src) > MAX_SRC_FILES or len(tests) > MAX_TEST_FILES:
        return None

    base = c["parents"][0] if c["parents"] else ""
    if not base:
        return None

    # Split the single streamed patch into per-file sections, then rebuild
    # the source and test patches from the in-memory text. No git call.
    sections = split_patch_by_file(c["patch"])
    src_names = [f for f in sections if is_source_path(f)]
    test_names = [f for f in sections if is_test_path(f) and f.endswith(".py")]

    patch = "".join(sections[f] for f in src_names)
    test_patch = "".join(sections[f] for f in test_names)

    if not patch.strip() or not test_patch.strip():
        return None
    if len(patch) > MAX_PATCH_CHARS:
        return None
    # Reject whitespace/import/comment-only churn: no behavioural signal.
    added = [ln for ln in patch.splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    if not added or all(MECHANICAL_RE.match(ln) for ln in added):
        return None

    # A linkable issue: prefer an explicit trailing (#N) PR reference.
    m = PR_RE.search(c["subject"])
    issue = int(m.group(1)) if m else None
    if issue is None:
        ids = ISSUE_RE.findall(c["subject"])
        issue = int(ids[0]) if ids else None

    # Problem statement: the commit subject stands in for the linked issue
    # title, the closest always-available proxy without API calls.
    subject = c["subject"].strip()
    if len(subject) > MAX_COMMIT_SUBJECT:
        subject = subject[:MAX_COMMIT_SUBJECT].rstrip() + "..."

    return Instance(
        repo=repo_name,
        instance_id=f"{repo_name}__{c['sha'][:12]}",
        base_commit=base,
        commit=c["sha"],
        problem_statement=subject,
        patch=patch,
        test_patch=test_patch,
        src_files=src_names,
        test_files=test_names,
        issue_number=issue,
        commit_subject=subject,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True, help="directory of cloned repos")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--repo", action="append", default=[],
                    help="restrict to these repo dir names (repeatable)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max commits to consider per repo")
    args = ap.parse_args()

    root = Path(args.repos).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = args.repo or sorted(p.name for p in root.iterdir() if (p / ".git").exists())
    all_instances: list[Instance] = []

    for name in wanted:
        repo_path = root / name
        if not (repo_path / ".git").exists():
            print(f"[{name}] SKIP: not a git repo", flush=True)
            continue
        commits = iter_commits(repo_path, args.limit)
        kept = 0
        for c in commits:
            c["_repo"] = repo_path
            inst = build_instance(name, c)
            if inst is not None:
                inst.patch_hash = content_hash(inst.patch)
                all_instances.append(inst)
                kept += 1
        print(f"[{name}] scanned={len(commits)} kept={kept}", flush=True)

    # Deduplicate on patch content: the same change often appears as both a
    # squash-merge commit and a merge commit.
    seen: set[tuple] = set()
    unique: list[Instance] = []
    for inst in all_instances:
        key = (inst.repo, inst.patch_hash, tuple(sorted(inst.test_files)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(inst)

    with out_path.open("w", encoding="utf-8") as fh:
        for inst in unique:
            fh.write(json.dumps(asdict(inst), ensure_ascii=False) + "\n")

    print(f"\nTOTAL kept={len(all_instances)} unique={len(unique)} -> {out_path}")
    by_repo: dict[str, int] = {}
    for inst in unique:
        by_repo[inst.repo] = by_repo.get(inst.repo, 0) + 1
    for repo, n in sorted(by_repo.items()):
        print(f"  {repo:<10} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
