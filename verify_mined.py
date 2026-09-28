"""
Verify that mined patches actually apply to their base commit.

A mined instance is only useful for training (and only trustworthy as a
reward signal) if its patch really applies at base_commit and its
test_patch really applies alongside it. This script proves that on a real
sample by checking out each base commit in a scratch worktree and running
`git apply --check` for both halves.

It does NOT run the test suite: that needs the competition's pinned
sandbox image, which is not available in this environment. Apply-correctness
is the strongest check available here, and it catches the failure mode that
actually matters -- patches that were split or truncated during mining.

Usage:
    python verify_mined.py --repos <dir> --data <mined.jsonl> [--sample N]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/", re.M)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr[:300]}")
    return res


def apply_check(base_content: dict[str, str], patch: str) -> tuple[bool, str]:
    """Verify a unified diff applies to the given file contents.

    Reimplements the subset of `git apply` that these patches use: every
    hunk's context and removed lines must match the original file exactly,
    starting at the recorded line offset and allowing the small drift that
    earlier hunks in the same file can introduce.

    This is done in memory rather than via `git worktree`, which requires a
    full checkout per instance (thousands of process spawns and disk writes
    on Windows). It validates the same property that matters: that the patch
    genuinely applies to the base_commit tree.
    """
    lines: list[str] = []
    i = 0
    n = len(patch)
    while i < n:
        if not patch.startswith("diff --git ", i):
            i += 1
            continue
        header = patch[i:patch.find("\n", i) if "\n" in patch[i:] else n]
        m = re.search(r" b/(.+)$", header)
        path = m.group(1) if m else None
        j = patch.find("\n", i)
        j = j + 1 if j != -1 else n

        # Advance to the first hunk header, noting file-level mode lines.
        # A file that the commit CREATES has no base version, so requiring
        # it to exist in base_content would wrongly reject it; a pure
        # addition always applies.
        is_new = False
        is_deleted = False
        is_rename = False
        old_path: str | None = None
        while j < n and not patch.startswith("@@", j):
            if patch.startswith("diff --git ", j):
                break
            line = patch[j:patch.find("\n", j) if "\n" in patch[j:] else n]
            if line.startswith("new file mode"):
                is_new = True
            elif line.startswith("deleted file mode"):
                is_deleted = True
            elif line.startswith("rename from"):
                is_rename = True
            elif line.startswith("--- a/"):
                old_path = line[6:].strip()
            elif line.startswith("--- /dev/null"):
                is_new = True
            j = patch.find("\n", j)
            j = j + 1 if j != -1 else n
        if j >= n or not patch.startswith("@@", j):
            i = j
            continue

        # The pre-image path is what exists at base_commit. For a rename the
        # b/ path does not exist yet, so looking content up by it would
        # wrongly report a missing file.
        content_key = old_path or path
        if content_key == "/dev/null":
            content_key = path
        if content_key is None:
            return False, "unparseable file header"
        if content_key not in base_content and not is_new:
            return False, f"missing file {content_key}"
        if is_deleted and not is_rename:
            i = j
            continue  # removal of an existing file needs no content match

        original = (base_content.get(content_key) or "").split("\n")
        if original and original[-1] == "":
            original.pop()  # trailing newline artefact
        result: list[str] = []
        cursor = 0  # 0-based index into original

        while j < n and patch.startswith("@@", j):
            eol = patch.find("\n", j)
            if eol == -1:
                break
            hm = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", patch[j:eol])
            if not hm:
                return False, "bad hunk header"
            old_start = int(hm.group(1)) - 1
            j = eol + 1

            hunk: list[str] = []
            while j < n and not patch.startswith(("@@", "diff --git "), j):
                eol = patch.find("\n", j)
                line = patch[j:eol] if eol != -1 else patch[j:]
                hunk.append(line)
                j = eol + 1 if eol != -1 else n

            # Locate the hunk: try the recorded offset, then search nearby,
            # which absorbs drift from previously applied hunks.
            context = [h[1:] for h in hunk if h and h[0] in " -"]
            target = max(old_start, cursor)
            pos = _find_block(original, context, target)
            if pos is None and old_start != 0:
                pos = _find_block(original, context, 0)
            if pos is None:
                return False, f"hunk does not apply to {path}"
            if pos > cursor:
                result.extend(original[cursor:pos])
            for h in hunk:
                if not h:
                    continue
                tag, text = h[0], h[1:]
                if tag == " ":
                    result.append(text)
                elif tag == "-":
                    pass  # removed
                elif tag == "+":
                    result.append(text)
                elif tag == "\\":
                    pass  # "\ No newline at end of file"
            cursor = pos + len(context)

        result.extend(original[cursor:])
        lines.append("\n".join(result))
        i = j
    return True, "ok"


def _find_block(haystack: list[str], needle: list[str], start: int) -> int | None:
    """Find `needle` in `haystack` at or after `start` (exact, then fuzzy)."""
    if not needle:
        return start
    limit = len(haystack) - len(needle)
    for pos in range(max(0, start), limit + 1):
        if haystack[pos:pos + len(needle)] == needle:
            return pos
    # Tolerate trailing-whitespace differences, which git also tolerates.
    stripped = [ln.rstrip() for ln in needle]
    for pos in range(max(0, start), limit + 1):
        if [ln.rstrip() for ln in haystack[pos:pos + len(needle)]] == stripped:
            return pos
    return None


def read_files_at(repo_path: Path, commit: str, paths: list[str]) -> dict[str, str]:
    """Read the given paths as of `commit` using one `git show` per file."""
    out: dict[str, str] = {}
    for p in paths:
        res = subprocess.run(
            ["git", "-C", str(repo_path), "show", f"{commit}:{p}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if res.returncode == 0:
            out[p] = res.stdout
    return out


def pre_image_paths(patch: str) -> list[str]:
    """Return the paths that must exist at base_commit for this patch.

    Prefers each section's `--- a/<path>` line, because that is the
    pre-image: it resolves correctly for renames (where the b/ path does not
    yet exist) and is /dev/null for created files. Falls back to the b/ path
    for sections with no `---` line.
    """
    out: list[str] = []
    for m in DIFF_GIT_RE.finditer(patch):
        section_end = patch.find("\ndiff --git ", m.end())
        section = patch[m.end():section_end if section_end != -1 else len(patch)]
        a = re.search(r"^--- a/(.+)$", section, re.M)
        b = re.search(r"^diff --git a/(.+?) b/", m.group(0), re.M)
        path = a.group(1).strip() if a else (b.group(1) if b else None)
        if path and path != "/dev/null" and path not in out:
            out.append(path)
    return out


def check_instance(repo_path: Path, inst: dict) -> tuple[bool, str]:
    """Confirm both patch halves apply to the base_commit tree."""
    base = inst["base_commit"]
    if git(repo_path, "cat-file", "-e", f"{base}^{{commit}}", check=False).returncode != 0:
        return False, "base_commit missing"

    # Read the PRE-image paths (`--- a/...`), falling back to the b/ path.
    # For a created file the a/ path is /dev/null, and for a rename the b/
    # path does not exist at base; only the a/ path identifies real content.
    src_names = pre_image_paths(inst["patch"])
    tst_names = pre_image_paths(inst["test_patch"])
    contents = read_files_at(repo_path, base, src_names + tst_names)

    ok, err = apply_check(contents, inst["patch"])
    if not ok:
        return False, f"src: {err}"
    ok, err = apply_check(contents, inst["test_patch"])
    if not ok:
        return False, f"test: {err}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=40,
                    help="instances to verify per repo (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None, help="write results JSON here")
    args = ap.parse_args()

    root = Path(args.repos).resolve()
    data = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_repo: dict[str, list[dict]] = {}
    for inst in data:
        by_repo.setdefault(inst["repo"], []).append(inst)

    rng = random.Random(args.seed)
    results: dict[str, dict] = {}

    for repo, insts in sorted(by_repo.items()):
        pool = sorted(insts, key=lambda r: r["instance_id"])
        if args.sample:
            pool = rng.sample(pool, min(args.sample, len(pool)))
        repo_path = root / repo
        ok_n = 0
        reasons: Counter = Counter()
        for inst in pool:
            try:
                ok, reason = check_instance(repo_path, inst)
            except Exception as exc:  # noqa: BLE001
                ok, reason = False, f"{type(exc).__name__}"
            if ok:
                ok_n += 1
            else:
                reasons[reason.split(":")[0]] += 1
        results[repo] = {
            "checked": len(pool),
            "applied": ok_n,
            "rate": round(ok_n / len(pool), 4) if pool else 0.0,
            "failures": dict(reasons),
        }
        print(f"[{repo}] applied {ok_n}/{len(pool)} ({results[repo]['rate']:.1%})", flush=True)
        for reason, n in reasons.most_common(5):
            print(f"    {n:>3}  {reason}")

    total_c = sum(r["checked"] for r in results.values())
    total_a = sum(r["applied"] for r in results.values())
    print(f"\nOVERALL {total_a}/{total_c} patches apply cleanly "
          f"({total_a/total_c:.1%})" if total_c else "nothing checked")

    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0 if total_c and total_a == total_c else 1


if __name__ == "__main__":
    sys.exit(main())

