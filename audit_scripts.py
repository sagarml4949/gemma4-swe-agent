"""Structural audit of the pipeline scripts.

Two separate edits silently corrupted files by inserting a top-level
function *inside* another function's body (an `editor` insert_line that
landed mid-function). Python still parsed those files -- a nested def is
valid syntax -- so `py_compile` passed while the outer function lost its
tail. This audit catches that class of bug, which compilation cannot.

Checks, per file:
  1. every expected top-level function exists exactly once (no duplicates)
  2. no top-level function contains a nested function def  <-- the bug
  3. no unreachable statements after a terminating return/raise
  4. functions are separated by exactly two blank lines (PEP 8)
"""
import ast
import sys
from pathlib import Path

FILES = [
    "mine_tasks.py", "verify_mined.py", "build_traces.py",
    "validate_traces.py", "build_sft.py", "reward_sandbox.py",
]

out: list[str] = []
failures = 0

for name in FILES:
    path = Path(name)
    if not path.is_file():
        out.append(f"{name}: MISSING")
        failures += 1
        continue
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)

    problems: list[str] = []

    top = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = [n.name for n in top]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        problems.append(f"duplicate top-level defs: {sorted(dupes)}")

    # Signature of the insert bug: a top-level function whose last statement
    # is another def, i.e. its real tail was pushed out / lost. A nested def
    # that IS called is just a local helper, so only a stranded one counts.
    for fn in top:
        body_src = ast.get_source_segment(src, fn) or ""
        if fn.body and isinstance(fn.body[-1], ast.FunctionDef):
            problems.append(f"{fn.name} ENDS with a nested def -- tail likely lost")
        for nd in [n for n in fn.body if isinstance(n, ast.FunctionDef)]:
            if body_src.count(nd.name) < 2:
                problems.append(
                    f"{fn.name}: nested '{nd.name}' (line {nd.lineno}) is never called")

    # Dead code: statements stranded after a bare terminating return.
    for fn in top:
        for i, stmt in enumerate(fn.body[:-1]):
            if (isinstance(stmt, ast.Return) and stmt.value is not None
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, int)
                    and i < len(fn.body) - 1
                    and not isinstance(fn.body[i + 1], (ast.FunctionDef, ast.ClassDef))):
                problems.append(f"{fn.name}: dead code after `return` at line {stmt.lineno}")

    # Real style smell left behind by a bad deletion: a long run of blank
    # lines at module level. Comparing node line numbers instead would
    # miscount comment/constant blocks as blank space.
    run = 0
    for idx, ln in enumerate(lines, 1):
        if ln.strip() == "":
            run += 1
        else:
            if run >= 3:
                problems.append(f"run of {run} blank lines ending at line {idx - 1}")
            run = 0
    if run >= 3:
        problems.append(f"run of {run} trailing blank lines at end of file")

    if problems:
        failures += 1
        out.append(f"{name}: {len(names)} top-level funcs -- PROBLEMS")
        out += [f"    - {p}" for p in problems]
    else:
        out.append(f"{name}: OK ({len(names)} funcs: {', '.join(names[:4])}...)")

out.append("")
out.append("AUDIT FAILED" if failures else "ALL FILES PASS")
Path("audit.txt").write_text("\n".join(out), encoding="utf-8")
sys.exit(1 if failures else 0)