"""
Build submission.zip from the submission root.

Follows the spec's deliverable contract: `agent.yaml` must sit at the zip
root, and `!include` paths must resolve relative to the file that
references them (so the directory layout is preserved verbatim in the
archive).

Dev-only tooling (validate_submission.py, this script) is excluded so the
archive matches the spec's file tree.

Usage:  python package_submission.py [submission_root] [output_zip]
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# Build tooling, not part of the submitted agent.
EXCLUDE_NAMES = {"package_submission.py", "validate_submission.py", "__pycache__"}
EXCLUDE_SUFFIXES = {".py", ".pyc", ".zip", ".log"}


def should_include(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDE_NAMES or part.endswith(".pyc") for part in rel.parts):
        return False
    return path.suffix.lower() not in EXCLUDE_SUFFIXES


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else root.parent / "submission.zip").resolve()

    members = [p for p in sorted(root.rglob("*")) if p.is_file() and should_include(p, root)]
    if not members:
        print(f"ERROR: nothing to package under {root}")
        return 1
    if not (root / "agent.yaml").is_file():
        print("ERROR: agent.yaml must exist at the submission root")
        return 1

    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            zf.write(p, arcname=str(p.relative_to(root)))

    print(f"Wrote {out} ({out.stat().st_size} bytes), {len(members)} files:")
    for p in members:
        print(f"  {p.relative_to(root)}")

    # Verify agent.yaml is at the archive root.
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    if "agent.yaml" not in names:
        print("ERROR: agent.yaml is not at the zip root")
        return 1
    bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
    if bad:
        print(f"ERROR: unsafe paths in archive: {bad}")
        return 1
    print("\nOK: agent.yaml at root, no path traversal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
