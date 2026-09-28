"""
Phase 1 validation harness.

Checks the submission scaffold against the packaging checklist from the
build spec, without needing the competition harness itself:

  1. Every YAML file parses.
  2. Every `!include <path>` resolves, relative to the file that
     references it, and does not escape the submission root.
  3. Every tool named in a prompt is actually declared in that agent's
     `tools:` list.
  4. The sub-agent is wired as an `agent_tool` of the root agent.
  5. eval_config.yaml caps per-task time well below the naive even split.

Usage:  python validate_submission.py [submission_root]
Exit code 0 = all checks pass, 1 = at least one failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(".").resolve()

# Path of the submission root is treated as the sandbox boundary.
class IncludeLoader(yaml.SafeLoader):
    """SafeLoader that resolves !include relative to the referencing file."""


def load_yaml(path: Path):
    """Load a YAML file with !include support.

    The loader is constructed from an open file handle so that
    ``loader.name`` is the real path, which is what !include resolution
    uses as its base directory. Passing a plain string to ``yaml.load``
    would leave ``loader.name`` as ``<unicode string>`` and break
    relative-path resolution.
    """
    with path.open("r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=IncludeLoader)


def _include_reader(loader: yaml.SafeLoader, node: yaml.Node):
    """Resolve !include relative to the file that references it.

    Non-YAML targets (e.g. prompts/*.md) are returned as raw text, since
    those are instruction bodies rather than config fragments.
    """
    raw = loader.construct_scalar(node)
    base = Path(loader.name).resolve().parent
    target = (base / raw).resolve()
    if not str(target).startswith(str(ROOT)):
        raise ValueError(f"!include escapes submission root: {raw} (from {loader.name})")
    if not target.is_file():
        raise FileNotFoundError(f"!include target does not exist: {raw} (from {loader.name})")
    if target.suffix.lower() not in (".yaml", ".yml"):
        return target.read_text(encoding="utf-8")
    # Nested includes resolve relative to this file, not the process CWD.
    return load_yaml(target)


IncludeLoader.add_constructor("!include", _include_reader)


def tool_names(text: str) -> set[str]:
    """Extract backticked identifiers that look like tool names."""
    return set(re.findall(r"`([a-z_][a-z0-9_]*)`", text))


RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    RESULTS.append((ok, msg))
    print(("  PASS  " if ok else "  FAIL  ") + msg)


KNOWN_TOOLS = {
    "run_command", "submit_patch", "get_status", "read_file", "edit_file",
    "write_file", "get_code_neighbors", "search_similar_code",
    "get_code_subgraph", "load_skill_resource", "code_analyzer",
}


def main() -> int:
    global ROOT
    ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(f"Validating submission root: {ROOT}\n")

    # ---- 1. YAML parses + includes resolve ------------------------------
    print("[1] YAML parse + !include resolution")
    yamls = sorted(ROOT.rglob("*.yaml")) + sorted(ROOT.rglob("*.yml"))
    parsed: dict[Path, object] = {}
    for y in yamls:
        try:
            parsed[y] = load_yaml(y)
            check(True, f"parsed {y.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001
            check(False, f"parsed {y.relative_to(ROOT)} -> {type(exc).__name__}: {exc}")

    if not parsed:
        print("\nNo YAML parsed; aborting.")
        return 1

    agent = next((d for p, d in parsed.items() if p.name == "agent.yaml"), None)
    sub = next((d for p, d in parsed.items() if p.name == "code_analyzer.yaml"), None)

    # ---- 2. Root agent essentials ---------------------------------------
    print("\n[2] Root agent essentials")
    if agent:
        check(agent.get("model") == "gemma-4-31b-it-qat-w4a16-ct",
              f"root model is gemma-4-31b-it-qat-w4a16-ct (got {agent.get('model')!r})")
        check("instruction" in agent, "root has an instruction")
        check("generation_config" in agent, "root has a generation_config")
        check(bool(agent.get("tools")), "root declares a non-empty tools list")
    else:
        check(False, "agent.yaml found at root")

    # ---- 3. Sub-agent wiring --------------------------------------------
    print("\n[3] Sub-agent wiring")
    if agent and sub:
        entries = agent.get("sub_agents") or []
        wired = [e for e in entries if isinstance(e, dict) and "agent_tool" in e]
        check(len(wired) == 1, f"root has exactly one sub_agents agent_tool entry (got {len(wired)})")
        if wired:
            e = wired[0]
            target = (ROOT / e["agent_tool"]).resolve()
            check(target == (ROOT / "sub_agents" / "code_analyzer.yaml").resolve(),
                  f"agent_tool points at sub_agents/code_analyzer.yaml (got {e['agent_tool']})")
            check(e.get("name") == "code_analyzer",
                  f"sub-agent exposed as 'code_analyzer' (got {e.get('name')!r})")
            check(bool(e.get("description")), "sub-agent entry has a description")

    # ---- 4. Sub-agent is read-only --------------------------------------
    print("\n[4] Sub-agent is read-only")
    if sub:
        st = set(sub.get("tools") or [])
        forbidden = {"edit_file", "write_file", "run_command", "submit_patch"}
        check(not (st & forbidden),
              f"code_analyzer has no mutating tools (found {sorted(st & forbidden)})")
        expected = {"read_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph"}
        check(expected <= st, f"code_analyzer has all 4 read-only tools (missing {sorted(expected - st)})")

    # ---- 5. Prompt/tool agreement ---------------------------------------
    print("\n[5] Every tool named in a prompt is declared for that agent")
    if agent:
        p = ROOT / "prompts" / "system.md"
        sys_txt = p.read_text(encoding="utf-8") if p.is_file() else ""
        mentioned = tool_names(sys_txt) & KNOWN_TOOLS
        declared = set(agent.get("tools") or [])
        # A sub-agent wired as `agent_tool` is callable by name, so it counts
        # as declared even though it is not in the `tools:` list.
        for e in agent.get("sub_agents") or []:
            if isinstance(e, dict) and e.get("name"):
                declared.add(e["name"])
        missing = mentioned - declared
        check(not missing, f"system.md references only declared tools (undeclared: {sorted(missing)})")
        print(f"         system.md references: {sorted(mentioned)}")

    if sub:
        p = ROOT / "prompts" / "analyzer.md"
        an_txt = p.read_text(encoding="utf-8") if p.is_file() else ""
        mentioned = tool_names(an_txt) & KNOWN_TOOLS
        missing = mentioned - set(sub.get("tools") or [])
        check(not missing, f"analyzer.md references only declared tools (undeclared: {sorted(missing)})")
        print(f"         analyzer.md references: {sorted(mentioned)}")

    # ---- 6. eval_config budget -------------------------------------------
    print("\n[6] eval_config.yaml per-task cap")
    ec_path = ROOT / "eval_config.yaml"
    if ec_path.is_file():
        ec = yaml.safe_load(ec_path.read_text(encoding="utf-8")) or {}
        cap = ec.get("per_task_time_limit_seconds")
        if isinstance(cap, (int, float)) and not isinstance(cap, bool):
            check(0 < cap <= 900, f"per-task cap {cap}s is <= 900s (15 min)")
        else:
            check(False, f"per_task_time_limit_seconds present and sane (got {cap!r})")
    else:
        check(False, "eval_config.yaml exists")

    # ---- 7. Skill resource ----------------------------------------------
    print("\n[7] Skill resource present")
    sk = ROOT / "skills" / "repo_navigation" / "SKILL.md"
    check(sk.is_file(), "skills/repo_navigation/SKILL.md exists")
    if sk.is_file():
        head = sk.read_text(encoding="utf-8")[:400]
        check("name:" in head and "description:" in head,
              "SKILL.md has frontmatter name+description")

    failed = [m for ok, m in RESULTS if not ok]
    print(f"\n{'='*60}\n{len(RESULTS)-len(failed)}/{len(RESULTS)} checks passed")
    for m in failed:
        print(f"  FAILED: {m}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

