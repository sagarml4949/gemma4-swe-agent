"""
Interactive / offline mock execution loop for SWE Agent submission.

Simulates the runtime execution of `agent.yaml` without requiring:
1. Docker daemon or competition sandbox images
2. 31B GPU hardware or live cloud model API

Demonstrates the step-by-step ADK agent life-cycle:
- Resolves YAML includes (prompts/system.md, sub_agents/code_analyzer.yaml)
- Dispatches bug localization to sub-agent
- Reads files and simulates search tools
- Generates surgical edits
- Performs self-verification pytest loop
- Submits final patch
"""

from __future__ import annotations

import json
from pathlib import Path


def run_mock_simulation():
    submission_dir = Path(r"C:\Users\nerel\Downloads\submission")
    agent_yaml_path = submission_dir / "agent.yaml"
    val_traces_path = Path(r"C:\Users\nerel\Downloads\swe_data\out\traces\edit_val.jsonl")

    print("=" * 65)
    print("MOCK SWE AGENT RUNTIME SIMULATION")
    print("=" * 65)

    if not agent_yaml_path.is_file():
        print(f"Error: {agent_yaml_path} does not exist.")
        return

    print(f"1. Loading Agent Config from: {agent_yaml_path.name}")
    print("   Model: gemma-4-31b-it-qat-w4a16-ct")
    print("   Sub-Agent registered: code_analyzer (read-only)")
    print("   Tools available: run_command, submit_patch, edit_file, read_file, get_status...")

    if not val_traces_path.is_file():
        print(f"Trace file not found: {val_traces_path}")
        return

    with open(val_traces_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        task = json.loads(first_line)

    instance_id = task["instance_id"]
    repo = task["repo"]
    problem = task["problem_statement"]
    analyzer_report = task["analyzer_report"]
    tool_calls = task["tool_calls"]

    print("\n" + "=" * 65)
    print(f"TASK DISPATCH: {instance_id} [{repo}]")
    print(f"ISSUE DESCRIPTION: {problem}")
    print("=" * 65)

    print("\n[Turn 1] Root Agent -> Sub-Agent Delegation")
    print(">> Assistant calls tool: code_analyzer(problem_statement=...)")
    print("<< Sub-Agent 'code_analyzer' executes (read-only search over graph & AST):")
    for line in analyzer_report.splitlines():
        print(f"   | {line}")

    print("\n[Turn 2] Root Agent: Inspecting pre-edit code")
    first_path = tool_calls[0]["args"]["path"]
    print(f">> Assistant calls tool: read_file(path='{first_path}')")
    print(f"<< Tool output: <loaded pre-edit contents of {first_path}>")

    print("\n[Turn 3] Root Agent: Surgical code edit")
    for i, call in enumerate(tool_calls, 1):
        args = call["args"]
        print(f">> Assistant calls tool: {call['tool']}(path='{args['path']}')")
        old_preview = args['old_string'].strip().replace('\n', ' ')[:60]
        new_preview = args['new_string'].strip().replace('\n', ' ')[:60]
        print(f"   Replacing: {old_preview}...")
        print(f"   With:      {new_preview}...")
        print(f"<< Tool output: Applied 1 change(s) in {args['path']}.")

    print("\n[Turn 4] Root Agent: Self-verification via pytest")
    print(f">> Assistant calls tool: run_command(command='python -m pytest tests/ -q')")
    print("<< Tool output: ==================== 1 passed in 0.42s ====================")

    print("\n[Turn 5] Root Agent: Final Patch Submission")
    print(">> Assistant calls tool: submit_patch()")
    print("<< Tool output: Patch successfully recorded. Workspace clean.")
    print("=" * 65)
    print(f"RESULT FOR {instance_id}: RESOLVED (Exit code 0, 0 broken tests)")
    print("=" * 65)


if __name__ == "__main__":
    run_mock_simulation()
