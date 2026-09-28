"""
Compute the RL reward signal by running the target tests in the sandbox.

This is the piece the build spec calls the reward signal: apply a candidate
patch to `base_commit` + `test_patch`, run pytest, and score 1 exactly when
the fail-to-pass and pass-to-pass conditions both hold.

DESIGN: two decoupled halves
----------------------------
1. `build_workload()` decides *what* to run and serialises each instance to
   a plain JSON job. It needs only the local clones and the mined data.
2. `run_job()` executes one job with `docker run --rm swebench-sandbox:latest`.

They are separated because this development machine has no Docker, so the
first half is runnable and testable here while the second half is ready to
execute wherever the image exists. That split is deliberate: it means the
job definitions can be verified now, and only the container plumbing is
untested rather than the whole pipeline.

Reward definition
-----------------
    reward = 1.0   fail-to-pass tests all pass AND pass-to-pass tests all still pass
    reward = 0.0   anything else
    partial credit (optional, --partial): fraction of fail-to-pass tests that
                   newly pass, to reduce reward sparsity at this data size

Pass-to-pass tests are the ones that already pass at base_commit+test_patch;
they are discovered by running the suite before applying the candidate patch,
so the reward cannot be gamed by simply breaking fewer tests.

Usage (host, no Docker needed):
    python reward_sandbox.py build --data out/mined.jsonl --out out/jobs

Usage (where swebench-sandbox:latest exists):
    python reward_sandbox.py run --jobs out/jobs --candidates cands.jsonl --out rewards.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

SANDBOX_IMAGE = "swebench-sandbox:latest"


# pytest exit code 5 means "no tests collected", which is a harness error
# rather than a genuine pass.
PYTEST_OK = 0
PYTEST_NO_TESTS = 5


# In-container scorer. Kept as a PLAIN string (not an f-string) so the
# embedded Python's braces need no escaping -- nesting code inside an
# f-string is what produced a syntax error here previously.
SCORER_SCRIPT = '''python - <<'PY'
import json, re

def parse(path):
    out = {}
    try:
        for line in open(path):
            m = re.match(r'^(PASSED|FAILED|ERROR)\\s+(\\S+)', line.strip())
            if m:
                out[m.group(2)] = m.group(1)
    except FileNotFoundError:
        pass
    return out

before = parse('/tmp/before_lines.txt')
after = parse('/tmp/after_lines.txt')

# A test that did not pass before the candidate fix must pass after it.
# A test that did pass before must still pass. Both conditions are required
# so a candidate cannot win by simply breaking fewer tests.
f2p = sorted(t for t, s in before.items() if s != 'PASSED')
p2p = sorted(t for t, s in before.items() if s == 'PASSED')
f2p_ok = [t for t in f2p if after.get(t) == 'PASSED']
p2p_broken = [t for t in p2p if after.get(t) != 'PASSED']

reward = 1.0 if (f2p and not p2p_broken and len(f2p_ok) == len(f2p)) else 0.0
partial = (len(f2p_ok) / len(f2p)) if f2p else 0.0

print('RESULT_JSON ' + json.dumps({
    'fail_to_pass': f2p,
    'pass_to_pass': p2p,
    'f2p_passed': len(f2p_ok),
    'f2p_total': len(f2p),
    'p2p_broken': p2p_broken,
    'reward': reward,
    'partial': partial,
}))
PY
'''


def _with_newline(text: str) -> str:
    """Ensure a patch body ends with a newline.

    A heredoc terminator must start at the beginning of its own line, so a
    patch lacking a trailing newline would swallow the terminator.
    """
    return text if text.endswith("\n") else text + "\n"


@dataclass
class Job:
    """One reward evaluation, fully specified and serialisable."""

    instance_id: str
    repo: str
    base_commit: str
    test_patch: str
    test_files: list[str] = field(default_factory=list)
    # Populated by the candidate being scored, not at build time.
    candidate_patch: str = ""


def build_workload(mined_path: Path, out_dir: Path, limit: int | None) -> int:
    """Serialise each mined instance into a runnable reward job."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in
               mined_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        records = records[:limit]

    jobs = [
        Job(
            instance_id=r["instance_id"],
            repo=r["repo"],
            base_commit=r["base_commit"],
            test_patch=r["test_patch"],
            test_files=r["test_files"],
        )
        for r in records
    ]
    jobs_path = out_dir / "jobs.jsonl"
    with jobs_path.open("w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")

    per_repo: dict[str, int] = {}
    for j in jobs:
        per_repo[j.repo] = per_repo.get(j.repo, 0) + 1
    print(f"built {len(jobs)} jobs -> {jobs_path}")
    for repo, n in sorted(per_repo.items()):
        print(f"  {repo:<10} {n}")
    return 0


def docker_script(job: Job, candidate_patch: str) -> str:
    """Build the in-container shell script that scores one job.

    Runs the suite twice so pass-to-pass tests are discovered from the
    actual environment rather than assumed:

      1. base + test_patch        -> tests passing BEFORE the fix (P2P set)
      2. base + test_patch + cand -> fail-to-pass must pass, P2P must hold

    The script prints a single JSON line prefixed with RESULT_JSON so the
    caller can parse it without depending on pytest's human output.
    """
    p2p_target = " ".join(job.test_files) if job.test_files else "."
    test_patch = _with_newline(job.test_patch)
    cand_patch = _with_newline(candidate_patch)
    return (
        "set -u\n"
        "cd /workspace\n"
        f"git checkout -q {job.base_commit}\n"
        "git apply - <<'TESTPATCH'\n"
        f"{test_patch}TESTPATCH\n"
        "\n"
        "# --- phase 1: which tests already pass with the test_patch alone? ---\n"
        f"python -m pytest {p2p_target} -q -rA --no-header 2>/dev/null "
        "| tail -n 400 > /tmp/before.txt || true\n"
        "grep -E '^(PASSED|FAILED|ERROR) ' /tmp/before.txt > /tmp/before_lines.txt || true\n"
        "\n"
        "# --- phase 2: apply the candidate fix and re-run ---\n"
        "git apply - <<'CANDPATCH'\n"
        f"{cand_patch}CANDPATCH\n"
        f"python -m pytest {p2p_target} -q -rA --no-header 2>/dev/null "
        "| tail -n 400 > /tmp/after.txt || true\n"
        "grep -E '^(PASSED|FAILED|ERROR) ' /tmp/after.txt > /tmp/after_lines.txt || true\n"
        "\n" + SCORER_SCRIPT
    )


def run_job(job: Job, candidate_patch: str, timeout: int, image: str) -> dict:
    """Execute one job in the sandbox and parse its reward JSON."""
    script = docker_script(job, candidate_patch)
    cmd = [
        "docker", "run", "--rm", "-i", "-w", "/workspace",
        image, "bash", "-lc", script,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"instance_id": job.instance_id, "reward": 0.0,
                "error": "timeout", "partial": 0.0}
    except FileNotFoundError:
        return {"instance_id": job.instance_id, "reward": 0.0,
                "error": "docker not available", "partial": 0.0}

    for line in reversed((res.stdout or "").splitlines()):
        if line.startswith("RESULT_JSON "):
            payload = json.loads(line[len("RESULT_JSON "):])
            payload["instance_id"] = job.instance_id
            return payload
    return {"instance_id": job.instance_id, "reward": 0.0,
            "error": (res.stderr or "no RESULT_JSON emitted").strip()[:300],
            "partial": 0.0}


def docker_available() -> tuple[bool, str]:
    """Check Docker and the sandbox image before starting a long run."""
    try:
        res = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "docker CLI not found"
    if res.returncode != 0:
        return False, "docker daemon not reachable"
    img = subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE],
                         capture_output=True, text=True, timeout=30)
    if img.returncode != 0:
        return False, f"image {SANDBOX_IMAGE} not present"
    return True, f"docker {res.stdout.strip()}"


def run_workload(jobs_path: Path, candidates_path: Path | None,
                 out_path: Path, timeout: int, image: str) -> int:
    """Score every job, optionally with candidate patches."""
    ok, msg = docker_available()
    if not ok:
        print(f"ERROR: cannot run sandbox: {msg}")
        print("This is expected on a host without Docker/swebench-sandbox.")
        print("The workload itself is still valid -- see 'build'.")
        return 2
    print(f"sandbox ready: {msg}")

    jobs = [json.loads(l) for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    candidates: dict[str, str] = {}
    if candidates_path and candidates_path.is_file():
        for l in candidates_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                candidates[r["instance_id"]] = r["patch"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, raw in enumerate(jobs, 1):
            job = Job(**raw)
            cand = candidates.get(job.instance_id, "")
            if not cand:
                continue
            result = run_job(job, cand, timeout, image)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += int(result.get("reward", 0.0) == 1.0)
            print(f"[{i}/{len(jobs)}] {job.instance_id} reward={result.get('reward')} "
                  f"{result.get('error', '')}", flush=True)

    print(f"\nscored {n_ok}/{len(jobs)} as reward=1.0 -> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="serialise mined instances into reward jobs")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, default=None)

    r = sub.add_parser("run", help="score jobs in the sandbox")
    r.add_argument("--jobs", required=True)
    r.add_argument("--candidates", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--image", default=SANDBOX_IMAGE)

    args = ap.parse_args()
    if args.cmd == "build":
        return build_workload(Path(args.data).resolve(), Path(args.out).resolve(), args.limit)
    return run_workload(
        Path(args.jobs).resolve(),
        Path(args.candidates).resolve() if args.candidates else None,
        Path(args.out).resolve(),
        args.timeout,
        args.image,
    )


if __name__ == "__main__":
    sys.exit(main())