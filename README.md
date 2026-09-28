# Gemma-4 Autonomous SWE Agent — post-training pipeline

Entry for the Gemma 4 Developer Agent Competition: post-train
`gemma-4-31b-it-qat-w4a16-ct` into an autonomous software-engineering agent that
reads a bug report, localizes the fix site, edits the minimal correct code,
verifies itself against the test suite, and submits a patch.

This repository contains everything behind that submission — the agent package,
the training-data pipeline, and the LoRA recipes.

## Contents

| Path | What it is |
|---|---|
| `submission/` | The competition deliverable: `agent.yaml` plus prompts, the read-only sub-agent, sampling config, skill, and `eval_config.yaml`. |
| `out/mined.jsonl` | 1522 mined `(problem_statement, patch, test_patch)` triples from four public repositories. |
| `out/traces/` | Localization, edit, and verification traces derived from those triples. |
| `out/sft/` | Multi-turn SFT transcripts for the two LoRA adapters (train + held-out validation). |
| `out/jobs/` | 1522 serialized sandbox jobs for the pass/fail reward runner. |
| `recipes/` | PEFT/TRL, Unsloth, and Axolotl training configurations for both adapters. |
| `*.py` | The pipeline: mining → verification → trace building → SFT → reward. |

## Verified status

Everything below was measured against real repository history or a real
re-execution, not estimated.

| Check | Scope | Result |
|---|---|---|
| Submission package | all files | **19/19** checks pass (`submission/validate_submission.py`) |
| Mined instances | four repos | **1522** unique (rich 475, httpx 425, requests 316, fastapi 306) |
| Mining determinism | two independent runs | identical output, SHA-256 `40B3B94F…1F981561` |
| Patch application | 60 per repo, random | **240/240 = 100%** |
| Edit-trace reconstruction | all 1498 traces | **2986/2987 = 100.0%** byte-exact vs the real post-commit file |
| SFT transcript integrity | all 2866 rows | prompt alignment + tool-turn integrity pass, **0** train/val overlap |
| Script structural audit | 6 pipeline scripts | **ALL FILES PASS** |

## Not yet done

Stated plainly so the repository is not mistaken for a trained system:

- **The LoRA adapters are not trained.** Training a 31B model needs
  multi-GPU hardware that was unavailable; `recipes/` is the ready-to-run
  configuration, not a result. No `adapters/` directory is shipped, and the
  submission therefore runs on the base model.
- **The reward signal is defined but not executed.** Running
  fail-to-pass/pass-to-pass needs the pinned competition sandbox image, and
  Docker was unavailable in the development environment. `test_patch` is mined
  and its clean application is verified; the pytest run itself is pending.
- **Patch application is verified as "applies cleanly", which is not the same
  as "makes the target tests pass."** The latter is the reward check that needs
  the sandbox.
- **Verification tool output in the SFT transcripts is idealized**
  (`VERIFY_OK_OBSERVATION = "Tests passed."`). The gold patch is proved to
  reconstruct the post-commit bytes exactly, but the pytest runner did not
  execute locally. Replace with captured stdout once the sandbox runner runs.
- **The transcript wire format is an assumption.** The competition harness
  README was not available in the development workspace, so transcripts use the
  standard OpenAI-style `messages` + `tool_calls` schema. All rendering is
  isolated in `build_sft.py::render_example`, so adapting it is a
  single-function change.

See `ATTRIBUTION.md` for upstream licensing of the mined data.

---

## Phase 2 — Training data construction

Turns public repository history into supervised traces and an RL reward
signal for post-training a SWE agent, per the competition build spec.

## What this produces

| Artifact | Contents |
|---|---|
| `out/mined.jsonl` | Mined `(problem_statement, patch, test_patch)` triples |
| `out/traces/localization_{train,val}.jsonl` | Problem statement → gold symbol report (trains `tool_lora`) |
| `out/traces/edit_{train,val}.jsonl` | Problem statement + analyzer report → `edit_file`/`write_file` calls (trains `main_lora`) |
| `out/traces/verification_{train,val}.jsonl` | The `run_command` that exercises `test_patch` + expected fail→pass |
| `out/verify_sample.txt` | Patch-application rate per repo |
| `out/vt_log.txt` | Edit-trace reconstruction rate |

## Data provenance

All data comes from the four public repositories named in the spec —
`fastapi/fastapi`, `Textualize/rich`, `psf/requests`, `encode/httpx` —
cloned at full history. No private or competition data is used, and nothing
here reads the hidden test set. Mining criteria, applied to every commit:

1. The commit jointly modifies real Python source **and** a `test_*.py` file.
2. Any type of linkable issue reference is captured (`#N`); GitHub REST is
   not used (unauthenticated limit is 60 req/hr, too few for four repos).
3. The problem statement is the commit/PR subject — the closest
   always-available proxy for the linked issue title.
4. Mechanical churn is rejected: docs, configs, lockfiles, vendored code,
   and hunks whose added lines are all comments/imports/blank.
5. Caps: ≤10 source files, ≤10 test files, patch ≤20 KB.

## Why it mines from local clones

Walking four repositories via the GitHub API would exhaust the
unauthenticated quota immediately. Two other performance decisions matter:

- **One `git log -p` per repository**, not one `git diff` per commit.
  Per-commit subprocesses made a full crawl take hours on Windows; the
  single-pass version takes ~1 s per repo (httpx: 1482 commits in 0.9 s).
- **One persistent `git cat-file --batch` process per repo** for reading
  pre-edit file contents. One `git show` per file measured 56 s CPU over
  473 s wall for ~1500 instances; the batch reader removes that overhead.

## Verification

Two independent checks, both re-runnable:

- `verify_mined.py` — replays each patch's hunks against the real
  `base_commit` file contents in memory (a `git worktree` per instance was
  too slow on Windows). Handles created files (`new file mode`) and
  renames (`--- a/` pre-image path).
- `validate_traces.py` — replays each edit trace against the real
  pre-edit content and compares the result to the actual post-commit
  content, proving the trace reproduces the commit byte-for-byte.

## Verified results

Measured on real repository history, not estimated:

| Check | Scope | Result |
|---|---|---|
| Mined instances | all four repos | **1522** unique (rich 475, httpx 425, requests 316, fastapi 306) |
| Patch application | 60 per repo, random | **240/240 = 100%** both patch halves apply at `base_commit` |
| Edit-trace reconstruction | all 1498 traces | **2986/2987 = 100.0%** byte-exact vs the real post-commit file |

Yield: 1390 localization traces (91.3% of instances), 1498 edit traces
(98.4%), 1522 verification traces. Held out 278 instances (20%,
stratified by repo) that are never trained on.

Skips are explicit and counted: 132 localization traces had no
recoverable gold symbol, and 24 edit traces were file renames or
deletions, which `edit_file`/`write_file` cannot express.

### Bugs these checks caught

Verification paid for itself; each of these was found, fixed, and re-measured:

1. **Created files rejected** — requiring every patched path to exist at
   `base_commit` wrongly failed every commit that adds a file or test.
   fastapi measured 46.7%; fixing this took it to 100% (overall 80.8% → 99.6%).
2. **Renames rejected** — the post-image path of a rename does not exist
   at base. Reading the `--- a/` pre-image path fixed the last failure.
3. **Multi-hunk edit traces wrong** — the post-image was indexed without
   the cumulative line delta of earlier hunks, so any multi-hunk file
   reconstructed incorrectly (57.6%). This was a genuine defect that
   direct inspection would have missed; only replaying caught it.
4. **Ambiguous `old_string`** — hunks whose context repeats would teach
   non-deterministic edits; context is now expanded until unique, or the
   instance is skipped.
5. **Created files rendered as `edit_file`** — now a full-content
   `write_file`, matching what the system prompt tells the agent to do.

## Known limitations

- The reward signal is defined but **not yet executed**: running
  fail-to-pass/pass-to-pass requires the pinned competition sandbox image,
  which is unavailable in the development environment. `test_patch` is
  mined and its application is verified; only the pytest run is pending.
- `patch`/`test_patch` application is verified as *applies cleanly*, which
  is not the same as *makes the target tests pass*. The latter is the
  reward check that needs the sandbox.
- 24 of 1522 instances (1.6%) yield no edit trace because the change is a
  file rename or deletion, which `edit_file`/`write_file` cannot express.
  These are skipped explicitly rather than mis-rendered.
- **Verification tool content is idealized**: in `build_coder_example`,
  the verification turn returns `"Tests passed."` (`VERIFY_OK_OBSERVATION`)
  as the tool output. The fix is mathematically proved to reconstruct the
  gold post-commit bytes, but the pytest runner did not execute locally.
  Replace with captured test stdout once the sandbox runner executes.
- **Wire format is an assumption**: competition harness README was not
  provided in the local workspace; transcripts use the standard OpenAI-style
  `messages` + `tool_calls` schema. All transcript rendering is isolated in
  `build_sft.py::render_example` so reformatting is a single-function change.

## Pipeline

```bash
python mine_tasks.py       --repos repos --out out/mined.jsonl
python verify_mined.py     --repos repos --data out/mined.jsonl --sample 60
python build_traces.py     --repos repos --data out/mined.jsonl --out out/traces
python validate_traces.py  --repos repos --traces out/traces --mined out/mined.jsonl
python verify_sample.py 60   # batch patch-application check, writes out/verify_sample.txt
python build_sft.py        --repos repos --traces out/traces --prompts ../submission --out out/sft --mined out/mined.jsonl
python check_sft.py        # verifies system prompt alignment & tool-turn integrity across all 2866 rows
python reward_sandbox.py build --data out/mined.jsonl --out out/jobs
# python reward_sandbox.py run   --jobs out/jobs/jobs.jsonl --candidates cands.jsonl --out rewards.jsonl  # requires Docker
```

## Reward design (for the RL stage)

`reward = 1` iff, starting from `base_commit`, applying a candidate patch
makes the fail-to-pass tests pass **and** keeps pass-to-pass tests passing;
`0` otherwise. Optional partial credit for "some previously failing tests
now pass" reduces reward sparsity at this dataset size. Candidates are
sampled only against these training instances — never the hidden test set.
The 20% held-out slice (`*_val.jsonl`, stratified by repo) is never trained
on, so internal scores stay honest.