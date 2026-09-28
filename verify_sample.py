"""Batch verification over a random sample; writes results to out/verify_sample.txt."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import verify_mined as v

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rs = [json.loads(l) for l in Path("out/mined.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by_repo: dict[str, list] = {}
for r in rs:
    by_repo.setdefault(r["repo"], []).append(r)

rng = random.Random(0)
out = []
tot_ok = tot = 0
for repo, insts in sorted(by_repo.items()):
    pool = rng.sample(sorted(insts, key=lambda x: x["instance_id"]), min(SAMPLE, len(insts)))
    ok_n = 0
    reasons: Counter = Counter()
    for inst in pool:
        try:
            ok, reason = v.check_instance(Path("repos") / repo, inst)
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, type(exc).__name__
        if ok:
            ok_n += 1
        else:
            reasons[reason.split(":")[0][:60]] += 1
    tot_ok += ok_n
    tot += len(pool)
    out.append(f"[{repo}] {ok_n}/{len(pool)} = {ok_n/len(pool):.1%}")
    for reason, n in reasons.most_common(6):
        out.append(f"      {n:>3}  {reason}")
out.append(f"OVERALL {tot_ok}/{tot} = {tot_ok/tot:.1%}" if tot else "none")
Path("out/verify_sample.txt").write_text("\n".join(out), encoding="utf-8")
