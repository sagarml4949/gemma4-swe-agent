# Attribution and upstream licensing

This repository contains **derived training data** mined from four public
open-source repositories. The source code of those projects is **not**
redistributed here (the `repos/` clones are excluded via `.gitignore` and are
re-created with the commands in `README.md`). What *is* committed are derived
artifacts — localized diff hunks, symbol names, and problem-statement metadata —
produced from the public git history of those projects.

## Upstream projects

| Repository | Upstream URL | License |
|---|---|---|
| `fastapi/fastapi` | https://github.com/fastapi/fastapi | MIT |
| `Textualize/rich` | https://github.com/Textualize/rich | MIT |
| `psf/requests` | https://github.com/psf/requests | Apache-2.0 |
| `encode/httpx` | https://github.com/encode/httpx | BSD-3-Clause |

Each derived record in `out/mined.jsonl`, `out/traces/*.jsonl`, and
`out/sft/*.jsonl` carries `repo` and `base_commit` fields identifying the exact
upstream origin, so any record can be traced back to the commit it came from.

## Redistribution status

The derived data consists of short code excerpts (individual hunks) and commit
subjects from MIT, BSD-3-Clause, and Apache-2.0 licensed projects. All three
licenses permit redistribution of derived source excerpts provided copyright
notices and license terms are preserved. Losing the original notices is the
practical risk here, which is why:

- every record retains its upstream `repo` + `base_commit`,
- the table above records the governing license for each project,
- and the full upstream license texts travel with the clones, which the
  reproduction commands in `README.md` fetch directly.

If you intend to redistribute the mined corpus further or use it commercially,
review the four upstream licenses first and retain this file alongside the data.

## Competition external-data rule

Per the competition's Competition-Specific Rules 2.6 (External Data), any
externally sourced data must be public and equally available to all
participants. Publishing this repository — including the mined corpus and the
scripts that regenerate it byte-for-byte — is intended to satisfy that
requirement for the paper track.
