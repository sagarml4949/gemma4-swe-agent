---
name: repo-navigation
description: >
  Strategies for using the code graph and embedding tools
  (get_code_neighbors, search_similar_code, get_code_subgraph) efficiently
  when localizing a bug in an unfamiliar repository, instead of defaulting
  to manual grep/read exploration.
---

# Repository navigation strategy

Manual exploration (reading files one by one, guessing at names) is the
most expensive way to find a bug in a large repo. Use it as a last resort,
not a first step.

## Order of operations

1. **Semantic search first.** `search_similar_code` on the problem
   statement's own wording (paraphrased into a few short queries) usually
   surfaces the right function even when you don't know its name.
2. **Expand along the graph, not the filesystem.** Once you have one
   plausible symbol, `get_code_neighbors` tells you what calls it and what
   it calls — this is almost always a better next step than opening
   neighboring files by guesswork.
3. **Use `get_code_subgraph` for multi-symbol changes.** If a fix plausibly
   touches more than one function (e.g. a shared helper and its two
   callers), pull the induced subgraph once rather than repeatedly calling
   `get_code_neighbors` on each symbol individually — it's cheaper and
   shows the interconnections directly.
4. **Read last.** Only `read_file` the specific symbols the graph/embedding
   tools pointed to, not entire modules.

## Common failure modes to avoid

- Treating the top embedding hit as certainly correct — always sanity check
  it against the actual reported symptom by reading the code.
- Re-running the same `search_similar_code` query with trivial rewordings
  expecting different results — vary the *content* of the query (different
  aspect of the behavior), not just its phrasing.
- Exploring outward indefinitely. If three hops of `get_code_neighbors`
  haven't produced a convincing fix site, stop and re-read the problem
  statement for a clue you missed rather than continuing to expand.
