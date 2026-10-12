# lunar-trn-baseline — Project Instructions

- Read `docs/handoff/` (newest first) before starting work. The kickoff handoff defines
  scope, decisions, the algorithm spec, datasets and acceptance tests.
- Method families follow the handoff §2.1 taxonomy (T-I, T-E, P-G, P-R). Classify a method
  by its comparison operation: if it extracts and describes primitives on both the image and
  the map side before matching, it is pattern matching (P). Otherwise it is template matching (T).
  Learning, rendering and Image-on-Map/Map-on-Image are attributes, not families.
- Scope boundary: this repo is algorithm and validation only. Do not import sim2moon,
  Isaac Sim or Unreal Engine code. The sim2moon integration point is the pure-Python
  `localize(...)` interface sketched in the handoff §8.
- Write all code comments, docstrings and log messages in English. Documents under
  `docs/` may be in Korean.
- Never commit datasets or large outputs (`data/`, `outputs/`, archives, EXR files).
- License hygiene:
  - Do not vendor GPL code. Run it out of process if it is needed.
  - Do not copy code from repositories that have no license file.
  - Do not use SuperGlue or SuperPoint weights (restrictive licenses).
- Every threshold that the source papers leave unpublished must be recorded with its
  rationale in `docs/decisions.md`.
- Log raw health signals, not only pass/fail verdicts. The downstream study builds ROCs
  from them.
- Commit or push only when the user asks.
