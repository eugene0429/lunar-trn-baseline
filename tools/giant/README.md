# GIANT SFN cross-implementation (V4) — out-of-process tools

NASA GIANT (https://github.com/nasa/giant, Apache-2.0 since v2.0) is used only as an external
reference implementation. It is **not** vendored; it is installed into its own virtualenv
because it needs Python >= 3.11 (this repository targets 3.10).

Setup (pin the release that was checked on 2026-10-10):

```bash
git clone --depth 1 --branch v2.0.0f https://github.com/nasa/giant.git ~/giant-src
uv venv --python 3.12 ~/giant-venv
uv pip install --python ~/giant-venv/bin/python "numpy==2.2.6" opencv-python-headless tqdm ~/giant-src
```

- `giant_shim.py`: import-time shims (no edits to GIANT): `typing.Self` alias and an empty
  stub for the optional RoMa matcher that GIANT imports unconditionally.
- `sfn_smoke.py`: synthetic 3 km descent smoke test of `SurfaceFeatureNavigation`, with two
  out-of-tree fixes for GIANT 2.0.0 (`compute_rays` integer bounds, `FeatureCatalog.compute_intersect`).
  Usage: `~/giant-venv/bin/python -I tools/giant/sfn_smoke.py tools/giant <sun_el_deg> <err_scale> <run_pnp 0|1> [check|fixjac]`
  (writes a cached truth image to `tools/giant/../work/`; run from a scratch copy if needed).

Known issues in GIANT 2.0.0 SFN: see `docs/validation/v4_giant_feasibility.md`.
