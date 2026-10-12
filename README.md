# lunar-trn-baseline

A standalone, validated baseline for lunar **map-relative terrain-relative navigation (TRN)**.
It follows the Mars 2020 Lander Vision System design: coarse FFT/phase-correlation matching,
then fine NCC landmark matching, then PnP-RANSAC.

The baseline will later be the measurement instrument for a study of how far down, and under
which lighting and terrain conditions, vision-based TRN can be trusted. That study runs on
synthetic data rendered by sim2moon and lives outside this repository.

- Primary validation data: StereoLunar, a lunar south pole dataset at 3.5–30.5 km with
  ground-truth camera poses.
- This repository has no dependency on sim2moon, Isaac Sim or Unreal Engine.

Start here: [`docs/handoff/2026-10-10-trn-baseline-kickoff.md`](docs/handoff/2026-10-10-trn-baseline-kickoff.md)

## Status

- P0 environment and loaders, P1 frame alignment (V1 passed, see
  [`docs/validation/v1_frames.md`](docs/validation/v1_frames.md)), P2 map generator,
  P3 coarse, P4 fine + pose and P5 gates/health signals are implemented.
- V2/V3 results: [`docs/validation/`](docs/validation/). Thresholds and data conventions:
  [`docs/decisions.md`](docs/decisions.md). Integration API: [`docs/interface.md`](docs/interface.md).

## Setup

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -m "not slow"        # unit tests (no data needed)
.venv/bin/python -m pytest                      # + synthetic end-to-end test
```

## Data (git-ignored, under `data/`)

```bash
# LOLA DEMs: PDS LDEM_875S_5M (StereoLunar's source DEM) and the PGDA 87S 5 m mosaic
.venv/bin/python scripts/fetch_dem.py pds875s pgda87s
# StereoLunar nadir subset via HTTP range requests (every 10th geometry, 3 illuminations)
.venv/bin/python scripts/fetch_stereolunar.py --archive nadir --block-stride 60
```

## Experiments

```bash
# V1 frame/pose consistency
.venv/bin/python scripts/run_v1.py --data data/stereolunar/nadir \
    --dems data/dem/ldem_875s_5m.img data/dem/ldem_87s_5mpp.tif --out outputs/v1/nadir
# V2 (matched illumination) and V3 (cross illumination) localisation runs, then tables
.venv/bin/python scripts/run_v2.py --data data/stereolunar/nadir --dem data/dem/ldem_875s_5m.img \
    --out outputs/v2/nadir --split eval --image-illums all --trials 8
.venv/bin/python scripts/analyze_v2.py outputs/v2/nadir
```

## Layout

```
src/lunar_trn/   frames, dem (ray casting), render (Hapke), maps, rectify, matching,
                 coarse, fine, pose, health, localize, experiments, datasets/, validation/
scripts/         data fetchers and experiment runners
docs/            handoff notes, decisions, validation reports, interface
tests/           pytest suite
data/, outputs/  local datasets and run outputs (git-ignored)
```
