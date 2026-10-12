import numpy as np

from lunar_trn.craters.detect import DetectorConfig, Detections, highpass, stable_regions


def _procedural(craters, sun_deg, size=256, seed=0, noise=0.02):
    """Crater-like blobs: the Sun-side wall is in shadow, the anti-Sun wall is bright."""
    rng = np.random.default_rng(seed)
    img = 0.5 + noise * rng.normal(size=(size, size))
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    s = np.array([np.cos(np.deg2rad(sun_deg)), np.sin(np.deg2rad(sun_deg))])  # towards the Sun, (col, row)
    for cx, cy, r in craters:
        dx, dy = xx - cx, yy - cy
        inside = np.hypot(dx, dy) <= r
        proj = (dx * s[0] + dy * s[1]) / r
        img[inside] = np.where(proj > 0.1, 0.03, 0.5 - 0.45 * proj)[inside]
    return np.clip(img, 0.0, 1.0)


CRATERS = [(60, 60, 8), (150, 70, 12), (200, 180, 16), (80, 190, 10), (130, 130, 6)]


def test_highpass_is_zero_mean_and_floored():
    hp = highpass(np.zeros((64, 64)), DetectorConfig())
    assert np.allclose(hp, 0.0)
    hp = highpass(_procedural(CRATERS, 30.0), DetectorConfig())
    assert abs(float(np.median(hp))) < 0.3 and np.isfinite(hp).all()


def test_stable_regions_find_shadows_and_lit_walls():
    cfg = DetectorConfig()
    hp = highpass(_procedural(CRATERS, 30.0), cfg)
    dark, bright = stable_regions(hp, -1, cfg), stable_regions(hp, +1, cfg)
    s = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))])
    for cx, cy, r in CRATERS:
        dd = np.hypot(*(dark.centroid - [cx, cy]).T)
        bd = np.hypot(*(bright.centroid - [cx, cy]).T)
        assert dd.min() < r and bd.min() < r
        k = int(np.argmin(dd))
        assert (dark.centroid[k] - [cx, cy]) @ s > 0  # shadow on the Sun side
    assert (dark.m0 >= cfg.min_area_px).all() and np.isfinite(dark.contrast).all()


def test_detections_container():
    d = Detections.from_ellipses([10.0, 20.0], [5.0, 6.0], [4.0, 8.0], [2.0, 8.0], [0.0, 0.3])
    assert len(d) == 2 and np.allclose(d.D_px, [2 * np.sqrt(8.0), 16.0]) and d.uv.shape == (2, 2)
    both = Detections.concat([d, Detections.empty(), d.subset([1])])
    assert len(both) == 3 and len(Detections.concat([])) == 0


from lunar_trn.camera import PinholeCamera
from lunar_trn.craters.detect import constant_sun_fn, detect_craters, estimate_illumination, pinhole_sun_fn


def _hits(det, craters, pixel_center=0.5):
    n = 0
    for cx, cy, r in craters:
        if len(det) == 0:
            continue
        d = np.hypot(det.u - pixel_center - cx, det.v - pixel_center - cy)
        k = int(np.argmin(d))
        n += int(d[k] <= 0.6 * r and 0.6 < det.D_px[k] / (2 * r) < 1.6)
    return n


def test_detects_procedural_craters_and_pairs_away_from_sun():
    det, h = detect_craters(_procedural(CRATERS, 30.0), constant_sun_fn(30.0))
    assert _hits(det, CRATERS) >= 4
    assert np.median(det.pair_dev_deg) <= 15.0
    assert abs((h["sun_img_angle_est_deg"] - 30.0 + 180.0) % 360.0 - 180.0) <= 15.0
    assert h["n_detections"] == len(det)


def test_detects_nothing_on_noise():
    rng = np.random.default_rng(5)
    det, _ = detect_craters(0.5 + 0.02 * rng.normal(size=(256, 256)), constant_sun_fn(30.0))
    assert len(det) <= 2


def test_detect_handles_black_nan_and_degenerate_sun():
    det, h = detect_craters(np.zeros((256, 256)), constant_sun_fn(30.0))
    assert len(det) == 0 and h["n_detections"] == 0
    img = _procedural(CRATERS, 30.0)
    img[100:140, 100:140] = np.nan
    det, _ = detect_craters(img, constant_sun_fn(30.0))
    assert np.isfinite(det.u).all()
    det, _ = detect_craters(_procedural(CRATERS, 30.0), lambda u, v: np.full((np.size(u), 2), np.nan))
    assert len(det) == 0


def test_cnav_illumination_estimate():
    rng = np.random.default_rng(2)
    ang = np.deg2rad(np.concatenate([200.0 + rng.normal(0, 8, 80), rng.uniform(0, 360, 20)]))
    mu, sig = estimate_illumination(ang)
    assert abs((np.degrees(mu) - 200.0 + 180.0) % 360.0 - 180.0) <= 5.0 and 0 < np.degrees(sig) < 30
    assert np.isnan(estimate_illumination(np.array([0.1]))[0])


def test_pinhole_sun_fn_matches_projected_sun_at_centre():
    f = 256 / np.tan(np.deg2rad(22.5))
    cam = PinholeCamera(np.array([[f, 0, 256.0], [0, f, 256.0], [0, 0, 1.0]]), 512, 512)
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # looking down -Z
    T[:3, 3] = [0.0, 0.0, 4000.0]
    sun = np.array([np.cos(np.deg2rad(20)) * np.cos(0.3), np.cos(np.deg2rad(20)) * np.sin(0.3), np.sin(np.deg2rad(20))])
    s = pinhole_sun_fn(cam, T, sun, 4000.0)(np.array([256.0]), np.array([256.0]))[0]
    sc = T[:3, :3].T @ sun
    expect = sc[:2] / np.linalg.norm(sc[:2])
    assert np.degrees(np.arccos(np.clip(s @ expect, -1, 1))) < 1.0


import pytest

from lunar_trn.craters.annotate import annotate_detections, catalog_recall
from lunar_trn.craters.catalog import extract_catalog
from lunar_trn.craters.synthetic import render_synthetic_scene


@pytest.fixture(scope="module")
def scene():
    return render_synthetic_scene()


@pytest.mark.slow
def test_detector_on_rendered_scene(scene):
    cat = extract_catalog(scene["dem"])
    cam, T = scene["cam"], scene["T"]
    fn = pinhole_sun_fn(cam, T, scene["sun"], 4000.0)
    det, _ = detect_craters(scene["image"], fn)
    assert len(det) >= 10
    ann = annotate_detections(det, cat, cam, T, fn, tol_rel=0.4, size_ratio_max=2.0)
    assert np.mean(ann["cat_index"] >= 0) >= 0.6
    rec = catalog_recall(det, cat, cam, T, scene["sun"], tol_rel=0.4, size_ratio_max=2.0)
    big = rec["D_px"] >= 12
    assert big.sum() >= 5 and rec["detected"][big].mean() >= 0.6
