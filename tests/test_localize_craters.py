import numpy as np
import pytest

from lunar_trn import frames
from lunar_trn.craters.annotate import project_catalog
from lunar_trn.craters.catalog import CraterCatalog, extract_catalog
from lunar_trn.craters.detect import Detections
from lunar_trn.craters.localize import CraterConfig, CraterMap, localize_craters
from lunar_trn.craters.synthetic import default_camera, flat_catalog, nadir_pose, render_synthetic_scene
from lunar_trn.experiments import score
from lunar_trn.validation.v1 import sun_from_az_el

CX, CY, ALT = -20000.0, 30000.0, 5000.0
BLACK = np.zeros((512, 512), np.float32)


def _prior(T, d_east, d_north, sigma_h=1000.0):
    enu = frames.enu_basis(T[:3, 3])
    Tp = T.copy()
    Tp[:3, 3] = T[:3, 3] + enu[:, 0] * d_east + enu[:, 1] * d_north
    cov = np.zeros((6, 6))
    cov[:3, :3] = enu @ np.diag([sigma_h**2, sigma_h**2, 40.0**2]) @ enu.T
    cov[3:, 3:] = np.eye(3) * np.deg2rad(0.01) ** 2
    return Tp, cov


def _world(seed=0):
    rng = np.random.default_rng(seed)
    cat = flat_catalog(rng, 600, (CX - 8000, CX + 8000, CY - 8000, CY + 8000), (60.0, 800.0))
    cam, T = default_camera(), nadir_pose(CX, CY, ALT)
    return cat, cam, T, sun_from_az_el(T[:3, 3], 135.0, 20.0), rng


def _true_detections(cat, cam, T, rng, noise=0.3, outliers=0.25, miss=0.2):
    p = project_catalog(cat, cam, T)
    sel = np.nonzero(p["in_image"] & (p["D_px"] >= 6) & (p["D_px"] <= 128))[0]
    sel = sel[rng.uniform(size=sel.size) > miss]
    n_out = int(outliers * sel.size)
    D_o = np.exp(rng.uniform(np.log(6.0), np.log(40.0), n_out))
    u = np.concatenate([p["u"][sel] + rng.normal(0, noise, sel.size), rng.uniform(0, 512, n_out)])
    v = np.concatenate([p["v"][sel] + rng.normal(0, noise, sel.size), rng.uniform(0, 512, n_out)])
    A = np.concatenate([p["A"][sel], D_o / 2])
    B = np.concatenate([p["B"][sel], D_o / 2])
    th = np.concatenate([p["theta"][sel], np.zeros(n_out)])
    return Detections.from_ellipses(u, v, A, B, th), sel.size


def test_injected_detections_recover_pose():
    cat, cam, T, sun, rng = _world()
    det, n_true = _true_detections(cat, cam, T, rng)
    Tp, cov = _prior(T, 2000.0, -1500.0)
    res = localize_craters(BLACK, cam.K, Tp, cov, CraterMap(cat, sun), seed=1, detections=det)
    gsd = ALT / cam.fx
    sc = score(res, T, map_px=gsd)
    assert res.family == "P-G" and res.status == "fix", res.health["pg"]
    assert sc["err_h_m"] <= gsd, sc
    assert res.health["pg"]["n_inliers"] >= 0.6 * n_true
    assert int(res.landmarks["inlier"].sum()) == res.health["pg"]["n_inliers"]
    strict = localize_craters(BLACK, cam.K, Tp, cov, CraterMap(cat, sun), CraterConfig(min_match_ratio=1.01),
                              seed=1, detections=det)
    assert strict.status == "no_fix" and strict.stage == "pose"  # gated, but the solution is kept
    assert np.allclose(strict.T_world_cam, res.T_world_cam)


def test_random_detections_do_not_fix():
    cat, cam, T, sun, rng = _world(1)
    n = 60
    D = np.exp(rng.uniform(np.log(8.0), np.log(60.0), n))
    det = Detections.from_ellipses(rng.uniform(0, 512, n), rng.uniform(0, 512, n), D / 2, D / 2, np.zeros(n))
    Tp, cov = _prior(T, 1000.0, 1000.0)
    res = localize_craters(BLACK, cam.K, Tp, cov, CraterMap(cat, sun), detections=det)
    assert res.status == "no_fix"
    assert res.health["pg"].get("n_inliers", 0) <= 4


def test_black_image_is_no_fix_at_detect():
    cat, cam, T, sun, _ = _world()
    Tp, cov = _prior(T, 500.0, 0.0)
    res = localize_craters(BLACK, cam.K, Tp, cov, CraterMap(cat, sun))
    assert res.status == "no_fix" and res.stage == "detect" and res.family == "P-G"
    assert "t_total_s" in res.health and "scene" in res.health


def test_empty_catalog_is_no_fix_at_match():
    cat, cam, T, sun, rng = _world()
    det, _ = _true_detections(cat, cam, T, rng)
    Tp, cov = _prior(T, 500.0, 0.0)
    res = localize_craters(BLACK, cam.K, Tp, cov, CraterMap(CraterCatalog.empty(), sun), detections=det)
    assert res.status == "no_fix" and res.stage == "match"


@pytest.mark.slow
def test_localize_craters_on_rendered_scene():
    s = render_synthetic_scene()
    cat = extract_catalog(s["dem"])
    cam, T = s["cam"], s["T"]
    Tp, cov = _prior(T, 700.0, -500.0)
    res = localize_craters(s["image"], cam.K, Tp, cov, CraterMap(cat, s["sun"]), seed=0)
    gsd = 4000.0 / cam.fx
    sc = score(res, T, map_px=gsd)
    assert res.status == "fix", res.health["pg"]
    assert sc["err_h_m"] <= 3 * gsd, sc
