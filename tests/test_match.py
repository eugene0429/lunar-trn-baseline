import numpy as np

from lunar_trn import frames
from lunar_trn.craters.annotate import project_catalog
from lunar_trn.craters.catalog import CraterCatalog
from lunar_trn.craters.detect import Detections
from lunar_trn.craters.match import MatchConfig, assign, select_candidates, vote_positions
from lunar_trn.craters.synthetic import default_camera, flat_catalog, nadir_pose
from lunar_trn.validation.v1 import sun_from_az_el

CX, CY, ALT = -20000.0, 30000.0, 5000.0


def _world(seed=0):
    rng = np.random.default_rng(seed)
    cat = flat_catalog(rng, 600, (CX - 8000, CX + 8000, CY - 8000, CY + 8000), (60.0, 800.0))
    cam, T = default_camera(), nadir_pose(CX, CY, ALT)
    return cat, cam, T, sun_from_az_el(T[:3, 3], 135.0, 20.0), rng


def _true_detections(cat, cam, T, rng, noise=0.3, outliers=0.3, miss=0.2):
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


def _shifted(T, d_east, d_north):
    enu = frames.enu_basis(T[:3, 3])
    Tp = T.copy()
    Tp[:3, 3] = T[:3, 3] + enu[:, 0] * d_east + enu[:, 1] * d_north
    return Tp


def test_vote_finds_true_offset():
    cat, cam, T, sun, rng = _world()
    det, _ = _true_detections(cat, cam, T, rng)
    Tp = _shifted(T, 2000.0, -1500.0)
    cand = select_candidates(cat, cam, Tp, 1000.0, sun, (6.0, 128.0), MatchConfig())
    assert len(cand.catalog) > 50
    vr = vote_positions(det, cand, cam, Tp, 1000.0, MatchConfig())
    truth = frames.enu_basis(Tp[:3, 3]).T @ (T[:3, 3] - Tp[:3, 3])
    err = np.hypot(vr["offsets"][0, 0] - truth[0], vr["offsets"][0, 1] - truth[1])
    assert err <= 2 * vr["bin_m"], (err, vr["bin_m"])
    assert vr["votes"][0] > 1.5 * vr["votes"][1]


def test_assign_ignores_invalid_predictions():
    det_uv = np.array([[10.0, 10.0], [10.5, 10.0], [100.0, 100.0]])
    det_D = np.array([10.0, 10.0, 10.0])
    pred_uv = np.array([[10.2, 10.0], [300.0, 300.0], [np.nan, np.nan]])
    pred_D = np.array([10.0, 10.0, np.nan])
    i, j, d = assign(det_uv, det_D, pred_uv, pred_D, 4.0, 0.4, 1.6)
    assert len(i) == 1 and j[0] == 0 and i[0] in (0, 1) and d[0] < 1.0
    assert len(assign(det_uv, 3 * det_D, pred_uv, pred_D, 4.0, 0.4, 1.6)[0]) == 0  # size mismatch
    assert len(assign(det_uv[:0], det_D[:0], pred_uv, pred_D, 4.0, 0.4, 1.6)[0]) == 0


def test_select_candidates_empty_and_size_floor():
    cat, cam, T, sun, _ = _world()
    e = select_candidates(CraterCatalog.empty(), cam, T, 1000.0, sun, (6.0, 128.0), MatchConfig())
    assert len(e.catalog) == 0 and e.n_window == 0 and e.pred["u"].shape == (0,)
    c = select_candidates(cat, cam, T, 1000.0, sun, (6.0, 128.0), MatchConfig(catalog_D_min_m=300.0))
    assert len(c.catalog) > 0 and (c.catalog.D >= 300.0).all()
    vr = vote_positions(Detections.empty(), c, cam, T, 1000.0, MatchConfig())
    assert vr["offsets"].shape == (0, 2) and vr["n_pairs"] == 0
