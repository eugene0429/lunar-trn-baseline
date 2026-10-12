import numpy as np

from lunar_trn import frames
from lunar_trn.craters.catalog import CraterCatalog, make_catalog, rim_frame


def _tiny_catalog(n=5):
    x = np.linspace(-1000.0, 1000.0, n)
    y = np.full(n, 500.0)
    X = frames.stereo_to_body(x, y, np.zeros(n))
    nrm = X / np.linalg.norm(X, axis=1, keepdims=True)
    e1, e2 = rim_frame(x, y, nrm)
    return make_catalog(X, nrm, e1, e2, np.full(n, 60.0), np.full(n, 50.0), np.zeros(n),
                        d_over_D=np.full(n, 0.15))


def test_rim_frame_is_orthonormal_and_polar_safe():
    x = np.array([0.0, 1000.0, -50000.0])
    y = np.array([0.0, -2000.0, 30000.0])
    X = frames.stereo_to_body(x, y, np.zeros(3))
    n = X / np.linalg.norm(X, axis=1, keepdims=True)
    e1, e2 = rim_frame(x, y, n)
    for v in (e1, e2):
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ij,ij->i", e1, n), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(e1, e2), n, atol=1e-12)


def test_make_catalog_fills_stereo_and_defaults():
    cat = _tiny_catalog()
    assert len(cat) == 5
    np.testing.assert_allclose(cat.x, np.linspace(-1000, 1000, 5), atol=1e-6)
    np.testing.assert_allclose(cat.D, 2 * np.sqrt(60 * 50.0))
    assert np.isnan(cat.depth).all() and (cat.id == np.arange(5)).all()


def test_save_load_round_trip(tmp_path):
    cat = _tiny_catalog()
    cat.meta = {"dem": "test", "pixel": 5.0}
    p = tmp_path / "cat.npz"
    cat.save(p)
    back = CraterCatalog.load(p)
    assert back.meta == cat.meta
    for k in ("X", "e1", "a", "psi", "x", "d_over_D", "id"):
        np.testing.assert_array_equal(getattr(back, k), getattr(cat, k))


def test_query_subset_concat_and_empty():
    cat = _tiny_catalog()
    sub, idx = cat.query_box(-600, 600, 0, 1000, return_index=True)
    assert (idx == np.array([1, 2, 3])).all() and len(sub) == 3
    both = CraterCatalog.concat([sub, cat.subset([0])], meta={"k": 1})
    assert len(both) == 4 and both.meta == {"k": 1}
    e = CraterCatalog.empty()
    assert len(e) == 0 and len(e.query_box(-1, 1, -1, 1)) == 0 and e.D.shape == (0,)
    assert len(CraterCatalog.concat([])) == 0


from lunar_trn.craters.catalog import CatalogConfig, extract_catalog
from lunar_trn.craters.synthetic import synthetic_crater_dem
from lunar_trn.dem import Dem


def _match_truth(cat, truth):
    rows = []
    for c in truth:
        if len(cat) == 0:
            rows.append((False, np.nan, np.nan, np.nan, 0))
            continue
        d = np.hypot(cat.x - c["x"], cat.y - c["y"])
        near = int(np.sum(d < 0.3 * c["D"]))
        k = int(np.argmin(d))
        ok = d[k] < 0.25 * c["D"] and abs(cat.D[k] / c["D"] - 1) < 0.3
        rows.append((ok, d[k] / c["D"], cat.D[k] / c["D"] - 1, cat.d_over_D[k] / c["dD"] - 1, near))
    return rows


def test_extract_recovers_planted_craters():
    dem, truth = synthetic_crater_dem(n=1600, n_craters=40, D_range=(60.0, 1500.0), seed=1)
    cat = extract_catalog(dem)
    rows = _match_truth(cat, truth)
    found = np.array([r[0] for r in rows])
    assert found.mean() >= 0.95, found.mean()
    cerr = np.array([r[1] for r in rows])[found]
    derr = np.array([r[2] for r in rows])[found]
    dderr = np.array([r[3] for r in rows])[found]
    assert np.percentile(cerr, 95) <= 0.05
    assert np.percentile(np.abs(derr), 95) <= 0.10
    assert abs(np.median(dderr)) <= 0.30
    assert max(r[4] for r in rows) <= 1  # de-duplicated across pyramid levels
    tx, ty, tD = (np.array([c[k] for c in truth]) for k in ("x", "y", "D"))
    spurious = [i for i in range(len(cat)) if np.min(np.hypot(tx - cat.x[i], ty - cat.y[i]) / tD) > 0.5]
    assert len(spurious) <= 0.1 * len(truth)
    assert np.isfinite(cat.X).all() and (np.einsum("ij,ij->i", cat.normal, cat.X) > 0).all()


def test_extract_finds_nothing_on_noise():
    rng = np.random.default_rng(3)
    n, px = 1200, 5.0
    H = 0.5 * rng.normal(size=(n, n)) + 0.01 * np.arange(n)[None, :] * px
    dem = Dem(H.astype(np.float32), -20000.0 - (n - 1) / 2 * px, 30000.0 + (n - 1) / 2 * px, px)
    assert len(extract_catalog(dem)) <= 2


def test_extract_tolerates_nodata():
    dem, _ = synthetic_crater_dem(n=1200, n_craters=15, D_range=(60.0, 600.0), seed=2)
    dem.heights[100:300, 100:300] = np.nan
    cat = extract_catalog(dem)
    assert len(cat) > 0
    for k in ("X", "a", "b", "psi", "depth", "d_over_D", "completeness"):
        assert np.isfinite(getattr(cat, k)).all(), k


def test_extract_core_is_half_open():
    dem, truth = synthetic_crater_dem(n=1200, n_craters=15, D_range=(60.0, 600.0), seed=4)
    full = extract_catalog(dem)
    xm = float(np.median(full.x))
    left = extract_catalog(dem, core=(-1e9, xm, -1e9, 1e9))
    right = extract_catalog(dem, core=(xm, 1e9, -1e9, 1e9))
    assert len(left) + len(right) == len(full)
    assert (left.x < xm).all() and (right.x >= xm).all()


from lunar_trn.craters.catalog import tile_cores


def test_tile_cores_partition_extent():
    ext = (-1000.0, 2500.0, -300.0, 1700.0)
    cores = tile_cores(ext, 1000.0)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(ext[0], ext[1], 2000), rng.uniform(ext[2], ext[3], 2000)])
    pts = np.vstack([pts, [[ext[1], ext[3]], [ext[0], ext[2]]]])  # the extreme corners
    hits = np.zeros(len(pts), int)
    for x0, x1, y0, y1 in cores:
        hits += (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
    assert (hits == 1).all()


from lunar_trn.craters.catalog import explain_rim


def test_explain_rim_reasons():
    dem, truth = synthetic_crater_dem(n=1200, n_craters=10, D_range=(150.0, 500.0), seed=6)
    c = truth[0]
    reason, rec = explain_rim(dem, c["x"], c["y"], c["D"] / 2)
    assert reason == "ok" and abs(2 * np.sqrt(rec["a"] * rec["b"]) / c["D"] - 1) < 0.1
    assert explain_rim(dem, c["x"], c["y"], c["D"] / 2, CatalogConfig(completeness_min=1.01))[0] == "completeness"
    flat, _ = synthetic_crater_dem(n=1200, n_craters=0, seed=7)
    assert explain_rim(flat, -20000.0, 30000.0, 100.0)[0] in ("no_crest", "few_points")
    assert explain_rim(flat, -20000.0 + 2990.0, 30000.0, 100.0)[0] == "edge"
