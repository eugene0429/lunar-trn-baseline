import numpy as np

from lunar_trn.craters.annotate import annotate_detections, catalog_recall, project_catalog
from lunar_trn.craters.detect import Detections, constant_sun_fn
from lunar_trn.craters.synthetic import default_camera, flat_catalog, nadir_pose
from lunar_trn.validation.v1 import sun_from_az_el

CX, CY = -20000.0, 30000.0


def _setup():
    rng = np.random.default_rng(0)
    cat = flat_catalog(rng, 300, (CX - 4000, CX + 4000, CY - 4000, CY + 4000), (60.0, 600.0))
    cam, T = default_camera(), nadir_pose(CX, CY, 5000.0)
    return cat, cam, T


def test_predicted_ellipses_annotate_perfectly():
    cat, cam, T = _setup()
    p = project_catalog(cat, cam, T)
    sel = np.nonzero(p["in_image"] & (p["D_px"] >= 6))[0]
    assert sel.size > 20
    det = Detections.from_ellipses(p["u"][sel], p["v"][sel], p["A"][sel], p["B"][sel], p["theta"][sel])
    ann = annotate_detections(det, cat, cam, T, constant_sun_fn(0.0))
    assert (ann["cat_index"] == sel).all() and np.allclose(ann["err_u"], 0, atol=1e-9)
    assert np.allclose(ann["size_ratio"], 1.0)
    rec = catalog_recall(det, cat, cam, T, sun_from_az_el(T[:3, 3], 90.0, 20.0), D_px_min=6.0)
    assert rec["detected"].all() and len(rec["cat_id"]) == sel.size
    assert np.all((rec["sun_el_deg"] > 15) & (rec["sun_el_deg"] < 25))


def test_far_or_wrong_size_detection_is_unmatched():
    cat, cam, T = _setup()
    p = project_catalog(cat, cam, T)
    k = int(np.nonzero(p["in_image"] & (p["D_px"] >= 10))[0][0])
    det = Detections.from_ellipses([p["u"][k] + 0.6 * p["D_px"][k], p["u"][k]], [p["v"][k], p["v"][k]],
                                   [p["A"][k], 3 * p["A"][k]], [p["B"][k], 3 * p["B"][k]], [0.0, 0.0])
    ann = annotate_detections(det, cat, cam, T, constant_sun_fn(0.0))
    assert ann["cat_index"][0] != k and ann["cat_index"][1] == -1
    empty = annotate_detections(Detections.empty(), cat, cam, T, constant_sun_fn(0.0))
    assert empty["cat_index"].shape == (0,)
