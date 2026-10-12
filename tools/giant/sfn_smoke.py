"""Synthetic smoke test of GIANT 2.0.0 surface feature navigation (SFN) for a lunar descent camera.

- Synthetic 3 km x 3 km DEM at 5 m posts near the lunar south pole (Moon-fixed frame, km).
- Truth image: rendered with GIANT's own ray tracer (full DEM, single-bounce shadows, McEwen BRDF).
- Feature catalog: 3x3 sub-tiles (81x81 posts, 400 m) cut from the same DEM.
- SFN run from a perturbed a-priori pose; compare matched feature pixels to truth projections.

Usage: python -I sfn_smoke.py <scripts_dir> [sun_elevation_deg]
"""
import sys
import time
from datetime import datetime

sys.path.insert(0, sys.argv[1])
import giant_shim  # noqa: F401,E402

import numpy as np  # noqa: E402

from giant.camera import Camera  # noqa: E402
from giant.camera_models import PinholeModel  # noqa: E402
from giant.image import OpNavImage  # noqa: E402
from giant.rotations import Rotation  # noqa: E402
from giant.ray_tracer.kdtree import KDTree  # noqa: E402
from giant.ray_tracer.shapes import Triangle64, Point  # noqa: E402
from giant.ray_tracer.scene import Scene, SceneObject, CorrectionsType  # noqa: E402
from giant.ray_tracer.rays import compute_rays  # noqa: E402
from giant.ray_tracer.illumination import McEwenIllumination  # noqa: E402
from giant.relative_opnav.estimators.sfn import (  # noqa: E402
    SurfaceFeatureNavigation, SurfaceFeatureNavigationOptions)
from giant.relative_opnav.estimators.sfn.surface_features import (  # noqa: E402
    SurfaceFeature, FeatureCatalog, VisibleFeatureFinder, VisibleFeatureFinderOptions)


# Out-of-tree fix 2: FeatureCatalog lacks ``compute_intersect`` so it fails the runtime-checkable
# ``Traceable`` protocol in Scene.trace ("Target 0 doesn't support ray tracing").
if not hasattr(FeatureCatalog, "compute_intersect"):
    FeatureCatalog.compute_intersect = lambda self, ray: self.trace(ray)


class PatchedSFN(SurfaceFeatureNavigation):
    """Out-of-tree fix: GIANT 2.0.0 SFN.compute_rays passes float bounds to rays.compute_rays,
    which rejects non-integer dtypes ("rows must be of integral type")."""

    def compute_rays(self, feature_catalog, feature_ind, temperature=0):
        image_locs = self.camera.model.project_onto_image(feature_catalog.feature_bounds[feature_ind],
                                                          temperature=temperature)
        local_min = np.floor(image_locs.min(axis=1)).astype(int)
        local_max = np.ceil(image_locs.max(axis=1)).astype(int)
        return (compute_rays(self.camera.model, (local_min[1], local_max[1]), (local_min[0], local_max[0]),
                             grid_size=self.grid_size, temperature=temperature), (local_min, local_max))


SUN_EL = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
rng = np.random.default_rng(0)
R_MOON = 1737.4  # km
POST = 0.005  # km (5 m)
N = 601  # posts per side -> 3 km

# ---- synthetic DEM (heights in metres on a local E/N grid) ----
ax = (np.arange(N) - N // 2) * POST * 1000.0  # metres
X, Y = np.meshgrid(ax, ax)  # rows = N (y), cols = E (x)
H = np.zeros_like(X)
for _ in range(120):
    cx, cy = rng.uniform(-1500, 1500, 2)
    r = rng.pareto(1.8) * 15 + 15
    d = np.hypot(X - cx, Y - cy) / r
    depth = 0.2 * r
    bowl = np.where(d < 1, depth * (d ** 2 - 1), 0.0)
    rim = np.where(d >= 1, 0.08 * r * np.exp(-((d - 1) / 0.35) ** 2), 0.0)
    H += bowl + rim
H += 0.002 * X + 0.001 * Y  # regional slope
H_km = H / 1000.0

# Moon-fixed local frame at the south pole: U = up, E x N = U
U = np.array([0.0, 0.0, -1.0])
E = np.array([1.0, 0.0, 0.0])
Nn = np.cross(U, E)
C0 = R_MOON * U  # tile centre on the sphere (km, Moon-fixed)
Xk, Yk = X / 1000.0, Y / 1000.0
curv = (Xk ** 2 + Yk ** 2) / (2 * R_MOON)
VERTS = (C0[None, None, :] + Xk[..., None] * E + Yk[..., None] * Nn
         + (H_km - curv)[..., None] * U)  # (N, N, 3) km


def grid_to_tris(verts: np.ndarray) -> Triangle64:
    n_r, n_c, _ = verts.shape
    idx = np.arange(n_r * n_c, dtype=np.uint32).reshape(n_r, n_c)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    facets = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                             np.stack([b, d, c], -1).reshape(-1, 3)])  # (v1-v0)x(v2-v0) = +U
    return Triangle64(verts.reshape(-1, 3).copy(), 1.0, facets, compute_reference_ellipsoid=False)


import os  # noqa: E402
CACHE = os.path.join(sys.argv[1], "..", "work", f"truth_img_el{SUN_EL:g}.npy")
ERR_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
RUN_PNP = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
t0 = time.time()
full_tris = grid_to_tris(VERTS)
print(f"full DEM facets: {full_tris.num_faces}, normal.U mean: {(full_tris.normals @ U).mean():.3f}")
full_kd = KDTree(full_tris, max_depth=16)
if not os.path.exists(CACHE):
    full_kd.build(force=True, print_progress=False)
print(f"KDTree built in {time.time() - t0:.1f} s")

# ---- feature catalog: 3x3 tiles of 81x81 posts ----
HALF = 40
features = []
for k, (ci, cj) in enumerate([(i, j) for i in (140, 300, 460) for j in (140, 300, 460)]):
    sub = VERTS[ci - HALF:ci + HALF + 1, cj - HALF:cj + HALF + 1]
    pts = sub.reshape(-1, 3)
    ctr = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - ctr)
    nrm = vt[2] if vt[2] @ U > 0 else -vt[2]
    center = VERTS[ci, cj]
    kd = KDTree(grid_to_tris(sub), max_depth=10)
    kd.build(force=True, print_progress=False)
    features.append(SurfaceFeature(kd, nrm, center, f"F{k:02d}", ground_sample_distance=POST))
fc = FeatureCatalog(features)
fc.feature_finder = VisibleFeatureFinder(fc, options=VisibleFeatureFinderOptions(
    gsd_scaling=3.0, incident_angle_maximum=89.5, reflectance_angle_maximum=80, percent_in_fov=90))

# ---- camera: K = [[618.04,0,256],[0,618.04,256],[0,0,1]], 512x512 ----
model = PinholeModel(focal_length=1.0, kx=618.04, ky=618.04, px=256.0, py=256.0,
                     n_rows=512, n_cols=512, field_of_view=2 * np.rad2deg(np.arctan(256 / 618.04)))


def look_at(cam_pos, target):
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = E - (E @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])  # rows = camera axes in Moon-fixed -> R_mf2cam


ALT = 3.0  # km
r_true = C0 + ALT * U + 0.15 * E - 0.1 * Nn
R_true = look_at(r_true, C0)
az = np.deg2rad(45.0)
sun_dir = (np.cos(np.deg2rad(SUN_EL)) * (np.cos(az) * E + np.sin(az) * Nn)
           + np.sin(np.deg2rad(SUN_EL)) * U)
sun_pos = sun_dir * 1.496e8
DATE = datetime(2027, 1, 1)


def make_scene(shape):
    tgt = SceneObject(shape, name="moon", position_function=lambda d: np.zeros(3),
                      orientation_function=lambda d: Rotation(np.eye(3)), corrections=CorrectionsType.NONE)
    sun = SceneObject(Point(sun_pos), name="sun", position_function=lambda d: sun_pos,
                      orientation_function=lambda d: Rotation(np.eye(3)), corrections=CorrectionsType.NONE)
    return Scene(target_objs=[tgt], light_obj=sun)


# ---- truth image via GIANT ray tracer on the full DEM ----
t0 = time.time()
assert np.allclose(Rotation(R_true).matrix, R_true), "Rotation(matrix) convention mismatch"
truth_img = OpNavImage(np.zeros((512, 512)), observation_date=DATE, rotation_inertial_to_camera=Rotation(R_true),
                       position=r_true, velocity=np.zeros(3), temperature=0.0, exposure_type="long")
if os.path.exists(CACHE):
    img = np.load(CACHE)
else:
    truth_scene = make_scene(full_kd)
    truth_scene.update(truth_img, corrections=CorrectionsType.NONE)
    rays, uv = compute_rays(model, (0, 511), (0, 511), grid_size=2)
    illum = McEwenIllumination()(truth_scene.get_illumination_inputs(rays))
    img = np.zeros((512, 512))
    np.add.at(img, (np.floor(uv[1] + 0.5).astype(int), np.floor(uv[0] + 0.5).astype(int)), illum)
    img = img / 4.0
    img = 200 * img / np.percentile(img, 99.5) + rng.normal(0, 1.0, img.shape)
    np.save(CACHE, img)
print(f"truth image ready in {time.time() - t0:.1f} s; dark-pixel fraction {(img < 3).mean():.2f}")

# ---- a priori pose: perturbed by ~30 m and ~0.2 deg ----
dpos = ERR_SCALE * np.array([0.030, -0.020, 0.010])
r_apr = r_true + dpos
dR = Rotation(ERR_SCALE * np.deg2rad([0.15, -0.1, 0.05])).matrix  # rotation vector
R_apr = dR @ R_true
apr_img = OpNavImage(img, observation_date=DATE, rotation_inertial_to_camera=Rotation(R_apr), position=r_apr,
                     velocity=np.zeros(3), temperature=0.0, exposure_type="long")
camera = Camera(images=[apr_img], model=model)
scene = make_scene(fc)
scene.update(apr_img, corrections=CorrectionsType.NONE)

truth_px = model.project_onto_image(R_true @ (np.array([f.body_fixed_center for f in features]).T
                                              - r_true[:, None]))
apr_px = model.project_onto_image(R_apr @ (np.array([f.body_fixed_center for f in features]).T
                                           - r_apr[:, None]))

opts = SurfaceFeatureNavigationOptions(search_region=25, second_search_region=6, run_pnp_solver=RUN_PNP,
                                       pnp_ransac_iterations=0, min_corr_score=0.3, brdf=McEwenIllumination(),
                                       position_sigma=0.1, attitude_sigma=np.deg2rad(0.5), measurement_sigma=0.5)
sfn = PatchedSFN(scene, camera, options=opts)
t0 = time.time()
sfn.estimate(camera.images[0])
print(f"SFN estimate in {time.time() - t0:.1f} s")

det = sfn.details[0]
obs = sfn.observed_bearings[0]
vis = sfn.visible_features[0]
print("visible features:", vis)
print("details keys:", list(det.keys()))
print(" feat  score   truth-apriori(px)      obs-apriori(px)   |obs-truth|px")
for n, fi in enumerate(vis):
    d0 = truth_px[:, fi] - apr_px[:, fi]
    d1 = obs[:, n] - apr_px[:, fi]
    e1 = np.linalg.norm(obs[:, n] - truth_px[:, fi])
    print(f" {features[fi].name}  {det['Correlation Scores'][n]:.3f}  ({d0[0]:6.2f},{d0[1]:6.2f})   "
          f"({d1[0]:6.2f},{d1[1]:6.2f})   {e1:7.3f}")
if det.get("PnP Solution"):
    # target moved in camera frame; recover camera pose in Moon-fixed frame
    tgt = scene.target_objs[0]
    R_est = tgt.orientation.matrix  # Moon-fixed -> camera
    r_est = -R_est.T @ tgt.position  # camera position in Moon-fixed (Moon centre at origin)
    print(f"position error  a priori {1000 * np.linalg.norm(r_apr - r_true):.1f} m -> "
          f"after PnP {1000 * np.linalg.norm(r_est - r_true):.1f} m")
    ang = lambda A, B: np.rad2deg(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))
    print(f"attitude error  a priori {ang(R_apr, R_true):.3f} deg -> after PnP {ang(R_est, R_true):.3f} deg")

if not RUN_PNP and len(sys.argv) > 5:
    # Isolate GIANT's PnP update with different measurement sets and check the reprojection vs truth.
    import copy
    if sys.argv[5] == "fixjac":
        # Out-of-tree test of a suspected bug in SFN._lls: the measurement-row Jacobian w.r.t. the shift has the
        # wrong sign (residual = observed - projected, jac = +J_cp R), and the a-priori rows return a constant
        # zero residual while their Jacobian is diag(1/sigma).
        import scipy.optimize
        import giant.relative_opnav.estimators.sfn.sfn_class as sfn_mod

        def _fixed_least_squares(fun, x0, jac=None, **kw):
            def f2(x):
                r = np.array(fun(x), dtype=float)
                r[-6:] = jac(x)[-6:] @ x
                return r

            def j2(x):
                J = np.array(jac(x), dtype=float)
                J[:-6, :3] *= -1  # shift columns: d(obs - proj)/d(shift) = -J_cp R
                return J
            return scipy.optimize.least_squares(f2, x0, jac=j2, **kw)
        sfn_mod.least_squares = _fixed_least_squares
    obs_first = obs.copy()
    no_f00 = obs_first.copy()
    no_f00[:, 0] = np.nan
    t_nan = truth_px[:, vis].copy()
    t_nan[:, np.isnan(obs_first[0])] = np.nan
    for mlabel, meas in (("truth pixels", truth_px[:, vis]), ("truth pixels, same NaNs as SFN", t_nan), ("SFN first-pass pixels", obs_first),
                         ("SFN first-pass minus F00", no_f00)):
        sfn.observed_bearings[0] = meas.copy()
        sfn.computed_bearings[0] = apr_px[:, vis].copy()
        dp, dq = sfn.pnp_solver(0, camera.images[0], 0)
        fc_try = copy.deepcopy(fc)
        fc_try.translate(dp)
        fc_try.rotate(dq)
        pred = model.project_onto_image(fc_try.feature_locations[vis].T)
        R_est = dq.matrix @ R_apr
        r_est = r_apr - R_apr.T @ dp
        ang = np.rad2deg(np.arccos(np.clip((np.trace(R_est @ R_true.T) - 1) / 2, -1, 1)))
        print(f"PnP[{mlabel}]: dpos_m={np.round(1000 * dp, 1)} drot_deg={np.round(np.rad2deg(dq.vector), 3)} "
              f"max_reproj_err_vs_truth={np.abs(pred - truth_px[:, vis]).max():.2f}px "
              f"pos_err={1000 * np.linalg.norm(r_est - r_true):.1f}m att_err={ang:.3f}deg")
