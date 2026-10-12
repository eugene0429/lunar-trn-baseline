# P-G 크레이터 패턴 매칭 기준선 (추적형 1단계) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** StereoLunar nadir 영상에서 T-I와 같은 하네스·사전 오차·채점으로 돌릴 수 있는 CNav식 크레이터 패턴 매칭(P-G) 기준선을 만들고, 보정 → V7(추적형) → H-b(조명 잔차) 비교까지 끝낸다.

**Architecture:** 새 하위 패키지 `src/lunar_trn/craters/`에 원뿔 대수, DEM 카탈로그 추출, CNav식 검출기, 위치 가설 투표 매칭, `localize_craters()`를 둔다. 자세 후단(`pose.ransac_position`, `pose.refine_pose`, `pose.mahalanobis_gate`)과 실험 도구(`experiments.sample_prior/score`)는 T-I 것을 그대로 쓴다. 실행기 `scripts/run_pg.py`는 `run_v2.py`와 같은 사전 표본 시드를 써서 T-I 실행과 짝지어 비교한다.

**Tech Stack:** Python 3.10, NumPy, SciPy(`ndimage`, `spatial.cKDTree`, `optimize.linear_sum_assignment`, `stats.beta`), OpenCV(`connectedComponents`, `resize`), pytest, matplotlib(그림만, dev 의존성).

**Spec:** `docs/superpowers/specs/2026-10-12-pg-crater-baseline-design.md` (실행자는 계획과 스펙을 함께 읽는다)

## Global Constraints

- Python 3.10. 새 런타임 의존성을 추가하지 않는다(`pyproject.toml`의 numpy/scipy/opencv/rasterio/OpenEXR/pyyaml만). 그림은 dev 의존성 matplotlib.
- 코드 주석·docstring·로그 메시지는 영어. `docs/` 문서는 한국어.
- **커밋하지 않는다.** 레포 규칙(CLAUDE.md): 커밋·푸시는 사용자가 요청할 때만. 각 태스크 끝의 "체크포인트"는 시험 실행과 변경 파일 확인만 한다.
- `data/`, `outputs/`는 커밋 대상이 아니다. 카탈로그는 `data/catalog/`에 둔다.
- T-I 알고리즘 코드(`coarse.py`, `fine.py`, `maps.py`, `matching.py`, `localize.py`의 기존 로직)는 바꾸지 않는다. 예외: `LocalizationResult`에 `family` 필드 추가(Task 1).
- 임계값은 **보정 표본(기하 순번 % 4 == 1)에서만** 정한다. 평가 표본으로 임계값을 바꾸지 않는다.
- 원문에 수치가 없는 임계값은 모두 `docs/decisions.md` §G에 값·근거·코드 위치를 남긴다.
- `health`에는 판정이 아니라 원신호를 남긴다.
- 영상 좌표는 `K` 프레임의 연속 좌표 `u = col + pixel_center`(StereoLunar는 0.5)다. `pose.project_uv`와 같다.
- 사전 표본 시드: `zlib.crc32(f"{gid}-{k_map}-{j}-{trial}-{seed}".encode())`, 가지치기 없는 P-G는 `k_map = j`(T-I V2 대각과 짝).
- 채점 허용치: `3 × experiments.fine_map_pixel(명목 대역 고도 / fx)` 미터(T-I와 같음).
- 이번 단계에서는 wdoppenberg 저장소의 코드를 가져오지 않는다(LIS는 2단계).
- GPL 코드 금지, 라이선스 파일 없는 저장소 코드 금지.
- 코드 블록 첫 줄의 `# <경로>`(또는 `# <경로> (append)` 등) 주석은 붙일 파일을 알려 주는 표시다. 파일에는 넣지 않는다. `(append)`는 파일 끝에 덧붙이고, `(replace ...)`는 말한 부분을 바꾼다.

## Review Focus

1. **현재 시야에서 무효한 테두리 원뿔**(카메라 뒤, oblique에서 수평선에 걸침, 테두리 평면을 뒤에서 봄): `valid=False`로 표시되고, 투표·할당에 NaN이 들어가지 않아야 한다. → Task 2 `test_rim_behind_camera_or_seen_from_below_is_invalid`, Task 9 `test_assign_ignores_invalid_predictions`.
2. **쓸 수 있는 크레이터가 없는 영상**(전부 검음, 균일, NaN 화소): 예외 없이 `no_fix`, `stage == "detect"`. → Task 7 `test_detect_handles_black_nan_and_degenerate_sun`, Task 10 `test_black_image_is_no_fix_at_detect`.
3. **사전 시야가 카탈로그 밖이거나 카탈로그가 빈 경우**: 예외 없이 `no_fix`, `stage == "match"`. → Task 10 `test_empty_catalog_is_no_fix_at_match`.
4. **DEM 무자료(NaN) 구역과 타일 경계**: 카탈로그에 NaN 항목이 없고, 타일 경계에서 크레이터가 중복되거나 빠지지 않는다. → Task 4 `test_extract_tolerates_nodata`, Task 5 `test_tile_cores_partition_extent`.
5. **모호한 장면**(무작위 검출, 반복 패턴): 자신 있는 오고정 대신 고정 없음. 태양이 광축 근처라 영상 안 태양 방향이 정의되지 않을 때도 예외 없음. → Task 10 `test_random_detections_do_not_fix`, Task 7 `test_detect_handles_black_nan_and_degenerate_sun`.

## File Structure

| 파일 | 책임 |
|---|---|
| `src/lunar_trn/localize.py` (수정) | `LocalizationResult.family` 필드 추가 |
| `src/lunar_trn/craters/__init__.py` | 패키지 docstring |
| `src/lunar_trn/craters/conics.py` | 테두리 타원 ↔ 영상 원뿔, 원뿔 중심·축, 직접 타원 적합, 타원 잔차 |
| `src/lunar_trn/craters/catalog.py` | `CraterCatalog`, `make_catalog`, `rim_frame`, `CatalogConfig`, `extract_catalog`, `tile_cores`, (Task 16) `prune_catalog` |
| `src/lunar_trn/craters/synthetic.py` | 합성 크레이터 DEM·평면 카탈로그·천저 자세·합성 장면 렌더 |
| `src/lunar_trn/craters/detect.py` | `DetectorConfig`, `Regions`, `Detections`, 고역 정규화, 안정 영역, 짝짓기, 조명 추정, `detect_craters`, 태양 방향 함수 |
| `src/lunar_trn/craters/annotate.py` | 정답 자세 기반 주석(검출 ↔ 카탈로그, 카탈로그 재현율) |
| `src/lunar_trn/craters/match.py` | `MatchConfig`, `Candidates`, `predict`, `select_candidates`, `vote_positions`, `assign` |
| `src/lunar_trn/craters/localize.py` | `CraterMap`, `CraterConfig`, `localize_craters` |
| `src/lunar_trn/craters/report.py` | 실행 기록 집계, 게이트 재판정, T-I 짝 비교 |
| `scripts/build_catalog.py` | 전 DEM 카탈로그(타일·병렬) |
| `scripts/run_pg.py` | P-G 실행기(검출 전용 보정 모드 포함) |
| `scripts/analyze_pg_calib.py` | 보정 분석(`detect`, `loc`) |
| `scripts/analyze_pg.py` | V7·H-b 표 |
| `scripts/pg_figures.py` | 오버레이·크기분포·재현율·V7 그림 |
| `tests/test_family.py`, `tests/test_conics.py`, `tests/test_catalog.py`, `tests/test_detect.py`, `tests/test_annotate.py`, `tests/test_match.py`, `tests/test_localize_craters.py`, `tests/test_report_pg.py` | 시험 |
| `docs/decisions.md` §G, `docs/validation/v7_pg_nadir.md`, `docs/interface.md`, `README.md`, `docs/handoff/2026-10-1x-pg-session.md` | 문서 |

모든 명령은 레포 루트(`~/lunar-trn-baseline`)에서 `.venv/bin/python`으로 실행한다.

---

### Task 1: `LocalizationResult.family`와 `craters` 패키지

**Files:**
- Modify: `src/lunar_trn/localize.py` (`LocalizationResult` 정의)
- Create: `src/lunar_trn/craters/__init__.py`
- Test: `tests/test_family.py`

**Interfaces:**
- Produces: `LocalizationResult(status, T_world_cam, cov, landmarks, health, stage, family="T-I")` — 기존 6개 위치 인자 호출은 그대로 동작한다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_family.py
import numpy as np

from lunar_trn.localize import LocalizationResult


def test_result_family_defaults_to_ti():
    r = LocalizationResult("no_fix", np.eye(4), np.full((6, 6), np.nan), {}, {}, "coarse")
    assert r.family == "T-I"


def test_result_family_can_be_set():
    r = LocalizationResult("fix", np.eye(4), np.eye(6), {}, {}, "pose", family="P-G")
    assert r.family == "P-G"


def test_craters_package_imports():
    import lunar_trn.craters  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_family.py -v`
Expected: FAIL (`AttributeError: ... has no attribute 'family'`, `ModuleNotFoundError: lunar_trn.craters`)

- [ ] **Step 3: Implement**

`src/lunar_trn/localize.py`의 `LocalizationResult`에 마지막 필드를 추가한다.

```python
# src/lunar_trn/localize.py (replace the LocalizationResult class)
@dataclass
class LocalizationResult:
    status: str  # "fix" or "no_fix"
    T_world_cam: np.ndarray
    cov: np.ndarray  # 6x6 (position world, attitude small-angle world)
    landmarks: dict
    health: dict
    stage: str  # last stage reached: "coarse", "fine" (T-I); "detect", "match", "pose" (P-G)
    family: str = "T-I"  # method family, handoff section 2.1
```

`src/lunar_trn/craters/__init__.py`:

```python
# src/lunar_trn/craters/__init__.py
"""P-G crater pattern matching (handoff section 2.1): catalog, detector, matcher, localiser."""
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_family.py -v && .venv/bin/python -m pytest -m "not slow" -q`
Expected: 3 passed; 기존 시험 전부 통과

- [ ] **Step 5: 체크포인트** — `git status --short`로 바뀐 파일이 위 두 개와 시험 파일뿐인지 확인한다. 커밋하지 않는다.

---

### Task 2: 원뿔 대수 (`conics.py`)

**Files:**
- Create: `src/lunar_trn/craters/conics.py`
- Test: `tests/test_conics.py`

**Interfaces:**
- Consumes: `lunar_trn.pose.project_uv(K, R_wc, c, X) -> (uv (N,2), z (N,))`
- Produces:
  - `plane_conic(a, b, psi) -> (N,3,3)`
  - `ellipse_to_conic(u, v, A, B, theta) -> (N,3,3)`
  - `conic_to_ellipse(C) -> (u, v, A, B, theta, valid)` (각 (N,))
  - `rim_axes(e1, e2, psi) -> (e_major (N,3), e_minor (N,3))`
  - `project_rims(K, R_wc, c, X_c, e1, e2, a, b, psi) -> dict` 키: `u, v, A, B, theta, D_px, z, uv0 (N,2), bias (N,2), valid`
  - `fit_ellipse_direct(x, y) -> (u, v, A, B, theta, ok)` (스칼라)
  - `ellipse_residual(x, y, u, v, A, B, theta) -> (N,)` 근사 부호 거리(입력 단위)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conics.py
import numpy as np

from lunar_trn.craters import conics
from lunar_trn.pose import project_uv

F = 256 / np.tan(np.deg2rad(22.5))
K = np.array([[F, 0, 256.0], [0, F, 256.0], [0, 0, 1.0]])
E1 = np.array([[1.0, 0.0, 0.0]])
E2 = np.array([[0.0, 1.0, 0.0]])


def _camera(alt, tilt_deg=0.0):
    """Camera above the origin of a flat world (ground z = 0), optical axis -Z tilted about +X."""
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    t = np.deg2rad(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Rx @ R, np.array([0.0, 0.0, alt])


def _ground_point(R, c, u, v):
    d = R @ np.linalg.inv(K) @ np.array([u, v, 1.0])
    return c + (-c[2] / d[2]) * d


def test_projected_rim_points_lie_on_image_ellipse():
    R, c = _camera(3500.0, tilt_deg=30.0)
    Xc = _ground_point(R, c, 400.0, 300.0)[None, :]
    a, b, psi = 500.0, 350.0, 0.4
    p = conics.project_rims(K, R, c, Xc, E1, E2, [a], [b], [psi])
    assert p["valid"][0]
    emaj, emin = conics.rim_axes(E1, E2, np.array([psi]))
    phi = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    P = Xc + a * np.cos(phi)[:, None] * emaj + b * np.sin(phi)[:, None] * emin
    uv, _ = project_uv(K, R, c, P)
    r = conics.ellipse_residual(uv[:, 0], uv[:, 1], p["u"][0], p["v"][0], p["A"][0], p["B"][0], p["theta"][0])
    assert np.max(np.abs(r)) < 1e-6 * p["A"][0] + 1e-6


def test_conic_ellipse_round_trip():
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, 512, 20), rng.uniform(0, 512, 20)
    B = rng.uniform(3, 40, 20)
    A = B * rng.uniform(1.2, 2.0, 20)
    th = rng.uniform(-np.pi / 2, np.pi / 2, 20)
    u2, v2, A2, B2, th2, ok = conics.conic_to_ellipse(conics.ellipse_to_conic(u, v, A, B, th))
    assert ok.all()
    np.testing.assert_allclose(np.column_stack([u2, v2, A2, B2]), np.column_stack([u, v, A, B]), rtol=1e-9, atol=1e-7)
    dth = (th2 - th + np.pi / 2) % np.pi - np.pi / 2
    assert np.max(np.abs(dth)) < 1e-7


def test_center_bias_zero_at_nadir_and_several_px_off_axis():
    R, c = _camera(3500.0)
    p = conics.project_rims(K, R, c, np.zeros((1, 3)), E1, E2, [500.0], [500.0], [0.0])
    assert np.hypot(*p["bias"][0]) < 1e-6
    R, c = _camera(3500.0, tilt_deg=30.0)
    Xc = _ground_point(R, c, 256.0, 256.0)[None, :]
    p = conics.project_rims(K, R, c, Xc, E1, E2, [500.0], [500.0], [0.0])
    assert 2.0 < np.hypot(*p["bias"][0]) < 8.0  # ~4 px for a 1 km crater at 3.5 km, 30 deg off nadir


def test_rim_behind_camera_or_seen_from_below_is_invalid():
    R, c = _camera(3500.0)
    behind = np.array([[0.0, 0.0, 5000.0]])  # above the camera, which looks down
    p = conics.project_rims(K, R, c, behind, E1, E2, [100.0], [100.0], [0.0])
    assert not p["valid"][0] and np.isnan(p["u"][0])
    flipped = conics.project_rims(K, R, c, np.zeros((1, 3)), E2, E1, [100.0], [100.0], [0.0])  # normal -Z
    assert not flipped["valid"][0]
    empty = conics.project_rims(K, R, c, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), [], [], [])
    assert empty["u"].shape == (0,) and empty["valid"].shape == (0,)


def test_fit_ellipse_direct_recovers_parameters():
    rng = np.random.default_rng(1)
    phi = rng.uniform(0, 2 * np.pi, 60)
    u0, v0, A, B, th = 1200.0, -300.0, 80.0, 50.0, 0.7
    x = u0 + A * np.cos(phi) * np.cos(th) - B * np.sin(phi) * np.sin(th)
    y = v0 + A * np.cos(phi) * np.sin(th) + B * np.sin(phi) * np.cos(th)
    u, v, A2, B2, th2, ok = conics.fit_ellipse_direct(x + rng.normal(0, 0.2, 60), y + rng.normal(0, 0.2, 60))
    assert ok
    assert abs(u - u0) < 0.5 and abs(v - v0) < 0.5 and abs(A2 - A) < 0.5 and abs(B2 - B) < 0.5
    assert abs((th2 - th + np.pi / 2) % np.pi - np.pi / 2) < 0.02
    assert not conics.fit_ellipse_direct(np.arange(4.0), np.arange(4.0))[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_conics.py -v`
Expected: FAIL (`ModuleNotFoundError: lunar_trn.craters.conics`)

- [ ] **Step 3: Implement `src/lunar_trn/craters/conics.py`**

```python
# src/lunar_trn/craters/conics.py
"""Crater rim conics: rim ellipses in 3D and their perspective images.

A crater rim is an ellipse on its rim plane, P(s, t) = X_c + s e1 + t e2, with semi-axes
``a >= b`` (m) and major-axis angle ``psi`` measured from ``e1`` towards ``e2``. Its image
is the conic C = H^-T C_p H^-1 with the plane-to-image homography
H = K [R^T e1, R^T e2, R^T (X_c - c)] (``R`` camera-to-world, ``c`` camera centre).

Image coordinates are continuous pixel coordinates of the ``K`` frame, as in
:func:`lunar_trn.pose.project_uv` (pixel index = u - pixel_center). The centre of an image
ellipse is not the image of the rim centre; :func:`project_rims` returns both so that
callers can correct this perspective centre bias.
"""

from __future__ import annotations

import numpy as np

from ..pose import project_uv


def _wrap_half_pi(theta):
    return (np.asarray(theta, float) + np.pi / 2) % np.pi - np.pi / 2


def plane_conic(a, b, psi) -> np.ndarray:
    """Conic matrices (N, 3, 3) of centred ellipses with semi-axes a, b and angle psi."""
    a, b, psi = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, float)) for x in (a, b, psi)))
    c, s = np.cos(psi), np.sin(psi)
    ia2, ib2 = 1.0 / a**2, 1.0 / b**2
    C = np.zeros(a.shape + (3, 3))
    C[..., 0, 0] = c * c * ia2 + s * s * ib2
    C[..., 1, 1] = s * s * ia2 + c * c * ib2
    C[..., 0, 1] = C[..., 1, 0] = c * s * (ia2 - ib2)
    C[..., 2, 2] = -1.0
    return C


def ellipse_to_conic(u, v, A, B, theta) -> np.ndarray:
    """Conic matrices (N, 3, 3) of image ellipses (centre u, v; semi-axes A, B; angle theta)."""
    Cp = plane_conic(A, B, theta)
    u, v = np.broadcast_arrays(np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float)))
    T = np.zeros(Cp.shape)
    T[..., 0, 0] = T[..., 1, 1] = T[..., 2, 2] = 1.0
    T[..., 0, 2] = -u
    T[..., 1, 2] = -v
    return np.swapaxes(T, -1, -2) @ Cp @ T


def conic_to_ellipse(C):
    """Centre (u, v), semi-axes A >= B, major-axis angle and validity of conics (N, 3, 3).

    Non-ellipses (hyperbolae, parabolae, imaginary or degenerate conics) are invalid and
    return NaN parameters.
    """
    C = np.asarray(C, float)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    M = C[..., :2, :2]
    m = C[..., :2, 2]
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] ** 2
    valid = np.isfinite(det) & (det > 0)
    sd = np.where(valid, det, 1.0)
    u = -(M[..., 1, 1] * m[..., 0] - M[..., 0, 1] * m[..., 1]) / sd
    v = -(M[..., 0, 0] * m[..., 1] - M[..., 0, 1] * m[..., 0]) / sd
    k = C[..., 2, 2] + m[..., 0] * u + m[..., 1] * v
    valid &= np.isfinite(k) & (k != 0)
    Q = M / np.where(valid, -k, 1.0)[..., None, None]
    Q = np.where(valid[..., None, None], Q, np.eye(2))
    w, V = np.linalg.eigh(Q)
    valid &= w[..., 0] > 0
    w = np.where(w > 0, w, np.nan)
    A = 1.0 / np.sqrt(w[..., 0])
    B = 1.0 / np.sqrt(w[..., 1])
    theta = _wrap_half_pi(np.arctan2(V[..., 1, 0], V[..., 0, 0]))
    nan = np.where(valid, 1.0, np.nan)
    return u * nan, v * nan, A * nan, B * nan, theta * nan, valid


def rim_axes(e1, e2, psi):
    """Major- and minor-axis unit vectors (N, 3) of rims with in-plane axes e1, e2."""
    psi = np.atleast_1d(np.asarray(psi, float))
    c, s = np.cos(psi)[:, None], np.sin(psi)[:, None]
    return c * e1 + s * e2, -s * e1 + c * e2


def project_rims(K, R_wc, c, X_c, e1, e2, a, b, psi) -> dict:
    """Image ellipses of N rims seen from a camera at ``c`` with camera-to-world rotation ``R_wc``.

    Returns ``u, v`` (ellipse centre), ``A, B, theta``, ``D_px = 2 sqrt(A B)``, ``z`` (camera
    depth of the rim centre), ``uv0`` (N, 2: image of the rim centre), ``bias`` (N, 2:
    ellipse centre minus ``uv0``) and ``valid``: an ellipse whose whole rim is in front of
    the camera and whose rim plane is seen from above (camera on the normal side).
    """
    K = np.asarray(K, float)
    R = np.asarray(R_wc, float)
    c = np.asarray(c, float)
    X_c = np.asarray(X_c, float).reshape(-1, 3)
    e1 = np.asarray(e1, float).reshape(-1, 3)
    e2 = np.asarray(e2, float).reshape(-1, 3)
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    psi = np.atleast_1d(np.asarray(psi, float))
    n = X_c.shape[0]
    if n == 0:
        z0 = np.zeros(0)
        return {"u": z0, "v": z0.copy(), "A": z0.copy(), "B": z0.copy(), "theta": z0.copy(),
                "D_px": z0.copy(), "z": z0.copy(), "uv0": np.zeros((0, 2)), "bias": np.zeros((0, 2)),
                "valid": np.zeros(0, bool)}
    cols = np.stack([e1 @ R, e2 @ R, (X_c - c) @ R], axis=-1)  # camera-frame columns R^T e1, R^T e2, R^T (X_c - c)
    H = K @ cols
    zc = (X_c - c) @ R[:, 2]
    emaj, emin = rim_axes(e1, e2, psi)
    front = (zc - np.hypot(a * (emaj @ R[:, 2]), b * (emin @ R[:, 2]))) > 0
    w = c - X_c
    above = np.einsum("ij,ij->i", np.cross(e1, e2), w) > 1e-3 * np.linalg.norm(w, axis=1)
    ok = front & above & np.isfinite(H).all(axis=(1, 2))
    Hinv = np.linalg.inv(np.where(ok[:, None, None], H, np.eye(3)))
    C = np.swapaxes(Hinv, -1, -2) @ plane_conic(a, b, psi) @ Hinv
    u, v, A, B, th, valid = conic_to_ellipse(C)
    valid &= ok
    uv0, _ = project_uv(K, R, c, X_c)
    nan = np.where(valid, 1.0, np.nan)
    u, v, A, B, th = u * nan, v * nan, A * nan, B * nan, th * nan
    return {"u": u, "v": v, "A": A, "B": B, "theta": th, "D_px": 2.0 * np.sqrt(A * B), "z": zc,
            "uv0": uv0, "bias": np.column_stack([u - uv0[:, 0], v - uv0[:, 1]]), "valid": valid}


def fit_ellipse_direct(x, y):
    """Direct least-squares ellipse fit (Fitzgibbon 1999; numerically stable form of Halir & Flusser 1998).

    Returns ``(u, v, A, B, theta, ok)``; ``ok`` is False for fewer than 5 points or when no
    ellipse solution exists.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    bad = (np.nan,) * 5 + (False,)
    if x.size < 5 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return bad
    mx, my = x.mean(), y.mean()
    sc = float(np.sqrt(np.mean((x - mx) ** 2 + (y - my) ** 2)))
    if sc <= 0:
        return bad
    xn, yn = (x - mx) / sc, (y - my) / sc
    D1 = np.column_stack([xn * xn, xn * yn, yn * yn])
    D2 = np.column_stack([xn, yn, np.ones_like(xn)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return bad
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    _, V = np.linalg.eig(M)
    V = np.real(V)
    cond = 4.0 * V[0] * V[2] - V[1] ** 2
    idx = np.nonzero(cond > 0)[0]
    if idx.size == 0:
        return bad
    a1 = V[:, idx[0]]
    qa, qb, qc, qd, qe, qf = np.concatenate([a1, T @ a1])
    Cn = np.array([[qa, qb / 2, qd / 2], [qb / 2, qc, qe / 2], [qd / 2, qe / 2, qf]])
    N = np.array([[1 / sc, 0, -mx / sc], [0, 1 / sc, -my / sc], [0, 0, 1.0]])
    u, v, A, B, th, ok = conic_to_ellipse((N.T @ Cn @ N)[None])
    return float(u[0]), float(v[0]), float(A[0]), float(B[0]), float(th[0]), bool(ok[0])


def ellipse_residual(x, y, u, v, A, B, theta) -> np.ndarray:
    """Approximate signed distance of points from an ellipse, in the units of x and y."""
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = np.asarray(x, float) - u, np.asarray(y, float) - v
    xp, yp = c * dx + s * dy, -s * dx + c * dy
    return (np.sqrt((xp / A) ** 2 + (yp / B) ** 2) - 1.0) * np.sqrt(A * B)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_conics.py -v`
Expected: 5 passed. `test_center_bias...`가 범위를 벗어나면 편향 크기를 출력해 보고, 연구 보고의 근사식(R²·sinθ·cosθ/h ≈ 31 m ≈ 4–5 px)과 비교해 구현 오류인지 확인한다(범위를 넓히지 말고 원인을 찾는다).

- [ ] **Step 5: 체크포인트** — `.venv/bin/python -m pytest -m "not slow" -q` 전부 통과. 커밋하지 않는다.

---

### Task 3: 카탈로그 자료구조 (`CraterCatalog`)

**Files:**
- Create: `src/lunar_trn/craters/catalog.py` (자료구조 부분)
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `frames.body_to_stereo`, `render.stereo_tangents`
- Produces:
  - `CraterCatalog` 필드: `X, normal, e1, e2 (N,3)`, `a, b, psi, x, y, h, depth, d_over_D, completeness, rim_h_std, fit_rms, seed_response (N,)`, `level, id (N,) int64`, `meta: dict`
  - 메서드: `len()`, `.D`, `.subset(sel)`, `.query_box(xmin, xmax, ymin, ymax, return_index=False)`, `.save(path)`, `CraterCatalog.load(path)`, `CraterCatalog.concat(cats, meta=None)`, `CraterCatalog.empty()`
  - `make_catalog(X, normal, e1, e2, a, b, psi, **extra) -> CraterCatalog` (x, y, h는 X에서 계산, 없는 원신호는 NaN, `level`=1, `id`=0..N-1)
  - `rim_frame(x, y, normal) -> (e1, e2)` (극점에서도 정의)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_catalog.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: FAIL (`ModuleNotFoundError: lunar_trn.craters.catalog`)

- [ ] **Step 3: Implement (catalog.py 첫 부분)**

```python
# src/lunar_trn/craters/catalog.py
"""Crater catalog extracted from a DEM: data structure, extraction and queries.

Each crater is a rim ellipse on its rim plane (see :mod:`lunar_trn.craters.conics`) plus raw
morphology signals. The catalog does not depend on the Sun, so one catalog per DEM serves
every illumination; the matcher selects craters by apparent size at run time.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import frames
from ..render import stereo_tangents

_VEC = ("X", "normal", "e1", "e2")
_SCALAR = ("a", "b", "psi", "x", "y", "h", "depth", "d_over_D", "completeness", "rim_h_std",
           "fit_rms", "seed_response")
_INT = ("level", "id")
_ALL = _VEC + _SCALAR + _INT


@dataclass
class CraterCatalog:
    X: np.ndarray  # (N, 3) rim-plane centre, body frame MOON_ME (m)
    normal: np.ndarray  # (N, 3) rim-plane unit normal, pointing up
    e1: np.ndarray  # (N, 3) in-plane axis: map-easting tangent made orthogonal to the normal
    e2: np.ndarray  # (N, 3) normal x e1
    a: np.ndarray  # (N,) semi-major axis (m)
    b: np.ndarray  # (N,) semi-minor axis (m)
    psi: np.ndarray  # (N,) major-axis angle from e1 towards e2 (rad)
    x: np.ndarray  # (N,) stereographic easting of the centre (m)
    y: np.ndarray  # (N,) stereographic northing of the centre (m)
    h: np.ndarray  # (N,) centre height above the reference sphere (m)
    depth: np.ndarray  # (N,) mean crest height minus inner floor, regional plane removed (m)
    d_over_D: np.ndarray  # (N,) depth / D with D = 2 sqrt(a b)
    completeness: np.ndarray  # (N,) fraction of azimuths with an accepted crest point
    rim_h_std: np.ndarray  # (N,) crest scatter about the rim plane (m)
    fit_rms: np.ndarray  # (N,) ellipse fit residual RMS (m)
    seed_response: np.ndarray  # (N,) scale-normalised LoG response of the seed (m)
    level: np.ndarray  # (N,) DEM decimation factor of the extraction level
    id: np.ndarray  # (N,) int64 identifier
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.a.shape[0])

    @property
    def D(self) -> np.ndarray:
        """Equivalent rim diameter 2 sqrt(a b) (m)."""
        return 2.0 * np.sqrt(self.a * self.b)

    @classmethod
    def empty(cls) -> "CraterCatalog":
        kw = {k: np.zeros((0, 3)) for k in _VEC} | {k: np.zeros(0) for k in _SCALAR}
        kw |= {k: np.zeros(0, np.int64) for k in _INT}
        return cls(**kw)

    def subset(self, sel) -> "CraterCatalog":
        sel = np.asarray(sel)
        return CraterCatalog(**{k: getattr(self, k)[sel] for k in _ALL}, meta=dict(self.meta))

    def query_box(self, xmin: float, xmax: float, ymin: float, ymax: float, return_index: bool = False):
        """Craters whose centre lies in the stereographic box (inclusive)."""
        idx = np.nonzero((self.x >= xmin) & (self.x <= xmax) & (self.y >= ymin) & (self.y <= ymax))[0]
        sub = self.subset(idx)
        return (sub, idx) if return_index else sub

    @classmethod
    def concat(cls, cats, meta: dict | None = None) -> "CraterCatalog":
        cats = [c for c in cats if c is not None and len(c)]
        out = cls(**{k: np.concatenate([getattr(c, k) for c in cats]) for k in _ALL}) if cats else cls.empty()
        out.meta = dict(meta or {})
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in _ALL}, meta=np.array([repr(self.meta)]))

    @classmethod
    def load(cls, path: str | Path) -> "CraterCatalog":
        with np.load(path, allow_pickle=False) as d:
            kw = {k: d[k] for k in _ALL}
            meta = ast.literal_eval(str(d["meta"][0]))
        return cls(**kw, meta=meta)


def rim_frame(x, y, normal):
    """Polar-safe in-plane axes: the map-easting tangent made orthogonal to the rim normal."""
    normal = np.asarray(normal, float).reshape(-1, 3)
    _, ux, _ = stereo_tangents(np.atleast_1d(x), np.atleast_1d(y))
    t = ux.reshape(-1, 3)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    e1 = t - np.sum(t * normal, axis=1, keepdims=True) * normal
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(normal, e1)


def make_catalog(X, normal, e1, e2, a, b, psi, **extra) -> CraterCatalog:
    """Catalog from rim geometry; morphology fields not given in ``extra`` are NaN."""
    X = np.asarray(X, float).reshape(-1, 3)
    n = X.shape[0]
    x, y, h = frames.body_to_stereo(X)
    kw = {"X": X, "normal": np.asarray(normal, float).reshape(-1, 3), "e1": np.asarray(e1, float).reshape(-1, 3),
          "e2": np.asarray(e2, float).reshape(-1, 3), "a": np.asarray(a, float).ravel(),
          "b": np.asarray(b, float).ravel(), "psi": np.asarray(psi, float).ravel(), "x": x, "y": y, "h": h}
    for k in ("depth", "d_over_D", "completeness", "rim_h_std", "fit_rms", "seed_response"):
        kw[k] = np.asarray(extra.get(k, np.full(n, np.nan)), float).ravel()
    kw["level"] = np.asarray(extra.get("level", np.ones(n)), np.int64).ravel()
    kw["id"] = np.asarray(extra.get("id", np.arange(n)), np.int64).ravel()
    return CraterCatalog(**kw, meta=dict(extra.get("meta", {})))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: 4 passed

- [ ] **Step 5: 체크포인트** — 빠른 시험 전부 통과. 커밋하지 않는다.

---

### Task 4: DEM 카탈로그 추출과 합성 지형 (`extract_catalog`, `synthetic.py`)

**Files:**
- Modify: `src/lunar_trn/craters/catalog.py` (추출 부분 추가)
- Create: `src/lunar_trn/craters/synthetic.py`
- Test: `tests/test_catalog.py` (시험 추가)

**Interfaces:**
- Consumes: `Dem`, `dem.bilinear`, `Dem.downsample`, `frames.stereo_to_body/body_to_stereo`, `conics.fit_ellipse_direct/ellipse_residual`, `rim_frame`
- Produces:
  - `CatalogConfig` (필드는 아래 코드; decisions G1–G4)
  - `extract_catalog(dem: Dem, cfg: CatalogConfig | None = None, core: tuple | None = None) -> CraterCatalog` — `core=(xmin, xmax, ymin, ymax)`이면 중심이 반열린 구간 `[xmin, xmax) × [ymin, ymax)` 안인 것만
  - `synthetic.crater_height(rho, D, dD, rim_frac=0.04)`
  - `synthetic.place_craters(rng, n, extent, D_range, dD_range, margin_r=3.0, spacing=1.15) -> list[dict(x, y, D, dD)]`
  - `synthetic.synthetic_crater_dem(n=1600, pixel=5.0, cx=-20000.0, cy=30000.0, n_craters=40, D_range=(60.0, 1500.0), dD_range=(0.08, 0.2), slope=(0.01, -0.005), noise_m=0.0, seed=0) -> (Dem, list[dict])`

- [ ] **Step 1: Write the failing tests (tests/test_catalog.py에 추가)**

```python
# tests/test_catalog.py (append)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: 새 시험 4개 FAIL (`ImportError: cannot import name 'CatalogConfig'`)

- [ ] **Step 3a: Implement `src/lunar_trn/craters/synthetic.py` (DEM 부분)**

```python
# src/lunar_trn/craters/synthetic.py
"""Synthetic crater terrain, catalogs and scenes for tests and controlled experiments."""

from __future__ import annotations

import numpy as np

from ..dem import Dem


def crater_height(rho, D: float, dD: float, rim_frac: float = 0.04) -> np.ndarray:
    """Radial height profile (m): parabolic bowl inside the rim crest, (r/rho)^3 ejecta outside.

    The crest is at rho = D/2 with height ``rim_frac * D``; crest-to-floor depth is ``dD * D``.
    """
    r = 0.5 * D
    rim = rim_frac * D
    rho = np.asarray(rho, float)
    inside = rim - dD * D * (1.0 - (rho / r) ** 2)
    outside = rim * (r / np.maximum(rho, r)) ** 3
    return np.where(rho <= r, inside, outside)


def place_craters(rng, n: int, extent, D_range, dD_range, margin_r: float = 3.0, spacing: float = 1.15,
                  tries_per_crater: int = 400) -> list[dict]:
    """Non-overlapping craters, log-uniform diameters, largest placed first."""
    xmin, xmax, ymin, ymax = extent
    Ds = np.sort(np.exp(rng.uniform(np.log(D_range[0]), np.log(D_range[1]), n)))[::-1]
    out: list[dict] = []
    for D in Ds:
        r = 0.5 * D
        lo_x, hi_x, lo_y, hi_y = xmin + margin_r * r, xmax - margin_r * r, ymin + margin_r * r, ymax - margin_r * r
        if lo_x >= hi_x or lo_y >= hi_y:
            continue
        for _ in range(tries_per_crater):
            x, y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            if all(np.hypot(x - c["x"], y - c["y"]) >= spacing * (r + 0.5 * c["D"]) for c in out):
                out.append({"x": float(x), "y": float(y), "D": float(D), "dD": float(rng.uniform(*dD_range))})
                break
    return out


def synthetic_crater_dem(n: int = 1600, pixel: float = 5.0, cx: float = -20000.0, cy: float = 30000.0,
                         n_craters: int = 40, D_range=(60.0, 1500.0), dD_range=(0.08, 0.2),
                         slope=(0.01, -0.005), noise_m: float = 0.0, seed: int = 0) -> tuple[Dem, list[dict]]:
    """Square DEM (n x n, centred at stereographic (cx, cy)) with planted bowl craters on a slope."""
    rng = np.random.default_rng(seed)
    x0 = cx - (n - 1) / 2 * pixel
    y0 = cy + (n - 1) / 2 * pixel
    xs = x0 + np.arange(n) * pixel
    ys = y0 - np.arange(n) * pixel
    X, Y = np.meshgrid(xs, ys)
    H = slope[0] * (X - cx) + slope[1] * (Y - cy)
    craters = place_craters(rng, n_craters, (xs[0], xs[-1], ys[-1], ys[0]), D_range, dD_range)
    for c in craters:
        w = int(np.ceil(2.0 * c["D"] / pixel))
        ci = int(round((c["x"] - x0) / pixel))
        ri = int(round((y0 - c["y"]) / pixel))
        sl = (slice(max(ri - w, 0), min(ri + w + 1, n)), slice(max(ci - w, 0), min(ci + w + 1, n)))
        H[sl] += crater_height(np.hypot(X[sl] - c["x"], Y[sl] - c["y"]), c["D"], c["dD"])
    if noise_m > 0:
        H += noise_m * rng.normal(size=H.shape)
    return Dem(H.astype(np.float32), x0, y0, pixel, name="synthetic_craters"), craters
```

- [ ] **Step 3b: Implement extraction (catalog.py에 추가)**

catalog.py의 import 블록을 아래로 바꾸고, 모듈 docstring 끝에 추출 절차 요약(씨앗 → 단면 → 적합 → 형태 → 필터·중복 제거)을 한 문단 붙인다.

```python
# src/lunar_trn/craters/catalog.py (replace the import block)
import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .. import frames
from ..dem import Dem, bilinear
from ..render import stereo_tangents
from .conics import ellipse_residual, fit_ellipse_direct
```

그다음 `make_catalog` 아래에 추가한다.

```python
# src/lunar_trn/craters/catalog.py (append)
@dataclass
class CatalogConfig:
    """DEM crater extraction settings (decisions G1-G4)."""

    levels: tuple = (1, 2, 4, 8, 16, 32, 64)  # DEM block-mean factors (5 m DEM -> 5 ... 320 m)
    radii_px: tuple = (5.0, 6.3, 7.9)  # seed radii per level, level pixels
    seed_sigma_frac: float = 0.45  # LoG sigma / seed radius: peak scale of rimmed bowls (not r / sqrt 2)
    seed_dD_min: float = 0.02  # LoG depth proxy 2 resp / D (resp ~ d/2 for a Gaussian bowl)
    n_azimuth: int = 48
    samples_per_radius: int = 12
    rho_max: float = 1.8  # profile extent, x seed radius
    rim_search: tuple = (0.5, 1.6)  # crest search interval, x seed radius
    inner_floor: float = 0.4  # floor = minimum inside this radius, x seed radius
    min_relief_m: float = 0.2  # crest must rise this much above the inner floor (rejects flat ground)
    detrend_annulus: tuple = (1.5, 2.5)  # regional plane fit annulus, x seed radius
    mad_k: float = 2.5  # crest outlier rejection, x 1.4826 MAD
    min_rim_points: int = 8
    size_consistency: tuple = (0.5, 1.6)  # fitted sqrt(a b) / seed radius
    centre_consistency: float = 0.5  # fitted centre offset / seed radius
    chunk: int = 20000  # seeds per vectorised profile batch
    # Filters (decisions G3)
    D_min_m: float = 50.0
    D_max_m: float = 6000.0  # 128 px at the 49 m/px top band
    completeness_min: float = 0.7
    ellipticity_max: float = 1.5
    dD_min: float = 0.04
    # De-duplication across levels (decisions G4)
    dedupe_centre: float = 0.3
    dedupe_size: float = 1.5


def level_margin_m(cfg: CatalogConfig, f: int, pixel: float) -> float:
    """Window margin that holds every profile of a seed centred in the core at level ``f``."""
    return (cfg.detrend_annulus[1] * max(cfg.radii_px) + 4.0) * f * pixel


def _crop(dem: Dem, xmin: float, xmax: float, ymin: float, ymax: float, align: int) -> Dem:
    """Sub-window of ``dem`` whose origin stays on the ``align``-pixel block grid of ``dem``."""
    rows, cols = dem.shape
    c0 = max(int(np.floor((xmin - dem.x0) / dem.pixel)) // align * align, 0)
    r0 = max(int(np.floor((dem.y0 - ymax) / dem.pixel)) // align * align, 0)
    c1 = min(int(np.ceil((xmax - dem.x0) / dem.pixel)) + 1, cols)
    r1 = min(int(np.ceil((dem.y0 - ymin) / dem.pixel)) + 1, rows)
    return Dem(dem.heights[r0:r1, c0:c1], dem.x0 + c0 * dem.pixel, dem.y0 - r0 * dem.pixel, dem.pixel,
               name=dem.name, radius=dem.radius)


def _fill_nan(h: np.ndarray):
    nan = ~np.isfinite(h)
    if not nan.any():
        return h.astype(np.float32), nan
    if nan.all():
        return np.zeros(h.shape, np.float32), nan
    idx = ndi.distance_transform_edt(nan, return_distances=False, return_indices=True)
    return h[tuple(idx)].astype(np.float32), nan


def _seeds(H: np.ndarray, pixel: float, cfg: CatalogConfig):
    radii = np.asarray(cfg.radii_px, float)
    sig = cfg.seed_sigma_frac * radii
    L = np.stack([np.float32(s * s) * ndi.gaussian_laplace(H, s) for s in sig])
    proxy = L / (radii[:, None, None] * pixel).astype(np.float32)  # 2 resp / D with D = 2 r pixel
    peak = (L == ndi.maximum_filter(L, size=(3, 5, 5), mode="nearest")) & (proxy >= cfg.seed_dD_min)
    k, i, j = np.nonzero(peak)
    r = radii[k]
    reach = cfg.detrend_annulus[1] * r + 1.0
    ok = (i >= reach) & (j >= reach) & (i < H.shape[0] - reach) & (j < H.shape[1] - reach)
    return i[ok].astype(float), j[ok].astype(float), r[ok], L[k[ok], i[ok], j[ok]].astype(float)


def _rim_candidates(H, i0, j0, r0, cfg: CatalogConfig):
    """Detrended radial profiles -> crest radius, crest height, validity per azimuth; inner floor."""
    A = cfg.n_azimuth
    th = 2.0 * np.pi * np.arange(A) / A
    rho = np.linspace(0.0, cfg.rho_max, int(round(cfg.rho_max * cfg.samples_per_radius)) + 1)
    ca, sa = np.cos(th), np.sin(th)
    dc = rho[None, None, :] * r0[:, None, None] * ca[None, :, None]  # column offsets (n, A, R)
    dr = -rho[None, None, :] * r0[:, None, None] * sa[None, :, None]  # row offsets (rows grow south)
    prof = bilinear(H, j0[:, None, None] + dc, i0[:, None, None] + dr)
    ann = np.linspace(cfg.detrend_annulus[0], cfg.detrend_annulus[1], 5)
    ac = ann[None, None, :] * r0[:, None, None] * ca[None, :, None]
    ar = -ann[None, None, :] * r0[:, None, None] * sa[None, :, None]
    ah = bilinear(H, j0[:, None, None] + ac, i0[:, None, None] + ar)
    n = i0.size
    G = np.stack([np.ones_like(ac), ac, ar], axis=-1).reshape(n, -1, 3)
    coef = np.linalg.solve(np.einsum("nki,nkj->nij", G, G),
                           np.einsum("nki,nk->ni", G, ah.reshape(n, -1))[..., None])[..., 0]
    prof = prof - (coef[:, 0, None, None] + coef[:, 1, None, None] * dc + coef[:, 2, None, None] * dr)
    lo = int(np.searchsorted(rho, cfg.rim_search[0]))
    hi = int(np.searchsorted(rho, cfg.rim_search[1], side="right"))
    seg = prof[:, :, lo:hi]
    k = np.argmax(np.nan_to_num(seg, nan=-np.inf), axis=2)
    crest = np.take_along_axis(seg, k[..., None], axis=2)[..., 0]
    inner = np.nanmin(prof[:, :, rho <= cfg.inner_floor], axis=(1, 2))
    valid = ((k > 0) & (k < seg.shape[2] - 1) & (crest - inner[:, None] > cfg.min_relief_m)
             & (crest > prof[:, :, -1]))
    return th, rho[lo + k], crest, valid, inner


def _fit_rim(dl: Dem, H, i0, j0, r0, th, crest_rho, crest, valid, inner, cfg: CatalogConfig):
    """Rim record of one seed and the reason code: "ok" or the first failed step."""
    if valid.sum() < cfg.min_rim_points:
        return None, "no_crest"
    rr = crest_rho[valid] * r0
    col = j0 + rr * np.cos(th[valid])
    row = i0 - rr * np.sin(th[valid])
    x, y = dl.colrow_to_xy(col, row)
    P = frames.stereo_to_body(x, y, bilinear(H, col, row))
    Pm = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - Pm)
    nrm = Vt[2] if Vt[2] @ Pm > 0 else -Vt[2]
    xs, ys = dl.colrow_to_xy(j0, i0)
    e1, e2 = rim_frame(np.array([xs]), np.array([ys]), nrm[None, :])
    e1, e2 = e1[0], e2[0]
    s, t = (P - Pm) @ e1, (P - Pm) @ e2
    keep = np.ones(s.size, bool)
    fit = None
    for _ in range(3):
        if keep.sum() < cfg.min_rim_points:
            return None, "few_points"
        u, v, A_, B_, psi, ok = fit_ellipse_direct(s[keep], t[keep])
        if not ok:
            return None, "fit"
        res = ellipse_residual(s, t, u, v, A_, B_, psi)
        fit = (u, v, A_, B_, psi, res, keep.copy())
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) <= cfg.mad_k * max(mad, 0.25 * dl.pixel)
        if np.array_equal(new, keep):
            break
        keep = new
    u, v, A_, B_, psi, res, keep = fit
    r_m = r0 * dl.pixel
    if not (cfg.size_consistency[0] <= np.sqrt(A_ * B_) / r_m <= cfg.size_consistency[1]):
        return None, "size"
    Xc = Pm + u * e1 + v * e2
    xc, yc, hc = frames.body_to_stereo(Xc)
    if np.hypot(xc - xs, yc - ys) > cfg.centre_consistency * r_m:
        return None, "centre"
    depth = float(np.mean(crest[valid][keep]) - inner)
    D = 2.0 * np.sqrt(A_ * B_)
    return {"X": Xc, "normal": nrm, "e1": e1, "e2": e2, "a": A_, "b": B_, "psi": psi,
            "x": float(xc), "y": float(yc), "h": float(hc), "depth": depth, "d_over_D": depth / D,
            "completeness": keep.sum() / th.size, "rim_h_std": float(np.std((P[keep] - Pm) @ nrm)),
            "fit_rms": float(np.sqrt(np.mean(res[keep] ** 2)))}, "ok"


def _dedupe(cat: CraterCatalog, cfg: CatalogConfig) -> CraterCatalog:
    if len(cat) < 2:
        return cat
    D = cat.D
    order = np.argsort(-(cat.completeness * np.abs(cat.seed_response) / D))
    tree = cKDTree(np.column_stack([cat.x, cat.y]))
    removed = np.zeros(len(cat), bool)
    for i in order:
        if removed[i]:
            continue
        for j in tree.query_ball_point([cat.x[i], cat.y[i]], cfg.dedupe_centre * D[i]):
            if j == i or removed[j]:
                continue
            dmin = min(D[i], D[j])
            if (np.hypot(cat.x[i] - cat.x[j], cat.y[i] - cat.y[j]) < cfg.dedupe_centre * dmin
                    and max(D[i], D[j]) / dmin < cfg.dedupe_size):
                removed[j] = True
    return cat.subset(~removed)


def extract_catalog(dem: Dem, cfg: CatalogConfig | None = None, core=None) -> CraterCatalog:
    """Crater catalog of a DEM window; ``core`` keeps centres in [xmin, xmax) x [ymin, ymax)."""
    cfg = cfg or CatalogConfig()
    recs: list[dict] = []
    for f in cfg.levels:
        base = dem
        if core is not None:  # only the part of the window that seeds in the core can reach
            m = level_margin_m(cfg, f, dem.pixel)
            base = _crop(dem, core[0] - m, core[1] + m, core[2] - m, core[3] + m, align=f)
        dl = base.downsample(f) if f > 1 else base
        if min(dl.shape) < 8 * max(cfg.radii_px):
            continue
        H, nan = _fill_nan(dl.heights)
        if nan.all():
            continue
        i0, j0, r0, resp = _seeds(H, dl.pixel, cfg)
        if nan.any() and i0.size:
            near = ndi.maximum_filter(nan, size=int(2 * cfg.detrend_annulus[1] * max(cfg.radii_px)) + 1)
            good = ~near[i0.astype(int), j0.astype(int)]
            i0, j0, r0, resp = i0[good], j0[good], r0[good], resp[good]
        for s0 in range(0, i0.size, cfg.chunk):
            sl = slice(s0, s0 + cfg.chunk)
            th, crest_rho, crest, valid, inner = _rim_candidates(H, i0[sl], j0[sl], r0[sl], cfg)
            for k in range(crest_rho.shape[0]):
                rec, _ = _fit_rim(dl, H, i0[sl][k], j0[sl][k], r0[sl][k], th, crest_rho[k], crest[k], valid[k],
                                  inner[k], cfg)
                if rec is not None:
                    rec["seed_response"] = float(resp[sl][k])
                    rec["level"] = int(f)
                    recs.append(rec)
    if recs:
        cat = make_catalog(np.array([r["X"] for r in recs]), np.array([r["normal"] for r in recs]),
                           np.array([r["e1"] for r in recs]), np.array([r["e2"] for r in recs]),
                           [r["a"] for r in recs], [r["b"] for r in recs], [r["psi"] for r in recs],
                           **{k: [r[k] for r in recs] for k in ("depth", "d_over_D", "completeness", "rim_h_std",
                                                                 "fit_rms", "seed_response", "level")})
    else:
        cat = CraterCatalog.empty()
    D = cat.D
    keep = (np.isfinite(D) & (D >= cfg.D_min_m) & (D <= cfg.D_max_m) & (cat.completeness >= cfg.completeness_min)
            & (cat.a / np.where(cat.b > 0, cat.b, np.nan) <= cfg.ellipticity_max) & (cat.d_over_D >= cfg.dD_min)
            & np.isfinite(cat.X).all(axis=1))
    cat = _dedupe(cat.subset(keep), cfg)
    if core is not None:
        xmin, xmax, ymin, ymax = core
        cat = cat.subset((cat.x >= xmin) & (cat.x < xmax) & (cat.y >= ymin) & (cat.y < ymax))
    cat.id = np.arange(len(cat), dtype=np.int64)
    cat.meta = {"dem": dem.name, "dem_pixel": float(dem.pixel), "config": asdict(cfg)}
    return cat
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: 8 passed. 실패하면 먼저 `cat`의 D·중심을 `truth`와 표로 찍어 어느 단계(씨앗/단면/적합/필터/중복 제거)에서 빠지는지 본다. 시험 기준(스펙 §10)을 낮추지 않는다. 기본값을 바꿔야 하면 그 이유를 Task 12의 decisions §G 초안에 메모해 둔다.

- [ ] **Step 5: 체크포인트** — 빠른 시험 전부 통과, `extract_catalog` 한 번 실행 시간(n=1600)을 기록해 둔다(Task 5 타일 크기 결정에 쓴다). 커밋하지 않는다.

---

### Task 5: 전 DEM 카탈로그 생성 (`tile_cores`, `build_catalog.py`)

**Files:**
- Modify: `src/lunar_trn/craters/catalog.py` (`tile_cores` 추가)
- Create: `scripts/build_catalog.py`
- Test: `tests/test_catalog.py` (시험 추가)

**Interfaces:**
- Consumes: `DemSource.read_window`, `DemSource.extent`, `extract_catalog`, `CraterCatalog.concat/save`
- Produces:
  - `tile_cores(extent, tile_m) -> list[(x0, x1, y0, y1)]` 반열린 코어, 마지막 코어의 상한은 `extent` 최댓값 + 1e-6
  - 파일 `data/catalog/ldem_875s_5m_craters.npz` (`meta`: dem, tile_m, margin_m, config, n, build_s)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog.py (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_catalog.py::test_tile_cores_partition_extent -v`
Expected: FAIL (`ImportError: cannot import name 'tile_cores'`)

- [ ] **Step 3a: Implement `tile_cores` (catalog.py)**

```python
# src/lunar_trn/craters/catalog.py (append)
def tile_cores(extent, tile_m: float) -> list[tuple[float, float, float, float]]:
    """Half-open tile cores [x0, x1) x [y0, y1) that partition ``extent`` = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = (float(v) for v in extent)
    xs = list(np.arange(xmin, xmax, tile_m)) or [xmin]
    ys = list(np.arange(ymin, ymax, tile_m)) or [ymin]
    out = []
    for x0 in xs:
        x1 = x0 + tile_m if x0 + tile_m < xmax else xmax + 1e-6
        for y0 in ys:
            y1 = y0 + tile_m if y0 + tile_m < ymax else ymax + 1e-6
            out.append((float(x0), float(x1), float(y0), float(y1)))
    return out
```

- [ ] **Step 3b: Implement `scripts/build_catalog.py`**

```python
# scripts/build_catalog.py
#!/usr/bin/env python3
"""Build the crater catalog of a whole DEM (tiled, parallel).

Each tile core is extracted from a window enlarged by a margin that holds the profiles of
the largest crater centred in the core, so tiles do not miss or duplicate craters.

    python scripts/build_catalog.py --dem data/dem/ldem_875s_5m.img \
        --out data/catalog/ldem_875s_5m_craters.npz --tile-km 10 --workers 32
    # development run on one area
    python scripts/build_catalog.py ... --bounds -20000 -10000 20000 30000 --out /tmp/cat_dev.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from lunar_trn.craters.catalog import CatalogConfig, CraterCatalog, extract_catalog, level_margin_m, tile_cores
from lunar_trn.dem import DemSource


def run_tile(job: dict):
    src = DemSource(job["dem"])
    x0, x1, y0, y1 = job["core"]
    m = job["margin_m"]
    try:
        dem = src.read_window(x0 - m, x1 + m, y0 - m, y1 + m)
    except ValueError:
        return None
    if not np.isfinite(dem.heights).any():
        return None
    return extract_catalog(dem, CatalogConfig(**job["cfg"]), core=job["core"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile-km", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    args = ap.parse_args(argv)
    cfg = CatalogConfig()
    src = DemSource(args.dem)
    extent = tuple(args.bounds) if args.bounds else src.extent
    # The coarsest level needs the widest margin; extract_catalog crops finer levels itself.
    margin = level_margin_m(cfg, max(cfg.levels), src.pixel)
    cores = tile_cores(extent, args.tile_km * 1000.0)
    jobs = [{"dem": args.dem, "core": c, "margin_m": margin, "cfg": dataclasses.asdict(cfg)} for c in cores]
    print(f"{len(jobs)} tiles, margin {margin:.0f} m", flush=True)
    t0 = time.time()
    cats = []
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=2) as pool:
        for n, cat in enumerate(pool.imap_unordered(run_tile, jobs), 1):
            if cat is not None and len(cat):
                cats.append(cat)
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} tiles, {sum(len(c) for c in cats)} craters, {time.time() - t0:.0f} s", flush=True)
    cat = CraterCatalog.concat(cats)
    order = np.lexsort((cat.x, -cat.y))
    cat = cat.subset(order)
    cat.id = np.arange(len(cat), dtype=np.int64)
    cat.meta = {"dem": src.name, "tile_m": args.tile_km * 1000.0, "margin_m": margin, "extent": list(extent),
                "config": dataclasses.asdict(cfg), "n": len(cat), "build_s": time.time() - t0}
    cat.save(args.out)
    D = cat.D
    print(f"saved {len(cat)} craters to {args.out}")
    if len(cat):
        print("D quantiles (m) 5/50/95/99:", np.round(np.percentile(D, [5, 50, 95, 99]), 1))
        for lv in np.unique(cat.level):
            print(f"  level {lv:3d}: {int(np.sum(cat.level == lv))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, then a timed development tile**

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: 9 passed

Run: `mkdir -p outputs/pg && .venv/bin/python scripts/build_catalog.py --dem data/dem/ldem_875s_5m.img --out outputs/pg/cat_dev.npz --bounds -10000 0 -10000 0 --workers 1 --tile-km 10`
Expected: 1 tile, 크레이터 수와 D 분위수 출력. 시간을 기록한다. 한 타일이 10분을 넘으면 `--tile-km 6`으로 줄여 다시 잰다.

- [ ] **Step 5: Build the full catalog**

Run: `mkdir -p data/catalog && .venv/bin/python scripts/build_catalog.py --dem data/dem/ldem_875s_5m.img --out data/catalog/ldem_875s_5m_craters.npz --tile-km 10 --workers 32 2>&1 | tee outputs/pg/build_catalog.log`
Expected: 모든 타일 처리, 크레이터 수 10⁴–10⁶ 범위, 수준별 개수 출력. 메모리가 부족하면 `--workers 16`.

- [ ] **Step 6: 체크포인트** — `outputs/pg/build_catalog.log`의 총수·D 분위수·실행 시간을 Task 15 보고서 메모로 옮겨 둔다. 커밋하지 않는다.

---

### Task 6: 검출기 1 — 고역 정규화와 안정 영역 (`detect.py` 앞부분)

**Files:**
- Create: `src/lunar_trn/craters/detect.py` (설정, `Regions`, `Detections`, `highpass`, `stable_regions`)
- Test: `tests/test_detect.py`

**Interfaces:**
- Produces:
  - `DetectorConfig` (필드는 아래 코드; decisions G5–G7)
  - `Regions` 필드 `m0, mx, my, mxx, mxy, myy, contrast, threshold` (각 (N,), 해당 피라미드 층의 화소 좌표 원모멘트), 속성 `.centroid (N,2)`, `len()`
  - `Detections` 필드 `u, v, A, B, theta, area_dark, area_bright, pair_dev_deg, contrast, level`, 속성 `.D_px`, `.uv (N,2)`, 메서드 `subset`, `empty()`, `concat(parts)`, `from_ellipses(u, v, A, B, theta)`
  - `highpass(image, cfg) -> ndarray` (국소 σ 단위)
  - `stable_regions(hp, polarity, cfg) -> Regions` (`polarity` +1 밝음, −1 어두움)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_detect.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_detect.py -v`
Expected: FAIL (`ModuleNotFoundError: lunar_trn.craters.detect`)

- [ ] **Step 3: Implement**

```python
# src/lunar_trn/craters/detect.py
"""CNav-style crater detector: stable dark/bright regions paired along the Sun direction.

Follows DLR CNav (Maass 2016; Trigo et al. 2018):

* "stable connected regions of above- and below-average intensity" from a threshold sweep,
  where stability means a small centroid movement between consecutive thresholds;
* each dark region is linked to its two nearest bright regions, excluding pairs where "the
  larger area is more than ten times the size of the smaller area";
* a pair must point away from the Sun (shadowed Sun-side wall -> lit anti-Sun wall);
* the crater ellipse is the principal-components ellipse "over the union of their
  constituent image support points".

Output image coordinates are continuous coordinates u = col + pixel_center. The thresholds
the papers leave open are in :class:`DetectorConfig` (decisions G5-G7).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..pose import project_uv


@dataclass
class DetectorConfig:
    """Detector settings (decisions G5-G7); kappa, beta, pair_angle_max_deg and D_min_px are calibrated."""

    pre_smooth_px: float = 1.0  # Gaussian pre-filter against pixel noise
    sigma_local_px: float = 25.0  # high-pass: local mean and std window (Gaussian sigma)
    std_floor: float = 0.02  # local std floor (image units) so flat or black areas do not explode
    thresholds: tuple = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)  # sweep, local-sigma units
    min_area_px: int = 3
    max_area_frac: float = 0.05
    stab_move_frac: float = 0.25  # stable: centroid move <= max(1 px, frac * equivalent radius)
    stab_area_ratio: float = 0.5  # and the next-threshold region keeps >= this area fraction
    pyramid_levels: int = 3
    level_bands_px: tuple = (32.0, 64.0)  # D_px limits between pyramid levels
    pair_neighbours: int = 2  # CNav: the two nearest bright regions
    pair_area_ratio_max: float = 10.0  # CNav
    pair_dist_k: float = 1.5  # centroid distance <= k * (sum of equivalent radii)
    pair_angle_max_deg: float = 40.0  # dark->bright vs anti-Sun direction (calibrated, G6)
    sun_source: str = "ephemeris"  # "ephemeris" (Sun + prior attitude) or "image" (CNav estimate)
    kappa: float = 1.0  # semi-axis scale of the moment ellipse (calibrated, G7)
    beta: float = 0.0  # centre shift towards anti-Sun, x semi-major axis (calibrated, G7)
    D_min_px: float = 6.0  # calibrated, G7
    D_max_px: float = 128.0
    dedupe_centre: float = 0.3
    dedupe_size: float = 1.5


@dataclass
class Regions:
    """Stable regions of one polarity at one pyramid level (raw moments in level pixels)."""

    m0: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    mxx: np.ndarray
    mxy: np.ndarray
    myy: np.ndarray
    contrast: np.ndarray  # mean |high-pass| over the region
    threshold: np.ndarray  # sweep threshold at which the region was taken

    def __len__(self) -> int:
        return int(self.m0.size)

    @property
    def centroid(self) -> np.ndarray:
        return np.column_stack([self.mx / self.m0, self.my / self.m0])


_DET_FIELDS = ("u", "v", "A", "B", "theta", "area_dark", "area_bright", "pair_dev_deg", "contrast", "level")


@dataclass
class Detections:
    u: np.ndarray  # ellipse centre, continuous image coordinates
    v: np.ndarray
    A: np.ndarray  # semi-major axis (px)
    B: np.ndarray  # semi-minor axis (px)
    theta: np.ndarray  # major-axis angle from +u towards +v (rad)
    area_dark: np.ndarray  # shadow region area (px)
    area_bright: np.ndarray  # lit region area (px)
    pair_dev_deg: np.ndarray  # dark->bright direction minus the anti-Sun direction (deg)
    contrast: np.ndarray  # mean |high-pass| of the pair (local-sigma units)
    level: np.ndarray  # pyramid level

    def __len__(self) -> int:
        return int(self.u.shape[0])

    @property
    def D_px(self) -> np.ndarray:
        return 2.0 * np.sqrt(self.A * self.B)

    @property
    def uv(self) -> np.ndarray:
        return np.column_stack([self.u, self.v])

    def subset(self, sel) -> "Detections":
        sel = np.asarray(sel)
        return Detections(**{k: getattr(self, k)[sel] for k in _DET_FIELDS})

    @classmethod
    def empty(cls) -> "Detections":
        return cls(**{k: np.zeros(0) for k in _DET_FIELDS})

    @classmethod
    def concat(cls, parts) -> "Detections":
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in _DET_FIELDS})

    @classmethod
    def from_ellipses(cls, u, v, A, B, theta) -> "Detections":
        """Detections given only as ellipses (injected detections, oracle tests)."""
        u = np.asarray(u, float).ravel()
        nan = np.full(u.size, np.nan)
        return cls(u=u, v=np.asarray(v, float).ravel(), A=np.asarray(A, float).ravel(),
                   B=np.asarray(B, float).ravel(), theta=np.asarray(theta, float).ravel(), area_dark=nan.copy(),
                   area_bright=nan.copy(), pair_dev_deg=nan.copy(), contrast=nan.copy(), level=np.zeros(u.size))


def highpass(image: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Image minus its local mean, divided by the floored local standard deviation."""
    I = np.nan_to_num(np.asarray(image, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if cfg.pre_smooth_px > 0:
        I = ndi.gaussian_filter(I, cfg.pre_smooth_px)
    mu = ndi.gaussian_filter(I, cfg.sigma_local_px)
    var = ndi.gaussian_filter(I * I, cfg.sigma_local_px) - mu * mu
    return (I - mu) / np.maximum(np.sqrt(np.maximum(var, 0.0)), cfg.std_floor)


def stable_regions(hp: np.ndarray, polarity: int, cfg: DetectorConfig) -> Regions:
    """CNav threshold sweep: the loosest stable region of every nested chain of components."""
    s = hp if polarity > 0 else -hp
    H, W = s.shape
    max_area = cfg.max_area_frac * H * W
    lev = []
    for t in cfg.thresholds:
        n, lab = cv2.connectedComponents((s > t).astype(np.uint8), connectivity=8)
        ys, xs = np.nonzero(lab)
        lbl = lab[ys, xs]
        xf, yf = xs.astype(np.float64), ys.astype(np.float64)

        def bc(w=None, lbl=lbl, n=n):
            return np.bincount(lbl, weights=w, minlength=n).astype(np.float64)

        lev.append({"n": n, "lab": lab, "l": lbl, "ys": ys, "xs": xs, "m0": bc(), "mx": bc(xf), "my": bc(yf),
                    "mxx": bc(xf * xf), "mxy": bc(xf * yf), "myy": bc(yf * yf), "sv": bc(np.abs(s[ys, xs]))})
    nl = len(lev)
    stable = [np.zeros(L["n"], bool) for L in lev]
    parent: list = [None] * nl
    for k in range(nl - 1):
        P, C = lev[k], lev[k + 1]
        par = np.zeros(C["n"], np.int64)
        par[C["l"]] = P["lab"][C["ys"], C["xs"]]  # nested thresholds: one parent per child
        parent[k + 1] = par
        if C["n"] <= 1:
            continue
        kids = np.arange(1, C["n"])
        kids = kids[np.argsort(C["m0"][kids], kind="stable")]
        best = np.zeros(P["n"], np.int64)
        best[par[kids]] = kids  # the largest child is written last
        pk = np.nonzero(best > 0)[0]
        pk = pk[pk > 0]
        ck = best[pk]
        pa, ca = P["m0"][pk], C["m0"][ck]
        move = np.hypot(C["mx"][ck] / ca - P["mx"][pk] / pa, C["my"][ck] / ca - P["my"][pk] / pa)
        ok = ((move <= np.maximum(1.0, cfg.stab_move_frac * np.sqrt(pa / np.pi))) & (ca / pa >= cfg.stab_area_ratio)
              & (pa >= cfg.min_area_px) & (pa <= max_area))
        stable[k][pk[ok]] = True
    keys = ("m0", "mx", "my", "mxx", "mxy", "myy", "sv")
    acc: dict = {key: [] for key in keys}
    thr = []
    for k in range(nl):
        st = np.nonzero(stable[k])[0]
        if k > 0 and st.size:
            st = st[~stable[k - 1][parent[k][st]]]  # a stable parent supersedes its child
        for key in keys:
            acc[key].append(lev[k][key][st])
        thr.append(np.full(st.size, float(cfg.thresholds[k])))
    cat = {key: (np.concatenate(v) if v else np.zeros(0)) for key, v in acc.items()}
    return Regions(m0=cat["m0"], mx=cat["mx"], my=cat["my"], mxx=cat["mxx"], mxy=cat["mxy"], myy=cat["myy"],
                   contrast=cat["sv"] / np.maximum(cat["m0"], 1.0),
                   threshold=np.concatenate(thr) if thr else np.zeros(0))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_detect.py -v`
Expected: 3 passed

- [ ] **Step 5: 체크포인트** — 빠른 시험 전부 통과. 커밋하지 않는다.

---

### Task 7: 검출기 2 — 짝짓기, 조명 추정, `detect_craters`

**Files:**
- Modify: `src/lunar_trn/craters/detect.py` (끝에 추가)
- Test: `tests/test_detect.py` (시험 추가)

**Interfaces:**
- Consumes: Task 6의 `DetectorConfig`, `Regions`, `Detections`, `highpass`, `stable_regions`; `pose.project_uv`
- Produces:
  - `pair_regions(dark, bright, cfg) -> (i_dark, j_bright, angle_rad)` 후보 변(방향 검사 전)
  - `estimate_illumination(angles_rad) -> (mu_rad, sigma_rad)` CNav 영상 추정(어두운→밝은 우세 방향 = 반태양)
  - `pinhole_sun_fn(cam, T_world_cam, sun_dir_world, z0) -> fn(u, v) -> (N,2)` 태양 쪽 영상 단위벡터(정의 안 되면 NaN)
  - `constant_sun_fn(angle_deg) -> fn(u, v) -> (N,2)`
  - `detect_craters(image, sun_fn, cfg=None, pixel_center=0.5) -> (Detections, health: dict)` — health 키: `n_dark_regions, n_bright_regions, n_candidate_edges, n_pairs, n_detections, sun_img_angle_deg, sun_img_angle_est_deg, sun_est_sigma_deg, det_D_px_p10, det_D_px_p50, det_D_px_p90`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_detect.py (append)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_detect.py -v`
Expected: 새 시험 5개 FAIL (`ImportError: cannot import name 'constant_sun_fn'`)

- [ ] **Step 3: Implement (detect.py 끝에 추가)**

```python
# src/lunar_trn/craters/detect.py (append)
def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi


def pair_regions(dark: Regions, bright: Regions, cfg: DetectorConfig):
    """CNav candidate edges: each dark region to its nearest bright regions (area-ratio and distance gated)."""
    if len(dark) == 0 or len(bright) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    pd, pb = dark.centroid, bright.centroid
    k = min(cfg.pair_neighbours, len(bright))
    dist, nb = cKDTree(pb).query(pd, k=k)
    dist = np.asarray(dist, float).reshape(len(pd), k).ravel()
    j = np.asarray(nb).reshape(len(pd), k).ravel()
    i = np.repeat(np.arange(len(pd)), k)
    ad, ab = dark.m0[i], bright.m0[j]
    ok = np.maximum(ad, ab) / np.minimum(ad, ab) <= cfg.pair_area_ratio_max
    ok &= dist <= cfg.pair_dist_k * (np.sqrt(ad / np.pi) + np.sqrt(ab / np.pi))
    vec = pb[j] - pd[i]
    return i[ok], j[ok], np.arctan2(vec[ok, 1], vec[ok, 0])


def estimate_illumination(angles) -> tuple[float, float]:
    """CNav: direction minimising the angular variance inside a pi-wide window of edge directions.

    ``angles`` are dark->bright edge directions; the result ``mu`` points away from the Sun.
    Returns (nan, nan) for fewer than three edges.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan"), float("nan")
    grid = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    dev = _wrap(a[None, :] - grid[:, None])
    inw = np.abs(dev) < np.pi / 2
    cnt = inw.sum(axis=1)
    mean = np.where(cnt > 0, (dev * inw).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 0, (((dev - mean[:, None]) ** 2) * inw).sum(axis=1) / np.maximum(cnt, 1), np.inf)
    var[cnt < max(3, 0.5 * a.size)] = np.inf
    g = int(np.argmin(var))
    if not np.isfinite(var[g]):
        return float("nan"), float("nan")
    return float(_wrap(grid[g] + mean[g])), float(np.sqrt(var[g]))


def pinhole_sun_fn(cam, T_world_cam: np.ndarray, sun_dir_world, z0: float):
    """Image direction towards the Sun at image points, for a camera with known attitude.

    A ground point at camera depth ``z0`` behind each pixel is moved slightly towards the
    Sun and both points are projected; this keeps the per-pixel perspective of the Sun
    direction (shadows converge to the Sun's vanishing point). Undefined -> NaN.
    """
    K = np.asarray(cam.K, float)
    T = np.asarray(T_world_cam, float)
    R, c = T[:3, :3], T[:3, 3]
    s = np.asarray(sun_dir_world, float)
    s = s / np.linalg.norm(s)
    Kinv = np.linalg.inv(K)

    def fn(u, v):
        u = np.atleast_1d(np.asarray(u, float))
        v = np.atleast_1d(np.asarray(v, float))
        d = (np.column_stack([u, v, np.ones_like(u)]) @ Kinv.T) @ R.T  # world rays with unit camera depth
        X1 = c + z0 * d
        p1, _ = project_uv(K, R, c, X1)
        p2, _ = project_uv(K, R, c, X1 + 0.01 * z0 * s)
        g = p2 - p1
        nrm = np.linalg.norm(g, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(nrm > 1e-6, g / nrm, np.nan)

    return fn


def constant_sun_fn(angle_deg: float):
    """Same image Sun direction everywhere (orthographic maps, procedural tests)."""
    vec = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))])
    return lambda u, v: np.tile(vec, (np.size(u), 1))


def _one_to_one(i, j, dev):
    keep, used_i, used_j = [], set(), set()
    for k in np.argsort(dev, kind="stable"):
        if i[k] in used_i or j[k] in used_j:
            continue
        used_i.add(i[k])
        used_j.add(j[k])
        keep.append(k)
    keep = np.asarray(keep, int)
    return i[keep], j[keep], dev[keep]


def _dedupe(det: Detections, cfg: DetectorConfig) -> Detections:
    if len(det) < 2:
        return det
    D = det.D_px
    tree = cKDTree(det.uv)
    removed = np.zeros(len(det), bool)
    for i in np.argsort(det.pair_dev_deg, kind="stable"):
        if removed[i]:
            continue
        for j in tree.query_ball_point(det.uv[i], cfg.dedupe_centre * D[i]):
            if j == i or removed[j]:
                continue
            dmin = min(D[i], D[j])
            if (np.hypot(*(det.uv[i] - det.uv[j])) < cfg.dedupe_centre * dmin
                    and max(D[i], D[j]) / dmin < cfg.dedupe_size):
                removed[j] = True
    return det.subset(~removed)


def detect_craters(image: np.ndarray, sun_fn, cfg: DetectorConfig | None = None,
                   pixel_center: float = 0.5) -> tuple[Detections, dict]:
    """Crater ellipses in ``image``; ``sun_fn(u, v)`` gives unit image vectors towards the Sun."""
    cfg = cfg or DetectorConfig()
    img = np.nan_to_num(np.asarray(image, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    H, W = img.shape
    n_lev = 0
    while n_lev < cfg.pyramid_levels and min(H, W) // (2**n_lev) >= 32:
        n_lev += 1
    edges = [cfg.D_min_px, *list(cfg.level_bands_px)[: max(n_lev - 1, 0)], cfg.D_max_px]
    health = {"n_dark_regions": 0, "n_bright_regions": 0, "n_candidate_edges": 0, "n_pairs": 0}
    mu, sig = float("nan"), float("nan")
    parts = []
    for lvl in range(n_lev):
        f = 2**lvl
        im = img if f == 1 else cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
        hp = highpass(im, cfg)
        dark, bright = stable_regions(hp, -1, cfg), stable_regions(hp, +1, cfg)
        health["n_dark_regions"] += len(dark)
        health["n_bright_regions"] += len(bright)
        i, j, ang = pair_regions(dark, bright, cfg)
        health["n_candidate_edges"] += int(i.size)
        if lvl == 0:
            mu, sig = estimate_illumination(ang)
        if i.size == 0:
            continue
        off = (f - 1) / 2.0 + pixel_center  # level pixel index -> full-resolution continuous coordinate
        if cfg.sun_source == "image":
            dev, lim = np.abs(_wrap(ang - mu)), sig
        else:
            cd = dark.centroid[i]
            s = sun_fn(f * cd[:, 0] + off, f * cd[:, 1] + off)
            dev, lim = np.abs(_wrap(ang - np.arctan2(-s[:, 1], -s[:, 0]))), np.deg2rad(cfg.pair_angle_max_deg)
        ok = np.isfinite(dev) & (dev <= lim)
        i, j, dev = _one_to_one(i[ok], j[ok], dev[ok])
        health["n_pairs"] += int(i.size)
        if i.size == 0:
            continue
        m0 = dark.m0[i] + bright.m0[j]
        cx = (dark.mx[i] + bright.mx[j]) / m0
        cy = (dark.my[i] + bright.my[j]) / m0
        sc2 = float(f * f)
        cxx = ((dark.mxx[i] + bright.mxx[j]) / m0 - cx * cx) * sc2 + (sc2 - 1.0) / 12.0
        cyy = ((dark.myy[i] + bright.myy[j]) / m0 - cy * cy) * sc2 + (sc2 - 1.0) / 12.0
        cxy = ((dark.mxy[i] + bright.mxy[j]) / m0 - cx * cy) * sc2
        half = 0.5 * (cxx + cyy)
        dlt = np.sqrt(np.maximum(0.25 * (cxx - cyy) ** 2 + cxy**2, 0.0))
        A = cfg.kappa * 2.0 * np.sqrt(half + dlt)
        B = cfg.kappa * 2.0 * np.sqrt(np.maximum(half - dlt, 1e-12))
        theta = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
        u, v = f * cx + off, f * cy + off
        if cfg.sun_source == "image":
            s2 = np.tile([-np.cos(mu), -np.sin(mu)], (u.size, 1))
        else:
            s2 = sun_fn(u, v)
        u = u - cfg.beta * A * s2[:, 0]  # shift towards anti-Sun: the shadow side biases the centroid
        v = v - cfg.beta * A * s2[:, 1]
        D = 2.0 * np.sqrt(A * B)
        upper = (D <= edges[lvl + 1]) if lvl == n_lev - 1 else (D < edges[lvl + 1])
        keep = np.isfinite(u) & np.isfinite(v) & (D >= edges[lvl]) & upper
        parts.append(Detections(u=u[keep], v=v[keep], A=A[keep], B=B[keep], theta=theta[keep],
                                area_dark=(dark.m0[i] * sc2)[keep], area_bright=(bright.m0[j] * sc2)[keep],
                                pair_dev_deg=np.rad2deg(dev)[keep],
                                contrast=(0.5 * (dark.contrast[i] + bright.contrast[j]))[keep],
                                level=np.full(int(keep.sum()), float(lvl))))
    det = _dedupe(Detections.concat(parts), cfg)
    centre = (W - 1) / 2.0 + pixel_center, (H - 1) / 2.0 + pixel_center
    sc = sun_fn(np.array([centre[0]]), np.array([centre[1]]))[0]
    health.update(n_detections=len(det), sun_img_angle_deg=float(np.degrees(np.arctan2(sc[1], sc[0]))),
                  sun_img_angle_est_deg=float(np.degrees(_wrap(mu + np.pi))) if np.isfinite(mu) else float("nan"),
                  sun_est_sigma_deg=float(np.degrees(sig)) if np.isfinite(sig) else float("nan"))
    q = np.percentile(det.D_px, [10, 50, 90]) if len(det) else [float("nan")] * 3
    health.update(det_D_px_p10=float(q[0]), det_D_px_p50=float(q[1]), det_D_px_p90=float(q[2]))
    return det, health
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_detect.py -v`
Expected: 8 passed. 절차적 영상에서 실패하면 `stable_regions`가 낸 영역과 짝을 그림으로 찍어(`matplotlib`, 스크래치패드) 어느 단계에서 빠지는지 본다.

- [ ] **Step 5: 체크포인트** — 빠른 시험 전부 통과. 커밋하지 않는다.

---

### Task 8: 정답 주석, 합성 장면, 실데이터 첫 확인 그림

**Files:**
- Create: `src/lunar_trn/craters/annotate.py`
- Modify: `src/lunar_trn/craters/synthetic.py` (평면 카탈로그·천저 자세·카메라·합성 장면 추가)
- Create: `scripts/pg_figures.py` (`overlay`, `catalog-dem`, `csfd` 부명령)
- Test: `tests/test_annotate.py`, `tests/test_detect.py` (느린 시험 추가)

**Interfaces:**
- Consumes: `conics.project_rims`, `catalog.make_catalog/rim_frame/extract_catalog`, `detect.*`, `render.render_perspective`, `validation.v1.sun_from_az_el`
- Produces:
  - `annotate.project_catalog(catalog, cam, T_world_cam, margin_px=0.0) -> dict` (`project_rims` 출력 + `in_image`: 타원 전체가 영상 안)
  - `annotate.annotate_detections(det, catalog, cam, T_true, sun_fn, tol_abs_px=2.0, tol_rel=0.25, size_ratio_max=1.5) -> dict` 키 `cat_index, cat_id, err_u, err_v, err_antisun_px, err_perp_px, size_ratio, D_px_pred, anti_u, anti_v` (각 (N_det,)); 오차 = 예측 − 검출
  - `annotate.catalog_recall(det, catalog, cam, T_true, sun_dir_world, tol_abs_px=2.0, tol_rel=0.25, size_ratio_max=1.5, D_px_min=4.0) -> dict` 키 `cat_id, D_px, D_m, d_over_D, completeness, sun_el_deg, detected`
  - `synthetic.flat_catalog(rng, n, extent, D_range, h=0.0) -> CraterCatalog`
  - `synthetic.nadir_pose(cx, cy, alt) -> T (4,4)`, `synthetic.default_camera(size=512) -> PinholeCamera`
  - `synthetic.render_synthetic_scene(alt=4000.0, sun_az=135.0, sun_el=20.0, n=1400, n_craters=150, D_range=(60.0, 600.0), seed=0, size=512) -> dict(dem, craters, cam, T, sun, image)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_annotate.py
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
```

```python
# tests/test_detect.py (append)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_annotate.py tests/test_detect.py -v`
Expected: FAIL (`ModuleNotFoundError: lunar_trn.craters.annotate`)

- [ ] **Step 3a: Implement synthetic additions (synthetic.py 끝에 추가)**

```python
# src/lunar_trn/craters/synthetic.py (append)
from .. import frames  # noqa: E402
from ..camera import PinholeCamera  # noqa: E402
from .catalog import CraterCatalog, make_catalog, rim_frame  # noqa: E402


def flat_catalog(rng, n: int, extent, D_range, h: float = 0.0) -> CraterCatalog:
    """Circular rims on the reference sphere (height ``h``), non-overlapping, log-uniform sizes."""
    cr = place_craters(rng, n, extent, D_range, (0.1, 0.2), margin_r=0.0, spacing=1.05)
    x = np.array([c["x"] for c in cr])
    y = np.array([c["y"] for c in cr])
    D = np.array([c["D"] for c in cr])
    X = frames.stereo_to_body(x, y, np.full(x.shape, h))
    nrm = X / np.linalg.norm(X, axis=1, keepdims=True)
    e1, e2 = rim_frame(x, y, nrm)
    return make_catalog(X, nrm, e1, e2, D / 2, D / 2, np.zeros(x.size),
                        d_over_D=[c["dD"] for c in cr], completeness=np.ones(x.size))


def nadir_pose(cx: float, cy: float, alt: float) -> np.ndarray:
    """Camera-to-world pose looking straight down from ``alt`` above stereographic (cx, cy)."""
    p = frames.stereo_to_body(cx, cy, alt)
    down = -p / np.linalg.norm(p)
    xc = np.cross(down, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([xc, np.cross(down, xc), down])
    T[:3, 3] = p
    return T


def default_camera(size: int = 512) -> PinholeCamera:
    """StereoLunar-like camera: 45 deg field of view, principal point at the image centre."""
    f = size / 2 / np.tan(np.deg2rad(22.5))
    return PinholeCamera(np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]), size, size)


def render_synthetic_scene(alt: float = 4000.0, sun_az: float = 135.0, sun_el: float = 20.0, n: int = 1400,
                           n_craters: int = 150, D_range=(60.0, 600.0), seed: int = 0, size: int = 512) -> dict:
    """Hapke render of a synthetic crater DEM from a nadir camera (min-max normalised like StereoLunar)."""
    from ..render import render_perspective
    from ..validation.v1 import sun_from_az_el

    cx, cy = -20000.0, 30000.0
    dem, craters = synthetic_crater_dem(n=n, cx=cx, cy=cy, n_craters=n_craters, D_range=D_range,
                                        dD_range=(0.12, 0.2), noise_m=0.2, seed=seed)
    cam, T = default_camera(size), nadir_pose(cx, cy, alt)
    sun = sun_from_az_el(T[:3, 3], sun_az, sun_el)
    img = render_perspective(dem, cam, T, sun, step=1, model="hapke")["image"]
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
    return {"dem": dem, "craters": craters, "cam": cam, "T": T, "sun": sun,
            "image": np.nan_to_num(img).astype(np.float32)}
```

- [ ] **Step 3b: Implement `src/lunar_trn/craters/annotate.py`**

```python
# src/lunar_trn/craters/annotate.py
"""Ground-truth annotation of crater detections (calibration and diagnostics only).

Catalog rims are projected with the true pose; a detection corresponds to the nearest
predicted ellipse centre within max(tol_abs_px, tol_rel * D_px) whose size ratio is within
``size_ratio_max``. Errors are prediction minus detection, in pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .catalog import CraterCatalog
from .conics import project_rims
from .detect import Detections


def project_catalog(catalog: CraterCatalog, cam, T_world_cam: np.ndarray, margin_px: float = 0.0) -> dict:
    T = np.asarray(T_world_cam, float)
    p = project_rims(cam.K, T[:3, :3], T[:3, 3], catalog.X, catalog.e1, catalog.e2, catalog.a, catalog.b,
                     catalog.psi)
    lo = cam.pixel_center - 0.5 + margin_px
    with np.errstate(invalid="ignore"):
        p["in_image"] = (p["valid"] & (p["u"] - p["A"] >= lo) & (p["u"] + p["A"] <= cam.width + lo - 2 * margin_px)
                         & (p["v"] - p["A"] >= lo) & (p["v"] + p["A"] <= cam.height + lo - 2 * margin_px))
    return p


def _nearest_ok(query_uv, query_D, ref_uv, ref_D, tol_abs_px, tol_rel, size_ratio_max):
    """Index into ref of the nearest compatible neighbour of each query point (-1 if none)."""
    n = len(query_uv)
    out = np.full(n, -1, np.int64)
    if n == 0 or len(ref_uv) == 0:
        return out
    k = min(8, len(ref_uv))
    dist, nb = cKDTree(ref_uv).query(query_uv, k=k)
    dist = np.asarray(dist).reshape(n, k)
    nb = np.asarray(nb).reshape(n, k)
    tol = np.maximum(tol_abs_px, tol_rel * query_D)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(ref_D[nb] / query_D[:, None]))
    ok = (dist <= tol) & (lr <= np.log(size_ratio_max))
    has = ok.any(axis=1)
    out[has] = nb[has, np.argmax(ok[has], axis=1)]
    return out


def annotate_detections(det: Detections, catalog: CraterCatalog, cam, T_true: np.ndarray, sun_fn,
                        tol_abs_px: float = 2.0, tol_rel: float = 0.25, size_ratio_max: float = 1.5) -> dict:
    n = len(det)
    nan = np.full(n, np.nan)
    out = {"cat_index": np.full(n, -1, np.int64), "cat_id": np.full(n, -1, np.int64), "err_u": nan.copy(),
           "err_v": nan.copy(), "err_antisun_px": nan.copy(), "err_perp_px": nan.copy(), "size_ratio": nan.copy(),
           "D_px_pred": nan.copy(), "anti_u": nan.copy(), "anti_v": nan.copy()}
    if n == 0 or len(catalog) == 0:
        return out
    p = project_catalog(catalog, cam, T_true)
    cand = np.nonzero(p["valid"])[0]
    if cand.size == 0:
        return out
    ref = _nearest_ok(det.uv, det.D_px, np.column_stack([p["u"][cand], p["v"][cand]]), p["D_px"][cand],
                      tol_abs_px, tol_rel, size_ratio_max)
    has = ref >= 0
    idx = cand[ref[has]]
    out["cat_index"][has] = idx
    out["cat_id"][has] = catalog.id[idx]
    out["err_u"][has] = p["u"][idx] - det.u[has]
    out["err_v"][has] = p["v"][idx] - det.v[has]
    out["size_ratio"][has] = p["D_px"][idx] / det.D_px[has]
    out["D_px_pred"][has] = p["D_px"][idx]
    anti = -sun_fn(det.u, det.v)
    out["anti_u"], out["anti_v"] = anti[:, 0], anti[:, 1]
    out["err_antisun_px"] = out["err_u"] * anti[:, 0] + out["err_v"] * anti[:, 1]
    out["err_perp_px"] = -out["err_u"] * anti[:, 1] + out["err_v"] * anti[:, 0]
    return out


def catalog_recall(det: Detections, catalog: CraterCatalog, cam, T_true: np.ndarray, sun_dir_world,
                   tol_abs_px: float = 2.0, tol_rel: float = 0.25, size_ratio_max: float = 1.5,
                   D_px_min: float = 4.0) -> dict:
    p = project_catalog(catalog, cam, T_true)
    sel = np.nonzero(p["in_image"] & (p["D_px"] >= D_px_min))[0]
    D = p["D_px"][sel]
    uv = np.column_stack([p["u"][sel], p["v"][sel]])
    detected = _nearest_ok(uv, D, det.uv, det.D_px, tol_abs_px, tol_rel, size_ratio_max) >= 0
    s = np.asarray(sun_dir_world, float)
    s = s / np.linalg.norm(s)
    return {"cat_id": catalog.id[sel], "D_px": D, "D_m": catalog.D[sel], "d_over_D": catalog.d_over_D[sel],
            "completeness": catalog.completeness[sel],
            "sun_el_deg": np.degrees(np.arcsin(np.clip(catalog.normal[sel] @ s, -1, 1))), "detected": detected}
```

- [ ] **Step 3c: Implement `scripts/pg_figures.py` (`overlay`, `catalog-dem`, `csfd`)**

그림 색은 `scripts/make_figures.py`와 같은 팔레트·바탕을 쓴다(카탈로그 = S1 파랑, 검출 = S2 주황). 두 계열이 있는 그림은 범례를 둔다.

```python
# scripts/pg_figures.py
#!/usr/bin/env python3
"""Figures for the P-G (crater) baseline.

    python scripts/pg_figures.py overlay --data data/stereolunar/nadir \
        --catalog data/catalog/ldem_875s_5m_craters.npz --geometries 13740 2640 4200 11220 --out docs/validation/fig
    python scripts/pg_figures.py catalog-dem --dem data/dem/ldem_875s_5m.img \
        --catalog data/catalog/ldem_875s_5m_craters.npz --box -10000 -5000 -10000 -5000 --out docs/validation/fig
    python scripts/pg_figures.py csfd --catalog data/catalog/ldem_875s_5m_craters.npz --out docs/validation/fig
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from lunar_trn.craters.annotate import project_catalog  # noqa: E402
from lunar_trn.craters.catalog import CraterCatalog  # noqa: E402
from lunar_trn.craters.detect import DetectorConfig, detect_craters, pinhole_sun_fn  # noqa: E402
from lunar_trn.datasets import stereolunar as sl  # noqa: E402
from lunar_trn.validation.v1 import footprint_bounds  # noqa: E402

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (same as make_figures.py)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)


def _ellipses(ax, u, v, A, B, th, color, pc, label):
    for k in range(len(u)):
        ax.add_patch(Ellipse((u[k] - pc, v[k] - pc), 2 * A[k], 2 * B[k], angle=np.degrees(th[k]), fill=False,
                             lw=1.0, ec=color, label=label if k == 0 else None))


def cmd_overlay(args):
    cat = CraterCatalog.load(args.catalog)
    frames_ = sl.list_frames(args.data)
    groups = {g[0].index: g for g in sl.assign_triplets(frames_) if g[0].illumination is not None}
    args.out.mkdir(parents=True, exist_ok=True)
    for gid in args.geometries:
        g = groups[gid]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        for ax, f in zip(axes, g):
            cam, T, img = f.camera(), f.T_world_cam, f.image()
            xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
            sub = cat.query_box(xmin, xmax, ymin, ymax)
            p = project_catalog(sub, cam, T)
            sel = p["in_image"] & (p["D_px"] >= args.min_px)
            alt = float(np.nanmedian(f.depth()[250:262, 250:262]))
            det, h = detect_craters(img, pinhole_sun_fn(cam, T, f.sun_direction(), alt), DetectorConfig(),
                                    pixel_center=cam.pixel_center)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            _ellipses(ax, p["u"][sel], p["v"][sel], p["A"][sel], p["B"][sel], p["theta"][sel], S1, cam.pixel_center,
                      f"catalog (true pose), {int(sel.sum())}")
            _ellipses(ax, det.u, det.v, det.A, det.B, det.theta, S2, cam.pixel_center, f"detections, {len(det)}")
            ax.set_xlim(-0.5, cam.width - 0.5)
            ax.set_ylim(cam.height - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"im {f.index}  illum {f.illumination}  alt {alt / 1000:.1f} km", color=TEXT, fontsize=10,
                         loc="left")
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower left")
        fig.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(args.out / f"pg_overlay_{gid}.png", dpi=110)
        plt.close(fig)


def cmd_catalog_dem(args):
    from lunar_trn.dem import DemSource

    cat = CraterCatalog.load(args.catalog)
    x0, x1, y0, y1 = args.box
    d = DemSource(args.dem).read_window(x0, x1, y0, y1)
    gy, gx = np.gradient(d.heights.astype(float), d.pixel)
    hs = np.clip(0.5 + 1.5 * (0.6 * gx - 0.6 * gy), 0, 1)
    sub = cat.query_box(x0, x1, y0, y1)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hs, cmap="gray", extent=(d.x0, d.x0 + d.shape[1] * d.pixel, d.y0 - d.shape[0] * d.pixel, d.y0))
    for lo, hi, col, lab in ((0, 80, S2, "D < 80 m"), (80, 200, S3, "80-200 m"), (200, 1e9, S1, ">= 200 m")):
        s = (sub.D >= lo) & (sub.D < hi)
        for k, i in enumerate(np.nonzero(s)[0]):
            ax.add_patch(Ellipse((sub.x[i], sub.y[i]), sub.D[i], sub.D[i], fill=False, lw=0.9, ec=col,
                                 label=f"{lab}, {int(s.sum())}" if k == 0 else None))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    style(ax, f"DEM catalog over hillshade ({len(sub)} craters)", "stereographic x (m)", "stereographic y (m)")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower left")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out / "pg_catalog_dem.png", dpi=110)
    plt.close(fig)


def cmd_csfd(args):
    cat = CraterCatalog.load(args.catalog)
    xmin, xmax, ymin, ymax = cat.meta.get("extent", [cat.x.min(), cat.x.max(), cat.y.min(), cat.y.max()])
    area_km2 = (xmax - xmin) * (ymax - ymin) / 1e6
    D = np.sort(cat.D)[::-1] / 1000.0
    N = np.arange(1, D.size + 1) / area_km2
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(D, N, color=S1, lw=2)
    ref = N[np.searchsorted(-D, -0.2)] * (D / 0.2) ** -2.0 if D.size else D
    ax.loglog(D, ref, color=TEXT2, lw=1, ls="--")
    ax.annotate("D^-2 (equilibrium)", (D[-1], ref[-1]), textcoords="offset points", xytext=(4, 4), color=TEXT2,
                fontsize=8)
    style(ax, f"Catalog cumulative size-frequency ({D.size} craters, {area_km2:.0f} km^2)", "D (km)",
          "N(>= D) per km^2")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out / "pg_csfd.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overlay")
    o.add_argument("--data", type=Path, required=True)
    o.add_argument("--catalog", required=True)
    o.add_argument("--geometries", type=int, nargs="+", required=True)
    o.add_argument("--min-px", type=float, default=6.0)
    o.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("catalog-dem")
    c.add_argument("--dem", required=True)
    c.add_argument("--catalog", required=True)
    c.add_argument("--box", type=float, nargs=4, required=True)
    c.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("csfd")
    f.add_argument("--catalog", required=True)
    f.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    {"overlay": cmd_overlay, "catalog-dem": cmd_catalog_dem, "csfd": cmd_csfd}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_annotate.py -v && .venv/bin/python -m pytest tests/test_detect.py -v -m slow`
Expected: annotate 2 passed, 느린 검출 시험 1 passed(렌더 포함 수십 초)

- [ ] **Step 5: 실데이터 첫 확인 (그림)**

Run:
```bash
.venv/bin/python scripts/pg_figures.py overlay --data data/stereolunar/nadir \
    --catalog data/catalog/ldem_875s_5m_craters.npz --geometries 13740 2640 4200 11220 --out outputs/pg/fig
.venv/bin/python scripts/pg_figures.py catalog-dem --dem data/dem/ldem_875s_5m.img \
    --catalog data/catalog/ldem_875s_5m_craters.npz --box -10000 -5000 -10000 -5000 --out outputs/pg/fig
.venv/bin/python scripts/pg_figures.py csfd --catalog data/catalog/ldem_875s_5m_craters.npz --out outputs/pg/fig
```
Expected: 그림 6장. 각 그림을 열어 본다(Read 도구로 PNG). 확인할 것:
1. 큰 크레이터의 카탈로그 타원(파랑)이 영상의 테두리와 겹치는가(스펙 M1 완료 조건). 체계적으로 어긋나면 좌표 변환(Task 2–4)을 의심한다.
2. 검출(주황)이 영상 크레이터에 붙는가, 그리고 카탈로그와 얼마나 겹치는가(M2).
3. `pg_catalog_dem.png`에서 D < 80 m 항목이 LOLA 궤적선을 따라 줄지어 있는 정도. 관찰을 Task 12의 decisions G3 근거에 적는다.
4. `pg_csfd.png`가 D⁻² 근처인지, 작은 쪽에서 어디서 꺾이는지.

- [ ] **Step 6: 체크포인트** — 빠른 시험 전부 통과, 그림은 `outputs/pg/fig/`에만 있다(문서용 복사는 Task 15). 커밋하지 않는다.

---

### Task 9: 추적 매칭 (`match.py`)

**Files:**
- Create: `src/lunar_trn/craters/match.py`
- Test: `tests/test_match.py`

**Interfaces:**
- Consumes: `CraterCatalog.query_box/subset/D`, `conics.project_rims`, `Detections`, `frames.enu_basis`, `validation.v1.footprint_bounds`
- Produces:
  - `MatchConfig` (필드는 아래 코드; decisions G8–G11. `catalog_D_min_m`은 H-c 민감도용 런타임 카탈로그 하한)
  - `Candidates(catalog, pred, n_window)`
  - `predict(catalog, K, R_wc, c) -> dict` (`project_rims` 출력)
  - `select_candidates(catalog, cam, T_prior, sigma_h, sun_dir_world, D_px_range, cfg) -> Candidates`
  - `vote_positions(det, cand, cam, T_prior, sigma_h, cfg) -> dict` 키 `offsets (K,2)` (사전 위치 ENU의 동·북 m), `votes (K,)`, `counts (K,)`, `n_pairs`, `bin_m`, `radius_m`
  - `assign(det_uv, det_D, pred_uv, pred_D, gate_px, gate_rel, size_ratio_max) -> (i_det, j_pred, dist_px)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_match.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_match.py -v`
Expected: FAIL (`ModuleNotFoundError: lunar_trn.craters.match`)

- [ ] **Step 3: Implement**

```python
# src/lunar_trn/craters/match.py
"""Tracking-mode crater matching: catalog prediction, position-hypothesis voting, assignment.

With the attitude known and a narrow vertical prior, one (detection, catalog crater) pair
fixes a horizontal camera-position hypothesis: the camera lies on the detection ray through
the crater centre, at the prior height. Hypotheses of size-compatible pairs are voted on a
horizontal grid around the prior; the strongest peaks are verified by gated one-to-one
assignment and the pose back end (:mod:`lunar_trn.craters.localize`). Thresholds the papers
leave open are in :class:`MatchConfig` (decisions G8-G11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .. import frames
from ..validation.v1 import footprint_bounds
from .catalog import CraterCatalog
from .conics import project_rims
from .detect import Detections


@dataclass
class MatchConfig:
    search_margin_m: float = 300.0  # search radius = 3 sigma_h + margin (same rule as T-I coarse, C3)
    catalog_D_min_m: float = 0.0  # run-time catalog size floor, 0 = as built (H-c sensitivity)
    size_pad: tuple = (0.7, 1.3)  # keep candidates whose predicted D_px is within pad x the detector range
    size_ratio_vote: float = 1.6  # size compatibility of a voting pair
    vote_bin_gsd: float = 2.0  # vote grid cell, x image GSD at the prior range
    n_hypotheses: int = 3
    exclusion_bins: float = 4.0  # peak suppression radius >= this many cells
    exclusion_fov_frac: float = 0.1  # and >= this fraction of the footprint width
    gate1_px: float = 4.0  # first assignment gate max(gate1_px, gate1_rel * D_px)
    gate1_rel: float = 0.4
    gate2_px: float = 2.0  # second assignment gate (refined pose)
    gate2_rel: float = 0.2
    size_ratio_gate: float = 1.6
    ransac_px: float = 3.0  # fixed-attitude position RANSAC threshold
    sigma_min_px: float = 0.5  # per-crater sigma = max(sigma_min_px, sigma_rel * D_px), calibrated (G11)
    sigma_rel: float = 0.05
    visibility_filter: bool = False  # d/D >= c_vis tan(e) / 4 (interior shadow possible), calibrated (G8)
    c_vis: float = 1.0
    image_margin_px: float = 0.0


@dataclass
class Candidates:
    catalog: CraterCatalog  # craters selected for this image
    pred: dict  # their prediction at the prior pose (project_rims output)
    n_window: int  # catalog craters in the search window (after catalog_D_min_m) before selection


def predict(catalog: CraterCatalog, K, R_wc, c) -> dict:
    return project_rims(K, R_wc, c, catalog.X, catalog.e1, catalog.e2, catalog.a, catalog.b, catalog.psi)


def _empty_candidates(K, R, c) -> Candidates:
    e = CraterCatalog.empty()
    return Candidates(e, predict(e, K, R, c), 0)


def select_candidates(catalog: CraterCatalog, cam, T_prior, sigma_h: float, sun_dir_world, D_px_range,
                      cfg: MatchConfig) -> Candidates:
    """Catalog craters near the prior footprint whose predicted size suits the detector."""
    T = np.asarray(T_prior, float)
    R, c = T[:3, :3], T[:3, 3]
    if len(catalog) == 0:
        return _empty_candidates(cam.K, R, c)
    reach = 3.0 * sigma_h + cfg.search_margin_m
    try:
        xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
    except ValueError:
        return _empty_candidates(cam.K, R, c)
    win = catalog.query_box(xmin - reach, xmax + reach, ymin - reach, ymax + reach)
    if cfg.catalog_D_min_m > 0:
        win = win.subset(win.D >= cfg.catalog_D_min_m)
    if len(win) == 0:
        return _empty_candidates(cam.K, R, c)
    pred = predict(win, cam.K, R, c)
    lo, hi = D_px_range
    with np.errstate(invalid="ignore"):
        keep = pred["valid"] & (pred["D_px"] >= cfg.size_pad[0] * lo) & (pred["D_px"] <= cfg.size_pad[1] * hi)
        if cfg.visibility_filter:
            s = np.asarray(sun_dir_world, float)
            s = s / np.linalg.norm(s)
            e = np.arcsin(np.clip(win.normal @ s, -1.0, 1.0))
            keep &= (e > 0) & (win.d_over_D >= cfg.c_vis * np.tan(e) / 4.0)
    idx = np.nonzero(keep)[0]
    return Candidates(win.subset(idx), {k: v[idx] for k, v in pred.items()}, len(win))


def vote_positions(det: Detections, cand: Candidates, cam, T_prior, sigma_h: float, cfg: MatchConfig) -> dict:
    """Strongest horizontal camera-position offsets (east, north at the prior) from pair votes."""
    out = {"offsets": np.zeros((0, 2)), "votes": np.zeros(0), "counts": np.zeros(0, int), "n_pairs": 0,
           "bin_m": float("nan"), "radius_m": float("nan")}
    if len(det) == 0 or len(cand.catalog) == 0:
        return out
    T = np.asarray(T_prior, float)
    R, p0 = T[:3, :3], T[:3, 3]
    east, north, up = frames.enu_basis(p0).T
    d = (np.column_stack([det.u, det.v, np.ones(len(det))]) @ np.linalg.inv(cam.K).T) @ R.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(det.D_px[:, None] / cand.pred["D_px"][None, :]))
    ii, jj = np.nonzero(lr <= np.log(cfg.size_ratio_vote))
    if ii.size == 0:
        return out
    X, di = cand.catalog.X[jj], d[ii]
    du = di @ up
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((X - p0) @ up) / np.where(du < -1e-9, du, np.nan)  # camera at the prior height on the ray
    C = X - t[:, None] * di - p0
    off = np.column_stack([C @ east, C @ north])
    radius = 3.0 * sigma_h + cfg.search_margin_m
    good = np.isfinite(off).all(axis=1) & (t > 0) & (np.hypot(off[:, 0], off[:, 1]) <= radius)
    off = off[good]
    out["n_pairs"], out["radius_m"] = int(off.shape[0]), float(radius)
    if off.shape[0] == 0:
        return out
    gsd = float(np.nanmedian(cand.pred["z"])) / cam.fx
    bin_m = max(cfg.vote_bin_gsd * gsd, 1.0)
    nb = max(int(np.ceil(2 * radius / bin_m)), 1)
    hist, xe, _ = np.histogram2d(off[:, 0], off[:, 1], bins=nb, range=[[-radius, radius], [-radius, radius]])
    sm = ndi.gaussian_filter(hist, 1.0, mode="constant")
    centres = 0.5 * (xe[:-1] + xe[1:])
    gx, gy = np.meshgrid(centres, centres, indexing="ij")
    excl = max(cfg.exclusion_bins * bin_m, cfg.exclusion_fov_frac * cam.width * gsd)
    work = sm.copy()
    offs, votes, counts = [], [], []
    for _ in range(cfg.n_hypotheses):
        k = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[k] <= 0:
            break
        o = np.array([centres[k[0]], centres[k[1]]])
        offs.append(o)
        votes.append(float(sm[k]))
        counts.append(int(np.sum(np.hypot(off[:, 0] - o[0], off[:, 1] - o[1]) <= 1.5 * bin_m)))
        work[np.hypot(gx - o[0], gy - o[1]) <= excl] = 0.0
    out.update(offsets=np.array(offs).reshape(-1, 2), votes=np.array(votes), counts=np.array(counts, int),
               bin_m=float(bin_m))
    return out


def assign(det_uv, det_D, pred_uv, pred_D, gate_px: float, gate_rel: float, size_ratio_max: float):
    """Gated one-to-one assignment (Hungarian) of detections to predicted craters."""
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    det_uv, pred_uv = np.asarray(det_uv, float).reshape(-1, 2), np.asarray(pred_uv, float).reshape(-1, 2)
    if len(det_uv) == 0 or len(pred_uv) == 0:
        return empty
    det_D, pred_D = np.asarray(det_D, float), np.asarray(pred_D, float)
    fin = np.isfinite(pred_uv).all(axis=1) & np.isfinite(pred_D) & (pred_D > 0)
    dist = np.hypot(det_uv[:, None, 0] - pred_uv[None, :, 0], det_uv[:, None, 1] - pred_uv[None, :, 1])
    gate = np.maximum(gate_px, gate_rel * det_D)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(det_D[:, None] / pred_D[None, :]))
        ok = fin[None, :] & (dist <= gate) & (lr <= np.log(size_ratio_max))
    if not ok.any():
        return empty
    rows, cols = np.nonzero(ok.any(axis=1))[0], np.nonzero(ok.any(axis=0))[0]
    sub = ok[np.ix_(rows, cols)]
    cost = np.where(sub, (dist / gate)[np.ix_(rows, cols)] + np.nan_to_num(lr[np.ix_(rows, cols)]), 1e6)
    ri, ci = linear_sum_assignment(cost)
    keep = sub[ri, ci]
    i, j = rows[ri[keep]], cols[ci[keep]]
    return i, j, dist[i, j]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_match.py -v`
Expected: 3 passed

- [ ] **Step 5: 체크포인트** — 빠른 시험 전부 통과. 커밋하지 않는다.

---

### Task 10: `localize_craters` (`craters/localize.py`)

**Files:**
- Create: `src/lunar_trn/craters/localize.py`
- Test: `tests/test_localize_craters.py`

**Interfaces:**
- Consumes: `detect.detect_craters/pinhole_sun_fn/Detections/DetectorConfig`, `match.*`, `conics.project_rims`, `pose.ransac_position/refine_pose/mahalanobis_gate/project_uv`, `health.scene_signals`, `localize.LocalizationResult`
- Produces:
  - `CraterMap(catalog: CraterCatalog, sun_dir_world: ndarray)`
  - `CraterConfig(detector, match, refine_passes=2, min_inliers_fix=5, min_match_ratio=0.0, max_sigma0=inf, cov_floor_h_px=0.0, cov_floor_v_px=0.0, pixel_center=0.5)` — 고정 = inlier ≥ `min_inliers_fix` ∧ `catalog_match_ratio` ≥ `min_match_ratio` ∧ `sigma0` ≤ `max_sigma0`
  - `localize_craters(image, K, prior_T_world_cam, prior_cov, crater_map, config=None, seed=0, detections=None) -> LocalizationResult` (`family="P-G"`, `stage ∈ {"detect","match","pose"}`; `stage == "pose"`이면 게이트와 무관하게 최선 해를 `T_world_cam`에 담는다)
  - `health["pg"]` 키: `n_catalog_window, n_catalog_candidates, gsd_prior_m`, 검출기 health 전체, `t_detect_s, vote_peak, vote_second, vote_ratio, n_vote_pairs, vote_bin_m, hypotheses (list of dict: offset_m, votes, count, n_assigned1, n_ransac1, n_inliers), hyp_inlier_margin, n_assigned, n_inliers, inlier_ratio, match_ratio, n_catalog_view, catalog_match_ratio, reproj_rms_px, sigma0, mahalanobis_gamma, reject_counts, t_match_s`
  - `landmarks` 키: `u, v, A, B, theta, D_px, area_dark, area_bright, pair_dev_deg, cat_id, cat_D_m, cat_d_over_D, cat_completeness, bias_u, bias_v, res_u, res_v, used, inlier, status`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_localize_craters.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_localize_craters.py -v`
Expected: FAIL (`ModuleNotFoundError: lunar_trn.craters.localize`)

- [ ] **Step 3: Implement**

```python
# src/lunar_trn/craters/localize.py
"""P-G crater pattern matching: single-image localisation in tracking mode.

``localize_craters(image, K, prior_T_world_cam, prior_cov, crater_map)`` detects crater
ellipses (CNav style), predicts catalog rims through the prior pose, votes horizontal
position hypotheses, verifies the strongest ones (gated assignment, fixed-attitude position
RANSAC, perspective-centre-bias correction, the shared 6-DoF refinement with the LVS
Mahalanobis gate) and gates the fix on the inlier count. It returns the T-I
:class:`~lunar_trn.localize.LocalizationResult` with ``family = "P-G"`` and raw signals
under ``health["pg"]``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import frames
from ..camera import PinholeCamera
from ..health import scene_signals
from ..localize import LocalizationResult
from ..pose import mahalanobis_gate, project_uv, ransac_position, refine_pose
from .catalog import CraterCatalog
from .conics import project_rims
from .detect import DetectorConfig, Detections, detect_craters, pinhole_sun_fn
from .match import Candidates, MatchConfig, assign, predict, select_candidates, vote_positions


@dataclass
class CraterMap:
    """Onboard P-G map: a Sun-independent crater catalog and the Sun direction (ephemeris)."""

    catalog: CraterCatalog
    sun_dir_world: np.ndarray


@dataclass
class CraterConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    refine_passes: int = 2
    min_inliers_fix: int = 5  # decisions G12 (CNav: at least five catalog craters)
    min_match_ratio: float = 0.0  # decisions G12: inliers / catalog craters in view (SLIM-style), calibrated
    max_sigma0: float = float("inf")  # decisions G12: a-posteriori unit-weight sigma, calibrated
    cov_floor_h_px: float = 0.0  # decisions G13, image-GSD pixels per horizontal axis
    cov_floor_v_px: float = 0.0
    pixel_center: float = 0.5


def _in_image(pred: dict, cam: PinholeCamera, margin: float) -> np.ndarray:
    lo = cam.pixel_center - 0.5 + margin
    with np.errstate(invalid="ignore"):
        return (pred["valid"] & (pred["u"] - pred["A"] >= lo) & (pred["u"] + pred["A"] <= cam.width + lo - 2 * margin)
                & (pred["v"] - pred["A"] >= lo) & (pred["v"] + pred["A"] <= cam.height + lo - 2 * margin))


def _depth_guess(p0: np.ndarray, R0: np.ndarray) -> float:
    up = p0 / np.linalg.norm(p0)
    alt = np.linalg.norm(p0) - frames.MOON_RADIUS_M
    return max(alt, 1.0) / max(-(R0[:, 2] @ up), 0.2)


def _refine(det: Detections, i, cat: CraterCatalog, keep, R, c, K, R0, sigma_att, cfg: CraterConfig):
    """Bias-corrected refinement of the pairs (det[i], cat) with the LVS Mahalanobis gate."""
    sig = np.maximum(cfg.match.sigma_min_px, cfg.match.sigma_rel * det.D_px[i])
    X = cat.X
    keep = keep.copy()
    gamma, rf, pr, uvc = float("nan"), None, None, None
    for _ in range(max(cfg.refine_passes, 1)):
        if keep.sum() < 3:
            break
        pr = project_rims(K, R, c, X, cat.e1, cat.e2, cat.a, cat.b, cat.psi)
        uvc = det.uv[i] - np.nan_to_num(pr["bias"])
        rf = refine_pose(uvc[keep], X[keep], K, R, c, sigma_px=sig[keep], att_prior_R=R0,
                         att_prior_sigma_rad=sigma_att)
        R, c = rf.R_wc, rf.position
        uvp, z = project_uv(K, R, c, X)
        res = uvp - uvc
        _, gamma, _ = mahalanobis_gate(res[keep], sig[keep])
        d2 = np.sum((res / sig[:, None]) ** 2, axis=1)
        keep = np.isfinite(d2) & (z > 0) & (d2 < gamma)
    if rf is None or keep.sum() < 3:
        return None
    pr = project_rims(K, R, c, X, cat.e1, cat.e2, cat.a, cat.b, cat.psi)
    uvc = det.uv[i] - np.nan_to_num(pr["bias"])
    rf = refine_pose(uvc[keep], X[keep], K, R, c, sigma_px=sig[keep], att_prior_R=R0, att_prior_sigma_rad=sigma_att)
    R, c = rf.R_wc, rf.position
    uvp, _ = project_uv(K, R, c, X)
    return {"R": R, "c": c, "cov": rf.cov, "keep": keep, "res": uvp - uvc, "bias": pr["bias"], "rms": rf.rms_px,
            "sigma0": rf.sigma0, "gamma": gamma, "n_inliers": int(keep.sum())}


def _verify(offset, det: Detections, cand: Candidates, cam: PinholeCamera, p0, enu, R0, sigma_att,
            cfg: CraterConfig, rng) -> dict:
    m = cfg.match
    rec = {"offset_m": [float(offset[0]), float(offset[1])], "n_assigned1": 0, "n_ransac1": 0, "n_inliers": 0,
           "state": None}
    c = p0 + enu[:, 0] * offset[0] + enu[:, 1] * offset[1]
    pred = predict(cand.catalog, cam.K, R0, c)
    vis = np.nonzero(_in_image(pred, cam, m.image_margin_px))[0]
    if vis.size < 2:
        return rec
    i, jv, _ = assign(det.uv, det.D_px, np.column_stack([pred["u"][vis], pred["v"][vis]]), pred["D_px"][vis],
                      m.gate1_px, m.gate1_rel, m.size_ratio_gate)
    j = vis[jv]
    rec["n_assigned1"] = int(i.size)
    if i.size < 2:
        return rec
    rr = ransac_position(det.uv[i] - np.nan_to_num(pred["bias"][j]), cand.catalog.X[j], cam.K, R0, m.ransac_px,
                         rng=rng)
    if rr.position is None:
        return rec
    rec["n_ransac1"] = int(rr.inliers.sum())
    best = _refine(det, i, cand.catalog.subset(j), rr.inliers, R0, rr.position, cam.K, R0, sigma_att, cfg)
    if best is None:
        return rec
    best.update(i=i, j=j)
    # Second assignment from the refined pose with a tighter gate (LVS: a second fine pass).
    pred2 = predict(cand.catalog, cam.K, best["R"], best["c"])
    vis2 = np.nonzero(_in_image(pred2, cam, m.image_margin_px))[0]
    i2, jv2, _ = assign(det.uv, det.D_px, np.column_stack([pred2["u"][vis2], pred2["v"][vis2]]),
                        pred2["D_px"][vis2], m.gate2_px, m.gate2_rel, m.size_ratio_gate)
    j2 = vis2[jv2]
    if i2.size >= 3:
        rr2 = ransac_position(det.uv[i2] - np.nan_to_num(pred2["bias"][j2]), cand.catalog.X[j2], cam.K, best["R"],
                              m.ransac_px, rng=rng)
        if rr2.position is not None:
            st2 = _refine(det, i2, cand.catalog.subset(j2), rr2.inliers, best["R"], rr2.position, cam.K, R0,
                          sigma_att, cfg)
            if st2 is not None and st2["n_inliers"] >= best["n_inliers"]:
                st2.update(i=i2, j=j2)
                best = st2
    rec.update(n_inliers=best["n_inliers"], state=best)
    return rec


def _landmarks(det: Detections, best: dict | None, cand: Candidates) -> dict:
    n = len(det)
    nan = np.full(n, np.nan)
    lm = {"u": det.u, "v": det.v, "A": det.A, "B": det.B, "theta": det.theta, "D_px": det.D_px,
          "area_dark": det.area_dark, "area_bright": det.area_bright, "pair_dev_deg": det.pair_dev_deg,
          "cat_id": np.full(n, -1, np.int64), "cat_D_m": nan.copy(), "cat_d_over_D": nan.copy(),
          "cat_completeness": nan.copy(), "bias_u": nan.copy(), "bias_v": nan.copy(), "res_u": nan.copy(),
          "res_v": nan.copy(), "used": np.zeros(n, bool), "inlier": np.zeros(n, bool),
          "status": np.array(["unassigned"] * n, dtype=object)}
    if best is not None:
        i, j, keep = best["i"], best["j"], best["keep"]
        cat = cand.catalog
        lm["cat_id"][i] = cat.id[j]
        lm["cat_D_m"][i] = cat.D[j]
        lm["cat_d_over_D"][i] = cat.d_over_D[j]
        lm["cat_completeness"][i] = cat.completeness[j]
        lm["bias_u"][i], lm["bias_v"][i] = best["bias"][:, 0], best["bias"][:, 1]
        lm["res_u"][i], lm["res_v"][i] = best["res"][:, 0], best["res"][:, 1]
        lm["used"][i] = True
        lm["inlier"][i[keep]] = True
        lm["status"][i] = "gated"
        lm["status"][i[keep]] = "inlier"
    return lm


def localize_craters(image: np.ndarray, K: np.ndarray, prior_T_world_cam: np.ndarray, prior_cov: np.ndarray,
                     crater_map: CraterMap, config: CraterConfig | None = None, seed: int = 0,
                     detections: Detections | None = None) -> LocalizationResult:
    """Localise one image against a crater catalog. ``detections`` replaces the detector (tests, oracle runs)."""
    cfg = config or CraterConfig()
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    H, W = np.asarray(image).shape
    cam = PinholeCamera(K, W, H, pixel_center=cfg.pixel_center)
    T = np.asarray(prior_T_world_cam, float).copy()
    p0, R0 = T[:3, 3].copy(), T[:3, :3].copy()
    P = np.asarray(prior_cov, float)
    enu = frames.enu_basis(p0)
    Pp = enu.T @ P[:3, :3] @ enu
    sigma_h = float(np.sqrt(0.5 * (Pp[0, 0] + Pp[1, 1])))
    sigma_att = max(float(np.sqrt(np.mean(np.diag(P)[3:]))) if P.shape[0] >= 6 else 0.0, 1e-6)
    dcfg = cfg.detector
    cand = select_candidates(crater_map.catalog, cam, T, sigma_h, crater_map.sun_dir_world,
                             (dcfg.D_min_px, dcfg.D_max_px), cfg.match)
    z0 = float(np.nanmedian(cand.pred["z"])) if len(cand.catalog) else _depth_guess(p0, R0)
    gsd = z0 / cam.fx
    pg: dict = {"n_catalog_window": cand.n_window, "n_catalog_candidates": len(cand.catalog), "gsd_prior_m": gsd}
    health: dict = {"scene": scene_signals(image, gsd, None), "prior_sigma_h_m": sigma_h, "pg": pg}
    nan6 = np.full((6, 6), np.nan)

    def finish(status, T_out, cov, landmarks, stage):
        health["t_total_s"] = time.perf_counter() - t0
        health["correction_from_prior_m"] = (T_out[:3, 3] - p0).tolist()
        return LocalizationResult(status, T_out, cov, landmarks, health, stage, family="P-G")

    t1 = time.perf_counter()
    if detections is None:
        det, dh = detect_craters(image, pinhole_sun_fn(cam, T, crater_map.sun_dir_world, z0), dcfg,
                                 pixel_center=cfg.pixel_center)
    else:
        det, dh = detections, {"n_detections": len(detections), "injected": True}
    pg.update(dh)
    pg["t_detect_s"] = time.perf_counter() - t1
    if len(det) < 2:
        return finish("no_fix", T, nan6, {}, "detect")
    if len(cand.catalog) < 2:
        return finish("no_fix", T, nan6, _landmarks(det, None, cand), "match")
    t2 = time.perf_counter()
    votes = vote_positions(det, cand, cam, T, sigma_h, cfg.match)
    vv = votes["votes"]
    pg.update(vote_peak=float(vv[0]) if vv.size else 0.0, vote_second=float(vv[1]) if vv.size > 1 else 0.0,
              n_vote_pairs=votes["n_pairs"], vote_bin_m=votes["bin_m"])
    pg["vote_ratio"] = pg["vote_second"] / pg["vote_peak"] if pg["vote_peak"] > 0 else float("nan")
    hyps = [_verify(o, det, cand, cam, p0, enu, R0, sigma_att, cfg, rng) for o in votes["offsets"]]
    pg["hypotheses"] = [{**{k: v for k, v in h.items() if k != "state"}, "votes": float(vt), "count": int(ct)}
                        for h, vt, ct in zip(hyps, votes["votes"], votes["counts"])]
    pg["t_match_s"] = time.perf_counter() - t2
    good = sorted((h for h in hyps if h["state"] is not None), key=lambda h: -h["n_inliers"])
    if not good:
        return finish("no_fix", T, nan6, _landmarks(det, None, cand), "match")
    best = good[0]["state"]
    pg["hyp_inlier_margin"] = good[0]["n_inliers"] - (good[1]["n_inliers"] if len(good) > 1 else 0)
    T_est = np.eye(4)
    T_est[:3, :3], T_est[:3, 3] = best["R"], best["c"]
    pred_f = predict(cand.catalog, cam.K, best["R"], best["c"])
    with np.errstate(invalid="ignore"):
        vis_f = _in_image(pred_f, cam, 0.0) & (pred_f["D_px"] >= dcfg.D_min_px) & (pred_f["D_px"] <= dcfg.D_max_px)
    n_inl, n_ass, n_view = best["n_inliers"], int(len(best["i"])), int(vis_f.sum())
    pg.update(n_assigned=n_ass, n_inliers=n_inl, inlier_ratio=n_inl / max(n_ass, 1),
              match_ratio=n_inl / max(len(det), 1), n_catalog_view=n_view,
              catalog_match_ratio=n_inl / max(n_view, 1), reproj_rms_px=best["rms"], sigma0=best["sigma0"],
              mahalanobis_gamma=best["gamma"], reject_counts={"unassigned": len(det) - n_ass, "gated": n_ass - n_inl})
    cov = np.asarray(best["cov"], float).copy()
    if np.all(np.isfinite(cov)):
        health["cov_landmark_only"] = cov[:3, :3].tolist()
        enu_e = frames.enu_basis(best["c"])
        gsd_e = float(np.nanmedian(pred_f["z"][vis_f])) / cam.fx if vis_f.any() else gsd
        sh, sv = cfg.cov_floor_h_px * gsd_e, cfg.cov_floor_v_px * gsd_e
        cov[:3, :3] += enu_e @ np.diag([sh**2, sh**2, sv**2]) @ enu_e.T
    gate_ok = (n_inl >= cfg.min_inliers_fix and pg["catalog_match_ratio"] >= cfg.min_match_ratio
               and best["sigma0"] <= cfg.max_sigma0)
    return finish("fix" if gate_ok else "no_fix", T_est, cov, _landmarks(det, best, cand), "pose")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_localize_craters.py -v -m "not slow" && .venv/bin/python -m pytest tests/test_localize_craters.py -v -m slow`
Expected: 4 passed, 이어서 느린 시험 1 passed. 무작위 검출 시험이 실패하면(오고정) 게이트를 넓히지 말고, `health["pg"]["hypotheses"]`로 어느 단계(1차 할당/RANSAC/2차 할당)가 우연 일치를 키우는지 확인한 뒤 고친다.

- [ ] **Step 5: 체크포인트** — `.venv/bin/python -m pytest -q` 전체(느린 시험 포함) 통과. 커밋하지 않는다.

---

### Task 11: 실행기 (`scripts/run_pg.py`)

**Files:**
- Create: `scripts/run_pg.py`

**Interfaces:**
- Consumes: `localize_craters`, `CraterMap`, `CraterConfig`, `detect_craters`, `pinhole_sun_fn`, `annotate_detections`, `catalog_recall`, `experiments.sample_prior/score/fine_map_pixel`, `stereolunar.Frame/list_frames/assign_triplets`
- Produces:
  - `runs.jsonl` 레코드(국지화 모드): `geometry_id, image_index, image_illum, band, alt_ground_m, gsd_m, map_px_fine, altitude_sphere_m, off_nadir_deg, trajectory, family, trial, status, stage, prior_err_*, err_e_m, err_n_m, err_u_m, err_h_m, err_att_deg, tol_m, verdict, t_run_s, cov_enu_diag_m2, cov_lm_enu_diag_m2(있으면), n_inliers_gt_consistent(있으면), catalog_D_min_m, health`
  - 검출 전용 모드(`--detect-only`): 레코드 `..., detect (health), n_det, n_det_matched, n_cat_view, n_cat_detected, t_detect_s`와 `annot/{gid}_{j}.npz`(키 `det_*`, `ann_*`, `rc_*`)
  - `config.json` (인자와 `CraterConfig`)

- [ ] **Step 1: Implement `scripts/run_pg.py`**

```python
# scripts/run_pg.py
#!/usr/bin/env python3
"""P-G (crater) runner for V7 / H-b on StereoLunar.

One job = one geometry (illumination triplet). Each requested image illumination j is
localised from ``--trials`` priors. The prior seed equals the one ``run_v2.py`` uses with
the map illumination set to j, so every run pairs with the T-I run of the same (geometry,
illumination, trial) and identical prior error.

    python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
        --out outputs/pg/v7_nadir --split eval --trials 6 --workers 40
    # detector calibration: detections + ground-truth annotation per image, no localisation
    python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
        --out outputs/pg/calib_detect --split calib --detect-only --workers 40
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from lunar_trn import frames
from lunar_trn.craters.annotate import annotate_detections, catalog_recall
from lunar_trn.craters.catalog import CraterCatalog
from lunar_trn.craters.detect import Detections, detect_craters, pinhole_sun_fn
from lunar_trn.craters.localize import CraterConfig, CraterMap, localize_craters
from lunar_trn.datasets import stereolunar as sl
from lunar_trn.experiments import fine_map_pixel, sample_prior, score
from lunar_trn.validation.v1 import footprint_bounds

NOMINAL_BANDS_M = [3500 + 3000 * k for k in range(10)]
_CAT: dict = {}


def _catalog(path: str) -> CraterCatalog:
    if path not in _CAT:
        _CAT[path] = CraterCatalog.load(path)
    return _CAT[path]


def _cfg_from(overrides: dict) -> CraterConfig:
    cfg = CraterConfig()
    for key, val in overrides.items():
        part, _, name = key.partition(".")
        if name:
            setattr(getattr(cfg, part), name, val)
        else:
            setattr(cfg, part, val)
    return cfg


def _image_info(f: sl.Frame):
    cam, T = f.camera(), f.T_world_cam
    c = cam.height // 2
    alt = float(np.nanmedian(f.depth()[c - 6:c + 6, c - 6:c + 6]))
    band = int(np.argmin([abs(alt - b) for b in NOMINAL_BANDS_M]))
    return cam, T, alt, band, fine_map_pixel(NOMINAL_BANDS_M[band] / cam.fx)


def run_job(job: dict) -> list[dict]:
    d = Path(job["dir"])
    idx = job["indices"]
    cat = _catalog(job["catalog"])
    cfg = _cfg_from(job["overrides"])
    out, annots = [], {}
    for j in job["illums"]:
        f = sl.Frame(d, idx[j], job["traj"], illumination=j, geometry_id=idx[0])
        cam, T, alt, band, map_px = _image_info(f)
        img, sun = f.image(), f.sun_direction()
        xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
        local = cat.query_box(xmin - 500.0, xmax + 500.0, ymin - 500.0, ymax + 500.0)
        sun_fn = pinhole_sun_fn(cam, T, sun, alt)
        base = {"geometry_id": idx[0], "image_index": idx[j], "image_illum": j, "band": band, "alt_ground_m": alt,
                "gsd_m": alt / cam.fx, "map_px_fine": map_px, "altitude_sphere_m": f.altitude_above_sphere(),
                "off_nadir_deg": f.off_nadir_deg(), "trajectory": job["traj"], "family": "P-G",
                "catalog_D_min_m": cfg.match.catalog_D_min_m}
        if job["detect_only"]:
            t1 = time.time()
            det, dh = detect_craters(img, sun_fn, cfg.detector, pixel_center=cam.pixel_center)
            kw = {"tol_rel": job["ann_tol_rel"], "size_ratio_max": job["ann_size_ratio"]}
            ann = annotate_detections(det, local, cam, T, sun_fn, **kw)
            rc = catalog_recall(det, local, cam, T, sun, **kw)
            annots[f"{idx[0]}_{j}"] = {
                **{f"det_{k}": getattr(det, k) for k in ("u", "v", "A", "B", "theta", "pair_dev_deg", "contrast",
                                                          "level", "area_dark", "area_bright")},
                **{f"ann_{k}": v for k, v in ann.items()}, **{f"rc_{k}": v for k, v in rc.items()}}
            out.append({**base, "detect": dh, "n_det": len(det), "n_det_matched": int((ann["cat_index"] >= 0).sum()),
                        "n_cat_view": int(len(rc["cat_id"])), "n_cat_detected": int(rc["detected"].sum()),
                        "t_detect_s": time.time() - t1})
            continue
        cmap = CraterMap(cat, sun)
        enu = frames.enu_basis(T[:3, 3])
        for trial in range(job["trials"]):
            k_map = j  # unpruned P-G pairs with the matched-illumination T-I run
            seed = zlib.crc32(f"{idx[0]}-{k_map}-{j}-{trial}-{job['seed']}".encode())
            rng = np.random.default_rng(seed)
            Tp, cov, perr = sample_prior(T, rng, job["prior_radius_m"], job["prior_vert_m"], job["prior_att_deg"])
            t1 = time.time()
            try:
                res = localize_craters(img, cam.K, Tp, cov, cmap, cfg, seed=seed)
            except Exception as err:  # noqa: BLE001
                out.append({**base, "trial": trial, "error": repr(err)})
                continue
            sc = score(res, T, map_px, tol_px=job["tol_px"])
            rec = {**base, "trial": trial, "status": res.status, "stage": res.stage, **perr, **sc,
                   "t_run_s": time.time() - t1, "cov_enu_diag_m2": np.diag(enu.T @ res.cov[:3, :3] @ enu).tolist(),
                   "health": res.health}
            if res.health.get("cov_landmark_only") is not None:
                lm_cov = np.asarray(res.health["cov_landmark_only"])
                rec["cov_lm_enu_diag_m2"] = np.diag(enu.T @ lm_cov @ enu).tolist()
            lm = res.landmarks
            if lm and lm["inlier"].any():
                # Ground-truth consistency of the inliers (analysis only, never fed back).
                det_all = Detections.from_ellipses(lm["u"], lm["v"], lm["A"], lm["B"], lm["theta"])
                gt = annotate_detections(det_all, local, cam, T, sun_fn)["cat_id"]
                rec["n_inliers_gt_consistent"] = int(np.sum(lm["inlier"] & (gt >= 0) & (gt == lm["cat_id"])))
            out.append(rec)
    if annots:
        p = Path(job["out"]) / "annot" / f"{idx[0]}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{f"{key}__{fld}": np.asarray(arr) for key, dct in annots.items()
                                  for fld, arr in dct.items()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--image-illums", default="0,1,2")
    ap.add_argument("--prior-radius-m", type=float, default=3000.0)
    ap.add_argument("--prior-vert-m", type=float, default=120.0)
    ap.add_argument("--prior-att-deg", type=float, default=0.0)
    ap.add_argument("--tol-px", type=float, default=3.0, help="false-fix tolerance in nominal fine-map pixels")
    ap.add_argument("--split", choices=["all", "calib", "eval"], default="all",
                    help="calib = geometries with ordinal % 4 == 1, eval = the rest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. detector.kappa=0.8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--ann-tol-rel", type=float, default=0.25)
    ap.add_argument("--ann-size-ratio", type=float, default=1.5)
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, _, val = item.partition("=")
        overrides[key] = json.loads(val)
    groups = [g for g in sl.assign_triplets(sl.list_frames(args.data)) if g[0].illumination is not None]
    if args.split == "calib":
        groups = [g for i, g in enumerate(groups) if i % 4 == 1]
    elif args.split == "eval":
        groups = [g for i, g in enumerate(groups) if i % 4 != 1]
    groups = groups[:: max(args.every, 1)]
    if args.limit:
        groups = groups[: args.limit]
    illums = [int(x) for x in args.image_illums.split(",")]
    jobs = [{"dir": str(args.data), "traj": args.data.name, "indices": [f.index for f in g], "illums": illums,
             "catalog": args.catalog, "trials": args.trials, "prior_radius_m": args.prior_radius_m,
             "prior_vert_m": args.prior_vert_m, "prior_att_deg": args.prior_att_deg, "tol_px": args.tol_px,
             "overrides": overrides, "seed": args.seed, "detect_only": args.detect_only,
             "ann_tol_rel": args.ann_tol_rel, "ann_size_ratio": args.ann_size_ratio, "out": str(args.out)}
            for g in groups]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "crater": dataclasses.asdict(_cfg_from(overrides))},
        indent=2))
    print(f"{len(groups)} geometries, {len(jobs)} jobs", flush=True)
    _catalog(args.catalog)  # load once before forking
    t0 = time.time()
    n_runs = 0
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=8) as pool, \
            (args.out / "runs.jsonl").open("w") as fh:
        for n, recs in enumerate(pool.imap_unordered(run_job, jobs), 1):
            for r in recs:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            n_runs += len(recs)
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} jobs, {n_runs} runs, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke run (both modes)**

Run:
```bash
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --out outputs/pg/smoke_detect --split calib --limit 3 --detect-only --workers 3
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --out outputs/pg/smoke_loc --split calib --limit 3 --trials 2 --workers 3
.venv/bin/python - <<'EOF'
import json, collections
for d in ("outputs/pg/smoke_detect", "outputs/pg/smoke_loc"):
    rs = [json.loads(l) for l in open(f"{d}/runs.jsonl")]
    print(d, len(rs), collections.Counter(r.get("verdict", r.get("stage", "detect")) for r in rs),
          [r.get("error") for r in rs if "error" in r][:3])
EOF
```
Expected: 검출 전용 9 레코드와 `annot/*.npz` 3개, 국지화 18 레코드, `error` 키 없음. 판정 분포는 아직 보정 전이라 기준이 없다.

- [ ] **Step 3: 체크포인트** — `runs.jsonl` 한 줄을 열어 `health.pg`의 원신호 키가 Task 10 인터페이스와 같은지 확인한다. 커밋하지 않는다.

---

### Task 12: 보정 A — 검출기·카탈로그 완전성 (보정 표본만)

**Files:**
- Modify: `src/lunar_trn/craters/catalog.py` (`explain_rim` 추가)
- Modify: `scripts/run_pg.py` (`--dem`이 있으면 카탈로그 누락 진단)
- Create: `scripts/analyze_pg_calib.py` (`detect` 부명령)
- Modify: `scripts/pg_figures.py` (`recall` 부명령)
- Modify: `src/lunar_trn/craters/detect.py`, `src/lunar_trn/craters/match.py` (보정값을 기본값으로)
- Modify: `docs/decisions.md` (§G 신설, G1–G8, G11)
- Test: `tests/test_catalog.py` (시험 추가)

**Interfaces:**
- Produces:
  - `catalog.explain_rim(dem, x, y, r_m, cfg=None) -> (reason: str, rec: dict | None)` — 사유: `"ok"`, `"edge"`, `"no_crest"`, `"few_points"`, `"fit"`, `"size"`, `"centre"`, `"D_min"`, `"D_max"`, `"completeness"`, `"ellipticity"`, `"dD"`
  - 검출 전용 `annot` npz에 `miss_reason, miss_D_m, miss_completeness, miss_d_over_D, miss_ellipticity` (카탈로그 대응이 없는 검출만 채움)
  - `outputs/pg/calibA*/calib_detect.json` 키: `kappa, beta, pair_angle_max_deg, D_min_px, sigma_min_px, sigma_rel, visibility_filter, c_vis, recall_by_D, recall_by_q, miss_reasons, miss_relax, set_flags`

**규칙 (평가 표본을 보기 전에 고정한다)**
- κ = (현재 κ) × median(D_px 예측 / D_px 검출), 대응 있는 검출만.
- β = median((β₀·A₀ + 반태양 축 오차) / A₁), A₁ = A₀ × κ 비. 오차 = 예측 − 검출.
- θ_pair = 대응 있는 짝의 방향 편차 98.5%ile을 5° 단위로 올림(T-I §E와 같은 정답 보존 규칙).
- D_min_px = 재현율 ≥ 0.5인 가장 작은 D_px 구간의 하한(구간 표본 ≥ 30).
- σ 모형: 보정 후 잔차의 축별 강건 표준편차(1.4826·MAD)를 D_px 구간별로 구하고, σ_rel = Σσ_b·D_b / ΣD_b²(D_b ≥ 10 px 구간), σ_min = 가장 작은 구간의 σ_b.
- 가시성 필터: q = d/D ÷ (tan e / 4). 재현율(q < 0.75) < 0.2이고 재현율(q ≥ 1) ≥ 2 × 재현율(q < 0.75)이면 켠다. c_vis = 재현율 ≥ 0.2가 되는 첫 q 구간 하한.
- 카탈로그 완화(한 번만): "검출됐지만 카탈로그에 없는" 검출(D_px ≥ D_min_px)의 사유 중 필터(`completeness`, `ellipticity`, `dD`) 하나가 30% 이상이면, 그 필터를 이 크레이터들의 80%가 통과하는 값으로 완화하고 카탈로그를 다시 만든다. `ok`(씨앗·중복 제거 누락)가 30% 이상이면 `seed_dD_min`을 절반으로 낮춘다. 완화한 경우 보정 A를 처음부터 한 번 더 돌린다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog.py (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_catalog.py::test_explain_rim_reasons -v`
Expected: FAIL (`ImportError: cannot import name 'explain_rim'`)

- [ ] **Step 3: Implement `explain_rim`**

```python
# src/lunar_trn/craters/catalog.py (append)
def explain_rim(dem: Dem, x: float, y: float, r_m: float, cfg: CatalogConfig | None = None):
    """Why a crater of radius ``r_m`` at stereographic (x, y) is (not) in the catalog.

    Runs the profile and fit steps with a seed placed exactly there, then the catalog filters.
    Returns the first failed step or "ok", and the rim record when the fit succeeded. A crater
    that is "ok" here but missing from the catalog was lost by LoG seeding or de-duplication.
    """
    cfg = cfg or CatalogConfig()
    rmin, rmax = min(cfg.radii_px), max(cfg.radii_px)
    usable = [f for f in cfg.levels if r_m / (f * dem.pixel) >= rmin]
    f = max(usable) if usable else min(cfg.levels)
    m = level_margin_m(cfg, f, dem.pixel)
    base = _crop(dem, x - m, x + m, y - m, y + m, align=f)
    dl = base.downsample(f) if f > 1 else base
    H, _ = _fill_nan(dl.heights)
    col, row = (float(v) for v in dl.xy_to_colrow(x, y))
    r0 = float(np.clip(r_m / dl.pixel, rmin, rmax))
    reach = cfg.detrend_annulus[1] * r0 + 1.0
    if not (reach <= col < H.shape[1] - reach and reach <= row < H.shape[0] - reach):
        return "edge", None
    th, crest_rho, crest, valid, inner = _rim_candidates(H, np.array([row]), np.array([col]), np.array([r0]), cfg)
    rec, reason = _fit_rim(dl, H, row, col, r0, th, crest_rho[0], crest[0], valid[0], inner[0], cfg)
    if rec is None:
        return reason, None
    D = 2.0 * np.sqrt(rec["a"] * rec["b"])
    for name, bad in (("D_min", D < cfg.D_min_m), ("D_max", D > cfg.D_max_m),
                      ("completeness", rec["completeness"] < cfg.completeness_min),
                      ("ellipticity", rec["a"] / rec["b"] > cfg.ellipticity_max),
                      ("dD", rec["d_over_D"] < cfg.dD_min)):
        if bad:
            return name, rec
    return "ok", rec
```

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: 10 passed

- [ ] **Step 4: Add the miss diagnosis to `scripts/run_pg.py`**

import에 `from lunar_trn.craters.catalog import CatalogConfig, CraterCatalog, explain_rim`, `from lunar_trn.dem import DemSource`를 쓰고(기존 `CraterCatalog` import 줄을 바꾼다), `_catalog` 아래에 추가한다.

```python
# scripts/run_pg.py (insert below _catalog)
_SRC: dict = {}


def _miss_diagnosis(det, ann, cam, T, dem_path: str, cat_cfg: dict | None, D_min_px: float) -> dict:
    """Why detections without a catalog counterpart are not in the catalog (calibration only)."""
    if dem_path not in _SRC:
        _SRC[dem_path] = DemSource(dem_path)
    n = len(det)
    out = {"miss_reason": np.full(n, "", dtype="<U12"), **{k: np.full(n, np.nan) for k in
           ("miss_D_m", "miss_completeness", "miss_d_over_D", "miss_ellipticity")}}
    sel = np.nonzero((ann["cat_index"] < 0) & (det.D_px >= D_min_px))[0]
    if sel.size == 0:
        return out
    xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
    dem = _SRC[dem_path].read_window(xmin - 8000.0, xmax + 8000.0, ymin - 8000.0, ymax + 8000.0)
    o, d = cam.rays_world(T, det.u[sel] - cam.pixel_center, det.v[sel] - cam.pixel_center)
    t = dem.raycast(np.broadcast_to(o, d.shape), d)
    ccfg = CatalogConfig(**cat_cfg) if cat_cfg else CatalogConfig()
    for k, s in enumerate(sel):
        if not np.isfinite(t[k]):
            out["miss_reason"][s] = "no_hit"
            continue
        x, y, _ = frames.body_to_stereo(o + t[k] * d[k])
        r_m = 0.5 * det.D_px[s] * t[k] * float(d[k] @ T[:3, 2]) / cam.fx
        reason, rec = explain_rim(dem, float(x), float(y), r_m, ccfg)
        out["miss_reason"][s] = reason
        if rec is not None:
            out["miss_D_m"][s] = 2.0 * np.sqrt(rec["a"] * rec["b"])
            out["miss_completeness"][s] = rec["completeness"]
            out["miss_d_over_D"][s] = rec["d_over_D"]
            out["miss_ellipticity"][s] = rec["a"] / rec["b"]
    return out
```

`run_job`의 검출 전용 분기에서 `annots[...] = {...}` 다음 줄에 추가한다.

```python
            if job.get("dem"):
                annots[f"{idx[0]}_{j}"].update(_miss_diagnosis(det, ann, cam, T, job["dem"], cat.meta.get("config"),
                                                               cfg.detector.D_min_px))
```

`main`에 `ap.add_argument("--dem", default=None, help="DEM for the catalog-miss diagnosis (--detect-only)")`를 더하고, 작업 사전에 `"dem": args.dem`을 넣는다.

- [ ] **Step 5: Create `scripts/analyze_pg_calib.py` (`detect`)**

```python
# scripts/analyze_pg_calib.py
#!/usr/bin/env python3
"""Calibration analysis of the P-G baseline (calibration split only, decisions section G).

    python scripts/analyze_pg_calib.py detect outputs/pg/calibA0   # kappa, beta, pair angle, recall, misses
    python scripts/analyze_pg_calib.py loc outputs/pg/calibB         # fix gate and covariance floor
Writes calib_detect.json / calib_loc.json into the run directory and prints the --set flags.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

D_EDGES = [4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128]
Q_EDGES = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]


def load_annots(run_dir: Path) -> dict:
    """Concatenate every image's annotation arrays (keys without the image prefix)."""
    acc: dict = {}
    for p in sorted((run_dir / "annot").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                acc.setdefault(k.split("__", 1)[1], []).append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def _rate_table(x, ok, edges, min_n=30):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(s.sum()),
                     "rate": float(ok[s].mean()) if s.sum() >= min_n else float("nan")})
    return rows


def cmd_detect(run_dir: Path) -> dict:
    det_cfg = json.loads((run_dir / "config.json").read_text())["crater"]["detector"]
    k0, b0 = det_cfg["kappa"], det_cfg["beta"]
    a = load_annots(run_dir)
    m = a["ann_cat_index"] >= 0
    A0 = a["det_A"]
    D0 = 2.0 * np.sqrt(a["det_A"] * a["det_B"])
    factor = float(np.median(a["ann_size_ratio"][m]))
    kappa = k0 * factor
    A1 = A0 * factor
    beta = float(np.median((b0 * A0[m] + a["ann_err_antisun_px"][m]) / A1[m]))
    pair = float(math.ceil(np.percentile(a["det_pair_dev_deg"][m], 98.5) / 5.0) * 5.0)
    along = a["ann_err_antisun_px"] - (beta * A1 - b0 * A0)
    perp = a["ann_err_perp_px"]
    D1 = D0 * factor
    sig_rows = []
    for lo, hi in zip(D_EDGES[:-1], D_EDGES[1:]):
        s = m & (D1 >= lo) & (D1 < hi)
        if s.sum() >= 30:
            r = np.concatenate([along[s] - np.median(along[s]), perp[s] - np.median(perp[s])])
            sig_rows.append({"lo": lo, "hi": hi, "n": int(s.sum()), "sigma_px": float(1.4826 * np.median(np.abs(r)))})
    big = [r for r in sig_rows if math.sqrt(r["lo"] * r["hi"]) >= 10]
    Dc = np.array([math.sqrt(r["lo"] * r["hi"]) for r in big])
    sg = np.array([r["sigma_px"] for r in big])
    sigma_rel = float((sg * Dc).sum() / (Dc**2).sum()) if big else 0.05
    sigma_min = float(sig_rows[0]["sigma_px"]) if sig_rows else 0.5
    rec_D = _rate_table(a["rc_D_px"], a["rc_detected"], D_EDGES)
    D_min = next((r["lo"] for r in rec_D if np.isfinite(r["rate"]) and r["rate"] >= 0.5), D_EDGES[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        q = a["rc_d_over_D"] / (np.tan(np.radians(a["rc_sun_el_deg"])) / 4.0)
    rec_q = _rate_table(np.nan_to_num(q, nan=-1.0), a["rc_detected"], Q_EDGES)
    low = (q < 0.75) & np.isfinite(q)
    high = (q >= 1.0) & np.isfinite(q)
    r_low = float(a["rc_detected"][low].mean()) if low.sum() else float("nan")
    r_high = float(a["rc_detected"][high].mean()) if high.sum() else float("nan")
    vis_on = bool(r_low < 0.2 and r_high >= 2 * r_low)
    c_vis = next((r["lo"] for r in rec_q if np.isfinite(r["rate"]) and r["rate"] >= 0.2), 0.0) if vis_on else 1.0
    out = {"kappa": kappa, "beta": beta, "pair_angle_max_deg": pair, "D_min_px": float(D_min),
           "sigma_min_px": sigma_min, "sigma_rel": sigma_rel, "visibility_filter": vis_on, "c_vis": float(c_vis),
           "n_detections": int(m.size), "n_matched": int(m.sum()), "size_factor": factor, "sigma_by_D": sig_rows,
           "recall_by_D": rec_D, "recall_by_q": rec_q, "recall_q_low": r_low, "recall_q_high": r_high}
    if "miss_reason" in a:
        reasons = a["miss_reason"][a["miss_reason"] != ""]
        cnt = Counter(reasons.tolist())
        n = max(len(reasons), 1)
        out["miss_reasons"] = {k: v / n for k, v in cnt.most_common()}
        relax = {}
        for name, key, q80 in (("completeness", "miss_completeness", 20), ("dD", "miss_d_over_D", 20),
                               ("ellipticity", "miss_ellipticity", 80)):
            if cnt.get(name, 0) / n >= 0.3:
                vals = a[key][a["miss_reason"] == name]
                relax[name] = float(np.percentile(vals[np.isfinite(vals)], q80))
        if cnt.get("ok", 0) / n >= 0.3:
            relax["seed_dD_min_factor"] = 0.5
        out["miss_relax"] = relax
    out["set_flags"] = [f"--set detector.kappa={kappa:.4f}", f"--set detector.beta={beta:.4f}",
                        f"--set detector.pair_angle_max_deg={pair:g}", f"--set detector.D_min_px={D_min:g}",
                        f"--set match.sigma_min_px={sigma_min:.3f}", f"--set match.sigma_rel={sigma_rel:.4f}",
                        f"--set match.visibility_filter={json.dumps(vis_on)}", f"--set match.c_vis={c_vis:g}"]
    (run_dir / "calib_detect.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["detect", "loc"])
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    out = cmd_detect(args.run_dir) if args.cmd == "detect" else cmd_loc(args.run_dir)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list) or k == "set_flags"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

(`cmd_loc`은 Task 13에서 추가한다. 그 전까지는 `detect`만 쓴다. 파일을 실행 가능한 상태로 두려면 Task 13 전까지 `cmd_loc = None` 한 줄을 `main` 위에 둔다.)

- [ ] **Step 6: Pass A0 — 느슨한 주석으로 κ·β·θ_pair·누락 사유**

Run:
```bash
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --dem data/dem/ldem_875s_5m.img --out outputs/pg/calibA0 --split calib --detect-only --workers 40 \
    --set detector.pair_angle_max_deg=90 --set detector.D_min_px=4 --ann-tol-rel 0.4 --ann-size-ratio 2.5
.venv/bin/python scripts/analyze_pg_calib.py detect outputs/pg/calibA0
```
Expected: 339 레코드(113 기하 × 3 조명), `calib_detect.json`. 출력의 `miss_reasons`, `miss_relax`를 본다.

- [ ] **Step 7: 카탈로그 완화 규칙 적용 (해당할 때만, 한 번)**

`miss_relax`가 비어 있지 않으면 `CatalogConfig` 기본값을 그 값으로 바꾸고(`seed_dD_min_factor`면 `seed_dD_min`을 절반으로), Task 5 Step 5의 명령으로 카탈로그를 다시 만든 뒤 Step 6을 반복한다. 두 번째에는 완화하지 않는다. 바꾼 값과 사유 비율을 decisions G3 근거에 적는다.

- [ ] **Step 8: Pass A1 — 보정된 κ·β·θ_pair로 재현율·σ 모형·가시성**

Run (`<...>`는 Step 6 또는 7의 `calib_detect.json` 값):
```bash
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --out outputs/pg/calibA1 --split calib --detect-only --workers 40 --set detector.D_min_px=4 \
    --set detector.kappa=<kappa> --set detector.beta=<beta> --set detector.pair_angle_max_deg=<pair_angle_max_deg>
.venv/bin/python scripts/analyze_pg_calib.py detect outputs/pg/calibA1
```
Expected: `size_factor`가 1에 가깝고(0.9–1.1), 새 `beta`가 입력 β와 거의 같다(±0.05). 아니면 A1 값으로 한 번 더 돌린다. 여기서 `D_min_px`, `sigma_min_px`, `sigma_rel`, `visibility_filter`, `c_vis`를 확정한다.

- [ ] **Step 9: 기본값 반영과 decisions §G**

`DetectorConfig`의 `kappa`, `beta`, `pair_angle_max_deg`, `D_min_px`와 `MatchConfig`의 `sigma_min_px`, `sigma_rel`, `visibility_filter`, `c_vis` 기본값을 확정값으로 바꾸고 주석에 "calibrated, decisions G…"를 남긴다. `docs/decisions.md` 끝에 다음 절을 추가하고 값을 채운다(값은 `calib_detect.json`에서 그대로 옮긴다).

```markdown
## G. P-G 크레이터 기준선 (보정 표본에서 정함)

스펙: `docs/superpowers/specs/2026-10-12-pg-crater-baseline-design.md`. 보정 분석: `scripts/analyze_pg_calib.py`.

| ID | 값 | 근거 | 코드 |
|---|---|---|---|
| G1 | 카탈로그 씨앗: DEM 피라미드 ×(1,2,4,8,16,32,64), 씨앗 반지름 5/6.3/7.9 층-px, LoG σ = 0.45·r, d/D 대리 ≥ 0.02 | 테두리 있는 사발형은 σ ≈ 0.44 r에서 LoG 최대(합성 시험). σ = r/√2로 두면 반지름을 0.6배로 과소평가해 큰 크레이터를 놓쳤다 | `catalog.py` |
| G2 | 테두리: 48방위, 단면 0–1.8r, 기울기 제거 고리 1.5–2.5r, 탐색 0.5–1.6r, MAD×2.5 | 비공개 값. 합성 시험에서 중심 오차 95%ile ≤ 0.05 D | `catalog.py` |
| G3 | 필터: D 50 m–6 km, 완결도 ≥ <값>, a/b ≤ <값>, d/D ≥ <값> | LOLA 5 m 극지 DEM 실측 채움률 ~10%(Barker), 궤적 흔적 지름 10–30 m. 누락 사유 분포: <miss_reasons>. 완화: <miss_relax 또는 "없음"> | `catalog.py` |
| G4 | 중복 제거: 중심 < 0.3 D, 크기비 < 1.5 | — | `catalog.py` |
| G5 | 검출 전처리: 가우스 1 px, 국소창 σ 25 px, 표준편차 하한 0.02, 문턱 0.5–2.5σ 7단계, 안정 = 이동 ≤ max(1, 0.25 r_eq)·면적비 ≥ 0.5, 피라미드 3층(경계 32/64 px) | CNav 임계 스윕(수치 비공개) | `detect.py` |
| G6 | 짝짓기: 최근접 밝은 영역 2개, 면적비 ≤ 10(CNav 원문), 거리 ≤ 1.5·(r_eq 합), 방향 편차 ≤ <pair_angle_max_deg>°, 태양 출처 = 천체력 | 대응 있는 짝의 98.5%ile | `detect.py` |
| G7 | 모멘트 타원 κ = <kappa>, β = <beta>, D_min_px = <D_min_px> | κ·β: 대응 검출의 크기비·반태양 축 오차 중앙값. D_min_px: 재현율 ≥ 0.5 | `detect.py` |
| G8 | 가시성 필터 <켬/끔>, c_vis = <c_vis> | 재현율(q<0.75) = <recall_q_low>, 재현율(q≥1) = <recall_q_high> | `match.py` |
| G9 | 위치 투표: 반경 3σ_h + 300 m(T-I C3와 같음), 칸 2 × 영상 GSD, 가우스 1칸, 상위 3봉우리, 배제 반경 max(4칸, 시야 폭 10%), 투표 크기 호환 ln 1.6 | 비공개 값(보정하지 않음). 저고도에서 사전 오차(±3 km)가 시야(2.9 km)보다 커 최근접 짝짓기가 불가능하다 | `match.py` |
| G10 | 할당 게이트: 1차 max(4 px, 0.4 D_px), 2차 max(2 px, 0.2 D_px), 크기비 ≤ 1.6, 위치 RANSAC 3 px | 비공개 값(보정하지 않음). 고정 판정은 G12가 맡는다 | `match.py` |
| G11 | 크레이터 σ = max(<sigma_min_px>, <sigma_rel>·D_px) px | 보정 후 잔차의 D_px 구간별 강건 표준편차 | `match.py` |
```

- [ ] **Step 10: Recall figure (`pg_figures.py recall`)**

```python
# scripts/pg_figures.py (insert above main)
def _annots(run_dir: Path) -> dict:
    acc: dict = {}
    for p in sorted((run_dir / "annot").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                acc.setdefault(k.split("__", 1)[1], []).append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def cmd_recall(args):
    a = _annots(args.run_dir)
    D, hit = a["rc_D_px"], a["rc_detected"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = a["rc_d_over_D"] / (np.tan(np.radians(a["rc_sun_el_deg"])) / 4.0)
    edges = np.array([4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128], float)
    centres = np.sqrt(edges[:-1] * edges[1:])

    def curve(mask):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = mask & (D >= lo) & (D < hi)
            out.append(hit[s].mean() if s.sum() >= 30 else np.nan)
        return np.array(out)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogx(centres, curve(np.ones_like(hit)), "-o", color=S1, lw=2, ms=5, label="all catalog craters in view")
    ax.semilogx(centres, curve(np.nan_to_num(q, nan=-1) >= 1.0), "-o", color=S2, lw=2, ms=5,
                label="interior shadow expected (q >= 1)")
    ax.set_ylim(0, 1)
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:g}" for e in edges])
    ax.minorticks_off()
    style(ax, "Detection probability vs crater size (calibration split)", "catalog crater diameter (px)",
          "fraction detected")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out / "pg_recall.png", dpi=110)
    plt.close(fig)
```

`main`에 부명령을 등록한다: `r = sub.add_parser("recall"); r.add_argument("--run-dir", type=Path, required=True); r.add_argument("--out", type=Path, required=True)`, 사전에 `"recall": cmd_recall`.

Run: `.venv/bin/python scripts/pg_figures.py recall --run-dir outputs/pg/calibA1 --out outputs/pg/fig` — 그림을 열어 곡선이 D_px와 함께 오르는지 확인한다.

- [ ] **Step 11: 체크포인트** — `.venv/bin/python -m pytest -m "not slow" -q` 통과(기본값이 바뀌어도 단위 시험은 주입 검출·합성 영상이라 영향이 없어야 한다. 실패하면 원인을 본다). 커밋하지 않는다.

---

### Task 13: 보정 B — 고정 게이트와 공분산 바닥 (카탈로그 하한 50 m, 80 m 두 조건)

**Files:**
- Modify: `scripts/analyze_pg_calib.py` (`cmd_loc` 추가)
- Modify: `src/lunar_trn/craters/localize.py` (`CraterConfig` 기본값)
- Modify: `docs/decisions.md` (G12, G13, G14, D_min 민감도 조건)

**Interfaces:**
- Consumes: Task 11 `runs.jsonl`(국지화 모드), Task 10 게이트 필드
- Produces: `calib_loc.json` 키 `n_min, s_max, r_min, retention, floor_h_px, floor_v_px, coverage_2sigma, by_band, set_flags`

**규칙 (평가 표본을 보기 전에 고정한다)**
- 게이트를 풀고 돌린다: `min_inliers_fix=2`, `min_match_ratio=0`, `max_sigma0=1e9`.
- N_min = 5 (CNav, 고정).
- s_max = 정답(오차 ≤ 허용치, inlier ≥ 5) 실행의 σ0 99.5%ile을 0.05 단위로 올림.
- r_min = inlier ≥ 5 ∧ σ0 ≤ s_max를 통과한 실행 중 오답(오차 > 허용치)이 하나도 남지 않는 가장 작은 매칭 비율(0.01 단위). 오답이 없으면 0.
- 정답 보존률(게이트 통과 정답 / inlier ≥ 5 정답)을 기록한다. 0.9 미만이면 그 사실과 대역별 손실을 decisions에 적는다(게이트는 그대로 둔다).
- 공분산 바닥: 게이트를 통과한 정답의 축별 수평 오차를 영상 GSD(`gsd_m`)로 나눈 RMS(수평 축 평균), 수직도 같은 방식. T-I D7b와 같은 절차.
- 50 m(기본 카탈로그)과 80 m(`match.catalog_D_min_m=80`) 조건마다 따로 정한다. 기본값에는 50 m 조건 값을 넣고, 80 m 조건 값은 decisions에 `--set` 묶음으로 남긴다.

- [ ] **Step 1: Implement `cmd_loc`** (`cmd_loc = None` 줄을 지우고 `main` 위에 둔다)

```python
# scripts/analyze_pg_calib.py (insert above main, replacing "cmd_loc = None")
def cmd_loc(run_dir: Path) -> dict:
    runs = [json.loads(line) for line in (run_dir / "runs.jsonl").open()]
    pose = [r for r in runs if r.get("stage") == "pose" and r["health"]["pg"].get("n_inliers", 0) >= 5]
    pg = lambda r, k: r["health"]["pg"].get(k)  # noqa: E731
    right = [r for r in pose if r["err_h_m"] <= r["tol_m"]]
    wrong = [r for r in pose if r["err_h_m"] > r["tol_m"]]
    s0 = np.array([pg(r, "sigma0") for r in right], float)
    s_max = float(math.ceil(np.percentile(s0, 99.5) / 0.05) * 0.05)
    w = [pg(r, "catalog_match_ratio") for r in wrong if pg(r, "sigma0") <= s_max]
    r_min = float(math.ceil((max(w) + 1e-9) * 100.0) / 100.0) if w else 0.0
    passed = [r for r in right if pg(r, "sigma0") <= s_max and pg(r, "catalog_match_ratio") >= r_min]
    retention = len(passed) / max(len(right), 1)
    eh = np.array([[r["err_e_m"] / r["gsd_m"], r["err_n_m"] / r["gsd_m"]] for r in passed])
    ev = np.array([r["err_u_m"] / r["gsd_m"] for r in passed])
    floor_h = float(np.sqrt(np.mean(eh**2))) if len(passed) else 0.0
    floor_v = float(np.sqrt(np.mean(ev**2))) if len(passed) else 0.0
    cover = []
    for r in passed:
        lm = r.get("cov_lm_enu_diag_m2")
        if lm is None:
            continue
        g = r["gsd_m"]
        sx, sy = math.sqrt(lm[0] + (floor_h * g) ** 2), math.sqrt(lm[1] + (floor_h * g) ** 2)
        cover.append(abs(r["err_e_m"]) <= 2 * sx and abs(r["err_n_m"]) <= 2 * sy)
    by_band = {}
    for b in sorted({r["band"] for r in runs if "band" in r}):
        rb = [r for r in runs if r.get("band") == b]
        ok = [r for r in rb if r.get("stage") == "pose" and pg(r, "n_inliers") >= 5 and pg(r, "sigma0") <= s_max
              and pg(r, "catalog_match_ratio") >= r_min]
        by_band[b] = {"runs": len(rb), "fix": len(ok), "correct": sum(r["err_h_m"] <= r["tol_m"] for r in ok),
                      "correct_n20": sum(r["err_h_m"] <= r["tol_m"] and pg(r, "n_inliers") >= 20 for r in ok)}
    out = {"n_min": 5, "s_max": s_max, "r_min": r_min, "retention": retention, "n_right": len(right),
           "n_wrong": len(wrong), "floor_h_px": floor_h, "floor_v_px": floor_v,
           "coverage_2sigma": float(np.mean(cover)) if cover else float("nan"), "by_band": by_band,
           "wrong_sigma0": sorted(float(pg(r, "sigma0")) for r in wrong),
           "wrong_match_ratio": sorted(float(pg(r, "catalog_match_ratio")) for r in wrong)}
    out["set_flags"] = ["--set min_inliers_fix=5", f"--set max_sigma0={s_max:g}", f"--set min_match_ratio={r_min:g}",
                        f"--set cov_floor_h_px={floor_h:.3f}", f"--set cov_floor_v_px={floor_v:.3f}"]
    (run_dir / "calib_loc.json").write_text(json.dumps(out, indent=2))
    return out
```

- [ ] **Step 2: Run calibration B (both catalog floors)**

Run:
```bash
LOOSE="--set min_inliers_fix=2 --set min_match_ratio=0 --set max_sigma0=1e9"
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --out outputs/pg/calibB --split calib --trials 2 --workers 40 $LOOSE
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --out outputs/pg/calibB_D80 --split calib --trials 2 --workers 40 $LOOSE --set match.catalog_D_min_m=80
.venv/bin/python scripts/analyze_pg_calib.py loc outputs/pg/calibB
.venv/bin/python scripts/analyze_pg_calib.py loc outputs/pg/calibB_D80
```
Expected: 각 678회. `calib_loc.json` 두 개. `coverage_2sigma`가 0.86–0.99 범위인지 본다(T-I D7b는 0.998).

- [ ] **Step 3: 기본값과 decisions**

`CraterConfig`의 `min_match_ratio`, `max_sigma0`, `cov_floor_h_px`, `cov_floor_v_px` 기본값을 `calibB` 값으로 바꾼다. `docs/decisions.md` §G 표에 추가한다.

```markdown
| G12 | 고정 = inlier ≥ 5 ∧ 카탈로그 매칭 비율 ≥ <r_min> ∧ σ0 ≤ <s_max> | inlier 5: CNav 남극(최소 5개). 매칭 비율: SLIM의 영역별 크레이터 매칭 비율 임계(수치 비공개)와 같은 역할. σ0: LVS Mahalanobis와 같은 일관성 검사. 사전 점검(보정 표본 29기하)에서 오고정은 σ0 1.93–2.36, 매칭 비율 0.03–0.08이었고 정답은 σ0 95%ile 1.51, 매칭 비율 5%ile 0.10이었다. 정답 보존률 <retention> | `craters/localize.py` |
| G13 | 공분산 바닥 수평 <floor_h_px>, 수직 <floor_v_px> 영상-GSD px | T-I D7b와 같은 절차. 2σ 포함률 <coverage_2sigma> | `craters/localize.py` |
| G14 | 채점 허용치 = 3 × 명목 정밀 지도 px(T-I와 같은 미터) | 계열 비교 | `scripts/run_pg.py` |
| G15 | 카탈로그 하한 민감도 조건: `match.catalog_D_min_m=80` + <calibB_D80의 set_flags> | H-c(카탈로그 최소 크기 하한)와 LOLA 궤적 흔적(10–30 m) 영향 분리 | `match.py` |
```

- [ ] **Step 4: 체크포인트** — 빠른 시험 통과. `calibB`의 대역별 `fix`/`correct`를 보고, 게이트가 특정 대역을 통째로 막는지 확인해 메모한다. 커밋하지 않는다.

---

### Task 14: 집계 (`craters/report.py`, `scripts/analyze_pg.py`)

**Files:**
- Create: `src/lunar_trn/craters/report.py`
- Create: `scripts/analyze_pg.py`
- Test: `tests/test_report_pg.py`

**Interfaces:**
- Produces:
  - `cp_upper(x, n, conf=0.95)`, `pct(values, q)`
  - `regate(run, n_min=None, r_min=None, s_max=None) -> "correct" | "false_fix" | "no_fix" | "error"`
  - `summarize(runs, **gates) -> dict` 키 `n_runs, n_correct, n_false_fix, n_no_fix, n_error, availability, false_fix_rate, false_fix_rate_ub95, false_fix_given_fix, false_fix_given_fix_ub95, err_h_p50_m, err_h_p95_m, err_h_p99_m, err_h_p95_px, err_u_abs_p95_m, n_inliers_p50, n_detections_p50, n_catalog_view_p50, t_run_p50_s`
  - `group_by(runs, key) -> dict`
  - `pair_with_ti(pg_runs, ti_runs) -> {"matched": {both, pg_only, ti_only, neither, unpaired}, "cross": [ {map_illum, image_illum, ti_runs, ti_availability, ti_false_fix, pg_runs, pg_availability, pg_false_fix} ]}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report_pg.py
from lunar_trn.craters.report import pair_with_ti, regate, summarize


def _run(verdict="correct", n_inl=10, cmr=0.2, s0=1.0, err=1.0, stage="pose", band=0, j=0, g=1, trial=0):
    return {"verdict": verdict, "stage": stage, "err_h_m": err, "tol_m": 3.0, "err_u_m": 0.5, "map_px_fine": 1.0,
            "band": band, "image_illum": j, "geometry_id": g, "trial": trial, "t_run_s": 1.0,
            "health": {"pg": {"n_inliers": n_inl, "catalog_match_ratio": cmr, "sigma0": s0, "n_detections": 50,
                              "n_catalog_view": 60}}}


def test_summarize_counts_and_bounds():
    runs = [_run(), _run(err=2.0), _run(err=0.5), _run("false_fix", err=10.0), _run("no_fix", stage="detect")]
    s = summarize(runs)
    assert (s["n_correct"], s["n_false_fix"], s["n_no_fix"]) == (3, 1, 1)
    assert abs(s["availability"] - 0.6) < 1e-12 and s["false_fix_rate_ub95"] > 0.2
    assert abs(s["err_h_p50_m"] - 1.0) < 1e-12


def test_regate():
    assert regate(_run(n_inl=6), n_min=8) == "no_fix"
    assert regate(_run("false_fix", err=10.0, cmr=0.05), r_min=0.1) == "no_fix"
    assert regate(_run("false_fix", err=10.0, cmr=0.5), r_min=0.1) == "false_fix"
    assert regate(_run(s0=1.0), s_max=0.5) == "no_fix"
    assert regate(_run("no_fix", stage="detect"), n_min=2) == "no_fix"
    assert regate({"error": "x"}) == "error"


def test_pair_with_ti():
    pg = [_run(j=0), _run("no_fix", j=1, stage="pose", n_inl=3)]
    common = {"geometry_id": 1, "trial": 0, "err_h_m": 1.0, "tol_m": 3.0, "err_u_m": 0.1, "map_px_fine": 1.0}
    ti = [{**common, "map_illum": 0, "image_illum": 0, "verdict": "no_fix"},
          {**common, "map_illum": 1, "image_illum": 1, "verdict": "correct"},
          {**common, "map_illum": 0, "image_illum": 1, "verdict": "no_fix"}]
    out = pair_with_ti(pg, ti)
    assert out["matched"] == {"both": 0, "pg_only": 1, "ti_only": 1, "neither": 0, "unpaired": 0}
    keys = {(c["map_illum"], c["image_illum"]) for c in out["cross"]}
    assert keys == {(0, 0), (0, 1), (1, 1)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_report_pg.py -v`
Expected: FAIL (`ModuleNotFoundError: lunar_trn.craters.report`)

- [ ] **Step 3: Implement `src/lunar_trn/craters/report.py`**

```python
# src/lunar_trn/craters/report.py
"""Aggregation of P-G runs: V7 tables, fix-gate re-evaluation and pairing with T-I runs.

Upper bounds on rates are one-sided 95 % Clopper-Pearson (about 3/N for zero events).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import beta


def cp_upper(x: int, n: int, conf: float = 0.95) -> float:
    if n == 0:
        return float("nan")
    return 1.0 if x >= n else float(beta.ppf(conf, x + 1, n - x))


def pct(values, q) -> float:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return float(np.percentile(a, q)) if a.size else float("nan")


def regate(run: dict, n_min=None, r_min=None, s_max=None) -> str:
    """Verdict of a recorded run under other fix gates (no gate given: the recorded verdict)."""
    if "error" in run:
        return "error"
    if n_min is None and r_min is None and s_max is None:
        return run.get("verdict", "error")
    if run.get("stage") != "pose":
        return "no_fix"
    pg = run.get("health", {}).get("pg", {})
    if n_min is not None and pg.get("n_inliers", 0) < n_min:
        return "no_fix"
    if r_min is not None and not pg.get("catalog_match_ratio", 0.0) >= r_min:
        return "no_fix"
    if s_max is not None and not pg.get("sigma0", np.inf) <= s_max:
        return "no_fix"
    return "correct" if run["err_h_m"] <= run["tol_m"] else "false_fix"


def summarize(runs: list[dict], **gates) -> dict:
    v = [regate(r, **gates) for r in runs]
    n = len(runs)
    n_ok, n_ff = v.count("correct"), v.count("false_fix")
    n_fix = n_ok + n_ff
    corr = [r for r, x in zip(runs, v) if x == "correct"]

    def pg(r, k):
        return r.get("health", {}).get("pg", {}).get(k)

    nan = float("nan")
    return {"n_runs": n, "n_correct": n_ok, "n_false_fix": n_ff, "n_no_fix": v.count("no_fix"),
            "n_error": v.count("error"), "availability": n_ok / n if n else nan,
            "false_fix_rate": n_ff / n if n else nan, "false_fix_rate_ub95": cp_upper(n_ff, n),
            "false_fix_given_fix": n_ff / n_fix if n_fix else nan,
            "false_fix_given_fix_ub95": cp_upper(n_ff, n_fix) if n_fix else nan,
            "err_h_p50_m": pct([r["err_h_m"] for r in corr], 50), "err_h_p95_m": pct([r["err_h_m"] for r in corr], 95),
            "err_h_p99_m": pct([r["err_h_m"] for r in corr], 99),
            "err_h_p95_px": pct([r["err_h_m"] / r["map_px_fine"] for r in corr], 95),
            "err_u_abs_p95_m": pct([abs(r["err_u_m"]) for r in corr], 95),
            "n_inliers_p50": pct([pg(r, "n_inliers") for r in corr], 50),
            "n_detections_p50": pct([pg(r, "n_detections") for r in runs], 50),
            "n_catalog_view_p50": pct([pg(r, "n_catalog_view") for r in runs], 50),
            "t_run_p50_s": pct([r.get("t_run_s") for r in runs], 50)}


def group_by(runs: list[dict], key: str) -> dict:
    out = defaultdict(list)
    for r in runs:
        out[r.get(key)].append(r)
    return dict(sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0)))


def pair_with_ti(pg_runs: list[dict], ti_runs: list[dict]) -> dict:
    """Contingency with the matched T-I runs and the T-I (map, image) illumination table next to P-G."""
    ti = {(r["geometry_id"], r["map_illum"], r["image_illum"], r["trial"]): r for r in ti_runs if "verdict" in r}
    cont = {"both": 0, "pg_only": 0, "ti_only": 0, "neither": 0, "unpaired": 0}
    for r in pg_runs:
        t = ti.get((r.get("geometry_id"), r.get("image_illum"), r.get("image_illum"), r.get("trial")))
        if t is None or "verdict" not in r:
            cont["unpaired"] += 1
            continue
        a, b = r["verdict"] == "correct", t["verdict"] == "correct"
        cont["both" if a and b else "pg_only" if a else "ti_only" if b else "neither"] += 1
    pg_by_j = group_by([r for r in pg_runs if "verdict" in r], "image_illum")
    cross = []
    for k, j in sorted({(r["map_illum"], r["image_illum"]) for r in ti_runs if "verdict" in r}):
        st = summarize([r for r in ti_runs if r["map_illum"] == k and r["image_illum"] == j and "verdict" in r])
        sp = summarize(pg_by_j.get(j, []))
        cross.append({"map_illum": k, "image_illum": j, "ti_runs": st["n_runs"], "ti_availability": st["availability"],
                      "ti_false_fix": st["n_false_fix"], "pg_runs": sp["n_runs"],
                      "pg_availability": sp["availability"], "pg_false_fix": sp["n_false_fix"]})
    return {"matched": cont, "cross": cross}
```

- [ ] **Step 4: Implement `scripts/analyze_pg.py`**

```python
# scripts/analyze_pg.py
#!/usr/bin/env python3
"""V7 / H-b tables of run_pg.py results (summary.json and summary.md next to runs.jsonl).

    python scripts/analyze_pg.py outputs/pg/v7_nadir --ti outputs/v2/nadir/runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lunar_trn.craters.report import group_by, pair_with_ti, pct, summarize

NOMINAL_BANDS_KM = [3.5 + 3 * k for k in range(10)]
ILLUM = ("side", "overhead", "back")
COLS = [("runs", "n_runs", "d"), ("avail", "availability", ".3f"), ("false fix", "n_false_fix", "d"),
        ("ub95", "false_fix_rate_ub95", ".4f"), ("err p50/p95 (m)", None, None), ("p95 (px)", "err_h_p95_px", ".2f"),
        ("|e_u| p95 (m)", "err_u_abs_p95_m", ".1f"), ("inliers p50", "n_inliers_p50", ".0f"),
        ("det p50", "n_detections_p50", ".0f"), ("cat view p50", "n_catalog_view_p50", ".0f"),
        ("t p50 (s)", "t_run_p50_s", ".2f")]


def _row(label, s):
    cells = [label]
    for _, key, fmt in COLS:
        cells.append(f"{s['err_h_p50_m']:.1f} / {s['err_h_p95_m']:.1f}" if key is None else format(s[key], fmt))
    return "| " + " | ".join(cells) + " |"


def _table(title, first, items):
    lines = [f"### {title}", "", "| " + " | ".join([first] + [c[0] for c in COLS]) + " |",
             "|" + "---|" * (len(COLS) + 1)]
    lines += [_row(lbl, s) for lbl, s in items]
    return lines + [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ti", type=Path, default=None, help="T-I runs.jsonl for the paired comparison")
    ap.add_argument("--alt-gates", default="5,20", help="inlier gates to re-evaluate with the recorded r/sigma0 gates")
    args = ap.parse_args(argv)
    runs = [json.loads(line) for line in (args.run_dir / "runs.jsonl").open()]
    crater = json.loads((args.run_dir / "config.json").read_text())["crater"]
    r_min, s_max = crater["min_match_ratio"], crater["max_sigma0"]
    bands = group_by(runs, "band")
    out = {"overall": summarize(runs), "by_band": {b: summarize(rs) for b, rs in bands.items()},
           "by_illum": {j: summarize(rs) for j, rs in group_by(runs, "image_illum").items()},
           "alt_gates": {n: {b: summarize(rs, n_min=int(n), r_min=r_min, s_max=s_max) for b, rs in bands.items()}
                         for n in args.alt_gates.split(",")}}
    lines = [f"# P-G results: {args.run_dir}", ""]
    lines += _table("By altitude band", "band (km)",
                    [(f"{NOMINAL_BANDS_KM[b]:.1f} (GSD {pct([r['gsd_m'] for r in bands[b]], 50):.1f} m)", s)
                     for b, s in out["by_band"].items()])
    lines += _table("By image illumination", "illumination",
                    [(f"{j} {ILLUM[j]}", s) for j, s in out["by_illum"].items()])
    lines += _table("Overall", "", [("all", out["overall"])])
    lines += ["### Availability under other inlier gates (same match-ratio and sigma0 gates)", "",
              "| band (km) | " + " | ".join(f"N>={n}: avail / false fix" for n in out["alt_gates"]) + " |",
              "|---|" + "---|" * len(out["alt_gates"])]
    for b in bands:
        lines.append(f"| {NOMINAL_BANDS_KM[b]:.1f} | " + " | ".join(
            f"{g[b]['availability']:.3f} / {g[b]['n_false_fix']}" for g in out["alt_gates"].values()) + " |")
    lines.append("")
    if args.ti:
        ti = [json.loads(line) for line in args.ti.open()]
        pair = pair_with_ti(runs, ti)
        out["pair_ti"] = pair
        m = pair["matched"]
        lines += ["### Paired with T-I (same geometry, illumination and prior)", "",
                  f"both correct {m['both']}, P-G only {m['pg_only']}, T-I only {m['ti_only']}, "
                  f"neither {m['neither']}, unpaired {m['unpaired']}", "",
                  "| map illum | image illum | T-I avail | T-I false fix | P-G avail | P-G false fix |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {c['map_illum']} | {c['image_illum']} | {c['ti_availability']:.3f} | {c['ti_false_fix']} | "
                  f"{c['pg_availability']:.3f} | {c['pg_false_fix']} |" for c in pair["cross"]]
        lines.append("")
    (args.run_dir / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    (args.run_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests and a dry analysis**

Run: `.venv/bin/python -m pytest tests/test_report_pg.py -v && .venv/bin/python scripts/analyze_pg.py outputs/pg/calibB --ti outputs/v2/nadir/runs.jsonl | head -40`
Expected: 3 passed. 보정 B 결과 표가 나온다(짝 비교는 보정 표본이라 `unpaired`가 대부분 — T-I V2 평가는 평가 표본만 돌렸다. 정상).

- [ ] **Step 6: 체크포인트** — 빠른 시험 통과. 커밋하지 않는다.

---

### Task 15: 평가 — V7·H-b·카탈로그 하한 민감도, 보고서

**Files:**
- Modify: `scripts/pg_figures.py` (`v7` 부명령)
- Create: `docs/validation/v7_pg_nadir.md`, `docs/validation/fig/pg_*.png`

**Interfaces:**
- Consumes: 보정된 기본값(Task 12, 13), `analyze_pg.py`, T-I 결과 `outputs/v2/nadir/runs.jsonl`, `outputs/v3/nadir_sweep*/runs.jsonl`

- [ ] **Step 1: Eval runs (평가 표본, 보정값 고정)**

Run (`<G15 set flags>`는 decisions G15에 적은 80 m 조건 묶음):
```bash
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --out outputs/pg/v7_nadir --split eval --trials 6 --workers 40
.venv/bin/python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
    --out outputs/pg/v7_nadir_D80 --split eval --trials 6 --workers 40 <G15 set flags>
.venv/bin/python scripts/analyze_pg.py outputs/pg/v7_nadir --ti outputs/v2/nadir/runs.jsonl
.venv/bin/python scripts/analyze_pg.py outputs/pg/v7_nadir_D80 --ti outputs/v2/nadir/runs.jsonl
```
Expected: 각 6,066회, `error` 0건. 짝 비교의 `unpaired`가 0이다(같은 평가 표본·시드).

- [ ] **Step 2: V7 figures (`pg_figures.py v7`)**

```python
# scripts/pg_figures.py (insert above main)
def _runs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def cmd_v7(args):
    from lunar_trn.craters.report import group_by, pct, summarize

    bands_km = np.array([3.5 + 3 * k for k in range(10)])
    pg, ti = _runs(args.pg / "runs.jsonl"), _runs(args.ti)
    ti_m = [r for r in ti if r.get("map_illum") == r.get("image_illum")]
    series = [("T-I (map Sun = image Sun)", ti_m, S1), ("P-G, catalog D >= 50 m", pg, S2)]
    if args.pg_alt:
        series.append(("P-G, catalog D >= 80 m", _runs(args.pg_alt / "runs.jsonl"), S3))
    args.out.mkdir(parents=True, exist_ok=True)
    # 1. Availability by band.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, rs, col in series:
        g = group_by(rs, "band")
        ax.plot(bands_km[list(g)], [summarize(v)["availability"] for v in g.values()], "-o", color=col, lw=2, ms=5,
                label=label)
    ax.set_ylim(0, 1.02)
    style(ax, "Availability by altitude band (eval split, nadir)", "nominal altitude (km)", "fraction correct fixes")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "pg_v7_availability.png", dpi=110)
    plt.close(fig)
    # 2. Matched craters and catalog craters in view vs altitude (P-G main condition).
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    g = group_by([r for r in pg if r.get("stage") == "pose"], "band")
    x = bands_km[list(g)]
    for key, col, label in (("n_catalog_view", S1, "catalog craters in view"), ("n_inliers", S2, "matched (inliers)")):
        p50 = [pct([r["health"]["pg"].get(key) for r in v], 50) for v in g.values()]
        p10 = [pct([r["health"]["pg"].get(key) for r in v], 10) for v in g.values()]
        p90 = [pct([r["health"]["pg"].get(key) for r in v], 90) for v in g.values()]
        ax.plot(x, p50, "-o", color=col, lw=2, ms=5, label=f"{label} (median, 10-90%)")
        ax.fill_between(x, p10, p90, color=col, alpha=0.15, lw=0)
    ax.set_yscale("log")
    style(ax, "Craters per image vs altitude (P-G, D >= 50 m)", "nominal altitude (km)", "count per image")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out / "pg_v7_counts.png", dpi=110)
    plt.close(fig)
    # 3. H-b: T-I availability vs map/image Sun angle; P-G has no map Sun (horizontal reference).
    sweep = ti + [r for p in args.sweeps for r in _runs(p)]
    ang = np.array([r.get("sun_angle_map_image_deg", np.nan) for r in sweep], float)
    ok = np.array([r.get("verdict") == "correct" for r in sweep])
    edges = np.arange(0, 181, 15)
    cen, av = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (ang >= lo) & (ang < hi)
        if s.sum() >= 50:
            cen.append(0.5 * (lo + hi))
            av.append(ok[s].mean())
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(cen, av, "-o", color=S1, lw=2, ms=5, label="T-I (rendered map at another Sun)")
    a_pg = summarize(pg)["availability"]
    ax.axhline(a_pg, color=S2, lw=2, label=f"P-G (Sun-independent catalog), {a_pg:.2f}")
    ax.set_ylim(0, 1.02)
    style(ax, "Illumination residual (H-b): all bands pooled", "Sun angle between map and image (deg)",
          "fraction correct fixes")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="center right")
    fig.tight_layout()
    fig.savefig(args.out / "pg_hb_sun_angle.png", dpi=110)
    plt.close(fig)
```

`import json`을 파일 위 import에 더하고, `main`에 부명령을 등록한다: `v = sub.add_parser("v7"); v.add_argument("--pg", type=Path, required=True); v.add_argument("--pg-alt", type=Path, default=None); v.add_argument("--ti", type=Path, required=True); v.add_argument("--sweeps", type=Path, nargs="*", default=[]); v.add_argument("--out", type=Path, required=True)`, 사전에 `"v7": cmd_v7`.

Run:
```bash
.venv/bin/python scripts/pg_figures.py v7 --pg outputs/pg/v7_nadir --pg-alt outputs/pg/v7_nadir_D80 \
    --ti outputs/v2/nadir/runs.jsonl --sweeps outputs/v3/nadir_sweep/runs.jsonl outputs/v3/nadir_sweep_wide/runs.jsonl \
    --out outputs/pg/fig
mkdir -p docs/validation/fig && cp outputs/pg/fig/pg_*.png docs/validation/fig/
```
Expected: 그림을 모두 열어 겹침·잘림이 없는지 본다(dataviz 7단계).

- [ ] **Step 3: Write `docs/validation/v7_pg_nadir.md`**

다음 구성으로 쓴다. 수치는 `summary.md`/`summary.json`, `calib_*.json`, `build_catalog.log`에서 옮긴다. 문장은 `docs/validation/v2_v3_nadir.md`와 같은 문체(짧은 평서문, 표 우선)로 쓴다.

```markdown
---
title: "V7 — P-G 크레이터 패턴 매칭 (StereoLunar nadir, 추적형)"
type: validation
lang: ko
created: <날짜>
---

# V7 — P-G 크레이터 패턴 매칭 (추적형)

## 결론
- 합격 판정(스펙 §1.3): 오고정 <n>건(상한95 <ub>), 정답 오차 95%ile 최대 <x> px(대역 <b>), 곡선 보고 여부.
- 대역별 가용성 요약과 T-I 대비.
- H-b: T-I가 0–2%였던 데이터셋 조명 교차(66–134°)에서 P-G 가용성.
- H-c: 카탈로그 하한 50 m 대 80 m의 저고도 가용성 차.

## 1. 방법
(표: 데이터, 실행 수, 카탈로그, 검출기, 매칭, 사전 오차, 판정, 실행 명령, 실행 시간)

## 2. 카탈로그
(총수, 수준별 개수, D 분위수, CSFD 그림 `fig/pg_csfd.png`, DEM 오버레이 `fig/pg_catalog_dem.png`, LOLA 흔적 관찰)

## 3. 검출 확률 (보정 표본)
(`fig/pg_recall.png`, D_px 구간별 재현율 표, q 구간 표, 카탈로그 누락 사유 분포)

## 4. V7 — 대역별
(`summary.md`의 대역 표, `fig/pg_v7_availability.png`, `fig/pg_v7_counts.png`, 영상 오버레이 예 `fig/pg_overlay_*.png`)

## 5. 고정 게이트
(G12 게이트, N=5/20 재판정 표, 정답·오답의 σ0·매칭 비율 분포)

## 6. T-I와 짝 비교, H-b
(짝 분할표, 조명 교차 표, `fig/pg_hb_sun_angle.png`)

## 7. 카탈로그 하한 민감도 (H-c)
(50 m 대 80 m 대역별 가용성·오고정)

## 8. 한계
- 같은 DEM·알베도 일정 → 낙관 편향. LOLA 궤적 흔적이 카탈로그와 영상 양쪽에 있다.
- 모멘트 타원(테두리 적합 아님), 추적형만, nadir만.
```

- [ ] **Step 4: 성공 기준 판정** — `summary.json`으로 스펙 §1.3의 세 항목을 확인해 보고서 결론에 적는다. 오고정이 1건이라도 있으면 합격이 아니다. 이때 평가 표본으로 게이트를 고치지 말고, 해당 실행의 원신호를 보고서에 적고 다음 단계 과제로 남긴다.

- [ ] **Step 5: 체크포인트** — 보고서의 모든 수치가 `summary.json`과 일치하는지 대조한다. 커밋하지 않는다.

---

### Task 16: 가지치기 옵션과 소규모 비교 (IM-2식)

**Files:**
- Modify: `src/lunar_trn/craters/catalog.py` (`prune_catalog`)
- Modify: `scripts/run_pg.py` (`--prune none|matched|cross`)
- Test: `tests/test_catalog.py` (시험 추가)
- Modify: `docs/validation/v7_pg_nadir.md` (§9 추가)

**Interfaces:**
- Produces:
  - `prune_catalog(catalog, src, sun_dir, center_xy, half_size_m, pixel, det_cfg=None, tol_abs_px=2.0, tol_rel=0.25, size_ratio_max=1.5) -> np.ndarray[bool]` (len(catalog), 정사 렌더에서 검출된 크레이터)
  - 레코드 키 `prune_map_illum`, `n_catalog_pruned`; 시드는 `k_map = prune_map_illum`(T-I (k, j) 실행과 짝)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog.py (append)
from lunar_trn.craters.catalog import prune_catalog
from lunar_trn.validation.v1 import sun_from_az_el


def test_prune_keeps_craters_visible_on_a_render():
    dem, truth = synthetic_crater_dem(n=1200, n_craters=25, D_range=(150.0, 500.0), dD_range=(0.12, 0.2), seed=8)
    cat = extract_catalog(dem)
    cx, cy = -20000.0, 30000.0
    from lunar_trn import frames

    sun = sun_from_az_el(frames.stereo_to_body(cx, cy, 0.0), 135.0, 20.0)
    vis = prune_catalog(cat, dem, sun, (cx, cy), 2500.0, 10.0)
    assert vis.shape == (len(cat),) and vis.dtype == bool
    inside = (np.abs(cat.x - cx) < 2300) & (np.abs(cat.y - cy) < 2300)
    outside = (np.abs(cat.x - cx) > 2600) | (np.abs(cat.y - cy) > 2600)  # beyond the 2.5 km render
    assert vis[inside].mean() >= 0.5 and not vis[outside].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_catalog.py::test_prune_keeps_craters_visible_on_a_render -v`
Expected: FAIL (`ImportError: cannot import name 'prune_catalog'`)

- [ ] **Step 3: Implement `prune_catalog`**

```python
# src/lunar_trn/craters/catalog.py (append)
def prune_catalog(catalog: CraterCatalog, src, sun_dir, center_xy, half_size_m: float, pixel: float, det_cfg=None,
                  tol_abs_px: float = 2.0, tol_rel: float = 0.25, size_ratio_max: float = 1.5) -> np.ndarray:
    """IM-2-style visibility pruning: craters the detector finds on an orthographic render with this Sun.

    This makes the catalog depend on the map Sun; it is an option for comparison only.
    """
    from ..maps import render_map, tangent_frame
    from .detect import DetectorConfig, constant_sun_fn, detect_craters

    m = render_map(src, center_xy, half_size_m, pixel, sun_dir)
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    _, Bm = tangent_frame(center_xy)
    st = s @ Bm
    angle = float(np.degrees(np.arctan2(-st[1], st[0])))  # map columns grow east, rows grow south
    img = np.nan_to_num(m.image.astype(np.float64))
    img = (img - img.min()) / max(float(img.max() - img.min()), 1e-12)  # min-max like the dataset images
    det, _ = detect_craters(img, constant_sun_fn(angle), det_cfg or DetectorConfig(), pixel_center=0.0)
    rows, cols = m.shape
    sub, idx = catalog.query_box(m.x0, m.x0 + (cols - 1) * pixel, m.y0 - (rows - 1) * pixel, m.y0, return_index=True)
    visible = np.zeros(len(catalog), bool)
    if len(det) == 0 or len(sub) == 0:
        return visible
    x, y = m.colrow_to_xy(det.u, det.v)
    dist, nb = cKDTree(np.column_stack([x, y])).query(np.column_stack([sub.x, sub.y]), k=1)
    D_det = det.D_px[nb] * pixel
    ok = (dist <= np.maximum(tol_abs_px * pixel, tol_rel * sub.D)) & (np.abs(np.log(D_det / sub.D)) <= np.log(size_ratio_max))
    visible[idx[ok]] = True
    return visible
```

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: 11 passed

- [ ] **Step 4: Add `--prune` to `scripts/run_pg.py`**

import에 `prune_catalog`, `from lunar_trn.dem import DemSource`(이미 있음)를 더한다. 국지화 분기의 `cmap = CraterMap(cat, sun)`부터 trial 루프 끝까지를 아래로 바꾼다.

```python
# scripts/run_pg.py (replace the localisation branch of run_job, from "cmap = CraterMap(cat, sun)")
        enu = frames.enu_basis(T[:3, 3])
        variants = [(j, cat, None)]  # (k_map for the seed, catalog, map illumination of the pruning)
        if job.get("prune", "none") != "none":
            if job["dem"] not in _SRC:
                _SRC[job["dem"]] = DemSource(job["dem"])
            half = 0.5 * max(xmax - xmin, ymax - ymin) + 2 * job["prior_radius_m"] + 800.0
            centre = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
            ks = [j] if job["prune"] == "matched" else [k for k in range(3) if k != j]
            variants = []
            for k in ks:
                sun_k = sl.Frame(d, idx[k], job["traj"], illumination=k, geometry_id=idx[0]).sun_direction()
                vis = prune_catalog(cat, _SRC[job["dem"]], sun_k, centre, half, map_px, cfg.detector)
                variants.append((k, cat.subset(vis), k))
        for k_map, cat_v, prune_k in variants:
            cmap = CraterMap(cat_v, sun)
            for trial in range(job["trials"]):
                seed = zlib.crc32(f"{idx[0]}-{k_map}-{j}-{trial}-{job['seed']}".encode())
                rng = np.random.default_rng(seed)
                Tp, cov, perr = sample_prior(T, rng, job["prior_radius_m"], job["prior_vert_m"], job["prior_att_deg"])
                t1 = time.time()
                try:
                    res = localize_craters(img, cam.K, Tp, cov, cmap, cfg, seed=seed)
                except Exception as err:  # noqa: BLE001
                    out.append({**base, "trial": trial, "prune_map_illum": prune_k, "error": repr(err)})
                    continue
                sc = score(res, T, map_px, tol_px=job["tol_px"])
                rec = {**base, "trial": trial, "prune_map_illum": prune_k, "n_catalog_pruned": len(cat_v),
                       "status": res.status, "stage": res.stage, **perr, **sc, "t_run_s": time.time() - t1,
                       "cov_enu_diag_m2": np.diag(enu.T @ res.cov[:3, :3] @ enu).tolist(), "health": res.health}
                if res.health.get("cov_landmark_only") is not None:
                    lm_cov = np.asarray(res.health["cov_landmark_only"])
                    rec["cov_lm_enu_diag_m2"] = np.diag(enu.T @ lm_cov @ enu).tolist()
                lm = res.landmarks
                if lm and lm["inlier"].any():
                    det_all = Detections.from_ellipses(lm["u"], lm["v"], lm["A"], lm["B"], lm["theta"])
                    gt = annotate_detections(det_all, local, cam, T, sun_fn)["cat_id"]
                    rec["n_inliers_gt_consistent"] = int(np.sum(lm["inlier"] & (gt >= 0) & (gt == lm["cat_id"])))
                out.append(rec)
```

`main`에 `ap.add_argument("--prune", choices=["none", "matched", "cross"], default="none")`를 더하고 작업 사전에 `"prune": args.prune`를 넣는다(가지치기에는 `--dem`이 필요하다: `if args.prune != "none" and not args.dem: ap.error("--prune needs --dem")`).

- [ ] **Step 5: Small comparison run and report section**

Run:
```bash
COMMON="--data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz --dem data/dem/ldem_875s_5m.img --split eval --every 4 --trials 2 --workers 40"
.venv/bin/python scripts/run_pg.py $COMMON --out outputs/pg/prune_none
.venv/bin/python scripts/run_pg.py $COMMON --out outputs/pg/prune_matched --prune matched
.venv/bin/python scripts/run_pg.py $COMMON --out outputs/pg/prune_cross --prune cross
for d in none matched cross; do .venv/bin/python scripts/analyze_pg.py outputs/pg/prune_$d --ti outputs/v2/nadir/runs.jsonl > /dev/null; done
```
`v7_pg_nadir.md`에 "## 9. 가지치기 옵션 (IM-2식) 비교"를 추가한다: 세 조건의 대역별 가용성·오고정 표, 가지친 카탈로그 크기, `prune_cross`에서 가용성이 `prune_matched`보다 떨어지는지(지도 태양 의존이 다시 생기는지)에 대한 한 문단.

- [ ] **Step 6: 체크포인트** — 빠른 시험 통과. 커밋하지 않는다.

---

### Task 17: 문서 마무리

**Files:**
- Modify: `docs/interface.md` (P-G 절), `README.md` (상태·명령·구조), `docs/superpowers/specs/2026-10-12-pg-crater-baseline-design.md` (`status: implemented`, 구현 중 바뀐 점), `docs/decisions.md` (§G 완결 확인)
- Create: `docs/handoff/<YYYY-MM-DD>-pg-session.md`

- [ ] **Step 1: `docs/interface.md`에 P-G 절 추가**

```markdown
## P-G (크레이터) — `localize_craters()`

```python
from lunar_trn.craters.catalog import CraterCatalog
from lunar_trn.craters.localize import CraterConfig, CraterMap, localize_craters

catalog = CraterCatalog.load("data/catalog/ldem_875s_5m_craters.npz")  # scripts/build_catalog.py, Sun-independent
result = localize_craters(image, K, prior_T_world_cam, prior_cov,
                          CraterMap(catalog=catalog, sun_dir_world=sun),  # Sun from the onboard ephemeris
                          CraterConfig())
```

| 항목 | 규약 |
|---|---|
| 지도 | `CraterMap(catalog, sun_dir_world)`. 카탈로그는 DEM에서 한 번 만든 테두리 타원 목록(태양 무관). 태양 방향은 검출기의 그림자 방향에만 쓴다 |
| 출력 | T-I와 같은 `LocalizationResult`, `family = "P-G"`, `stage ∈ {"detect", "match", "pose"}`. `stage == "pose"`이면 게이트에 걸려도 최선 해를 담는다 |
| 고정 | inlier ≥ 5 ∧ 카탈로그 매칭 비율 ≥ `min_match_ratio` ∧ σ0 ≤ `max_sigma0` (decisions G12) |
| 공분산 | 크레이터 잔차 공분산(σ0 > 1이면 σ0²) + 공통모드 바닥(G13). 바닥 이전 값은 `health["cov_landmark_only"]` |
| 원신호 | `health["pg"]`: 카탈로그(창·후보·시야 안), 검출(영역·짝·검출 수·태양 방향 천체력/영상 추정), 투표(봉우리·비·가설별 inlier), 자세(inlier, 비율, 매칭 비율, RMS, σ0, γ), 시간 |
| 랜드마크 | 검출 타원, 짝지은 카탈로그 id·D·d/D, 중심 편향, 잔차, `inlier` |
| EKF 연동 | 위치 갱신은 `T_world_cam[:3, 3]`, `cov[:3, :3]`. 랜드마크 갱신은 `inlier` 크레이터의 (u − bias, v − bias) ↔ `X`(테두리 평면 중심), σ = max(σ_min, σ_rel·D_px)·σ0 |
```
```

- [ ] **Step 2: `README.md` 갱신** — 상태 목록에 "P-G 추적형 1단계(V7) 구현·검증, 결과 `docs/validation/v7_pg_nadir.md`"를 추가하고, Experiments 절에 `build_catalog.py`, `run_pg.py`, `analyze_pg.py` 명령을, Layout에 `craters/`를 추가한다.

- [ ] **Step 3: 스펙 갱신** — `status: implemented`로 바꾸고, 스펙 §15(계획 단계 변경)에 이어 구현·보정 중에 바뀐 점(카탈로그 완화 적용 여부, 기본값 변경)을 덧붙인다.

- [ ] **Step 4: 핸드오프 작성** — `docs/handoff/<날짜>-pg-session.md`를 `2026-10-11-trn-baseline-session1.md`와 같은 구성으로 쓴다: 결과 요약 표, 데이터에 대해 새로 안 것, 알고리즘에서 정한 것(G 요약), 코드와 실행(명령·시간), 아직 안 한 것(LIS 2단계: Christian 불변량 — wdoppenberg MIT 수학을 출처 표기해 재작성, oblique, 학습 검출기 속성 실험, 테두리 부화소 정제), 주의, 다음 세션 시작 프롬프트.

- [ ] **Step 5: 최종 검증**

Run: `.venv/bin/python -m pytest -q`
Expected: 전부 통과(느린 시험 포함). `git status --short`로 새·바뀐 파일 목록을 확인해 사용자에게 보고한다. 커밋은 사용자가 요청할 때만 한다.
