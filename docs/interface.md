---
title: "localize() 인터페이스 — sim2moon 항법 플러그인 연동용 (초안)"
type: interface
status: draft
lang: ko
created: 2026-10-10
---

# `localize()` 인터페이스 (초안)

핸드오프 §8의 초안을 구현한 것이다. sim2moon·Isaac·UE 의존이 없고 numpy, scipy, OpenCV만 쓴다. P10에서 필드 이름을 고정한다.

## 호출

```python
from lunar_trn.localize import localize, LocalizerConfig, MapSet
from lunar_trn.maps import render_maps
from lunar_trn.dem import DemSource

# Onboard maps (once per landing area and Sun geometry)
src = DemSource("data/dem/ldem_875s_5m.img")
fine, coarse = render_maps(src, center_xy=(x_m, y_m), half_size_m=12_000.0,
                           pixels=[10.0, 20.0], sun_dir=sun_world)
maps = MapSet(coarse=coarse, fine=fine)

result = localize(
    image,              # HxW float32 in [0, 1]
    K,                  # 3x3; pixel (col, row) centre is at u = col + 0.5
    prior_T_world_cam,  # 4x4 camera-to-world, Moon-fixed MOON_ME frame (m)
    prior_cov,          # 6x6: position (m^2, world) then attitude (rad^2, world small-angle)
    maps,
    LocalizerConfig(),  # optional
)
```

## 입력 규약

| 입력 | 규약 |
|---|---|
| 좌표계 | MOON_ME (DE421) 달 고정 직교좌표, 원점 달 중심, 단위 m. 지도 격자는 남극 평사도법(R = 1,737.4 km) |
| 카메라 | 핀홀. 광축 +Z, +X 오른쪽, +Y 아래. 픽셀 (col, row)의 연속 좌표는 (col + 0.5, row + 0.5) |
| 영상 | 2D float, 밝기 척도는 자유(국소 정규화). 선형 복사량이면 더 좋다 |
| 사전 공분산 | 위치 블록으로 거친 탐색 반경(3σ + 300 m)을 정한다. 자세 블록은 정밀 단계의 MAP 사전으로 쓴다. 3×3이거나 자세 블록이 0이면 자세를 아는 것으로 본다 |
| 지도 | `MapSet(coarse, fine)`. 정밀 지도 픽셀 ≈ 카메라 GSD, 거친 지도는 2배. `TrnMap`은 영상·고도·그림자·태양 방향을 담는다 |

## 출력 `LocalizationResult`

| 필드 | 내용 |
|---|---|
| `status` | `"fix"` (정밀 inlier ≥ 20) 또는 `"no_fix"` |
| `T_world_cam` | 추정 자세. `no_fix`이면 마지막으로 도달한 단계의 값 |
| `cov` | 6×6 공분산(위치 m², 자세 rad²). 랜드마크 잡음 항(σ0 > 1이면 σ0²로 키움)에 공통모드 바닥(수평 축마다 0.47 정밀-지도 px, 수직 0.085 px)을 더한 값이다. 바닥 이전 값은 `health["cov_landmark_only"]` |
| `landmarks` | 랜드마크별 배열: 특징 픽셀, Harris 응답, 피크 높이·폭, 부화소 오프셋, PSR, 부피크 비, 영상 좌표 `u, v`, 3D 지도점 `X`, 기각 사유 `status`, `used`, `inlier`, 재투영 잔차 |
| `health` | 원신호 사전. 아래 표 |
| `stage` | `"coarse"` 또는 `"fine"` |

### `health`의 원신호 (핸드오프 §7 대응)

| 분류 | 키 |
|---|---|
| 영상·장면 | `scene.entropy_bits`, `scene.grad_cov_lambda_min`, `scene.grad_cov_lambda_max`, `scene.grad_cov_tr_inv`, `scene.dark_fraction_002/005`, `scene.mean_intensity`, `scene.gsd_ratio_cam_over_map` |
| 거친 단계 | `coarse.templates[*]`: `peak`, `peak_fit`, `width`, `width_min`, `psr`, `second_ratio`, `phasecorr_response`(OpenCV `phaseCorrelate`), `offset_px`, `reject`, `geom_inlier`, `geom_residual_px`. 집계: `n_templates`, `n_peak_ok`, `n_geom_inliers`, `n_geom_outliers`, `search_radius_px`, `correction_m` |
| 정밀 단계 | `fine.n_features`(시도), `fine.n_matched`(피크 시험 통과), `fine.n_used`(추정기 입력 ≤ 100), `fine.reject_counts`(사유별: `peak_shape`, `peak_low`, `peak_wide`, `peak_on_border`, `geometric`, `mahalanobis`, `outside_map`, `no_template`), `fine.peak_height_used_median`, `fine.mahalanobis_gamma`, `fine.ransac_threshold_px` |
| 자세 | `fine.n_inliers`, `fine.inlier_ratio`, `fine.reproj_rms_px`, `fine.sigma0`, `cov`, `correction_from_prior_m` |
| 시간 | `t_coarse_s`, `t_total_s` |

채점(정답 대비 오차, 판정)은 `lunar_trn.experiments.score()`가 따로 한다. 고도, 태양 방향, 궤적 유형은 실험 기록(`runs.jsonl`)에 함께 남긴다.

## EKF 연동 메모

- 위치 갱신만 쓰려면 `T_world_cam[:3, 3]`와 `cov[:3, :3]`을 측정으로 넣는다. 공통모드 바닥이 들어 있어 StereoLunar 평가에서 2σ 포함률이 99.8%다(약간 보수적).
- 랜드마크 단위 갱신을 쓰려면 `landmarks`에서 `inlier`인 것만 골라 (u, v) ↔ X를 측정으로 넣는다. R의 기본값은 σ = 0.5 지도 px(영상 px로 환산) × σ0다. 이때 랜드마크 오차가 서로 독립이라는 가정은 틀리다. 공통모드 오차(~0.5 px)를 상태(지도 오프셋 바이어스)나 R 부풀림으로 따로 다뤄야 한다.
- 지도 생성(`render_maps`)은 무겁다(수 초–수십 초). 비행 중에 매 영상마다 하지 말고, 착륙 지역과 태양 조건별로 미리 만든다.
