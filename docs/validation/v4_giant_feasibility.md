---
title: "V4 교차 구현(NASA GIANT SFN) 가능성 조사"
type: validation
status: feasibility
lang: ko
created: 2026-10-10
---

# V4 교차 구현 — NASA GIANT SFN 가능성 조사

## 결론

- **쓸 수 있다. 다만 별도 프로세스로 돌리고 일부를 고쳐 써야 한다.** 남은 연결 작업은 16–24시간으로 본다(P7, 다음 세션).
- 라이선스는 Apache-2.0이다(v2.0.0, 태그 `v2.0.0f`, 커밋 `2c2225a9`, 2025-09-03). 단 `pyproject.toml`에는 아직 `license = "NASA-1.3"`가 남아 있다. v1.x는 NOSA 1.3이었으므로 반드시 v2.0.0f 이후를 고정한다.
- **Python 3.11 이상이 필요하다.** 3.10에서는 문법 오류가 난다(`split_camera.py`의 `out[*leading_slices, ...]`). 이 레포(3.10)에 넣지 않고 별도 venv와 하위 프로세스로 쓴다. 설치 방법은 `tools/giant/README.md`에 있다.

## 확인한 사항

| 항목 | 결과 |
|---|---|
| 설치 | Python 3.12 + `numpy==2.2.6` 고정. 선언되지 않은 의존성 `opencv-python-headless`, `tqdm`을 추가하고, RoMa(torch) 강제 import는 빈 모듈 shim으로 우회 |
| 입력 대응 | 핀홀 K(fx = fy = 618.04, c = 256), 달 고정 자세, DEM(달 고정 정점 → `Triangle64` → `KDTree`), 태양 방향(1 AU 점광원) 모두 그대로 넣을 수 있다 |
| 출력 | 특징별 관측·예측 픽셀, 상관 점수, 상관면, 피크 위치. 우리 원신호 기록에 바로 쓸 수 있다 |
| 하강 적용성 | 기하는 일반적이다. 3 km 합성 하강 장면에서 동작했다 |
| 합성 시험 (태양 20°, 사전 오차 없음) | 특징 9개 중 8개가 0.04–0.6 px 이내 |
| 합성 시험 (태양 20°, 사전 37 m·0.19°) | 9개 중 7개 정합, 그중 5개가 0.1–0.3 px |
| 합성 시험 (태양 5°/2°) | 6개 중 4개 / 5개 중 1개 |

## 결함과 대응 (GIANT 2.0.0)

1. `SurfaceFeatureNavigation.compute_rays`가 실수 경계를 넘겨 `rows must be of integral type` 오류가 난다. 하위 클래스로 고친다(`tools/giant/sfn_smoke.py`).
2. `FeatureCatalog`에 `compute_intersect`가 없어 장면 추적이 실패한다. 메서드를 덧붙인다.
3. **PnP 갱신(`_lls`)이 틀렸다.** 위치 이동 열의 Jacobian 부호가 반대이고, 사전 행의 잔차가 항상 0이다. 완전한 입력에서도 49 m 오차가 났고, 부호를 고치면 3 m가 됐다. **V4에서는 GIANT의 특징 정합 결과(픽셀)만 쓰고 자세는 우리 추정기로 푼다.**

## V4 설계 시 주의

- GIANT 템플릿은 특징 자기 타일만으로 렌더한다. 타일 밖에서 드리우는 그림자가 없다. 기본 반사 모델은 McEwen(Lambert/Lommel-Seeliger 혼합)이다. 우리 지도(Hapke, 12 km 그림자 확장)와 렌더러가 다르다는 점을 결과 해석에 넣는다.
- 기본 `incident_angle_maximum = 70°`이면 극지 태양에서 모든 특징이 빠진다. 89.5°로 바꿔야 한다. 이 값과 `min_corr_score`(기본 0.3), 특징 크기 등 우리가 정하는 값은 `docs/decisions.md`에 남긴다.
- "우주 마스크"(Otsu + (0,0)에서 flood fill)는 천체가 검은 배경 앞에 있다고 가정한다. 하강 영상에서는 템플릿 가장자리에만 영향을 준다.
- SFN 단위 시험이 GIANT에 없다. 우리 쪽에서 합성 장면 시험을 먼저 붙인다.

## 다음 작업 (추정 16–24시간)

| 작업 | 시간 |
|---|---|
| LOLA 평사도법 타일 → 달 고정 특징 타일·KD 트리 | 4–6 h |
| 자세·좌표계 어댑터 | 2–3 h |
| 하위 프로세스 실행기(NPZ/JSON 입출력, 랜드마크 ID 공유) | 4–6 h |
| 우리 정합기와 비교(같은 입력, 조명 일치 조건), 저태양 시험 | 4–8 h |
