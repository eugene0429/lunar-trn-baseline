---
title: "달 지도 기반 TRN 기준선 — 1차 세션 핸드오프 (P0–P5, V1–V3)"
type: handoff
status: active
lang: ko
created: 2026-10-11
---

# 1차 세션 핸드오프 (P0–P5, V1–V3)

착수 핸드오프(`2026-10-10-trn-baseline-kickoff.md`)의 P0–P5를 구현하고 V1·V2·V3(일부)를 마쳤다. 다음 세션은 이 문서와 `docs/decisions.md`를 먼저 읽는다. 끝의 §7에 시작 프롬프트가 있다.

## 1. 결과 요약

| 시험 | 결과 | 문서 |
|---|---|---|
| V1 좌표계 | 합격(nadir 450, oblique 196 기하). 깊이 잔차 중앙값 0.00 / −0.02 m, 영상면 이동 95%ile 0.009 / 0.022 px. 원천 DEM은 PDS `LDEM_875S_5M`, 좌표계는 MOON_ME | `docs/validation/v1_frames.md` |
| V2a 거친 | 합격. 대역별 99%ile 39–84 m (기준 ≲ 200 m) | `docs/validation/v2_v3_nadir.md` |
| V2b 정밀 | 합격. 대역별 95%ile 0.81–0.84 지도 px | 같은 문서 |
| V2c 무결성 | 합격. 18,198회 오고정 0건(상한95 0.02%) | 같은 문서 |
| V3 조명 | 태양 각도 차 ≤ 28°는 영향 거의 없음, 42°에서 고고도 ≥ 0.98, 56°에서 0.2–0.4, ≥ 70°에서 ≈ 0. 데이터셋 조명 쌍(66–134°)은 0–2%. 스윕 7,140회 오고정 0건 | 같은 문서 §4–5 |
| V4 교차 구현 | 가능성만 확인(GIANT 2.0.0, Python 3.11+ 별도 프로세스, 결함 3개) | `docs/validation/v4_giant_feasibility.md` |

- 가용성(조명 일치): 3.5 km 76%, 6.5 km 95%, 9.5 km 97%, 12.5 km 99%, 15.5 km 이상 100%.
- 저고도 실패의 원인은 사전 오차에 비례하는 시차(남극 km급 기복)와 그림자다.

## 2. 데이터셋에 대해 새로 알게 된 것 (로더에 반영됨)

1. EXR 깊이는 **z-깊이**(half float)다. ReadMe의 "시선 방향 거리"가 아니다.
2. 픽셀 중심은 `u = col + 0.5`다.
3. 조명 순서는 인덱스로 정할 수 없다. 배치 시작이 6의 배수가 아닌 구간이 있다. `assign_triplets()`로 같은 자세끼리 묶는다.
4. npz에 metadata가 없다. 태양 방향은 생성 코드 식으로 복원한다. 세 조명은 (330°, 20°), (250°, 20°), (180°, 15°)다.
5. JPG는 영상마다 최소–최대 정규화된 8비트다.
6. 원천 DEM은 PDS `LDEM_875S_5M`(2017)이다. PGDA 87S 5 m 모자이크(2021)는 0.3 px급으로 어긋난다.
7. oblique 궤적의 천저 이탈은 27–59°(중앙값 40°)로, 논문의 20–35°보다 넓다.
8. 태양 방위·고도는 **시선 목표점**의 국지 ENU 기준이다. 카메라 바로 아래 점을 쓰면 oblique에서 방위가 ±70–110° 틀어진다(`Frame.boresight_point()`로 수정).

## 3. 알고리즘에서 정한 것 (자세한 값은 `docs/decisions.md`)

- 지도: Hapke 정사 렌더, 정밀 = 명목 GSD, 거친 = 1.5×GSD, 정확한 2.5D 지평선 그림자, 태양 쪽 12 km 확장.
- 거친: 128 px 템플릿 9개, NCC(FFT), 자세 고정 위치 RANSAC(기하 검정).
- 정밀: Harris 150개, 21×21 NCC, 128×128 탐색, 쌍이차 부화소, 피크 게이트(높이 ≥ 0.60, FWHM ≤ 12 px — 보정 표본으로 정함), ≤ 100개 → 위치 RANSAC → 6자유도 GN(자세는 MAP 사전) → Mahalanobis 게이트 → inlier ≥ 20.
- 공분산: 랜드마크 잡음 + 공통모드 바닥(0.47 px/축). 바닥이 없으면 3.6배 과신한다.

## 4. 코드와 실행

```
src/lunar_trn/  frames, dem(피라미드 광선추적), render(Hapke), maps(지도 생성), rectify,
                matching(NCC·피크), coarse, fine, pose, health, localize(인터페이스), experiments,
                datasets/stereolunar, validation/v1
scripts/        fetch_dem, fetch_stereolunar(HTTP range 부분 추출), run_v1, run_v2, analyze_calib,
                analyze_v2, make_figures
tools/giant/    GIANT SFN 시험 스크립트(별도 venv, 레포에 GIANT 코드 없음)
```

- 시험: `.venv/bin/python -m pytest` (40개, 합성 종단 시험 포함)
- 데이터(커밋 안 됨): `data/dem/ldem_875s_5m.{img,lbl}`, `data/dem/ldem_87s_5mpp.tif`, `data/stereolunar/{nadir,oblique}/`
- 결과(커밋 안 됨): `outputs/v1/`, `outputs/calib/`, `outputs/v2/`, `outputs/v3/`. 실행별 원신호는 `runs.jsonl`에 있다.
- 실행 시간(64코어): V1 nadir 25분, V1 oblique 약 20분(40 프로세스), 보정 30분, V2/V3 본 실행 72분, 방위 스윕 67분 + 42분.

## 5. 아직 안 한 것 / 다음 순서 제안

1. **V3 고도 오프셋 스윕**: 방위 스윕은 끝났다. 태양 고도 차(±5–10°)와 방위·고도 결합을 `--sun-delta-el`로 돌린다.
2. **oblique V2, dynamic V1·V2**: oblique는 V1만 했다(데이터는 `data/stereolunar/oblique`). dynamic은 아직 받지 않았다(`scripts/fetch_stereolunar.py --archive dynamic --block-stride 120` 후 `--indices-file`로 삼중쌍 보충).
3. **P7 / V4**: GIANT SFN 연결(16–24시간 추정). 정합 픽셀만 쓰고 자세는 우리 추정기로 푼다.
4. **P9 연속 영상 필터**: 저고도 가용성은 사전 오차가 작을수록 크게 오른다(< 1 km에서 93%). StereoLunar에는 연속 궤적이 없으므로 합성 시퀀스나 DDL-F1로 시험한다.
5. **거친 단계 강건화(선택)**: 저고도·큰 사전 오차에서의 시차 한계. 다중 가설 와핑이나 2회 거친(`coarse_passes=2`, 구현만 됨) 효과를 보정 표본으로 먼저 본다. LVS 충실 기준선과 구분해 옵션으로 둔다.
6. **그림자 확장 거리**를 지형 기복에 맞춰 적응형으로 한다(B4 한계).
7. **P8 DDL-F1**, **P10 인터페이스 고정**(`docs/interface.md` 초안).

## 6. 주의

- 지도 생성이 가장 무겁다(3.5 km 대역에서 지도 한 쌍에 수십 초–수 분). 실행기는 작업마다 지도를 새로 만든다.
- 보정 표본(기하 순번 % 4 == 1)으로 임계값을 정했다. 평가 표본으로 임계값을 바꾸지 않는다.
- 커밋은 사용자가 요청할 때만 한다. 이번 세션 결과물은 아직 커밋하지 않았다.

## 7. 다음 세션 시작 프롬프트

```text
~/lunar-trn-baseline 레포에서 작업한다. docs/handoff/2026-10-11-trn-baseline-session1.md와 docs/decisions.md를 먼저 읽어라.
이번 세션 목표:
(1) V3 태양 고도 오프셋 스윕을 돌리고 docs/validation/v2_v3_nadir.md를 갱신한다.
(2) StereoLunar oblique V2와 dynamic V1·V2를 돌려 궤적 유형별 결과를 보고한다.
(3) NASA GIANT SFN(tools/giant, docs/validation/v4_giant_feasibility.md)을 별도 프로세스로 연결해 V4를 한다.
임계값은 보정 표본에서만 정하고 docs/decisions.md에 남긴다. 커밋은 내가 요청할 때만 한다.
```
