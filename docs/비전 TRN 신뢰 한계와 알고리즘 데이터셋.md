# 지도 해상도가 비전 TRN의 고도 하한을 정한다

달·행성 하강에서 지도 기반 비전 TRN이 실제로 끝난 고도는 조명이나 플룸보다 먼저 **탑재 지도의 해상도와 내용**이 정했다.

- **지도 기반 위치추정**: 실측된 종료점은 세 곳이다.
  - New Shepard 비행 영상 후처리: 20 m/px 지도는 4,000 m, 3 m/px 지도는 750 m에서 마지막 갱신을 냈다. 수치의 출처는 Blue Origin 최종보고서다.
  - Masten Xombie 로켓 시험: LVS가 1 m/px 지도로 ~85 m까지 매칭하다가 "the scale difference precluded successful feature matching"으로 끝났다.
  - Mars 2020: LVS는 6 m/px 지도로 500 m까지 위치를 추정했다.
- **크레이터 TRN**: 지도의 크레이터 내용이 바닥나는 지점에서 멈췄다. Chang'e-3 영상을 쓴 DLR CNav는 ~350 m, 남극 렌더에서는 ~1 km(전역 카탈로그만 쓰면 ~4 km)였다.
- **"종료 고도 ÷ 지도 GSD"는 상수가 아니다.**
  - 시야각 12.2°의 좁은 카메라(New Shepard)는 200–250, LVS형 시스템은 ~85 이하였다. 카메라 시야각과 매처 설정에 따라 달라진다.
  - Draper의 해석에서는 지도 0.5–2 m/px 사이에서 하한 차이가 작았다. 4 m/px에서 하한이 "nearly doubles"했고, 8 m/px에서는 매칭에 실패했다. 즉 비례가 아니라 **꺾임**이 있다.
- **하한 위에서는 조명 불일치가 신뢰 범위를 정한다.**
  - LVS는 운용 범위를 "LCAM to Map Delta Sun Angle" 0–20°로 묶었다.
  - LION은 고도각 차 10°, 방위각 차 20°를 넘으면 성능이 크게 나빠졌다.
  - 낮은 태양에서 방위각 차가 30–60°를 넘으면 SIFT·NCC급 매칭이 깨진다.
  - 같은 크레이터 매칭법의 성공률이 중위도 100%에서 남극 72%로 떨어졌다.
- **지형 질감과 저고도 VO**
  - 지형 질감은 같은 알고리즘의 유효해 비율을 99%에서 71%로 깎았다.
  - 저고도 VO는 질감 소실과 프레임 간 운동으로 끝났다(Ingenuity 72번 비행 12 m, New Shepard 하강 마지막 ~20초).
  - 플룸은 수십 m 아래의 부차 요인이다.
- **이론과 비행의 차이**
  - Setterfield의 Cramér–Rao 한계는 **완만한** 저하를 예측한다. 그러나 실제 오차는 추정기 편향이 지배했다.
  - 비행에서는 갱신이 한순간에 끊기는 **절벽**이 관측됐다. 절벽은 오정합 영역에서 생긴다.
- **신뢰 게이트의 작동점은 화성 LVS의 것 하나뿐이다.** "inlier 20개 이상" 게이트가 오탐률 2×10⁻⁸, 검출률 83%를 냈다(Johnson et al. 2025). **고도별 정답 고정 확률과 오고정률 곡선은 달·화성 어디에도 없다**(2026-10). 이것이 sim2moon 연구가 채울 수 있는 핵심 공백이다.
- **de Gerlache에 대한 예측**: sim2moon의 60° 카메라와 5–7 m/px 지도라면 절대 갱신은 실측 사례의 범위로 보아 **~0.2–1 km**에서 끝날 것이다. DEM의 실효 해상도(20–40 m)를 쓰면 수 km까지 올라갈 수 있다.
- **절대 TRN의 분류**: 정합 전에 기본 요소(크레이터, 키포인트)를 양쪽에서 추출·기술하는지에 따라 **템플릿 매칭**과 **패턴 매칭** 두 갈래로 나뉜다. 관측 신호(밝기 대 고도)는 그 안의 하위 축이다. 딥러닝, 기준 영상의 렌더 여부, Image-on-Map/Map-on-Image는 계열이 아니라 속성이다. 근거는 Rahuman & Lee 2026(IJASS) 리뷰이고, 리뷰의 느슨한 분류 적용은 고쳐 썼다(질문 5).
- **기준선과 자료**: 비행 TRN 코드는 공개되지 않았지만 기준선과 자료는 충분하다.
  - 알고리즘:
    - 밝기 템플릿(T-I): 공개 논문 수치로 다시 짠 LVS형 2단 MRL, NASA GIANT 표면특징항법(Apache-2.0)
    - 크레이터 패턴(P-G): Christian 불변량 크레이터 식별
    - 밝기 특징 패턴(P-R): Kornia의 LightGlue/LoFTR
    - 탐색적 비교군인 고도 템플릿(T-E): 영상 복원 DEM + ICP
    - 상대 항법 대조군: OpenVINS
  - 지도·지형: PGDA 78 오차지도와 clone DEM, PGDA 104 SfS DEM, LROC NAC 태양경도별 모자이크
  - 비행 데이터: New Shepard NS-13 비행 데이터(DDL-F1: IMU, TRN 카메라, LiDAR)가 NASA Open Data Portal에 공개되어 있다.

*표기: [추론]은 이 보고서의 해석, [산술]은 인용한 수치로 직접 계산한 값, [언론]은 기술 문헌이 아닌 회사·언론 발표, [요약 확인]은 원문이 아니라 초록이나 검색 요약으로만 확인한 내용이다. 각 절 제목 끝의 (질문 n)은 요청의 (1)–(5)에 대응한다.*

> **2026-10-06 갱신 — KAIST 원문 추출 반영.** 부록의 1·2순위 자료 대부분과 빈칸 검색(Web of Science, IEEE Xplore, Scopus, RISS)의 원문 추출 결과([`../source_extraction_2026-10-06.md`](../source_extraction_2026-10-06.md))로 본문을 고쳤다. 원문 확인 표시는 **[원문]**이다.
>
> **결론이 바뀐 곳**
> 1. "종료 고도 ≈ 200–250 × 지도 GSD"는 카메라 특이적인 규칙으로 강등했다.
>    - VISINAV의 230 m는 실험자가 정한 설정값이다.
>    - New Shepard 카메라는 시야각 12.2°였다.
>    - Xombie 로켓 시험에서는 1 m/px 지도로 ~85 m까지 실측 매칭했다.
> 2. de Gerlache 예측을 1.0–1.75 km에서 ~0.2–1 km로 다시 계산했다.
> 3. 신뢰 게이트의 실측 작동점(LVS inlier 20 → 오탐률 2×10⁻⁸, 검출률 83%)과 조명 지표 임계(0.3)를 넣었다(Johnson et al. 2025).
> 4. 남극 크레이터 항법 하한을 ~1 km로 확정했다(Woicke & Krüger 2023).
> 5. 고도별 성능을 부분적으로 비교한 Smith et al. 2020, 공개 비행 데이터 DDL-F1, 국내 선행연구를 추가했다.

> **2026-10-10 갱신 — 절대 TRN 분류 기준 확정.**
> - **근거 자료**: 이전 판 부록 #31인 Rahuman & Lee 2026 리뷰 원문(사용자 제공 PDF)과 공개 원문 세 편(Johnson & Montgomery 2008, Johnson & Ivanov 2011, Kim et al. 2025)
> - **반영한 곳**
>   - 질문 5의 기준선을 계열 코드(T-I/T-E/P-G/P-R)로 다시 묶었다.
>   - 상대 항법(VO/VIO)은 절대 TRN 비교군이 아닌 대조군으로 분리했다.
> - **새로 추가한 내용**
>   - 공개된 가중치 라이선스: LightGlue는 Apache-2.0이고 SuperPoint 가중치는 제한 라이선스다.
>   - 국내 선행연구 두 편
> - **결론이 바뀐 곳**: 없다.
>   - 리뷰는 고도별 신뢰도를 정량화하지 않는다.
>   - 리뷰는 저고도의 "ground-sampling differences"를 미해결 과제로 꼽는다. 이는 이 보고서의 공백 판정을 뒷받침한다.
> - **상세 근거**: [`../research_notes/비전 TRN 신뢰 한계와 알고리즘 데이터셋/absolute_trn_taxonomy.md`](../research_notes/비전%20TRN%20신뢰%20한계와%20알고리즘%20데이터셋/absolute_trn_taxonomy.md)
> - **출처란 신설과 부록 정리**
>   - 원문을 확인해 본문에 쓴 자료 36건을 부록에서 빼고, 본문 인용 전체와 함께 [출처](#출처)로 모았다.
>   - 부록에는 미확인 자료 40건과 빈칸 검색 후보만 남겼다.
>   - 서지는 Crossref·DataCite·arXiv·NTRS API로 맞췄다.
> - **서지 정정 두 건**
>   - "Fritz et al. 2020"(AIAA 2020-0370)의 제1저자는 Smith, K. W.다. Crossref·Semantic Scholar·OpenAlex 기준이다. 본문 표기를 Smith et al. 2020으로 바꿨다.
>   - SCALPSS PDS 묶음의 DOI를 10.17189/1522847(Mars 2020 ECAM 묶음이었다)에서 10.17189/xbv8-7s71로 바로잡았다.

---

## 비행 기록에서 절대 TRN은 지도 GSD의 200–250배 고도에서 멈췄다 (질문 1)

### 범위와 판정 기준

비행했거나 비행 시험을 거친 비전 TRN은 예외 없이 명시적 운용 범위 안에서만 신뢰를 주장했다. 가장 자세히 공개된 것은 Mars 2020 Lander Vision System(LVS)의 범위다 ([Johnson, LSIC 2021](https://lsic.jhuapl.edu/uploadedDocs/presentations/764-1.09_Johnson_20210302_LuNaMaps_LVS_v0.pdf)).

| 항목 | LVS 운용 범위 (원문) |
|---|---|
| 태양 | "sun elevation between 25˚ & 55˚", "sun azimuth between 240˚ & 310˚" |
| 자세 | 천저 이탈각 45° 미만, 각속도 50°/s 미만 |
| 지형 | "slope < 15˚", "terrain relief < 150m", "appearance entropy > 4" |
| 고도 | 4,200 m에서 시작해 "2000m or back-shell separation"까지 |

비행 재구성 궤적에서 LVS는 3,699 m AGL에서 유효해졌고 backshell 분리는 2,161 m AGL이었다 ([Way et al., NTRS 20210024480](https://ntrs.nasa.gov/api/citations/20210024480/downloads/Mars2020_Flight_Mechanics_Simulation_Paper_v3.pdf)). 실제로 비행한 LVS 구간은 비전 실패가 아니라 EDL 사건으로 끝났다는 뜻이다. 요구 정확도 60 m에 대해 목표에서 5 m 떨어진 곳에 내렸다 ([NTRS 20230007014](https://ntrs.nasa.gov/citations/20230007014)).

유효성은 매칭·inlier 수, 매칭 비율, 상관 피크 품질, 필터 혁신 게이트로 판정한다. 그러나 **임계값이 공개된 사례는 드물다**. LVS는 coarse 단계의 landmark 총수를 "a set of parameterized criteria"로 판정한다고만 쓴다. 공개된 것은 비행 실적뿐이다. 첫 fine 영상에서 150개를 매칭해 inlier 92개가 남았고, fine 영상 42장이 모두 inlier 80개 이상이었다 ([Johnson et al. 2023](https://doi.org/10.2514/1.G006780)). 수치가 공개된 예외는 세 가지 정도다.

- MER DIMES: 상관값·피크 폭 검사 ([Johnson et al. 2007](https://robotics.jpl.nasa.gov/media/documents/DIMES_IJCV_20070117.pdf))
- Draper IBAL: inlier가 "less than 8"이거나 영상 사이 고도 점프가 450 m를 넘으면 추정을 버린다 ([Maggio et al. 2023](https://arxiv.org/abs/2302.08011))
- DLR CNav: χ² 임계값 30 ([Trigo et al. 2018](https://elib.dlr.de/118510/1/hybrid_optical_navigation_by_crater_detection_for_pinpoint_landing_%28elib%29.pdf))

### 끝나거나 열화된 지점

아래 표는 실제 자료에서 비전 TRN이 끝나거나 열화된 지점을 높은 고도부터 정리한 것이다.

| 시스템 (자료 유형) | 끝나거나 열화된 고도 | 보고된 지표 | 명시된 원인 | 출처 |
|---|---|---|---|---|
| Draper IBAL, 풍선 (비행 후처리) | 29–23 km와 그 아래 | 자이로 없이 처리한 프레임 비율이 63–96%에서 0–3%로 하락 | 20°/s를 넘는 회전, 매단 줄의 가림. 질감 문제는 아님 | [Maggio 2023](https://arxiv.org/abs/2302.08011) |
| Draper IBAL, 풍선 | 9 → 4.5 km 쪽 | outlier:inlier 비 증가 | "shadows, lack of distinct texture on the ground, and … sparse amount of landmarks in our database" | [Maggio 2023](https://arxiv.org/abs/2302.08011) |
| Draper IBAL, New Shepard NS-23 | 8.5 → 3.5 km 구간 후반 | 매칭이 "more difficult" | 산 그림자가 DB 촬영 시각과 불일치. 저고도 3D 지형에서 homography 가정 붕괴 | [Maggio 2023](https://arxiv.org/abs/2302.08011) |
| JPL MRL, New Shepard 2회 (후처리). TRN 카메라 12.2°×12.2°, 1024², 천저 이탈 15° [원문] | **4,000 m** (20 m/px 지도) | 마지막 영상 갱신. AIAA 2022-1830 Fig.5에서 inlier가 산악 구간 60–80, 끝에서 0으로 급감 [원문] | "image/map scale mismatch". 2022-1830의 표현은 "The drop off in matches at the end of descent is due to the increasingly large mismatch in resolution between image and map" [원문] | [Blue Origin 최종보고서](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf) |
| Mars 2020 LVS (탑재) | ~2.2 km AGL에서 위치 인계, **추정은 500 m까지 계속** (6 m/px 지도) | 첫 fine 영상 150 매치 중 inlier 92, fine 영상 42장 모두 inlier ≥80 [원문] | 인계는 backshell 분리 일정. 비전 실패가 아님 | [Way et al.](https://ntrs.nasa.gov/api/citations/20210024480/downloads/Mars2020_Flight_Mechanics_Simulation_Paper_v3.pdf) |
| JPL MRL, New Shepard (후처리) | **750 m** (3 m/px 지도) | 마지막 갱신 | 축척 불일치 (보고서가 암시) | [Blue Origin 최종보고서](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf) |
| JAXA SLIM VBN (탑재) | **500 m** | 마지막 측정, 수평 오차 1 m 미만 | 설계된 일정. 실패가 아님 | [Ishida et al. 2025](https://doi.org/10.1016/j.actaastro.2024.11.002) |
| DLR CNav, Chang'e-3 영상 (후처리) | **~350 m** | 추적 상실 | "lack of map data content … the crater detector used is beginning to be confused by surface clutter" | [Maass et al. 2020](https://doi.org/10.2514/1.G004850) |
| **JPL LVS, Masten Xombie VTVL (탑재, 2014)** | **~85 m** (1 m/px 지도, 300 m에서 시작) | 영상당 최대 49 특징, 인계 시 수평 오차 <2 m/축 [원문] | "image matches generated for almost 20 s after the end of nominal LVS operations until at approximately 85 m altitude the scale difference precluded successful feature matching" | [Trawny et al. 2015](https://doi.org/10.2514/6.2015-4418) |
| DLR CNav, 남극 렌더 (시뮬레이션) | **~1 km** (전역 카탈로그만 쓰면 ~4 km) | 매칭 크레이터 25 → 10–14 (사례 영상) | 극지 DEM 렌더 수치 문제. "currently limited only by DEM related rendering capabilities" [원문] | [Woicke & Krüger 2023](https://doi.org/10.2514/6.2023-0690) |
| VISINAV 관측로켓 (후처리) | **230 m** (1 m/px 지도). 실험자가 정한(emulated) 중단이며 **실측 한계가 아님** [원문] | 직전까지 영상당 최대 80개 ML | "the number of features within the FOV becomes too small, and the difference in resolution between camera and map too significant" | [Mourikis et al. 2007](https://www.roboticsproceedings.org/rss03/p19.pdf) |
| JPL VO, New Shepard (후처리) | 하강 마지막 ~20초 (고도 미공개) | VO 갱신 실패 | "high frame-to-frame motion" | [Blue Origin 최종보고서](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf) |
| Draper TRN, New Shepard (탑재) | 착륙 직전 | 잔차 증가. TRN 처리가 멈춰 마지막 유효값을 유지 | "highly dynamic phase of flight coupled with foreign object debris on the lens" | [Fritz et al. 2022](https://ntrs.nasa.gov/api/citations/20210023718/downloads/AIAA_SciTech2022_TP2_NAVDQG.pdf) |
| IM-2 크레이터 추적 (탑재, [언론]) | 최종 접근 (고도 미공개) | 미공개 | LRO 100 km 영상 기반 지도가 "could not accurately account for how craters appear at lower altitudes with south pole lighting conditions" | [Space.com 2025-05-14](https://www.space.com/astronomy/moon/heres-why-the-private-athena-lunar-lander-toppled-over-on-the-moon) |
| DLR ATON 헬기 (지구 시험) | 저고도 호버 | 필터 갱신이 멈추고 오차가 "slightly" 커짐 | 크레이터 표적이 두 카메라 영상 모두에서 사라짐 | [Theil et al. 2017](https://elib.dlr.de/114457/1/ATON_DLRK_2017.pdf) |
| Ingenuity 72번 비행 (탑재) | 12 m로 상승, 이륙 ~20초 후 | 미공개 (보도 자료) | "Lack of surface texture gave the navigation system too little information" | [JPL 2024-12-11](https://www.jpl.nasa.gov/news/nasa-performs-first-aircraft-accident-investigation-on-another-world/) |
| Chang'e-6 하강 카메라 (후처리) | 8.6 m 아래 마지막 10장 | 정합에 쓸 수 없음 | 엔진 플룸 | [Shu et al. 2026](https://doi.org/10.1038/s43247-025-03074-7) |

### 종료 원인 네 가지

종료 원인은 네 갈래로 모인다.

1. **지도 축척과 내용.** 지도 기반 위치추정은 카메라 GSD가 지도 GSD보다 지나치게 촘촘해지거나 지도에 쓸 내용이 바닥나는 지점에서 끝났다.
2. **특징 결핍.** VO와 영상 매칭이 끝났다.
3. **외관 불일치.** 조명과 지도 출처가 다르면 매칭이 깎였다.
4. **프레임 간 운동.** 가장 낮은 고도의 VO가 이것으로 끝났다.

플룸이 영상을 끝낸 기록은 Chang'e-6의 마지막 8.6 m뿐이다.

### 하한은 지도 GSD로 정해지지만 비례 상수는 카메라마다 다르다

[산술] 실측된 종료 고도를 지도 GSD로 나누면 다음과 같다.

| 사례 | 카메라 | 종료 고도 ÷ 지도 GSD | 종료 시 지도 GSD ÷ 카메라 GSD | 영상 폭에 담긴 지도 픽셀 수 |
|---|---|---|---|---|
| New Shepard, 20 m/px | 12.2°, 1024 px [원문] | 4,000 ÷ 20 = 200 | ≈24 | ≈43 |
| New Shepard, 3 m/px | 같음 | 750 ÷ 3 = 250 | ≈19 | ≈53 |
| Xombie, 1 m/px (실측 종료) | 미확인 (LVS형) | 85 ÷ 1 = 85 | 90°·1024 px이면 ≈6 | 90°이면 ≈170 |
| Mars 2020, 6 m/px (종료 아님, 500 m에서 정상 추정 중) | 90°, 1024 px | ≤ 500 ÷ 6 ≈ 83 | ≥6.1 | ≤167 |
| VISINAV, 1 m/px | — | 230 (**설정값이라 제외**) | — | — |

**읽는 법**
- "종료 고도 ≈ 200–250 × 지도 GSD"는 시야각 12.2° 카메라에서만 맞는다. 넓은 시야각의 LVS형에서는 같은 지도로 2–3배 더 낮게 내려갔다.
- 같은 화소 수라면 두 해석은 같은 말이다. 하나는 "카메라 GSD가 지도 GSD보다 r배 촘촘해지면 끝난다"이고, 다른 하나는 "영상 폭에 담기는 지도 픽셀이 N_px/r개로 줄면 끝난다"다.
  - 관측된 r은 ~6(LVS형, 카메라 사양 미확인)에서 ~20–24(New Shepard)까지 퍼져 있다.
  - New Shepard는 영상 폭이 지도 픽셀 ~40–55개가 될 때 끝났다. 원인은 템플릿과 탐색창을 채울 지도 내용이 부족해진 것으로 보인다 [추론]. LVS fine 템플릿은 21×21, 탐색 영역은 128×128 지도 픽셀이다 [원문].
- Mourikis et al.(2009)은 일반론으로 "The cutoff point is mission-dependent; for the case of Mars descent, this occurs during the powered descent phase, nominally between 1000 m and 500 m above ground level"이라고 썼다 [원문].
- **지도 해상도에 대한 꺾임 (Draper 해석)**: "There is little difference in the lower altitude limit from 0.5 – 2 m/pix resolution satellite maps but as the resolution degrades to 4 m/pix, the minimum altitude at which the IPA is able to match features … nearly doubles", "The simulation runs using the 8 m/pixel satellite image failed to find any matches" ([Smith et al. 2020](https://doi.org/10.2514/6.2020-0370) [원문]). 그림에 축 눈금이 없어 고도 수치는 인용할 수 없다.
  - [추론] 하한은 "지도가 정하는 부분"과 "카메라·알고리즘이 정하는 부분" 가운데 큰 쪽이다. 지도가 충분히 촘촘하면 하한이 더 내려가지 않는다.

[추론] 따라서 하한은 비행 중 신뢰 지표가 아니라 **임무 지도 준비와 카메라 설계**가 함께 정한다. SLIM은 저고도 영역용 크레이터 지도를 따로 만들어 500 m까지 내려왔다. 원문은 "LRO topographic data was used for low-altitude areas (PPD3, CST2, VLD1, and VLD2) because higher topographic resolution data is required at lower altitudes"라고 쓴다 ([Ishida et al. 2025](https://doi.org/10.1016/j.actaastro.2024.11.002)). CNav의 350 m도 원문이 "the maps' resolution is no longer sufficient below that point to automatically extract craters"라고 쓰는 지도 해상도 한계다 [원문].

### 오차는 줄다가 끊긴다

정확도와 고도의 관계를 보면 저하가 완만하지 않고 **절벽**처럼 끊긴다.

- **SLIM**: Table 3의 14회 측정(15 km → 500 m)에서 픽셀 단위 오차는 0.16–2.61 px로 거의 일정했다 [산술]. VBN 출력의 픽셀당 미터는 PPD1의 ~38 m/px에서 VLD2의 ~0.5 m/px로 줄었다 [산술]. 미터 단위 오차는 영역별 지도 축척을 따라 줄었을 뿐 저고도에서 커지지 않았다. 요구조건(<3 px)에 가장 가까웠던 것은 4 km의 VLD1 #2로, ΔX −2.37 px, 고도 오차 −2.41%(−101.2 m)였다 ([Ishida et al. 2025](https://doi.org/10.1016/j.actaastro.2024.11.002)).
- **VISINAV**: 첫 랜드마크 세트를 "within about 5 seconds" 안에 2,722 m 오차에서 16.9 m로 수렴시켰다. 230 m에서는 5.1 m였고, 접지 오차는 6.4 m·0.16 m/s였다. IMU만 쓰면 9,169.5 m·32.70 m/s였다 ([Mourikis et al. 2007](https://www.roboticsproceedings.org/rss03/p19.pdf)).
- **New Shepard**: 마지막에 쓴 지도가 접지 오차를 정했다. 20 m/px 지도는 60 m, 3 m/px 지도는 5 m와 15 m 미만이었다. 지도 갱신이 끝난 뒤 VO는 속도를 "between +/- 12 m/s per axis"로 묶었지만 "the absolute position error is unobservable"이었다 ([Blue Origin 최종보고서](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf)).

[추론] 지도 기반 정확도는 종료 직전까지 지도 픽셀 0.2–3개 수준을 유지한다. 종료 뒤에는 갱신이 아예 나오지 않는다.

### 시스템은 "답하지 않을 권리"로 신뢰를 지켰다

MER DIMES의 설계 요구는 두 가지였다. 틀린 속도를 낼 확률은 0.1% 미만이어야 했고, 그 대가로 "DIMES was allowed to not report a velocity 10% of the time"이었다 ([Johnson et al. 2007](https://robotics.jpl.nasa.gov/media/documents/DIMES_IJCV_20070117.pdf)). 비행 중에도 실제로 특징을 거부했다.

- MER-A: 피크 폭 39 px인 특징을 버렸다. 받아들인 특징은 2.4–4.3 px였다.
- MER-B: 상관값 0.68인 특징을 버렸다. 받아들인 특징은 0.76–0.82였다. MER-B 착륙지는 "a very bland terrain spotted with a few craters"였고, 대비가 16–31 dn/pix로 MER-A의 42–56에 크게 못 미쳤다.

MOC 영상으로 만든 몬테카를로에서는 착륙 타원 가중 유효율이 Gusev 99%, Meridiani 71%였다. Meridiani의 hematite bright smooth plains는 50%(22/44)까지 떨어졌다. 저자들은 변수를 "tuned to Gusev"했으며 Meridiani에 맞췄다면 90%를 넘었을 것이라고 썼다. 오차(평균 + 3σ)는 3.73 m/s와 3.74 m/s로 같았다. **밋밋한 지형은 정확도가 아니라 가용성을 깎았다.** 게이트가 보수적이어서 틀린 추정을 내보내지 않았기 때문이다.

Ingenuity는 반대 사례다. 대신 속도를 잴 수단이 없는 VO가 질감 결핍을 만나자 틀린 속도로 추락했다. IM-3가 "an additional lighting-independent sensor for surface velocity measurements"를 더한 것도 같은 교훈이다 ([Spaceflight Now 2025-05-14](https://spaceflightnow.com/2025/05/14/intuitive-machines-ceo-points-to-issues-that-prevented-upright-touchdown-during-im-2-moon-landing/)).

[추론] 신뢰 분석은 **가용성**(고정 없음)과 **무결성**(틀린 고정)을 따로 재야 한다.

### 남극: 비행 증거는 하나뿐이고 그마저 열화됐다

남극 비전 TRN의 비행 증거는 IM-2(Mons Mouton, 2025) 하나다. CEO는 세 원인을 들었다 [언론] ([Space.com 2025-05-14](https://www.space.com/astronomy/moon/heres-why-the-private-athena-lunar-lander-toppled-over-on-the-moon)).

1. 고도계: "signal noise and distortion"
2. 조명: "low-angle sunlight created long shadows and dim lighting conditions"
3. 크레이터 인식: LRO 100 km 영상으로 만든 지도와 저고도 모습의 불일치

반면 앞서 Tim Crain은 "I'm really proud of how well our crater tracking system did in this very unusual lighting condition"이라고 말했다 ([Space.com 2025-03-07](https://www.space.com/the-universe/moon/private-intuitive-machines-moon-lander-fell-over-inside-crater-at-lunar-south-pole-photo-reveals)). [추론] 두 발언을 합치면 크레이터 추적은 하강 대부분에서 동작했고 저고도에서 나빠졌다고 읽힌다. 다만 그 고도는 공개되지 않았다.

지도 범위도 운용을 묶었다. 궤도를 돌수록 지면 궤적이 바뀌어 착륙 지점이 "outside of the lander's onboard optical navigation maps"로 나가므로 IM-2는 세 궤도 이상 기다릴 수 없었다. IM-3는 12궤도와 확장된 크레이터 지도로 대응한다 ([Aerospace America 2025-07-22](https://aerospaceamerica.aiaa.org/how-intuitive-machines-is-planning-to-make-its-third-moon-landing-a-successful-one/)).

NASA LuNaMaps는 그 배경을 정리했다 ([Restrepo et al. 2022](https://ntrs.nasa.gov/api/citations/20210024816/downloads/LuNaMaps_Scitech2022_Nov19.pdf); [LuNaMaps Technology Transfer Plan](https://ntrs.nasa.gov/api/citations/20240012328/downloads/LuNaMaps_TechTransfer.pdf)).

- 극지 궤도 영상은 그림자가 지배한다.
- "Appearance changes from small changes in sun angle result in much greater sensitivity in image matching algorithms. Thus, TRN algorithms become very sensitive to the timing at high latitude regions."
- 지도는 "constructed to match as closely as possible the lighting conditions present during landing"이어야 하고, 영역 밖에서 드리우는 그림자 때문에 더 넓은 맥락이 필요하다.
- "5 m/pix"로 배포되는 LOLA DEM은 NAC 영상과 비교하면 실제 해상도가 20–40 m/pix다.
- 남극 근처 5 m LDEM의 채움률은 ~10%뿐이다.

**남극 크레이터 항법 (Woicke & Krüger 2023)** [원문]
- 남극 접근을 렌더해 "CNav can be operated successfully during a south pole landing approach until approximately 1 km above the landing site"를 보였다. 하한의 원인은 극지 DEM 렌더의 수치 문제다.
- 전역 카탈로그만 쓰면 ~4 km에서 마지막 매칭이 났다.
- 운용 조건은 lit 지형 위 궤적, "local Sun elevation shall be lower than 40°", 크레이터 존재다.
- Shackleton 테두리의 국소 태양 고도는 연중 약 ±2° 안이다(2024-12-16 최대 1.55°).
- 그림자 구간에서는 "no CNav solutions are possible for more than 30 s"였다. 고정 판정은 영상 크레이터 가운데 최소 5개가 카탈로그에 있을 때다.

**극지 조명 지상 헬기 시험 (Wang et al. 2026, CJA)** [원문]
- 둔황에서 태양 고도 2–17°, 2 km → 100 m 조건으로 시험했다.
- 고도대별 랜드마크 DB를 6/2/1 m/px로 따로 만들었다. 시야당 3–5개를 매칭했다.
- 수평 오차는 평균 28.8 m(<50 m), 저고도 구간에서 평균 5.7 m였다.
- NCC 성공 임계는 0.35로, 원문은 이를 "the matching threshold used by the Mars 2020 lander"라고 쓴다.
- 실패 사례는 "relatively flat, with a large proportion of weak texture and completely shadowed regions, resulting in a grayscale image computed as all zeros during real-time rendering"이었다 ([Wang et al. 2026](https://doi.org/10.1016/j.cja.2026.104149)).

긍정적인 남극 자료는 이것들과 하드웨어 루프 시뮬레이션뿐이다. Blue Origin 시연은 고도 2,000 m에서 태양 고도 5°, 2 m/pixel 지도 조건으로 돌렸고 "LVS/TRN and the BlueNav-L worked at the lunar south pole with lunar imagery and at low sun elevations"라고 보고했다 ([Blue Origin 최종보고서 p.21](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf)). [추론] LVS가 실제로 비행한 조명 범위는 25–55°였다. 그러니 태양 고도 수 도의 남극에서 절대 TRN을 믿을 비행 선례는 열화된 IM-2밖에 없다.

Firefly Blue Ghost M1은 Space-ng 소프트웨어로 "comparing live imagery from two onboard navigation cameras against simulated lunar terrain rendered in real time"하며 내려가 100 m 목표 안에 착륙했다 ([TechTimes 2026-06-27](https://www.techtimes.com/articles/319197/20260627/firefly-aerospace-brings-moon-landing-ai-navigation-house-after-blue-ghost-success.htm); [CNN 2025-03-01](https://www.cnn.com/2025/03/01/science/blue-ghost-mission-moon-landing/index.html)). 운용 고도, 매칭 통계, 고도별 오차는 공개되지 않았다.

### 플룸은 부차적이다

플룸은 이 그림에서 마지막 수십 m의 요인이다. Mars Sample Retrieval Lander용 ELViS 설계는 50 m 아래에서 영상을 쓰지 않는다. 근거는 실측이 아니라 "the **expected** altitude at which dust motion will be visible"이다 ([Setterfield et al. 2024](https://doi.org/10.2514/6.2024-0315)). New Shepard 2차 비행의 헬륨 퍼지 카메라는 "A clear field of view … from liftoff through descent"를 유지했다 ([Blue Origin 최종보고서](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf)).

---

## 이론은 완만한 저하를, 비행은 절벽을 보여 준다 — 절벽은 오정합 영역에 있다 (질문 2)

### 성능 한계: Fisher 정보는 "보이는 픽셀 수 × 기울기 공분산 ÷ 잡음"이다

하강 고도와 GSD 의존성을 함께 풀어 낸 한계는 Setterfield, Hewitt & Trawny의 "Fundamental Limits of Map Relative Localization During Spacecraft Descent and Landing"(JGCD 44(7), 2021) 하나뿐이다 ([Setterfield et al. 2021](https://doi.org/10.2514/1.G005634)).

- **유도 방식**: 지형을 성가신(nuisance) 매개변수로 두고 Schur 보수 L_δ = I_δ − I_cᵀI_z⁻¹I_c로 변위 정보를 구한다.
- **기대값**: E[L_δ] ≈ K·(M_l N_l/σ_η²)·(1 − σ_ηo²σ_ηl²/(σ_η²(σ_ηo²+σ_ηl²)))·Σ∇ (식 62). 여기서 "Σ∇ is the covariance of pixel image gradients"이다. 이 근사는 두 GSD가 같을 때만 K = 1로 정확하다.
- **고도의 역할**: 고도는 착륙선 GSD Δd_l로만 들어온다. "Optimal results are achieved when orbital and lander-fixed pixels are the same size"이다.
- **지도 GSD의 포화**: 지도 GSD를 키우는 효과는 "plateaus when Δd ≥ 11 m/px"이다.
- **불가능 조건**: 지도나 영상 한쪽의 잡음이 무한대면 "localization is not possible"이다.
- **원문 추가 확인** [원문]
  - 식 (62)의 근사는 "given only to facilitate intuition and should not be used in implementation"이다.
  - Σ∇는 "taken over a copy of Zm that was down-sampled to a ground sample distance of Δdr"로 구했다. 창 크기와 기울기 연산자는 원문에 없다.
  - Fig.8b의 확률적 하한 ρ는 fine 모드에서 1,950 m 0.33 m → 4,200 m ≈0.8 m다.
  - Jezero 시험에서 실제 오차는 "dominated by estimator bias"(ML 평균 bias 0.18–0.33 m, stochastic 0.02–0.16 m)였다. 원문은 "a general method of predicting estimator bias remains elusive"라고 쓴다.
  - 기존 엔트로피·기복 지수는 "heuristic measures … useful in rejecting potential outliers, but cannot be used to quantitatively predict the performance"라고 평가한다.
  - [추론] CRB 하나로 신뢰 고도를 정할 수 없다는 근거다.

일반 영상 정합 이론도 같은 구조다.

- **Robinson & Milanfar(2004)**: 평행이동 정합의 Fisher 정보는 영상 내용(기울기 또는 스펙트럼 에너지)에만 달려 있다. 질감이 한 방향뿐이면(조리개 문제) FIM이 특이해져 "any unbiased estimator will have infinite variance"이다. 이들은 tr(J⁻¹)를 영상 내용 기반 성능 예측자로 제안하면서도, 비편향 추정기가 일반적으로 없으므로 순수 CRB는 "overly optimistic"하다고 경고한다 ([Robinson & Milanfar 2004](https://users.soe.ucsc.edu/~milanfar/publications/journal/MotionPerformanceFinal.pdf)).
- **Aguerrebere 외(2016)**: Ziv–Zakai 한계로 세 SNR 영역을 갈랐다 ([Aguerrebere et al. 2016](https://arxiv.org/abs/1602.01541)).
  - CRB가 맞는 고SNR 영역
  - 성능이 "degrades dramatically below a given threshold"하는 중간 영역
  - 정렬 자체가 불가능한 저SNR 영역

  프레임을 더 쓰면 문턱 SNR이 옮겨 간다.
- **지형 고도 프로파일 TAN**: Bergman(1997)의 사후 CRB는 참 위치의 지형 기울기를 측정 행렬로 하는 Riccati 재귀로 정리된다 ([Bergman 1997](https://www.diva-portal.org/smash/get/diva2:315986/FULLTEXT02)).
- **레이더 TAN**: Haydon 외(2025)는 지형 상관길이가 150 m 이상(중앙 경사 ≈2–7°)이면 성능이 괜찮고, 중앙 경사가 10°를 넘으면 "caution is recommended"라고 정리했다 ([Haydon et al. 2025](https://navi.ion.org/content/72/3/navi.707)).

[추론] 이 구조에서 세 가지가 나온다.

1. **오프라인 예측자.** Σ∇는 템플릿 위에서 합한 영상 구조 텐서다. 예측 착륙선 GSD와 잡음에서 지도 패치의 λ_min과 tr⁻¹을 계산하면 예측자가 된다. λ_min은 평행 능선이나 명암 경계 그림자 같은 방향성 퇴화도 잡는다.
2. **고도 의존성.** 착륙선 GSD가 지도 GSD와 같아지는 교차 고도 h* 아래에서는 발자국에 담기는 지도 픽셀 수가 h²에 비례해 줄어든다. 그러므로 CRB만으로는 **완만한** h² 저하가 예측된다. 비행에서 본 절벽은 문턱(모호성) 영역에서 온다.
3. **아무도 계산하지 않은 한계.** 행성 TRN에 Ziv–Zakai, Barankin, Weiss–Weinstein 같은 대오차 한계를 적용한 논문은 찾지 못했다. Setterfield 한계도 연관이 옳다고 가정한다.

플룸 먼지와 저대비는 이 식에 두 경로로 들어간다. Σ∇를 낮추고 σ_η를 높인다. 둘 다 정보량을 곱셈으로 줄여 작동점을 문턱 쪽으로 민다. 이것이 플룸을 TRN 신뢰도에 잇는 정량 고리다.

### 오프라인 적합성: 비행에 쓰인 지표는 LVS의 "외관 엔트로피" 하나다

실제로 비행에 쓰인 지형 적합성 지표는 LVS의 "appearance entropy"뿐이다. 원 출처인 Johnson 외(2017, AAS 17-038)는 1,500 m × 1,500 m coarse landmark 발자국을 지형 평가 단위로 삼고, "bland regions that have appearances with entropy as low as 3.0 (roughly a contrast of 8 DN within a 256 DN image)"이라고 쓴다 ([Johnson et al. 2017](https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/46186)).

같은 논문의 민감도 실험은 이 지표의 예측력을 보여 주는 거의 유일한 정량 증거다.

- **대비**: 엔트로피 5(대비 32 DN)인 지도에서 엔트로피를 1씩 낮췄더니 "LVS worked down to an entropy of 2 which is 4 DN of contrast"였다. 후보지가 모두 5 이상이므로 "a factor of 8 margin against reduced contrast"다.
- **노출**: 평균 밝기 128 DN에서 16 DN까지 동작했다.
- **밋밋한 영역**: 첫 coarse 영상이 "two incorrect matches out of 5"를 냈다.
- **초기 오차**: 수평 5 km, 수직 350 m까지 수렴했다.

**정의와 운용 범위의 원문** [원문]
- AIAA 2015-0851은 "Image entropy measures the information content in an image and is defined as E = -p*log(p) where p is the probability of a brightness value in the image"라고 정의한다(합 기호와 창 크기는 원문에 없다). JGCD 2023 논문에는 엔트로피가 0건이다.
- 2015 현장시험 운용 범위는 엔트로피 4.0–7.5였다. 시험은 3.0–7.5에서 했고, 엔트로피 ≈3 지형에서는 "Landmark matching still succeeds although there are fewer fine matches than is desired"였다 ([Johnson et al. 2015](https://doi.org/10.2514/6.2015-0851)).
- Mars 2020의 최종 운용 범위표(AIAA 2022-1214 Fig.6)는 다음과 같다.
  - Image Entropy ≥4
  - Terrain Relief 0–150 m, Slope 0–15°
  - **LCAM to Map Delta Sun Angle 0–20°**
  - TRN Altitudes 2,000–4,200 m
  - Off-nadir ≤45°, Angular Rate ≤50°/s
  - 현장시험이 덮은 범위는 이보다 넓다(엔트로피 1–7, 기복 0–373 m, 태양각 차 1–121°, 고도 1,163–4,740 m).
  - 99%ile 수평 오차는 coarse 끝 158.0 m, 6초 16.0 m, 10초 14.3 m였다 ([Johnson et al. 2022](https://doi.org/10.2514/6.2022-1214)).

[추론] 보정 쌍(H=5↔32 DN, 3↔8 DN, 2↔4 DN, 1↔2 DN)은 정확히 H = log₂(대비 DN)를 따른다. 이 지표는 거의 확실히 1차 히스토그램 엔트로피이며, 논문은 공식과 창 크기를 주지 않는다. 그렇다면 공간 구조를 보지 못한다. 엔트로피가 높아도 반복되는 사구 지대나 한 방향 능선 질감은 통과한다. Johnson의 스윕은 대비를 전역으로 줄였으므로 Σ∇도 같은 비율로 줄었다. 따라서 "H ≥ 2" 경계는 그 지형에서만 성립하는 Σ∇ 경계이기도 하다.

미사일 장면 매칭에는 더 오랜 전통이 있다.

- **DSMAC (Tomahawk)**: 임무 계획 단계에서 "Scene Area Suitability Data Base"를 쓴다. 장면마다 "the likelihood that the true peak will be larger than all the sidelobes"를 예측하고, 지도와 자기 자신을 10⁶–10¹⁰번 이진 비교해 상관 잡음을 추정한다. 이를 바탕으로 이진화 정답 확률 P_cba와 갱신 정답 확률 P_cu를 낸다 ([Irani & Christ 1994](https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V15-N03/15-03-Irani.pdf)).
  - "moving shadows or seasonal changes in foliage" 같은 시간적 불안정성을 예측해야 한다.
  - 예측 품질에 대한 교훈이 결정적이다. "the predicted P_cba is never less than 0.5. In contrast, the measured P_cba also includes values that are less than 0.5; in some areas the measured P_cba is even zero."
- **Mostafavi & Smith(1978)**: 기하 왜곡 아래 상관의 오획득 확률을 다뤘다 ([Mostafavi & Smith 1978](https://doi.org/10.1109/TAES.1978.308610) [원문]).
  - 닫힌 식이 아니다. 상관 피크와 사이드로브의 평균·분산식 (5)(15)(19)(22)에 가우스 근사를 써서 단일 오프셋 P_FA와 P_CA를 **수치로** 구한다.
  - 원문은 "the Gaussian approximation is least accurate in the tails"라고 경고한다.
  - 핵심 결론: "there is an image size that minimizes the probability of false acquisition. This optimum image size is proportional to the width of the image autocorrelation function and decreases with increasing geometric distortion."
  - [추론] 하강에서는 영상 발자국이 고도에 비례해 줄어든다. 그러면 오획득 확률이 최소가 되는 고도대가 따로 있을 수 있다. 검증할 가설이다.
- **중국의 景象匹配适配性 연구**: 沈林成 외(2010, 航空学报 31(3))의 원 제목은 「景象匹配辅助组合导航中景象区域适配性研究进展」이며 **리뷰 논문**이다 [원문] ([Shen et al. 2010](https://hkxb.buaa.edu.cn/EN/abstract/abstract9465.shtml)).
  - 지표군을 분류만 하고 정의식이나 예측력 수치는 주지 않는다. 분류는 네 갈래다.
    - 화소 통계: 엔트로피, 분산, 거칠기, 독립화소수
    - 에지: 에지 밀도, 영교차 밀도
    - 상관면: 상관길이, 8-이웃 피크비, 피크 첨예도, 부피크/주피크비, 반복패턴
    - 영상 해석
  - 적합성은 "以匹配概率来衡量"(매칭 확률로 잰다)고 하며, 지표의 유효성은 "지표값–매칭확률 곡선의 단조성"으로 평가하자고 제안한다.
  - 한계로는 "基于像素单元的区域适配性特征指标 … 未考虑到实时图与基准图的差别与联系"(화소 통계 지표는 실시간 영상과 기준 영상의 차이를 반영하지 못한다)를 지적한다.
  - CN106709501A는 국소 지표인 피크-사이드로브비(峰值旁瓣比, PSR)와 피크 뾰족함이 매칭 성능과 "明显的单调性"을 보이고 전역 현저성은 그렇지 않음을 확인했다. 반복패턴 지수 cf_i = P_i/s도 썼다 ([CN106709501A](https://patents.google.com/patent/CN106709501A/zh)).
  - Yu 외(2024, CJA)는 반복·약질감 장면의 매칭 신뢰도를 상관 피크로 예측했다 ([Yu et al. 2024](https://doi.org/10.1016/j.cja.2024.06.024) [원문]).
    - 지표 정의
      - 상대 피크 높이 d = H − z_H. z_H는 적응 범위 경계의 평균 높이다.
      - 피크 기울기 Slope = r/d. r = √(ab)이고, a와 b는 2D 가우스 적합에서 얻는다.
      - 부피크/주피크 비 Ratio = d_sub/d_main
    - 판정 규칙: d_main > 0.1을 기본 조건으로 두고, Ratio < 0.8 또는 Slope < 35이면 신뢰로 판정한다(SEN1-2 기준).
    - 성능: 기존 지표가 5 px 허용에서 정밀도 ~80–90%였고, 제안법은 이를 ~5–10%p 올렸다.
  - **고전 지표의 예측력은 낮다** — Yang 외(2026, CJA 39(2):103719)의 SAR–광학 매칭 실험 [원문]
    - 적합 영역 선택의 정밀도(PMA)는 에지 밀도 0.40, 엔트로피 0.35, 분산 0.42였다. 매칭 결과로 만든 라벨 통계는 0.79, 학습 분할(MLSNet)은 0.82였다.
    - [추론] 엔트로피·분산 같은 단일 통계로 신뢰를 예측하면 크게 틀린다. 실제 매칭 몬테카를로로 보정해야 한다는 DSMAC의 교훈과 같다.

[추론] 달에서 "시간적 변화"는 거의 전부 조명이다. DSMAC의 P_cu 틀(지도 자기상관 사이드로브 통계 + 조명 변화 예측)은 달 TRN에 그대로 옮길 수 있다. DSMAC의 교훈은 적합성 예측자를 지도 통계만으로 믿지 말고 렌더 영상 몬테카를로로 검증하라는 것이다.

[추론] 이 연구가 지도와 예측 태양각 렌더로 프레임마다 계산할 후보 지표는 여섯 가지다.

1. 히스토그램 엔트로피 (LVS와 비교 가능)
2. 예측 GSD에서의 기울기 FIM λ_min과 tr(J⁻¹)
3. 주변 영역과의 자기상관 부피크비·PSR
4. 반복패턴 지수
5. 검출 가능 크기 이상의 크레이터 수
6. 착륙 태양각에서의 그림자 비율

**행성 착륙에 이 지표들을 적용한 논문, 그리고 어떤 적합성 지표든 TRN 실패를 예측하는 ROC/AUC를 보고한 논문은 찾지 못했다.**

### 크레이터 항법: 보이는 크레이터 수는 지도가 따라오는 동안 고도와 무관하다

**최소 개수.** Christian, Derksen & Watkins(2021)에 따르면 테두리 원뿔 전체를 검증에 쓰면 "with only three craters (instead of the usual ≥ 5 craters)"로 식별할 수 있다. 중심 좌표만 쓰는 CNav식 방법은 삼중항과 검증용 2개, 모두 5개가 필요하다 ([Christian et al. 2021](https://arxiv.org/abs/2009.01228)).

**모양 가정.** 작은 크레이터의 25–50%는 타원율이 1.1을 넘는다. ε = 1.1 크레이터를 원으로 가정하면 천저 영상에서 법선 오차가 20°를 넘는다.

**식별 몬테카를로.** 타원 적합 오차가 0에서 3 px로 커질 때 정답 식별률은 다음과 같았다. 시나리오마다 100회씩 돌렸고, 모든 경우에서 오식별은 0건이었다.

| 시나리오 | 0 px | 1.5 px | 3 px |
|---|---|---|---|
| 150 km, 국소 서술자 | 100% | 94% | 83% |
| 600 km, 전역 서술자 | 100% | 99% | 95% |

실패는 오식별이 아니라 "no match"나 "fewer than three craters"로 나타났다. 카메라 위치 RSS 오차는 150 km에서 0.5 px일 때 116 m, 3 px일 때 923 m로 커졌다. 다만 이 결과는 1–2 km 이상 크레이터만 담은 Robbins 카탈로그 기반의 궤도 고도 결과다.

[산술] "100회 중 0건"이 보장하는 오식별 확률은 95% 신뢰로 ≲3%까지다(rule of three). 무결성급(10⁻⁴–10⁻⁷) 오고정률을 보이려면 단순 몬테카를로가 아니라 중요도 표본추출이나 해석적 꼬리가 필요하다.

**고도 하한 모형.** 달의 작은 크레이터는 평형 누적 크기분포를 따라 반경 r 이상 개수가 r⁻²에 비례한다 ([Minton et al. 2019](https://arxiv.org/abs/1902.07746)). 가장 젊은 지형에서도 5 m 이상 크레이터가 km²당 10³개를 넘는다 ([LunarNav](https://arxiv.org/abs/2301.01350)). 이를 천저 카메라 기하와 합치면 다음 [추론] 모형이 나온다.

- **카메라 한계 영역**: 최소 검출 크기가 카메라 GSD로 정해지는 동안 시야 안 크레이터 수 N_vis ≈ C·(N_pix/k_px)²는 고도와 무관하다. 여기서 C는 크기분포 계수, N_pix는 영상 폭의 픽셀 수, k_px는 검출기가 크레이터로 잡는 최소 픽셀 지름이다.
- **지도 한계 영역**: 지도의 최소 크레이터 크기 D_map이 지배하는 고도 아래에서는 N_vis가 h²로 줄어든다.
- **종료 고도**: h_fail = ½·cot(θ/2)·√(N_min·D_map²/C). 여기서 θ는 카메라 시야각, N_min은 고정에 필요한 최소 크레이터 수(테두리 검증 3, 중심 검증 5)다.

Firefly의 GNC 설계자가 "A hundred kilometers up, one kilometer up, or 10 meters up — when you look at the surface, you see craters"라고 한 자기유사성과 같은 그림이다 ([Freeman 2025, 2차 인용](https://www.opsjournal.org/DocumentLibrary/Uploads/Freeman%20RH_JSOC_Q2B_2025.pdf)).

[산술] 시야각 60°, N_min = 3, km²당 10³개(5 m 이상), 기울기 −2가 수십 m 규모까지 유지된다고 가정하면(특정 지역에 대해서는 미검증) h_fail은 다음과 같다.

| 지도의 최소 크레이터 D_map | h_fail |
|---|---|
| 5 m | ≈ 50 m |
| 25 m | ≈ 240 m |
| 50 m | ≈ 470 m |
| 100 m | ≈ 950 m |

이 범위는 CNav의 350 m, 남극 CNav의 ~1 km와 같은 자릿수다 ([Woicke & Krüger 2023](https://doi.org/10.2514/6.2023-0690)). 크레이터 TRN의 고도 하한은 카메라의 성질이 아니라 **지도 제품의 설계 변수(최소 수록 크레이터 크기)**라는 뜻이다.

**하강 영상의 검출 확률 자료는 없다.** 크레이터 지름을 픽셀로 잰 값에 대한 검출 확률 곡선은 하강 영상에 대해 찾지 못했다. 집계 수치만 있다.

- STELLA는 CRESENT-365(100 km 궤도) 영상 15,283장에서 자세 추정을 내지 못한 비율이 판마다 다르다.
  - arXiv판: 2,070장(≈13.5%). "largest failures occurring in poorly-illuminated and sparse-crater areas" ([McLeod et al. 2025](https://arxiv.org/abs/2509.20748))
  - Astrodynamics 출판판: STELLA-core가 11,213장, STELLA-core-6(대응점 6개 게이트)이 8,179장에서 추정을 냈다. 미산출은 ≈26.6%와 ≈46.5%다 [산술].
  - 오차는 태양각 0–80°에서 일정하다가 80–90°에서 급증했다 ([McLeod et al. 2026](https://doi.org/10.1007/s42064-025-0294-0) [원문]).
- **NAC 영상의 작은 크레이터 검출 (Fairweather et al. 2022)** [원문]
  - 지름 ≥20 m에서 입사각 41.8–66.9°(0°=정오)에 걸쳐 Recall 0.86–0.98, Precision 0.68–0.90이었다. 오검출 비율은 0.10–0.32다 [산술].
  - 원문은 "between ∼50° and 70° are the most reliable", ">80° … large shadows"라고 쓴다.
  - 1 km 미만 카탈로그를 자체 제작할 때 조명별 오검출률의 기준이 된다 ([Fairweather et al. 2022](https://doi.org/10.1029/2021EA002177)).
- **전역 카탈로그의 한계 (Hamel et al. 2025)** [원문]
  - 원문: "the system does not work reliably at altitudes of 20 km or less using Moon crater databases available in the literature … despite the image processing function detection process still correctly detecting craters … due to the sparseness of the database"
  - 검출은 되는데 **지도가 비어서** 실패한 것이다. 궤적 맞춤 증강 DB로 5 km까지 시연했다.
- DLR CNav의 고고도 측정 정확도는 "limited by the camera resolution and not by the map resolution"이다 ([Trigo et al. 2018](https://elib.dlr.de/118510/1/hybrid_optical_navigation_by_crater_detection_for_pinpoint_landing_%28elib%29.pdf)).

### 온라인 신뢰도·무결성: 비행 게이트는 개수와 피크 모양을 보고, 보호수준은 아직 행성 TRN에 오지 않았다

비행과 시험에 쓰인 게이트는 단순하다.

- **LVS**: coarse landmark 총수와 fine 누적 inlier 수를 본다. 한 장이라도 최소 inlier에 못 미치면 coarse를 다시 한다 ([Johnson et al. 2023](https://doi.org/10.2514/1.G006780)).
- **ALHAT/Morpheus HRN**: "quality metrics on the correlation scores, the measurement covariance, and the correlation peak quality (sharpness)"를 본다 [요약 확인] ([Carson et al., NTRS 20170009170](https://ntrs.nasa.gov/api/citations/20170009170/downloads/20170009170.pdf)).
- **DLR CNav**: NIS T² = zᵀS⁻¹z가 "should follow a χ²-distribution with n_z-degrees of freedom … (in this case 6) … A threshold value of 30 was used"이다 ([Trigo et al. 2018](https://elib.dlr.de/118510/1/hybrid_optical_navigation_by_crater_detection_for_pinpoint_landing_%28elib%29.pdf)).
- **DSMAC Block II**: 연속 세 프레임의 피크 위치를 관성 예측과 비교해 "at least two of the three positions are consistent"일 때만 갱신한다. Block IIA는 관성으로 이동시킨 상관면을 합산해 사이드로브를 평탄하게 만든다 ([Irani & Christ 1994](https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V15-N03/15-03-Irani.pdf)).

[산술] χ²(6) 분포에서 30을 넘을 확률은 e⁻¹⁵(1 + 15 + 112.5) ≈ 3.9×10⁻⁵이다. CNav 게이트의 정상 오거부율이 그 정도라는 뜻이다.

**LVS 게이트의 원문 세부** [원문]
- 이상치 분류: "Red (correlation peak mis shaped), purple (correlation peak low or wide), yellow (geometric) and orange (Mahalanobis distance)" ([Johnson et al. 2022](https://doi.org/10.2514/6.2022-1214)). 피크 높이·폭 임계 수치는 공개되지 않았다("specified").
- Mahalanobis 게이트: d² = rᵀS⁻¹r < γ, γ = q3 + 1.5·(q3 − q1). 원문 예는 γ = 6.07, 포함 확률 95.2%다 ([Johnson et al. 2023](https://doi.org/10.2514/1.G006780)).
- 2015 현장시험: "a threshold of 200 landmark matches easily separates the matches that meet the 40m requirement from those that don't". 운용 범위 안 128회 중 7회가 적은 매칭과 큰 오차를 냈다 ([Johnson et al. 2015](https://doi.org/10.2514/6.2015-0851)).

**신뢰 게이트의 유일한 실측 작동점 — Johnson et al. 2025, "Challenges and Solutions for Early Morning Terrain Relative Navigation on Mars"** (AIAA 2025-2076, [doi:10.2514/6.2025-2076](https://doi.org/10.2514/6.2025-2076)) [원문]
- **inlier 게이트**: "at least 20 estimated inliers effectively removes false positives (2 × 10⁻⁸ false positive rate) while still achieving an 83% true positive rate" (Fig.12c). Mars 2020도 "a threshold of 20 inliers was required for a position update"였다.
- **이상치 내성**: "tolerate outlier fractions exceeding 60% above a minimum number of about 40 candidate feature matches"
- **조명 지표**: 같은 고도 지도를 두 조명 조건으로 렌더한다(알베도 일정 가정). fine 랜드마크 크기 패치마다 셰이딩 차이의 표준편차를 구하고, 그 평균을 지표로 쓴다. 원문은 "a maximum illumination metric threshold of 0.3 will bound the effect of illumination"이다.
- **이른 아침 시험** (태양 고도는 같고 방위만 180° 다름)
  - Anza-Borrego 현장시험: coarse 5개 중 4개가 맞았고, fine은 150개 중 39개만 맞았다.
  - CaSSIS 화성 실영상: fine inlier가 40까지, 한 표적은 ~20까지 떨어져 "too close to failure"였다.
  - 결론: "there is a problem with early morning landing but only if the terrain is rough"
- [추론] 이 연구가 처음으로 주는 것은 두 가지다. 오고정률을 실제로 측정한 게이트, 그리고 지도 렌더로 계산하는 사전 조명 지표다. 달 남극에 같은 방식(inlier 수 게이트 + 조명 지표)을 적용해 고도별로 보정하는 것이 이 연구의 직접 연장선이다.

**TERCOM 오고정 제거 (Uslu et al. 2026, IJASS 27(4))** [원문]
- **minima ratio**: R = |Score_best − Score_second| / Score_second × 100. 두 최솟값이 같으면 R = 0(완전 모호)이다.
- **라벨**: 수평 오차 90 m 미만을 "reliable"로 정했다.
- **작동점**: B/G = 1, 2, 5에서 R_MAD 임계는 32, 15, 4다. B/G 1과 5에서 진양성률은 각각 ~95%, ~99%다.
- **결과**: "reduced more than 44% of false fixes" ([Uslu et al. 2026](https://doi.org/10.1007/s42405-026-01126-6))

**오연관 무결성 (Zhu, Joerger & Meurer 2020, PLANS)** [원문]
- **연관 순열**: N_L개 랜드마크의 순열 A_i마다 혁신 γ_i ~ N(b_i, Σ_i)를 둔다.
- **올바른 연관 확률**: P(CA_k) ≥ P(Γ_0 < min_i b_iᵀΣ_i⁻¹b_i / 4)로 하한을 준다.
- **무결성 위험**: 이를 P(HMI) ≤ 1 − (1 − P(HMI ∣ CA))·P(CA)에 넣는다.
- **수치 예**: N_L = 3, 잡음 1 px이면 오연관 확률을 10⁻⁷ 아래로 묶을 수 있다 ([Zhu et al. 2020](https://doi.org/10.1109/PLANS46316.2020.9109919)).
- [추론] 크레이터·랜드마크 연관의 무결성 위험을 고도별로 계산하는 틀로 그대로 쓸 수 있다.

[추론] NIS 게이트는 사전 예측과 어긋나는 고정만 잡는다. 사전 공분산이 작고 근처에 반복 구조가 있으면 예측 타원 안에 떨어지는 오고정은 그대로 통과한다. 이 문제에는 피크 유일성 지표가 따로 필요하다.

- **PSR (Bolme 외)**: 정상 추적에서 20–60, ≈7이면 가림이나 실패다 [요약 확인] ([Bolme et al. 2010](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf)).
- **OpenCV `phaseCorrelate`의 `response`**: 피크 주변 5×5 합을 정규화한 값으로, 피크가 하나일 때 1이고 "smaller when there are multiple peaks"이다 ([OpenCV imgproc.hpp](https://raw.githubusercontent.com/opencv/opencv/4.x/modules/imgproc/include/opencv2/imgproc.hpp)).
- **TERCOM minima ratio**: 점수 지도에서 가장 강한 두 극소값의 비다. 45개 후보 특징 중 가장 믿을 만했고, "reduced more than 44% of false fixes"였다 [요약 확인] ([IJASS 2026](https://link.springer.com/article/10.1007/s42405-026-01126-6)).

지도 기반 비전 항법의 EKF는 "significantly inconsistent (overconfident) covariance estimates"를 낸다는 보고도 있다 ([Zhu & Taylor 2017](https://ieeexplore.ieee.org/document/7858648/)).

무결성 이론은 준비되어 있지만 행성 TRN에 들어오지 않았다. Zhu, Meurer & Günther(2022)는 무결성 위험 IR과 보호수준을 정의하고 CAT I 착륙 요구(IR ≤ 10⁻⁷, AL ≤ 10 m)를 예로 든다 ([Zhu et al. 2022](https://elib.dlr.de/147041/1/Integrity_of_visual_navigation_ION_NAVIGATION_v4.pdf)). 이들이 꼽은 비전 고유의 장애는 다음과 같다.

- 잡음이 아닌 편향 오차("sometimes even more than half")
- 연관 오류("for N_p points, there are maximum N_p! association possibilities")
- 잔차가 전역 최소와 거의 같아 잡기 어려운 국소 최소

이들은 영상 밝기, 특징, 위치 세 영역에 감시를 두자고 제안한다.

- **RAIM식 PnP 보호수준**: 보호수준이 키포인트 수의 −1/2승으로 줄어든다. 활주로 키포인트 4개로는 1 km 거리에서 진행방향 PL이 700 m를 넘고, 30개면 26 m 아래다 ([Bruvik et al. 2026](https://arxiv.org/abs/2608.10023)).
- **데이터 기반 PL**: IR 0.01을 만족했지만 오경보율이 0.40–0.47이었다. 가용성 비용이 크다는 뜻이다 ([Gupta & Gao 2021](https://arxiv.org/abs/2101.06379)).

**달이나 화성 TRN에 보호수준·무결성 틀을 적용한 연구는 찾지 못했다.**

[추론] 선례가 있는 층을 쌓으면 다음 온라인 신뢰도 스택이 된다.

1. 영상 밝기 감시: 전역 대비·엔트로피, 노출, 먼지 헤이즈 지표
2. 랜드마크별 피크 높이·폭·PSR·부피크비
3. 기하 검증 뒤 영상별 inlier 수와 비율, 또는 크레이터 삼중항 수용과 테두리 재투영
4. 다중 프레임 일관성
5. 필터 NIS
6. 남은 랜드마크 수와 배치로 계산한 보호수준

마지막 층은 n^(−1/2) 스케일링 때문에 특징 결핍을 곧바로 "가용성 없음" 플래그로 바꾼다. 어느 층도 달 TRN 실패에 대한 ROC 증거가 없다.

### "고도별 가용성"을 정식화한 연구는 없다

영어·중국어 검색 어디에서도 P(정답 고정 ∣ h, 지형, 태양) 또는 P(오고정 ∣ h)를 달·화성 하강에 대해 정식화한 연구는 나오지 않았다. 2026-10-06 KAIST 빈칸 검색에서도 판단이 유지됐다. Web of Science·IEEE Xplore·Scopus에서 가용성·신뢰도, 조명, 적합성, 고도·운용 범위, 피인용 추적을 검색했고, RISS 국내 문헌도 봤다. 다만 아래 표처럼 **부분 정량화**는 몇 편 있다. 지금 있는 고도 한계는 네 종류뿐이다.

| 종류 | 사례 |
|---|---|
| 신뢰도 외의 이유로 정한 설계 창 | LVS의 EDL 일정, ELViS의 "expected" 50 m |
| 단일 실패 고도 | CNav ~350 m. Hamel 외(2025)의 "Operations from altitude of 20 km or lower require an enhanced database" [요약 확인] ([Hamel et al. 2025](https://doi.org/10.2514/6.2025-2075)) |
| 정확도만 다룬 CRB | Setterfield 2021 |
| 고정 궤도 고도의 식별 몬테카를로 | Christian 2021 (150 km, 600 km) |
| 지도 GSD별 최저 매칭 고도의 반정량 비교 | Smith et al. 2020: 0.5–2 m/px 차이 작음, 4 m/px 약 두 배, 8 m/px 실패. 그림 눈금이 없음 [원문] |
| 고도 × 태양 고도 오차표 | Hamel et al. 2025: 20/50/100 km × 최대 태양 고도 3–40°의 1σ 오차 [원문] |
| 게이트 작동점(고도 무관) | Johnson et al. 2025: inlier ≥20 → 오탐률 2×10⁻⁸, 검출률 83% [원문] |

GSFC autoNGC도 달 TRN을 40 → 15 km에서만 모사했고, 15 km 아래는 "a faster TRN measurement processing time may be required"라고만 적었다 ([Shoemaker et al. 2022](https://ntrs.nasa.gov/api/citations/20210024481/downloads/autoNGC_SciTech2022_Shoemaker_final.pdf)). 가장 가까운 개념 틀은 DSMAC의 장면별 P_cu다.

---

## 민감도 실험은 지도–하강 조명 방위각 차이를 1순위 파괴 변수로 지목한다 (질문 3)

### 조명: 방위각 차이 30–60°에서 고전 매칭이 깨진다

**JPL/LuNaMaps 남극 실험 (Georgakis & Ansar 2024).** 조명을 가장 체계적으로 쓸어 본 연구다 ([Georgakis & Ansar 2024](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf)).

- 정사 실험: 남극 LOLA DEM 150 × 150 km를 10 m/px hillshade로 렌더했다. 태양 고도 0–4°, 방위 0–330°를 쓸었다.
- 원근 실험: Blender Cycles로 3 km 고도, 시야각 100°, 태양 고도 2–6°에서 렌더했다.
- 비교 대상: SIFT, MegaDepth로 학습한 LoFTR, 달 렌더로 미세조정한 LoFTR-fine.

결과는 다음과 같다.

| 방법 | 결과 |
|---|---|
| SIFT | "SIFT starts failing at 60° of azimuth difference" |
| 기성 LoFTR | "starts struggling when azimuth difference is beyond 90°, with RA-5 falling at or below 1.5%" |
| LoFTR-fine | 가장 어려운 180° 방위·3° 고도 차이에서도 "RA-1 (41.8% of 479 matches)" |

자세 15° 이하로 학습한 모델은 45° 자세에서도 성능이 떨어지긴 했지만 정확한 매칭을 많이 냈다.

**실제 LROC 다조명 영상 (Xie 외 2025).** 세 지역의 321쌍으로 비교했다 ([Xie et al. 2025](https://doi.org/10.3390/rs17132302)).

- 크레이터 이웃구조 매칭(CNSFM)은 Chang'e-3 지역과 Apollo 11 지역에서 100%, **Shackleton 근처 남극 지역에서 72.3%**를 기록했다.
- 남극에서 성공한 최대 방위각 차이는 CNSFM 147.7°, WSSF 62°, HAPCG 47°, ML-HLMO 34°, **SIFT 33°**였다.
- 남극 실패의 원인은 "low solar elevation angles … azimuthal differences approaching 180 degrees result in overlapping image regions predominantly covered by shadows"였다.
- 적도에서는 반대로 태양이 높으면 과노출로 대비가 사라졌다.

**화성 회전익 유사 사례 (Pisanti 외 2025).** HiRISE 기반 렌더(0.25 m/px), 고도 64–200 m 쿼리 4,500장으로 지도 태양 고도를 2–90° 범위에서 쓸었다 ([Pisanti et al. 2025](https://arxiv.org/abs/2502.09795)).

- 조명이 일치하면 DEM 기하를 더한 Geo-LoFTR이 "87% @1m accurate"로 SIFT보다 63%p 높았다.
- 쿼리 태양 고도 10°에서는 54–63%였다.
- 2°에서는 "all the methods significantly degrade … extensive shadow coverage may saturate the geometric information"이었다.

후속 MARTIAN 연구는 실제 Ingenuity 비행 영상에서 Acc@5m 89.4%를 보고했다 ([MARTIAN 2026](https://arxiv.org/abs/2605.29647)).

**절대 태양 고도의 두 극단.** 태양 고도는 지도–하강 차이와 별개로도 실패를 만든다. DLR Maass(2016)는 Kaguya DEM 렌더 134,784장으로 태양 고도 5–65°와 카메라 고도각 5–90°를 쓸었다. 그림자 분할 크레이터 검출기는 "almost certain for all but the lowest camera elevation of 5° and illumination elevations above 55°"였다 ([Maass 2016](https://elib.dlr.de/104630/1/art_10.1007_s12567-016-0129-1%20Maass.pdf)). 정리하면 다음과 같다.

| 극단 | 실패하는 대상 | 근거 |
|---|---|---|
| 높은 태양(55° 초과) | 그림자 기반 크레이터 검출 | Maass 2016 |
| 매우 낮은 태양(5° 이하) | 지도를 맞춰도 모든 매처 | Pisanti 2025 |

**학습 검출기와 밝기 변화.**

- Andolfo 외(2025)의 YOLO 크레이터 검출기는 태양 방향 네 개로만 학습했을 때, 처음 보는 방위에서 "crater matching performance significantly decreased"였고 ~0.5 px 편향이 남았다. 재학습 뒤에는 편향 없이 ~2 px(1σ)가 되었다 ([Andolfo et al. 2025](https://doi.org/10.3390/aerospace12030195)).
- LunaNet은 밝기 ±30%에 흔들리지 않았다. 고전 검출기는 검출 수가 늘었지만 "greater percentage incorrectly matched"였다 ([Downes et al. 2020](https://arxiv.org/abs/2007.07702)).

**비행 시스템은 렌더 지도로 문제를 피했다.** SPLICE의 New Shepard 비행에서는 거의 천정 태양에서 찍은 USGS 영상 DB가 "failed to correlate features in the mountainous regions"였다. 비행 태양각으로 레이트레이싱한 Blender DB는 ">400 unique features … over 400 continuous seconds"를 냈다. 지도를 2시간 조명 구간으로 나눴더니 09:31 비행에 10:30 DB가 맞게 골라졌다. 다만 저자들도 "analysis on the size of the sun angle change required for a new database has not been performed"라고 쓴다 ([Smith et al. 2022](https://ntrs.nasa.gov/citations/20210023848)).

ELViS 논문(2024)은 "landing in the early morning (beyond the currently tolerable limits of illumination differences between map and descent images)"을 검토 중이라고만 쓰고 수치를 주지 않는다 ([Trawny et al. 2024](https://doi.org/10.2514/6.2024-0314)). 그 정량 분석은 Johnson et al. 2025에 있다(§질문 2 온라인 신뢰도).

**원문으로 확인한 조명 허용 범위** [원문] — 지표가 서로 다르므로 직접 비교하지 말 것

| 출처 | 허용·파괴 조건 | 성격 |
|---|---|---|
| Mars 2020 LVS 운용 범위 ([Johnson et al. 2022](https://doi.org/10.2514/6.2022-1214) Fig.6) | **LCAM–지도 태양각 차 0–20°**. 현장시험은 1–121°를 덮었고 "except when there were large illumination differences or extreme terrain relief" 외에는 요구를 만족 | 운용 범위 |
| LVS 지도 제작 ([Cheng et al. 2021](https://doi.org/10.1029/2020EA001560)) | 지도 원천 영상은 착륙 시각 태양의 고도 ±15°, 방위 ±35° 안. 대기 τ < 1.0 | 지도 제작 요구. **근거 시험은 원문에 없음** |
| LVS 2015 현장시험 ([Johnson et al. 2015](https://doi.org/10.2514/6.2015-0851)) | 태양 고도 30–50°(시험 18–50°). 방위는 시험 날짜로 고정되어 통제하지 못함 | 운용 범위 |
| Johnson et al. 2025 (화성 이른 아침) | 렌더 조명 지표 ≤ 0.3. 방위 180° 차는 **거친 지형에서만** 문제 | 사전 지표 + 현장·실영상 |
| LION ([Delaune et al. 2016](https://doi.org/10.1016/j.robot.2016.01.007)) | "Performances are more severely affected above 10 deg of elevation change, and 20 deg of azimuth". 다만 20 km에서 방위 90° 차에도 초기 3-RMS 1,700 m를 680 m로 줄임. 조건당 63회 | 몬테카를로 |
| Mars 헬기 절대 위치추정 ([Brockers et al. 2022](https://doi.org/10.1109/AERO53065.2022.9843673)) | 지도 태양 경사 90° 기준, 질의 30°/150°에서 5 m 성공률 SIFT ≈9/15%, SuperPoint ≈14/35%. 60–120°에서는 ≈90% 이상 (그림 판독) | 렌더 (AirSim) |
| Draper ([Steffes et al. 2019](https://doi.org/10.2514/6.2019-1178)) | 템플릿 매칭은 태양각 ±5° 변화에서 오차 <0.25 px("very robust"). SIFT는 "even 1deg of change results in several pixels of error" | 궤도(NRHO) 렌더 |
| 크레이터 항법 ([Hamel et al. 2025](https://doi.org/10.2514/6.2025-2075)) | 태양 고도 3–40°에서 동작, 최적 10–20°. 20 km·3°에서 1σ 790/537/1,409 m, 10° 이상에서 ~60–110 m | SurRender 몬테카를로 |
| 남극 CNav ([Woicke & Krüger 2023](https://doi.org/10.2514/6.2023-0690)) | 국소 태양 고도 < 40°, lit 지형 위 궤적 | 운용 조건 |
| 지도 없는 TRN ([Christian et al. 2021](https://doi.org/10.2514/1.A34875)) | 낮은 위상각(opposition effect, g < 8° 언급)에서 랜드마크 수 감소 | 실영상 |
| LIMA ([Rothenberger et al. 2025](https://doi.org/10.2514/6.2025-2073)) | 남극 10 m/px 렌더(고도 0–4°, 방위 0–330°) 집계 RA-1: SIFT 13.8%, LoFTR 17.7%, 달 미세조정 LoFTR 84.9%(미관측 영역 70.6%). **각도별 성공률은 원문에 없음** | 렌더 |

[추론] 운용 시스템들이 공통으로 쓰는 허용폭은 "지도–하강 태양 고도 차 ~10–15°, 방위 차 ~20–35°, 합성 각도 차 ~20°"다. 이 안에서는 고전 매처도 동작한다. 달 남극처럼 태양 고도가 0–3°이면 방위만 조금 달라도 그림자 배치가 크게 바뀐다. 그래서 같은 각도 허용폭이 훨씬 좁아질 수 있다(LuNaMaps의 "very sensitive to the timing"). 이것이 RQ3의 핵심이다.

[추론] 남극에서는 태양 방위가 한 삭망월에 360° 돈다. 고전 매처의 허용 방위차가 30–60°라면, 지도 한 장으로 쓸 수 있는 착륙 창은 삭망월마다 며칠 수준일 것이다. 직접 측정된 값은 아니다.

### 고도·해상도: 하한은 축척 불일치와 카탈로그 완전성 중 먼저 오는 쪽이 정한다

**영상 매칭 TRN.** 하한은 지도 GSD와 카메라가 함께 정한다(§질문 1의 표).
- **실측 종료점**: New Shepard 두 점(20 m/px → 4,000 m, 3 m/px → 750 m, 시야각 12.2°)과 Xombie(1 m/px → ~85 m)다.
- **해석의 꺾임**: Smith et al. 2020은 지도 0.5–2 m/px에서 하한 차이가 작고, 4 m/px에서 약 두 배, 8 m/px에서 실패를 보였다.
- **이론**: Setterfield의 GSD 일치 최적점이다. SPLICE 운용 개념도 수동 TRN을 "until the ground sample distance (GSD) per pixel exceeds the GSD of the onboard reference maps"까지만 쓴다.
- **화성 VISINAV 이론값**: Mourikis et al. 2009는 화성의 차단점을 일반론으로 1,000–500 m AGL로 보았다. 7 m/px 지도 구간(3.8–3.1 km)은 영상당 ML ~40개, 1 m/px 지도 구간(1.6 km–230 m)은 ~80개였다 [원문].
- **특징 반복률**: Delaune et al. 2016에서 Harris–Laplace 반복률은 685 m에서 95.8%, 400 m에서 55%였다. 고도가 내려가 축척이 벌어지면 반복률이 급락한다 [원문].
- **크레이터 증강 DB**: Hamel et al. 2025의 증강 DB 측정 오차(1σ)는 다음과 같다 [원문].
  - 20 km(GSD 55 m) → 5 km(GSD 14 m)에서 AT·CT 수 m, R 5–20 m였다.
  - 조명이 다른 검증 모자이크에서는 R 오차가 140–170 m로 커졌다.

**크레이터 카탈로그 TRN.** 하한은 카탈로그의 최소 크기가 정한다.

| 연구 | 카탈로그 | 한계 |
|---|---|---|
| Andolfo 2025 | 기존 카탈로그 | "down to altitudes of approximately 10 km". 그 아래에서는 카탈로그 크레이터가 없어 매칭 실패. ~25 km에서는 크레이터가 "very shallow"해 검출 안 됨 ([Andolfo et al. 2025](https://doi.org/10.3390/aerospace12030195)) |
| Hamel 2025 | 전역 공개 DB | 저궤도에서 100 m 이하 정확도. 20 km 아래는 궤적별 증강 DB 필요, 5 km까지 시연 [요약 확인] ([Hamel et al. 2025](https://doi.org/10.2514/6.2025-2075)) |
| DLR 남극 연구 | CNav 카탈로그 | **~1 km** (렌더 DEM 아티팩트 한계). 전역 카탈로그만 쓰면 ~4 km [원문] ([Woicke & Krüger 2023](https://doi.org/10.2514/6.2023-0690)). ESA GNC판 초록의 "at least 600 m"는 원문 수치와 다름 |

**해상도 바닥과 프레임률.**

- Pisanti 연구의 64 m 바닥은 HiRISE 질감 해상도가 정한 것이다. Geo-LoFTR은 64–200 m에서 성능이 일정했다.
- Cowan 외(2025)의 남극 광학흐름 VO는 영상 크기 1024²에서 성능이 포화했다. 프레임률은 "degrades sharply below 1 Hz"였다 ([Cowan et al. 2025](https://arxiv.org/abs/2511.17720)).

### 지형: 같은 알고리즘에서 30%p 가까이 차이가 난다

DIMES의 Gusev 99% 대 Meridiani 71%는 같은 알고리즘에서 지형만 바꿔 28%p가 빠진 가장 깨끗한 증거다 ([Cheng et al. 2004](https://robotics.jpl.nasa.gov/media/documents/DIMES-ai-space.pdf)). 남극 매칭 성공률 72.3%(Xie 2025)도 같은 방향이다.

원문 추출로 더해진 지형 증거 [원문]
- **LVS 2015**: 엔트로피 ≈3인 지형에서도 매칭은 성공했지만 fine 매칭 수가 부족했다. 실패한 7회는 모두 매칭 수가 적었다(200개 미만).
- **LVS 2022**: 현장시험에서 요구를 못 맞춘 경우는 큰 조명 차와 "extreme terrain relief"뿐이었다.
- **Johnson 2025**: 이른 아침(방위 180° 차) 문제는 "only if the terrain is rough"였다. 기복이 그림자를 만들어야 조명 차가 매칭을 깨뜨린다는 뜻이다. 그러므로 **지형 기복과 조명 잔차는 곱으로 작용하는 상호작용 항**으로 다뤄야 한다 [추론].
- **Wang 2026**: "relatively flat, with a large proportion of weak texture and completely shadowed regions"에서 렌더 영상이 0이 되어 실패했다.
- **국내 연구**: 구평모·박영범·박찬국(2016, 한국항공우주학회지 44(7))은 레이더 고도계–DEM 방식의 달 착륙 지형상대항법에서 **영역기반 지형 험준도 지수**와 성능의 관계를 분석했다. 영상 TRN은 아니지만 지형 지수–성능 관계를 다룬 국내 선례다 [RISS 서지].

PANGU로 Malapert 지역을 렌더해 4 km → 100 m 하강을 모사한 Cowan 외(2025)의 속도 오차는 다음과 같았다 ([Cowan et al. 2025](https://arxiv.org/abs/2511.17720)).

| 지형 | 속도 오차 | 상대 오차 |
|---|---|---|
| 평탄 | 1.49 m/s | 3.25% |
| 경사 | 4.82 m/s | 4.53% |
| 봉우리 | 2.07 m/s | 3.45% |

기복은 평면 와핑 가정도 깨뜨린다. "The 2D-2D homography assumption … begins to break down when 3D structures such as mountains are viewed from low altitudes" ([Maggio et al. 2023](https://arxiv.org/abs/2302.08011)).

지도 오차도 지형 변수다. LRO 궤도 오차(~7/7/0.5 m)는 20 m/pix보다 촘촘한 LOLA DEM에서 "flag-poles"와 "pot-holes" 인공물을 만든다 ([Restrepo et al. 2022](https://ntrs.nasa.gov/api/citations/20210024816/downloads/LuNaMaps_Scitech2022_Nov19.pdf)).

### 기하와 운동: 저고도 VO의 실제 구속 조건

**천저 이탈각은 비교적 무해하다.** Maass(2016)에서 실패는 카메라 고도각 5°(천저 이탈 85°)에서만 났다.

**저고도 VO는 운동에서 무너진다.** Draper IBAL은 자이로 보조를 끄면 처리할 수 있는 프레임 비율이 크게 줄었다 ([Maggio et al. 2023](https://arxiv.org/abs/2302.08011)).

| 설정 | 33–29 km | 29–23 km | 23–4.5 km |
|---|---|---|---|
| 자이로 보조 (4 Hz, 2 Hz) | 100% | 100% | 100% |
| 자이로 없음, 4 Hz | 100% | 96% | 1–3% |
| 자이로 없음, 2 Hz | 100% | 63% | 0–1% |

Cowan(2025)은 카메라 잡음보다 IMU·상태 잡음이 더 해롭다고 보고했다. "small angular velocities generate large image motion"이기 때문이다.

**초기 오차 포획 범위.** LVS는 3.2 km 초기 오차를 ~200 m로 줄였다 ([Johnson et al. 2023](https://doi.org/10.2514/1.G006780)). Candan & Servadio는 5 km 초기 오차에서 50–100 m로 수렴했다 ([Candan & Servadio 2026](https://arxiv.org/abs/2606.14776)).

### 변수별 파괴점

| 변수 | 파괴점 | 근거 자료 | 출처 |
|---|---|---|---|
| 지도–하강 태양 방위각 차이 (저태양) | SIFT 최대 성공 33°(실제 극지 쌍), 60°에서 실패(hillshade). 기성 LoFTR ~90°. 달 미세조정 LoFTR은 180°에서도 RA-1 41.8% | LROC 실영상, LOLA 10 m/px 렌더 | [Xie 2025](https://doi.org/10.3390/rs17132302); [Georgakis & Ansar 2024](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf) |
| 지도 GSD·카탈로그 완전성 | 20 m/px → 4 km, 3 m/px → 750 m(시야각 12.2°), 1 m/px → ~85 m(Xombie). Smith et al. 2020: 4 m/px에서 하한 약 두 배, 8 m/px 실패. 크레이터 카탈로그 ~350 m(CE-3 재생), ~1 km(남극 렌더), ~4 km(남극·전역 카탈로그만), ~10–20 km(전역 카탈로그) | 비행 후처리, 시뮬레이션 | [Blue Origin](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf); [Maass 2020](https://doi.org/10.2514/1.G004850); [Andolfo 2025](https://doi.org/10.3390/aerospace12030195) |
| 지형 질감 | 유효해 99% 대 71%, 극지 72.3% 대 100% | DIMES 몬테카를로, LROC | [Cheng 2004](https://robotics.jpl.nasa.gov/media/documents/DIMES-ai-space.pdf); [Xie 2025](https://doi.org/10.3390/rs17132302) |
| 절대 태양 고도 | 55° 초과 시 그림자 기반 크레이터 검출 실패. 2–5°에서 모든 매처 저하 | 렌더 134,784장, MARTIAN | [Maass 2016](https://elib.dlr.de/104630/1/art_10.1007_s12567-016-0129-1%20Maass.pdf); [Pisanti 2025](https://arxiv.org/abs/2502.09795) |
| 프레임률·회전 | 1 Hz 미만 급락. 자이로 없으면 23 km 아래 프레임 0–3%만 처리 | PANGU, 풍선 비행 | [Cowan 2025](https://arxiv.org/abs/2511.17720); [Maggio 2023](https://arxiv.org/abs/2302.08011) |
| 천저 이탈각 | 85°에서만 실패. ≤15° 학습 모델이 45°에서도 동작 | 렌더 | [Maass 2016](https://elib.dlr.de/104630/1/art_10.1007_s12567-016-0129-1%20Maass.pdf); [Georgakis & Ansar 2024](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf) |
| 설계자가 고른 운용 태양 창 | LVS 25–55°(2015 시험 기준 30–50°), autoNGC 20–40°, 남극 CNav 국소 태양 고도 <40°, 크레이터 항법 최적 10–20°(Hamel) | 설계 | [LSIC](https://lsic.jhuapl.edu/uploadedDocs/presentations/764-1.09_Johnson_20210302_LuNaMaps_LVS_v0.pdf); [autoNGC](https://ntrs.nasa.gov/api/citations/20210024481/downloads/autoNGC_SciTech2022_Shoemaker_final.pdf) |
| 지도–하강 조명 차 (운용 시스템) | LVS 태양각 차 ≤20°. LION 고도 차 >10°·방위 차 >20°에서 악화. Johnson 2025 조명 지표 ≤0.3 | 운용 범위, 몬테카를로 | [Johnson 2022](https://doi.org/10.2514/6.2022-1214); [Delaune 2016](https://doi.org/10.1016/j.robot.2016.01.007); [Johnson 2025](https://doi.org/10.2514/6.2025-2076) |

공개된 정량 민감도 자료를 낸 곳은 JPL(LVS, ELViS, Geo-LoFTR, LuNaMaps), DLR(CNav, TRON), Draper/SPLICE뿐이다. ESA/PANGU 산업 캠페인, SLIM 지상 시험, IM, Astrobotic의 정량 조명·고도 스윕은 찾지 못했다.

### 플룸: 정량 연결은 아직 비어 있다

플룸을 TRN 가용성에 정량적으로 이은 연구는 없다. Johnson 2017은 대비 스윕 뒤에 "Future work will link contrast reduction to optical opacity"라고 남겼고, 후속 연구는 찾지 못했다 ([Johnson et al. 2017](https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/46186)).

[추론] 대비 32 DN → 4 DN까지 동작했다는 그 스윕이 플룸 소광을 TRN 가용성에 잇는 가장 직접적인 다리다. 저고도 열화 요인 가운데 플룸은 Σ∇와 σ_η를 바꾸는 한 축으로 넣으면 충분하다.

---

## 남은 공백은 "고도·조건별 정답 고정 확률" 곡면 하나로 모인다 (질문 4)

### 공백 목록

흩어진 공백들은 결국 하나의 산출물이 없다는 사실로 수렴한다. 문헌에 없는 것은 다음과 같다.

1. 달·화성 하강에서 정답 고정 확률이나 오고정률을 고도의 함수로 그린 곡선
2. 행성 지도 기반 위치추정이나 크레이터 매칭에 대한 대오차 한계(ZZB 등)
3. 엔트로피·PSR·inlier 비·NIS 같은 적합성·온라인 지표가 실제 실패를 얼마나 맞히는지 보여 주는 ROC
   - 예외는 하나다. 화성 LVS의 inlier ≥20 게이트에 대한 작동점(오탐률 2×10⁻⁸, 검출률 83%; Johnson et al. 2025)이다.
   - 달, 고도별, 다른 지표에 대한 ROC는 없다.
   - 고전 적합성 지표(엔트로피, 분산, 에지 밀도)의 정밀도가 0.35–0.42에 그친다는 SAR–광학 결과가 있다(Yang et al. 2026).
4. 같은 남극 DEM 위에서 조명·고도·지형을 함께 쓸고 절대 TRN의 성공률 곡면을 보고한 연구. 상호작용도 미지다. 예컨대 그림자가 분해되는 저고도에서 방위각 허용치가 좁아지는가에 대한 자료가 없다 ([Georgakis & Ansar 2024](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf)와 [Xie 2025](https://doi.org/10.3390/rs17132302) 모두 단일 고도).
5. 태양 고도 1° 아래에서 지도를 맞춘 절대 TRN 연구. 상대 광학흐름만 0.4°와 1.35°에서 다뤄졌다 ([Cowan et al. 2025](https://arxiv.org/abs/2511.17720)).
6. 0–3.5 km 남극 하강 영상 시퀀스와 자세 진실값을 갖춘 공개 벤치마크. 가장 낮은 StereoLunar도 3.5 km부터다 ([StereoLunar](https://arxiv.org/abs/2510.18172)).
7. 행성 TRN에 적용한 보호수준·무결성 틀
8. de Gerlache를 덮는 1 km 미만 크레이터 카탈로그, 그리고 하강 영상에서 픽셀 지름별 검출 확률 곡선
9. 플룸 광학두께를 TRN 가용성에 잇는 모형 (부차)

### 제안하는 정식화

[추론] 이 연구가 낼 수 있는 핵심 산출물은 두 함수다.

- 가용성 A(h) = P(고정이 나오고 검증됨 ∣ h)
- 무결성 I(h) = 1 − P(오고정이 수용됨 ∣ h)

조건 변수는 지형 부류(Σ∇, 크레이터 밀도), 태양 고도·방위, 지도–하강 조명 잔차, 지도 GSD와 최소 수록 크레이터, 그리고 부차 요인인 플룸 광학두께다. 이미 발표된 세 도구가 각 부분을 채운다.

| 도구 | 채우는 것 |
|---|---|
| Setterfield E[L_δ](h) | 소오차 영역의 정확도 |
| (픽셀 수 × Σ∇ / σ²)에 대한 EZZB식 문턱 | 대오차가 시작되는 지점 |
| 위의 크레이터 개수 모형 N_vis(h) 대 N_min | 크레이터 TRN의 가용성 |

이를 렌더 하강 몬테카를로로 검증하되, 각 고정을 **정답 / 고정 없음 / 오고정** 셋 중 하나로 채점한다(DSMAC P_cu 방식). 오고정률은 반드시 신뢰구간과 함께 보고한다. N회에서 0건이면 95% 상한이 ≈3/N이다.

### 검증할 가설

[추론] 검증할 가설은 다음과 같다.

- **상관 TRN**: GSD 일치 고도 근처에 넓은 평탄부가 있고, 그 아래에서 h²에 따라 떨어진다.
- **크레이터 TRN**: 고도와 무관한 평탄부가 있고, 지도 한계 교차 고도 아래에서 이차로 붕괴한다.
- **공통**: 둘 다 조명 인자(극지 저태양 그림자)가 곱해진다. 가장 낮은 고도에서만 플룸 인자가 곱해진다.

### de Gerlache 연구에 대한 구체적 예측

[산술] 카메라 GSD = 2h·tan(FOV/2)/1024를 써서, 실측 종료 조건 두 가지를 sim2moon에 옮겼다.
- 종료 조건 A: 지도 GSD ÷ 카메라 GSD = r에서 끝난다.
  - r ≈ 6: LVS형 넓은 카메라. Xombie 85 m @1 m/px에 90° 카메라를 가정했고, Mars 2020에서는 500 m에서도 동작 중이었다.
  - r ≈ 20–24: New Shepard의 12.2° 카메라
- 종료 조건 B: 경험칙 "종료 고도 ≈ 200–250 × 지도 GSD"를 카메라와 무관하게 적용한다. 이전 판의 방식이며, 카메라 특이적이라 **신뢰도가 낮다**.

sim2moon의 현재 하향 카메라는 수평 시야각 60°다 ([sim2moon 카메라 설정](/home/sim2real/sim2moon/config/environment/chandrayaan3_paired_dataset.yaml)). 1024 px로 올린다고 가정한다.

| 지도 GSD | GSD 일치 고도 (60°) | 조건 A, r = 6 | 조건 A, r = 20–24 | 조건 B (참고) |
|---|---|---|---|---|
| 5 m/px (명목) | ~4.4 km | ~740 m | ~185–220 m | 1.0–1.25 km |
| 7 m/px (sim2moon 격자) | ~6.2 km | ~1.0 km | ~260–310 m | 1.4–1.75 km |
| 20 m/px (실효 하한) | ~17.7 km | ~3.0 km | ~740–890 m | 4–5 km |
| 40 m/px (실효 상한) | ~35 km | ~5.9 km | ~1.5–1.8 km | 8–10 km |

**읽는 법**
- 명목 지도(5–7 m/px)라면 절대 갱신은 **~0.2–1 km**에서 끝날 것으로 예상된다.
- 지도의 실효 해상도가 20–40 m라면 ~0.7–6 km까지 올라간다.
- 범위가 넓은 이유는 r이 카메라·매처 설정에 따라 4배 가까이 퍼져 있기 때문이다. **r 자체가 이 연구에서 측정할 첫 번째 양**(RQ1)이다.
- 고도 격자는 ~10 km에서 ~50 m까지 로그 간격으로 잡고, 지도 GSD(1/2/5/10/20/40 m)와 카메라 시야각(40/60/90°)을 독립변수로 둔다.

크레이터 카탈로그의 최소 크기도 독립변수로 쓸어야 한다.
- 시야각 45° 천저 카메라의 시야 폭은 ≈0.83 × 고도다 [산술].
- Robbins 카탈로그(1–2 km 이상)는 ~5–10 km 위에서만 쓸모가 있다.
  - Woicke & Krüger 2023의 남극 시험에서도 전역 카탈로그만으로는 ~4 km가 끝이었다 [원문].
  - Hamel et al. 2025는 공개 DB로는 20 km 아래에서 신뢰할 수 없다고 했다 [원문].
- AI 기반 LU5M812TGT(0.4 km 이상)는 수 km까지 내려온다.
- 1 km 아래에서는 수십 m 크레이터가 필요하다. 그런 카탈로그는 없으므로 SfS DEM이나 NAC 모자이크에서 직접 만들어야 한다 ([Robbins 2018](https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018); [LU5M812TGT](https://zenodo.org/records/13990480)).
  - NAC 자동 검출의 조명별 정밀도는 0.68–0.90이다(Fairweather et al. 2022). 자체 카탈로그에도 10–30%의 거짓 크레이터가 섞일 수 있다는 뜻이며, 이것이 오고정 원인이 된다.

### 국내 선행연구 (RISS·해외 DB, 2026-10-06 검색, 2026-10-10 추가)

**영상 TRN의 신뢰도를 고도·조명의 함수로 정량화한 국내 연구는 찾지 못했다.** 2026-10-10에 원문을 읽은 국내 리뷰(Rahuman & Lee 2026)도 이를 정량화하지 않는다. 가까운 연구는 다음과 같다. 표시가 없는 행은 [서지만 확인, 본문 미열람]이다.

| 연구 | 내용 | 이 연구와의 관계 |
|---|---|---|
| 구평모·박영범·박찬국 (2016) 「영역기반 지형 험준도 지수를 이용한 달착륙선의 일괄처리방식 지형상대항법 성능분석」, 한국항공우주학회지 44(7) | 레이더 고도계–DEM TRN의 지형 지수–성능 관계 | 지형 적합성 지표의 국내 선례 |
| 박영범·박찬국·권재욱·류동영 (2017) 「임무조건에 따른 달 착륙선 항법시스템 요구성능 분석」, 한국항공우주학회지 45(9) | 임무 조건별 항법 요구성능 | 착지 오차 예산과 A(h)의 연결 |
| 고형우·전하민·김효중·허재석·유창경 (2024) 「카메라 시야각에 따른 크레이터 기반 영상항법 성능 분석」, KSAS 2024 | 시야각별 크레이터 항법 성능 | 시야각 독립변수(RQ1·RQ2)의 국내 선례 |
| 최재훈·김효중·유창경 (2025) 「크레이터 영상항법의 고위도 방위각 추정 오차 보정」, KSAS 2025 | 고위도 크레이터 항법의 오차 | 남극 조건 |
| 황연하·박봉균·탁민제 (2012) 「달 착륙선의 정밀 착륙을 위한 DSMAC방법 기반의 지형 상대 항법 연구」, KSAS 2012 | DSMAC형 장면 매칭 TRN | DSMAC P_cu 틀의 국내 적용 |
| 김재인 (2022) 「화성 지형상대항법을 위한 하강 데이터셋 생성과 랜드마크 추출 방법」, 대한원격탐사학회지 38(6) | 하강 데이터셋 생성 | 렌더 데이터셋 구성 |
| 박우상·정유연·나윤주·방효충 (2016) KSAS; Park, Jung, Bang (2016) ICCAS, doi:10.1109/ICCAS.2016.7832378 (KAIST) | 크레이터 탐지·매칭 지상검증 플랫폼, 크레이터 매칭 TRN | 크레이터 기준선 |
| 김기덕·정유연·방효충 외 (2017) KSAS 두 편 | 동력하강 전 크레이터 + 관성 영상항법, 영상 잡음 아래 크레이터 매칭 특성 | 잡음 민감도 |
| 류동영 외 (2022) 「달 착륙 항법 종합시험장치를 이용한 지상비행 시험」, KSAS 2022; 문귀원·이훈희·류동영 (2020) | 항우연 달 착륙 항법 시험 | 국내 시험 기반 |
| Choe & Park (2024) IEEE TAES 60(3) (서울대); Kim, Park & Bang (2025) IJASS (KAIST) | LiDAR-관성 희소 DEM 절대위치, 레이더 고도계 TRN 오차 추정 | Setterfield 2021·Johnson 2023 피인용 |
| Rahuman & Lee (2026) "A Review of Vision-Based Terrain Relative Navigation for Planetary Descent and Landing of Space Vehicles", IJASS 27:3482–3501 (경북대) [원문] | 비전 TRN을 템플릿 매칭·패턴 매칭·딥러닝 통합으로 나눈 리뷰. 유사도 지표 표, 임무별 방법 표, 탑재 프로세서 비교 | 이 보고서 절대 TRN 분류(질문 5)의 1차 근거. 고도별 신뢰도는 정량화하지 않고 저고도 GSD 차이를 미해결 과제로 꼽는다 |
| Kim, Nam, Lee & Oh (2025) "Verification of Vision-Based Terrain-Referenced Navigation Using the Iterative Closest Point Algorithm Through Flight Testing", *Sensors* 25:5813 (국방과학연구소) [요약 확인] | Cessna 비행시험. 연속 영상 SGM 스테레오 DEM을 DTED4(5 m)와 ICP로 정합. 3–5 km에서 장축 σ 4.3–5.1 m | 수동 고도 템플릿(T-E) 계열의 유일한 비행 근거. 탐색적 비교군의 기준선 |

### 가장 큰 타당성 위협

시뮬레이터의 진실 지형과 지도가 같은 DEM에서 나온다는 점이다.

- **원천 DEM**: sim2moon의 de Gerlache 지형은 PGDA 78의 `SL2_final_adj_5mpp_surf.tif`를 6.61 m 격자, 3025 × 3025(~20 km)로 다시 표본화한 것이다. 타일 안 고도 범위는 −1,859 m에서 +1,765 m다 ([sim2moon 지형 계약](/home/sim2real/sim2moon/config/terrain/de_gerlache_rim2_clean.json)).
- **보간 비율**: 이 원천 LDEM의 5 m 픽셀 ~90%는 LOLA 궤적 사이를 보간한 값이다 ([PGDA 78](https://pgda.gsfc.nasa.gov/products/78)).
- **실효 해상도**: LuNaMaps는 "5 m/pix" 극지 LOLA DEM의 실제 해상도를 20–40 m/pix로 본다 ([LuNaMaps TT Plan](https://ntrs.nasa.gov/api/citations/20240012328/downloads/LuNaMaps_TechTransfer.pdf)).

[추론] 결과는 두 가지다.

1. **지도와 진실이 같으면** 매칭이 비현실적으로 좋게 나온다(inverse crime).
2. **저고도 영상의 내용**은 실제 지형이 아니라 6.6 m 면의 셰이딩과 재질 질감이 결정한다. 예컨대 시야각 60°에서 고도 100 m 영상 폭은 ~115 m로, 격자 꼭짓점 17개 정도가 걸친다 [산술].

경험칙을 실효 해상도 20–40 m에 적용하면 하한은 4–10 km까지 올라갈 수 있다 [산술]. 이 위협은 세 가지로 줄일 수 있다.

- 진실 지형은 PGDA 104의 SfS DEM이나 NAC 질감 표면으로 그린다.
- 탑재 지도는 PGDA 78의 clone DEM 가운데 하나로 따로 만든다.
- 렌더 광도는 NAC 태양경도별 모자이크와 비교해 검증한다.

### 실험 설계

[추론] 세 연구의 설계를 합치면 어느 논문도 내지 못한 성공률 곡면이 나온다.

- Georgakis & Ansar의 지도–쿼리 태양 고도·방위 요인 격자
- Maass의 태양 고도 × 카메라 고도각 격자와 "N개 이상 카탈로그 랜드마크" 성공 기준
- Maggio의 고도대별 처리 프레임 비율

**독립변수**

- 고도
- 실제 de Gerlache 하강 태양 고도(대략 0–3°)
- 지도 렌더 태양의 잔차: Δ방위 0/15/30/45/60/90/135/180°, Δ고도 0/±1/±2°
- 천저 이탈각 0–45°
- 프레임률과 각속도
- 초기 위치 오차 0.1–5 km
- 프레임별 지형 공변량: Σ∇, 엔트로피, 그림자 비율

**종속변수**

- TRN 자체 게이트 통과 여부와 inlier 수
- Acc@X m, RA-1/RA-5
- 마지막 유효 갱신 고도
- 고도대별 처리 프레임 비율
- 온라인 신뢰도 스택 각 층의 원값. 이로부터 고도대별 ROC를 계산한다.

Isaac RTX 결과는 최소 한 개의 외부 렌더러와 교차 확인해야 한다. 학습 매처 연구들은 학습 조명 분포에 대한 sim-to-real·분포 밖 민감도가 강하다는 것을 보여 준다.

---

## 절대 TRN은 템플릿 매칭과 패턴 매칭 두 갈래이고, 갈래마다 공개 기준선과 남극 지도 사다리가 있다 (질문 5)

### 절대 TRN 분류 기준 (2026-10-10 확정)

최신 리뷰인 Rahuman & Lee 2026(IJASS, 경북대)은 SLIM 팀의 Ishii et al. 2017을 따라 절대 TRN을 두 갈래로 나눈다. 기준은 "Based on how the visual matches the reference map"이다 [원문].

- **템플릿 매칭**: "Direct correlation of image patches with the reference map using intensity or similarity metrics"
- **패턴 매칭**: "Identification of geometric landmarks (e.g., craters) and matching them with pre-mapped patterns"

리뷰는 여기에 딥러닝 통합을 셋째 갈래로 더한다. 그러나 리뷰 자신의 Table 4 "Pipeline Integration" 열은 모든 학습 방법을 두 파이프라인 중 하나의 부품(전단 검출기, 전처리, 기술자, 유사도)으로 적는다. 그래서 이 보고서는 학습 여부를 계열이 아니라 구현 속성으로 둔다 [추론].

리뷰 Table 2는 OSIRIS-REx NFT와 SPLICE를 패턴 매칭으로 분류한다. 그러나 둘 다 랜드마크 DB를 쓸 뿐, 정합 연산은 렌더하거나 와핑한 패치의 상관이다(IBAL: Maggio et al. 2023). 리뷰가 "랜드마크를 쓰면 패턴"이라는 기준을 섞은 것으로 보아, 이 보고서는 판정을 비교 연산 하나로 고정한다 [추론].

- **1차 기준 (계열을 정한다)**: 정합 전에 관측과 지도 **양쪽에서** 기본 요소(크레이터, 키포인트, 능선, 그림자 영역)를 추출해 그 기술자나 불변량끼리 비교하는가?
  - 아니오 → **템플릿 매칭**. 원시 신호 패치를 유사도로 직접 비교한다. LVS 정밀 단계는 Harris로 템플릿 *위치*만 고르므로 여기에 든다.
  - 예 → **패턴 매칭**. 요소 사이의 대응 목록을 만들고 기하 검증으로 거른다.
- **2차 기준 (하위 계열을 정한다)**: 관측 신호가 밝기(수동 영상)인가, 고도(LiDAR·고도계, 또는 영상에서 복원한 DEM)인가?
  - 리뷰도 "Template matching incorporates contour maps, optical images, DEMs"라고 쓰고 TERCOM을 템플릿 매칭에 넣는다 [원문].
- **계열을 나누지 않는 속성** — 실험 변수나 기록 항목으로 쓴다.
  - 정합 구조: Image-on-Map / Map-on-Image (리뷰 Fig. 2)
  - 기준 영상 생성: 정사 사진 / 태양 맞춘 렌더 정사 / 실시간 시점 렌더
  - 사전 정보: lost-in-space / 추적형
  - 유사도·기술자: NCC, POC, MI, VCC / SIFT, 학습 기술자, 사영 불변량
  - 구현: 수작업 / 학습
  - 계층: 단일 / coarse→fine

| 코드 | 계열 | 지도 | 비행·시험 사례 | 하한을 정할 축 (가설) | 조명 의존 |
|---|---|---|---|---|---|
| **T-I** | 밝기 템플릿 매칭 | 정사영상(사진·렌더) 또는 DEM+알베도 | Mars 2020 LVS, SPLICE/IBAL, GIANT/Retina, Blue Ghost(Space-ng), OSIRIS-REx NFT, DSMAC | 지도/카메라 GSD 비, 질감 엔트로피 | 크다. 지도–하강 태양 잔차에 민감 |
| **T-E** | 고도 템플릿 매칭 | DEM | 능동: ALHAT LiDAR DEM 상관(지구 현장시험). 수동: 국방과학연구소 항공기 시험(SGM 스테레오 DEM + ICP) | DEM 실효 해상도, 기복 대비 측정·복원 잡음 | 능동은 없다. 수동은 복원 단계에만 있다 |
| **P-G** | 기하 랜드마크(크레이터) 패턴 매칭 | 크레이터 목록 | SLIM(비행, 500 m까지), DLR CNav, LunaNet | 카탈로그 최소 크기, 시야 안 크레이터 수 | 중간. 검출 단계에서 그림자 영향 |
| **P-R** | 밝기 특징 패턴 매칭 | 키포인트·기술자 DB | 연구 단계. Chang'e-6 하강 영상 재처리(BOFDAFT) | 기술자의 조명·축척 불변성 | 크다. SIFT는 60°, 기성 LoFTR은 90°에서 무너진다 |

지도 없는 상대 항법(VO/VIO, IM-1, DIMES)은 범위 밖이다. 리뷰 Table 2에서는 "[Mapless]"로 분류된다. 이들은 절대 TRN 비교의 계열이 아니라, 절대 갱신이 끊긴 뒤 구간의 대조군이다.

**리뷰가 이 연구에 주는 정성 근거** [원문]
- **고도**: 결론에서 "TRN is generally more reliable at higher altitudes"라고 쓰고, 저고도의 "challenges related to ground-sampling differences"를 미해결 과제로 꼽는다. 고도별 수치는 없고 운용 고도 표도 없다.
- **크레이터**: LunaNet의 한계로 크레이터 "visibility decreases significantly at lower altitudes"를 든다.
- **지형별 적합성**: "Template Matching is the optimal choice for texture-deficient or flat terrains"; "Pattern Matching excels in environments rich in distinct, stable landmarks"
  - 정량 근거가 없고 LVS도 엔트로피 ≥4를 요구한다. 그래서 아래 권장 조합에서 가설 H-a·H-b로 시험한다.

**주의 — 리뷰의 몇 서술은 1차 원문과 다르다.**
- Table 3의 "NCC: + Robust to lighting"은 밝기 이득·오프셋에 대한 강건성이다. 태양 기하가 바뀌는 경우에는 맞지 않는다(LVS의 ΔSun ≤20° 운용 제한).
- SLIM이 "4 km and 500 m"에서 "< 1 m"를 냈다는 서술은 원문상 500 m(VLD2)에만 해당한다. VLD1(4 km)은 수 m–10 m였다.
- 상세 대조는 [`absolute_trn_taxonomy.md`](../research_notes/비전%20TRN%20신뢰%20한계와%20알고리즘%20데이터셋/absolute_trn_taxonomy.md)에 있다.

### 기준 알고리즘

LVS/ELViS, SPLICE, Draper IBAL, Astrobotic OPAL, Firefly/Space-ng 가운데 공개 코드가 있는 시스템은 없다. 이들은 모두 T-I 계열이고 같은 처방을 쓴다.

1. 사전 자세로 보일 지도 패치를 예측한다.
2. 패치를 렌더하거나 와핑한다.
3. FFT나 NCC로 찾는다.
4. P3P/PnP-RANSAC으로 이상치를 거른다.
5. 랜드마크 픽셀이나 자세를 EKF에 넣는다.

NASA TRN 교육 자료도 탑재 지도를 "typically an image rendered under the expected day-of-flight lighting conditions"로 정의한다 ([Restrepo & Johnson, TRN Basics](https://ntrs.nasa.gov/api/citations/20210010726/downloads/TRN_Basics_LSICWorkshop_March2%20%28002%29.pptx.pdf)). 아래 표는 2026-10-06 기준 저장소 메타데이터로 확인한 절대 TRN 기준선이다. 2026-10-10에 계열 코드로 다시 묶고 리뷰·교차 확인 자료의 다섯 행을 더했다.

| 계열 · 속성 | 기준선 | 코드·라이선스 | 기록할 건전성 신호 | 출처 |
|---|---|---|---|---|
| T-I · Image-on-Map, 정사 지도 | LVS형 2단 MRL 재구현: coarse FFT/위상상관(큰 패치 몇 개) → fine NCC(작은 패치 ~150개) → P3P/PnP-RANSAC → EKF. **원문 공개 매개변수** [원문]: 영상 64×64 부분창 N개(coarse 64, fine 150), 5×5 Harris, coarse 템플릿 148→128×128을 1024×1024 지도에 FFT, 21×21 국소 정규화, fine 21×21 템플릿을 128×128 탐색 영역에서 래스터 탐색, 3×3 이차곡면 부화소, Mahalanobis γ = q3 + 1.5 IQR(≈6.07), 갱신 조건 inlier ≥20, NCC 성공 임계 0.35(Wang 2026이 Mars 2020 값으로 인용) | 비공개. 공개 논문 수치로 재구현 | 시도·매칭 landmark 수, inlier 수·비, NCC 피크 높이·폭, `phaseCorrelate` `response`, 재투영 RMS, 카메라/지도 GSD 비 | [Johnson et al. 2023](https://doi.org/10.2514/1.G006780); [LSIC](https://lsic.jhuapl.edu/uploadedDocs/presentations/764-1.09_Johnson_20210302_LuNaMaps_LVS_v0.pdf) |
| T-I · Map-on-Image, 실시간 렌더 템플릿 | NASA GIANT Surface Feature Navigation: DEM maplet에서 템플릿을 렌더 → 2D NCC → 선택적 PnP | **Apache-2.0**, Python, v2.0.0 (2025-09) | `min_corr_score`, `gsd_scaling`(허용 GSD 비), 입사·반사각 한계, `percent_in_fov`, PnP-RANSAC | [nasa/giant](https://github.com/nasa/giant); [sfn_class.py](https://github.com/nasa/giant/blob/main/giant/relative_opnav/estimators/sfn/sfn_class.py) |
| T-I · Map-on-Image, 템플릿 DB | Draper IBAL 재구현: 템플릿 DB → homography 와핑 → 상관 → P3P-RANSAC | 비공개. 파이프라인은 arXiv에 공개 | inlier 8개 미만 거부, 450 m 고도 점프 거부, 랜드마크 각도 필터 | [Maggio et al. 2023](https://arxiv.org/abs/2302.08011) |
| T-I · 극지, 실시간 렌더 + masked NCC | Wang et al. 2026 2단 오프라인–온라인 프레임워크: 고도대별 랜드마크 DB(6/2/1 m/px), 실시간 렌더 + masked NCC | 미확인 (CC BY 논문) | NCC 피크 >0.35, 렌더–기준 >0.6 | [doi:10.1016/j.cja.2026.104149](https://doi.org/10.1016/j.cja.2026.104149) |
| T-I · 실시간 시점 렌더 | 렌더-대조 (Blue Ghost/Space-ng 개념): EKF 사전 자세에서 기대 영상을 실시간 렌더 | 독점. 개념만 공개 | 렌더–실영상 상관 | [TechTimes](https://www.techtimes.com/articles/319197/20260627/firefly-aerospace-brings-moon-landing-ai-navigation-house-after-blue-ghost-success.htm) |
| T-I · 유사도 지표 대안 | 리뷰 Table 3의 NCC 대안: POC(위상상관, 날카로운 피크), MI(비선형 광도·다중모달), VCC(기울기 부호화 상관, Hayabusa2 리허설 데이터에서 NCC보다 작은 오차, Ohira et al. 2022), ZNCC. 같은 T-I 틀에서 지표만 바꾸는 속성 실험용 | POC는 OpenCV `phaseCorrelate`, 나머지는 자체 구현 | 피크 높이, PSR, 부피크/주피크 비 | [Rahuman & Lee 2026](https://doi.org/10.1007/s42405-026-01139-1); [Ohira et al. 2022](https://doi.org/10.1016/j.actaastro.2020.10.013) |
| T-E · 수동 (탐색적) | 영상 복원 DEM + 기준 DEM 정합. 국방과학연구소 항공기 비행시험: 연속 영상 SGM 스테레오(기선 ≥120 m) → DTED Level 4(5 m)와 회전 불변·평행이동 ICP → 위치 갱신으로 EKF 입력. 3/4/5 km에서 장축 σ 4.28/4.66/5.10 m [요약 확인]. 달 착륙 비행 이력 없음 | 비공개. OpenCV `StereoSGBM` + Open3D ICP로 재구현 가능 | ICP 잔차 RMS·수렴, 지형 기복 지수, 복원 유효 픽셀 비 | [Kim et al. 2025](https://doi.org/10.3390/s25185813) |
| T-E · 능동 (지표 참고) | ALHAT flash LiDAR DEM을 5 m 기준 DEM과 정규화 상관 + 이차곡면 부화소. "under any lighting conditions", 위치 오차 <90 m, 초기 오차 1.6 km까지 정합 성능 저하 없음 [원문]. 비전 연구에서는 신뢰 지표 정의의 참고로만 쓴다 | 비공개 | 상관 피크 높이·폭·비, Peak-to-Valley, 지형 기복 지수(TRI) | [Johnson & Ivanov 2011](https://www-robotics.jpl.nasa.gov/media/documents/FT3_LidarTRN_Paper_v12.pdf) |
| P-G · lost-in-space 식별 | Christian 투영 불변량 lost-in-space 식별 (제3자 구현, Chang'e-5 착륙 영상으로 시연) | **GPL-3.0**, 커밋 1개 (2025-01) | 삼중항 수용, PnP 재투영 지표 | [vas-byte/Crater-Identification](https://github.com/vas-byte/Crater-Identification) |
| P-G · lost-in-space 대안 | 별 식별을 본뜬 크레이터 삼각형(지름 비, 최소·최대 내각, 방향)과 USGS 카탈로그 8,673개, 오식별 확률 검사로 거부 (Hanak & Crain 2010). 크레이터 5–6개 미만이면 유일 식별이 어렵다 | 논문 기반 자체 구현. GPL 회피용 | 식별된 삼각형 수, 오식별 확률 | [NTRS 20090032647](https://ntrs.nasa.gov/citations/20090032647) |
| P-G · 학습 검출기 | Ellipse R-CNN 테두리 검출 + 패턴 매칭 (500 km 이하 궤도) | **MIT**. 데이터 생성에 Airbus SurRender 필요 | 검출·매칭 수, 타원 적합 잔차 | [wdoppenberg/crater-detection](https://github.com/wdoppenberg/crater-detection) |
| P-G · 카탈로그 생성 | DeepMoon: DEM에서 크레이터 검출 (사람 카탈로그 92% 재현) | 코드 **MIT**, 데이터 CC BY-SA 4.0 | 자체 DEM으로 1 km 미만 카탈로그를 만드는 용도 | [silburt/DeepMoon](https://github.com/silburt/DeepMoon) |
| P-G · 참고만 | FederNet (Mask R-CNN + 불변량 + EKF), STELLA CBPE | **라이선스 파일 없음** → 재배포 불가, 참고만 | — | [FederNet](https://github.com/sirbastiano/FederNet); [CBPE](https://github.com/s-mcleod/crater_based_navigation_pipeline) |
| P-G · 그림자 형상 [추론] | 그림자를 예측 가능한 랜드마크로 쓰는 절대 위치추정 (Kaufmann et al. 2015; Kuga & Kojima 2018) | 미확인 | — | [doi:10.1109/AERO.2015.7119045](https://doi.org/10.1109/AERO.2015.7119045) |
| P-R · 실영상 검증 | BOFDAFT (Liu et al. 2026, PSS 106221): Chang'e-6 하강 영상–DOM 정합에 성공한 유일한 두 방법 중 하나. 나머지(HAPCG, RIFT, SIFT, SuperPoint+SuperGlue, PSO-SIFT)는 실패 | 미확인 | 재투영 오차(평균 1.77 px) | [doi:10.1016/j.pss.2025.106221](https://doi.org/10.1016/j.pss.2025.106221) |
| P-R · 계층 검색 | VLAD 전역 기술자로 지도 타일을 검색(coarse)한 뒤 SIFT로 정합(fine). 검색 타일 수를 늘리면 정답 비율이 오른다 (Li Sanchez 2024). Mars 2020의 coarse→fine 구조를 특징 기반으로 옮긴 형태 | 석사논문 공개, 코드 미확인 | 검색 순위, 정합 inlier 수 | [Li Sanchez 2024](https://d-scholarship.pitt.edu/41874/) |
| P-R · 학습 | LightGlue, LoFTR, DISK 등 (Kornia가 LoFTR·LightGlue·DISK·DeDoDe·ALIKED·XFeat·AdaLAM을 묶음) | **Apache-2.0** (코드·LightGlue 가중치·DISK). ALIKED는 BSD-3. **SuperPoint 가중치·추론 파일은 제한 라이선스** → ALIKED/DISK와 조합 | `matching_scores0`(LightGlue 기본 `filter_threshold` 0.1), `mconf`, 신뢰도 분위, RANSAC inlier 비 | [LightGlue](https://github.com/cvg/LightGlue/blob/main/lightglue/lightglue.py); [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/src/loftr/utils/coarse_matching.py); [Kornia](https://github.com/kornia/kornia/tree/main/kornia/feature) |
| P-R · 학습 (피함) | SuperGlue | 학술·비영리·비상업 전용 → 피함 | — | [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py) |
| T-I·P-R 전처리 · 학습 영상 변환 | 하위 NCC/SIFT 정합에 최적화한 학습 영상 변환 | MIT | 변환 전후 상관 | [Fragoso et al. 2021](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1126/scirobotics.abf3320?fields=title,abstract); [GitHub](https://github.com/connorlee77/seasonally-invariant-image-transform) |
| 미분류 · 조명 강건 (방법 원문 미확인) | Tong et al. 2025, "Illumination Robust Landing Point Visual Localization for Lunar Lander With High-Resolution Map Generation" (IEEE JSTARS 18:1577–1591) — 저해상도 지도와 하강–지도 조명 변화를 직접 다룸 | 미확인 | — | [doi:10.1109/JSTARS.2024.3498351](https://doi.org/10.1109/JSTARS.2024.3498351) |

**상대 항법 대조군 (절대 TRN 계열 아님).** 절대 갱신이 끊긴 뒤 구간을 메우는 소스다. 계열 비교에는 넣지 않고, 하한 아래의 착지 오차 예산(질문 1의 관성 구간 산술)과 전환 실험에 쓴다.

| 종류 | 기준선 | 코드·라이선스 | 기록할 건전성 신호 | 출처 |
|---|---|---|---|---|
| VIO | OpenVINS (MSCKF) | **GPL-3.0** | 갱신별 χ² 거부(`chi2_multipler` × 95% 분위), 트랙 수, 자체 공분산 | [UpdaterMSCKF.cpp](https://github.com/rpng/open_vins/blob/master/ov_msckf/src/update/UpdaterMSCKF.cpp) |
| VIO | JPL x / xVIO (range-VIO, 레이저 고도계 융합) | Apache-2.0. 2022년 이후 갱신 없음 | — | [jpl-x/x](https://github.com/jpl-x/x) |
| VIO | Kimera-VIO / Basalt / ORB-SLAM3 / VINS-Mono | BSD-2 / BSD-3 / GPL-3.0 / GPL-3.0 | `TrackingStatus`(FEW_MATCHES 등), ORB-SLAM3 inlier 15·50 규칙, VINS 재설정 조건. 판정이 아니라 원값을 기록 | [Kimera](https://github.com/MIT-SPARK/Kimera-VIO/blob/master/include/kimera-vio/frontend/Tracker-definitions.h); [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Tracking.cc); [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/vins_estimator/src/estimator.cpp) |
| VO (비행 게이트 공개) | Nova-C(IM) DPOS VO: 최소 매치 10, 목표 겹침 50% (Molina et al. 2022). Mars Helicopter MAVeN: RANSAC 뒤 추적 특징 m = 40 미만이거나 빈 타일 k = 3 초과이면 새 기준 프레임, 트랙 10프레임, 경사 1–3° 평탄 가정 (Bayard et al. 2019) | 논문 기반 자체 구현 | 매치 수, 빈 타일 수, 트랙 길이 [원문] | [Molina et al.](https://doi.org/10.1007/978-3-031-51928-4_58); [Bayard et al. 2019](https://doi.org/10.2514/6.2019-1411) |
| VO | DIMES/Cowan형 흐름 + 고도계 → 속도 | 논문 기반 자체 구현 (수백 줄) | 상관 피크 폭, 흐름 잔차 RMS, 영상쌍 사이 속도 일관성 | [Johnson et al. 2007](https://robotics.jpl.nasa.gov/media/documents/DIMES_IJCV_20070117.pdf); [Cowan et al. 2025](https://arxiv.org/abs/2511.17720) |

**권장 조합** [추론]. 목적은 계열 비교다. 그래서 핵심 3계열에 기준선을 하나씩 두고, T-I에만 독립 구현을 하나 더 둔다.

- **T-I**
  - LVS형 2단 MRL: Image-on-Map, 태양 맞춘 렌더 정사
  - GIANT SFN: Map-on-Image, 실시간 렌더 템플릿
  - 두 구현의 차이가 곧 속성 실험(정합 구조·기준 생성)의 한 축이다. 유사도 지표(NCC/POC/MI/VCC)도 같은 틀에서 바꾼다.
- **P-G**: 자체 DEM 카탈로그 + 추적형 예측-매칭-갱신(CNav TM형). 초기화는 Christian 불변량(GPL, 별도 프로세스)이나 Hanak & Crain 삼각형으로 한다.
- **P-R**: 고전 SIFT/ORB + RANSAC과 학습 LightGlue(ALIKED/DISK)·LoFTR. 후단의 PnP-RANSAC은 T-I와 같은 것을 쓴다.
- **T-E (탐색적)**: 스테레오/SfM 복원 DEM + DEM 상관 또는 평행이동 ICP. 지구 항공기 시험(Kim et al. 2025)은 있지만 달 비행 이력은 없다. StereoLunar의 정답 깊이로 정합 절반을 먼저 검증할 수 있다.
- **상대 항법 대조군**: OpenVINS와 자체 흐름-고도계 추정기
- **검증 가설**
  - H-a: 질감이 적은 지형에서 T-I의 가용성이 P-G보다 높다. 리뷰의 정성 주장이다.
  - H-b: 크레이터가 많고 지도–하강 조명 잔차가 큰 조건에서 P-G의 무결성이 T-I보다 높다.
  - H-c: 시스템 하한은 T-I의 GSD 비 하한과 P-G의 카탈로그 하한 중 먼저 오는 쪽이다.
- **공정성 조건**
  - 모든 계열의 지도(정사 렌더, DEM, 크레이터 목록, 특징 DB)를 같은 원천 DEM에서 파생한다. 진실 지형은 다른 실현값으로 만든다.
  - 사전 오차와 채점(정답 / 고정 없음 / 오고정)을 같게 한다.
  - 원값 신뢰 신호와 영상당 계산 시간을 기록한다.

GIANT의 계보는 GSFC Retina다. autoNGC 프로세서 루프 시험(MicroZed 1 GHz ARM)에서 40 km 영상 한 장에 ~6초가 걸렸다. GPS와 함께 영상당 랜드마크 3–5개를 쓴 15 km 결과는 3σ 반경 1.6 m였다 ([Shoemaker et al. 2022](https://ntrs.nasa.gov/api/citations/20210024481/downloads/autoNGC_SciTech2022_Shoemaker_final.pdf)).

**학습 매처의 남극 성능.** 달 남극에서 학습 매처의 결과는 미세조정 여부에 달렸다. 기성 LoFTR은 방위 차 90°를 넘으면 쓸 수 없었고 미세조정판만 180°를 버텼다. 그 모델의 매칭 신뢰도는 "typically correlates with the accuracy of each match"였지만 코드는 공개되지 않았다 ([Georgakis & Ansar 2024](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf)). 후속인 LIMA(AIAA 2025-2073)는 유료다.

**라이선스 위생.**

- GPL 코드(OpenVINS, vas-byte)는 별도 프로세스나 ROS2 노드로 둔다.
- 라이선스가 없는 저장소는 저장소 안으로 들이지 않는다.
- SuperGlue는 쓰지 않는다. SuperPoint 가중치·추론 파일도 제한 라이선스이므로 LightGlue는 ALIKED(BSD-3)나 DISK(Apache-2.0)와 조합한다 ([LightGlue README](https://github.com/cvg/LightGlue)).

### 15상태 MEKF 플러그인과의 통합

sim2moon 항법기는 15상태 곱셈형 EKF다 ([lander_navigation_ekf.py](/home/sim2real/sim2moon/robots/lander_navigation_ekf.py)). NIS 통계와 잔차 편집을 이미 갖추고 있다. 다만 센서 역할 목록 `NAV_ROLES`는 `("imu", "attitude", "altimeter", "doppler")`뿐이라 카메라 역할을 새로 더해야 한다 ([lander_navigation.py](/home/sim2real/sim2moon/robots/lander_navigation.py)).

비행 사례는 두 결합 수준으로 나뉜다.

- **강결합**: 랜드마크 픽셀이나 시선 방향을 직접 넣는다. LVS, VISINAV-MSCKF, GIANT/autoNGC, Candan & Servadio가 이 방식이다.
- **약결합**: PnP 자세와 그 공분산을 넣는다. CNav의 EPnP + T² 게이트, IBAL의 PnP 선택지가 이 방식이다 ([Trigo et al. 2018](https://elib.dlr.de/118510/1/hybrid_optical_navigation_by_crater_detection_for_pinpoint_landing_%28elib%29.pdf); [Candan & Servadio 2026](https://arxiv.org/abs/2606.14776)).

[추론] MEKF에는 랜드마크 픽셀(또는 단위 시선) 갱신이 가장 깔끔하다.

- 측정 잡음은 R_i = σ_corr²·I로 둔다. σ_corr는 NCC 부화소 적합이나 검출 통계에서 얻고, 지도 위치 오차를 카메라로 투영한 항을 더한다.
- 각 갱신에 NIS χ² 게이트를 걸면, 기준선 사이에 비교 가능한 건전성 신호가 그대로 기록된다.
- PnP를 쓰면 6×6 공분산을 RANSAC inlier 야코비안에서 유도해야 한다. EPnP 오차가 사실상 3자유도로 상관된다는 CNav의 경고도 지켜야 한다.

지도 오차 예산도 중요하다. 지도 랜드마크를 융합한 로버 연구에서는 지도 σ가 1.5 m 이상이면 EKF 이득이 사라졌다 ([Duran & Leloglu 2026](https://navi.ion.org/content/73/1/navi.782)).

현 하향 카메라(96 × 128, 2 Hz)는 이 연구에 너무 작고 느리다. SLIM VBN은 512 × 512, LVS LCAM은 1024²였다. 광학흐름 VO는 1 Hz 아래에서 급락하므로 512² 이상과 4–10 Hz를 출발점으로 삼아야 한다 [추론].

Isaac Sim은 EKF 사전 자세에서 기대 영상을 렌더할 수 있다. 그러니 Blue Ghost식 렌더-대조 지도가 이 시뮬레이터에서는 가장 싸다.

### 지도·데이터·렌더링 자원

| 용도 | 자원 | 내용 | 라이선스·접근 | 출처 |
|---|---|---|---|---|
| 현재 진실 지형 + 지도 오차 주입 | PGDA 78, SL2 "de Gerlache rim" | 5 m LDEM(sim2moon 원천), `_toterr`(총 Z RMS), `_ldec`(LOLA 반사 수), 경사, **100개 clone DEM**. 중앙 Z 오차 ~0.30–0.50 m, 픽셀 ~90% 보간 | NASA GSFC 공개 | [PGDA 78](https://pgda.gsfc.nasa.gov/products/78); [README](https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README) |
| 고충실 진실·지도 | PGDA 104 / Zenodo 17954508 | de Gerlache Rim 2 5 m SfS DEM(2.19 GB), hillshade, **최대조명 정사모자이크**, SDEM−LDEM 차이, 다기선 경사·거칠기 | CC BY 4.0 | [PGDA 104](https://pgda.gsfc.nasa.gov/products/104); [Zenodo](https://zenodo.org/records/17954508) |
| 실효 해상도·오차 층 | PGDA 90 | 10/20/30/40/60/80/120/240 m LDEM, `_ERR`, `_EFFRES`, 영구음영, 거칠기 | 공개 COG | [PGDA 90](https://pgda.gsfc.nasa.gov/products/90) |
| 접근 경로 지도 | PGDA 92 | 60–80°S 띠, ~30 m SfS DEM (LuNaMaps) | 공개 | [PGDA 92](https://pgda.gsfc.nasa.gov/products/92) |
| 같은 지형·다른 태양의 실영상 | LROC NAC 남극 제어 모자이크 | 1 m/px, **태양경도 10° 구간별** 모자이크(예: CM_015는 424장) | PDS | [CM_015 README](https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/BDR/NAC_POLE/NAC_POLE_SOUTH_CM_015/NAC_POLE_SOUTH_CM_015_README.TXT) |
| de Gerlache 저태양 실영상 | `NAC_ROI_DEGERRIMLOB` | 1/5/20 m. NAC 51장, 태양경도 205–235°, 절대 위치 정확도 5–20 m | PDS | [LROC RDR](https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_ROI_DEGERRIMLOB) |
| 인접 고해상도 DTM | `NAC_DTM_SHACKRDGE02` | Shackleton–de Gerlache 연결 능선. 정사 1.2/3.5 m, LOLA 대비 RMS 1.06 m | PDS | [LROC RDR](https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_DTM_SHACKRDGE02) |
| 크레이터 카탈로그 | Robbins 2018; LU5M812TGT; Povilaitis 5–20 km 남극 shapefile; Fairweather NAC 평가지역; EXPLORE 2022 | ~1.3 M (1–2 km 이상), ~5 M (0.4 km 이상), 20 m급(평가지역 한정) | LU5M812TGT·Fairweather·EXPLORE는 CC BY 4.0 | [Robbins](https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018); [LU5M812TGT](https://zenodo.org/records/13990480); [Povilaitis](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LROC_5TO20KM_CRATERS_SPOLE); [Fairweather](https://zenodo.org/records/6386198); [EXPLORE](https://zenodo.org/records/7096475) |
| **지구 하강 비행 데이터 (TRN 카메라)** | **DDL-F1 — Blue Origin New Shepard NS-13 (2020-10)** | IMU, TRN 카메라 영상, 거리·속도 LiDAR. 단일 ZIP(`DDL-F1_Dataset-20201013.zip`). 비행 1 진실값 정확도는 "better than ±0.25 m 1-sigma"(AIAA 2022-1829). 카메라는 12.2° 크롭·1024², 천저 이탈 15° [원문]. **실제 하강 영상 + 진실값 + 공개된 TRN 결과(AIAA 2022-1830, 4,000 m/750 m 종료)** 를 함께 갖춘 유일한 공개 자료. 진실 궤적 포함 여부와 형식은 내려받아 확인 필요 | 미국 정부 저작물(NASA Open Data Portal, 관리자 K. Somervill) | [data.nasa.gov DDL-F1](https://data.nasa.gov/dataset/deorbit-descent-and-landing-flight-1-ddl-f1); [catalog.data.gov](https://catalog.data.gov/dataset/deorbit-descent-and-landing-flight-1-ddl-f1) |
| 비행 지도 + 영상 + 궤적 회귀 시험 | Mars 2020 LCAM + USGS CTX 6 m 정사/20 m DTM + NAIF `m2020_edl_v01` | 공개된 유일한 "비행 영상 + 비행 지도 + 재구성 궤적" 세트. 궤적은 IMU를 접지에서 역적분, 끝점 ~7 m 차이 | PDS·USGS·NAIF | [LCAM](https://doi.org/10.17189/hkah-6a45); [CTX 정사](https://astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0); [NAIF](https://naif.jpl.nasa.gov/pub/naif/MARS2020/kernels/spk/m2020_edl_v01.bsp) |
| 달 실제 하강 영상 | Chang'e-3/4/5/6 LCAM (CLEP) | CE-4는 1024², 45.3°, 10 fps. 텔레메트리는 공개되지 않아 사진측량 궤적(GCP RMS ~0.7/1.0 m)을 직접 복원. ASP에 CE-3 예제 있음 | CLEP 등록 필요 (조건 미확인) | [Liu et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6760200/); [ASP CE-3](https://stereopipeline.readthedocs.io/en/latest/examples/change3.html) |
| 실하강 크레이터 라벨 | CE-5 LCAM 132장 | 테두리 타원 라벨 | CC BY 4.0 | [Zenodo 11326450](https://zenodo.org/records/11326450) |
| CLPS 궤적 | NAIF CLPS SPICE | Blue Ghost 궤적은 탑재 "Vision Nav" 필터 출력이라 비전 채점에는 순환적. IM-1 SPK는 3σ 100 m | 공개 | [BGM-1 SPK](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/clps/clps_spice/spice_kernels/spk/clps_to19d_bgm1_edl_rec_250302_v01.bsp); [IM-1 SPK](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/clps/clps_spice/spice_kernels/spk/clps_to2im_im1_edl_rec_240222_v01.bsp) |
| 플룸 (부차) | Blue Ghost SCALPSS PDS (2026-09-17 공개) | ~28 m부터 시작. 외부표정은 아직 SPICE에 없음 | PDS | [DOI 10.17189/xbv8-7s71](https://doi.org/10.17189/xbv8-7s71); [NASA 2025-03-13](https://www.nasa.gov/general/nasa-cameras-on-blue-ghost-capture-first-of-its-kind-moon-landing-footage/) |
| 교차 렌더러 비교 | StereoLunar; MoonAnything; CRESENT-365 | 남극 LOLA 5 m, 3.5–30.5 km, Hapke+SurRender (Zenodo 11.43 GB); MoonAnything은 표본만 공개; CRESENT-365는 100 km 궤도 | StereoLunar·MoonAnything CC BY 4.0, CRESENT-365는 **CC BY-NC** | [StereoLunar](https://zenodo.org/records/21133765); [MoonAnything](https://arxiv.org/abs/2604.00682); [CRESENT-365](https://zenodo.org/records/23021341) |
| 렌더러 | Isaac RTX, Blender Cycles, GIANT 렌더, NASA Vira (C++), PANGU, SurRender, OmniLRS | OmniLRS는 Isaac Sim 5.0 기반이고 같은 PGDA DEM을 쓰지만 로버 전용 | Vira NOSA, PANGU는 ESA 과제 라이선스, SurRender 독점, OmniLRS BSD-3 | [Vira](https://github.com/nasa/vira); [PANGU](https://www.star-dundee.com/products/pangu-planet-and-asteroid-natural-scene-generation-utility/); [SurRender](https://www.airbus.com/en/products-services/space/space-customer-support/surrendersoftware); [OmniLRS](https://github.com/OmniLRS/OmniLRS) |
| 지도 제작 도구 | ISIS (10.0.0_LTS), Ames Stereo Pipeline (`sfs`), GDAL, SPICE | PGDA 104 파이프라인 그대로 | ASP Apache-2.0 | [ISIS3](https://github.com/DOI-USGS/ISIS3); [ASP](https://github.com/NeoGeographyToolkit/StereoPipeline); [ASP SfS](https://stereopipeline.readthedocs.io/en/latest/sfs_usage.html) |
| 광도 모델 | Sato 2014 Hapke 지도 (70°S–70°N만), Lunar-Lambert | 남극은 상수 Hapke나 Lunar-Lambert로 대체 | 논문 | [Sato et al. 2014](https://doi.org/10.1002/2013JE004580); [McEwen 1991](https://doi.org/10.1016/0019-1035%2891%2990053-V) |

### 실제 외관 지도를 만드는 파이프라인

[추론] 실제 외관의 다해상도·다조명 기준 지도는 다음 순서로 만들 수 있다.

1. de Gerlache Rim 2 위의 NAC EDR을 모은다.
2. ISIS `lronac2isis`/`lronaccal`/`spiceinit`로 들여온다. LuNaMaps 권고대로 LOLA SPK를 쓰고 NAC 시각 오프셋을 보정한다. 보정 전에는 "as much as 100 meters"였다.
3. ASP `bundle_adjust` + `mapproject`로 SL2 LDEM이나 SfS SDEM 위에 태양 기하별 정사영상을 1/2/5/10/20 m로 만든다.
4. `gdalwarp`으로 sim2moon 투영(−88.23°, −64.63° 중심 구면 정사, 25.37° 회전)에 맞춘다.

`NAC_ROI_DEGERRIMLOB`의 범위(−89.02…−87.79°, −83.18…−41.31°)에는 sim2moon 투영 중심이 들어간다. 다만 두 영역이 실제로 겹친다고 가정하기 전에 GIS로 교차를 확인해야 한다.

NAC 태양경도별 모자이크 사이의 템플릿 매칭은 남극에서 방위각 차이에 따라 상관이 얼마나 빨리 무너지는지를 **실제 영상으로** 보정해 준다. 공개된 하강 영상으로는 얻을 수 없는 자료다.

**Isaac/MDL에서 Hapke 같은 사용자 BRDF가 지원되는지는 확인하지 못했다.** 렌더 지도의 광도를 NAC 모자이크와 비교하는 LuNaMaps식 "Rendered DEM to NAC image comparison" 검사가 필요하다.

---

## 결론

"어느 고도까지 비전 TRN을 믿을 수 있는가"의 답은 카메라나 알고리즘의 고유 성질이 아니다. 대부분 **지도 제품과 카메라의 설계 변수**가 정한다. 지도 GSD와 그 실효 해상도, 최소 수록 크레이터 크기, 카메라 시야각(같은 지도에서 종료 고도가 2–3배 달라진다), 그리고 지도를 렌더한 조명 시점이다. 비행 증거가 하나같이 지도 준비가 하한을 정했다고 말하는 이유다. 남극에서는 문제의 축이 하나 더 생긴다. 태양 방위가 한 달에 한 바퀴 돌기 때문에, 고도 하한에 닿기 전에 **조명 창이 먼저 닫힐 수 있다**. 그러므로 이 연구의 결과물은 고도 하나가 아니다. (고도, 지도–하강 조명 잔차, 태양 고도, 지형 질감) 위에서 정의된 가용성 A와 무결성 I의 곡면이어야 한다. 독립변수의 중심도 절대 태양각보다 "렌더 지도와 실제 하강 조명의 잔차"에 두는 것이 비행 관행(태양 맞춘 렌더 지도)과 맞다.

둘째 함의는 신뢰가 정확도(CRB)의 문제가 아니라 **오고정의 문제**라는 것이다. 비행 시스템은 "답하지 않을 권리"로 무결성을 샀다. DIMES는 10% 미보고를 허용했고 IBAL은 inlier 8개 미만을 버렸다. 실패한 시스템은 대신 쓸 소스 없이 틀린 답을 냈다(Ingenuity, IM-2). 유일하게 측정된 작동점은 화성 LVS의 inlier ≥20 게이트다. 오탐률 2×10⁻⁸에 검출률 83%로, 무결성을 위해 가용성 17%를 포기한 셈이다. 시뮬레이션 연구만이 낼 수 있는 기여는 세 가지다. 오고정률을 신뢰구간과 함께 측정하는 것, 이미 쓰이는 온라인 지표(inlier 비, PSR, minima ratio, NIS, 보호수준)의 고도대별 ROC를 보정하는 것, 그리고 진실 지형과 탑재 지도를 서로 다른 DEM 실현값에서 만들어 "같은 DEM으로 채점하는" 낙관 편향을 제거하는 것이다. 이 셋을 갖추면 결과는 sim2moon 안에 머물지 않는다. LuNaMaps나 IM-3급 남극 착륙선이 지도 해상도와 착륙 시각 창을 정할 때 직접 쓸 수 있는 설계 곡선이 된다.

셋째 함의는 하한을 정하는 변수가 절대 TRN 계열마다 다르다는 것이다. 밝기 템플릿 매칭(T-I)에서는 지도 GSD 비가, 크레이터 패턴 매칭(P-G)에서는 카탈로그 최소 크기가 하한을 정한다. 그래서 같은 조건에서 계열을 나란히 돌리면 어느 계열이 어디서 먼저 무너지는지가 갈린다. 그 결과가 계열을 병행하거나 전환하는 설계의 정량 근거가 된다. 최신 리뷰도 "hybrid architectures capable of dynamically switching between template, pattern, or mapless methods based on real-time terrain analysis"를 권한다. 다만 전환 기준은 제시하지 않는다(Rahuman & Lee 2026) [원문].

---

## 출처

본문에서 인용하거나 사실 확인에 쓴 자료다(2026-10-10 기준). 아직 원문을 확인하지 못한 자료는 [부록](#부록-아직-원문을-확인하지-못한-자료-2026-10-10-정리)에 있다.

- **‡**: 원문을 확인했다. KAIST 도서관 기관 구독(2026-09-30·10-06)이나 사용자가 제공한 PDF(2026-10-10)로 확인했고, 기록은 아래 바.의 KAIST 기록에 있다.
- **[초록]**: 초록이나 요약만 확인했다.
- **[2차]**: 다른 문헌을 거쳐 인용했다.
- **[서지]**: 서지만 확인했다.
- **표시 없음**: 공개 링크로 내용을 확인한 자료다. 확인 범위는 조사 노트에 적었다.
- 링크가 여럿이면 DOI가 정본이다. 나머지는 본문에서 쓴 공개 전문, 사전본, 코드 위치다.
- 서지 메타데이터는 2026-10-10에 Crossref, DataCite, arXiv, NTRS API로 맞췄다.

### 가. 논문·학회 발표·보고서

- Aaron, S. B., Conway, D., Clouse, D., et al. (2022). Performance Analysis of Terrain Relative Navigation Using Blue Origin New Shepard Suborbital Flight Telemetry. *AIAA SCITECH 2022 Forum, AIAA 2022-1830*. doi:[10.2514/6.2022-1830](https://doi.org/10.2514/6.2022-1830) ‡
- Abdhul Rahuman, M. A., Lee, K. (2026). A Review of Vision-Based Terrain Relative Navigation for Planetary Descent and Landing of Space Vehicles. *International Journal of Aeronautical and Space Sciences* 27(4):3482–3501. doi:[10.1007/s42405-026-01139-1](https://doi.org/10.1007/s42405-026-01139-1) ‡
- Aguerrebere, C., Delbracio, M., Bartesaghi, A., et al. (2016). Fundamental Limits in Multi-image Alignment. *arXiv* [1602.01541](https://arxiv.org/abs/1602.01541)
- Andolfo, S., Genova, A., Buonomo, F. V., et al. (2025). Neural Network-Aided Optical Navigation for Precise Lunar Descent Operations. *Aerospace* 12(3):195. doi:[10.3390/aerospace12030195](https://doi.org/10.3390/aerospace12030195) [초록]
- Bayard, D. S., Conway, D. T., Brockers, R., et al. (2019). Vision-Based Navigation for the NASA Mars Helicopter. *AIAA Scitech 2019 Forum, AIAA 2019-1411*. doi:[10.2514/6.2019-1411](https://doi.org/10.2514/6.2019-1411) ‡
- Bergman, N. (1997). On the Cramér–Rao Bound for Terrain-Aided Navigation. *Linköping University 기술 보고서* LiTH-ISY-R-1970 — [링크](https://www.diva-portal.org/smash/get/diva2:315986/FULLTEXT02)
- Bieniawski, S., Polaha, J., Friia, B., et al. (2022). Blue Origin De-orbit Descent and Landing Tipping Point: Program Final Report. *NASA 계약자 보고서*. [NTRS 20210026314](https://ntrs.nasa.gov/citations/20210026314) — [링크](https://ntrs.nasa.gov/api/citations/20210026314/downloads/Blue%20Origin%20De-Orbit%20Descent%20Landing%20Tipping%20Point%20Final%20Report.pdf)
- Bieniawski, S. R., Lewis, B., Friia, B., et al. (2022). New Shepard Flight Test Results from Blue Origin De-Orbit Descent and Landing Tipping Point. *AIAA SCITECH 2022 Forum, AIAA 2022-1829*. doi:[10.2514/6.2022-1829](https://doi.org/10.2514/6.2022-1829) ‡
- Bolme, D., Beveridge, J. R., Draper, B. A., et al. (2010). Visual object tracking using adaptive correlation filters. *2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*. doi:[10.1109/cvpr.2010.5539960](https://doi.org/10.1109/cvpr.2010.5539960) — [링크](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf)
- Brockers, R., Proenca, P., Delaune, J., et al. (2022). On-board Absolute Localization Based on Orbital Imagery for a Future Mars Science Helicopter. *2022 IEEE Aerospace Conference (AERO)*. doi:[10.1109/AERO53065.2022.9843673](https://doi.org/10.1109/AERO53065.2022.9843673) ‡
- Bruvik, O. B., Valentin, R., Schlichting, M. R., et al. (2026). Protection Levels for Vision-Based Pose Estimation. *arXiv* [2608.10023](https://arxiv.org/abs/2608.10023)
- Candan, B., Servadio, S. (2026). Deep Learning-Based Lunar Crater Terrain Relative Navigation. *arXiv* [2606.14776](https://arxiv.org/abs/2606.14776)
- Carson, J. M., III, Robertson, E. A., Trawny, N., et al. (2015). Flight Testing ALHAT Precision Landing Technologies Integrated Onboard the Morpheus Rocket Vehicle. *AIAA SPACE 2015 Conference and Exposition*. [NTRS 20170009170](https://ntrs.nasa.gov/citations/20170009170) — [링크](https://ntrs.nasa.gov/api/citations/20170009170/downloads/20170009170.pdf) [초록]
- Chen, A., Casoliva, J., Montgomery, J., et al. (2021). Assessment of M2020 Terrain Relative Landing Accuracy: Flight Performance vs Predicts. *JPL 사전본*. [NTRS 20230007014](https://ntrs.nasa.gov/citations/20230007014)
- Cheng, Y., Goguen, J., Johnson, A., et al. (2004). The Mars exploration rovers descent image motion estimation system. *IEEE Intelligent Systems* 19(3):13–21. doi:[10.1109/mis.2004.18](https://doi.org/10.1109/mis.2004.18) — [링크](https://robotics.jpl.nasa.gov/media/documents/DIMES-ai-space.pdf)
- Cheng, Y., Ansar, A., Johnson, A. (2021). Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System. *Earth and Space Science* 8(8), e2020EA001560. doi:[10.1029/2020EA001560](https://doi.org/10.1029/2020EA001560) ‡
- Christian, J. A., Hong, L., McKee, P., et al. (2021). Image-Based Lunar Terrain Relative Navigation Without a Map: Measurements. *Journal of Spacecraft and Rockets* 58(1):164–181. doi:[10.2514/1.A34875](https://doi.org/10.2514/1.A34875) ‡
- Christian, J. A., Derksen, H., Watkins, R. (2021). Lunar Crater Identification in Digital Images. *The Journal of the Astronautical Sciences* 68(4):1056–1144. doi:[10.1007/s40295-021-00287-8](https://doi.org/10.1007/s40295-021-00287-8) — [링크](https://arxiv.org/abs/2009.01228)
- 中国科学院沈阳自动化研究所. 一种图像匹配系统的景象匹配区域选择与基准图优化方法. 중국 공개특허 CN106709501A — [링크](https://patents.google.com/patent/CN106709501A/zh)
- Cowan, S., Fanti, P., Williams, L. B. S., et al. (2025). Vision-Guided Optic Flow Navigation for Small Lunar Missions. *arXiv* [2511.17720](https://arxiv.org/abs/2511.17720)
- Daftry, S., Chen, Z., Cheng, Y., et al. (2023). LunarNav: Crater-based Localization for Long-range Autonomous Lunar Rover Navigation. *arXiv* [2301.01350](https://arxiv.org/abs/2301.01350)
- Danehy, P. M., Tyrrell, O. K., Weisberger, J. M., et al. (2026). Firefly Blue Ghost Mission 1: SCALPSS 1.1 Short Focal Length Camera Descent Imaging. *AIAA SCITECH 2026 Forum, AIAA 2026-1311*. doi:[10.2514/6.2026-1311](https://doi.org/10.2514/6.2026-1311) ‡
- Delaune, J., Le Besnerais, G., Voirin, T., et al. (2016). Visual–inertial navigation for pinpoint planetary landing using scale-based landmark matching. *Robotics and Autonomous Systems* 78:63–82. doi:[10.1016/j.robot.2016.01.007](https://doi.org/10.1016/j.robot.2016.01.007) ‡
- Downes, L. M., Steiner, T. J., How, J. P. (2020). Lunar Terrain Relative Navigation Using a Convolutional Neural Network for Visual Crater Detection. *arXiv* [2007.07702](https://arxiv.org/abs/2007.07702)
- Duran, H., Leloglu, U. M. (2026). Maximum-Likelihood Aerial-to-Ground Landmark Matching for Drift Suppression in Lunar Visual–Inertial Navigation. *NAVIGATION: Journal of the Institute of Navigation* 73(1):navi.782. doi:[10.33012/navi.782](https://doi.org/10.33012/navi.782) — [링크](https://navi.ion.org/content/73/1/navi.782)
- Fairweather, J. H., Lagain, A., Servis, K., et al. (2022). Automatic Mapping of Small Lunar Impact Craters Using LRO‐NAC Images. *Earth and Space Science* 9(7), e2021EA002177. doi:[10.1029/2021EA002177](https://doi.org/10.1029/2021EA002177) ‡
- Fragoso, A. T., Lee, C. T., McCoy, A. S., et al. (2021). A seasonally invariant deep transform for visual terrain-relative navigation. *Science Robotics* 6(55), eabf3320. doi:[10.1126/scirobotics.abf3320](https://doi.org/10.1126/scirobotics.abf3320) [초록]
- Freeman, R. H. (2025). Soft Landing on the Moon: The Ups and Downs of Robotic Lunar Landers. *Journal of Space Operations & Communicator* 21(2) — [링크](https://www.opsjournal.org/DocumentLibrary/Uploads/Freeman%20RH_JSOC_Q2B_2025.pdf) [2차]
- Fritz, M. P., Doll, J. A., Ward, K. C., et al. (2022). Post-Flight Performance Analysis of Navigation and Advanced Guidance Algorithms on a Terrestrial Suborbital Rocket Flight. *AIAA SciTech Forum*. [NTRS 20210023718](https://ntrs.nasa.gov/citations/20210023718) — [링크](https://ntrs.nasa.gov/api/citations/20210023718/downloads/AIAA_SciTech2022_TP2_NAVDQG.pdf)
- Georgakis, G., Ansar, A. (2024). Learning Illumination Invariant Features for Lunar South Pole with Deep Learning. *Space Imaging Workshop*, Atlanta, 2024-10-07–09 — [링크](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf)
- Grethen, C., Gasparini, S., Morin, G., et al. (2025). Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset. *arXiv* [2510.18172](https://arxiv.org/abs/2510.18172)
- Grethen, C., Shi, Y., Gasparini, S., et al. (2026). MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data. *arXiv* [2604.00682](https://arxiv.org/abs/2604.00682)
- Gupta, S., Gao, G. X. (2021). Data-Driven Protection Levels for Camera and 3D Map-based Safe Urban Localization. *arXiv* [2101.06379](https://arxiv.org/abs/2101.06379)
- Hamel, J.-F., Garant, A., Godin, C., et al. (2025). Performance Assessment of Crater-Based Navigation for Autonomous Moon Landing. *AIAA SCITECH 2025 Forum, AIAA 2025-2075*. doi:[10.2514/6.2025-2075](https://doi.org/10.2514/6.2025-2075) ‡
- Hanak, C., Crain, T. (2010). Crater Identification Algorithm for the Lost in Low Lunar Orbit Scenario. *33rd Annual AAS Guidance and Control Conference*. [NTRS 20090032647](https://ntrs.nasa.gov/citations/20090032647)
- Haydon, T., Huang, A., Humphreys, T. E. (2025). Analysis of the Trusted Inertial Terrain-Aided Navigation Measurement Function. *NAVIGATION: Journal of the Institute of Navigation* 72(3):navi.707. doi:[10.33012/navi.707](https://doi.org/10.33012/navi.707) — [링크](https://navi.ion.org/content/72/3/navi.707) [초록]
- Hines, G. D., Amzajerdian, F., Gragossian, A., et al. (2025). Demonstration of Navigation Doppler Lidar Capabilities Onboard the First Commercial Lunar Lander. *AIAA SCITECH 2025 Forum, AIAA 2025-1115*. doi:[10.2514/6.2025-1115](https://doi.org/10.2514/6.2025-1115) ‡
- Hough, K., Mohammadi, M. E., Stephans, T., et al. (2024). Verification and Validation of Lunar Terrain Relative Navigation Maps for Precision Landing. *AIAA SCITECH 2024 Forum, AIAA 2024-0311*. doi:[10.2514/6.2024-0311](https://doi.org/10.2514/6.2024-0311) ‡
- Irani, G. B., Christ, J. P. (1994). Image Processing for Tomahawk Scene Matching. *Johns Hopkins APL Technical Digest* 15(3):250–264 — [링크](https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V15-N03/15-03-Irani.pdf)
- Ishida, T., Fukuda, S., Kariya, K., et al. (2025). Vision-based navigation and obstacle detection flight results in SLIM lunar landing. *Acta Astronautica* 226:772–781. doi:[10.1016/j.actaastro.2024.11.002](https://doi.org/10.1016/j.actaastro.2024.11.002) ‡
- Ishii, H., et al. (2017). The robust spacecraft location estimation algorithm toward the misdetection crater and the undetected crater in SLIM. *ISTS/ISSFD 2017*, ISTS-2017-d-067 [2차]
- Johnson, A., Willson, R., Cheng, Y., et al. (2007). Design Through Operation of an Image-Based Velocity Estimation System for Mars Landing. *International Journal of Computer Vision* 74(3):319–341. doi:[10.1007/s11263-006-0022-z](https://doi.org/10.1007/s11263-006-0022-z) — [링크](https://robotics.jpl.nasa.gov/media/documents/DIMES_IJCV_20070117.pdf)
- Johnson, A. E., Montgomery, J. F. (2008). Overview of Terrain Relative Navigation Approaches for Precise Lunar Landing. *2008 IEEE Aerospace Conference*. doi:[10.1109/AERO.2008.4526302](https://doi.org/10.1109/AERO.2008.4526302) [초록]
- Johnson, A., Ivanov, T. (2011). Analysis and Testing of a LIDAR-Based Approach to Terrain Relative Navigation for Precise Lunar Landing. *AIAA Guidance, Navigation, and Control Conference, AIAA 2011-6578*. doi:[10.2514/6.2011-6578](https://doi.org/10.2514/6.2011-6578) — [링크](https://www-robotics.jpl.nasa.gov/media/documents/FT3_LidarTRN_Paper_v12.pdf)
- Johnson, A. E., Cheng, Y., Montgomery, J. F., et al. (2015). Real-Time Terrain Relative Navigation Test Results from a Relevant Environment for Mars Landing. *AIAA Guidance, Navigation, and Control Conference, AIAA 2015-0851*. doi:[10.2514/6.2015-0851](https://doi.org/10.2514/6.2015-0851) ‡
- Johnson, A. E., Aaron, S. B., Ansari, H., et al. (2022). Mars 2020 Lander Vision System Flight Performance. *AIAA SCITECH 2022 Forum, AIAA 2022-1214*. doi:[10.2514/6.2022-1214](https://doi.org/10.2514/6.2022-1214) ‡
- Johnson, A. E., Cheng, Y., Trawny, N., et al. (2023). Implementation of a Map Relative Localization System for Planetary Landing. *Journal of Guidance, Control, and Dynamics* 46(4):618–637. doi:[10.2514/1.G006780](https://doi.org/10.2514/1.G006780) ‡
- Johnson, A. E., Trawny, N., Setterfield, T. P., et al. (2025). Challenges and Solutions for Early Morning Terrain Relative Navigation on Mars. *AIAA SCITECH 2025 Forum, AIAA 2025-2076*. doi:[10.2514/6.2025-2076](https://doi.org/10.2514/6.2025-2076) ‡
- Johnson, A., Aaron, S., Chang, J., et al. (2017). The Lander Vision System for Mars 2020 Entry Descent and Landing. *AAS Guidance, Navigation and Control Conference*, AAS 17-038 (JPL hdl:2014/46186) — [링크](https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/46186)
- Johnson, A. (2021-03-02). Mars 2020 Lander Vision System (발표 자료). *LSIC Workshop for Lunar Mapping for Precision Landing* — [링크](https://lsic.jhuapl.edu/uploadedDocs/presentations/764-1.09_Johnson_20210302_LuNaMaps_LVS_v0.pdf)
- Kaufmann, H., Lingenauber, M., Bodenmueller, T., et al. (2015). Shadow-based matching for precise and robust absolute self-localization during lunar landings. *2015 IEEE Aerospace Conference*. doi:[10.1109/AERO.2015.7119045](https://doi.org/10.1109/AERO.2015.7119045) [초록]
- Kim, T., Nam, S., Lee, H., et al. (2025). Verification of Vision-Based Terrain-Referenced Navigation Using the Iterative Closest Point Algorithm Through Flight Testing. *Sensors* 25(18):5813. doi:[10.3390/s25185813](https://doi.org/10.3390/s25185813) [초록]
- Kuga, T., Kojima, H. (2018). Self-position estimation using terrain shadows for precise planetary landing. *Acta Astronautica* 148:345–354. doi:[10.1016/j.actaastro.2018.05.002](https://doi.org/10.1016/j.actaastro.2018.05.002) [초록]
- Li Sanchez, H. A. (2024). Terrain-Relative Navigation for Precision Landings based on a Hierarchical Localization Approach. *석사논문, University of Pittsburgh* — [링크](https://d-scholarship.pitt.edu/41874/) [2차]
- Liu, J., Ren, X., Yan, W., et al. (2019). Descent trajectory reconstruction and landing site positioning of Chang’E-4 on the lunar farside. *Nature Communications* 10(1), 4229. doi:[10.1038/s41467-019-12278-3](https://doi.org/10.1038/s41467-019-12278-3) — [링크](https://pmc.ncbi.nlm.nih.gov/articles/PMC6760200/)
- Liu, S., Xu, C., Chu, G., et al. (2026). Block-optimized FDAFT feature matching for lunar landing navigation and positioning: Validation with Chang'E−6 data. *Planetary and Space Science* 270:106221. doi:[10.1016/j.pss.2025.106221](https://doi.org/10.1016/j.pss.2025.106221) ‡
- Maass, B. (2016). Robust approximation of image illumination direction in a segmentation-based crater detection algorithm for spacecraft navigation. *CEAS Space Journal* 8(4):303–314. doi:[10.1007/s12567-016-0129-1](https://doi.org/10.1007/s12567-016-0129-1) — [링크](https://elib.dlr.de/104630/1/art_10.1007_s12567-016-0129-1%20Maass.pdf)
- Maass, B., Woicke, S., Oliveira, W. M., et al. (2020). Crater Navigation System for Autonomous Precision Landing on the Moon. *Journal of Guidance, Control, and Dynamics* 43(8):1414–1431. doi:[10.2514/1.G004850](https://doi.org/10.2514/1.G004850) ‡
- Maddock, R., Munk, M., Danehy, P. M., et al. (2026). Firefly Blue Ghost Mission 1: SCALPSS 1.1 Payload and Mission Overview. *AIAA SCITECH 2026 Forum, AIAA 2026-1308*. doi:[10.2514/6.2026-1308](https://doi.org/10.2514/6.2026-1308) ‡
- Maggio, D., Mario, C., Streetman, B., et al. (2023). Vision-Based Terrain Relative Navigation on High-Altitude Balloon and Sub-Orbital Rocket. *arXiv* [2302.08011](https://arxiv.org/abs/2302.08011)
- McEwen, A. S. (1991). Photometric functions for photoclinometry and other applications. *Icarus* 92(2):298–311. doi:[10.1016/0019-1035(91)90053-V](https://doi.org/10.1016/0019-1035(91)90053-V)
- McLeod, S., Chng, C.-K., Rodda, M., et al. (2026). AI-enabled crater-based navigation for lunar mapping. *Astrodynamics* 10(3):483–505. doi:[10.1007/s42064-025-0294-0](https://doi.org/10.1007/s42064-025-0294-0) — [링크](https://arxiv.org/abs/2509.20748) ‡
- Minton, D. A., Fassett, C. I., Hirabayashi, M., et al. (2019). The equilibrium size-frequency distribution of small craters reveals the effects of distal ejecta on lunar landscape morphology. *arXiv* [1902.07746](https://arxiv.org/abs/1902.07746)
- Molina, G., Hansen, M., Getchius, J., et al. (2024). Visual Odometry For Precision Lunar Landing. *Advances in the Astronautical Sciences*. doi:[10.1007/978-3-031-51928-4_58](https://doi.org/10.1007/978-3-031-51928-4_58) — Proc. 44th AAS GN&C Conference (2022), AAS 22-113, pp. 1021–1042 ‡
- Mostafavi, H., Smith, F. W. (1978). Image Correlation with Geometric Distortion Part 1: Acquisition Performance. *IEEE Transactions on Aerospace and Electronic Systems* AES-14(3):487–493. doi:[10.1109/TAES.1978.308610](https://doi.org/10.1109/TAES.1978.308610) — Part II: pp. 494–500 ‡
- Mourikis, A., Trawny, N., Roumeliotis, S., et al. (2007). Vision-Aided Inertial Navigation for Precise Planetary Landing: Analysis and Experiments. *Robotics: Science and Systems III*. doi:[10.15607/rss.2007.iii.019](https://doi.org/10.15607/rss.2007.iii.019) — [링크](https://www.roboticsproceedings.org/rss03/p19.pdf)
- Mourikis, A., Trawny, N., Roumeliotis, S., et al. (2009). Vision-Aided Inertial Navigation for Spacecraft Entry, Descent, and Landing. *IEEE Transactions on Robotics* 25(2):264–280. doi:[10.1109/TRO.2009.2012342](https://doi.org/10.1109/TRO.2009.2012342) ‡
- Ohira, G., Kashioka, S., Takao, Y., et al. (2022). Autonomous image-based navigation using vector code correlation algorithm for distant small body exploration. *Acta Astronautica* 196:400–413. doi:[10.1016/j.actaastro.2020.10.013](https://doi.org/10.1016/j.actaastro.2020.10.013)
- Pierrottet, D., Monaco, J. F., Sandford, S. P. (2025). Navigation Doppler LiDAR Lunar Landing Data Analysis and Trajectory Reconstruction. *AIAA SCITECH 2025 Forum, AIAA 2025-1116*. doi:[10.2514/6.2025-1116](https://doi.org/10.2514/6.2025-1116) ‡
- Pisanti, D., Hewitt, R., Brockers, R., et al. (2025). Geometry-aided Vision-based Localization of Future Mars Helicopters in Challenging Illumination Conditions. *arXiv* [2502.09795](https://arxiv.org/abs/2502.09795)
- Pisanti, D., Georgakis, G. (2026). MARTIAN: A Rendering Framework for Aerial Mars Imagery from HiRISE Orbital Data. *arXiv* [2605.29647](https://arxiv.org/abs/2605.29647)
- Restrepo, C., Johnson, A. (2021). Terrain Relative Navigation Basics. *LSIC Workshop on Lunar Mapping for Precision Landing (발표 자료)*. [NTRS 20210010726](https://ntrs.nasa.gov/citations/20210010726) — [링크](https://ntrs.nasa.gov/api/citations/20210010726/downloads/TRN_Basics_LSICWorkshop_March2%20%28002%29.pptx.pdf)
- Restrepo, C. I., Petro, N., Barker, M., et al. (2022). Building Lunar Maps for Terrain Relative Navigation and Hazard Detection Applications. *AIAA SCITECH 2022 Forum, AIAA 2022-0356*. doi:[10.2514/6.2022-0356](https://doi.org/10.2514/6.2022-0356) — [링크](https://ntrs.nasa.gov/api/citations/20210024816/downloads/LuNaMaps_Scitech2022_Nov19.pdf) ‡
- Restrepo, C., Petro, N., Barker, M., et al. (2024). Technology Transfer Plan: LuNaMaps Project. *4th Space Imaging Workshop*. [NTRS 20240012328](https://ntrs.nasa.gov/citations/20240012328) — [링크](https://ntrs.nasa.gov/api/citations/20240012328/downloads/LuNaMaps_TechTransfer.pdf)
- Robinson, D., Milanfar, P. (2004). Fundamental Performance Limits in Image Registration. *IEEE Transactions on Image Processing* 13(9):1185–1199. doi:[10.1109/TIP.2004.832923](https://doi.org/10.1109/TIP.2004.832923) — [링크](https://users.soe.ucsc.edu/~milanfar/publications/journal/MotionPerformanceFinal.pdf)
- Rothenberger, N., Georgakis, G., Cheng, Y., et al. (2025). Illumination Invariant Image Matching for Lunar TRN. *AIAA SCITECH 2025 Forum, AIAA 2025-2073*. doi:[10.2514/6.2025-2073](https://doi.org/10.2514/6.2025-2073) ‡
- Sato, H., Robinson, M. S., Hapke, B., et al. (2014). Resolved Hapke parameter maps of the Moon. *Journal of Geophysical Research: Planets* 119(8):1775–1805. doi:[10.1002/2013JE004580](https://doi.org/10.1002/2013JE004580)
- Setterfield, T. P., Hewitt, R. A., Trawny, N. (2021). Fundamental Limits of Map Relative Localization During Spacecraft Descent and Landing. *Journal of Guidance, Control, and Dynamics* 44(7):1239–1252. doi:[10.2514/1.G005634](https://doi.org/10.2514/1.G005634) ‡
- Setterfield, T. P., Conway, D., Chen, P.-T., et al. (2024). Enhanced Lander Vision System (ELViS) Algorithms for Pinpoint Landing of the Mars Sample Retrieval Lander. *AIAA SCITECH 2024 Forum, AIAA 2024-0315*. doi:[10.2514/6.2024-0315](https://doi.org/10.2514/6.2024-0315) ‡
- 沈林成, 卜彦龙, 徐昕, 潘亮 (Shen, L., Bu, Y., Xu, X., Pan, L.) (2010). 景象匹配辅助组合导航中景象区域适配性研究进展 (Research on Matching-area Suitability for Scene Matching Aided Navigation). *航空学报* 31(3):553–563 — [링크](https://hkxb.buaa.edu.cn/EN/abstract/abstract9465.shtml) ‡
- Shoemaker, M. A., Hur-Diaz, S., Liounis, A. J., et al. (2022). Terrain Relative Navigation in a Lunar Landing Scenario Using autoNGC. *AIAA SciTech Forum and Exposition*. [NTRS 20210024481](https://ntrs.nasa.gov/citations/20210024481) — [링크](https://ntrs.nasa.gov/api/citations/20210024481/downloads/autoNGC_SciTech2022_Shoemaker_final.pdf)
- Shu, S., Lin, L., Hou, B., et al. (2026). Intelligent vision-guided trajectory reconstruction enables rapid localization and characterization of the Chang’E-6 landing site. *Communications Earth & Environment* 7(1), 61. doi:[10.1038/s43247-025-03074-7](https://doi.org/10.1038/s43247-025-03074-7) — 온라인 2025-12-17
- Smith, K., Olguin, A., Fritz, M., et al. (2020). Operational Constraint Analysis of Terrain Relative Navigation for Landing Applications. *AIAA Scitech 2020 Forum, AIAA 2020-0370*. doi:[10.2514/6.2020-0370](https://doi.org/10.2514/6.2020-0370) — 제1저자 정정: 2026-10-06 KAIST 추출 기록은 Fritz를 제1저자로 적었으나 Crossref·Semantic Scholar·OpenAlex 모두 Smith가 제1저자 ‡
- Smith, K. W., Anastas, N., Olguin, A. S., et al. (2022). Building Maps for Terrain Relative Navigation Using Blender: An Open-Source Approach. *AIAA SciTech Forum*. [NTRS 20210023848](https://ntrs.nasa.gov/citations/20210023848)
- Steffes, S. R., Monterroza, F., Benhacine, L., et al. (2019). Optical Terrain Relative Navigation Approaches to Lunar Orbit, Descent and Landing. *AIAA Scitech 2019 Forum, AIAA 2019-1178*. doi:[10.2514/6.2019-1178](https://doi.org/10.2514/6.2019-1178) ‡
- Steiner, T. J., Brady, T. M., Hoffman, J. A. (2015). Graph-based terrain relative navigation with optimal landmark database selection. *2015 IEEE Aerospace Conference*. doi:[10.1109/AERO.2015.7119053](https://doi.org/10.1109/AERO.2015.7119053) ‡
- Theil, S., Krüger, H., Maass, B., et al. (2017). ATON – Autonomous Terrain-based Optical Navigation for Exploration Missions: Recent Flight Test Results. *Deutscher Luft- und Raumfahrtkongress (DLRK) 2017*, DLR elib 114457 — [링크](https://elib.dlr.de/114457/1/ATON_DLRK_2017.pdf)
- Tong, X., Feng, Y., Ye, Z., et al. (2025). Illumination Robust Landing Point Visual Localization for Lunar Lander With High-Resolution Map Generation. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* 18:1577–1591. doi:[10.1109/JSTARS.2024.3498351](https://doi.org/10.1109/JSTARS.2024.3498351) [초록]
- Trawny, N., Benito, J., Tweddle, B. E., et al. (2015). Flight testing of terrain-relative navigation and large-divert guidance on a VTVL rocket. *AIAA SPACE 2015 Conference and Exposition, AIAA 2015-4418*. doi:[10.2514/6.2015-4418](https://doi.org/10.2514/6.2015-4418) ‡
- Trawny, N., Johnson, A. E., Bailey, E. S., et al. (2024). The Enhanced Lander Vision System for Mars Sample Return Lander Entry Descent and Landing. *AIAA SCITECH 2024 Forum, AIAA 2024-0314*. doi:[10.2514/6.2024-0314](https://doi.org/10.2514/6.2024-0314) ‡
- Trigo, G. F., Maass, B., Krüger, H., et al. (2018). Hybrid optical navigation by crater detection for lunar pin-point landing: trajectories from helicopter flight tests. *CEAS Space Journal* 10(4):567–581. doi:[10.1007/s12567-017-0188-y](https://doi.org/10.1007/s12567-017-0188-y) — [링크](https://elib.dlr.de/118510/1/hybrid_optical_navigation_by_crater_detection_for_pinpoint_landing_%28elib%29.pdf)
- Uslu, I. E., Tanyeri, B., Bekar, U. C., et al. (2026). A Statistical Framework for False-Fix Elimination in TERCOM Using Real Flight Test Data. *International Journal of Aeronautical and Space Sciences* 27(4):3623–3641. doi:[10.1007/s42405-026-01126-6](https://doi.org/10.1007/s42405-026-01126-6) — [링크](https://link.springer.com/article/10.1007/s42405-026-01126-6) ‡
- Wang, S., Chen, Z., Yu, G., et al. (2026). A two-stage offline-online visual navigation framework for high-precision lunar polar landing. *Chinese Journal of Aeronautics* 39(9):104149. doi:[10.1016/j.cja.2026.104149](https://doi.org/10.1016/j.cja.2026.104149) ‡
- Way, D. W., Dutta, S., Zumwalt, C. H., et al. (2022). Assessment of the Mars 2020 Entry, Descent, and Landing Simulation. *AIAA SciTech Forum*. [NTRS 20210024480](https://ntrs.nasa.gov/citations/20210024480) — [링크](https://ntrs.nasa.gov/api/citations/20210024480/downloads/Mars2020_Flight_Mechanics_Simulation_Paper_v3.pdf)
- Woicke, S., Krüger, H. (2023). Performance of a Crater Navigation Method for Lunar South Pole Landing. *AIAA SCITECH 2023 Forum, AIAA 2023-0690*. doi:[10.2514/6.2023-0690](https://doi.org/10.2514/6.2023-0690) ‡
- Xie, B., Liu, B., Di, K., et al. (2025). Robust Feature Matching of Multi-Illumination Lunar Orbiter Images Based on Crater Neighborhood Structure. *Remote Sensing* 17(13):2302. doi:[10.3390/rs17132302](https://doi.org/10.3390/rs17132302)
- Xiu, Y., Zhu, S., Xu, R., et al. (2023). Optimal crater landmark selection based on optical navigation performance factors for planetary landing. *Chinese Journal of Aeronautics* 36(3):254–270. doi:[10.1016/j.cja.2022.08.009](https://doi.org/10.1016/j.cja.2022.08.009) ‡
- Yang, C., Ye, Y., Liu, R., et al. (2026). Suitable area selection method based on scene matching level segmentation. *Chinese Journal of Aeronautics* 39(2):103719. doi:[10.1016/j.cja.2025.103719](https://doi.org/10.1016/j.cja.2025.103719) ‡
- Yu, Z., Zhu, B., Han, Z., et al. (2024). Reliability analysis of multimodal scene matching based on correlation peaks. *Chinese Journal of Aeronautics* 37(9):433–447. doi:[10.1016/j.cja.2024.06.024](https://doi.org/10.1016/j.cja.2024.06.024) ‡
- Zhu, Z., Taylor, C. (2017). Conservative Uncertainty Estimation in Map-Based Vision-Aided Navigation. *IEEE Transactions on Aerospace and Electronic Systems* 53(2):941–949. doi:[10.1109/TAES.2017.2667278](https://doi.org/10.1109/TAES.2017.2667278) — [링크](https://ieeexplore.ieee.org/document/7858648/) ‡
- Zhu, C., Joerger, M., Meurer, M. (2020). Quantifying Feature Association Error in Camera-based Positioning. *2020 IEEE/ION Position, Location and Navigation Symposium (PLANS)*. doi:[10.1109/PLANS46316.2020.9109919](https://doi.org/10.1109/PLANS46316.2020.9109919) ‡
- Zhu, C., Meurer, M., Günther, C. (2022). Integrity of Visual Navigation—Developments, Challenges, and Prospects. *NAVIGATION: Journal of the Institute of Navigation* 69(2):navi.518. doi:[10.33012/navi.518](https://doi.org/10.33012/navi.518) — [링크](https://elib.dlr.de/147041/1/Integrity_of_visual_navigation_ION_NAVIGATION_v4.pdf)

### 나. 국내 문헌 (서지만 확인한 것 포함)

원문을 확인한 국내 연구(Rahuman & Lee 2026, Kim et al. 2025)는 가.에 실었다.

- Choe, Y., Park, C. G. (2024). LiDAR-Inertial-Based Absolute Positioning With Sparse DEM for Accurate Lunar Landing. *IEEE Transactions on Aerospace and Electronic Systems* 60(3):3482–3495 (서울대) [서지]
- Kim, S., Park, J., Bang, H. (2025). Enhanced Terrain-Referenced Navigation Through Adaptive Radar Altimeter Error Estimation. *International Journal of Aeronautical and Space Sciences* (KAIST) [서지]
- 고형우, 전하민, 김효중, 허재석, 유창경 (2024). 카메라 시야각에 따른 크레이터 기반 영상항법 성능 분석. *한국항공우주학회 학술발표회* [서지]
- 구평모, 박영범, 박찬국 (2016). 영역기반 지형 험준도 지수를 이용한 달착륙선의 일괄처리방식 지형상대항법 성능분석. *한국항공우주학회지* 44(7) [서지]
- 김기덕, 정유연, 방효충 외 (2017). 동력하강 전 크레이터 + 관성 영상항법, 영상 잡음 아래 크레이터 매칭 특성 (두 편). *한국항공우주학회 학술발표회* [서지]
- 김재인 (2022). 화성 지형상대항법을 위한 하강 데이터셋 생성과 랜드마크 추출 방법. *대한원격탐사학회지* 38(6) [서지]
- 류동영 외 (2022). 달 착륙 항법 종합시험장치를 이용한 지상비행 시험. *한국항공우주학회 학술발표회* [서지]
- 문귀원, 이훈희, 류동영 (2020). 달 착륙선 항법용 영상의 사전처리 소개. *컴퓨팅의 실제 논문지* 26(6) [서지]
- 박영범, 박찬국, 권재욱, 류동영 (2017). 임무조건에 따른 달 착륙선 항법시스템 요구성능 분석. *한국항공우주학회지* 45(9) [서지]
- 박우상, 정유연, 나윤주, 방효충 (2016). 크레이터 탐지·매칭 지상검증 플랫폼 연구. *한국항공우주학회 학술발표회*. 영문판: Park, W., Jung, Y., Bang, H. (2016). Terrain relative navigation for precise lunar landing using crater matching algorithm. *ICCAS 2016*, doi:[10.1109/ICCAS.2016.7832378](https://doi.org/10.1109/ICCAS.2016.7832378) [서지]
- 최재훈, 김효중, 유창경 (2025). 크레이터 영상항법의 고위도 방위각 추정 오차 보정. *한국항공우주학회 학술발표회* [서지]
- 황연하, 박봉균, 탁민제 (2012). 달 착륙선의 정밀 착륙을 위한 DSMAC방법 기반의 지형 상대 항법 연구. *한국항공우주학회 학술발표회* [서지]

### 다. 데이터·지도 제품

- Bertone, S., McKenna, T. E., Barker, M. K., et al. (2025). Enhanced Topography Models for selected Lunar South Pole regions with Shape-from-Shading. *NASA GSFC PGDA Product 104* (Zenodo 17954508 보관본), CC BY 4.0 — [링크 1](https://pgda.gsfc.nasa.gov/products/104), [링크 2](https://zenodo.org/records/17954508)
- Chng, C.-K., McLeod, S., Chin, T.-J. (2026). CRESENT-365. *Zenodo* 23021341, CC BY-NC 4.0 — [링크](https://zenodo.org/records/23021341)
- EXPLORE (2022). EXPLORE Expert Data Challenges 2022 – Craters dataset. *Zenodo* 7096475, CC BY 4.0 — [링크](https://zenodo.org/records/7096475)
- Fairweather, J. H., et al. (2022). Supporting Material for Automatic Mapping of Small Lunar Impact Craters Using LROC NAC Images. *Zenodo* 6386198, CC BY 4.0 — [링크](https://zenodo.org/records/6386198)
- Grethen, C., Gasparini, S., Morin, G., et al. (2026). StereoLunar dataset. *Zenodo* 21133765, CC BY 4.0 — [링크](https://zenodo.org/records/21133765)
- La Grassa, R. (2024). LU5M812TGT: An AI-Powered Global Database of Impact Craters ≥0.4 km on the Moon. *Zenodo* 13990480, CC BY 4.0 — [링크](https://zenodo.org/records/13990480)
- LROC Team. NAC_DTM_SHACKRDGE02 — Shackleton–de Gerlache 능선 NAC DTM. *LROC RDR* — [링크](https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_DTM_SHACKRDGE02)
- LROC Team. NAC_POLE_SOUTH_CM_015 — LROC NAC 남극 제어 모자이크(태양경도 10° 구간). *PDS* LROLRC_2001 README — [링크](https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/BDR/NAC_POLE/NAC_POLE_SOUTH_CM_015/NAC_POLE_SOUTH_CM_015_README.TXT)
- LROC Team. NAC_ROI_DEGERRIMLOB — de Gerlache 림 NAC 모자이크. *LROC RDR* — [링크](https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_ROI_DEGERRIMLOB)
- Lueg, R. T. (2026). CLPS Task Order 19D Blue Ghost Lander SCALPSS Bundle. *NASA PDS 데이터셋*. doi:[10.17189/xbv8-7s71](https://doi.org/10.17189/xbv8-7s71)
- Maki, J. N. (2020). Mars 2020 Entry, Descent, and Landing (EDL) and Lander Visual System Cameras Bundle, raw products. *NASA PDS 데이터셋*. doi:[10.17189/hkah-6a45](https://doi.org/10.17189/hkah-6a45)
- NASA (2020). Deorbit, Descent, and Landing Flight 1 (DDL-F1) — Blue Origin New Shepard NS-13 비행 데이터(IMU, TRN 카메라, LiDAR, 진실 궤적). *NASA Open Data Portal* — [링크 1](https://data.nasa.gov/dataset/deorbit-descent-and-landing-flight-1-ddl-f1), [링크 2](https://catalog.data.gov/dataset/deorbit-descent-and-landing-flight-1-ddl-f1)
- NASA GSFC PGDA Product 78. LOLA 5 m/px 남극 사이트 지형(LDEM, 오차 지도, clone DEM) — [링크 1](https://pgda.gsfc.nasa.gov/products/78), [링크 2](https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README)
- NASA GSFC PGDA Product 90. LOLA 다해상도 LDEM과 오차·실효 해상도 층 — [링크](https://pgda.gsfc.nasa.gov/products/90)
- NASA GSFC PGDA Product 92. 60–80°S 띠 ~30 m SfS DEM (LuNaMaps) — [링크](https://pgda.gsfc.nasa.gov/products/92)
- NASA NAIF. CLPS SPICE EDL 재구성 SPK — Blue Ghost M1(clps_to19d_bgm1_edl_rec_250302_v01.bsp), IM-1(clps_to2im_im1_edl_rec_240222_v01.bsp) — [링크 1](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/clps/clps_spice/spice_kernels/spk/clps_to19d_bgm1_edl_rec_250302_v01.bsp), [링크 2](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/clps/clps_spice/spice_kernels/spk/clps_to2im_im1_edl_rec_240222_v01.bsp)
- NASA NAIF. Mars 2020 EDL 재구성 궤적 SPK (m2020_edl_v01.bsp) — [링크](https://naif.jpl.nasa.gov/pub/naif/MARS2020/kernels/spk/m2020_edl_v01.bsp)
- Povilaitis, R. Z., et al. SHAPEFILE_LROC_5TO20KM_CRATERS_SPOLE — 남극 5–20 km 크레이터 shapefile. *LROC RDR* — [링크](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LROC_5TO20KM_CRATERS_SPOLE)
- Robbins, S. J. (2018). Lunar Crater Database. *USGS Astrogeology* — [링크](https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018)
- Rodda, M. (2024). Chang'e 5 Landing Camera Crater Detection Dataset. *Zenodo* 11326450, CC BY 4.0 — [링크](https://zenodo.org/records/11326450)
- USGS Astrogeology. Mars 2020 TRN CTX Orthorectified Image Mosaic (JEZ_ctx_B_soc_008, 6 m) — [링크](https://astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0)

### 라. 코드·소프트웨어

- Airbus. SurRender 렌더러 (독점) — [링크](https://www.airbus.com/en/products-services/space/space-customer-support/surrendersoftware)
- connorlee77/seasonally-invariant-image-transform — Fragoso et al. 2021 학습 코드, MIT — [링크](https://github.com/connorlee77/seasonally-invariant-image-transform)
- cvg/LightGlue — 코드·가중치 Apache-2.0 (SuperPoint 가중치는 제한 라이선스) — [링크 1](https://github.com/cvg/LightGlue/blob/main/lightglue/lightglue.py), [링크 2](https://github.com/cvg/LightGlue)
- DOI-USGS/ISIS3 — USGS ISIS — [링크](https://github.com/DOI-USGS/ISIS3)
- HKUST-Aerial-Robotics/VINS-Mono — GPL-3.0 — [링크](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/vins_estimator/src/estimator.cpp)
- jpl-x/x — JPL x / xVIO (range-VIO), Apache-2.0 — [링크](https://github.com/jpl-x/x)
- kornia/kornia — `kornia.feature` (LoFTR, LightGlue, DISK, DeDoDe, ALIKED, XFeat, AdaLAM), Apache-2.0 — [링크](https://github.com/kornia/kornia/tree/main/kornia/feature)
- magicleap/SuperGluePretrainedNetwork — 비상업 전용 — [링크](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
- MIT-SPARK/Kimera-VIO — BSD-2 — [링크](https://github.com/MIT-SPARK/Kimera-VIO/blob/master/include/kimera-vio/frontend/Tracker-definitions.h)
- NASA GSFC. GIANT — Goddard Image Analysis and Navigation Tool (Surface Feature Navigation 포함), Apache-2.0 — [링크 1](https://github.com/nasa/giant), [링크 2](https://github.com/nasa/giant/blob/main/giant/relative_opnav/estimators/sfn/sfn_class.py)
- nasa/vira — NASA Vira 렌더러(C++) — [링크](https://github.com/nasa/vira)
- NeoGeographyToolkit/StereoPipeline — Ames Stereo Pipeline(ASP), Apache-2.0. 문서: Chang'e-3 예제, `sfs` 사용법 — [링크 1](https://stereopipeline.readthedocs.io/en/latest/examples/change3.html), [링크 2](https://github.com/NeoGeographyToolkit/StereoPipeline), [링크 3](https://stereopipeline.readthedocs.io/en/latest/sfs_usage.html)
- OmniLRS/OmniLRS — Isaac Sim 기반 달 로버 시뮬레이터, BSD-3 — [링크](https://github.com/OmniLRS/OmniLRS)
- OpenCV `imgproc.hpp` (`phaseCorrelate` response 정의), Apache-2.0 — [링크](https://raw.githubusercontent.com/opencv/opencv/4.x/modules/imgproc/include/opencv2/imgproc.hpp)
- rpng/open_vins — OpenVINS (MSCKF), GPL-3.0 — [링크](https://github.com/rpng/open_vins/blob/master/ov_msckf/src/update/UpdaterMSCKF.cpp)
- s-mcleod/crater_based_navigation_pipeline — STELLA 계열 CBPE (라이선스 파일 없음) — [링크](https://github.com/s-mcleod/crater_based_navigation_pipeline)
- silburt/DeepMoon — DEM 크레이터 검출 U-Net, MIT — [링크](https://github.com/silburt/DeepMoon)
- sirbastiano/FederNet — Mask R-CNN + 불변량 + EKF (라이선스 파일 없음) — [링크](https://github.com/sirbastiano/FederNet)
- STAR-Dundee. PANGU — Planet and Asteroid Natural Scene Generation Utility (ESA 과제 라이선스) — [링크](https://www.star-dundee.com/products/pangu-planet-and-asteroid-natural-scene-generation-utility/)
- UZ-SLAMLab/ORB_SLAM3 — GPL-3.0 — [링크](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Tracking.cc)
- vas-byte/Crater-Identification — Christian 사영 불변량 크레이터 식별의 제3자 구현, GPL-3.0 — [링크](https://github.com/vas-byte/Crater-Identification)
- wdoppenberg/crater-detection — Ellipse R-CNN 크레이터 검출과 패턴 매칭, MIT — [링크](https://github.com/wdoppenberg/crater-detection)
- zju3dv/LoFTR — Apache-2.0 — [링크](https://github.com/zju3dv/LoFTR/blob/master/src/loftr/utils/coarse_matching.py)

### 마. 언론·회사 발표

- Aerospace America (2025-07-22). How Intuitive Machines is planning to make its third moon landing a successful one — [링크](https://aerospaceamerica.aiaa.org/how-intuitive-machines-is-planning-to-make-its-third-moon-landing-a-successful-one/)
- CNN (2025-03-01). Blue Ghost mission moon landing — [링크](https://www.cnn.com/2025/03/01/science/blue-ghost-mission-moon-landing/index.html)
- NASA (2025-03-13). NASA Cameras on Blue Ghost Capture First-of-its-Kind Moon Landing Footage — [링크](https://www.nasa.gov/general/nasa-cameras-on-blue-ghost-capture-first-of-its-kind-moon-landing-footage/)
- NASA JPL (2024-12-11). NASA Performs First Aircraft Accident Investigation on Another World — [링크](https://www.jpl.nasa.gov/news/nasa-performs-first-aircraft-accident-investigation-on-another-world/)
- Space.com (2025-03-07). Private Intuitive Machines moon lander fell over inside crater at lunar south pole, photo reveals — [링크](https://www.space.com/the-universe/moon/private-intuitive-machines-moon-lander-fell-over-inside-crater-at-lunar-south-pole-photo-reveals)
- Space.com (2025-05-14). Here's why the private Athena lunar lander toppled over on the moon — [링크](https://www.space.com/astronomy/moon/heres-why-the-private-athena-lunar-lander-toppled-over-on-the-moon)
- Spaceflight Now (2025-05-14). Intuitive Machines CEO points to issues that prevented upright touchdown during IM-2 moon landing — [링크](https://spaceflightnow.com/2025/05/14/intuitive-machines-ceo-points-to-issues-that-prevented-upright-touchdown-during-im-2-moon-landing/)
- Tech Times (2026-06-27). Firefly Aerospace brings moon landing AI navigation in-house after Blue Ghost success — [링크](https://www.techtimes.com/articles/319197/20260627/firefly-aerospace-brings-moon-landing-ai-navigation-house-after-blue-ghost-success.htm)

### 바. 이 연구의 내부 자료

- KAIST 원문 추출 기록 (2026-10-06) — 1·2·3순위 37건과 빈칸 검색 D1–D6 — [링크](../source_extraction_2026-10-06.md)
- KAIST 원문 검증 기록 (2026-09-30) — 1차 검증 21건 — [링크](../source_verification_2026-09-30.md)
- 조사 노트: 비행 증거, 신뢰도 이론·지표, 민감도 실험, 참조 알고리즘, 데이터셋, 절대 TRN 분류 기준(2026-10-10) — [링크](../research_notes/비전%20TRN%20신뢰%20한계와%20알고리즘%20데이터셋/)
- sim2moon 코드·설정: 항법 플러그인 경계, 15상태 MEKF, 카메라 설정, de Gerlache 지형 계약 — [링크 1](/home/sim2real/sim2moon/robots/lander_navigation.py), [링크 2](/home/sim2real/sim2moon/robots/lander_navigation_ekf.py), [링크 3](/home/sim2real/sim2moon/config/environment/chandrayaan3_paired_dataset.yaml), [링크 4](/home/sim2real/sim2moon/config/terrain/de_gerlache_rim2_clean.json)

---

## 부록. 아직 원문을 확인하지 못한 자료 (2026-10-10 정리)

이 부록에는 원문을 아직 확인하지 못한 자료만 남겼다.

- **옮긴 자료**: 원문을 확인해 본문 작성과 사실 확인에 쓴 자료는 [출처](#출처)로 옮겼다. 1순위 11건, 2순위 19건, 3순위 6건이다.
- **확인 기록**: 각 자료에서 무엇을 확인했는지는 [`../source_extraction_2026-10-06.md`](../source_extraction_2026-10-06.md)(KAIST 2026-10-06)와 [`../source_verification_2026-09-30.md`](../source_verification_2026-09-30.md)(KAIST 2026-09-30)에 있다.
- **번호**: 이전 판과 같게 유지했다. 빠진 번호는 확인을 마쳐 출처로 옮긴 항목이다.

우선순위의 뜻은 다음과 같다.

- **1순위**: "어느 고도·어떤 조건까지"에 숫자를 직접 주는 자료
- **2순위**: 신뢰도 지표의 공식이나 기준선 재구현에 필요한 자료
- **3순위**: 맥락, 데이터셋, 부차 요인
- **★**: 2026-10-10의 계열별 기준선 비교 방향(질문 5)에 직접 필요한 자료. 다음 KAIST Primo 조사는 ★부터 한다.

접근 메모는 2026-09-30·2026-10-06의 접근 시도 결과다. "ARC"는 AIAA Aerospace Research Central을 뜻하며, KAIST 기관 프록시로 열람에 성공한 경로다.

**구조적으로 막힌 것**
- #3 LVS Field Test(AAS 2020): NTRS에 파일이 없다. 학회록 *Advances in the Astronautical Sciences* 인쇄본의 소장 권호를 확인하거나 국외 원문복사를 신청해야 한다.
- #11, #24, #41 SLIM 특집호: KAIST에 J-STAGE 구독이 없다. 국외 원문복사나 JSASS 문의로 받아야 한다.

### 1순위

| # | 서지 | DOI·식별자 (접근) | 확인할 것 |
|---|---|---|---|
| 3 | Johnson, A., Villaume, N., Umsted, C., Kourchians, A., Sternberg, D., Trawny, N., Cheng, Y., Geipel, E., Montgomery, J. (2020). "The Mars 2020 Lander Vision System Field Test." AAS GN&C 2020 | NTRS 20220000862 / 20210007612 (PDF 없음; AAS 학회록, 상호대차 필요 가능) | 600회 넘는 헬기 실시간 시험의 고도·지형·태양 조건별 오차, off-nominal 시험 |
| ★11 | 石田 貴行, 福田 盛介, 狩谷 和季, 澤井 秀次郎, 坂井 真一郎, 原 誠一, 日置 和之, 下城 洋人 (2025). 「SLIM画像照合航法のクレータマップ」. *Aerospace Technology Japan* 24: s113–s121 | doi:10.2322/astj.24.s113 (J-STAGE 구독자 인증. 2026-09-30 같은 특집호가 기관 프록시로도 막힘 → 상호대차·JSASS 문의) | 영역별 크레이터 지도 해상도, 매칭 비율과 영역별 임계, 지도 크레이터의 비행 영상 재현성과 조명 |

### 2순위

| # | 서지 | DOI·식별자 (접근) | 확인할 것 |
|---|---|---|---|
| 24 | 伊藤 琢博, 植田 聡史, 横田 健太朗, 坂井 真一郎, et al. (2025). 「SLIMピンポイント月着陸における誘導航法制御の開発およびフライト結果」. *Aerospace Technology Japan* 24: s103–s112 | doi:10.2322/astj.24.s103 (J-STAGE 인증) | GN&C 비행 결과, 500 m 아래 항법 소스, VBN 수용 논리 |
| ★29 | Theil, S., Ammann, N., Andert, F., et al. (2018). ATON (Autonomous Terrain-based Optical Navigation) flight test results. *CEAS Space Journal* 10:325–341 | doi:10.1007/s12567-018-0201-0 (Springer) | 크레이터·그림자·특징추적·3D 매칭의 UKF 융합. 다중 기준선 융합의 참고 |
| ★32 | Johnson, A. E., Aaron, S., et al. (2016). "Design and Analysis of Map Relative Localization for Access to Hazardous Landing Sites on Mars." AIAA SciTech 2016 | NTRS 20170008287 (PDF 없음; DOI 미확인) | 지도 해상도·조명별 MRL 몬테카를로 설계와 오차 예산 |
| ★36 | Sato, H., Robinson, M. S., Hapke, B., Denevi, B. W., Boyd, A. K. (2014). "Resolved Hapke parameter maps of the Moon." *JGR Planets* 119:1775–1805 | doi:10.1002/2013JE004580 (Wiley) | 남극 렌더에 상수로 쓸 Hapke 값, 고지와 바다의 값 |
| ★37 | McEwen, A. S. (1991). "Photometric functions for photoclinometry and other applications." *Icarus* 92:298–311 | doi:10.1016/0019-1035(91)90053-V | 셰이더용 Lunar-Lambert 위상 계수 |
| 38 | Gläser, P., Scholten, F., De Rosa, D., Marco Figuera, R., Oberst, J. (2014). "Illumination conditions at the lunar south pole using high resolution Digital Terrain Models from LOLA." *Icarus* 243:78–90 | doi:10.1016/j.icarus.2014.08.013 | 하강 태양 기하 표본화용 고해상도 남극 조명 시계열 |
| 39 | Barker, M. K., et al. (2020). "Improved LOLA Elevation Maps for South Pole Landing Sites: Error Estimates and Their Impact on Illumination Conditions." | NTRS 20205009660 (공개, 미열람 → 도서관 불필요) | DEM 오차가 예측 그림자로 번지는 정도. 렌더 지도가 하강 조명에 얼마나 맞을 수 있는지의 한계 |

### 3순위

| # | 서지 | DOI·식별자 (접근) | 확인할 것 |
|---|---|---|---|
| 41 | 植田 聡史, 伊藤 琢博, 横田 健太朗, 坂井 真一郎 (2025). 「ピンポイント月着陸のためのSLIM着陸降下シーケンス設計」. *Aerospace Technology Japan* 24: s91–s102 | doi:10.2322/astj.24.s91 (J-STAGE 인증) | VBN이 500 m에서 끝난 이유, 하강 태양 고도 설계 |
| 42 | Seybold, Chen, Bellutta, Johnson, et al. (2007). "Suborbital Flight Test of a Prototype Terrain-Relative Navigation System." AIAA 2007-2896 | doi:10.2514/6.2007-2896 | 2006 관측로켓 시험의 카메라·조명, 고도별 영상 품질 |
| 43 | Johnson, Willson, Goguen, Alexander, et al. (2005). "Field Testing of the Mars Exploration Rovers Descent Image Motion Estimation System." ICRA 2005 | doi:10.1109/ROBOT.2005.1570807 (JPL PDF `DIMES_ICRA05_FINAL.pdf`가 공개되어 있을 수 있음) | 시험 회차별 유효율(구름 그림자가 움직인 Kelso Run 3) |
| 45 | Dever, C., Hamilton, L., Truax, R., Wholey, L., Bergeron, K. (2017). "Guided-Airdrop Vision-Based Navigation." AIAA 2017-3723 | doi:10.2514/6.2017-3723 | IBAL 템플릿 DB 구성, 상관 세부, 실패·재포획 통계 |
| 46 | Trawny, N., Mourikis, A. I., Roumeliotis, S. I., Johnson, A. E., Montgomery, J. F. (2007). "Vision-aided inertial navigation for pin-point landing using observations of mapped landmarks." *J. Field Robotics* 24(5):357–378 | DOI 미확인 | 지도 랜드마크 EKF 갱신식 |
| 47 | Getchius, J., et al. (2022). "Hazard Detection and Avoidance for the Nova-C Lander." Proc. 44th AAS GN&C Conf., 921–943 | doi:10.1007/978-3-031-51928-4_53 (arXiv 2204.00660) | HRN 카메라 사용 고도와 조명 가정 |
| 48 | Kumar, S., Chakrabarti, D., Rallapalli, A., Bharat Kumar, G. V. P. (2026). "Real-Time Retargeting-Based Robust Polynomial Guidance for Chandrayaan-3 Lunar Landing Mission." *JGCD* 49(6):1759–1772 | doi:10.2514/1.G008733 | LPDC/LHVC 자료와 사용 고도가 보고되었는지 |
| 49 | Zhu, S., Xiu, Y., Zhang, N., Xu, R., Cui, P. (2020). "Crater-based attitude and position estimation for planetary exploration with weighted measurement uncertainty." *Acta Astronautica* 176:216–232 | doi:10.1016/j.actaastro.2020.06.033 | 크레이터 픽셀 크기·타원 적합별 중심 오차 공분산 |
| 51 | Zhu, C., Steinmetz, C., Belabbas, B., Meurer, M. (2019). "Six degrees-of-freedom dilution of precision for integrity of camera-based localization"; "Feature error model for integrity of pattern-based visual positioning." ION GNSS+ 2019 | DOI 미확인 | 랜드마크 배치의 DOP형 기하 인자 → 크레이터 기하 가용성 지표 |
| 52 | Johnson, A. E., Montgomery, J. F. (2008). "Overview of Terrain Relative Navigation Approaches for Precise Lunar Landing." IEEE Aerospace Conf. | doi:10.1109/AERO.2008.4526302 (2026-10-10 초록만 확인) | TRN 계열별 설계 운용 고도와 이유 |
| 53 | Henry, S., Christian, J. A. (2023). "Absolute Triangulation Algorithms for Space Exploration." *JGCD* 46(1):21–46 | arXiv 2205.12197 공개 (도서관 불필요) | 랜드마크 시선 위치 고정의 공분산 → 크레이터 DOP |
| 54 | Haydon, T., Huang, A., Humphreys, T. E. (2025). "Analysis of the Trusted Inertial Terrain-Aided Navigation Measurement Function." *NAVIGATION* 72(3) | navi.707 (오픈. 본문은 도구 요약으로만 확인) | TAN CRLB 유도 전체와 선형화가 유효한 지형 통계 임계 |
| 55 | (저자 미확인) (2019). "Path Planning of Underwater Terrain-Aided Navigation …" *Marine Technology Society Journal* 53(2) | Ingenta | terrain Fisher information content(TFIC)의 정확한 정의 |
| 56 | Cannon, M. W. (1974). "TERCOM Performance: Analysis and Simulation." | DTIC AD-783804 (다운로드 차단) | σ_T, 상관길이, 프로파일 길이, 잡음별 고전 오고정 확률 |
| 58 | Downes, L., Steiner, T. J., How, J. P. (2020). "Deep Learning Crater Detection for Lunar Terrain Relative Navigation." AIAA SciTech 2020 | doi:10.2514/6.2020-1838 (MIT DSpace 공개본 있음) | 잡음·밝기 강건성, 학습 데이터, 카탈로그 매칭 세부 |
| ★59 | Li, S., et al. (2025). "Lunar Visual Localization Method Based on Crater Geohash Encoding and Consistency Matching." *Remote Sensing* 17(9):1493 | doi:10.3390/rs17091493 (오픈액세스) | 빠른 크레이터 매칭 기준선의 알고리즘 세부 |
| ★60 | Del Prete, R., Renga, A. (2022). "A Novel Visual-Based Terrain Relative Navigation System for Planetary Applications Based on Mask R-CNN and Projective Invariants." *Aerotecnica Missili & Spazio* | doi:10.1007/s42496-022-00139-0 | FederNet 정확도와 운용 고도, 코드 라이선스를 요청할 수 있는지 |
| 62 | Chu, G. (2026). "Impact Mechanism and Quantitative Analysis of Elevation Errors in Lunar Crater-Based Visual Navigation." *J. Phys.: Conf. Ser.* | DOI 미확인 (CC BY) | DEM·크레이터 고도 오차 대 위치 오차 곡선 |
| 63 | Krüger, H., Theil, S., Sagliano, M., Hartkopf, S. (2014). "On-ground testing optical navigation systems for exploration missions." 9th ESA GNC Conf. | DLR elib 94964 | TRON 하드웨어 루프 CNav의 조명·궤적별 결과 |
| 64 | ESA GNC / ICATT 학회록 (GMV, Airbus DS, Thales Alenia; NPAL / PILOT / Argonaut) | esa-gnc.eu (특정 논문 미식별) | PANGU 기반 유럽 VBN의 태양 고도·지형·지도 오차 몬테카를로 강건성 |
| ★65 | Lorenz, D. A., et al. (2017). "Lessons learned from OSIRIS-REx autonomous navigation using natural feature tracking." IEEE Aerospace Conf. | DOI 미확인 | 소천체에서의 지도–외관 불일치 유사 사례 (선택) |
| 66 | Weisberger, J. M., et al. (2026). "Firefly Blue Ghost Mission 1: SCALPSS 1.1 Long Focal Length Camera Descent Imaging." AIAA 2026-1310 (같은 특집의 1308·1311은 2026-09-30에 원문 확인 → 출처) | doi:10.2514/6.2026-1310 | 장초점 카메라 모델·장착, 고도 대 프레임, 스테레오 DEM. 플룸은 부차 요인 |
| 67 | Hines, G., Amzajerdian, F., et al. (2025). "Demonstration of Navigation Doppler Lidar Capabilities Onboard the First Commercial Lunar Lander." AIAA 2025-1115; Pierrottet, D. F., Monaco, J. F., Sandford, S. P. (2025). "Navigation Doppler LiDAR Lunar Landing Data Analysis and Trajectory Reconstruction." AIAA 2025-1116 | doi:10.2514/6.2025-1115 | IM-1 궤적 재구성 방법과 정확도. CLPS SPK의 3σ 100 m와 비교 (두 편 모두 2026-09-30 원문 열람. 먼지 질문만 확인했고 궤적 재구성 정확도는 미확인) |
| 68 | Malin, M. C., et al. (2017). "The Mars Science Laboratory (MSL) Mast cameras and Descent imager …" *Earth and Space Science* 4:506–539 | doi:10.1002/2016EA000252 (오픈액세스, 브라우저) | MARDI 프레임 수, 프레임률, 노출, 고도 범위 |
| 69 | Liu, Z., et al. (2014). "Descending and landing trajectory recovery of Chang'e-3 lander using descent images." *National Remote Sensing Bulletin* 18:981–987 (중국어) | doi:10.11834/jrs.20144071 | CE-3 LCAM 고도 범위, 궤적 복원 방법과 정확도 |
| 70 | Wan, W., et al. (2019). "Descent trajectory recovery of Chang'E-4 lander based on descent images." *ISPRS Archives* XLII-2/W13:1457 | 오픈액세스 PDF | CE-4 궤적의 독립 검증 |
| 72 | Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., Torrence, M. H. (2011). "Illumination conditions of the lunar polar regions using LOLA topography." *Icarus* 211:1066–1081 | doi:10.1016/j.icarus.2010.10.030 | PGDA 78의 사이트 명명(SL2, Site11) 방법 |
| 73 | Povilaitis, R. Z., et al. (2018). "Crater density differences: Exploring regional resurfacing, secondary crater populations, and crater saturation equilibrium on the moon." *PSS* 162:41–51 | doi:10.1016/j.pss.2017.05.006 | 5–20 km 카탈로그의 남극 완전성 |
| 74 | Head, J. W., et al. (2010). "Global Distribution of Large Lunar Craters …" *Science* 329:1504–1507 | doi:10.1126/science.1195050 | 20 km 이상 카탈로그의 극지 처리 |
| 75 | Grethen, C., et al. (2026). "MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data." Proc. ACM MMSys 2026, 457–463 | doi:10.1145/3793853.3799814 (arXiv판은 열람) | 출판판 Hapke 값, 전체 데이터 호스팅 계획 |
| 76 | Shu, S., et al. *Commun. Earth Environ.* 7:61 보충자료; Liu, J., et al. (2019) *Nat. Commun.* 10 보충자료 | 본문은 열람 | 임무별 프레임–고도 표, 프레임별 재구성 궤적 |

### 빈칸 검색에서 나온 미열람 후보 (2026-10-06 KAIST 검색)

초록이나 DB 요약까지만 본 자료다. 서지는 [`../source_extraction_2026-10-06.md`](../source_extraction_2026-10-06.md) D1–D6에 있다.

| 서지 | DOI·식별자 | 확인할 것 |
|---|---|---|
| ★ Kaufmann, H., Lingenauber, M., Bodenmueller, T., Suppa, M. (2015). "Shadow-based matching for precise and robust absolute self-localization during lunar landings." IEEE Aerospace Conf. | doi:10.1109/AERO.2015.7119045 | 그림자 영역을 영상과 지도 양쪽에서 추출해 비교하는지 확인한다. 질문 5의 P-G 판정이 지금은 [추론]이다. 조명 범위별 성능도 본다 |
| ★ Kuga, T., Kojima, H. (2018). "Self-position estimation using terrain shadows for precise planetary landing." *Acta Astronautica* 148:345–354 | doi:10.1016/j.actaastro.2018.05.002 | 위와 같은 판정, 그림자 기반 위치추정의 고도·태양 고도 범위 |
| ★ Rothenberger, N., et al. (2026). "Stitching Illumination-Matched Maps: A LIMA-Based Pipeline for Robust Reference Map Generation." AIAA SciTech 2026 | 미확인 | 조명을 맞춘 기준 지도의 생성 절차. 기준선 레포의 지도 생성기(P2)에 쓴다 |
| Ansalone, L., Grava, E., Curti, F. (2015). "Experimental results of a TRN algorithm using a simulated lunar scenario." *Acta Astronautica* 116:78–92 | 미표시 | 축소 달 지형 실험의 조명·고도별 결과 |
| Zhang, L., Liu, N. (2024). "TRN in the Permanently Shadowed Regions at the Lunar Poles with DEM Data and SIFT Algorithm." ISAPE 2024 | doi:10.1109/ISAPE62431.2024.10840427 | 영구음영지역에서 DEM과 SIFT를 쓴 TRN 성능 |
| Shu (2026). 「基于光照控制的月球表面精确着陆点定位算法」. 航空学报 | 미표시 | 조명 제어 기반 착륙점 위치추정 |
| Inoue, T., Kojima, H. (2024). "Performance Evaluation of Lunar Lander Self-Positioning Estimation Method Improved by Incorporating Crater Size." *Trans. JSASS* | 미표시 | 크레이터 크기를 반영한 위치추정의 성능 |
| Yu, L. (2016). "A method for reliability analysis and measurement error modeling based on machine learning in scene matching navigation (SMN)." 西北工业大学学报 | 미표시 | 장면 매칭 신뢰도와 오차 모형 |
| Martin et al. (2013) IAC; Mammarella et al. (2011) IAC | 미표시 | 유럽 비전 항법의 조명·고도 결과 |
| 본문 §국내 선행연구의 KSAS·한국항공우주학회지 논문들 | RISS | 서지만 확인했다. 본문을 열람해 고도·조명 변수를 다룬 정도를 본다 |

### 공개본은 있으나 아직 읽지 않은 것

| 항목 | 공개 경로 | 확인할 것 |
|---|---|---|
| Carson et al. ALHAT 비행시험 (AIAA SPACE 2015) | NTRS 20170009170 | 본문의 인용은 요약 수준이다. 직접 내려받아 HRN 상관 품질 임계 수치를 확인한다 |

### 도서관에서 구할 수 없는 항목

- Intuitive Machines IM-2 이상 조사 보고서: 공개되지 않음
- Firefly / Space-ng Blue Ghost TRN 기술 논문: AIAA ARC 검색에서도 없음
- ESA ADCSS 2021 "PILOT latest developments, challenges and perspectives": 영상만 있음 ([Indico](https://indico.esa.int/event/393/contributions/6561/))
