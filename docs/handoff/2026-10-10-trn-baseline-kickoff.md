---
title: "달 절대 TRN 계열별 기준선 — 착수 핸드오프"
type: handoff
status: active
lang: ko
created: 2026-10-10
updated: 2026-10-10 (계열 분류 반영)
---

# 달 절대 TRN 계열별 기준선 — 착수 핸드오프

이 문서는 이 레포(`~/lunar-trn-baseline`)에서 첫 개발 세션을 시작하기 위한 인계 문서다. 다음 세션은 이 문서만 읽고 바로 착수할 수 있어야 한다. 끝의 §11에 바로 붙여 넣을 시작 프롬프트가 있다.

---

## 1. 이 레포가 하는 일

**목적: 어느 정도 검증된 달 절대 TRN 기준선을 계열별로 독립 개발하고 검증한다.**
- 첫 계열은 밝기 템플릿 매칭(T-I, LVS형)이다.
- 이어서 크레이터 패턴 매칭(P-G)과 밝기 특징 패턴 매칭(P-R)을 만든다.
- 탐색적 비교군으로 고도 템플릿 매칭(T-E)을 둔다.
- 계열 정의는 §2.1에 있다.

이 기준선들은 나중에 아래 본 연구에서 측정 도구로 쓴다.

- **본 연구**: 비전 기반 TRN을 어느 고도·어떤 조건까지 신뢰할 수 있는가를 정량적으로 분석한다.
  - 데이터는 sim2moon의 Unreal Engine(UE) 렌더로 만든 합성 데이터를 쓴다.
  - 결과는 가용성 A(h, c)와 무결성 I(h, c)의 곡면으로 낸다.
  - 본 연구는 이 레포 밖(`~/sim2moon-experiments`)에서 진행한다.
- **이 레포의 범위**
  - 알고리즘 구현
  - sim2moon과 무관한 외부 데이터로 검증
  - 나중에 sim2moon 항법 플러그인이 부를 수 있는 순수 Python 인터페이스
- **이 레포의 범위 밖**
  - sim2moon, Isaac Sim, UE에 대한 코드 의존. 이 레포는 그것들 없이 설치·시험되어야 한다.
  - 저고도(<3.5 km) 달 데이터. 본 연구에서 sim2moon이 직접 만든다.
  - 플룸·먼지 모델링

**왜 독립 레포인가**
- 기준선 검증이 sim2moon 렌더러에 순환적으로 의존하면 안 된다. 검증 데이터(StereoLunar)는 SurRender로 렌더한 외부 데이터셋이다.
- 알고리즘 코드를 시뮬레이터 레포와 분리하면, 나중에 실비행 데이터(DDL-F1, Mars 2020)에도 같은 코드를 그대로 돌릴 수 있다.

---

## 2. 이미 내린 결정과 근거

| 결정 | 근거 |
|---|---|
| **절대 TRN은 템플릿 매칭(T) 대 패턴 매칭(P), 그 안에서 밝기(I) 대 고도(E)·기하(G)·기술자(R)로 나눈다** (§2.1) | Rahuman & Lee 2026(IJASS) 리뷰의 이분법을 받되, 판정을 "정합 전에 양쪽에서 요소를 추출·기술하는가" 하나로 고정했다. 딥러닝, 렌더 여부, Image-on-Map/Map-on-Image는 계열이 아니라 속성이다 |
| **첫 기준선은 T-I: LVS형 2단 지도 매칭(MRL)이다** (거친 FFT·위상상관 → 정밀 NCC → PnP-RANSAC → 선택적 EKF) | 화성에서 비행했고(Mars 2020 LVS), NASA 달 TRN 계획(SPLICE, LuNaMaps)도 같은 방식이다. 매개변수가 원문에 공개되어 있다(§4). 학습 모델이 아니라 결정적이다. 본 연구에 필요한 신뢰 신호(피크 높이·폭, inlier 수)가 자연스럽게 나온다 |
| **주 검증 데이터는 StereoLunar다** | 달 남극 3.5–30.5 km, 영상마다 정답 카메라 자세. 같은 자세에 조명 3종이 있다. sim2moon과 다른 렌더러(SurRender + Hapke)다. CC BY 4.0 |
| **T-I의 교차 구현은 NASA GIANT 표면특징항법이다** | 같은 NCC 핵심부의 독립 공개 구현이다(Apache-2.0). LVS형은 Image-on-Map에 정사 지도를 쓰고, GIANT는 Map-on-Image에 실시간 렌더 템플릿을 쓴다. 그래서 둘의 비교가 속성 실험의 한 축도 된다 |
| **P-G(크레이터)는 T-I 다음 계열이다** | 쓸 만한 공개 검증 데이터(CRESENT-365)는 100 km 궤도용이고 CC BY-NC라 상업 과제에 걸린다. 그래서 StereoLunar 영상에 같은 DEM에서 뽑은 자체 카탈로그를 쓴다. Robbins 카탈로그(1–2 km 이상)로는 3.5 km 이하에서 크레이터가 금방 바닥난다 |
| **P-R(밝기 특징)은 고전판과 학습판을 함께 둔다** | 고전판은 SIFT/ORB + RANSAC, 학습판은 LightGlue(ALIKED/DISK)·LoFTR이다. 기성 학습 모델은 남극 저태양에서 방위각 차 90°를, SIFT는 60°를 넘으면 무너진다. 고전판이 계열의 하한 기준선이다. 학습 매처를 T-I 정밀 단계의 교체 부품으로 쓰는 실험은 그 뒤에 한다 |
| **T-E(고도 템플릿)는 탐색적 비교군이다** | 달 착륙 비행 이력은 없다. 근거는 지구 항공기 비행시험이다(국방과학연구소, Kim et al. 2025: SGM 스테레오 DEM + 평행이동 ICP, 3–5 km에서 σ 3.3–5.1 m). StereoLunar의 정답 깊이로 정합 절반을 먼저 검증할 수 있다 |
| SuperGlue, SuperPoint 가중치, FederNet, STELLA 코드는 쓰지 않는다 | SuperGlue·SuperPoint는 비상업·제한 라이선스, FederNet·STELLA는 라이선스 파일이 없다. LightGlue 코드·가중치와 DISK는 Apache-2.0, ALIKED는 BSD-3다 |

조사 근거 문서(로컬, 커밋 안 됨). 경로는 `~/sim2moon-experiments/docs/experiments/vision_trn_reliability/` 기준이다.

| 파일 | 내용 |
|---|---|
| `reports/비전 TRN 신뢰 한계와 알고리즘 데이터셋.md` | 주 보고서. 질문 5가 알고리즘·데이터 목록 |
| `source_extraction_2026-10-06.md` | KAIST 원문 추출. LVS 매개변수의 원문 인용이 여기에 있다(A1, A4, A6, D5) |
| `source_verification_2026-09-30.md` | 1차 원문 검증 기록 |
| `research_notes/비전 TRN 신뢰 한계와 알고리즘 데이터셋/absolute_trn_taxonomy.md` | **절대 TRN 분류 기준의 근거**(2026-10-10). 리뷰 원문 인용, 판정 사례, 리뷰와 다르게 판정한 곳 |
| `research_notes/비전 TRN 신뢰 한계와 알고리즘 데이터셋/reference_algorithms.md` | 기준선 후보와 공개 코드 목록(저장소, 라이선스) |
| `research_notes/비전 TRN 신뢰 한계와 알고리즘 데이터셋/datasets_and_resources.md` | 데이터셋 목록 |

### 2.1 절대 TRN 계열과 이 레포의 기준선 로드맵

**판정 규칙**: 정합 전에 관측과 지도 **양쪽에서** 기본 요소(크레이터, 키포인트, 능선)를 추출해 기술자나 불변량끼리 비교하는가?
- 아니오 → 템플릿 매칭(T). 원시 신호 패치를 유사도로 직접 비교한다.
- 예 → 패턴 매칭(P)

판정에서 주의할 점
- 랜드마크 DB를 쓴다고 패턴 매칭이 되는 것은 아니다. LVS 정밀 단계, GIANT SFN, Draper IBAL, OSIRIS-REx NFT는 패치를 상관하므로 T다.
- 하위 축은 관측 신호다. 밝기(I)인가, 고도(E)인가? 고도는 LiDAR·고도계, 또는 영상에서 복원한 DEM이다.

| 코드 | 계열 | 지도 | 이 레포의 기준선 | 하한을 정할 축 (가설) | 순서 |
|---|---|---|---|---|---|
| **T-I** | 밝기 템플릿 매칭 | 정사영상(렌더) 또는 DEM+알베도 | LVS형 2단 MRL(§4) + GIANT SFN | 지도/카메라 GSD 비, 지도–하강 조명 잔차 | 1 |
| **P-G** | 크레이터 패턴 매칭 | 크레이터 목록(같은 DEM에서 추출) | 추적형 예측-매칭-갱신(CNav TM형) + 타원 검출기. lost-in-space 초기화는 Hanak & Crain 삼각형(자체 구현)이나 Christian 불변량(GPL, 별도 프로세스) | 카탈로그 최소 크기, 시야 안 크레이터 수 | 2 |
| **P-R** | 밝기 특징 패턴 매칭 | 키포인트·기술자 DB 또는 정사영상 | SIFT/ORB + RANSAC, LightGlue(ALIKED/DISK)·LoFTR. 후단의 PnP-RANSAC은 T-I와 같은 것을 쓴다 | 기술자의 조명·축척 불변성 | 3 |
| **T-E** | 고도 템플릿 매칭 (탐색적) | DEM | 스테레오/SfM 복원 DEM + DEM 정규화 상관(Johnson & Ivanov 2011식) 또는 평행이동 ICP(Kim et al. 2025식) | DEM 실효 해상도, 기복 대비 복원 잡음 | 4 |

- **계열이 아닌 속성**: 실험 변수로 쓰고, 결과마다 기록한다.
  - 정합 구조: Image-on-Map / Map-on-Image
  - 기준 생성: 정사 사진 / 태양 맞춘 렌더 정사 / 실시간 시점 렌더
  - 사전 정보: lost-in-space / 추적형
  - 유사도: NCC / POC / MI / VCC
  - 구현: 수작업 / 학습
- **범위 밖**: 지도 없는 상대 항법(VO/VIO). 본 연구 쪽 대조군이다.
- **공통 하네스 원칙**
  - 모든 계열이 같은 `localize()`(§8)를 구현한다.
  - 지도는 모두 같은 원천 DEM에서 파생한다.
  - 사전 오차, 채점 규칙(§6), 원값 신뢰 신호 기록 방식(§7)을 같게 한다.
  - 영상당 계산 시간도 기록한다.
- **계열 비교에서 시험할 가설**
  - H-a: 질감이 적은 지형에서 T-I의 가용성이 P-G보다 높다. 리뷰의 정성 주장이다.
  - H-b: 크레이터가 많고 조명 잔차가 큰 조건에서 P-G의 무결성이 T-I보다 높다.
  - H-c: 시스템 하한은 T-I의 GSD 비 하한과 P-G의 카탈로그 하한 중 먼저 오는 쪽이다.

---

## 3. 데이터

### 3.1 StereoLunar — 주 검증 데이터 (2026-10-10 확인)

- **출처**
  - Zenodo record **21133765**: https://zenodo.org/records/21133765
  - 논문: Grethen et al., "Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset", ICCV 2025 Workshops, pp. 3751–3760, arXiv:2510.18172
  - 코드: https://github.com/clementinegrethen/StereoLunar
- **파일**: `nadir.zip` 3.73 GB, `oblique.zip` 4.19 GB, `dynamic.zip` 3.49 GB, `ReadMe.txt`, 논문 PDF. 라이선스는 CC BY 4.0이다.
- **구성** (ReadMe 원문 기준)
  - 영상마다 다음 세 파일이 있다.
    - `im_XXXXX.jpg`
    - `.npz`: `intrinsics`(3×3 K), `cam2world`(4×4, 카메라→월드), `width`, `height`, `metadata`(태양 방향, 렌더 매개변수, 고도, 궤적 유형 등)
    - `.exr`: 깊이 지도. 카메라 중심에서 시선 방향 거리(m), 광축 = +Z, 무효값은 inf
  - 스테레오 쌍은 짝수·홀수 연번이다(`im_00000 ↔ im_00001`). 이 레포는 단일 영상 절대 위치추정이므로 한 장씩 쓴다.
  - **조명 3종이 같은 기하로 연달아 있다.** (00000, 00001)은 조명 1, (00002, 00003)은 같은 자세의 조명 2, (00004, 00005)는 조명 3이다.
  - 궤적 유형은 셋이다.
    - Nadir: 수직 하강, 기선은 고도의 ~10%
    - Oblique: 기울어진 시점
    - Dynamic: 고도 ±30%, 롤 ±10° 등 변동
- **렌더 설정** (논문 원문 기준)
  - 지형은 "the LOLA South Pole DEM at 5 m/px"다. 고도 범위는 −4,350 ~ +1,850 m(달 평균 반경 기준)다.
  - 카메라 위치는 "uniformly sampled across the lunar south polar cap (from −90° to −87°) and the full longitude range"다.
  - 반사는 Hapke BRDF이며 "assume constant albedo"다.
  - 카메라는 시야각 45°, 512×512다. 일부는 1024²이고, 가우스 PSF가 걸려 있다.
  - 조명 (방위, 입사)은 측면 (150°, 160°), 머리 위 (250°, 20°), 역광 (360°, 165°)이다. 논문의 각도 정의는 직접 확인해야 한다.
  - 자세는 "camera-to-world poses in Moon-fixed Frame"이다.
  - 고도 대역별 GSD(512 px, 45° 기준):

    | 고도 (km) | 3.5 | 6.2 | 9.5 | 12.8 | 16.1 | 19.4 | 22.7 | 26.0 | 29.2 | 30.5 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | GSD (m/px) | 5.7 | 10.0 | 15.4 | 20.7 | 26.0 | 31.4 | 36.7 | 42.1 | 47.2 | 49.3 |

- **⚠️ 미확인 — 첫 작업으로 풀어야 한다**
  1. **"Moon-fixed Frame"이 정확히 무엇인가.** LOLA DEM은 MOON_ME(평균 지구/회전축) 좌표계다. MOON_PA나 IAU_MOON이면 표면에서 수백 m가 어긋날 수 있다(ME–PA 차는 ~0.8 km급). `npz` metadata와 GitHub 코드를 먼저 보고, §6의 V1 시험으로 확정한다.
  2. **원천 DEM 제품의 정확한 이름.** 후보는 LOLA PDS의 남극 5 m 극지 입체투영 LDEM(이름 예: `LDEM_875S_5M`, 남위 87.5° 이남)이다. 다만 StereoLunar 샘플링이 남위 87°까지이므로 더 넓은 제품일 수도 있다. LOLA PDS(https://pds-geosciences.wustl.edu/missions/lro/lola.htm)와 PGDA(https://pgda.gsfc.nasa.gov/)에서 범위와 고도 범위(−4,350 ~ +1,850 m)가 맞는 제품을 찾는다. 찾지 못하면 저자에게 문의한다.
  3. **Hapke 매개변수 값**: 탑재 지도를 렌더할 때 같은 값을 쓸지, 일부러 다르게 할지 정한다.

### 3.2 탑재 지도 원천

- LOLA 남극 5 m DEM: §3.1의 제품을 특정한 뒤 사용한다.
- 지도는 이 레포에서 직접 렌더한 정사영상이다. 같은 DEM을 Hapke(또는 Lunar-Lambert)로, 지정한 태양 방향에서 1/5/10/20 m/px로 렌더한다.
- 지도와 영상이 같은 DEM이므로 낙관 편향이 있다. 하지만 **이 레포의 목적은 구현 검증**이라 문제가 없고, 오히려 통제된 조건이다. "진실 ≠ 지도" 구성은 본 연구(sim2moon)에서 한다.
  - 참고: PGDA 78 clone DEM, PGDA 104 SfS DEM

### 3.3 선택 — 실비행 데이터 (같은 구현의 실영상 확인용)

- **DDL-F1** (Blue Origin New Shepard NS-13, 2020-10-13, 미국 텍사스) — 2026-10-10에 ZIP 목차와 README를 확인했다.
  - **받는 법**
    - CKAN API `https://data.nasa.gov/api/3/action/package_show?id=deorbit-descent-and-landing-flight-1-ddl-f1`의 리소스 URL(`https://data.nasa.gov/docs/legacy/DDL-F1_Dataset-20201013.zip`)은 S3 서명 URL로 302 리디렉션된다.
    - 크기는 **713,127,663 B**다.
  - **구성**: `Flight1_Catered_Dataset-20201013/`
    - `README.txt`
    - `Data/truth.csv` (16 MB, 100 Hz, `TIME_NANOSECONDS_TAI`, ECEF 위치·속도, 쿼터니언 `truth_quat_CON2ECEF[1..4]`, Breckenridge 정의·스칼라가 마지막)
    - `Data/dlc.csv` (IMU 증분 속도·각도)
    - `Data/commercial_lidar.csv` (4빔 거리·도플러)
    - `Images/image_<TAI ns>.000000.{png,pgm}` 각 410장, 1 Hz
  - **카메라** (README 원문)
    - Blackfly BFS-PGE-16S2M-CS, 1440×1080
    - 내부 행렬 A = [[4794.31, 0, 720.03], [0, 4800.63, 540.00], [0, 0, 1]]
    - 왜곡 k1 = −0.07577, k2 = 0.02939, k3 = −0.23080, p1 = −0.006624, p2 = 0.0003305
    - CAHVOR 모델도 있다.
    - 차량 항법 중심(CON)에서 카메라 위치 [−0.09649, 0.27776, −1.54774] m와 `att_dcm_CON_Camera`가 README에 있다.
  - **사건**: T0 = 1602596210210000000 ns (TAI)
    - 원점(apogee) T+246.6 s, 106.7 km
    - 감속(brakes) T+406.3 s, 6.2 km
    - 착지 T+447.8 s
    - 진실값의 정확도는 비행 1에서 ±0.25 m 1σ다(AIAA 2022-1829).
  - **영상 범위** (진실값으로 계산): 하강 영상 194장. 30 km 이하 68장, 5 km 이하 34장, 1 km 이하 15장, 마지막 ~17 m. 착지 직전 영상은 초점이 안 맞는다(표본 확인).
  - **태양**: NS-13은 방위 103.62°, 고도 6.65°였다. JPL 지도의 원천 영상은 이와 다른 조명에서 찍혔다(AIAA 2022-1830).
  - **탑재 지도**: 포함되지 않았다. JPL은 20 m/px와 3 m/px 지도를 썼다. USGS 항공영상(NAIP)과 3DEP DEM으로 직접 만들어야 한다.
  - **재현 목표**: JPL 결과는 두 가지다.
    - 매칭 수가 해상도 불일치로 줄어든다.
    - 마지막 갱신이 20 m/px에서 ~4,000 m, 3 m/px에서 ~750 m다(Blue Origin 최종보고서, NTRS 20210026314).
- **Mars 2020 LCAM**
  - PDS의 LCAM 영상(약 87장, 수량 재확인 필요)
  - NAIF `m2020_edl_v01.bsp`(위치)와 `m2020_edl_v01.bc`(자세): https://naif.jpl.nasa.gov/pub/naif/MARS2020/kernels/
  - 비행에 쓴 지도: USGS CTX 6 m 정사영상과 20 m DTM
  - "실영상 + 비행 지도 + 재구성 궤적"을 다 갖춘 유일한 세트다.

---

## 4. 알고리즘 사양 — 원문으로 확인된 LVS 매개변수

1차 출처는 오픈액세스다. 구현 전에 반드시 직접 읽는다.

- **Johnson et al. 2023**, "Implementation of a Map Relative Localization System for Planetary Landing", *JGCD* 46(4):618–637, doi:10.2514/1.G006780 — 알고리즘 본체
- **Johnson et al. 2022**, AIAA 2022-1214 — 비행 성능, 이상치 분류, 운용 범위
- **Johnson et al. 2015**, AIAA 2015-0851 — 현장시험, 엔트로피 정의, 200 매치 경계
- **Johnson et al. 2025**, AIAA 2025-2076 — inlier 20 게이트의 오탐률·검출률, 조명 지표

아래 인용은 `source_extraction_2026-10-06.md`의 원문 추출에서 가져왔다. 기호 일부는 PDF 추출에서 구조로 판독한 것이다.

| 단계 | 원문 매개변수 |
|---|---|
| 특징 선택 | "N 64 × 64 pixel subwindows … For coarse, N = 64; for fine, N = 150", "5 × 5 Harris corner detector" |
| 거친 단계 | "148 × 148 pixel coarse template"(와핑), "128 × 128 coarse template and 1024 × 1024 map"(FFT). Mars 2020: 영상 3장 × 큰 패치 5개를 12 m/px 지도와 매칭해 초기 3.2 km 오차를 ~200 m로 줄임 |
| 정규화 | "local spatial filter that removes the mean and divides by the standard deviation of a 21 × 21 window" |
| 정밀 단계 | "raster scans a 21 × 21 pixel template … across the 128 × 128 pixel search area". 영상당 작은 패치 최대 150개를 6 m/px 지도와 매칭 |
| 부화소 | "fitting a biquadratic function to the 3 × 3 correlation surface" |
| 피크 이상치 | "Peaks that are below a specified height threshold or above a specified width threshold are treated as outliers … Concave up peak fits and subpixel fits that have peaks more than 1 pixel from the match pixel are also rejected." **높이·폭 임계 수치는 비공개** → 직접 정하고 기록한다 |
| 거친 단계 기하 이상치 | "\|ax − mx\| < T and \|ay − my\| < T" (T 수치 비공개) |
| Mahalanobis 게이트 | d² = rᵀS⁻¹r < γ, γ = q3 + 1.5·(q3 − q1). 원문 예 γ = 6.07, 포함 확률 95.2% |
| 추정기 입력 | "a 100 landmark match count threshold used to limit processing by the estimator" (2022) |
| 갱신 게이트 | Mars 2020: "a threshold of 20 inliers was required for a position update". Johnson 2025: "at least 20 estimated inliers effectively removes false positives (2 × 10⁻⁸ false positive rate) while still achieving an 83% true positive rate", "tolerate outlier fractions exceeding 60% above a minimum number of about 40 candidate feature matches" |
| 모드 전환 | 거친 단계가 끝나면 landmark 총수를 "parameterized criteria"로 판정해 정밀 단계로 가거나 재시도. 정밀 단계 직전 "fine converging" 상태는 누적 inlier가 임계를 넘으면 진입하고, 한 장이라도 최소 inlier에 못 미치면 coarse를 재시도 (수치 비공개) |
| 지도 해상도 선택 | "The map resolution was set to 6 m/pixel so that it matched the LCAM pixel size at the middle of the LVS operational altitude interval (3 km)" |
| 참고 임계 | NCC 성공 0.35. Wang et al. 2026(CJA)이 "the matching threshold used by the Mars 2020 lander"로 인용한 2차 정보 |
| 운용 범위 (Mars 2020) | 엔트로피 ≥4, 기복 ≤150 m, 경사 ≤15°, **LCAM–지도 태양각 차 ≤20°**, 고도 2,000–4,200 m, 천저 이탈 ≤45°, 각속도 ≤50°/s |
| 비행 성능 (Mars 2020) | 첫 fine 영상 150 매치 → inlier 92. fine 영상 42장 모두 inlier ≥80. 99%ile 수평 오차는 coarse 끝 158 m, 6 s 16.0 m, 10 s 14.3 m. 잔차 RMS ~1.2 px |

**이 레포에서 정할 것** (원문에 없음)
- 피크 높이·폭 임계값
- 거친 단계 기하 임계 T
- 모드 전환 수치
- 측정 잡음 R(부화소 적합에서 유도)

→ 정한 값과 근거를 `docs/decisions.md`에 남긴다(파일은 아직 없음).

---

## 5. 구현 단계 제안

| 단계 | 내용 | 완료 조건 |
|---|---|---|
| **P0 환경·데이터** | Python 패키지 골격(`src/lunar_trn`), 의존성 고정. StereoLunar `nadir.zip` 일부와 DEM 확보. `data/`는 git 밖 | 표본 영상·npz·exr을 읽는 로더와 단위 시험 |
| **P1 좌표계 정합** | §3.1 미확인 1·2를 푼다. DEM → Moon-fixed 3D, 카메라 투영 | V1 통과 |
| **P2 지도 생성기** | DEM에서 정사지도(1/5/10/20 m/px)를 지정 태양으로 렌더. 지도 픽셀 ↔ 달 고정 좌표 변환 | 렌더 지도가 StereoLunar 영상과 같은 조명에서 시각적으로 일치 |
| **P3 거친 단계** | 큰 템플릿 와핑 → FFT 위상상관 → 기하 이상치 제거 → 위치 보정 | V2a 통과 |
| **P4 정밀 단계와 자세** | Harris 특징 → 21×21 NCC → 부화소 → 피크 이상치 → PnP-RANSAC → 위치·공분산 | V2b 통과 |
| **P5 게이트와 신뢰 신호** | §7의 신호를 모두 기록. inlier 20 게이트, Mahalanobis 게이트, 정답 / 고정 없음 / 오고정 판정 | V2c 통과 |
| **P6 조명 시험** | 같은 자세의 조명 3종 × 지도 조명 몇 종 | V3 보고 |
| **P7 교차 구현** | NASA GIANT SFN으로 같은 입력 처리 | V4 보고 |
| **P8 (선택) 실비행** | DDL-F1 지도 제작 → JPL 결과 재현 | V5 |
| **P9 (선택) 시퀀스 필터** | 연속 영상 EKF(IMU 없이 등속 모델 또는 정답 기반 사전) | 필요할 때 |
| **P10 인계** | sim2moon 항법 플러그인용 인터페이스 고정(§8) | 인터페이스 문서와 예제 |

**계열 확장 (T-I의 P5까지 끝난 뒤)**

| 단계 | 내용 | 완료 조건 |
|---|---|---|
| **F1 공통 하네스** | T-I 코드에서 계열 공통부(좌표계, 지도 파생, 채점, 신호 기록)를 분리한다. 계열 무관 실행기: 같은 사전 오차 표본, 채점, 신호 기록, 계산 시간. `trn_map`이 정사 타일, DEM, 크레이터 목록, 특징 DB를 같은 DEM에서 파생 | T-I가 하네스 위에서 V2c를 그대로 재현 |
| **F2 P-G** | DEM에서 크레이터 카탈로그 추출(DeepMoon식, MIT, 또는 고전 림 검출) → 영상 타원 검출 → 추적형 매칭 → PnP. lost-in-space 초기화 | V7 |
| **F3 P-R** | 고전 SIFT/ORB → 학습 LightGlue(ALIKED/DISK)·LoFTR. 같은 PnP 후단 | V8 |
| **F4 T-E (탐색적)** | ① StereoLunar 정답 깊이로 DEM을 만들어 정합만 검증 ② 스테레오 쌍 SGM 복원 깊이로 교체 | V9 |
| **F5 계열 비교** | 같은 StereoLunar 표본에서 네 계열을 실행. H-a/H-b/H-c와 조명 3종 결과 보고 | 고도 대역 × 조명별 가용성·무결성 표 |

---

## 6. 검증 시험과 합격 기준

| ID | 시험 | 합격 기준 |
|---|---|---|
| **V1 좌표계** | 정답 자세로 DEM 점을 영상에 투영한다. ① 깊이 지도(exr)와 DEM 광선 교차 거리를 비교하고, ② 같은 조명으로 렌더한 지도와 실영상의 에지가 겹치는지 본다 | 깊이 차의 중앙값 < DEM 1–2 픽셀(5–10 m). 계통 오프셋이 수백 m면 좌표계 불일치(ME/PA)를 의심 |
| **V2a 거친 단계** | 조명이 일치하는 조건에서, 정답에 수평 ±3 km·수직 ±60–120 m 초기 오차를 더해 출발 | 수평 오차 99%ile ≲ 200 m. LVS의 158 m가 기준 |
| **V2b 정밀 단계** | V2a 뒤 단일 영상 PnP | 수평 오차가 지도 픽셀 2–3개 이내(고도 대역별로 보고). inlier ≥20이 대부분 |
| **V2c 무결성** | 같은 조건에서 반복 실행(대역별 수백–수천 회) | inlier ≥20 게이트를 통과한 오고정(오차 > 허용치)이 0건이고, 그 신뢰구간 상한(≈3/N)을 함께 보고 |
| **V3 조명** | 지도 조명 고정 × 영상 조명 3종 | 대역별 가용성과 오차 저하를 보고. 문헌 경향(LVS 태양각 차 ≤20°, 고전 매처 30–60°에서 붕괴)과 방향이 맞는지 |
| **V4 교차 구현** | GIANT SFN과 같은 입력 | 조명 일치 조건에서 위치 오차가 같은 자릿수 |
| **V5 실비행 (선택)** | DDL-F1 | 매칭 수가 해상도 불일치에 따라 줄고, 종료 고도가 JPL 값(20 m/px ~4 km, 3 m/px ~750 m)과 같은 자릿수 |
| **V6 렌더러 교차 (본 연구 쪽)** | 같은 자세를 sim2moon UE로 렌더해 같은 기준선 실행 | 두 렌더러의 성능 차이를 보고. 이 값이 본 연구의 렌더러 편향 항이 된다 |
| **V7 P-G** | 조명 일치 조건, 같은 사전 오차. 추적형과 lost-in-space를 따로 실행 | 정답 고정 시 오차가 지도 픽셀 수 개 이내. 매칭된 크레이터 수가 고도에 따라 줄어드는 곡선 보고. 오고정 0건과 신뢰구간 상한 |
| **V8 P-R** | V3과 같은 조명 3종 | 방위각 차가 커질 때 고전판이 학습판보다 먼저 무너지는지(문헌: SIFT 60°, 기성 LoFTR 90°) |
| **V9 T-E** | ① 정답 깊이 DEM ② 복원 깊이 DEM | ①에서 수평 오차가 DEM 픽셀 수 개 이내. ②에서 저하량을 복원 오차 기여로 보고 |

**채점 규칙**: 고정마다 **정답 / 고정 없음 / 오고정** 셋 중 하나로 판정한다. 오고정 허용치는 지도 픽셀 수(예: 3 px)와 미터로 함께 기록한다.

---

## 7. 반드시 기록할 신뢰 신호

본 연구는 이 값들의 원값으로 고도대별 ROC를 그린다. 판정만 남기지 말고 원값을 남긴다.

- **영상·장면**
  - 히스토그램 엔트로피(LVS 정의 E = −Σ p log p)
  - 영상 기울기 공분산 Σ∇의 λ_min과 tr(Σ∇⁻¹)
  - 그림자 비율
  - 카메라 GSD ÷ 지도 GSD
- **거친 단계**
  - 패치별 위상상관 피크(OpenCV `phaseCorrelate` response)
  - 피크 대 사이드로브 비(PSR)
  - 부피크/주피크 비
  - 기하 이상치 수
- **정밀 단계**
  - 시도한 특징 수, 매칭 수
  - 패치별 NCC 피크 높이·폭
  - 부화소 오프셋
  - 이상치 사유별 개수(피크 모양, 낮음·넓음, 기하, Mahalanobis)
- **자세**
  - inlier 수·비
  - 재투영 RMS
  - PnP 공분산
  - 사전 대비 보정량
- **채점**
  - 정답 대비 수평·수직 오차
  - 정답 / 고정 없음 / 오고정 판정
  - 고도, 태양 방향, 궤적 유형

---

## 8. sim2moon 연동 인터페이스 (P10에서 고정, 지금은 초안)

sim2moon 항법기는 교체할 수 있다(2026-10-10 확인).

- **교체 지점**: `robots_settings.landers.<i>.navigation.plugin = "module:Class"`. 프로토콜은 `~/sim2moon/robots/lander_navigation.py`의 `Navigator`다.
- **기본 필터**: 15상태 부분 오차상태 곱셈형 EKF, `robots/lander_navigation_ekf.py:EKFNavigator`
- **카메라 역할 없음**: 항법 센서 역할은 `NAV_ROLES = ("imu", "attitude", "altimeter", "doppler")`뿐이다. TRN을 연결하려면 sim2moon 쪽에 카메라 역할과 측정 갱신을 추가해야 한다. 이것은 본 연구의 작업이다.

이 레포는 그 안에서 측정 함수처럼 불릴 수 있어야 하고, Isaac·UE 의존은 두지 않는다.

```python
# Draft interface (English names; decide details in P10)
result = localize(
    image,            # HxW float32, radiometrically linear if possible
    K,                # 3x3 intrinsics
    prior_T_world_cam,  # 4x4 prior pose
    prior_cov,        # 6x6 prior covariance (position + attitude)
    trn_map,          # onboard map object (ortho tiles + DEM + metadata)
)
# result.status in {"fix", "no_fix"}
# result.T_world_cam, result.cov (6x6)
# result.landmarks: per-landmark pixel, map 3D point, correlation stats
# result.health: dict of the raw signals listed in section 7
# result.family: "T-I" | "T-E" | "P-G" | "P-R" (section 2.1)
```

모든 계열이 이 시그니처를 구현한다. 계열마다 다른 신호(크레이터 수, 기술자 점수, ICP 잔차)는 `health` 안에 계열 접두어를 붙여 기록한다.

EKF 갱신은 랜드마크 픽셀 단위로 넣는 것이 가장 깔끔하다. R은 상관 부화소 적합에서 유도한다. 위치 갱신만 넣으려면 PnP 공분산을 쓴다.

---

## 9. 코드·레포 규칙

- **언어와 도구**: Python 3.10+.
  - 수치·영상: numpy, scipy, opencv-python
  - DEM·영상 입출력: rasterio 또는 GDAL, OpenEXR(또는 imageio)
  - 시험: pytest
  - 좌표계: 필요하면 spiceypy
- **주석·docstring·로그 메시지는 영어로 쓴다**(사용자 규칙). `docs/`는 한국어 가능.
- **데이터와 대용량 산출물은 커밋하지 않는다**(`data/`, `outputs/`, `*.zip`, `*.exr`는 .gitignore).
- **라이선스 위생**
  - GPL 코드(OpenVINS, Christian 식별기 구현 등)는 이 레포에 섞지 않는다. 필요하면 별도 프로세스로 쓴다.
  - 라이선스 파일이 없는 저장소의 코드는 가져오지 않는다.
  - SuperGlue와 SuperPoint 가중치는 쓰지 않는다.
- 커밋·푸시는 사용자가 요청할 때만 한다.

---

## 10. 열린 질문

1. StereoLunar의 좌표계와 원천 DEM (§3.1) — **P1 전에 반드시 해결**
2. 정사지도 렌더러를 무엇으로 할 것인가
   - 간단한 Hapke/Lunar-Lambert 셰이딩 + 그림자 광선 추적을 직접 구현한다(권장, 의존성 최소).
   - Blender Cycles를 쓴다.
   - NASA Vira를 쓴다.
3. 렌더 지도 그림자: 영역 밖에서 드리우는 그림자 때문에 지도 타일보다 넓은 DEM 맥락이 필요하다(LuNaMaps 권고).
4. StereoLunar 조명 각도의 정의(방위 기준, "incidence" 정의)
5. GIANT를 그대로 쓸 수 있는가, 아니면 하강용으로 손봐야 하는가. GIANT는 궤도·소천체 OpNav용으로 만들어졌다.
6. P-G 카탈로그 추출 방법: DeepMoon(학습, MIT)으로 할지, DEM 곡률·림 검출 같은 고전 방법으로 할지. 추출 방식이 곧 P-G의 하한을 정하므로 F2 전에 정한다.
7. T-E의 복원 방식: StereoLunar 쌍(기선 ~10% 고도) SGM으로 할지, 연속 영상 SfM으로 할지

---

## 11. 다음 세션 시작 프롬프트

```text
~/lunar-trn-baseline 레포에서 작업한다. 먼저 docs/handoff/2026-10-10-trn-baseline-kickoff.md를 끝까지 읽어라.
이번 세션 목표는 P0와 P1이다.
(1) Python 패키지 골격과 의존성을 만든다.
(2) StereoLunar(Zenodo 21133765)에서 nadir.zip을 받아 data/에 둔다. 일부만 받을 수 있으면 그렇게 한다.
(3) 영상·npz·exr 로더를 만들고 단위 시험을 단다.
(4) 핸드오프 §3.1의 미확인 1·2를 푼다. 즉 "Moon-fixed Frame"의 정체와 원천 LOLA 남극 5 m DEM 제품을 특정해 받는다.
(5) V1 좌표계 시험을 구현해 결과를 docs/에 남긴다.
코드 주석은 영어로 쓴다. 데이터는 커밋하지 않는다. 커밋은 내가 요청할 때만 한다.
```
