# 원/달러 환율 시계열 분석 (2021~2026)

ECB 기준환율을 바탕으로 원/달러 일별 환율(2021-01-04 ~ 2026-09-30, 1,472영업일)을 분석해 추세, 상승·하락 국면의 비대칭, 급등락, 변동성, 달력 효과를 살펴본 프로젝트입니다. 보너스로 시계열 분해와 베이스라인 예측도 포함했습니다.

👉 **분석 결과는 [REPORT.md](REPORT.md)에 있습니다.**

## 폴더 구조

```
krw-usd-analysis/
├── data/
│   ├── usdkrw_daily.csv          # 정제 전 일별 환율 (date, usd_krw), 2020-07-01 ~ 2026-09-30
│   └── raw/usdkrw_<연도>.json     # API 원본 응답
├── images/                       # 리포트에 들어가는 그래프 8개
├── outputs/                      # 리포트에 인용한 수치 (summary.json, *.csv)
├── download_data.py              # 데이터 수집 스크립트
├── analysis.py                   # 정제·분석·시각화 스크립트
├── REPORT.md                     # 분석 리포트
├── requirements.txt
└── README.md
```

## 실행 방법

Python 3.10 이상이 필요합니다.

```bash
# 1) 가상환경 (선택)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 데이터 수집 (선택: data/usdkrw_daily.csv가 이미 포함돼 있음)
python download_data.py

# 4) 분석 실행
python analysis.py
```

`analysis.py`를 실행하면 `images/`에 그래프 8개가, `outputs/`에 결과 파일이 만들어지고, 주요 표가 콘솔에 출력됩니다. Anaconda를 쓴다면 `conda create -n krw python=3.11` 후 같은 순서로 실행하면 됩니다.

테스트 환경: Python 3.13, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2, python-dateutil 2.9.0

## 데이터 출처 · 수집 방법 · 라이선스

| 항목 | 내용 |
|---|---|
| 원천 | 유럽중앙은행(ECB) 유로 기준환율(euro foreign exchange reference rates). 매 영업일 14:10 CET경 집계 |
| 수집 경로 | [Frankfurter API](https://frankfurter.dev) (`/v1/{시작}..{끝}?base=USD&symbols=KRW`), API 키 불필요 |
| 값 | USD/KRW 교차환율 = (EUR/KRW) ÷ (EUR/USD), Frankfurter가 계산한 파생값 |
| 기간 | 2020-07-01 ~ 2026-09-30 수집, 분석은 2021-01-01 ~ 2026-09-30 |
| 라이선스 | [ECB 통계 재사용 정책](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)에 따라 무료 재사용 가능, 출처 표기 필요("Source: ECB statistics"). ECB는 기준환율을 정보 제공용으로만 공개하며 거래 목적 사용은 권장하지 않음 |

> 처음에는 미국 연준 FRED의 DEXKOUS 시리즈를 쓰려 했으나, 작업일(2026-10-22)에 FRED가 점검 중이어서 ECB 기준환율로 대체했습니다. 서울 외환시장 종가와의 차이는 REPORT.md 3.1절에 정리했습니다.

## 그래프 목록

| 파일 | 내용 |
|---|---|
| `01_trend_moving_average.png` | 환율 추이 + 20/120일 이동평균 + 주요 이벤트 |
| `02_swings_up_slow_down_fast.png` | 상승·하락 국면(7% 지그재그) 기간·속도 비교 |
| `03_daily_change.png` | 일별 변화율과 분포(정규분포 대비 두꺼운 꼬리) |
| `04_rolling_volatility.png` | 20일 롤링 변동성과 연간 변동성 |
| `05_monthly_heatmap.png` | 연도 × 월 월간 변화율 |
| `06_calendar_effects.png` | 월별·요일별 평균 변화율 |
| `07_decomposition.png` | [보너스 A] 추세·계절·잔차 분해 |
| `08_baseline_forecast.png` | [보너스 B] 베이스라인 예측 vs 실제 |
