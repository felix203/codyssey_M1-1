"""
원/달러 환율 시계열 분석 (분석 기간: 2021-01-01 ~ 2026-09-30)

실행 순서
  1) python download_data.py   -> data/usdkrw_daily.csv
  2) python analysis.py        -> images/*.png, outputs/*.csv, outputs/summary.json

적용 기법
  - 이동평균(20/60/120 영업일), 일별 변화율, 20일 롤링 변동성(연율화)
  - 구간별(연도별) 통계, 월별/요일별 집계
  - 상승/하락 국면(스윙) 구분: 7% 되돌림 기준 지그재그로 국면별 기간·속도 비교
  - [보너스 A] 월평균 환율의 고전적 가법 분해(2x12 중심이동평균 추세 + 월별 계절지수 + 잔차)
  - [보너스 B] 베이스라인 예측(단순 지속, 드리프트, 20일 평균) 및 오차 비교

외부 의존성은 pandas / numpy / matplotlib 뿐이다(statsmodels 없이 분해를 직접 구현).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.easter import easter
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "usdkrw_daily.csv"
IMG_DIR = BASE_DIR / "images"
OUT_DIR = BASE_DIR / "outputs"

START, END = "2021-01-01", "2026-09-30"
TRADING_DAYS = 252      # 변동성 연율화 기준
HOLDOUT = 40            # 예측 평가 구간: 마지막 40영업일(약 2개월)
Z_THRESHOLD = 5.0       # 이상치 후보: 일별 변화율의 로버스트 z-score 절대값 > 5
SWING_THRESHOLD = 0.07  # 상승/하락 국면 구분: 직전 고점(저점)에서 7% 이상 되돌리면 국면 전환
LEVEL_LINE = 1400       # 시장에서 자주 언급되는 심리적 기준선

SOURCE_NOTE = "자료: ECB 유로 기준환율 기반 USD/KRW 교차환율 (Frankfurter API) · 분석: analysis.py"

# 주요 이벤트(리포트 '이벤트 대조'에 사용). 날짜는 해당 이벤트가 ECB 기준환율(매 영업일 14:10 CET경 집계)에 처음 반영되는 영업일.
EVENTS: list[tuple[str, str]] = [
    ("2022-10-21", "2022년 고점: 미 연준 급격한 금리 인상기"),
    ("2022-11-11", "미 CPI 둔화 발표 다음 날 (-4.5%, 기간 최대 낙폭)"),
    ("2024-12-04", "비상계엄 선포(12/3 밤) 직후"),
    ("2025-04-10", "미 상호관세 90일 유예 발표 직후"),
    ("2025-12-24", "외환당국 고강도 구두개입·세제 대책 (-2.2%)"),
    ("2026-07-01", "기간 최고점: 중동 긴장·외국인 주식 매도"),
    ("2026-07-16", "한국은행 기준금리 인상(2.75%)"),
]

# dataviz 기본 팔레트(라이트 모드)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
UP, DOWN = "#e34948", "#2a78d6"          # 환율 상승(원화 약세) / 하락(원화 강세)
CONTEXT = "#bdbcb6"                      # 배경 맥락용 회색
DIVERGING = LinearSegmentedColormap.from_list(
    "krw_div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19b9a", "#c23534"]
)
WEEKDAYS = ["월", "화", "수", "목", "금"]


def setup_style() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic", "Noto Sans KR"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "savefig.dpi": 150,
    })


def save(fig: plt.Figure, fname: str, title: str, subtitle: str | None = None) -> None:
    """제목/부제/출처를 일정한 위치에 붙여 저장한다."""
    h = fig.get_figheight()
    ty = 1 - 0.14 / h
    fig.text(0.012, ty, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    top = ty - 0.36 / h
    if subtitle:
        fig.text(0.012, ty - 0.34 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
        top = ty - 0.62 / h - 0.17 * subtitle.count("\n") / h
    fig.text(0.012, 0.08 / h, SOURCE_NOTE, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.30 / h, 1, top))
    fig.savefig(IMG_DIR / fname)
    plt.close(fig)


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 1. 데이터 로드 · 기본 정보 · 결측치
# ---------------------------------------------------------------------------
def target_holidays(years) -> pd.DatetimeIndex:
    """ECB(TARGET) 휴무일: 1/1, 성금요일, 부활절 월요일, 5/1, 12/25, 12/26."""
    days = []
    for y in years:
        e = pd.Timestamp(easter(y))
        days += [pd.Timestamp(y, 1, 1), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1),
                 pd.Timestamp(y, 5, 1), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26)]
    return pd.DatetimeIndex(days)


def load_and_clean() -> tuple[pd.Series, pd.Series, dict]:
    raw = pd.read_csv(DATA_CSV)
    info: dict = {"columns": list(raw.columns), "raw_rows": int(len(raw))}
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["usd_krw"] = pd.to_numeric(raw["usd_krw"], errors="coerce")

    info["unparsable_dates"] = int(raw["date"].isna().sum())
    info["duplicated_dates"] = int(raw["date"].duplicated().sum())
    raw = raw.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").set_index("date")

    full = raw["usd_krw"]
    period = full[START:END]
    info.update({
        "period_start": fmt_date(period.index.min()),
        "period_end": fmt_date(period.index.max()),
        "n_obs": int(period.notna().sum()),
        "nan_values": int(period.isna().sum()),
        "nonpositive_values": int((period <= 0).sum()),
        "weekend_rows": int((period.index.dayofweek >= 5).sum()),
    })

    # 결측 영업일: 월~금 달력과 대조 -> ECB 휴무일인지 확인
    weekdays = pd.bdate_range(START, END)
    missing = weekdays.difference(period.dropna().index)
    hol = target_holidays(range(int(START[:4]), int(END[:4]) + 1))
    info["missing_weekdays"] = int(len(missing))
    info["missing_on_ecb_holiday"] = int(missing.isin(hol).sum())
    info["missing_other_dates"] = [fmt_date(d) for d in missing if d not in hol]

    # 처리: 값이 NaN인 행은 제거(보간하지 않음). 휴무일은 '거래가 없던 날'이므로 채우지 않는다.
    full = full.dropna()
    full = full[full > 0]
    return full, full[START:END], info


# ---------------------------------------------------------------------------
# 2. 이상치 점검
# ---------------------------------------------------------------------------
def check_outliers(full: pd.Series, period: pd.Series) -> tuple[pd.DataFrame, dict]:
    chg = (full.pct_change() * 100)[START:END].dropna()
    med = chg.median()
    robust_sigma = 1.4826 * (chg - med).abs().median()
    z = (chg - med) / robust_sigma
    nxt = chg.shift(-1)
    # 데이터 오류형 패턴: 크게 튄 다음 날 거의 같은 크기로 되돌아감(spike-and-revert)
    reverted = (np.sign(nxt) == -np.sign(chg)) & (nxt.abs() >= 0.8 * chg.abs())
    flagged = z.abs() > Z_THRESHOLD

    # 수준 기준 점검: 전후 21영업일 중앙값과 5% 이상 괴리
    med21 = period.rolling(21, center=True, min_periods=11).median()
    level_dev = (period / med21 - 1) * 100

    table = pd.DataFrame({
        "usd_krw": period.reindex(chg.index), "change_pct": chg, "robust_z": z,
        "next_day_change_pct": nxt, "spike_and_revert": reverted,
        "level_dev_from_21d_median_pct": level_dev.reindex(chg.index),
    })[flagged]
    table["decision"] = np.where(table["spike_and_revert"], "검토 필요(되돌림)", "실제 시장 움직임으로 보고 유지")
    info = {
        "robust_sigma_pct": round(float(robust_sigma), 4),
        "flagged_count": int(flagged.sum()),
        "flagged_spike_and_revert": int((flagged & reverted).sum()),
        "level_dev_over_5pct": int((level_dev.abs() > 5).sum()),
        "max_abs_level_dev_pct": round(float(level_dev.abs().max()), 2),
    }
    return table.round(3), info


# ---------------------------------------------------------------------------
# 3. 지표 계산
# ---------------------------------------------------------------------------
def build_frame(full: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"rate": full})
    f["chg"] = f["rate"].pct_change() * 100
    for w in (20, 60, 120):
        f[f"ma{w}"] = f["rate"].rolling(w).mean()
    f["vol20"] = f["chg"].rolling(20).std() * np.sqrt(TRADING_DAYS)
    f["ret20"] = f["rate"].pct_change(20) * 100
    return f[START:END].copy()


def yearly_stats(full: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    year_end = full.groupby(full.index.year).last()
    g = df.groupby(df.index.year)
    out = pd.DataFrame({
        "n": g["rate"].size(),
        "start": g["rate"].first(),
        "end": g["rate"].last(),
        "mean": g["rate"].mean(),
        "min": g["rate"].min(),
        "min_date": g["rate"].idxmin().map(fmt_date),
        "max": g["rate"].max(),
        "max_date": g["rate"].idxmax().map(fmt_date),
        "range": g["rate"].max() - g["rate"].min(),
        "vol_ann_pct": g["chg"].std() * np.sqrt(TRADING_DAYS),
        "days_abs_chg_over_1pct": g["chg"].apply(lambda x: int((x.abs() > 1).sum())),
        "share_above_ma120_pct": g.apply(lambda x: (x["rate"] > x["ma120"]).mean() * 100),
        "share_days_ge_1400_pct": g["rate"].apply(lambda x: (x >= LEVEL_LINE).mean() * 100),
    })
    out["yoy_change_pct"] = [(year_end[y] / year_end[y - 1] - 1) * 100 for y in out.index]
    out.index.name = "year"
    return out.round(2)


def distribution_stats(df: pd.DataFrame) -> dict:
    c = df["chg"].dropna()
    sd = c.std()
    up, down = c[c > 0], c[c < 0]
    return {
        "mean_pct": round(float(c.mean()), 4), "std_pct": round(float(sd), 4),
        "skew": round(float(c.skew()), 3), "excess_kurtosis": round(float(c.kurt()), 3),
        "share_beyond_2sd_pct": round(float((c.abs() > 2 * sd).mean() * 100), 2),
        "normal_expected_beyond_2sd_pct": 4.55,
        "share_beyond_3sd_pct": round(float((c.abs() > 3 * sd).mean() * 100), 2),
        "normal_expected_beyond_3sd_pct": 0.27,
        "up_days_share_pct": round(float((c > 0).mean() * 100), 2),
        "mean_up_move_pct": round(float(up.mean()), 4), "mean_down_move_pct": round(float(down.mean()), 4),
        "max_up_pct": round(float(c.max()), 3), "max_up_date": fmt_date(c.idxmax()),
        "max_down_pct": round(float(c.min()), 3), "max_down_date": fmt_date(c.idxmin()),
    }


def acf(x: pd.Series, lags: int = 10) -> dict:
    x = x.dropna().to_numpy()
    x = x - x.mean()
    denom = (x ** 2).sum()
    vals = {k: round(float((x[k:] * x[:-k]).sum() / denom), 4) for k in range(1, lags + 1)}
    return {"acf": vals, "bound_95": round(1.96 / np.sqrt(len(x)), 4)}


def ma_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    d = (df["ma20"] > df["ma120"]).astype(int)
    cross = d.diff().fillna(0)
    rows = [(fmt_date(t), "상향 돌파(골든크로스)" if v > 0 else "하향 돌파(데드크로스)", round(df.loc[t, "rate"], 2))
            for t, v in cross[cross != 0].items()]
    return pd.DataFrame(rows, columns=["date", "type", "usd_krw"])


def swing_analysis(s: pd.Series, thr: float = SWING_THRESHOLD) -> tuple[pd.DataFrame, dict]:
    """지그재그: 직전 극값에서 thr 이상 반대로 움직이면 그 극값을 국면 전환점으로 확정."""
    piv: list[tuple[pd.Timestamp, float]] = []
    hi_t = lo_t = s.index[0]
    hi_v = lo_v = s.iloc[0]
    direction, ext_t, ext_v = 0, None, None
    for t, v in s.iloc[1:].items():
        if direction == 0:
            if v > hi_v:
                hi_t, hi_v = t, v
            if v < lo_v:
                lo_t, lo_v = t, v
            if hi_v / lo_v - 1 >= thr:
                if lo_t < hi_t:
                    piv.append((lo_t, lo_v)); direction = 1; ext_t, ext_v = hi_t, hi_v
                else:
                    piv.append((hi_t, hi_v)); direction = -1; ext_t, ext_v = lo_t, lo_v
        elif direction == 1:
            if v > ext_v:
                ext_t, ext_v = t, v
            elif v <= ext_v * (1 - thr):
                piv.append((ext_t, ext_v)); direction = -1; ext_t, ext_v = t, v
        else:
            if v < ext_v:
                ext_t, ext_v = t, v
            elif v >= ext_v * (1 + thr):
                piv.append((ext_t, ext_v)); direction = 1; ext_t, ext_v = t, v
    piv.append((ext_t, ext_v))           # 마지막 극값은 아직 7% 반전이 없어 '미확정'
    pos = {t: i for i, t in enumerate(s.index)}
    rows = []
    for k, ((t0, v0), (t1, v1)) in enumerate(zip(piv[:-1], piv[1:])):
        n = pos[t1] - pos[t0]
        chg = (v1 / v0 - 1) * 100
        rows.append({"start": fmt_date(t0), "end": fmt_date(t1), "from": round(v0, 2), "to": round(v1, 2),
                     "direction": "상승(원화 약세)" if chg > 0 else "하락(원화 강세)", "change_pct": round(chg, 2),
                     "trading_days": n, "pct_per_day": round(chg / n, 3),
                     "confirmed": k < len(piv) - 2})
    sw = pd.DataFrame(rows)
    up, dn = sw[sw["change_pct"] > 0], sw[sw["change_pct"] < 0]
    info = {
        "threshold_pct": round(thr * 100, 1), "n_up": int(len(up)), "n_down": int(len(dn)),
        "up_avg_days": round(float(up["trading_days"].mean()), 1), "down_avg_days": round(float(dn["trading_days"].mean()), 1),
        "up_avg_pct_per_day": round(float(up["pct_per_day"].mean()), 3),
        "down_avg_pct_per_day": round(float(dn["pct_per_day"].mean()), 3),
        "speed_ratio_down_vs_up": round(float(dn["pct_per_day"].abs().mean() / up["pct_per_day"].mean()), 2),
        "lows": [round(v, 2) for v in sw.loc[sw["change_pct"] > 0, "from"]],
        "highs": [round(v, 2) for v in sw.loc[sw["change_pct"] > 0, "to"]],
    }
    return sw, info


def linear_trend(df: pd.DataFrame) -> dict:
    t = np.asarray((df.index - df.index[0]).days, dtype=float) / 365.25
    y = df["rate"].to_numpy()
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"slope_krw_per_year": round(float(slope), 2), "r2": round(float(r2), 3)}


def calendar_effects(full: pd.Series, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month_end = full.resample("ME").last()
    mchg = (month_end.pct_change() * 100)[START:END]
    pivot = pd.DataFrame({"year": mchg.index.year, "month": mchg.index.month, "chg": mchg.values}) \
        .pivot(index="year", columns="month", values="chg")
    cal = pd.DataFrame({
        "mean_pct": pivot.mean(), "median_pct": pivot.median(), "std_pct": pivot.std(),
        "up_share_pct": (pivot > 0).sum() / pivot.notna().sum() * 100, "n_years": pivot.notna().sum(),
    })
    cal.index.name = "month"

    c = df["chg"].dropna()
    g = c.groupby(c.index.dayofweek)
    wk = pd.DataFrame({"mean_pct": g.mean(), "std_pct": g.std(), "n": g.size(), "mean_abs_pct": g.apply(lambda x: x.abs().mean())})
    wk["t_stat"] = wk["mean_pct"] / (wk["std_pct"] / np.sqrt(wk["n"]))
    wk["ci95_pct"] = 1.96 * wk["std_pct"] / np.sqrt(wk["n"])
    wk.index = [WEEKDAYS[i] for i in wk.index]
    wk.index.name = "weekday"
    return pivot.round(3), cal.round(3), wk.round(4)


def decompose_monthly(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """월평균 환율을 고전적 가법 모형 y = 추세 + 계절 + 잔차로 분해."""
    m = df["rate"].resample("MS").mean()
    w = np.r_[0.5, np.ones(11), 0.5] / 12                       # 2x12 중심이동평균 가중치
    trend = m.rolling(13, center=True).apply(lambda x: float(np.dot(x, w)), raw=True)
    detr = m - trend
    idx = detr.groupby(detr.index.month).mean()
    idx -= idx.mean()                                            # 12개월 합이 0이 되도록 정규화
    seasonal = pd.Series(m.index.month.map(idx).astype(float), index=m.index)
    resid = m - trend - seasonal
    dec = pd.DataFrame({"observed": m, "trend": trend, "seasonal": seasonal, "resid": resid})

    v = dec.dropna()
    fs = max(0.0, 1 - v["resid"].var() / (v["seasonal"] + v["resid"]).var())
    ft = max(0.0, 1 - v["resid"].var() / (v["trend"] + v["resid"]).var())
    n_per_month = detr.dropna().groupby(detr.dropna().index.month).size()
    info = {
        "months": int(len(m)), "months_with_trend": int(trend.notna().sum()),
        "obs_per_calendar_month_min": int(n_per_month.min()), "obs_per_calendar_month_max": int(n_per_month.max()),
        "seasonal_index": {int(k): round(float(val), 2) for k, val in idx.items()},
        "seasonal_amplitude_krw": round(float(idx.max() - idx.min()), 2),
        "trend_range_krw": round(float(trend.max() - trend.min()), 2),
        "resid_std_krw": round(float(v["resid"].std()), 2),
        "seasonal_strength": round(float(fs), 3), "trend_strength": round(float(ft), 3),
    }
    return dec.round(3), info


def baseline_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = df["rate"]
    train, test = y.iloc[:-HOLDOUT], y.iloc[-HOLDOUT:]
    h = np.arange(1, HOLDOUT + 1)
    last = train.iloc[-1]
    sigma1 = train.diff().iloc[-250:].std()                     # 최근 1년 일별 변화폭의 표준편차(원)
    fc = pd.DataFrame(index=test.index)
    fc["actual"] = test
    fc["naive"] = last                                           # 내일도 오늘과 같다(랜덤워크)
    fc["drift"] = last + h * (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)   # 과거 평균 기울기 연장
    fc["ma20"] = train.iloc[-20:].mean()                         # 최근 20일 평균 유지
    fc["naive_lo95"] = last - 1.96 * sigma1 * np.sqrt(h)
    fc["naive_hi95"] = last + 1.96 * sigma1 * np.sqrt(h)

    rows = []
    for name in ("naive", "drift", "ma20"):
        e = fc["actual"] - fc[name]
        rows.append({"model": name, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                     "MAPE_pct": (e.abs() / fc["actual"]).mean() * 100})
    metrics = pd.DataFrame(rows).set_index("model").round(3)

    # 롤링 1일 앞 예측(마지막 250영업일): 매일 다시 예측했을 때의 평균 오차
    one = pd.DataFrame({"actual": y, "naive": y.shift(1), "ma5": y.rolling(5).mean().shift(1),
                        "ma20": y.rolling(20).mean().shift(1),
                        "drift": y.shift(1) + (y.shift(1) - y.shift(250)) / 249}).iloc[-250:]
    roll = {k: round(float((one["actual"] - one[k]).abs().mean()), 3) for k in ("naive", "ma5", "ma20", "drift")}

    info = {
        "train_end": fmt_date(train.index[-1]), "test_start": fmt_date(test.index[0]), "test_end": fmt_date(test.index[-1]),
        "last_train_value": round(float(last), 2), "sigma_1day_krw": round(float(sigma1), 3),
        "naive_95_coverage_pct": round(float(((fc["actual"] >= fc["naive_lo95"]) & (fc["actual"] <= fc["naive_hi95"])).mean() * 100), 1),
        "rolling_1step_MAE_last250": roll,
    }
    return fc.round(3), metrics, info


# ---------------------------------------------------------------------------
# 4. 시각화
# ---------------------------------------------------------------------------
def year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", which="minor", length=2)


def thousands(ax, axis="y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def plot_trend(df: pd.DataFrame, yearly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(df.index, df["rate"], color=CONTEXT, lw=0.9, label="일별 환율")
    ax.plot(df.index, df["ma20"], color=C1, lw=1.6, label="20일 이동평균")
    ax.plot(df.index, df["ma120"], color=C2, lw=2.0, label="120일 이동평균")
    ax.axhline(LEVEL_LINE, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(df.index[0], LEVEL_LINE + 4, f"{LEVEL_LINE:,}원", color=INK2, fontsize=8.5, va="bottom")

    for i, (d, label) in enumerate(EVENTS, start=1):
        t = pd.Timestamp(d)
        if t not in df.index:
            t = df.index[df.index.get_indexer([t], method="nearest")[0]]
        yv = df.loc[t, "rate"]
        ax.scatter([t], [yv], s=34, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"{i}", (t, yv), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, fontweight="bold", color=INK)

    # 끝점 직접 라벨
    last = df.index[-1]
    for col, color, name in (("ma20", C1, "20일"), ("ma120", C2, "120일")):
        ax.annotate(name, (last, df[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(df.index[0], df.index[-1] + pd.Timedelta(days=55))
    year_axis(ax)
    thousands(ax)
    ax.set_ylabel("원 / 달러")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    if EVENTS:
        lines = "\n".join(f"{i}  {fmt_date(d)[2:].replace('-', '.')}  {lab}" for i, (d, lab) in enumerate(EVENTS, start=1))
        ax.text(0.995, 0.03, lines, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color=INK2,
                linespacing=1.55, bbox=dict(boxstyle="round,pad=0.6", facecolor=SURFACE, edgecolor=GRID))
    first, lastv = df["rate"].iloc[0], df["rate"].iloc[-1]
    save(fig, "01_trend_moving_average.png",
         f"원/달러 환율 추이와 이동평균 ({df.index[0]:%Y.%m}~{df.index[-1]:%Y.%m})",
         f"{first:,.0f}원 → {lastv:,.0f}원 ({(lastv / first - 1) * 100:+.1f}%) · 기간 최고 {df['rate'].max():,.0f}원 "
         f"({df['rate'].idxmax():%Y.%m.%d}) · 번호는 주요 이벤트(이벤트가 ECB 고시 환율에 처음 반영된 날)")


def plot_swings(df: pd.DataFrame, sw: pd.DataFrame, info: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(df.index, df["rate"], color=CONTEXT, lw=0.9, label="일별 환율")
    for _, r in sw.iterrows():
        t0, t1 = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        up = r["change_pct"] > 0
        ax.plot([t0, t1], [r["from"], r["to"]], color=UP if up else DOWN, lw=2.4,
                ls="-" if r["confirmed"] else (0, (4, 2)), solid_capstyle="round")
        mid_t = t0 + (t1 - t0) / 2
        mid_v = (r["from"] + r["to"]) / 2
        tag = "" if r["confirmed"] else " (미확정)"
        ax.annotate(f"{r['change_pct']:+.1f}%\n{r['trading_days']}영업일{tag}", (mid_t, mid_v),
                    xytext=(-10, 8) if up else (10, 4), textcoords="offset points",
                    ha="right" if up else "left", va="bottom", fontsize=8.5, color=INK, linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    for _, r in sw.iterrows():
        for t, v in ((r["start"], r["from"]), (r["end"], r["to"])):
            ax.scatter([pd.Timestamp(t)], [v], s=26, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1)
    ax.plot([], [], color=UP, lw=2.4, label="상승 국면(원화 약세)")
    ax.plot([], [], color=DOWN, lw=2.4, label="하락 국면(원화 강세)")
    year_axis(ax)
    thousands(ax)
    ax.set_xlim(df.index[0] - pd.Timedelta(days=20), df.index[-1] + pd.Timedelta(days=20))
    ax.set_ylabel("원 / 달러")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    save(fig, "02_swings_up_slow_down_fast.png", "오를 때는 천천히, 내릴 때는 빠르게: 상승·하락 국면 비교",
         f"직전 고점(저점)에서 {info['threshold_pct']:.0f}% 이상 되돌리면 국면 전환으로 판정 · "
         f"상승 국면 평균 {info['up_avg_days']:.0f}영업일 vs 하락 국면 평균 {info['down_avg_days']:.0f}영업일 · "
         f"하루 평균 속도는 하락이 {info['speed_ratio_down_vs_up']:.1f}배 빠름")


def plot_daily_change(df: pd.DataFrame, dist: dict) -> None:
    c = df["chg"].dropna()
    sd = c.std()
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True,
                                  gridspec_kw={"width_ratios": [4, 1], "wspace": 0.04})
    ax.vlines(c.index, 0, c.values, colors=np.where(c > 0, UP, DOWN), lw=0.8)
    for k in (2, -2):
        ax.axhline(k * sd, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(c.index[0], 2 * sd + 0.05, "+2σ", color=INK2, fontsize=8.5)
    ax.text(c.index[0], -2 * sd - 0.05, "-2σ", color=INK2, fontsize=8.5, va="top")
    placed: list[tuple[pd.Timestamp, float]] = []
    for t in c.abs().nlargest(6).index:          # 절대값 상위 6일에 날짜 라벨
        v = c[t]
        near = [p for p in placed if abs((p[0] - t).days) < 120 and np.sign(p[1]) == np.sign(v)]
        dx = (34 if t > near[0][0] else -34) if near else 0
        ax.annotate(f"{t:%y.%m.%d}\n{v:+.2f}%", (t, v), xytext=(dx, 6 if v > 0 else -6), textcoords="offset points",
                    ha="center", va="bottom" if v > 0 else "top", fontsize=7.5, color=INK2)
        placed.append((t, v))
    year_axis(ax)
    ax.set_xlim(c.index[0] - pd.Timedelta(days=20), c.index[-1] + pd.Timedelta(days=20))
    ax.set_ylabel("전일 대비 변화율 (%)")
    ax.axhline(0, color=INK2, lw=0.6)
    lo, hi = c.min() * 1.25, c.max() * 1.6
    ax.set_ylim(lo, hi)

    bins = np.linspace(lo, hi, 61)
    axh.hist(c, bins=bins, orientation="horizontal", color=CONTEXT, edgecolor=SURFACE, linewidth=0.5)
    xs = np.linspace(lo, hi, 300)
    normal = len(c) * (bins[1] - bins[0]) * np.exp(-0.5 * ((xs - c.mean()) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    axh.plot(normal, xs, color=C1, lw=1.6, label="같은 σ의 정규분포")
    axh.tick_params(labelleft=False)
    axh.set_xlabel("일수")
    axh.legend(loc="upper right", fontsize=8)
    save(fig, "03_daily_change.png", "일별 변화율: 대부분 조용하지만 가끔 크게 튄다",
         f"σ = {sd:.2f}% · 초과첨도 {dist['excess_kurtosis']:.1f} · |변화율|>2σ 비율 {dist['share_beyond_2sd_pct']:.1f}% "
         f"(정규분포라면 4.6%) · |변화율|>3σ 비율 {dist['share_beyond_3sd_pct']:.2f}% (정규분포라면 0.27%) · 빨강=원화 약세, 파랑=원화 강세")


def plot_volatility(df: pd.DataFrame, yearly: pd.DataFrame) -> None:
    v = df["vol20"].dropna()
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(v.index, v.values, color=C1, lw=1.4, label="20일 롤링 변동성(연율화)")
    # 연평균 계단선
    for y, row in yearly.iterrows():
        s, e = pd.Timestamp(f"{y}-01-01"), min(pd.Timestamp(f"{y}-12-31"), v.index[-1])
        ax.hlines(row["vol_ann_pct"], s, e, color=C2, lw=2.2, label="연간 변동성" if y == yearly.index[0] else None)
        ax.text(s + (e - s) / 2, row["vol_ann_pct"] + 0.5, f"{y}년 {row['vol_ann_pct']:.1f}%", ha="center",
                va="bottom", fontsize=8.5, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    for t in [v.idxmax()]:
        ax.annotate(f"최고 {v[t]:.1f}%\n({t:%Y.%m.%d})", (t, v[t]), xytext=(8, -4), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="top")
    year_axis(ax)
    ax.set_xlim(v.index[0] - pd.Timedelta(days=20), v.index[-1] + pd.Timedelta(days=20))
    ax.set_ylabel("연율화 변동성 (%)")
    ax.set_ylim(0, v.max() * 1.15)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "04_rolling_volatility.png", "변동성은 일정하지 않다: 20일 롤링 변동성",
         "일별 변화율의 20영업일 표준편차 × √252 · 주황 선은 해당 연도 전체의 변동성")


def plot_monthly_heatmap(pivot: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.9))
    data = pivot.reindex(columns=range(1, 13))
    vmax = np.nanmax(np.abs(data.values))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(data.values, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(12), [f"{m}월" for m in range(1, 13)])
    ax.set_yticks(range(len(data.index)), [str(y) for y in data.index])
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(data.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for (i, j), val in np.ndenumerate(data.values):
        if np.isnan(val):
            continue
        dark = abs(val) > vmax * 0.55
        ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8.5, color="white" if dark else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("월간 변화율 (%)", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "05_monthly_heatmap.png", "연도 × 월 월간 변화율 (월말 대비 월말)",
         "빨강 = 환율 상승(원화 약세), 파랑 = 환율 하락(원화 강세) · 같은 달이라도 해마다 방향이 엇갈린다")


def plot_calendar_effects(pivot: pd.DataFrame, cal: pd.DataFrame, wk: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    months = np.arange(1, 13)
    for j, m in enumerate(months):
        vals = pivot[m].dropna().values
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0]
        a1.scatter(m + np.asarray(jitter), vals, s=16, color=CONTEXT, zorder=2, label="개별 연도" if j == 0 else None)
    a1.bar(months, cal["mean_pct"], width=0.55, color=np.where(cal["mean_pct"] > 0, UP, DOWN), alpha=0.9,
           zorder=3, label="평균")
    a1.axhline(0, color=INK2, lw=0.6)
    a1.set_xticks(months, [f"{m}월" for m in months])
    a1.set_ylabel("월간 변화율 (%)")
    a1.set_title("월별: 평균 막대와 연도별 값(점)")
    a1.legend(loc="upper left", fontsize=8.5)

    x = np.arange(len(wk))
    a2.bar(x, wk["mean_pct"], width=0.55, color=np.where(wk["mean_pct"] > 0, UP, DOWN), zorder=3)
    a2.errorbar(x, wk["mean_pct"], yerr=wk["ci95_pct"], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_xticks(x, wk.index)
    a2.set_ylabel("평균 일별 변화율 (%)")
    a2.set_title("요일별: 평균과 95% 신뢰구간")
    save(fig, "06_calendar_effects.png", "달력 효과가 있을까? 월별·요일별 평균 변화율",
         "관측 연도가 5~6개뿐이라 월별 평균은 한두 해의 큰 움직임에 좌우된다 · 요일별 오차막대가 0을 포함하면 '차이 없음'과 구분이 어렵다")


def plot_decomposition(dec: pd.DataFrame, info: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    a0, a1, a2 = axes
    a0.plot(dec.index, dec["observed"], color=CONTEXT, lw=1.6, marker="o", ms=3, label="월평균 환율(관측)")
    a0.plot(dec.index, dec["trend"], color=C2, lw=2.2, label="추세 (2×12 중심이동평균)")
    a0.set_ylabel("원 / 달러")
    thousands(a0)
    a0.legend(loc="upper left", fontsize=9)
    a0.set_title("관측값과 추세")
    lim = np.nanmax(np.abs(dec[["seasonal", "resid"]].values)) * 1.15
    a1.bar(dec.index, dec["seasonal"], width=20, color=C1)
    a1.axhline(0, color=INK2, lw=0.6)
    a1.set_ylim(-lim, lim)
    a1.set_ylabel("원")
    a1.set_title(f"계절 성분 (매년 같은 패턴 반복, 진폭 {info['seasonal_amplitude_krw']:.1f}원)")
    a2.bar(dec.index, dec["resid"], width=20, color=np.where(dec["resid"] > 0, UP, DOWN))
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_ylim(-lim, lim)
    a2.set_ylabel("원")
    a2.set_title(f"잔차 (표준편차 {info['resid_std_krw']:.1f}원) · 계절 성분과 같은 눈금")
    year_axis(a2)
    save(fig, "07_decomposition.png", "[보너스 A] 시계열 분해: 추세 · 계절 · 잔차 (월평균, 가법 모형)",
         f"추세 강도 {info['trend_strength']:.2f} · 계절 강도 {info['seasonal_strength']:.2f} (0~1, 1에 가까울수록 해당 성분이 뚜렷함) · "
         f"달마다 관측치 {info['obs_per_calendar_month_min']}~{info['obs_per_calendar_month_max']}개")


def plot_forecast(df: pd.DataFrame, fc: pd.DataFrame, metrics: pd.DataFrame, info: dict) -> None:
    hist = df["rate"].iloc[-(HOLDOUT + 160):-HOLDOUT]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(hist.index, hist.values, color=INK2, lw=1.3, label="학습 구간 실제값")
    ax.fill_between(fc.index, fc["naive_lo95"], fc["naive_hi95"], color=C1, alpha=0.12, lw=0,
                    label="단순 지속 예측 95% 범위")
    ax.plot(fc.index, fc["actual"], color=INK, lw=1.8, label="평가 구간 실제값")
    ax.plot(fc.index, fc["naive"], color=C1, lw=2, label=f"단순 지속 (MAE {metrics.loc['naive', 'MAE']:.1f}원)")
    ax.plot(fc.index, fc["drift"], color=C2, lw=2, ls=(0, (5, 2)), label=f"드리프트 (MAE {metrics.loc['drift', 'MAE']:.1f}원)")
    ax.plot(fc.index, fc["ma20"], color=C3, lw=2, ls=(0, (1.5, 1.5)), label=f"20일 평균 (MAE {metrics.loc['ma20', 'MAE']:.1f}원)")
    ax.axvline(fc.index[0], color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(fc.index[0], ax.get_ylim()[1], "  예측 시작", va="top", fontsize=8.5, color=INK2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
    thousands(ax)
    ax.set_ylabel("원 / 달러")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    save(fig, "08_baseline_forecast.png", f"[보너스 B] 베이스라인 예측: 마지막 {HOLDOUT}영업일 ({info['test_start']} ~ {info['test_end']})",
         f"{info['train_end']}까지의 데이터만으로 이후 {HOLDOUT}영업일을 한 번에 예측 · 실제값이 95% 범위 안에 든 비율 {info['naive_95_coverage_pct']:.0f}%")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    setup_style()

    full, period, data_info = load_and_clean()
    outliers, outlier_info = check_outliers(full, period)
    df = build_frame(full)
    yearly = yearly_stats(full, df)
    dist = distribution_stats(df)
    top_moves = df["chg"].dropna().abs().nlargest(10).index
    top = df.loc[top_moves, ["rate", "chg"]].assign(weekday=[WEEKDAYS[d.dayofweek] for d in top_moves]).round(3)
    top.index = top.index.map(fmt_date)
    crosses = ma_crossovers(df)
    swings, swing_info = swing_analysis(df["rate"])
    pivot, cal, wk = calendar_effects(full, df)
    dec, dec_info = decompose_monthly(df)
    fc, metrics, fc_info = baseline_forecast(df)

    v = df["vol20"].dropna()
    vol_info = {
        "vol20_max_pct": round(float(v.max()), 2), "vol20_max_date": fmt_date(v.idxmax()),
        "vol20_min_pct": round(float(v.min()), 2), "vol20_min_date": fmt_date(v.idxmin()),
        "corr_level_vs_vol20": round(float(df[["rate", "vol20"]].corr().iloc[0, 1]), 3),
        "corr_ret20_vs_vol20": round(float(df[["ret20", "vol20"]].corr().iloc[0, 1]), 3),
        "mean_vol20_when_ret20_pos": round(float(df.loc[df["ret20"] > 0, "vol20"].mean()), 2),
        "mean_vol20_when_ret20_neg": round(float(df.loc[df["ret20"] < 0, "vol20"].mean()), 2),
    }
    summary = {
        "data": data_info, "outliers": outlier_info, "trend": {
            "first": round(float(df["rate"].iloc[0]), 2), "first_date": fmt_date(df.index[0]),
            "last": round(float(df["rate"].iloc[-1]), 2), "last_date": fmt_date(df.index[-1]),
            "total_change_pct": round(float((df["rate"].iloc[-1] / df["rate"].iloc[0] - 1) * 100), 2),
            "min": round(float(df["rate"].min()), 2), "min_date": fmt_date(df["rate"].idxmin()),
            "max": round(float(df["rate"].max()), 2), "max_date": fmt_date(df["rate"].idxmax()),
            "days_above_level_line_pct": round(float((df["rate"] >= LEVEL_LINE).mean() * 100), 2),
            "first_close_above_level_line": fmt_date(df.index[df["rate"] >= LEVEL_LINE][0]) if (df["rate"] >= LEVEL_LINE).any() else None,
            **linear_trend(df), "ma20_ma120_crossovers": int(len(crosses)),
        },
        "top10_daily_moves_down_count": int((top["chg"] < 0).sum()),
        "swings": swing_info,
        "distribution": dist, "acf_daily_change": acf(df["chg"]), "volatility": vol_info,
        "decomposition": dec_info, "forecast": fc_info, "forecast_metrics": metrics.to_dict(orient="index"),
    }

    yearly.to_csv(OUT_DIR / "yearly_stats.csv", encoding="utf-8-sig")
    top.to_csv(OUT_DIR / "top10_daily_moves.csv", encoding="utf-8-sig")
    crosses.to_csv(OUT_DIR / "ma20_ma120_crossovers.csv", index=False, encoding="utf-8-sig")
    swings.to_csv(OUT_DIR / "swings.csv", index=False, encoding="utf-8-sig")
    outliers.to_csv(OUT_DIR / "outlier_candidates.csv", encoding="utf-8-sig")
    pivot.to_csv(OUT_DIR / "monthly_change_pivot.csv", encoding="utf-8-sig")
    cal.to_csv(OUT_DIR / "calendar_month_stats.csv", encoding="utf-8-sig")
    wk.to_csv(OUT_DIR / "weekday_stats.csv", encoding="utf-8-sig")
    dec.to_csv(OUT_DIR / "decomposition_monthly.csv", encoding="utf-8-sig")
    fc.to_csv(OUT_DIR / "forecast_holdout.csv", encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "forecast_metrics.csv", encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_trend(df, yearly)
    plot_swings(df, swings, swing_info)
    plot_daily_change(df, dist)
    plot_volatility(df, yearly)
    plot_monthly_heatmap(pivot)
    plot_calendar_effects(pivot, cal, wk)
    plot_decomposition(dec, dec_info)
    plot_forecast(df, fc, metrics, fc_info)

    pd.set_option("display.width", 200)
    print("== 데이터 정보 ==", json.dumps(data_info, ensure_ascii=False, indent=1), sep="\n")
    print("\n== 이상치 후보 ==", outliers.to_string(), json.dumps(outlier_info, ensure_ascii=False), sep="\n")
    print("\n== 연도별 통계 ==", yearly.to_string(), sep="\n")
    print("\n== 일별 변화율 상위 10 ==", top.to_string(), sep="\n")
    print("\n== 20일/120일 이동평균 교차 ==", crosses.to_string(index=False), sep="\n")
    print("\n== 상승/하락 국면 ==", swings.to_string(index=False), json.dumps(swing_info, ensure_ascii=False), sep="\n")
    print("\n== 월별 ==", cal.to_string(), "\n== 요일별 ==", wk.to_string(), sep="\n")
    print("\n== 예측 오차 ==", metrics.to_string(), sep="\n")
    print("\n== 요약(summary.json) ==", json.dumps({k: summary[k] for k in ("trend", "distribution", "acf_daily_change", "volatility", "decomposition", "forecast")}, ensure_ascii=False, indent=1), sep="\n")
    print(f"\n그래프 {len(list(IMG_DIR.glob('*.png')))}개 -> {IMG_DIR}")


if __name__ == "__main__":
    main()
