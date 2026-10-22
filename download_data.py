"""
원/달러 일별 환율 수집 스크립트

출처: Frankfurter API (https://frankfurter.dev)
      - 유럽중앙은행(ECB)이 매 영업일 14:10(CET)경 집계해 16:00경 공표하는 유로 기준환율(euro foreign exchange reference rates)을
        USD 기준으로 다시 계산해 제공하는 무료 오픈소스 API (API 키 불필요)
      - USD/KRW = (EUR/KRW) / (EUR/USD) 교차환율

실행:  python download_data.py
결과:  data/usdkrw_daily.csv  (date, usd_krw)
       data/raw/usdkrw_<연도>.json  (API 원본 응답, 연도별)
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_CSV = BASE_DIR / "data" / "usdkrw_daily.csv"

START = date(2020, 7, 1)    # 2021년 초부터 120일 이동평균을 계산하기 위해 6개월 앞서 수집
END = date(2026, 9, 30)     # 분석 종료일(완결된 마지막 월말)

# 기본 주소가 안 되면 예비 주소로 재시도
HOSTS = ["https://api.frankfurter.dev/v1", "https://api.frankfurter.app"]


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "krw-usd-analysis/1.0 (educational project)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_range(start: date, end: date) -> dict:
    last_error: Exception | None = None
    for host in HOSTS:
        url = f"{host}/{start.isoformat()}..{end.isoformat()}?base=USD&symbols=KRW"
        try:
            return fetch_json(url)
        except Exception as e:  # 네트워크 오류 시 다음 주소로
            last_error = e
    raise RuntimeError(f"데이터를 받지 못했습니다: {last_error}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, float] = {}

    # 기간이 길면 응답이 커지므로 연도별로 나눠 요청
    for year in range(START.year, END.year + 1):
        s = max(START, date(year, 1, 1))
        e = min(END, date(year, 12, 31))
        payload = fetch_range(s, e)
        (RAW_DIR / f"usdkrw_{year}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        n = 0
        for d, v in payload.get("rates", {}).items():
            if "KRW" in v:
                rows[d] = float(v["KRW"])
                n += 1
        print(f"{year}: {n}개 영업일 수신 ({s} ~ {e})")
        time.sleep(0.5)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "usd_krw"])
        for d in sorted(rows):
            w.writerow([d, rows[d]])
    print(f"저장 완료: {OUT_CSV} (총 {len(rows)}행)")


if __name__ == "__main__":
    main()
