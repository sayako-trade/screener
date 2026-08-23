"""
5日線-10日線クロス スクリーナー

やること:
  1. JPXの上場銘柄一覧を取得（30日ごとに更新）
  2. yfinanceで全銘柄の日足を取得
  3. 5日線と10日線のクロス当日の銘柄を判定
  4. 結果を docs/index.html に書き出す

価格帯・売買代金の絞り込みは画面側でやるので、ここでは
「クロスした銘柄」を全部書き出しておく。
"""

import datetime as dt
import io
import json
import os
import sys
import time
import zoneinfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Windowsのコンソールはcp932なので、そのままだと「〜」などで落ちる
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

JST = zoneinfo.ZoneInfo("Asia/Tokyo")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
DOCS_DIR = os.path.join(ROOT, "docs")
TICKER_CACHE = os.path.join(DATA_DIR, "tickers.json")
LAST_DATE = os.path.join(DATA_DIR, "last_date.txt")
TEMPLATE = os.path.join(ROOT, "template.html")
OUTPUT = os.path.join(DOCS_DIR, "index.html")

JPX_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

TICKER_MAX_AGE_DAYS = 30   # 銘柄一覧を取り直す間隔
CHUNK = 150                # yfinanceに一度に投げる銘柄数
RANGE_DAYS = 120           # 高値安値レンジの期間（約6か月）
TURNOVER_DAYS = 20         # 売買代金の平均を取る日数
PROBE = ["7203.T", "8306.T", "9432.T"]   # データ更新チェック用
RETRIES = 3                # 1チャンクあたりの取得リトライ回数
RETRY_WAIT = 30            # リトライの待ち秒数（回を追うごとに伸ばす）
MAX_FAIL_RATIO = 0.1       # これを超える割合で取れなかったら結果を書き換えない
                           # チャンクごと失敗した分と、銘柄単位で欠けた分の合計で見る


# ---------------------------------------------------------------- 銘柄一覧

def load_tickers():
    """JPXの上場銘柄一覧を取得。キャッシュが新しければそれを使う。"""
    if os.path.exists(TICKER_CACHE):
        age = time.time() - os.path.getmtime(TICKER_CACHE)
        if age < TICKER_MAX_AGE_DAYS * 86400:
            with open(TICKER_CACHE, encoding="utf-8") as f:
                cached = json.load(f)
            print(f"銘柄一覧: キャッシュを使用 ({len(cached)}銘柄)")
            return cached

    print("銘柄一覧: JPXから取得中...")
    res = requests.get(JPX_URL, timeout=60)
    res.raise_for_status()
    df = pd.read_excel(io.BytesIO(res.content))

    # ETF・REIT・出資証券などを除外して、内国株式だけ残す
    df = df[df["市場・商品区分"].astype(str).str.contains("内国株式", na=False)]

    tickers = {}
    for code, name in zip(df["コード"], df["銘柄名"]):
        code = str(code).strip()
        if code and code.lower() != "nan":
            tickers[code] = str(name).strip()

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TICKER_CACHE, "w", encoding="utf-8") as f:
        json.dump(tickers, f, ensure_ascii=False, indent=1)

    print(f"銘柄一覧: {len(tickers)}銘柄を保存")
    return tickers


# ---------------------------------------------------------- データ更新チェック

def market_date():
    """主要銘柄をのぞいて、最新の日足がいつ分か調べる。"""
    df = yf.download(
        PROBE, period="10d", interval="1d",
        auto_adjust=False, progress=False, threads=True,
    )
    if df is None or df.empty:
        return None
    return df.index[-1].date()


def already_done(latest):
    if not os.path.exists(LAST_DATE):
        return False
    with open(LAST_DATE, encoding="utf-8") as f:
        return f.read().strip() == latest.isoformat()


# -------------------------------------------------------------------- 判定

def judge(d):
    """1銘柄分の日足から、クロス当日かどうかと各条件を判定する。"""
    if len(d) < RANGE_DAYS + 5:
        return None

    close = d["Close"]
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    if pd.isna(ma20.iloc[-1]) or pd.isna(ma10.iloc[-2]):
        return None

    # --- クロス当日か
    gc = ma5.iloc[-2] <= ma10.iloc[-2] and ma5.iloc[-1] > ma10.iloc[-1]
    dc = ma5.iloc[-2] >= ma10.iloc[-2] and ma5.iloc[-1] < ma10.iloc[-1]
    if not (gc or dc):
        return None

    kind = "gc" if gc else "dc"

    o = float(d["Open"].iloc[-1])
    h = float(d["High"].iloc[-1])
    l = float(d["Low"].iloc[-1])
    c = float(close.iloc[-1])
    m20 = float(ma20.iloc[-1])

    up5 = ma5.iloc[-1] > ma5.iloc[-2]
    up10 = ma10.iloc[-1] > ma10.iloc[-2]

    # --- 条件チェック（満たしていない理由を集める）
    fails = []

    if kind == "gc":
        if not up5 and not up10:
            fails.append("5日線・10日線が下向き")
        elif not up5:
            fails.append("5日線が下向き")
        elif not up10:
            fails.append("10日線が下向き")
        # またぐ or 20日線の上に載っている = 足全体が20日線より下でなければOK
        if h < m20:
            fails.append("20日線より下に離れている")
        if c <= o:
            fails.append("陽線でない")
    else:
        if up5 and up10:
            fails.append("5日線・10日線が上向き")
        elif up5:
            fails.append("5日線が上向き")
        elif up10:
            fails.append("10日線が上向き")
        # またぐ or 20日線にぶら下がっている = 足全体が20日線より上でなければOK
        if l > m20:
            fails.append("20日線より上に離れている")
        if c >= o:
            fails.append("陰線でない")

    # --- 直近6か月レンジの中の位置（0%が安値、100%が高値）
    window = d.iloc[-RANGE_DAYS:]
    lo = float(window["Low"].min())
    hi = float(window["High"].max())
    pos = 50.0 if hi <= lo else (c - lo) / (hi - lo) * 100

    # --- 売買代金（20日平均・億円）
    tv = (d["Close"] * d["Volume"]).iloc[-TURNOVER_DAYS:].mean()
    turnover = float(tv) / 1e8 if pd.notna(tv) else 0.0

    return {
        "kind": kind,
        "close": round(c, 1),
        "pos": round(pos, 1),
        "turnover": round(turnover, 1),
        "matched": len(fails) == 0,
        "fails": fails,
    }


# ------------------------------------------------------------------ 取得

def fetch(chunk):
    """1チャンク取得する。Yahooはまとめて叩くとレート制限してくるので待って粘る。"""
    for attempt in range(RETRIES):
        try:
            raw = yf.download(
                chunk, period="1y", interval="1d", group_by="ticker",
                auto_adjust=False, progress=False, threads=True,
            )
            if raw is not None and not raw.empty:
                return raw
            reason = "空のデータが返った"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"

        if attempt < RETRIES - 1:
            wait = RETRY_WAIT * (attempt + 1)
            print(f"    取得失敗（{reason}）。{wait}秒待って再試行", flush=True)
            time.sleep(wait)
        else:
            print(f"    取得失敗（{reason}）。このチャンクは諦めます", flush=True)

    return None


def scan(tickers):
    """全銘柄を取得して判定する。戻り値は (結果, データを取れなかった銘柄数)。

    チャンクごと落ちるだけでなく、チャンクは成功しても中の1銘柄だけが
    レート制限で全部NaNになって返ってくることがある。どちらも
    「取れなかった銘柄」として数えて、呼び出し側で多すぎないか見る。
    """
    symbols = [f"{code}.T" for code in tickers]
    results = []
    missing = 0

    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        print(f"  {i + 1}〜{i + len(chunk)} / {len(symbols)}", flush=True)

        raw = fetch(chunk)
        if raw is None:
            missing += len(chunk)
            continue

        multi = isinstance(raw.columns, pd.MultiIndex)

        for sym in chunk:
            try:
                d = raw[sym] if multi else raw
            except KeyError:
                missing += 1
                continue
            d = d.dropna(subset=["Close"])
            if d.empty:
                missing += 1
                continue

            try:
                r = judge(d)
            except Exception:
                continue

            if r:
                code = sym[:-2]
                r["code"] = code
                r["name"] = tickers.get(code, code)
                results.append(r)

    return results, missing


# ------------------------------------------------------------------ 出力

def render(results, latest):
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()

    now = dt.datetime.now(JST)
    wd = "月火水木金土日"[latest.weekday()]

    html = html.replace("__DATE__", f"{latest.month}/{latest.day}（{wd}）")
    html = html.replace("__UPDATED__", now.strftime("%Y-%m-%d %H:%M"))
    html = html.replace('"__DATA__"', json.dumps(results, ensure_ascii=False))

    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    with open(LAST_DATE, "w", encoding="utf-8") as f:
        f.write(latest.isoformat())


# -------------------------------------------------------------------- main

def main():
    force = "--force" in sys.argv

    latest = market_date()
    if latest is None:
        print("株価データを取得できませんでした。今回は更新しません。")
        return 0

    today = dt.datetime.now(JST).date()
    print(f"最新の日足: {latest}  （今日: {today}）")

    if not force:
        if latest < today:
            print("本日分のデータがまだありません。休場か、反映待ちです。")
            return 0
        if already_done(latest):
            print("この日付はすでに処理済みです。")
            return 0

    tickers = load_tickers()
    print(f"判定開始: {len(tickers)}銘柄")

    results, missing = scan(tickers)

    if missing:
        print(f"データを取れなかった銘柄: {missing} / {len(tickers)}")
    if missing > len(tickers) * MAX_FAIL_RATIO:
        print("取りこぼしが多すぎます。前回の結果を残したまま終了します。")
        return 1

    matched = sum(1 for r in results if r["matched"])
    print(f"クロス: {len(results)}銘柄  うち条件一致: {matched}銘柄")

    render(results, latest)
    print(f"書き出し完了: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
