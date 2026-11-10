# -*- coding: utf-8 -*-
"""
数据源层：从 freqtrade 获取 K 线。

支持两种来源：
  1. api     —— freqtrade webserver 模式 POST /api/v1/pair_history（live_mode=true，
               由 freqtrade 直接向交易所拉最新 K 线，实盘唯一可靠的实时源）
  2. feather —— 直接读 freqtrade 的 feather 文件（通常为 download-data 的静态快照）

统一输出为规范化的 DataFrame：
    timestamps(tz-naive UTC) | open | high | low | close | volume
且**只包含已收盘 K 线**。
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

_REQUIRED = ["timestamps", "open", "high", "low", "close", "volume"]


class DataSourceError(RuntimeError):
    pass


# ------------------------------------------------------------------
# 通用处理
# ------------------------------------------------------------------
def _normalize(df: pd.DataFrame, ts_col_candidates=("timestamps", "date", "time", "datetime")) -> pd.DataFrame:
    """统一列名 + 时间戳转为 tz-naive UTC"""
    ts_col = None
    for c in ts_col_candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamps"})
            ts_col = "timestamps"
        else:
            raise DataSourceError(f"找不到时间列，现有列: {list(df.columns)}")

    out = df.rename(columns={ts_col: "timestamps"}).copy()
    ts = pd.to_datetime(out["timestamps"], utc=True, errors="coerce")
    out["timestamps"] = ts.dt.tz_convert(None)          # tz-naive UTC

    keep = [c for c in _REQUIRED if c != "timestamps"]
    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise DataSourceError(f"缺少必需列: {missing}")

    out = out[["timestamps"] + keep]
    out = out.dropna(subset=["timestamps"]).sort_values("timestamps").reset_index(drop=True)
    return out


def _closed_only(df: pd.DataFrame, now: float = None) -> pd.DataFrame:
    """只保留已收盘的 K 线：open_ts + tf <= now"""
    tf_sec = C.tf_seconds()
    now = now if now is not None else time.time()
    # 先降到秒精度再转 int64：直接 astype("int64") 在 datetime64[us]/[ns]
    # 下得到的是微秒/纳秒，除以 1e9 会缩水，导致未收盘 K 线被误判为已收盘
    opens = df["timestamps"].astype("datetime64[s]").astype("int64")
    return df[opens + tf_sec <= now + 1].reset_index(drop=True)


# ------------------------------------------------------------------
# 来源 1：freqtrade webserver API（POST /pair_history）
# ------------------------------------------------------------------
def fetch_from_api(limit: int = None, pair: str = None) -> pd.DataFrame:
    """
    通过 webserver 模式的 POST /api/v1/pair_history 取 K 线。

    live_mode=True：由 freqtrade 直接向交易所拉取最新 K 线（等同实盘数据源）。
    strategy / freqaimodel 留空不传：不跑策略分析，只要原始 OHLCV。
    timerange 用「当前时刻 - 所需根数」限定起点，避免接口默认回拉 30 天
    （5m 周期 30 天要翻 9 页 API，慢且无必要）。

    :param pair: 指定产品，默认 C.PAIR（多产品实例逐个传入）
    """
    pair = pair or C.PAIR
    # 需要 LOOKBACK+LONG_LEN 根【已收盘】K 线；接口返回的最后一根通常是
    # 未收盘 K 线（会被 _closed_only 丢掉），故多要 2 根作为缓冲。
    limit = limit or (C.LOOKBACK + C.LONG_LEN + 2)
    tf_sec = C.tf_seconds()
    start_ts = int(time.time()) - (limit + 2) * tf_sec

    payload = {
        "pair": pair,
        "timeframe": C.TIMEFRAME,
        "timerange": f"{start_ts}-",        # 10 位 unix 秒，"-" 表示到当前
        "live_mode": True,
        "columns": ["date", "open", "high", "low", "close", "volume"],
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{C.API_BASE}/api/v1/pair_history"

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    token = base64.b64encode(f"{C.API_USERNAME}:{C.API_PASSWORD}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=C.API_TIMEOUT_SEC) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "ignore")
        except Exception:                                    # noqa: BLE001
            pass
        raise DataSourceError(f"API 请求失败 {url}: HTTP {e.code} {detail}") from e
    except Exception as e:                                   # noqa: BLE001
        raise DataSourceError(f"API 请求失败 {url}: {type(e).__name__}: {e}") from e

    cols = result.get("columns") or []
    rows = result.get("data") or []
    if not cols or not rows:
        raise DataSourceError(f"API 返回空数据 (columns={len(cols)}, rows={len(rows)})")

    df = pd.DataFrame(rows, columns=cols)
    df = _normalize(df)
    df = _closed_only(df)
    # 不强制 LOOKBACK：实盘冷启动时数据可能不足，>= MIN_LOOKBACK 即可，
    # 实际窗口长度由 bridge 按"有多少用多少"降级处理。
    if len(df) < C.MIN_LOOKBACK:
        raise DataSourceError(
            f"API 返回已收盘 K 线仅 {len(df)} 根，少于 MIN_LOOKBACK={C.MIN_LOOKBACK}"
        )
    return df


# ------------------------------------------------------------------
# 来源 2：freqtrade feather 文件
# ------------------------------------------------------------------
def fetch_from_feather(paths: list = None, pair: str = None):
    pair = pair or C.PAIR
    paths = paths or C.feather_paths(pair)
    errors = []
    for p in paths:
        if not p or not os.path.exists(p):
            errors.append(f"{p} (不存在)")
            continue
        try:
            df = pd.read_feather(p)
            df = _normalize(df)
            df = _closed_only(df)
            if len(df) < C.MIN_LOOKBACK:
                errors.append(f"{p} (仅 {len(df)} 根 < MIN_LOOKBACK={C.MIN_LOOKBACK})")
                continue
            return df, p
        except Exception as e:                               # noqa: BLE001
            errors.append(f"{p} ({type(e).__name__}: {e})")
    raise DataSourceError("feather 读取失败: " + " | ".join(errors))


def staleness(df: pd.DataFrame) -> float:
    """最后一根已收盘 K 线距现在的**周期倍数**；>1 说明已滞后"""
    tf_sec = C.tf_seconds()
    last_open = int(df["timestamps"].iloc[-1].value // 10 ** 9)
    age = time.time() - (last_open + tf_sec)
    return age / tf_sec


# ------------------------------------------------------------------
# 统一入口
# ------------------------------------------------------------------
def load_klines(limit: int = None, source: str = None, pair: str = None):
    """
    按优先级获取 K 线。日志输出由 config.VERBOSE 统一控制。

    :param source: 强制指定 "api" / "feather"；None 则按 C.LIVE_SOURCE_PRIORITY
    :param pair: 指定产品，默认 C.PAIR（多产品实例逐个传入）
    :return: (DataFrame, meta dict)
    """
    pair = pair or C.PAIR
    limit = limit or max(C.LOOKBACK, C.LOOKBACK + C.LONG_LEN)
    order = [source] if source else list(C.LIVE_SOURCE_PRIORITY)

    errors = []
    for src in order:
        try:
            if src == "api":
                df = fetch_from_api(limit=limit, pair=pair)
                origin_path = f"api@{C.API_BASE}"
            elif src == "feather":
                df, origin_path = fetch_from_feather(pair=pair)
            else:
                raise DataSourceError(f"未知数据源: {src}")

            if limit and len(df) > limit:
                df = df.iloc[-limit:].reset_index(drop=True)

            stale = staleness(df)
            meta = {
                "source": src, "path": origin_path, "pair": pair,
                "timeframe": C.TIMEFRAME,
                "rows": len(df), "stale_ratio": round(stale, 2),
                "warn": None,
            }
            if stale >= C.STALE_TOLERANCE:
                meta["warn"] = (
                    f"数据已滞后 {stale:.1f} 个周期（最后 K 线 "
                    f"{df['timestamps'].iloc[-1]}），可能不是实时数据"
                )
            if C.VERBOSE:
                print(f"[data] {src} <- {origin_path} | {len(df)} 根 | "
                      f"末端 {df['timestamps'].iloc[-1]} | 滞后 {stale:.2f} 周期")
                if meta["warn"]:
                    print(f"[data] ⚠ {meta['warn']}")
            return df, meta
        except Exception as e:                               # noqa: BLE001
            errors.append(f"{src}: {e}")
            if C.VERBOSE:
                print(f"[data] {src} 失败: {e}")

    raise DataSourceError("所有数据源均失败 -> " + " || ".join(errors))


if __name__ == "__main__":
    df, meta = load_klines(limit=C.LOOKBACK + C.LONG_LEN)
    print(df.tail(3).to_string())
    print(meta)
