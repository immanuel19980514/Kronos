# -*- coding: utf-8 -*-
"""
预测结果可视化。

参考 examples/prediction_example.py 的绘图逻辑，为守护进程输出：
  历史 K 线（LOOKBACK 根）+ 长期预测（LONG_LEN 根）+ 短期预测（动态 N 根）
  的代表路径（各采样路径均值），并以短期路径 min~max 区间带展示模型分歧，
  保存为 PNG 到 C.FIGURE_DIR。

文件名包含本次本地预测的关键信息：
  <pair>_<timeframe>_<origin时间>_<signal>_conf<x.xx>_slen<N>_<生成时刻>.png
  例：BTCUSDTUSDT_30m_20260724_0150_long_conf1.00_slen60_20261015_213001.png
"""
import os
import time

import numpy as np
import pandas as pd

import config as C
import store

import matplotlib
matplotlib.use("Agg")                       # 无窗口环境，只保存文件不弹窗
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Windows 下中文与负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# K 线颜色（绿涨红跌，与 freqtrade / 交易所口径一致）
_CANDLE_UP = "#26a69a"
_CANDLE_DOWN = "#ef5350"


def _candles(ax, t, df):
    """在 ax 上画历史 K 线蜡烛图（绿涨红跌）。t 为时间序列，df 含 OHLC 列"""
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    ax.xaxis_date()                            # 声明日期轴，后续 datetime 曲线可正常叠加
    x = mdates.date2num(pd.DatetimeIndex(pd.to_datetime(t)).to_pydatetime())
    width = C.tf_seconds() / 86400 * 0.7       # 蜡烛宽度（日期轴单位：天）
    colors = np.where(c >= o, _CANDLE_UP, _CANDLE_DOWN)
    eps = max(c.mean() * 1e-6, 1e-12)          # 十字星实体也保证可见
    ax.bar(x, np.maximum(np.abs(c - o), eps), width,
           bottom=np.minimum(o, c), color=colors, edgecolor=colors, lw=0.3)
    ax.vlines(x, l, h, colors=colors, lw=0.6)  # 上下影线
    ax.plot([], [], color=_CANDLE_UP, lw=3,
            label=f"历史 K 线({len(df)}根, 绿涨红跌)")   # 图例代理


def _mean(paths: list, col: str) -> np.ndarray:
    """各路径取均值（代表路径，与 compute_metrics 口径一致）"""
    return np.mean([p[col].to_numpy(dtype=float) for p in paths], axis=0)


def _minmax(paths: list, col: str):
    """跨路径 min / max（展示模型分歧带）"""
    arr = np.stack([p[col].to_numpy(dtype=float) for p in paths])
    return arr.min(axis=0), arr.max(axis=0)


def _pred_times(origin_ts: int, n: int) -> pd.DatetimeIndex:
    """预测区的时间轴：origin 之后逐根顺延"""
    tf_sec = C.tf_seconds()
    return pd.to_datetime([origin_ts + (i + 1) * tf_sec for i in range(n)], unit="s")


def figure_name(main: dict) -> str:
    """按预测关键信息构造文件名"""
    pair_tag = main["pair"].replace("/", "").replace(":", "")
    origin_tag = pd.Timestamp(main["origin_dt"]).strftime("%Y%m%d_%H%M")
    now_tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")   # 生成时刻，避免同 origin 覆盖
    return (f"{pair_tag}_{main['timeframe']}_{origin_tag}_"
            f"{main['signal']}_conf{main['confidence']:.2f}"
            f"_slen{main['short_len']}_{now_tag}.png")


def save_figure(df_hist: pd.DataFrame, long_paths: list, short_paths: list,
                main: dict, out_dir: str = None) -> str:
    """
    绘制「历史 + 长期预测 + 短期预测」图并保存，返回图片完整路径。

    :param df_hist: 历史 K 线（含 timestamps 列，取最后 LOOKBACK 根）
    :param long_paths: 长期预测路径列表（每条一个 DataFrame，由 bridge 返回）
    :param short_paths: 短期预测路径列表
    :param main: A 表结果字典（含 pair/timeframe/origin/signal/confidence 等）
    """
    out_dir = out_dir or C.FIGURE_DIR
    os.makedirs(out_dir, exist_ok=True)

    hist = df_hist.iloc[-C.LOOKBACK:].reset_index(drop=True)
    hist_t = pd.to_datetime(hist["timestamps"]).reset_index(drop=True)
    origin_ts = main["origin_ts"]
    long_t = _pred_times(origin_ts, len(long_paths[0]))
    short_t = _pred_times(origin_ts, len(short_paths[0]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # ---------------- 价格面板 ----------------
    _candles(ax1, hist_t, hist)                                  # 历史 K 线
    ax1.axvline(hist_t.iloc[-1], color="gray", ls="--", lw=0.8)     # 预测起点
    ax1.axhline(main["ref_close"], color="gray", ls=":", lw=0.8)    # 基准收盘价

    l_mean = _mean(long_paths, "close")
    ax1.plot(long_t, l_mean, color="red", lw=1.2,
             label=f"长期预测({main['long_len']}根)")

    # 只画短期路径分歧带：开仓决策依据的是短期预测，且短期有多条采样路径
    # （长期预测固定 sample_count=1，min~max 退化为一条线，分歧带无意义）
    s_mean = _mean(short_paths, "close")
    s_min, s_max = _minmax(short_paths, "close")
    ax1.fill_between(short_t, s_min, s_max, color="darkorange", alpha=0.15,
                     label="短期路径分歧带(min~max)")
    ax1.plot(short_t, s_mean, color="darkorange", lw=2.0,
             label=f"短期预测({main['short_len']}根)")
    ax1.axvline(short_t[-1], color="darkorange", ls=":", lw=1.0)    # 短期窗口右边界

    ax1.set_title(f"{main['pair']} {main['timeframe']} | origin {main['origin_dt']} | "
                    f"R/F/A={main['s_ret']:.4f}/{main['s_mfe']:.4f}/{main['s_mae']:.4f} | "
                    f"ER/UP={main['s_er']:.2f}/{main['s_up_prob']:.2f} | "
                    f"R/T {main['s_regime']}/{main['s_trend']} | "
                  f"S/C={main['signal']}/{main['confidence']:.2f}",
                  fontsize=12)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ---------------- 成交量面板 ----------------
    ax2.plot(hist_t, hist["volume"], color="#1f77b4", lw=1.0, label="历史 volume")
    ax2.plot(long_t, _mean(long_paths, "volume"), color="red", lw=1.0,
             label="预测 volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Time (UTC)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, figure_name(main))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# 总图：把一段时间内的所有预测序列画在同一个面板里
# ------------------------------------------------------------------
def _fmt_local(ts, fmt="%m-%d %H:%M") -> str:
    """unix 秒 -> 本地时间字符串（总图区间用本地时间标注，与输出时间点口径一致）"""
    return time.strftime(fmt, time.localtime(ts)) if ts is not None else "最早"


def _many_figure_name(kind: str, start, end) -> str:
    """总图文件名，命名风格与 figure_name 一致，区间为本地时间"""
    pair_tag = C.PAIR.replace("/", "").replace(":", "")
    now_tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")   # 生成时刻，避免覆盖
    return (f"{pair_tag}_{C.TIMEFRAME}_many_{kind}_"
            f"{_fmt_local(start, '%Y%m%d_%H%M')}-{_fmt_local(end, '%Y%m%d_%H%M')}"
            f"_{now_tag}.png")


def _many_figure(kind: str, df_hist: pd.DataFrame = None,
                 start: int = None, end: int = None, out_dir: str = None):
    """
    输出「总图」：读取明细表 origin_ts 落在 [start, end) 区间内的所有预测序列，
    与历史 K 线整合在同一个面板里 —— 最近 LOOKBACK 根历史 K 线（蜡烛图）为
    实际走势底色，预测序列按 origin_ts 顺序逐条叠加（半透明线 + 起点标记），
    每条序列以 (pair, timeframe, origin_ts) 唯一确定。

    :param df_hist: 历史 K 线（含 timestamps 列），展示最后 LOOKBACK 根
    :return: 图片完整路径；区间内无数据时返回 None
    """
    out_dir = out_dir or C.FIGURE_DIR
    os.makedirs(out_dir, exist_ok=True)

    table = C.TABLE_LONG if kind == "long" else C.TABLE_SHORT
    rows = store.load_list_range(table, start, end)
    if not rows:
        return None

    # 按 origin_ts 分组：同一 origin 的所有 step 构成一条预测序列
    groups = {}
    for r in rows:
        groups.setdefault(r["origin_ts"], []).append(r)

    color = "red" if kind == "long" else "darkorange"
    label = "长期" if kind == "long" else "短期"

    fig, ax = plt.subplots(figsize=(12, 6))

    # 历史 K 线：与 save_figure 口径一致，展示最后 LOOKBACK 根（蜡烛图）
    if df_hist is not None and len(df_hist):
        hist = df_hist.iloc[-C.LOOKBACK:]
        _candles(ax, hist["timestamps"], hist)

    for origin in sorted(groups):
        seq = sorted(groups[origin], key=lambda r: r["step"])
        t = pd.to_datetime([r["ts"] for r in seq], unit="s")
        close = [r["close"] for r in seq]
        ax.plot(t, close, color=color, lw=1.0, alpha=0.45)
        ax.plot(t[0], close[0], marker=".", ms=3, color=color, alpha=0.7)  # 各序列起点

    # 图例只放一条代理线，避免几十条序列把图例撑爆
    ax.plot([], [], color=color, lw=1.2,
            label=f"{label}预测路径（共{len(groups)}条）")
    ax.set_title(f"{C.PAIR} {C.TIMEFRAME} | {label}预测总图 | "
                 f"{_fmt_local(start)} ~ {_fmt_local(end)}（本地时间）", fontsize=12)
    ax.set_ylabel("Price")
    ax.set_xlabel("Time (UTC)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, _many_figure_name(kind, start, end))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def many_figure_long(df_hist: pd.DataFrame = None, start: int = None,
                     end: int = None, **kw):
    """长期预测总图（数据源 kr_predict_list_long + 历史 K 线）"""
    return _many_figure("long", df_hist, start, end, **kw)


def many_figure_short(df_hist: pd.DataFrame = None, start: int = None,
                      end: int = None, **kw):
    """短期预测总图（数据源 kr_predict_list_short + 历史 K 线）"""
    return _many_figure("short", df_hist, start, end, **kw)
