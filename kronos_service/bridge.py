# -*- coding: utf-8 -*-
"""
Kronos 推理桥接层。

职责：
  1. 加载本地模型权重（离线，不联网）
  2. 对给定历史窗口做长/短期两段预测
  3. 把预测路径压缩为策略可用的信号指标（收益率 / MFE / MAE / 效率比 / 共识度 …）

关于 sample_count：
  模型内部对 sample_count 条路径做 mean（model/kronos.py），拿不到逐条路径。
  因此这里改为「把同一窗口复制 N 份放入 batch，sample_count=1」——
  等价于 N 条独立随机采样路径，且保留了每条路径的结果，可计算跨路径共识度。
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import config as C  # noqa: E402

# 离线模式：只用本地权重，不访问 HF
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, C.KRONOS_ROOT)  # noqa: E402  为了 import model

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


def _first_dir_with(candidates: list, filename: str = "model.safetensors") -> str:
    """取第一个包含指定权重文件的目录"""
    tried = []
    for d in candidates:
        if not d:
            continue
        p = os.path.join(d, filename)
        if os.path.isdir(d) and os.path.exists(p):
            return d
        tried.append(f"{d} (缺 {filename})")
    raise FileNotFoundError("找不到可用的模型目录:\n  " + "\n  ".join(tried))


# ------------------------------------------------------------------
# 指标计算
# ------------------------------------------------------------------
def compute_metrics(paths: list, ref_close: float) -> dict:
    """
    :param paths: list[DataFrame]，每个 DF 含 open/high/low/close/volume/amount，
                  行数 = 预测长度，按时间升序
    :param ref_close: 基准价（窗口最后一根实际收盘价）
    """
    #stack =[路径1[close1,close2,...,closeN],路径2[close1,close2,...,closeN],路径3[close1,close2,...,closeN],...]
    closes = np.stack([p["close"].to_numpy(dtype=float) for p in paths])
    highs = np.stack([p["high"].to_numpy(dtype=float) for p in paths])
    lows = np.stack([p["low"].to_numpy(dtype=float) for p in paths])

    # 代表路径 = 各路径均值，然后用均值计算ret，mfe，mae。或者可以先算每一条路径的ret，mfe，mae，再取均值。
    # c = [close1,close2,...,closeN]
    # h = [high1,high2,...,highN]
    # l = [low1,low2,...,lowN]
    c, h, l = closes.mean(axis=0), highs.mean(axis=0), lows.mean(axis=0)

    ret = float(c[-1] / ref_close - 1.0)          # 预期收益率,有正，也可能负
    mfe = float(h.max() / ref_close - 1.0)        # 最大有利波动，如果做空，则mfe和mae要对换
    mae = float(l.min() / ref_close - 1.0)        # 最大不利波动（通常为负）

    # 效率比 ER = |净位移| / 路径总行程。→1 单边趋势，→0 来回震荡
    #算er有两种方式，均值路径上算，各自算再平均。
    #方式一：均值路径上算
    # seq = np.concatenate([[ref_close], c])  #把 基准价 （窗口最后一根真实收盘价）拼到预测代表路径`c` （各采样路径的均值收盘价，长度 N）前面，得到长度 N+1 的完整序列
    # travel = float(np.abs(np.diff(seq)).sum()) #`np.diff` 得到相邻两根的价差（N 个），取绝对值求和 = 路径总行程 （单位：价格，如 USDT）
    # net = float((seq[-1] - seq[0]) / ref_close) #净位移，除以ref_close基准价，归一化，
    # er = float(abs(seq[-1] - seq[0]) / travel) if travel > 0 else 0.0  #效率比，净位移除以路径总行程，归一化
    # amplitude = float((h.max() - l.min()) / ref_close) #预测区间的 总振幅包络 ：代表路径最高点减最低点，归一化

    #注意，short_len过短，会导致ER计算中travel偏小，导致er计算错误。

    # 方式二（当前采用）：逐条采样路径各自算 ER 再取均值。
    # 不采用方式一（均值路径上算）的原因：ER 是路径的非线性函数，
    # 均值会把跨路径的反向抖动抵消掉（travel 偏小），ER 被系统性抬高，
    # 震荡市可能被误判为趋势。
    seqs = np.concatenate([np.full((closes.shape[0], 1), ref_close), closes], axis=1)  # (路径数, N+1)，首列为基准价
    travels = np.abs(np.diff(seqs, axis=1)).sum(axis=1)          # 各路径总行程（价格单位）
    er_vals = np.divide(np.abs(seqs[:, -1] - seqs[:, 0]), travels,
                        out=np.zeros(closes.shape[0]), where=travels > 0)  # 各路径 ER，travel=0 时记 0
    er = float(er_vals.mean())
    #travel = float(travel_vals.mean())
    #net = float((c[-1] - ref_close) / ref_close)                 # 净位移，归一化
    #amplitude = float((h.max() - l.min()) / ref_close)           # 总振幅包络，归一化

    # 多路径分歧：3 条采样路径对"终点涨跌"的看法一致吗？
    #因为我们对short_len上的分期尽量保持在一个单边行情上，所以这里算这个是意义的
    # consensus 对 sample_count 敏感，sample_count 越大，consensus 越准确。

    #取每条路径 最后一根预测收盘价 ，算各自相对基准价的收益率。
    # `closes` 形状是 (路径数, N)，所以`per` 是一维数组，长度=路径数。
    # 例如 sample_count=3 时可能是`[+0.012, +0.008, -0.003]`
    per = closes[:, -1] / ref_close - 1.0 

    if len(paths) >= 2:
        up_prob = float((per > 0).mean()) #看涨概率 ：几条路径终点高于现价。`[+, +, -]` → 0.67。这是 Monte Carlo 采样最直观的产物。
        path_std = float(per.std(ddof=0)) #终点收益率的标准差 ，分歧的绝对尺度
        consensus = float(abs(per.mean()) / (path_std + 1e-6)) #共识度 = |平均收益率| / 分歧度 ，本质上是一个信噪比（类似 t 统计量/夏普比率的思路）：
    else:
        # 单路径：无分歧信息，共识度不参与判定（见 decide_signal）
        up_prob = 1.0 if per[0] > 0 else 0.0
        path_std = 0.0
        consensus = 1.0 # 单路径：无分歧信息，共识度为1.0

    # ---- 逐根 K 线的横截面分歧（多路径在每一根预测 K 线上的发散程度）----
    # 上面的 per/path_std/consensus 只度量"终点"分歧，中间过程的信息会丢失。
    # 这里逐根度量：
    #   band_width  平均带宽 = 各路径在同一根 K 线上的 std 的均值（归一化）
    #               → 模型整体不确定度，越大越该降仓位
    #   band_slope  带宽随时间的增速（归一化，线性拟合斜率）
    #               → 预测锥张开快慢；越大表示越远的 K 线越不可信
    #   shape_corr  各路径"逐根收益"的两两平均相关系数
    #               → 形态/转折位置是否一致；低相关 = 连转折点都谈不拢
    #   mfe_std     各路径 MFE 的标准差 → 止盈位的不确定性
    #   mae_std     各路径 MAE 的标准差 → 止损位的不确定性（对风控最关键）
    #   er_std      各路径 ER 的标准差 → 形态判断本身的确定性
    # 单路径无分歧可言，全部置 0（与 path_std 哨兵值口径一致）。
    # if len(paths) >= 2:
    #     bar_std = closes.std(axis=0)                 # (N,) 每根 K 线的跨路径 std
    #     band_width = float(bar_std.mean() / ref_close)
    #     if bar_std.shape[0] >= 2:
    #         band_slope = float(np.polyfit(np.arange(bar_std.shape[0]), bar_std, 1)[0]
    #                            / ref_close)
    #     else:
    #         band_slope = 0.0
    #     mfe_std = float((highs.max(axis=1) / ref_close - 1.0).std(ddof=0))
    #     mae_std = float((lows.min(axis=1) / ref_close - 1.0).std(ddof=0))
    #     er_std = float(er_vals.std(ddof=0))
    #     rets = np.diff(closes, axis=1)               # (P, N-1) 各路径逐根收益
    #     # 需至少 2 个收益点、且每条路径都有波动，否则相关系数无定义（nan）
    #     if rets.shape[1] >= 2 and bool((rets.std(axis=1) > 1e-12).all()):
    #         corr = np.corrcoef(rets)
    #         p = len(paths)
    #         shape_corr = float((corr.sum() - p) / (p * (p - 1)))  # 非对角元均值
    #     else:
    #         shape_corr = 0.0
    # else:
    #     band_width = band_slope = 0.0
    #     mfe_std = mae_std = er_std = shape_corr = 0.0

    trend = "up" if ret > C.TREND_EPS else ("down" if ret < -C.TREND_EPS else "flat")
    if er >= C.ER_TREND and abs(ret) > C.TREND_EPS:
        regime = "trend" #单边趋势
    elif er < C.ER_CHOP:
        regime = "chop" #来回震荡
    else:
        regime = "mixed" #混合趋势，多个单边趋势

    return {
        "ret": ret, "mfe": mfe, "mae": mae, "close": float(c[-1]),
        "er": er,
        "up_prob": up_prob, "path_std": path_std, "consensus": consensus,
        # 逐根横截面分歧
        # "band_width": band_width, "band_slope": band_slope,
        # "shape_corr": shape_corr, "mfe_std": mfe_std,
        # "mae_std": mae_std, "er_std": er_std,
        "trend": trend, "regime": regime,
    }


def decide_signal(short_m: dict):
    """综合短期指标给出方向。策略侧可自行覆盖此逻辑。"""

    #最大有利波动 > 1.2 × 最大不利波动 （`MIN_RR=1.2` ）
    
    rr_long  = short_m["mfe"] > C.MIN_RR * abs(short_m["mae"])
    rr_short = abs(short_m["mae"]) > C.MIN_RR * short_m["mfe"]
    rr_ok = rr_long if short_m["ret"] > 0 else rr_short

    # 短期趋势必须是单边趋势
    regime_ok = short_m["regime"] == "trend" 

    # 单路径（`path_std==0` 哨兵值）没有分歧证据，直接放行；多路径要求短期共识度 ≥ 1.0（平均收益要盖过终点分歧
    cons_ok = (short_m["path_std"] == 0.0) or (short_m["consensus"] >= C.CONS_MIN)

    signal = "flat"
    #看短期预期收益率 定方向，`THETA=0.0035` 是最小预期收益门槛（覆盖手续费+滑点后还得有肉）
    if rr_ok and regime_ok : #and cons_ok
        if short_m["ret"] > C.THETA:
            signal = "long"
        elif short_m["ret"] < -C.THETA:
            signal = "short"

    #置信度=利润空间大小 × 共识强度
    if signal == "flat":
        confidence = 0.0
    else:
        confidence = float(
            min(abs(short_m["ret"]) / C.THETA, 1.0)
            * min(short_m["consensus"] / C.CONS_MIN, 1.0)
        )

    return signal, confidence, int(rr_ok)


# ------------------------------------------------------------------
# ZigZag 转折点
# ------------------------------------------------------------------
def zigzag(df: pd.DataFrame, min_bars_between: int = 5) -> list:
    """
    在一段 K 线序列上寻找 zigzag 转折点（基于 high/low 摆动极值）。

    转折阈值自适应：threshold = ATR(ZZ_ATR_PERIOD) * ZZ_ATR_MULT（价格单位，
    参数取自 config）。ATR 由序列末尾最近 ZZ_ATR_PERIOD 根的真实波幅
    （TR = max(high-low, |high-prev_close|, |low-prev_close|)）均值求得，
    随周期/价格/波动率自动缩放。

    算法：
      - 上行摆动中，用 high 创新高来延伸极值；当某根 K 线的 low 自极值
        高点回撤 >= threshold 时，确认该极值高点为转折点，方向翻转。
      - 下行摆动中，用 low 创新低来延伸极值；当某根 K 线的 high 自极值
        低点反弹 >= threshold 时，确认该极值低点为转折点，方向翻转。
      - 相邻转折点间隔不足 min_bars_between 根视为噪音，不记录该点
        （但方向仍然翻转，因为回撤事实上已发生）。

    :param df: K 线 DataFrame（按时间升序），必须含 high/low/close 列
    :param min_bars_between: 相邻转折点最小间隔 K 线数
    :return: [(idx, price, kind), ...] 按时间升序；
             price 为转折点所在 K 线的 high（'H'）或 low（'L'）
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("zigzag 入参必须是 pd.DataFrame（需含 high/low/close 列）")
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(high)
    if n < 2:
        return []

    # ---- ATR 自适应阈值（价格单位）----
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
    period = min(C.ZZ_ATR_PERIOD, n)
    delta = float(tr[-period:].mean()) * C.ZZ_ATR_MULT   # 最小回撤幅度（含倍数）

    pivots = []                      # [(idx, price, kind), ...]
    direction = 0                    # 0=未定向, 1=上行找高点, -1=下行找低点
    ext_idx, ext_price = 0, high[0]

    for i in range(1, n):
        if direction == 1:           # 上行中：high 创新高则延伸极值
            if high[i] > ext_price:
                ext_idx, ext_price = i, high[i]
            if low[i] <= ext_price - delta:          # 自极值高点回撤达标
                if not pivots or ext_idx - pivots[-1][0] >= min_bars_between:
                    pivots.append((ext_idx, ext_price, "H"))
                direction, ext_idx, ext_price = -1, i, low[i]
        elif direction == -1:        # 下行中：low 创新低则延伸极值
            if low[i] < ext_price:
                ext_idx, ext_price = i, low[i]
            if high[i] >= ext_price + delta:         # 自极值低点反弹达标
                if not pivots or ext_idx - pivots[-1][0] >= min_bars_between:
                    pivots.append((ext_idx, ext_price, "L"))
                direction, ext_idx, ext_price = 1, i, high[i]
        else:                        # 未定向：先按上行跟踪，low 回撤达标即确认首高点
            if high[i] > ext_price:
                ext_idx, ext_price = i, high[i]
            if low[i] <= ext_price - delta:
                pivots.append((ext_idx, ext_price, "H"))
                direction, ext_idx, ext_price = -1, i, low[i]
    return pivots


# ------------------------------------------------------------------
# 桥接类
# ------------------------------------------------------------------
class KronosBridge:
    def __init__(self):
        C.assert_context()
        torch.set_num_threads(C.CPU_THREADS)

        tok_dir = _first_dir_with(C.TOKENIZER_DIR_CANDIDATES)
        model_dir = _first_dir_with(C.MODEL_DIR_CANDIDATES)
        if C.VERBOSE:
            print(f"[bridge] tokenizer <- {tok_dir}")
            print(f"[bridge] model     <- {model_dir}")

        t0 = time.perf_counter()
        self.tokenizer = KronosTokenizer.from_pretrained(tok_dir)
        self.model = Kronos.from_pretrained(model_dir)
        self.predictor = KronosPredictor(
            self.model, self.tokenizer, device=C.DEVICE, max_context=C.MAX_CONTEXT
        )
        self.model.eval()
        if C.VERBOSE:
            n = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"[bridge] 加载完成 {n:.1f}M 参数, {time.perf_counter() - t0:.1f}s, "
                  f"device={C.DEVICE}")

    # ---------------- 推理 ----------------
    def _predict_batch(self, series: list, pred_len: int, sample_count: int):
        """
        批量预测，每个窗口预测 sample_count 条路径
        pred_len 预测长度，单位秒,长周期，短周期都使用这个函数预测，回测时同时预测每个窗口，也是这个

        :param series: list[(x_df, x_ts)]，x_df 列含 open/high/low/close/volume
        :return: list[list[DataFrame]]  外层对应 series，内层为 sample_count 条路径
        """
        tf_delta = pd.Timedelta(seconds=C.tf_seconds())
        flat_x, flat_x_ts, flat_y_ts, owner = [], [], [], []
        for i, (x_df, x_ts) in enumerate(series):
            #最后一根已知 K 线的下一个时间点为起点，按 K 线周期铺出未来 pred_len根 K 线的时间戳序列，
            y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + tf_delta, periods=pred_len, freq=tf_delta))
            for _ in range(sample_count):
                flat_x.append(x_df)
                flat_x_ts.append(x_ts)
                flat_y_ts.append(y_ts)
                owner.append(i)

        #预测时，自己手动复制sample_count份，然后调用predict_batch时传入sample_count=1
        # 便于获取多条原始预测路径，然后测算他们的共识度
        #返回的outs是list[DataFrame]，对应一个窗口的预测结果路径，每个DataFrame的列是 timestamps, close, high, low, open, volume
        outs = self.predictor.predict_batch(
            flat_x, flat_x_ts, flat_y_ts,
            pred_len=pred_len, T=C.TEMPERATURE, top_k=C.TOP_K, top_p=C.TOP_P,
            sample_count=1, verbose=C.VERBOSE,
        )

        #根据owner（1，2，3...）分组，返回list[outs]，每个outs是一个窗口的sample_count条预测结果
        groups = [[] for _ in series]
        for o, df in zip(owner, outs):
            groups[o].append(df)
        return groups

    # ---------------- 单次完整预测 ----------------
    def build_series(self, df: pd.DataFrame) -> tuple:
        """
        校验单个产品窗口并构造预测输入，供多产品批处理复用。

        :param df: 完整 K 线（已收盘），按时间升序
        :return: ((x_df, x_ts), info)，
                 info = {lookback, ref_close, origin_ts, origin_dt}
        """
        n = len(df)
        if n < C.MIN_LOOKBACK:
            raise ValueError(
                f"K 线不足：{n} < MIN_LOOKBACK={C.MIN_LOOKBACK}，本轮不预测"
            )

        # 实际窗口长度：目标 LOOKBACK，但数据不够时降级为全部可用 K 线
        lookback = min(n, C.LOOKBACK)
        if lookback < C.LOOKBACK and C.VERBOSE:
            print(f"[bridge] ⚠ 历史 K 线 {n} 根 < LOOKBACK={C.LOOKBACK}，"
                  f"降级使用 {lookback} 根预测（质量可能下降，~200 根以上才接近稳定）")

        #构造预测的窗口数据，获取最后的一根K线作为参考价格，获取最后k线的open时间作为origin_ts，单位秒，如14：00：00的秒格式
        x_df, x_ts = self._window(df, n - 1)
        info = {
            "lookback": lookback,
            "ref_close": float(x_df["close"].iloc[-1]),
            "origin_ts": int(x_ts.iloc[-1].value // 10 ** 9),
            "origin_dt": str(x_ts.iloc[-1]),
        }
        return (x_df, x_ts), info

    def run_once(self, df: pd.DataFrame, pair: str = None) -> dict:
        """
        :param df: 完整 K 线（已收盘），按时间升序；
                   至少 MIN_LOOKBACK 根，不足 LOOKBACK 时"有多少用多少"
        :param pair: 指定产品，默认 C.PAIR（多产品实例逐个传入）
        :return: {"main": {...}, "long_rows": [...], "short_rows": [...]}
        """
        pair = pair or C.PAIR
        (x_df, x_ts), info = self.build_series(df)

        t0 = time.perf_counter()
        #long_groups list[list[DataFrame]]  外层对应 series，内层为 sample_count 条路径
        #short_groups series都是1，所以为：[[df1,df2,df3,df4,df5]]
        long_groups, short_groups = self.predict_many([(x_df, x_ts)])

        #构建结果，因为这里只预测了一个窗口，所以long_groups[0]和short_groups[0]只包含一个DataFrame，回测哪里才有多个窗口的场景
        #long_groups[0] ：为 sample_count 条路径的预测结果
        res = self.build_result(
            long_groups[0], short_groups[0], info["ref_close"], info["origin_ts"],
            info["origin_dt"], lookback=info["lookback"], pair=pair,
        )
        res["long_paths"] = long_groups[0]       # 原始采样路径（供绘图等扩展用）
        res["short_paths"] = short_groups[0]
        res["timings"] = self.last_timings

        if C.VERBOSE:
            short_m = res["short_m"]
            print(f"[bridge] 预测完成 {time.perf_counter() - t0:.1f}s | "
                  f"ref={info['ref_close']:.1f} | "
                  f"S(ret={short_m['ret']:+.4f} er={short_m['er']:.2f} "
                  f"regime={short_m['regime']}) signal={res['main']['signal']} "
                  f"(confidence={res['main']['confidence']:.2f})")
        return res

    # ---------------- 窗口构造 ----------------
    @staticmethod
    def _window(df: pd.DataFrame, i: int):
        """取以第 i 根为结尾的窗口 -> (x_df, x_ts)

        窗口长度固定取 C.LOOKBACK；数据不足时起点自动钳到 0
        （"有多少用多少"，实际根数 < C.LOOKBACK 时由 run_once 记录到 A 表）。
        """
        start = max(0, i - C.LOOKBACK + 1)
        w = df.iloc[start: i + 1]
        x_df = w[["open", "high", "low", "close", "volume"]].copy()
        x_df["amount"] = x_df["volume"] * x_df[["high", "low", "close"]].mean(axis=1)
        return x_df.reset_index(drop=True), w["timestamps"].reset_index(drop=True)

    # ---------------- 批量预测（长 + 短） ----------------
    def get_short_len(self, x_df: pd.DataFrame, long_paths: list) -> int:
        """
        动态决定短期预测长度：
          1. 长期预测 60 根 + 最后 ZZ_HIST_BARS(10) 根历史 K 线组成序列
          2. zigzag 基于摆动方向上的 high/low 极值找转折点
             （阈值 = ATR(10)*multiplier 自适应，最小间隔 ZZ_MIN_BARS）
          3. 跳过属于历史区的转折点，取第一个落在预测区的转折点，
             其在预测区中的位置（第几根）即为短期长度
          4. 预测区没有转折点 -> 返回 C.LONG_LEN（覆盖整个长期视野）
        """
        n_hist = min(C.ZZ_HIST_BARS, len(x_df))
        # 近期历史 K 线（保留完整 OHLC，zigzag 使用 high/low/close）
        hist = x_df.iloc[-n_hist:].reset_index(drop=True)
        # 预测区代表路径 = 各路径均值（与 compute_metrics 口径一致）
        pred = pd.DataFrame({
            "high": np.mean([p["high"].to_numpy(dtype=float) for p in long_paths], axis=0),
            "low": np.mean([p["low"].to_numpy(dtype=float) for p in long_paths], axis=0),
            "close": np.mean([p["close"].to_numpy(dtype=float) for p in long_paths], axis=0),
        })
        # 将10条历史记录 + 60条长期预测结果（以将多条路径平均）组合为新的k线，去求zigzag折线
        seq = pd.concat([hist, pred], ignore_index=True)

        #求zigzag折线
        pivots = zigzag(seq, min_bars_between=C.ZZ_MIN_BARS)

        #没有转折点，返回LONG_LEN，有转折点，则分析跳过属于历史区的转折点，取第一个落在预测区的转折点
        for idx, _price, _kind in pivots:
            if idx >= n_hist:                    # 转折点落在预测区
                length = int((idx - n_hist + 1) * 1.2) #延后20%
                length = min(length, C.LONG_LEN)
                #如果length小于SHORT_LEN，就取SHORT_LEN的
                length = max(length, int(C.SHORT_LEN))
                return length
        
        return C.LONG_LEN


    def predict_many(self, series: list):
        """
        series: list[(x_df, x_ts)] -> (long_groups, short_groups)

        支持多产品/多窗口混批：predict_batch 要求**同批历史长度一致**，
        因此按 (历史长度) 分组分批，不同长度各自成批（不做截断，保留各自
        可用历史，冷启动降级时才不会丢数据）。

        短期长度只按主产品（series[0]，daemon 的 all_pairs 保证 PAIR 在首位）
        的「长期预测 + 最近历史 K 线」zigzag 转折点计算一次，
        其余窗口（辅助产品/其余 origin）统一复用同一长度，
        再按 (历史长度, 短期长度) 合并为一批推理。

        return long_groups, short_groups
                long_groups list[list[DataFrame]]  外层对应 series，内层为 sample_count 条路径

        耗时写 self.last_timings，供 daemon 打印（批量 vs 串行模式对比用）。
        """
        t_all = time.perf_counter()

        # ---------------- 长期预测：按历史长度分组 ----------------
        t0 = time.perf_counter()
        long_groups = [None] * len(series)
        buckets = {}
        for j, (x_df, _x_ts) in enumerate(series):
            buckets.setdefault(len(x_df), []).append(j)

        long_batches = []
        for ln, idxs in sorted(buckets.items(), reverse=True):
            outs = self._predict_batch([series[j] for j in idxs],
                                       C.LONG_LEN, C.SAMPLE_COUNT)
            for j, g in zip(idxs, outs):
                long_groups[j] = g
            long_batches.append(f"{ln}根×{len(idxs)}窗口")
        t_long = time.perf_counter() - t0

        # 短期长度：只算主产品（series[0]）一次，所有窗口复用同一长度
        main_len = self.get_short_len(series[0][0], long_groups[0])
        lengths = [main_len] * len(series)

        if C.SINGLE_ROLLOUT:
            #直接复制长期预测结果，不进行短期预测
            short_groups = [
                [p.iloc[:n].reset_index(drop=True) for p in g]
                for n, g in zip(lengths, long_groups)
            ]
            self.last_timings = {
                "windows": len(series), "long_sec": round(t_long, 2),
                "short_sec": 0.0, "total_sec": round(time.perf_counter() - t_all, 2),
                "long_batches": long_batches, "short_batches": [],
            }
            return long_groups, short_groups

        # ---------------- 短期预测：按 (历史长度, 短期长度) 分组 ----------------
        t0 = time.perf_counter()
        short_groups = [None] * len(series)
        buckets2 = {}
        for j, n in enumerate(lengths):
            buckets2.setdefault((len(series[j][0]), n), []).append(j)

        short_batches = []
        for (ln, n), idxs in sorted(buckets2.items(), reverse=True):
            outs = self._predict_batch([series[j] for j in idxs], n, C.SAMPLE_COUNT)
            for j, g in zip(idxs, outs):
                short_groups[j] = g
            short_batches.append(f"{ln}根→{n}根×{len(idxs)}窗口")
        t_short = time.perf_counter() - t0

        self.last_timings = {
            "windows": len(series),
            "long_sec": round(t_long, 2),
            "short_sec": round(t_short, 2),
            "total_sec": round(time.perf_counter() - t_all, 2),
            "long_batches": long_batches,
            "short_batches": short_batches,
        }
        return long_groups, short_groups

    # ---------------- 结果组装 ----------------
    @staticmethod
    def build_result(long_paths, short_paths, ref_close, origin_ts, origin_dt,
                     lookback: int = None, pair: str = None):
        """由预测路径组装 A 表字典 + B/C 表明细行

        :param lookback: 本次预测实际使用的历史窗口长度（降级时 < C.LOOKBACK），
                         原样写入 A 表 lookback 列，便于事后按窗口长度评估命中率
        :param pair: 指定产品，默认 C.PAIR（多产品实例逐个传入）
        """
        pair = pair or C.PAIR
        lookback = lookback or C.LOOKBACK
        #long_m = compute_metrics(long_paths, ref_close) #长期的可以不分析
        short_m = compute_metrics(short_paths, ref_close)
        signal, confidence, rr_ok = decide_signal(short_m)

        main = {
            "pair": pair, "timeframe": C.TIMEFRAME,
            "origin_ts": origin_ts, "origin_dt": origin_dt,
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ref_close": ref_close,
            "lookback": lookback,          # 实际窗口长度（数据不足时可能 < 配置值 420）
            "sample_count": C.SAMPLE_COUNT,
            "long_len": C.LONG_LEN,
            "short_len": len(short_paths[0]),   # 实际使用的动态短期长度
            "signal": signal, "confidence": confidence, "min_rr_ok": rr_ok,
        }
        for k, v in short_m.items():
            main[f"s_{k}"] = v
            

        return {
            "main": main,
            "long_rows": KronosBridge._to_rows(long_paths, origin_ts, pair),
            "short_rows": KronosBridge._to_rows(short_paths, origin_ts, pair),
            "short_m": short_m,
        }

    # ---------------- 明细行 ----------------
    @staticmethod
    def _to_rows(paths: list, origin_ts: int, pair: str = None) -> list:
        """各路径取均值，生成逐根预测 K 线明细"""
        pair = pair or C.PAIR
        cols = ["open", "high", "low", "close", "volume", "amount"]
        agg = {c: np.mean([p[c].to_numpy(dtype=float) for p in paths], axis=0)
               for c in cols if c in paths[0].columns}
        n = len(agg["close"])
        tf_sec = C.tf_seconds()
        rows = []
        for i in range(n):
            ts = origin_ts + (i + 1) * tf_sec
            rows.append([
                pair, C.TIMEFRAME, origin_ts, i + 1, ts,
                str(pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(None)),
                *[float(agg[c][i]) if c in agg else None for c in cols],
            ])
        return rows


if __name__ == "__main__":
    import datasource

    b = KronosBridge()
    df, meta = datasource.load_klines(limit=C.LOOKBACK + 10)
    res = b.run_once(df)
    m = res["main"]
    print("\n--- A 表字段 ---")
    for k, v in m.items():
        print(f"  {k:14s} = {v}")
    print(f"\n长期明细 {len(res['long_rows'])} 行, 短期明细 {len(res['short_rows'])} 行")
