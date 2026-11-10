# -*- coding: utf-8 -*-
"""
Kronos 预测准确度评估脚本。

评估时，只跑长期，短期只用截取，sample_count=1

核心能力：对任意一组历史 K 线，批量预测后拿"预测" vs "真实"做度量，
可针对不同模型权重（微调前后）快速出同口径对比。

用法：
    python evaluate.py -c config/config_btc_5m_evaluate.json --feather-path data/BTC_USDT_USDT-5m-futures.feather --start 2026-09-01 --end 2026-10-27 --step 60 --limit 100 --save-csv data/btc_5m_evaluate.csv --save-fig

    python evaluate.py -c config/config.json --feather-path data/xx.csv \
        --model-dir models/Kronos-base --step 1

    每日数据库的验证
    python evaluate.py -c config/config_btc_5m_evaluate.json --from-db --source api 
    --start 2026-10-29 --end 2026-10-29 --step 1 --limit 300 --save-fig

输出：
    * 终端打印汇总指标（方向准确率、RMSE、信号命中率...）
    * --save-csv 写出逐行对比表
    * --compare 支持两次运行同一时间范围、同一个 feather，
        只是换 --model-dir，用来"微调前后"并排对比。

指标说明（全部基于**短期预测**，即 bridge.build_result 里的 short_paths）：
    dir_acc      方向准确率 = (预测↑ ∧ 真实↑ + 预测↓ ∧ 真实↓) / N
    dir_auc      用预测 ret 当 score 的方向 AUC（= 真阳性率在所有阈值下的面积）
    close_mae    最终收盘价 MAE（绝对误差 / 基准价，归一化）
    close_rmse   最终收盘价 RMSE（同上，根号）
    bar_mae      逐根 K 线 close 的 MAE（对 N 根取平均）
    bar_rmse     逐根 K 线 close 的 RMSE
    signal_hit   信号命中率：signal=long 时真实>0 算对；short 时真实<0 算对
    win_rate     有交易(signal≠flat)样本里，信号命中的比例
    mean_real_ret 所有样本的真实收益率均值 vs 仅交易样本的真实收益率均值
    ret_corr     预测 ret 与真实 ret 的 Pearson 相关系数

    MFE/MAE/ER 真实对比（预测值 vs 同 short_len 窗口真实走势）：
    mfe_bias     mean(pred_mfe - real_mfe)，>0 = 模型高估上行空间
    mae_bias     mean(pred_mae - real_mae)，>0 = 模型低估下行风险（止损会被打穿）
    er_bias      mean(pred_er  - real_er )，>0 = 模型以为比实际更单边
    mfe_corr / mae_corr / er_corr   预测值与真实值的相关性

    置信类指标校准（无真实对应物，看"说话算不算数"）：
    brier        up_prob 的 Brier score = mean((p - y)²)，越小越好（瞎猜 0.5 → 0.25）
    ece          up_prob 期望校准误差 = Σ 桶权重 × |桶内预测概率均值 - 实际频率|
    cons_spearman consensus 与方向命中的秩相关（>2/√N 视为显著单调）

模式：
    默认（重新推理）：读 feather/CSV，对 origins 跑 bridge.predict_many 再对比真实。
    --from-db：不重推理，直接读 sqlite kr_predict_main（+kr_predict_list_short
               拿逐根预测 close），与真实 K 线对比——度量的是线上实际发出的预测。

# 重新推理并出图（你要的场景：10 次预测）
python evaluate.py -c config/config_btc_5m_evaluate.json \
    --feather-path kronos_service\data\BTC_USDT_USDT-5m-futures.feather \
    --limit 10 --step 3 --save-fig

# 指定输出目录
--save-fig kronos_service\data

# 指定文件名
--save-fig kronos_service\data\my_cmp.png
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ---- -c 提前扫描：在 import config 之前设置环境变量（与 daemon.py 一致）----
for _i, _a in enumerate(sys.argv):
    if _a in ("-c", "--config") and _i + 1 < len(sys.argv):
        os.environ["KRONOS_CONFIG"] = sys.argv[_i + 1]
        break

import config as C  # noqa: E402 — 模块 import 时自动执行 load()
import datasource  # noqa: E402
import store  # noqa: E402

# API 单次下载超过该根数时提示改用 feather：freqtrade 的 live 下载持有
# 进程级 _loop_lock，长时间下载会让 /pair_history 完全卡死、连小请求都超时。
API_SAFE_MAX_BARS = 2000


def _warn_api_need(source: str, need: int):
    """source 为 None 或 api 时都会优先走 API；超过阈值就提示改用 feather"""
    if source != "feather" and need > API_SAFE_MAX_BARS:
        print(f"[evaluate] ⚠ 需从 API 拉取约 {need} 根 K 线（>{API_SAFE_MAX_BARS}）。"
              f"长 timerange 会让 freqtrade 的 live 下载长时间持有 _loop_lock，"
              f"导致后续请求全部超时。建议改用 --feather-path 或 --source feather。")

try:
    from bridge import KronosBridge  # noqa: E402
except ModuleNotFoundError:
    KronosBridge = None                # --from-db 模式不需要模型/torch


def _ts_arg(df: pd.DataFrame, s: str) -> pd.Timestamp:
    """命令行时间字符串转 Timestamp，时区与 df.timestamps 对齐，避免比较报错"""
    t = pd.Timestamp(s)
    if df["timestamps"].dt.tz is not None and t.tz is None:
        t = t.tz_localize("UTC")
    return t


# ======================================================================
# 1. 批量预测
# ======================================================================
def pick_origins(df: pd.DataFrame, limit: int, step: int,
                 start: str = None, end: str = None) -> list:
    """选出所有可以作为预测起点的索引（df 内部位置，不是 origin_ts）
    
    limit：最多返回 limit 个起点
    step：每个 step步长，跳过中间的k线不预测
    start：从这个时间开始
    end：到这个时间结束

    总 K 线 16732 
    可用起点索引范围 299 ~ 16671（ 16373 根） 前面的300根，后面的一段用来验证的60根
    `step=30` 时候选 origin 数 `(16671-299)//30 + 1` = 545 
    `limit=400` 实际只跑 400 次窗口预测 400 个 origin 覆盖的 K 线跨度 400×30 = 11980 根 ≈ 42 天

    注意：每个 origin 会同时做 长期（60 根）+ 短期 两段预测，`SAMPLE_COUNT=5` 条采样路径。
    所以模型实际 rollout 次数是`400 × 2 × 5 = 4000` 条路径，按`--batch 8` （默认）分 50 批送 GPU。

    另外，选取的origin尽量靠近最近的
    """
    n = len(df)                                     # 总k线条数
    lo = C.LOOKBACK - 1                             # 至少要有 LOOKBACK 根历史
    if start:
        lo = max(lo, int((df["timestamps"] < _ts_arg(df, start)).sum()))
    # i 之后至少还要留出 LONG_LEN 根当真实走势对照（保守）
    hi = n - 1 - C.LONG_LEN
    if end:
        hi = min(hi, int((df["timestamps"] <= _ts_arg(df, end)).sum()) - 1)
    if hi < lo:
        return []
    idxs = list(range(lo, hi + 1, step))
    if limit and len(idxs) > limit:
        idxs = idxs[-limit:]                        # 取最近的
    return idxs


def run_inference(df: pd.DataFrame, origins: list,
                 bridge: KronosBridge, batch: int = 8) -> list:
    """
    对每个 origin 跑预测，返回 list[eval_row]，每个元素形如：
        {
            "idx":         df 内部位置,
            "origin_ts":   预测发起时间,
            "ref_close":   基准价（窗口最后一根收盘价）,
            "short_len":   本次短期预测长度（动态值）,
            "pred_ret":    短期预期收益率（预测的）,
            "pred_signal": long / short / flat,
            "pred_conf":   置信度,
            "pred_paths":  short_paths（list[DataFrame]，供更细粒度对比）,
        }
    """
    results = []
    for bi in range(0, len(origins), batch):
        chunk = origins[bi:bi + batch]
        series = [bridge._window(df, i) for i in chunk]
        refs = [float(x["close"].iloc[-1]) for x, _ in series]
        ots = [int(ts.iloc[-1].value // 10 ** 9) for _, ts in series]
        odts = [str(ts.iloc[-1]) for _, ts in series]

        long_groups, short_groups = bridge.predict_many(series)

        for j in range(len(chunk)):
            # 只拿 short_paths，动态 short_len
            res = bridge.build_result(long_groups[j], short_groups[j],
                                      refs[j], ots[j], odts[j])
            short_m = res["short_m"]
            mean_closes = np.stack(
                [p["close"].to_numpy(dtype=float) for p in short_groups[j]]
            ).mean(axis=0)
            results.append({
                "idx": chunk[j],
                "origin_ts": ots[j],
                "ref_close": refs[j],
                "short_len": len(short_groups[j][0]),
                "pred_ret": short_m["ret"],
                "pred_mfe": short_m["mfe"],
                "pred_mae": short_m["mae"],
                "pred_er": short_m["er"],
                "pred_up_prob": short_m["up_prob"],
                "pred_consensus": short_m["consensus"],
                "pred_path_std": short_m["path_std"],
                "pred_close_last": float(mean_closes[-1]),
                "pred_bar_closes": mean_closes,      # ndarray, 长度=short_len
                "pred_signal": res["main"]["signal"],
                "pred_conf": res["main"]["confidence"],
                "pred_paths": short_groups[j],
            })
    return results


# ======================================================================
# 2. 真实走势 + 逐行对比
# ======================================================================
def attach_real(df: pd.DataFrame, rows: list) -> list:
    """给每个结果行拼上真实走势（同一 short_len），并算真实 ret/MFE/MAE/ER"""
    out = []
    for r in rows:
        i = r["idx"]
        sl = r["short_len"]
        # 真实走势 = origin 之后紧接的 sl 根 K 线
        real = df.iloc[i + 1: i + 1 + sl].reset_index(drop=True)
        if len(real) < sl:                           # 数据越界，跳过
            continue
        ref = r["ref_close"]
        real_closes = real["close"].to_numpy(dtype=float)
        real_close_last = float(real_closes[-1])
        real_ret = real_close_last / ref - 1.0
        real_mfe = float(real["high"].max()) / ref - 1.0
        real_mae = float(real["low"].min()) / ref - 1.0
        # 真实 ER：单条已实现路径 |净位移| / 总行程（首列拼基准价，与预测侧口径一致）
        seq = np.concatenate([[ref], real_closes])
        travel = float(np.abs(np.diff(seq)).sum())
        real_er = float(abs(seq[-1] - seq[0]) / travel) if travel > 0 else 0.0

        r = dict(r)
        r.update({
            "real_ret": real_ret,
            "real_close_last": real_close_last,
            "real_mfe": real_mfe,
            "real_mae": real_mae,
            "real_er": real_er,
            # 逐根真实 close 序列（ndarray，长度=sl）
            "real_closes": real_closes,
        })
        out.append(r)
    return out


def attach_pos(df: pd.DataFrame, rows: list) -> list:
    """--from-db 模式：按 origin_ts 把 db 行映射到 df 内部位置"""
    pos = {int(t.value // 10 ** 9): i for i, t in enumerate(df["timestamps"])}
    out = []
    for r in rows:
        i = pos.get(r["origin_ts"])
        if i is None:
            continue
        r = dict(r)
        r["idx"] = i
        out.append(r)
    return out


# ======================================================================
# 3. 度量指标
# ======================================================================
def _auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """简化版 ROC AUC（正样本在负样本之前出现的频率）"""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U 统计量
    ranks = np.concatenate([pos, neg])
    order = ranks.argsort().argsort() + 1            # 1-based rank
    n_pos = len(pos)
    U = order[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * len(neg)))


def _safe(x):
    return float(x) if not np.isnan(x) else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) > 1e-12 and np.std(b) > 1e-12:
        return float(np.corrcoef(a, b)[0, 1])
    return float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """秩相关（numpy 实现，避免 scipy 依赖）；|rho| > 2/√N 大致相当于 5% 显著"""
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    return _corr(rx, ry)


def calibrate_up_prob(rows: list, n_bins: int = 5) -> dict:
    """
    up_prob 校准：预测的"上涨概率" vs 实际上涨频率。
    完美校准 → 每桶 mean_pred ≈ realized_freq（45° 对角线）。
    gap > 0 = 过度自信；gap < 0 = 过度保守。
    """
    p = np.array([r["pred_up_prob"] for r in rows], dtype=float)
    y = np.array([r["real_ret"] > 0 for r in rows], dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        if m.sum() == 0:
            continue
        table.append({
            "bin": f"[{lo:.1f},{hi:.1f}]",
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "realized_freq": float(y[m].mean()),
            "gap": float(p[m].mean() - y[m].mean()),
        })
    brier = float(np.mean((p - y) ** 2))
    ece = float(sum(t["n"] * abs(t["gap"]) for t in table) / len(rows))
    return {"bins": table, "brier": brier, "ece": ece}


def calibrate_consensus(rows: list, n_bins: int = 5) -> dict:
    """
    consensus 单调性验证：按分位数分桶，看方向命中率是否随共识度上升。
    只在 signal≠flat 的样本上评估（flat 是"不下注"，方向命中无意义）；
    若非 flat 样本太少则退化为全样本。
    """
    sig = np.array([r["pred_signal"] for r in rows])
    use = [r for r, s in zip(rows, sig) if s != "flat"]
    scope = "non_flat"
    if len(use) < 20:
        use = rows
        scope = "all"
    p = np.array([r["pred_ret"] for r in use], dtype=float)
    c = np.array([r["pred_consensus"] for r in use], dtype=float)
    y = np.array([r["real_ret"] for r in use], dtype=float)
    hit = (np.sign(p) == np.sign(y)).astype(float)

    q = np.quantile(c, np.linspace(0, 1, n_bins + 1))
    table = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (c >= lo) & (c <= hi if hi == q[-1] else c < hi)
        if m.sum() == 0:
            continue
        table.append({
            "consensus_range": f"[{lo:.2f},{hi:.2f}]",
            "n": int(m.sum()),
            "hit_rate": float(hit[m].mean()),
        })
    rho = _spearman(c, hit)
    sig_flag = bool(abs(rho) > 2 / np.sqrt(len(use))) if not np.isnan(rho) else False
    return {"bins": table, "spearman_rho": rho, "significant": sig_flag, "scope": scope}


def summarize(rows: list) -> tuple:
    """
    把 attach_real 后的 rows 做汇总。
    :return: (单行 DataFrame, extras)，extras 里是校准分桶明细表
    """
    if not rows:
        return pd.DataFrame([{"n": 0}]), {}

    pred_ret = np.array([r["pred_ret"] for r in rows], dtype=float)
    real_ret = np.array([r["real_ret"] for r in rows], dtype=float)
    ref_closes = np.array([r["ref_close"] for r in rows], dtype=float)
    real_last = np.array([r["real_close_last"] for r in rows], dtype=float)
    pred_last = np.array([r["pred_close_last"] for r in rows], dtype=float)

    # --- 方向准确率 ---
    pred_dir = np.sign(pred_ret)
    real_dir = np.sign(real_ret)
    # flat(0) 预测不算方向
    non_flat = pred_dir != 0
    if non_flat.sum() > 0:
        dir_hit = ((pred_dir[non_flat] > 0) & (real_dir[non_flat] > 0)
                   | (pred_dir[non_flat] < 0) & (real_dir[non_flat] < 0)).mean()
    else:
        dir_hit = float("nan")

    # --- 方向 AUC ---
    y_dir = (real_ret > 0).astype(int)
    auc = _auc_binary(y_dir, pred_ret)

    # --- 最终收盘价误差（归一化到 ref）---
    diff = (pred_last - real_last) / ref_closes
    close_mae = float(np.abs(diff).mean())
    close_rmse = float(np.sqrt((diff ** 2).mean()))

    # --- 逐根 close 误差（db 模式下可能缺 pred_bar_closes，逐行跳过）---
    bar_abs = []
    bar_sq = []
    for r in rows:
        pred_c = r.get("pred_bar_closes")
        if pred_c is None or len(pred_c) != len(r["real_closes"]):
            continue
        d = (pred_c - r["real_closes"]) / r["ref_close"]
        bar_abs.append(np.abs(d).mean())
        bar_sq.append((d ** 2).mean())
    bar_mae = float(np.mean(bar_abs)) if bar_abs else float("nan")
    bar_rmse = float(np.sqrt(np.mean(bar_sq))) if bar_sq else float("nan")

    # --- 信号命中率 ---
    signals = np.array([r["pred_signal"] for r in rows])
    trades = signals != "flat"                       # 实际"下注"的样本
    if trades.sum() > 0:
        hit = np.zeros(trades.sum(), dtype=int)
        # signal=long 时真实 ret>0 命中；signal=short 时真实 ret<0 命中
        long_idx = (signals[trades] == "long")
        short_idx = (signals[trades] == "short")
        real_trades = real_ret[trades]
        hit[long_idx] = (real_trades[long_idx] > 0)
        hit[short_idx] = (real_trades[short_idx] < 0)
        signal_hit = float(hit.mean())
        win_rate = signal_hit
        mean_real_trade = float(real_trades.mean())
        mean_real_win = float(real_trades[hit == 1].mean()) if hit.sum() > 0 else float("nan")
        mean_real_loss = float(real_trades[hit == 0].mean()) if (hit == 0).sum() > 0 else float("nan")
    else:
        signal_hit = win_rate = float("nan")
        mean_real_trade = mean_real_win = mean_real_loss = float("nan")

    # --- 相关性 ---
    ret_corr = _corr(pred_ret, real_ret)

    # --- MFE / MAE / ER 真实对比 ---
    pred_mfe = np.array([r["pred_mfe"] for r in rows], dtype=float)
    real_mfe = np.array([r["real_mfe"] for r in rows], dtype=float)
    pred_mae = np.array([r["pred_mae"] for r in rows], dtype=float)
    real_mae = np.array([r["real_mae"] for r in rows], dtype=float)
    pred_er = np.array([r["pred_er"] for r in rows], dtype=float)
    real_er = np.array([r["real_er"] for r in rows], dtype=float)
    mfe_bias = float((pred_mfe - real_mfe).mean())
    mae_bias = float((pred_mae - real_mae).mean())
    er_bias = float((pred_er - real_er).mean())

    # --- 置信类指标校准 ---
    up_cal = calibrate_up_prob(rows)
    cons_cal = calibrate_consensus(rows)

    # --- 分布 ---
    flat_pct = float((signals == "flat").mean())
    long_pct = float((signals == "long").mean())
    short_pct = float((signals == "short").mean())

    row = {
        "n": len(rows),
        "mean_short_len": float(np.mean([r["short_len"] for r in rows])),
        "dir_acc": _safe(dir_hit),
        "dir_auc": _safe(auc),
        "close_mae": _safe(close_mae),
        "close_rmse": _safe(close_rmse),
        "bar_mae": _safe(bar_mae),
        "bar_rmse": _safe(bar_rmse),
        "signal_hit": _safe(signal_hit),
        "win_rate": _safe(win_rate),
        "mean_real_ret": _safe(float(real_ret.mean())),
        "mean_real_trade": _safe(mean_real_trade),
        "mean_real_win": _safe(mean_real_win),
        "mean_real_loss": _safe(mean_real_loss),
        "flat_pct": _safe(flat_pct),
        "long_pct": _safe(long_pct),
        "short_pct": _safe(short_pct),
        "ret_corr": _safe(ret_corr),
        # MFE/MAE/ER 真实对比
        "mfe_bias": _safe(mfe_bias),
        "mae_bias": _safe(mae_bias),
        "er_bias": _safe(er_bias),
        "mfe_corr": _safe(_corr(pred_mfe, real_mfe)),
        "mae_corr": _safe(_corr(pred_mae, real_mae)),
        "er_corr": _safe(_corr(pred_er, real_er)),
        # 校准
        "brier": _safe(up_cal["brier"]),
        "ece": _safe(up_cal["ece"]),
        "cons_spearman": _safe(cons_cal["spearman_rho"]),
        # 原始数组（方便后续画图/调试）
        "_pred_ret": pred_ret,
        "_real_ret": real_ret,
        "_pred_signal": signals,
    }
    extras = {"up_prob_cal": up_cal, "consensus_cal": cons_cal}
    return pd.DataFrame([row]), extras


# ======================================================================
# 4. 逐行明细 DataFrame（供 --save-csv）
# ======================================================================
def detail_table(rows: list) -> pd.DataFrame:
    """把 rows 展开成可存 CSV 的表格（不存 pred_paths 这种大对象）"""
    drop = {"pred_paths", "real_closes", "pred_bar_closes"}
    out = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in drop}
        out.append(d)
    return pd.DataFrame(out)


# ======================================================================
# 4b. --from-db：读 sqlite 已存预测（不重新推理）
# ======================================================================
def load_db_rows(pair: str, timeframe: str, start: str = None,
                 end: str = None, limit: int = 0) -> list:
    """
    从 kr_predict_main 读 daemon 已写入的预测，附 kr_predict_list_short
    的逐根预测 OHLC（供 bar 级误差与 K 线绘图）。返回与 run_inference
    相同结构的 rows（idx 待 attach_pos 填充）。
    """
    sql = (f"SELECT origin_ts, ref_close, short_len, "
           f"s_ret, s_mfe, s_mae, s_er, s_up_prob, s_consensus, s_path_std, "
           f"s_close, signal, confidence "
           f"FROM {C.TABLE_MAIN} WHERE pair=? AND timeframe=?")
    params = [pair, timeframe]
    if start:
        sql += " AND origin_ts>=?"
        params.append(int(pd.Timestamp(start, tz="UTC").timestamp()))
    if end:
        sql += " AND origin_ts<=?"
        params.append(int(pd.Timestamp(end, tz="UTC").timestamp()))
    sql += " ORDER BY origin_ts DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"

    conn = store.connect()
    m = pd.read_sql_query(sql, conn, params=params)
    if len(m) == 0:
        conn.close()
        return []
    m = m.sort_values("origin_ts").reset_index(drop=True)

    # 逐根预测 OHLC（范围查询，避免 IN 变量数超限）
    s = pd.read_sql_query(
        f"SELECT origin_ts, step, open, high, low, close FROM {C.TABLE_SHORT} "
        f"WHERE pair=? AND timeframe=? AND origin_ts BETWEEN ? AND ? "
        f"ORDER BY origin_ts, step",
        conn, params=[pair, timeframe,
                      int(m["origin_ts"].min()), int(m["origin_ts"].max())])
    conn.close()
    bar_map, path_map = {}, {}
    for ots, g in s.groupby("origin_ts"):
        g = g.sort_values("step")
        bar_map[int(ots)] = g["close"].to_numpy(dtype=float)
        path_map[int(ots)] = g[["open", "high", "low", "close"]].reset_index(drop=True)

    rows = []
    for r in m.itertuples():
        rows.append({
            "idx": None,
            "origin_ts": int(r.origin_ts),
            "ref_close": float(r.ref_close),
            "short_len": int(r.short_len),
            "pred_ret": float(r.s_ret),
            "pred_mfe": float(r.s_mfe),
            "pred_mae": float(r.s_mae),
            "pred_er": float(r.s_er),
            "pred_up_prob": float(r.s_up_prob),
            "pred_consensus": float(r.s_consensus),
            "pred_path_std": float(r.s_path_std),
            "pred_close_last": float(r.s_close),
            "pred_bar_closes": bar_map.get(int(r.origin_ts)),
            # 单条路径：DB 里存的就是各路径均值，直接当作一条路径画 K 线
            "pred_paths": ([path_map[int(r.origin_ts)]]
                           if int(r.origin_ts) in path_map else None),
            "pred_signal": r.signal,
            "pred_conf": float(r.confidence),
        })
    return rows


# ======================================================================
# 4c. 预测 vs 实际 可视化
# ======================================================================
def _default_fig_path(n: int, pair: str) -> str:
    """默认输出到 data 目录，文件名带样本数与生成时刻"""
    tag = pair.replace("/", "").replace(":", "")
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_HERE, "data",
                        f"eval_pred_{tag}_{C.TIMEFRAME}_{n}p_{ts}.png")


# 绿涨红跌：实际 K 线用深色实心，预测 K 线用同色半透明
_UP_DARK, _DOWN_DARK = "#128a72", "#d32f2f"
_UP, _DOWN = "#26a69a", "#ef5350"


def plot_predictions(df: pd.DataFrame, rows: list, out_path: str,
                     max_panels: int = 20) -> list:
    """
    把若干次预测的 short_len 窗口与实际 K 线画成对比图，返回图片路径列表。

    max_panels 控制单张图最多放几个面板：
        rows 数 > max_panels 时自动分页，输出 _p1、_p2 ... 多张图；
        max_panels <= 0 时全部塞进一张（面板多、图会很大）。

    每个面板 = 一次预测，两层 K 线叠在同一 x 位置：
        预测 K 线：半透明绿涨红跌，实体加宽（宽度 = 1 根 K 线）
        实际 K 线：深色实心绿涨红跌，实体减窄（宽度 = 0.5 根），画在上层
        灰色虚线 = 预测基准价 ref_close
    预测与实际同向时颜色叠加，反向时外围露出预测色 → 一眼看出偏差。
    --from-db 模式没有预测 OHLC，退化为橙色折线（预测 close）。
    """
    if max_panels and len(rows) > max_panels:
        pages = [rows[i:i + max_panels] for i in range(0, len(rows), max_panels)]
    else:
        pages = [rows]

    outs = []
    for idx, page in enumerate(pages, 1):
        suffix = "" if len(pages) == 1 else f"_p{idx}"
        page_path = (out_path[:-4] + suffix + ".png"
                     if out_path.lower().endswith(".png") else out_path + suffix)
        outs.append(_plot_page(df, page, page_path, len(pages), idx))
    return outs


def _plot_page(df: pd.DataFrame, rows: list, out_path: str,
               total_pages: int = 1, page_idx: int = 1) -> str:
    """绘制一页对比图并保存，返回图片路径"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(rows)
    if n == 0:
        return out_path
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 2.6 * nrows),
                             squeeze=False)
    tf_sec = C.tf_seconds()

    for k, r in enumerate(rows):
        ax = axes[k // ncols][k % ncols]
        i, sl = r["idx"], r["short_len"]
        real = df.iloc[i + 1: i + 1 + sl]
        if len(real) < sl:
            continue
        rt = pd.DatetimeIndex(pd.to_datetime(real["timestamps"]))
        x = mdates.date2num(rt.to_pydatetime())

        w = tf_sec / 86400 * 0.7
        ac = real["close"].to_numpy(dtype=float)
        eps = max(float(ac.mean()) * 1e-6, 1e-12)
        pt = pd.to_datetime(
            [r["origin_ts"] + (j + 1) * tf_sec for j in range(sl)], unit="s")

        # ---- 预测 K 线：半透明绿涨红跌，实体加宽，先画（在下层）----
        if r.get("pred_paths"):
            po = np.mean([p["open"].to_numpy(dtype=float)
                          for p in r["pred_paths"]], axis=0)
            ph = np.mean([p["high"].to_numpy(dtype=float)
                          for p in r["pred_paths"]], axis=0)
            pl = np.mean([p["low"].to_numpy(dtype=float)
                          for p in r["pred_paths"]], axis=0)
            pc = np.mean([p["close"].to_numpy(dtype=float)
                          for p in r["pred_paths"]], axis=0)
            pcol = np.where(pc >= po, _UP, _DOWN)
            ax.bar(pt, np.maximum(np.abs(pc - po), eps), w,
                   bottom=np.minimum(po, pc), color=pcol, edgecolor=pcol,
                   alpha=0.32, lw=0.3, zorder=2,
                   label="预测 K 线(半透明)" if k == 0 else None)
            ax.vlines(pt, pl, ph, colors=pcol, lw=0.9, alpha=0.32, zorder=2)
        elif r.get("pred_bar_closes") is not None:
            # DB 模式只有预测 close，退化为折线
            pm = np.asarray(r["pred_bar_closes"], dtype=float)[:sl]
            if len(pm) == sl:
                ax.plot(pt, pm, color="darkorange", lw=1.6, marker="o",
                        ms=2.5, zorder=3,
                        label="预测 close(DB)" if k == 0 else None)

        # ---- 实际 K 线：深色实心，实体减窄，画在上层 ----
        ao = real["open"].to_numpy(dtype=float)
        ah = real["high"].to_numpy(dtype=float)
        al = real["low"].to_numpy(dtype=float)
        acol = np.where(ac >= ao, _UP_DARK, _DOWN_DARK)
        ax.bar(x, np.maximum(np.abs(ac - ao), eps), w * 0.5,
               bottom=np.minimum(ao, ac), color=acol, edgecolor=acol,
               lw=0.3, zorder=3,
               label="实际 K 线(深色)" if k == 0 else None)
        ax.vlines(x, al, ah, colors=acol, lw=1.0, zorder=3)

        ax.axhline(r["ref_close"], color="gray", ls=":", lw=0.8)
        ax.set_xlim(x[0] - w, x[-1] + w)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

        # ---- 标题：方向对错 + 期间收益 ----
        hit = np.sign(r["pred_ret"]) == np.sign(r["real_ret"])
        mark = "方向对" if hit else "方向错"
        color = "#26a69a" if hit else "#ef5350"
        ax.set_title(
            f"{pd.Timestamp(r['origin_ts'], unit='s').strftime('%m-%d %H:%M')} "
            f"{r['pred_signal']}({r['short_len']}根) {mark} "
            f"pred {r['pred_ret']:+.4f} / real {r['real_ret']:+.4f}",
            fontsize=8, color=color)

    # 空面板隐藏
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=3)
    page_tag = f"第 {page_idx}/{total_pages} 页，" if total_pages > 1 else ""
    fig.suptitle(f"预测 vs 实际（{page_tag}{n} 次 × {C.TIMEFRAME}，"
                 f"绿色=方向对 / 红色=方向错）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ======================================================================
# 5. 主流程
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Kronos 预测准确度评估")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--from-db", action="store_true", default=False,
                    help="不重新推理：直接读 sqlite kr_predict_main 已存预测做评估")
    
    ap.add_argument("--pair", default=C.PAIR,
                    help=f"产品（默认配置 PAIR = {C.PAIR}）")
    ap.add_argument("--timeframe", default=C.TIMEFRAME,
                    help=f"时间粒度（默认 {C.TIMEFRAME}）")

    ap.add_argument("--source", default=None, choices=[None, "api", "feather"],
                    help="K 线数据源（默认按配置文件config.json中的优先级，先 api 后 feather）")
    ap.add_argument("--feather-path", default=None,
                    help="K 线 CSV/feather 文件路径（列：timestamps,open,high,low,close,volume,amount）")

    ap.add_argument("--start", default=None, help="起始时间 ISO，如 2025-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=300,
                    help="最多评估多少个 origin（取最后 N 个）")
    ap.add_argument("--step", type=int, default=3,
                    help="每隔几根 K 线评估一次 origin（=1 表示每根都评）")

    ap.add_argument("--batch", type=int, default=8,
                    help="一次送 bridge.predict_many 的窗口数")
    ap.add_argument("--save-csv", default=None,
                    help="写出逐行对比表 CSV 路径")

    ap.add_argument("--save-fig", nargs="?", const="", default=None,
                    help="输出「预测 vs 实际」对比图；给目录则自动命名，"
                         "不给值则存到 kronos_service/data/")
    ap.add_argument("--fig-panels", type=int, default=20,
                    help="单张对比图最多几个面板，超出自动分页（0=全部放一张，默认 20）")
    ap.add_argument("--fig-limit", type=int, default=0,
                    help="对比图总共画多少次预测（从 --limit 结果里取最近 N 个；"
                         "0=全部，建议取 --fig-panels 的倍数）")

    ap.add_argument("--compare", action="store_true",
                    help="连续跑两次（先 base，再 --model-dir 指定的），并排对比")    
    ap.add_argument("--model-dir", default=None,
                    help="指定另一个 KronosPredictor 目录（覆盖 config.KRONOS_ROOT）")
    
    args = ap.parse_args()

    C.TIMEFRAME = args.timeframe
    C.assert_context()
    print(f"[evaluate] 配置 <- {C.CONFIG_PATH}")
    print(f"[evaluate] LOOKBACK={C.LOOKBACK} LONG_LEN={C.LONG_LEN} "
          f"SHORT_LEN={C.SHORT_LEN} SAMPLE_COUNT={C.SAMPLE_COUNT} "
          f"SINGLE_ROLLOUT={C.SINGLE_ROLLOUT} DEVICE={C.DEVICE}")

    def _load_df_file(path):
        """从 CSV/feather 文件加载 K 线，复用 datasource 的列名归一化
        （freqtrade feather 的时间列名是 date，不是 timestamps）"""
        df_ = pd.read_feather(path) if path.endswith(".feather") else pd.read_csv(path)
        return datasource._normalize(df_)

    def _fig_path(n: int, suffix: str = "") -> str:
        """--save-fig 的值是目录则自动命名，是 .png 则直接使用"""
        v = args.save_fig
        name = os.path.basename(_default_fig_path(n, args.pair))
        if suffix:
            name = name.replace(".png", f"_{suffix}.png")
        if not v:
            return os.path.join(_HERE, "data", name)
        if v.lower().endswith(".png"):
            return v if not suffix else v.replace(".png", f"_{suffix}.png")
        return os.path.join(v, name)

    def _maybe_fig(rows_: list, suffix: str = ""):
        if args.save_fig is None or not rows_:
            return
        if args.fig_limit:
            rows_ = rows_[-args.fig_limit:]
        try:
            paths = plot_predictions(df, rows_, _fig_path(len(rows_), suffix),
                                     max_panels=args.fig_panels)
            print(f"[evaluate] 对比图 {len(paths)} 张（{len(rows_)} 次预测）:")
            for p in paths:
                print(f"[evaluate]   -> {p}")
        except Exception as e:                               # noqa: BLE001
            print(f"[evaluate] 绘图失败（不影响指标）: {type(e).__name__}: {e}")

    # --- from-db 模式：不重推理，度量线上已发出的预测 ---
    if args.from_db:
        # 先不截断：limit 只应限制「可验证」的样本数，
        # 若先取最新 N 条，它们可能还没走完 short_len，会导致对齐为 0
        rows = load_db_rows(args.pair, C.TIMEFRAME, args.start, args.end, 0)
        print(f"[evaluate] DB 预测 {len(rows)} 条 <- {C.DB_PATH} "
              f"(pair={args.pair}, tf={C.TIMEFRAME})")
        if not rows:
            return
        if args.feather_path:
            df = _load_df_file(args.feather_path)
            src = args.feather_path
        else:
            # 按 DB 里最早的 origin 倒推所需 K 线根数，从 API/feather 拉取
            tf_sec = C.tf_seconds()
            need = int((time.time() - min(r["origin_ts"] for r in rows)) / tf_sec) \
                + max(r["short_len"] for r in rows) + 10
            _warn_api_need(args.source, need)
            df, meta = datasource.load_klines(limit=need, source=args.source,
                                              pair=args.pair)
            src = meta["source"]
        print(f"[evaluate] K线 <- {src} | {len(df)} 根 | "
              f"{df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
        rows = attach_pos(df, rows)
        rows = attach_real(df, rows)
        print(f"[evaluate] 能对齐真实走势的样本 {len(rows)} 条")
        if not rows:
            print("[evaluate] 对齐为 0：K 线数据未覆盖 DB 预测的时间范围，"
                  "检查数据源是否为最新（可试 --source api 或指定 --feather-path）")
            return
        if args.limit and len(rows) > args.limit:
            rows = rows[-args.limit:]        # 取最近 limit 条「可验证」样本
        summary, extras = summarize(rows)
        _print_summary(summary.iloc[0], extras)
        if args.save_csv:
            detail_table(rows).to_csv(args.save_csv, index=False)
        _maybe_fig(rows)
        return

    # --- 加载 K 线（重新推理模式）---
    if args.feather_path:
        df = _load_df_file(args.feather_path)
        path = args.feather_path
    else:
        # 与 from-db 一致：用 --pair / --source（不指定时按 _source 优先级
        # 先 api 后 feather），并按需要的根数反推起点
        tf_sec = C.tf_seconds()
        if args.start:
            need = int((time.time()
                        - pd.Timestamp(args.start, tz="UTC").timestamp()) / tf_sec) \
                + C.LOOKBACK + C.LONG_LEN + 10
        else:
            need = (C.LOOKBACK + C.LONG_LEN
                    + max(1, args.limit or 1) * max(1, args.step) + 10)
        _warn_api_need(args.source, need)
        df, meta = datasource.load_klines(limit=need, source=args.source,
                                          pair=args.pair)
        path = f"{meta['source']}@{args.pair} ({meta['rows']} 根)"
        if args.end:
            df = df[df["timestamps"] <= pd.Timestamp(args.end, tz="UTC")]
            df = df.reset_index(drop=True)
    print(f"[evaluate] 数据 <- {path}")
    print(f"[evaluate] {len(df)} 根 | {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")

    if KronosBridge is None:
        print("[evaluate] 未安装 torch/模型依赖，仅 --from-db 模式可用")
        return

    origins = pick_origins(df, args.limit, args.step, args.start, args.end)
    print(f"[evaluate] origins = {len(origins)} (step={args.step}, 区间={args.start}~{args.end})")
    if not origins:
        print("[evaluate] 没有可用 origin，退出")
        return

    # --- 单次评估 ---
    def _run_one(tag: str, model_dir: str = None):
        prev_root = C.KRONOS_ROOT
        if model_dir:
            C.KRONOS_ROOT = model_dir
        t0 = time.perf_counter()
        bridge = KronosBridge()
        rows = run_inference(df, origins, bridge, args.batch)
        rows = attach_real(df, rows)
        summary, extras = summarize(rows)
        elapsed = time.perf_counter() - t0
        print(f"\n{'='*68}\n[{tag}] 耗时 {elapsed:.1f}s | 有效样本 {len(rows)}")
        _print_summary(summary.iloc[0], extras)
        C.KRONOS_ROOT = prev_root
        return summary, rows

    # config.KRONOS_ROOT 对应 base 模型（finetune 前 / 默认）
    if args.compare:
        s_base, rows_base = _run_one("BASE", None)
        if not args.model_dir:
            print("--compare 需要同时给 --model-dir 指定微调后模型目录")
            return
        s_new, rows_new = _run_one("FINETUNED", args.model_dir)
        merged = pd.concat([s_base.assign(tag="base"), s_new.assign(tag="finetuned")],
                           ignore_index=True)
        _print_compare(merged)
        if args.save_csv:
            detail_table(rows_base).to_csv(args.save_csv.replace(".csv", "_base.csv"),
                                            index=False)
            detail_table(rows_new).to_csv(args.save_csv, index=False)
        _maybe_fig(rows_base, "base")
        _maybe_fig(rows_new, "ft")
    else:
        _, rows = _run_one("RUN", args.model_dir)
        if args.save_csv:
            detail_table(rows).to_csv(args.save_csv, index=False)
        _maybe_fig(rows)


def _print_summary(row: pd.Series, extras: dict = None):
    """单模型汇总打印"""
    keys = ["n", "mean_short_len", "dir_acc", "dir_auc",
            "close_mae", "close_rmse", "bar_mae", "bar_rmse",
            "mfe_bias", "mae_bias", "er_bias",
            "mfe_corr", "mae_corr", "er_corr",
            "signal_hit", "win_rate",
            "mean_real_ret", "mean_real_trade", "mean_real_win", "mean_real_loss",
            "flat_pct", "long_pct", "short_pct", "ret_corr",
            "brier", "ece", "cons_spearman"]
    print("-" * 40)
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if k == "n" or k == "mean_short_len":
            print(f"  {k:<18} {v:.0f}")
        elif k.endswith("_pct"):
            print(f"  {k:<18} {v*100:>5.1f}%")
        elif k.startswith("mean_real") or k.endswith("_corr") or k.endswith("_bias") \
                or k in ("ret_corr", "cons_spearman"):
            print(f"  {k:<18} {v:+.4f}")
        else:
            print(f"  {k:<18} {v:.4f}")
    print("-" * 40)
    if extras:
        _print_cal(extras)


def _print_cal(extras: dict):
    """校准分桶明细"""
    up = extras.get("up_prob_cal")
    if up and up["bins"]:
        print(f"up_prob 校准 (Brier={up['brier']:.4f}, ECE={up['ece']:.4f}):")
        print(f"  {'bin':<11}{'n':>6}{'mean_pred':>10}{'realized':>10}{'gap':>8}")
        for t in up["bins"]:
            print(f"  {t['bin']:<11}{t['n']:>6}{t['mean_pred']:>10.2f}"
                  f"{t['realized_freq']:>10.2f}{t['gap']:>+8.2f}")
    cons = extras.get("consensus_cal")
    if cons and cons["bins"]:
        sig = "显著" if cons["significant"] else "不显著"
        print(f"consensus 单调性 (Spearman ρ={cons['spearman_rho']:+.3f}, {sig}, "
              f"样本={cons['scope']}):")
        print(f"  {'range':<15}{'n':>6}{'hit_rate':>10}")
        for t in cons["bins"]:
            print(f"  {t['consensus_range']:<15}{t['n']:>6}{t['hit_rate']:>10.2f}")
    print("-" * 40)


def _print_compare(df: pd.DataFrame):
    """双模型并排打印，突出显著差异"""
    metrics = ["dir_acc", "dir_auc",
               "close_mae", "close_rmse", "bar_mae", "bar_rmse",
               "mfe_bias", "mae_bias", "er_bias",
               "mfe_corr", "mae_corr", "er_corr",
               "signal_hit", "win_rate",
               "mean_real_ret", "mean_real_trade", "ret_corr",
               "brier", "ece", "cons_spearman"]
    lower_better_set = {"close_mae", "close_rmse", "bar_mae", "bar_rmse",
                        "brier", "ece"}
    print(f"\n{'metric':<16} {'base':>10} {'finetuned':>10} {'Δ(ft-base)':>12} {'better':>8}")
    print("-" * 60)
    base = df[df["tag"] == "base"].iloc[0]
    new = df[df["tag"] == "finetuned"].iloc[0]
    for m in metrics:
        if m not in base.index or m not in new.index:
            continue
        b, n = float(base[m]), float(new[m])
        d = n - b
        if m.endswith("_bias"):
            # bias 越接近 0 越好
            better = "finetuned" if abs(n) < abs(b) else "base"
        else:
            lower_better = m in lower_better_set
            better = "finetuned" if (d > 0 and not lower_better) \
                or (d < 0 and lower_better) else "base"
        print(f"  {m:<14} {b:>10.4f} {n:>10.4f} {d:>+12.4f} {better:>8}")
    print("-" * 60)


if __name__ == "__main__":
    main()
