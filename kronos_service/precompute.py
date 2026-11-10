# -*- coding: utf-8 -*-
"""
历史批量预计算 —— 为 freqtrade 回测生成逐根 K 线的预测记录。

每个 origin(i) 使用**截止第 i 根**的窗口预测，不含未来数据，因此回测无前视偏差。
结果与 daemon 写同一批表（同主键），回测时策略读取方式完全一致。

⚠ CPU 成本：单根 origin（长60 + 短12，3 路径）实测约 30s（8 线程 CPU），
   批量化可小幅摊薄但仍是「小时级」。请用 --limit / --step 控制规模，
   先 --dry-run 看 ETA 再决定。快速试跑可用 --sample-count 1。

用法：
    python precompute.py --dry-run                    # 只看计划与预估
    python precompute.py --limit 100 --batch 8
    python precompute.py --limit 300 --step 2 --sample-count 1   # 快速模式
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import config as C  # noqa: E402

import pandas as pd  # noqa: E402

import datasource  # noqa: E402
import store  # noqa: E402
from bridge import KronosBridge  # noqa: E402


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def build_origins(df, limit, step, start, end):
    """返回可作为 origin 的索引列表（需满足 idx >= LOOKBACK-1）"""
    n = len(df)
    lo = C.LOOKBACK - 1
    if start:
        lo = max(lo, int((df["timestamps"] < pd.Timestamp(start)).sum()))
    hi = n - 1
    if end:
        hi = min(hi, int((df["timestamps"] <= pd.Timestamp(end)).sum()) - 1)
    idxs = list(range(lo, hi + 1, step))
    if limit and len(idxs) > limit:
        idxs = idxs[-limit:]
    return idxs


def main():
    ap = argparse.ArgumentParser(description="Kronos 历史批量预计算")
    ap.add_argument("--timeframe", default=C.TIMEFRAME)
    ap.add_argument("--limit", type=int, default=200, help="最多计算多少根 origin")
    ap.add_argument("--step", type=int, default=1, help="每隔几根算一次")
    ap.add_argument("--batch", type=int, default=8, help="每批并行多少个窗口")
    ap.add_argument("--sample-count", type=int, default=C.SAMPLE_COUNT)
    ap.add_argument("--start", default=None, help="起始时间 ISO，如 2026-07-20")
    ap.add_argument("--end", default=None)
    ap.add_argument("--feather-path", default=None, help="显式指定 feather 文件")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    C.TIMEFRAME = args.timeframe
    C.SAMPLE_COUNT = args.sample_count
    # 批量预计算有自己的进度输出（[precompute] ...），底层模块统一静默，
    # 避免每个 origin 都刷一条 [store]/[bridge] 日志
    C.VERBOSE = False
    C.assert_context()

    paths = [args.feather_path] if args.feather_path else None
    df, path = datasource.fetch_from_feather(paths)
    print(f"[precompute] 数据 <- {path}")
    print(f"[precompute] {len(df)} 根  {_fmt(df['timestamps'].iloc[0])} ~ "
          f"{_fmt(df['timestamps'].iloc[-1])}")

    origins = build_origins(df, args.limit, args.step, args.start, args.end)
    if not origins:
        print("[precompute] 无可用 origin（历史不足 LOOKBACK）")
        return
    print(f"[precompute] 计划 {len(origins)} 个 origin: "
          f"{_fmt(df['timestamps'].iloc[origins[0]])} ~ "
          f"{_fmt(df['timestamps'].iloc[origins[-1]])} | "
          f"timeframe={C.TIMEFRAME} paths={C.SAMPLE_COUNT} "
          f"long={C.LONG_LEN} short={C.SHORT_LEN}")

    # 粗略 ETA：以实测为基准（单 origin 长60+短12，3 路径 ≈ 30s
    # @5m / Kronos-small / 8 线程 CPU），批量摊薄系数约 batch^0.35（CPU 上收益有限）。
    # 仅作参考，循环中会依据首轮真实测速滚动修正。
    est_core = len(origins) * 30.0 * (C.SAMPLE_COUNT / 3.0)
    est = est_core / (args.batch ** 0.35)
    print(f"[precompute] 预估耗时 ≈ {est/3600:.2f} 小时 "
          f"（粗估，实际以首轮测速为准）")
    if args.dry_run:
        print("[precompute] --dry-run，退出")
        return

    store.ensure_schema()
    bridge = KronosBridge()
    n_ok = 0
    t_start = time.perf_counter()

    for bi in range(0, len(origins), args.batch):
        chunk = origins[bi:bi + args.batch]
        series = [bridge._window(df, i) for i in chunk]
        refs = [float(x["close"].iloc[-1]) for x, _ in series]
        ots = [int(ts.iloc[-1].value // 10 ** 9) for _, ts in series]
        odts = [str(ts.iloc[-1]) for _, ts in series]

        long_groups, short_groups = bridge.predict_many(series)

        for j in range(len(chunk)):
            res = bridge.build_result(long_groups[j], short_groups[j],
                                      refs[j], ots[j], odts[j])
            store.save_prediction(res["main"], res["long_rows"], res["short_rows"])
            n_ok += 1

        done = bi + len(chunk)
        speed = (time.perf_counter() - t_start) / done
        remain = (len(origins) - done) * speed
        print(f"[precompute] {done}/{len(origins)} | 平均 {speed:.1f}s/origin | "
              f"剩余 ≈ {remain/60:.1f} min | 最新 origin={odts[-1]}")

    print(f"[precompute] 完成 {n_ok} 条，用时 "
          f"{(time.perf_counter()-t_start)/60:.1f} min")
    print("[precompute] 表计数:", store.table_counts())


if __name__ == "__main__":
    main()
