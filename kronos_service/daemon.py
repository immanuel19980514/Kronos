# -*- coding: utf-8 -*-
"""
Kronos 预测守护进程。

每根 K 线收盘后延迟 CLOSE_DELAY_SEC 秒：
    读取 K 线 -> 长/短期两段预测 -> 预测结果写入 sqlite 三张表

用法：
    python daemon.py -c config/config.json   # 默认配置
    python daemon.py -c config/config_btc_30m.json  # 自定义配置文件
    python daemon.py --once                  # 只跑一次（调试用）
    python daemon.py --source feather
    python daemon.py --timeframe 30m
    python daemon.py --reset                 # 忽略已存在的记录，强制重算
"""
import argparse
import logging
import os
import sys
import time
import traceback

# ---- -c 提前扫描：在 import config 之前设置环境变量 ----
for i, a in enumerate(sys.argv):
    if a in ("-c", "--config") and i + 1 < len(sys.argv):
        os.environ["KRONOS_CONFIG"] = sys.argv[i + 1]
        break

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import config as C  # noqa: E402 — 模块 import 时自动执行 load()

import datasource  # noqa: E402
import store  # noqa: E402
import visualize  # noqa: E402
from bridge import KronosBridge  # noqa: E402


# ------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    os.makedirs(C.LOG_DIR, exist_ok=True)
    path = os.path.join(C.LOG_DIR, f"daemon_{C.TIMEFRAME}.log")
    logger = logging.getLogger("kronos")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"日志文件: {path}")
    return logger


# ------------------------------------------------------------------
# 时间对齐
# ------------------------------------------------------------------
def _fmt(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) + "Z"


def next_run_at(now: float, tf_sec: int, delay: int) -> int:
    """下一个「收盘 + delay」的时刻"""
    return (int(now // tf_sec) + 1) * tf_sec + delay


def wait_until(target: float, stop_check=None) -> bool:
    while True:
        remain = target - time.time()
        if remain <= 0:
            return True
        if stop_check and stop_check():
            return False
        time.sleep(min(remain, 1.0))


def _many_figure_window(origin: int):
    """
    判断本轮是否到达「输出总图」的时间点（本地时间 FIGURE_MANY_PERIOD 整点边界）。

    触发条件：本轮处理的 K 线的下一根开盘时刻恰好落在输出边界上
    （period=4 时每天的 0/4/8/12/16/20 点），每个边界只触发一次。
    用 origin 而不是当前墙钟判断，预测耗时不影响触发，也不会同边界重复输出。

    :return: (start, end) unix 秒 = [上一个输出点, 当前输出点)；未触发返回 (None, None)
    """
    period = C.FIGURE_MANY_PERIOD
    tf_sec = C.tf_seconds()

    step = period * 3600
    next_open = origin + tf_sec
    lt = time.localtime(next_open)
    sec_of_day = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    if sec_of_day % step != 0:
        return None, None
    return next_open - step, next_open


# ------------------------------------------------------------------
# 单产品取数
# ------------------------------------------------------------------
def _load_pair(pair, logger, expected_origin, source, force):
    """
    取单个产品的 K 线（含重试 + 新鲜度校验）。

    :return: (df, meta, reason)；成功时 reason=None
    """
    df, meta = None, None
    for attempt in range(1, C.POLL_RETRY + 1):
        try:
            df, meta = datasource.load_klines(
                limit=C.LOOKBACK + C.LONG_LEN + 10, source=source, pair=pair)  #limit=C.LOOKBACK + C.LONG_LEN + 10，其实LOOKBACK就够，+LONG_LEN为了回测比照，+10为了余量保险
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[{pair}] 取数失败({attempt}/{C.POLL_RETRY}): {e}")
            df = None

        if df is not None:
            last_open = int(df["timestamps"].iloc[-1].value // 10 ** 9)
            if last_open >= expected_origin:
                logger.info(f"[{pair}] k线最新时间 {_fmt(last_open)} >= "
                            f"期望时间 {_fmt(expected_origin)}，可以处理")
                break
            logger.warning(
                f"[{pair}] 数据末端 {_fmt(last_open)} < 期望 {_fmt(expected_origin)}，"
                f"{C.POLL_RETRY_INTERVAL}s 后重试 ({attempt}/{C.POLL_RETRY})")
        if attempt < C.POLL_RETRY:
            time.sleep(C.POLL_RETRY_INTERVAL)

    if df is None:
        return None, None, "no_data"

    origin = int(df["timestamps"].iloc[-1].value // 10 ** 9)
    if origin < expected_origin:
        if not force:
            logger.error(f"[{pair}] 数据未更新到目标 K 线（末端 {_fmt(origin)} < "
                         f"{_fmt(expected_origin)}），跳过该产品。"
                         f"若 freqtrade 未运行/数据为历史快照，可加 --force")
            return None, meta, "stale_data"
        logger.warning(f"[{pair}] --force：跳过新鲜度检查（数据末端 {_fmt(origin)}）")

    if meta and meta.get("warn"):
        logger.warning(f"[{pair}] {meta['warn']}")

    logger.info(f"[{pair}] 数据源={meta['source']} 行数={meta['rows']} "
                f"末端={_fmt(origin)} 滞后={meta['stale_ratio']}周期")
    return df, meta, None


# ------------------------------------------------------------------
# 单轮
# ------------------------------------------------------------------
def run_once(bridge: KronosBridge, logger, source: str = None,
             force: bool = False) -> dict:
    """
    执行一次「取数(全部产品) -> 预测(批量或逐个) -> 写库」，返回统计信息。

    产品范围 = PAIR + INFORMATIVE_PAIRS；不同产品历史长度可能不同，
    由 bridge.predict_many 按长度分批，不做截断。
    """
    tf_sec = C.tf_seconds()
    expected_origin = (int(time.time() // tf_sec) - 1) * tf_sec
    pairs = C.all_pairs()

    # ---------------- 1. 取数 ----------------
    t_load0 = time.perf_counter()
    items = []                        # [{pair, df, series, info}]
    for pair in pairs:
        df, _meta, reason = _load_pair(pair, logger, expected_origin, source, force)
        if df is None:
            if pair == C.PAIR:
                logger.error(f"主产品 {pair} 取数失败（{reason}），本轮跳过")
                return {"ok": False, "reason": reason}
            logger.warning(f"辅助产品 {pair} 跳过（{reason}）")
            continue
        try:
            series, info = bridge.build_series(df)
        except ValueError as e:
            if pair == C.PAIR:
                logger.error(f"主产品 {pair} 无法预测: {e}")
                return {"ok": False, "reason": "too_few_bars"}
            logger.warning(f"辅助产品 {pair} 跳过：{e}")
            continue
        items.append({"pair": pair, "df": df, "series": series, "info": info})
    t_load = time.perf_counter() - t_load0

    if not items:
        return {"ok": False, "reason": "no_data"}

    # ---------------- 2. 预测 ----------------
    t_infer0 = time.perf_counter()
    if C.BATCH_PREDICT and len(items) > 1:
        mode = f"批量({len(items)}产品)"
        long_groups, short_groups = bridge.predict_many(
            [it["series"] for it in items])
        for k, it in enumerate(items):
            it["long"], it["short"] = long_groups[k], short_groups[k]
        tim = bridge.last_timings
        logger.info(
            f"⏱ 推理[{mode}] 合计 {tim['total_sec']:.1f}s | 窗口={tim['windows']} | "
            f"长期 {tim['long_sec']:.1f}s 批次={tim['long_batches']} | "
            f"短期 {tim['short_sec']:.1f}s 批次={tim['short_batches']}")
    else:
        mode = "逐个"
        per_sec = {}
        for it in items:
            t0 = time.perf_counter()
            long_groups, short_groups = bridge.predict_many([it["series"]])
            it["long"], it["short"] = long_groups[0], short_groups[0]
            per_sec[it["pair"]] = time.perf_counter() - t0
            logger.info(f"⏱ [{it['pair']}] 推理 {per_sec[it['pair']]:.1f}s")
        # 串行合计：后续换成 BATCH_PREDICT=true 可直接对比
        logger.info(f"⏱ 推理[逐个] 合计 {sum(per_sec.values()):.1f}s | "
                    f"明细={ {p: round(v, 1) for p, v in per_sec.items()} }")
    t_infer = time.perf_counter() - t_infer0

    # ---------------- 3. 组装 + 写库 ----------------
    t_save0 = time.perf_counter()
    saved = []
    for it in items:
        info = it["info"]
        res = bridge.build_result(
            it["long"], it["short"], info["ref_close"], info["origin_ts"],
            info["origin_dt"], lookback=info["lookback"], pair=it["pair"])
        main = res["main"]
        if main["origin_ts"] != info["origin_ts"]:
            logger.warning(f"[{it['pair']}] origin 不一致 "
                           f"{main['origin_ts']} != {info['origin_ts']}")
        store.save_prediction(main, res["long_rows"], res["short_rows"])

        res["long_paths"] = it["long"]
        res["short_paths"] = it["short"]
        it["res"] = res
        saved.append(it)

        logger.info(
            f"✅ [{it['pair']}] 写入 origin={_fmt(info['origin_ts'])} | "
            f"ref={main['ref_close']:.2f} | "
            f"S(ret={main['s_ret']:+.4f}, er={main['s_er']:.2f}, "
            f"regime={main['s_regime']}) "
            f"->signal={main['signal']} confidence={main['confidence']:.2f}")
    t_save = time.perf_counter() - t_save0

    # ---------------- 4. 绘图（仅主产品，避免多产品图量翻倍）----------------
    primary = next((it for it in saved if it["pair"] == C.PAIR), saved[0])
    main = primary["res"]["main"]
    origin = primary["info"]["origin_ts"]
    t_fig = 0.0
    if C.FIGURE:
        try:
            t0 = time.perf_counter()
            fig_path = visualize.save_figure(primary["df"],
                                             primary["res"]["long_paths"],
                                             primary["res"]["short_paths"], main)
            logger.info(f"图已保存 -> {fig_path}")

            # 每天定时输出总图：仅在 FIGURE_MANY_PERIOD 整点边界触发，
            # 区间为 [上一个输出时间点, 当前输出时间点)
            m_start, m_end = _many_figure_window(origin)
            if m_start is not None:
                long_fig = visualize.many_figure_long(primary["df"], m_start, m_end)
                short_fig = visualize.many_figure_short(primary["df"], m_start, m_end)
                if long_fig and short_fig:
                    logger.info(f"总图已保存 -> {long_fig} | {short_fig}")
            t_fig = time.perf_counter() - t0
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"绘图失败（不影响预测流程）: {e}")

    logger.info(f"⏱ 本轮耗时 取数 {t_load:.1f}s | 推理 {t_infer:.1f}s（{mode}） | "
                f"写库 {t_save:.1f}s | 绘图 {t_fig:.1f}s | "
                f"产品={[it['pair'] for it in saved]}")

    return {"ok": True, "origin": origin, "main": main,
            "pairs": [it["pair"] for it in saved]}


# ------------------------------------------------------------------
# 主循环
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Kronos 预测守护进程")
    ap.add_argument("-c", "--config", default=None, help="配置文件路径（默认 config/config.json）")
    ap.add_argument("--timeframe", default=C.TIMEFRAME)
    ap.add_argument("--source", default=None, choices=[None, "api", "feather"], help="数据源,默认先api后feather")
    ap.add_argument("--once", action="store_true", help="只跑一轮")
    ap.add_argument("--reset", action="store_true", help="忽略已存在记录，强制重算")
    ap.add_argument("--verbose", action="store_true", default=C.VERBOSE, help="详细输出")
    ap.add_argument("--force", action="store_true", help="跳过 K 线新鲜度检查（freqtrade 未运行时测试用）")
    args = ap.parse_args()

    C.TIMEFRAME = args.timeframe
    # --verbose 开关统一写入 config，底层模块(store/datasource/bridge)
    # 全部直接读 C.VERBOSE，不再逐方法传参
    C.VERBOSE = args.verbose

    #窗口长度不超过模型上下文上限
    C.assert_context()

    logger = setup_logger()
    delay = C.close_delay()          # 按周期错峰（CLOSE_DELAY_BY_TF）

    logger.info("=" * 68)
    logger.info(f"Kronos 守护进程启动 | 主产品={C.PAIR} 辅助={list(C.INFORMATIVE_PAIRS)} "
                f"timeframe={C.TIMEFRAME} "
                f"回看K线长度={C.LOOKBACK} 长期预测K线长度={C.LONG_LEN} 短期预测K线长度（默认，默认X5）={C.SHORT_LEN} "
                f"预测次数={C.SAMPLE_COUNT} 预测设备CPU/GPU={C.DEVICE} "
                f"是否短期直接截取长期预测={C.SINGLE_ROLLOUT}")
    logger.info(f"错峰延迟={delay}s（CLOSE_DELAY_BY_TF={C.CLOSE_DELAY_BY_TF}）| "
                f"批量推理={C.BATCH_PREDICT}")

    logger.info(f"数据源优先级={list(C.LIVE_SOURCE_PRIORITY) if not args.source else [args.source]}")
    logger.info(f"数据库={C.DB_PATH}")

    store.ensure_schema()
    bridge = KronosBridge()
    logger.info(f"模型加载完成，threads={C.CPU_THREADS}")

    tf_sec = C.tf_seconds()
    last_origin = None if args.reset else store.latest_origin()
    logger.info(f"数据库中最新 origin = "
                f"{_fmt(last_origin) if last_origin else '无'}")

    #1.单次预测模式
    if args.once:
        t0 = time.perf_counter()
        stat = run_once(bridge, logger, args.source, force=args.force)
        logger.info(f"单次预测结束: {stat.get('ok')} {stat.get('reason', '')} 耗时 {time.perf_counter() - t0:.2f}s")
        return

    #2.循环预测模式
    while True:
        #循环时间的控制，先计算下一个执行时间点，然后等待下一个时间点
        # delay 按周期错峰：5m 先跑，30m/4h 依次延后，避免同一整点抢 CPU
        target = next_run_at(time.time(), tf_sec, delay)
        logger.info(f"下次执行: {_fmt(target)} （等待 {target - time.time():.0f}s）")
        wait_until(target)

        try:
            #同样地，run_once执行一次预测流程
            t0 = time.perf_counter()
            stat = run_once(bridge, logger, args.source, force=args.force)
            logger.info(f"单次预测结束: {stat.get('ok')} {stat.get('reason', '')} 耗时 {time.perf_counter() - t0:.2f}s")
            if stat.get("ok"):
                if last_origin is not None and stat["origin"] <= last_origin:
                    logger.info(f"origin {_fmt(stat['origin'])} 已存在，跳过")
                last_origin = stat["origin"]
        except Exception:                                    # noqa: BLE001
            logger.error("本轮异常，继续下一轮:\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
