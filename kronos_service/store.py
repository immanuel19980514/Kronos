# -*- coding: utf-8 -*-
"""
SQLite 读写层。

设计要点（与 freqtrade 共享 advanced2.sqlite）：
  * **短连接**：每次写入 = connect -> insert -> commit -> close，
    绝不在进程内长期持有连接，避免与 freqtrade 长时间抢锁。
  * **WAL + busy_timeout**：读不阻塞写，写冲突时自动等待重试。
  * **幂等**：以 (pair, timeframe, origin_ts) / (..., step) 唯一键 INSERT OR REPLACE，
    重复运行不会产生重复行。
"""
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

# ------------------------------------------------------------------
# 表结构定义
# ------------------------------------------------------------------
MAIN_COLS = [
    "pair", "timeframe", "origin_ts", "origin_dt", "created_at",
    "ref_close", "lookback", "sample_count", "long_len", "short_len",

    # 短期(12根)：入场时机
    "s_ret", "s_mfe", "s_mae", "s_close",
    "s_er", "s_up_prob", "s_path_std", "s_consensus",
    # 逐根横截面分歧（多路径中间过程的发散程度）
    # "s_band_width", "s_band_slope", "s_shape_corr",
    # "s_mfe_std", "s_mae_std", "s_er_std",
    "s_trend", "s_regime",
    # 综合信号
    "signal", "confidence", "min_rr_ok",
]

LIST_COLS = [
    "pair", "timeframe", "origin_ts", "step", "ts", "ts_dt",
    "open", "high", "low", "close", "volume", "amount",
]

    # s_band_width REAL,              -- 平均带宽 = 各路径逐根std均值/基准(整体不确定度)
    # s_band_slope REAL,              -- 带宽随时间增速(归一化)，越大=越远的K线越不可信
    # s_shape_corr REAL,              -- 各路径逐根收益的两两平均相关系数(形态一致性)
    # s_mfe_std    REAL,              -- 各路径 MFE 的标准差(止盈位不确定性)
    # s_mae_std    REAL,              -- 各路径 MAE 的标准差(止损位不确定性)
    # s_er_std     REAL,              -- 各路径 ER 的标准差(形态判断本身的确定性)
    
_SQL_MAIN = f"""
CREATE TABLE IF NOT EXISTS {C.TABLE_MAIN} (
    pair        TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    origin_ts   INTEGER NOT NULL,   -- 预测发起K线的开盘时间(unix秒, UTC)
    origin_dt   TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    ref_close   REAL    NOT NULL,   -- 基准价 = 窗口最后一根实际收盘价
    lookback    INTEGER,
    sample_count INTEGER,
    long_len    INTEGER,
    short_len   INTEGER,

    s_ret       REAL,               -- 短期预期收益率 = 预测期末/基准 - 1
    s_mfe       REAL,               -- 短期最大有利波动(Max Favorable Excursion)
    s_mae       REAL,               -- 短期最大不利波动(Max Adverse Excursion)
    s_close     REAL,               -- 短期预测期末收盘价
    s_er        REAL,               -- 效率比 = |净位移|/路径总行程, 越接近1越单边
    s_up_prob   REAL,               -- 跨路径上涨比例
    s_path_std  REAL,               -- 跨路径收益标准差
    s_consensus REAL,               -- 共识度 = |mean|/(std+eps)

    s_trend     TEXT,               -- up / down / flat
    s_regime    TEXT,               -- trend / chop / mixed


    signal      TEXT,               -- long / short / flat
    confidence  REAL,
    min_rr_ok   INTEGER,

    PRIMARY KEY (pair, timeframe, origin_ts)
);
"""

# TABLE_LONG or TABLE_SHORT
_SQL_LIST = """
CREATE TABLE IF NOT EXISTS {table} (
    pair       TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    origin_ts  INTEGER NOT NULL,
    step       INTEGER NOT NULL,   -- 第几根预测K线(从1开始)
    ts         INTEGER NOT NULL,   -- 该预测K线的开盘时间(unix秒, UTC)
    ts_dt      TEXT    NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     REAL,
    amount     REAL,
    PRIMARY KEY (pair, timeframe, origin_ts, step)
);
"""

_SQL_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{C.TABLE_MAIN}_q "
    f"ON {C.TABLE_MAIN}(pair, timeframe, origin_ts DESC);",
    f"CREATE INDEX IF NOT EXISTS idx_{C.TABLE_LONG}_q "
    f"ON {C.TABLE_LONG}(pair, timeframe, origin_ts DESC);",
    f"CREATE INDEX IF NOT EXISTS idx_{C.TABLE_SHORT}_q "
    f"ON {C.TABLE_SHORT}(pair, timeframe, origin_ts DESC);",
]


def connect() -> sqlite3.Connection:
    """
    建立短连接（
        busy_timeout 必设；
        WAL 尽量设）。

    WAL 可提升与 freqtrade 的并发容忍度；但若库文件正被其他程序
    占用（DB 工具 / webui / 残留会话），设置 WAL 可能报 disk I/O error。
    此时降级为默认 journal 模式继续写，不中断预测流程。
    """
    os.makedirs(os.path.dirname(C.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(C.DB_PATH, timeout=C.DB_BUSY_TIMEOUT_MS / 1000.0)
    conn.execute(f"PRAGMA busy_timeout={C.DB_BUSY_TIMEOUT_MS};")
    try:
        mode = str(conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]).lower()
        if mode == "wal":
            conn.execute("PRAGMA synchronous=NORMAL;")
        else:
            print(f"[store] ⚠ journal_mode={mode}（WAL 未生效，可能有其他连接占用）")
    except sqlite3.OperationalError as e:
        print(f"[store] ⚠ WAL 不可用（{e}），使用默认 journal 模式")
    return conn


def _sql_col_defs(create_sql: str) -> dict:
    """
    从 CREATE TABLE 语句中解析出「列名 -> 列定义」，用于自动补列。
    仅取顶层列定义，跳过 PRIMARY/UNIQUE/FOREIGN/CHECK/CONSTRAINT 等表级约束。
    """
    body = create_sql[create_sql.index("(") + 1: create_sql.rindex(")")]
    defs = {}
    for line in body.splitlines():
        line = line.split("--")[0].strip().rstrip(",")
        if not line:
            continue
        name = line.split()[0]
        if name.upper() in ("PRIMARY", "UNIQUE", "FOREIGN", "CHECK", "CONSTRAINT"):
            continue
        defs[name] = line
    return defs


def ensure_schema() -> None:
    """
    建表 + 自动补列（均幂等）。

    已有库若缺列（表结构升级），用 ALTER TABLE ADD COLUMN 补齐，
    不影响既有数据；SQLite 的 ADD COLUMN 对新增列一律填 NULL。
    日志输出由 config.VERBOSE 统一控制。
    """
    conn = connect()
    try:
        conn.execute(_SQL_MAIN)
        conn.execute(_SQL_LIST.format(table=C.TABLE_LONG))
        conn.execute(_SQL_LIST.format(table=C.TABLE_SHORT))
        for idx in _SQL_INDEXES:
            conn.execute(idx)
        conn.commit()

        # 补列：对比 CREATE TABLE 定义与库中实际列
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({C.TABLE_MAIN})")}
        added = []
        for col, definition in _sql_col_defs(_SQL_MAIN).items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {C.TABLE_MAIN} ADD COLUMN {definition}")
                added.append(col)
        if added:
            conn.commit()
            if C.VERBOSE:
                print(f"[store] 补列 {len(added)} 个 -> {', '.join(added)}")

        if C.VERBOSE:
            print(f"[store] schema ok -> {C.DB_PATH} "
                  f"({C.TABLE_MAIN}, {C.TABLE_LONG}, {C.TABLE_SHORT})")
    finally:
        conn.close()


def _insert_many(conn, table: str, cols: list, rows: list) -> int:
    if not rows:
        return 0
    placeholders = ",".join("?" * len(cols))
    sql = (f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
           f"VALUES ({placeholders})")
    conn.executemany(sql, rows)
    return len(rows)


def save_prediction(main: dict, long_rows: list, short_rows: list) -> dict:
    """
    一次性写入 A/B/C 三张表，单个事务，写完 commit 并关闭连接。
    日志输出由 config.VERBOSE 统一控制。

    :return: 写入统计
    """
    main_row = [main.get(c) for c in MAIN_COLS]
    stat = {}
    conn = connect()
    try:
        stat["main"] = _insert_many(conn, C.TABLE_MAIN, MAIN_COLS, [main_row])
        stat["long"] = _insert_many(conn, C.TABLE_LONG, LIST_COLS, long_rows)
        stat["short"] = _insert_many(conn, C.TABLE_SHORT, LIST_COLS, short_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    if C.VERBOSE:
        print(f"[store] 写入 预测K线时间={main.get('origin_ts')} "
              f"A表数量={stat['main']} B表数量={stat['long']} C表数量={stat['short']}")
    return stat


# ------------------------------------------------------------------
# 读取（供校验 / freqtrade 策略参考）
# ------------------------------------------------------------------
def latest_origin():
    """返回最新一条 A 表记录的时间戳，用于跳过重复预测"""
    conn = connect()
    try:
        cur = conn.execute(
            f"SELECT origin_ts FROM {C.TABLE_MAIN} WHERE pair=? AND timeframe=? "
            f"ORDER BY origin_ts DESC LIMIT 1", (C.PAIR, C.TIMEFRAME))
        row = cur.fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def read_result(origin_ts: int):
    """读取指定 origin 的 A 表一行（dict）"""
    conn = connect()
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"SELECT * FROM {C.TABLE_MAIN} WHERE pair=? AND timeframe=? AND origin_ts=?",
            (C.PAIR, C.TIMEFRAME, origin_ts))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def load_list_range(table: str, start: int = None, end: int = None) -> list:
    """
    读取明细表（B/C 表）origin_ts 落在 [start, end) 区间内的所有预测行，
    按 (origin_ts, step) 排序返回 dict 列表。start/end 为 None 表示不限制。
    """
    pair = C.PAIR
    timeframe = C.TIMEFRAME
    
    sql = (f"SELECT origin_ts, step, ts, open, high, low, close, volume, amount "
           f"FROM {table} WHERE pair=? AND timeframe=?")
    args = [pair, timeframe]
    if start is not None:
        sql += " AND origin_ts >= ?"
        args.append(int(start))
    if end is not None:
        sql += " AND origin_ts < ?"
        args.append(int(end))
    sql += " ORDER BY origin_ts, step"
    conn = connect()
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def table_counts() -> dict:
    conn = connect()
    try:
        out = {}
        for t in (C.TABLE_MAIN, C.TABLE_LONG, C.TABLE_SHORT):
            out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return out
    finally:
        conn.close()


if __name__ == "__main__":
    ensure_schema()
    print(table_counts())
    print("latest origin:", latest_origin())
