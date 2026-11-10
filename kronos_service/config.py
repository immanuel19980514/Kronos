# -*- coding: utf-8 -*-
"""
Kronos 预测服务 —— 集中配置（从 config/config.json 加载）

所有可配置项集中在 JSON 文件；本模块负责：
  1. 加载 JSON 中的扁平键值对为模块级全局变量（供 C.X 访问）
  2. 计算派生值（DB_PATH, LOG_DIR, MODEL_DIR_CANDIDATES 等）
  3. 提供工具函数 tf_seconds / assert_context / feather_paths

切换周期 / 交易标的 / 守护进程参数只需修改 JSON；模型权重路径的优先级列表写死在这里
（涉及 HF 缓存路径展开，不适合 JSON 配置）。
"""
import json
import os
import re
import sys

# ============================================================
# 1. 加载 JSON 配置
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_HERE, "config", "config.json")


def _resolve(p):
    """相对路径先尝试 kronos_service 目录下，再保持原样让系统按 CWD 解析"""
    if os.path.isabs(p):
        return p
    in_here = os.path.join(_HERE, p)
    return in_here if os.path.isfile(in_here) else p


def _find_config_path():
    """查找配置文件：KRONOS_CONFIG 环境变量 > 默认路径（均支持相对 kronos_service）"""
    env = os.environ.get("KRONOS_CONFIG")
    if env:
        env = _resolve(env)
        if os.path.isfile(env):
            return env
    if os.path.isfile(_DEFAULT_CONFIG):
        return _DEFAULT_CONFIG
    print(f"[config] ❌ 找不到配置文件（尝试过：KRONOS_CONFIG={env}, default={_DEFAULT_CONFIG}）",
          file=sys.stderr)
    sys.exit(1)


def _strip_comments(text: str) -> str:
    """
    去除 JSON 中的 // 单行 和 /* */ 块注释，保留字符串内的 // / /* 不动。
    正则思路：优先匹配字符串（捕获组 1/2 完整保留），其次匹配注释（捕获组 3 丢弃）。
    """
    pattern = re.compile(
        r'"(?:\\.|[^"\\])*"'        # 双引号字符串
        r"|'(?:\\.|[^'\\])*'"       # 单引号字符串（JSON 不标准但容错）
        r"|/\*.*?\*/"               # 块注释
        r"|//[^\n]*"                # 单行注释
        , re.DOTALL
    )
    out = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            out.append(text[last:m.start()])
        # 字符串（首字符为 " 或 '）原样保留；注释丢弃
        ch = text[m.start()]
        if ch in ('"', "'"):
            out.append(m.group())
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def load(path=None):
    """加载 JSON 配置并注入全局命名空间。daemon.py -c 参数会间接调用本函数（通过环境变量）。"""
    if path is None:
        path = _find_config_path()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(_strip_comments(raw))

    g = globals()
    for key, val in data.items():
        g[key] = val
    g["CONFIG_PATH"] = path
    _post_load()


def _post_load():
    """JSON 加载后，计算所有派生值和不可序列化的硬编码列表"""
    g = globals()

    # 多产品 / 批量 / 错峰 的兜底默认值（旧配置文件缺少这些键也能跑）
    g.setdefault("INFORMATIVE_PAIRS", [])
    g.setdefault("CLOSE_DELAY_BY_TF", {})
    g.setdefault("BATCH_PREDICT", True)

    # 派生路径
    g["FREQTRADE_USER_DATA"] = os.path.join(g["FREQTRADE_ROOT"], "user_data")
    g["DB_PATH"] = os.path.join(g["KRONOS_ROOT"], "kronos_service", "db", "kronos.sqlite")
    g["LOG_DIR"] = os.path.join(g["KRONOS_ROOT"], "kronos_service", "logs")
    g["FIGURE_DIR"] = os.path.join(g["KRONOS_ROOT"], "kronos_service", "figures")

    # 模型权重候选（硬编码展开 HF 缓存路径）
    g["MODEL_DIR_CANDIDATES"] = [
        os.path.join(g["KRONOS_ROOT"], "models", "Kronos-small"),
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--NeoQuasar--Kronos-small"
            r"\snapshots\901c26c1332695a2a8f243eb2f37243a37bea320"
        ),
        os.environ.get("KRONOS_MODEL_DIR", ""),
    ]
    g["TOKENIZER_DIR_CANDIDATES"] = [
        os.path.join(g["KRONOS_ROOT"], "models", "Kronos-Tokenizer-base"),
        os.environ.get("KRONOS_TOKENIZER_DIR", ""),
    ]

    # 周期秒数表（不适合 JSON，硬编码）
    g["TIMEFRAME_SECONDS"] = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
        "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
    }


# ============================================================
# 2. 工具函数（不受 JSON 影响，保持不变）
# ============================================================
def tf_seconds(tf: str = None) -> int:
    """周期 -> 秒"""
    tf = tf or TIMEFRAME
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"不支持的 timeframe: {tf}")
    return TIMEFRAME_SECONDS[tf]


def assert_context():
    """校验窗口长度不超过模型上下文"""
    total = LOOKBACK + LONG_LEN
    if total > MAX_CONTEXT:
        raise ValueError(
            f"LOOKBACK({LOOKBACK}) + LONG_LEN({LONG_LEN}) = {total} 超过 "
            f"MAX_CONTEXT({MAX_CONTEXT})，窗口长度超过模型上下文上限，会被滚动丢弃最早的历史"
        )


def all_pairs() -> list:
    """本轮要预测的产品：主产品 + INFORMATIVE_PAIRS（去重、保持顺序）"""
    out = []
    for p in [PAIR] + list(INFORMATIVE_PAIRS):
        if p and p not in out:
            out.append(p)
    return out


def close_delay(tf: str = None) -> int:
    """
    按周期错峰：同一整点 5m/30m/4h 的实例都会触发，用不同延迟让它们串行，
    避免同时争抢 CPU（未配置的周期回落到 CLOSE_DELAY_SEC）。
    """
    tf = tf or TIMEFRAME
    return int(CLOSE_DELAY_BY_TF.get(tf, CLOSE_DELAY_SEC))


def feather_paths(pair: str = None) -> list:
    """
    按 freqtrade 命名规则推导可能的 feather 路径（按优先级）。

    freqtrade 规则：
      futures: <datadir>/<exchange>/futures/<PAIR>_<BASE>_<QUOTE>-<tf>-futures.feather
      spot   : <datadir>/<exchange>/<BASE>_<QUOTE>-<tf>.feather

    :param pair: 指定产品，默认 C.PAIR（多产品实例用）
    """
    pair = pair or PAIR
    tf = TIMEFRAME
    datadir = os.path.join(FREQTRADE_USER_DATA, "data", EXCHANGE)
    base, quote = pair.split("/")[0], pair.split("/")[1].split(":")[0]
    fut = os.path.join(datadir, "futures", f"{base}_{quote}_{quote}-{tf}-futures.feather")
    spot = os.path.join(datadir, f"{base}_{quote}-{tf}.feather")
    return [fut, spot] if TRADING_MODE == "futures" else [spot, fut]


# ============================================================
# 3. 模块加载时自动执行
# ============================================================
load()
