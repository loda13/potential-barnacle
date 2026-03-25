#!/usr/bin/env python3
"""
股票K线技术分析工具
支持A股/美股，计算MA、RSI、MACD、布林带、KDJ、ADX、ATR、OBV、CCI、SuperTrend、PSAR、Ichimoku等技术指标
输出买卖信号、多指标共振分析和综合胜率提示
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional, Callable, Any

import pandas as pd
import numpy as np

# 长线基本面扫雷模块
try:
    from fundamentals import fetch_dilution_analysis, fetch_quality_metrics, check_delisting_risk, detect_value_trap
except ImportError:
    fetch_dilution_analysis = None
    fetch_quality_metrics = None
    check_delisting_risk = None
    detect_value_trap = None

# 聪明钱动向模块
try:
    from smart_money import fetch_insider_transactions, fetch_institutional_holdings, calc_smart_money_confirmation
except ImportError:
    fetch_insider_transactions = None
    fetch_institutional_holdings = None
    calc_smart_money_confirmation = None

# 行业相对强弱与财报波动模块
try:
    from sector_analysis import calc_sector_relative_strength, fetch_earnings_volatility, check_structural_bear_market
except ImportError:
    calc_sector_relative_strength = None
    fetch_earnings_volatility = None
    check_structural_bear_market = None


REQUEST_TIMEOUT = 10
CHART_FETCH_BUFFER_DAYS = 1200  # 长线分析：支持 3-5 年历史数据

# ============ 仓位管理参数 ============
MAX_RISK_PER_TRADE = 0.02      # 每笔交易最大风险比例 (1-2%)
DEFAULT_ACCOUNT_SIZE = 100000  # 默认账户规模（10万元）
CHANDELIER_ATR_MULTIPLIER = 2.5  # Chandelier Exit ATR倍数
CHANDELIER_LOOKBACK = 22      # Chandelier Exit 回溯天数
SLIPPAGE_PCT = 0.002          # 滑点 0.2%
ADV_POSITION_LIMIT = 0.01    # 仓位不超过 ADV 的 1%

# ============ 持仓管理参数 ============
PORTFOLIO_FILE = 'portfolio.json'
MAX_HOLDINGS = 5


def normalize_symbol(code: str) -> str:
    """将用户输入的股票代码转换为数据源可识别的标准代码。"""
    if code.isdigit() and len(code) == 6:
        if code.startswith(('6', '5', '9')):
            return f"{code}.SS"
        return f"{code}.SZ"
    return code.upper()


def call_with_suppressed_output(func: Callable[[], Any]) -> Any:
    """抑制第三方库向stdout/stderr输出的噪音。"""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        return func()


def get_history_window_days(period: str, days: int) -> int:
    """为技术指标预留更长的取数窗口。"""
    multiplier = 10 if period == 'w' else 2
    return max(days + 200, days * multiplier, CHART_FETCH_BUFFER_DAYS)


def parse_chart_response(data: dict, code: str, period: str, days: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """解析 Yahoo chart API 返回。"""
    chart = data.get('chart', {})
    error = chart.get('error')
    if error:
        description = error.get('description') or error.get('code') or '未知错误'
        return None, f"股票代码无效或无数据: {code} ({description})"

    results = chart.get('result') or []
    if not results:
        return None, f"股票代码无效或无数据: {code}"

    result = results[0]
    timestamps = result.get('timestamp')
    indicators = result.get('indicators', {})
    quotes = indicators.get('quote') or []
    quote = quotes[0] if quotes else None

    if not timestamps or not quote:
        return None, f"无有效行情数据: {code}"

    df = pd.DataFrame({
        'date': pd.to_datetime(timestamps, unit='s'),
        'open': quote.get('open'),
        'high': quote.get('high'),
        'low': quote.get('low'),
        'close': quote.get('close'),
        'volume': quote.get('volume')
    })

    df = clean_stock_data(df)
    if df.empty:
        return None, f"无有效数据: {code}"

    if period == 'w':
        # 周线数据：API 已通过 interval='1wk' 返回周线数据
        # 仅在 API 返回日线时才 resample（兼容旧逻辑）
        if len(df) > days * 3:
            weekly = (
                df.set_index('date')
                .resample('W-FRI')
                .agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                })
                .dropna()
                .reset_index()
            )
            df = clean_stock_data(weekly)

    df = df.tail(days).reset_index(drop=True)
    if df.empty:
        return None, f"数据不足: {code}"

    return df[['date', 'open', 'high', 'low', 'close', 'volume']], None


# ============ 数据清洗函数 ============

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """统一数据清洗流程

    Args:
        df: 原始数据DataFrame

    Returns:
        清洗后的DataFrame
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # 确保必要列存在
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()  # 返回空DataFrame表示数据无效

    # 转换数值类型
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 删除包含NaN的行
    df = df.dropna(subset=required_cols)

    # 过滤异常数据（价格为负或为零）
    df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['open'] > 0)]

    # 确保high >= low
    df = df[df['high'] >= df['low']]

    # 按日期排序
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    return df


# ============ 验证函数 ============

def validate_stock_code(code: str) -> Tuple[bool, str]:
    """验证股票代码格式

    Returns:
        (is_valid, error_message)
    """
    if not code:
        return False, "股票代码不能为空"

    # A股：6位数字
    if re.match(r'^\d{6}$', code):
        return True, ""

    # 美股：1-5个大写字母
    if re.match(r'^[A-Z]{1,5}$', code):
        return True, ""

    # 港股：4位数字 + .HK
    if re.match(r'^\d{4}\.HK$', code):
        return True, ""

    # 台股：4位数字 + .TW
    if re.match(r'^\d{4}\.TW$', code):
        return True, ""

    return False, f"股票代码格式无效: {code}\n  A股: 6位数字 (如 600519)\n  美股: 1-5个大写字母 (如 AAPL)\n  港股: 4位数字.HK (如 0700.HK)\n  台股: 4位数字.TW (如 2330.TW)"


def validate_data(df: pd.DataFrame, min_rows: int = 30) -> Tuple[bool, str, pd.DataFrame]:
    """验证数据是否满足分析要求

    Args:
        df: 数据DataFrame
        min_rows: 最小数据行数（默认30，满足基本计算需求）

    Returns:
        (is_valid, error_message, cleaned_df)
    """
    if df is None or df.empty:
        return False, "无法获取股票数据，请检查股票代码是否正确", pd.DataFrame()

    # 统一数据清洗
    df_clean = clean_stock_data(df)

    if df_clean.empty:
        return False, "数据清洗后无有效数据，请检查数据质量", pd.DataFrame()

    if len(df_clean) < min_rows:
        return False, f"数据不足，当前仅 {len(df_clean)} 条有效数据，建议使用 -d {min_rows + 50} 或更大的天数参数（至少需要 {min_rows} 条数据）", pd.DataFrame()

    # 数据量建议提示（返回在success message中）
    return True, "", df_clean


def validate_indicators(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """验证指标计算结果是否在合理范围内

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    last = df.iloc[-1]

    # 检查关键指标是否为NaN
    nan_indicators = []
    for col in ['MA5', 'MA10', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 'ADX', 'ATR']:
        if col in df.columns and pd.isna(last[col]):
            nan_indicators.append(col)
    if nan_indicators:
        warnings.append(f"以下指标计算结果为空: {', '.join(nan_indicators)}")

    # RSI范围检查 (0-100)
    if 'RSI' in df.columns and pd.notna(last['RSI']):
        if last['RSI'] < 0 or last['RSI'] > 100:
            warnings.append(f"RSI值异常: {last['RSI']:.2f} (应在0-100之间)")

    # KDJ 已移除

    # ADX范围检查 (0-100)
    if 'ADX' in df.columns and pd.notna(last['ADX']):
        if last['ADX'] < 0 or last['ADX'] > 100:
            warnings.append(f"ADX值异常: {last['ADX']:.2f} (应在0-100之间)")

    # CCI范围检查 (通常-300到+300)
    if 'CCI' in df.columns and pd.notna(last['CCI']):
        if last['CCI'] < -500 or last['CCI'] > 500:
            warnings.append(f"CCI值异常: {last['CCI']:.2f}")

    return len(warnings) == 0, warnings


def safe_fetch_yfinance_chart(code: str, period: str, days: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """使用 Yahoo chart API 获取行情数据。"""
    try:
        import requests
    except ImportError:
        return None, "请安装requests库"

    def _do_request(verify=True):
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT, verify=verify)
        if response.status_code != 200:
            return None, f"API请求失败: HTTP {response.status_code}"
        return parse_chart_response(response.json(), code, period, days)

    try:
        yf_code = normalize_symbol(code)
        end_time = int(time.time())
        start_time = end_time - get_history_window_days(period, days) * 86400

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_code}"

        interval_map = {'d': '1d', 'w': '1wk', 'm': '1mo'}
        interval = interval_map.get(period, '1d')

        params = {
            'period1': start_time,
            'period2': end_time,
            'interval': interval,
            'events': 'history'
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

        return _do_request()

    except requests.exceptions.SSLError:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return _do_request(verify=False)
        except Exception:
            return None, f"获取数据失败: {code} - SSLError"

    except Exception as e:
        return None, f"获取数据失败: {code} - {type(e).__name__}"


def safe_fetch_yfinance(code: str, period: str, days: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """兼容旧调用名，实际走稳定的 chart API。"""
    return safe_fetch_yfinance_chart(code, period, days)


def safe_fetch_akshare(code: str, period: str, days: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """安全获取akshare数据，带错误处理

    Returns:
        (dataframe, error_message)
    """
    try:
        import akshare as ak
    except ImportError:
        return None, "请安装akshare: pip install akshare"

    try:
        # 判断市场
        if code.startswith(('6', '5', '9')):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"

        # 获取数据
        period_str = "daily" if period == 'd' else "weekly"
        df = call_with_suppressed_output(lambda: ak.stock_zh_a_hist(symbol=symbol, period=period_str, adjust="qfq"))

        # 标准化列名 - 处理不同版本的列名
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if '日期' in col or 'date' in col_lower:
                column_mapping[col] = 'date'
            elif '开盘' in col or 'open' in col_lower:
                column_mapping[col] = 'open'
            elif '收盘' in col or 'close' in col_lower:
                column_mapping[col] = 'close'
            elif '最高' in col or 'high' in col_lower:
                column_mapping[col] = 'high'
            elif '最低' in col or 'low' in col_lower:
                column_mapping[col] = 'low'
            elif '成交' in col and ('量' in col or 'volume' in col_lower):
                column_mapping[col] = 'volume'

        df = df.rename(columns=column_mapping)

        if 'date' not in df.columns:
            # 尝试使用第一列作为日期
            df = df.rename(columns={df.columns[0]: 'date'})

        df['date'] = pd.to_datetime(df['date'])

        # 取最近N天
        df = df.tail(days).reset_index(drop=True)

        return df[['date', 'open', 'high', 'low', 'close', 'volume']], None

    except Exception as e:
        error_msg = str(e).lower()
        if 'connection' in error_msg or 'network' in error_msg or 'timeout' in error_msg or 'proxy' in error_msg:
            return None, "网络连接失败，请检查网络后重试"
        else:
            return None, f"获取数据失败: {code}"


def generate_demo_data(code: str, period: str, days: int) -> pd.DataFrame:
    """生成演示数据（用于测试）"""
    effective_days = max(days, 300)  # 确保足够数据用于200日均线等计算
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp('2024-01-15'), periods=effective_days, freq='D')

    # 模拟股票走势
    base_price = 100 if not code.isdigit() else float(code[:3])
    prices = [base_price]
    for i in range(1, effective_days):
        # 添加趋势和随机波动
        trend = 0.05 if i < effective_days // 2 else -0.03  # 先涨后跌
        change = np.random.randn() * 2 + trend
        prices.append(max(1, prices[-1] + change))

    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + abs(np.random.randn() * 1.5) for p in prices],
        'low': [p - abs(np.random.randn() * 1.5) for p in prices],
        'close': [p + np.random.randn() * 0.5 for p in prices],
        'volume': [1000000 + np.random.randint(-200000, 200000) for _ in range(effective_days)]
    })

    return df


def fetch_stock_data(code: str, period: str, days: int, retry: int = 2, demo: bool = False) -> pd.DataFrame:
    """获取股票数据，优先使用 Yahoo chart API，A 股失败后尝试 akshare。"""
    if demo:
        print("使用演示数据模式")
        return generate_demo_data(code, period, days)

    is_a_share = code.isdigit() and len(code) == 6

    for attempt in range(retry):
        df, error = safe_fetch_yfinance(code, period, days)
        if df is not None:
            print("数据源: Yahoo chart API")
            return df

        if is_a_share:
            print("Yahoo行情获取失败，正在尝试akshare...")
            df, error = safe_fetch_akshare(code, period, days)
            if df is not None:
                print("数据源: akshare")
                return df

        if error and ("rate" in error.lower() or "429" in error or "http 429" in error.lower()):
            wait_time = (attempt + 1) * 3
            print(f"API限速，等待{wait_time}秒后重试... ({attempt + 1}/{retry})")
            time.sleep(wait_time)
        elif error and ("connection" in error.lower() or "timeout" in error.lower()):
            wait_time = (attempt + 1) * 2
            print(f"网络异常({error})，等待{wait_time}秒后重试... ({attempt + 1}/{retry})")
            time.sleep(wait_time)
        else:
            raise ValueError(error if error else "获取数据失败，请检查股票代码或网络连接")

    raise ValueError("多次重试后仍无法获取数据，请稍后再试")


def fetch_weekly_confirmation(code: str, demo: bool = False) -> dict:
    """
    获取周线数据并判断周线趋势确认

    Args:
        code: 股票代码
        demo: 是否使用演示数据

    Returns:
        dict: {
            'weekly_trend': 'uptrend' | 'downtrend' | 'sideways',
            'weekly_bos_signal': 1 | -1 | 0,
            'weekly_choch_signal': 1 | -1 | 0,
            'weekly_adx': float,
            'explanation': str
        }
    """
    default_result = {
        'weekly_trend': 'sideways',
        'weekly_bos_signal': 0,
        'weekly_choch_signal': 0,
        'weekly_adx': 0.0,
        'weekly_50ma': 0.0,
        'weekly_200ma': 0.0,
        'weekly_golden_cross': False,
        'long_term_trend_health': 'unknown',
        'explanation': '周线数据不足'
    }

    try:
        # 获取周线数据（1500天 ≈ 215周，覆盖200周均线计算）
        df_weekly = fetch_stock_data(code, period='w', days=1500, demo=demo)

        if df_weekly is None or len(df_weekly) < 30:
            return default_result

        # 计算周线指标
        df_weekly = calculate_indicators(df_weekly)

        if len(df_weekly) < 20:
            return default_result

        last_weekly = df_weekly.iloc[-1]

        # 获取周线 BoS 和 ChoCh 信号
        weekly_bos_signal = int(last_weekly.get('bos_signal', 0))
        weekly_choch_signal = int(last_weekly.get('choch_signal', 0))
        weekly_adx = float(last_weekly.get('ADX', 0.0))

        if pd.isna(weekly_adx):
            weekly_adx = 0.0

        # 判断周线趋势
        ma5 = last_weekly.get('MA5', 0)
        ma10 = last_weekly.get('MA10', 0)
        ma20 = last_weekly.get('MA20', 0)

        if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20):
            if ma5 > ma10 > ma20:
                weekly_trend = 'uptrend'
                trend_text = '上升趋势'
            elif ma5 < ma10 < ma20:
                weekly_trend = 'downtrend'
                trend_text = '下降趋势'
            else:
                weekly_trend = 'sideways'
                trend_text = '震荡'
        else:
            weekly_trend = 'sideways'
            trend_text = '震荡'

        # 生成解释
        explanation = f"周线{trend_text} (ADX={weekly_adx:.1f})"
        if weekly_bos_signal == 1:
            explanation += ", BoS看多"
        elif weekly_bos_signal == -1:
            explanation += ", BoS看空"

        if weekly_choch_signal == 1:
            explanation += ", ChoCh看多"
        elif weekly_choch_signal == -1:
            explanation += ", ChoCh看空"

        # 计算50周/200周均线（长线趋势健康检查）
        weekly_50ma = 0.0
        weekly_200ma = 0.0
        weekly_golden_cross = False
        long_term_trend_health = 'unknown'

        if len(df_weekly) >= 200:
            weekly_50ma = float(df_weekly['close'].rolling(50).mean().iloc[-1])
            weekly_200ma = float(df_weekly['close'].rolling(200).mean().iloc[-1])
            if not pd.isna(weekly_50ma) and not pd.isna(weekly_200ma) and weekly_200ma > 0:
                weekly_golden_cross = weekly_50ma > weekly_200ma
                if weekly_golden_cross:
                    long_term_trend_health = 'healthy'
                    explanation += f" | 长线健康 (50W={weekly_50ma:.2f} > 200W={weekly_200ma:.2f})"
                else:
                    long_term_trend_health = 'broken'
                    explanation += f" | ⚠ 长线破位 (50W={weekly_50ma:.2f} < 200W={weekly_200ma:.2f})"
            else:
                weekly_50ma = 0.0
                weekly_200ma = 0.0
        elif len(df_weekly) >= 50:
            weekly_50ma = float(df_weekly['close'].rolling(50).mean().iloc[-1])
            if pd.isna(weekly_50ma):
                weekly_50ma = 0.0
            long_term_trend_health = 'insufficient'
            explanation += " | 200周均线数据不足"

        return {
            'weekly_trend': weekly_trend,
            'weekly_bos_signal': weekly_bos_signal,
            'weekly_choch_signal': weekly_choch_signal,
            'weekly_adx': weekly_adx,
            'weekly_50ma': weekly_50ma,
            'weekly_200ma': weekly_200ma,
            'weekly_golden_cross': weekly_golden_cross,
            'long_term_trend_health': long_term_trend_health,
            'explanation': explanation
        }

    except Exception:
        return default_result


def fetch_monthly_confirmation(code: str, demo: bool = False) -> dict:
    """
    获取月线数据并判断月线趋势确认

    Args:
        code: 股票代码
        demo: 是否使用演示数据

    Returns:
        dict: 月线趋势确认信息
    """
    default_result = {
        'monthly_trend': 'sideways',
        'monthly_ma12': 0.0,
        'monthly_ma24': 0.0,
        'monthly_adx': 0.0,
        'monthly_golden_cross': False,
        'explanation': '月线数据不足'
    }

    try:
        # 获取月线数据（1800天 ≈ 60个月 ≈ 5年）
        df_monthly = fetch_stock_data(code, period='m', days=1800, demo=demo)

        if df_monthly is None or len(df_monthly) < 24:
            return default_result

        # 计算月线指标
        df_monthly = calculate_indicators(df_monthly)

        if len(df_monthly) < 12:
            return default_result

        last_monthly = df_monthly.iloc[-1]

        # 获取月线均线（MA10≈10月, MA20≈20月）和 ADX
        monthly_ma10 = float(last_monthly.get('MA10', 0.0))
        monthly_ma20 = float(last_monthly.get('MA20', 0.0))
        monthly_adx = float(last_monthly.get('ADX', 0.0))

        if pd.isna(monthly_ma10):
            monthly_ma10 = 0.0
        if pd.isna(monthly_ma20):
            monthly_ma20 = 0.0
        if pd.isna(monthly_adx):
            monthly_adx = 0.0

        # 判断月线趋势
        if monthly_ma10 > 0 and monthly_ma20 > 0:
            if monthly_ma10 > monthly_ma20:
                monthly_trend = 'uptrend'
                trend_text = '上升趋势'
            elif monthly_ma10 < monthly_ma20:
                monthly_trend = 'downtrend'
                trend_text = '下降趋势'
            else:
                monthly_trend = 'sideways'
                trend_text = '震荡'
        else:
            monthly_trend = 'sideways'
            trend_text = '震荡'

        explanation = f"月线{trend_text} (MA10={monthly_ma10:.2f}, MA20={monthly_ma20:.2f})"

        monthly_golden_cross = monthly_ma10 > monthly_ma20 if (monthly_ma10 > 0 and monthly_ma20 > 0) else False

        return {
            'monthly_trend': monthly_trend,
            'monthly_ma12': monthly_ma10,
            'monthly_ma24': monthly_ma20,
            'monthly_adx': monthly_adx,
            'monthly_golden_cross': monthly_golden_cross,
            'explanation': explanation
        }

    except Exception:
        return default_result


def fetch_valuation_metrics(code: str) -> dict:
    """获取估值指标 (PE/PS 历史百分位)"""
    default_result = {
        'current_pe': None, 'current_ps': None,
        'pe_percentile': None, 'ps_percentile': None,
        'valuation_signal': 'fair', 'signal_text': '估值数据不可用',
        'available': False, 'source': '无数据'
    }

    is_a_share = code.isdigit() and len(code) == 6

    try:
        import yfinance as yf
        yf_code = normalize_symbol(code)
        ticker = yf.Ticker(yf_code)

        info = call_with_suppressed_output(lambda: ticker.info)
        if not info:
            return default_result

        current_pe = info.get('trailingPE')
        current_ps = info.get('priceToSalesTrailing12Months')

        if current_pe and current_pe < 0:
            current_pe = None

        if not current_pe and not current_ps:
            default_result['source'] = '该标的无估值数据'
            return default_result

        hist = call_with_suppressed_output(lambda: ticker.history(period='5y', interval='1wk'))
        if hist is None or hist.empty or len(hist) < 52:
            default_result['source'] = '历史数据不足'
            return default_result

        pe_percentile = None
        ps_percentile = None

        hist_prices = hist['Close'].dropna()
        if len(hist_prices) == 0:
            return default_result

        current_price = hist_prices.iloc[-1]

        # 计算历史 PE 分位数
        if current_pe and current_pe > 0:
            current_eps = current_price / current_pe
            if current_eps > 0:
                hist_pe = hist_prices / current_eps
                hist_pe = hist_pe[hist_pe > 0]
                if len(hist_pe) > 0:
                    pe_percentile = float((hist_pe < current_pe).sum() / len(hist_pe) * 100)

        # 计算历史 PS 分位数
        if current_ps and current_ps > 0:
            current_sps = current_price / current_ps
            if current_sps > 0:
                hist_ps = hist_prices / current_sps
                hist_ps = hist_ps[hist_ps > 0]
                if len(hist_ps) > 0:
                    ps_percentile = float((hist_ps < current_ps).sum() / len(hist_ps) * 100)

        # 估值信号判定
        signal = 'fair'
        signal_text = '估值合理'

        if pe_percentile is not None and ps_percentile is not None:
            if pe_percentile < 20 and ps_percentile < 30:
                signal = 'undervalued'
                signal_text = f'低估（PE {pe_percentile:.0f}%分位，PS {ps_percentile:.0f}%分位）'
            elif pe_percentile > 80 or ps_percentile > 80:
                signal = 'overvalued'
                signal_text = f'高估（PE {pe_percentile:.0f}%分位，PS {ps_percentile:.0f}%分位）'
            else:
                signal_text = f'合理（PE {pe_percentile:.0f}%分位，PS {ps_percentile:.0f}%分位）'
        elif pe_percentile is not None:
            if pe_percentile < 20:
                signal = 'undervalued'
                signal_text = f'低估（PE {pe_percentile:.0f}%分位）'
            elif pe_percentile > 80:
                signal = 'overvalued'
                signal_text = f'高估（PE {pe_percentile:.0f}%分位）'
            else:
                signal_text = f'合理（PE {pe_percentile:.0f}%分位）'
        elif ps_percentile is not None:
            if ps_percentile < 30:
                signal = 'undervalued'
                signal_text = f'低估（PS {ps_percentile:.0f}%分位）'
            elif ps_percentile > 80:
                signal = 'overvalued'
                signal_text = f'高估（PS {ps_percentile:.0f}%分位）'
            else:
                signal_text = f'合理（PS {ps_percentile:.0f}%分位）'

        data_years = round(len(hist_prices) / 52, 1)

        return {
            'current_pe': round(current_pe, 2) if current_pe else None,
            'current_ps': round(current_ps, 2) if current_ps else None,
            'pe_percentile': round(pe_percentile, 1) if pe_percentile is not None else None,
            'ps_percentile': round(ps_percentile, 1) if ps_percentile is not None else None,
            'valuation_signal': signal,
            'signal_text': signal_text,
            'data_years': data_years,
            'available': True,
            'source': 'yfinance' + (' (A股数据可能不完整)' if is_a_share else '')
        }

    except Exception:
        default_result['source'] = '估值数据获取失败'
        return default_result


def fetch_analyst_consensus(code: str) -> dict:
    """获取分析师评级共识

    数据源优先级:
    1. yfinance (美股/港股/A股)
    2. akshare (A股备选，使用stock_rank_forecast_cninfo)

    yfinance获取:
    - recommendationKey: strong_buy/buy/hold/sell/strong_sell
    - targetMeanPrice: 目标均价
    - numberOfAnalystOpinions: 分析师数量

    akshare获取(仅A股):
    - 使用stock_rank_forecast_cninfo接口
    - 汇总买入/卖出评级比例

    Returns:
        {
            'rating': 'buy'|'hold'|'sell',
            'rating_text': '买入居多（建议加仓）',
            'target_price': 目标价,
            'analyst_count': 分析师数量,
            'source': 'Yahoo主流研报'|'东方财富研报'
        }
    """
    default_result = {
        'rating': 'hold',
        'rating_text': '暂无评级数据',
        'target_price': None,
        'analyst_count': 0,
        'source': '无数据'
    }

    # 判断是否为A股
    is_a_share = code.isdigit() and len(code) == 6

    # 尝试使用 yfinance 获取
    try:
        import yfinance as yf
        yf_code = normalize_symbol(code)
        ticker = yf.Ticker(yf_code)
        info = call_with_suppressed_output(lambda: ticker.info)

        if info:
            rec_key = info.get('recommendationKey', '')
            target_price = info.get('targetMeanPrice')
            analyst_count = info.get('numberOfAnalystOpinions', 0)

            # 映射评级
            rating_map = {
                'strong_buy': ('buy', '强烈买入（建议重点加仓）'),
                'buy': ('buy', '买入居多（建议加仓）'),
                'hold': ('hold', '持仓观望'),
                'sell': ('sell', '卖出为主（建议减仓）'),
                'strong_sell': ('sell', '强烈卖出（建议清仓）')
            }

            if rec_key in rating_map:
                rating, rating_text = rating_map[rec_key]
                return {
                    'rating': rating,
                    'rating_text': rating_text,
                    'target_price': target_price,
                    'analyst_count': analyst_count if analyst_count else 0,
                    'source': 'Yahoo主流研报'
                }

    except Exception:
        pass

    if is_a_share:
        try:
            import akshare as ak
            df = call_with_suppressed_output(lambda: ak.stock_rank_forecast_cninfo(symbol=code))

            if df is not None and not df.empty:
                # 统计评级（假设列名包含"评级"或类似字段）
                # 该接口返回预测数据，我们需要统计看多/看空比例
                buy_count = 0
                sell_count = 0
                hold_count = 0

                for _, row in df.iterrows():
                    # 尝试从数据中提取评级信息
                    rating_str = str(row.get('评级', row.get('rating', ''))).lower()
                    if '买' in rating_str or 'buy' in rating_str:
                        buy_count += 1
                    elif '卖' in rating_str or 'sell' in rating_str:
                        sell_count += 1
                    else:
                        hold_count += 1

                total = buy_count + sell_count + hold_count
                if total > 0:
                    if buy_count > sell_count and buy_count > hold_count:
                        return {
                            'rating': 'buy',
                            'rating_text': '买入居多（建议加仓）',
                            'target_price': None,
                            'analyst_count': total,
                            'source': '东方财富研报'
                        }
                    elif sell_count > buy_count and sell_count > hold_count:
                        return {
                            'rating': 'sell',
                            'rating_text': '卖出为主（建议减仓）',
                            'target_price': None,
                            'analyst_count': total,
                            'source': '东方财富研报'
                        }
                    else:
                        return {
                            'rating': 'hold',
                            'rating_text': '持仓观望',
                            'target_price': None,
                            'analyst_count': total,
                            'source': '东方财富研报'
                        }

        except Exception:
            pass

    return default_result


# ============ 新增功能: 期权数据 ============

def fetch_options_data(code: str) -> dict:
    """获取期权数据 (IV, P/C Ratio)

    数据源: yfinance (仅美股/港股部分标的)

    Returns:
        {
            'iv': float,              # 当前隐含波动率 (最近到期日 ATM 期权)
            'iv_rank': float,         # IV Rank (过去252日IV百分位)
            'iv_percentile': float,   # IV 百分位
            'put_call_ratio': float,  # 看跌/看涨比率
            'pc_ratio_change': float, # P/C Ratio 近5日变化（简化为0）
            'available': bool,        # 数据是否可用
            'source': str             # 数据源
        }
    """
    default_result = {
        'iv': None,
        'iv_rank': None,
        'iv_percentile': None,
        'put_call_ratio': None,
        'pc_ratio_change': 0,
        'available': False,
        'source': '无数据'
    }

    # A股不支持期权数据
    if code.isdigit() and len(code) == 6:
        default_result['source'] = 'A股暂不支持期权数据'
        return default_result

    try:
        import yfinance as yf
        yf_code = normalize_symbol(code)
        ticker = yf.Ticker(yf_code)

        # 获取可用的到期日
        expirations = call_with_suppressed_output(lambda: ticker.options)
        if not expirations or len(expirations) == 0:
            default_result['source'] = '该标的暂无期权数据'
            return default_result

        # 使用最近到期日
        nearest_exp = expirations[0]

        # 获取期权链
        opt_chain = call_with_suppressed_output(lambda: ticker.option_chain(nearest_exp))
        if opt_chain is None:
            return default_result

        calls = opt_chain.calls
        puts = opt_chain.puts

        if calls.empty or puts.empty:
            return default_result

        # 获取当前价格用于选择 ATM 期权
        info = call_with_suppressed_output(lambda: ticker.info)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')

        if current_price is None:
            # 使用最近的收盘价
            hist = call_with_suppressed_output(lambda: ticker.history(period='1d'))
            if hist.empty:
                return default_result
            current_price = hist['Close'].iloc[-1]

        # 选择 ATM 期权（最接近当前价格的）
        calls['strike_diff'] = abs(calls['strike'] - current_price)
        puts['strike_diff'] = abs(puts['strike'] - current_price)

        atm_call = calls.loc[calls['strike_diff'].idxmin()]
        atm_put = puts.loc[puts['strike_diff'].idxmin()]

        # 计算当前 IV（ATM call 和 put 的平均值）
        call_iv = atm_call.get('impliedVolatility', 0)
        put_iv = atm_put.get('impliedVolatility', 0)
        current_iv = (call_iv + put_iv) / 2 if call_iv and put_iv else (call_iv or put_iv or 0)

        # 计算 P/C Ratio（使用未平仓合约）
        total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
        total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0

        pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        # IV Rank 和 IV Percentile（简化计算，使用当前IV作为参考）
        # 实际应用中需要历史IV数据，这里简化处理
        iv_percentile = min(100, max(0, current_iv * 100))  # 简化：将IV转换为百分位
        iv_rank = iv_percentile  # 简化：IV Rank ≈ IV Percentile

        return {
            'iv': round(current_iv, 4) if current_iv else None,
            'iv_rank': round(iv_rank, 1) if iv_rank else None,
            'iv_percentile': round(iv_percentile, 1) if iv_percentile else None,
            'put_call_ratio': round(pc_ratio, 2) if pc_ratio else None,
            'pc_ratio_change': 0,  # 简化，需要历史数据才能计算
            'available': True,
            'source': 'yfinance期权数据'
        }

    except Exception:
        default_result['source'] = '期权数据获取失败'
        return default_result


def detect_fake_breakout(df: pd.DataFrame, options_data: dict, breakout_direction: str = 'up') -> dict:
    """检测假突破信号

    Args:
        df: 价格数据
        options_data: 期权数据（来自 fetch_options_data）
        breakout_direction: 'up' 或 'down'

    Returns:
        {
            'is_fake_breakout': bool,
            'warning_level': str,  # 'none' | 'low' | 'medium' | 'high'
            'signals': [],         # 触发的信号列表
            'details': str
        }
    """
    result = {
        'is_fake_breakout': False,
        'warning_level': 'none',
        'signals': [],
        'details': '无假突破风险'
    }

    if not options_data.get('available', False):
        result['details'] = '期权数据不可用，无法检测假突破'
        return result

    try:
        pc_ratio = options_data.get('put_call_ratio', 0)
        iv = options_data.get('iv', 0)

        if breakout_direction == 'up':
            # 价格向上突破时的假突破检测
            signals = []

            # 1. P/C Ratio 异常飙升 (>1.5 或近期涨幅>50%)
            if pc_ratio > 1.5:
                signals.append('P/C Ratio异常高(>1.5)')
            if options_data.get('pc_ratio_change', 0) > 0.5:
                signals.append('P/C Ratio近期飙升(>50%)')

            # 2. IV 飙升（市场恐慌）
            if iv > 0.6:  # IV > 60% 为高波动
                signals.append('IV异常高(>60%)')

            if signals:
                result['is_fake_breakout'] = True
                result['signals'] = signals
                result['warning_level'] = 'high' if len(signals) >= 2 else 'medium'
                result['details'] = f"假突破风险: {', '.join(signals)}"

        else:  # breakout_direction == 'down'
            # 价格向下突破时的假跌破检测
            signals = []

            # 1. P/C Ratio 下降（市场不恐慌）
            if pc_ratio < 0.7:
                signals.append('P/C Ratio较低(<0.7)，市场不恐慌')

            # 2. IV 高位（可能已恐慌到位）
            if iv > 0.5:
                signals.append('IV较高(>50%)，恐慌可能见顶')

            if signals:
                result['is_fake_breakout'] = True
                result['signals'] = signals
                result['warning_level'] = 'medium'
                result['details'] = f"可能假跌破: {', '.join(signals)}"

    except Exception:
        pass

    return result


# ============ 新增功能: 筹码分布 ============

def fetch_chip_distribution(code: str) -> dict:
    """获取筹码分布数据

    数据源: akshare.stock_cyq_em (东方财富筹码分布)

    Returns:
        {
            'price': 当前价,
            'cost_range': [(价格区间, 占比), ...],
            'concentration': 集中度描述,
            'main_cost_area': 主力成本区,
            'float_ratio': 浮筹比例,
            'profit_ratio': 获利盘比例,
            'source': '东方财富筹码分布'
        }
    """
    default_result = {
        'price': None,
        'cost_range': [],
        'concentration': '未知',
        'main_cost_area': None,
        'float_ratio': None,
        'profit_ratio': None,
        'source': '获取失败'
    }

    # 判断是否为A股
    is_a_share = code.isdigit() and len(code) == 6
    if not is_a_share:
        default_result['source'] = '仅支持A股'
        return default_result

    try:
        import akshare as ak

        # 获取筹码分布数据
        df = call_with_suppressed_output(lambda: ak.stock_cyq_em(symbol=code))

        if df is None or df.empty:
            default_result['source'] = '数据为空'
            return default_result

        # 解析数据 - 东方财富筹码分布返回的格式通常是:
        # 价位, 持股比例, 日期 等列
        # 需要根据实际返回格式进行调整

        # 尝试找到价位和比例列
        price_col = None
        ratio_col = None

        for col in df.columns:
            col_lower = str(col).lower()
            if '价' in col or 'price' in col_lower:
                price_col = col
            if '比' in col or 'ratio' in col_lower or '比例' in col:
                ratio_col = col

        if price_col is None or ratio_col is None:
            # 尝试使用默认列名
            if len(df.columns) >= 2:
                price_col = df.columns[0]
                ratio_col = df.columns[1]

        if price_col and ratio_col:
            # 转换数据
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            df[ratio_col] = pd.to_numeric(df[ratio_col], errors='coerce')
            df = df.dropna()

            if not df.empty:
                # 获取当前价格（最新价）
                current_price = df[price_col].iloc[-1] if len(df) > 0 else None

                # 获取筹码分布区间
                cost_range = []
                for _, row in df.iterrows():
                    price = row[price_col]
                    ratio = row[ratio_col]
                    if pd.notna(price) and pd.notna(ratio):
                        cost_range.append((round(price, 2), round(ratio, 2)))

                # 计算主力成本区（持股比例最高的价位区间）
                if cost_range:
                    # 排序找到最高比例的价位
                    sorted_by_ratio = sorted(cost_range, key=lambda x: x[1], reverse=True)
                    main_cost = sorted_by_ratio[0][0] if sorted_by_ratio else None

                    # 计算集中度（90%筹码区间）
                    sorted_by_price = sorted(cost_range, key=lambda x: x[0])
                    total_ratio = 0
                    low_price = None
                    high_price = None

                    for price, ratio in sorted_by_price:
                        total_ratio += ratio
                        if low_price is None:
                            low_price = price
                        if total_ratio <= 90:
                            high_price = price
                        else:
                            break

                    concentration = "集中" if current_price and (high_price - low_price) / current_price < 0.2 else "分散"

                    # 估算浮筹比例（通常取最近20%价位的筹码）
                    recent_range = cost_range[-10:] if len(cost_range) >= 10 else cost_range
                    float_ratio = sum(r for _, r in recent_range) / len(recent_range) if recent_range else None

                    # 估算获利盘（当前价格以上的筹码比例）
                    if current_price:
                        profit_ratio = sum(r for p, r in cost_range if p > current_price)
                    else:
                        profit_ratio = None

                    return {
                        'price': current_price,
                        'cost_range': cost_range[:20],  # 限制返回数量
                        'concentration': concentration,
                        'main_cost_area': f"{main_cost-2}-{main_cost+2}" if main_cost else None,
                        'float_ratio': round(float_ratio, 1) if float_ratio else None,
                        'profit_ratio': round(profit_ratio, 1) if profit_ratio else None,
                        'source': '东方财富筹码分布'
                    }

    except ImportError:
        default_result['source'] = '请安装akshare: pip install akshare'
    except Exception as e:
        error_msg = str(e).lower()
        if 'network' in error_msg or 'connection' in error_msg:
            default_result['source'] = '网络连接失败'
        else:
            default_result['source'] = f'获取失败: {type(e).__name__}'

    return default_result


# ============ 主力资金流向 ============

def fetch_fund_flow(code: str, df: pd.DataFrame = None) -> dict:
    """获取主力资金流向

    数据源:
    - A股: akshare.stock_fund_flow_industry 或 volume分析
    - 美股: volume + price change分析

    Returns:
        {
            'main_inflow_today': 当日主力净流入,
            'main_inflow_5d': 5日主力净流入,
            'signal': '净流入'/'净流出'/'观望',
            'trend': '持续流入'/'持续流出'/'反转',
            'volume_ratio': 量比,
            'source': 数据来源
        }
    """
    default_result = {
        'main_inflow_today': None,
        'main_inflow_5d': None,
        'main_inflow_10d': None,
        'signal': '未知',
        'trend': '观望',
        'volume_ratio': None,
        'source': '获取失败'
    }

    is_a_share = code.isdigit() and len(code) == 6

    # 方法1: 尝试akshare行业资金流（仅A股）
    if is_a_share:
        try:
            import akshare as ak

            # 获取行业资金流
            df_flow = call_with_suppressed_output(lambda: ak.stock_fund_flow_industry())

            if df_flow is not None and not df_flow.empty:
                # 尝试找到目标股票对应的行业
                # 这里简化处理，返回行业资金流数据
                default_result['source'] = '东方财富行业资金流'

                # 如果有大盘数据，计算简单的净流入
                if df is not None and len(df) >= 5:
                    # 基于成交量和价格变化估算
                    recent = df.tail(5)
                    avg_volume = df['volume'].rolling(5).mean().iloc[-1] if len(df) >= 5 else df['volume'].mean()
                    current_volume = df['volume'].iloc[-1]

                    # 计算价格变化
                    price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0

                    # 估算净流入（简化模型）
                    volume_diff = current_volume - avg_volume
                    if volume_diff > 0 and price_change > 0:
                        main_inflow = volume_diff * df['close'].iloc[-1] / 10000  # 万元
                        default_result['main_inflow_today'] = round(main_inflow, 2)
                        default_result['signal'] = '净流入'

                        # 5日累计
                        flow_5d = 0
                        for i in range(min(5, len(df))):
                            v = df['volume'].iloc[-(i+1)]
                            p = df['close'].iloc[-(i+1)]
                            pc = df['close'].iloc[-(i+1)] - df['open'].iloc[-(i+1)]
                            if pc > 0:
                                flow_5d += v * p / 10000
                        default_result['main_inflow_5d'] = round(flow_5d, 2)
                    elif volume_diff < 0 and price_change < 0:
                        default_result['signal'] = '净流出'
                        default_result['main_inflow_today'] = round(volume_diff * df['close'].iloc[-1] / 10000, 2)
                    else:
                        default_result['signal'] = '观望'

                    # 量比
                    default_result['volume_ratio'] = round(current_volume / avg_volume, 2) if avg_volume > 0 else 1

                return default_result

        except ImportError:
            pass
        except Exception:
            pass

    # 方法2: 基于成交量分析（通用）
    if df is not None and len(df) >= 20:
        try:
            # 计算各项指标
            avg_volume_5 = df['volume'].tail(5).mean()
            avg_volume_20 = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1

            # 近期涨跌分析
            recent_5d = df.tail(5)
            up_days = sum(1 for i in range(len(recent_5d)) if recent_5d['close'].iloc[-(i+1)] > recent_5d['open'].iloc[-(i+1)])
            up_volume = sum(recent_5d['volume'].iloc[-(i+1)] for i in range(len(recent_5d)) if recent_5d['close'].iloc[-(i+1)] > recent_5d['open'].iloc[-(i+1)])
            down_volume = sum(recent_5d['volume'].iloc[-(i+1)] for i in range(len(recent_5d)) if recent_5d['close'].iloc[-(i+1)] <= recent_5d['open'].iloc[-(i+1)])

            # 资金流向判断
            if up_volume > down_volume * 1.2:
                signal = '净流入'
                trend = '持续流入' if up_days >= 3 else '反弹'
            elif down_volume > up_volume * 1.2:
                signal = '净流出'
                trend = '持续流出' if (5 - up_days) >= 3 else '回调'
            else:
                signal = '观望'
                trend = '震荡'

            # 估算金额（使用成交量*价格/10000 = 万元）
            avg_price = df['close'].tail(5).mean()
            net_flow = (up_volume - down_volume) * avg_price / 10000

            return {
                'main_inflow_today': round(net_flow * 0.2, 2) if signal == '净流入' else round(net_flow * 0.2, 2),
                'main_inflow_5d': round(net_flow, 2),
                'main_inflow_10d': round(net_flow * 2, 2) if len(df) >= 10 else None,
                'signal': signal,
                'trend': trend,
                'volume_ratio': round(volume_ratio, 2),
                'source': '技术分析'
            }

        except Exception:
            pass

    default_result['source'] = '数据不足'
    return default_result


# ============ 实时新闻/舆情 ============

def fetch_news_sentiment(code: str) -> dict:
    """获取新闻和舆情分析

    数据源: yfinance.Ticker.news

    情绪判断规则:
    - 利好关键词: buy/upgrade/beat/raise/target/bullish/growth/profit/上调/增持
    - 利空关键词: sell/downgrade/miss/cut/bearish/loss/warning/risk/下调/减持

    Returns:
        {
            'news': [...],
            'sentiment': '利好'/'利空'/'中性',
            'sentiment_score': -10到+10,
            'summary': '简要总结',
            'source': 'Yahoo Finance'
        }
    """
    default_result = {
        'news': [],
        'sentiment': '中性',
        'sentiment_score': 0,
        'summary': '暂无新闻数据',
        'source': '获取失败'
    }

    is_a_share = code.isdigit() and len(code) == 6

    try:
        import yfinance as yf
        yf_code = normalize_symbol(code)
        ticker = yf.Ticker(yf_code)
        news_data = call_with_suppressed_output(lambda: ticker.news)

        if not news_data:
            default_result['source'] = '暂无新闻'
            return default_result

        # 解析新闻
        news_list = []
        for item in news_data[:5]:  # 取最近5条
            title = item.get('title', '')
            source = item.get('publisher', '未知来源')
            time_str = item.get('time', '')
            if time_str and hasattr(time_str, 'strftime'):
                time_str = time_str.strftime('%m-%d')
            elif time_str:
                time_str = str(time_str)[:10]

            news_list.append({
                'title': title,
                'source': source,
                'time': time_str,
                'url': item.get('link', '')
            })

        # 情绪分析
        bullish_keywords = ['buy', 'upgrade', 'beat', 'raise', 'target', 'bullish', 'growth',
                           'profit', '上调', '增持', '推荐', '看好', '买入', '强买', '超配']
        bearish_keywords = ['sell', 'downgrade', 'miss', 'cut', 'bearish', 'loss', 'warning',
                           'risk', '下调', '减持', '看空', '卖出', '弱卖', '低配', '警告']

        bullish_count = 0
        bearish_count = 0

        for item in news_data[:10]:
            title = str(item.get('title', '')).lower()
            summary = str(item.get('summary', '')).lower()

            for kw in bullish_keywords:
                if kw.lower() in title or kw.lower() in summary:
                    bullish_count += 1
                    break

            for kw in bearish_keywords:
                if kw.lower() in title or kw.lower() in summary:
                    bearish_count += 1
                    break

        # 计算情绪分数
        sentiment_score = (bullish_count - bearish_count) * 2
        sentiment_score = max(-10, min(10, sentiment_score))

        if sentiment_score > 3:
            sentiment = '利好'
        elif sentiment_score < -3:
            sentiment = '利空'
        else:
            sentiment = '中性'

        # 生成总结
        summary = f"{bullish_count}条利好，{bearish_count}条利空"
        if sentiment == '利好':
            summary += "，市场情绪偏多"
        elif sentiment == '利空':
            summary += "，市场情绪偏空"
        else:
            summary += "，市场观望为主"

        return {
            'news': news_list,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'summary': summary,
            'source': 'Yahoo Finance'
        }

    except ImportError:
        default_result['source'] = '请安装yfinance'
    except Exception as e:
        default_result['source'] = f'获取失败: {type(e).__name__}'

    return default_result


# ============ 大盘复盘+板块 ============

def get_market_review() -> dict:
    """获取大盘复盘数据

    指数代码:
    - 上证指数: ^SSEC
    - 深证成指: ^SZCOMP
    - 创业板指: 399006.SZ
    - 纳斯达克: ^IXIC
    - 道琼斯: ^DJI

    Returns:
        {
            'indices': {...},
            'sectors': [...],
            'market_status': '普涨'/'普跌'/'分化'/'震荡'
        }
    """
    default_result = {
        'indices': {},
        'sectors': [],
        'market_status': '获取失败'
    }

    try:
        index_codes = {
            '上证指数': '^SSEC',
            '深证成指': '^SZCOMP',
            '创业板指': '399006.SZ',
            '纳斯达克': '^IXIC',
            '道琼斯': '^DJI'
        }

        indices = {}
        changes = []

        for name, code in index_codes.items():
            hist, _ = safe_fetch_yfinance_chart(code, 'd', 5)
            if hist is not None and len(hist) >= 2:
                current_price = hist['close'].iloc[-1]
                prev_price = hist['close'].iloc[-2]
                change = (current_price - prev_price) / prev_price * 100
                indices[name] = {
                    'code': code,
                    'price': round(current_price, 2),
                    'change': round(change, 2)
                }
                changes.append(change)

        market_status = '数据不足'
        if changes:
            avg_change = sum(changes) / len(changes)
            pos_count = sum(1 for c in changes if c > 0)
            neg_count = len(changes) - pos_count

            if pos_count >= len(changes) * 0.7 and avg_change > 0.5:
                market_status = '普涨'
            elif neg_count >= len(changes) * 0.7 and avg_change < -0.5:
                market_status = '普跌'
            elif abs(avg_change) < 0.5:
                market_status = '震荡'
            else:
                market_status = '分化'

        default_result['indices'] = indices
        default_result['market_status'] = market_status

        try:
            import akshare as ak
            df_sector = call_with_suppressed_output(lambda: ak.stock_fund_flow_industry())

            if df_sector is not None and not df_sector.empty:
                sector_name_col = None
                change_col = None
                inflow_col = None

                for col in df_sector.columns:
                    col_str = str(col)
                    if '名称' in col_str or '行业' in col_str:
                        sector_name_col = col
                    elif '涨跌幅' in col_str or '涨幅' in col_str:
                        change_col = col
                    elif '净流入' in col_str or '流入' in col_str:
                        inflow_col = col

                if sector_name_col and change_col:
                    sectors = []
                    for _, row in df_sector.head(5).iterrows():
                        name = row.get(sector_name_col, '')
                        change = row.get(change_col, 0)
                        inflow = row.get(inflow_col, 0) if inflow_col else 0
                        name_str = str(name).strip()
                        if not name_str or re.fullmatch(r'[-+]?\d+(\.\d+)?', name_str):
                            continue
                        sectors.append({
                            'name': name_str,
                            'change': round(float(change), 2) if pd.notna(change) else 0,
                            'inflow': round(float(inflow), 2) if pd.notna(inflow) else 0
                        })
                    default_result['sectors'] = sectors
        except Exception:
            pass

        return default_result
    except Exception as e:
        default_result['market_status'] = f'获取失败: {type(e).__name__}'
    return default_result


# ============ 黑天鹅与熔断模块 (Circuit Breaker) ============

def fetch_macro_risk_data(code: str) -> dict:
    """获取宏观风险数据：VIX + 基准指数 MA200 状态

    根据股票类型选择基准指数：
    - A股 → 沪深300 (000300.SS)
    - 美股 → SPY
    - 港股 → ^HSI

    Returns:
        {
            'benchmark': {
                'code': str,           # 基准指数代码
                'name': str,           # 基准指数名称
                'price': float,        # 当前价格
                'ma200': float,        # 200日均线
                'below_ma200': bool,   # 是否跌破 MA200
                'deviation_pct': float # 偏离 MA200 百分比
            },
            'vix': {
                'current': float,      # 当前 VIX
                'prev_close': float,   # 前收盘
                'change_pct': float,   # 日内变化百分比
                'spike': bool          # 是否飙升 > 15%
            },
            'available': bool,
            'source': str
        }
    """
    default_result = {
        'benchmark': {
            'code': '', 'name': '', 'price': None, 'ma200': None,
            'below_ma200': False, 'deviation_pct': 0
        },
        'vix': {
            'current': None, 'prev_close': None, 'change_pct': 0, 'spike': False
        },
        'available': False,
        'source': '未获取'
    }

    try:
        # 根据股票代码判断市场类型，选择基准指数
        normalized = normalize_symbol(code)
        if normalized.endswith('.SS') or normalized.endswith('.SZ'):
            benchmark_code = '000300.SS'
            benchmark_name = '沪深300'
        elif normalized.endswith('.HK'):
            benchmark_code = '^HSI'
            benchmark_name = '恒生指数'
        else:
            benchmark_code = 'SPY'
            benchmark_name = 'S&P 500 (SPY)'

        # 获取基准指数数据（250天，用于计算 MA200）
        bm_df, bm_err = safe_fetch_yfinance_chart(benchmark_code, 'd', 250)

        if bm_df is not None and len(bm_df) >= 200:
            bm_close = bm_df['close']
            ma200 = bm_close.rolling(200).mean().iloc[-1]
            current_price = bm_close.iloc[-1]
            below_ma200 = current_price < ma200
            deviation_pct = ((current_price - ma200) / ma200) * 100

            default_result['benchmark'] = {
                'code': benchmark_code,
                'name': benchmark_name,
                'price': round(current_price, 2),
                'ma200': round(ma200, 2),
                'below_ma200': below_ma200,
                'deviation_pct': round(deviation_pct, 2)
            }
        elif bm_df is not None and len(bm_df) > 0:
            # 数据不足200天，仅记录价格
            default_result['benchmark'] = {
                'code': benchmark_code,
                'name': benchmark_name,
                'price': round(bm_df['close'].iloc[-1], 2),
                'ma200': None,
                'below_ma200': False,
                'deviation_pct': 0
            }

        # 获取 VIX 数据（2天，用于计算日内变化）
        vix_df, vix_err = safe_fetch_yfinance_chart('^VIX', 'd', 5)

        if vix_df is not None and len(vix_df) >= 2:
            vix_current = vix_df['close'].iloc[-1]
            vix_prev = vix_df['close'].iloc[-2]
            vix_change_pct = ((vix_current - vix_prev) / vix_prev) * 100
            vix_spike = vix_change_pct > 15

            default_result['vix'] = {
                'current': round(vix_current, 2),
                'prev_close': round(vix_prev, 2),
                'change_pct': round(vix_change_pct, 2),
                'spike': vix_spike
            }

        default_result['available'] = True
        default_result['source'] = 'Yahoo Finance'

    except Exception:
        default_result['source'] = '获取失败'

    return default_result


def check_gap_destruction(df: pd.DataFrame, hvn_zones: list) -> dict:
    """检测个股跳空毁灭：开盘价向下跳空穿越 HVN 且幅度 > 1.5x ATR

    Args:
        df: OHLCV DataFrame (包含 ATR 列)
        hvn_zones: 高成交量节点 [(low, high, ratio), ...]

    Returns:
        {
            'triggered': bool,         # 是否触发
            'gap_size': float,         # 跳空幅度（绝对值）
            'gap_atr_ratio': float,    # 跳空幅度 / ATR
            'breached_hvn': tuple,     # 被穿越的 HVN 区域
            'details': str             # 描述
        }
    """
    default_result = {
        'triggered': False,
        'gap_size': 0,
        'gap_atr_ratio': 0,
        'breached_hvn': None,
        'details': '无跳空毁灭'
    }

    if len(df) < 2:
        return default_result

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        today_open = last['open']
        prev_close = prev['close']
        atr = last.get('ATR')

        if pd.isna(atr) or atr <= 0:
            return default_result

        # 计算向下跳空幅度
        gap_size = prev_close - today_open  # 正值 = 向下跳空

        if gap_size <= 0:
            default_result['details'] = '无向下跳空'
            return default_result

        gap_atr_ratio = gap_size / atr

        # 条件1: 跳空幅度 > 1.5x ATR
        if gap_atr_ratio <= 1.5:
            default_result['gap_size'] = round(gap_size, 2)
            default_result['gap_atr_ratio'] = round(gap_atr_ratio, 2)
            default_result['details'] = f'向下跳空 {gap_size:.2f} ({gap_atr_ratio:.1f}x ATR)，未达阈值'
            return default_result

        # 条件2: 开盘价跌穿 HVN 区域
        breached_hvn = None
        for lo, hi, ratio in hvn_zones:
            # 前收盘在 HVN 上方或内部，今开盘跌穿 HVN 下沿
            if prev_close >= lo and today_open < lo:
                breached_hvn = (lo, hi, ratio)
                break

        if breached_hvn is None:
            default_result['gap_size'] = round(gap_size, 2)
            default_result['gap_atr_ratio'] = round(gap_atr_ratio, 2)
            default_result['details'] = f'向下跳空 {gap_size:.2f} ({gap_atr_ratio:.1f}x ATR)，但未穿越 HVN'
            return default_result

        # 两个条件同时满足 → 触发
        return {
            'triggered': True,
            'gap_size': round(gap_size, 2),
            'gap_atr_ratio': round(gap_atr_ratio, 2),
            'breached_hvn': breached_hvn,
            'details': f'跳空毁灭: 向下跳空 {gap_size:.2f} ({gap_atr_ratio:.1f}x ATR)，穿越 HVN ({breached_hvn[0]:.2f}-{breached_hvn[1]:.2f})'
        }

    except Exception:
        return default_result


def check_circuit_breaker(code: str, df: pd.DataFrame, hvn_zones: list,
                           demo: bool = False) -> dict:
    """黑天鹅与熔断检查（最高优先级）

    在所有个股分析之前运行，两层防线：
    1. 大盘环境阻断：基准指数跌破 MA200 或 VIX 飙升 > 15%
    2. 个股跳空毁灭：向下跳空穿越 HVN 且幅度 > 1.5x ATR

    Args:
        code: 股票代码
        df: OHLCV DataFrame
        hvn_zones: 高成交量节点
        demo: 是否为演示模式

    Returns:
        {
            'macro_triggered': bool,
            'gap_triggered': bool,
            'any_triggered': bool,
            'macro_data': dict,
            'gap_data': dict,
            'override_decision': str | None,
            'override_text': str,
            'override_action': str
        }
    """
    result = {
        'macro_triggered': False,
        'gap_triggered': False,
        'any_triggered': False,
        'macro_data': None,
        'gap_data': None,
        'override_decision': None,
        'override_text': '',
        'override_action': ''
    }

    # 1. 大盘环境阻断
    if demo:
        result['macro_data'] = {
            'available': False, 'source': '演示模式已跳过',
            'benchmark': {'code': '', 'name': '', 'price': None, 'ma200': None, 'below_ma200': False, 'deviation_pct': 0},
            'vix': {'current': None, 'prev_close': None, 'change_pct': 0, 'spike': False}
        }
    else:
        macro_data = fetch_macro_risk_data(code)
        result['macro_data'] = macro_data

        if macro_data.get('available'):
            bm = macro_data['benchmark']
            vix = macro_data['vix']

            below_ma200 = bm.get('below_ma200', False)
            vix_spike = vix.get('spike', False)

            if below_ma200 or vix_spike:
                result['macro_triggered'] = True
                result['any_triggered'] = True
                result['override_decision'] = 'CIRCUIT_BREAKER_MACRO'
                result['override_text'] = '宏观风险熔断，禁止买入'
                result['override_action'] = '禁止买入'

    # 2. 个股跳空毁灭检测
    gap_data = check_gap_destruction(df, hvn_zones)
    result['gap_data'] = gap_data

    if gap_data['triggered']:
        result['gap_triggered'] = True
        result['any_triggered'] = True
        # 跳空毁灭优先级更高（直接影响个股）
        result['override_decision'] = 'CIRCUIT_BREAKER_GAP'
        result['override_text'] = '逻辑证伪，切勿接飞刀，观望或止损'
        result['override_action'] = '观望或止损'

    return result


# ============ 长线风控函数 ============

def calc_max_drawdown_analysis(df: pd.DataFrame) -> dict:
    """历史最大回撤分析"""
    try:
        close = df['close']
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax * 100
        max_dd = float(drawdown.min())
        ath = float(cummax.iloc[-1])
        current = float(close.iloc[-1])
        ath_distance = (current - ath) / ath * 100 if ath > 0 else 0
        current_dd = float(drawdown.iloc[-1])

        return {
            'max_drawdown': round(max_dd, 1),
            'ath': round(ath, 2),
            'ath_distance': round(ath_distance, 1),
            'current_drawdown': round(current_dd, 1),
            'dd_buffer': round(current_dd - max_dd, 1),  # 距最大回撤还有多少空间
            'available': True
        }
    except Exception:
        return {'max_drawdown': 0, 'ath': 0, 'ath_distance': 0, 'current_drawdown': 0, 'dd_buffer': 0, 'available': False}


def calc_relative_strength(code: str, df: pd.DataFrame) -> dict:
    """相对强弱分析：对比基准指数（1年/3年）"""
    default_result = {
        'benchmark_name': '', 'stock_1y_return': None, 'benchmark_1y_return': None,
        'relative_1y': None, 'stock_3y_return': None, 'benchmark_3y_return': None,
        'relative_3y': None, 'signal': 'neutral', 'signal_text': '数据不足', 'available': False
    }

    try:
        # 选择基准
        normalized = normalize_symbol(code)
        if normalized.endswith('.SS') or normalized.endswith('.SZ'):
            benchmark_code, benchmark_name = '000300.SS', '沪深300'
        elif normalized.endswith('.HK'):
            benchmark_code, benchmark_name = '^HSI', '恒生指数'
        else:
            benchmark_code, benchmark_name = 'SPY', 'S&P 500'

        # 获取基准周线数据（3年 ≈ 156周）
        bm_df, err = safe_fetch_yfinance_chart(benchmark_code, 'w', 780)
        if bm_df is None or bm_df.empty:
            default_result['benchmark_name'] = benchmark_name
            return default_result

        stock_close = df['close']
        bm_close = bm_df['close']

        # 计算 1 年收益率（约 52 周）
        stock_1y = None
        bm_1y = None
        relative_1y = None
        if len(stock_close) >= 52 and len(bm_close) >= 52:
            stock_1y = (stock_close.iloc[-1] / stock_close.iloc[-52] - 1) * 100
            bm_1y = (bm_close.iloc[-1] / bm_close.iloc[-52] - 1) * 100
            relative_1y = stock_1y - bm_1y

        # 计算 3 年收益率（约 156 周）
        stock_3y = None
        bm_3y = None
        relative_3y = None
        if len(stock_close) >= 156 and len(bm_close) >= 156:
            stock_3y = (stock_close.iloc[-1] / stock_close.iloc[-156] - 1) * 100
            bm_3y = (bm_close.iloc[-1] / bm_close.iloc[-156] - 1) * 100
            relative_3y = stock_3y - bm_3y

        # 信号判定
        signal = 'neutral'
        signal_text = '相对强弱数据不足'

        if relative_1y is not None and relative_3y is not None:
            if relative_1y > 0 and relative_3y > 0:
                signal = 'outperform'
                signal_text = f'长期相对强势，1年和3年均跑赢{benchmark_name}'
            elif relative_1y < 0 and relative_3y < 0:
                signal = 'underperform'
                signal_text = f'长期相对弱势，1年和3年均跑输{benchmark_name}，长线需谨慎'
            elif relative_1y > 0:
                signal = 'neutral'
                signal_text = f'近1年跑赢但3年跑输{benchmark_name}，趋势改善中'
            else:
                signal = 'neutral'
                signal_text = f'近1年跑输但3年跑赢{benchmark_name}，近期走弱'
        elif relative_1y is not None:
            if relative_1y > 0:
                signal = 'outperform'
                signal_text = f'近1年跑赢{benchmark_name}'
            else:
                signal = 'underperform'
                signal_text = f'近1年跑输{benchmark_name}'

        return {
            'benchmark_name': benchmark_name,
            'stock_1y_return': round(stock_1y, 1) if stock_1y is not None else None,
            'benchmark_1y_return': round(bm_1y, 1) if bm_1y is not None else None,
            'relative_1y': round(relative_1y, 1) if relative_1y is not None else None,
            'stock_3y_return': round(stock_3y, 1) if stock_3y is not None else None,
            'benchmark_3y_return': round(bm_3y, 1) if bm_3y is not None else None,
            'relative_3y': round(relative_3y, 1) if relative_3y is not None else None,
            'signal': signal,
            'signal_text': signal_text,
            'available': True
        }

    except Exception:
        return default_result


def calc_dca_zone(df: pd.DataFrame, valuation: dict, sr_data: dict) -> dict:
    """长线定投建议区间（结合周线支撑+估值）"""
    try:
        current_price = float(df['close'].iloc[-1])
        last = df.iloc[-1]

        # 区间下限：MA50 或支撑位
        ma50 = last.get('MA50')
        support_levels = sr_data.get('support', [])

        if pd.notna(ma50) and ma50 > 0:
            zone_low = float(ma50)
            low_reason = 'MA50 长线支撑'
        elif support_levels:
            zone_low = float(support_levels[0])
            low_reason = '周线支撑位'
        else:
            zone_low = current_price * 0.9
            low_reason = '当前价下方10%'

        # 区间上限：MA20 或当前价
        ma20 = last.get('MA20')
        if pd.notna(ma20) and ma20 > 0:
            zone_high = float(ma20)
            high_reason = 'MA20'
        else:
            zone_high = current_price
            high_reason = '当前价'

        # 确保 zone_low < zone_high
        if zone_low >= zone_high:
            zone_low, zone_high = min(zone_low, zone_high), max(zone_low, zone_high)
            if zone_low == zone_high:
                zone_low = zone_high * 0.95

        # 估值调整
        pe_pct = valuation.get('pe_percentile') if valuation.get('available') else None
        valuation_adjusted = False
        adjust_reason = ''

        if pe_pct is not None:
            if pe_pct < 30:
                # 低估：区间上移 5%（更积极）
                zone_high = zone_high * 1.05
                valuation_adjusted = True
                adjust_reason = f'估值低估({pe_pct:.0f}%分位)，区间上移'
            elif pe_pct > 70:
                # 高估：区间下移 5%（更保守）
                zone_high = zone_high * 0.95
                zone_low = zone_low * 0.95
                valuation_adjusted = True
                adjust_reason = f'估值偏高({pe_pct:.0f}%分位)，区间下移'

        zone_reason = f'{low_reason} ~ {high_reason}'
        if adjust_reason:
            zone_reason += f'，{adjust_reason}'

        # 判断当前价位
        if current_price < zone_low:
            position_text = '低于区间，可积极建仓'
        elif current_price > zone_high:
            position_text = '高于区间，建议等待回调'
        else:
            position_text = '区间内，可分批建仓'

        return {
            'zone_low': round(zone_low, 2),
            'zone_high': round(zone_high, 2),
            'zone_reason': zone_reason,
            'valuation_adjusted': valuation_adjusted,
            'position_text': position_text,
            'available': True
        }

    except Exception:
        return {'zone_low': 0, 'zone_high': 0, 'zone_reason': '数据不足', 'valuation_adjusted': False, 'position_text': '', 'available': False}


# ============ 仓位管理核心函数 ============

def calc_chandelier_exit(df: pd.DataFrame, atr_multiplier: float = CHANDELIER_ATR_MULTIPLIER,
                          lookback: int = CHANDELIER_LOOKBACK) -> dict:
    """
    计算 Chandelier Exit（吊灯止损）

    做多止损 = 最高价(lookback日) - (atr_multiplier * ATR)
    做空止损 = 最低价(lookback日) + (atr_multiplier * ATR)

    Args:
        df: OHLCV DataFrame (必须包含 ATR 列)
        atr_multiplier: ATR倍数，默认2.5
        lookback: 回溯天数，默认22日（约一个月）

    Returns:
        {
            'long_stop': float,       # 做多止损价
            'short_stop': float,      # 做空止损价
            'highest_high': float,    # 区间最高价
            'lowest_low': float,      # 区间最低价
            'atr': float,             # 当前ATR值
            'atr_pct': float,         # ATR占价格百分比
            'atr_multiplier': float   # 使用的ATR倍数
        }
    """
    default_result = {
        'long_stop': None,
        'short_stop': None,
        'highest_high': None,
        'lowest_low': None,
        'atr': None,
        'atr_pct': None,
        'atr_multiplier': atr_multiplier
    }

    if len(df) < lookback:
        return default_result

    try:
        last = df.iloc[-1]
        current_price = last['close']

        # 获取ATR
        atr = last.get('ATR')
        if pd.isna(atr):
            return default_result

        # 计算区间最高价和最低价
        lookback_df = df.tail(lookback)
        highest_high = lookback_df['high'].max()
        lowest_low = lookback_df['low'].min()

        # 计算止损价
        long_stop = highest_high - (atr_multiplier * atr)
        short_stop = lowest_low + (atr_multiplier * atr)

        # ATR占价格百分比
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        return {
            'long_stop': round(long_stop, 2),
            'short_stop': round(short_stop, 2),
            'highest_high': round(highest_high, 2),
            'lowest_low': round(lowest_low, 2),
            'atr': round(atr, 2),
            'atr_pct': round(atr_pct, 2),
            'atr_multiplier': atr_multiplier
        }

    except Exception:
        return default_result


def calc_position_sizing(current_price: float, stop_loss: float,
                          account_size: float = DEFAULT_ACCOUNT_SIZE,
                          max_risk_pct: float = MAX_RISK_PER_TRADE,
                          adv: float = None) -> dict:
    """
    波动率平价建仓模型

    公式：建议买入股数 = (账户规模 * MAX_RISK_PER_TRADE) / (当前价格 - 止损价)
    流动性限制：仓位不超过 ADV 的 1%

    Args:
        current_price: 当前价格
        stop_loss: 止损价
        account_size: 账户规模
        max_risk_pct: 每笔交易最大风险比例
        adv: 20日平均成交量（股数）

    Returns:
        {
            'max_risk_amount': float,     # 本次交易最大承担风险金额
            'stop_loss': float,           # 止损价
            'stop_loss_pct': float,       # 止损幅度百分比
            'suggested_shares': int,      # 建议买入股数（向下取整到100股）
            'position_value': float,      # 仓位市值
            'position_pct': float,        # 占账户比例
            'volatility_warning': str,    # 波动率警告
            'risk_reward_warning': str,   # 盈亏比警告
            'liquidity_warning': str,     # 流动性警告
            'is_valid': bool              # 仓位是否有效
        }
    """
    default_result = {
        'max_risk_amount': 0,
        'stop_loss': stop_loss,
        'stop_loss_pct': 0,
        'suggested_shares': 0,
        'position_value': 0,
        'position_pct': 0,
        'volatility_warning': '',
        'risk_reward_warning': '',
        'liquidity_warning': '',
        'is_valid': False
    }

    if current_price is None or stop_loss is None or current_price <= 0:
        return default_result

    try:
        # 计算风险金额
        max_risk_amount = account_size * max_risk_pct

        # 计算止损幅度
        stop_loss_pct = (current_price - stop_loss) / current_price

        # 每股风险
        risk_per_share = current_price - stop_loss

        if risk_per_share <= 0:
            default_result['risk_reward_warning'] = '⚠ 止损价高于当前价，无法计算仓位'
            return default_result

        # 计算建议买入股数（向下取整到100股）
        raw_shares = max_risk_amount / risk_per_share
        suggested_shares = int(raw_shares // 100) * 100  # 向下取整到100股

        # 流动性惩罚：仓位不超过 ADV 的 1%
        liquidity_warning = ''
        if adv and adv > 0 and not pd.isna(adv):
            max_shares_by_adv = int(adv * ADV_POSITION_LIMIT)
            max_shares_by_adv = (max_shares_by_adv // 100) * 100
            if max_shares_by_adv > 0 and suggested_shares > max_shares_by_adv:
                suggested_shares = max_shares_by_adv
                liquidity_warning = f"⚠ 流动性限制: 仓位上限 {max_shares_by_adv} 股 (ADV {adv:,.0f} 的 1%)"

        # 仓位市值
        position_value = suggested_shares * current_price

        # 占账户比例
        position_pct = (position_value / account_size) * 100 if account_size > 0 else 0

        # 波动率警告
        volatility_warning = ''
        if stop_loss_pct > 0.05:
            volatility_warning = f"⚠ 当前波动率过高，止损幅度 {stop_loss_pct*100:.1f}% > 5%，建议谨慎入场或降低仓位"

        # 盈亏比警告
        risk_reward_warning = ''
        if suggested_shares < 100:
            risk_reward_warning = f"⚠ 止损幅度过大({stop_loss_pct*100:.1f}%)，建议买入数量不足100股，盈亏比极差"
        elif suggested_shares == 0:
            risk_reward_warning = f"⚠ 止损幅度过大({stop_loss_pct*100:.1f}%)，无法建仓，盈亏比极差"

        return {
            'max_risk_amount': round(max_risk_amount, 2),
            'stop_loss': round(stop_loss, 2),
            'stop_loss_pct': round(stop_loss_pct, 4),
            'suggested_shares': suggested_shares,
            'position_value': round(position_value, 2),
            'position_pct': round(position_pct, 2),
            'volatility_warning': volatility_warning,
            'risk_reward_warning': risk_reward_warning,
            'liquidity_warning': liquidity_warning,
            'is_valid': suggested_shares >= 100
        }

    except Exception:
        return default_result


# ============ 精确买卖/止损/目标价 + 检查清单 ============

def calculate_price_targets(df: pd.DataFrame, resonance: dict, consensus: dict,
                           support: list, resistance: list, contrarian: dict = None,
                           account_size: float = DEFAULT_ACCOUNT_SIZE) -> dict:
    """计算精确的买卖价格和止损价

    买入价计算逻辑:
    - 最优先: 回调到看多Order Block上沿
    - 优先: 回调到支撑位附近
    - 次选: 回调到MA10/MA20均线
    - 备选: 当前价格 * 0.98

    止损价计算逻辑:
    - 优先: Chandelier Exit（吊灯止损）
    - 次选: 跌破关键支撑位
    - 备选: ATR止损 (close - 2*ATR)

    目标价计算逻辑:
    - 优先: 目标价来自机构共识
    - 次选: 上涨至前高/压力位
    - 备选: 上涨空间 = 止损幅度的2倍

    仓位计算:
    - 使用波动率平价模型: 建议买入股数 = MAX_RISK_PER_TRADE / (当前价格 - 止损价)
    """
    last = df.iloc[-1]
    current_price = last['close']

    default_result = {
        'buy_price': None,
        'stop_loss': None,
        'target_price': None,
        'risk_reward': None,
        'position_info': None,
        'buy_reason': '',
        'stop_reason': '',
        'target_reason': '',
        'chandelier_info': None
    }

    try:
        # ========== 买入价 ==========
        buy_price = None
        buy_reason = ''

        # 方案0 (最高优先级): Order Block zone
        if contrarian and contrarian.get('sr_institutional', {}).get('order_blocks'):
            bullish_obs = [ob for ob in contrarian['sr_institutional']['order_blocks']
                           if ob['type'] == 'bullish' and ob['price_high'] < current_price]
            if bullish_obs:
                nearest_ob = max(bullish_obs, key=lambda x: x['price_high'])
                buy_price = nearest_ob['price_high']
                buy_reason = f'回调至看多Order Block({nearest_ob["price_low"]:.2f}-{nearest_ob["price_high"]:.2f})'

        # 方案1: 支撑位附近
        if buy_price is None and support:
            nearest_support = min(support, key=lambda x: abs(x - current_price))
            if nearest_support < current_price:
                buy_price = nearest_support
                buy_reason = f'回调至支撑位{nearest_support}'
            elif nearest_support <= current_price * 1.02:
                buy_price = current_price
                buy_reason = '接近支撑位'

        # 方案2: MA均线
        if buy_price is None:
            ma10 = last.get('MA10')
            ma20 = last.get('MA20')

            if ma10 and ma10 < current_price:
                buy_price = ma10
                buy_reason = f'回调至MA10({ma10:.2f})'
            elif ma20 and ma20 < current_price:
                buy_price = ma20
                buy_reason = f'回调至MA20({ma20:.2f})'

        # 方案3: 当前价格98折
        if buy_price is None:
            buy_price = current_price * 0.98
            buy_reason = '2%回调'

        # ========== 止损价（优先使用 Chandelier Exit）==========
        stop_loss = None
        stop_reason = ''

        # 方案1: Chandelier Exit（吊灯止损）
        chandelier = calc_chandelier_exit(df)
        chandelier_info = chandelier

        if chandelier['long_stop'] is not None:
            stop_loss = chandelier['long_stop']
            stop_reason = f"Chandelier Exit({chandelier['atr_multiplier']}x ATR, 区间最高{chandelier['highest_high']:.2f})"

        # 方案2: 跌破支撑位下方（作为备选）
        if stop_loss is None and support:
            valid_supports = [s for s in support if s < buy_price]
            if valid_supports:
                stop_loss = min(valid_supports) * 0.98
                stop_reason = f'跌破支撑位({min(valid_supports):.2f})'

        # 方案3: ATR止损（最后备选）
        if stop_loss is None:
            atr = last.get('ATR')
            if pd.notna(atr):
                stop_loss = buy_price - 2 * atr
                stop_reason = f'ATR止损({2*atr:.2f})'

        # ========== 目标价 ==========
        target_price = None
        target_reason = ''

        # 方案1: 机构目标价
        if consensus and consensus.get('target_price'):
            target_price = consensus['target_price']
            target_reason = '机构目标价'

        # 方案2: 压力位
        if target_price is None and resistance:
            valid_resistances = [r for r in resistance if r > buy_price]
            if valid_resistances:
                target_price = min(valid_resistances)
                target_reason = f'第一压力位({target_price:.2f})'

        # 方案3: 前高
        if target_price is None:
            high_60 = df['high'].tail(60).max()
            if high_60 > current_price:
                target_price = high_60
                target_reason = f'60日最高({high_60:.2f})'

        # 方案4: 2倍止损空间
        if target_price is None and stop_loss:
            risk = buy_price - stop_loss
            target_price = buy_price + risk * 2
            target_reason = '2倍风险报酬'

        # ========== 风险报酬比 ==========
        risk_reward = None
        if stop_loss and buy_price:
            risk = buy_price - stop_loss
            if risk > 0 and target_price:
                reward = target_price - buy_price
                risk_reward = round(reward / risk, 1)

        # ========== 滑点复核 ==========
        risk_reward_after_slippage = None
        slippage_ev_warning = ''
        if buy_price and stop_loss and target_price:
            slippage_buy = buy_price * (1 + SLIPPAGE_PCT)
            slippage_stop = stop_loss * (1 - SLIPPAGE_PCT)
            slippage_risk = slippage_buy - slippage_stop
            slippage_reward = target_price - slippage_buy

            if slippage_risk > 0:
                risk_reward_after_slippage = round(slippage_reward / slippage_risk, 1)

            if risk_reward_after_slippage is not None and risk_reward_after_slippage < 2.0:
                slippage_ev_warning = f"⚠ 滑点后盈亏比 {risk_reward_after_slippage:.1f} < 2.0，EV 不足"

        # ========== 仓位计算（波动率平价模型）==========
        adv = last.get('ADV20')
        adv_val = adv if pd.notna(adv) else None

        position_info = None
        if stop_loss:
            position_info = calc_position_sizing(current_price, stop_loss, account_size, adv=adv_val)

        return {
            'buy_price': round(buy_price, 2) if buy_price else None,
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'target_price': round(target_price, 2) if target_price else None,
            'risk_reward': risk_reward,
            'risk_reward_after_slippage': risk_reward_after_slippage,
            'slippage_ev_warning': slippage_ev_warning,
            'position_info': position_info,
            'buy_reason': buy_reason,
            'stop_reason': stop_reason,
            'target_reason': target_reason,
            'chandelier_info': chandelier_info
        }

    except Exception as e:
        return default_result


def generate_checklist(df: pd.DataFrame, resonance: dict, signals: dict,
                      indicators: dict, contrarian: dict = None) -> list:
    """生成12项买入检查清单

    Returns:
        [(检查项, 是否通过, 原因), ...]
    """
    last = df.iloc[-1]
    results = []

    # 1. 长线均线趋势（MA50/MA200）
    ma50 = last.get('MA50')
    ma200 = last.get('MA200')
    close = last.get('close')

    if pd.notna(ma50) and pd.notna(ma200) and pd.notna(close):
        is_bullish = close > ma50 > ma200
        if is_bullish:
            reason = f"价格{close:.1f}>MA50{ma50:.1f}>MA200{ma200:.1f}"
        else:
            reason = f"未满足长线多头排列"
        results.append(("长线均线多头排列", is_bullish, reason))
    else:
        results.append(("长线均线多头排列", False, "数据不足"))

    # 2. 乖离率（基于MA50）
    if pd.notna(ma50):
        bias = (last['close'] - ma50) / ma50 * 100
        is_ok = abs(bias) < 15  # 长线放宽至15%
        reason = f"乖离率{bias:.1f}%" if is_ok else f"乖离率过高({bias:.1f}%)"
        results.append(("乖离率合理(<15%)", is_ok, reason))
    else:
        results.append(("乖离率合理(<15%)", False, "MA50数据不足"))

    # 3. RSI未超买
    rsi = last.get('RSI')
    if pd.notna(rsi):
        is_ok = rsi < 75
        reason = f"RSI={rsi:.1f}" if is_ok else f"RSI超买({rsi:.1f})"
        results.append(("RSI未超买(<75)", is_ok, reason))
    else:
        results.append(("RSI未超买(<75)", False, "RSI数据不足"))

    # 4. KDJ未超买
    # KDJ 已移除（长线投资不需要）

    # 5. MACD多头运行
    macd = last.get('MACD')
    macd_signal = last.get('MACD_signal')
    if pd.notna(macd) and pd.notna(macd_signal):
        is_bullish = macd > macd_signal
        reason = "MACD多头运行" if is_bullish else "MACD空头运行"
        results.append(("MACD多头运行", is_bullish, reason))
    else:
        results.append(("MACD多头运行", False, "MACD数据不足"))

    # 6. 布林带未突破上轨
    bb_upper = last.get('BB_upper')
    if pd.notna(bb_upper):
        is_ok = last['close'] < bb_upper * 1.02
        reason = "布林带轨道内" if is_ok else "接近上轨"
        results.append(("布林带未突破上轨", is_ok, reason))
    else:
        results.append(("布林带未突破上轨", False, "布林带数据不足"))

    # 7. ADX趋势向上
    adx = last.get('ADX')
    adx_pdi = last.get('ADX_PDI')
    adx_ndi = last.get('ADX_NDI')
    if pd.notna(adx) and pd.notna(adx_pdi) and pd.notna(adx_ndi):
        is_trending = adx > 20 and adx_pdi > adx_ndi
        reason = f"ADX={adx:.1f}多头趋势" if is_trending else f"ADX={adx:.1f}趋势不明"
        results.append(("ADX趋势向上(>20)", is_trending, reason))
    else:
        results.append(("ADX趋势向上(>20)", False, "ADX数据不足"))

    # 8. 成交量配合
    if len(df) >= 5:
        avg_vol = df['volume'].tail(5).mean()
        current_vol = df['volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        is_ok = vol_ratio > 0.8
        reason = f"量比{vol_ratio:.2f}" if is_ok else f"量比过低({vol_ratio:.2f})"
        results.append(("成交量配合(量比>0.8)", is_ok, reason))
    else:
        results.append(("成交量配合(量比>0.8)", False, "数据不足"))

    # 9. 价格在支撑位上方
    support, _ = find_support_resistance(df)
    if support:
        nearest = min(support)
        is_above = last['close'] > nearest * 0.97
        reason = f"距支撑位{((last['close']-nearest)/nearest*100):.1f}%"
        results.append(("价格在支撑位上方", is_above, reason))
    else:
        results.append(("价格在支撑位上方", True, "无明显支撑"))

    # 10. 技术面共振
    res = resonance.get('resonance', 'neutral')
    is_ok = res in ['strong_buy', 'buy']
    reason = f"技术面{res}" if is_ok else "技术面偏弱"
    results.append(("技术面共振偏多", is_ok, reason))

    # 11. Z-Score未超买
    if contrarian and contrarian.get('zscore', {}).get('sufficient_data'):
        zs = contrarian['zscore']['zscore']
        is_ok = zs < 2.0
        reason = f"Z-Score={zs:.2f}" if is_ok else f"Z-Score超买({zs:.2f})"
    else:
        is_ok = True
        reason = "数据不足，跳过"
    results.append(("Z-Score未超买(<2.0)", is_ok, reason))

    # 12. 无看空背离
    if contrarian and contrarian.get('divergence'):
        bearish_count = contrarian['divergence'].get('bearish_div_count', 0)
        is_ok = bearish_count == 0
        reason = "无看空背离" if is_ok else f"{bearish_count}重看空背离"
    else:
        is_ok = True
        reason = "数据不足，跳过"
    results.append(("无看空背离", is_ok, reason))

    return results


# ============ SMC + EV 严格决策树 ============

def generate_smc_decision_tree(df: pd.DataFrame, price_targets: dict,
                                contrarian: dict, distribution: dict,
                                chandelier: dict) -> dict:
    """基于 SMC（聪明钱概念）+ EV（期望值）的严格决策树

    决策逻辑:
    1. SELL: 满足任意 1 个卖出条件（优先级最高）
    2. BUY: 必须同时满足 4 个买入条件
    3. HOLD: 其他所有情况

    Args:
        df: OHLCV DataFrame (包含 ChoCh/BoS 列)
        price_targets: 来自 calculate_price_targets()
        contrarian: 来自 calc_contrarian_signals()
        distribution: 来自 calc_distribution_patterns()
        chandelier: 来自 calc_chandelier_exit()

    Returns:
        {
            'decision': 'BUY' | 'SELL' | 'HOLD_WITH_POSITION' | 'HOLD_EMPTY',
            'decision_text': str,
            'confidence': 'HIGH' | 'MEDIUM' | 'LOW',
            'buy_conditions': dict,
            'sell_conditions': dict,
            'buy_details': dict,
            'sell_details': dict,
            'reasoning': list,
            'warnings': list
        }
    """
    last = df.iloc[-1]
    current_price = last['close']
    reasoning = []
    warnings = []

    # ========== 买入条件检查 ==========

    # 多时间框架共振检查（周线+月线三重确认）
    weekly_confirm = df.attrs.get('weekly_confirmation', {})
    monthly_confirm = df.attrs.get('monthly_confirmation', {})
    multi_timeframe_resonance = False

    if weekly_confirm and monthly_confirm:
        current_bos = last.get('bos_signal', 0)
        current_choch = last.get('choch_signal', 0)
        weekly_bos = weekly_confirm.get('weekly_bos_signal', 0)
        weekly_choch = weekly_confirm.get('weekly_choch_signal', 0)
        weekly_trend = weekly_confirm.get('weekly_trend', 'sideways')
        monthly_trend = monthly_confirm.get('monthly_trend', 'sideways')

        # 三重共振条件：当前周期有信号 + 周线趋势同向 + 月线趋势同向
        if current_bos == 1 or current_choch == 1:
            if weekly_trend == 'uptrend' and monthly_trend == 'uptrend':
                multi_timeframe_resonance = True
        # 双重共振（降级）：当前周期 + 周线同向
        elif (current_bos == 1 and weekly_bos == 1) or \
             (current_choch == 1 and weekly_choch == 1):
            if monthly_trend != 'downtrend':
                multi_timeframe_resonance = True

    # 条件 1: ChoCh 确认（下降趋势中出现 Higher High）+ 多时间框架共振
    choch_signal = last.get('choch_signal', 0)
    choch_strength = last.get('choch_strength', 0)
    choch_confirmed = (choch_signal == 1 and choch_strength > 50 and multi_timeframe_resonance)

    choch_detail = ''
    if choch_confirmed:
        choch_detail = f"看多 ChoCh 确认 (强度: {choch_strength:.1f}, 多时间框架共振)"
    elif choch_signal == 1 and choch_strength > 50 and not multi_timeframe_resonance:
        choch_detail = f"ChoCh 信号强但多时间框架未共振 (强度: {choch_strength:.1f})"
    elif choch_signal == 1:
        choch_detail = f"ChoCh 信号弱 (强度: {choch_strength:.1f} < 50)"
    elif choch_signal == -1:
        choch_detail = f"看空 ChoCh (强度: {choch_strength:.1f})"
    else:
        choch_detail = "无明确趋势特性改变"

    # 条件 2: 回踩确认（价格测试 HVN 或 FVG）
    hvn_zones = []
    if contrarian:
        hvn_zones = contrarian.get('volume_exhaustion', {}).get('hvn_zones', [])

    fvg_zones = []
    if contrarian:
        sr_inst = contrarian.get('sr_institutional', {})
        fvg_zones = sr_inst.get('fvg_zones', [])

    bullish_fvgs = [f for f in fvg_zones if f['type'] == 'bullish' and not f.get('filled', True)]

    price_in_hvn = any(lo <= current_price <= hi for lo, hi, _ in hvn_zones)
    price_in_fvg = any(f['bottom'] <= current_price <= f['top'] for f in bullish_fvgs)
    price_testing_zone = price_in_hvn or price_in_fvg

    zone_detail = ''
    if price_in_hvn:
        matched = [(lo, hi) for lo, hi, _ in hvn_zones if lo <= current_price <= hi]
        if matched:
            zone_detail = f"价格在 HVN 区域 ({matched[0][0]:.2f}-{matched[0][1]:.2f})"
    elif price_in_fvg:
        matched = [f for f in bullish_fvgs if f['bottom'] <= current_price <= f['top']]
        if matched:
            zone_detail = f"价格在看多 FVG ({matched[0]['bottom']:.2f}-{matched[0]['top']:.2f})"
    else:
        zone_detail = "价格未在 HVN 或 FVG 区域内"

    # 条件 3: R-Multiple > 2.0
    risk_reward = price_targets.get('risk_reward') if price_targets else None
    risk_reward_valid = (risk_reward is not None and risk_reward >= 2.0)

    rr_detail = ''
    if risk_reward is not None:
        rr_detail = f"R-Multiple = {risk_reward:.1f}"
        if not risk_reward_valid:
            rr_detail += " (< 2.0，强制观望)"
    else:
        rr_detail = "无法计算盈亏比"

    # 条件 4: Z-Score 验证
    zscore_data = contrarian.get('zscore', {}) if contrarian else {}
    zscore = zscore_data.get('zscore', 0)
    sufficient_data = zscore_data.get('sufficient_data', False)
    zscore_valid = (not sufficient_data) or (-2 <= zscore <= 2)

    zscore_detail = ''
    if not sufficient_data:
        zscore_detail = "数据不足，跳过验证"
    elif zscore_valid:
        zscore_detail = f"Z = {zscore:.2f} (正常区间)"
    else:
        zscore_detail = f"Z = {zscore:.2f} (极端区域，不宜操作)"

    buy_conditions = {
        'choch_confirmed': choch_confirmed,
        'price_testing_zone': price_testing_zone,
        'risk_reward_valid': risk_reward_valid,
        'zscore_valid': zscore_valid
    }

    buy_details = {
        'choch': choch_detail,
        'zone': zone_detail,
        'risk_reward': rr_detail,
        'zscore': zscore_detail
    }

    # ========== 卖出条件检查 ==========

    # 条件 1: 跌破 Chandelier Exit 止损
    long_stop = chandelier.get('long_stop') if chandelier else None
    stop_loss_triggered = (long_stop is not None and current_price < long_stop)

    stop_detail = ''
    if long_stop is not None:
        if stop_loss_triggered:
            stop_detail = f"当前价 {current_price:.2f} < 止损价 {long_stop:.2f}"
        else:
            stop_detail = f"当前价 {current_price:.2f} > 止损价 {long_stop:.2f}"
    else:
        stop_detail = "止损价未计算"

    # 条件 2: 派发模式（放量滞涨）
    dist_score = distribution.get('distribution_score', 0) if distribution else 0
    dist_patterns = distribution.get('patterns', []) if distribution else []
    distribution_detected = (dist_score >= 70 or 'volume_climax' in dist_patterns)

    dist_detail = ''
    if distribution_detected:
        triggered = []
        if 'volume_climax' in dist_patterns:
            triggered.append('放量滞涨')
        if dist_score >= 70:
            triggered.append(f'派发评分 {dist_score}')
        dist_detail = f"派发信号: {', '.join(triggered)}"
    else:
        dist_detail = f"派发评分 {dist_score} 分"

    sell_conditions = {
        'stop_loss_triggered': stop_loss_triggered,
        'distribution_detected': distribution_detected
    }

    sell_details = {
        'stop_loss': stop_detail,
        'distribution': dist_detail
    }

    # ========== 决策逻辑 ==========
    # SELL 优先于 BUY（止损无条件执行）
    if any(sell_conditions.values()):
        decision = 'SELL'
        decision_text = '清仓卖出'
        confidence = 'HIGH'

        if stop_loss_triggered:
            reasoning.append('价格跌破 Chandelier Exit 止损线，趋势破坏')
            warnings.append('止损已触发，必须立即离场')
        if distribution_detected:
            reasoning.append('检测到高位派发模式，机构正在退出')
            if 'volume_climax' in dist_patterns:
                warnings.append('放量滞涨: 成交量放大但价格不涨，典型派发特征')

    elif all(buy_conditions.values()):
        decision = 'BUY'
        decision_text = '买入'
        confidence = 'HIGH'

        reasoning.append(f'结构确认: {choch_detail}')
        reasoning.append(f'回踩确认: {zone_detail}')
        reasoning.append(f'盈亏比优秀: {rr_detail}')

        # 附加风险提示
        if price_targets and price_targets.get('position_info', {}).get('volatility_warning'):
            warnings.append(price_targets['position_info']['volatility_warning'])

    else:
        # HOLD — 区分持仓观望和空仓观望
        if long_stop and current_price > long_stop:
            decision = 'HOLD_WITH_POSITION'
            decision_text = '持仓观望'
        else:
            decision = 'HOLD_EMPTY'
            decision_text = '空仓观望'

        # 置信度取决于缺失条件数量
        missing = sum(1 for v in buy_conditions.values() if not v)
        confidence = 'LOW' if missing >= 3 else 'MEDIUM'

        # 说明缺失的条件
        if not choch_confirmed:
            reasoning.append('缺少结构确认: ChoCh 信号未触发')
        if not price_testing_zone:
            reasoning.append('缺少回踩确认: 价格未在 HVN/FVG 区域')
        if not risk_reward_valid:
            reasoning.append(f'盈亏比不足: {rr_detail}')
        if not zscore_valid:
            reasoning.append(f'Z-Score 极端: {zscore_detail}')

    return {
        'decision': decision,
        'decision_text': decision_text,
        'confidence': confidence,
        'buy_conditions': buy_conditions,
        'sell_conditions': sell_conditions,
        'buy_details': buy_details,
        'sell_details': sell_details,
        'reasoning': reasoning,
        'warnings': warnings
    }


def generate_decision_dashboard(df: pd.DataFrame, price_targets: dict,
                                contrarian: dict, distribution: dict,
                                chandelier: dict, circuit_breaker: dict = None,
                                long_term_context: dict = None) -> dict:
    """决策仪表盘 (基于 SMC + EV 严格决策树)

    否决优先级链：
    1. 熔断（宏观风险/跳空证伪）→ 最高优先级
    2. 长线趋势熔断（50W < 200W）→ 严格禁止买入
    3. 价值陷阱否决（低估值+恶化基本面）→ 否决买入
    4. 宏观熊市降级（大盘结构性熊市）→ 买入降级为观望
    5. SMC 决策树 → 正常决策流程

    Returns:
        {
            'decision': str,
            'decision_text': str,
            'confidence': str,
            'buy_conditions': dict,
            'sell_conditions': dict,
            'buy_details': dict,
            'sell_details': dict,
            'reasoning': list,
            'warnings': list,
            'action': str,
            'verdict': str,
            'core_conclusion': str,
            'circuit_breaker': dict,
            'long_term_bear_mode': bool
        }
    """
    # 熔断检查（最高优先级，覆写一切）
    if circuit_breaker and circuit_breaker.get('any_triggered'):
        override = circuit_breaker['override_decision']
        override_text = circuit_breaker['override_text']
        override_action = circuit_breaker['override_action']

        reasoning = []
        warnings = []

        if circuit_breaker.get('macro_triggered'):
            macro = circuit_breaker.get('macro_data', {})
            bm = macro.get('benchmark', {})
            vix = macro.get('vix', {})
            if bm.get('below_ma200'):
                reasoning.append(f"基准指数 {bm.get('name', '')} 跌破 MA200 ({bm.get('price', 0):.2f} < {bm.get('ma200', 0):.2f})")
            if vix.get('spike'):
                reasoning.append(f"VIX 日内飙升 {vix.get('change_pct', 0):.1f}% (> 15%)")
            warnings.append('宏观风险极高，全局禁止任何买入操作')

        if circuit_breaker.get('gap_triggered'):
            gap = circuit_breaker.get('gap_data', {})
            reasoning.append(f"向下跳空 {gap.get('gap_size', 0):.2f} ({gap.get('gap_atr_ratio', 0):.1f}x ATR) 穿越 HVN")
            warnings.append('逻辑证伪，切勿接飞刀，无视任何超卖反弹指标')

        return {
            'decision': override,
            'decision_text': override_text,
            'confidence': 'HIGH',
            'buy_conditions': {},
            'sell_conditions': {},
            'buy_details': {},
            'sell_details': {},
            'reasoning': reasoning,
            'warnings': warnings,
            'action': override_action,
            'verdict': override_text,
            'core_conclusion': f"⚠ 熔断: {override_text}",
            'circuit_breaker': circuit_breaker
        }

    # ===== 长线趋势熔断（优先级仅次于熔断）=====
    ltc = long_term_context or {}
    weekly_conf = ltc.get('weekly_confirmation', {})
    monthly_conf = ltc.get('monthly_confirmation', {})
    long_term_health = weekly_conf.get('long_term_trend_health', 'unknown')

    if long_term_health == 'broken':
        reasoning = []
        warnings = []
        w50 = weekly_conf.get('weekly_50ma', 0)
        w200 = weekly_conf.get('weekly_200ma', 0)

        if w50 > 0 and w200 > 0:
            deviation = (w50 - w200) / w200 * 100
            reasoning.append(f"50周均线({w50:.2f}) < 200周均线({w200:.2f})，偏离 {deviation:.1f}%")

        override_text = '致命警告：长线结构性熊市，禁止买入'
        monthly_also_broken = not monthly_conf.get('monthly_golden_cross', True)
        if monthly_also_broken and monthly_conf.get('monthly_ma12', 0) > 0:
            override_text = '致命警告：月线+周线双破位，严禁抄底'
            reasoning.append(f"月线MA10({monthly_conf.get('monthly_ma12', 0):.2f}) < MA20({monthly_conf.get('monthly_ma24', 0):.2f})")

        warnings.append('长线结构性熊市，全局禁止任何买入操作，等待50周均线重新站上200周均线')

        return {
            'decision': 'LONG_TERM_BEAR',
            'decision_text': override_text,
            'confidence': 'HIGH',
            'buy_conditions': {},
            'sell_conditions': {},
            'buy_details': {},
            'sell_details': {},
            'reasoning': reasoning,
            'warnings': warnings,
            'action': '禁止买入',
            'verdict': override_text,
            'core_conclusion': f"⚠ 长线熔断: {override_text}",
            'long_term_bear_mode': True
        }

    # ===== 价值陷阱否决 =====
    fundamentals = ltc.get('fundamentals', {})
    value_trap = fundamentals.get('value_trap', {})
    if value_trap.get('is_trap') and value_trap.get('veto_buy'):
        return {
            'decision': 'VALUE_TRAP',
            'decision_text': value_trap.get('warning_text', '价值陷阱警告：盈利能力恶化，否决买入'),
            'confidence': 'HIGH',
            'buy_conditions': {},
            'sell_conditions': {},
            'buy_details': {},
            'sell_details': {},
            'reasoning': value_trap.get('reasons', ['低估值 + 基本面恶化 = 价值陷阱']),
            'warnings': [value_trap.get('warning_text', '价值陷阱警告')],
            'action': '否决买入',
            'verdict': '价值陷阱',
            'core_conclusion': f"⚠ 价值陷阱: {value_trap.get('warning_text', '')}",
            'long_term_bear_mode': False
        }

    smc = generate_smc_decision_tree(df, price_targets, contrarian, distribution, chandelier)

    # 映射到兼容接口
    decision_map = {
        'BUY': {'action': '买入', 'verdict': '买入'},
        'SELL': {'action': '清仓', 'verdict': '清仓'},
        'HOLD_WITH_POSITION': {'action': '持仓观望', 'verdict': '观望'},
        'HOLD_EMPTY': {'action': '空仓观望', 'verdict': '观望'}
    }
    mapped = decision_map.get(smc['decision'], {'action': '观望', 'verdict': '观望'})

    # 生成一句话结论
    buy_ok = sum(1 for v in smc['buy_conditions'].values() if v)
    buy_total = len(smc['buy_conditions'])
    core_conclusion = f"SMC决策: {smc['decision_text']} (买入条件 {buy_ok}/{buy_total})"
    if smc['reasoning']:
        core_conclusion += f" — {smc['reasoning'][0]}"

    return {
        **smc,
        'action': mapped['action'],
        'verdict': mapped['verdict'],
        'core_conclusion': core_conclusion
    }


def generate_position_strategy(df: pd.DataFrame, current_price: float, entry_price: float,
                                dashboard: dict, chandelier: dict, contrarian: dict,
                                distribution: dict, price_targets: dict,
                                dd_analysis: dict = None) -> dict:
    """持仓应对策略 — 根据浮盈/浮亏自动生成操作建议。

    返回 dict，包含 'mode'('loss'/'profit'), 以及各子模块结果。
    """
    if entry_price is None or entry_price <= 0 or current_price is None or current_price <= 0:
        return {'available': False}

    pnl_pct = (current_price - entry_price) / entry_price * 100
    decision = dashboard.get('decision', 'HOLD_EMPTY')
    circuit_triggered = dashboard.get('circuit_breaker_triggered', False)

    # ── 趋势状态判断 ──
    last = df.iloc[-1]
    ma50 = last.get('MA50') if 'MA50' in df.columns else None
    ma200 = last.get('MA200') if 'MA200' in df.columns else None
    choch_signal = last.get('choch_signal', 0)

    long_stop = chandelier.get('long_stop')
    stop_breached = (long_stop is not None and current_price < long_stop)

    # 死叉判断
    death_cross = False
    if ma50 is not None and ma200 is not None and not pd.isna(ma50) and not pd.isna(ma200):
        death_cross = ma50 < ma200

    # 趋势状态: 'intact' / 'damaged' / 'broken'
    broken_count = sum([
        stop_breached,
        choch_signal == -1,
        death_cross,
        decision in ('SELL', 'LONG_TERM_BEAR', 'VALUE_TRAP'),
    ])
    if broken_count >= 3 or circuit_triggered:
        trend_status = 'broken'
        trend_text = '趋势破坏'
    elif broken_count >= 1:
        trend_status = 'damaged'
        trend_text = '趋势受损'
    else:
        trend_status = 'intact'
        trend_text = '趋势完好'

    # ── 止损距离 ──
    if long_stop is not None:
        stop_dist_pct = (current_price - long_stop) / current_price * 100
        if stop_breached:
            stop_urgency = 'breached'
        elif stop_dist_pct < 2:
            stop_urgency = 'critical'
        elif stop_dist_pct < 5:
            stop_urgency = 'warning'
        else:
            stop_urgency = 'safe'
    else:
        stop_dist_pct = None
        stop_urgency = 'unknown'

    if pnl_pct < 0:
        return _calc_loss_strategy(
            pnl_pct, trend_status, trend_text, stop_urgency, stop_dist_pct,
            long_stop, current_price, entry_price, decision, circuit_triggered,
            contrarian, distribution, price_targets, df, dd_analysis
        )
    else:
        return _calc_profit_strategy(
            pnl_pct, trend_status, trend_text, stop_urgency, stop_dist_pct,
            long_stop, current_price, entry_price, decision, circuit_triggered,
            contrarian, distribution, price_targets, df
        )


def _calc_loss_strategy(pnl_pct, trend_status, trend_text, stop_urgency, stop_dist_pct,
                        long_stop, current_price, entry_price, decision, circuit_triggered,
                        contrarian, distribution, price_targets, df, dd_analysis):
    """亏损应对策略计算"""
    abs_loss = abs(pnl_pct)

    # 亏损分级（参照历史最大回撤动态修正）
    max_dd = 0
    if dd_analysis and dd_analysis.get('available'):
        max_dd = abs(dd_analysis.get('max_drawdown', 0))

    # 动态阈值：历史最大回撤的 30%/60%/90%
    if max_dd > 0:
        t_shallow = max(5, max_dd * 0.3)
        t_moderate = max(15, max_dd * 0.6)
        t_deep = max(30, max_dd * 0.9)
    else:
        t_shallow, t_moderate, t_deep = 5, 15, 30

    if abs_loss < t_shallow:
        loss_level = 'shallow'
        loss_text = f'浅度亏损 (<{t_shallow:.0f}%)'
    elif abs_loss < t_moderate:
        loss_level = 'moderate'
        loss_text = f'中度亏损 ({t_shallow:.0f}%-{t_moderate:.0f}%)'
    elif abs_loss < t_deep:
        loss_level = 'deep'
        loss_text = f'深度亏损 ({t_moderate:.0f}%-{t_deep:.0f}%)'
    else:
        loss_level = 'extreme'
        loss_text = f'极度亏损 (>{t_deep:.0f}%)'

    # 减仓建议
    if circuit_triggered:
        reduce_pct = 100
        reduce_reason = '熔断触发，立即清仓'
    elif decision in ('SELL', 'LONG_TERM_BEAR', 'VALUE_TRAP'):
        reduce_pct = 100
        reduce_reason = 'SMC决策为卖出，建议清仓'
    elif trend_status == 'broken' and loss_level in ('deep', 'extreme'):
        reduce_pct = 50
        reduce_reason = '趋势破坏+深度亏损，建议减仓50%'
    elif trend_status == 'damaged' and loss_level == 'moderate':
        reduce_pct = 30
        reduce_reason = '趋势受损+中度亏损，建议减仓30%'
    else:
        reduce_pct = 0
        reduce_reason = ''

    # 加仓评估
    contrarian_score = contrarian.get('composite_score', 0) if contrarian else 0
    zscore_val = None
    try:
        zscore_data = contrarian.get('zscore', {}) if contrarian else {}
        zscore_val = zscore_data.get('zscore')
    except Exception:
        pass

    add_conditions = []
    add_ok = True

    if loss_level == 'extreme':
        add_ok = False
        add_conditions.append('✗ 极度亏损禁止加仓')
    else:
        if trend_status == 'broken':
            add_ok = False
            add_conditions.append('✗ 趋势已破坏')
        else:
            add_conditions.append('✓ 趋势未破坏')

        if contrarian_score >= 65:
            add_conditions.append(f'✓ 左侧评分 {contrarian_score} ≥ 65')
        else:
            add_ok = False
            add_conditions.append(f'✗ 左侧评分 {contrarian_score} < 65')

        if zscore_val is not None:
            if zscore_val < -1.5:
                add_conditions.append(f'✓ Z-Score {zscore_val:.2f} < -1.5')
            else:
                add_ok = False
                add_conditions.append(f'✗ Z-Score {zscore_val:.2f} ≥ -1.5（未超卖）')
        else:
            add_conditions.append('— Z-Score 数据不足，跳过')

    # 加仓价位和股数
    add_price = None
    add_shares = None
    if add_ok and long_stop:
        add_price = price_targets.get('buy_price', current_price)
        try:
            sizing = calc_position_sizing(current_price, long_stop)
            add_shares = sizing.get('suggested_shares', 0)
        except Exception:
            pass

    # 心理建设
    bias_map = {
        'shallow': ('锚定效应', '你可能在用买入价锚定判断，而非客观评估当前价值。', '若趋势完好，浅度回调是正常波动。'),
        'moderate': ('处置效应', '亏损时倾向于持有等待回本，而非理性评估。', '问自己：若今天才看到这只股票，你还会买入吗？'),
        'deep': ('沉没成本谬误', '已亏损的钱不应影响未来决策。', '深度亏损时，每一分钱都是新的投资决策。'),
        'extreme': ('鸵鸟心态', '极度亏损时容易回避现实，拒绝止损。', '最坏情景：若趋势持续恶化，亏损可能进一步扩大。'),
    }
    bias_name, bias_desc, rational_action = bias_map.get(loss_level, bias_map['moderate'])

    return {
        'available': True,
        'mode': 'loss',
        'pnl_pct': round(pnl_pct, 2),
        'loss_level': loss_level,
        'loss_text': loss_text,
        'trend_status': trend_status,
        'trend_text': trend_text,
        'long_stop': long_stop,
        'stop_dist_pct': round(stop_dist_pct, 2) if stop_dist_pct is not None else None,
        'stop_urgency': stop_urgency,
        'add_ok': add_ok,
        'add_conditions': add_conditions,
        'add_price': add_price,
        'add_shares': add_shares,
        'reduce_pct': reduce_pct,
        'reduce_reason': reduce_reason,
        'bias_name': bias_name,
        'bias_desc': bias_desc,
        'rational_action': rational_action,
    }


def _calc_profit_strategy(pnl_pct, trend_status, trend_text, stop_urgency, stop_dist_pct,
                          long_stop, current_price, entry_price, decision, circuit_triggered,
                          contrarian, distribution, price_targets, df):
    """盈利应对策略计算"""
    # 盈利分级
    if pnl_pct < 10:
        profit_level = 'small'
        profit_text = '小幅盈利 (<10%)'
    elif pnl_pct < 25:
        profit_level = 'moderate'
        profit_text = '中等盈利 (10%-25%)'
    elif pnl_pct < 50:
        profit_level = 'large'
        profit_text = '大幅盈利 (25%-50%)'
    else:
        profit_level = 'excessive'
        profit_text = '超额盈利 (>50%)'

    # 分批止盈目标
    target_price = price_targets.get('target_price')
    take_profit_levels = []
    if target_price and target_price > entry_price:
        progress = (current_price - entry_price) / (target_price - entry_price) * 100
        tp1 = entry_price + (target_price - entry_price) * 0.5
        tp2 = entry_price + (target_price - entry_price) * 0.8
        take_profit_levels = [
            {'price': round(tp1, 2), 'progress': 50, 'action': '减仓1/3', 'reached': current_price >= tp1},
            {'price': round(tp2, 2), 'progress': 80, 'action': '再减1/3', 'reached': current_price >= tp2},
            {'price': round(target_price, 2), 'progress': 100, 'action': '清仓', 'reached': current_price >= target_price},
        ]
    else:
        progress = None

    # VWAP偏离检查
    last = df.iloc[-1]
    vwap = last.get('VWAP') if 'VWAP' in df.columns else None
    vwap_deviation = None
    vwap_alert = False
    if vwap and not pd.isna(vwap) and vwap > 0:
        vwap_deviation = (current_price - vwap) / vwap * 100
        # 估算sigma（用ATR/VWAP近似）
        atr = last.get('ATR')
        if atr and not pd.isna(atr):
            vwap_sigma = atr / vwap * 100
            vwap_sigmas = vwap_deviation / vwap_sigma if vwap_sigma > 0 else 0
            vwap_alert = vwap_sigmas > 3
        else:
            vwap_sigmas = None
    else:
        vwap_sigmas = None

    # 移动止损锁定利润
    locked_profit_pct = None
    if long_stop and long_stop > entry_price:
        locked_profit_pct = round((long_stop - entry_price) / entry_price * 100, 2)

    # 派发预警
    dist_score = distribution.get('distribution_score', 0) if distribution else 0
    if dist_score >= 70:
        dist_alert = 'critical'
        dist_text = '强烈派发信号'
    elif dist_score >= 40:
        dist_alert = 'warning'
        dist_text = '出现派发迹象'
    elif dist_score >= 20:
        dist_alert = 'watch'
        dist_text = '轻微派发迹象'
    else:
        dist_alert = 'none'
        dist_text = '无派发信号'

    # 加仓禁区检查
    rsi = last.get('RSI') if 'RSI' in df.columns else None
    zscore_val = None
    try:
        zscore_data = contrarian.get('zscore', {}) if contrarian else {}
        zscore_val = zscore_data.get('zscore')
    except Exception:
        pass

    add_ban_reasons = []
    if pnl_pct > 20:
        add_ban_reasons.append(f'浮盈 {pnl_pct:.1f}% > 20%')
    if vwap_sigmas is not None and vwap_sigmas > 2:
        add_ban_reasons.append(f'VWAP偏离 {vwap_sigmas:.1f}σ > 2σ')
    if dist_score > 40:
        add_ban_reasons.append(f'派发评分 {dist_score} > 40')
    if zscore_val is not None and zscore_val > 2:
        add_ban_reasons.append(f'Z-Score {zscore_val:.2f} > 2（超买）')
    if rsi is not None and not pd.isna(rsi) and rsi > 70:
        add_ban_reasons.append(f'RSI {rsi:.1f} > 70（超买）')

    add_banned = len(add_ban_reasons) >= 2

    return {
        'available': True,
        'mode': 'profit',
        'pnl_pct': round(pnl_pct, 2),
        'profit_level': profit_level,
        'profit_text': profit_text,
        'trend_status': trend_status,
        'trend_text': trend_text,
        'long_stop': long_stop,
        'stop_dist_pct': round(stop_dist_pct, 2) if stop_dist_pct is not None else None,
        'stop_urgency': stop_urgency,
        'locked_profit_pct': locked_profit_pct,
        'take_profit_levels': take_profit_levels,
        'progress': round(progress, 1) if progress is not None else None,
        'vwap_deviation': round(vwap_deviation, 2) if vwap_deviation is not None else None,
        'vwap_sigmas': round(vwap_sigmas, 2) if vwap_sigmas is not None else None,
        'vwap_alert': vwap_alert,
        'dist_score': dist_score,
        'dist_alert': dist_alert,
        'dist_text': dist_text,
        'add_banned': add_banned,
        'add_ban_reasons': add_ban_reasons,
    }


# ============ 技术指标计算函数 ============

def calc_sma(series: pd.Series, length: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=length).mean()


def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=length, adjust=False).mean()


def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """RSI相对强弱指标 - 使用Wilder平滑法

    Args:
        series: 价格序列
        length: 计算周期，默认14

    Returns:
        RSI值序列
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    # 使用EMA（Wilder方法）- alpha = 1/length
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    # 防止除零错误
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    # 处理无穷大和NaN
    rsi = rsi.replace([np.inf, -np.inf], np.nan)

    return rsi


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD指标"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger_bands(series: pd.Series, length: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """布林带"""
    middle = calc_sma(series, length)
    std = series.rolling(window=length).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标（纯pandas/numpy实现）

    带有错误处理，单个指标计算失败不会影响其他指标
    """
    df = df.copy()

    # 移动平均线（含长线均线）
    try:
        df['MA5'] = calc_sma(df['close'], 5)
        df['MA10'] = calc_sma(df['close'], 10)
        df['MA20'] = calc_sma(df['close'], 20)
        df['MA50'] = calc_sma(df['close'], 50)    # 50周均线（长线关键均线）
        df['MA200'] = calc_sma(df['close'], 200)   # 200周均线（长线关键均线）
    except Exception:
        df['MA5'] = np.nan
        df['MA10'] = np.nan
        df['MA20'] = np.nan
        df['MA50'] = np.nan
        df['MA200'] = np.nan

    # RSI
    try:
        df['RSI'] = calc_rsi(df['close'], 14)
    except Exception:
        df['RSI'] = np.nan

    # MACD
    try:
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calc_macd(df['close'])
    except Exception:
        df['MACD'] = np.nan
        df['MACD_signal'] = np.nan
        df['MACD_hist'] = np.nan

    # 布林带
    try:
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = calc_bollinger_bands(df['close'])
    except Exception:
        df['BB_upper'] = np.nan
        df['BB_middle'] = np.nan
        df['BB_lower'] = np.nan

    # ========== 新增指标（纯pandas实现） ==========

    # KDJ指标已移除（长线投资不需要）

    # ADX指标（平均趋向指数）
    try:
        df = calc_adx(df)
    except Exception:
        df['ADX'] = np.nan
        df['ADX_PDI'] = np.nan
        df['ADX_NDI'] = np.nan

    # ATR指标（平均真实波幅）
    try:
        df['ATR'] = calc_atr(df)
    except Exception:
        df['ATR'] = np.nan

    # OBV指标（能量潮）
    try:
        df['OBV'] = calc_obv(df)
        df['OBV_MA'] = calc_sma(df['OBV'], 20)
    except Exception:
        df['OBV'] = np.nan
        df['OBV_MA'] = np.nan

    # CCI指标（顺势指标）
    try:
        df['CCI'] = calc_cci(df)
    except Exception:
        df['CCI'] = np.nan

    # SuperTrend指标
    try:
        df = calc_supertrend(df)
    except Exception:
        df['SuperTrend'] = np.nan
        df['SuperTrend_dir'] = np.nan

    # PSAR指标（抛物线转向）
    try:
        df = calc_psar(df)
    except Exception:
        df['PSAR'] = np.nan
        df['PSAR_dir'] = np.nan

    # Ichimoku云图（一目均衡表）
    try:
        df = calc_ichimoku(df)
    except Exception:
        df['ICH_TENKAN'] = np.nan
        df['ICH_KIJUN'] = np.nan
        df['ICH_SSA'] = np.nan
        df['ICH_SSB'] = np.nan

    # MA200 + 标准差（左侧交易Z-Score用）
    try:
        df['MA200'] = calc_sma(df['close'], 200)
        df['MA200_std'] = df['close'].rolling(200).std()
    except Exception:
        df['MA200'] = np.nan
        df['MA200_std'] = np.nan

    # ADV (20日平均成交量)
    try:
        df['ADV20'] = df['volume'].rolling(20).mean()
    except Exception:
        df['ADV20'] = np.nan

    # ========== 新增右侧确认指标 ==========

    # Break of Structure (BoS) 检测
    try:
        df = calc_break_of_structure(df, lookback=20)
    except Exception:
        df['bos_signal'] = 0
        df['bos_level'] = np.nan
        df['bos_strength'] = 0

    # Change of Character (ChoCh) 检测
    try:
        df = calc_change_of_character(df, lookback=20)
    except Exception:
        df['choch_signal'] = 0
        df['choch_strength'] = 0
        df['choch_trend'] = 'sideways'

    # VWAP 偏离度分析
    try:
        df = calc_vwap_deviation(df, window=20)
    except Exception:
        df['vwap'] = np.nan
        df['vwap_std'] = np.nan
        df['vwap_deviation'] = np.nan
        df['vwap_signal'] = 'normal'

    return df


# KDJ 指标已移除（长线投资不需要短期噪音指标）


def calc_adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """ADX平均趋向指数计算 - 修正Series索引问题

    Args:
        df: 包含high, low, close的DataFrame
        n: 计算周期，默认14

    Returns:
        添加了ADX指标的DataFrame
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM 和 -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # 保持索引一致 - 修正点
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # TR (True Range)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 平滑
    atr = tr.ewm(alpha=1/n, adjust=False).mean()

    # 防止除零
    atr_safe = atr.replace(0, np.inf)
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_safe

    # DX
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.inf)  # 防止除零
    dx = 100 * abs(plus_di - minus_di) / di_sum
    df['ADX'] = dx.ewm(alpha=1/n, adjust=False).mean()
    df['ADX_PDI'] = plus_di
    df['ADX_NDI'] = minus_di

    return df


def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR平均真实波幅计算"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1/n, adjust=False).mean()


def calc_obv(df: pd.DataFrame) -> pd.Series:
    """OBV能量潮计算 - 向量化优化

    Args:
        df: 包含close, volume的DataFrame

    Returns:
        OBV序列
    """
    close = df['close']
    volume = df['volume']

    # 向量化计算：根据价格变化方向决定成交量正负
    direction = np.sign(close.diff())
    direction.iloc[0] = 1  # 第一个值设为正

    # 累加得到OBV
    obv = (direction * volume).cumsum()

    return obv


def calc_cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """CCI顺势指标计算

    Args:
        df: 包含high, low, close的DataFrame
        n: 计算周期，默认20

    Returns:
        CCI序列
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window=n).mean()
    md = tp.rolling(window=n).apply(lambda x: np.abs(x - x.mean()).mean())

    # 防止除零
    md_safe = md.replace(0, np.inf)
    cci = (tp - ma) / (0.015 * md_safe)

    return cci


def calc_supertrend(df: pd.DataFrame, n: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend指标计算"""
    high = df['high']
    low = df['low']
    close = df['close']

    # 基础上下轨
    hl2 = (high + low) / 2
    atr = calc_atr(df, n)

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # 初始化
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    # 第一个有效值
    first_valid = atr.first_valid_index()
    if first_valid is None:
        df['SuperTrend'] = np.nan
        df['SuperTrend_dir'] = np.nan
        return df

    first_idx = df.index.get_loc(first_valid)
    supertrend.iloc[first_idx] = upper_band.iloc[first_idx]
    direction.iloc[first_idx] = -1

    for i in range(first_idx + 1, len(df)):
        # 上轨调整
        if upper_band.iloc[i] < supertrend.iloc[i-1] or close.iloc[i-1] > supertrend.iloc[i-1]:
            upper_band.iloc[i] = upper_band.iloc[i]
        else:
            upper_band.iloc[i] = supertrend.iloc[i-1]

        # 下轨调整
        if lower_band.iloc[i] > supertrend.iloc[i-1] or close.iloc[i-1] < supertrend.iloc[i-1]:
            lower_band.iloc[i] = lower_band.iloc[i]
        else:
            lower_band.iloc[i] = supertrend.iloc[i-1]

        # 方向判断
        if direction.iloc[i-1] == -1:  # 之前是空头
            if close.iloc[i] > supertrend.iloc[i-1]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
        else:  # 之前是多头
            if close.iloc[i] < supertrend.iloc[i-1]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]

    df['SuperTrend'] = supertrend
    df['SuperTrend_dir'] = direction

    return df


def calc_psar(df: pd.DataFrame, af_start: float = 0.02, af_inc: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """PSAR抛物线转向指标计算"""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    length = len(df)
    psar = np.zeros(length)
    psar_dir = np.ones(length)  # 1=多头, -1=空头
    ep = np.zeros(length)  # 极值点
    af = np.zeros(length)  # 加速因子

    # 初始化
    psar[0] = high[0]
    psar_dir[0] = -1
    ep[0] = low[0]
    af[0] = af_start

    for i in range(1, length):
        if psar_dir[i-1] == 1:  # 多头
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])

            if low[i] < psar[i]:  # 反转
                psar_dir[i] = -1
                psar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                psar_dir[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_inc, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # 空头
            psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
            psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])

            if high[i] > psar[i]:  # 反转
                psar_dir[i] = 1
                psar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                psar_dir[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_inc, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

    df['PSAR'] = psar
    df['PSAR_dir'] = psar_dir

    return df


def calc_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """Ichimoku一目均衡表计算"""
    high = df['high']
    low = df['low']

    # 转换线 (Conversion Line / Tenkan-sen)
    tenkan_high = high.rolling(window=tenkan).max()
    tenkan_low = low.rolling(window=tenkan).min()
    df['ICH_TENKAN'] = (tenkan_high + tenkan_low) / 2

    # 基准线 (Base Line / Kijun-sen)
    kijun_high = high.rolling(window=kijun).max()
    kijun_low = low.rolling(window=kijun).min()
    df['ICH_KIJUN'] = (kijun_high + kijun_low) / 2

    # 先行带A (Leading Span A / Senkou Span A)
    df['ICH_SSA'] = ((df['ICH_TENKAN'] + df['ICH_KIJUN']) / 2).shift(kijun)

    # 先行带B (Leading Span B / Senkou Span B)
    senkou_high = high.rolling(window=senkou).max()
    senkou_low = low.rolling(window=senkou).min()
    df['ICH_SSB'] = ((senkou_high + senkou_low) / 2).shift(kijun)

    return df


def calc_vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算价格相对 VWAP 的偏离度 (基于 20 日滚动窗口)

    VWAP = Σ(Close × Volume) / Σ(Volume) (滚动 20 日)
    偏离度 = (Close - VWAP) / VWAP_std

    返回:
    - vwap: 成交量加权平均价 (20 日滚动)
    - vwap_std: VWAP 标准差
    - vwap_deviation: 偏离度 (σ)
    - vwap_signal: 'extreme_high' (>3σ), 'high' (>2σ), 'normal', 'low' (<-2σ), 'extreme_low' (<-3σ)
    """
    df = df.copy()
    df['vwap'] = np.nan
    df['vwap_std'] = np.nan
    df['vwap_deviation'] = np.nan
    df['vwap_signal'] = 'normal'

    try:
        if len(df) < window + 5:
            return df

        # 计算滚动 VWAP
        price_volume = (df['close'] * df['volume']).rolling(window=window).sum()
        volume_sum = df['volume'].rolling(window=window).sum()

        # 避免除零
        df['vwap'] = np.where(volume_sum > 0, price_volume / volume_sum, np.nan)

        # 计算 VWAP 标准差 (使用收盘价的滚动标准差作为近似)
        df['vwap_std'] = df['close'].rolling(window=window).std()

        # 计算偏离度 (σ)
        df['vwap_deviation'] = np.where(
            df['vwap_std'] > 0,
            (df['close'] - df['vwap']) / df['vwap_std'],
            0
        )

        # 分类信号
        def classify_vwap_signal(deviation):
            if pd.isna(deviation):
                return 'normal'
            if deviation > 3:
                return 'extreme_high'
            elif deviation > 2:
                return 'high'
            elif deviation < -3:
                return 'extreme_low'
            elif deviation < -2:
                return 'low'
            else:
                return 'normal'

        df['vwap_signal'] = df['vwap_deviation'].apply(classify_vwap_signal)

    except Exception:
        pass

    return df


def calc_distribution_patterns(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    识别机构派发 (Distribution) 模式

    派发特征:
    1. 放量滞涨 (Volume Climax): 成交量创新高但价格不涨
    2. 顶背离 (Bearish Divergence): 价格新高但 RSI/MACD 不创新高
    3. 破位下跌 (Breakdown): 跌破 MA20 且 MACD 死叉

    返回:
    - distribution_score: 派发强度 (0-100)
    - patterns: ['volume_climax', 'bearish_divergence', 'breakdown']
    - signal: 'strong_distribution' | 'distribution' | 'neutral'
    - signal_text: 中文描述
    """
    result = {
        'distribution_score': 0,
        'patterns': [],
        'signal': 'neutral',
        'signal_text': '无派发迹象',
        'details': []
    }

    try:
        if len(df) < lookback + 10:
            return result

        tail = df.tail(lookback + 5)
        score = 0
        patterns = []
        details = []

        # 1. 检测放量滞涨 (Volume Climax)
        vol_ma20 = tail['volume'].rolling(20).mean()
        recent_5 = tail.tail(5)

        if len(recent_5) >= 5:
            recent_vol = recent_5['volume'].values
            recent_close = recent_5['close'].values
            last_vol_ma = vol_ma20.iloc[-1]

            if not pd.isna(last_vol_ma) and last_vol_ma > 0:
                # 成交量创新高
                vol_surge = any(v > last_vol_ma * 2 for v in recent_vol[-3:])

                # 价格涨幅有限
                price_change_pct = (recent_close[-1] - recent_close[0]) / recent_close[0] * 100

                if vol_surge and -2 < price_change_pct < 2:
                    score += 35
                    patterns.append('volume_climax')
                    details.append(f'放量滞涨: 成交量激增但价格涨幅仅{price_change_pct:.1f}%')

        # 2. 检测顶背离 (Bearish Divergence)
        if 'RSI' in tail.columns and 'MACD' in tail.columns:
            close_series = tail['close'].reset_index(drop=True)
            rsi_series = tail['RSI'].reset_index(drop=True)
            macd_series = tail['MACD'].reset_index(drop=True)

            # 找到波段高点
            swing_points = _find_swing_points(close_series, order=2)
            highs = [(idx, val) for idx, val, typ in swing_points if typ == 'high']

            if len(highs) >= 2:
                i1, p1 = highs[-2]
                i2, p2 = highs[-1]

                # 价格创新高
                if p2 > p1:
                    rsi1 = rsi_series.iloc[i1]
                    rsi2 = rsi_series.iloc[i2]
                    macd1 = macd_series.iloc[i1]
                    macd2 = macd_series.iloc[i2]

                    # RSI 或 MACD 未创新高
                    rsi_div = not pd.isna(rsi1) and not pd.isna(rsi2) and rsi2 < rsi1
                    macd_div = not pd.isna(macd1) and not pd.isna(macd2) and macd2 < macd1

                    if rsi_div or macd_div:
                        score += 40
                        patterns.append('bearish_divergence')
                        div_type = 'RSI' if rsi_div else 'MACD'
                        details.append(f'顶背离: 价格新高但{div_type}未创新高')

        # 3. 检测破位下跌 (Breakdown)
        if 'MA20' in tail.columns and 'MACD' in tail.columns and 'MACD_SIGNAL' in tail.columns:
            last = tail.iloc[-1]
            prev = tail.iloc[-2] if len(tail) >= 2 else last

            close = last['close']
            ma20 = last['MA20']
            macd = last['MACD']
            macd_signal = last['MACD_SIGNAL']
            prev_macd = prev['MACD']
            prev_macd_signal = prev['MACD_SIGNAL']

            # 跌破 MA20
            breakdown_ma = not pd.isna(ma20) and close < ma20

            # MACD 死叉
            macd_death_cross = (
                not pd.isna(macd) and not pd.isna(macd_signal) and
                not pd.isna(prev_macd) and not pd.isna(prev_macd_signal) and
                prev_macd >= prev_macd_signal and macd < macd_signal
            )

            if breakdown_ma and macd_death_cross:
                score += 25
                patterns.append('breakdown')
                details.append(f'破位下跌: 跌破MA20({ma20:.2f})且MACD死叉')

        # 汇总结果
        result['distribution_score'] = min(100, score)
        result['patterns'] = patterns

        if score >= 70:
            result['signal'] = 'strong_distribution'
            result['signal_text'] = '强烈派发信号'
        elif score >= 40:
            result['signal'] = 'distribution'
            result['signal_text'] = '派发迹象'
        else:
            result['signal'] = 'neutral'
            result['signal_text'] = '无明显派发'

        result['details'] = details

    except Exception:
        pass

    return result


# ============ 左侧交易模块 (Contrarian / Left-Side Trading) ============

def calc_mean_reversion_zscore(df: pd.DataFrame, ma_period: int = 200) -> dict:
    """均值回归Z-Score分析

    计算价格偏离200日均线的标准差倍数，识别极端超卖区域。
    """
    result = {
        'zscore': 0.0, 'ma200': None, 'deviation_pct': 0.0,
        'signal': 'neutral', 'signal_text': '数据不足',
        'score': 50, 'sufficient_data': False
    }
    try:
        if len(df) < ma_period + 20:
            return result

        ma200 = df['MA200'].iloc[-1]
        std200 = df['MA200_std'].iloc[-1]
        close = df['close'].iloc[-1]

        if pd.isna(ma200) or pd.isna(std200) or std200 == 0:
            return result

        zscore = (close - ma200) / std200
        deviation_pct = (close - ma200) / ma200 * 100
        result['zscore'] = round(zscore, 2)
        result['ma200'] = round(ma200, 2)
        result['deviation_pct'] = round(deviation_pct, 1)
        result['sufficient_data'] = True

        if zscore < -2.5:
            result.update(signal='extreme_oversold', score=95,
                          signal_text=f'极度超卖(Z={zscore:.2f})，强烈左侧买入信号')
        elif zscore < -2.0:
            result.update(signal='oversold', score=80,
                          signal_text=f'超卖区(Z={zscore:.2f})，左侧买入信号')
        elif zscore < -1.0:
            result.update(signal='mild_oversold', score=65,
                          signal_text=f'偏低(Z={zscore:.2f})，关注左侧机会')
        elif zscore > 2.5:
            result.update(signal='extreme_overbought', score=5,
                          signal_text=f'极度超买(Z={zscore:.2f})，左侧卖出信号')
        elif zscore > 2.0:
            result.update(signal='overbought', score=15,
                          signal_text=f'超买区(Z={zscore:.2f})，警惕回调')
        else:
            result.update(signal='neutral', score=50,
                          signal_text=f'中性区间(Z={zscore:.2f})')
    except Exception:
        pass
    return result


def _find_swing_points(series: pd.Series, order: int = 3) -> list:
    """找摆动高低点，返回 [(index, value), ...]"""
    points = []
    values = series.values
    for i in range(order, len(values) - order):
        if all(values[i] <= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] <= values[i + j] for j in range(1, order + 1)):
            points.append((i, values[i], 'low'))
        if all(values[i] >= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] >= values[i + j] for j in range(1, order + 1)):
            points.append((i, values[i], 'high'))
    return points

# PLACEHOLDER_MODULES_CONTINUE


def calc_volume_exhaustion(df: pd.DataFrame, lookback: int = 20) -> dict:
    """量能衰竭分析 — 检测放量滞跌、缩量阴跌、高量节点(HVN)等转折模式。"""
    result = {
        'pattern': 'normal', 'pattern_text': '正常',
        'hvn_zones': [], 'vol_trend': 'flat',
        'vol_price_divergence': False, 'score': 50, 'details': '无明显量能异常'
    }
    try:
        if len(df) < lookback + 5:
            return result

        vol_ma20 = df['volume'].rolling(20).mean()
        recent = df.tail(5).copy()
        recent_vol = recent['volume'].values
        recent_close = recent['close'].values
        last_vol_ma = vol_ma20.iloc[-1]

        if pd.isna(last_vol_ma) or last_vol_ma == 0:
            return result

        # --- 模式检测 ---
        last3_vol = recent_vol[-3:]
        last3_close = recent_close[-3:]
        total_drop_3d = (last3_close[-1] - last3_close[0]) / last3_close[0] * 100

        # 放量滞跌: 近3日量 > 1.5x MA20 但总跌幅 < 1%
        if all(v > last_vol_ma * 1.5 for v in last3_vol) and -1.0 < total_drop_3d < 1.0:
            result.update(pattern='high_vol_stall', pattern_text='放量滞跌',
                          score=80, details='大量成交但价格不再下跌，空方力竭信号')

        # 缩量阴跌: 连续5日量递减+价格递减
        elif all(recent_vol[i] < recent_vol[i - 1] for i in range(1, 5)) and \
             all(recent_close[i] < recent_close[i - 1] for i in range(1, 5)):
            result.update(pattern='declining_vol_drop', pattern_text='缩量阴跌',
                          score=60, details='量价齐缩，抛压衰竭中，关注企稳信号')

        # 放量长阴后缩量: 某日量>2x MA20且跌>3%, 随后缩量
        else:
            for i in range(max(0, len(df) - 6), len(df) - 2):
                vol_i = df['volume'].iloc[i]
                chg_i = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1] * 100 if i > 0 else 0
                if vol_i > last_vol_ma * 2 and chg_i < -3:
                    after_vols = df['volume'].iloc[i + 1:i + 3]
                    if len(after_vols) >= 2 and all(v < vol_i for v in after_vols):
                        result.update(pattern='volume_climax', pattern_text='放量长阴后缩量',
                                      score=75, details='恐慌性抛售后量能萎缩，可能见底')
                        break

        # --- 量价背离 ---
        low_20 = df['close'].tail(20).min()
        if df['close'].iloc[-1] <= low_20 * 1.005 and df['volume'].iloc[-1] < last_vol_ma:
            result['vol_price_divergence'] = True
            if result['score'] < 70:
                result['score'] = 70
                result['details'] += '；量价背离(创新低但缩量)'

        # --- 量能趋势 ---
        vol_10 = df['volume'].tail(10).values
        if len(vol_10) >= 10:
            slope = np.polyfit(range(len(vol_10)), vol_10, 1)[0]
            if slope > last_vol_ma * 0.02:
                result['vol_trend'] = 'increasing'
            elif slope < -last_vol_ma * 0.02:
                result['vol_trend'] = 'decreasing'

        # --- HVN (高量节点) ---
        hvn_df = df.tail(60) if len(df) >= 60 else df
        price_min, price_max = hvn_df['low'].min(), hvn_df['high'].max()
        if price_max > price_min:
            n_bins = 10
            bin_size = (price_max - price_min) / n_bins
            bins = []
            for b in range(n_bins):
                lo = price_min + b * bin_size
                hi = lo + bin_size
                mask = (hvn_df['close'] >= lo) & (hvn_df['close'] < hi)
                vol_sum = hvn_df.loc[mask, 'volume'].sum()
                bins.append((round(lo, 2), round(hi, 2), vol_sum))
            avg_vol = sum(b[2] for b in bins) / n_bins if n_bins > 0 else 0
            result['hvn_zones'] = [(lo, hi, round(v / avg_vol, 1))
                                   for lo, hi, v in bins if avg_vol > 0 and v > avg_vol * 1.5]
    except Exception:
        pass
    return result


def find_anchor_point(df: pd.DataFrame) -> dict:
    """自动识别 Volume Profile 的锚定点

    优先级（从高到低）：
    1. 最近一次 BoS（结构突破）点
    2. 近期最大放量日（成交量 > 3倍MA20）
    3. 财报跳空日（Gap > 3%）
    4. Fallback: 60天前

    Returns:
        {
            'anchor_type': 'bos' | 'volume_climax' | 'earnings_gap' | 'fallback',
            'anchor_index': int,           # 锚定点在 df 中的索引
            'anchor_date': str,            # 锚定日期
            'anchor_price': float,         # 锚定点价格
            'anchor_details': str,         # 锚定点说明
            'days_since_anchor': int       # 距今天数
        }
    """
    default_result = {
        'anchor_type': 'fallback',
        'anchor_index': max(0, len(df) - 60),
        'anchor_date': str(df.iloc[max(0, len(df) - 60)]['date']) if len(df) > 0 else '',
        'anchor_price': df.iloc[max(0, len(df) - 60)]['close'] if len(df) > 0 else 0,
        'anchor_details': '默认锚定点（60天前）',
        'days_since_anchor': min(60, len(df))
    }

    if len(df) < 30:
        return default_result

    try:
        # 计算 BoS（复用已有函数的逻辑）
        vol_ma20 = df['volume'].rolling(20).mean()

        # 1. 查找最近一次 BoS 点
        # BoS 定义：突破前一个波段高点/低点，且成交量放大
        lookback = min(60, len(df) - 10)
        for i in range(len(df) - 1, max(lookback, 10), -1):
            window = df.iloc[max(0, i-20):i+1]
            if len(window) < 10:
                continue

            # 简化的 BoS 检测：收盘价创 window新高或新低，且成交量放大
            current_close = df.iloc[i]['close']
            current_vol = df.iloc[i]['volume']
            vol_ma = vol_ma20.iloc[i] if i < len(vol_ma20) else None

            window_high = window['high'].max()
            window_low = window['low'].min()

            is_breakout_up = current_close >= window_high * 0.995 and current_vol and vol_ma and current_vol > vol_ma * 1.3
            is_breakout_down = current_close <= window_low * 1.005 and current_vol and vol_ma and current_vol > vol_ma * 1.3

            if is_breakout_up or is_breakout_down:
                return {
                    'anchor_type': 'bos',
                    'anchor_index': i,
                    'anchor_date': str(df.iloc[i]['date']),
                    'anchor_price': df.iloc[i]['close'],
                    'anchor_details': f"结构突破点({'向上' if is_breakout_up else '向下'})",
                    'days_since_anchor': len(df) - 1 - i
                }

        # 2. 查找近期最大放量日
        vol_ratio = df['volume'] / vol_ma20
        vol_ratio = vol_ratio.fillna(0)

        # 找成交量比值最大的那一天（排除最近5天，避免噪音）
        for i in range(len(df) - 6, max(lookback, 10), -1):
            if vol_ratio.iloc[i] > 3.0:
                return {
                    'anchor_type': 'volume_climax',
                    'anchor_index': i,
                    'anchor_date': str(df.iloc[i]['date']),
                    'anchor_price': df.iloc[i]['close'],
                    'anchor_details': f"放量日(量比={vol_ratio.iloc[i]:.1f}x)",
                    'days_since_anchor': len(df) - 1 - i
                }

        # 如果没有 >3x 的，找最大的
        max_vol_idx = vol_ratio.iloc[max(lookback, 10):len(df)-5].idxmax()
        if max_vol_idx is not None and vol_ratio.iloc[max_vol_idx] > 2.0:
            return {
                'anchor_type': 'volume_climax',
                'anchor_index': max_vol_idx,
                'anchor_date': str(df.iloc[max_vol_idx]['date']),
                'anchor_price': df.iloc[max_vol_idx]['close'],
                'anchor_details': f"最大放量日(量比={vol_ratio.iloc[max_vol_idx]:.1f}x)",
                'days_since_anchor': len(df) - 1 - max_vol_idx
            }

        # 3. 查找财报跳空日
        df['gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        for i in range(len(df) - 1, max(lookback, 10), -1):
            gap = df.iloc[i]['gap']
            vol = df.iloc[i]['volume']
            vol_ma = vol_ma20.iloc[i] if i < len(vol_ma20) else None
            if gap > 3.0 and vol and vol_ma and vol > vol_ma * 1.2:
                gap_dir = '向上' if df.iloc[i]['open'] > df.iloc[i-1]['close'] else '向下'
                return {
                    'anchor_type': 'earnings_gap',
                    'anchor_index': i,
                    'anchor_date': str(df.iloc[i]['date']),
                    'anchor_price': df.iloc[i]['close'],
                    'anchor_details': f"跳空日({gap_dir}跳空{gap:.1f}%)",
                    'days_since_anchor': len(df) - 1 - i
                }

        # 4. Fallback
        fallback_idx = max(0, len(df) - 60)
        return {
            'anchor_type': 'fallback',
            'anchor_index': fallback_idx,
            'anchor_date': str(df.iloc[fallback_idx]['date']),
            'anchor_price': df.iloc[fallback_idx]['close'],
            'anchor_details': '默认锚定点（60天前）',
            'days_since_anchor': len(df) - 1 - fallback_idx
        }

    except Exception:
        return default_result


def calc_anchored_volume_profile(df: pd.DataFrame, value_area_pct: float = 0.7) -> dict:
    """计算锚定筹码分布 (Anchored Volume Profile)

    从自动识别的锚定点开始，计算至今的筹码分布

    Args:
        df: OHLCV DataFrame
        value_area_pct: Value Area 包含的成交量比例，默认70%

    Returns:
        {
            'anchor_info': dict,           # 来自 find_anchor_point()
            'poc': float,                  # Point of Control - 最大成交量价位
            'vah': float,                  # Value Area High
            'val': float,                  # Value Area Low
            'total_volume': float,         # 总成交量
            'price_range': (low, high),    # 价格区间
            'hvn_zones': [(low, high, vol_ratio), ...],  # 高成交量节点
            'lnv_zones': [(low, high, vol_ratio), ...],  # 低成交量节点
            'support_levels': [float, ...], # 基于VP的支撑位
            'resistance_levels': [float, ...], # 基于VP的压力位
            'interpretation': str           # 筹码分布解读
        }
    """
    default_result = {
        'anchor_info': None,
        'poc': None,
        'vah': None,
        'val': None,
        'total_volume': 0,
        'price_range': (None, None),
        'hvn_zones': [],
        'lnv_zones': [],
        'support_levels': [],
        'resistance_levels': [],
        'interpretation': '数据不足，无法计算筹码分布'
    }

    if len(df) < 30:
        return default_result

    try:
        # 1. 找锚定点
        anchor_info = find_anchor_point(df)
        anchor_idx = anchor_info['anchor_index']

        # 2. 从锚定点到现在的数据
        vp_df = df.iloc[anchor_idx:].copy()

        if len(vp_df) < 10:
            return default_result

        # 3. 价格分箱 - 使用 (high + low) / 2 作为代表价格
        vp_df['typical_price'] = (vp_df['high'] + vp_df['low']) / 2

        price_min = vp_df['low'].min()
        price_max = vp_df['high'].max()

        if price_max <= price_min:
            return default_result

        # 自适应分箱数量（约100个bins，但根据数据量调整）
        n_bins = min(100, max(20, len(vp_df) // 2))
        bin_size = (price_max - price_min) / n_bins

        # 4. 计算每个价格区间的成交量
        bins_volume = []
        for b in range(n_bins):
            bin_low = price_min + b * bin_size
            bin_high = bin_low + bin_size
            bin_mid = (bin_low + bin_high) / 2

            # 统计落在该区间的K线成交量
            # 使用 typical_price 判断
            mask = (vp_df['typical_price'] >= bin_low) & (vp_df['typical_price'] < bin_high)
            vol = vp_df.loc[mask, 'volume'].sum()

            bins_volume.append({
                'low': bin_low,
                'high': bin_high,
                'mid': bin_mid,
                'volume': vol
            })

        total_volume = sum(b['volume'] for b in bins_volume)
        if total_volume == 0:
            return default_result

        # 5. 计算 POC (Point of Control) - 成交量最大的区间
        poc_bin = max(bins_volume, key=lambda x: x['volume'])
        poc = poc_bin['mid']

        # 6. 计算 Value Area (VAH/VAL) - 包含70%成交量的区间
        sorted_bins = sorted(bins_volume, key=lambda x: x['volume'], reverse=True)
        va_volume = 0
        va_bins = []
        target_va_volume = total_volume * value_area_pct

        for b in sorted_bins:
            va_bins.append(b)
            va_volume += b['volume']
            if va_volume >= target_va_volume:
                break

        vah = max(b['high'] for b in va_bins)
        val = min(b['low'] for b in va_bins)

        # 7. 识别 HVN (High Volume Nodes) 和 LNV (Low Volume Nodes)
        avg_volume = total_volume / n_bins
        hvn_zones = []
        lnv_zones = []

        for b in bins_volume:
            vol_ratio = b['volume'] / avg_volume if avg_volume > 0 else 0
            if vol_ratio > 1.5:
                hvn_zones.append((round(b['low'], 2), round(b['high'], 2), round(vol_ratio, 2)))
            elif vol_ratio < 0.5:
                lnv_zones.append((round(b['low'], 2), round(b['high'], 2), round(vol_ratio, 2)))

        # 8. 基于VP的支撑压力位
        current_price = df.iloc[-1]['close']

        # 支撑位：当前价下方的HVN
        support_levels = sorted([b[0] for b in hvn_zones if b[0] < current_price], reverse=True)[:3]

        # 压力位：当前价上方的HVN
        resistance_levels = sorted([b[1] for b in hvn_zones if b[1] > current_price])[:3]

        # 9. 筹码分布解读
        interpretation = ''
        if current_price > vah:
            interpretation = f'当前价({current_price:.2f})在VA上方，高位筹码集中，注意派发风险'
        elif current_price < val:
            interpretation = f'当前价({current_price:.2f})在VA下方，低位筹码集中，关注反弹机会'
        elif abs(current_price - poc) / poc < 0.02:
            interpretation = f'当前价({current_price:.2f})接近POC({poc:.2f})，筹码集中，即将方向选择'
        else:
            interpretation = f'当前价在VA内，POC={poc:.2f}'

        return {
            'anchor_info': anchor_info,
            'poc': round(poc, 2),
            'vah': round(vah, 2),
            'val': round(val, 2),
            'total_volume': total_volume,
            'price_range': (round(price_min, 2), round(price_max, 2)),
            'hvn_zones': hvn_zones,
            'lnv_zones': lnv_zones,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'interpretation': interpretation
        }

    except Exception:
        return default_result


def calc_macro_volume_profile(df: pd.DataFrame, value_area_pct: float = 0.7) -> dict:
    """3年宏观筹码分布 — 使用完整历史数据（不依赖锚点），识别长线核心支撑/阻力区。

    Args:
        df: 完整 OHLCV DataFrame（建议 3-5 年日线数据）
        value_area_pct: Value Area 包含的成交量比例，默认70%

    Returns:
        dict: 宏观 POC/VAH/VAL/HVN 及长线支撑阻力
    """
    default_result = {
        'poc': None, 'vah': None, 'val': None,
        'hvn_zones': [], 'lnv_zones': [],
        'macro_support': [], 'macro_resistance': [],
        'total_volume': 0, 'data_days': 0,
        'interpretation': '数据不足，无法计算宏观筹码分布',
        'available': False
    }

    if len(df) < 120:
        return default_result

    try:
        vp_df = df.copy()
        vp_df['typical_price'] = (vp_df['high'] + vp_df['low']) / 2

        price_min = vp_df['low'].min()
        price_max = vp_df['high'].max()

        if price_max <= price_min:
            return default_result

        n_bins = min(150, max(30, len(vp_df) // 5))
        bin_size = (price_max - price_min) / n_bins

        bins_volume = []
        for b in range(n_bins):
            bin_low = price_min + b * bin_size
            bin_high = bin_low + bin_size
            bin_mid = (bin_low + bin_high) / 2
            mask = (vp_df['typical_price'] >= bin_low) & (vp_df['typical_price'] < bin_high)
            vol = vp_df.loc[mask, 'volume'].sum()
            bins_volume.append({'low': bin_low, 'high': bin_high, 'mid': bin_mid, 'volume': vol})

        total_volume = sum(b['volume'] for b in bins_volume)
        if total_volume == 0:
            return default_result

        poc_bin = max(bins_volume, key=lambda x: x['volume'])
        poc = poc_bin['mid']

        sorted_bins = sorted(bins_volume, key=lambda x: x['volume'], reverse=True)
        va_volume = 0
        va_bins = []
        target_va_volume = total_volume * value_area_pct
        for b in sorted_bins:
            va_bins.append(b)
            va_volume += b['volume']
            if va_volume >= target_va_volume:
                break

        vah = max(b['high'] for b in va_bins)
        val = min(b['low'] for b in va_bins)

        avg_volume = total_volume / n_bins
        hvn_zones = []
        lnv_zones = []
        for b in bins_volume:
            vol_ratio = b['volume'] / avg_volume if avg_volume > 0 else 0
            if vol_ratio > 1.5:
                hvn_zones.append((round(b['low'], 2), round(b['high'], 2), round(vol_ratio, 2)))
            elif vol_ratio < 0.5:
                lnv_zones.append((round(b['low'], 2), round(b['high'], 2), round(vol_ratio, 2)))

        current_price = df.iloc[-1]['close']
        macro_support = sorted([h[0] for h in hvn_zones if h[0] < current_price], reverse=True)[:5]
        macro_resistance = sorted([h[1] for h in hvn_zones if h[1] > current_price])[:5]

        data_years = len(df) / 250
        interpretation = f'{data_years:.1f}年宏观筹码分布: POC={poc:.2f}'
        if current_price > vah:
            interpretation += f', 当前价({current_price:.2f})在宏观VA上方，历史高位套牢盘较少'
        elif current_price < val:
            interpretation += f', 当前价({current_price:.2f})在宏观VA下方，历史低位筹码密集区'
        else:
            interpretation += f', 当前价在宏观VA内 ({val:.2f}-{vah:.2f})'

        return {
            'poc': round(poc, 2), 'vah': round(vah, 2), 'val': round(val, 2),
            'hvn_zones': hvn_zones, 'lnv_zones': lnv_zones,
            'macro_support': macro_support, 'macro_resistance': macro_resistance,
            'total_volume': total_volume, 'data_days': len(df),
            'interpretation': interpretation, 'available': True
        }

    except Exception:
        return default_result


def calc_volatility_regime(df: pd.DataFrame) -> dict:
    """波动率收缩/扩张分析 — ATR百分位、布林带挤压、脉冲信号检测。"""
    result = {
        'regime': 'normal', 'regime_text': '正常波动',
        'atr_percentile': 50.0, 'bb_width_percentile': 50.0,
        'is_squeeze': False, 'is_pulse': False,
        'score': 50, 'details': ''
    }
    try:
        lookback = min(120, len(df))
        if lookback < 30:
            return result

        # ATR 百分位
        atr_series = df['ATR'].tail(lookback)
        if atr_series.isna().all():
            return result
        atr_pct = atr_series.rank(pct=True).iloc[-1] * 100
        result['atr_percentile'] = round(atr_pct, 1)

        # BB 宽度百分位
        bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        bb_series = bb_width.tail(lookback)
        if bb_series.isna().all():
            bb_pct = 50.0
        else:
            bb_pct = bb_series.rank(pct=True).iloc[-1] * 100
        result['bb_width_percentile'] = round(bb_pct, 1)

        # 挤压检测
        is_squeeze = atr_pct < 20 and bb_pct < 20
        result['is_squeeze'] = is_squeeze

        # 脉冲检测: 当前ATR百分位>30 且 前5日均ATR百分位<20
        atr_pct_series = atr_series.rank(pct=True) * 100
        prev5_avg = atr_pct_series.iloc[-6:-1].mean() if len(atr_pct_series) >= 6 else 50
        is_pulse = atr_pct > 30 and prev5_avg < 20
        result['is_pulse'] = is_pulse

        close = df['close'].iloc[-1]
        bb_lower = df['BB_lower'].iloc[-1] if 'BB_lower' in df.columns else None

        if is_pulse:
            # 脉冲方向
            price_dir = df['close'].iloc[-1] > df['close'].iloc[-3] if len(df) >= 3 else True
            if price_dir:
                result.update(regime='pulse', regime_text='脉冲信号(收缩后向上扩张)',
                              score=85, details='波动率极低后首次放大，方向向上，左侧入场信号')
            else:
                result.update(regime='pulse', regime_text='脉冲信号(收缩后向下扩张)',
                              score=30, details='波动率极低后首次放大，方向向下，警惕破位')
        elif is_squeeze:
            near_lower = bb_lower is not None and not pd.isna(bb_lower) and close < bb_lower * 1.02
            if near_lower:
                result.update(regime='squeeze', regime_text='布林带挤压(价格靠近下轨)',
                              score=75, details='波动率极低+价格靠近下轨，蓄势待发')
            else:
                result.update(regime='squeeze', regime_text='布林带挤压',
                              score=60, details='波动率极低，等待方向选择')
        elif atr_pct > 80:
            result.update(regime='expansion', regime_text='高波动扩张',
                          score=50, details='波动率处于高位，趋势可能延续')
    except Exception:
        pass
    return result


# PLACEHOLDER_SR_AND_DIVERGENCE


def calc_adaptive_params(df: pd.DataFrame, market_stage: str = '震荡整理') -> dict:
    """
    自适应周期参数计算中心 — 基于四维度动态调整所有周期参数

    四维度：
    1. 波动率制度（ATR百分位）
    2. 趋势强度（ADX）
    3. 流动性状态（量比 = 当日成交量/ADV20）
    4. 市场阶段（筑底/启动/派发/下行/震荡）

    Args:
        df: OHLCV DataFrame（必须已计算 ATR, ADX, ADV20）
        market_stage: 市场阶段（来自 determine_market_stage()）

    Returns:
        dict: 包含所有自适应周期参数和诊断信息
    """
    # 默认返回值（数据不足时使用固定周期）
    default_result = {
        'bos_lookback': 20, 'choch_lookback': 20, 'chandelier_lookback': 22,
        'vol_exhaust_lookback': 20, 'divergence_lookback': 30,
        'volatility_factor': 1.0, 'trend_factor': 1.0, 'liquidity_factor': 1.0,
        'stage_offset': 0, 'composite_factor': 1.0,
        'atr_percentile': 50.0, 'adx': 25.0, 'volume_ratio': 1.0,
        'market_stage': market_stage, 'regime': 'normal',
        'explanation': '数据不足，使用固定周期'
    }

    try:
        if len(df) < 60:
            return default_result

        last = df.iloc[-1]

        # ========== 维度1: 波动率制度 ==========
        vol_regime = calc_volatility_regime(df)
        atr_pct = vol_regime['atr_percentile']
        regime = vol_regime['regime']

        # 波动率缩放因子：挤压期用短周期，扩张期用长周期
        if regime == 'squeeze':
            volatility_factor = 0.5  # 挤压期：周期减半（20→10）
        elif regime == 'pulse':
            volatility_factor = 0.7  # 脉冲期：周期缩短30%
        elif regime == 'expansion':
            volatility_factor = 1.5  # 扩张期：周期延长50%
        else:  # normal
            if atr_pct < 30:
                volatility_factor = 0.7
            elif atr_pct > 70:
                volatility_factor = 1.3
            else:
                volatility_factor = 1.0

        # ========== 维度2: 趋势强度 ==========
        adx = last.get('ADX', 25.0)
        if pd.isna(adx):
            adx = 25.0

        # 趋势缩放因子：强趋势用长周期，弱趋势用短周期
        if adx > 40:
            trend_factor = 1.5  # 极强趋势：延长50%
        elif adx > 25:
            trend_factor = 1.2  # 强趋势：延长20%
        elif adx < 15:
            trend_factor = 0.6  # 极弱趋势：缩短40%
        elif adx < 20:
            trend_factor = 0.8  # 弱趋势：缩短20%
        else:
            trend_factor = 1.0

        # ========== 维度3: 流动性状态 ==========
        current_vol = last.get('volume', 0)
        adv20 = last.get('ADV20', 0)

        if adv20 > 0 and not pd.isna(adv20):
            volume_ratio = current_vol / adv20
        else:
            volume_ratio = 1.0

        # 流动性缩放因子：高流动性用短周期，低流动性用长周期
        if volume_ratio > 2.0:
            liquidity_factor = 0.7  # 高流动性：缩短30%
        elif volume_ratio > 1.5:
            liquidity_factor = 0.9
        elif volume_ratio < 0.5:
            liquidity_factor = 1.3  # 低流动性：延长30%
        elif volume_ratio < 0.8:
            liquidity_factor = 1.1
        else:
            liquidity_factor = 1.0

        # ========== 维度4: 市场阶段 ==========
        # 不同阶段的基础周期偏移（加法调整）
        stage_offset_map = {
            '左侧筑底': -5,    # 筑底期：缩短周期，捕捉早期信号
            '右侧启动': 0,     # 启动期：标准周期
            '高位派发': +5,    # 派发期：延长周期，避免假突破
            '下行通道': +3,    # 下行期：延长周期，过滤噪音
            '震荡整理': 0      # 震荡期：标准周期
        }
        stage_offset = stage_offset_map.get(market_stage, 0)

        # ========== 综合缩放因子 ==========
        # 加权平均：波动率40%，趋势30%，流动性30%
        composite_factor = (
            volatility_factor * 0.4 +
            trend_factor * 0.3 +
            liquidity_factor * 0.3
        )

        # 限制在 [0.5, 1.5] 范围内
        composite_factor = max(0.5, min(1.5, composite_factor))

        # ========== 计算自适应周期 ==========
        # 基础周期 × 综合因子 + 阶段偏移
        bos_base = 20
        choch_base = 20
        chandelier_base = 22
        vol_exhaust_base = 20
        divergence_base = 30

        bos_lookback = int(bos_base * composite_factor + stage_offset)
        choch_lookback = int(choch_base * composite_factor + stage_offset)
        chandelier_lookback = int(chandelier_base * composite_factor + stage_offset)
        vol_exhaust_lookback = int(vol_exhaust_base * composite_factor + stage_offset)
        divergence_lookback = int(divergence_base * composite_factor + stage_offset)

        # 确保最小值（避免周期过短）
        bos_lookback = max(10, bos_lookback)
        choch_lookback = max(10, choch_lookback)
        chandelier_lookback = max(11, chandelier_lookback)
        vol_exhaust_lookback = max(10, vol_exhaust_lookback)
        divergence_lookback = max(15, divergence_lookback)

        # 生成解释
        regime_text_map = {
            'squeeze': '挤压期',
            'pulse': '脉冲期',
            'expansion': '扩张期',
            'normal': '正常'
        }
        explanation = f"{regime_text_map.get(regime, '正常波动')} + ADX={adx:.0f} + 量比={volume_ratio:.1f}x → 周期调整为 {composite_factor:.1f}x"

        return {
            'bos_lookback': bos_lookback,
            'choch_lookback': choch_lookback,
            'chandelier_lookback': chandelier_lookback,
            'vol_exhaust_lookback': vol_exhaust_lookback,
            'divergence_lookback': divergence_lookback,
            'volatility_factor': round(volatility_factor, 2),
            'trend_factor': round(trend_factor, 2),
            'liquidity_factor': round(liquidity_factor, 2),
            'stage_offset': stage_offset,
            'composite_factor': round(composite_factor, 2),
            'atr_percentile': round(atr_pct, 1),
            'adx': round(adx, 1),
            'volume_ratio': round(volume_ratio, 2),
            'market_stage': market_stage,
            'regime': regime,
            'explanation': explanation
        }

    except Exception:
        return default_result


def find_support_resistance_institutional(df: pd.DataFrame) -> dict:
    """机构级支撑压力位分析 — Order Block、FVG、流动性扫荡检测。"""
    # 传统S/R作为基线
    recent = df.tail(20)
    support_levels = sorted(set(round(v, 2) for v in recent['low'].nsmallest(3)))[:2]
    resistance_levels = sorted(set(round(v, 2) for v in recent['high'].nlargest(3)), reverse=True)[:2]

    result = {
        'support': support_levels, 'resistance': resistance_levels,
        'order_blocks': [], 'fvg_zones': [], 'liquidity_sweeps': [],
        'score': 50, 'signal_text': '传统支撑压力位'
    }
    try:
        current_price = df['close'].iloc[-1]
        scan_len = min(60, len(df) - 1)
        scan_df = df.tail(scan_len + 1).reset_index(drop=True)

        # --- Order Blocks ---
        for i in range(1, len(scan_df) - 1):
            prev = scan_df.iloc[i - 1]
            curr = scan_df.iloc[i]
            nxt = scan_df.iloc[i + 1]
            # 看多OB: 阴线后紧跟强阳线(收盘>阴线开盘)
            if curr['close'] < curr['open'] and nxt['close'] > curr['open'] and nxt['close'] > nxt['open']:
                result['order_blocks'].append({
                    'type': 'bullish', 'price_low': round(curr['low'], 2),
                    'price_high': round(curr['high'], 2),
                    'strength': round(abs(nxt['close'] - nxt['open']) / max(abs(curr['close'] - curr['open']), 0.01), 1)
                })
            # 看空OB: 阳线后紧跟强阴线(收盘<阳线开盘)
            if curr['close'] > curr['open'] and nxt['close'] < curr['open'] and nxt['close'] < nxt['open']:
                result['order_blocks'].append({
                    'type': 'bearish', 'price_low': round(curr['low'], 2),
                    'price_high': round(curr['high'], 2),
                    'strength': round(abs(nxt['open'] - nxt['close']) / max(abs(curr['close'] - curr['open']), 0.01), 1)
                })
        # 只保留最近5个
        result['order_blocks'] = result['order_blocks'][-5:]

        # --- FVG (公允价值缺口) ---
        for i in range(len(scan_df) - 2):
            c0 = scan_df.iloc[i]
            c2 = scan_df.iloc[i + 2]
            # 看多FVG: 向上跳空
            if c2['low'] > c0['high']:
                filled = current_price <= c2['low'] and current_price >= c0['high']
                result['fvg_zones'].append({
                    'type': 'bullish', 'top': round(c2['low'], 2),
                    'bottom': round(c0['high'], 2), 'filled': filled
                })
            # 看空FVG: 向下跳空
            if c2['high'] < c0['low']:
                filled = current_price >= c2['high'] and current_price <= c0['low']
                result['fvg_zones'].append({
                    'type': 'bearish', 'top': round(c0['low'], 2),
                    'bottom': round(c2['high'], 2), 'filled': filled
                })
        result['fvg_zones'] = result['fvg_zones'][-5:]

        # --- Liquidity Sweep (假破位) ---
        for i in range(20, len(scan_df) - 2):
            low_20 = scan_df['low'].iloc[i - 20:i].min()
            high_20 = scan_df['high'].iloc[i - 20:i].max()
            curr = scan_df.iloc[i]
            # 扫底: 跌破20日低点后2日内收回
            if curr['low'] < low_20:
                for j in range(1, min(3, len(scan_df) - i)):
                    if scan_df['close'].iloc[i + j] > low_20:
                        result['liquidity_sweeps'].append({
                            'type': 'sweep_low', 'price': round(low_20, 2),
                            'recovered': True,
                            'date': str(scan_df['date'].iloc[i]) if 'date' in scan_df.columns else ''
                        })
                        break
            # 扫顶: 突破20日高点后2日内收回
            if curr['high'] > high_20:
                for j in range(1, min(3, len(scan_df) - i)):
                    if scan_df['close'].iloc[i + j] < high_20:
                        result['liquidity_sweeps'].append({
                            'type': 'sweep_high', 'price': round(high_20, 2),
                            'recovered': True,
                            'date': str(scan_df['date'].iloc[i]) if 'date' in scan_df.columns else ''
                        })
                        break
        result['liquidity_sweeps'] = result['liquidity_sweeps'][-5:]

        # --- 评分 ---
        score = 50
        # 价格在看多OB区间
        for ob in result['order_blocks']:
            if ob['type'] == 'bullish' and ob['price_low'] <= current_price <= ob['price_high']:
                score += 25
                break
        # 近期未回补看多FVG
        for fvg in result['fvg_zones']:
            if fvg['type'] == 'bullish' and not fvg['filled'] and fvg['bottom'] <= current_price:
                score += 20
                break
        # 近期扫底
        recent_sweeps = [s for s in result['liquidity_sweeps']
                         if s['type'] == 'sweep_low' and s['recovered']]
        if recent_sweeps:
            score += 30
        result['score'] = min(100, max(0, score))

        if score >= 80:
            result['signal_text'] = '强机构级支撑(OB+FVG+假破位共振)'
        elif score >= 65:
            result['signal_text'] = '机构级支撑信号'
        elif score <= 30:
            result['signal_text'] = '机构级压力区'
        else:
            result['signal_text'] = '传统支撑压力位'
    except Exception:
        pass
    return result


def calc_break_of_structure(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    检测市场结构突破 (Break of Structure)

    BoS 定义:
    - 上升 BoS: 价格突破前一个波段高点 (Higher High)
    - 下降 BoS: 价格跌破前一个波段低点 (Lower Low)

    参数:
    - lookback: 20 天 (保守模式)

    返回:
    - bos_signal: 1 (bullish BoS), -1 (bearish BoS), 0 (no BoS)
    - bos_level: 突破的关键价位
    - bos_strength: 突破强度 (0-100)
    """
    df = df.copy()
    df['bos_signal'] = 0
    df['bos_level'] = np.nan
    df['bos_strength'] = 0

    try:
        if len(df) < lookback + 10:
            return df

        # 计算成交量均线用于确认
        vol_ma20 = df['volume'].rolling(20).mean()

        # 遍历最近的数据点
        for i in range(lookback + 5, len(df)):
            window = df.iloc[i-lookback:i+1]
            close_series = window['close'].reset_index(drop=True)

            # 找到波段高低点
            swing_points = _find_swing_points(close_series, order=3)
            highs = [(idx, val) for idx, val, typ in swing_points if typ == 'high']
            lows = [(idx, val) for idx, val, typ in swing_points if typ == 'low']

            current_close = df.iloc[i]['close']
            current_vol = df.iloc[i]['volume']
            vol_ma = vol_ma20.iloc[i]

            # 检测上升 BoS (突破前一个波段高点)
            if len(highs) >= 2:
                prev_high_idx, prev_high_val = highs[-2]
                last_high_idx, last_high_val = highs[-1]

                # 确认突破: 收盘价站稳 + 成交量放大
                if current_close > last_high_val and not pd.isna(vol_ma) and vol_ma > 0:
                    if current_vol > vol_ma:  # 成交量确认
                        strength = min(100, ((current_close - last_high_val) / last_high_val) * 100 * 10)
                        df.loc[df.index[i], 'bos_signal'] = 1
                        df.loc[df.index[i], 'bos_level'] = last_high_val
                        df.loc[df.index[i], 'bos_strength'] = strength

            # 检测下降 BoS (跌破前一个波段低点)
            if len(lows) >= 2:
                prev_low_idx, prev_low_val = lows[-2]
                last_low_idx, last_low_val = lows[-1]

                # 确认跌破: 收盘价跌破 + 成交量放大
                if current_close < last_low_val and not pd.isna(vol_ma) and vol_ma > 0:
                    if current_vol > vol_ma:  # 成交量确认
                        strength = min(100, ((last_low_val - current_close) / last_low_val) * 100 * 10)
                        df.loc[df.index[i], 'bos_signal'] = -1
                        df.loc[df.index[i], 'bos_level'] = last_low_val
                        df.loc[df.index[i], 'bos_strength'] = strength

    except Exception:
        pass

    return df


def calc_change_of_character(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    检测趋势性质改变 (Change of Character)

    ChoCh 定义:
    - 上升趋势中出现 Lower Low (LL) → 可能转为下降趋势
    - 下降趋势中出现 Higher High (HH) → 可能转为上升趋势

    参数:
    - lookback: 20 天 (保守模式)

    返回:
    - choch_signal: 1 (bullish ChoCh), -1 (bearish ChoCh), 0 (no ChoCh)
    - choch_strength: 信号强度 (0-100)
    - choch_trend: 当前趋势 ('uptrend', 'downtrend', 'sideways')
    """
    df = df.copy()
    df['choch_signal'] = 0
    df['choch_strength'] = 0
    df['choch_trend'] = 'sideways'

    try:
        if len(df) < lookback + 10:
            return df

        # 计算 ADX 和 MA 用于趋势识别
        if 'ADX' not in df.columns or 'MA5' not in df.columns or 'MA20' not in df.columns:
            return df

        # 计算成交量均线用于确认
        vol_ma20 = df['volume'].rolling(20).mean()

        # 计算 ATR 用于标准化强度
        if 'ATR' not in df.columns:
            return df

        # 遍历最近的数据点
        for i in range(lookback + 5, len(df)):
            window = df.iloc[i-lookback:i+1]
            close_series = window['close'].reset_index(drop=True)

            # 识别当前趋势
            adx = df.iloc[i]['ADX']
            ma5 = df.iloc[i]['MA5']
            ma20 = df.iloc[i]['MA20']

            if pd.isna(adx) or pd.isna(ma5) or pd.isna(ma20):
                continue

            trend = 'sideways'
            if adx > 25:
                if ma5 > ma20:
                    trend = 'uptrend'
                elif ma5 < ma20:
                    trend = 'downtrend'

            df.loc[df.index[i], 'choch_trend'] = trend

            # 找到波段高低点
            swing_points = _find_swing_points(close_series, order=3)
            highs = [(idx, val) for idx, val, typ in swing_points if typ == 'high']
            lows = [(idx, val) for idx, val, typ in swing_points if typ == 'low']

            current_vol = df.iloc[i]['volume']
            vol_ma = vol_ma20.iloc[i]
            atr = df.iloc[i]['ATR']

            if pd.isna(vol_ma) or pd.isna(atr) or vol_ma == 0 or atr == 0:
                continue

            # 上升趋势中检测 Lower Low (看空 ChoCh)
            if trend == 'uptrend' and len(lows) >= 2:
                prev_low_idx, prev_low_val = lows[-2]
                last_low_idx, last_low_val = lows[-1]

                # Lower Low 形成
                if last_low_val < prev_low_val:
                    # 计算强度 (相对 ATR 标准化)
                    ll_magnitude = (prev_low_val - last_low_val) / atr
                    strength = min(100, ll_magnitude * 100)

                    # 成交量确认
                    if current_vol > vol_ma and strength > 50:
                        df.loc[df.index[i], 'choch_signal'] = -1
                        df.loc[df.index[i], 'choch_strength'] = strength

            # 下降趋势中检测 Higher High (看多 ChoCh)
            if trend == 'downtrend' and len(highs) >= 2:
                prev_high_idx, prev_high_val = highs[-2]
                last_high_idx, last_high_val = highs[-1]

                # Higher High 形成
                if last_high_val > prev_high_val:
                    # 计算强度 (相对 ATR 标准化)
                    hh_magnitude = (last_high_val - prev_high_val) / atr
                    strength = min(100, hh_magnitude * 100)

                    # 成交量确认
                    if current_vol > vol_ma and strength > 50:
                        df.loc[df.index[i], 'choch_signal'] = 1
                        df.loc[df.index[i], 'choch_strength'] = strength

    except Exception:
        pass

    return df


def calc_triple_divergence(df: pd.DataFrame, lookback: int = 30) -> dict:
    """三重背离检测 — RSI、OBV、MACD柱状图与价格的背离。"""
    result = {
        'rsi_divergence': 'none', 'obv_divergence': 'none', 'macd_divergence': 'none',
        'rsi_divergence_text': '无', 'obv_divergence_text': '无', 'macd_divergence_text': '无',
        'divergence_count': 0, 'bearish_div_count': 0,
        'signal': 'none', 'signal_text': '无背离信号',
        'score': 50, 'details': []
    }
    try:
        if len(df) < lookback + 5:
            return result

        tail = df.tail(lookback).reset_index(drop=True)
        close = tail['close']

        # 找摆动低点和高点
        swing_points = _find_swing_points(close, order=2)
        lows = [(i, v) for i, v, t in swing_points if t == 'low']
        highs = [(i, v) for i, v, t in swing_points if t == 'high']

        def _check_bullish_div(indicator_series, lows_list):
            """检查看多背离: 价格更低低点 + 指标更高低点"""
            if len(lows_list) < 2:
                return 'none'
            i1, p1 = lows_list[-2]
            i2, p2 = lows_list[-1]
            ind1 = indicator_series.iloc[i1]
            ind2 = indicator_series.iloc[i2]
            if pd.isna(ind1) or pd.isna(ind2):
                return 'none'
            if p2 < p1 and ind2 > ind1:
                return 'bullish'
            return 'none'

        def _check_bearish_div(indicator_series, highs_list):
            """检查看空背离: 价格更高高点 + 指标更低高点"""
            if len(highs_list) < 2:
                return 'none'
            i1, p1 = highs_list[-2]
            i2, p2 = highs_list[-1]
            ind1 = indicator_series.iloc[i1]
            ind2 = indicator_series.iloc[i2]
            if pd.isna(ind1) or pd.isna(ind2):
                return 'none'
            if p2 > p1 and ind2 < ind1:
                return 'bearish'
            return 'none'

        indicators_map = {
            'rsi': tail['RSI'] if 'RSI' in tail.columns else None,
            'obv': tail['OBV'] if 'OBV' in tail.columns else None,
            'macd': tail['MACD_hist'] if 'MACD_hist' in tail.columns else None,
        }

        bullish_count = 0
        bearish_count = 0
        text_map = {'rsi': 'RSI', 'obv': 'OBV', 'macd': 'MACD柱'}

        for key, series in indicators_map.items():
            if series is None or series.isna().all():
                continue
            bull = _check_bullish_div(series, lows)
            bear = _check_bearish_div(series, highs)
            if bull == 'bullish':
                result[f'{key}_divergence'] = 'bullish'
                result[f'{key}_divergence_text'] = f'{text_map[key]}看多背离 ▲'
                result['details'].append(f'{text_map[key]}: 价格创新低但{text_map[key]}未创新低')
                bullish_count += 1
            elif bear == 'bearish':
                result[f'{key}_divergence'] = 'bearish'
                result[f'{key}_divergence_text'] = f'{text_map[key]}看空背离 ▼'
                result['details'].append(f'{text_map[key]}: 价格创新高但{text_map[key]}未创新高')
                bearish_count += 1

        result['divergence_count'] = bullish_count
        result['bearish_div_count'] = bearish_count

        # 综合信号
        if bullish_count >= 3:
            result.update(signal='triple_bullish', score=95,
                          signal_text=f'三重看多背离(RSI+OBV+MACD)')
        elif bullish_count == 2:
            result.update(signal='double_bullish', score=80,
                          signal_text=f'双重看多背离')
        elif bullish_count == 1:
            result.update(signal='single_bullish', score=65,
                          signal_text=f'单一看多背离')
        elif bearish_count >= 3:
            result.update(signal='triple_bearish', score=5,
                          signal_text=f'三重看空背离(RSI+OBV+MACD)')
        elif bearish_count == 2:
            result.update(signal='double_bearish', score=15,
                          signal_text=f'双重看空背离')
        elif bearish_count == 1:
            result.update(signal='single_bearish', score=35,
                          signal_text=f'单一看空背离')
    except Exception:
        pass
    return result


def generate_contrarian_caveats(contrarian: dict, df: pd.DataFrame) -> list:
    """检测左侧指标可能失效的场景，生成风险警告。"""
    caveats = []
    try:
        # ATR持续扩张 + 价格持续下跌
        vr = contrarian.get('volatility_regime', {})
        if vr.get('atr_percentile', 50) > 90:
            recent_chg = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0
            if recent_chg < -5:
                caveats.append('⚠ 趋势性下跌中(ATR>90%+近5日跌>5%)，均值回归可能失效')

        # 连续大跌
        if len(df) >= 5:
            big_drops = sum(1 for i in range(-5, 0)
                           if (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1] * 100 < -5)
            if big_drops >= 3:
                caveats.append('⚠ 恐慌性抛售(近5日有3日跌幅>5%)，左侧抄底风险极高')

        # Z-Score极端恶化
        zs = contrarian.get('zscore', {})
        if zs.get('sufficient_data') and zs.get('zscore', 0) < -3.0:
            caveats.append('⚠ Z-Score极端偏离(<-3.0)，可能反映基本面恶化而非单纯超卖')
    except Exception:
        pass
    return caveats


def calc_contrarian_signals(df: pd.DataFrame, account_size: float = DEFAULT_ACCOUNT_SIZE) -> dict:
    """左侧交易信号综合分析 — 聚合5个模块，生成综合评分和仓位建议。

    仓位计算使用波动率平价模型，不再使用固定百分比。
    """
    zscore = calc_mean_reversion_zscore(df)
    vol_exhaust = calc_volume_exhaustion(df)
    vol_regime = calc_volatility_regime(df)
    sr_inst = find_support_resistance_institutional(df)
    divergence = calc_triple_divergence(df)

    # 加权聚合
    composite = (
        zscore['score'] * 0.25 +
        divergence['score'] * 0.25 +
        vol_exhaust['score'] * 0.20 +
        vol_regime['score'] * 0.15 +
        sr_inst['score'] * 0.15
    )
    composite = round(composite)

    # 信号判定
    if composite >= 80:
        signal = 'strong_contrarian_buy'
        signal_text = '强烈左侧买入信号'
    elif composite >= 65:
        signal = 'contrarian_buy'
        signal_text = '左侧买入信号'
    elif composite <= 20:
        signal = 'contrarian_sell'
        signal_text = '左侧卖出信号'
    elif composite <= 35:
        signal = 'mild_contrarian_sell'
        signal_text = '偏空，不宜左侧抄底'
    else:
        signal = 'neutral'
        signal_text = '左侧信号中性，暂无明确机会'

    # 仓位建议（使用波动率平价模型）
    position_advice = {
        'position_info': None,          # 波动率平价仓位信息
        'chandelier_info': None,        # Chandelier Exit 信息
        'confirm_conditions': [],       # 右侧确认条件
        'action': 'hold'                # 建议操作
    }

    # 计算止损价（使用 Chandelier Exit）
    current_price = df['close'].iloc[-1]
    chandelier = calc_chandelier_exit(df)
    stop_loss = chandelier.get('long_stop')

    if signal in ['strong_contrarian_buy', 'contrarian_buy'] and stop_loss:
        # 计算仓位
        position_info = calc_position_sizing(current_price, stop_loss, account_size)
        position_advice['position_info'] = position_info
        position_advice['chandelier_info'] = chandelier

        if signal == 'strong_contrarian_buy':
            position_advice['confirm_conditions'] = ['MACD金叉', '站上5日线', '放量突破']
            position_advice['action'] = 'buy'
        else:
            position_advice['confirm_conditions'] = ['KDJ金叉', '站上10日线']
            position_advice['action'] = 'buy_cautious'
    else:
        position_advice['action'] = 'hold'
        position_advice['position_info'] = None

    # 风险警告
    result = {
        'zscore': zscore, 'volume_exhaustion': vol_exhaust,
        'volatility_regime': vol_regime, 'sr_institutional': sr_inst,
        'divergence': divergence, 'composite_score': composite,
        'signal': signal, 'signal_text': signal_text,
        'position_advice': position_advice
    }
    result['caveats'] = generate_contrarian_caveats(result, df)
    return result

def analyze_trend(df: pd.DataFrame) -> str:
    """分析当前趋势"""
    if len(df) < 20:
        return "数据不足"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # MA排列判断
    ma5_up = last['MA5'] > prev['MA5']
    ma10_up = last['MA10'] > prev['MA10']

    # 多头排列：MA5 > MA10 > MA20
    if last['MA5'] > last['MA10'] > last['MA20']:
        if ma5_up and ma10_up:
            return "强势上涨"
        return "上涨趋势"
    # 空头排列：MA5 < MA10 < MA20
    elif last['MA5'] < last['MA10'] < last['MA20']:
        if not ma5_up and not ma10_up:
            return "强势下跌"
        return "下跌趋势"
    # 交叉震荡
    else:
        return "震荡整理"


def detect_signals(df: pd.DataFrame) -> dict:
    """检测买卖信号（所有指标）"""
    signals = {
        'buy': [],
        'sell': [],
        'golden_cross': [],
        'death_cross': []
    }

    if len(df) < 30:
        return signals

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # MA金叉死叉
    if pd.notna(prev['MA5']) and pd.notna(prev['MA10']) and pd.notna(last['MA5']) and pd.notna(last['MA10']):
        if prev['MA5'] <= prev['MA10'] and last['MA5'] > last['MA10']:
            signals['golden_cross'].append('MA5上穿MA10')
            signals['buy'].append('MA金叉')
        if prev['MA5'] >= prev['MA10'] and last['MA5'] < last['MA10']:
            signals['death_cross'].append('MA5下穿MA10')
            signals['sell'].append('MA死叉')

    # MACD金叉死叉
    if pd.notna(prev['MACD']) and pd.notna(prev['MACD_signal']) and pd.notna(last['MACD']) and pd.notna(last['MACD_signal']):
        if prev['MACD'] <= prev['MACD_signal'] and last['MACD'] > last['MACD_signal']:
            signals['golden_cross'].append('MACD金叉')
            signals['buy'].append('MACD金叉')
        if prev['MACD'] >= prev['MACD_signal'] and last['MACD'] < last['MACD_signal']:
            signals['death_cross'].append('MACD死叉')
            signals['sell'].append('MACD死叉')

    # RSI超买超卖
    if pd.notna(last['RSI']):
        if last['RSI'] < 30:
            signals['buy'].append('RSI超卖')
        elif last['RSI'] > 70:
            signals['sell'].append('RSI超买')

    # 布林带
    if pd.notna(last['BB_lower']) and pd.notna(last['BB_upper']):
        if last['close'] < last['BB_lower']:
            signals['buy'].append('跌破下轨(可能反弹)')
        elif last['close'] > last['BB_upper']:
            signals['sell'].append('突破上轨(可能回调)')

    # ========== 新增指标信号 ==========

    # KDJ 已移除（长线投资不需要）

    # ADX趋势强度信号
    if pd.notna(last['ADX']) and pd.notna(last['ADX_PDI']) and pd.notna(last['ADX_NDI']):
        if last['ADX'] > 25:  # 趋势明显
            if last['ADX_PDI'] > last['ADX_NDI']:
                signals['buy'].append('ADX多头趋势')
            else:
                signals['sell'].append('ADX空头趋势')

    # CCI信号
    if pd.notna(last['CCI']) and pd.notna(prev['CCI']):
        if last['CCI'] < -100:
            signals['buy'].append('CCI超卖')
        elif last['CCI'] > 100:
            signals['sell'].append('CCI超买')
        # CCI穿越
        if prev['CCI'] <= -100 and last['CCI'] > -100:
            signals['buy'].append('CCI上穿-100')
        if prev['CCI'] >= 100 and last['CCI'] < 100:
            signals['sell'].append('CCI下穿100')

    # SuperTrend信号
    if pd.notna(last['SuperTrend_dir']) and pd.notna(prev['SuperTrend_dir']):
        if last['SuperTrend_dir'] == 1 and prev['SuperTrend_dir'] == -1:
            signals['buy'].append('SuperTrend转多')
        elif last['SuperTrend_dir'] == -1 and prev['SuperTrend_dir'] == 1:
            signals['sell'].append('SuperTrend转空')

    # PSAR信号
    if pd.notna(last['PSAR_dir']) and pd.notna(prev['PSAR_dir']):
        if last['PSAR_dir'] == 1 and prev['PSAR_dir'] == -1:
            signals['buy'].append('PSAR转多')
        elif last['PSAR_dir'] == -1 and prev['PSAR_dir'] == 1:
            signals['sell'].append('PSAR转空')

    # OBV信号
    if pd.notna(last['OBV']) and pd.notna(last['OBV_MA']) and pd.notna(prev['OBV']) and pd.notna(prev['OBV_MA']):
        if prev['OBV'] <= prev['OBV_MA'] and last['OBV'] > last['OBV_MA']:
            signals['buy'].append('OBV上穿均线')
        elif prev['OBV'] >= prev['OBV_MA'] and last['OBV'] < last['OBV_MA']:
            signals['sell'].append('OBV下穿均线')

    # Ichimoku云图信号
    if pd.notna(last['ICH_TENKAN']) and pd.notna(last['ICH_KIJUN']) and pd.notna(prev['ICH_TENKAN']) and pd.notna(prev['ICH_KIJUN']):
        # 转换线与基准线交叉
        if prev['ICH_TENKAN'] <= prev['ICH_KIJUN'] and last['ICH_TENKAN'] > last['ICH_KIJUN']:
            signals['buy'].append('一目均衡表金叉')
        elif prev['ICH_TENKAN'] >= prev['ICH_KIJUN'] and last['ICH_TENKAN'] < last['ICH_KIJUN']:
            signals['sell'].append('一目均衡表死叉')
        # 价格与云的关系
        if pd.notna(last['ICH_SSA']) and pd.notna(last['ICH_SSB']):
            cloud_top = max(last['ICH_SSA'], last['ICH_SSB'])
            cloud_bottom = min(last['ICH_SSA'], last['ICH_SSB'])
            if last['close'] > cloud_top:
                signals['buy'].append('价格在云上方')
            elif last['close'] < cloud_bottom:
                signals['sell'].append('价格在云下方')

    return signals


def analyze_indicator_signals(df: pd.DataFrame) -> Dict[str, dict]:
    """分析各指标单独的信号状态"""
    if len(df) < 30:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    indicators = {}

    # MA
    if pd.notna(last['MA5']) and pd.notna(last['MA10']) and pd.notna(last['MA20']):
        if last['MA5'] > last['MA10'] > last['MA20']:
            ma_signal = 'buy'
            ma_status = '多头排列'
        elif last['MA5'] < last['MA10'] < last['MA20']:
            ma_signal = 'sell'
            ma_status = '空头排列'
        else:
            ma_signal = 'neutral'
            ma_status = '纠缠震荡'
        indicators['MA'] = {'signal': ma_signal, 'status': ma_status, 'weight': 1}

    # RSI
    if pd.notna(last['RSI']):
        if last['RSI'] < 30:
            rsi_signal = 'buy'
            rsi_status = f'超卖区({last["RSI"]:.1f})'
        elif last['RSI'] > 70:
            rsi_signal = 'sell'
            rsi_status = f'超买区({last["RSI"]:.1f})'
        else:
            rsi_signal = 'neutral'
            rsi_status = f'中性区({last["RSI"]:.1f})'
        indicators['RSI'] = {'signal': rsi_signal, 'status': rsi_status, 'weight': 1}

    # MACD (辅助指标，仅用于背离检测，决策权重降低)
    if pd.notna(last['MACD']) and pd.notna(last['MACD_signal']):
        # 仅在极端值时输出信号，平时保持中性
        macd_hist = last['MACD_hist']
        if macd_hist > 0 and abs(macd_hist) > abs(df['MACD_hist'].tail(20).mean()) * 2:
            macd_signal = 'buy'
            macd_status = f'强势多头(hist={macd_hist:.2f})'
        elif macd_hist < 0 and abs(macd_hist) > abs(df['MACD_hist'].tail(20).mean()) * 2:
            macd_signal = 'sell'
            macd_status = f'强势空头(hist={macd_hist:.2f})'
        else:
            macd_signal = 'neutral'
            macd_status = f'震荡运行(hist={macd_hist:.2f})'
        indicators['MACD'] = {'signal': macd_signal, 'status': macd_status, 'weight': 1, 'is_auxiliary': True}

    # KDJ 已移除（长线投资不需要短期噪音指标）

    # ADX
    if pd.notna(last['ADX']) and pd.notna(last['ADX_PDI']) and pd.notna(last['ADX_NDI']):
        trend_strength = '强' if last['ADX'] > 25 else '弱'
        if last['ADX_PDI'] > last['ADX_NDI']:
            adx_signal = 'buy'
            adx_status = f'多头{trend_strength}趋势(ADX={last["ADX"]:.1f})'
        elif last['ADX_PDI'] < last['ADX_NDI']:
            adx_signal = 'sell'
            adx_status = f'空头{trend_strength}趋势(ADX={last["ADX"]:.1f})'
        else:
            adx_signal = 'neutral'
            adx_status = f'无方向(ADX={last["ADX"]:.1f})'
        indicators['ADX'] = {'signal': adx_signal, 'status': adx_status, 'weight': 2}

    # CCI
    if pd.notna(last['CCI']):
        if last['CCI'] < -100:
            cci_signal = 'buy'
            cci_status = f'超卖区({last["CCI"]:.1f})'
        elif last['CCI'] > 100:
            cci_signal = 'sell'
            cci_status = f'超买区({last["CCI"]:.1f})'
        else:
            cci_signal = 'neutral'
            cci_status = f'中性区({last["CCI"]:.1f})'
        indicators['CCI'] = {'signal': cci_signal, 'status': cci_status, 'weight': 1}

    # SuperTrend
    if pd.notna(last['SuperTrend_dir']):
        if last['SuperTrend_dir'] == 1:
            st_signal = 'buy'
            st_status = f'多头趋势(支撑:{last["SuperTrend"]:.2f})'
        else:
            st_signal = 'sell'
            st_status = f'空头趋势(压力:{last["SuperTrend"]:.2f})'
        indicators['SuperTrend'] = {'signal': st_signal, 'status': st_status, 'weight': 2}

    # PSAR
    if pd.notna(last['PSAR_dir']):
        if last['PSAR_dir'] == 1:
            psar_signal = 'buy'
            psar_status = f'多头止损:{last["PSAR"]:.2f}'
        else:
            psar_signal = 'sell'
            psar_status = f'空头止损:{last["PSAR"]:.2f}'
        indicators['PSAR'] = {'signal': psar_signal, 'status': psar_status, 'weight': 1}

    # OBV
    if pd.notna(last['OBV']) and pd.notna(last['OBV_MA']):
        if last['OBV'] > last['OBV_MA']:
            obv_signal = 'buy'
            obv_status = '资金流入'
        else:
            obv_signal = 'sell'
            obv_status = '资金流出'
        indicators['OBV'] = {'signal': obv_signal, 'status': obv_status, 'weight': 1}

    # Ichimoku
    if pd.notna(last['ICH_TENKAN']) and pd.notna(last['ICH_KIJUN']) and pd.notna(last['ICH_SSA']) and pd.notna(last['ICH_SSB']):
        cloud_top = max(last['ICH_SSA'], last['ICH_SSB'])
        cloud_bottom = min(last['ICH_SSA'], last['ICH_SSB'])

        if last['close'] > cloud_top:
            if last['ICH_TENKAN'] > last['ICH_KIJUN']:
                ichi_signal = 'buy'
                ichi_status = '云上多头'
            else:
                ichi_signal = 'neutral'
                ichi_status = '云上回调'
        elif last['close'] < cloud_bottom:
            if last['ICH_TENKAN'] < last['ICH_KIJUN']:
                ichi_signal = 'sell'
                ichi_status = '云下空头'
            else:
                ichi_signal = 'neutral'
                ichi_status = '云下反弹'
        else:
            ichi_signal = 'neutral'
            ichi_status = '云中震荡'
        indicators['Ichimoku'] = {'signal': ichi_signal, 'status': ichi_status, 'weight': 2}

    # 布林带
    if pd.notna(last['BB_upper']) and pd.notna(last['BB_lower']):
        bb_width = (last['BB_upper'] - last['BB_lower']) / last['BB_middle'] * 100
        if last['close'] < last['BB_lower']:
            bb_signal = 'buy'
            bb_status = f'下轨下方(带宽:{bb_width:.1f}%)'
        elif last['close'] > last['BB_upper']:
            bb_signal = 'sell'
            bb_status = f'上轨上方(带宽:{bb_width:.1f}%)'
        else:
            bb_signal = 'neutral'
            bb_status = f'轨道内(带宽:{bb_width:.1f}%)'
        indicators['BOLL'] = {'signal': bb_signal, 'status': bb_status, 'weight': 1}

    return indicators


def calculate_resonance(indicators: Dict[str, dict]) -> dict:
    """计算多指标共振信号

    辅助指标 (is_auxiliary=True) 不计入主共振计算，仅作为辅助信息记录
    """
    buy_score = 0
    sell_score = 0
    total_weight = 0
    buy_indicators = []
    sell_indicators = []
    neutral_indicators = []
    auxiliary_buy = []
    auxiliary_sell = []

    for name, info in indicators.items():
        weight = info.get('weight', 1)
        is_auxiliary = info.get('is_auxiliary', False)

        if is_auxiliary:
            # 辅助指标不计入主共振，仅记录
            if info['signal'] == 'buy':
                auxiliary_buy.append(name)
            elif info['signal'] == 'sell':
                auxiliary_sell.append(name)
            continue

        total_weight += weight
        if info['signal'] == 'buy':
            buy_score += weight
            buy_indicators.append(name)
        elif info['signal'] == 'sell':
            sell_score += weight
            sell_indicators.append(name)
        else:
            neutral_indicators.append(name)

    # 计算共振比例
    buy_ratio = buy_score / total_weight if total_weight > 0 else 0
    sell_ratio = sell_score / total_weight if total_weight > 0 else 0

    # 判断共振信号
    if buy_ratio >= 0.7:
        resonance = 'strong_buy'
        signal_text = '强烈买入共振'
    elif buy_ratio >= 0.5:
        resonance = 'buy'
        signal_text = '买入共振'
    elif sell_ratio >= 0.7:
        resonance = 'strong_sell'
        signal_text = '强烈卖出共振'
    elif sell_ratio >= 0.5:
        resonance = 'sell'
        signal_text = '卖出共振'
    else:
        resonance = 'neutral'
        signal_text = '多空分歧'

    return {
        'resonance': resonance,
        'signal_text': signal_text,
        'buy_ratio': buy_ratio,
        'sell_ratio': sell_ratio,
        'buy_indicators': buy_indicators,
        'sell_indicators': sell_indicators,
        'neutral_indicators': neutral_indicators,
        'buy_score': buy_score,
        'sell_score': sell_score,
        'total_weight': total_weight,
        'auxiliary_buy': auxiliary_buy,  # 辅助指标买入信号
        'auxiliary_sell': auxiliary_sell  # 辅助指标卖出信号
    }


def calculate_win_rate(indicators: Dict[str, dict], resonance: dict, df: pd.DataFrame) -> dict:
    """计算综合胜率提示"""
    # 基于历史回测的胜率权重（模拟数据，实际应基于回测）
    win_rate_weights = {
        'MACD': {'base': 65, 'trend': 0.8},
        'RSI': {'base': 55, 'trend': 0.5},
        'KDJ': {'base': 55, 'trend': 0.5},
        'ADX': {'base': 60, 'trend': 0.7},
        'SuperTrend': {'base': 65, 'trend': 0.85},
        'PSAR': {'base': 60, 'trend': 0.75},
        'Ichimoku': {'base': 62, 'trend': 0.8},
        'CCI': {'base': 55, 'trend': 0.5},
        'OBV': {'base': 50, 'trend': 0.5},
        'BOLL': {'base': 52, 'trend': 0.5},
        'MA': {'base': 55, 'trend': 0.6},
    }

    # 计算趋势强度（基于ADX）
    trend_strength = 0.5
    if 'ADX' in indicators and pd.notna(df.iloc[-1]['ADX']):
        adx_value = df.iloc[-1]['ADX']
        trend_strength = min(adx_value / 50, 1.0)  # 归一化

    # 计算综合胜率
    total_rate = 0
    total_weight = 0

    for name, info in indicators.items():
        if name in win_rate_weights:
            weight = info.get('weight', 1)
            base_rate = win_rate_weights[name]['base']
            trend_factor = win_rate_weights[name]['trend']

            # 趋势行情中提高趋势指标的胜率
            adjusted_rate = base_rate + (trend_strength * trend_factor * 10)

            if info['signal'] == resonance['resonance'].replace('strong_', '').replace('_', ''):
                total_rate += adjusted_rate * weight
            else:
                total_rate += adjusted_rate * weight * 0.8  # 降低非共振指标权重

            total_weight += weight

    # 最终胜率计算
    if total_weight > 0:
        final_rate = total_rate / total_weight
        # 根据共振程度调整
        if resonance['resonance'] in ['strong_buy', 'strong_sell']:
            final_rate *= 1.1  # 强共振提升胜率
        elif resonance['resonance'] == 'neutral':
            final_rate *= 0.8  # 分歧降低胜率
    else:
        final_rate = 50

    final_rate = min(max(final_rate, 30), 85)  # 限制在合理范围

    # 风险提示
    risk_warning = []
    if resonance['resonance'] == 'neutral':
        risk_warning.append('多空分歧较大，建议观望')
    if len(resonance['buy_indicators']) > 0 and len(resonance['sell_indicators']) > 0:
        risk_warning.append(f"买入指标({len(resonance['buy_indicators'])})与卖出指标({len(resonance['sell_indicators'])})冲突")
    if trend_strength < 0.3:
        risk_warning.append('趋势不明朗，震荡行情')

    # 操作建议
    if resonance['resonance'] in ['strong_buy', 'buy']:
        suggestion = '建议：可考虑逢低建仓，设置止损'
    elif resonance['resonance'] in ['strong_sell', 'sell']:
        suggestion = '建议：可考虑减仓或观望，控制风险'
    else:
        suggestion = '建议：观望为主，等待明确信号'

    return {
        'win_rate': final_rate,
        'trend_strength': trend_strength,
        'risk_warnings': risk_warning,
        'suggestion': suggestion
    }


def find_support_resistance(df: pd.DataFrame) -> Tuple[list, list]:
    """寻找支撑位和压力位"""
    if len(df) < 20:
        return [], []

    recent = df.tail(20)

    # 支撑位：近期低点
    support_levels = []
    lows = recent['low'].nsmallest(3)
    for low in lows:
        support_levels.append(round(low, 2))

    # 压力位：近期高点
    resistance_levels = []
    highs = recent['high'].nlargest(3)
    for high in highs:
        resistance_levels.append(round(high, 2))

    # 去重并排序
    support_levels = sorted(list(set(support_levels)))
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)

    return support_levels[:2], resistance_levels[:2]


def generate_summary(df: pd.DataFrame, trend: str, signals: dict,
                     support: list, resistance: list, resonance: dict = None) -> str:
    """生成一句话总结"""
    last = df.iloc[-1]

    parts = []

    # 共振信号
    if resonance:
        res_icon = {'strong_buy': '▲▲', 'buy': '▲', 'neutral': '○',
                    'sell': '▼', 'strong_sell': '▼▼'}
        icon = res_icon.get(resonance['resonance'], '○')
        parts.append(f"{icon}{resonance['signal_text']}")

    # 趋势
    parts.append(f"当前{trend}")

    # 信号
    if signals['buy']:
        parts.append(f"买入信号：{'+'.join(signals['buy'][:3])}")
    if signals['sell']:
        parts.append(f"卖出信号：{'+'.join(signals['sell'][:3])}")

    # RSI状态
    rsi = last['RSI']
    if pd.notna(rsi):
        if rsi < 30:
            parts.append(f"RSI={rsi:.1f}超卖")
        elif rsi > 70:
            parts.append(f"RSI={rsi:.1f}超买")
        else:
            parts.append(f"RSI={rsi:.1f}")

    # 支撑压力
    if support:
        parts.append(f"支撑{support[0]}")
    if resistance:
        parts.append(f"压力{resistance[0]}")

    return " | ".join(parts)


def format_volume(volume) -> str:
    """格式化成交量"""
    if volume >= 1e8:
        return f"{volume/1e8:.2f}亿"
    elif volume >= 1e4:
        return f"{volume/1e4:.2f}万"
    else:
        return f"{volume:.0f}"


def format_table(rows: List[Tuple[str, str, str]], headers: Tuple[str, str, str] = ("指标", "最新值", "信号")) -> str:
    """格式化表格输出

    Args:
        rows: [(指标名, 值, 信号状态), ...]
        headers: 表头

    Returns:
        格式化的表格字符串
    """
    # 计算每列宽度
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # 构建表格
    def make_border():
        parts = ["┌"]
        for i, w in enumerate(col_widths):
            parts.append("─" * (w + 2))
            if i < len(col_widths) - 1:
                parts.append("┬")
        parts.append("┐")
        return "".join(parts)

    def make_row(cells):
        parts = ["│"]
        for i, (cell, w) in enumerate(zip(cells, col_widths)):
            parts.append(f" {str(cell):{w}} ")
            if i < len(cells) - 1:
                parts.append("│")
        parts.append("│")
        return "".join(parts)

    def make_middle_border():
        parts = ["├"]
        for i, w in enumerate(col_widths):
            parts.append("─" * (w + 2))
            if i < len(col_widths) - 1:
                parts.append("┼")
        parts.append("┤")
        return "".join(parts)

    def make_bottom_border():
        parts = ["└"]
        for i, w in enumerate(col_widths):
            parts.append("─" * (w + 2))
            if i < len(col_widths) - 1:
                parts.append("┴")
        parts.append("┘")
        return "".join(parts)

    lines = [make_border(), make_row(headers), make_middle_border()]
    for row in rows:
        lines.append(make_row(row))
    lines.append(make_bottom_border())

    return "\n".join(lines)


def build_optional_context(code: str, df: pd.DataFrame, demo: bool = False) -> dict:
    """按需获取扩展信息，失败时只回落到默认值。"""
    context = {
        'market_review': {'indices': {}, 'sectors': [], 'market_status': '已跳过'},
        'chip_dist': {'source': '已跳过'},
        'fund_flow': fetch_fund_flow(code, df),
        'news_sentiment': {'news': [], 'sentiment': '中性', 'sentiment_score': 0, 'summary': '暂无新闻数据', 'source': '为保障流程稳定已跳过'},
        'consensus': {'rating': 'hold', 'rating_text': '暂无评级数据', 'target_price': None, 'analyst_count': 0, 'source': '为保障流程稳定已跳过'},
        'options_data': {'available': False, 'source': '已跳过'},
        'volume_profile': None,
        'macro_vp': None,
        'fundamentals': {'dilution': {}, 'quality': {}, 'delisting': {}, 'value_trap': {}},
        'smart_money': {'insider': {}, 'institutional': {}, 'confirmation': {}},
        'sector': {'rs': {}, 'earnings_vol': {}, 'macro_bear': {}},
        'notes': [],
    }

    if demo:
        context['notes'].append('演示模式已跳过联网扩展数据')
        context['chip_dist'] = {'source': '演示模式已跳过'}
        context['market_review']['market_status'] = '演示模式已跳过'
        context['news_sentiment']['source'] = '演示模式已跳过'
        context['consensus']['source'] = '演示模式已跳过'
        context['options_data']['source'] = '演示模式已跳过'
        # Volume Profile 可以在 demo 模式下计算
        context['volume_profile'] = calc_anchored_volume_profile(df)
        context['macro_vp'] = calc_macro_volume_profile(df)
        return context

    try:
        context['market_review'] = get_market_review()
    except Exception:
        context['notes'].append('大盘复盘获取失败，已跳过')

    try:
        context['chip_dist'] = fetch_chip_distribution(code)
    except Exception:
        context['chip_dist'] = {'source': '获取失败'}

    # 获取期权数据
    try:
        context['options_data'] = fetch_options_data(code)
    except Exception:
        context['options_data'] = {'available': False, 'source': '获取失败'}

    # 计算锚定筹码分布
    try:
        context['volume_profile'] = calc_anchored_volume_profile(df)
    except Exception:
        context['volume_profile'] = None

    # 计算3年宏观筹码分布
    try:
        context['macro_vp'] = calc_macro_volume_profile(df)
    except Exception:
        context['macro_vp'] = None

    # 基本面扫雷（稀释、质量、退市风险）
    if fetch_dilution_analysis is not None:
        try:
            context['fundamentals']['dilution'] = fetch_dilution_analysis(code)
        except Exception:
            pass
    if fetch_quality_metrics is not None:
        try:
            context['fundamentals']['quality'] = fetch_quality_metrics(code)
        except Exception:
            pass
    if check_delisting_risk is not None:
        try:
            context['fundamentals']['delisting'] = check_delisting_risk(code, df)
        except Exception:
            pass

    # 聪明钱动向（内部人士交易 + 机构持股）
    if fetch_insider_transactions is not None:
        try:
            context['smart_money']['insider'] = fetch_insider_transactions(code)
        except Exception:
            pass
    if fetch_institutional_holdings is not None:
        try:
            context['smart_money']['institutional'] = fetch_institutional_holdings(code)
        except Exception:
            pass

    # 行业相对强弱
    if calc_sector_relative_strength is not None:
        try:
            context['sector']['rs'] = calc_sector_relative_strength(code, df)
        except Exception:
            pass

    # 财报波动统计
    if fetch_earnings_volatility is not None:
        try:
            context['sector']['earnings_vol'] = fetch_earnings_volatility(code)
        except Exception:
            pass

    # 宏观结构性熊市检查
    if check_structural_bear_market is not None:
        try:
            context['sector']['macro_bear'] = check_structural_bear_market()
        except Exception:
            pass

    return context


def _print_loss_response(strategy: dict, entry_price: float, current_price: float):
    """打印亏损应对策略"""
    print(f"\n{'='*70}")
    print(f"  ║           持仓应对策略 — 亏损管理                              ║")
    print(f"{'='*70}")

    print(f"\n  【浮亏概览】")
    print(f"    买入均价: {entry_price:.2f}")
    print(f"    当前价格: {current_price:.2f}")
    print(f"    浮亏幅度: {strategy['pnl_pct']:.2f}%")
    print(f"    亏损等级: {strategy['loss_text']}")
    print(f"    趋势状态: {strategy['trend_text']}")

    print(f"\n  【止损距离监控】")
    if strategy['long_stop']:
        print(f"    Chandelier止损价: {strategy['long_stop']:.2f}")
        if strategy['stop_dist_pct'] is not None:
            print(f"    距离止损: {strategy['stop_dist_pct']:.2f}%")
        urgency_map = {
            'safe': '✓ 安全（距离>5%）',
            'warning': '⚠ 警告（距离2%-5%）',
            'critical': '⚠⚠ 危险（距离<2%）',
            'breached': '✗✗ 已击穿',
            'unknown': '— 数据不足'
        }
        print(f"    紧急度: {urgency_map.get(strategy['stop_urgency'], '未知')}")
    else:
        print(f"    止损价: 数据不足")

    print(f"\n  【加仓评估】")
    if strategy['add_ok']:
        print(f"    ✓ 满足加仓条件")
        for cond in strategy['add_conditions']:
            print(f"      {cond}")
        if strategy['add_price'] and strategy['add_shares']:
            print(f"    建议价位: {strategy['add_price']:.2f}")
            print(f"    建议股数: {strategy['add_shares']} 股")
    else:
        print(f"    ✗ 不满足加仓条件")
        for cond in strategy['add_conditions']:
            print(f"      {cond}")

    if strategy['reduce_pct'] > 0:
        print(f"\n  【减仓建议】")
        print(f"    建议减仓: {strategy['reduce_pct']}%")
        print(f"    理由: {strategy['reduce_reason']}")

    print(f"\n  【心理建设】")
    print(f"    认知偏差: {strategy['bias_name']}")
    print(f"    偏差描述: {strategy['bias_desc']}")
    print(f"    理性操作: {strategy['rational_action']}")

    print(f"\n  ── 策略总结 ──")
    if strategy['reduce_pct'] > 0:
        print(f"  当前建议: 减仓 {strategy['reduce_pct']}%")
    elif strategy['add_ok']:
        print(f"  当前建议: 可考虑加仓，但需等待右侧确认")
    else:
        print(f"  当前建议: 持仓观望，严守止损")


def _print_profit_response(strategy: dict, entry_price: float, current_price: float):
    """打印盈利应对策略"""
    print(f"\n{'='*70}")
    print(f"  ║           持仓应对策略 — 盈利管理                              ║")
    print(f"{'='*70}")

    print(f"\n  【浮盈概览】")
    print(f"    买入均价: {entry_price:.2f}")
    print(f"    当前价格: {current_price:.2f}")
    print(f"    浮盈幅度: {strategy['pnl_pct']:.2f}%")
    print(f"    盈利等级: {strategy['profit_text']}")
    print(f"    趋势状态: {strategy['trend_text']}")

    print(f"\n  【分批止盈】")
    if strategy['take_profit_levels']:
        if strategy['progress'] is not None:
            print(f"    目标价进度: {strategy['progress']:.1f}%")
        for tp in strategy['take_profit_levels']:
            status = '✓' if tp['reached'] else '—'
            print(f"    [{status}] {tp['progress']}%进度 @ {tp['price']:.2f} → {tp['action']}")
    else:
        print(f"    目标价数据不足，建议手动设定止盈位")

    if strategy['vwap_alert']:
        print(f"\n    ⚠ VWAP偏离 {strategy['vwap_sigmas']:.1f}σ > 3σ，建议即时减仓")

    print(f"\n  【移动止损】")
    if strategy['long_stop']:
        print(f"    Chandelier止损价: {strategy['long_stop']:.2f}")
        if strategy['locked_profit_pct'] is not None and strategy['locked_profit_pct'] > 0:
            print(f"    已锁定利润: {strategy['locked_profit_pct']:.2f}%")
        else:
            print(f"    已锁定利润: 0% (止损价低于成本)")
    else:
        print(f"    止损价: 数据不足")

    print(f"\n  【派发预警】")
    alert_icon = {'none': '✓', 'watch': '—', 'warning': '⚠', 'critical': '⚠⚠'}.get(strategy['dist_alert'], '—')
    print(f"    [{alert_icon}] 派发评分: {strategy['dist_score']} 分")
    print(f"    信号: {strategy['dist_text']}")
    if strategy['dist_alert'] in ('warning', 'critical'):
        print(f"    → 建议密切关注，考虑分批减仓")

    print(f"\n  【加仓禁区】")
    if strategy['add_banned']:
        print(f"    ✗ 当前处于加仓禁区（触发 {len(strategy['add_ban_reasons'])} 项）")
        for reason in strategy['add_ban_reasons']:
            print(f"      • {reason}")
    else:
        print(f"    ✓ 未触发加仓禁区")

    print(f"\n  ── 策略总结 ──")
    if strategy['dist_alert'] == 'critical':
        print(f"  当前建议: 强烈派发信号，建议减仓或清仓")
    elif strategy['vwap_alert']:
        print(f"  当前建议: VWAP严重偏离，建议即时减仓")
    elif strategy['take_profit_levels']:
        reached = [tp for tp in strategy['take_profit_levels'] if tp['reached']]
        if reached:
            print(f"  当前建议: 已达 {reached[-1]['progress']}% 目标，{reached[-1]['action']}")
        else:
            print(f"  当前建议: 持仓待涨，按计划分批止盈")
    else:
        print(f"  当前建议: 持仓待涨，严守移动止损")


def print_analysis(df: pd.DataFrame, code: str, period: str, demo: bool = False, stress_test: bool = False, entry_price: float = None):
    """打印分析结果（表格格式）

    输出顺序：大盘复盘 → 指标表格 → 筹码/资金流 → 新闻情绪 → 买卖点+清单 → 决策仪表盘 → 持仓应对策略 → 综合结论
    """
    # 计算指标
    df = calculate_indicators(df)
    last = df.iloc[-1]

    # ========== 0. 黑天鹅与熔断检查（最高优先级）==========
    vol_exhaust_early = calc_volume_exhaustion(df)
    hvn_zones_early = vol_exhaust_early.get('hvn_zones', [])
    circuit_breaker = check_circuit_breaker(code, df, hvn_zones_early, demo=demo)

    # 验证指标计算结果
    is_valid, warnings = validate_indicators(df)
    if warnings:
        print("\n注意：部分指标计算异常")
        for w in warnings:
            print(f"  - {w}")

    # 分析
    trend = analyze_trend(df)
    signals = detect_signals(df)
    sr_data = find_support_resistance_institutional(df)
    support = sr_data['support']
    resistance = sr_data['resistance']

    # 多指标分析
    indicators = analyze_indicator_signals(df)
    resonance = calculate_resonance(indicators)
    win_rate_info = calculate_win_rate(indicators, resonance, df)
    context = build_optional_context(code, df, demo=demo)

    # 获取周线确认
    weekly_confirmation = fetch_weekly_confirmation(code, demo=demo)
    df.attrs['weekly_confirmation'] = weekly_confirmation

    # 获取月线确认
    monthly_confirmation = fetch_monthly_confirmation(code, demo=demo)
    df.attrs['monthly_confirmation'] = monthly_confirmation

    # 左侧交易信号分析
    contrarian = calc_contrarian_signals(df)

    # 计算自适应周期参数
    distribution = calc_distribution_patterns(df)
    market_stage = determine_market_stage(resonance, contrarian, distribution, last)
    adaptive_params = calc_adaptive_params(df, market_stage)
    df.attrs['adaptive_params'] = adaptive_params

    # 使用自适应参数重新计算关键指标
    try:
        df = calc_break_of_structure(df, lookback=adaptive_params['bos_lookback'])
    except Exception:
        pass

    try:
        df = calc_change_of_character(df, lookback=adaptive_params['choch_lookback'])
    except Exception:
        pass

    # 重新获取最新数据（因为 BoS/ChoCh 可能已更新）
    last = df.iloc[-1]

    summary = generate_summary(df, trend, signals, support, resistance, resonance)

    # 打印结果
    period_name = "日线" if period == 'd' else "周线"
    print(f"\n{'='*70}")
    print(f"  股票代码: {code} | 周期: {period_name} | 综合技术分析")
    print(f"{'='*70}")
    for note in context['notes']:
        print(f"  说明: {note}")

    # ========== 0. 熔断警告（最高优先级）==========
    if circuit_breaker['any_triggered']:
        print(f"\n{'='*70}")
        print(f"  ║  ⚠⚠⚠  黑天鹅熔断 (Circuit Breaker)  ⚠⚠⚠              ║")
        print(f"{'='*70}")

        if circuit_breaker['macro_triggered']:
            macro = circuit_breaker['macro_data']
            bm = macro.get('benchmark', {})
            vix = macro.get('vix', {})
            print(f"\n  【大盘环境阻断】")
            if bm.get('below_ma200') and bm.get('ma200'):
                print(f"    基准指数: {bm['name']} = {bm['price']:.2f}")
                print(f"    MA200: {bm['ma200']:.2f} (偏离: {bm['deviation_pct']:+.2f}%)")
                print(f"    → 基准指数跌破 200 日均线")
            if vix.get('spike'):
                print(f"    VIX: {vix['current']:.2f} (日内变化: {vix['change_pct']:+.1f}%)")
                print(f"    → VIX 日内飙升超过 15%")
            print(f"\n    ⚠ 宏观风险熔断，全局禁止买入")

        if circuit_breaker['gap_triggered']:
            gap = circuit_breaker['gap_data']
            print(f"\n  【个股跳空毁灭】")
            print(f"    向下跳空: {gap['gap_size']:.2f} ({gap['gap_atr_ratio']:.1f}x ATR)")
            if gap.get('breached_hvn'):
                hvn = gap['breached_hvn']
                print(f"    穿越 HVN: {hvn[0]:.2f}-{hvn[1]:.2f} (量比: {hvn[2]:.1f}x)")
            print(f"\n    ⚠ 逻辑证伪，切勿接飞刀，观望或止损")

        print(f"{'='*70}")

    # ========== 1. 大盘复盘 ==========
    print(f"\n【大盘复盘】")
    market_review = context['market_review']
    if market_review.get('indices'):
        for name, data in market_review['indices'].items():
            change_str = f"+{data['change']:.2f}%" if data['change'] > 0 else f"{data['change']:.2f}%"
            print(f"  {name}: {data['price']} ({change_str})")
        print(f"  市场状态: {market_review.get('market_status', '未知')}")

        if market_review.get('sectors'):
            print(f"\n  热门板块(前5):")
            for sector in market_review['sectors']:
                change_s = f"+{sector['change']:.2f}%" if sector['change'] > 0 else f"{sector['change']:.2f}%"
                inflow_s = f"+{sector['inflow']:.1f}亿" if sector['inflow'] > 0 else f"{sector['inflow']:.1f}亿"
                print(f"    {sector['name']}: {change_s} 主力{inflow_s}")
    else:
        print(f"  暂无法获取大盘数据 ({market_review.get('market_status', '未知')})")

    # ========== 2. 最新数据和技术指标 ==========
    print(f"\n【最新数据】")
    print(f"  日期: {last['date'].strftime('%Y-%m-%d')}")
    print(f"  收盘: {last['close']:.2f}")
    print(f"  成交量: {format_volume(last['volume'])}")
    if pd.notna(last['ATR']):
        print(f"  ATR(14): {last['ATR']:.2f}")

    # 技术指标表格
    print(f"\n【技术指标】")
    table_rows = []

    # 基础指标（长线均线）
    if pd.notna(last['MA50']):
        table_rows.append(("MA50", f"{last['MA50']:.2f}", "长线支撑"))
    if pd.notna(last['MA200']):
        table_rows.append(("MA200", f"{last['MA200']:.2f}", "牛熊分界"))

    # 短期均线（仅供参考）
    if pd.notna(last['MA5']):
        ma_signal = "多头排列" if last['MA5'] > last['MA10'] > last['MA20'] else ("空头排列" if last['MA5'] < last['MA10'] < last['MA20'] else "-")
        table_rows.append(("MA5", f"{last['MA5']:.2f}", ma_signal))
    if pd.notna(last['MA10']):
        table_rows.append(("MA10", f"{last['MA10']:.2f}", "-"))
    if pd.notna(last['MA20']):
        table_rows.append(("MA20", f"{last['MA20']:.2f}", "-"))

    # RSI
    if pd.notna(last['RSI']):
        rsi_signal = "超卖" if last['RSI'] < 30 else ("超买" if last['RSI'] > 70 else "中性")
        table_rows.append(("RSI(14)", f"{last['RSI']:.2f}", rsi_signal))

    # MACD
    if pd.notna(last['MACD']) and pd.notna(last['MACD_signal']):
        macd_signal = "▲ 多头运行" if last['MACD'] > last['MACD_signal'] else "▼ 空头运行"
        table_rows.append(("MACD", f"{last['MACD']:.2f}", macd_signal))

    # 布林带
    if pd.notna(last['BB_upper']) and pd.notna(last['BB_lower']):
        bb_signal = "突破上轨" if last['close'] > last['BB_upper'] else ("跌破下轨" if last['close'] < last['BB_lower'] else "轨道内")
        table_rows.append(("BOLL", f"{last['BB_middle']:.2f}", bb_signal))

    # KDJ
    # KDJ 已移除（长线投资不需要）

    # ADX
    if pd.notna(last['ADX']):
        adx_signal = "多头趋势" if last['ADX_PDI'] > last['ADX_NDI'] else "空头趋势"
        table_rows.append(("ADX", f"{last['ADX']:.1f}", adx_signal))

    # CCI
    if pd.notna(last['CCI']):
        cci_signal = "超买" if last['CCI'] > 100 else ("超卖" if last['CCI'] < -100 else "中性")
        table_rows.append(("CCI", f"{last['CCI']:.1f}", cci_signal))

    # SuperTrend
    if pd.notna(last['SuperTrend']):
        st_signal = "▲ 多头" if last['SuperTrend_dir'] == 1 else "▼ 空头"
        table_rows.append(("SuperTrend", f"{last['SuperTrend']:.2f}", st_signal))

    # PSAR
    if pd.notna(last['PSAR']):
        psar_signal = "▲ 多头" if last['PSAR_dir'] == 1 else "▼ 空头"
        table_rows.append(("PSAR", f"{last['PSAR']:.2f}", psar_signal))

    # OBV
    if pd.notna(last['OBV']):
        obv_signal = "资金流入" if last['OBV'] > last['OBV_MA'] else "资金流出"
        table_rows.append(("OBV", format_volume(last['OBV']), obv_signal))

    # Ichimoku
    if pd.notna(last['ICH_TENKAN']):
        ichi_signal = "云上" if last['close'] > max(last['ICH_SSA'], last['ICH_SSB']) else ("云下" if last['close'] < min(last['ICH_SSA'], last['ICH_SSB']) else "云中")
        table_rows.append(("Ichimoku", f"{last['ICH_TENKAN']:.1f}", ichi_signal))

    print(format_table(table_rows))

    # 多指标共振分析
    print(f"\n【多指标共振】")
    res_icon = {'strong_buy': '▲▲', 'buy': '▲', 'neutral': '○',
                'sell': '▼', 'strong_sell': '▼▼'}
    icon = res_icon.get(resonance['resonance'], '○')
    print(f"  共振信号: {icon} {resonance['signal_text']}")
    print(f"  买入指标({len(resonance['buy_indicators'])}): {', '.join(resonance['buy_indicators']) if resonance['buy_indicators'] else '无'}")
    print(f"  卖出指标({len(resonance['sell_indicators'])}): {', '.join(resonance['sell_indicators']) if resonance['sell_indicators'] else '无'}")

    # 综合胜率提示
    print(f"\n【综合胜率提示】")
    print(f"  预估胜率: {win_rate_info['win_rate']:.1f}%")
    print(f"  趋势强度: {win_rate_info['trend_strength']*100:.0f}%")
    if win_rate_info['risk_warnings']:
        print(f"  风险提示: {'; '.join(win_rate_info['risk_warnings'])}")
    print(f"  {win_rate_info['suggestion']}")

    # 买卖信号表格
    print(f"\n【交叉信号】")
    cross_rows = []
    if signals['golden_cross']:
        for gc in signals['golden_cross']:
            cross_rows.append((gc, "金叉", "▲ 买入"))
    if signals['death_cross']:
        for dc in signals['death_cross']:
            cross_rows.append((dc, "死叉", "▼ 卖出"))
    if cross_rows:
        print(format_table(cross_rows, ("信号", "类型", "方向")))
    else:
        print("  暂无金叉/死叉信号")

    # 支撑压力位
    print(f"\n【支撑压力位】")
    print(f"  支撑位: {', '.join(map(str, support)) if support else '暂无'}")
    print(f"  压力位: {', '.join(map(str, resistance)) if resistance else '暂无'}")

    # ========== 3. 筹码分布 + 主力资金流向 ==========
    print(f"\n【筹码分布与资金流向】")

    # 筹码分布
    chip_dist = context['chip_dist']
    if chip_dist.get('source') == '东方财富筹码分布' and chip_dist.get('price'):
        print(f"  当前股价: {chip_dist['price']:.2f}")
        if chip_dist.get('main_cost_area'):
            print(f"  主力成本区: {chip_dist['main_cost_area']}元区间")
        print(f"  筹码集中度: {chip_dist.get('concentration', '未知')}")
        if chip_dist.get('profit_ratio') is not None:
            print(f"  获利盘: {chip_dist['profit_ratio']:.1f}%")
        if chip_dist.get('float_ratio') is not None:
            print(f"  浮筹比例: 约{chip_dist['float_ratio']:.1f}%")
    else:
        print(f"  筹码分布: {chip_dist.get('source', '获取失败')}")

    # 主力资金流向
    fund_flow = context['fund_flow']
    if fund_flow.get('source') and fund_flow.get('source') != '获取失败':
        inflow_today = fund_flow.get('main_inflow_today')
        inflow_5d = fund_flow.get('main_inflow_5d')
        signal = fund_flow.get('signal', '未知')
        trend_f = fund_flow.get('trend', '观望')
        vol_ratio = fund_flow.get('volume_ratio')

        if inflow_today is not None:
            inflow_str = f"+{inflow_today:.0f}万" if inflow_today > 0 else f"{inflow_today:.0f}万"
            print(f"  当日主力净流入: {inflow_str}")
        if inflow_5d is not None:
            inflow_5d_str = f"+{inflow_5d:.0f}万" if inflow_5d > 0 else f"{inflow_5d:.0f}万"
            print(f"  5日累计净流入: {inflow_5d_str}")
        print(f"  资金动向: {signal} ({trend_f})")
        if vol_ratio:
            print(f"  量比: {vol_ratio:.2f}")
        print(f"  数据来源: {fund_flow.get('source', '技术分析')}")
    else:
        print("  主力资金流: 数据不足")

    # ========== 4. 新闻舆情 ==========
    print(f"\n【市场舆情】")
    news_sentiment = context['news_sentiment']
    if news_sentiment.get('news'):
        print(f"  最新新闻(最近5条):")
        for i, news in enumerate(news_sentiment['news'][:5], 1):
            title = news.get('title', '')[:40]
            source = news.get('source', '未知')
            print(f"    {i}. {title} ({source})")

        sentiment = news_sentiment.get('sentiment', '中性')
        score = news_sentiment.get('sentiment_score', 0)
        summary = news_sentiment.get('summary', '')
        print(f"\n  情绪分析: {sentiment} (评分: {score:+d})")
        print(f"  总结: {summary}")
    else:
        print(f"  暂无新闻数据 ({news_sentiment.get('source', '无数据')})")

    # ========== 5. 分析师共识（机构评级） ==========
    print(f"\n【机构评级】")
    consensus = context['consensus']
    print(f"  机构评级: {consensus['rating_text']}")
    if consensus['target_price']:
        print(f"  目标价: {consensus['target_price']:.2f}")
    if consensus['analyst_count'] > 0:
        print(f"  分析师数量: {consensus['analyst_count']}位")
    print(f"  数据来源: 基于{consensus['source']}汇总")

    # ========== 5.1 锚定筹码分布 (Volume Profile) ==========
    volume_profile = context.get('volume_profile')
    if volume_profile and volume_profile.get('poc'):
        anchor = volume_profile.get('anchor_info', {})
        print(f"\n【锚定筹码分布 (Volume Profile)】")
        print(f"  锚定点: {anchor.get('anchor_details', '未知')} ({anchor.get('anchor_date', '')})")
        print(f"  距今: {anchor.get('days_since_anchor', 0)} 天")
        print(f"  POC (控制点): {volume_profile['poc']:.2f}")
        print(f"  VAH (价值区高点): {volume_profile['vah']:.2f}")
        print(f"  VAL (价值区低点): {volume_profile['val']:.2f}")

        if volume_profile.get('hvn_zones'):
            print(f"  高成交量节点 (HVN):")
            for lo, hi, ratio in volume_profile['hvn_zones'][:3]:
                print(f"    • {lo:.2f} - {hi:.2f} (量比: {ratio:.1f}x)")

        if volume_profile.get('support_levels'):
            print(f"  VP支撑位: {', '.join(f'{s:.2f}' for s in volume_profile['support_levels'])}")
        if volume_profile.get('resistance_levels'):
            print(f"  VP压力位: {', '.join(f'{r:.2f}' for r in volume_profile['resistance_levels'])}")

        print(f"  → {volume_profile.get('interpretation', '')}")

    # ========== 5.2 期权数据 ==========
    options_data = context.get('options_data', {})
    if options_data.get('available'):
        print(f"\n【期权数据分析】")
        if options_data.get('iv') is not None:
            iv_pct = options_data['iv'] * 100
            print(f"  隐含波动率 (IV): {iv_pct:.1f}%")
        if options_data.get('iv_rank') is not None:
            print(f"  IV Rank: {options_data['iv_rank']:.1f}")
        if options_data.get('put_call_ratio') is not None:
            print(f"  P/C Ratio: {options_data['put_call_ratio']:.2f}")
        print(f"  数据来源: {options_data.get('source', 'yfinance')}")

        # 假突破检测
        # 判断当前是否有突破（价格接近近期高点/低点）
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]

        near_high = current_price >= recent_high * 0.98
        near_low = current_price <= recent_low * 1.02

        if near_high:
            fake_out = detect_fake_breakout(df, options_data, 'up')
            if fake_out['is_fake_breakout']:
                print(f"\n  ⚠ 假突破警告: {fake_out['details']}")
        elif near_low:
            fake_out = detect_fake_breakout(df, options_data, 'down')
            if fake_out['is_fake_breakout']:
                print(f"\n  ⚠ 假跌破警告: {fake_out['details']}")
    elif options_data.get('source') and '不支持' not in options_data.get('source', ''):
        print(f"\n【期权数据分析】")
        print(f"  {options_data.get('source', '暂无期权数据')}")

    # ========== 5.5 左侧交易信号 ==========
    print(f"\n{'='*70}")
    print(f"  ║           左侧交易信号 (逆势分析)                            ║")
    print(f"{'='*70}")

    # Z-Score
    zs = contrarian['zscore']
    if zs['sufficient_data']:
        print(f"  均值回归Z-Score: {zs['zscore']:.2f} (偏离MA200: {zs['deviation_pct']:+.1f}%)")
        print(f"    → {zs['signal_text']}")
    else:
        print(f"  均值回归Z-Score: 数据不足(需200日以上)")

    # Triple Divergence
    div = contrarian['divergence']
    print(f"\n  三重背离检测:")
    print(f"    RSI背离: {div['rsi_divergence_text']}")
    print(f"    OBV背离: {div['obv_divergence_text']}")
    print(f"    MACD背离: {div['macd_divergence_text']}")
    print(f"    → {div['signal_text']}")

    # Volume Exhaustion
    vol = contrarian['volume_exhaustion']
    print(f"\n  量能衰竭分析:")
    print(f"    模式: {vol['pattern_text']}")
    if vol['hvn_zones']:
        hvn_str = ', '.join(f"{z[0]:.2f}-{z[1]:.2f}" for z in vol['hvn_zones'][:3])
        print(f"    高量节点: {hvn_str}")
    print(f"    → {vol['details']}")

    # Volatility Regime
    vr = contrarian['volatility_regime']
    print(f"\n  波动率状态:")
    print(f"    ATR百分位: {vr['atr_percentile']:.0f}% | 布林带宽百分位: {vr['bb_width_percentile']:.0f}%")
    print(f"    → {vr['regime_text']}")

    # Institutional S/R
    sr = contrarian['sr_institutional']
    if sr['order_blocks']:
        print(f"\n  机构级支撑压力:")
        for ob in sr['order_blocks'][:3]:
            ob_type = "看多OB" if ob['type'] == 'bullish' else "看空OB"
            print(f"    {ob_type}: {ob['price_low']:.2f}-{ob['price_high']:.2f}")
    if sr['fvg_zones']:
        for fvg in sr['fvg_zones'][:2]:
            fvg_type = "看多FVG" if fvg['type'] == 'bullish' else "看空FVG"
            filled = "(已回补)" if fvg['filled'] else "(未回补)"
            print(f"    {fvg_type}: {fvg['bottom']:.2f}-{fvg['top']:.2f} {filled}")
    if sr['liquidity_sweeps']:
        for ls in sr['liquidity_sweeps'][:2]:
            ls_type = "扫底" if ls['type'] == 'sweep_low' else "扫顶"
            recovered = "已收回" if ls['recovered'] else "未收回"
            print(f"    假破位({ls_type}): {ls['price']:.2f} ({recovered})")

    # Composite Score
    score = contrarian['composite_score']
    score_bar = "█" * (score // 5) + "░" * (20 - score // 5)
    print(f"\n  左侧综合评分: [{score_bar}] {score}分")
    print(f"  信号: {contrarian['signal_text']}")

    # Position Advice (波动率平价模型)
    pa = contrarian['position_advice']
    position_info = pa.get('position_info')
    chandelier_info = pa.get('chandelier_info')

    if position_info and position_info.get('is_valid'):
        print(f"\n  仓位建议 (波动率平价模型):")

        # Chandelier Exit 止损计算
        if chandelier_info and chandelier_info.get('long_stop'):
            print(f"    Chandelier Exit 止损计算:")
            print(f"      区间最高价({CHANDELIER_LOOKBACK}日): {chandelier_info['highest_high']:.2f}")
            print(f"      ATR(14): {chandelier_info['atr']:.2f}")
            print(f"      止损价: {chandelier_info['highest_high']:.2f} - ({chandelier_info['atr_multiplier']} × {chandelier_info['atr']:.2f}) = {position_info['stop_loss']:.2f}")
            print(f"      止损幅度: {position_info['stop_loss_pct']*100:.1f}%")

        # 风险控制参数
        print(f"\n    风险控制参数:")
        print(f"      账户规模: {DEFAULT_ACCOUNT_SIZE:,.0f} 元")
        print(f"      单笔最大风险: {MAX_RISK_PER_TRADE*100:.1f}%")
        print(f"      本次交易最大承担风险金额: {position_info['max_risk_amount']:,.0f} 元")

        # 建议仓位
        print(f"\n    建议仓位:")
        print(f"      建议买入数量: {position_info['suggested_shares']} 股")
        print(f"      仓位市值: {position_info['position_value']:,.0f} 元")
        print(f"      占账户比例: {position_info['position_pct']:.1f}%")

        if pa.get('confirm_conditions'):
            print(f"\n    右侧确认条件: {', '.join(pa['confirm_conditions'])}")

        # 波动率警告
        if position_info.get('volatility_warning'):
            print(f"\n    {position_info['volatility_warning']}")
        if position_info.get('risk_reward_warning'):
            print(f"    {position_info['risk_reward_warning']}")
    else:
        print(f"\n  仓位建议: 暂不建议左侧入场，等待更明确信号")
        if position_info and position_info.get('risk_reward_warning'):
            print(f"    {position_info['risk_reward_warning']}")

    # Risk Caveats
    if contrarian.get('caveats'):
        print(f"\n  风险警告:")
        for caveat in contrarian['caveats']:
            print(f"    {caveat}")

    # ========== 5.6 右侧确认信号 (BoS & ChoCh) ==========
    print(f"\n{'='*70}")
    print(f"  ║           右侧确认信号 (趋势跟随)                            ║")
    print(f"{'='*70}")

    # Break of Structure (BoS)
    bos_signal = last.get('bos_signal', 0)
    bos_level = last.get('bos_level', np.nan)
    bos_strength = last.get('bos_strength', 0)

    print(f"\n  Break of Structure (BoS):")
    if bos_signal == 1:
        print(f"    状态: 看多突破 ✓")
        print(f"    突破价位: {bos_level:.2f}")
        print(f"    突破强度: {bos_strength:.1f}/100")
        print(f"    → 价格突破前一个波段高点，右侧确认信号")
    elif bos_signal == -1:
        print(f"    状态: 看空突破 ✗")
        print(f"    突破价位: {bos_level:.2f}")
        print(f"    突破强度: {bos_strength:.1f}/100")
        print(f"    → 价格跌破前一个波段低点，趋势转弱")
    else:
        print(f"    状态: 无突破")
        print(f"    → 价格未突破关键波段高低点")

    # Change of Character (ChoCh)
    choch_signal = last.get('choch_signal', 0)
    choch_strength = last.get('choch_strength', 0)
    choch_trend = last.get('choch_trend', 'sideways')

    print(f"\n  Change of Character (ChoCh):")
    print(f"    当前趋势: {choch_trend}")
    if choch_signal == 1:
        print(f"    状态: 看多转折 ✓")
        print(f"    转折强度: {choch_strength:.1f}/100")
        print(f"    → 下降趋势中出现 Higher High，可能转为上升趋势")
    elif choch_signal == -1:
        print(f"    状态: 看空转折 ✗")
        print(f"    转折强度: {choch_strength:.1f}/100")
        print(f"    → 上升趋势中出现 Lower Low，可能转为下降趋势")
    else:
        print(f"    状态: 无转折")
        print(f"    → 趋势性质未改变")

    # ChoCh 检测逻辑伪代码
    print(f"\n  ChoCh 检测逻辑 (Python 伪代码):")
    print(f"    1. 识别当前趋势:")
    print(f"       IF ADX > 25 AND MA5 > MA20 THEN trend = 'uptrend'")
    print(f"       ELIF ADX > 25 AND MA5 < MA20 THEN trend = 'downtrend'")
    print(f"       ELSE trend = 'sideways'")
    print(f"")
    print(f"    2. 检测反向 swing point:")
    print(f"       IF trend == 'uptrend' AND 出现 Lower Low THEN")
    print(f"           choch_signal = 'bearish'")
    print(f"           choch_strength = (LL 幅度 / ATR) × 100")
    print(f"")
    print(f"       IF trend == 'downtrend' AND 出现 Higher High THEN")
    print(f"           choch_signal = 'bullish'")
    print(f"           choch_strength = (HH 幅度 / ATR) × 100")
    print(f"")
    print(f"    3. 确认信号有效性:")
    print(f"       IF choch_strength > 50 AND 成交量 > MA20_volume THEN")
    print(f"           RETURN 'confirmed_choch'")
    print(f"       ELSE")
    print(f"           RETURN 'weak_choch'")

    # ========== 5.7 VWAP 偏离度分析 ==========
    print(f"\n{'='*70}")
    print(f"  ║           VWAP 偏离度分析 (机构级止盈)                      ║")
    print(f"{'='*70}")

    vwap = last.get('vwap', np.nan)
    vwap_deviation = last.get('vwap_deviation', np.nan)
    vwap_signal = last.get('vwap_signal', 'normal')

    print(f"\n  VWAP (20日滚动): {vwap:.2f}" if not pd.isna(vwap) else "\n  VWAP: 数据不足")
    print(f"  当前价格: {last['close']:.2f}")

    if not pd.isna(vwap_deviation):
        print(f"  偏离度: {vwap_deviation:.2f}σ")
        print(f"  信号: {vwap_signal}")

        if vwap_signal == 'extreme_high':
            print(f"  → 价格严重偏离 VWAP (>3σ)，建议执行强制减仓")
        elif vwap_signal == 'high':
            print(f"  → 价格偏离 VWAP (>2σ)，考虑部分止盈")
        elif vwap_signal == 'extreme_low':
            print(f"  → 价格严重低于 VWAP (<-3σ)，可能存在超卖机会")
        elif vwap_signal == 'low':
            print(f"  → 价格低于 VWAP (<-2σ)，关注反弹机会")
        else:
            print(f"  → 价格在 VWAP 正常区间内")
    else:
        print(f"  偏离度: 数据不足")

    # ========== 5.8 高位派发模式识别 ==========
    print(f"\n{'='*70}")
    print(f"  ║           高位派发模式识别 (机构退出)                        ║")
    print(f"{'='*70}")

    distribution = calc_distribution_patterns(df)
    dist_score = distribution['distribution_score']
    dist_patterns = distribution['patterns']
    dist_signal = distribution['signal']

    score_bar = "█" * (dist_score // 5) + "░" * (20 - dist_score // 5)
    print(f"\n  派发风险评分: [{score_bar}] {dist_score}分")
    print(f"  信号: {distribution['signal_text']}")

    if dist_patterns:
        print(f"\n  检测到的派发模式:")
        for pattern in dist_patterns:
            if pattern == 'volume_climax':
                print(f"    ✗ 放量滞涨 (Volume Climax)")
            elif pattern == 'bearish_divergence':
                print(f"    ✗ 顶背离 (Bearish Divergence)")
            elif pattern == 'breakdown':
                print(f"    ✗ 破位下跌 (Breakdown)")

    if distribution['details']:
        print(f"\n  详细信息:")
        for detail in distribution['details']:
            print(f"    {detail}")

    if dist_score >= 70:
        print(f"\n  ⚠️  警告: 强烈派发信号，建议减仓或清仓")
    elif dist_score >= 40:
        print(f"\n  ⚠️  注意: 出现派发迹象，密切关注")

    # ========== 5.9 估值水位 (PE/PS 历史分位数) ==========
    print(f"\n{'='*70}")
    print(f"  ║           估值水位 (PE/PS 历史分位数)                        ║")
    print(f"{'='*70}")

    valuation = fetch_valuation_metrics(code)
    if valuation.get('available'):
        print(f"\n  【当前估值】")
        if valuation['current_pe'] is not None:
            print(f"    动态市盈率 (PE): {valuation['current_pe']:.1f}")
        if valuation['current_ps'] is not None:
            print(f"    市销率 (PS): {valuation['current_ps']:.1f}")
        print(f"    数据覆盖: {valuation.get('data_years', 0)} 年")

        print(f"\n  【历史分位】")
        if valuation['pe_percentile'] is not None:
            print(f"    PE 百分位: {valuation['pe_percentile']:.1f}%")
        if valuation['ps_percentile'] is not None:
            print(f"    PS 百分位: {valuation['ps_percentile']:.1f}%")

        # 估值信号
        signal_icon = {'undervalued': '▼', 'fair': '○', 'overvalued': '▲'}.get(valuation['valuation_signal'], '○')
        print(f"\n  【估值信号】")
        print(f"    {signal_icon} {valuation['signal_text']}")

        # 估值+筹码联合确认
        hvn_zones = contrarian.get('volume_exhaustion', {}).get('hvn_zones', [])
        current_price = last['close']
        price_in_hvn = any(lo <= current_price <= hi for lo, hi, _ in hvn_zones)
        pe_pct = valuation.get('pe_percentile')

        print(f"\n  【估值+筹码联合确认】")
        if price_in_hvn and pe_pct is not None and pe_pct < 30:
            print(f"    ✓ 价格在底部筹码密集区 + PE 历史低估({pe_pct:.0f}%分位) → 高胜率长线买点")
        elif price_in_hvn and pe_pct is not None and pe_pct < 50:
            print(f"    ○ 价格在筹码密集区 + PE 估值合理({pe_pct:.0f}%分位) → 可关注")
        elif price_in_hvn:
            print(f"    ○ 价格在筹码密集区，但估值偏高 → 等待估值回落")
        else:
            print(f"    ✗ 估值+筹码联合条件未满足")

        print(f"    数据来源: {valuation['source']}")
    else:
        print(f"\n  {valuation.get('source', '估值数据不可用')}")

    # ========== 5.10 自适应周期参数 ==========
    print(f"\n{'='*70}")
    print(f"  ║           自适应周期参数 (四维度动态调整)                    ║")
    print(f"{'='*70}")

    adaptive = df.attrs.get('adaptive_params', {})
    if adaptive:
        regime_text_map = {
            'squeeze': '挤压期',
            'pulse': '脉冲期',
            'expansion': '扩张期',
            'normal': '正常'
        }
        regime_text = regime_text_map.get(adaptive['regime'], '正常')

        print(f"\n  【市场诊断】")
        print(f"    市场阶段: {adaptive['market_stage']}")
        print(f"    波动率制度: {regime_text} (ATR百分位: {adaptive['atr_percentile']:.1f}%)")
        print(f"    趋势强度: ADX={adaptive['adx']:.1f}")
        print(f"    流动性状态: 量比={adaptive['volume_ratio']:.2f}x")

        print(f"\n  【缩放因子】")
        print(f"    波动率维度: {adaptive['volatility_factor']:.2f}x (权重40%)")
        print(f"    趋势维度: {adaptive['trend_factor']:.2f}x (权重30%)")
        print(f"    流动性维度: {adaptive['liquidity_factor']:.2f}x (权重30%)")
        print(f"    市场阶段偏移: {adaptive['stage_offset']:+d}天")
        print(f"    综合缩放因子: {adaptive['composite_factor']:.2f}x")

        print(f"\n  【调整后周期】")
        print(f"    BoS/ChoCh 回溯期: {adaptive['bos_lookback']}天 (基础20天)")
        print(f"    Chandelier Exit: {adaptive['chandelier_lookback']}天 (基础22天)")
        print(f"    三重背离: {adaptive['divergence_lookback']}天 (基础30天)")

        print(f"\n  说明: {adaptive['explanation']}")
    else:
        print(f"\n  数据不足，使用固定周期参数")

    # ========== 5.10 多时间框架共振 ==========
    print(f"\n{'='*70}")
    print(f"  ║           多时间框架共振 (周线 × 月线 三重确认)               ║")
    print(f"{'='*70}")

    weekly = df.attrs.get('weekly_confirmation', {})
    monthly = df.attrs.get('monthly_confirmation', {})

    trend_icon_map = {
        'uptrend': '▲',
        'downtrend': '▼',
        'sideways': '○'
    }
    trend_text_map = {
        'uptrend': '上升趋势',
        'downtrend': '下降趋势',
        'sideways': '震荡'
    }

    # 当前周期信号
    current_bos = last.get('bos_signal', 0)
    current_choch = last.get('choch_signal', 0)

    # 周线趋势
    if weekly and weekly.get('weekly_trend'):
        w_trend = weekly['weekly_trend']
        w_icon = trend_icon_map.get(w_trend, '○')
        w_text = trend_text_map.get(w_trend, '震荡')
        print(f"\n  【周线趋势】")
        print(f"    趋势方向: {w_icon} {w_text}")
        print(f"    周线 ADX: {weekly['weekly_adx']:.1f}")
        print(f"    周线 BoS: {weekly['weekly_bos_signal']}, ChoCh: {weekly['weekly_choch_signal']}")
    else:
        w_trend = 'sideways'
        print(f"\n  【周线趋势】 数据不足")

    # 月线趋势
    if monthly and monthly.get('monthly_trend'):
        m_trend = monthly['monthly_trend']
        m_icon = trend_icon_map.get(m_trend, '○')
        m_text = trend_text_map.get(m_trend, '震荡')
        print(f"\n  【月线趋势】")
        print(f"    趋势方向: {m_icon} {m_text}")
        print(f"    月线 ADX: {monthly['monthly_adx']:.1f}")
        print(f"    说明: {monthly['explanation']}")
    else:
        m_trend = 'sideways'
        print(f"\n  【月线趋势】 数据不足")

    # 三重共振判断
    print(f"\n  【三重共振分析】")

    # 判断各周期趋势方向
    current_trend = 'sideways'
    ma5 = last.get('MA5', 0)
    ma20 = last.get('MA20', 0)
    if pd.notna(ma5) and pd.notna(ma20):
        if ma5 > ma20:
            current_trend = 'uptrend'
        elif ma5 < ma20:
            current_trend = 'downtrend'

    c_icon = trend_icon_map.get(current_trend, '○')
    c_text = trend_text_map.get(current_trend, '震荡')
    w_icon2 = trend_icon_map.get(w_trend, '○')
    m_icon2 = trend_icon_map.get(m_trend, '○')

    print(f"    当前周期: {c_icon} {c_text}")
    print(f"    周线:     {w_icon2} {trend_text_map.get(w_trend, '震荡')}")
    print(f"    月线:     {m_icon2} {trend_text_map.get(m_trend, '震荡')}")

    # 三重共振条件
    all_uptrend = (current_trend == 'uptrend' and w_trend == 'uptrend' and m_trend == 'uptrend')
    all_downtrend = (current_trend == 'downtrend' and w_trend == 'downtrend' and m_trend == 'downtrend')

    if all_uptrend:
        print(f"\n  ✓✓✓ 三重多头共振确认 — 强烈看多信号")
    elif all_downtrend:
        print(f"\n  ✗✗✗ 三重空头共振确认 — 强烈看空信号")
    elif current_trend == w_trend and current_trend != 'sideways':
        print(f"\n  ✓✓ 双重共振（当前+周线同向），月线未确认")
    else:
        print(f"\n  ✗ 多时间框架未共振，建议等待趋势同向确认")

    # ========== 5.11 长线趋势健康检查 ==========
    print(f"\n{'='*70}")
    print(f"  ║           长线趋势健康检查 (50W/200W 均线)                   ║")
    print(f"{'='*70}")

    long_term_health = weekly.get('long_term_trend_health', 'unknown')
    w50 = weekly.get('weekly_50ma', 0)
    w200 = weekly.get('weekly_200ma', 0)
    w_golden = weekly.get('weekly_golden_cross', False)

    if long_term_health == 'broken':
        print(f"\n  ⚠⚠⚠ 致命警告：长线结构性熊市，禁止买入！⚠⚠⚠")
        print(f"\n  【50周/200周均线状态】")
        if w50 > 0 and w200 > 0:
            deviation = (w50 - w200) / w200 * 100
            print(f"    50周均线: {w50:.2f}")
            print(f"    200周均线: {w200:.2f}")
            print(f"    偏离度: {deviation:.1f}%")
            print(f"    状态: ✗ 死叉 (50W < 200W)")
        print(f"\n  【月线趋势状态】")
        m_golden = monthly.get('monthly_golden_cross', False)
        if not m_golden and monthly.get('monthly_ma12', 0) > 0:
            print(f"    月线MA10: {monthly.get('monthly_ma12', 0):.2f}")
            print(f"    月线MA20: {monthly.get('monthly_ma24', 0):.2f}")
            print(f"    状态: ✗ 月线也破位")
            print(f"\n  ⚠ 月线+周线双破位，严禁抄底")
        else:
            print(f"    月线状态: {monthly.get('explanation', '未知')}")
        print(f"\n  【趋势健康度评级】")
        print(f"    评级: 破位 (Broken)")
        print(f"    建议: 等待50周均线重新站上200周均线后再考虑")
    elif long_term_health == 'healthy':
        print(f"\n  ✓ 长线趋势健康")
        print(f"\n  【50周/200周均线状态】")
        if w50 > 0 and w200 > 0:
            deviation = (w50 - w200) / w200 * 100
            print(f"    50周均线: {w50:.2f}")
            print(f"    200周均线: {w200:.2f}")
            print(f"    偏离度: +{deviation:.1f}%")
            print(f"    状态: ✓ 金叉 (50W > 200W)")
        print(f"\n  【月线趋势状态】")
        print(f"    {monthly.get('explanation', '未知')}")
        print(f"\n  【趋势健康度评级】")
        print(f"    评级: 健康 (Healthy)")
        print(f"    建议: 长线趋势向上，可关注回调买点")
    else:
        print(f"\n  【50周/200周均线状态】")
        print(f"    数据不足，无法计算200周均线")
        print(f"    需要至少 1500 天历史数据")

    # ========== 5.12 宏观筹码分布（3年）==========
    macro_vp = context.get('macro_vp')
    if macro_vp and macro_vp.get('available'):
        print(f"\n{'='*70}")
        print(f"  ║           宏观筹码分布（3年历史）                            ║")
        print(f"{'='*70}")
        print(f"\n  数据覆盖: {macro_vp['data_days']} 天 ({macro_vp['data_days']/250:.1f} 年)")
        print(f"  POC (控制点): {macro_vp['poc']:.2f}")
        print(f"  VAH (价值区高点): {macro_vp['vah']:.2f}")
        print(f"  VAL (价值区低点): {macro_vp['val']:.2f}")

        if macro_vp.get('macro_support'):
            print(f"\n  历史长线核心支撑区:")
            for i, s in enumerate(macro_vp['macro_support'][:5], 1):
                print(f"    S{i}: {s:.2f}")

        if macro_vp.get('macro_resistance'):
            print(f"\n  历史长线核心阻力区:")
            for i, r in enumerate(macro_vp['macro_resistance'][:5], 1):
                print(f"    R{i}: {r:.2f}")

        print(f"\n  → {macro_vp.get('interpretation', '')}")

    # ========== 5.13 基本面体检 ==========
    fundamentals = context.get('fundamentals', {})
    dilution = fundamentals.get('dilution', {})
    quality = fundamentals.get('quality', {})
    delisting = fundamentals.get('delisting', {})

    if dilution.get('available') or quality.get('available') or delisting.get('risk_level', 'none') != 'none':
        print(f"\n{'='*70}")
        print(f"  ║           基本面体检（稀释+质量+退市风险）                   ║")
        print(f"{'='*70}")

        # 估值水位（已在前面输出，这里引用）
        if valuation.get('available'):
            print(f"\n  【估值水位】")
            if valuation['current_pe'] is not None:
                print(f"    PE: {valuation['current_pe']:.1f} (历史 {valuation['pe_percentile']:.0f}% 分位)")
            if valuation['current_ps'] is not None:
                print(f"    PS: {valuation['current_ps']:.1f} (历史 {valuation['ps_percentile']:.0f}% 分位)")
            print(f"    信号: {valuation['signal_text']}")

        # 稀释检测
        if dilution.get('available'):
            print(f"\n  【稀释检测】")
            print(f"    年化稀释率: {dilution['dilution_rate']:+.2f}%")
            if dilution['severe_dilution']:
                print(f"    ⚠ {dilution['warning_text']}")
            elif dilution['warning_text']:
                print(f"    {dilution['warning_text']}")
            else:
                print(f"    ✓ 流通股本稳定，无明显稀释")

        # 质量检测
        if quality.get('available'):
            print(f"\n  【质量检测（ROE/FCF）】")
            if quality['latest_roe'] is not None:
                print(f"    最新 ROE: {quality['latest_roe']:.1f}%")
            if quality['latest_fcf_margin'] is not None:
                print(f"    最新 FCF 利润率: {quality['latest_fcf_margin']:.1f}%")
            if quality['quality_deteriorating']:
                print(f"    ⚠ {quality['warning_text']}")
            elif quality['warning_text']:
                print(f"    {quality['warning_text']}")
            else:
                print(f"    ✓ 盈利能力健康")

        # 退市风险
        if delisting.get('risk_level', 'none') != 'none':
            print(f"\n  【退市风险】")
            print(f"    风险等级: {delisting['risk_level'].upper()}")
            if delisting['warning_text']:
                print(f"    ⚠ {delisting['warning_text']}")
            for detail in delisting.get('details', []):
                print(f"    • {detail}")

        # 价值陷阱判定
        if detect_value_trap is not None:
            try:
                value_trap = detect_value_trap(valuation, quality)
                if value_trap.get('is_trap'):
                    print(f"\n  【价值陷阱判定】")
                    print(f"    ⚠⚠⚠ {value_trap['warning_text']}")
                    for reason in value_trap.get('reasons', []):
                        print(f"    • {reason}")
            except Exception:
                pass

    # ========== 5.14 聪明钱动向 ==========
    smart_money = context.get('smart_money', {})
    insider = smart_money.get('insider', {})
    institutional = smart_money.get('institutional', {})

    if insider.get('available') or institutional.get('available'):
        print(f"\n{'='*70}")
        print(f"  ║           聪明钱动向（内部人士+机构持股）                    ║")
        print(f"{'='*70}")

        # 内部人士交易
        if insider.get('available'):
            print(f"\n  【内部人士交易（近6月）】")
            print(f"    买入次数: {insider['total_buys']}  |  卖出次数: {insider['total_sells']}")
            activity_map = {'net_buying': '净买入', 'net_selling': '净卖出', 'mixed': '买卖交替'}
            print(f"    净活动: {activity_map.get(insider['net_activity'], '未知')}")

            if insider['large_sells']:
                print(f"\n    大额卖出（> $1M）:")
                for sell in insider['large_sells'][:5]:
                    print(f"      • {sell['name']} ({sell['title']}): ${sell['value']:,.0f} ({sell['date']})")

            if insider['insider_selling_alert']:
                print(f"\n    ⚠ {insider['alert_text']}")
            elif insider['alert_text']:
                print(f"    {insider['alert_text']}")

        # 机构持股
        if institutional.get('available'):
            print(f"\n  【机构持股】")
            print(f"    机构数量: {institutional['total_holders']}")
            if institutional['top_holders']:
                print(f"    前5大机构:")
                for h in institutional['top_holders'][:5]:
                    pct = f"{h['pct_out']:.2f}%" if h['pct_out'] else ''
                    print(f"      • {h['name']}: {pct}")

        # 聪明钱综合确认
        if calc_smart_money_confirmation is not None:
            try:
                macro_support_levels = context.get('macro_vp', {}).get('macro_support', [])
                current_price = df.iloc[-1]['close']
                at_support = any(abs(current_price - s) / current_price < 0.05 for s in macro_support_levels)
                confirmation = calc_smart_money_confirmation(insider, institutional, at_support)
                level_icon = {'high': '✓✓', 'medium': '✓', 'low': '○', 'warning': '⚠'}.get(confirmation['confirmation_level'], '○')
                print(f"\n  【聪明钱综合判定】")
                print(f"    {level_icon} {confirmation['signal_text']}")
                for d in confirmation['details']:
                    print(f"    • {d}")
            except Exception:
                pass

    # ========== 5.15 行业相对强弱与宏观风控 ==========
    sector = context.get('sector', {})
    rs = sector.get('rs', {})
    earnings_vol = sector.get('earnings_vol', {})
    macro_bear = sector.get('macro_bear', {})

    if rs.get('available') or earnings_vol.get('available') or macro_bear.get('available'):
        print(f"\n{'='*70}")
        print(f"  ║           行业相对强弱与宏观风控                             ║")
        print(f"{'='*70}")

        # 行业 RS 线
        if rs.get('available'):
            print(f"\n  【行业相对强弱】")
            print(f"    对比标的: {rs['sector_name']}")
            print(f"    1年RS变化: {rs['rs_1y_change']:+.2f}%")
            if rs['rs_divergence']:
                print(f"    ⚠ {rs['warning_text']}")
            elif rs['warning_text']:
                print(f"    {rs['warning_text']}")
            if rs.get('is_fallback'):
                print(f"    ⚠ 行业匹配失败，已降级为大盘宏观相对强弱对比")

        # 财报波动统计
        if earnings_vol.get('available'):
            print(f"\n  【财报波动统计（近8次）】")
            print(f"    平均跳空: {earnings_vol['avg_gap']:.1f}%  |  最大跳空: {earnings_vol['max_gap']:.1f}%")
            print(f"    平均日内回撤: {earnings_vol['avg_drawdown']:.1f}%  |  最大回撤: {earnings_vol['max_drawdown']:.1f}%")
            print(f"    {earnings_vol['summary_text']}")

        # 宏观结构性熊市
        if macro_bear.get('available'):
            print(f"\n  【宏观结构性熊市检查】")
            for name, data in macro_bear['indices'].items():
                status = '✓' if data['above_ma200'] else '✗'
                slope_icon = '↑' if data['ma200_slope'] > 0 else '↓'
                print(f"    {status} {name}: ${data['price']:.2f} (MA200: ${data['ma200']:.2f}, 斜率: {slope_icon}{data['ma200_slope']:+.2f}%)")
            if macro_bear['structural_bear']:
                print(f"\n    ⚠ {macro_bear['warning_text']}")

    # 如果触发长线熊市模式，跳过所有短线分析
    if long_term_health == 'broken':
        print(f"\n{'='*70}")
        print(f"  ║  ⚠ 长线熊市模式：已跳过所有短线分析                         ║")
        print(f"{'='*70}")
        print(f"\n  说明: 50周均线 < 200周均线，长线结构性熊市")
        print(f"  已跳过: 日线SMC买入条件、支撑阻力位、短线盈亏比、左侧/右侧信号、仓位建议")
        print(f"  建议: 等待周线金叉后再考虑，或关注基本面是否出现拐点")
        print(f"\n{'='*70}\n")
        return  # 提前结束，不输出短线分析

    # ========== 5.13 压力测试 (如果启用) ==========
    if stress_test:
        print(f"\n{'='*70}")
        print(f"  ║           压力测试 (15% 回撤场景)                            ║")
        print(f"{'='*70}")

        stress_result = stress_test_drawdown(df, code, drawdown_pct=0.15)

        print(f"\n  当前价格: {last['close']:.2f}")
        print(f"  回撤后价格 (-15%): {stress_result['scenario_price']:.2f}")

        # VaR 计算
        if stress_result['var_formula']:
            print(f"\n  VaR (95% 置信度):")
            print(f"    公式: {stress_result['var_formula']}")
            print(f"    VaR₉₅%: {stress_result['var_95']:.4f} ({stress_result['var_95']*100:.2f}%)")
            print(f"    预期损失: {stress_result['expected_loss']:.4f}")

        # 支撑位
        if stress_result['support_levels']:
            print(f"\n  下方支撑位:")
            for i, support in enumerate(stress_result['support_levels'][:5], 1):
                distance_pct = (support - stress_result['scenario_price']) / stress_result['scenario_price'] * 100
                print(f"    S{i}: {support:.2f} (距回撤价 {distance_pct:+.1f}%)")
        else:
            print(f"\n  下方支撑位: 无明确支撑")

        # 对冲策略
        hedge = stress_result['hedge_strategy']
        print(f"\n  对冲策略建议:")
        print(f"    行动: {hedge['action']}")
        print(f"    理由: {hedge['reason']}")

        if hedge['hedge_instruments']:
            print(f"\n  对冲工具:")
            for instrument in hedge['hedge_instruments']:
                print(f"    • {instrument}")

    # ========== 6. 长线风控分析 + 定投建议 ==========

    # 6.1 历史回撤分析
    print(f"\n【长线风控分析】")
    dd_analysis = calc_max_drawdown_analysis(df)
    if dd_analysis.get('available'):
        print(f"  历史最高价 (ATH): {dd_analysis['ath']:.2f}")
        print(f"  距 ATH 跌幅: {dd_analysis['ath_distance']:+.1f}%")
        print(f"  历史最大回撤: {dd_analysis['max_drawdown']:.1f}%")
        print(f"  当前回撤位置: {dd_analysis['current_drawdown']:.1f}% (距最大回撤还有 {dd_analysis['dd_buffer']:.1f}% 空间)")
    else:
        print(f"  回撤数据不足")

    # 6.2 相对强弱分析
    print(f"\n【相对强弱分析】")
    rs = calc_relative_strength(code, df)
    if rs.get('available'):
        print(f"  基准指数: {rs['benchmark_name']}")
        if rs.get('relative_1y') is not None:
            s1 = rs['stock_1y_return']
            b1 = rs['benchmark_1y_return']
            d1 = rs['relative_1y']
            icon1 = '▲' if d1 > 0 else '▼'
            print(f"  近 1 年: 股票 {s1:+.1f}% vs 基准 {b1:+.1f}% → {icon1} {'跑赢' if d1 > 0 else '跑输'} {abs(d1):.1f}%")
        if rs.get('relative_3y') is not None:
            s3 = rs['stock_3y_return']
            b3 = rs['benchmark_3y_return']
            d3 = rs['relative_3y']
            icon3 = '▲' if d3 > 0 else '▼'
            print(f"  近 3 年: 股票 {s3:+.1f}% vs 基准 {b3:+.1f}% → {icon3} {'跑赢' if d3 > 0 else '跑输'} {abs(d3):.1f}%")

        signal_icon = {'outperform': '▲', 'underperform': '▼', 'neutral': '○'}.get(rs['signal'], '○')
        print(f"\n  结论: {signal_icon} {rs['signal_text']}")
    else:
        print(f"  {rs.get('signal_text', '数据不足')}")

    # 6.3 长线定投建议区间
    print(f"\n【长线定投建议区间】")
    try:
        price_targets = calculate_price_targets(df, resonance, consensus, support, resistance, contrarian=contrarian)
        valuation = fetch_valuation_metrics(code)
        dca_zone = calc_dca_zone(df, valuation, sr_data)

        if dca_zone.get('available'):
            print(f"  建议区间: {dca_zone['zone_low']:.2f} - {dca_zone['zone_high']:.2f}")
            print(f"  区间依据: {dca_zone['zone_reason']}")
            current_price = last['close']
            if dca_zone['zone_low'] <= current_price <= dca_zone['zone_high']:
                print(f"  当前价位: {current_price:.2f} (区间内，可分批建仓)")
            elif current_price < dca_zone['zone_low']:
                print(f"  当前价位: {current_price:.2f} (低于区间，积极建仓)")
            else:
                print(f"  当前价位: {current_price:.2f} (高于区间，等待回调)")
        else:
            print(f"  数据不足，无法计算定投区间")
    except Exception:
        print(f"  计算定投区间失败")

    # 操作检查清单
    print(f"\n【操作检查清单】")
    try:
        checklist = generate_checklist(df, resonance, signals, indicators, contrarian=contrarian)
        pass_count = sum(1 for _, passed, _ in checklist if passed)
        for item, passed, reason in checklist:
            status = "✓" if passed else "✗"
            print(f"  [{status}] {item}: {reason}")
        print(f"\n  通过: {pass_count}/12 项")
        if pass_count >= 10:
            print(f"  建议: 可以执行买入")
        elif pass_count >= 8:
            print(f"  建议: 谨慎观望，逢低布局")
        else:
            print(f"  建议: 暂不建议买入")
    except Exception as e:
        print(f"  生成检查清单失败")

    # ========== 7. SMC 决策仪表盘 (严格决策树) ==========
    print(f"\n{'='*70}")
    print(f"  ║               SMC 决策仪表盘 (严格决策树)                      ║")
    print(f"{'='*70}")

    try:
        # 计算派发模式和 Chandelier Exit（使用自适应参数）
        distribution = calc_distribution_patterns(df)
        adaptive = df.attrs.get('adaptive_params', {})
        chandelier_lookback = adaptive.get('chandelier_lookback', 22)
        chandelier = calc_chandelier_exit(df, lookback=chandelier_lookback)

        # 生成决策
        # 计算价值陷阱（用于决策仪表盘否决）
        value_trap_result = {}
        if detect_value_trap is not None:
            try:
                value_trap_result = detect_value_trap(valuation, context.get('fundamentals', {}).get('quality', {}))
            except Exception:
                pass

        long_term_context = {
            'weekly_confirmation': weekly_confirmation,
            'monthly_confirmation': monthly_confirmation,
            'valuation': valuation,
            'fundamentals': {
                **context.get('fundamentals', {}),
                'value_trap': value_trap_result
            },
            'smart_money': context.get('smart_money', {}),
            'sector_rs': context.get('sector', {}).get('rs', {})
        }
        dashboard = generate_decision_dashboard(df, price_targets, contrarian, distribution, chandelier,
                                                circuit_breaker=circuit_breaker,
                                                long_term_context=long_term_context)

        # 输出决策
        decision_text = dashboard.get('decision_text', '观望')
        confidence = dashboard.get('confidence', 'LOW')
        confidence_text = {'HIGH': '高', 'MEDIUM': '中', 'LOW': '低'}.get(confidence, '低')

        print(f"\n  最终决策: 【{decision_text}】 置信度: {confidence_text}")

        # 买入条件检查
        print(f"\n  ── 买入条件 ──────────────────────────────────")
        buy_conds = dashboard.get('buy_conditions', {})
        buy_details = dashboard.get('buy_details', {})

        for key, label, detail_key in [
            ('choch_confirmed', 'ChoCh 确认', 'choch'),
            ('price_testing_zone', '回踩确认', 'zone'),
            ('risk_reward_valid', '盈亏比', 'risk_reward'),
            ('zscore_valid', 'Z-Score', 'zscore')
        ]:
            status = "✓" if buy_conds.get(key, False) else "✗"
            detail = buy_details.get(detail_key, '')
            print(f"    [{status}] {label}: {detail}")

        # 卖出条件检查
        print(f"\n  ── 卖出条件 ──────────────────────────────────")
        sell_conds = dashboard.get('sell_conditions', {})
        sell_details = dashboard.get('sell_details', {})

        for key, label, detail_key in [
            ('stop_loss_triggered', '止损触发', 'stop_loss'),
            ('distribution_detected', '派发识别', 'distribution')
        ]:
            status = "✓" if sell_conds.get(key, False) else "✗"
            detail = sell_details.get(detail_key, '')
            print(f"    [{status}] {label}: {detail}")

        # 决策理由
        reasoning = dashboard.get('reasoning', [])
        if reasoning:
            print(f"\n  决策理由:")
            for reason in reasoning:
                print(f"    • {reason}")

        # 风险警告
        warnings = dashboard.get('warnings', [])
        if warnings:
            print(f"\n  ⚠ 风险警告:")
            for warning in warnings:
                print(f"    • {warning}")

    except Exception as e:
        print(f"  生成决策仪表盘失败: {e}")

    print(f"{'='*70}")

    # ========== 7.5 持仓应对策略 ==========
    # 判断是否有持仓（根据决策推断）
    decision = dashboard.get('decision', 'HOLD_EMPTY')
    has_position = decision in ('HOLD_WITH_POSITION', 'SELL', 'LONG_TERM_BEAR', 'VALUE_TRAP')

    # 获取 entry_price（优先级：命令行参数 > Volume Profile POC > 60日均价）
    pos_entry_price = None
    entry_source = ''

    if entry_price is not None:
        pos_entry_price = entry_price
        entry_source = '命令行参数'
        has_position = True  # 指定了买入价则视为有持仓
    elif has_position:
        try:
            # 尝试从 Volume Profile 获取 POC
            vol_profile = context.get('volume_profile', {})
            if vol_profile and vol_profile.get('poc'):
                pos_entry_price = vol_profile['poc']
                entry_source = 'Volume Profile POC'
            else:
                # 使用 60 日均价
                if len(df) >= 60:
                    pos_entry_price = df['close'].tail(60).mean()
                    entry_source = '60日均价估算'
                else:
                    pos_entry_price = df['close'].mean()
                    entry_source = f'{len(df)}日均价估算'
        except Exception:
            pass

    if has_position and pos_entry_price and pos_entry_price > 0:
        try:
            current_price = last['close']

            # 计算回撤分析
            dd_analysis = calc_max_drawdown_analysis(df)

            # 生成持仓策略
            position_strategy = generate_position_strategy(
                df, current_price, pos_entry_price, dashboard,
                chandelier, contrarian, distribution, price_targets,
                dd_analysis=dd_analysis
            )

            if position_strategy.get('available'):
                if position_strategy['mode'] == 'loss':
                    _print_loss_response(position_strategy, pos_entry_price, current_price)
                else:
                    _print_profit_response(position_strategy, pos_entry_price, current_price)

                if entry_price is None:
                    print(f"\n  说明: 买入均价来源于 {entry_source}")
                    print(f"  提示: 使用 --entry-price 参数可指定实际买入均价")
        except Exception:
            # 持仓策略失败不影响主流程
            pass

    # ========== 8. 综合结论（研报风格） ==========
    print(f"\n【综合结论】(研报风格)")
    try:
        # 使用 generate_summary 作为后备，同时尝试 generate_final_conclusion
        final_conclusion = generate_summary(df, trend, signals, support, resistance, resonance)
        if consensus and consensus.get('rating') != 'hold':
            # 如果有机构评级，生成更详细的结论
            tech_direction = resonance.get('resonance', 'neutral')
            tech_text = {'strong_buy': '技术面强烈看多', 'buy': '技术面偏多',
                        'neutral': '技术面震荡', 'sell': '技术面偏空', 'strong_sell': '技术面强烈看空'}.get(tech_direction, '技术面震荡')

            consensus_text = {'buy': '机构看涨', 'hold': '机构观望', 'sell': '机构看跌'}.get(consensus.get('rating', 'hold'), '机构观望')

            target = consensus.get('target_price')
            if target:
                target_text = f", 目标价{target:.2f}元"
            else:
                target_text = ""

            final_conclusion = f"{tech_text}+{consensus_text}，{win_rate_info.get('suggestion', '建议观望')}{target_text}"

        print(f"  {final_conclusion}")
    except Exception as e:
        print(f"  {summary}")

    print(f"{'='*70}\n")

    # 静默写入决策日志
    try:
        append_decision_log(code, df, price_targets, dashboard, context)
    except Exception:
        pass


# ============ 批量分析与压力测试模块 ============

def analyze_portfolio(codes: list, period: str = 'w', days: int = 1000, demo: bool = False, stress_test: bool = False) -> dict:
    """
    批量分析多只股票

    参数:
    - codes: 股票代码列表 ['0700.HK', '1810.HK', 'QQQ', 'TSLA', 'NVDA']
    - period: 周期 ('d' 或 'w')
    - days: 回溯天数
    - demo: 是否使用演示数据
    - stress_test: 是否执行压力测试

    返回:
    - results: Dict[code, analysis_result]
    - comparison: 横向对比数据
    """
    results = {}
    print(f"\n开始批量分析 {len(codes)} 只股票...\n")

    for idx, code in enumerate(codes, 1):
        print(f"[{idx}/{len(codes)}] 正在分析 {code}...")
        try:
            # 获取数据
            df = fetch_stock_data(code, period, days, demo=demo)
            if df.empty:
                results[code] = {'error': '无法获取数据'}
                continue

            # 数据清洗
            is_valid, error_msg, df_clean = validate_data(df, min_rows=30)
            if not is_valid:
                results[code] = {'error': error_msg}
                continue

            # 计算指标
            df_clean = calculate_indicators(df_clean)

            # 分析信号
            signals = detect_signals(df_clean)
            sr_inst = find_support_resistance_institutional(df_clean)
            indicators = analyze_indicator_signals(df_clean)
            resonance = calculate_resonance(indicators)
            win_rate_info = calculate_win_rate(indicators, resonance, df_clean)
            contrarian = calc_contrarian_signals(df_clean)
            distribution = calc_distribution_patterns(df_clean)

            # 计算自适应周期参数
            last_row = df_clean.iloc[-1]
            market_stage = determine_market_stage(resonance, contrarian, distribution, last_row)
            adaptive_params = calc_adaptive_params(df_clean, market_stage)
            df_clean.attrs['adaptive_params'] = adaptive_params

            # 使用自适应参数重新计算关键指标
            try:
                df_clean = calc_break_of_structure(df_clean, lookback=adaptive_params['bos_lookback'])
            except Exception:
                pass

            try:
                df_clean = calc_change_of_character(df_clean, lookback=adaptive_params['choch_lookback'])
            except Exception:
                pass

            # 获取周线确认
            weekly_confirmation = fetch_weekly_confirmation(code, demo=demo)
            df_clean.attrs['weekly_confirmation'] = weekly_confirmation

            # 获取月线确认
            monthly_confirmation = fetch_monthly_confirmation(code, demo=demo)
            df_clean.attrs['monthly_confirmation'] = monthly_confirmation

            # 价格目标
            consensus = {'rating': 'hold', 'rating_text': '暂无', 'target_price': None, 'analyst_count': 0, 'source': '批量模式'}
            price_targets = calculate_price_targets(
                df_clean, resonance, consensus, sr_inst['support'], sr_inst['resistance'], contrarian=contrarian
            )

            # 决策仪表盘（含熔断检查，使用自适应参数）
            chandelier_lookback = adaptive_params.get('chandelier_lookback', 22)
            chandelier = calc_chandelier_exit(df_clean, lookback=chandelier_lookback)
            vol_exhaust_p = calc_volume_exhaustion(df_clean)
            hvn_zones_p = vol_exhaust_p.get('hvn_zones', [])
            circuit_breaker_p = check_circuit_breaker(code, df_clean, hvn_zones_p, demo=demo)
            dashboard = generate_decision_dashboard(
                df_clean, price_targets, contrarian, distribution, chandelier,
                circuit_breaker=circuit_breaker_p,
                long_term_context={
                    'weekly_confirmation': weekly_confirmation,
                    'monthly_confirmation': monthly_confirmation,
                }
            )

            # 可选上下文（用于结果存储）
            optional_ctx = build_optional_context(code, df_clean, demo)

            # 压力测试
            stress_result = None
            if stress_test:
                stress_result = stress_test_drawdown(df_clean, code, drawdown_pct=0.15)

            # 汇总结果
            analysis = {
                'df': df_clean,
                'signals': signals,
                'sr_inst': sr_inst,
                'indicators': indicators,
                'resonance': resonance,
                'win_rate_info': win_rate_info,
                'contrarian': contrarian,
                'distribution': distribution,
                'price_targets': price_targets,
                'dashboard': dashboard,
                'optional_ctx': optional_ctx,
                'stress_result': stress_result
            }
            results[code] = analysis
            print(f"  ✓ {code} 分析完成")

        except Exception as e:
            results[code] = {'error': str(e)}
            print(f"  ✗ {code} 分析失败: {e}")

    # 生成横向对比表
    print("\n生成对比表...")
    comparison = generate_comparison_table(results)

    return {'results': results, 'comparison': comparison}


def generate_comparison_table(results: dict) -> pd.DataFrame:
    """
    生成多股票对比表

    列:
    - 股票代码
    - 当前阶段
    - 多头强度
    - 空头强度
    - 共振信号
    - 左侧信号得分
    - BoS 状态
    - ChoCh 状态
    - 派发风险
    - 建议操作
    """
    rows = []

    for code, analysis in results.items():
        if 'error' in analysis:
            rows.append({
                '股票代码': code,
                '当前阶段': '数据错误',
                '多头强度': 0,
                '空头强度': 0,
                '共振信号': 'N/A',
                '左侧得分': 0,
                'BoS状态': 'N/A',
                'ChoCh状态': 'N/A',
                '派发风险': 0,
                '建议操作': '无法分析'
            })
            continue

        try:
            df = analysis['df']
            resonance = analysis['resonance']
            contrarian = analysis['contrarian']
            distribution = analysis['distribution']
            dashboard = analysis['dashboard']

            # 获取最新数据
            last = df.iloc[-1]

            # 判断当前阶段
            stage = determine_market_stage(
                resonance, contrarian, distribution, last
            )

            # BoS 状态
            bos_signal = last.get('bos_signal', 0)
            if bos_signal == 1:
                bos_status = '看多突破'
            elif bos_signal == -1:
                bos_status = '看空突破'
            else:
                bos_status = '无突破'

            # ChoCh 状态
            choch_signal = last.get('choch_signal', 0)
            if choch_signal == 1:
                choch_status = '看多转折'
            elif choch_signal == -1:
                choch_status = '看空转折'
            else:
                choch_status = '无转折'

            # 计算买入/卖出强度（基于条件满足数量）
            buy_conds = dashboard.get('buy_conditions', {})
            sell_conds = dashboard.get('sell_conditions', {})

            buy_strength = sum(1 for v in buy_conds.values() if v) * 25  # 0-100
            sell_strength = sum(1 for v in sell_conds.values() if v) * 50  # 0-100

            rows.append({
                '股票代码': code,
                '当前阶段': stage,
                '多头强度': buy_strength,
                '空头强度': sell_strength,
                '共振信号': resonance['resonance'],
                '左侧得分': contrarian['composite_score'],
                'BoS状态': bos_status,
                'ChoCh状态': choch_status,
                '派发风险': distribution['distribution_score'],
                '建议操作': dashboard['action']
            })

        except Exception as e:
            rows.append({
                '股票代码': code,
                '当前阶段': '解析错误',
                '多头强度': 0,
                '空头强度': 0,
                '共振信号': 'N/A',
                '左侧得分': 0,
                'BoS状态': 'N/A',
                'ChoCh状态': 'N/A',
                '派发风险': 0,
                '建议操作': '无法分析'
            })

    return pd.DataFrame(rows)


def determine_market_stage(resonance: dict, contrarian: dict, distribution: dict, last_row) -> str:
    """
    判断股票当前所处阶段

    阶段:
    - 左侧筑底: 左侧信号强 + 共振中性/看多
    - 右侧启动: BoS 看多 + 共振看多
    - 高位派发: 派发风险高 + 共振看空
    - 下行通道: 共振看空 + 无左侧信号
    """
    resonance_signal = resonance['resonance']
    contrarian_score = contrarian['composite_score']
    distribution_score = distribution['distribution_score']
    bos_signal = last_row.get('bos_signal', 0)

    # 高位派发
    if distribution_score >= 60:
        return '高位派发'

    # 右侧启动
    if bos_signal == 1 and resonance_signal in ['strong_buy', 'buy']:
        return '右侧启动'

    # 左侧筑底
    if contrarian_score >= 65 and resonance_signal in ['neutral', 'buy', 'strong_buy']:
        return '左侧筑底'

    # 下行通道
    if resonance_signal in ['sell', 'strong_sell'] and contrarian_score < 50:
        return '下行通道'

    # 震荡整理
    return '震荡整理'


def stress_test_drawdown(df: pd.DataFrame, code: str, drawdown_pct: float = 0.15) -> dict:
    """
    模拟价格下跌 X% 的压力测试

    参数:
    - df: 股票数据
    - code: 股票代码
    - drawdown_pct: 回撤幅度 (默认 15%)

    返回:
    - scenario_price: 回撤后的价格
    - support_levels: 下方支撑位列表
    - hedge_strategy: 对冲策略建议
    - var_95: 95% 置信度下的 VaR
    - expected_loss: 预期损失
    """
    result = {
        'scenario_price': 0,
        'support_levels': [],
        'hedge_strategy': {},
        'var_95': 0,
        'expected_loss': 0,
        'var_formula': ''
    }

    try:
        current_price = df['close'].iloc[-1]
        scenario_price = current_price * (1 - drawdown_pct)
        result['scenario_price'] = scenario_price

        # 1. 计算 VaR (使用历史模拟法)
        returns = df['close'].pct_change().dropna()
        if len(returns) > 0:
            var_95_pct = np.percentile(returns, 5)
            var_95 = var_95_pct * current_price
            result['var_95'] = var_95
            result['expected_loss'] = abs(var_95) * drawdown_pct

            # LaTeX 公式 (使用 Unicode 符号)
            mean_return = returns.mean()
            std_return = returns.std()
            result['var_formula'] = f"VaR₉₅% = μ + z₀.₀₅ × σ = {mean_return:.4f} + (-1.645) × {std_return:.4f} = {var_95_pct:.4f}"

        # 2. 找到下方支撑位
        sr_inst = find_support_resistance_institutional(df)
        support_levels = sr_inst.get('support', [])
        result['support_levels'] = [s for s in support_levels if s < current_price]

        # 3. 生成对冲策略
        result['hedge_strategy'] = generate_hedge_strategy(code, scenario_price, result['support_levels'])

    except Exception:
        pass

    return result


def generate_hedge_strategy(code: str, scenario_price: float, support_levels: list) -> dict:
    """
    生成对冲策略建议 (纯文字建议)

    策略:
    1. 如果 scenario_price 接近支撑位 → 建议"持仓观望，准备补仓"
    2. 如果 scenario_price 跌破所有支撑位 → 建议"止损离场"
    3. 如果是 QQQ/NVDA (科技股) → 建议"买入 VIX 看涨期权对冲"
    4. 如果是港股 (腾讯/小米) → 建议"关注恒生指数走势，考虑做空恒指期货"

    返回:
    - action: 'hold_and_watch' | 'add_position' | 'stop_loss' | 'hedge_with_options'
    - reason: 策略理由
    - hedge_instruments: 对冲工具列表
    """
    result = {
        'action': 'hold_and_watch',
        'reason': '',
        'hedge_instruments': []
    }

    try:
        # 检查是否接近支撑位
        near_support = False
        if support_levels:
            closest_support = max(support_levels)
            if abs(scenario_price - closest_support) / closest_support < 0.05:
                near_support = True
                result['action'] = 'add_position'
                result['reason'] = f'回撤后价格({scenario_price:.2f})接近关键支撑位({closest_support:.2f})，可考虑补仓'
            elif scenario_price < min(support_levels):
                result['action'] = 'stop_loss'
                result['reason'] = f'回撤后价格({scenario_price:.2f})跌破所有支撑位，建议止损离场'
            else:
                result['action'] = 'hold_and_watch'
                result['reason'] = f'回撤后价格({scenario_price:.2f})仍在支撑区间内，持仓观望'
        else:
            result['action'] = 'hold_and_watch'
            result['reason'] = '无明确支撑位数据，建议持仓观望'

        # 根据股票类型推荐对冲工具
        code_upper = code.upper()

        # 美股科技股
        if code_upper in ['QQQ', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
            result['hedge_instruments'].append('买入 VIX 看涨期权 (对冲市场波动风险)')
            result['hedge_instruments'].append(f'买入 {code_upper} 看跌期权 (直接对冲个股风险)')
            result['hedge_instruments'].append('做空纳斯达克 100 期货 (NQ) (对冲科技板块风险)')

        # 港股
        elif '.HK' in code_upper:
            result['hedge_instruments'].append('关注恒生指数走势，考虑做空恒指期货 (HSI)')
            result['hedge_instruments'].append('买入恒生指数看跌期权')
            result['hedge_instruments'].append('配置美元资产对冲港币汇率风险')

        # A股
        elif code_upper.isdigit() and len(code_upper) == 6:
            result['hedge_instruments'].append('做空沪深 300 期货 (IF) 或中证 500 期货 (IC)')
            result['hedge_instruments'].append('买入沪深 300 看跌期权')
            result['hedge_instruments'].append('配置国债或货币基金降低组合波动')

        else:
            result['hedge_instruments'].append('根据股票所属市场选择相应的指数期货或期权对冲')

    except Exception:
        result['reason'] = '对冲策略生成失败'

    return result


def print_portfolio_analysis(results: dict, period: str = 'w'):
    """
    打印批量分析结果 (完整输出模式)

    输出:
    1. 多空力量对比表
    2. 各股票详细分析
    3. 风险压力测试结果 (如果启用)
    """
    print("\n" + "="*80)
    print("【投资组合分析】")
    print("="*80)

    # 1. 打印对比表
    comparison_df = results['comparison']
    print("\n多空力量对比表:")
    print(comparison_df.to_string(index=False))

    # 2. 逐个打印详细分析
    for code, analysis in results['results'].items():
        if 'error' in analysis:
            print(f"\n{'='*80}")
            print(f"【{code}】")
            print(f"{'='*80}")
            print(f"分析失败: {analysis['error']}\n")
            continue

        print(f"\n{'='*80}")
        print(f"【{code} 详细分析】")
        print(f"{'='*80}")

        # 复用 print_analysis 的核心逻辑
        df = analysis['df']
        print_analysis(
            df, code, period,
            demo=False,
            stress_test=(analysis.get('stress_result') is not None)
        )


# ============ 执行日志 ============

def append_decision_log(code: str, df: pd.DataFrame, price_targets: dict,
                         dashboard: dict, context: dict):
    """静默追加决策日志到 trading_decision_log.csv

    每次脚本运行结束时调用，将结构化决策数据追加到本地 CSV 文件。
    写入失败不影响主流程。
    """
    import csv

    try:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_decision_log.csv')

        last = df.iloc[-1]

        # 提取 IV
        iv = ''
        options_data = context.get('options_data', {}) if context else {}
        if options_data.get('iv') is not None:
            iv = options_data['iv']

        # 提取 POC
        poc = ''
        vp = context.get('volume_profile') if context else None
        if vp and vp.get('poc') is not None:
            poc = vp['poc']

        # 提取仓位信息
        position_info = price_targets.get('position_info', {}) if price_targets else {}
        suggested_shares = position_info.get('suggested_shares', '') if position_info else ''

        # 提取熔断状态
        cb = dashboard.get('circuit_breaker') if dashboard else None
        cb_triggered = cb.get('any_triggered', False) if cb else False

        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'code': code,
            'date': str(last.get('date', '')),
            'close': round(last['close'], 2),
            'atr': round(last['ATR'], 2) if pd.notna(last.get('ATR')) else '',
            'iv': iv,
            'poc': poc,
            'adv20': round(last['ADV20'], 0) if pd.notna(last.get('ADV20')) else '',
            'buy_price': price_targets.get('buy_price', '') if price_targets else '',
            'stop_loss': price_targets.get('stop_loss', '') if price_targets else '',
            'target_price': price_targets.get('target_price', '') if price_targets else '',
            'risk_reward': price_targets.get('risk_reward', '') if price_targets else '',
            'risk_reward_after_slippage': price_targets.get('risk_reward_after_slippage', '') if price_targets else '',
            'decision': dashboard.get('decision', '') if dashboard else '',
            'action': dashboard.get('action', '') if dashboard else '',
            'suggested_shares': suggested_shares,
            'circuit_breaker': cb_triggered
        }

        file_exists = os.path.exists(log_path)
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    except Exception:
        pass  # 日志写入失败不影响主流程


# ============ 持仓管理 ============


def _get_portfolio_path() -> str:
    """返回 portfolio.json 的绝对路径（与脚本同目录）"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), PORTFOLIO_FILE)


def load_portfolio() -> dict:
    """加载持仓文件。文件不存在则返回空结构。"""
    path = _get_portfolio_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data.get('holdings'), dict):
            data['holdings'] = {}
        if not isinstance(data.get('account_size'), (int, float)):
            data['account_size'] = DEFAULT_ACCOUNT_SIZE
        return data
    except Exception:
        return {'version': 1, 'account_size': DEFAULT_ACCOUNT_SIZE, 'holdings': {}}


def save_portfolio(portfolio: dict) -> bool:
    """原子写入持仓 JSON（先写临时文件再 rename，防止写入中断损坏文件）。"""
    path = _get_portfolio_path()
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        print(f"  警告: 持仓文件保存失败: {e}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False


def calc_weighted_avg_cost(old_price: float, old_qty: int,
                            new_price: float, new_qty: int) -> float:
    """计算加权平均成本。"""
    total_qty = old_qty + new_qty
    if total_qty <= 0:
        return old_price
    return (old_price * old_qty + new_price * new_qty) / total_qty


def portfolio_add(code: str, price: float, quantity: int) -> None:
    """添加新持仓或加仓（自动计算加权平均成本）。"""
    code = code.upper()

    # 验证股票代码格式
    is_valid, err = validate_stock_code(code)
    if not is_valid:
        print(f"错误: {err}")
        return

    if price <= 0 or quantity <= 0:
        print("错误: 价格和数量必须大于0")
        return

    pf = load_portfolio()
    holdings = pf.setdefault('holdings', {})

    if code in holdings:
        # 已有持仓：计算加权平均成本
        old = holdings[code]
        old_price = old['entry_price']
        old_qty = old['quantity']
        new_avg = calc_weighted_avg_cost(old_price, old_qty, price, quantity)
        new_qty = old_qty + quantity

        print(f"\n加仓 {code}:")
        print(f"  原持仓: {old_qty} 股 @ {old_price:.2f}")
        print(f"  本次买入: {quantity} 股 @ {price:.2f}")
        if new_avg < old_price:
            reduction = (old_price - new_avg) / old_price * 100
            print(f"  新均价: {new_avg:.4f}（成本降低 {reduction:.1f}%）")
        elif new_avg > old_price:
            increase = (new_avg - old_price) / old_price * 100
            print(f"  新均价: {new_avg:.4f}（成本上升 {increase:.1f}%）")
        else:
            print(f"  新均价: {new_avg:.4f}（成本不变）")
        print(f"  总持仓: {new_qty} 股")

        holdings[code]['entry_price'] = round(new_avg, 4)
        holdings[code]['quantity'] = new_qty
        holdings[code]['last_update'] = datetime.now().strftime('%Y-%m-%d')
    else:
        # 新建持仓
        if len(holdings) >= MAX_HOLDINGS:
            print(f"  警告: 持仓已达 {MAX_HOLDINGS} 只上限，建议控制持仓数量")

        holdings[code] = {
            'entry_price': round(price, 4),
            'quantity': quantity,
            'first_buy_date': datetime.now().strftime('%Y-%m-%d'),
            'last_update': datetime.now().strftime('%Y-%m-%d'),
            'notes': '',
        }
        print(f"\n新建持仓 {code}: {quantity} 股 @ {price:.2f}")

    if save_portfolio(pf): print("  持仓已保存。")


def portfolio_remove(code: str, quantity: int = None) -> None:
    """减仓或清仓。quantity=None 表示全部清仓。"""
    code = code.upper()

    if quantity is not None and quantity <= 0:
        print("错误: 卖出数量必须大于0")
        return

    is_valid, err = validate_stock_code(code)
    if not is_valid:
        print(f"错误: {err}")
        return

    pf = load_portfolio()
    holdings = pf.get('holdings', {})

    if code not in holdings:
        print(f"错误: 持仓中未找到 {code}")
        return

    holding = holdings[code]
    current_qty = holding['quantity']

    if quantity is None or quantity >= current_qty:
        # 全部清仓
        if quantity is not None and quantity > current_qty:
            print(f"  提示: 卖出数量 ({quantity}) 超过持仓 ({current_qty})，执行全部清仓")
        del holdings[code]
        print(f"\n已清仓 {code}（原持仓 {current_qty} 股）")
    else:
        remaining = current_qty - quantity
        holdings[code]['quantity'] = remaining
        holdings[code]['last_update'] = datetime.now().strftime('%Y-%m-%d')
        print(f"\n减仓 {code}: 卖出 {quantity} 股，剩余 {remaining} 股 @ {holding['entry_price']:.2f}")

    if save_portfolio(pf): print("  持仓已更新。")


def portfolio_list() -> None:
    """打印当前持仓列表（表格格式）。"""
    pf = load_portfolio()
    holdings = pf.get('holdings', {})

    print(f"\n{'='*60}")
    print(f"  当前持仓")
    print(f"{'='*60}")

    if not holdings:
        print("  （暂无持仓）")
        print(f"\n  提示: 使用 'hold add <代码> --price <价格> --qty <数量>' 添加持仓")
        print(f"{'='*60}")
        return

    print(f"  {'代码':<12} {'均价':>10} {'持仓数量':>10} {'首次买入':>12} {'最后更新':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

    for code, h in holdings.items():
        print(f"  {code:<12} {h['entry_price']:>10.4f} {h['quantity']:>10} "
              f"{h.get('first_buy_date', '-'):>12} {h.get('last_update', '-'):>12}")

    print(f"{'='*60}")
    print(f"  共 {len(holdings)} 只持仓")
    print(f"{'='*60}")


def calc_breakeven_analysis(entry_price: float, quantity: int,
                             current_price: float, strategy: dict,
                             price_targets: dict,
                             circuit_triggered: bool = False,
                             decision: str = '') -> dict:
    """回本路径分析 — 评估持仓/补仓/止损换股三条路径，推荐最高效方案。

    strategy: 来自 generate_position_strategy() 的返回值（loss mode）
    price_targets: 来自 calculate_price_targets() 的返回值
    circuit_triggered: 是否触发熔断（来自 dashboard）
    decision: SMC决策字符串（来自 dashboard，如 'SELL', 'HOLD_WITH_POSITION'）
    """
    if entry_price <= 0 or current_price <= 0 or quantity <= 0:
        return {'available': False}

    pnl_pct = (current_price - entry_price) / entry_price * 100
    if pnl_pct >= 0:
        return {'available': False}  # 已回本，不需要回本分析

    # ── 路径1：持有等待 ──
    required_gain_pct = (entry_price / current_price - 1) * 100  # 例：亏20% → 需涨25%
    trend_status = strategy.get('trend_status', 'damaged')

    if trend_status == 'intact' and required_gain_pct < 15:
        hold_feasibility = 'high'
        hold_feasibility_text = '高'
    elif trend_status == 'broken' or required_gain_pct > 25:
        hold_feasibility = 'low'
        hold_feasibility_text = '低'
    else:
        hold_feasibility = 'medium'
        hold_feasibility_text = '中'

    hold_path = {
        'required_gain_pct': round(required_gain_pct, 1),
        'feasibility': hold_feasibility,
        'feasibility_text': hold_feasibility_text,
    }

    # ── 路径2：补仓摊低成本 ──
    add_ok = strategy.get('add_ok', False)
    add_price = strategy.get('add_price') or price_targets.get('buy_price', current_price)
    add_shares = strategy.get('add_shares') or 0
    loss_level = strategy.get('loss_level', 'moderate')

    if add_ok and add_price and add_shares and add_shares > 0:
        new_avg = calc_weighted_avg_cost(entry_price, quantity, add_price, add_shares)
        cost_reduction_pct = (entry_price - new_avg) / entry_price * 100
        new_required_gain_pct = (new_avg / current_price - 1) * 100
        capital_required = add_price * add_shares

        if loss_level in ('shallow', 'moderate'):
            add_feasibility, add_feasibility_text = 'high', '高'
        elif loss_level == 'deep':
            add_feasibility, add_feasibility_text = 'medium', '中'
        else:
            add_feasibility, add_feasibility_text = 'low', '低'

        add_path = {
            'available': True,
            'add_price': round(add_price, 2),
            'add_shares': add_shares,
            'new_avg_cost': round(new_avg, 4),
            'cost_reduction_pct': round(cost_reduction_pct, 1),
            'new_required_gain_pct': round(new_required_gain_pct, 1),
            'capital_required': round(capital_required, 2),
            'feasibility': add_feasibility,
            'feasibility_text': add_feasibility_text,
        }
    else:
        if add_ok:
            # 条件满足但仓位计算失败（止损价不可用）
            add_path = {
                'available': False,
                'feasibility': 'blocked',
                'feasibility_text': '仓位计算失败',
                'blocked_reasons': ['✗ 无法计算建议股数（止损价不可用）'],
            }
        else:
            add_conditions = strategy.get('add_conditions', [])
            add_path = {
                'available': False,
                'feasibility': 'blocked',
                'feasibility_text': '条件未满足',
                'blocked_reasons': add_conditions,
            }

    # ── 路径3：止损换股 ──
    realized_loss = (current_price - entry_price) * quantity
    capital_freed = current_price * quantity
    # 新标的需涨多少才能弥补亏损
    new_target_gain_pct = (entry_price / current_price - 1) * 100

    if circuit_triggered or decision in ('SELL', 'LONG_TERM_BEAR', 'VALUE_TRAP'):
        rotate_feasibility = 'high'
        rotate_feasibility_text = '高（强烈建议）'
    elif trend_status == 'broken' and loss_level in ('deep', 'extreme'):
        rotate_feasibility = 'high'
        rotate_feasibility_text = '高'
    else:
        rotate_feasibility = 'low'
        rotate_feasibility_text = '低'

    rotate_path = {
        'realized_loss': round(realized_loss, 2),
        'realized_loss_pct': round(pnl_pct, 1),
        'capital_freed': round(capital_freed, 2),
        'new_target_gain_pct': round(new_target_gain_pct, 1),
        'feasibility': rotate_feasibility,
        'feasibility_text': rotate_feasibility_text,
    }

    # ── 推荐决策 ──
    if circuit_triggered or decision in ('SELL', 'LONG_TERM_BEAR', 'VALUE_TRAP'):
        recommendation = 'rotate'
        recommendation_text = '止损换股'
        recommendation_reason = '熔断触发或SMC决策明确看空'
    elif trend_status == 'broken' and loss_level in ('deep', 'extreme'):
        recommendation = 'rotate'
        recommendation_text = '止损换股'
        recommendation_reason = '趋势破坏且亏损深重'
    elif add_ok and loss_level in ('shallow', 'moderate'):
        recommendation = 'add'
        recommendation_text = '补仓摊低成本'
        recommendation_reason = '技术条件满足，补仓可显著降低回本门槛'
    elif trend_status == 'intact' and required_gain_pct < 15:
        recommendation = 'hold'
        recommendation_text = '持仓等待'
        recommendation_reason = '趋势完好，所需涨幅在合理范围'
    elif add_ok:
        recommendation = 'add'
        recommendation_text = '补仓摊低成本'
        recommendation_reason = '技术条件满足补仓'
    else:
        recommendation = 'hold'
        recommendation_text = '持仓等待'
        recommendation_reason = '暂不满足加仓条件，耐心等待趋势修复'

    return {
        'available': True,
        'pnl_pct': round(pnl_pct, 1),
        'hold_path': hold_path,
        'add_path': add_path,
        'rotate_path': rotate_path,
        'recommendation': recommendation,
        'recommendation_text': recommendation_text,
        'recommendation_reason': recommendation_reason,
    }


def _print_breakeven_analysis(analysis: dict) -> None:
    """打印回本路径分析结果。"""
    if not analysis.get('available'):
        return

    print(f"\n{'─'*60}")
    print(f"  【回本路径分析】当前浮亏 {analysis['pnl_pct']:.1f}%")
    print(f"{'─'*60}")

    hold = analysis['hold_path']
    add = analysis['add_path']
    rotate = analysis['rotate_path']
    rec = analysis['recommendation']

    # 路径1：持有
    marker = '★ 推荐' if rec == 'hold' else '  '
    print(f"\n  {marker} 路径1：持有等待回本")
    print(f"    需从当前价上涨: +{hold['required_gain_pct']:.1f}%")
    print(f"    可行性: {hold['feasibility_text']}")

    # 路径2：补仓
    print(f"\n  {'★ 推荐' if rec == 'add' else '  '} 路径2：补仓摊低成本")
    if add.get('available'):
        print(f"    建议买入: {add['add_shares']} 股 @ {add['add_price']:.2f}")
        print(f"    新均价: {add['new_avg_cost']:.4f}（成本降低 {add['cost_reduction_pct']:.1f}%）")
        print(f"    新均价回本需上涨: +{add['new_required_gain_pct']:.1f}%")
        print(f"    所需资金: {add['capital_required']:,.0f}")
        print(f"    可行性: {add['feasibility_text']}")
    else:
        reasons = add.get('blocked_reasons', [])
        print(f"    状态: 条件未满足")
        for r in reasons:
            print(f"      {r}")

    # 路径3：止损换股
    print(f"\n  {'★ 推荐' if rec == 'rotate' else '  '} 路径3：止损换股")
    print(f"    实现亏损: {rotate['realized_loss']:,.0f}（{rotate['realized_loss_pct']:.1f}%）")
    print(f"    释放资金: {rotate['capital_freed']:,.0f}")
    print(f"    新标的需涨: +{rotate['new_target_gain_pct']:.1f}% 才能弥补亏损")
    print(f"    可行性: {rotate['feasibility_text']}")

    # 总推荐
    print(f"\n{'─'*60}")
    print(f"  ★ 最优路径: 【{analysis['recommendation_text']}】")
    print(f"  理由: {analysis['recommendation_reason']}")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser(
        description='股票K线技术分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 分析贵州茅台日K线，最近60天
  python stock_analyzer.py 600519 -d 60

  # 分析苹果股票周K线，最近30周
  python stock_analyzer.py AAPL -p w -d 30

  # 分析腾讯控股
  python stock_analyzer.py 0700.HK -d 60

  # 批量分析多只股票
  python stock_analyzer.py --portfolio 0700.HK 1810.HK QQQ TSLA NVDA -d 60

  # 执行压力测试
  python stock_analyzer.py NVDA -d 60 --stress-test
        '''
    )

    parser.add_argument('code', nargs='?', help='股票代码（A股6位数字，美股代码如AAPL）')
    parser.add_argument('-p', '--period', choices=['d', 'w', 'm'], default='w',
                        help='K线周期：d=日线，w=周线，m=月线 (默认: w)')
    parser.add_argument('-d', '--days', type=int, default=1000,
                        help='分析天数 (默认: 1000，约3-5年)')
    parser.add_argument('--demo', action='store_true',
                        help='使用演示数据模式（无需网络）')
    parser.add_argument('--portfolio', nargs='+',
                        help='批量分析多只股票 (例: --portfolio 0700.HK 1810.HK QQQ TSLA NVDA)')
    parser.add_argument('--stress-test', action='store_true',
                        help='执行 15%% 回撤压力测试')
    parser.add_argument('--entry-price', type=float, default=None,
                        help='持仓买入均价（用于持仓应对策略，不指定则使用筹码成本中心估算）')

    args = parser.parse_args()

    try:
        # 批量分析模式
        if args.portfolio:
            codes = [c.upper() for c in args.portfolio]
            results = analyze_portfolio(codes, args.period, args.days, args.demo, args.stress_test)
            print_portfolio_analysis(results, period=args.period)
            return

        # 单股票分析模式
        if not args.code:
            parser.error('请提供股票代码或使用 --portfolio 进行批量分析')

        args.code = args.code.upper()

        # 验证股票代码格式
        is_valid, error_msg = validate_stock_code(args.code)
        if not is_valid:
            print(f"错误: {error_msg}")
            sys.exit(1)

        # 获取数据
        print(f"正在获取 {args.code} 数据...")
        df = fetch_stock_data(args.code, args.period, args.days, demo=args.demo)

        if df.empty:
            print("错误：无法获取股票数据，请检查股票代码是否正确")
            sys.exit(1)

        # 数据清洗和验证
        is_valid, error_msg, df_clean = validate_data(df, min_rows=30)
        if not is_valid:
            print(f"错误: {error_msg}")
            sys.exit(1)

        print(f"成功获取 {len(df_clean)} 条有效数据")

        # 数据量建议提示
        if len(df_clean) < 100:
            print(f"提示: 数据量较少（{len(df_clean)}条），结论仅供参考，建议使用 -d 120 获取更多数据")

        # 分析并输出
        print_analysis(df_clean, args.code, args.period, demo=args.demo, stress_test=args.stress_test, entry_price=args.entry_price)

    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(0)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e).lower()
        if 'connection' in error_msg or 'network' in error_msg or 'timeout' in error_msg:
            print("错误: 网络连接失败，请检查网络后重试")
        elif 'rate' in error_msg or 'limit' in error_msg:
            print("错误: API请求频率限制，请稍后重试")
        else:
            print(f"发生未知错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
