#!/usr/bin/env python3
"""
股票K线技术分析工具
支持A股/美股，计算MA、RSI、MACD、布林带、KDJ、ADX、ATR、OBV、CCI、SuperTrend、PSAR、Ichimoku等技术指标
输出买卖信号、多指标共振分析和综合胜率提示
"""

import argparse
import contextlib
import io
import sys
import time
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional, Callable, Any

import pandas as pd
import numpy as np


REQUEST_TIMEOUT = 10
CHART_FETCH_BUFFER_DAYS = 320


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
    for col in ['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'KDJ_K', 'ADX', 'ATR']:
        if col in df.columns and pd.isna(last[col]):
            nan_indicators.append(col)
    if nan_indicators:
        warnings.append(f"以下指标计算结果为空: {', '.join(nan_indicators)}")

    # RSI范围检查 (0-100)
    if 'RSI' in df.columns and pd.notna(last['RSI']):
        if last['RSI'] < 0 or last['RSI'] > 100:
            warnings.append(f"RSI值异常: {last['RSI']:.2f} (应在0-100之间)")

    # KDJ范围检查 (通常0-100，允许略微超出)
    for col in ['KDJ_K', 'KDJ_D', 'KDJ_J']:
        if col in df.columns and pd.notna(last[col]):
            if last[col] < -20 or last[col] > 120:
                warnings.append(f"{col}值异常: {last[col]:.2f}")

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

    try:
        yf_code = normalize_symbol(code)
        end_time = int(time.time())
        start_time = end_time - get_history_window_days(period, days) * 86400

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_code}"
        params = {
            'period1': start_time,
            'period2': end_time,
            'interval': '1d',
            'events': 'history'
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            return None, f"API请求失败: HTTP {response.status_code}"

        data = response.json()
        return parse_chart_response(data, code, period, days)

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
        else:
            raise ValueError(error if error else "获取数据失败，请检查股票代码或网络连接")

    raise ValueError("多次重试后仍无法获取数据，请稍后再试")


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


# ============ 精确买卖/止损/目标价 + 检查清单 ============

def calculate_price_targets(df: pd.DataFrame, resonance: dict, consensus: dict,
                           support: list, resistance: list, contrarian: dict = None) -> dict:
    """计算精确的买卖价格和止损价

    买入价计算逻辑:
    - 最优先: 回调到看多Order Block上沿
    - 优先: 回调到支撑位附近
    - 次选: 回调到MA10/MA20均线
    - 备选: 当前价格 * 0.98

    止损价计算逻辑:
    - 优先: 跌破关键支撑位
    - 次选: ATR止损 (close - 2*ATR)
    - 备选: 买入价 * 0.95

    目标价计算逻辑:
    - 优先: 目标价来自机构共识
    - 次选: 上涨至前高/压力位
    - 备选: 上涨空间 = 止损幅度的2倍
    """
    last = df.iloc[-1]
    current_price = last['close']

    default_result = {
        'buy_price': None,
        'stop_loss': None,
        'target_price': None,
        'risk_reward': None,
        'position_size': None,
        'buy_reason': '',
        'stop_reason': '',
        'target_reason': ''
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

        # ========== 止损价 ==========
        stop_loss = None
        stop_reason = ''

        # 方案1: 跌破支撑位下方
        if support:
            valid_supports = [s for s in support if s < buy_price]
            if valid_supports:
                stop_loss = min(valid_supports) * 0.98
                stop_reason = f'跌破支撑位({min(valid_supports):.2f})'

        # 方案2: ATR止损
        if stop_loss is None:
            atr = last.get('ATR')
            if pd.notna(atr):
                stop_loss = buy_price - 2 * atr
                stop_reason = f'ATR止损({2*atr:.2f})'

        # 方案3: 5%止损
        if stop_loss is None:
            stop_loss = buy_price * 0.95
            stop_reason = '5%固定止损'

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
        if target_price is None:
            risk = buy_price - stop_loss
            target_price = buy_price + risk * 2
            target_reason = '2倍风险报酬'

        # ========== 风险报酬比 ==========
        risk = buy_price - stop_loss
        reward = target_price - buy_price

        if risk > 0:
            risk_reward = round(reward / risk, 1)
        else:
            risk_reward = None

        # ========== 建议仓位 ==========
        if risk_reward:
            if risk_reward >= 3:
                position_size = 30
            elif risk_reward >= 2:
                position_size = 20
            elif risk_reward >= 1:
                position_size = 15
            else:
                position_size = 10
        else:
            position_size = 10

        return {
            'buy_price': round(buy_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target_price': round(target_price, 2),
            'risk_reward': risk_reward,
            'position_size': position_size,
            'buy_reason': buy_reason,
            'stop_reason': stop_reason,
            'target_reason': target_reason
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

    # 1. MA多头排列
    ma5 = last.get('MA5')
    ma10 = last.get('MA10')
    ma20 = last.get('MA20')

    if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20):
        is_bullish = ma5 > ma10 > ma20
        reason = f"MA5>{ma10:.1f}>MA20{ma10:.1f}" if is_bullish else "非多头排列"
        results.append(("MA均线多头排列", is_bullish, reason))
    else:
        results.append(("MA均线多头排列", False, "数据不足"))

    # 2. 乖离率
    if pd.notna(ma20):
        bias = (last['close'] - ma20) / ma20 * 100
        is_ok = abs(bias) < 8
        reason = f"乖离率{bias:.1f}%" if is_ok else f"乖离率过高({bias:.1f}%)"
        results.append(("乖离率合理(<8%)", is_ok, reason))
    else:
        results.append(("乖离率合理(<8%)", False, "MA20数据不足"))

    # 3. RSI未超买
    rsi = last.get('RSI')
    if pd.notna(rsi):
        is_ok = rsi < 75
        reason = f"RSI={rsi:.1f}" if is_ok else f"RSI超买({rsi:.1f})"
        results.append(("RSI未超买(<75)", is_ok, reason))
    else:
        results.append(("RSI未超买(<75)", False, "RSI数据不足"))

    # 4. KDJ未超买
    kdj_j = last.get('KDJ_J')
    if pd.notna(kdj_j):
        is_ok = kdj_j < 85
        reason = f"KDJ_J={kdj_j:.1f}" if is_ok else f"KDJ超买({kdj_j:.1f})"
        results.append(("KDJ未超买(<85)", is_ok, reason))
    else:
        results.append(("KDJ未超买(<85)", False, "KDJ数据不足"))

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


# ============ LLM决策仪表盘 ============

def generate_decision_dashboard(resonance: dict, fund_flow: dict,
                                sentiment: dict, consensus: dict,
                                win_rate: dict, contrarian: dict = None) -> dict:
    """生成LLM风格决策仪表盘（零成本实现）

    评分权重:
    - 技术面(30%): strong_buy=90, buy=70, neutral=50, sell=30, strong_sell=10
    - 资金面(25%): 持续流入=80, 流入=65, 观望=50, 流出=35, 持续流出=20
    - 舆情(10%): 利好=80, 中性=50, 利空=20
    - 机构(10%): buy=85, hold=50, sell=15
    - 左侧信号(25%): 来自contrarian composite_score

    Returns:
        {
            'core_conclusion': str,
            'bullish_score': int,
            'bearish_score': int,
            'sideways_score': int,
            'verdict': str,
            'action': str,
            'confidence': str,
            'factors': dict
        }
    """
    # 技术面评分 (40%)
    tech_score = {
        'strong_buy': 90,
        'buy': 70,
        'neutral': 50,
        'sell': 30,
        'strong_sell': 10
    }.get(resonance.get('resonance', 'neutral'), 50)

    # 资金面评分 (30%)
    fund_signal = fund_flow.get('signal', '未知')
    fund_trend = fund_flow.get('trend', '观望')

    if fund_trend == '持续流入':
        fund_score = 80
    elif fund_signal == '净流入':
        fund_score = 65
    elif fund_trend == '反弹':
        fund_score = 60
    elif fund_signal == '净流出':
        fund_score = 35
    elif fund_trend == '持续流出':
        fund_score = 20
    else:
        fund_score = 50

    # 舆情评分 (15%)
    sent_score = {
        '利好': 80,
        '中性': 50,
        '利空': 20
    }.get(sentiment.get('sentiment', '中性'), 50)

    # 机构评分 (15%)
    consensus_rating = consensus.get('rating', 'hold') if consensus else 'hold'
    institution_score = {
        'buy': 85,
        'hold': 50,
        'sell': 15
    }.get(consensus_rating, 50)

    # 左侧信号评分 (25%)
    contrarian_score = 50
    if contrarian and contrarian.get('composite_score') is not None:
        contrarian_score = contrarian['composite_score']

    # 计算综合评分
    bullish_score = int(tech_score * 0.30 + fund_score * 0.25 + sent_score * 0.10 +
                        institution_score * 0.10 + contrarian_score * 0.25)
    bearish_score = 100 - bullish_score

    # 震荡评分（基于ADX或市场状态）
    sideways_score = min(bullish_score, bearish_score) * 0.5

    # 判断多空
    if bullish_score >= 70:
        verdict = '看多'
    elif bearish_score >= 70:
        verdict = '看空'
    else:
        verdict = '震荡'

    # 行动建议
    if verdict == '看多':
        if bullish_score >= 85:
            action = '加仓'
        else:
            action = '买入'
    elif verdict == '看空':
        if bearish_score >= 85:
            action = '清仓'
        else:
            action = '减仓'
    else:
        action = '持仓观望'

    # 置信度
    diff = abs(bullish_score - bearish_score)
    if diff >= 40:
        confidence = '高'
    elif diff >= 20:
        confidence = '中'
    else:
        confidence = '低'

    # 核心结论
    tech_dir = '偏多' if tech_score >= 60 else ('偏空' if tech_score <= 40 else '中性')
    fund_dir = '流入' if fund_score >= 60 else ('流出' if fund_score <= 40 else '观望')
    sent_dir = sentiment.get('sentiment', '中性')
    consensus_dir = {'buy': '看涨', 'hold': '观望', 'sell': '看跌'}.get(consensus_rating, '观望')
    contrarian_dir = '左侧看多' if contrarian_score >= 65 else ('左侧看空' if contrarian_score <= 35 else '左侧中性')

    core_conclusion = f"技术面{tech_dir}+资金{fund_dir}+舆情{sent_dir}+机构{consensus_dir}+{contrarian_dir}，综合评分{bullish_score}分"

    return {
        'core_conclusion': core_conclusion,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'sideways_score': int(sideways_score),
        'verdict': verdict,
        'action': action,
        'confidence': confidence,
        'factors': {
            '技术面': tech_dir,
            '资金面': fund_dir,
            '舆情': sent_dir,
            '机构': consensus_dir,
            '左侧信号': contrarian_dir
        }
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

    # 移动平均线
    try:
        df['MA5'] = calc_sma(df['close'], 5)
        df['MA10'] = calc_sma(df['close'], 10)
        df['MA20'] = calc_sma(df['close'], 20)
    except Exception:
        df['MA5'] = np.nan
        df['MA10'] = np.nan
        df['MA20'] = np.nan

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

    # KDJ指标
    try:
        df = calc_kdj(df)
    except Exception:
        df['KDJ_K'] = np.nan
        df['KDJ_D'] = np.nan
        df['KDJ_J'] = np.nan

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


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ指标计算 - 添加除零保护

    Args:
        df: 包含high, low, close的DataFrame
        n: RSV计算周期，默认9
        m1: K值平滑周期，默认3
        m2: D值平滑周期，默认3

    Returns:
        添加了KDJ指标的DataFrame
    """
    low_n = df['low'].rolling(window=n).min()
    high_n = df['high'].rolling(window=n).max()

    # 除零保护：当high_n == low_n时，RSV取平衡点50
    denom = high_n - low_n
    rsv = np.where(denom != 0,
                   (df['close'] - low_n) / denom * 100,
                   50.0)  # 平衡点
    rsv = pd.Series(rsv, index=df.index).fillna(50)

    # K值 = RSV的M1日指数移动平均
    df['KDJ_K'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    # D值 = K值的M2日指数移动平均
    df['KDJ_D'] = df['KDJ_K'].ewm(alpha=1/m2, adjust=False).mean()
    # J值 = 3*K - 2*D
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']

    return df


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


def calc_contrarian_signals(df: pd.DataFrame) -> dict:
    """左侧交易信号综合分析 — 聚合5个模块，生成综合评分和仓位建议。"""
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

    # 仓位建议 (左侧入场、右侧确认)
    position_advice = {
        'initial_position': 0, 'confirm_position': 0,
        'confirm_conditions': [], 'stop_loss': None
    }
    if signal == 'strong_contrarian_buy':
        position_advice.update(
            initial_position=10, confirm_position=20,
            confirm_conditions=['MACD金叉', '站上5日线', '放量突破']
        )
    elif signal == 'contrarian_buy':
        position_advice.update(
            initial_position=5, confirm_position=15,
            confirm_conditions=['KDJ金叉', '站上10日线']
        )

    # 止损: 最近支撑位下方 1*ATR
    try:
        atr = df['ATR'].iloc[-1]
        supports = sr_inst.get('support', [])
        if supports and not pd.isna(atr):
            position_advice['stop_loss'] = round(min(supports) - atr, 2)
    except Exception:
        pass

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

    # KDJ信号
    if pd.notna(last['KDJ_K']) and pd.notna(last['KDJ_D']) and pd.notna(prev['KDJ_K']) and pd.notna(prev['KDJ_D']):
        # KDJ金叉/死叉
        if prev['KDJ_K'] <= prev['KDJ_D'] and last['KDJ_K'] > last['KDJ_D']:
            signals['golden_cross'].append('KDJ金叉')
            signals['buy'].append('KDJ金叉')
        if prev['KDJ_K'] >= prev['KDJ_D'] and last['KDJ_K'] < last['KDJ_D']:
            signals['death_cross'].append('KDJ死叉')
            signals['sell'].append('KDJ死叉')
        # 超买超卖
        if last['KDJ_J'] < 20:
            signals['buy'].append('KDJ超卖')
        elif last['KDJ_J'] > 80:
            signals['sell'].append('KDJ超买')

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

    # MACD
    if pd.notna(last['MACD']) and pd.notna(last['MACD_signal']):
        if last['MACD'] > last['MACD_signal'] and last['MACD_hist'] > 0:
            macd_signal = 'buy'
            macd_status = '多头运行'
        elif last['MACD'] < last['MACD_signal'] and last['MACD_hist'] < 0:
            macd_signal = 'sell'
            macd_status = '空头运行'
        else:
            macd_signal = 'neutral'
            macd_status = '震荡'
        indicators['MACD'] = {'signal': macd_signal, 'status': macd_status, 'weight': 2}

    # KDJ
    if pd.notna(last['KDJ_K']) and pd.notna(last['KDJ_D']):
        if last['KDJ_K'] > last['KDJ_D'] and last['KDJ_J'] < 80:
            kdj_signal = 'buy'
            kdj_status = f'K>D,J={last["KDJ_J"]:.1f}'
        elif last['KDJ_K'] < last['KDJ_D'] and last['KDJ_J'] > 20:
            kdj_signal = 'sell'
            kdj_status = f'K<D,J={last["KDJ_J"]:.1f}'
        elif last['KDJ_J'] < 20:
            kdj_signal = 'buy'
            kdj_status = f'超卖,J={last["KDJ_J"]:.1f}'
        elif last['KDJ_J'] > 80:
            kdj_signal = 'sell'
            kdj_status = f'超买,J={last["KDJ_J"]:.1f}'
        else:
            kdj_signal = 'neutral'
            kdj_status = f'中性,J={last["KDJ_J"]:.1f}'
        indicators['KDJ'] = {'signal': kdj_signal, 'status': kdj_status, 'weight': 1}

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
    """计算多指标共振信号"""
    buy_score = 0
    sell_score = 0
    total_weight = 0
    buy_indicators = []
    sell_indicators = []
    neutral_indicators = []

    for name, info in indicators.items():
        weight = info.get('weight', 1)
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
        'total_weight': total_weight
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
        'notes': [],
    }

    if demo:
        context['notes'].append('演示模式已跳过联网扩展数据')
        context['chip_dist'] = {'source': '演示模式已跳过'}
        context['market_review']['market_status'] = '演示模式已跳过'
        context['news_sentiment']['source'] = '演示模式已跳过'
        context['consensus']['source'] = '演示模式已跳过'
        return context

    try:
        context['market_review'] = get_market_review()
    except Exception:
        context['notes'].append('大盘复盘获取失败，已跳过')

    try:
        context['chip_dist'] = fetch_chip_distribution(code)
    except Exception:
        context['chip_dist'] = {'source': '获取失败'}

    return context


def print_analysis(df: pd.DataFrame, code: str, period: str, demo: bool = False, stress_test: bool = False):
    """打印分析结果（表格格式）

    输出顺序：大盘复盘 → 指标表格 → 筹码/资金流 → 新闻情绪 → 买卖点+清单 → 决策仪表盘 → 综合结论
    """
    # 计算指标
    df = calculate_indicators(df)
    last = df.iloc[-1]

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

    # 左侧交易信号分析
    contrarian = calc_contrarian_signals(df)

    summary = generate_summary(df, trend, signals, support, resistance, resonance)

    # 打印结果
    period_name = "日线" if period == 'd' else "周线"
    print(f"\n{'='*70}")
    print(f"  股票代码: {code} | 周期: {period_name} | 综合技术分析")
    print(f"{'='*70}")
    for note in context['notes']:
        print(f"  说明: {note}")

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

    # 基础指标
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
    if pd.notna(last['KDJ_K']):
        kdj_signal = "超买" if last['KDJ_J'] > 80 else ("超卖" if last['KDJ_J'] < 20 else "中性")
        table_rows.append(("KDJ", f"K={last['KDJ_K']:.1f}", kdj_signal))

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

    # Position Advice
    pa = contrarian['position_advice']
    if pa['initial_position'] > 0:
        print(f"\n  仓位建议 (左侧入场、右侧确认):")
        print(f"    第一步: 左侧试探仓 {pa['initial_position']}%")
        print(f"    第二步: 右侧确认后加仓至 {pa['initial_position'] + pa['confirm_position']}%")
        print(f"    确认条件: {', '.join(pa['confirm_conditions'])}")
        if pa.get('stop_loss'):
            print(f"    止损价: {pa['stop_loss']:.2f}")
    else:
        print(f"\n  仓位建议: 暂不建议左侧入场，等待更明确信号")

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

    # ========== 5.9 压力测试 (如果启用) ==========
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

    # ========== 6. 买卖点分析 + 检查清单 ==========
    print(f"\n【买卖点分析】")

    try:
        price_targets = calculate_price_targets(df, resonance, consensus, support, resistance, contrarian=contrarian)
        print(f"  建议买入价: {price_targets['buy_price']:.2f}元 ({price_targets.get('buy_reason', '')})")
        print(f"  建议止损价: {price_targets['stop_loss']:.2f}元 ({price_targets.get('stop_reason', '')})")
        print(f"  目标价: {price_targets['target_price']:.2f}元 ({price_targets.get('target_reason', '')})")

        if price_targets.get('risk_reward'):
            print(f"  风险报酬比: 1:{price_targets['risk_reward']}")
        print(f"  建议仓位: {price_targets.get('position_size', 10)}%")

    except Exception as e:
        print(f"  计算买卖点失败")

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

    # ========== 7. LLM决策仪表盘 ==========
    _fund_flow = fund_flow if isinstance(fund_flow, dict) else {'signal': '未知', 'trend': '观望'}
    _news_sentiment = news_sentiment if isinstance(news_sentiment, dict) else {'sentiment': '中性', 'sentiment_score': 0}

    print(f"\n{'='*70}")
    print(f"  ║               LLM 决策仪表盘 (模拟)                         ║")
    print(f"{'='*70}")

    try:
        dashboard = generate_decision_dashboard(resonance, _fund_flow,
                                                _news_sentiment,
                                                consensus if consensus else {},
                                                win_rate_info,
                                                contrarian=contrarian)

        print(f"  一句话结论:")
        print(f"  {dashboard.get('core_conclusion', '综合分析中...')}")
        print()

        # 评分条
        bullish = dashboard.get('bullish_score', 50)
        bearish = dashboard.get('bearish_score', 50)

        # 绘制简单条形图
        b_bar = "█" * (bullish // 5) + "░" * (20 - bullish // 5)
        be_bar = "█" * (bearish // 5) + "░" * (20 - bearish // 5)

        print(f"  评分:  [{b_bar}] 看多 {bullish}分")
        print(f"         [{be_bar}] 看空 {bearish}分")
        print()

        verdict = dashboard.get('verdict', '震荡')
        action = dashboard.get('action', '观望')
        confidence = dashboard.get('confidence', '低')

        print(f"  行动建议: 【{action}】  置信度: {confidence}")
        print()

        print(f"  影响因素:")
        factors = dashboard.get('factors', {})
        for key, value in factors.items():
            icon = "▲" if '多' in str(value) or '涨' in str(value) or '流入' in str(value) else ("▼" if '空' in str(value) or '跌' in str(value) or '流出' in str(value) else "○")
            print(f"    {key}  {icon} {value}")

    except Exception as e:
        print(f"  生成决策仪表盘失败")

    print(f"{'='*70}")

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


# ============ 批量分析与压力测试模块 ============

def analyze_portfolio(codes: list, period: str = 'd', days: int = 60, demo: bool = False, stress_test: bool = False) -> dict:
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
            df = fetch_stock_data(code, period, days, demo)
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

            # 价格目标
            price_targets = calculate_price_targets(
                df_clean, code, resonance, sr_inst, contrarian, demo
            )

            # 决策仪表盘
            optional_ctx = build_optional_context(code, df_clean, demo)
            dashboard = generate_decision_dashboard(
                resonance,
                optional_ctx.get('fund_flow', {}),
                optional_ctx.get('sentiment', {}),
                optional_ctx.get('consensus', {}),
                win_rate_info,
                contrarian
            )

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

            rows.append({
                '股票代码': code,
                '当前阶段': stage,
                '多头强度': dashboard['bullish_score'],
                '空头强度': dashboard['bearish_score'],
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


def print_portfolio_analysis(results: dict):
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
            df, code, 'd',
            demo=False,
            stress_test=(analysis.get('stress_result') is not None)
        )


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
    parser.add_argument('-p', '--period', choices=['d', 'w'], default='d',
                        help='K线周期：d=日线，w=周线 (默认: d)')
    parser.add_argument('-d', '--days', type=int, default=60,
                        help='分析天数 (默认: 60)')
    parser.add_argument('--demo', action='store_true',
                        help='使用演示数据模式（无需网络）')
    parser.add_argument('--portfolio', nargs='+',
                        help='批量分析多只股票 (例: --portfolio 0700.HK 1810.HK QQQ TSLA NVDA)')
    parser.add_argument('--stress-test', action='store_true',
                        help='执行 15%% 回撤压力测试')

    args = parser.parse_args()

    try:
        # 批量分析模式
        if args.portfolio:
            codes = [c.upper() for c in args.portfolio]
            results = analyze_portfolio(codes, args.period, args.days, args.demo, args.stress_test)
            print_portfolio_analysis(results)
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
        print_analysis(df_clean, args.code, args.period, demo=args.demo, stress_test=args.stress_test)

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
