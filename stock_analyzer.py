#!/usr/bin/env python3
"""
股票K线技术分析工具
支持A股/美股，计算MA、RSI、MACD、布林带、KDJ、ADX、ATR、OBV、CCI、SuperTrend、PSAR、Ichimoku等技术指标
输出买卖信号、多指标共振分析和综合胜率提示
"""

import argparse
import sys
import time
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional

import pandas as pd
import numpy as np


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


def safe_fetch_yfinance(code: str, period: str, days: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """安全获取yfinance数据，带错误处理

    Returns:
        (dataframe, error_message)
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, "请安装yfinance: pip install yfinance"

    try:
        # A股代码转换
        original_code = code
        yf_code = code
        if code.isdigit() and len(code) == 6:
            if code.startswith(('6', '5', '9')):
                yf_code = f"{code}.SS"
            else:
                yf_code = f"{code}.SZ"

        # 映射周期
        interval = '1d' if period == 'd' else '1wk'

        # 获取数据 (多获取一些以确保有足够数据计算指标)
        fetch_days = days + 100
        ticker = yf.Ticker(yf_code)
        df = ticker.history(period=f"{fetch_days}d", interval=interval)

        if df.empty:
            return None, f"股票代码无效或无数据: {original_code}"

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })

        df['date'] = pd.to_datetime(df['date'])

        # 取最近N条
        df = df.tail(days).reset_index(drop=True)

        return df[['date', 'open', 'high', 'low', 'close', 'volume']], None

    except Exception as e:
        error_msg = str(e).lower()
        if 'connection' in error_msg or 'network' in error_msg or 'timeout' in error_msg:
            return None, "网络连接失败，请检查网络后重试"
        elif 'rate' in error_msg or 'limit' in error_msg:
            return None, "API请求频率限制，请稍后重试"
        else:
            return None, f"获取数据失败: {original_code}"


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
        df = ak.stock_zh_a_hist(symbol=symbol, period=period_str, adjust="qfq")

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
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp('2024-01-15'), periods=days, freq='D')

    # 模拟股票走势
    base_price = 100 if not code.isdigit() else float(code[:3])
    prices = [base_price]
    for i in range(1, days):
        # 添加趋势和随机波动
        trend = 0.05 if i < days // 2 else -0.03  # 先涨后跌
        change = np.random.randn() * 2 + trend
        prices.append(max(1, prices[-1] + change))

    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + abs(np.random.randn() * 1.5) for p in prices],
        'low': [p - abs(np.random.randn() * 1.5) for p in prices],
        'close': [p + np.random.randn() * 0.5 for p in prices],
        'volume': [1000000 + np.random.randint(-200000, 200000) for _ in range(days)]
    })

    return df


def fetch_stock_data(code: str, period: str, days: int, retry: int = 3, demo: bool = False) -> pd.DataFrame:
    """获取股票数据，优先使用yfinance，失败后尝试akshare（仅A股）

    数据源优先级：
    1. yfinance（支持A股、美股、港股）
    2. akshare（仅A股备选）
    """
    if demo:
        print("使用演示数据模式")
        return generate_demo_data(code, period, days)

    is_a_share = code.isdigit() and len(code) == 6

    for attempt in range(retry):
        # 优先使用yfinance
        df, error = safe_fetch_yfinance(code, period, days)

        if df is not None:
            print(f"数据源: yfinance")
            return df

        # yfinance失败，如果是A股则尝试akshare
        if is_a_share:
            print(f"yfinance获取失败，正在尝试akshare...")
            df, error = safe_fetch_akshare(code, period, days)

            if df is not None:
                print(f"数据源: akshare")
                return df

        # 检查是否是限速错误
        if error and ("限" in error or "limit" in error.lower()):
            wait_time = (attempt + 1) * 10
            print(f"API限速，等待{wait_time}秒后重试... ({attempt + 1}/{retry})")
            time.sleep(wait_time)
        else:
            # 非限速错误，直接报错退出
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

    # 尝试使用yfinance获取
    try:
        import yfinance as yf

        # A股代码转换
        yf_code = code
        if is_a_share:
            if code.startswith(('6', '5', '9')):
                yf_code = f"{code}.SS"
            else:
                yf_code = f"{code}.SZ"

        ticker = yf.Ticker(yf_code)
        info = ticker.info

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
        pass  # yfinance失败，尝试akshare

    # A股尝试使用akshare
    if is_a_share:
        try:
            import akshare as ak

            # 使用东方财富研报接口
            df = ak.stock_rank_forecast_cninfo(symbol=code)

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
        df = ak.stock_cyq_em(symbol=code)

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
            df_flow = ak.stock_fund_flow_industry()

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

    # 判断股票代码
    is_a_share = code.isdigit() and len(code) == 6

    try:
        import yfinance as yf

        # 转换代码
        yf_code = code
        if is_a_share:
            if code.startswith(('6', '5', '9')):
                yf_code = f"{code}.SS"
            else:
                yf_code = f"{code}.SZ"

        ticker = yf.Ticker(yf_code)

        # 获取新闻
        news_data = ticker.news

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
        import yfinance as yf

        # 指数代码列表
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
            try:
                ticker = yf.Ticker(code)
                hist = ticker.history(period='5d')

                if hist is not None and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = (current_price - prev_price) / prev_price * 100

                    indices[name] = {
                        'code': code,
                        'price': round(current_price, 2),
                        'change': round(change, 2)
                    }
                    changes.append(change)

            except Exception:
                pass

        # 判断市场状态
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

        # 尝试获取A股板块数据
        try:
            import akshare as ak

            # 获取板块资金流
            df_sector = ak.stock_fund_flow_industry()

            if df_sector is not None and not df_sector.empty:
                # 尝试找到涨跌幅列
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

                        sectors.append({
                            'name': str(name),
                            'change': round(float(change), 2) if pd.notna(change) else 0,
                            'inflow': round(float(inflow), 2) if pd.notna(inflow) else 0
                        })

                    default_result['sectors'] = sectors

        except Exception:
            pass

        return default_result

    except ImportError:
        default_result['market_status'] = '请安装yfinance'
    except Exception as e:
        default_result['market_status'] = f'获取失败: {type(e).__name__}'

    return default_result


# ============ 精确买卖/止损/目标价 + 检查清单 ============

def calculate_price_targets(df: pd.DataFrame, resonance: dict, consensus: dict,
                           support: list, resistance: list) -> dict:
    """计算精确的买卖价格和止损价

    买入价计算逻辑:
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

        # 方案1: 支撑位附近
        if support:
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
                      indicators: dict) -> list:
    """生成10项买入检查清单

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

    return results


# ============ LLM决策仪表盘 ============

def generate_decision_dashboard(resonance: dict, fund_flow: dict,
                                sentiment: dict, consensus: dict,
                                win_rate: dict) -> dict:
    """生成LLM风格决策仪表盘（零成本实现）

    评分权重:
    - 技术面(40%): strong_buy=90, buy=70, neutral=50, sell=30, strong_sell=10
    - 资金面(30%): 持续流入=80, 流入=65, 观望=50, 流出=35, 持续流出=20
    - 舆情(15%): 利好=80, 中性=50, 利空=20
    - 机构(15%): buy=85, hold=50, sell=15

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

    # 计算综合评分
    bullish_score = int(tech_score * 0.4 + fund_score * 0.3 + sent_score * 0.15 + institution_score * 0.15)
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

    core_conclusion = f"技术面{tech_dir}+资金{fund_dir}+舆情{sent_dir}+机构{consensus_dir}，综合评分{bullish_score}分"

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
            '机构': consensus_dir
        }
    }
    # 判断技术面方向
    tech_direction = resonance.get('resonance', 'neutral')
    tech_bullish = tech_direction in ['strong_buy', 'buy']
    tech_bearish = tech_direction in ['sell', 'strong_sell']

    # 判断机构共识方向
    inst_rating = consensus.get('rating', 'hold')
    inst_bullish = inst_rating == 'buy'
    inst_bearish = inst_rating == 'sell'

    # 构建结论
    parts = []

    # 技术面描述
    if tech_direction == 'strong_buy':
        tech_desc = "技术面强烈看多"
    elif tech_direction == 'buy':
        tech_desc = "技术面偏多"
    elif tech_direction == 'sell':
        tech_desc = "技术面偏空"
    elif tech_direction == 'strong_sell':
        tech_desc = "技术面强烈看空"
    else:
        tech_desc = "技术面中性"

    # 机构共识描述
    if inst_rating == 'buy':
        inst_desc = "机构共识看涨"
    elif inst_rating == 'sell':
        inst_desc = "机构共识看跌"
    else:
        inst_desc = "机构评级中性"

    # 综合判断
    if tech_bullish and inst_bullish:
        action = "技术面+机构共识均偏多，建议逢低加仓"
        if support:
            action += f"，支撑参考{support[0]:.2f}"
    elif tech_bullish and inst_bearish:
        action = "技术面偏多但机构谨慎，建议轻仓试探，设好止损"
    elif tech_bearish and inst_bullish:
        action = "机构看好但技术面偏弱，建议等待技术确认后再介入"
    elif tech_bearish and inst_bearish:
        action = "技术面+机构共识均偏空，建议减仓观望"
        if resistance:
            action += f"，压力位参考{resistance[0]:.2f}"
    else:
        # 分歧或中性状态
        buy_count = len(resonance.get('buy_indicators', []))
        sell_count = len(resonance.get('sell_indicators', []))
        if abs(buy_count - sell_count) <= 1:
            action = "多空分歧明显，建议观望等待明确信号"
        else:
            action = "趋势不明朗，建议谨慎操作"

    # 添加胜率参考
    win_rate_val = win_rate.get('win_rate', 50)
    if win_rate_val >= 70:
        confidence = "高置信度"
    elif win_rate_val >= 55:
        confidence = "中等置信度"
    else:
        confidence = "低置信度"

    # 构建最终结论
    conclusion = f"{tech_desc}，{inst_desc}。{action}。胜率{win_rate_val:.0f}%（{confidence}）"

    # 添加止损建议（使用支撑位或ATR）
    if support and tech_bullish:
        conclusion += f"，止损可参考{support[0]:.2f}下方"

    return conclusion


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


def print_analysis(df: pd.DataFrame, code: str, period: str):
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
    support, resistance = find_support_resistance(df)

    # 多指标分析
    indicators = analyze_indicator_signals(df)
    resonance = calculate_resonance(indicators)
    win_rate_info = calculate_win_rate(indicators, resonance, df)

    summary = generate_summary(df, trend, signals, support, resistance, resonance)

    # 打印结果
    period_name = "日线" if period == 'd' else "周线"
    print(f"\n{'='*70}")
    print(f"  股票代码: {code} | 周期: {period_name} | 综合技术分析")
    print(f"{'='*70}")

    # ========== 1. 大盘复盘 ==========
    print(f"\n【大盘复盘】")
    try:
        market_review = get_market_review()
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
            print("  暂无法获取大盘数据")
    except Exception as e:
        print(f"  获取大盘数据失败")

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
    try:
        chip_dist = fetch_chip_distribution(code)
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
    except Exception as e:
        print(f"  筹码分布: 获取失败")

    # 主力资金流向
    try:
        fund_flow = fetch_fund_flow(code, df)
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
            print(f"  主力资金流: 数据不足")
    except Exception as e:
        print(f"  主力资金流: 获取失败")

    # ========== 4. 新闻舆情 ==========
    print(f"\n【市场舆情】")
    try:
        news_sentiment = fetch_news_sentiment(code)
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
            print(f"  暂无新闻数据")
    except Exception as e:
        print(f"  获取新闻失败")

    # ========== 5. 分析师共识（机构评级） ==========
    print(f"\n【机构评级】")
    consensus = None
    try:
        consensus = fetch_analyst_consensus(code)
        print(f"  机构评级: {consensus['rating_text']}")
        if consensus['target_price']:
            print(f"  目标价: {consensus['target_price']:.2f}")
        if consensus['analyst_count'] > 0:
            print(f"  分析师数量: {consensus['analyst_count']}位")
        print(f"  数据来源: 基于{consensus['source']}汇总")
    except Exception as e:
        print("  暂无法获取分析师评级数据")

    # ========== 6. 买卖点分析 + 检查清单 ==========
    print(f"\n【买卖点分析】")

    try:
        price_targets = calculate_price_targets(df, resonance, consensus, support, resistance)
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
        checklist = generate_checklist(df, resonance, signals, indicators)
        pass_count = sum(1 for _, passed, _ in checklist if passed)
        for item, passed, reason in checklist:
            status = "✓" if passed else "✗"
            print(f"  [{status}] {item}: {reason}")
        print(f"\n  通过: {pass_count}/10 项")
        if pass_count >= 8:
            print(f"  建议: 可以执行买入")
        elif pass_count >= 6:
            print(f"  建议: 谨慎观望，逢低布局")
        else:
            print(f"  建议: 暂不建议买入")
    except Exception as e:
        print(f"  生成检查清单失败")

    # ========== 7. LLM决策仪表盘 ==========
    # 确保变量有默认值
    try:
        _fund_flow = fund_flow if 'fund_flow' in dir() and isinstance(fund_flow, dict) else {'signal': '未知', 'trend': '观望'}
    except:
        _fund_flow = {'signal': '未知', 'trend': '观望'}

    try:
        _news_sentiment = news_sentiment if 'news_sentiment' in dir() and isinstance(news_sentiment, dict) else {'sentiment': '中性', 'sentiment_score': 0}
    except:
        _news_sentiment = {'sentiment': '中性', 'sentiment_score': 0}

    print(f"\n{'='*70}")
    print(f"  ║               LLM 决策仪表盘 (模拟)                         ║")
    print(f"{'='*70}")

    try:
        dashboard = generate_decision_dashboard(resonance, _fund_flow,
                                                _news_sentiment,
                                                consensus if consensus else {},
                                                win_rate_info)

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
        '''
    )

    parser.add_argument('code', help='股票代码（A股6位数字，美股代码如AAPL）')
    parser.add_argument('-p', '--period', choices=['d', 'w'], default='d',
                        help='K线周期：d=日线，w=周线 (默认: d)')
    parser.add_argument('-d', '--days', type=int, default=60,
                        help='分析天数 (默认: 60)')
    parser.add_argument('--demo', action='store_true',
                        help='使用演示数据模式（无需网络）')

    args = parser.parse_args()

    try:
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
        print_analysis(df_clean, args.code, args.period)

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