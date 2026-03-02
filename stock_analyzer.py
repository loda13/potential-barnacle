#!/usr/bin/env python3
"""
股票K线技术分析工具
支持A股/美股，计算MA、RSI、MACD、布林带等技术指标，输出买卖信号和趋势分析
"""

import argparse
import sys
import time
from typing import Tuple

import pandas as pd
import numpy as np


def fetch_data_yfinance(code: str, period: str, days: int) -> pd.DataFrame:
    """使用yfinance获取股票数据"""
    import yfinance as yf

    # A股代码转换
    original_code = code
    if code.isdigit() and len(code) == 6:
        if code.startswith(('6', '5', '9')):
            code = f"{code}.SS"
        else:
            code = f"{code}.SZ"

    # 映射周期
    interval = '1d' if period == 'd' else '1wk'

    # 获取数据 (多获取一些以确保有足够数据计算指标)
    fetch_days = days + 50
    ticker = yf.Ticker(code)
    df = ticker.history(period=f"{fetch_days}d", interval=interval)

    if df.empty:
        raise ValueError(f"无法获取股票数据: {original_code}")

    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'Close': 'close',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })

    # 取最近N条
    df = df.tail(days).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def fetch_data_akshare(code: str, period: str, days: int) -> pd.DataFrame:
    """使用akshare获取A股数据"""
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请安装akshare: pip install akshare")

    # 判断市场
    if code.startswith(('6', '5', '9')):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"

    # 获取数据
    period_str = "daily" if period == 'd' else "weekly"
    df = ak.stock_zh_a_hist(symbol=symbol, period=period_str, adjust="qfq")

    # 取最近N天
    df = df.tail(days).reset_index(drop=True)

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
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


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
    """获取股票数据，自动判断A股或美股"""
    if demo:
        print("使用演示数据模式")
        return generate_demo_data(code, period, days)

    is_a_share = code.isdigit() and len(code) == 6

    last_error = None

    for attempt in range(retry):
        try:
            if is_a_share:
                # A股优先用akshare
                try:
                    return fetch_data_akshare(code, period, days)
                except Exception as e:
                    print(f"akshare获取失败: {e}")
                    # 备用yfinance
                    return fetch_data_yfinance(code, period, days)
            else:
                return fetch_data_yfinance(code, period, days)
        except Exception as e:
            last_error = e
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"API限速，等待{wait_time}秒后重试... ({attempt + 1}/{retry})")
                time.sleep(wait_time)
            else:
                raise

    raise last_error


# ============ 技术指标计算函数 ============

def calc_sma(series: pd.Series, length: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=length).mean()


def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=length, adjust=False).mean()


def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """RSI相对强弱指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
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
    """计算所有技术指标"""
    df = df.copy()

    # 移动平均线
    df['MA5'] = calc_sma(df['close'], 5)
    df['MA10'] = calc_sma(df['close'], 10)
    df['MA20'] = calc_sma(df['close'], 20)

    # RSI
    df['RSI'] = calc_rsi(df['close'], 14)

    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calc_macd(df['close'])

    # 布林带
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calc_bollinger_bands(df['close'])

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
    """检测买卖信号"""
    signals = {
        'buy': [],
        'sell': [],
        'golden_cross': [],
        'death_cross': []
    }

    if len(df) < 20:
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

    return signals


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
                     support: list, resistance: list) -> str:
    """生成一句话总结"""
    last = df.iloc[-1]

    parts = []

    # 趋势
    parts.append(f"当前{trend}")

    # 信号
    if signals['buy']:
        parts.append(f"买入信号：{'+'.join(signals['buy'])}")
    if signals['sell']:
        parts.append(f"卖出信号：{'+'.join(signals['sell'])}")

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


def print_analysis(df: pd.DataFrame, code: str, period: str):
    """打印分析结果"""
    if len(df) < 20:
        print("错误：数据不足，请增加天数参数")
        return

    # 计算指标
    df = calculate_indicators(df)
    last = df.iloc[-1]

    # 分析
    trend = analyze_trend(df)
    signals = detect_signals(df)
    support, resistance = find_support_resistance(df)
    summary = generate_summary(df, trend, signals, support, resistance)

    # 打印结果
    period_name = "日线" if period == 'd' else "周线"
    print(f"\n{'='*50}")
    print(f"  股票代码: {code} | 周期: {period_name}")
    print(f"{'='*50}")

    # 最新数据
    print(f"\n【最新数据】")
    print(f"  日期: {last['date'].strftime('%Y-%m-%d')}")
    print(f"  收盘: {last['close']:.2f}")
    print(f"  成交量: {format_volume(last['volume'])}")

    # 技术指标
    print(f"\n【技术指标】")
    print(f"  MA5:  {last['MA5']:.2f}" if pd.notna(last['MA5']) else "  MA5:  N/A")
    print(f"  MA10: {last['MA10']:.2f}" if pd.notna(last['MA10']) else "  MA10: N/A")
    print(f"  MA20: {last['MA20']:.2f}" if pd.notna(last['MA20']) else "  MA20: N/A")
    print(f"  RSI:  {last['RSI']:.2f}" if pd.notna(last['RSI']) else "  RSI:  N/A")
    print(f"  MACD: {last['MACD']:.4f}" if pd.notna(last['MACD']) else "  MACD: N/A")
    print(f"  布林上轨: {last['BB_upper']:.2f}" if pd.notna(last['BB_upper']) else "  布林上轨: N/A")
    print(f"  布林下轨: {last['BB_lower']:.2f}" if pd.notna(last['BB_lower']) else "  布林下轨: N/A")

    # 趋势分析
    print(f"\n【趋势分析】")
    print(f"  当前趋势: {trend}")

    # 买卖信号
    print(f"\n【买卖信号】")
    if signals['golden_cross']:
        print(f"  金叉信号: {', '.join(signals['golden_cross'])}")
    if signals['death_cross']:
        print(f"  死叉信号: {', '.join(signals['death_cross'])}")
    if signals['buy']:
        print(f"  买入信号: {', '.join(signals['buy'])}")
    if signals['sell']:
        print(f"  卖出信号: {', '.join(signals['sell'])}")
    if not any([signals['buy'], signals['sell'], signals['golden_cross'], signals['death_cross']]):
        print("  暂无明确信号")

    # 支撑压力位
    print(f"\n【支撑压力位】")
    print(f"  支撑位: {', '.join(map(str, support)) if support else '暂无'}")
    print(f"  压力位: {', '.join(map(str, resistance)) if resistance else '暂无'}")

    # 一句话总结
    print(f"\n【总结】")
    print(f"  {summary}")
    print(f"{'='*50}\n")


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
        # 获取数据
        print(f"正在获取 {args.code} 数据...")
        df = fetch_stock_data(args.code, args.period, args.days, demo=args.demo)

        if df.empty:
            print("错误：无法获取股票数据")
            sys.exit(1)

        print(f"成功获取 {len(df)} 条数据")

        # 分析并输出
        print_analysis(df, args.code, args.period)

    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()