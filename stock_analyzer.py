#!/usr/bin/env python3
"""
股票K线技术分析工具
支持A股/美股，计算MA、RSI、MACD、布林带、KDJ、ADX、ATR、OBV、CCI、SuperTrend、PSAR、Ichimoku等技术指标
输出买卖信号、多指标共振分析和综合胜率提示
"""

import argparse
import sys
import time
from typing import Tuple, Dict, List

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
                    # 简化错误提示，避免打印冗长的堆栈信息
                    error_msg = str(e)
                    if 'proxy' in error_msg.lower() or 'connection' in error_msg.lower():
                        print("akshare网络连接失败，正在切换备用数据源...")
                    else:
                        print(f"akshare获取失败，正在切换备用数据源...")
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
    """计算所有技术指标（纯pandas/numpy实现）"""
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

    # ========== 新增指标（纯pandas实现） ==========

    # KDJ指标
    df = calc_kdj(df)

    # ADX指标（平均趋向指数）
    df = calc_adx(df)

    # ATR指标（平均真实波幅）
    df['ATR'] = calc_atr(df)

    # OBV指标（能量潮）
    df['OBV'] = calc_obv(df)
    df['OBV_MA'] = calc_sma(df['OBV'], 20)

    # CCI指标（顺势指标）
    df['CCI'] = calc_cci(df)

    # SuperTrend指标
    df = calc_supertrend(df)

    # PSAR指标（抛物线转向）
    df = calc_psar(df)

    # Ichimoku云图（一目均衡表）
    df = calc_ichimoku(df)

    return df


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ指标计算"""
    low_n = df['low'].rolling(window=n).min()
    high_n = df['high'].rolling(window=n).max()

    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    # K值 = RSV的M1日移动平均
    df['KDJ_K'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    # D值 = K值的M2日移动平均
    df['KDJ_D'] = df['KDJ_K'].ewm(alpha=1/m2, adjust=False).mean()
    # J值 = 3*K - 2*D
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']

    return df


def calc_adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """ADX平均趋向指数计算"""
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM 和 -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # TR (True Range)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 平滑
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean() / atr

    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
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
    """OBV能量潮计算"""
    close = df['close']
    volume = df['volume']

    obv = pd.Series(0.0, index=df.index)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv


def calc_cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """CCI顺势指标计算"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window=n).mean()
    md = tp.rolling(window=n).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (tp - ma) / (0.015 * md)
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


def print_analysis(df: pd.DataFrame, code: str, period: str):
    """打印分析结果"""
    # 根据周期类型设置最小数据点要求
    min_data_points = 30
    if len(df) < min_data_points:
        if period == 'w':
            suggested_days = min_data_points * 7 + 30  # 周线需要更多日历天数
            print(f"错误：数据不足，当前仅获取 {len(df)} 条周线数据，至少需要 {min_data_points} 条")
            print(f"建议：使用 -d {suggested_days} 或更大的天数参数")
        elif period == 'm':
            suggested_days = min_data_points * 30 + 60  # 月线需要更多日历天数
            print(f"错误：数据不足，当前仅获取 {len(df)} 条月线数据，至少需要 {min_data_points} 条")
            print(f"建议：使用 -d {suggested_days} 或更大的天数参数")
        else:
            print("错误：数据不足，请增加天数参数（至少需要30天）")
        return

    # 计算指标
    df = calculate_indicators(df)
    last = df.iloc[-1]

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
    print(f"\n{'='*60}")
    print(f"  股票代码: {code} | 周期: {period_name}")
    print(f"{'='*60}")

    # 最新数据
    print(f"\n【最新数据】")
    print(f"  日期: {last['date'].strftime('%Y-%m-%d')}")
    print(f"  收盘: {last['close']:.2f}")
    print(f"  成交量: {format_volume(last['volume'])}")
    if pd.notna(last['ATR']):
        print(f"  ATR(14): {last['ATR']:.2f}")

    # 基础技术指标
    print(f"\n【基础指标】")
    print(f"  MA5:  {last['MA5']:.2f}" if pd.notna(last['MA5']) else "  MA5:  N/A")
    print(f"  MA10: {last['MA10']:.2f}" if pd.notna(last['MA10']) else "  MA10: N/A")
    print(f"  MA20: {last['MA20']:.2f}" if pd.notna(last['MA20']) else "  MA20: N/A")
    print(f"  RSI:  {last['RSI']:.2f}" if pd.notna(last['RSI']) else "  RSI:  N/A")
    print(f"  MACD: {last['MACD']:.4f}" if pd.notna(last['MACD']) else "  MACD: N/A")
    print(f"  布林上轨: {last['BB_upper']:.2f}" if pd.notna(last['BB_upper']) else "  布林上轨: N/A")
    print(f"  布林下轨: {last['BB_lower']:.2f}" if pd.notna(last['BB_lower']) else "  布林下轨: N/A")

    # 新增技术指标
    print(f"\n【新增指标】")
    # KDJ
    if pd.notna(last['KDJ_K']):
        print(f"  KDJ: K={last['KDJ_K']:.2f}, D={last['KDJ_D']:.2f}, J={last['KDJ_J']:.2f}")
    # ADX
    if pd.notna(last['ADX']):
        print(f"  ADX: {last['ADX']:.2f} (+DI:{last['ADX_PDI']:.2f}, -DI:{last['ADX_NDI']:.2f})")
    # CCI
    if pd.notna(last['CCI']):
        print(f"  CCI: {last['CCI']:.2f}")
    # SuperTrend
    if pd.notna(last['SuperTrend']):
        direction = "多头" if last['SuperTrend_dir'] == 1 else "空头"
        print(f"  SuperTrend: {last['SuperTrend']:.2f} ({direction})")
    # PSAR
    if pd.notna(last['PSAR']):
        direction = "多头" if last['PSAR_dir'] == 1 else "空头"
        print(f"  PSAR: {last['PSAR']:.2f} ({direction})")
    # OBV
    if pd.notna(last['OBV']):
        obv_trend = "↑" if last['OBV'] > last['OBV_MA'] else "↓"
        print(f"  OBV: {format_volume(last['OBV'])} {obv_trend}")
    # Ichimoku
    if pd.notna(last['ICH_TENKAN']):
        print(f"  Ichimoku: 转换线={last['ICH_TENKAN']:.2f}, 基准线={last['ICH_KIJUN']:.2f}")

    # 趋势分析
    print(f"\n【趋势分析】")
    print(f"  当前趋势: {trend}")

    # 各指标信号详情
    print(f"\n【各指标信号】")
    signal_icons = {'buy': '▲', 'sell': '▼', 'neutral': '○'}
    for name, info in indicators.items():
        icon = signal_icons.get(info['signal'], '○')
        print(f"  {icon} {name}: {info['status']}")

    # 多指标共振分析
    print(f"\n【多指标共振】")
    res_icon = {'strong_buy': '▲▲', 'buy': '▲', 'neutral': '○',
                'sell': '▼', 'strong_sell': '▼▼'}
    icon = res_icon.get(resonance['resonance'], '○')
    print(f"  共振信号: {icon} {resonance['signal_text']}")
    print(f"  买入指标({len(resonance['buy_indicators'])}): {', '.join(resonance['buy_indicators']) if resonance['buy_indicators'] else '无'}")
    print(f"  卖出指标({len(resonance['sell_indicators'])}): {', '.join(resonance['sell_indicators']) if resonance['sell_indicators'] else '无'}")
    print(f"  中性指标({len(resonance['neutral_indicators'])}): {', '.join(resonance['neutral_indicators']) if resonance['neutral_indicators'] else '无'}")

    # 综合胜率提示
    print(f"\n【综合胜率提示】")
    print(f"  预估胜率: {win_rate_info['win_rate']:.1f}%")
    print(f"  趋势强度: {win_rate_info['trend_strength']*100:.0f}%")
    if win_rate_info['risk_warnings']:
        print(f"  风险提示: {'; '.join(win_rate_info['risk_warnings'])}")
    print(f"  {win_rate_info['suggestion']}")

    # 买卖信号汇总
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
    print(f"{'='*60}\n")


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