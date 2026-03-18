"""
行业相对强弱与财报波动模块 — 行业RS线、申万行业分类、财报跳空统计

所有函数遵循优雅降级原则：网络/数据异常时返回默认值，不中断主流程。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 美股/港股 → 行业 ETF 静态映射
SECTOR_ETF_MAP = {
    # 半导体
    'NVDA': 'SMH', 'AMD': 'SMH', 'INTC': 'SMH', 'TSM': 'SMH',
    'AVGO': 'SMH', 'QCOM': 'SMH', 'MU': 'SMH', 'MRVL': 'SMH',
    # 科技
    'AAPL': 'XLK', 'MSFT': 'XLK', 'CRM': 'XLK', 'ORCL': 'XLK',
    # 通信
    'GOOG': 'XLC', 'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC',
    # 消费
    'TSLA': 'XLY', 'AMZN': 'XLY', 'NIO': 'XLY', 'BABA': 'XLY',
    'HD': 'XLY', 'MCD': 'XLY',
    # 金融
    'JPM': 'XLF', 'GS': 'XLF', 'BAC': 'XLF', 'MS': 'XLF',
    # 能源
    'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE',
    # 医疗
    'LLY': 'XLV', 'JNJ': 'XLV', 'PFE': 'XLV', 'UNH': 'XLV',
    # 港股 → 中概科技
    '0700.HK': 'KWEB', '9988.HK': 'KWEB', '9618.HK': 'KWEB',
    '1810.HK': 'KWEB', '3690.HK': 'KWEB', '9999.HK': 'KWEB',
    '2318.HK': 'EWH', '0005.HK': 'EWH', '1299.HK': 'EWH',
}

# 大盘 fallback
MARKET_FALLBACK = {
    'us': 'SPY',
    'hk': '^HSI',
    'ashare': 'sh000300',
}


def _is_ashare(code: str) -> bool:
    return code.endswith('.SS') or code.endswith('.SZ')


def _fetch_yahoo_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """通过 yfinance 获取历史数据。"""
    try:
        import yfinance as yf
        import io, contextlib
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f'{days}d')
        if hist is None or hist.empty:
            return None
        hist = hist.reset_index()
        hist.columns = [c.lower() for c in hist.columns]
        return hist
    except Exception:
        return None


def get_ashare_sector_index(code: str) -> dict:
    """通过 akshare 获取 A 股所属申万行业分类，返回对应行业指数数据。

    流程：
    1. ak.stock_individual_info_em() 获取行业信息
    2. 匹配申万行业指数
    3. ak.index_zh_a_hist() 获取行业指数历史数据

    降级：akshare 失败 → fallback 到沪深300

    Returns:
        {
            'sector_name': str,
            'sector_index_code': str,
            'sector_data': DataFrame or None,
            'is_fallback': bool,
            'available': bool
        }
    """
    default = {
        'sector_name': '沪深300', 'sector_index_code': 'sh000300',
        'sector_data': None, 'is_fallback': True, 'available': False
    }

    try:
        import akshare as ak

        # 提取纯数字代码
        raw_code = code.replace('.SS', '').replace('.SZ', '')

        # 获取个股信息（含行业分类）
        info_df = ak.stock_individual_info_em(symbol=raw_code)
        if info_df is None or info_df.empty:
            return default

        # 查找行业字段
        sector_name = None
        for _, row in info_df.iterrows():
            item = str(row.get('item', ''))
            if '行业' in item:
                sector_name = str(row.get('value', ''))
                break

        if not sector_name:
            return default

        # 获取申万行业指数列表
        try:
            sw_index = ak.index_stock_info()
            if sw_index is not None and not sw_index.empty:
                # 模糊匹配行业名
                match = sw_index[sw_index['index_name'].str.contains(sector_name[:2], na=False)]
                if not match.empty:
                    idx_code = str(match.iloc[0]['index_code'])
                    # 获取行业指数历史数据
                    end_date = datetime.now().strftime('%Y%m%d')
                    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')
                    idx_data = ak.index_zh_a_hist(symbol=idx_code, period='daily',
                                                  start_date=start_date, end_date=end_date)
                    if idx_data is not None and not idx_data.empty:
                        idx_data.columns = [c.lower().replace('日期', 'date').replace('收盘', 'close') for c in idx_data.columns]
                        return {
                            'sector_name': sector_name,
                            'sector_index_code': idx_code,
                            'sector_data': idx_data,
                            'is_fallback': False,
                            'available': True
                        }
        except Exception:
            pass

        # 降级：使用沪深300
        return {**default, 'sector_name': sector_name + '（行业指数获取失败）'}

    except Exception:
        return default


def calc_sector_relative_strength(code: str, df: pd.DataFrame) -> dict:
    """计算个股相对行业 ETF 的 RS 线。

    RS = 个股价格 / 行业ETF价格（归一化到起始点=100）

    Returns:
        {
            'available': bool,
            'sector_etf': str,
            'sector_name': str,
            'is_fallback': bool,
            'rs_1y_change': float,       # RS线1年变化 (%)
            'rs_new_low': bool,          # RS线是否创1年新低
            'rs_divergence': bool,       # 个股涨但RS创新低
            'warning_text': str,
            'source': str
        }
    """
    default = {
        'available': False, 'sector_etf': '', 'sector_name': '',
        'is_fallback': False, 'rs_1y_change': 0.0,
        'rs_new_low': False, 'rs_divergence': False,
        'warning_text': '', 'source': '数据不可用'
    }

    try:
        if df is None or len(df) < 60:
            return default

        is_ashare = _is_ashare(code)
        sector_name = ''
        is_fallback = False
        etf_data = None

        if is_ashare:
            # A 股：akshare 申万行业分类
            sector_info = get_ashare_sector_index(code)
            sector_name = sector_info.get('sector_name', '')
            is_fallback = sector_info.get('is_fallback', True)

            if sector_info.get('available') and sector_info.get('sector_data') is not None:
                etf_data = sector_info['sector_data']
                sector_etf = sector_info['sector_index_code']
            else:
                # fallback 到沪深300
                is_fallback = True
                sector_etf = '000300'
                try:
                    import akshare as ak
                    end_date = datetime.now().strftime('%Y%m%d')
                    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')
                    etf_data = ak.index_zh_a_hist(symbol=sector_etf, period='daily',
                                                  start_date=start_date, end_date=end_date)
                    if etf_data is not None and not etf_data.empty:
                        etf_data.columns = [c.lower().replace('日期', 'date').replace('收盘', 'close') for c in etf_data.columns]
                except Exception:
                    etf_data = None
        else:
            # 美股/港股：静态映射
            clean_code = code.upper().replace('.HK', '')
            if code.endswith('.HK'):
                clean_code = code
            sector_etf = SECTOR_ETF_MAP.get(clean_code, SECTOR_ETF_MAP.get(code, ''))

            if not sector_etf:
                sector_etf = 'SPY'
                is_fallback = True

            etf_data = _fetch_yahoo_data(sector_etf, days=400)
            sector_name = sector_etf

        if etf_data is None or len(etf_data) < 30:
            if is_fallback and is_ashare:
                return {**default, 'warning_text': '行业匹配失败，已降级为大盘宏观相对强弱对比',
                        'is_fallback': True, 'sector_name': sector_name}
            return default

        # 对齐数据长度
        stock_close = df['close'].values
        etf_close_col = 'close' if 'close' in etf_data.columns else etf_data.columns[-1]
        etf_close = etf_data[etf_close_col].values

        min_len = min(len(stock_close), len(etf_close), 250)
        stock_close = stock_close[-min_len:]
        etf_close = etf_close[-min_len:]

        # 归一化 RS 线
        stock_norm = stock_close / stock_close[0] * 100
        etf_norm = etf_close / etf_close[0] * 100
        rs_line = stock_norm / etf_norm * 100

        rs_1y_change = (rs_line[-1] / rs_line[0] - 1) * 100
        rs_current = rs_line[-1]
        rs_min = np.min(rs_line)
        rs_new_low = rs_current <= rs_min * 1.01  # 接近或创新低

        # 个股涨但 RS 创新低 = 背离
        stock_up = stock_close[-1] > stock_close[0]
        rs_divergence = stock_up and rs_new_low

        warning = ''
        if rs_divergence:
            warning = f'跑输同业：资金正在流向竞争对手，非行业龙头标的，降低预期'
        elif rs_new_low:
            warning = f'RS线创新低，相对行业表现弱势'
        elif rs_1y_change < -10:
            warning = f'过去1年相对行业跑输 {abs(rs_1y_change):.1f}%'

        fallback_note = ''
        if is_fallback and is_ashare:
            fallback_note = '（行业匹配失败，已降级为大盘宏观相对强弱对比）'

        return {
            'available': True,
            'sector_etf': sector_etf,
            'sector_name': sector_name + fallback_note,
            'is_fallback': is_fallback,
            'rs_1y_change': round(rs_1y_change, 2),
            'rs_new_low': rs_new_low,
            'rs_divergence': rs_divergence,
            'warning_text': warning,
            'source': 'akshare 申万行业' if is_ashare else f'yfinance ({sector_etf})'
        }

    except Exception:
        return default


def fetch_earnings_volatility(code: str) -> dict:
    """计算过去 2 年（最近 8 次财报）发布后的跳空幅度和最大日内回撤。

    数据源: yfinance Ticker.earnings_dates + 日线数据

    Returns:
        {
            'available': bool,
            'earnings_count': int,
            'avg_gap': float,            # 平均跳空幅度 (%)
            'max_gap': float,            # 最大跳空幅度 (%)
            'avg_drawdown': float,       # 平均日内回撤 (%)
            'max_drawdown': float,       # 最大日内回撤 (%)
            'summary_text': str,
            'source': str
        }
    """
    default = {
        'available': False, 'earnings_count': 0,
        'avg_gap': 0.0, 'max_gap': 0.0,
        'avg_drawdown': 0.0, 'max_drawdown': 0.0,
        'summary_text': '财报波动数据不可用', 'source': '数据不可用'
    }

    try:
        import yfinance as yf
        import io, contextlib

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            ticker = yf.Ticker(code)
            earnings_dates = ticker.earnings_dates

        if earnings_dates is None or earnings_dates.empty:
            return default

        # 过滤过去 2 年的财报日期
        cutoff = datetime.now() - timedelta(days=730)
        earnings_dates.index = pd.to_datetime(earnings_dates.index, errors='coerce')
        past_earnings = earnings_dates[earnings_dates.index < datetime.now()]
        past_earnings = past_earnings[past_earnings.index >= cutoff]

        if len(past_earnings) == 0:
            return default

        # 获取日线数据
        hist = ticker.history(period='2y')
        if hist is None or hist.empty:
            return default

        hist = hist.reset_index()
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)

        gaps = []
        drawdowns = []

        for earn_date in past_earnings.index[:8]:
            earn_date = earn_date.tz_localize(None) if earn_date.tzinfo else earn_date
            # 找到财报后的第一个交易日
            post_days = hist[hist['Date'] > earn_date].head(2)
            pre_days = hist[hist['Date'] <= earn_date].tail(1)

            if len(pre_days) == 0 or len(post_days) == 0:
                continue

            pre_close = float(pre_days.iloc[0]['Close'])
            post_open = float(post_days.iloc[0]['Open'])
            post_high = float(post_days.iloc[0]['High'])
            post_low = float(post_days.iloc[0]['Low'])

            if pre_close == 0:
                continue

            # 跳空幅度
            gap = (post_open - pre_close) / pre_close * 100
            gaps.append(gap)

            # 日内最大回撤（从开盘到最低）
            intraday_dd = (post_low - post_open) / post_open * 100 if post_open > 0 else 0
            drawdowns.append(intraday_dd)

        if not gaps:
            return default

        avg_gap = np.mean([abs(g) for g in gaps])
        max_gap = max([abs(g) for g in gaps])
        avg_dd = np.mean([abs(d) for d in drawdowns])
        max_dd = max([abs(d) for d in drawdowns])

        summary = f"历史财报波动率：平均跳空 {avg_gap:.1f}%，最大跳空 {max_gap:.1f}%，最大单日回撤曾达 {max_dd:.1f}%，做好预期管理"

        return {
            'available': True,
            'earnings_count': len(gaps),
            'avg_gap': round(avg_gap, 2),
            'max_gap': round(max_gap, 2),
            'avg_drawdown': round(avg_dd, 2),
            'max_drawdown': round(max_dd, 2),
            'summary_text': summary,
            'source': 'yfinance 财报日历'
        }

    except Exception:
        return default


def check_structural_bear_market() -> dict:
    """检查大盘指数长线趋势（200日均线方向），判断是否处于结构性熊市。

    检查: SPY, QQQ, 沪深300

    Returns:
        {
            'available': bool,
            'indices': dict,  # {name: {price, ma200, above_ma200, ma200_slope}}
            'structural_bear': bool,
            'bear_count': int,
            'warning_text': str,
            'source': str
        }
    """
    default = {
        'available': False, 'indices': {},
        'structural_bear': False, 'bear_count': 0,
        'warning_text': '', 'source': '数据不可用'
    }

    try:
        indices_to_check = [
            ('SPY', 'S&P 500'),
            ('QQQ', 'NASDAQ 100'),
        ]

        results = {}
        bear_count = 0

        for symbol, name in indices_to_check:
            data = _fetch_yahoo_data(symbol, days=300)
            if data is None or len(data) < 200:
                continue

            close = data['close'].values
            ma200 = np.mean(close[-200:])
            current = close[-1]
            above = current > ma200

            # MA200 斜率（最近 20 天 vs 之前 20 天）
            ma200_recent = np.mean(close[-20:])
            ma200_prev = np.mean(close[-40:-20])
            slope = (ma200_recent - ma200_prev) / ma200_prev * 100 if ma200_prev > 0 else 0

            results[name] = {
                'price': round(current, 2),
                'ma200': round(ma200, 2),
                'above_ma200': above,
                'ma200_slope': round(slope, 2)
            }

            if not above and slope < 0:
                bear_count += 1

        structural_bear = bear_count >= 1
        warning = ''
        if structural_bear:
            bear_names = [n for n, d in results.items() if not d['above_ma200'] and d['ma200_slope'] < 0]
            warning = f"宏观结构性熊市：{', '.join(bear_names)} 跌破200日均线且斜率向下，建议增加现金储备"

        return {
            'available': len(results) > 0,
            'indices': results,
            'structural_bear': structural_bear,
            'bear_count': bear_count,
            'warning_text': warning,
            'source': 'yfinance 大盘指数'
        }

    except Exception:
        return default
