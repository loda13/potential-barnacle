"""
基本面扫雷模块 — 稀释检测、质量检测（ROE/FCF）、退市风险、价值陷阱否决

所有函数遵循优雅降级原则：网络/数据异常时返回默认值，不中断主流程。
"""

import pandas as pd
import numpy as np


def _get_ticker(code: str):
    """懒加载 yfinance Ticker 对象。"""
    try:
        import yfinance as yf
        return yf.Ticker(code)
    except Exception:
        return None


def fetch_dilution_analysis(code: str) -> dict:
    """检测过去 5 年流通股本变化趋势（股东权益稀释）。

    数据源: yfinance Ticker.quarterly_balance_sheet → 'Ordinary Shares Number'

    Returns:
        {
            'available': bool,
            'shares_trend': list[dict],  # [{year, shares}]
            'dilution_rate': float,      # 年化稀释率 (%)
            'severe_dilution': bool,     # 连续3年以上净增发>5%/年
            'warning_text': str,
            'source': str
        }
    """
    default = {
        'available': False, 'shares_trend': [], 'dilution_rate': 0.0,
        'severe_dilution': False, 'warning_text': '', 'source': '数据不可用'
    }

    try:
        ticker = _get_ticker(code)
        if ticker is None:
            return default

        bs = ticker.quarterly_balance_sheet
        if bs is None or bs.empty:
            return default

        # 尝试多种字段名
        shares_row = None
        for key in ['Ordinary Shares Number', 'Share Issued', 'Common Stock']:
            if key in bs.index:
                shares_row = bs.loc[key]
                break

        if shares_row is None:
            return default

        # 按年聚合（取每年最新季度）
        shares_data = shares_row.dropna().sort_index()
        if len(shares_data) < 4:
            return default

        yearly = {}
        for date, val in shares_data.items():
            year = date.year
            yearly[year] = float(val)

        years = sorted(yearly.keys())
        if len(years) < 2:
            return default

        shares_trend = [{'year': y, 'shares': yearly[y]} for y in years]

        # 计算年化稀释率
        yoy_changes = []
        consecutive_dilution = 0
        max_consecutive = 0
        for i in range(1, len(years)):
            prev = yearly[years[i - 1]]
            curr = yearly[years[i]]
            if prev > 0:
                change = (curr - prev) / prev * 100
                yoy_changes.append(change)
                if change > 5:
                    consecutive_dilution += 1
                    max_consecutive = max(max_consecutive, consecutive_dilution)
                else:
                    consecutive_dilution = 0

        avg_dilution = np.mean(yoy_changes) if yoy_changes else 0.0
        severe = max_consecutive >= 3

        warning = ''
        if severe:
            warning = '警告：检测到持续的股东权益稀释，长线持有风险极高'
        elif avg_dilution > 3:
            warning = '注意：流通股本呈扩张趋势，关注稀释影响'

        return {
            'available': True,
            'shares_trend': shares_trend,
            'dilution_rate': round(avg_dilution, 2),
            'severe_dilution': severe,
            'warning_text': warning,
            'source': 'yfinance 季度资产负债表'
        }

    except Exception:
        return default


def fetch_quality_metrics(code: str) -> dict:
    """获取 3 年 ROE 和 FCF 利润率趋势。

    数据源: yfinance Ticker.quarterly_financials / .quarterly_balance_sheet / .quarterly_cashflow

    Returns:
        {
            'available': bool,
            'roe_trend': list[dict],       # [{year, roe}]
            'fcf_trend': list[dict],       # [{year, fcf_margin}]
            'latest_roe': float,
            'latest_fcf_margin': float,
            'quality_deteriorating': bool,  # ROE连续下降或FCF为负
            'quality_signal': str,          # 'healthy' / 'warning' / 'deteriorating'
            'warning_text': str,
            'source': str
        }
    """
    default = {
        'available': False, 'roe_trend': [], 'fcf_trend': [],
        'latest_roe': None, 'latest_fcf_margin': None,
        'quality_deteriorating': False, 'quality_signal': 'unknown',
        'warning_text': '', 'source': '数据不可用'
    }

    try:
        ticker = _get_ticker(code)
        if ticker is None:
            return default

        # 获取财务数据
        financials = ticker.quarterly_financials
        balance = ticker.quarterly_balance_sheet
        cashflow = ticker.quarterly_cashflow

        if financials is None or balance is None:
            return default

        # --- ROE 计算 ---
        roe_yearly = {}
        net_income_row = None
        equity_row = None

        for key in ['Net Income', 'Net Income Common Stockholders']:
            if key in financials.index:
                net_income_row = financials.loc[key]
                break

        for key in ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']:
            if key in balance.index:
                equity_row = balance.loc[key]
                break

        if net_income_row is not None and equity_row is not None:
            common_dates = net_income_row.dropna().index.intersection(equity_row.dropna().index)
            for date in common_dates:
                ni = float(net_income_row[date])
                eq = float(equity_row[date])
                if eq != 0:
                    year = date.year
                    if year not in roe_yearly:
                        roe_yearly[year] = []
                    roe_yearly[year].append(ni / eq * 100)

        # 年化 ROE（取各季度平均）
        roe_trend = []
        for year in sorted(roe_yearly.keys()):
            avg_roe = np.mean(roe_yearly[year])
            roe_trend.append({'year': year, 'roe': round(avg_roe, 2)})

        # --- FCF 利润率计算 ---
        fcf_yearly = {}
        if cashflow is not None and not cashflow.empty:
            fcf_row = None
            revenue_row = None

            for key in ['Free Cash Flow']:
                if key in cashflow.index:
                    fcf_row = cashflow.loc[key]
                    break

            for key in ['Total Revenue', 'Operating Revenue']:
                if key in financials.index:
                    revenue_row = financials.loc[key]
                    break

            if fcf_row is not None and revenue_row is not None:
                common_dates = fcf_row.dropna().index.intersection(revenue_row.dropna().index)
                for date in common_dates:
                    fcf = float(fcf_row[date])
                    rev = float(revenue_row[date])
                    if rev != 0:
                        year = date.year
                        if year not in fcf_yearly:
                            fcf_yearly[year] = []
                        fcf_yearly[year].append(fcf / rev * 100)

        fcf_trend = []
        for year in sorted(fcf_yearly.keys()):
            avg_margin = np.mean(fcf_yearly[year])
            fcf_trend.append({'year': year, 'fcf_margin': round(avg_margin, 2)})

        if not roe_trend and not fcf_trend:
            return default

        # 判断质量趋势
        latest_roe = roe_trend[-1]['roe'] if roe_trend else None
        latest_fcf = fcf_trend[-1]['fcf_margin'] if fcf_trend else None

        roe_declining = False
        if len(roe_trend) >= 3:
            recent_3 = [r['roe'] for r in roe_trend[-3:]]
            roe_declining = recent_3[-1] < recent_3[0] and recent_3[-1] < recent_3[-2]

        fcf_negative = latest_fcf is not None and latest_fcf < 0

        quality_deteriorating = roe_declining or fcf_negative
        if quality_deteriorating:
            quality_signal = 'deteriorating'
            warning = '盈利能力恶化'
            if roe_declining:
                warning += f'（ROE 连续下降至 {latest_roe:.1f}%）'
            if fcf_negative:
                warning += f'（FCF 利润率为负 {latest_fcf:.1f}%）'
        elif latest_roe is not None and latest_roe < 5:
            quality_signal = 'warning'
            warning = f'ROE 偏低（{latest_roe:.1f}%），盈利能力一般'
        else:
            quality_signal = 'healthy'
            warning = ''

        return {
            'available': True,
            'roe_trend': roe_trend,
            'fcf_trend': fcf_trend,
            'latest_roe': latest_roe,
            'latest_fcf_margin': latest_fcf,
            'quality_deteriorating': quality_deteriorating,
            'quality_signal': quality_signal,
            'warning_text': warning,
            'source': 'yfinance 季度财务报表'
        }

    except Exception:
        return default


def check_delisting_risk(code: str, df: pd.DataFrame) -> dict:
    """检查退市风险。

    - 美股：最近 30 个交易日收盘价持续 < $1
    - A 股：连续 20 日 < ¥1（面值退市）或 ST/*ST 标记

    Returns:
        {
            'risk_level': 'none' / 'warning' / 'critical',
            'warning_text': str,
            'details': list[str]
        }
    """
    default = {'risk_level': 'none', 'warning_text': '', 'details': []}

    try:
        if df is None or len(df) < 20:
            return default

        details = []
        risk_level = 'none'
        close_prices = df['close'].tail(30)

        # 判断市场类型
        is_ashare = code.endswith('.SS') or code.endswith('.SZ')
        is_us = not is_ashare and not code.endswith('.HK')

        if is_us:
            # 美股：连续 30 日 < $1
            days_below_1 = (close_prices < 1.0).sum()
            if days_below_1 >= 30:
                risk_level = 'critical'
                details.append(f'连续 {days_below_1} 个交易日收盘价低于 $1，面临退市风险')
            elif days_below_1 >= 15:
                risk_level = 'warning'
                details.append(f'近 30 日有 {days_below_1} 天收盘价低于 $1，需关注退市风险')

        if is_ashare:
            # A 股：连续 20 日 < ¥1
            days_below_1 = (close_prices.tail(20) < 1.0).sum()
            if days_below_1 >= 20:
                risk_level = 'critical'
                details.append(f'连续 20 个交易日收盘价低于 ¥1，触发面值退市条件')
            elif days_below_1 >= 10:
                risk_level = 'warning'
                details.append(f'近 20 日有 {days_below_1} 天收盘价低于 ¥1，面值退市风险')

            # ST/*ST 检查
            raw_code = code.replace('.SS', '').replace('.SZ', '')
            try:
                import akshare as ak
                info = ak.stock_individual_info_em(symbol=raw_code)
                if info is not None:
                    name_row = info[info['item'] == '股票简称']
                    if not name_row.empty:
                        name = str(name_row.iloc[0]['value'])
                        if 'ST' in name.upper():
                            risk_level = 'critical' if risk_level != 'critical' else risk_level
                            details.append(f'股票名称含 ST 标记（{name}），存在退市风险')
            except Exception:
                pass

        warning = ''
        if risk_level == 'critical':
            warning = '退市风险极高，强烈建议回避'
        elif risk_level == 'warning':
            warning = '存在退市风险信号，需密切关注'

        return {
            'risk_level': risk_level,
            'warning_text': warning,
            'details': details
        }

    except Exception:
        return default


def detect_value_trap(valuation: dict, quality: dict) -> dict:
    """价值陷阱检测：低估值 + 恶化基本面 = 价值陷阱。

    Args:
        valuation: fetch_valuation_metrics() 的返回值
        quality: fetch_quality_metrics() 的返回值

    Returns:
        {
            'is_trap': bool,
            'veto_buy': bool,
            'warning_text': str,
            'reasons': list[str]
        }
    """
    default = {'is_trap': False, 'veto_buy': False, 'warning_text': '', 'reasons': []}

    try:
        if not valuation.get('available') or not quality.get('available'):
            return default

        pe_pct = valuation.get('pe_percentile')
        ps_pct = valuation.get('ps_percentile')
        quality_bad = quality.get('quality_deteriorating', False)

        # 低估值判定：PE 或 PS 百分位 < 30%
        low_valuation = False
        reasons = []

        if pe_pct is not None and pe_pct < 30:
            low_valuation = True
            reasons.append(f'PE 百分位 {pe_pct:.0f}%（看似低估）')
        if ps_pct is not None and ps_pct < 30:
            low_valuation = True
            reasons.append(f'PS 百分位 {ps_pct:.0f}%（看似低估）')

        if low_valuation and quality_bad:
            reasons.append(quality.get('warning_text', '盈利能力恶化'))
            return {
                'is_trap': True,
                'veto_buy': True,
                'warning_text': '价值陷阱警告：盈利能力恶化，否决买入',
                'reasons': reasons
            }

        return default

    except Exception:
        return default
