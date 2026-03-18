"""
聪明钱动向模块 — 内部人士交易、机构持股趋势、聪明钱确认逻辑

所有函数遵循优雅降级原则：网络/数据异常时返回默认值，不中断主流程。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _get_ticker(code: str):
    """懒加载 yfinance Ticker 对象。"""
    try:
        import yfinance as yf
        return yf.Ticker(code)
    except Exception:
        return None


def fetch_insider_transactions(code: str) -> dict:
    """获取最近 6 个月内部人士交易数据。

    数据源: yfinance Ticker.insider_transactions

    Returns:
        {
            'available': bool,
            'transactions': list[dict],
            'net_activity': str,           # 'net_buying' / 'net_selling' / 'mixed'
            'total_buys': int,
            'total_sells': int,
            'large_sells': list[dict],     # 大额卖出 (> $1M)
            'insider_selling_alert': bool,
            'alert_text': str,
            'source': str
        }
    """
    default = {
        'available': False, 'transactions': [],
        'net_activity': 'unknown', 'total_buys': 0, 'total_sells': 0,
        'large_sells': [], 'insider_selling_alert': False,
        'alert_text': '', 'source': '数据不可用'
    }

    try:
        ticker = _get_ticker(code)
        if ticker is None:
            return default

        insider_df = ticker.insider_transactions
        if insider_df is None or insider_df.empty:
            return default

        # 过滤最近 6 个月
        cutoff = datetime.now() - timedelta(days=180)
        if 'Start Date' in insider_df.columns:
            insider_df['Start Date'] = pd.to_datetime(insider_df['Start Date'], errors='coerce')
            insider_df = insider_df[insider_df['Start Date'] >= cutoff]

        if insider_df.empty:
            return {**default, 'available': True, 'source': 'yfinance (近6月无交易)'}

        transactions = []
        total_buys = 0
        total_sells = 0
        large_sells = []
        executive_sells = 0

        executive_titles = ['CEO', 'CFO', 'COO', 'CTO', 'President', 'Founder',
                            'Chief', 'Director', 'Officer', 'VP']

        for _, row in insider_df.iterrows():
            text = str(row.get('Text', ''))
            insider_name = str(row.get('Insider', ''))
            shares = row.get('Shares', 0)
            value = row.get('Value', 0)
            title = str(row.get('Position', ''))

            is_sale = 'Sale' in text or 'Sell' in text
            is_purchase = 'Purchase' in text or 'Buy' in text

            txn = {
                'name': insider_name,
                'title': title,
                'type': 'sell' if is_sale else ('buy' if is_purchase else 'other'),
                'shares': int(shares) if pd.notna(shares) else 0,
                'value': float(value) if pd.notna(value) else 0,
                'date': str(row.get('Start Date', ''))[:10]
            }
            transactions.append(txn)

            if is_sale:
                total_sells += 1
                if pd.notna(value) and abs(float(value)) > 1_000_000:
                    large_sells.append(txn)
                if any(t.lower() in title.lower() for t in executive_titles):
                    executive_sells += 1
            elif is_purchase:
                total_buys += 1

        # 判断净活动
        if total_sells > total_buys * 2:
            net_activity = 'net_selling'
        elif total_buys > total_sells * 2:
            net_activity = 'net_buying'
        else:
            net_activity = 'mixed'

        # 内部人士派发预警
        insider_alert = executive_sells >= 3 or len(large_sells) >= 3
        alert_text = ''
        if insider_alert:
            alert_text = '内部人士派发预警：核心高管大规模净抛售'
        elif net_activity == 'net_selling':
            alert_text = '内部人士净卖出，关注动向'

        return {
            'available': True,
            'transactions': transactions[:20],
            'net_activity': net_activity,
            'total_buys': total_buys,
            'total_sells': total_sells,
            'large_sells': large_sells,
            'insider_selling_alert': insider_alert,
            'alert_text': alert_text,
            'source': 'yfinance 内部人士交易'
        }

    except Exception:
        return default


def fetch_institutional_holdings(code: str) -> dict:
    """获取机构持股数据。

    数据源: yfinance Ticker.institutional_holders

    Returns:
        {
            'available': bool,
            'top_holders': list[dict],
            'total_holders': int,
            'trend': str,  # 'accumulating' / 'distributing' / 'stable'
            'source': str
        }
    """
    default = {
        'available': False, 'top_holders': [], 'total_holders': 0,
        'trend': 'unknown', 'source': '数据不可用'
    }

    try:
        ticker = _get_ticker(code)
        if ticker is None:
            return default

        holders_df = ticker.institutional_holders
        if holders_df is None or holders_df.empty:
            return default

        top_holders = []
        for _, row in holders_df.head(10).iterrows():
            holder = {
                'name': str(row.get('Holder', '')),
                'shares': int(row.get('Shares', 0)) if pd.notna(row.get('Shares')) else 0,
                'pct_out': float(row.get('% Out', 0)) if pd.notna(row.get('% Out')) else 0,
                'value': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else 0,
            }
            top_holders.append(holder)

        # 简单趋势判断（基于 Date Reported 变化）
        trend = 'stable'
        if 'Date Reported' in holders_df.columns:
            recent_dates = pd.to_datetime(holders_df['Date Reported'], errors='coerce').dropna()
            if len(recent_dates) > 0:
                latest = recent_dates.max()
                if (datetime.now() - latest).days < 90:
                    trend = 'accumulating'  # 近期有报告 = 活跃持仓

        return {
            'available': True,
            'top_holders': top_holders,
            'total_holders': len(holders_df),
            'trend': trend,
            'source': 'yfinance 机构持股'
        }

    except Exception:
        return default


def calc_smart_money_confirmation(insider: dict, institutional: dict,
                                  at_macro_support: bool = False) -> dict:
    """聪明钱综合确认。

    组合逻辑：
    - 股价在宏观支撑位 + 机构建仓 + 内部人士净买入 → "高确定性长线击球区"
    - 内部人士大规模抛售 → "内部人士派发预警"

    Returns:
        {
            'confirmation_level': str,  # 'high' / 'medium' / 'low' / 'warning'
            'signal_text': str,
            'details': list[str]
        }
    """
    details = []
    level = 'low'

    insider_buying = insider.get('net_activity') == 'net_buying'
    insider_selling = insider.get('insider_selling_alert', False)
    inst_accumulating = institutional.get('trend') == 'accumulating'

    if insider_selling:
        level = 'warning'
        details.append('内部人士大规模抛售')

    if at_macro_support and inst_accumulating and insider_buying:
        level = 'high'
        details.append('宏观支撑位 + 机构建仓 + 内部人士净买入')
    elif at_macro_support and (inst_accumulating or insider_buying):
        level = 'medium' if level != 'warning' else level
        details.append('宏观支撑位 + 部分聪明钱信号')
    elif inst_accumulating:
        level = 'medium' if level != 'warning' else level
        details.append('机构持续建仓')

    signal_map = {
        'high': '高确定性长线击球区',
        'medium': '聪明钱信号偏正面',
        'low': '聪明钱信号不明确',
        'warning': '内部人士派发预警，谨慎'
    }

    return {
        'confirmation_level': level,
        'signal_text': signal_map.get(level, '未知'),
        'details': details
    }
