# 股票长线投资分析系统

命令行股票分析工具，支持A股/美股/港股/台股。从短线交易工具升级为**长线投资与基本面体检系统**，集成 SMC 决策树、多周期共振、基本面扫雷、聪明钱追踪、行业相对强弱和多层否决机制。

## 安装

```bash
pip3 install -r requirements.txt
```

## 使用

```bash
# A股日线（隆基绿能，最近120天）
python3 stock_analyzer.py 601012 -d 120

# 美股（英伟达）
python3 stock_analyzer.py NVDA -d 120

# 港股（腾讯控股）
python3 stock_analyzer.py 0700.HK -d 120

# 批量分析多只股票
python3 stock_analyzer.py --portfolio 0700.HK 1810.HK QQQ TSLA NVDA -d 120

# 压力测试（15% 回撤场景）
python3 stock_analyzer.py NVDA -d 120 --stress-test

# 持仓应对策略（指定买入均价）
python3 stock_analyzer.py NVDA -d 120 --entry-price 170.5

# 演示模式（离线，不拉取网络数据）
python3 stock_analyzer.py TSLA --demo
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `code` | 股票代码（A股6位数字，美股如AAPL，港股如0700.HK，台股如2330.TW） | 必填 |
| `-p, --period` | K线周期：`d`=日线，`w`=周线 | `d` |
| `-d, --days` | 分析天数（建议≥120） | `60` |
| `--demo` | 演示模式，使用模拟数据 | 关闭 |
| `--portfolio` | 批量分析多只股票 | 无 |
| `--stress-test` | 执行 15% 回撤压力测试 | 关闭 |
| `--entry-price` | 持仓买入均价（用于持仓应对策略） | 自动估算 |

## 系统架构

### 文件结构

```
stock_analyzer.py      # 主程序（~7900行）
├── fundamentals.py    # 基本面扫雷：稀释检测、ROE/FCF质量、退市风险、价值陷阱否决
├── smart_money.py     # 聪明钱动向：内部人士交易、机构持股、综合确认
├── sector_analysis.py # 行业相对强弱：行业RS线、财报波动统计、宏观熊市检查
└── trading_decision_log.csv  # 执行日志（自动追加）
```

### 否决优先级链（从高到低）

```
1. 黑天鹅熔断 (Circuit Breaker)         ← 最高优先级，覆写一切
   ├─ 大盘环境阻断: 基准指数跌破 MA200 或 VIX 飙升 >15%
   └─ 个股跳空毁灭: 向下跳空 >1.5x ATR 穿越 HVN

2. 长线趋势熔断                          ← 50周均线 < 200周均线
   ├─ 红色致命警告: "长线结构性熊市，禁止买入"
   ├─ 完全跳过所有日线 SMC 分析（切断抄底念想）
   └─ 仅输出: 长线均线偏离度 + 基本面数据

3. 价值陷阱否决                          ← 低估值 + 恶化基本面
   └─ PE低估 + ROE下降/FCF为负 → "价值陷阱警告，否决买入"

4. SMC 严格决策树                        ← 核心决策引擎
   ├─ 买入 (BUY): 4个条件必须全部满足
   │   ├─ ChoCh 确认 + 多时间框架共振
   │   ├─ 回踩确认 (价格在 HVN 或 FVG 区域)
   │   ├─ R-Multiple ≥ 2.0
   │   └─ Z-Score 在 [-2, 2]
   ├─ 卖出 (SELL): 满足任意1个即触发
   └─ 观望 (HOLD): 其他所有情况

5. 波动率平价仓位管理                    ← 精确计算建仓量
```

### 分析模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **长线趋势** | stock_analyzer.py | 50周/200周均线金叉检测、月线趋势确认、长线熔断 |
| **宏观筹码分布** | stock_analyzer.py | 3-5年全量数据计算宏观 POC/VAH/VAL/HVN，识别历史核心支撑/阻力区 |
| **估值水位** | stock_analyzer.py | PE/PS 历史百分位（5年），低估/合理/高估信号 |
| **稀释检测** | fundamentals.py | 5年流通股本变化趋势，持续增发→红色警告 |
| **质量检测** | fundamentals.py | 3年 ROE/FCF 利润率趋势，恶化→价值陷阱否决 |
| **退市风险** | fundamentals.py | 美股 <$1、A股面值退市、ST标记检测 |
| **内部人士交易** | smart_money.py | 近6月高管买卖统计，大规模净抛售→派发预警 |
| **机构持股** | smart_money.py | 前5大机构持仓、建仓/派发趋势 |
| **行业RS线** | sector_analysis.py | 个股 vs 行业ETF相对强弱（A股自动匹配申万行业） |
| **财报波动** | sector_analysis.py | 过去8次财报跳空幅度与日内回撤统计 |
| **宏观熊市** | sector_analysis.py | SPY/QQQ 200日均线方向，结构性熊市检测 |
| **技术指标** | stock_analyzer.py | MA, RSI, MACD, 布林带, ADX, ATR, OBV, CCI, SuperTrend, PSAR, Ichimoku, VWAP |
| **左侧交易** | stock_analyzer.py | Z-Score、量能衰竭、波动率状态、三重背离、机构级支撑压力 |
| **右侧交易** | stock_analyzer.py | BoS、ChoCh、多时间框架共振（日线×周线×月线） |
| **风控** | stock_analyzer.py | Chandelier Exit 止损、黑天鹅熔断、VIX 监控、回撤分析 |
| **持仓应对策略** | stock_analyzer.py | 亏损/盈利分级管理、分批止盈、VWAP偏离预警、加仓禁区检测 |

### 输出区块顺序

**正常模式（长线趋势健康）：**
```
熔断警告 → 大盘复盘 → 技术指标 → 多指标共振 → 交叉信号 → 支撑压力位
→ 筹码分布/资金流 → 新闻舆情 → 机构评级 → 锚定筹码分布 → 期权数据
→ 估值水位 → 自适应周期参数 → 多时间框架共振
→ 长线趋势健康检查（50W/200W均线）
→ 宏观筹码分布（3年历史）
→ 基本面体检（稀释+质量+退市风险）
→ 聪明钱动向（内部人士+机构持股）
→ 行业相对强弱与宏观风控
→ 左侧交易信号 → 右侧确认信号 → VWAP偏离度 → 派发模式
→ 长线风控 → 相对强弱 → 定投建议 → 操作检查清单
→ SMC决策仪表盘 → 持仓应对策略 → 综合结论
```

**长线熊市模式（50W < 200W 触发）：**
```
⚠⚠⚠ 致命警告：长线结构性熊市，禁止买入！⚠⚠⚠
→ 长线均线偏离度 → 宏观筹码分布 → 基本面体检
→ [跳过所有短线分析] → 结束
```

### A股行业映射

A股通过 akshare 自动获取申万行业分类，匹配对应行业指数计算 RS 线。如果 akshare 获取失败，自动降级到沪深300指数并输出黄色警告。

美股/港股使用静态 ETF 映射字典（SMH/XLK/XLY/XLF/KWEB 等），未匹配的股票 fallback 到 SPY/^HSI。

## 全局参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_RISK_PER_TRADE` | 2% | 每笔交易最大风险比例 |
| `DEFAULT_ACCOUNT_SIZE` | 100,000 | 默认账户规模（元） |
| `CHANDELIER_ATR_MULTIPLIER` | 2.5 | Chandelier Exit ATR 倍数 |
| `CHANDELIER_LOOKBACK` | 22 | Chandelier Exit 回溯天数 |
| `SLIPPAGE_PCT` | 0.2% | 滑点模拟 |
| `ADV_POSITION_LIMIT` | 1% | 仓位不超过 ADV 的比例 |
| `CHART_FETCH_BUFFER_DAYS` | 1200 | 长线分析数据窗口（3-5年） |

## 依赖

- `pandas`, `numpy` — 核心计算
- `requests` — Yahoo chart API
- `yfinance` — 期权数据、财务报表、内部人士交易、机构持股（懒加载）
- `akshare` — A股数据回退、筹码分布、资金流、申万行业分类（懒加载）

## 优雅降级

所有扩展数据源（基本面、聪明钱、行业RS、新闻、期权等）遵循优雅降级原则：网络/数据异常时返回默认值，不中断主流程。核心分析仅需 OHLCV 数据。

## 常见问题

**Q: 提示"无法获取股票数据"？**
A: 检查网络连接，或使用 `--demo` 模式测试。

**Q: A股代码如何输入？**
A: 直接输入6位数字，如 `600519`、`000876`

**Q: 港股如何输入？**
A: 使用 yfinance 格式，如 `0700.HK`（腾讯）、`1810.HK`（小米）

**Q: 为什么系统总是输出"持仓观望"？**
A: SMC 决策树要求 4 个买入条件全部满足才会输出买入信号。这是设计上的严格性，避免在震荡市中频繁交易。

**Q: 长线趋势熔断触发后看不到技术分析？**
A: 这是设计意图。50周均线 < 200周均线意味着长线结构性熊市，系统会完全跳过日线级别的 SMC 分析，只展示长线均线偏离度和基本面数据，防止抄底冲动。

## License

MIT
