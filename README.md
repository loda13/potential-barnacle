# 股票K线技术分析工具

命令行股票技术分析工具，支持A股和美股。

## 安装

```bash
pip install pandas numpy yfinance akshare
```

## 使用

```bash
# A股日线
python stock_analyzer.py 600519 -d 60

# 美股周线
python stock_analyzer.py AAPL -p w -d 30

# 演示模式（离线）
python stock_analyzer.py 600519 --demo
```

## 参数

| 参数 | 说明 |
|------|------|
| `code` | 股票代码（A股6位数字，美股如AAPL） |
| `-p` | 周期：d=日线，w=周线 |
| `-d` | 分析天数 |
| `--demo` | 演示模式 |

## 输出

- MA5/10/20 移动平均线
- RSI 相对强弱指标
- MACD 指标
- 布林带
- 趋势判断
- 买卖信号
- 支撑压力位
- 一句话总结