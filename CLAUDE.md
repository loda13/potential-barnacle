# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language CLI stock technical analysis tool (`stock_analyzer.py`, ~2900 lines, single-file Python app). Supports A-shares (China), US stocks, and HK stocks. Outputs multi-indicator resonance analysis, fund flow, news sentiment, buy/sell signals, and an LLM decision dashboard — all to stdout.

## Commands

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run analysis (examples)
python3 stock_analyzer.py 600519 -d 60          # A-share daily
python3 stock_analyzer.py AAPL -p w -d 240       # US stock weekly
python3 stock_analyzer.py 0700.HK -d 60          # HK stock
python3 stock_analyzer.py TSLA --demo             # Offline demo mode (no network)
```

There are no tests, linter, or build steps configured.

## Architecture

Everything lives in a single file `stock_analyzer.py` with this flow:

1. **Entry** — `main()` parses args, validates stock code, fetches data, runs analysis
2. **Data fetching** — `fetch_stock_data()` tries Yahoo chart API first (`safe_fetch_yfinance_chart`); for A-shares, falls back to `akshare`. Includes retry with backoff on rate limits. `--demo` generates synthetic data via `generate_demo_data()`.
3. **Data cleaning/validation** — `clean_stock_data()` and `validate_data()` ensure OHLCV integrity
4. **Technical indicators** — Pure pandas/numpy implementations (no pandas_ta dependency): MA, RSI, MACD, Bollinger, KDJ, ADX, ATR, OBV, CCI, SuperTrend, PSAR, Ichimoku. All `calc_*` functions take a DataFrame and return Series/DataFrame.
5. **Signal detection** — `detect_signals()` and `analyze_indicator_signals()` classify each indicator as buy/sell/neutral
6. **Resonance & scoring** — `calculate_resonance()` aggregates signals; `calculate_win_rate()` estimates probability
7. **Extended data** (network, graceful degradation) — `fetch_chip_distribution()`, `fetch_fund_flow()`, `fetch_news_sentiment()`, `get_market_review()`, `fetch_analyst_consensus()`. These are gathered in `build_optional_context()` and failures never interrupt the main flow.
8. **Decision outputs** — `calculate_price_targets()`, `generate_checklist()`, `generate_decision_dashboard()`
9. **Rendering** — `print_analysis()` orchestrates all of the above and prints formatted tables/sections to stdout

### Key design patterns

- **Graceful degradation**: Extended data sources (news, fund flow, chips, market review) fail silently. Core analysis only needs OHLCV data from Yahoo/akshare.
- **Symbol normalization**: `normalize_symbol()` converts raw codes (e.g., `600519`) to Yahoo format (`600519.SS`). A-share codes starting with 6/5/9 → `.SS` (Shanghai), others → `.SZ` (Shenzhen).
- **Output suppression**: `call_with_suppressed_output()` wraps noisy third-party calls (akshare) to keep stdout clean.
- **All UI text is in Chinese** — maintain this convention for any user-facing strings.

### Dependencies

- `pandas`, `numpy` — core computation
- `requests` — Yahoo chart API calls
- `yfinance` — analyst consensus, news sentiment (imported lazily)
- `akshare` — A-share fallback data, chip distribution, fund flow (imported lazily)
