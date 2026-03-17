# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language CLI stock technical analysis tool (`stock_analyzer.py`, ~5800 lines, single-file Python app). Supports A-shares (China), US stocks, HK stocks, and TW stocks. Built on SMC (Smart Money Concepts) and EV (Expected Value) principles with a strict decision tree, circuit breaker risk control, volatility-parity position sizing, and execution logging.

## Commands

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run analysis (examples)
python3 stock_analyzer.py 601012 -d 120         # A-share (LONGi Green Energy)
python3 stock_analyzer.py NVDA -d 120            # US stock
python3 stock_analyzer.py 0700.HK -d 120         # HK stock
python3 stock_analyzer.py TSLA --demo            # Offline demo mode (no network)
python3 stock_analyzer.py --portfolio 0700.HK 1810.HK QQQ TSLA NVDA -d 120  # Batch
python3 stock_analyzer.py NVDA -d 120 --stress-test  # With stress test
```

There are no tests, linter, or build steps configured.

## Architecture

Everything lives in a single file `stock_analyzer.py` with this decision chain (highest priority first):

### Decision Chain

1. **Circuit Breaker** (highest priority) — `check_circuit_breaker()` runs before all analysis
   - `fetch_macro_risk_data()`: benchmark index MA200 + VIX spike detection
   - `check_gap_destruction()`: gap down > 1.5x ATR through HVN
   - If triggered, overrides all buy signals globally

2. **SMC Decision Tree** — `generate_smc_decision_tree()` + `generate_decision_dashboard()`
   - BUY requires ALL 4: ChoCh confirmed, price testing HVN/FVG, R-Multiple ≥ 2.0, Z-Score in [-2,2]
   - SELL on ANY 1: Chandelier Exit breached, or distribution pattern detected
   - HOLD: everything else (distinguishes HOLD_WITH_POSITION vs HOLD_EMPTY)

3. **Position Sizing** — `calc_position_sizing()` with `calc_chandelier_exit()`
   - Formula: shares = (account × MAX_RISK_PER_TRADE) / (price - stop_loss)
   - Liquidity cap: position ≤ 1% of ADV (20-day average volume)
   - Slippage check: buy +0.2%, stop -0.2%, recheck EV ≥ 2.0

4. **Execution Log** — `append_decision_log()` silently appends to `trading_decision_log.csv`

### Data Flow

```
main() → fetch_stock_data() → validate_data() → print_analysis()
  ├─ calculate_indicators(df)           # Technical indicators + ADV20
  ├─ check_circuit_breaker()            # Macro risk + gap destruction
  ├─ build_optional_context()           # Market review, fund flow, options, Volume Profile
  │   ├─ fetch_options_data()           # IV, P/C Ratio
  │   └─ calc_anchored_volume_profile() # POC/VAH/VAL/HVN from anchor point
  ├─ calc_contrarian_signals()          # Left-side: Z-Score, divergence, vol exhaustion
  ├─ calculate_price_targets()          # Buy/stop/target + slippage check + position sizing
  ├─ generate_decision_dashboard()      # SMC decision tree (with circuit breaker override)
  └─ append_decision_log()              # Silent CSV append
```

### Key Modules

- **Technical indicators** — Pure pandas/numpy: MA, RSI, MACD, Bollinger, KDJ, ADX, ATR, OBV, CCI, SuperTrend, PSAR, Ichimoku, VWAP, MA200, ADV20. KDJ/MACD are auxiliary (`is_auxiliary=True`), excluded from main resonance.
- **Anchored Volume Profile** — `find_anchor_point()` auto-detects anchor (BoS > volume climax > earnings gap > fallback 60d), `calc_anchored_volume_profile()` computes POC/VAH/VAL/HVN/LNV.
- **Options data** — `fetch_options_data()` gets IV/P/C Ratio via yfinance; `detect_fake_breakout()` flags suspicious breakouts.
- **Left-side trading** — Z-Score, volume exhaustion, volatility regime, institutional S/R (Order Blocks, FVG), triple divergence.
- **Right-side trading** — `calc_break_of_structure()` (BoS), `calc_change_of_character()` (ChoCh).
- **Exit strategies** — VWAP deviation, distribution patterns (volume climax, bearish divergence, breakdown).
- **Batch analysis** — `analyze_portfolio()` + `print_portfolio_analysis()` with comparison table.

### Global Constants

```python
MAX_RISK_PER_TRADE = 0.02       # 2% risk per trade
DEFAULT_ACCOUNT_SIZE = 100000   # 100k CNY
CHANDELIER_ATR_MULTIPLIER = 2.5
CHANDELIER_LOOKBACK = 22
SLIPPAGE_PCT = 0.002            # 0.2% slippage
ADV_POSITION_LIMIT = 0.01      # 1% of ADV
```

### Key design patterns

- **Graceful degradation**: Extended data sources (news, fund flow, chips, options, VIX) fail silently. Core analysis only needs OHLCV data.
- **Circuit breaker override**: Macro risk and gap destruction can override any buy signal at the top of the decision chain.
- **Symbol normalization**: `normalize_symbol()` converts raw codes to Yahoo format. A-share 6/5/9 → `.SS`, others → `.SZ`.
- **Output suppression**: `call_with_suppressed_output()` wraps noisy third-party calls.
- **All UI text is in Chinese** — maintain this convention for any user-facing strings.
- **Silent logging**: `append_decision_log()` writes to CSV without terminal output; failures never interrupt main flow.

### Dependencies

- `pandas`, `numpy` — core computation
- `requests` — Yahoo chart API calls
- `yfinance` — options data, analyst consensus (imported lazily)
- `akshare` — A-share fallback data, chip distribution, fund flow (imported lazily)
