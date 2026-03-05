# STAR Market Stock Prediction

Machine learning model for predicting next-day returns on China's STAR Market (科创板), achieving **89.9% annualized return** over backtesting period (2019–2025).

## Results

| Metric | Value |
|--------|-------|
| Annualized Return | 89.9% |
| Daily Alpha vs STAR 50 | +0.54% |
| Maximum Drawdown | 31.8% |
| Backtest Period | Jul 2019 – Aug 2025 |

## Approach

### Data
- **556,000+ datapoints** across **589 stocks** on the STAR Market
- Daily OHLCV data plus order flow metrics from Wind/Tushare

### Features (8 engineered)
- `return_1day` — Previous day return
- `price_amplitude_2days` — 2-day price range / close
- `high_low_price_amplitude_diff_5days` — 5-day high-low spread trend
- `trade_count_1day` — Number of trades
- `amount_per_trade_5days` — Average trade size (5-day)
- `turnover_adjusted_by_price_amplitude_1day` — Volume normalized by volatility
- `big_order_opening_5days` — Large order flow at open
- `small_order_2days` — Retail order flow proxy

### Model
- **Random Forest** classifier predicting next-day return direction
- Compared against XGBoost baseline
- Chronological train/test split (no lookahead bias)

### Evaluation
- Walk-forward backtesting
- Feature importance analysis
- Risk metrics: Sharpe ratio, max drawdown, win rate

## Repository Structure

```
star_market/
├── src/
│   └── ml.ipynb          # Main notebook: EDA, feature engineering, model training, backtesting
├── tests/                # Unit tests
└── README.md
```

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Random Forest)
- XGBoost (baseline comparison)
- matplotlib, seaborn (visualization)

## Usage

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Run notebook
jupyter notebook src/ml.ipynb
```

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Not financial advice.

