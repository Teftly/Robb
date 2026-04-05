"""
Stock price forecasting with TimesFM 2.5.

Pipeline:
  1. Fetch real OHLCV data via yfinance (falls back to synthetic GBM if blocked)
  2. Load TimesFM 2.5 pretrained checkpoint (falls back to random weights if blocked)
  3. Forecast log-returns, convert back to price levels
  4. Plot history + forecast with quantile band -> stock_forecast.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TICKERS      = ["AAPL", "MSFT", "GOOGL"]
HISTORY_DAYS = 252          # ~1 trading year of context
HORIZON      = 30           # forecast 30 trading days (~6 weeks)
PLOT_TAIL    = 60           # show last N history days in chart for clarity

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data — try yfinance, fall back to Geometric Brownian Motion
# ─────────────────────────────────────────────────────────────────────────────
def gbm_prices(n: int, s0: float = 150.0, mu: float = 0.0003,
               sigma: float = 0.018, seed: int = 0) -> np.ndarray:
    """Simulate stock prices via Geometric Brownian Motion."""
    rng = np.random.default_rng(seed)
    returns = mu + sigma * rng.standard_normal(n)
    return s0 * np.exp(np.cumsum(returns)).astype(np.float32)

price_data: dict[str, np.ndarray] = {}
data_source = "live"

try:
    import yfinance as yf
    print("Fetching price data from Yahoo Finance …")
    for ticker in TICKERS:
        df = yf.Ticker(ticker).history(period="2y")
        prices = df["Close"].values[-HISTORY_DAYS:].astype(np.float32)
        if len(prices) < 30:
            raise ValueError(f"Too few rows for {ticker}")
        price_data[ticker] = prices
    print(f"  Loaded {[len(v) for v in price_data.values()]} days for {TICKERS}")

except Exception as e:
    print(f"Yahoo Finance unavailable ({e.__class__.__name__}). Using synthetic GBM data.\n")
    data_source = "synthetic GBM"
    seeds   = [1, 2, 3]
    s0s     = [182.0, 415.0, 172.0]
    sigmas  = [0.017, 0.016, 0.019]
    for ticker, seed, s0, sigma in zip(TICKERS, seeds, s0s, sigmas):
        price_data[ticker] = gbm_prices(HISTORY_DAYS, s0=s0, sigma=sigma, seed=seed)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load TimesFM 2.5
# ─────────────────────────────────────────────────────────────────────────────
import timesfm
from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
import torch

model_source = "pretrained"

try:
    print("\nLoading TimesFM 2.5 pretrained checkpoint …")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )
    tfm.compile(ForecastConfig(
        max_context=HISTORY_DAYS,
        max_horizon=HORIZON,
        normalize_inputs=True,
        per_core_batch_size=len(TICKERS),
    ))
    print("Checkpoint loaded and compiled.")

    def do_forecast(log_return_inputs):
        points, quantiles = tfm.forecast(horizon=HORIZON, inputs=log_return_inputs)
        return points, quantiles

except Exception as e:
    print(f"Pretrained checkpoint unavailable ({e.__class__.__name__}). Using random-weight model.\n")
    model_source = "random weights"
    _module = TimesFM_2p5_200M_torch_module()
    _module.eval()

    def do_forecast(log_return_inputs):
        raw = _module.forecast_naive(horizon=HORIZON, inputs=log_return_inputs)
        # raw[i]: (horizon, n_quantiles+1) or (horizon,)
        points = np.stack([r[:HORIZON, 0] if r.ndim == 2 else r[:HORIZON] for r in raw])
        lo     = np.stack([r[:HORIZON, 0]  if r.ndim == 2 else r[:HORIZON] for r in raw])
        hi     = np.stack([r[:HORIZON, -1] if r.ndim == 2 else r[:HORIZON] for r in raw])
        # pack into same shape as tfm.forecast: (n, horizon, n_quantiles)
        quantiles = np.stack([lo, hi], axis=-1)
        return points, quantiles

# ─────────────────────────────────────────────────────────────────────────────
# 3. Forecast on log-returns, convert back to price
# ─────────────────────────────────────────────────────────────────────────────
print("\nForecasting …")

log_return_inputs = []
last_prices       = []

for ticker in TICKERS:
    prices = price_data[ticker]
    log_returns = np.diff(np.log(prices)).astype(np.float32)
    log_return_inputs.append(log_returns)
    last_prices.append(prices[-1])

point_forecasts, quantile_forecasts = do_forecast(log_return_inputs)

def returns_to_prices(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(log_returns))

results = {}
for i, ticker in enumerate(TICKERS):
    pt   = returns_to_prices(last_prices[i], point_forecasts[i])
    lo   = returns_to_prices(last_prices[i], quantile_forecasts[i, :, 0])
    hi   = returns_to_prices(last_prices[i], quantile_forecasts[i, :, -1])
    results[ticker] = {"point": pt, "lo": lo, "hi": hi}
    pct = (pt[-1] / last_prices[i] - 1) * 100
    print(f"  {ticker:5s}  last=${last_prices[i]:.2f}  "
          f"forecast end=${pt[-1]:.2f}  ({pct:+.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(TICKERS), 1, figsize=(12, 4 * len(TICKERS)), sharex=False)
if len(TICKERS) == 1:
    axes = [axes]

hist_tail_x  = np.arange(PLOT_TAIL)
forecast_x   = np.arange(PLOT_TAIL - 1, PLOT_TAIL - 1 + HORIZON)

for ax, ticker in zip(axes, TICKERS):
    prices = price_data[ticker]
    tail   = prices[-PLOT_TAIL:]
    pt     = results[ticker]["point"]
    lo     = results[ticker]["lo"]
    hi     = results[ticker]["hi"]

    ax.plot(hist_tail_x, tail, color="steelblue", label="History (tail)")
    ax.plot(forecast_x,  pt,   color="tomato", linewidth=2, label="Forecast (point)")
    ax.fill_between(forecast_x, lo, hi, alpha=0.25, color="tomato", label="Quantile band")
    ax.axvline(PLOT_TAIL - 1, color="gray", linestyle="--", linewidth=0.8)

    ax.set_title(f"{ticker}  —  last ${prices[-1]:.2f}  →  forecast ${pt[-1]:.2f}")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8)

model_label = f"TimesFM 2.5 ({model_source})"
data_label  = f"data: {data_source}"
fig.suptitle(f"{model_label} | {data_label}", fontsize=11, y=1.01)
axes[-1].set_xlabel("Trading days")

plt.tight_layout()
out_path = "stock_forecast.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved to {out_path}")
