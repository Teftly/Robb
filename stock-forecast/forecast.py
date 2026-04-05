"""TimesFM 2.5 forecast demo: sine wave with noise.

Tries to load the real checkpoint from HuggingFace.
Falls back to random-weight model (pipeline demo) if the network is blocked.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 1. Synthetic time series ──────────────────────────────────────────────────
np.random.seed(42)
n_history = 200
t = np.arange(n_history, dtype=np.float32)
history = (np.sin(2 * np.pi * t / 30) + 0.2 * np.random.randn(n_history)).astype(np.float32)
horizon = 50

# ── 2. Load TimesFM 2.5 ───────────────────────────────────────────────────────
import timesfm
from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module

using_pretrained = False
forecast_values = None
quantile_lo = quantile_hi = None

try:
    print("Attempting to load TimesFM 2.5 (200M) from HuggingFace …")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )
    using_pretrained = True
    print("Checkpoint loaded successfully.")

    # ── 3a. Compile ───────────────────────────────────────────────────────────
    print("Compiling …")
    tfm.compile(
        ForecastConfig(
            max_context=n_history,
            max_horizon=horizon,
            normalize_inputs=True,
            per_core_batch_size=1,
        )
    )

    # ── 4a. Forecast ──────────────────────────────────────────────────────────
    print("Forecasting …")
    point_forecasts, quantile_forecasts = tfm.forecast(
        horizon=horizon,
        inputs=[history],
    )
    forecast_values = point_forecasts[0]
    if quantile_forecasts is not None and quantile_forecasts.shape[-1] >= 2:
        quantile_lo = quantile_forecasts[0, :, 0]
        quantile_hi = quantile_forecasts[0, :, -1]

except Exception as e:
    print(f"\nCould not load pretrained checkpoint: {e}")
    print("Falling back to random-weight model (pipeline demo only — not a trained forecast).\n")

    # ── 3b. Instantiate with random weights ───────────────────────────────────
    import torch
    model = TimesFM_2p5_200M_torch_module()
    model.eval()

    # ── 4b. Forecast via forecast_naive ───────────────────────────────────────
    print("Running forecast_naive with uninitialised weights …")
    raw_outputs = model.forecast_naive(horizon=horizon, inputs=[history])
    # raw_outputs[0] shape: (horizon + possible extra, n_quantiles+1)
    out = raw_outputs[0]
    if out.ndim == 2:
        forecast_values = out[:horizon, 0]         # point estimate column
        if out.shape[1] >= 2:
            quantile_lo = out[:horizon, 0]
            quantile_hi = out[:horizon, -1]
    else:
        forecast_values = out[:horizon]

# ── 5. Print results ──────────────────────────────────────────────────────────
label = "pretrained checkpoint" if using_pretrained else "random-weight demo"
print(f"\nForecast ({horizon} steps, {label}):")
print(np.round(forecast_values, 4))

# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

hist_x = np.arange(n_history)
fore_x = np.arange(n_history, n_history + horizon)

ax.plot(hist_x, history, label="History", color="steelblue")
ax.plot(fore_x, forecast_values, label=f"Forecast ({label})", color="tomato", linewidth=2)

if quantile_lo is not None and quantile_hi is not None:
    ax.fill_between(fore_x, quantile_lo, quantile_hi,
                    alpha=0.25, color="tomato", label="Quantile band")

ax.axvline(n_history - 1, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Time step")
ax.set_ylabel("Value")
title = "TimesFM 2.5 — sine-wave forecast"
if not using_pretrained:
    title += " (random weights — pipeline demo)"
ax.set_title(title)
ax.legend()
plt.tight_layout()

out_path = "forecast.png"
fig.savefig(out_path, dpi=150)
print(f"\nChart saved to {out_path}")
