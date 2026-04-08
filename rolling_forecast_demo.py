from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import timesfm


def require_mps() -> torch.device:
  if not torch.backends.mps.is_built():
    raise RuntimeError("PyTorch was not built with MPS support.")
  if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available on this machine.")
  return torch.device("mps")


def move_timesfm_to_mps(model: timesfm.TimesFM_2p5_200M_torch,
                        device: torch.device) -> None:
  model.model.device = device
  model.model.device_count = 1
  model.model.to(device)
  model.model.eval()


def build_demo_series(num_points: int, seed: int = 7) -> np.ndarray:
  rng = np.random.default_rng(seed)
  x = np.linspace(0, 6 * np.pi, num_points, dtype=np.float32)
  trend = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
  noise = rng.normal(loc=0.0, scale=0.08, size=num_points).astype(np.float32)
  return 0.7 * np.sin(x) + 0.25 * np.cos(0.4 * x) + 0.35 * trend + noise


def load_model(num_points: int) -> timesfm.TimesFM_2p5_200M_torch:
  device = require_mps()
  torch.set_float32_matmul_precision("high")

  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,
  )
  move_timesfm_to_mps(model, device)

  model.compile(
    timesfm.ForecastConfig(
      max_context=num_points,
      max_horizon=num_points,
      normalize_inputs=True,
      use_continuous_quantile_head=True,
      force_flip_invariance=True,
      infer_is_positive=False,
      fix_quantile_crossing=True,
      per_core_batch_size=num_points,
    )
  )
  return model


def generate_rollouts(model: timesfm.TimesFM_2p5_200M_torch,
                      series: np.ndarray) -> np.ndarray:
  num_points = len(series)
  rollouts = np.full((num_points, num_points), np.nan, dtype=np.float32)
  context = model.forecast_config.max_context
  values: list[np.ndarray] = []
  masks: list[np.ndarray] = []
  start_indices: list[int] = []

  for k in range(num_points):
    # The upstream inference path needs at least one context value. For k=0,
    # seed with the first observed value and start forecasting at x=1.
    start_idx = k if k > 0 else 1
    start_indices.append(start_idx)
    rollouts[k, :start_idx] = series[:start_idx]

    value = np.pad(
      series[:start_idx],
      (context - start_idx, 0),
      mode="constant",
      constant_values=0.0,
    ).astype(np.float32)
    mask = np.array(
      [True] * (context - start_idx) + [False] * start_idx,
      dtype=bool,
    )
    values.append(value)
    masks.append(mask)

  if model.compiled_decode is None:
    raise RuntimeError("Model is not compiled.")

  # Warmup once to avoid timing first-dispatch overhead on MPS.
  model.compiled_decode(1, values[:1], masks[:1])
  point_forecast, _ = model.compiled_decode(num_points - 1, values, masks)

  for k, start_idx in enumerate(start_indices):
    if start_idx < num_points:
      rollouts[k, start_idx:] = point_forecast[k, :num_points - start_idx]

  return rollouts


def plot_rollouts(series: np.ndarray,
                  rollouts: np.ndarray,
                  output_path: str) -> None:
  num_points = len(series)
  x = np.arange(num_points)
  cmap = plt.get_cmap("viridis")
  residuals = rollouts - series[None, :]

  fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 10),
    sharex=True,
    constrained_layout=True,
    height_ratios=(2.2, 1.4),
  )
  ax_forecast, ax_error = axes
  ax_forecast.plot(x, series, color="black", linewidth=3, label="actual")

  for k in range(num_points):
    color = cmap(k / max(1, num_points - 1))
    ax_forecast.plot(x, rollouts[k], color=color, alpha=0.12, linewidth=1)
    ax_error.plot(x, residuals[k], color=color, alpha=0.12, linewidth=1)

  highlighted = sorted(set([0, num_points // 4, num_points // 2,
                            (3 * num_points) // 4, num_points - 1]))
  for k in highlighted:
    color = cmap(k / max(1, num_points - 1))
    ax_forecast.plot(
      x,
      rollouts[k],
      color=color,
      alpha=0.95,
      linewidth=2,
      label=f"cutoff {k}",
    )
    ax_error.plot(
      x,
      residuals[k],
      color=color,
      alpha=0.95,
      linewidth=2,
      label=f"cutoff {k}",
    )

  ax_forecast.set_title("Rolling TimesFM Forecasts Against Actual Series")
  ax_forecast.set_ylabel("value")
  ax_forecast.grid(True, alpha=0.25)
  ax_forecast.legend(loc="best")

  ax_error.axhline(0.0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
  ax_error.set_title("Forecast Minus Actual")
  ax_error.set_xlabel("time index")
  ax_error.set_ylabel("error")
  ax_error.grid(True, alpha=0.25)

  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Generate a rolling-origin TimesFM visualization on a demo series."
  )
  parser.add_argument(
    "--num-points",
    type=int,
    default=48,
    help="Length of the synthetic demo series.",
  )
  parser.add_argument(
    "--output",
    default="rolling_forecast_demo.png",
    help="Path to the output PNG plot.",
  )
  args = parser.parse_args()

  num_points = args.num_points
  output_path = args.output

  print("Loading model on MPS...")
  model = load_model(num_points=num_points)
  series = build_demo_series(num_points=num_points)

  start = time.perf_counter()
  rollouts = generate_rollouts(model, series)
  elapsed_ms = (time.perf_counter() - start) * 1000

  plot_rollouts(series, rollouts, output_path)

  print(f"Series shape: {series.shape}")
  print(f"Rollout matrix shape: {rollouts.shape}")
  print(f"Total rollout time: {elapsed_ms:.1f} ms")
  print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
  main()
