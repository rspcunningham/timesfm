from __future__ import annotations

import time

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
  # The upstream module only auto-selects CUDA or CPU, so override it for Apple
  # Silicon before compile() builds its decode closure.
  model.model.device = device
  model.model.device_count = 1
  model.model.to(device)
  model.model.eval()


def build_demo_inputs() -> list[np.ndarray]:
  x = np.linspace(0, 16 * np.pi, 512, dtype=np.float32)
  trend = np.linspace(0, 1, 512, dtype=np.float32)
  return [
    np.sin(x) + 0.1 * trend,
    0.5 * np.cos(0.5 * x) + 0.2 * np.sin(2 * x) + 0.05 * trend,
  ]


def main() -> None:
  device = require_mps()
  torch.set_float32_matmul_precision("high")
  inputs = build_demo_inputs()

  print(f"Using device: {device}")
  print("Loading TimesFM 2.5 from Hugging Face...")

  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,
  )
  move_timesfm_to_mps(model, device)

  forecast_config = timesfm.ForecastConfig(
    max_context=512,
    max_horizon=128,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=False,
    fix_quantile_crossing=True,
    per_core_batch_size=len(inputs),
  )

  print("Compiling decode path...")
  model.compile(forecast_config)

  horizon = 12

  # One warmup pass helps avoid timing the first MPS dispatch cost.
  model.forecast(horizon=horizon, inputs=[arr.copy() for arr in inputs])

  start = time.perf_counter()
  point_forecast, quantile_forecast = model.forecast(
    horizon=horizon,
    inputs=[arr.copy() for arr in inputs],
  )
  elapsed_ms = (time.perf_counter() - start) * 1000

  print(f"Point forecast shape: {point_forecast.shape}")
  print(f"Quantile forecast shape: {quantile_forecast.shape}")
  print(f"Inference time: {elapsed_ms:.1f} ms")
  print("First series point forecast:")
  print(np.array2string(point_forecast[0], precision=4))
  print("First series q10/q50/q90 for first 3 steps:")
  print(np.array2string(quantile_forecast[0, :3, [1, 5, 9]], precision=4))


if __name__ == "__main__":
  main()
