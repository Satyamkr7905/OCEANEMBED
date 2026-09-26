"""Checkpoint loading, MC-dropout inference, and depth-wise metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from track2_model_engine.config import STANDARD_DEPTHS
from track2_model_engine.dataset.zarr_dataset import SyntheticOceanDataset, ZarrDataset, make_dataloader
from track2_model_engine.losses.area_weighted_loss import masked_area_mse
from track2_model_engine.models.heads import ModelOutput
from track2_model_engine.models.layers import set_mc_dropout
from track2_model_engine.train import OceanEmbedLitModule


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.model if hasattr(module, "model") else module


@torch.no_grad()
def mc_predict(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    lon: torch.Tensor,
    *,
    n_samples: int = 50,
) -> dict[str, torch.Tensor]:
    """Mean and epistemic std over `n_samples` stochastic forward passes."""
    core = _unwrap(model)
    set_mc_dropout(core, True)
    model.eval()
    thetas: list[torch.Tensor] = []
    sps: list[torch.Tensor] = []
    alea_t: list[torch.Tensor] = []
    alea_s: list[torch.Tensor] = []
    for _ in range(n_samples):
        try:
            out: ModelOutput = core(inputs, lon)
        except TypeError:
            out = core(inputs)
        thetas.append(out.delta_theta)
        sps.append(out.delta_sp)
        alea_t.append(out.sigma_theta())
        alea_s.append(out.sigma_sp())
    set_mc_dropout(core, False)
    stack_t = torch.stack(thetas, dim=0)
    stack_s = torch.stack(sps, dim=0)
    mu_t = stack_t.mean(0)
    mu_s = stack_s.mean(0)
    epi_t = stack_t.std(0, unbiased=False)
    epi_s = stack_s.std(0, unbiased=False)
    aleatoric_t = torch.stack(alea_t, dim=0).mean(0)
    aleatoric_s = torch.stack(alea_s, dim=0).mean(0)
    return {
        "delta_theta": mu_t,
        "delta_sp": mu_s,
        "sigma_theta": torch.sqrt(epi_t.pow(2) + aleatoric_t.pow(2) + 1e-12),
        "sigma_sp": torch.sqrt(epi_s.pow(2) + aleatoric_s.pow(2) + 1e-12),
        "epistemic_theta": epi_t,
        "epistemic_sp": epi_s,
        "aleatoric_theta": aleatoric_t,
        "aleatoric_sp": aleatoric_s,
    }


def depthwise_rmse(pred: torch.Tensor, true: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    se = (pred - true).pow(2)
    out = []
    for k in range(pred.size(1)):
        w = valid[:, k]
        denom = w.sum().clamp_min(1e-6)
        out.append(torch.sqrt((se[:, k] * w).sum() / denom).item())
    return np.asarray(out, dtype=np.float64)


def coverage(pred: torch.Tensor, true: torch.Tensor, sigma: torch.Tensor, valid: torch.Tensor, k: float) -> float:
    inside = ((true - pred).abs() <= k * sigma).to(dtype=pred.dtype) * valid
    return float(inside.sum() / valid.sum().clamp_min(1e-6))


def load_from_checkpoint(ckpt: Path, map_location: str | torch.device = "cpu") -> OceanEmbedLitModule:
    return OceanEmbedLitModule.load_from_checkpoint(str(ckpt), map_location=map_location)


def run_eval(
    ckpt: Path,
    *,
    data_root: Path,
    split: str = "test",
    mc_samples: int = 50,
    batch_size: int = 1,
    synthetic: bool = False,
    device: str | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    module = load_from_checkpoint(ckpt, map_location=device)
    module.to(device)
    module.eval()
    if synthetic:
        loader = make_dataloader(SyntheticOceanDataset(8, seed=3), batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        ds = ZarrDataset(data_root, split)  # type: ignore[arg-type]
        loader = make_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    sse_t = torch.zeros(15)
    sse_s = torch.zeros(15)
    wsum = torch.zeros(15)
    cov1 = 0.0
    cov2 = 0.0
    n_cov = 0.0
    mse_clim_t = []

    for batch in tqdm(loader, desc=f"eval-{split}"):
        batch = batch.to(torch.device(device))
        pred = mc_predict(module, batch.inputs, batch.lon, n_samples=mc_samples)
        valid = batch.valid
        diff_t = (pred["delta_theta"] - batch.delta_theta).pow(2)
        diff_s = (pred["delta_sp"] - batch.delta_sp).pow(2)
        sse_t += (diff_t * valid).sum(dim=(0, 2, 3)).cpu()
        sse_s += (diff_s * valid).sum(dim=(0, 2, 3)).cpu()
        wsum += valid.sum(dim=(0, 2, 3)).cpu()
        cov1 += coverage(pred["delta_theta"], batch.delta_theta, pred["sigma_theta"], valid, 1.0) * valid.size(0)
        cov2 += coverage(pred["delta_theta"], batch.delta_theta, pred["sigma_theta"], valid, 2.0) * valid.size(0)
        n_cov += valid.size(0)
        mse_clim_t.append(masked_area_mse(batch.delta_theta.new_zeros(batch.delta_theta.shape), batch.delta_theta, valid, batch.lat).item())

    rmse_t = torch.sqrt(sse_t / wsum.clamp_min(1e-6)).numpy()
    rmse_s = torch.sqrt(sse_s / wsum.clamp_min(1e-6)).numpy()
    mse_model = float(np.nanmean(rmse_t**2))
    mse_clim = float(np.mean(mse_clim_t)) if mse_clim_t else float("nan")
    css = 1.0 - mse_model / mse_clim if mse_clim and mse_clim > 0 else float("nan")
    report = {
        "checkpoint": str(ckpt),
        "split": split,
        "mc_samples": mc_samples,
        "depths_m": list(STANDARD_DEPTHS),
        "rmse_theta": rmse_t.tolist(),
        "rmse_sp": rmse_s.tolist(),
        "climatology_skill_score_theta": css,
        "coverage_1sigma": cov1 / max(n_cov, 1e-6),
        "coverage_2sigma": cov2 / max(n_cov, 1e-6),
        "nominal_1sigma": 0.68,
        "nominal_2sigma": 0.95,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MC-dropout evaluation")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("oceanembed"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--mc", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("oceanembed/metadata/eval_report.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_eval(
        args.ckpt,
        data_root=args.data_root,
        split=args.split,
        mc_samples=args.mc,
        batch_size=args.batch_size,
        synthetic=args.synthetic,
        out_json=args.out,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
