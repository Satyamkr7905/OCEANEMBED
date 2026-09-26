"""PyTorch Lightning training harness for OceanEmbed and the two baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from track2_model_engine.config import load_model_config
from track2_model_engine.dataset.zarr_dataset import (
    Batch,
    SyntheticOceanDataset,
    ZarrDataset,
    make_dataloader,
)
from track2_model_engine.losses.physics_loss import OceanEmbedLoss
from track2_model_engine.models.ocean_embed import build_model

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install pytorch-lightning / lightning: pip install lightning") from exc


class OceanEmbedLitModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "oceanembed",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        model_kwargs: dict[str, Any] | None = None,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_name, **(model_kwargs or {}))
        self.loss_fn = OceanEmbedLoss(**(loss_kwargs or {}))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def forward(self, inputs: torch.Tensor, lon: torch.Tensor | None = None):
        if lon is None:
            return self.model(inputs)
        try:
            return self.model(inputs, lon)
        except TypeError:
            return self.model(inputs)

    def _shared_step(self, batch: Batch, stage: str) -> torch.Tensor:
        out = self.forward(batch.inputs, batch.lon)
        breakdown = self.loss_fn(
            out.delta_theta,
            out.delta_sp,
            batch.delta_theta,
            batch.delta_sp,
            batch.clim_theta,
            batch.clim_sp,
            batch.valid,
            batch.lat,
        )
        self.log(f"{stage}/loss", breakdown.total, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.inputs.size(0))
        self.log(f"{stage}/mse_theta", breakdown.mse_theta, on_epoch=True, batch_size=batch.inputs.size(0))
        self.log(f"{stage}/mse_sp", breakdown.mse_sp, on_epoch=True, batch_size=batch.inputs.size(0))
        self.log(f"{stage}/inversion", breakdown.inversion, on_epoch=True, batch_size=batch.inputs.size(0))
        return breakdown.total

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, int(self.trainer.max_epochs)))
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def _datasets(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[Dataset, Dataset]:
    if args.synthetic:
        n = max(4, args.batch_size * 4)
        return SyntheticOceanDataset(n, seed=0), SyntheticOceanDataset(n, seed=1)
    root = Path(args.data_root)
    train = ZarrDataset(root, "train")
    val = ZarrDataset(root, "val")
    return train, val


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train OceanEmbed / baselines")
    p.add_argument("--model", default="oceanembed", choices=["oceanembed", "mlp", "unet"])
    p.add_argument("--data-root", type=Path, default=Path("oceanembed"))
    p.add_argument("--synthetic", action="store_true", help="Smoke-train on synthetic NIO tensors")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default=None)
    p.add_argument("--ckpt-dir", type=Path, default=Path("oceanembed/checkpoints"))
    p.add_argument("--fast-dev-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_model_config()
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    lcfg = cfg["loss"]
    batch_size = args.batch_size or int(tcfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(tcfg["num_workers"])
    if args.synthetic:
        num_workers = 0
    max_epochs = args.max_epochs or int(tcfg["max_epochs"])
    lr = args.lr or float(tcfg["lr"])
    precision = args.precision or tcfg["precision"]

    model_kwargs: dict[str, Any]
    if args.model == "mlp":
        model_kwargs = {"hidden": int(mcfg["mlp_hidden"]), "dropout": float(mcfg["dropout"])}
    elif args.model == "unet":
        model_kwargs = {"base": int(mcfg["unet_base"]), "dropout": float(mcfg["dropout"])}
    else:
        model_kwargs = {
            "stream_a_dim": int(mcfg["stream_a_dim"]),
            "stream_b_dim": int(mcfg["stream_b_dim"]),
            "latent_dim": int(mcfg["latent_dim"]),
            "n_tokens": int(mcfg["n_depth_tokens"]),
            "n_heads": int(mcfg["n_heads"]),
            "dropout": float(mcfg["dropout"]),
        }

    module = OceanEmbedLitModule(
        model_name=args.model,
        lr=lr,
        weight_decay=float(tcfg["weight_decay"]),
        model_kwargs=model_kwargs,
        loss_kwargs={
            "w_theta": float(lcfg["w_theta"]),
            "w_sp": float(lcfg["w_sp"]),
            "lambda_inv": float(lcfg["lambda_inv"]),
            "lambda_area": float(lcfg["lambda_area"]),
            "inversion_eps": float(lcfg["inversion_eps"]),
            "barrier_ds_threshold": float(lcfg["barrier_ds_threshold"]),
        },
    )
    train_ds, val_ds = _datasets(args, cfg)
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=f"{args.model}-{{epoch:03d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        ckpt,
        EarlyStopping(monitor="val/loss", mode="min", patience=12),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=float(tcfg["gradient_clip_val"]),
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=5,
        default_root_dir=str(args.ckpt_dir.parent / "lightning_logs"),
    )
    trainer.fit(module, train_loader, val_loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
