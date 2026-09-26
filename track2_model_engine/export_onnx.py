"""Export OceanEmbed (or a baseline) to ONNX and TorchScript."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from track2_model_engine.config import N_CHANNELS, N_LAT, N_LON, WINDOW_DAYS, target_longitudes
from track2_model_engine.models.heads import ModelOutput
from track2_model_engine.models.layers import set_mc_dropout
from track2_model_engine.models.ocean_embed import build_model
from track2_model_engine.train import OceanEmbedLitModule


class ExportWrapper(torch.nn.Module):
    """ONNX/TorchScript-friendly module that returns a tuple of tensors."""

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.core = core
        self.register_buffer("lon", torch.from_numpy(target_longitudes()).float(), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            out: ModelOutput = self.core(x, self.lon)
        except TypeError:
            out = self.core(x)
        return out.delta_theta, out.delta_sp, out.sigma_theta(), out.sigma_sp()


def load_core(ckpt: Path | None, model_name: str) -> torch.nn.Module:
    if ckpt is not None:
        module = OceanEmbedLitModule.load_from_checkpoint(str(ckpt), map_location="cpu")
        core = module.model
    else:
        core = build_model(model_name)
    core.eval()
    set_mc_dropout(core, False)
    return core


def export(
    *,
    ckpt: Path | None,
    model_name: str,
    out_dir: Path,
    opset: int = 17,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(load_core(ckpt, model_name))
    wrapper.eval()
    dummy = torch.zeros(1, WINDOW_DAYS, N_CHANNELS, N_LAT, N_LON, dtype=torch.float32)

    ts_path = out_dir / f"{model_name}.ts"
    traced = torch.jit.trace(wrapper, dummy, strict=False)
    traced.save(str(ts_path))

    onnx_path = out_dir / f"{model_name}.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["surface_window"],
        output_names=["delta_theta", "delta_sp", "sigma_theta", "sigma_sp"],
        dynamic_axes={
            "surface_window": {0: "batch"},
            "delta_theta": {0: "batch"},
            "delta_sp": {0: "batch"},
            "sigma_theta": {0: "batch"},
            "sigma_sp": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return onnx_path, ts_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export ONNX + TorchScript")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--model", default="oceanembed")
    p.add_argument("--out-dir", type=Path, default=Path("oceanembed/export"))
    p.add_argument("--opset", type=int, default=17)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    onnx_path, ts_path = export(ckpt=args.ckpt, model_name=args.model, out_dir=args.out_dir, opset=args.opset)
    print(f"ONNX: {onnx_path}")
    print(f"TorchScript: {ts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
