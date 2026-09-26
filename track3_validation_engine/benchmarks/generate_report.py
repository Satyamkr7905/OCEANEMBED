"""Export JSON and Markdown performance matrices for OceanEmbed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from track3_validation_engine.benchmarks.skill_scores import SkillReport


def _skill_table_md(report: SkillReport) -> str:
    lines = [
        f"### {report.name}",
        "",
        f"- RMSE (all): **{report.rmse_all:.4f}**",
        f"- Bias (all): **{report.bias_all:.4f}**",
        f"- Pearson r (all): **{report.pearson_all:.4f}**",
    ]
    if report.css is not None:
        lines.append(f"- Climatology Skill Score: **{report.css:.4f}**")
    lines += [
        "",
        "| Depth (m) | N | RMSE | Bias | r |",
        "|---:|---:|---:|---:|---:|",
    ]
    for z, n, rmse, bias, r in zip(report.depths, report.n, report.rmse, report.bias, report.pearson_r):
        def fmt(v: float) -> str:
            return "—" if v != v else f"{v:.4f}"  # NaN check

        lines.append(f"| {z:.0f} | {n} | {fmt(rmse)} | {fmt(bias)} | {fmt(r)} |")
    lines.append("")
    return "\n".join(lines)


def _walk(obj: Any) -> Any:
    if isinstance(obj, SkillReport):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _walk(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def write_report(
    payload: Mapping[str, Any],
    out_dir: Path | str,
    *,
    stem: str = "oceanembed_validation",
) -> tuple[Path, Path]:
    """Write ``stem.json`` and ``stem.md`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": "OceanEmbed SIH26066",
        "notes": {
            "argo_min_depth_m": 5,
            "zero_metre": "validated against OISST, not ARGO",
            "teos10": "SA_from_SP then CT_from_pt(SA, theta); density gsw.rho(SA, CT, p)",
            "css": "1 - MSE_model / MSE_clim",
        },
        "results": _walk(dict(payload)),
    }
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")

    md: list[str] = [
        "# OceanEmbed validation report",
        "",
        f"Generated: {body['generated_utc']}",
        "",
        "## Protocol",
        "",
        "- ARGO in-situ scores use **z ≥ 5 m** (pump cutoff).",
        "- **0 m** is scored against OISST (skin/sub-skin; 0.1–0.3 °C bulk offset is a known limitation).",
        "- TEOS-10 conversions are vectorized through `gsw`.",
        "- TCHP uses dynamic density ρ(S_A, Θ, p) and C_p = 4178 J kg⁻¹ K⁻¹.",
        "- MLD uses Δρ = 0.03 kg m⁻³ from 10 m; Z20 is the 20 °C Conservative Temperature isotherm.",
        "",
        "## Scores",
        "",
    ]
    results = body["results"]
    for key, value in results.items():
        md.append(f"## {key}")
        md.append("")
        if isinstance(value, dict) and "rmse" in value and "depths" in value:
            md.append(_skill_table_md(SkillReport(**{k: value[k] for k in SkillReport.__dataclass_fields__ if k in value})))
        elif isinstance(value, dict):
            for nested_k, nested in value.items():
                if isinstance(nested, dict) and "rmse" in nested and "depths" in nested:
                    fields = {k: nested[k] for k in SkillReport.__dataclass_fields__ if k in nested}
                    fields.setdefault("extra", nested.get("extra", {}))
                    md.append(_skill_table_md(SkillReport(**fields)))
                else:
                    md.append(f"- **{nested_k}**: `{nested}`")
                    md.append("")
        else:
            md.append(f"`{value}`")
            md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path
