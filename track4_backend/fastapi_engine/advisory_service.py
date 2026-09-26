"""INCOIS operational maritime advisory generator.

The language model never invents physical numbers. OceanEmbed's PyTorch
reconstruction and TEOS-10 engine already compute θ, Sp, TCHP, MLD, Z20,
the inversion flag, and uncertainty. This module only narrates those values
as a standardized INCOIS bulletin.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENGINE_DIR = Path(__file__).resolve().parent
_ENV_CANDIDATES = (
    _ENGINE_DIR / ".env",
    _ENGINE_DIR.parent / ".env",
    Path.cwd() / ".env",
)

SYSTEM_PROMPT = """You are the Chief Oceanographic Duty Officer at INCOIS
(Indian National Centre for Ocean Information Services), Ministry of Earth
Sciences (MoES), Hyderabad.

You issue official Operational Maritime Advisory Bulletins for the North
Indian Ocean. You do NOT calculate, estimate, or invent any physical number.
Every numeric value you write MUST be copied verbatim from the telemetry
JSON provided by OceanEmbed (PyTorch reconstruction + TEOS-10). If a field
is missing, write "not reported" — never guess.

Standard interpretation rules (apply these; do not recompute):
- TCHP > 80 kJ/cm²: Extreme cyclone intensification fuel. Rapid intensification
  is favoured if a tropical cyclone is nearby.
- TCHP 50–80 kJ/cm²: Moderate intensification potential. Monitor closely.
- TCHP < 50 kJ/cm²: Depleted heat / cold-wake regime. Intensification is
  suppressed; upwelling or prior storm passage is likely.
- Inversion Flag = True: Strong freshwater barrier layer typical of the
  northern Bay of Bengal. Wind mixing is inhibited; subsurface heat can
  remain trapped beneath a thin mixed layer.
- Inversion Flag = False: No thermal inversion flagged at this station.

Respond ONLY in markdown with exactly these four sections, in this order,
using these headings:

## 1. EXECUTIVE SUMMARY
Exactly two sentences. Location, date, and the headline risk.

## 2. CYCLONE FUEL & INTENSIFICATION RISK
Interpret TCHP with the rules above. Quote the supplied TCHP and SST.

## 3. THERMOHALINE & STRATIFICATION STATUS
Discuss MLD, Z20, and the barrier-layer / inversion flag. Quote the supplied
values. Mention reconstruction uncertainty if provided.

## 4. MARITIME OPERATIONAL DIRECTIVES
Actionable recommendations for:
- IMD (cyclone warning / intensity desk)
- Fisheries (small-craft and fishing fleet)
- Shipping (commercial routing and port operations)

Tone: formal, concise, operational. No preamble, no closing signature block,
no extra sections.
"""

GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash")
OPENAI_MODEL = "gpt-4o-mini"


def _load_local_env() -> None:
    """Load GEMINI/OPENAI keys from nearby .env files without overriding the process."""
    for path in _ENV_CANDIDATES:
        if not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key in {"GEMINI_API_KEY", "OPENAI_API_KEY"} and value and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


def _api_key(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _iso(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _num(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "not reported"


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def classify_tchp(tchp: float | None) -> tuple[str, str]:
    """Return (label, rule text) for a TCHP value in kJ/cm²."""
    if tchp is None:
        return "UNKNOWN", "TCHP was not reported; intensification fuel cannot be classified."
    if tchp > 80:
        return "EXTREME", "TCHP exceeds 80 kJ/cm² — extreme cyclone intensification fuel."
    if tchp >= 50:
        return "MODERATE", "TCHP is between 50 and 80 kJ/cm² — moderate intensification potential."
    return "DEPLETED", "TCHP is below 50 kJ/cm² — depleted heat / cold-wake regime."


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def fallback_advisory(telemetry: dict[str, Any]) -> str:
    """Deterministic INCOIS bulletin used when no LLM key is present or a call fails."""
    lat = _num(telemetry.get("lat"), 2)
    lon = _num(telemetry.get("lon"), 2)
    day = _iso(telemetry.get("date", "unknown"))
    sst = _num(telemetry.get("sst"), 2)
    tchp = _safe_float(telemetry.get("tchp"))
    mld = _num(telemetry.get("mld"), 1)
    z20 = _num(telemetry.get("z20"), 1)
    uncertainty = _num(telemetry.get("uncertainty"), 2)
    inversion = _bool(telemetry.get("inversion_flag", False))
    tchp_text = _num(telemetry.get("tchp"), 1)
    band, rule = classify_tchp(tchp)

    if inversion:
        barrier = (
            "Inversion Flag is **True**: a strong freshwater barrier layer is indicated, "
            "consistent with northern Bay of Bengal monsoon runoff. Wind-driven mixing is "
            "inhibited and subsurface heat can remain trapped below a thin mixed layer."
        )
        fish = (
            "Fisheries: expect a shallow mixed layer with a sharp density cap. Small craft "
            "should treat sudden squalls as high-risk; the barrier layer can keep sea-surface "
            "conditions deceptively calm until wind stress increases."
        )
    else:
        barrier = (
            "Inversion Flag is **False**: no thermal inversion is flagged at this station. "
            "Stratification is dominated by the reported mixed-layer and 20 °C isotherm depths."
        )
        fish = (
            "Fisheries: mixed-layer depth is as reported; advise fleet to follow IMD sea-state "
            "bulletins and avoid the cyclone-risk polygon if TCHP is elevated."
        )

    if band == "EXTREME":
        exec_2 = (
            f"Tropical Cyclone Heat Potential of {tchp_text} kJ/cm² constitutes extreme "
            "oceanic fuel for rapid intensification."
        )
        imd = (
            "IMD: elevate intensity-watch posture. If a tropical cyclone is within the basin, "
            "treat rapid intensification as the baseline scenario while the system remains "
            "over this heat reservoir."
        )
        shipping = (
            "Shipping: reroute commercial traffic away from the high-TCHP filament and "
            "prepare ports for a possible upgrade in cyclone warning."
        )
    elif band == "MODERATE":
        exec_2 = (
            f"Tropical Cyclone Heat Potential of {tchp_text} kJ/cm² indicates moderate "
            "intensification potential and requires continued monitoring."
        )
        imd = (
            "IMD: maintain standard intensity surveillance. Re-evaluate if the cyclone track "
            "lingers over this water mass for more than 12 hours."
        )
        shipping = (
            "Shipping: keep published routes but raise readiness for course adjustments "
            "if IMD upgrades the warning."
        )
    else:
        exec_2 = (
            f"Tropical Cyclone Heat Potential of {tchp_text} kJ/cm² indicates a depleted "
            "or cold-wake heat reservoir that suppresses intensification."
        )
        imd = (
            "IMD: intensification is unlikely over this water mass. Prefer weakening or "
            "steady-state guidance unless the storm leaves the cold wake."
        )
        shipping = (
            "Shipping: residual swell and cross-seas from prior storm passage are the "
            "primary hazard; cyclone fuel is not."
        )

    return f"""## 1. EXECUTIVE SUMMARY

INCOIS OceanEmbed sounding at {lat}°N, {lon}°E on {day} reports SST {sst} °C, TCHP {tchp_text} kJ/cm², MLD {mld} m, and Z20 {z20} m. {exec_2}

## 2. CYCLONE FUEL & INTENSIFICATION RISK

**Classification: {band}.** {rule} Quoted telemetry: TCHP = {tchp_text} kJ/cm², SST = {sst} °C. These values were computed by the OceanEmbed PyTorch / TEOS-10 engine and are not re-derived here.

## 3. THERMOHALINE & STRATIFICATION STATUS

Mixed-layer depth (MLD) = {mld} m. Depth of the 20 °C isotherm (Z20) = {z20} m. {barrier} Reconstruction uncertainty (theta) = {uncertainty} °C.

## 4. MARITIME OPERATIONAL DIRECTIVES

- {imd}
- {fish}
- {shipping}

*Bulletin generated by OceanEmbed INCOIS Duty Desk (offline template — LLM unavailable).*
"""


def _user_prompt(telemetry: dict[str, Any]) -> str:
    tchp = _safe_float(telemetry.get("tchp"))
    band, rule = classify_tchp(tchp)
    inversion = _bool(telemetry.get("inversion_flag", False))
    return (
        "Issue the INCOIS Operational Maritime Advisory Bulletin from this telemetry. "
        "Copy every number exactly. Do not add any number that is not listed.\n\n"
        f"- Latitude: {_num(telemetry.get('lat'), 2)} °N\n"
        f"- Longitude: {_num(telemetry.get('lon'), 2)} °E\n"
        f"- Date: {_iso(telemetry.get('date', 'unknown'))}\n"
        f"- SST: {_num(telemetry.get('sst'), 2)} °C\n"
        f"- TCHP: {_num(telemetry.get('tchp'), 1)} kJ/cm²\n"
        f"- TCHP class (pre-applied rule): {band} — {rule}\n"
        f"- MLD: {_num(telemetry.get('mld'), 1)} m\n"
        f"- Z20: {_num(telemetry.get('z20'), 1)} m\n"
        f"- Inversion Flag: {inversion}\n"
        f"- Uncertainty (potential temperature): {_num(telemetry.get('uncertainty'), 2)} °C\n"
    )


def _call_gemini(api_key: str, user_prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.25,
                    max_output_tokens=1400,
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 — fall through to next model / provider
            last_error = exc
            logger.warning("Gemini model %s failed: %s", model, exc)
    if last_error:
        raise last_error
    raise RuntimeError("Gemini returned an empty advisory")


def _call_openai(api_key: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.25,
        max_tokens=1400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty advisory")
    return text


def _hydrate_keys_from_settings() -> None:
    try:
        from .config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001
        return
    if settings.gemini_api_key and not _api_key("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key.strip()
    if settings.openai_api_key and not _api_key("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key.strip()


def generate_ocean_advisory(telemetry: dict[str, Any]) -> str:
    """Return a 4-section INCOIS markdown bulletin for the given telemetry."""
    _load_local_env()
    _hydrate_keys_from_settings()
    prompt = _user_prompt(telemetry)

    gemini_key = _api_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            return _call_gemini(gemini_key, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini advisory failed, trying next provider: %s", exc)

    openai_key = _api_key("OPENAI_API_KEY")
    if openai_key:
        try:
            return _call_openai(openai_key, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI advisory failed, using offline template: %s", exc)

    return fallback_advisory(telemetry)


def advisory_source() -> str:
    """Which generator will be attempted first given current environment."""
    _load_local_env()
    _hydrate_keys_from_settings()
    if _api_key("GEMINI_API_KEY"):
        return "gemini"
    if _api_key("OPENAI_API_KEY"):
        return "openai"
    return "offline"
