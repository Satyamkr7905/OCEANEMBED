import type { OceanProfile } from "@/types/ocean";

/** UNESCO EOS-80 density (kg m-3) at atmospheric pressure, then σ_θ = ρ − 1000. */
export function sigmaTheta(thetaC: number, sp: number): number {
  const T = thetaC;
  const S = sp;
  const rhoW =
    999.842594 +
    6.793952e-2 * T -
    9.09529e-3 * T ** 2 +
    1.001685e-4 * T ** 3 -
    1.120083e-6 * T ** 4 +
    6.536332e-9 * T ** 5;
  const A =
    0.824493 -
    4.0899e-3 * T +
    7.6438e-5 * T ** 2 -
    8.2467e-7 * T ** 3 +
    5.3875e-9 * T ** 4;
  const B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T ** 2;
  const C = 4.8314e-4;
  const rho = rhoW + S * A + S ** 1.5 * B + C * S ** 2;
  return rho - 1000;
}

export function interpolateAt(depths: number[], values: number[], z: number): number | null {
  if (depths.length === 0 || values.length === 0) return null;
  if (z <= depths[0]) return values[0];
  if (z >= depths[depths.length - 1]) return values[values.length - 1];
  for (let i = 0; i < depths.length - 1; i += 1) {
    if (z >= depths[i] && z <= depths[i + 1]) {
      const span = depths[i + 1] - depths[i];
      const t = span === 0 ? 0 : (z - depths[i]) / span;
      return values[i] + t * (values[i + 1] - values[i]);
    }
  }
  return null;
}

export function nearestIndex(depths: number[], z: number): number {
  let best = 0;
  let dist = Number.POSITIVE_INFINITY;
  depths.forEach((d, i) => {
    const delta = Math.abs(d - z);
    if (delta < dist) {
      dist = delta;
      best = i;
    }
  });
  return best;
}

export interface GradientSample {
  zMid: number;
  dTdz: number;
}

/** Finite-difference dT/dz (°C m-1). Negative values mean cooling with depth. */
export function temperatureGradient(depths: number[], theta: number[]): GradientSample[] {
  const out: GradientSample[] = [];
  for (let i = 0; i < depths.length - 1; i += 1) {
    const dz = depths[i + 1] - depths[i];
    if (dz <= 0) continue;
    out.push({
      zMid: (depths[i] + depths[i + 1]) / 2,
      dTdz: (theta[i + 1] - theta[i]) / dz,
    });
  }
  return out;
}

/** Isothermal layer depth: first depth where T ≤ T_surface − ΔT (default 0.5 °C). */
export function isothermalLayerDepth(depths: number[], theta: number[], deltaT = 0.5): number {
  if (theta.length === 0) return 0;
  const surface = theta[0];
  for (let i = 1; i < depths.length; i += 1) {
    if (theta[i] <= surface - deltaT) {
      const prev = depths[i - 1];
      const span = depths[i] - prev;
      const drop = theta[i - 1] - theta[i];
      if (drop <= 0) return depths[i];
      const frac = (theta[i - 1] - (surface - deltaT)) / drop;
      return prev + span * Math.min(1, Math.max(0, frac));
    }
  }
  return depths[depths.length - 1] ?? 0;
}

const RHO = 1025;
const CP = 3985;
const KJ_CM2 = 1e7;

export interface HeatLayer {
  key: string;
  label: string;
  z0: number;
  z1: number;
  kjcm2: number;
}

export const TCHP_LAYERS: Array<Pick<HeatLayer, "key" | "label" | "z0" | "z1">> = [
  { key: "0-50", label: "0–50 m", z0: 0, z1: 50 },
  { key: "50-100", label: "50–100 m", z0: 50, z1: 100 },
  { key: "100-200", label: "100–200 m", z0: 100, z1: 200 },
  { key: "200-500", label: "200–500 m", z0: 200, z1: 500 },
];

function thetaAt(depths: number[], theta: number[], z: number): number {
  return interpolateAt(depths, theta, z) ?? theta[0] ?? 0;
}

/** Layer heat relative to 26 °C, same units as TCHP (kJ cm-2). Only T > 26 contributes. */
export function layerHeatContent(depths: number[], theta: number[], z0: number, z1: number): number {
  const nodes = [z0, ...depths.filter((z) => z > z0 && z < z1), z1];
  let acc = 0;
  for (let i = 0; i < nodes.length - 1; i += 1) {
    const za = nodes[i];
    const zb = nodes[i + 1];
    const ta = thetaAt(depths, theta, za);
    const tb = thetaAt(depths, theta, zb);
    const ha = Math.max(0, ta - 26);
    const hb = Math.max(0, tb - 26);
    acc += 0.5 * (ha + hb) * (zb - za);
  }
  return (RHO * CP * acc) / KJ_CM2;
}

export function tchpLayerBreakdown(profile: OceanProfile): HeatLayer[] {
  return TCHP_LAYERS.map((layer) => ({
    ...layer,
    kjcm2: layerHeatContent(profile.depths, profile.potentialTemperature, layer.z0, layer.z1),
  }));
}

export function derivedSounding(profile: OceanProfile) {
  const sigma = profile.depths.map((_, i) =>
    sigmaTheta(profile.potentialTemperature[i], profile.practicalSalinity[i]),
  );
  const gradient = temperatureGradient(profile.depths, profile.potentialTemperature);
  const ild = isothermalLayerDepth(profile.depths, profile.potentialTemperature);
  const blt = profile.blt != null && profile.blt > 0 ? profile.blt : Math.max(0, ild - profile.mld);
  const core = gradient.reduce<GradientSample | null>((best, sample) => {
    if (!best || sample.dTdz < best.dTdz) return sample;
    return best;
  }, null);
  return {
    sigma,
    gradient,
    ild,
    blt,
    thermoclineCore: core,
    layers: tchpLayerBreakdown(profile),
  };
}

export function sampleAtDepth(profile: OceanProfile, z: number) {
  const i = nearestIndex(profile.depths, z);
  const theta = interpolateAt(profile.depths, profile.potentialTemperature, z) ?? profile.potentialTemperature[i];
  const sp = interpolateAt(profile.depths, profile.practicalSalinity, z) ?? profile.practicalSalinity[i];
  const unc = interpolateAt(profile.depths, profile.uncertaintyTheta, z) ?? profile.uncertaintyTheta[i];
  return {
    z,
    theta,
    sp,
    sigma: sigmaTheta(theta, sp),
    uncertainty: unc,
  };
}
