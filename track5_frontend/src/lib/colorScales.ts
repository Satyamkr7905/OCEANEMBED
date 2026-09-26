/** cmocean-inspired palettes for NIO operational layers. */

export type RGBA = [number, number, number, number];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function sampleStops(stops: number[][], t: number): RGBA {
  const x = Math.min(1, Math.max(0, t));
  const scaled = x * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const c0 = stops[i];
  const c1 = stops[i + 1];
  return [
    Math.round(lerp(c0[0], c1[0], f)),
    Math.round(lerp(c0[1], c1[1], f)),
    Math.round(lerp(c0[2], c1[2], f)),
    255,
  ];
}

/** Sequential blues matching the 10% → 100% chart ramp. */
const BLUES = [
  [236, 244, 252],
  [214, 230, 246],
  [184, 212, 238],
  [146, 192, 230],
  [102, 166, 220],
  [62, 140, 208],
  [40, 112, 190],
  [28, 84, 166],
  [20, 60, 132],
  [12, 36, 86],
];

export const SCALES: Record<string, { stops: number[][]; vmin: number; vmax: number; units: string }> = {
  sst: { stops: BLUES, vmin: 24, vmax: 32, units: "°C" },
  sss: { stops: BLUES, vmin: 31, vmax: 37, units: "psu" },
  sla: { stops: BLUES, vmin: -0.25, vmax: 0.25, units: "m" },
  winds: { stops: BLUES, vmin: 0, vmax: 18, units: "m s⁻¹" },
  theta: { stops: BLUES, vmin: 8, vmax: 31, units: "°C" },
  sp: { stops: BLUES, vmin: 32, vmax: 36.5, units: "psu" },
  sigma_theta: { stops: BLUES, vmin: 0, vmax: 1.2, units: "°C" },
  tchp: { stops: BLUES, vmin: 20, vmax: 140, units: "kJ cm⁻²" },
  mld: { stops: BLUES, vmin: 10, vmax: 80, units: "m" },
  z20: { stops: BLUES, vmin: 40, vmax: 180, units: "m" },
  blt: { stops: BLUES, vmin: 0, vmax: 40, units: "m" },
  cip: { stops: BLUES, vmin: 0, vmax: 1, units: "" },
};

export function colorFor(layer: string, value: number | null | undefined): RGBA {
  if (value === null || value === undefined || Number.isNaN(value)) return [0, 0, 0, 0];
  const scale = SCALES[layer] ?? SCALES.theta;
  const t = (value - scale.vmin) / (scale.vmax - scale.vmin);
  return sampleStops(scale.stops, t);
}

export function legendGradient(layer: string): string {
  const stops = (SCALES[layer] ?? SCALES.theta).stops;
  const last = Math.max(1, stops.length - 1);
  const parts = stops.map((c, i) => `rgb(${c[0]}, ${c[1]}, ${c[2]}) ${Math.round((i / last) * 100)}%`);
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

export function legendTicks(layer: string): { vmin: number; vmax: number; units: string } {
  const s = SCALES[layer] ?? SCALES.theta;
  return { vmin: s.vmin, vmax: s.vmax, units: s.units };
}

export function gridToImageData(values: Array<Array<number | null>>, layer: string): ImageData {
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const img = new ImageData(cols, rows);
  for (let j = 0; j < rows; j += 1) {
    const srcRow = values[rows - 1 - j];
    for (let i = 0; i < cols; i += 1) {
      const [r, g, b, a] = colorFor(layer, srcRow[i]);
      const o = (j * cols + i) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = a;
    }
  }
  return img;
}

export function imageDataToDataUrl(img: ImageData): string {
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}
