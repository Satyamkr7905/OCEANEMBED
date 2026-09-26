"""Numerical tests for Track 3 diagnostics, matchers, and skill scores.

Run::

    python -m track3_validation_engine.test_diagnostics
    pytest track3_validation_engine/test_diagnostics.py
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from track3_validation_engine.benchmarks.generate_report import write_report
from track3_validation_engine.benchmarks.skill_scores import (
    climatology_skill_score,
    cube_skill,
    depthwise_skill,
    inversion_brier,
    mse,
    pearson_r,
    rmse,
)
from track3_validation_engine.constants import (
    ARGO_MIN_DEPTH_M,
    CP_TCHP,
    KJ_CM2_PER_J_M2,
    RAMA_SENSOR_DEPTHS,
    STANDARD_DEPTHS,
    z_1m,
)
from track3_validation_engine.diagnostics.indices import blt_m, ild_m, mld_m, tchp_kj_cm2, z20_depth
from track3_validation_engine.diagnostics.pchip_profiler import pchip_sample, pchip_to_z
from track3_validation_engine.validation.argo_matcher import ArgoProfile, match_argo_to_model, match_surface_to_oisst
from track3_validation_engine.validation.rama_matcher import RamaRecord, match_rama_to_model

try:
    import gsw
except ImportError:
    gsw = None


class PchipTests(unittest.TestCase):
    def test_linear_profile_recovered_at_1m(self) -> None:
        z = STANDARD_DEPTHS
        t = 30.0 - 0.02 * z
        t_1m = pchip_to_z(t[:, None], z_src=z, depth_axis=0)[:, 0]
        zq = z_1m()
        expected = 30.0 - 0.02 * zq
        finite = zq <= z[-1]
        np.testing.assert_allclose(t_1m[finite], expected[finite], atol=1e-3)

    def test_monotonic_decreasing_stays_monotonic(self) -> None:
        z = STANDARD_DEPTHS
        t = 30.0 - 0.025 * z
        t_1m = pchip_to_z(t[:, None], depth_axis=0)[:, 0]
        self.assertTrue(np.all(np.diff(t_1m[np.isfinite(t_1m)]) <= 1e-9))

    def test_rama_sensor_sampling(self) -> None:
        z = STANDARD_DEPTHS
        t = 28.0 - 0.01 * z
        sampled = pchip_sample(t[:, None], z, RAMA_SENSOR_DEPTHS, depth_axis=0).reshape(-1)
        np.testing.assert_allclose(sampled, 28.0 - 0.01 * RAMA_SENSOR_DEPTHS, atol=1e-3)


class IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.z = z_1m()
        self.ct = (30.0 - 0.02 * self.z)[:, None]
        self.rho = np.full_like(self.ct, 1025.0)

    def test_z20_and_d26(self) -> None:
        z20 = z20_depth(self.ct, self.z)
        np.testing.assert_allclose(z20, [500.0], atol=1.0)

    def test_tchp_matches_analytic_integral(self) -> None:
        # Θ = 30 - 0.02 z  →  26 °C at 200 m. ρ = 1025 constant.
        tchp = tchp_kj_cm2(self.ct, self.rho, self.z)
        energy = 1025.0 * CP_TCHP * 400.0  # ∫_0^200 (4 - 0.02z) dz = 400
        expected = energy * KJ_CM2_PER_J_M2
        np.testing.assert_allclose(tchp, [expected], rtol=0.02)

    def test_mld_linear_density(self) -> None:
        rho = 1024.0 + 0.001 * np.maximum(self.z - 10.0, 0.0)
        mld = mld_m(rho[:, None], self.z)
        np.testing.assert_allclose(mld, [40.0], atol=1.0)

    def test_ild(self) -> None:
        ild = ild_m(self.ct, self.z)
        # T(10)=29.8, target 29.6 → z = 20 m
        np.testing.assert_allclose(ild, [20.0], atol=1.0)

    def test_blt_requires_ild_deeper_than_mld(self) -> None:
        ct = np.full((self.z.size, 1), 29.0)
        ct[self.z > 80] = 29.0 - 0.05 * (self.z[self.z > 80][:, None] - 80.0)
        rho = 1024.0 + 0.003 * np.maximum(self.z - 10.0, 0.0)
        blt = blt_m(ct, rho[:, None], self.z)
        self.assertTrue(np.isfinite(blt[0]))
        self.assertGreater(float(blt[0]), 0.0)


class SkillTests(unittest.TestCase):
    def test_perfect_model_css_is_one(self) -> None:
        obs = np.array([1.0, 2.0, 3.0])
        pred = obs.copy()
        clim = np.array([0.0, 0.0, 0.0])
        css = climatology_skill_score(mse(pred, obs), mse(clim, obs))
        self.assertAlmostEqual(css, 1.0)

    def test_climatology_predictor_css_is_zero(self) -> None:
        obs = np.array([1.0, 3.0, 5.0])
        clim = np.full_like(obs, obs.mean())
        css = climatology_skill_score(mse(clim, obs), mse(clim, obs))
        self.assertAlmostEqual(css, 0.0)

    def test_rmse_and_pearson(self) -> None:
        x = np.linspace(0, 1, 50)
        self.assertAlmostEqual(rmse(x, x), 0.0)
        self.assertAlmostEqual(pearson_r(x, 2 * x + 1), 1.0, places=6)

    def test_inversion_brier_perfect(self) -> None:
        z = STANDARD_DEPTHS
        t = (30 - 0.02 * z)[:, None]
        self.assertAlmostEqual(inversion_brier(t, t), 0.0)

    def test_cube_skill_depth_axis(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(15, 8, 8))
        pred = truth + 0.1
        report = cube_skill(pred, truth, name="toy")
        self.assertEqual(len(report.rmse), 15)
        self.assertGreater(report.rmse_all, 0.0)


class ArgoGuardTests(unittest.TestCase):
    def test_rejects_samples_shallower_than_5m(self) -> None:
        z = STANDARD_DEPTHS
        theta = np.broadcast_to((28 - 0.01 * z)[:, None, None], (15, 4, 4)).copy()
        sp = np.full_like(theta, 34.5)
        times = np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]").astype("datetime64[ns]")
        theta_t = np.stack([theta, theta], axis=0)
        sp_t = np.stack([sp, sp], axis=0)
        lat = 5.125 + 0.25 * np.arange(4)
        lon = 45.125 + 0.25 * np.arange(4)
        prof = ArgoProfile(
            time=np.datetime64("2020-01-01T06:00:00"),
            lat=float(lat[1]),
            lon=float(lon[1]),
            depth=np.array([0.0, 2.0, 10.0, 20.0, 50.0]),
            temp=np.array([29.0, 28.9, 28.0, 27.8, 27.2]),
            psal=np.array([33.0, 33.1, 34.0, 34.2, 34.5]),
            source="synthetic",
        )
        table = match_argo_to_model(
            [prof],
            theta_t,
            sp_t,
            times,
            lat=lat,
            lon=lon,
        )
        self.assertTrue((table.rows["depth"] >= ARGO_MIN_DEPTH_M).all())
        self.assertIn("dt_0d", set(table.rows["dt_bin"]))
        self.assertGreater(table.notes["skipped_shallow_samples"], 0)

    def test_surface_oisst_path(self) -> None:
        model0 = np.array([28.0, 28.2, 27.5])
        oisst = np.array([28.1, 28.0, 27.6])
        report = match_surface_to_oisst(model0, oisst)
        self.assertEqual(report.depths[0], 0.0)
        self.assertGreater(report.n[0], 0)


class RamaTests(unittest.TestCase):
    def test_model_interpolated_to_sensor_depths(self) -> None:
        z = STANDARD_DEPTHS
        lat = 5.125 + 0.25 * np.arange(8)
        lon = 45.125 + 0.25 * np.arange(8)
        iy = int(np.argmin(np.abs(lat - 15.0)))
        ix = int(np.argmin(np.abs(lon - 90.0)))
        # 15N 90E is outside this tiny 8x8 patch; place the mooring on-grid instead.
        site_lat, site_lon = float(lat[2]), float(lon[3])
        theta = np.broadcast_to((29 - 0.012 * z)[:, None, None], (15, 8, 8)).copy()
        sp = np.full_like(theta, 34.8)
        times = np.array(["2020-06-01T00:00:00"], dtype="datetime64[ns]")
        theta_t = theta[None, ...]
        sp_t = sp[None, ...]
        rama_z = RAMA_SENSOR_DEPTHS
        rec = RamaRecord(
            site="toy",
            lat=site_lat,
            lon=site_lon,
            time=times,
            depth=rama_z,
            temp=(29 - 0.012 * rama_z)[None, :],
            psal=np.full((1, rama_z.size), 34.8),
        )
        match = match_rama_to_model(rec, theta_t, sp_t, times, lat=lat, lon=lon)
        self.assertFalse(match.model_on_rama.empty)
        np.testing.assert_array_equal(sorted(match.model_on_rama["depth"].unique()), sorted(rama_z))
        err = np.nanmean(np.abs(match.model_on_rama["temp_model"] - match.model_on_rama["temp_rama"]))
        self.assertLess(err, 0.05)
        self.assertIn("rama_to_standard", set(match.rama_on_standard["direction"]))


class ReportTests(unittest.TestCase):
    def test_json_and_markdown_written(self) -> None:
        obs = np.linspace(1, 15, 15)
        pred = obs + 0.2
        report = depthwise_skill(pred, obs, STANDARD_DEPTHS, name="demo")
        with tempfile.TemporaryDirectory() as tmp:
            jp, mp = write_report({"theta": report}, tmp, stem="demo")
            self.assertTrue(jp.exists() and mp.exists())
            payload = json.loads(Path(jp).read_text(encoding="utf-8"))
            self.assertIn("results", payload)
            self.assertIn("demo", Path(mp).read_text(encoding="utf-8"))


@unittest.skipIf(gsw is None, "gsw is not installed")
class Teos10Tests(unittest.TestCase):
    def test_vectorized_matches_scalar_and_budget(self) -> None:
        from track3_validation_engine.diagnostics.teos10_vectorized import convert_theta_sp

        z = STANDARD_DEPTHS
        nlat, nlon = 20, 24
        lat = 5.125 + 0.25 * np.arange(nlat)
        lon = 45.125 + 0.25 * np.arange(nlon)
        theta = np.broadcast_to((28.0 - 0.015 * z)[:, None, None], (15, nlat, nlon)).copy()
        sp = np.full_like(theta, 34.5)
        fields = convert_theta_sp(theta, sp, lat, lon, z)
        self.assertEqual(fields.sa.shape, theta.shape)
        self.assertTrue(np.isfinite(fields.rho).mean() > 0.9)
        iy, ix = 3, 5
        p = gsw.p_from_z(-z, np.full_like(z, lat[iy]))
        sa = gsw.SA_from_SP(sp[:, iy, ix], p, np.full_like(z, lon[ix]), np.full_like(z, lat[iy]))
        ct = gsw.CT_from_pt(sa, theta[:, iy, ix])
        rho = gsw.rho(sa, ct, p)
        np.testing.assert_allclose(fields.sa[:, iy, ix], sa, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(fields.ct[:, iy, ix], ct, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(fields.rho[:, iy, ix], rho, rtol=1e-10, atol=1e-10)
        n_cols = nlat * nlon
        scaled = fields.elapsed_s * (24_000 / n_cols)
        self.assertLess(scaled, 2.0, msg=f"scaled runtime {scaled:.3f}s exceeds loose 2 s CI bound")


def main() -> None:
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
