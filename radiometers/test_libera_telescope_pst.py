"""Regression tests for :mod:`libera_telescope_pst`.

Run with::

    pytest test_libera_telescope_pst.py            # fast tests (slow deselected)
    pytest test_libera_telescope_pst.py -m slow    # only the FFT compute
    pytest test_libera_telescope_pst.py -m ''      # everything

``addopts`` in pytest.ini is what deselects the slow test by default; ``-m`` on
the command line overrides it.

Most tests use a small hand-built ``DiffractionProfile`` so they never trigger
the (~1-4 s) FFT. The single test that validates the real numerical calculation
is marked ``slow``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import libera_telescope_pst as m


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_profile() -> m.DiffractionProfile:
    """A tiny, self-consistent diffraction profile (no FFT required).

    ``u`` is built as ``sin(angle) / lambda_ref`` so that rescaling back to the
    reference wavelength must recover the original angles.
    """
    angle = np.array([0.0, 0.5, 1.0, 5.0, 30.0, 60.0])
    diff = np.array([1.0, 5e-1, 1e-1, 1e-2, 1e-5, 1e-8])
    ref_um = m._DIFF_REF_WAVELENGTH_UM
    u = np.sin(np.radians(angle)) / (ref_um * 1e-6)
    return m.DiffractionProfile(angle_deg=angle, u=u, diff_signal=diff,
                                ref_wavelength_um=ref_um)


@pytest.fixture
def patched_profile(monkeypatch, synthetic_profile) -> m.DiffractionProfile:
    """Make the model use the synthetic profile instead of the real cache.

    ``get_diffraction_profile`` reads its path from config.toml, so the model
    tests stub it out rather than pointing it at a temporary file. The static
    term cache is cleared too, since it would otherwise hold entries built from
    the real profile.
    """
    monkeypatch.setattr(m, "get_diffraction_profile", lambda: synthetic_profile)
    m._STATIC_TERM_CACHE.clear()
    yield synthetic_profile
    m._STATIC_TERM_CACHE.clear()


# -----------------------------------------------------------------------------
# Persistence round-trip
# -----------------------------------------------------------------------------
def test_cache_roundtrip(tmp_path, synthetic_profile):
    path = tmp_path / "roundtrip.npz"
    m.save_diffraction_profile(synthetic_profile, path)
    loaded = m._load_diffraction_profile(path)
    np.testing.assert_allclose(loaded.angle_deg, synthetic_profile.angle_deg)
    np.testing.assert_allclose(loaded.u, synthetic_profile.u)
    np.testing.assert_allclose(loaded.diff_signal, synthetic_profile.diff_signal)
    assert loaded.ref_wavelength_um == synthetic_profile.ref_wavelength_um


# -----------------------------------------------------------------------------
# Wavelength scaling of the diffraction pattern
# -----------------------------------------------------------------------------
def test_scale_at_reference_wavelength_recovers_angles(synthetic_profile):
    angle, diff = m.scale_diffraction_to_wavelength(
        synthetic_profile, synthetic_profile.ref_wavelength_um)
    # last point is the flat 180 deg anchor; everything before must match input
    assert angle[-1] == pytest.approx(180.0)
    assert diff[-1] == pytest.approx(synthetic_profile.diff_signal[-1])
    np.testing.assert_allclose(angle[:-1], synthetic_profile.angle_deg, atol=1e-9)
    np.testing.assert_allclose(diff[:-1], synthetic_profile.diff_signal)


def test_scale_longer_wavelength_stretches_and_drops(synthetic_profile):
    # At 50 um, u*lambda = 2.5*sin(angle); points with 2.5*sin(angle) > 1
    # (angle >= ~23.6 deg -> the 30 and 60 deg entries) have no valid angle.
    angle, diff = m.scale_diffraction_to_wavelength(synthetic_profile, 50.0)
    interior = angle[:-1]  # drop the 180 deg anchor
    # two tail points dropped from the original six
    assert interior.size == synthetic_profile.angle_deg.size - 2
    # ascending, and each surviving small angle is pushed outward vs 20 um
    assert np.all(np.diff(angle) > 0)
    assert interior[2] > synthetic_profile.angle_deg[2]  # the 1 deg point moved out


# -----------------------------------------------------------------------------
# Geometric illumination factor + clamp
# -----------------------------------------------------------------------------
def test_geometric_on_axis_is_unity():
    assert m.geometric_model(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-9)


def test_geometric_clamp_bounds_result():
    angles = np.linspace(0.0, 90.0, 400)
    clamped = m.geometric_model(angles, clamp=True)
    raw = m.geometric_model(angles, clamp=False)
    # np.interp holds the boundary value, so nothing goes negative either way
    assert raw.min() >= 0.0
    # clamp is exactly a clip to [0, 1] ...
    assert np.all(clamped >= 0.0) and np.all(clamped <= 1.0)
    np.testing.assert_allclose(clamped, np.clip(raw, 0.0, 1.0))
    # ... and it does something: the raw curve overshoots 1 near the axis
    assert raw.max() > 1.0


# -----------------------------------------------------------------------------
# Full PST model
# -----------------------------------------------------------------------------
COEFFS = dict(a0=-14.39, a1=-2.53, a2=-0.37, lp0=-10.58, lp1=-3.14)
# gc0 is the secondary's effective grating spacing [um]; the ring is only
# defined where wavelength < gc0, so the ring tests use SW-like wavelengths
# rather than the 20 um used for the scatter-only tests.
RING = dict(gc0=5.410, gw0=-0.5, gw1=0.2, ga0=-8.0, ga1=-0.5)
RING_WL = 1.0


def test_pst_two_components_finite_and_nonnegative(patched_profile):
    angles = np.logspace(-2, np.log10(90.0), 200)
    res = m.pst_two_components(angles, 20.0, **COEFFS)
    assert np.all(np.isfinite(res.pst))
    assert np.all(res.pst >= 0.0)


def test_pst_two_components_sum_to_total(patched_profile):
    angles = np.logspace(-2, 1, 150)
    res = m.pst_two_components(angles, 20.0, **COEFFS)
    np.testing.assert_allclose(res.pst, res.surf + res.part + res.diff)


def test_pst_two_components_broadcasts_wavelength(patched_profile):
    # Parallel arrays, one entry per measurement, is the whole-dataset fit case.
    angles = np.array([2.0, 4.0, 8.0, 2.0])
    wavelengths = np.array([0.5, 1.0, 2.0, 4.0])
    res = m.pst_two_components(angles, wavelengths, **COEFFS)
    assert res.pst.shape == angles.shape
    # Each point must equal the same model evaluated one point at a time.
    for i, (a, w) in enumerate(zip(angles, wavelengths)):
        one = m.pst_two_components(a, w, **COEFFS)
        assert res.pst[i] == pytest.approx(float(one.pst))


def test_pst_two_components_geo_applies_to_scatter_not_diffraction(patched_profile):
    angles = np.linspace(0.1, 20.0, 50)
    res = m.pst_two_components(angles, 20.0, **COEFFS)
    assert np.all(res.geo_factor >= 0.0) and np.all(res.geo_factor <= 1.0)
    # diff is the raw interpolated profile, untouched by geo
    prof_angle, prof_diff = m.scale_diffraction_to_wavelength(patched_profile, 20.0)
    np.testing.assert_allclose(res.diff, np.interp(angles, prof_angle, prof_diff))


def test_pst_three_components_adds_ring_outside_geo(patched_profile):
    angles = np.linspace(0.1, 20.0, 800)
    two = m.pst_two_components(angles, RING_WL, **COEFFS)
    three = m.pst_three_components(angles, RING_WL, **COEFFS, **RING)
    # Ring is added outside the geometric factor, like diffraction
    np.testing.assert_allclose(three.pst, two.pst + three.ring)
    np.testing.assert_allclose(three.surf, two.surf)
    np.testing.assert_allclose(three.part, two.part)


def test_pst_three_components_ring_peaks_at_diffraction_angle(patched_profile):
    angles = np.linspace(0.1, 20.0, 2000)
    res = m.pst_three_components(angles, RING_WL, **COEFFS, **RING)
    # Centre follows first-order diffraction off the secondary, asin(lam/gc0)
    expected_centre = np.degrees(np.arcsin(RING_WL / RING["gc0"]))
    expected_amp = np.exp(RING["ga0"]
                          + RING["ga1"]*(RING_WL - m.RING_PIVOT_UM))
    assert res.ring.max() == pytest.approx(expected_amp, rel=1e-3)
    assert angles[np.argmax(res.ring)] == pytest.approx(expected_centre, abs=0.05)


def test_pst_three_components_ring_moves_out_with_wavelength(patched_profile):
    angles = np.linspace(0.1, 25.0, 3000)
    centres = []
    for wl in (0.4, 1.0, 2.0):
        res = m.pst_three_components(angles, wl, **COEFFS, **RING)
        centres.append(angles[np.argmax(res.ring)])
    assert centres[0] < centres[1] < centres[2]


@pytest.mark.parametrize("wavelength_um", [0.3, 1.0, 1.152, 2.0, 5.0])
def test_pst_three_components_width_stays_positive(patched_profile, wavelength_um):
    # The old linear width crossed zero at gw0/gw1 (1.15 um for the original
    # 0.622/0.540 pair), dividing by zero there and flipping sign beyond it.
    # The log-space form cannot reach zero at any wavelength.
    width = np.exp(RING["gw0"]
                   - RING["gw1"]*(wavelength_um - m.RING_PIVOT_UM))
    assert width > 0.0

    angles = np.linspace(0.1, 25.0, 500)
    res = m.pst_three_components(angles, wavelength_um, **COEFFS, **RING)
    assert np.all(np.isfinite(res.ring))
    assert np.all(res.ring >= 0.0)
    assert np.all(np.isfinite(res.pst))


@pytest.mark.parametrize("wavelength_um", [5.410, 6.0, 20.0, 100.0])
def test_pst_three_components_no_ring_beyond_grating_cutoff(patched_profile,
                                                            wavelength_um):
    # asin(lam/gc0) has no solution for lam >= gc0. The artifact is gone by
    # then, so the ring must switch off cleanly instead of returning NaN.
    angles = np.linspace(0.1, 25.0, 500)
    three = m.pst_three_components(angles, wavelength_um, **COEFFS, **RING)
    two = m.pst_two_components(angles, wavelength_um, **COEFFS)
    assert np.all(three.ring == 0.0)
    assert np.all(np.isfinite(three.pst))
    np.testing.assert_allclose(three.pst, two.pst)


def test_pst_three_components_cutoff_is_per_point(patched_profile):
    # A dataset spanning the cutoff must keep its ring below gc0 and drop it
    # above, rather than the whole array going NaN together.
    angles = np.full(4, 10.0)
    wavelengths = np.array([1.0, 2.0, RING["gc0"], 20.0])
    res = m.pst_three_components(angles, wavelengths, **COEFFS, **RING)
    assert np.all(np.isfinite(res.pst))
    assert res.ring[0] > 0.0 and res.ring[1] > 0.0
    assert res.ring[2] == 0.0 and res.ring[3] == 0.0


def test_pst_three_components_no_warnings_beyond_cutoff(patched_profile):
    # The clip before arcsin is what keeps this from emitting an invalid-value
    # RuntimeWarning on the out-of-range points.
    angles = np.linspace(0.1, 25.0, 200)
    wavelengths = np.linspace(0.4, 30.0, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m.pst_three_components(angles, wavelengths, **COEFFS, **RING)


def test_pst_three_components_width_shrinks_with_wavelength(patched_profile):
    # gw1 > 0 means the ring narrows as wavelength grows; measure the FWHM
    # directly off the evaluated ring rather than trusting the formula.
    angles = np.linspace(0.1, 25.0, 20000)
    widths = []
    for wl in (0.4, 1.0, 2.0):
        ring = m.pst_three_components(angles, wl, **COEFFS, **RING).ring
        above = angles[ring >= 0.5 * ring.max()]
        widths.append(above.max() - above.min())
    assert widths[0] > widths[1] > widths[2]


# -----------------------------------------------------------------------------
# Surface roughness from the integrated PSD
# -----------------------------------------------------------------------------
def _roughness_by_quadrature(a0, a1, a2, f_min, f_max, solid_angle_sr):
    """sigma**2 by direct numerical integration, to check the closed form."""
    from scipy.integrate import quad
    f0 = m.PSD_PIVOT_FREQ

    def integrand(f):
        log_f = np.log(f / f0)
        return 2.0 * np.pi * f * np.exp(a0 + a1*log_f + a2*log_f**2)

    return quad(integrand, f_min, f_max, limit=200)[0] / solid_angle_sr


# A representative band; each channel's real band comes from parts.f in the fit.
BAND = dict(f_min=0.0147, f_max=0.4837)


@pytest.mark.parametrize("a0,a1,a2", [
    (-14.37, -2.675, -0.507),   # Total channel global fit
    (-14.39, -2.530, -0.370),   # the model's default start values
    (-12.00, -2.000, 0.0),      # a2 == 0 and b == a1+2 == 0 together
    (-13.00, -4.000, -0.10),    # weak curvature
    (-10.00, 0.500, -0.90),     # positive slope, strong curvature
])
def test_roughness_closed_form_matches_quadrature(a0, a1, a2):
    res = m.mirror_roughness_from_psd(a0, a1, a2, **BAND, solid_angle_sr=1.0)
    expected = _roughness_by_quadrature(a0, a1, a2, res.f_min, res.f_max, 1.0)
    assert res.psd_integral == pytest.approx(expected, rel=1e-8)
    # sigma is self-consistent with the integral it reports (quadrature itself
    # is only good to ~1e-8, so it cannot be the reference for this one).
    assert res.sigma_um == pytest.approx(np.sqrt(res.psd_integral), rel=1e-12)
    assert res.sigma_nm == pytest.approx(res.sigma_um * 1e3, rel=1e-12)
    # the band it reports back is the one it was given
    assert (res.f_min, res.f_max) == (BAND["f_min"], BAND["f_max"])


def test_roughness_scales_as_inverse_sqrt_solid_angle():
    coeffs = (-14.37, -2.675, -0.507)
    base = m.mirror_roughness_from_psd(*coeffs, **BAND, solid_angle_sr=1.0).sigma_nm
    for omega in (1e-1, 1e-2, 1e-3):
        s = m.mirror_roughness_from_psd(*coeffs, **BAND,
                                        solid_angle_sr=omega).sigma_nm
        assert s / base == pytest.approx(1.0 / np.sqrt(omega), rel=1e-12)


def test_roughness_band_widening_converges_for_negative_curvature():
    # a2 < 0 makes the integrand fall off at both ends, so the result is only
    # weakly sensitive to the band and converges as it widens. This is what lets
    # different channels' bands be compared against each other at all.
    coeffs = (-14.37, -2.675, -0.507)
    band = m.mirror_roughness_from_psd(*coeffs, **BAND, solid_angle_sr=1.0).sigma_nm
    wide = m.mirror_roughness_from_psd(*coeffs, solid_angle_sr=1.0,
                                       f_min=1e-6, f_max=1e4).sigma_nm
    assert wide > band                    # widening can only add power
    assert wide / band < 1.15             # but only by ~9% here


def test_roughness_band_is_required():
    coeffs = (-14.37, -2.675, -0.507)
    with pytest.raises(TypeError):
        m.mirror_roughness_from_psd(*coeffs)
    with pytest.raises(TypeError):
        m.mirror_roughness_from_psd(*coeffs, f_min=0.0147)


def test_roughness_defaults_to_erf_solid_angle():
    coeffs = (-14.37, -2.675, -0.507)
    default = m.mirror_roughness_from_psd(*coeffs, **BAND)
    explicit = m.mirror_roughness_from_psd(
        *coeffs, **BAND, solid_angle_sr=m.ERF_RECEIVER_SOLID_ANGLE_SR)
    assert default.solid_angle_sr == m.ERF_RECEIVER_SOLID_ANGLE_SR
    assert default.sigma_nm == pytest.approx(explicit.sigma_nm, rel=1e-15)


def test_roughness_rejects_bad_inputs():
    with pytest.raises(ValueError, match="solid_angle_sr"):
        m.mirror_roughness_from_psd(-14.0, -2.6, -0.5, **BAND,
                                    solid_angle_sr=0.0)
    with pytest.raises(ValueError, match="f_min"):
        m.mirror_roughness_from_psd(-14.0, -2.6, -0.5, solid_angle_sr=1.0,
                                    f_min=2.0, f_max=1.0)
    # a2 > 0 means the answer is set by f_max, not by the data
    with pytest.raises(ValueError, match="curves upward"):
        m.mirror_roughness_from_psd(-14.0, -2.6, 0.5, **BAND,
                                    solid_angle_sr=1.0)


# -----------------------------------------------------------------------------
# The real numerical diffraction calculation (slow: runs the FFT)
# -----------------------------------------------------------------------------
@pytest.mark.slow
def test_compute_diffraction_profile_is_normalised_and_decreasing():
    prof = m.compute_diffraction_profile()
    assert prof.angle_deg[0] == pytest.approx(0.0)
    assert prof.diff_signal[0] == pytest.approx(1.0)          # peak-normalised
    assert np.all(prof.diff_signal > 0) and np.all(prof.diff_signal <= 1.0)
    # Non-decreasing (the smallest log-spaced bins can share a near-axis angle).
    assert np.all(np.diff(prof.angle_deg) >= 0)
    assert prof.angle_deg.size > 100
