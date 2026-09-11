"""
libera_telescope_pst
====================

Python 3.11 port of ``libera_model_telescope_pst.pro``.

The model builds the telescope point-source transmittance (PST) from up to
three additive components (``pst_two_components`` / ``pst_three_components``):

    pst = geo * (surface_scatter + particulate) + diffraction [+ ring]

* **surface_scatter** - smooth-surface scatter from the mirror's power spectral
  density, evaluated at the surface spatial frequency ``f = sin(theta)/lambda``
  and carrying the standard ``1/lambda**4`` prefactor.
* **particulate** - wavelength-and-angle power law from contamination, which
  sets the scatter floor once the roughness term dies off beyond ~3 um.
* **diffraction** - the Fraunhofer diffraction pattern of the pupil, computed
  once numerically and rescaled to any wavelength via a wavelength-independent
  lookup in ``u = sin(theta) / lambda``. Not vignetted.
* **ring** - a Gaussian bump used to fit the extra signal seen in the SW
  channel. Added outside the geometric factor, like the diffraction term.

IDL -> Python translation notes (called out inline where they occur):

    gen_array(a, b, n)        -> np.linspace(a, b, n)
    gen_array(a, b, n, /log)  -> np.geomspace(a, b, n)
    cmreplicate(v, n) + T     -> np.meshgrid(v, v)
    fft(m, /center)           -> np.fft.fftshift(fft2(m))   (magnitude only)
    interpol(y, x, xnew)      -> np.interp(xnew, x, y)  (clamps to the end
                                 values outside the data range)
    replicate({...}, n)       -> dataclass holding parallel arrays
    save/restore (.sav)       -> np.savez / np.load (.npz)  + in-memory cache

The IDL code cached the diffraction result in an ``.sav`` file because the FFT
is the one genuinely expensive step (the grid is 4001x4001 and 4001 is prime,
so the transform has no fast factorization - it runs ~1-4 s). We keep that idea
but persist to a portable, array-native ``.npz`` and add an in-process cache so
repeated calls in a fit loop neither re-read disk nor recompute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from scipy import fft as sfft
from scipy.special import erf
from types import SimpleNamespace
import numpy as np
import functools
from libera_config import load_config

# -----------------------------------------------------------------------------
# Pupil / telescope geometry (all lengths in metres unless noted)
# -----------------------------------------------------------------------------

# Diffraction-calculation grid
_DIFF_NPTS = 4001          # grid points per axis
_DIFF_WIDTH_M = 40e-3      # physical grid width [m]
_DIFF_REF_WAVELENGTH_UM = 20.0  # wavelength the pattern is computed at [um]

# Pupil apertures used in the diffraction calculation
_DIFF_PRIMARY_STOP_DIA_M = 18.1e-3   # outer clear aperture
_DIFF_OBSCURATION_DIA_M = 10.74e-3   # central (secondary) obscuration
# _DIFF_SPIDER_HALF_WIDTH_DEG = 2.6 / 2.0  # half angular width of each spider vane
_DIFF_SPIDER_WIDTH_M = 0.59e-3       # Width of the spider vanes

# Default location for the cached diffraction lookup table
# _DEFAULT_CACHE = Path(__file__).with_name("libera_telescope_modeled_diffraction.npz")

# -----------------------------------------------------------------------------
# Diffraction: numerical calculation, persistence, and wavelength scaling
# -----------------------------------------------------------------------------
@dataclass
class DiffractionProfile:
    """Radially-averaged diffraction pattern, stored wavelength-independently.

    Attributes
    ----------
    angle_deg : np.ndarray
        Off-axis angle [deg] at the reference wavelength.
    u : np.ndarray
        ``sin(angle) / lambda`` [1/m]. This is the wavelength-independent
        coordinate that lets a single calculation serve every wavelength.
    diff_signal : np.ndarray
        Peak-normalised diffraction irradiance [-].
    ref_wavelength_um : float
        Wavelength the pattern was computed at [um].
    """

    angle_deg: np.ndarray
    u: np.ndarray
    diff_signal: np.ndarray
    ref_wavelength_um: float = _DIFF_REF_WAVELENGTH_UM


def compute_diffraction_profile() -> DiffractionProfile:
    """Numerically compute the pupil diffraction pattern (the expensive step).

    Ports ``libera_model_pst_light_numerical_diffraction_calculation``. The
    Fraunhofer irradiance is ``|FFT(pupil)|**2``; because we only ever use its
    magnitude and normalise to the peak, neither the FFT normalisation constant
    nor the input origin matters, so a single ``fftshift`` on the output is the
    exact equivalent of IDL's ``fft(..., /center)``.
    """
    wavelength_m = _DIFF_REF_WAVELENGTH_UM * 1e-6
    npts = _DIFF_NPTS

    # 1-D position grid, then a 2-D (x, y) mesh: gen_array -> linspace,
    # cmreplicate(...) + transpose -> meshgrid.
    x1d = np.linspace(-_DIFF_WIDTH_M / 2.0, _DIFF_WIDTH_M / 2.0, npts)
    x, y = np.meshgrid(x1d, x1d)
    r = np.sqrt(x**2 + y**2)

    # Angular grid conjugate to the position grid. The angular sample spacing is
    # set by the total grid extent: dtheta = asin(lambda / L).
    dtheta_deg = np.degrees(np.arcsin(wavelength_m / (x1d.max() - x1d.min())))
    theta_1d = np.arange(npts) * dtheta_deg
    theta_1d -= theta_1d.mean()
    xt, yt = np.meshgrid(theta_1d, theta_1d)
    r_theta_deg = np.sqrt(xt**2 + yt**2)

    # Azimuth used only to place the (approximate) spider vanes.
    az_deg = np.degrees(np.arctan2(y, x))

    # Build the pupil mask: clear aperture, central obscuration, and three
    # spider vanes at 0, +120, -120 deg. The vane placement is deliberately
    # approximate (it follows azimuth lines), as in the original.
    mask_aperture = r < _DIFF_PRIMARY_STOP_DIA_M / 2.0
    mask_obscuration = r > _DIFF_OBSCURATION_DIA_M / 2.0

    mask_spiders = np.ones_like(r, dtype=bool)
    for vane_deg in (0.0, 120.0, -120.0):
        a = np.radians(vane_deg)
        u =  x*np.cos(a) + y*np.sin(a)
        v = -x*np.sin(a) + y*np.cos(a)
        mask_spiders &= ~((np.abs(v) < _DIFF_SPIDER_WIDTH_M/2.0) & (u > 0.0))

    # shw = _DIFF_SPIDER_HALF_WIDTH_DEG
    # mask_spiders = (
    #     ((az_deg <= -shw) | (az_deg > shw))
    #     & ((az_deg <= 120.0 - shw) | (az_deg > 120.0 + shw))
    #     & ((az_deg <= -120.0 - shw) | (az_deg > -120.0 + shw))
    # )

    pupil = (mask_aperture & mask_obscuration & mask_spiders).astype(np.float64)

    # Fraunhofer irradiance, DC-centred and peak-normalised.
    # scipy's pocketfft with workers=-1 is the fast path for this prime size.
    field = sfft.fftshift(np.abs(sfft.fft2(pupil, workers=-1)) ** 2)
    field /= field.max()

    # --- Radial average onto a log-spaced angular grid -----------------------
    r_flat = r_theta_deg.ravel()
    sig_flat = field.ravel()
    order = np.argsort(r_flat)
    r_flat = r_flat[order]
    sig_flat = sig_flat[order]

    n_flat = r_flat.size
    # gen_array(..., /log) -> geomspace; round to integer sample indices, dedupe.
    idx = np.unique(np.round(np.geomspace(1.0, n_flat - 1.0, 2001)).astype(np.int64))

    angle_deg = np.empty(idx.size - 1)
    diff_signal = np.empty(idx.size - 1)
    for m in range(idx.size - 1):
        # IDL array slices are inclusive of the upper bound, so add 1 to match.
        sl = slice(idx[m], idx[m + 1] + 1)
        angle_deg[m] = r_flat[sl].mean()
        diff_signal[m] = sig_flat[sl].mean()

    # Prepend the on-axis point (angle 0, normalised signal 1).
    angle_deg = np.concatenate(([0.0], angle_deg))
    diff_signal = np.concatenate(([1.0], diff_signal))

    # Store against the wavelength-independent coordinate u = sin(theta)/lambda.
    u = np.sin(np.radians(angle_deg)) / wavelength_m

    return DiffractionProfile(angle_deg=angle_deg, u=u, diff_signal=diff_signal)


def save_diffraction_profile(profile: DiffractionProfile, path: Path | str) -> None:
    """Persist a diffraction profile to a portable ``.npz`` (replaces IDL save)."""
    np.savez(
        path,
        angle_deg=profile.angle_deg,
        u=profile.u,
        diff_signal=profile.diff_signal,
        ref_wavelength_um=profile.ref_wavelength_um,
    )


def _default_diffraction_cache_path() -> Path:
    """Location of the cached diffraction lookup table, from config.toml."""
    paths = load_config()
    filename = 'libera_telescope_modeled_diffraction.npz'
    return paths.analysis_dir / filename


def _load_diffraction_profile(path: Path | str | None = None) -> DiffractionProfile:
    """Load a stored profile. ``path=None`` uses the configured cache location."""
    if path is None:
        path = _default_diffraction_cache_path()

    with np.load(path) as data:
        return DiffractionProfile(
            angle_deg=data["angle_deg"],
            u=data["u"],
            diff_signal=data["diff_signal"],
            ref_wavelength_um=float(data["ref_wavelength_um"]),
        )


@functools.lru_cache(maxsize=None)
def get_diffraction_profile() -> DiffractionProfile:
    """Return the diffraction profile, computing and caching it as needed.

    Behaviour:
      * if the ``.npz`` cache exists, load it;
      * otherwise compute it once and write the cache.
    The ``lru_cache`` keeps it resident for the life of the process, so a fit
    loop calling ``pst_two_components`` many times pays the cost once. Delete
    the ``.npz`` (and call ``get_diffraction_profile.cache_clear()``) to force
    a recompute.
    """
    path = _default_diffraction_cache_path()

    if path.exists():
        return _load_diffraction_profile(path)
    profile = compute_diffraction_profile()
    save_diffraction_profile(profile, path)
    return profile


def scale_diffraction_to_wavelength(
    profile: DiffractionProfile, wavelength_um: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale the stored profile to ``wavelength_um``.

    Returns ``(angle_deg, diff_signal)`` with a flat anchor added at 180 deg so
    interpolation to large angles never extrapolates off the end.

    Ports ``libera_model_pst_light_diffraction_model``. Because the pattern is
    stored in ``u``, the angle axis is simply ``theta = asin(u * lambda)``. Any
    stored point with ``u * lambda > 1`` has no diffraction angle at this (longer)
    wavelength - the pattern is compressed - so we drop those tail points. At the
    reference wavelength nothing is dropped and the result is identical to IDL;
    this only differs for lambda > 20 um, where IDL would have produced NaNs.
    """
    arg = profile.u * (wavelength_um * 1e-6)
    valid = np.abs(arg) <= 1.0

    angle = np.degrees(np.arcsin(arg[valid]))
    diff = profile.diff_signal[valid]

    order = np.argsort(angle)  # keep ascending for interpolation
    angle = angle[order]
    diff = diff[order]

    # Flat anchor at 180 deg (large-angle floor for the interpolation).
    angle = np.append(angle, 180.0)
    diff = np.append(diff, diff[-1])
    return angle, diff


# -----------------------------------------------------------------------------
# Geometric illumination (vignetting) factor
# -----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _geometric_curve() -> tuple[np.ndarray, np.ndarray]:
    """Compute the vignetting curve on its native 0-20 deg grid.

    This is the expensive part of ``geometric_model`` - 50 angles x a 200x200
    pupil grid - and it depends on nothing at all, so it is computed once per
    process and reused. Returns ``(angle_deg, geo)``.
    """
    npts = 200
    x1d = np.linspace(-12.0, 12.0, npts)
    x, y = np.meshgrid(x1d, x1d)

    angle_deg = np.linspace(0.0, 20.0, 50)
    angle_rad = np.radians(angle_deg)
    geo = np.empty_like(angle_rad)

    for m, ang in enumerate(angle_rad):
        t = np.tan(ang)
        mask0 = np.sqrt(x**2 + y**2) < 20.0 / 2.0                         # mirror
        mask1 = np.sqrt((x + 9.886 * t) ** 2 + y**2) < 18.0 / 2.0         # aperture
        mask2 = np.sqrt((x + 13.933 * t) ** 2 + y**2) > 11.0 / 2.0        # obscuration
        mask3 = np.sqrt((x + 52.868 * t) ** 2 + y**2) < 21.382 / 2.0      # baffle
        geo[m] = np.count_nonzero(mask0 & mask1 & mask2 & mask3)

    geo /= geo[0]  # normalise to on-axis
    return angle_deg, geo


def geometric_model(angles_deg: np.ndarray, *, clamp: bool = True) -> np.ndarray:
    """Fraction of the primary mirror that stays illuminated off-axis.

    Ports ``libera_model_pst_geometric_model``. For a set of field angles it
    counts the pupil-grid cells that survive the AND of the mirror, aperture,
    obscuration, and baffle masks (each shifted by ``disp * tan(angle)``),
    normalises to the on-axis value, and interpolates to the requested angles.
    1 = fully illuminated, 0 = fully vignetted. Units here are millimetres, as
    in the original.

    Parameters
    ----------
    clamp : bool, default True
        The underlying grid only spans 0-20 deg. ``np.interp`` holds the
        boundary value beyond it (so the factor stays >= 0 - no negative
        surface-scatter), which means the only thing the clamp still trims is a
        small (<0.1%) discretisation overshoot slightly above 1 near the axis.
        With ``clamp=True`` the result is bounded to ``[0, 1]``; set
        ``clamp=False`` to return the raw interpolated values.
    """
    angle_deg, geo = _geometric_curve()
    geo_factor = np.interp(np.asarray(angles_deg, dtype=float), angle_deg, geo)
    if clamp:
        geo_factor = np.clip(geo_factor, 0.0, 1.0)
    return geo_factor


# -----------------------------------------------------------------------------
# Two- and three-component scatter models
#
# The older model treated k_scat and c_scat as 2nd-order polynomials in
# wavenumber. That was purely empirical, and it had two problems: the
# polynomial drove the PST negative below ~0.32 um, and k_scat/c_scat were ~97%
# anticorrelated so neither was individually meaningful.
#
# These models instead follow the smooth-surface scattering theory the surface
# term was already built on:
#
#     pst_surf = PSD(f) * geo * 16*pi^2 / wavelength_um^4
#
# where f = sin(angle)/wavelength_um is the surface spatial frequency [1/um]
# and PSD(f) is the mirror's power spectral density. In that theory the PSD is
# a property of the *mirror*, not of wavelength - all of the wavelength
# dependence lives in f and in the 1/lambda^4 prefactor. The apparent drift of
# k_scat and c_scat with wavelength is mostly an artifact of forcing a straight
# line onto a curved PSD, because each measurement wavelength samples a
# different sliding band of f (0.359 um covers f = 0.097-0.388, 2.376 um covers
# f = 0.015-0.059 - they do not even overlap).
#
# A residual wavelength trend survives after allowing PSD curvature, and it is
# flat below ~1 um then climbs sharply. That is the signature of a second,
# wavelength-independent scatter component (particulate / contamination),
# taking over as the roughness term dies off as 1/lambda^4.
#
#     pst = geo * [ PSD(f)/lambda^4 * 16*pi^2  +  particulate(angle) ] + diff
#
# PSD and the particulate term are both parameterised in log space so the
# amplitudes cannot go negative, and both are pivoted near the centre of the
# measured range so the coefficients come out close to uncorrelated.
# -----------------------------------------------------------------------------

# Pivot points for the log-space polynomials. These are chosen near the centre
# of the measured data so that the fitted coefficients are close to
# uncorrelated; they are a reparameterisation only and do not change the model.
PSD_PIVOT_FREQ = 0.08    # spatial frequency pivot f0 [1/um]
PART_PIVOT_DEG = 4.0     # particulate angle pivot [deg]
RING_PIVOT_UM = 0.43     # SW ring wavelength pivot [um]

# Effective receiver solid angle of the ERF scattered-light measurement [sr].
# The model is fit to a peak-normalised PST, while the 16*pi^2/lambda^4 * S(f)
# form it uses is Church's smooth-surface BRDF; the two differ by this factor
# (PST = BRDF * Omega). It therefore only enters when converting the fitted PSD
# back to an absolute surface roughness - it does not affect the PST model or
# the fit itself.
ERF_RECEIVER_SOLID_ANGLE_SR = 7.66e-4

# Cache for the parts of the model that do not depend on any fit parameter.
# Both the diffraction term and the geometric illumination factor depend only
# on (angle, wavelength), so they are constant across a fit - recomputing them
# every iteration is the dominant cost otherwise.
_STATIC_TERM_CACHE = {}
_STATIC_TERM_CACHE_MAX = 32


def _static_pst_terms(angle_deg, wavelength_um):
    """Return the (diffraction, geometric factor) terms for this set of points.

    Neither depends on the fit parameters, so they are cached on the values of
    the independent variables and reused across fit iterations.

    The diffraction term is grouped by wavelength: it depends only on (angle,
    wavelength), so one scaled curve serves every point sharing a wavelength,
    and the scale/interp cost is paid once per *distinct* wavelength rather
    than once per point.
    """
    angle_deg = np.asarray(angle_deg, dtype=float)
    wavelength_um = np.asarray(wavelength_um, dtype=float)

    key = (angle_deg.shape, angle_deg.tobytes(),
           wavelength_um.shape, wavelength_um.tobytes())

    if key not in _STATIC_TERM_CACHE:
        profile = get_diffraction_profile()
        ang_flat = np.ravel(angle_deg)
        wl_flat = np.ravel(wavelength_um)
        diff_flat = np.empty(ang_flat.shape, dtype=float)

        uniq_wl, inv = np.unique(wl_flat, return_inverse=True)
        inv = np.ravel(inv)  # numpy>=2.0 can return inv shaped like the input
        for i in range(uniq_wl.size):
            sel = inv == i
            diff_angle, diff_sig = scale_diffraction_to_wavelength(
                profile, float(uniq_wl[i])
            )
            diff_flat[sel] = np.interp(ang_flat[sel], diff_angle, diff_sig)

        pst_diff = diff_flat.reshape(angle_deg.shape)
        geo_factor = geometric_model(angle_deg)

        if len(_STATIC_TERM_CACHE) >= _STATIC_TERM_CACHE_MAX:
            _STATIC_TERM_CACHE.clear()
        _STATIC_TERM_CACHE[key] = (pst_diff, geo_factor)

    return _STATIC_TERM_CACHE[key]


def pst_two_components(angle_deg, wavelength_um, a0, a1, a2, lp0, lp1):
    """Evaluate the two-component scatter model and return its pieces separately.

    Parameters
    ----------
    angle_deg, wavelength_um : float or array_like
        Off-axis angle [deg] and wavelength [um]. Broadcast together, so pass
        parallel arrays (one entry per measurement) to fit a whole dataset.
    a0, a1, a2 : float
        Surface PSD coefficients. The PSD is a quadratic in log-log space,
        pivoted at PSD_PIVOT_FREQ:

            ln PSD = a0 + a1*ln(f/f0) + a2*ln(f/f0)**2

        a0 is ln(PSD) at the pivot frequency, a1 is the local power-law slope
        there (so a1 = -c_scat in the old parameterisation), and a2 is the
        curvature. Setting a2=0 recovers a pure power-law PSD.
    lp0, lp1 : float
        Particulate scatter coefficients:

            ln pst_part = lp0 + lp1*ln(angle/PART_PIVOT_DEG)

        lp0 is ln(PST) contribution at the pivot angle and lp1 is the angular
        slope.

    Returns
    -------
    SimpleNamespace with pst, surf, part, diff, geo_factor, f, psd - all
    broadcast to the common shape of the inputs. `f` is the surface spatial
    frequency [1/um] each point corresponds to, which is useful for checking
    how far a prediction sits outside the measured range.
    """
    angle_deg = np.asarray(angle_deg, dtype=float)
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    angle_deg, wavelength_um = np.broadcast_arrays(angle_deg, wavelength_um)

    pst_diff, geo_factor = _static_pst_terms(angle_deg, wavelength_um)

    # Surface spatial frequency [1/um]. Clipped away from zero so the logs stay
    # finite if this is ever evaluated at exactly 0 deg.
    beta = np.sin(np.radians(angle_deg))
    f = np.clip(beta / wavelength_um, 1e-12, None)

    # Surface roughness scatter: curved PSD in f, with the standard 1/lambda^4
    # smooth-surface prefactor. Reflectance is folded into a0, as elsewhere.
    log_f = np.log(f / PSD_PIVOT_FREQ)
    psd = np.exp(a0 + a1*log_f + a2*log_f**2)
    pst_surf = psd * 16.0 * np.pi**2 / wavelength_um**4

    # Particulate scatter. In the diffractive regime the wavelength and angular
    # exponents are not independent: BRDF ~ lam^(3-p) * theta^(p-5) for a local
    # size-distribution slope dN/da ~ a^-p. Writing lp1 = p-5 fixes the wavelength
    # exponent at 3-p = -2-lp1, so this costs no extra free parameter.
    log_angle = np.log(np.clip(angle_deg, 1e-6, None) / PART_PIVOT_DEG)
    pst_part = np.exp(lp0 + lp1*log_angle) * wavelength_um**(-2.0 - lp1)

    # Both scatter terms see the same vignetting; diffraction does not.
    pst_total = geo_factor*(pst_surf + pst_part) + pst_diff

    return SimpleNamespace(pst=pst_total,
                           surf=geo_factor*pst_surf,
                           part=geo_factor*pst_part,
                           diff=pst_diff,
                           geo_factor=geo_factor,
                           f=f,
                           psd=psd)


def pst_three_components(angle_deg, wavelength_um, a0, a1, a2, lp0, lp1,
                         gc0, gw0, gw1, ga0, ga1,
                         ):
    """Two-component model plus the SW Gaussian ring.

    The SW channel shows extra signal in a narrow annulus that neither scatter
    term nor diffraction accounts for. It is added outside the geometric factor,
    like the diffraction term, since it is not a mirror-scatter contribution.

    All three ring properties vary with wavelength, and the two that must stay
    positive - the width and the amplitude - are parameterised in log space so
    they cannot reach zero or go negative anywhere in the fit.

    Parameters
    ----------
    angle_deg, wavelength_um, a0, a1, a2, lp0, lp1
        As for ``pst_two_components``.
    gc0 : float
        Effective grating spacing [um] of the secondary, setting the ring
        centre by ``asin(wavelength/gc0)``. The ring moves outward with
        wavelength, as first-order diffraction should. There is no first order
        for ``wavelength >= gc0``, and the artifact has faded by then in any
        case, so the ring is simply zero at those wavelengths - passing LW data
        is safe and contributes no ring rather than NaN.
    gw0, gw1 : float
        Log-space ring width [deg], pivoted at ``RING_PIVOT_UM``:
        ``width = exp(gw0 - gw1*(wavelength_um - RING_PIVOT_UM))``. The width is
        ``exp(gw0)`` at the pivot and falls by a factor ``exp(gw1)`` per micron.
        Log space keeps it strictly positive.
    ga0, ga1 : float
        Log-space ring amplitude, pivoted the same way:
        ``amp = exp(ga0 + ga1*(wavelength_um - RING_PIVOT_UM))``. The peak height
        is ``exp(ga0)`` at the pivot, positive by construction, and changes by a
        factor ``exp(ga1)`` per micron.

    Returns
    -------
    SimpleNamespace with the same fields as ``pst_two_components`` plus `ring`,
    with `pst` now including the ring.
    """
    parts = pst_two_components(angle_deg, wavelength_um, a0, a1, a2, lp0, lp1)

    # Calculate the center of the Gaussian annulus
    # We're using a model of diffraction from the secondary
    # d = 5.410   # Fit to the secondary grating spacing in microns
    #
    # First order only exists while wavelength < gc0. Past that there is no
    # diffraction angle to solve for - and physically the artifact has died out
    # by then anyway - so the ring is switched off rather than allowed to become
    # NaN. The argument is clipped before arcsin so the invalid points produce a
    # harmless number instead of a warning; `ring_exists` then zeroes them.
    sin_center = np.asarray(wavelength_um, dtype=float) / gc0
    ring_exists = sin_center < 1.0
    g_center = np.degrees(np.arcsin(np.clip(sin_center, -1.0, 1.0)))

    # Both the width and the amplitude are log-linear in wavelength, measured
    # from RING_PIVOT_UM rather than from zero. The pivot is a
    # reparameterisation only - it does not change the model - but it puts each
    # intercept in the middle of the 0.36-0.7 um range where the ring actually
    # has signal, instead of extrapolating it to zero wavelength. Without it
    # gw0/gw1 and ga0/ga1 come out ~98% anticorrelated and neither is
    # individually meaningful, the same problem the PSD and particulate pivots
    # exist to avoid.
    d_wavelength = np.asarray(wavelength_um, dtype=float) - RING_PIVOT_UM

    # Allow the Gaussian width to change with wavelength, in log space so it
    # stays strictly positive. A plain linear form crosses zero at some
    # wavelength (1.15 um for the old 0.622/0.540 pair, i.e. inside the SW
    # range), dividing by zero there and silently flipping sign beyond it -
    # g_width**2 hides the flip as an unphysical widening rather than an error.
    # The width is exp(gw0) at the pivot and shrinks by exp(gw1) per micron.
    g_width = np.exp(gw0 - gw1*d_wavelength)

    # Gaussian height. Strictly positive by construction: exp(ga0) at the pivot,
    # changing by a factor exp(ga1) per micron.
    g_amp = np.exp(ga0 + ga1*d_wavelength)

    angle_deg = np.asarray(angle_deg, dtype=float)
    pst_ring = g_amp*np.exp(-0.5*(angle_deg - g_center)**2 / g_width**2)

    # No first order at or beyond gc0: the ring contributes nothing there.
    pst_ring = np.where(ring_exists, pst_ring, 0.0)

    # Broadcast to the model shape in case angle_deg is the scalar of the pair.
    pst_ring = np.broadcast_to(pst_ring, parts.pst.shape)

    parts.ring = pst_ring
    parts.pst = parts.pst + pst_ring
    return parts


def mirror_roughness_from_psd(a0, a1, a2, *, f_min, f_max,
                              solid_angle_sr=ERF_RECEIVER_SOLID_ANGLE_SR):
    """Band-limited rms surface roughness implied by the fitted PSD coefficients.

    Integrates the surface PSD that ``pst_two_components`` fits, turning the
    scattered-light measurement into a mirror roughness estimate. For an
    isotropic 2-D PSD the rms roughness over a band of spatial frequencies is

        sigma**2 = integral of 2*pi*f * S(f) df,  from f_min to f_max

    With ``ln S = a0 + a1*L + a2*L**2`` and ``L = ln(f/PSD_PIVOT_FREQ)``, the
    substitution ``f = f0*exp(L)`` turns this into

        sigma**2 = 2*pi*f0**2 * integral of exp(a0 + (a1+2)*L + a2*L**2) dL

    which is a Gaussian integral in ``L`` whenever ``a2 < 0`` (the usual fitted
    case - the PSD curves down in log-log), so it evaluates in closed form via
    error functions with no numerical quadrature. ``a2 == 0`` reduces to the
    plain exponential/linear result, which is handled separately.

    Because ``a2 < 0`` makes the integrand fall off faster than any power law at
    both ends, the result is only weakly sensitive to the band: for the Total
    channel fit, widening from the measured band to effectively unbounded moves
    sigma by ~7%.

    The band has no default because each channel's data spans a different range
    of ``f = sin(angle)/wavelength``, and integrating outside what a given
    channel measured is extrapolation. Take it from the fit itself:

        parts = pst_two_components(angle_deg, wavelength_um, a0, a1, a2, lp0, lp1)
        rough = mirror_roughness_from_psd(a0, a1, a2,
                                          f_min=parts.f.min(),
                                          f_max=parts.f.max())

    Calibration
    -----------
    ``a0`` is fit against a peak-normalised PST, whereas the
    ``16*pi**2/lambda**4 * S(f)`` form the model uses is Church's smooth-surface
    *BRDF*. The two differ by the effective receiver solid angle:
    ``PST = BRDF * solid_angle_sr``. The fitted PSD is therefore smaller than the
    true PSD by that factor, so the roughness scales as
    ``1/sqrt(solid_angle_sr)`` - at the ERF value of 7.66e-4 sr this is a factor
    of ~36 on sigma, so it is not a detail. Note that reflectance is also folded
    into ``a0`` by the model, which absorbs a further factor of order unity.

    Parameters
    ----------
    a0, a1, a2 : float
        Fitted PSD coefficients from ``pst_two_components``.
    solid_angle_sr : float
        Effective receiver solid angle [sr] relating the measured PST to BRDF.
        Defaults to ``ERF_RECEIVER_SOLID_ANGLE_SR``, the ERF setup value.
    f_min, f_max : float
        Integration band [1/um]. Required - use the range the channel's data
        actually covers, e.g. ``parts.f.min()`` / ``parts.f.max()``.

    Returns
    -------
    SimpleNamespace with sigma_nm, sigma_um, psd_integral (= sigma**2 [um**2]),
    and the f_min / f_max / solid_angle_sr actually used.
    """
    if not f_max > f_min > 0.0:
        raise ValueError("need 0 < f_min < f_max")
    if not solid_angle_sr > 0.0:
        raise ValueError("solid_angle_sr must be positive")

    f0 = PSD_PIVOT_FREQ
    log_min = np.log(f_min / f0)
    log_max = np.log(f_max / f0)

    # The +2 comes from the 2*pi*f df measure of the 2-D integral pulling two
    # extra factors of f = f0*exp(L) through the substitution.
    b = a1 + 2.0

    if a2 == 0.0:
        # Pure power-law PSD. b == 0 is the log-divergent-slope special case
        # where the integrand is flat in L.
        if b == 0.0:
            integral = np.exp(a0) * (log_max - log_min)
        else:
            integral = (np.exp(a0 + b*log_max) - np.exp(a0 + b*log_min)) / b
    elif a2 < 0.0:
        # Completing the square: a0 + b*L + a2*L**2
        #   = (a0 - b**2/(4*a2)) - k*(L + b/(2*a2))**2,  k = -a2 > 0
        k = -a2
        centre = b / (2.0 * a2)
        integral = (
            np.exp(a0 - b**2 / (4.0 * a2))
            * 0.5 * np.sqrt(np.pi / k)
            * (erf(np.sqrt(k) * (log_max + centre))
               - erf(np.sqrt(k) * (log_min + centre)))
        )
    else:
        # a2 > 0 curves the PSD *up* in log-log, so the integral is dominated by
        # the upper limit and grows without bound as f_max increases. The band
        # integral is still finite and correct, but it is an extrapolation
        # artifact rather than a roughness, so say so instead of returning it.
        raise ValueError(
            f"a2 = {a2:.3g} > 0: the PSD curves upward in log-log, so the "
            "roughness is set by f_max rather than by the data. Refit with "
            "fit_curvature=False, or narrow the band deliberately."
        )

    # Undo the PST -> BRDF scaling, then sigma**2 = 2*pi*f0**2 * integral.
    sigma_sq_um2 = 2.0 * np.pi * f0**2 * integral / solid_angle_sr
    sigma_um = np.sqrt(sigma_sq_um2)

    return SimpleNamespace(sigma_nm=sigma_um * 1e3,
                           sigma_um=sigma_um,
                           psd_integral=sigma_sq_um2,
                           f_min=f_min,
                           f_max=f_max,
                           solid_angle_sr=solid_angle_sr)
