import pandas as pd
import numpy as np
import math
import warnings
import pbrr_conversions as pbr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from types import SimpleNamespace
from matplotlib.ticker import FuncFormatter

from libera_config import load_config

# Directories in config.toml that hold raw instrument data. These are NOT in
# the repository - the ERF log files, laser spectra and MODTRAN scene files are
# large and live on LASP storage. Everything from level_01 onward runs from the
# committed CSVs in data/, so the SRF analysis reproduces without them; only the
# level_00/level_01 reduction stages and the individual spectrum analyses need
# them. See raw_data_available.
RAW_DATA_PATH_KEYS = ("erf_osa_spectrum_dir", "erf_ccs_spectrum_dir",
                      "erf_grating_spectrum_dir", "erf_log_file_dir",
                      "erf_misc_file_dir", "erf_ceres_scene_dir")


def raw_data_available(*, keys=RAW_DATA_PATH_KEYS, report=True):
    """
    True if every raw-data directory named in config.toml exists on this
    machine, False if any of them is missing or absent from config.toml.

    The notebooks call this once at the top to set their raw_data_available
    flag, so that the cells re-running the upstream reduction skip themselves
    with an explanation rather than raising on a missing path, while every cell
    downstream of the committed CSVs in data/ runs either way. Set the notebook
    flag by hand to override the detection in either direction.

    Reports which paths failed rather than just returning False, so a partial
    setup (some of the ERF data tree present, some not) is diagnosable instead
    of looking like a total absence.
    """

    paths = load_config()
    missing = [k for k in keys
               if not getattr(paths, k, None) or not Path(getattr(paths, k)).is_dir()]

    if report and missing:
        print("Raw instrument data not found - these config.toml paths do not resolve:")
        for k in missing:
            print(f"  {k} = {getattr(paths, k, '<not in config.toml>')}")
        print("The analysis from level_01 onward runs from the CSVs in data/, "
              "so only the cells that re-reduce the raw files are affected.")

    return not missing


def gaussian(x, amplitude, center, fwhm, offset):
    return amplitude * np.exp(-4 * np.log(2) * (x - center) ** 2 / fwhm ** 2) + offset

def gaussian_no_offset(x, amplitude, center, fwhm):
    return amplitude * np.exp(-4 * np.log(2) * (x - center) ** 2 / fwhm ** 2)

def tidy_up_header(df):
    #Clean up the column names
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r"[\[(]", "_", regex=True)    # replace [ and ( with _
        .str.replace(r"[\])]", "",  regex=True)    # drop ] )
        .str.replace(" _", " ",  regex=True)       # replace any double spaces with single space
        .str.strip()                               # trailing/leading whitespace, e.g. "signal " -> "signal"
        .str.replace(" ", "_")                     # "peak wavelength" -> "peak_wavelength"
    )
    return df

def log_x_axis_decimal(ax):
    # matplotlib's default log-axis formatter renders ticks as powers of ten
    # (e.g. 10^0, 10^1); this shows plain decimals instead (e.g. 0.3, 1, 3, 10).
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))

def fit_plane(*, x, y, z, sd=None, n_boot=500, seed=None, absolute_sd=False):
    """
    Fit a 2D plane z = a*x + b*y + c via linear least squares, with both
    an analytic (Gaussian-noise-assumed) parameter uncertainty and a
    bootstrap-resampled uncertainty that doesn't assume Gaussian/homoscedastic
    noise -- useful for reflecting the influence of outliers or non-uniform
    noise on the fit stability, without rejecting any points.

    Parameters
    ----------
    x, y, z : array_like
        1D arrays of equal length.
    sd : array_like, optional
        Per-point uncertainty on z, used as weights w = 1/sd**2. Default None
        weights every point equally, which reproduces the unweighted fit
        exactly. Must be finite and strictly positive.
    n_boot : int, optional
        Number of bootstrap resamples (case resampling, with replacement).
        Default 500. Set to 0 to skip bootstrapping.
    seed : int, optional
        Seed for the bootstrap random number generator, for reproducibility.
    absolute_sd : bool, optional
        How to read the sd values when scaling the analytic covariance.

        False (default) treats sd as known only up to a constant factor: it
        sets the relative weighting between points, and the overall size of
        the error bars comes from the residuals via cov = redchi*inv(A'WA).
        This is the right choice for l_sd, which is the within-half-cycle
        sample scatter and so is proportional to, not equal to, the
        uncertainty on the dwell mean. It also matches scipy.curve_fit's
        default.

        True treats sd as an absolute 1-sigma uncertainty and leaves the
        covariance as inv(A'WA). Use it only once the sd values are known
        to be correctly scaled -- comparing redchi against 1 is the test.

    Returns
    -------
    SimpleNamespace with fields:
        a, b, c                            : fitted plane coefficients
        a_err, b_err, c_err                : analytic 1-sigma uncertainty (Gaussian-noise assumption)
        a_err_boot, b_err_boot, c_err_boot : bootstrap 1-sigma uncertainty (distribution-free)
        rmse                               : root-mean-square residual (unweighted)
        chisq, redchi                      : weighted chi-square and its reduced value. With
                                             absolute_sd=False redchi is 1 by construction only
                                             for the scaling, so the useful diagnostic is the
                                             unscaled redchi reported here: far from 1 means the
                                             supplied sd is off by that factor squared.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(x)

    # Weights. Unit weights reproduce the unweighted fit bit for bit, so
    # existing callers that pass no sd are unaffected.
    if sd is None:
        w = np.ones(n)
    else:
        sd = np.asarray(sd, dtype=float)
        if sd.shape != z.shape:
            raise ValueError("sd must be the same length as z.")
        usable = np.isfinite(sd) & (sd > 0)
        if not usable.any():
            # Nothing usable anywhere, so sd carries no relative weighting
            # information and unit weights are the honest reading rather than a
            # fudge. This happens when a detector's telemetry freezes for a
            # whole file - every sample identical, so every l_sd is exactly
            # zero - and it keeps such a file behaving as it did before
            # weighting instead of taking down the batch. Same convention as
            # weighted_mean_and_stddev, which drops to an unweighted mean when
            # it sees a zero. The file still has to be caught downstream on its
            # own merits; this only decides how to weight it.
            w = np.ones(n)
        elif not usable.all():
            # A mix is genuinely ambiguous - some points would get infinite
            # weight relative to their neighbours - so refuse rather than guess.
            raise ValueError(f"sd has {int((~usable).sum())} of {n} entries that are zero, "
                             "negative or non-finite, mixed with usable ones. A zero would "
                             "give that point infinite weight; drop or floor those points "
                             "before fitting.")
        else:
            w = sd**-2.0

    # The 2D plane fit. Weighting by scaling both sides by sqrt(w) is the
    # standard trick: it turns the weighted normal equations into an ordinary
    # least-squares problem, so the rank check still means what it did.
    def _fit(xx, yy, zz, ww):
        A = np.column_stack([xx, yy, np.ones_like(xx)])
        rw = np.sqrt(ww)
        coeffs, _, rank, _ = np.linalg.lstsq(A*rw[:, None], zz*rw, rcond=None)
        return coeffs, A, rank

    # Do the fit and extract the fit coeffs
    coeffs, A, rank = _fit(x, y, z, w)
    a, b, c = coeffs

    # Error if the plane fit isn't well constrained; like if the data is in a line
    if rank < 3:
        raise ValueError("Design matrix is rank-deficient (x,y points may be collinear) "
                          "-- plane fit is not well constrained.")

    # Note A @ coeffs calculates the fit for the set of x and y points
    residuals = z - A @ coeffs
    dof = n - 3
    chisq = np.sum(w * residuals**2)
    redchi = chisq / dof
    cov = np.linalg.inv((A*w[:, None]).T @ A)
    if not absolute_sd:
        cov = redchi * cov
    a_err, b_err, c_err = np.sqrt(np.diag(cov))
    rmse = np.sqrt(np.mean(residuals**2))

    # Run bootstrapping to calculate the bootstrap error. The resampled points
    # carry their own weights, so the bootstrap is weighted the same way the
    # fit is and needs no absolute_sd equivalent - it never used sd for scale.
    a_err_boot = b_err_boot = c_err_boot = np.nan
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boot_coeffs = np.empty((n_boot, 3))
        for i in range(n_boot):
            idx = rng.integers(0, n, n)   # case resampling: draw rows with replacement
            boot_coeffs[i], _, _ = _fit(x[idx], y[idx], z[idx], w[idx])
        a_err_boot, b_err_boot, c_err_boot = boot_coeffs.std(axis=0, ddof=1)

    # Return the fit coefficients, basic error estimates and the bootstrap error estimates
    return SimpleNamespace(
        a=a, b=b, c=c,
        a_err=a_err, b_err=b_err, c_err=c_err,
        a_err_boot=a_err_boot, b_err_boot=b_err_boot, c_err_boot=c_err_boot,
        rmse=rmse, chisq=chisq, redchi=redchi)

def fit_radiance_field(meas_data, *, share_angular=True, fit_drift=True,
                       fit_gradients=True, n_iter=3, pbrr_index=0):
    """
    Fit one radiance field to every detector in a file at once.

    Every detector in a file is looking at the same source, so the field they
    measure is one object. This fits it as one object instead of fitting each
    detector separately and dividing afterwards:

        l = l0 * r_i * (1 + gx*x + gy*y + gp*pitch + gw*yaw + gt*(t - t_ref))

    l0        radiance seen by PBR-R at x=y=0, pitch=yaw=0, at the file's mean
              time [W m-2 sr-1]
    r_i       responsivity of detector i relative to PBR-R. r=1 means it reads
              the same radiance, r=0.5 means it reads half. This is the
              measurand, so it comes out as a fitted parameter with its own
              uncertainty rather than as a quotient of two separate fits -
              which also means no E[A/B] != E[A]/E[B] bias and no need to
              reason about numerator/denominator correlation by hand.
    gx, gy    fractional radiance gradient per mm
    gp, gw    fractional gradient per degree of pitch/yaw. The manipulator only
              tips the science radiometers, so these do not act on PBR-R.
    gt        fractional drift per minute

    The gradients are FRACTIONAL, and that is what makes sharing them across
    detectors legitimate: a fractional gradient is a property of the source
    field, an absolute one (W m-2 sr-1 per mm) would differ per detector with
    its responsivity and could not be shared.

    Sharing is worth doing because it is a real constraint - five detectors
    measuring one gradient - and because of the timing. PBR-R is taken in one
    contiguous block in the middle of a file while each science radiometer is
    taken twice, bracketing it. That leaves each channel with an effective
    measurement time a few minutes either side of PBR-R's (SW -3 min, Total
    -1, LW +1, SSW +3), so a source drift biases the channels by different
    amounts and in opposite directions. Fitting the two passes separately and
    dividing cannot see that; gt removes it. Measured drift is around
    0.17 %/hr, significant at >3 sigma in 77% of files, which puts the
    channel-to-channel bias it causes at roughly 150 ppm - larger than most
    terms in the uncertainty budget.

    The two passes per channel are also what make gt identifiable at all. With
    a single pass each, a linear drift would be absorbed entirely into the
    r_i and the fit would be degenerate in that direction.

    Noise model: one sigma per detector, solved from the fit's own residuals by
    iteration. There is no per-point uncertainty to inherit, and the detectors
    differ in noise by orders of magnitude, so a pooled chi-square with equal
    weights would let one detector dominate for no reason. A by-product is that
    sigma per detector per file is a data quality metric in its own right: a
    frozen or out-of-band channel shows up immediately.

    Parameters
    ----------
    meas_data : DataFrame
        One file's points, from erf_log_file_read_and_parse. Needs
        detector_index, manip_x_mm, manip_y_mm, manip_pitch_deg,
        manip_yaw_deg, gps_time_s and l. Apply the monitor correction first.
    share_angular : bool, default True
        Share gp/gw across the science radiometers. Tested against per-channel
        angular gradients on all 783 files: dAIC is +12 in 100% of them, which
        is exactly the penalty for the 6 extra parameters, so the per-channel
        version fits no better at all. The angular gradient is a property of
        the source, not of the individual telescopes.
    fit_drift : bool, default True
        Include gt. Turn off to reproduce a no-drift field.
    fit_gradients : bool, default True
        Include gx, gy, gp, gw. Turn off to hold all four spatial/angular
        gradients at zero instead of fitting them - e.g. to show a "before"
        fit for comparison against the full fit's reduced residuals.
    n_iter : int, default 3
        Refit passes for the per-detector sigma.
    pbrr_index : int, default 0
        detector_index of the reference radiometer, whose r is fixed at 1.

    Returns
    -------
    SimpleNamespace with the fitted values and 1-sigma uncertainties (l0/l0_sd,
    gx/gx_sd, ...), r and r_sd as dicts keyed by detector_index, sigma and
    rel_sigma per detector, the reference time t_ref, fit statistics
    (redchi, aic, bic, nvarys, ndata), and the raw lmfit result.

    """
    from lmfit import Parameters, minimize

    # Copy the data and trim out bad points
    d = meas_data.loc[np.isfinite(meas_data["l"])].copy()
    det = d["detector_index"].astype(int).values

    # List of detector indexs in this measurement
    present = sorted(set(det))
    if pbrr_index not in present:
        raise ValueError(f"detector_index {pbrr_index} (the reference) is not in meas_data.")

    # List of channels other than PBR-R
    others = [i for i in present if i != pbrr_index]
    if not others:
        raise ValueError("Need at least one detector besides the reference.")

    # Extract the key arrays for fitting
    t_ref = d["gps_time_s"].mean()
    x  = d["manip_x_mm"].values
    y  = d["manip_y_mm"].values
    pi = d["manip_pitch_deg"].values
    w  = d["manip_yaw_deg"].values
    dt = (d["gps_time_s"].values - t_ref)/60.0
    z  = d["l"].values

    # Boolean array identifying the PBR-R points
    is_ref = det == pbrr_index

    # Create the parameter structure
    pars = Parameters()

    # Get the mean PBR-R measured radiance to set the radiance level
    l0_guess = z[is_ref].mean()
    if not np.isfinite(l0_guess) or l0_guess == 0:
        raise ValueError("Reference detector has no usable signal to set the scale.")
    pars.add('l0', value=l0_guess)
    if fit_gradients:
        pars.add('gx', value=0.0)
        pars.add('gy', value=0.0)
        if share_angular:
            pars.add('gp', value=0.0)
            pars.add('gw', value=0.0)
        else:
            for i in others:
                pars.add(f'gp_{i}', value=0.0)
                pars.add(f'gw_{i}', value=0.0)
    if fit_drift:
        pars.add('gt', value=0.0)

    # Calculate the mean ratio between the PBR-R radiance and each channel's measured radiance
    for i in others:
        pars.add(f'r_{i}', value=z[det == i].mean()/l0_guess)

    # This is the fitting model
    def model(p):
        # The fit parameters
        v = p.valuesdict()
        # This array is the ratio of a channel to the PBR-R
        r = np.where(is_ref, 1.0, 0.0)
        gp = np.zeros(len(d))
        gw = np.zeros(len(d))
        for i in others:
            m = det == i
            r[m] = v[f'r_{i}']
            # The angular terms act only where the manipulator tips the
            # detector, so they stay zero on the reference. The
            # angular terms are in the detector loop because
            # there's an option to vary them per channel. They're left at
            # their initialized zero here when fit_gradients is off, since
            # gp/gw (or gp_{i}/gw_{i}) were never added to pars in that case.
            if fit_gradients:
                gp[m] = v['gp'] if share_angular else v[f'gp_{i}']
                gw[m] = v['gw'] if share_angular else v[f'gw_{i}']
        # This captures the spatial/angular dependence of the radiance field.
        # gx/gy are likewise left at zero (not looked up in v) when
        # fit_gradients is off.
        gx_val = v['gx'] if fit_gradients else 0.0
        gy_val = v['gy'] if fit_gradients else 0.0
        shape = 1.0 + gx_val*x + gy_val*y + gp*pi + gw*w
        if fit_drift:
            # If fitting drift then let the field change with time
            shape = shape + v['gt']*dt
        return v['l0'] * r * shape

    # Start each detector's sigma at its own spread, then let the residuals
    # refine it. Constant within a detector: across one file the signal is
    # near constant apart from the small gradients, so there is no signal
    # dependence for a per-point sigma to capture.
    sigma = {}
    for i in present:
        # Get all the radiance points for this detector
        zi = z[det == i]
        # If there's more than three points calcualte the stddev of the 
        # radiance measurements, otherwise set it to 1. Set the minimum
        # value to 1e-30 in case the data is wonky
        sigma[i] = max(zi.std(), 1e-30) if len(zi) > 3 else 1.0

    # Perform the fit, recalculate the sigma from the residual between
    # the model and the radiance for each channel and refit. We do
    # this because the sigma is used for the weighting the radiance
    # points
    out = None
    for it in range(n_iter):
        sig = np.array([sigma[i] for i in det])
        out = minimize(lambda p: (z - model(p))/sig, pars, nan_policy='omit')
        pars = out.params
        fit_output = model(pars)
        res = z - fit_output
        for i in present:
            m = det == i
            if m.sum() > 3:
                sigma[i] = max(np.sqrt(np.mean(res[m]**2)), 1e-30)

    # All of the following code builds the returned data structure
    def val(name):
        p = out.params[name]
        return p.value, (p.stderr if p.stderr is not None else np.nan)

    res_ns = SimpleNamespace(result=out, t_ref=t_ref, fit_output=fit_output,
                             redchi=out.redchi, aic=out.aic, bic=out.bic,
                             nvarys=out.nvarys, ndata=out.ndata,
                             share_angular=share_angular, fit_drift=fit_drift,
                             fit_gradients=fit_gradients,
                             detectors=present, pbrr_index=pbrr_index)
    v, s = val('l0')
    res_ns.l0, res_ns.l0_sd = v, s
    if fit_gradients:
        for name in ('gx', 'gy'):
            v, s = val(name)
            setattr(res_ns, name, v)
            setattr(res_ns, name + '_sd', s)
        for name in (('gp', 'gw') if share_angular else ()):
            v, s = val(name)
            setattr(res_ns, name, v)
            setattr(res_ns, name + '_sd', s)
    else:
        for name in (('gx', 'gy') + (('gp', 'gw') if share_angular else ())):
            setattr(res_ns, name, 0.0)
            setattr(res_ns, name + '_sd', np.nan)
    if fit_drift:
        res_ns.gt, res_ns.gt_sd = val('gt')
    else:
        res_ns.gt, res_ns.gt_sd = 0.0, np.nan

    res_ns.r = {pbrr_index: 1.0}
    res_ns.r_sd = {pbrr_index: 0.0}
    for i in others:
        res_ns.r[i], res_ns.r_sd[i] = val(f'r_{i}')
        if not share_angular:
            if fit_gradients:
                setattr(res_ns, f'gp_{i}', out.params[f'gp_{i}'].value)
                setattr(res_ns, f'gw_{i}', out.params[f'gw_{i}'].value)
            else:
                setattr(res_ns, f'gp_{i}', 0.0)
                setattr(res_ns, f'gw_{i}', 0.0)

    res_ns.sigma = dict(sigma)
    res_ns.rel_sigma = {}
    for i in present:
        mean_i = z[det == i].mean()
        res_ns.rel_sigma[i] = sigma[i]/abs(mean_i) if mean_i else np.nan
    return res_ns


def weighted_mean_and_stddev(*, x, sd):

    x = np.asarray(x)
    sd = np.asarray(sd)
    n = len(x)

    if len(x) != len(sd):
        raise ValueError("Inequal elements of data points and weights.")

    # Make sure there are no zero elements of sd
    if np.all(sd != 0):
        # Calculate weights from stddev
        weights = sd ** -2

        # Calculate the weighted mean
        x_mn = (x*weights).sum()/weights.sum()
    
        # Calculate the weighted standard deviation
        x_sd = math.sqrt(1/weights.sum())
    else:
        # If any sd elements are zero then ignore the sd terms
        # and calc the mean and stddev from just x
        x_mn = np.mean(x)
        x_sd = np.std(x)

    return SimpleNamespace(x_mn=x_mn, x_sd=x_sd)


def analyze_osa207_spectrum(*, file,
                            wavelength_um_opa,
                            show_plot=False):
    """
    Loads and fits the laser wavelength from a OSA207 file
    """

    
    paths = load_config()
    sp = pd.read_csv(paths.erf_osa_spectrum_dir / file)

    if sp["signal"].sum() > 0 and len(sp) > 100:

        # Get an estimate for the FWHM, uses scaling found in the data
        fwhm_est = (3.142*(wavelength_um_opa) + 7.204*(wavelength_um_opa)**2)/1000

        # Locate the line in the data before choosing the fit window, instead of
        # centring the window on the OPA setpoint. The setpoint is not reliable
        # enough for that: across the 672 ERF spectra with a resolvable line the
        # measured line sits anywhere from -1.7 to +2.5 x fwhm_est from the
        # setpoint (median +0.3, so biased red but running both ways). Centring
        # on the setpoint clipped the line on ~21% of LST and ~27% of science
        # radiometer spectra, which left curve_fit fitting the rising flank
        # alone - biasing the fitted centre by ~15nm typically and up to ~200nm,
        # and failing outright (maxfev) on at least one file.
        #
        # Searched over +/-3 fwhm_est to cover both tails of that spread with
        # margin, wider than the +/-2 fwhm_est that actually gets fitted. The
        # trace is median-smoothed over ~1/8 of a FWHM first so that a single
        # noisy sample cannot be taken for the line: the OSA delivers a near
        # constant ~40 samples per fwhm_est across 1-14um, so this is a few
        # samples at every wavelength.
        search = sp.loc[(sp["wavelength[um]"] > wavelength_um_opa - 3*fwhm_est) &
                        (sp["wavelength[um]"] < wavelength_um_opa + 3*fwhm_est)]

        if len(search) > 20:
            # Sample spacing is uniform in wavenumber, so it is measured locally
            # rather than assumed. Note the OSA writes its trace in DESCENDING
            # wavelength order, hence the abs().
            spacing = np.abs(np.diff(search["wavelength[um]"].values)).mean()
            n_smooth = max(3, int(round(0.125*fwhm_est/spacing)))
            smoothed = search["signal"].rolling(n_smooth, center=True, min_periods=1).median()
            smoothed = smoothed.values
            i_peak = int(np.argmax(smoothed))
            wl_line = search["wavelength[um]"].iloc[i_peak]

            # Does that peak actually belong to a line? Over the search range a
            # real line falls back below half its height on BOTH sides of the
            # peak, however broad or absorption-carved it is. A monotonic
            # background ramp - which is all the OSA sees beyond ~11um - never
            # does on its rising side: its peak sits at the edge of the search
            # range, not inside it.
            #
            # This is the test that separates them, and peak height is not: on
            # the 12.79um files the ramp reaches 36-49x the background scatter
            # while containing no line at all. Nor does asking how much signal
            # remains at the window edges work - a genuine CO2-carved line at
            # 4.077um leaves 37% of its peak at the blue window edge, against
            # 90% for a ramp, so any threshold there either rejects real broad
            # lines or accepts ramps.
            #
            # Index order is irrelevant here (the OSA trace is descending), only
            # that the signal comes back down on each side of the peak.
            base_level = np.median(smoothed)
            half_level = base_level + 0.5*(smoothed[i_peak] - base_level)
            below_half = smoothed < half_level
            line_bracketed = bool(below_half[:i_peak].any() and below_half[i_peak+1:].any())
        else:
            # Too little to search. Fall back to the setpoint, and treat it as
            # having no line rather than asserting one is there.
            wl_line = wavelength_um_opa
            line_bracketed = False

        # Integration window for the moments, and the annulus the background is
        # measured in. Both are placed on the located line, not the setpoint.
        #
        # Half-width 1.5 x fwhm_est, chosen by measuring the tradeoff over all
        # 693 OSA spectra: a wider window makes the second moment increasingly
        # sensitive to the background (95th-percentile width error per 1-sigma
        # baseline error is 7.4% at 2.0 fwhm_est but 4.2% at 1.5), while a
        # narrower one starts truncating the line (width falls to 0.95 of its
        # 2.0 value at 1.0 fwhm_est, 5th percentile 0.76). At 1.5 the centroid
        # sits within 0.05nm (median) of its 2.0 value, so this trades
        # essentially no accuracy for half the background sensitivity. The
        # truncation does bias the width low by ~1.1% for a Gaussian line, but
        # it is the same window at every wavelength, so that bias is consistent
        # across the SRF rather than varying point to point.
        half_width = 1.5*fwhm_est

        wl = sp["wavelength[um]"].values
        sig = sp["signal"].values

        in_window  = np.abs(wl - wl_line) < half_width
        in_annulus = ((np.abs(wl - wl_line) > half_width + 0.25*fwhm_est) &
                      (np.abs(wl - wl_line) < half_width + 1.50*fwhm_est))

        if in_window.sum() > 20 and in_annulus.sum() > 8:

            w, s = wl[in_window], sig[in_window]

            # Background from a LINEAR fit across both wings, not a constant.
            # The OSA trace carries a sloping background that grows strong beyond
            # ~9um, and a slope biases the centroid where a constant offset
            # largely cancels by symmetry. Fitting the slope cuts the
            # 95th-percentile centroid sensitivity from 8.3nm per 1-sigma
            # baseline error down to 1.3nm. The annulus lies outside the
            # integration window, where a line of this width has fallen below
            # ~0.02% of its peak, so it is line-free.
            base_coeffs = np.polyfit(wl[in_annulus], sig[in_annulus], 1)
            baseline = np.polyval(base_coeffs, w)

            # Noise level, taken as the scatter about that background fit
            noise_sd = np.std(sig[in_annulus] - np.polyval(base_coeffs, wl[in_annulus]))

            # Clipped at zero so that background-level noise cannot contribute
            # negative weight to the moments.
            y = np.clip(s - baseline, 0, None)

            # Reject traces with no resolvable laser line before taking any
            # moments. Two tests are needed, because amplitude alone does not
            # separate them: beyond ~11um the OSA sees no line at all, only a
            # rising thermal background, and that ramp can carry a large
            # peak-to-noise ratio (36-49 on the 12.79um files) while looking
            # nothing like a line.
            #
            #   amp_over_noise - peak height above background in units of the
            #                    background scatter. Real lines sit at ~620
            #                    (median) and only 1% fall below 13, so a
            #                    threshold of 10 catches only genuinely weak
            #                    traces.
            #   line_bracketed - whether the peak comes back down on both sides
            #                    within the search range, established above.
            #                    This is what rules out a background ramp.
            #
            # A rejected trace returns a zero centre, which leaves the caller's
            # wavelength_source at its sentinel so that the grating spectrum can
            # take over, and level_01 drops the row if it cannot.
            amp_over_noise = y.max()/noise_sd if noise_sd > 0 else np.inf

            if y.sum() > 0 and amp_over_noise >= 10 and line_bracketed:

                # Centroid and second moment, in place of a Gaussian fit. The
                # laser output itself is Gaussian, but atmospheric absorption on
                # the path to the spectrometer (mainly H2O and CO2) carves
                # structure into the line, and that structure is not symmetric.
                # A Gaussian misdescribes those profiles badly - the fit residual
                # reaches 25-30% of the line amplitude in the 2.7, 4.3 and 6.3um
                # bands, against ~1% in the windows between them - and the fitted
                # centre then disagrees with the centroid by up to hundreds of
                # nm. The moments assume no shape, so they describe the absorbed
                # line as actually measured.
                center = float((w*y).sum()/y.sum())
                variance = float((y*(w - center)**2).sum()/y.sum())
                fwhm = 2.3548*math.sqrt(max(variance, 0.0))

                # Same quantity as center now; kept as its own field so callers
                # that ask for the centroid explicitly still get it.
                wavelength_centroid_um = center

                # Peak height above background. Still called gaussian_amp
                # because level_01 filters on the wavelength_gauss_amp column
                # this feeds - there is no Gaussian fit here any more.
                amplitude = float(y.max())

                # Uncertainties, two contributions added in quadrature:
                #
                #   noise    - per-point scatter propagated through the moment
                #              estimators. For c = sum(w y)/sum(y) we have
                #              dc/dy_i = (w_i - c)/sum(y), and for the variance
                #              dV/dy_i = ((w_i - c)^2 - V)/sum(y).
                #   baseline - the background level is itself uncertain by
                #              noise_sd, and that shifts every point coherently.
                #              Taken as the larger of the two +/-1 sigma shifts,
                #              since it is a systematic rather than random error.
                sum_y = y.sum()
                center_noise_sd = noise_sd*math.sqrt(((w - center)**2).sum())/sum_y
                var_noise_sd = noise_sd*math.sqrt((((w - center)**2 - variance)**2).sum())/sum_y
                fwhm_noise_sd = (2.3548*var_noise_sd/(2*math.sqrt(variance))
                                 if variance > 0 else np.nan)

                center_base_sd, fwhm_base_sd = 0.0, 0.0
                for db in (-noise_sd, noise_sd):
                    y2 = np.clip(s - (baseline + db), 0, None)
                    if y2.sum() <= 0:
                        continue
                    c2 = float((w*y2).sum()/y2.sum())
                    v2 = float((y2*(w - c2)**2).sum()/y2.sum())
                    center_base_sd = max(center_base_sd, abs(c2 - center))
                    fwhm_base_sd = max(fwhm_base_sd,
                                       abs(2.3548*math.sqrt(max(v2, 0.0)) - fwhm))

                center_sd = math.sqrt(center_noise_sd**2 + center_base_sd**2)
                fwhm_sd = math.sqrt(fwhm_noise_sd**2 + fwhm_base_sd**2)

            else:
                # Background subtraction left nothing positive, so there is no
                # line here to take moments of. Note this is only the numerical
                # guard - a significance-based rejection of line-free traces
                # (and the OSA/grating crossover that goes with it) is still
                # outstanding.
                center = 0
                wavelength_centroid_um = 0
                fwhm = 0
                amplitude = 0
                center_sd = np.nan
                fwhm_sd = np.nan

            if show_plot:
                plt.plot(wl, sig, label='Data', color='blue')
                plt.plot(w, baseline, label='Background', linestyle=':', color='green')
                plt.axvspan(wl_line - half_width, wl_line + half_width,
                            color='orange', alpha=0.2, label='Moment window')
                if center:
                    plt.axvline(center, color='red', linestyle='--', linewidth=1,
                                label=f'Centroid {center:.4f} um')
                plt.xlim(wl_line - 4*fwhm_est, wl_line + 4*fwhm_est)
                plt.title(file)
                plt.ylabel("Signal")
                plt.xlabel("Wavelength [um]")
                plt.axhline(0, color='black', linewidth=0.5, zorder=1)
                plt.legend()
                plt.savefig(paths.figure_dir / (file[0:-4] + '.png'), bbox_inches="tight", dpi=200)
                plt.show()

        else:
            # Too few points in the window or the annulus to take moments.
            center = 0
            wavelength_centroid_um = 0
            fwhm = 0
            amplitude = 0
            center_sd = np.nan
            fwhm_sd = np.nan

        # RSS a 1ppm uncertainty with the measurement uncertainty
        center_sd = math.sqrt(center_sd**2 + (1e-6*center)**2)

        return SimpleNamespace(wavelength_center_um=center,
                               wavelength_center_um_sd=center_sd,
                               wavelength_centroid_um=wavelength_centroid_um,
                               wavelength_fwhm_um=abs(fwhm),
                               wavelength_fwhm_um_sd=fwhm_sd,
                               gaussian_amp=amplitude)

    else:

        # NaN rather than 0 for the uncertainties: there is no fit here, and a
        # zero uncertainty reads as "perfectly known", which would give this
        # row infinite weight anywhere the value is used as one.
        return SimpleNamespace(wavelength_center_um=0,
                               wavelength_center_um_sd=np.nan,
                               wavelength_centroid_um=0,
                               wavelength_fwhm_um=0,
                               wavelength_fwhm_um_sd=np.nan,
                               gaussian_amp=0)

# 2nd order polynomial (highest degree first, wavelength in nm) fit to the
# CCS200's wavelength error against the Ocean Optics HG-1 line list, used by
# analyze_ccs200_spectrum to correct measured line centers. Generated by
# analyze_ccs200_calibration_spectrum() - regenerate and re-paste this if that
# calibration spectrum, the HG-1 line list, or the fit method there changes.
CCS200_WAVELENGTH_ERROR_POLY_NM = np.array([-2.95443166e-06, 3.65271257e-03, -1.58205218e+00])

# 1-sigma CCS200 wavelength uncertainty (nm), estimated from the scatter of
# the HG-1 lines around CCS200_WAVELENGTH_ERROR_POLY_NM rather than from the
# individual (unreliable) Gaussian-fit center_sd values. Generated by
# analyze_ccs200_calibration_spectrum() - regenerate alongside the polynomial
# above.
CCS200_WAVELENGTH_ERROR_SD_NM = 0.1255

# Type B standard uncertainty on the IR (OSA207-measured) wavelength scale,
# FRACTIONAL - the shift it describes is wl*(1 + f), so the uncertainty in um
# is this times the wavelength. This is a COMMON term - one unknown scale error
# applied to every OSA-measured wavelength at once - so it does not average
# down over measurements and must not be folded into u_resp_noise or any other
# random term. It is registered as class "Wavelength" with
# session_variable = 0, which is the combination that means "varies with
# wavelength, does not average down".
#
# Fractional rather than a fixed number of nm because that is what the two
# independent anchors say - see the wavelength-offset discussion in
# science_radiometer_srf_campaign_correct. In brief: the steep band edges
# fix the offset at 4-5um, the SW etalon fringe fixes it over 0.4-3.0um, and a
# constant shift in um is excluded against those two (chi2 57.6 on three
# campaigns, against 5.0 for fractional).
#
# Why there is a residual at all. The per-campaign offsets fitted in
# science_radiometer_analyze_level_02 are referenced to the unweighted
# campaign mean, which removes the campaign-to-campaign differences but cannot
# establish where the mean itself sits. Three contributions, in quadrature,
# with sigma_f = 2.43e-3 the sample sd of the three per-campaign fractional
# offsets (+2.78/-1.11/-1.68 e-3):
#
#   1.85e-3  uncertainty in the reference: sigma_f/sqrt(3) times
#            t(2 dof, 68.27%) = 1.32, the t factor because a sigma from three
#            samples is itself poorly known.
#   0.33e-3  model consistency, the rms disagreement between the channels'
#            independent estimates and the shared value applied, over the
#            eight channel-campaign combinations with enough steep points.
#   2.43e-3  Delta, the FIXED part of the OSA-versus-radiometer sampling
#            difference. If the OSA pickoff systematically samples a different
#            part of the spatially chirped OPA beam than the radiometer sees,
#            every campaign shares that error and no averaging over campaigns
#            touches it. Nothing in the ERF dataset bounds it, so it is taken
#            equal to sigma_f: there is no reason to believe the systematic
#            sampling offset is smaller than the campaign-to-campaign scatter
#            it produces. This is the conservative judgement in the term and it
#            dominates.
#
# This is NOT a statement about the OSA207's wavelength axis, which is sound. A
# 2.7e-3 error in that axis would be 11.5nm at 4.26um, and the CO2 nu3
# band-centre fiducial at 2349.14cm-1 - a real P/R gap, because the Q branch of
# a Sigma-Sigma transition is forbidden - puts the axis right to ~1nm absolute
# there and campaign-stable to <=0.5nm, with H2O structure at 2.7 and 6.3um
# agreeing. The term covers which part of the beam reached which instrument,
# not what the spectrometer did with the light it got.
#
# Above the 4.6um anchor the fractional form is EXTRAPOLATION: there is no
# steep edge past 5.8um and no second ruler, so nothing here is measured beyond
# that point. It is carried on rather than frozen as the conservative choice,
# which means the term grows to 25nm at 8um and 40nm at 13um and dominates LW's
# in-band budget. A measurement above 5um could only reduce it.
#
# What would shrink it: an independent measurement of the filter edge positions
# would bound Delta directly, and a beam-translation experiment at ~4.5um (scan
# an aperture across the beam, watch the OSA centroid) would measure the chirp
# that is presumed to cause it.
IR_WAVELENGTH_SCALE_U_FRAC = 3.07e-3

# Longest OPA setpoint (um) at which the OSA207 is asked for a wavelength;
# beyond this the grating spectrometer is used instead. Set where the OSA is
# provably dead rather than merely suspect: by the line test in
# analyze_osa207_spectrum, 0 of 13 traces with a setpoint of 11-12um fail, while
# 9 of 9 fail over 12-13um and 11 of 11 above 13um - past 12um the laser line
# has moved beyond the OSA's ~13um sensitivity limit and the trace holds nothing
# but a rising thermal background. The previous crossover of 13um therefore had
# the OSA fitting that background on every file between 12 and 13um, returning
# centres ~1um too long.
#
# This is a coarse backstop on the setpoint, not the real arbiter - the setpoint
# runs up to +1.2um from the measured line at this end of the range, so it only
# approximates which files the OSA can handle. The per-file line test does the
# actual work, and files it rejects fall back to the grating as well. Tighten
# this to 11.0 if the OSA's wavelength scale near its 13um limit turns out to be
# less trustworthy than the grating, which would cost the 13 files at 11-12um
# their OSA precision.
OSA_MAX_WAVELENGTH_UM = 12.0

def analyze_ccs200_calibration_spectrum():
    """
    This just loads and analyzes a wavelength calibration measurement taken of the
    Ocean Optics HG-1 with the CCS200
    """

    paths = load_config()

    file = 'CCS200_Spectrum_1464724756_OceanOpticsHG1.csv'
    sp = tidy_up_header(pd.read_csv(paths.analysis_dir / file))

    # The list of HG-1 lines
    wl_list = [253.652,
               313.155,
               365.015,
               404.656,
               435.833,
               546.074,
               696.543,
               706.722,
               727.294,
               738.393,
               763.511,
               772.376,
               794.818,
               826.452,
               852.144,
               912.297]

    # Create a column to hold the data fit
    sp["fit"] = 0.0*sp["signal"]

    fit_data = []
    window_width = 4
    for wl in wl_list:

        # Calculate the width of the window to use
        wl_min = wl - window_width
        wl_max = wl + window_width

        # Trim the data        
        sp_tmp = sp.loc[(sp["wavelength_nm"] > wl_min) & (sp["wavelength_nm"] < wl_max) & (sp["signal"] < 4.95)].copy()

        # Calculate the centroid
        wavelength_centroid_nm = (sp_tmp["wavelength_nm"]*sp_tmp["signal"]).sum()/sp_tmp["signal"].sum()

        # initial guesses for [amplitude, center, fwhm, offset]
        p0 = [sp_tmp["signal"].max(), wavelength_centroid_nm, 0.9]

        if wl > 650:
            # Trim the bottom range of the data to trim out the asymmetric trails at long wavelengths
            sp_tmp = sp_tmp.loc[sp["signal"] > 0.3*p0[0]]

        popt, pcov = curve_fit(gaussian_no_offset, sp_tmp["wavelength_nm"], sp_tmp["signal"], p0=p0)
        amplitude, center, fwhm = popt

        # Calculate the corrected center using the fit created in this function
        # This is to verify that the calculation is correct
        center_corr = center - np.polyval(CCS200_WAVELENGTH_ERROR_POLY_NM, center)

        # 1-sigma uncertainties, same parameter order as p0. No per-point
        # uncertainty is supplied for the spectrometer trace, so curve_fit's
        # default absolute_sigma=False applies: pcov is scaled by the reduced
        # chi-square, taking the error scale from the residuals. Same convention
        # as absolute_sd=False in fit_plane.
        amplitude_sd, center_sd, fwhm_sd = np.sqrt(np.diag(pcov))

        sp["fit"] = sp["fit"] + gaussian_no_offset(sp["wavelength_nm"], amplitude, center, fwhm)

        fit_data.append({
            "wavelength_nm":                wl,
            "fit_amp":                      amplitude,
            "fit_center":                   center,
            "fit_center_corr":              center_corr,
            "fit_fwhm":                     fwhm,
            "fit_amp_sd":                   amplitude_sd,
            "fit_center_sd":                center_sd,
            "fit_fwhm_sd":                  fwhm_sd,
            "wavelength_error_nm":          center - wl,
            "corr_wavelength_error_nm":     center_corr - wl})
        
    # Change to a dataframe
    fit_data = pd.DataFrame(fit_data)

    # Fit a 2nd order polynomial to the wavelength error as a function of
    # wavelength. Unweighted: the formal fit_center_sd values are much
    # smaller than the actual point-to-point scatter around a smooth trend
    # (e.g. the 313nm line has center_sd=0.05nm but sits ~0.25nm off a
    # weighted fit), so they don't reflect which lines are actually more
    # trustworthy - weighting by them just lets whichever line happens to
    # have the smallest formal error dominate the fit.
    poly_order = 2
    poly_coeffs = np.polyfit(fit_data["wavelength_nm"], fit_data["wavelength_error_nm"], deg=poly_order)
    print()
    print()
    print("CCS200 wavelength error polynomial coefficients (highest degree first):")

    # Calculate the polynomial
    fit_data["poly_fit"] = np.polyval(poly_coeffs, fit_data["wavelength_nm"])

    print(poly_coeffs)
    print("This is the value hardcoded as CCS200_WAVELENGTH_ERROR_POLY_NM - "
          "regenerate and re-paste it there if this calibration spectrum, the "
          "HG-1 line list, or the fit method above changes.")

    # Estimate the CCS200's wavelength uncertainty from how much the lines
    # scatter around the polynomial, rather than from the (unreliable, see
    # above) individual fit_center_sd values. Divide the sum of squared
    # residuals by (n_lines - n_poly_params) rather than (n_lines - 1): this
    # is the reduced-chi-square convention (same as curve_fit's
    # absolute_sigma=False elsewhere in this file) and accounts for the fact
    # that the polynomial itself already absorbed some of the scatter.
    residual_nm = fit_data["wavelength_error_nm"] - fit_data["poly_fit"]
    dof = len(residual_nm) - (poly_order + 1)
    wavelength_error_sd_nm = np.sqrt((residual_nm**2).sum() / dof)
    print(f"CCS200 wavelength error std (nm), hardcoded as CCS200_WAVELENGTH_ERROR_SD_NM: "
          f"{wavelength_error_sd_nm:.4f}")
    print()
    print()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax1.plot(sp["wavelength_nm"],sp["signal"],linewidth=0.5,label="Data")
    ax1.plot(sp["wavelength_nm"],sp["fit"],linewidth=0.5,label="Fit")
    ax1.set_xlabel('CCS200 Wavelength [nm]')
    ax1.set_ylabel('Signal')
    ax1.set_xlim(200,1000)
    ax1.set_ylim(-0.1,6)

    wl_grid = np.linspace(200, 1000, 500)
    ax2.plot([200,1000],[0,0],color='black', zorder=1, linewidth=1)
    fit_str = f"2nd order fit, residual = {wavelength_error_sd_nm:.4f} nm"
    ax2.plot(wl_grid, np.polyval(poly_coeffs, wl_grid), color='blue', zorder=2, linewidth=1, label=fit_str)
    ax2.scatter(fit_data["wavelength_nm"], fit_data["wavelength_error_nm"], color='red', s=10, label='Uncorrected Centers')
    ax2.scatter(fit_data["wavelength_nm"], fit_data["corr_wavelength_error_nm"], color='green', s=10, label='Corrected Centers')        
    ax2.set_xlabel('CCS200 Wavelength [nm]')
    ax2.set_ylabel('CCS200 Error [nm]')
    ax2.set_xlim(200,1000)
    ax2.set_ylim(-2,2)
    ax2.legend()

    fig.savefig(paths.figure_dir / ('erf_ccs200_wavelength_calibration.png'), bbox_inches="tight", dpi=200)
    fig.show()


def analyze_ccs200_spectrum(*, file,
                            wavelength_um_opa,
                            show_plot=False):
    """
    Loads and fits the laser wavelength from a CCS200 file
    """

    paths = load_config()
    sp = pd.read_csv(paths.erf_ccs_spectrum_dir / file, header=None, names=["wavelength_nm", "signal"])

    if sp["signal"].sum() > 0:

        # Convert the wavelength to nm
        wavelength_nm_opa = wavelength_um_opa*1e3

        # Calculate the width of the window to use
        window_width = 7 + 20*wavelength_um_opa

        wl_min = wavelength_nm_opa - window_width
        wl_max = wavelength_nm_opa + window_width

        # Trim the data        
        sp = sp.loc[(sp["wavelength_nm"] > wl_min) & (sp["wavelength_nm"] < wl_max)]

        # Calculate the centroid
        wavelength_centroid_nm = (sp["wavelength_nm"]*sp["signal"]).sum()/sp["signal"].sum()

        # initial guesses for [amplitude, center, fwhm, offset]
        p0 = [sp["signal"].max(), wavelength_centroid_nm, 1.0, 0.0]

        popt, pcov = curve_fit(gaussian, sp["wavelength_nm"], sp["signal"], p0=p0)
        amplitude, center, fwhm, offset = popt

        # 1-sigma uncertainties, same parameter order as p0. No per-point
        # uncertainty is supplied for the spectrometer trace, so curve_fit's
        # default absolute_sigma=False applies: pcov is scaled by the reduced
        # chi-square, taking the error scale from the residuals. Same convention
        # as absolute_sd=False in fit_plane.
        amplitude_sd, center_sd, fwhm_sd, offset_sd = np.sqrt(np.diag(pcov))

        y_fit = gaussian(sp["wavelength_nm"], amplitude, center, fwhm, offset)

        # Correct for the CCS200's own wavelength calibration error, fit
        # against the Ocean Optics HG-1 line list in
        # analyze_ccs200_calibration_spectrum. The true wavelength isn't
        # known here (that's what we're measuring), so the polynomial is
        # evaluated at the measured center as the best available estimate -
        # it's slowly varying, so this is a good approximation.
        center = center - np.polyval(CCS200_WAVELENGTH_ERROR_POLY_NM, center)

        # RSS the CCS200's own wavelength calibration uncertainty (the
        # scatter of the HG-1 lines around CCS200_WAVELENGTH_ERROR_POLY_NM,
        # see analyze_ccs200_calibration_spectrum) with this fit's own
        # uncertainty.
        center_sd = math.sqrt(center_sd**2 + CCS200_WAVELENGTH_ERROR_SD_NM**2)

        if show_plot: 
            #plt.ion()                    # non-blocking, like IDL — script keeps running
            #fig, ax = plt.subplots()
            plt.plot(sp["wavelength_nm"], sp["signal"], label='Data', color='blue')
            plt.plot(sp["wavelength_nm"], y_fit, label='Fit', linestyle='--', color='red')
            plt.title(file)
            plt.axhline(0, color='black', linewidth=0.5, zorder=1)
            plt.ylabel("Signal")
            plt.xlabel("Wavelength [nm]")
            plt.legend()
            plt.savefig(paths.figure_dir / (file[0:-4] + '.png'), bbox_inches="tight", dpi=200)
            plt.show()
            
        return SimpleNamespace(wavelength_center_um=center/1e3,
                               wavelength_center_um_sd=center_sd/1e3,
                               wavelength_centroid_um=wavelength_centroid_nm/1e3,
                               wavelength_fwhm_um=abs(fwhm)/1e3,
                               wavelength_fwhm_um_sd=fwhm_sd/1e3,
                               gaussian_amp=amplitude)
    else:
        # NaN rather than 0 for the uncertainty: there is no fit here, and a
        # zero uncertainty reads as "perfectly known", which would give this
        # row infinite weight anywhere the value is used as one.
        return SimpleNamespace(wavelength_center_um=0,
                               wavelength_center_um_sd=np.nan,
                               wavelength_centroid_um=0,
                               wavelength_fwhm_um=0,
                               wavelength_fwhm_um_sd=np.nan,
                               gaussian_amp=0)

def estimate_grating_spectrum_wavelength_error():
    """
    Estimates the error in the wavelength for the grating spectrum
    """
    
    # Load the data
    paths = load_config()
    filename = "Libera_ERF_analysis_summary_level_01.csv"
    df = tidy_up_header(pd.read_csv(paths.analysis_dir / filename))

    # Sort by wavelength
    df.sort_values(by="wavelength_um", inplace=True)

    # Get the data points with a grating wavelength
    df = df.loc[(df["wavelength_um_grating"] != 0) & (df["wavelength_source"] == 2)]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax1.plot([6,15],[6,15],color='black', zorder=1, linewidth=1)
    ax1.scatter(df["wavelength_um"], df["wavelength_um_grating"], color='red', s=8, zorder=2)
    ax1.set_title('ERF Grating Spectrometer')
    ax1.set_xlabel('OSA Wavelength [um]')
    ax1.set_ylabel('Grating Wavelength [um]')
    ax1.set_xlim(6,15)
    ax1.set_ylim(6,15)

    df["wl_error"] = df["wavelength_um_grating"] - df["wavelength_um"]
    ax2.plot([6,15],[0,0],color='black', zorder=1, linewidth=1)
    ax2.scatter(df["wavelength_um"], df["wl_error"], color='red', s=8, zorder=2)
    ax2.set_title('ERF Grating Spectrometer Error')
    ax2.set_xlabel('OSA Wavelength [um]')
    ax2.set_ylabel('Grating Wavelength Difference [um]')
    ax2.set_xlim(6,15)
    ax2.set_ylim(-1,1)

    # Calculate the error for wavelength greater than 10um
    df = df.loc[df["wavelength_um"] >= 10]
    print(f"Estimated Grating Wavelength Error = {(df['wl_error'].std()):0.3f}um")

def analyze_grating_spectrum(*, file,
                            wavelength_um_opa,
                            show_plot=False):
    
    paths = load_config()
    #sp = pd.read_csv(paths.erf_grating_spectrum_dir / file, header=None, names=["wavelength_nm", "signal"])
    sp = tidy_up_header(pd.read_csv(paths.erf_grating_spectrum_dir / file))

    if sp["signal"].sum() > 0:

        # Get an estimate for the FWHM, uses scaling found in the data
        fwhm_est = (3.142*(wavelength_um_opa) + 7.204*(wavelength_um_opa)**2)/1000

        # Get a range of 2 FWHM around the peak
        wl_min = wavelength_um_opa - 4*fwhm_est
        wl_max = wavelength_um_opa + 4*fwhm_est

        # Trim the data
        sp = sp.loc[(sp["wavelength_um"] > wl_min) & (sp["wavelength_um"] < wl_max)]

        # Calculate the centroid
        wavelength_centroid_um = (sp["wavelength_um"]*sp["signal"]).sum()/sp["signal"].sum()

        # initial guesses for [amplitude, center, fwhm, offset]
        p0 = [sp["signal"].max(), wavelength_centroid_um, 1.0, 0.0]

        popt, pcov = curve_fit(gaussian, sp["wavelength_um"], sp["signal"], p0=p0)
        amplitude, center, fwhm, offset = popt

        # 1-sigma uncertainties, same parameter order as p0. curve_fit's default
        # absolute_sigma=False applies, so pcov is scaled by the reduced
        # chi-square and the error scale comes from the residuals - no per-point
        # uncertainty is available for the grating trace.
        amplitude_sd, center_sd, fwhm_sd, offset_sd = np.sqrt(np.diag(pcov))

        # RSS a 0.2um uncertainty with the fit uncertainty
        grating_error = 0.2
        center_sd = math.sqrt(center_sd**2 + grating_error**2)

        y_fit = gaussian(sp["wavelength_um"], amplitude, center, fwhm, offset)

        if show_plot:
            #plt.ion()                    # non-blocking, like IDL — script keeps running
            #fig, ax = plt.subplots()
            plt.plot(sp["wavelength_um"], sp["signal"], label='Data', color='blue')
            plt.plot(sp["wavelength_um"], y_fit, label='Fit', linestyle='--', color='red')
            plt.title(file)
            plt.ylabel("Signal")
            plt.xlabel("Wavelength [um]")
            plt.legend()
            plt.savefig(paths.figure_dir / (file[0:-4] + '.png'), bbox_inches="tight", dpi=200)
            plt.show()
        return SimpleNamespace(wavelength_center_um=center,
                               wavelength_center_um_sd=center_sd,
                               wavelength_centroid_um=wavelength_centroid_um,
                               wavelength_fwhm_um=abs(fwhm),
                               wavelength_fwhm_um_sd=fwhm_sd,
                               gaussian_amp=amplitude)
    else:
        # NaN rather than 0 for the uncertainties: there is no fit here, and a
        # zero uncertainty reads as "perfectly known", which would give this
        # row infinite weight anywhere the value is used as one.
        return SimpleNamespace(wavelength_center_um=0,
                               wavelength_center_um_sd=np.nan,
                               wavelength_centroid_um=0,
                               wavelength_fwhm_um=0,
                               wavelength_fwhm_um_sd=np.nan,
                               gaussian_amp=0)

def get_scirad_conversions(*, fpe_name):

    match fpe_name:
        case 'emfpe':
            # This is the original EM FPE configuration with the initial flight detector 
            # boards (SN01-SN04), started July 2024 to Feb 2025
            conv_file = 'libera_science_radiometer_conversions_emfpe_v4.csv'       
        case 'emfpeb': 
            # This is the short-lived testing configuration with the final flights detector
            # boards (SN05-SN08) and the EM FPE 
            conv_file = 'libera_science_radiometer_conversions_emfpeb_v0.csv'       
        case 'fmfpe':
            # The flight FPE or CFPE. We're on the v03 conversions, this adds the offset
            # of the substrate temperature relative to the telescope temperature
            conv_file = 'libera_flight_radiometer_conversions_v03.csv'       
    
    paths = load_config()
    conv = pd.read_csv(paths.conversions_and_calibrations_dir / conv_file)
    conv = conv.set_index("channel_name")

    return conv

def load_erf_logfile(*, file, 
                     ap_dia_mm=10, 
                     wavelength_um=1,
                     fpe_name='fmfpe'): 
    """
    Loads an ERF file and converts the readings to radiance

    Inputs:
        file                ERF log filename
        ap_dia_mm           ERF integrating sphere aperture, 10, 12, or 13
        wavelength_um       The laser wavelength [um]
        fpe_name            Which FPE we're using: fmfpe, emfpe, emfpeb, or fmcfpe
    """

    # Load the ERF data
    paths = load_config()
    df = tidy_up_header(pd.read_csv(paths.erf_log_file_dir / file))

    # Get the correct telemetry conversions for the FPE housekeeping
    if fpe_name == 'fmfpe':
        coeffs = pd.read_csv(paths.conversions_and_calibrations_dir / "fpe_adc_hk_conversions.csv", header=None)
    elif fpe_name in ('emfpe', 'emfpeb'):
        coeffs = pd.read_csv(paths.conversions_and_calibrations_dir / "emfpe_adc_hk_conversions.csv", header=None)
    elif fpe_name == 'fmcfpe':
        coeffs = pd.read_csv(paths.conversions_and_calibrations_dir / "cfpe_adc_hk_conversions.csv", header=None)
    else:
        coeffs = pd.read_csv(paths.conversions_and_calibrations_dir / "fpe_adc_hk_conversions.csv", header=None)

    #Convert the rad bench temp
    df["sci_rad_bench_temp_c"] = coeffs.iloc[7,0] + \
                                 coeffs.iloc[7,1]*df["sci_rad_bench_temp_dn"] + \
                                 coeffs.iloc[7,2]*df["sci_rad_bench_temp_dn"]**2 + \
                                 coeffs.iloc[7,3]*df["sci_rad_bench_temp_dn"]**3

    #Convert the ltz1000 temp
    df["sci_rad_lt1000_temp_c"] = coeffs.iloc[8,0] + \
                                  coeffs.iloc[8,1]*df["sci_rad_lt1000_temp_dn"] + \
                                  coeffs.iloc[8,2]*df["sci_rad_lt1000_temp_dn"]**2 + \
                                  coeffs.iloc[8,3]*df["sci_rad_lt1000_temp_dn"]**3

    #Convert the CAES board temp
    df["sci_rad_caes_temp_c"] = coeffs.iloc[9,0] + \
                                coeffs.iloc[9,1]*df["sci_rad_caes_temp_dn"] + \
                                coeffs.iloc[9,2]*df["sci_rad_caes_temp_dn"]**2 + \
                                coeffs.iloc[9,3]*df["sci_rad_caes_temp_dn"]**3

    #Convert the heater drive voltage
    df["sci_rad_heater_adc_v"] = coeffs.iloc[6,0] + \
                                 coeffs.iloc[6,1]*df["sci_rad_heater_adc_dn"] + \
                                 coeffs.iloc[6,2]*df["sci_rad_heater_adc_dn"]**2 + \
                                 coeffs.iloc[6,3]*df["sci_rad_heater_adc_dn"]**3

    # Convert science radiometer DN to radiance for the four science radiometers:
    if fpe_name in ('emfpe', 'emfpeb', 'fmfpe'):

        # Get the median dark active PWM level used as part of the conversion
        active_dc0 = df["sci_rad_pwm0_dn"].median()/12500.0
        active_dc1 = df["sci_rad_pwm1_dn"].median()/12500.0
        active_dc2 = df["sci_rad_pwm2_dn"].median()/12500.0
        active_dc3 = df["sci_rad_pwm3_dn"].median()/12500.0

        # Next convert the raw DN values into power   

        # Load the conversion coefficients 
        conv = get_scirad_conversions(fpe_name=fpe_name)

        # Calculate nW per DN for channel 0/SW
        df["nw_per_dn0"] = conv.loc["sw","nw_per_dn0"] + \
                        conv.loc["sw","nw_per_dn1"]*(df["sci_rad_bench_temp_c"]  - conv.loc["sw","nw_per_dn_t0"]) + \
                        conv.loc["sw","nw_per_dn2"]*((df["sci_rad_bench_temp_c"] - conv.loc["sw","nw_per_dn_t0"])**2) + \
                        conv.loc["sw","nw_per_dn3"]*active_dc0 + \
                        conv.loc["sw","nw_per_dn4"]*(active_dc0**2) + \
                        conv.loc["sw","nw_per_dn5"]*(df["sci_rad_bench_temp_c"] - conv.loc["sw","nw_per_dn_t0"])*active_dc0

        # Calculate nW per DN for channel 1/Total
        df["nw_per_dn1"] = conv.loc["total","nw_per_dn0"] + \
                        conv.loc["total","nw_per_dn1"]*(df["sci_rad_bench_temp_c"]  - conv.loc["total","nw_per_dn_t0"]) + \
                        conv.loc["total","nw_per_dn2"]*((df["sci_rad_bench_temp_c"] - conv.loc["total","nw_per_dn_t0"])**2) + \
                        conv.loc["total","nw_per_dn3"]*active_dc1 + \
                        conv.loc["total","nw_per_dn4"]*(active_dc1**2) + \
                        conv.loc["total","nw_per_dn5"]*(df["sci_rad_bench_temp_c"] - conv.loc["total","nw_per_dn_t0"])*active_dc1
        
        # Calculate nW per DN for channel 2/LW
        df["nw_per_dn2"] = conv.loc["lw","nw_per_dn0"] + \
                        conv.loc["lw","nw_per_dn1"]*(df["sci_rad_bench_temp_c"]  - conv.loc["lw","nw_per_dn_t0"]) + \
                        conv.loc["lw","nw_per_dn2"]*((df["sci_rad_bench_temp_c"] - conv.loc["lw","nw_per_dn_t0"])**2) + \
                        conv.loc["lw","nw_per_dn3"]*active_dc2 + \
                        conv.loc["lw","nw_per_dn4"]*(active_dc2**2) + \
                        conv.loc["lw","nw_per_dn5"]*(df["sci_rad_bench_temp_c"] - conv.loc["lw","nw_per_dn_t0"])*active_dc2

        # Calculate nW per DN for channel 3/SSW
        df["nw_per_dn3"] = conv.loc["ssw","nw_per_dn0"] + \
                        conv.loc["ssw","nw_per_dn1"]*(df["sci_rad_bench_temp_c"]  - conv.loc["ssw","nw_per_dn_t0"]) + \
                        conv.loc["ssw","nw_per_dn2"]*((df["sci_rad_bench_temp_c"] - conv.loc["ssw","nw_per_dn_t0"])**2) + \
                        conv.loc["ssw","nw_per_dn3"]*active_dc3 + \
                        conv.loc["ssw","nw_per_dn4"]*(active_dc3**2) + \
                        conv.loc["ssw","nw_per_dn5"]*(df["sci_rad_bench_temp_c"] - conv.loc["ssw","nw_per_dn_t0"])*active_dc3

        # Next, correct the nw per dn for the optical-electrical equivalance. The value
        # of conv#.oe_ineq is the in-equivalance in percent and they are all positive.
        # This is because the detectors measure too high. So the nw_per_dn# conversion
        # is correct DOWN by this amount
        df["nw_per_dn0"] = df["nw_per_dn0"]*(1 - conv.loc["sw","optical_electrical_inequiv_percent"]/100)
        df["nw_per_dn1"] = df["nw_per_dn1"]*(1 - conv.loc["total","optical_electrical_inequiv_percent"]/100)
        df["nw_per_dn2"] = df["nw_per_dn2"]*(1 - conv.loc["lw","optical_electrical_inequiv_percent"]/100)
        df["nw_per_dn3"] = df["nw_per_dn3"]*(1 - conv.loc["ssw","optical_electrical_inequiv_percent"]/100)
        
        # Convert dn to nW for each radiometer
        df["nw0_sw"] = df["sci_rad_pwm0_dn"]*df["nw_per_dn0"]
        df["nw1_to"] = df["sci_rad_pwm1_dn"]*df["nw_per_dn1"]
        df["nw2_lw"] = df["sci_rad_pwm2_dn"]*df["nw_per_dn2"]
        df["nw3_ss"] = df["sci_rad_pwm3_dn"]*df["nw_per_dn3"]

        # Convert from nW to W m-2 sr-1
        df["l0_sw"] = 1e-9*df["nw0_sw"]/(conv.loc["sw","collection_area"]*conv.loc["sw","solid_angle"])
        df["l1_to"] = 1e-9*df["nw1_to"]/(conv.loc["total","collection_area"]*conv.loc["total","solid_angle"])
        df["l2_lw"] = 1e-9*df["nw2_lw"]/(conv.loc["lw","collection_area"]*conv.loc["lw","solid_angle"])
        df["l3_ss"] = 1e-9*df["nw3_ss"]/(conv.loc["ssw","collection_area"]*conv.loc["ssw","solid_angle"])

    elif fpe_name == 'fmcfpe':

        # Get the median dark active PWM level used as part of the conversion
        active_dc0 = df["em_pwm_dn"].median()/12500.0

        # Next convert the raw DN values into power   

        # Load the conversion coefficients 
        conv = get_scirad_conversions(fpe_name=fpe_name)

        # Calculate nW per DN for channel 0/SW
        df["nw_per_dn"] = conv.loc["swcr","nw_per_dn0"] + \
                          conv.loc["swcr","nw_per_dn1"]*(df["sci_rad_bench_temp_c"]  - conv.loc["swcr","nw_per_dn_t0"]) + \
                          conv.loc["swcr","nw_per_dn2"]*((df["sci_rad_bench_temp_c"] - conv.loc["swcr","nw_per_dn_t0"])**2) + \
                          conv.loc["swcr","nw_per_dn3"]*active_dc0 + \
                          conv.loc["swcr","nw_per_dn4"]*(active_dc0**2) + \
                          conv.loc["swcr","nw_per_dn5"]*(df["sci_rad_bench_temp_c"] - conv.loc["swcr","nw_per_dn_t0"])*active_dc0

        # Next, correct the nw per dn for the optical-electrical equivalance. The value
        # of conv#.oe_ineq is the in-equivalance in percent and they are all positive.
        # This is because the detectors measure too high. So the nw_per_dn# conversion
        # is correct DOWN by this amount
        df["nw_per_dn"] = df["nw_per_dn"]*(1 - conv.loc["swcr","optical_electrical_inequiv_percent"]/100)
        
        # Convert dn to nW for each radiometer
        df["nw_swcr"] = df["em_pwm_dn"]*df["nw_per_dn"]

        # Convert from nW to W m-2 sr-1
        df["l_swcr"] = 1e-9*df["nw_swcr"]/(conv.loc["swcr","collection_area"]*conv.loc["swcr","solid_angle"])

    elif fpe_name in ('p01','p02'):

        # We'll add the conversions here later

        print("LST Data.")

    # Convert the PBR-R readings
    pbr_dark_dn = df["pbrr_pwm_dn"].median()
    pbr_pcb_temp = df["pbrr_board_t_c"].mean()
    pbr_sn = '17'
    pbr_pow_conv = pbr.dn_to_pow(pbr_dark_dn=pbr_dark_dn,
                                 pbr_sn=pbr_sn,
                                 pbr_pcb_temp=pbr_pcb_temp)            

    df["c_pbr_nw_per_dn"] = 1e9*pbr_pow_conv.c_pbr_w_per_dn
    df["pbr_pow_nw"] = df["c_pbr_nw_per_dn"]*df["pbrr_pwm_dn"]

    # If aperture diameter isn't set then use 12mm
    if ap_dia_mm is None:
        ap_dia_mm = 13

    # Use OPA wavelength if the wavelength wasn't passed
    if wavelength_um is None:
        wavelength_um = df["opa_wavelength_um"].median()

    # Convert the PBR-R snout PRT resistanes to temperatures
    # PRT parameters. This is based on the PRT formula:
    #       R(T) = R0*(1+a*T+b*T^2)
    # where we solved for T and is good for T>0C
    r0 = 1000           # nominal resistance in ohms
    a = 3.9083e-3
    b = -5.775e-7
    if df['pbrr_ap1_prt_r_ohms'].median() > 0:
        df['pbrr_ap1_prt_t_c'] = (np.sqrt(r0*(4*b*df['pbrr_ap1_prt_r_ohms'] + (a**2)*r0 - 4*b*r0)) - a*r0)/(2*b*r0)
    else:
        df['pbrr_ap1_prt_t_c'] = 0

    if df['pbrr_ap2_prt_r_ohms'].median() > 0:
        df['pbrr_ap2_prt_t_c'] = (np.sqrt(r0*(4*b*df['pbrr_ap2_prt_r_ohms'] + (a**2)*r0 - 4*b*r0)) - a*r0)/(2*b*r0)
    else:
        df['pbrr_ap2_prt_t_c'] = 0

    # Use the control temperatures for the temperature correction 
    # since they are always there. We'll probably add a correction
    # to them using the PRTs later.
    baffle_temp1_c = df["pbrr_ap1_t_c"].median()
    baffle_temp2_c = df["pbrr_ap2_t_c"].median()
    baffle_temp_c = (baffle_temp1_c + baffle_temp2_c)/2

    pbr_rad_conv = pbr.radiance_conversion(wavelength_um=wavelength_um, 
                                           ap_dia_mm=ap_dia_mm, 
                                           baffle_temp_c=baffle_temp_c)

    df["pbr_l"] = 1e-9*df["pbr_pow_nw"]*pbr_rad_conv.c_pbrr_pow_to_rad
    
    # Rename some items that started out as extra and later got renamed to prevent errors
    if "extra_1" in df.columns:
        df = df.rename(columns={"extra_1": "laser_setpoint"})

    if "extra_2" in df.columns:
        df = df.rename(columns={"extra_2": "sigma_delta_output"})

    pbr_conv = SimpleNamespace(**vars(pbr_pow_conv), **vars(pbr_rad_conv),
                        baffle_temp_c=baffle_temp_c,
                        wavelength_um=wavelength_um,
                        ap_dia_mm=ap_dia_mm)

    return df, pbr_conv


def load_erf_lst_logfile(*, file, 
                     ap_dia_mm=10, 
                     wavelength_um=1,
                     fpe_name='p02'): 
    """
    Loads an ERF LST file and converts the readings to radiance

    Inputs:
        file                ERF log filename
        ap_dia_mm           ERF integrating sphere aperture, 10, 12, or 13
        wavelength_um       The laser wavelength [um]
        fpe_name            Which FPE we're using: fmfpe, emfpe, emfpeb, or fmcfpe
    """

    # Load the ERF data
    paths = load_config()
    df = tidy_up_header(pd.read_csv(paths.erf_log_file_dir / file))

    # Convert science radiometer DN to radiance for the four science radiometers:
    if fpe_name == 'p02':

        # Get the median dark active PWM level used as part of the conversion. 
        # Note the max PWM for p02 is 10,000 dn
        active_dc = df["em_pwm_dn"].median()/10000.0

        conv_file = "libera_lst_radiometer_conversions_p02_v01.csv"
        paths = load_config()
        conv = pd.read_csv(paths.conversions_and_calibrations_dir / conv_file)
        conv = conv.set_index("channel_name")

        # Calculate nW per DN for channel 0/SW
        df["nw_per_dn"] = conv.loc["lst","nw_per_dn0"] + \
                          conv.loc["lst","nw_per_dn1"]*(df["em_telescope_temp_c"]  - conv.loc["lst","nw_per_dn_t0"]) + \
                          conv.loc["lst","nw_per_dn2"]*((df["em_telescope_temp_c"] - conv.loc["lst","nw_per_dn_t0"])**2) + \
                          conv.loc["lst","nw_per_dn3"]*active_dc + \
                          conv.loc["lst","nw_per_dn4"]*(active_dc**2) + \
                          conv.loc["lst","nw_per_dn5"]*(df["em_telescope_temp_c"] - conv.loc["lst","nw_per_dn_t0"])*active_dc

        # Next, correct the nw per dn for the optical-electrical equivalance. The value
        # of conv#.oe_ineq is the in-equivalance in percent and they are all positive.
        # This is because the detectors measure too high. So the nw_per_dn# conversion
        # is correct DOWN by this amount
        df["nw_per_dn"] = df["nw_per_dn"]*(1 - conv.loc["lst","optical_electrical_inequiv_percent"]/100)
        
        # Convert dn to nW
        df["nw"] = df["em_pwm_dn"]*df["nw_per_dn"]

        # Convert from nW to W m-2 sr-1
        df["l"] = 1e-9*df["nw"]/(conv.loc["lst","collection_area"]*conv.loc["lst","solid_angle"])

    else:
        print("Incorrect drive electronics.")

    # Convert the PBR-R readings
    pbr_dark_dn = df["pbrr_pwm_dn"].median()
    pbr_pcb_temp = df["pbrr_board_t_c"].mean()
    pbr_sn = '17'
    pbr_pow_conv = pbr.dn_to_pow(pbr_dark_dn=pbr_dark_dn,
                                 pbr_sn=pbr_sn,
                                 pbr_pcb_temp=pbr_pcb_temp)            

    df["c_pbr_nw_per_dn"] = 1e9*pbr_pow_conv.c_pbr_w_per_dn
    df["pbr_pow_nw"] = df["c_pbr_nw_per_dn"]*df["pbrr_pwm_dn"]

    # If aperture diameter isn't set then use 12mm
    if ap_dia_mm is None:
        ap_dia_mm = 13

    # Use OPA wavelength if the wavelength wasn't passed
    if wavelength_um is None:
        wavelength_um = df["opa_wavelength_um"].median()

    # Convert the PBR-R snout PRT resistanes to temperatures
    # PRT parameters. This is based on the PRT formula:
    #       R(T) = R0*(1+a*T+b*T^2)
    # where we solved for T and is good for T>0C
    r0 = 1000           # nominal resistance in ohms
    a = 3.9083e-3
    b = -5.775e-7
    if df['pbrr_ap1_prt_r_ohms'].median() > 0:
        df['pbrr_ap1_prt_t_c'] = (np.sqrt(r0*(4*b*df['pbrr_ap1_prt_r_ohms'] + (a**2)*r0 - 4*b*r0)) - a*r0)/(2*b*r0)
    else:
        df['pbrr_ap1_prt_t_c'] = 0

    if df['pbrr_ap2_prt_r_ohms'].median() > 0:
        df['pbrr_ap2_prt_t_c'] = (np.sqrt(r0*(4*b*df['pbrr_ap2_prt_r_ohms'] + (a**2)*r0 - 4*b*r0)) - a*r0)/(2*b*r0)
    else:
        df['pbrr_ap2_prt_t_c'] = 0

    # Use the control temperatures for the temperature correction 
    # since they are always there. We'll probably add a correction
    # to them using the PRTs later.
    baffle_temp1_c = df["pbrr_ap1_t_c"].median()
    baffle_temp2_c = df["pbrr_ap2_t_c"].median()
    baffle_temp_c = (baffle_temp1_c + baffle_temp2_c)/2

    pbr_rad_conv = pbr.radiance_conversion(wavelength_um=wavelength_um, 
                                           ap_dia_mm=ap_dia_mm, 
                                           baffle_temp_c=baffle_temp_c)

    df["pbr_l"] = 1e-9*df["pbr_pow_nw"]*pbr_rad_conv.c_pbrr_pow_to_rad
    
    # Rename some items that started out as extra and later got renamed to prevent errors
    if "extra_1" in df.columns:
        df = df.rename(columns={"extra_1": "laser_setpoint"})

    if "extra_2" in df.columns:
        df = df.rename(columns={"extra_2": "sigma_delta_output"})

    pbr_conv = SimpleNamespace(**vars(pbr_pow_conv), **vars(pbr_rad_conv),
                        baffle_temp_c=baffle_temp_c,
                        wavelength_um=wavelength_um,
                        ap_dia_mm=ap_dia_mm)

    return df, pbr_conv

def erf_log_file_read_and_parse(*,file,
                                ap_dia_mm=10,
                                wavelength_um=1,
                                fpe_name='fmfpe'):
    """
    This function actually loads the ERF data and performs the DC subtraction for each dwell
    """

    from scipy.interpolate import interp1d
    import statistics

    # Load the ERF data
    # df is the actual data
    # pbr_conv holds the PBR-R conversions and uncertainties
    df, pbr_conv = load_erf_logfile(file=file,
                          ap_dia_mm=ap_dia_mm, 
                          wavelength_um=wavelength_um,
                          fpe_name=fpe_name)

    # Throw out all the data with "active measurement = 0"
    df = df.loc[(df["active_measurement"] == 1)]

    # Loop through each active measurement
    meas_data = []
    for i in range(df["measurement_index"].max() + 1):

        # Get the data with the current measurement index
        df_tmp = df.loc[(df["measurement_index"] == i)].copy()

        # The the number of data points, analyze if there are more than 20
        npts = len(df_tmp)
        # print(f"Loop {i} Npts {npts}")

        if npts >= 20:

            # Get the current detector index
            detector_index = df_tmp["detector_index"].median()

            match detector_index:
                case 0:
                    # PBR-R
                    l_col = "pbr_l"
                    # Boxcar, not Hanning. The averaging window is the second
                    # half of the half-cycle, and PBR-R settles by sample 8 of
                    # 80 - so the window is already clear of the transition and
                    # the taper was only costing noise-equivalent bandwidth. A
                    # boxcar over the same 40 samples cuts the measurement to
                    # measurement scatter by 6.6% with the chop amplitude
                    # unchanged to 10 ppm.
                    han_filter = 0
                case 1:
                    # SSW
                    l_col = "l3_ss"
                    han_filter = 0
                case 2:
                    # LW
                    l_col = "l2_lw"
                    han_filter = 0
                case 3:
                    # Total
                    l_col = "l1_to"
                    han_filter = 0
                case 4:
                    # SW
                    l_col = "l0_sw"
                    han_filter = 0
                case _:  
                    # If there is a invalid detector index then just exit
                    print("Invalid Detector Index ")
                    return df, []
            
            # Extract out as a plain numpy array
            shutter = df_tmp["shutter_a"].values

            # --- transitions: where shutter_a differs from the previous point (circular) ---
            transitions = shutter != np.roll(shutter, 1)
            itmp = np.where(transitions)[0]

            # --- median spacing between transitions (also circular difference) ---
            # half_cycle_pts = np.median(itmp - np.roll(itmp, 1))

            # Number of points per half-cycle
            half_cycle_pts = statistics.mode(itmp[1:] - itmp[:-1])

            # --- round to nearest multiple of 4 ---
            half_cycle_pts = 4 * round(half_cycle_pts / 4.0)

            # --- cumulative count of transitions -> increments each half-cycle ---
            df_tmp["measurement_index"] = np.cumsum(transitions)

            # --- second half of the shutter cycle ---
            navg_pts = int(np.floor(half_cycle_pts / 2.0))

            # Create the hanning window
            if han_filter:
                avg_kernel = np.hanning(navg_pts + 2)[1:-1]
            else:
                avg_kernel = np.ones(navg_pts)

            # Normalize avg_kernel
            avg_kernel = avg_kernel/avg_kernel.sum()

            # Loop through each active measurement
            dc_sub_data = []
            for j in range(df_tmp["measurement_index"].max() + 1):
                df_tmp2 = df_tmp.loc[(df_tmp["measurement_index"] == j)].copy()
                df_tmp2 = df_tmp2[-navg_pts:]
                
                dc_sub_data.append({
                    "time":     (df_tmp2["gps_time_s"]*avg_kernel).sum(),
                    "l":        (df_tmp2[l_col]*avg_kernel).sum(),
                    "shutter":   df_tmp2["shutter_a"].median(),
                    "monitor":  (df_tmp2["monitor_a"]*avg_kernel).sum(),
                })

            # Change to a dataframe
            dc_sub_data = pd.DataFrame(dc_sub_data)

            # Get the time array. Then throw out the first and last points so that
            # every point is bracketed
            time = dc_sub_data["time"].iloc[1:-1].values

            # Subset the open and closed data
            closed = dc_sub_data[dc_sub_data["shutter"] == 0]
            open_  = dc_sub_data[dc_sub_data["shutter"] == 1]

            # Make sure each open and closed cycle in "time" is bracketed
            for name, subset in [("closed", closed), ("open", open_)]:
                subset_time = subset["time"].values
                if time.min() < subset_time.min() or time.max() > subset_time.max():
                    raise ValueError(
                        f"Interpolation times fall outside the {name} shutter time range: "
                        f"time range [{time.min():.3f}, {time.max():.3f}] vs "
                        f"{name} range [{subset_time.min():.3f}, {subset_time.max():.3f}]"
            )

            # Get the light and dark data interpolated to those times
            est_dark_radiance  = np.interp(time, closed["time"].values, closed["l"].values)
            est_open_radiance  = np.interp(time, open_["time"].values, open_["l"].values)
            monitor            = np.interp(time, dc_sub_data["time"].values, dc_sub_data["monitor"].values)

            # Calculate the DC subtraction
            dc_sub_radiance     = est_dark_radiance - est_open_radiance

            npts = len(dc_sub_radiance)
            for j in range(npts):
                meas_data.append({
                        "gps_time_s":           time[j],
                        "measurement_index":    i,
                        "detector_index":       detector_index,
                        "l":                    dc_sub_radiance[j],
                        "monitor":              monitor[j],
                        "manip_x_mm":           df_tmp['manip_x_mm'].median(),
                        "manip_y_mm":           df_tmp['manip_y_mm'].median(),
                        "manip_pitch_deg":      df_tmp['manip_pitch_deg'].median(),
                        "manip_yaw_deg":        df_tmp['manip_yaw_deg'].median(),
                        "opa_wavelength_um":    df_tmp['opa_wavelength_um'].median(),
                    })

    # Return the raw data and the analyzed data as a dataframe
    return df, pd.DataFrame(meas_data), pbr_conv

def erf_lst_log_file_read_and_parse(*,file,
                                ap_dia_mm=10,
                                wavelength_um=1,
                                fpe_name='fmfpe'):
    """
    This function actually loads the ERF data and performs the DC subtraction for each dwell
    """

    from scipy.interpolate import interp1d
    import statistics

    # Load the ERF data
    # df is the actual data
    # pbr_conv holds the PBR-R conversions and uncertainties
    df, pbr_conv = load_erf_lst_logfile(file=file,
                          ap_dia_mm=ap_dia_mm, 
                          wavelength_um=wavelength_um,
                          fpe_name=fpe_name)

    # Throw out all the data with "active measurement = 0"
    df = df.loc[(df["active_measurement"] == 1)]

    # Loop through each active measurement
    meas_data = []
    for i in range(df["measurement_index"].max() + 1):

        # Get the data with the current measurement index
        df_tmp = df.loc[(df["measurement_index"] == i)].copy()

        # The the number of data points, analyze if there are more than 20
        npts = len(df_tmp)
        # print(f"Loop {i} Npts {npts}")

        if npts >= 20:

            # Get the current detector index
            detector_index = df_tmp["detector_index"].median()

            match detector_index:
                case 0:
                    # PBR-R
                    l_col = "pbr_l"
                    # Boxcar, not Hanning. The averaging window is the second
                    # half of the half-cycle, and PBR-R settles by sample 8 of
                    # 80 - so the window is already clear of the transition and
                    # the taper was only costing noise-equivalent bandwidth. A
                    # boxcar over the same 40 samples cuts the measurement to
                    # measurement scatter by 6.6% with the chop amplitude
                    # unchanged to 10 ppm.
                    han_filter = 0
                case 5:
                    # LST
                    l_col = "l"
                    han_filter = 0
                case _:  
                    # If there is a invalid detector index then just exit
                    print("Invalid Detector Index ")
                    return df, []
            
            # Extract out as a plain numpy array
            shutter = df_tmp["shutter_a"].values

            # --- transitions: where shutter_a differs from the previous point (circular) ---
            transitions = shutter != np.roll(shutter, 1)
            itmp = np.where(transitions)[0]

            # --- median spacing between transitions (also circular difference) ---
            # half_cycle_pts = np.median(itmp - np.roll(itmp, 1))

            # Number of points per half-cycle
            half_cycle_pts = statistics.mode(itmp[1:] - itmp[:-1])

            # --- round to nearest multiple of 4 ---
            half_cycle_pts = 4 * round(half_cycle_pts / 4.0)

            # --- cumulative count of transitions -> increments each half-cycle ---
            df_tmp["measurement_index"] = np.cumsum(transitions)

            # --- second half of the shutter cycle ---
            navg_pts = int(np.floor(half_cycle_pts / 2.0))

            # Create the hanning window
            if han_filter:
                avg_kernel = np.hanning(navg_pts + 2)[1:-1]
            else:
                avg_kernel = np.ones(navg_pts)

            # Normalize avg_kernel
            avg_kernel = avg_kernel/avg_kernel.sum()

            # Loop through each active measurement
            dc_sub_data = []
            for j in range(df_tmp["measurement_index"].max() + 1):
                df_tmp2 = df_tmp.loc[(df_tmp["measurement_index"] == j)].copy()
                df_tmp2 = df_tmp2[-navg_pts:]
                
                dc_sub_data.append({
                    "time":     (df_tmp2["gps_time_s"]*avg_kernel).sum(),
                    "l":        (df_tmp2[l_col]*avg_kernel).sum(),
                    "shutter":   df_tmp2["shutter_a"].median(),
                    "monitor":  (df_tmp2["monitor_a"]*avg_kernel).sum(),
                })

            # Change to a dataframe
            dc_sub_data = pd.DataFrame(dc_sub_data)

            # Get the time array. Then throw out the first and last points so that
            # every point is bracketed
            time = dc_sub_data["time"].iloc[1:-1].values

            # Subset the open and closed data
            closed = dc_sub_data[dc_sub_data["shutter"] == 0]
            open_  = dc_sub_data[dc_sub_data["shutter"] == 1]

            # Make sure each open and closed cycle in "time" is bracketed
            for name, subset in [("closed", closed), ("open", open_)]:
                subset_time = subset["time"].values
                if time.min() < subset_time.min() or time.max() > subset_time.max():
                    raise ValueError(
                        f"Interpolation times fall outside the {name} shutter time range: "
                        f"time range [{time.min():.3f}, {time.max():.3f}] vs "
                        f"{name} range [{subset_time.min():.3f}, {subset_time.max():.3f}]"
            )

            # Get the light and dark data interpolated to those times
            est_dark_radiance  = np.interp(time, closed["time"].values, closed["l"].values)
            est_open_radiance  = np.interp(time, open_["time"].values, open_["l"].values)
            monitor            = np.interp(time, dc_sub_data["time"].values, dc_sub_data["monitor"].values)

            # Calculate the DC subtraction
            dc_sub_radiance     = est_dark_radiance - est_open_radiance

            npts = len(dc_sub_radiance)
            for j in range(npts):
                meas_data.append({
                        "gps_time_s":           time[j],
                        "measurement_index":    i,
                        "detector_index":       detector_index,
                        "l":                    dc_sub_radiance[j],
                        "monitor":              monitor[j],
                        "manip_x_mm":           df_tmp['manip_x_mm'].median(),
                        "manip_y_mm":           df_tmp['manip_y_mm'].median(),
                        "manip_pitch_deg":      df_tmp['manip_pitch_deg'].median(),
                        "manip_yaw_deg":        df_tmp['manip_yaw_deg'].median(),
                        "opa_wavelength_um":    df_tmp['opa_wavelength_um'].median(),
                    })

    # Return the raw data and the analyzed data as a dataframe
    return df, pd.DataFrame(meas_data), pbr_conv

def science_radiometer_cal_single(*,row,summary_data):
    """
    Analyzes a single ERF measurement
    """

    from scipy.interpolate import interp1d
    import statistics

    # The file paths
    paths = load_config()

    # Initially just use the OPA wavelength, ovewrite if there is a spectrum file
    wavelength_um = row.wavelength_um
    wavelength_um_fwhm = 0
    wavelength_gauss_amp = 0
    wavelength_center_um_sd = 0
    wavelength_fwhm_um_sd = 0
    wavelength_source = 10

    # Analyze the laser spectrum
    # Make sure there's a spectrum filename present
    if pd.notna(row.spectrum_file):

        # Analyze a CCS spectrum
        if "ccs200" in row.spectrum_file.lower():
            #print(f"CCS200: {row.spectrum_file}")
            wl_fit = analyze_ccs200_spectrum(file=row.spectrum_file, 
                                             wavelength_um_opa=row.wavelength_um,
                                             show_plot=False)
            wavelength_um = wl_fit.wavelength_center_um
            wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
            wavelength_gauss_amp = wl_fit.gaussian_amp
            wavelength_center_um_sd = wl_fit.wavelength_center_um_sd
            wavelength_fwhm_um_sd = wl_fit.wavelength_fwhm_um_sd
            wavelength_source = 1

        # Analyze a OSA spectrum
        if ("osa207" in row.spectrum_file.lower()) and (row.wavelength_um <= OSA_MAX_WAVELENGTH_UM) and (row.wavelength_um > 1):
            #print(f"OSA207: {row.spectrum_file}")
            wl_fit = analyze_osa207_spectrum(file=row.spectrum_file,
                                             wavelength_um_opa=row.wavelength_um,
                                             show_plot=False)

            # A trace with no resolvable line comes back with a zero centre.
            # Leave wavelength_source at its sentinel in that case, so that the
            # grating below takes over, and level_01 drops the row if there is
            # no grating spectrum either.
            if wl_fit.wavelength_center_um > 0:
                wavelength_um = wl_fit.wavelength_center_um
                wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
                wavelength_gauss_amp = wl_fit.gaussian_amp
                wavelength_center_um_sd = wl_fit.wavelength_center_um_sd
                wavelength_fwhm_um_sd = wl_fit.wavelength_fwhm_um_sd
                wavelength_source = 2
    
    # Analyze a grating spectrum
    grating_wavelength_um = 0
    grating_wavelength_um_fwhm = 0
    if pd.notna(row.grating_spectrum_file):
        if "grating" in row.grating_spectrum_file.lower():
            wl_fit = analyze_grating_spectrum(file=row.grating_spectrum_file,
                                              wavelength_um_opa=row.wavelength_um,
                                              show_plot=False)
            grating_wavelength_um = wl_fit.wavelength_center_um
            grating_wavelength_um_fwhm = wl_fit.wavelength_fwhm_um

            # Use the grating wavelength whenever the OSA cannot supply one:
            # either the OPA is beyond the OSA's usable range, or the OSA had no
            # resolvable line and left wavelength_source at its sentinel. The
            # grating also returns a zero centre when it fails, so that is
            # guarded rather than assigning a zero wavelength downstream.
            if ((row.wavelength_um > OSA_MAX_WAVELENGTH_UM or wavelength_source == 10)
                    and grating_wavelength_um > 0):
                wavelength_um = grating_wavelength_um
                wavelength_um_fwhm = grating_wavelength_um_fwhm
                wavelength_gauss_amp = wl_fit.gaussian_amp
                wavelength_center_um_sd = wl_fit.wavelength_center_um_sd
                wavelength_fwhm_um_sd = wl_fit.wavelength_fwhm_um_sd
                wavelength_source = 3

    # This routine reads and does the DC subtraction for each file
    df, meas_data, pbr_conv = erf_log_file_read_and_parse(file=row.log_file,
                                                ap_dia_mm=row.port_size_mm,
                                                wavelength_um=wavelength_um,
                                                fpe_name=row.driver)

    # If the monitor signal is large enough (greater than 0.2V) then:
    # Now convert the relative monitor fluctuations to a monitor correction that
    # we'll apply to each detector measurement
    if meas_data["monitor"].mean() > 0.2:
        meas_data["monitor_correction"] = meas_data["monitor"]/meas_data["monitor"].mean()
        monitor_corr = 1
    else:
        meas_data["monitor_correction"] = 1
        monitor_corr = 1

    # Now apply the correction. So if the monitor is lower than the mean the correction
    # is less that 1, and this division increases the measured radiance
    meas_data["l"]  = meas_data["l"]/meas_data["monitor_correction"]

    # Global fit of the radiance field: one field, every detector at once.
    #
    # This is the alternative to the per-detector plane fits above. It gives the
    # responsivity of each science radiometer relative to PBR-R directly as a
    # fitted parameter (gf_resp below) rather than as scirad_l/pbrr_l, and it
    # carries a drift term that the separate fits cannot represent. See
    # fit_radiance_field for why the shared gradients are legitimate and why the
    # drift matters.
    #
    # Both are computed for now so the two can be compared row by row. The plane
    # fit results are still what feed scirad_l / pbrr_l.
    field = fit_radiance_field(meas_data)

    # The list of ERF scattered light correction files
    # we're initially using the total channel for the LW
    corr_files = ['scirad_ssw_pst_corr.csv',
                  'scirad_total_pst_corr.csv',
                  'scirad_total_pst_corr.csv',
                  'scirad_sw_pst_corr.csv']

    # Get the correct columns for the ERF corrections. This stores the correction and the associated
    # uncertainty, but it is not applied to the data yet.
    match row.port_size_mm: 
        case 10:
            erf_cor_col = 'loss_10'
            erf_uc_col  = 'loss_10_sd'
        case 12:
            erf_cor_col = 'loss_12'
            erf_uc_col  = 'loss_12_sd'
        case 13:
            erf_cor_col = 'loss_13'
            erf_uc_col  = 'loss_13_sd'
        case _:
            print("Invalid Sphere Diameter " + str(row.port_size_mm))
            breakpoint()

    # Get the ERF out of field correction for each channel
    c_scirad_out_of_field = np.zeros(5)
    u_scirad_out_of_field = np.zeros(5)     
    for i in range(1,5):

        # Load the ERF correction file
        erf_corr_df = tidy_up_header(pd.read_csv(paths.analysis_dir  / corr_files[i-1]))

        # Get the science radiometer out of field corrections
        scirad_out_of_field_error = np.interp(wavelength_um, erf_corr_df["wavelength_um"], erf_corr_df[erf_cor_col]/100)

        # This is the correction, we divide the ratio by this number to do the correction:
        # For example if the correction is 1% then scirad_out_of_field_error = 0.01
        # Then c_scirad_out_of_field = 1 - 0.01 = 0.99
        # Then we divide the ratio by c_scirad_out_of_field to increase the signal by 1%
        c_scirad_out_of_field[i] = 1 - scirad_out_of_field_error

        # u_scirad_out_of_field is
        # a *fractional* uncertainty on c_scirad_out_of_field (e.g. 0.0005 =
        # 0.05%), but was being used downstream (level_02/level_03) directly
        # as if it were already an absolute SRF-domain uncertainty - it never
        # got scaled by the channel's own response. Since
        # srf = gf_resp / c_scirad_out_of_field, propagating a fractional
        # uncertainty in the denominator gives an absolute uncertainty of
        # roughly srf * u_fractional (c_scirad_out_of_field is close to 1),
        # so this now multiplies by field.r[i] (the channel's raw response
        # relative to PBR-R, before the out-of-field correction is even
        # applied) to convert it to that absolute uncertainty. Confirmed
        # this masked itself wherever a channel's SRF was near 1 (unscaled
        # and scaled versions coincide there), but left the term
        # artificially large - not shrinking toward 0 the way the channel's
        # own response does - at every channel's own out-of-band edges.
        u_scirad_out_of_field[i] = np.interp(wavelength_um, erf_corr_df["wavelength_um"], erf_corr_df[erf_uc_col]/100) * field.r[i]

    summary_data.append({
            "filename":                     row.log_file,
            "int_sphere":                   row.int_sphere,
            "ap_dia_mm":                    row.port_size_mm,
            "driver":                       row.driver,
            "alignment_session":            row.alignment_session,
            "measurement_campaign":         row.measurement_campaign,
            "beam_dither":                  row.beam_dither,
            "filter":                       row.filter,
            "spectrum_filename":            row.spectrum_file,
            "gps_time_s":                   meas_data["gps_time_s"].mean(),
            "monitor_mn":                   meas_data["monitor"].mean(),
            "monitor_sd":                   meas_data["monitor"].std(),
            "sigma_delta_output_mean":      df["sigma_delta_output"].mean(),
            "sigma_delta_output_sd":        df["sigma_delta_output"].std(),
            "sci_rad_bench_temp_c":         df["sci_rad_bench_temp_c"].mean(),
            "wavelength_um":                wavelength_um,
            "wavelength_center_um_sd":      wavelength_center_um_sd,
            "wavelength_gauss_amp":         wavelength_gauss_amp,
            "wavelength_source":            wavelength_source,
            "wavelength_um_fwhm":           wavelength_um_fwhm,
            "wavelength_fwhm_um_sd":        wavelength_fwhm_um_sd,
            "wavelength_um_opa":            row.wavelength_um,
            "wavelength_um_grating":        grating_wavelength_um,
            # --- PBR-R Uncertainties ---------------------------------
            "u_pbrr_ap_align":              pbr_conv.u_pbrr_ap_align, 
            "u_pbrr_stray":                 pbr_conv.u_pbrr_stray,
            "u_pbrr_ap_d":                  pbr_conv.u_pbrr_ap_d,
            "u_pbrr_det_ap":                pbr_conv.u_pbrr_det_ap,
            "u_pbrr_det_ap_diff":           pbr_conv.u_pbrr_det_ap_diff,
            "u_pbrr_ent_ap":                pbr_conv.u_pbrr_ent_ap,
            "u_pbrr_ent_ap_diff":           pbr_conv.u_pbrr_ent_ap_diff,
            "u_pbrr_nonequiv":              pbr_conv.u_pbrr_nonequiv,
            "u_pbrr_nonlinear":             pbr_conv.u_pbrr_nonlinear,
            "u_pbrr_rhtr":                  pbr_conv.u_pbrr_rhtr,
            "u_pbrr_rtop":                  pbr_conv.u_pbrr_rtop,
            "u_pbrr_rtrace":                pbr_conv.u_pbrr_rtrace,
            "u_pbrr_vacnt":                 pbr_conv.u_pbrr_vacnt,
            "u_pbrr_vref":                  pbr_conv.u_pbrr_vref,
            # --- global radiance field fit ---------------------------------
            # gf_resp is the measurand: this detector's responsivity relative
            # to PBR-R, fitted directly rather than formed as scirad_l/pbrr_l.
            "gf_resp_ss":                   field.r[1],                     # SSW Responsivity relative to PBR-R [-]
            "gf_resp_lw":                   field.r[2],                     # LW Responsivity relative to PBR-R [-]
            "gf_resp_to":                   field.r[3],                     # Total Responsivity relative to PBR-R [-]
            "gf_resp_sw":                   field.r[4],                     # SW Responsivity relative to PBR-R [-]
            "gf_resp_sd_ss":                field.r_sd[1],                  # SSW Responsivity relative to PBR-R k=1 uncertainty [-]
            "gf_resp_sd_lw":                field.r_sd[2],                  # LW Responsivity relative to PBR-R k=1 uncertainty [-]
            "gf_resp_sd_to":                field.r_sd[3],                  # Total Responsivity relative to PBR-R k=1 uncertainty [-]
            "gf_resp_sd_sw":                field.r_sd[4],                  # SW Responsivity relative to PBR-R k=1 uncertainty [-]
            "gf_l0":                        field.l0,                       # PBR-R radiance at field centre, mean time [W m-2 sr-1]
            "gf_l0_sd":                     field.l0_sd,                    # [W m-2 sr-1]
            "gf_gx":                        field.gx,                       # Fractional radiance x gradient [mm-1]
            "gf_gx_sd":                     field.gx_sd,                    # [mm-1]
            "gf_gy":                        field.gy,                       # Fractional radiance y gradient [mm-1]
            "gf_gy_sd":                     field.gy_sd,                    # [mm-1]
            "gf_gp":                        field.gp,                       # Fractional radiance pitch gradient [deg-1]
            "gf_gp_sd":                     field.gp_sd,                    # [deg-1]
            "gf_gw":                        field.gw,                       # Fractional radiance yaw gradient [deg-1]
            "gf_gw_sd":                     field.gw_sd,                    # [deg-1]
            "gf_gt":                        field.gt,                       # Fractional source drift [min-1]
            "gf_gt_sd":                     field.gt_sd,                    # [min-1]            
            # Per-detector noise solved from the fit residuals. rel_sigma is a
            # data quality metric: out-of-band or frozen channels stand out.
            "gf_sigma_ss":                  field.sigma[1],                 # SSW detector's residual scatter [W m-2 sr-1]
            "gf_sigma_lw":                  field.sigma[2],                 # LW detector's residual scatter [W m-2 sr-1]
            "gf_sigma_to":                  field.sigma[3],                 # Total detector's residual scatter [W m-2 sr-1]
            "gf_sigma_sw":                  field.sigma[4],                 # SW detector's residual scatter [W m-2 sr-1]            
            "gf_sigma_pbrr":                field.sigma[field.pbrr_index],  # PBR-R residual scatter [W m-2 sr-1]            
            "gf_redchi":                    field.redchi,                   # Reduced chi-square of the global fit [-]
            "gf_ndata":                     field.ndata,                    # Points in the global fit
            "gf_nvarys":                    field.nvarys,                   # Free parameters in the global fit
            # The ERF out of field corrections
            "c_scirad_out_of_field_ss":     c_scirad_out_of_field[1],       # SSW channel out of field correction, divide gf_resp_ss by this to correct [-]
            "c_scirad_out_of_field_lw":     c_scirad_out_of_field[2],       # LW channel out of field correction, divide gf_resp_lw by this to correct [-]
            "c_scirad_out_of_field_to":     c_scirad_out_of_field[3],       # Total channel out of field correction, divide gf_resp_to by this to correct [-]
            "c_scirad_out_of_field_sw":     c_scirad_out_of_field[4],       # SW channel out of field correction, divide gf_resp_sw by this to correct [-]
            "u_scirad_out_of_field_ss":     u_scirad_out_of_field[1],       # SSW channel out of field correction uncertainty [-]
            "u_scirad_out_of_field_lw":     u_scirad_out_of_field[2],       # LW channel out of field correction uncertainty [-]
            "u_scirad_out_of_field_to":     u_scirad_out_of_field[3],       # Total channel out of field correction uncertainty [-]
            "u_scirad_out_of_field_sw":     u_scirad_out_of_field[4]        # SW channel out of field correction uncertainty [-]
            })

    return summary_data

def lst_cal_single(*,row,summary_data):
    """
    Analyzes a single ERF measurement taken with the LST
    """

    from scipy.interpolate import interp1d
    import statistics

    # The file paths
    paths = load_config()

    # Initially just use the OPA wavelength, ovewrite if there is a spectrum file
    wavelength_um = row.wavelength_um
    wavelength_um_fwhm = 0
    wavelength_gauss_amp = 0
    wavelength_center_um_sd = 0
    wavelength_fwhm_um_sd = 0
    wavelength_source = 10

    # Analyze the laser spectrum
    # Make sure there's a spectrum filename present
    if pd.notna(row.spectrum_file):

        # Analyze a CCS spectrum
        if "ccs200" in row.spectrum_file.lower():
            #print(f"CCS200: {row.spectrum_file}")
            wl_fit = analyze_ccs200_spectrum(file=row.spectrum_file, 
                                             wavelength_um_opa=row.wavelength_um,
                                             show_plot=False)
            wavelength_um = wl_fit.wavelength_center_um
            wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
            wavelength_gauss_amp = wl_fit.gaussian_amp
            wavelength_center_um_sd = wl_fit.wavelength_center_um_sd
            wavelength_fwhm_um_sd = wl_fit.wavelength_fwhm_um_sd
            wavelength_source = 1

        # Analyze a OSA spectrum
        if ("osa207" in row.spectrum_file.lower()) and (row.wavelength_um <= OSA_MAX_WAVELENGTH_UM) and (row.wavelength_um > 1):
            #print(f"OSA207: {row.spectrum_file}")
            wl_fit = analyze_osa207_spectrum(file=row.spectrum_file,
                                             wavelength_um_opa=row.wavelength_um,
                                             show_plot=False)

            # A trace with no resolvable line comes back with a zero centre.
            # Leave wavelength_source at its sentinel in that case, so that the
            # grating below takes over, and level_01 drops the row if there is
            # no grating spectrum either.
            if wl_fit.wavelength_center_um > 0:
                wavelength_um = wl_fit.wavelength_center_um
                wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
                wavelength_gauss_amp = wl_fit.gaussian_amp
                wavelength_center_um_sd = wl_fit.wavelength_center_um_sd
                wavelength_fwhm_um_sd = wl_fit.wavelength_fwhm_um_sd
                wavelength_source = 2
    
    # Analyze a grating spectrum
    grating_wavelength_um = 0
    grating_wavelength_um_fwhm = 0
    if pd.notna(row.grating_spectrum_file):
        if "grating" in row.grating_spectrum_file.lower():
            wl_fit = analyze_grating_spectrum(file=row.grating_spectrum_file,
                                              wavelength_um_opa=row.wavelength_um,
                                              show_plot=False)
            grating_wavelength_um = wl_fit.wavelength_center_um
            grating_wavelength_um_fwhm = wl_fit.wavelength_fwhm_um

            # Use the grating wavelength whenever the OSA cannot supply one:
            # either the OPA is beyond the OSA's usable range, or the OSA had no
            # resolvable line and left wavelength_source at its sentinel. The
            # grating also returns a zero centre when it fails, so that is
            # guarded rather than assigning a zero wavelength downstream.
            if ((row.wavelength_um > OSA_MAX_WAVELENGTH_UM or wavelength_source == 10)
                    and grating_wavelength_um > 0):
                wavelength_um = grating_wavelength_um
                wavelength_um_fwhm = grating_wavelength_um_fwhm
                wavelength_gauss_amp = wl_fit.gaussian_amp
                wavelength_center_um_sd = wl_fit.wavelength_center_um_sd
                wavelength_fwhm_um_sd = wl_fit.wavelength_fwhm_um_sd
                wavelength_source = 3

    # This routine reads and does the DC subtraction for each file
    df, meas_data, pbr_conv = erf_lst_log_file_read_and_parse(file=row.log_file,
                                                ap_dia_mm=row.port_size_mm,
                                                wavelength_um=wavelength_um,
                                                fpe_name=row.driver)

    # If the monitor signal is large enough (greater than 0.2V) then:
    # Now convert the relative monitor fluctuations to a monitor correction that
    # we'll apply to each detector measurement
    if meas_data["monitor"].mean() > 0.2:
        meas_data["monitor_correction"] = meas_data["monitor"]/meas_data["monitor"].mean()
        monitor_corr = 1
    else:
        meas_data["monitor_correction"] = 1
        monitor_corr = 1

    # Now apply the correction. So if the monitor is lower than the mean the correction
    # is less that 1, and this division increases the measured radiance
    meas_data["l"]  = meas_data["l"]/meas_data["monitor_correction"]

    # Global fit of the radiance field: one field, both detectors at once.
    #
    # This gives the LST's responsivity relative to PBR-R (gf_resp below)
    # directly as a fitted parameter rather than as a quotient of two separate
    # fits, and it carries the source drift term. See fit_radiance_field for
    # why the shared gradients are legitimate and why the drift matters.
    #
    # No LST-specific version of fit_radiance_field is needed: it fits whatever
    # detectors are present, and an LST file is structurally a science
    # radiometer file with three of the four channels removed - the LST is
    # measured in two passes bracketing the PBR-R block with the same
    # manipulator dither pattern, which is what keeps the drift term
    # identifiable (see that function's docstring).
    #
    # Two caveats from having only one non-reference detector: share_angular has
    # no effect (shared and per-channel gp/gw are then the same fit), and gp/gw
    # are more weakly constrained than in the flight fit, where four channels'
    # dithers constrain the one pair.
    field = fit_radiance_field(meas_data)

    # detector_index of the LST. Taken from the fit rather than hardcoded so it
    # cannot drift out of sync with erf_lst_log_file_read_and_parse (which
    # assigns 5); an LST file has exactly one detector besides PBR-R.
    lst_detectors = [i for i in field.detectors if i != field.pbrr_index]
    if len(lst_detectors) != 1:
        raise ValueError(f"Expected one LST detector besides PBR-R, found {lst_detectors}.")
    lst_index = lst_detectors[0]

    # ERF scattered light (out of field) correction file. The LST is a model of
    # the Total channel - optics from the same coating run, detector from the
    # same fab run - so the Total channel's correction is used. The LST has its
    # own optics and baffle geometry, so this is an approximation until the
    # LST's own PST is measured, but the Total channel should be close.
    corr_file = 'scirad_total_pst_corr.csv'

    # Get the correct columns for the ERF corrections. This stores the correction and the associated
    # uncertainty, but it is not applied to the data yet.
    match row.port_size_mm:
        case 10:
            erf_cor_col = 'loss_10'
            erf_uc_col  = 'loss_10_sd'
        case 12:
            erf_cor_col = 'loss_12'
            erf_uc_col  = 'loss_12_sd'
        case 13:
            erf_cor_col = 'loss_13'
            erf_uc_col  = 'loss_13_sd'
        case _:
            print("Invalid Sphere Diameter " + str(row.port_size_mm))
            breakpoint()

    # Load the ERF correction file
    erf_corr_df = tidy_up_header(pd.read_csv(paths.analysis_dir / corr_file))

    # Get the LST out of field correction at this wavelength
    scirad_out_of_field_error = np.interp(wavelength_um, erf_corr_df["wavelength_um"], erf_corr_df[erf_cor_col]/100)

    # This is the correction, we divide the ratio by this number to do the correction:
    # For example if the correction is 1% then scirad_out_of_field_error = 0.01
    # Then c_scirad_out_of_field = 1 - 0.01 = 0.99
    # Then we divide the ratio by c_scirad_out_of_field to increase the signal by 1%
    c_scirad_out_of_field = 1 - scirad_out_of_field_error

    # The correction file's uncertainty column is a *fractional* uncertainty on
    # c_scirad_out_of_field (e.g. 0.0005 = 0.05%), so it is scaled by the LST's
    # own response here to turn it into the absolute, SRF-domain uncertainty the
    # downstream levels consume. Since srf = gf_resp / c_scirad_out_of_field and
    # c_scirad_out_of_field is close to 1, propagating a fractional uncertainty
    # in the denominator gives an absolute uncertainty of roughly
    # srf * u_fractional. Without this scaling the term would not shrink toward
    # zero the way the LST's own response does, leaving it artificially large
    # out of band. Matches the flight treatment in science_radiometer_cal_single.
    u_scirad_out_of_field = np.interp(wavelength_um, erf_corr_df["wavelength_um"], erf_corr_df[erf_uc_col]/100) * field.r[lst_index]

    summary_data.append({
            "filename":                     row.log_file,
            "int_sphere":                   row.int_sphere,
            "ap_dia_mm":                    row.port_size_mm,
            "driver":                       row.driver,
            "alignment_session":            row.alignment_session,
            "measurement_campaign":         row.measurement_campaign,
            "beam_dither":                  row.beam_dither,
            "filter":                       row.filter,
            "spectrum_filename":            row.spectrum_file,
            "gps_time_s":                   meas_data["gps_time_s"].mean(),
            "monitor_mn":                   meas_data["monitor"].mean(),
            "monitor_sd":                   meas_data["monitor"].std(),
            "sigma_delta_output_mean":      df["sigma_delta_output"].mean(),
            "sigma_delta_output_sd":        df["sigma_delta_output"].std(),
            # The LST's own telescope temperature. The flight summary records
            # sci_rad_bench_temp_c, but load_erf_lst_logfile does not derive
            # that column - em_telescope_temp_c is the LST's temperature and is
            # what its radiance conversion is keyed to.
            "em_telescope_temp_c":          df["em_telescope_temp_c"].mean(),
            "wavelength_um":                wavelength_um,
            "wavelength_center_um_sd":      wavelength_center_um_sd,
            "wavelength_gauss_amp":         wavelength_gauss_amp,
            "wavelength_source":            wavelength_source,
            "wavelength_um_fwhm":           wavelength_um_fwhm,
            "wavelength_fwhm_um_sd":        wavelength_fwhm_um_sd,
            "wavelength_um_opa":            row.wavelength_um,
            "wavelength_um_grating":        grating_wavelength_um,
            # --- PBR-R Uncertainties ---------------------------------
            "u_pbrr_ap_align":              pbr_conv.u_pbrr_ap_align,
            "u_pbrr_stray":                 pbr_conv.u_pbrr_stray,
            "u_pbrr_ap_d":                  pbr_conv.u_pbrr_ap_d,
            "u_pbrr_det_ap":                pbr_conv.u_pbrr_det_ap,
            "u_pbrr_det_ap_diff":           pbr_conv.u_pbrr_det_ap_diff,
            "u_pbrr_ent_ap":                pbr_conv.u_pbrr_ent_ap,
            "u_pbrr_ent_ap_diff":           pbr_conv.u_pbrr_ent_ap_diff,
            "u_pbrr_nonequiv":              pbr_conv.u_pbrr_nonequiv,
            "u_pbrr_nonlinear":             pbr_conv.u_pbrr_nonlinear,
            "u_pbrr_rhtr":                  pbr_conv.u_pbrr_rhtr,
            "u_pbrr_rtop":                  pbr_conv.u_pbrr_rtop,
            "u_pbrr_rtrace":                pbr_conv.u_pbrr_rtrace,
            "u_pbrr_vacnt":                 pbr_conv.u_pbrr_vacnt,
            "u_pbrr_vref":                  pbr_conv.u_pbrr_vref,
            # --- global radiance field fit ---------------------------------
            # gf_resp is the measurand: the LST's responsivity relative to
            # PBR-R, fitted directly rather than formed as lst_l/pbrr_l. There
            # is only one channel, so no channel suffix here - unlike the
            # flight summary, which carries _ss/_lw/_to/_sw.
            "gf_resp":                      field.r[lst_index],             # LST Responsivity relative to PBR-R [-]
            "gf_resp_sd":                   field.r_sd[lst_index],          # LST Responsivity relative to PBR-R k=1 uncertainty [-]
            "gf_l0":                        field.l0,                       # PBR-R radiance at field centre, mean time [W m-2 sr-1]
            "gf_l0_sd":                     field.l0_sd,                    # [W m-2 sr-1]
            "gf_gx":                        field.gx,                       # Fractional radiance x gradient [mm-1]
            "gf_gx_sd":                     field.gx_sd,                    # [mm-1]
            "gf_gy":                        field.gy,                       # Fractional radiance y gradient [mm-1]
            "gf_gy_sd":                     field.gy_sd,                    # [mm-1]
            "gf_gp":                        field.gp,                       # Fractional radiance pitch gradient [deg-1]
            "gf_gp_sd":                     field.gp_sd,                    # [deg-1]
            "gf_gw":                        field.gw,                       # Fractional radiance yaw gradient [deg-1]
            "gf_gw_sd":                     field.gw_sd,                    # [deg-1]
            "gf_gt":                        field.gt,                       # Fractional source drift [min-1]
            "gf_gt_sd":                     field.gt_sd,                    # [min-1]
            # Per-detector noise solved from the fit residuals. rel_sigma is the
            # data quality metric that actually varies with the data: gf_redchi
            # is pinned to ndata/(ndata - nvarys) by the sigma iteration (the
            # sigmas are themselves the residual RMS), so it is the same number
            # for every file with the same point count and carries no
            # fit-quality information.
            "gf_sigma_lst":                 field.sigma[lst_index],         # LST detector's residual scatter [W m-2 sr-1]
            "gf_sigma_pbrr":                field.sigma[field.pbrr_index],  # PBR-R residual scatter [W m-2 sr-1]
            "gf_rel_sigma_lst":             field.rel_sigma[lst_index],     # LST residual scatter relative to its own mean signal [-]
            "gf_redchi":                    field.redchi,                   # Reduced chi-square of the global fit [-]
            "gf_ndata":                     field.ndata,                    # Points in the global fit
            "gf_nvarys":                    field.nvarys,                   # Free parameters in the global fit
            # The ERF out of field correction
            "c_scirad_out_of_field":        c_scirad_out_of_field,          # LST out of field correction, divide gf_resp by this to correct [-]
            "u_scirad_out_of_field":        u_scirad_out_of_field           # LST out of field correction uncertainty [-]
            })

    return summary_data

# The SW channel carries an etalon fringe in its response. Its source has not
# been identified optically, but it behaves like a simple two-surface cavity
# and is described by a SINGLE parameter, the optical thickness n*d.
#
# For such a cavity the transmission maxima satisfy 2*n*d*nu = m for integer m,
# so the fringes are equally spaced in WAVENUMBER and - the part that does the
# work here - a maximum necessarily sits at nu = 0, leaving no free phase.
# That was tested against the data rather than assumed: fitting a cosine to the
# residual about a smooth baseline over 0.40-2.25um, adding a free phase
# parameter gives F = 0.78, p = 0.38, i.e. it is not justified (chi2 88868
# locked against 88536 free). Two independent routes agree on the value:
#
#   cosine fit to the detrended residual   2*n*d = 16.50 +/- 0.17 um
#   chi2 scan of the full node fit         2*n*d = 16.50 um
#
# and it is stable across sub-bands (16.485 +/- 0.432, 16.514 +/- 0.153,
# 16.394 +/- 0.109), so no dispersion is detectable at the ~1% level. The
# implied physical thickness is a few um for any plausible index (5.3um at
# n=1.55, 2.4um at n=3.42), so this is a film, coating layer or bond line, not
# a bulk optic.
#
# Fringe amplitude runs 3.0e-4 at 0.45um rising to ~4.4e-3 at 2.2um, and
# accounts for 27% of the detrended variance in the band.
SW_ETALON_TWO_ND_UM = 16.499

# Fringe spacing follows from the optical thickness - it is not an independent
# quantity. Used only to lay out the node grid below.
SW_ETALON_PERIOD_CM1 = 1e4/SW_ETALON_TWO_ND_UM

# The fringe is carried by the parametric term in SRF_FRINGE_TERMS, not by the
# nodes, so this block only has to describe the smooth ENVELOPE underneath it.
# Two nodes per fringe cycle was chosen by 5-fold cross-validation over
# envelope densities of 0.5, 1, 1.5, 2 and 3 per cycle, all with the fringe
# term active, against the previous nodes-only grid:
#
#   case                       held-out chi by zone           fringe recovered
#                        0.4-.75  .75-1.4  1.4-2.25  2.25-4.6
#   158 nodes, no fringe    10.9     13.3       3.6      15.3   67 77 83 57 38 %
#   0.5/cyc + fringe         6.8      6.8       4.8      14.8   98 96 88 90 65 %
#   2  /cyc + fringe         6.8      6.9       3.1      15.3   96 95 95 85 57 %
#   3  /cyc + fringe         6.9      7.1       3.4      15.6   95 96 95 86 50 %
#
# The pure-envelope grid (0.5/cycle) has the best pooled score but regresses at
# 1.4-2.25um, where some of the structure is real and non-periodic and the
# nodes need enough freedom to follow it. Two per cycle improves every zone
# against the nodes-only grid and regresses in none, with 12 fewer free
# parameters and a lower reported uncertainty (median 2.51e-4 against 2.71e-4).
#
# Note the fringe term itself spans a wider range than this block - see
# SW_FRINGE_RANGE_UM.
SW_ETALON_NODES_PER_CYCLE = 2
SW_ETALON_RANGE_UM = (0.75, 2.25)

# Range over which the parametric fringe is applied. Wider than the node block
# above and deliberately so: the etalon is a real element in the path, so it
# acts across the whole SW passband rather than stopping where the node
# refinement does. Confining it to 0.40-2.25um cost real signal - recovery in
# 1.8-2.25um, where the fringe is strongest, was 28% with the taper closing at
# 2.25um and 65% with it at 4.60um, at unchanged cross-validation score.
SW_FRINGE_RANGE_UM = (0.40, 4.60)

# Node blocks for the Total channel, which needs the opposite treatment in two
# places. Total has essentially NO structure to resolve: testing the fit
# residual for coherence - |weighted mean residual| against its own standard
# error in +/-10% wavelength windows - gives ratios of 0.0-0.6 in every band,
# and 0 of 224 narrow (+/-3%) windows show a mean residual above 3x its
# standard error. Consistent with that, pooled cross-validation moves only
# 4.22 to 3.9 across a 4x range in node count (77 to 318 nodes), so grid choice
# is worth about 10% here against the 12.8 -> 10.4 the fringe term bought on SW.
#
# The two changes that do earn their place, in opposite directions:
#
#   UV turn-on   0.348um is Total's steepest feature (dSRF/dlog10(wl) = 18.6)
#                and the only real structure in the channel. Refining it takes
#                that zone's held-out chi from 5.9 to 4.3. Finer still (120
#                cm-1, 149 UV nodes) reaches 3.8, but that is 3.5x the nodes
#                against only 48 measurements in the band, so 240 cm-1 is where
#                this stops.
#
#   0.42-4.5um   COARSENED, which improves the fit rather than degrading it:
#                held-out chi 3.3 -> 2.4 and 2.7 -> 2.2, and the reported
#                uncertainty falls. With no structure to lose, extra freedom
#                only lets the fit chase noise. Note this is the one change
#                that shifts what GCV selects (smoothing 7.4e-4 -> 4.1e-4,
#                edof 136 -> 99); the UV refinement leaves it untouched at
#                7.44e-4, confirming the penalty is mesh-invariant.
#
# Net: 149 nodes -> 119, pooled held-out chi 4.15 -> 3.72, median uncertainty
# 2.95e-4 -> 2.77e-4. Above 4.5um the hand grid is left alone - every variant
# tried scored identically there.
#
# No parametric term is registered for Total: the cross-channel search found
# no power at the SW etalon period (0.007-0.012 against a median of 0.014 over
# the identical band and the same 212 measurements, i.e. below the noise floor,
# where SW reads 0.321) and no other significant periodicity.
TO_UV_RANGE_UM = (0.31, 0.42)
TO_UV_STEP_CM1 = 240.0
TO_MID_RANGE_UM = (0.42, 4.50)
TO_MID_STEP_CM1 = 480.0

# SSW: coarsen the flat plateau only. Unlike Total, SSW cannot be coarsened
# broadly - it has a genuine roll-off from 3.4um (dSRF/dlog10(wl) reaches -3.4
# by 3.6um) and a steep turn-on at 0.72um (+79 at 0.72, +32 at 0.75), and a
# coarse block run across either of those wrecks the fit. Restricted to the
# genuinely flat 1.0-3.0um it is a small improvement in every zone:
#
#   grid                 nodes   held-out chi by zone
#                                .69-.86  .86-1  1-3   3-4   4-4.9
#   current                155      13.5    4.3   2.8   4.8    68.9
#   1.0-3.0um @ 240 cm-1   138      10.3    4.2   2.8   4.0    68.5
#
# with the median uncertainty unchanged at 1.26e-4. Extending the block to
# 3.4um degrades 1-3um to 4.5 and 3-4um to 7.5, which is the roll-off being
# lost; 480 cm-1 scores marginally better pooled but raises the median
# uncertainty to 1.33e-4, so 240 is the choice.
#
# Note the gain shows up mostly in the 0.69-0.86um TURN-ON, not in the band
# that was coarsened. Fewer nodes on the plateau lets GCV settle on a lighter
# penalty, and the edge is what benefits - the same coupling seen on Total.
#
# All of this is nearly beside the point for SSW. Its pooled score is dominated
# by the 4.0-4.9um zone at 68.5, which no grid touches: that region's residuals
# are wavelength errors of order 10-18nm, not fitting errors. See
# science_radiometer_srf_node_fit's notes on the wavelength problem.
SS_FLAT_RANGE_UM = (1.0, 3.0)
SS_FLAT_STEP_CM1 = 240.0

def _wavenumber_node_block(*, range_um, step_cm1):
    """
    Node positions uniform in wavenumber across range_um at the given step.
    Returned in um, ascending.

    Wavenumber rather than wavelength because that is the coordinate the
    measurement's own resolution is uniform in: the laser line as recorded by
    the OSA207 has a width of 62-86 cm-1 across 1-16um (an FTS resolves
    uniformly in wavenumber, so this is the laser itself), which makes a
    wavenumber-uniform grid the resolution-matched choice for every channel.
    Below ~1um the recorded width rises to 113-163 cm-1, but there the
    measurement is made with the CCS200 grating spectrometer, whose instrument
    function is not negligible, so that figure is an upper bound on the laser.
    """

    nu_lo, nu_hi = 1e4/range_um[1], 1e4/range_um[0]
    nu = np.arange(nu_lo, nu_hi + step_cm1, step_cm1)
    nu = nu[(nu >= nu_lo) & (nu <= nu_hi)]

    return np.sort(1e4/nu)

def _uniform_node_block(*, range_um, step_um):
    """
    Node positions uniform in WAVELENGTH across range_um at the given step,
    starting at range_um[0] and never exceeding range_um[1]. Returned in um,
    ascending.

    Wavelength-uniform rather than the wavenumber-uniform default of
    _wavenumber_node_block, and used only for LW's 3.9-5.2um edge, where
    cross-validation picked a wavelength-uniform 0.15um grid over the
    wavenumber-uniform alternatives (held-out chi 27.4 against 31.1 for the
    best wavenumber step, 55cm-1, and with 9 nodes rather than 12). Over a span
    this narrow the two coordinates barely differ, so this is a marginal
    preference rather than a statement about the measurement's resolution.
    """

    x = np.arange(range_um[0], range_um[1] + 0.5*step_um, step_um)

    return x[x <= range_um[1] + 1e-9]

def _log_node_block(*, range_um, n_nodes):
    """
    n_nodes positions spaced uniformly in log wavelength across range_um,
    endpoints included. Returned in um, ascending.

    Used where a channel is far out of band and the grid's job is to describe a
    slowly varying leakage level over a wide span rather than to resolve a
    feature. Log spacing there beats both wavelength- and wavenumber-uniform
    because neither end of the span is privileged: LW's out-of-band region runs
    from 0.25 to 3.9um, a factor of 16, and a grid uniform in either coordinate
    puts nearly all of its nodes at one end of that.
    """

    return np.geomspace(range_um[0], range_um[1], n_nodes)

def _etalon_node_block(*, range_um, period_cm1, nodes_per_cycle):
    """
    Node positions uniform in wavenumber across range_um, at nodes_per_cycle
    per fringe of the given period. Returned in um, ascending.
    """

    nu_lo, nu_hi = 1e4/range_um[1], 1e4/range_um[0]
    step = period_cm1/nodes_per_cycle
    nu = np.arange(nu_lo, nu_hi + step, step)
    nu = nu[(nu >= nu_lo) & (nu <= nu_hi)]

    return np.sort(1e4/nu)

# Per-channel node grids for the SRF spline fit, ported from the IDL
# libera_cal_erf_science_radiometer_srf_wl_grid. These were placed by hand over
# six years of measurement campaigns, and the placement is the physics content
# of the fit: node density is where structure is believed to exist. The SW grid
# is roughly 3x denser than Total's through 0.4-1.0um, where SW has a genuine
# etalon-like oscillation; every grid tightens up around the UV turn-on
# (0.30-0.40um) and around the 4.2-5.0um band edges.
#
# Measured against this dataset the grids sit at about one node per laser
# linewidth, which is the resolution limit of the measurement:
#
#   band          laser fwhm (median)   SW node spacing
#   0.25-0.5um            2.2 nm            5.5 nm
#   0.5-1.0               6.1              10
#   1-2                  14.3              27
#   2-4                  75.8             100
#   4-7                 210               160
#   7-10                451               500
#   10-13               859               500
#
# In the 4-7 and 10-13um rows the nodes are finer than the linewidth, so the
# fit there is asking for detail the measurement cannot resolve. That is what
# the smoothing penalty below is for.
SRF_NODE_GRIDS = {
    "sw": np.array([
        0.248, 0.259, 0.269, 0.279, 0.289, 0.299, 0.305, 0.310, 0.315, 0.320,
        0.328, 0.332, 0.334, 0.340, 0.345, 0.350, 0.355, 0.360, 0.365, 0.369,
        0.374, 0.380, 0.384, 0.391, 0.395, 0.400, 0.405, 0.410, 0.420, 0.430,
        0.440, 0.450, 0.460, 0.475, 0.490, 0.500, 0.510, 0.520, 0.527,
        0.538, 0.549, 0.558, 0.570, 0.579, 0.590, 0.599, 0.607, 0.617,
        0.637, 0.650, 0.660, 0.670, 0.680, 0.690, 0.700, 0.710, 0.720, 0.730,
        0.740, 0.750, 0.760, 0.770, 0.780, 0.790, 0.800, 0.810, 0.820, 0.830,
        0.840, 0.850, 0.860, 0.870, 0.880, 0.890, 0.900, 0.910, 0.920, 0.930,
        0.940, 0.950, 0.960, 0.970, 0.980, 0.990, 1.000, 1.020, 1.036, 1.063,
        1.080, 1.100, 1.120, 1.140, 1.160, 1.180, 1.200, 1.220, 1.240, 1.260,
        1.280, 1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600, 1.650, 1.700,
        1.750, 1.800, 1.850, 1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200,
        2.250, 2.300, 2.350, 2.400, 2.450, 2.500, 2.600, 2.700, 2.800, 2.900,
        3.000, 3.100, 3.200, 3.300, 3.400, 3.500, 3.600, 3.700, 3.800, 3.900,
        4.000, 4.100, 4.200, 4.300, 4.420, 4.500, 4.600, 4.700, 4.800, 4.900,
        5.000, 5.200, 5.400, 5.600, 5.800, 6.000, 6.200, 6.400, 6.600, 6.800,
        7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 10.500, 11.000, 11.500,
        12.000, 13.000, 13.900, 14.700, 15.600]),
    "to": np.array([
        0.248, 0.259, 0.269, 0.279, 0.289, 0.299, 0.305, 0.310, 0.315, 0.320,
        0.326, 0.332, 0.334, 0.340, 0.345, 0.350, 0.355, 0.360, 0.365, 0.369,
        0.374, 0.380, 0.384, 0.391, 0.395, 0.400, 0.405, 0.410, 0.420, 0.430,
        0.440, 0.460, 0.480, 0.500, 0.520, 0.540, 0.560, 0.580, 0.600, 0.620,
        0.640, 0.660, 0.680, 0.700, 0.720, 0.740, 0.760, 0.780, 0.800, 0.820,
        0.840, 0.860, 0.880, 0.900, 0.920, 0.940, 0.960, 0.980, 1.000, 1.020,
        1.040, 1.060,
        1.080, 1.100, 1.120, 1.140, 1.160, 1.180, 1.200, 1.220, 1.240, 1.260,
        1.280, 1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600, 1.650, 1.700,
        1.750, 1.800, 1.850, 1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200,
        2.250, 2.300, 2.350, 2.400, 2.450, 2.500, 2.600, 2.700, 2.800, 2.900,
        3.000, 3.100, 3.200, 3.300, 3.400, 3.500, 3.600, 3.700, 3.800, 3.900,
        4.000, 4.100, 4.200, 4.300, 4.420, 4.500, 4.600, 4.700, 4.800, 4.900,
        5.000, 5.200, 5.400, 5.600, 5.800, 6.000, 6.200, 6.400, 6.600, 6.800,
        7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 10.500, 11.000, 11.500,
        12.100, 12.500, 12.930, 13.450, 13.980, 14.740, 15.600]),
    "ss": np.array([
        0.248, 0.259, 0.269, 0.279, 0.289, 0.299, 0.305, 0.310, 0.315, 0.320,
        0.326, 0.332, 0.334, 0.340, 0.345, 0.350, 0.355, 0.360, 0.365, 0.369,
        0.374, 0.380, 0.384, 0.391, 0.395, 0.400, 0.405, 0.410, 0.420, 0.430,
        0.440, 0.460, 0.480, 0.500, 0.520, 0.540, 0.560, 0.580, 0.600, 0.620,
        0.640, 0.660, 0.670, 0.680, 0.690, 0.700, 0.720, 0.730, 0.740, 0.750,
        0.760, 0.770, 0.780, 0.790, 0.800, 0.820,
        0.840, 0.860, 0.880, 0.900, 0.920, 0.940, 0.960, 0.980, 1.000, 1.020,
        1.040, 1.060,
        1.080, 1.100, 1.120, 1.140, 1.160, 1.180, 1.200, 1.220, 1.240, 1.260,
        1.280, 1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600, 1.650, 1.700,
        1.750, 1.800, 1.850, 1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200,
        2.250, 2.300, 2.350, 2.400, 2.450, 2.500, 2.600, 2.700, 2.800, 2.900,
        3.000, 3.100, 3.200, 3.300, 3.400, 3.500, 3.600, 3.700, 3.800, 3.900,
        4.000, 4.100, 4.200, 4.300, 4.420, 4.530, 4.600, 4.700, 4.800, 4.900,
        5.050, 5.200, 5.400, 5.600, 5.800, 6.000, 6.200, 6.400, 6.600, 6.800,
        7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 10.500, 11.000, 11.500,
        12.100, 12.500, 12.930, 13.450, 13.980, 14.740, 15.600]),
    "lw": np.array([
        0.248, 0.259, 0.269, 0.279, 0.289, 0.299, 0.305, 0.310, 0.315, 0.320,
        0.326, 0.332, 0.334, 0.340, 0.345, 0.350, 0.355, 0.360, 0.365, 0.369,
        0.374, 0.380, 0.384, 0.391, 0.395, 0.400, 0.405, 0.410, 0.420, 0.430,
        0.440, 0.460, 0.480, 0.500, 0.520, 0.540, 0.560, 0.580, 0.600, 0.620,
        0.640, 0.660, 0.680, 0.700, 0.720, 0.740, 0.760, 0.780, 0.800, 0.820,
        0.840, 0.860, 0.880, 0.900, 0.920, 0.940, 0.960, 0.980, 1.000, 1.020,
        1.040, 1.060,
        1.080, 1.100, 1.120, 1.140, 1.160, 1.180, 1.200, 1.220, 1.240, 1.260,
        1.280, 1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600, 1.650, 1.700,
        1.750, 1.800, 1.850, 1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200,
        2.250, 2.300, 2.350, 2.400, 2.450, 2.500, 2.600, 2.700, 2.800, 2.900,
        3.000, 3.100, 3.200, 3.300, 3.400, 3.500, 3.600, 3.700, 3.800, 3.900,
        4.000, 4.100, 4.200, 4.300, 4.420, 4.500, 4.550, 4.600, 4.650, 4.700,
        4.750, 4.800, 4.850, 4.900, 4.950, 5.140, 5.400,
        5.600, 5.700, 5.800, 6.000, 6.200, 6.400, 6.600, 6.800,
        7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 10.500, 11.000, 11.500,
        12.100, 12.500, 12.930, 13.450, 13.980, 14.740, 15.600]),
    "lst": np.array([
        0.234, 0.250, 0.265, 0.272, 0.290, 0.312, 0.327,
        0.340, 0.345, 0.350, 0.355, 0.360, 0.365, 0.369,
        0.380, 0.390, 0.400, 0.420, 0.440, 0.460, 0.480, 0.500, 0.525, 0.550,
        0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725, 0.750, 0.775, 0.800,
        0.850, 0.900, 0.950,
        1.000, 1.200, 1.400, 1.600, 1.800,
        2.000, 2.250, 2.500, 2.750,
        3.000, 3.250, 3.500, 3.750,
        4.000, 4.250, 4.500, 4.750,
        5.000, 5.500, 6.000, 6.500, 7.000, 8.000, 9.000, 10.000, 11.000, 12.500,
        15.5]),
    }

# Replace SW's hand-placed nodes across the etalon band with the wavenumber-
# uniform block, keeping the hand grid either side of it. This is done here
# rather than by editing the literal above so that the fringe period and the
# nodes-per-cycle choice stay in one place and the grid follows if either is
# revised. Net effect on SW: 173 hand nodes -> 158, and because the penalty is
# now mesh-invariant the smaller node count does not change the smoothing GCV
# selects.
_sw_lo, _sw_hi = SW_ETALON_RANGE_UM
SRF_NODE_GRIDS["sw"] = np.sort(np.unique(np.concatenate([
    SRF_NODE_GRIDS["sw"][(SRF_NODE_GRIDS["sw"] < _sw_lo) | (SRF_NODE_GRIDS["sw"] > _sw_hi)],
    _etalon_node_block(range_um=SW_ETALON_RANGE_UM,
                       period_cm1=SW_ETALON_PERIOD_CM1,
                       nodes_per_cycle=SW_ETALON_NODES_PER_CYCLE)])))
del _sw_lo, _sw_hi

# Total: refine the UV turn-on, coarsen 0.42-4.5um, leave the rest of the hand
# grid in place. See the comment on TO_UV_RANGE_UM for the evidence.
SRF_NODE_GRIDS["to"] = np.sort(np.unique(np.concatenate([
    SRF_NODE_GRIDS["to"][(SRF_NODE_GRIDS["to"] < TO_UV_RANGE_UM[0]) |
                         (SRF_NODE_GRIDS["to"] > TO_MID_RANGE_UM[1])],
    _wavenumber_node_block(range_um=TO_UV_RANGE_UM, step_cm1=TO_UV_STEP_CM1),
    _wavenumber_node_block(range_um=TO_MID_RANGE_UM, step_cm1=TO_MID_STEP_CM1)])))

# LW zone boundaries and densities. LW is the one channel whose hand grid was
# badly misallocated, because it was laid out on the same template as the others
# without accounting for LW being out of band over three quarters of its span:
# 111 of its 154 nodes sat below 3.9um, where the SRF never exceeds 0.0015,
# while only 17 covered the 3.9-5.2um edge across which it rises to 0.78 - the
# steepest edge in the instrument.
#
# Chosen by 5-fold cross-validation on held-out chi, 12 seeds, against the
# campaign- and wavelength-corrected level_02 data. Zone by zone:
#
#   0.25-1.0um  Pure noise: 164 measurements spanning +/-3e-5 with no structure
#               resolvable above ~6e-5, against a peak in-band SRF of 0.78. It
#               carried 58 nodes, which were fitting that noise - held-out chi
#               in the UV fell 8.0 -> 1.6 and in the visible 8.8 -> 4.8 on
#               coarsening to 6. Monotone in node count all the way down; 6
#               rather than 4 only as insurance against structure the data
#               cannot see.
#
#   1.0-3.9um   A real out-of-band leakage bump, rising from zero to 1.2e-3
#               near 2.2um and falling back to a 2.2e-4 shelf - scatter/u of
#               40 about a constant, so it is genuine structure and needs
#               nodes. The hand grid's failure here was placement, not count:
#               it put 16 nodes into 1.0-1.3um where the SRF is flat zero and
#               thinned out over the bump itself. 30 log-uniform nodes beat
#               the 53 hand-placed ones, 59.5 -> 17.8 in 1-2um. Optimum is
#               broad over 30-40 and collapses past 50 (chi 41 at N=50).
#               Structure-aware placements - sparse/dense/sparse over the bump
#               - were all tried and all lost to plain log-uniform.
#
#   3.9-5.2um   The edge. Held-out chi is strongly NON-monotone in density
#               here: 0.15um spacing gives 27, while both coarser (0.25um ->
#               121) and finer (0.05um -> 895) are far worse. The hand grid
#               effectively sat on the fine side, stepping 0.05um across
#               4.42-4.95um before jumping to 5.14, which is why it scored 51.
#               The measurements allow about 4.5 points per 0.1um here, so
#               0.05um spacing leaves ~2 per node - too few to pin a
#               near-vertical edge, and the fit starts chasing the
#               measurement-to-measurement wavelength scatter instead.
#
#   5.2-15.6um  Left exactly as hand-placed. Every alternative tested was
#               worse; the hand grid is already dense over 5.4-7um where the
#               post-edge structure is and coarse beyond, which is right.
#
# Net: 154 nodes -> 69, held-out chi median 60.7 -> 13.6, and the instability
# is gone - the baseline's pooled chi ranged to 670 depending on the CV split
# (nodes outnumbering local data, so a held-out fold could leave a node
# unconstrained), while the new grid runs 13.6 median against a 14.8 maximum.
LW_OOB_NOISE_RANGE_UM = (0.248, 1.0)
LW_OOB_NOISE_NODES = 6
LW_OOB_LEAK_RANGE_UM = (1.0, 3.9)
LW_OOB_LEAK_NODES = 30
LW_EDGE_RANGE_UM = (3.9, 5.2)
LW_EDGE_STEP_UM = 0.15

# SSW: coarsen the flat plateau, leave both edges as hand-placed. See the
# comment on SS_FLAT_RANGE_UM.
SRF_NODE_GRIDS["ss"] = np.sort(np.unique(np.concatenate([
    SRF_NODE_GRIDS["ss"][(SRF_NODE_GRIDS["ss"] < SS_FLAT_RANGE_UM[0]) |
                         (SRF_NODE_GRIDS["ss"] > SS_FLAT_RANGE_UM[1])],
    _wavenumber_node_block(range_um=SS_FLAT_RANGE_UM, step_cm1=SS_FLAT_STEP_CM1)])))

# SW and SSW out-of-band and flat-plateau blocks. Both channels' hand grids
# carried the same misallocation LW's did, if less extremely: SW had 25 nodes
# above 5um where its SRF never exceeds 0.0018, and SSW had 28 below 0.42um plus
# 27 above 5um, 55 of its 138 nodes describing regions where it is blind.
#
# These grids were originally tuned BEFORE the per-campaign wavelength
# correction existed, on data still carrying the 12-20nm campaign error, and
# both channels' band edges sit at 3.9-5.0um squarely inside the affected band -
# so they were re-checked once the correction was in. The edges turned out to be
# fine exactly as hand-placed: unlike LW, where the optimum was NON-monotone and
# the hand grid sat on the wrong side of it, coarsening these edges is
# catastrophic (held-out chi at 0.30um spacing is 92 for SW and 126 for SSW,
# against 11 and 18 for the hand nodes). The falling edge of a channel dropping
# from 0.5 to 0.0009 needs every node it has. So only the out-of-band and flat
# regions changed:
#
#   SW  0.42-0.75um   flat, 0.735-0.773, on 60 measurements. 31 nodes -> 11
#                     wavenumber-uniform. Improves its own zone 7.83 -> 6.97 and
#                     the UV turn-on below it 7.13 -> 6.22.
#       5.0-15.6um    out of band, |SRF| <= 0.0018 on 206 measurements.
#                     25 nodes -> 6, zone chi 14.24 -> 13.21.
#
#   SSW 0.248-0.42um  out of band, |SRF| <= 1e-4. 28 nodes -> 4, and the zone
#                     halves, 2.63 -> 1.41.
#       5.0-15.6um    out of band. 27 nodes -> 14, zone 20.45 -> 19.57. Note
#                     this one wants MORE nodes than SW's equivalent: 6 there
#                     costs 21.9 against 19.6 at 14, the reverse of SW, so the
#                     two were tuned separately rather than sharing a constant.
#
# Net, over 12 CV seeds: SW 143 nodes -> 104 with pooled chi 11.33 -> 10.87,
# SSW 138 -> 100 with 17.95 -> 16.75. Modest in chi but a third of the nodes
# gone, and no zone materially worse.
SW_FLAT_RANGE_UM = (0.42, 0.749)
SW_FLAT_STEP_CM1 = 960.0
SW_OOB_IR_RANGE_UM = (5.0, 15.6)
SW_OOB_IR_NODES = 6
SS_OOB_UV_RANGE_UM = (0.248, 0.42)
SS_OOB_UV_NODES = 4
SS_OOB_IR_RANGE_UM = (5.0, 15.6)
SS_OOB_IR_NODES = 14

# LST zone boundaries and densities. LST is a broadband channel like Total: a
# steep UV turn-on and then an almost featureless plateau, the SRF rising only
# 0.875 to 1.02 across 0.4-15.5um. Its hand grid was not badly misallocated the
# way LW's was, but it had 48 nodes describing that plateau on 356 measurements.
#
# The Total grid was tried directly as a starting point and is WORSE than LST's
# own hand grid (held-out chi 4.57 against 3.30) - Total carries 579
# measurements to LST's 356, so its density does not transfer. Chosen instead by
# 5-fold cross-validation on held-out chi, 12 seeds:
#
#   0.234-0.29um  The declining tail below the shelf, SRF 0.07 down to 0.010 on
#                 11 measurements. Coarsened from 5 nodes to 2 with no loss
#                 (tail chi 4.72 -> 4.56, and the turn-on improves too). Every
#                 uniform rule tried here was far worse - log-uniform over
#                 0.234-0.330 sends the tail to 77 - because the region is too
#                 sparsely measured for a rule to place nodes better than hand.
#
#   0.30-0.42um   The shelf and the turn-on, left exactly as hand-placed. The
#                 turn-on is the steepest thing in the channel (dlnSRF/dlnwl
#                 reaches 102 at 0.333um) and the hand nodes beat every
#                 wavenumber-uniform block tried, at 17 to 44 fewer nodes. The
#                 shelf's chi of 7.3 is a floor, not a resolution limit: it does
#                 not improve at any node count between 1 and 5, so it is
#                 scatter-limited where the SRF is only 0.003-0.010.
#
#   0.42-4.5um    18 log-uniform nodes for the plateau proper.
#
#   4.5-15.5um    5 log-uniform nodes. This is the reduced IR density - only 35
#                 measurements between 5 and 9um and 14 above 9um - and it is
#                 also where the node count matters least, the SRF being
#                 smooth and monotone. Dropping to 3 is marginally better
#                 pooled (2.95 against 2.96) but degrades 9-14um from 1.43 to
#                 1.94, so 5 is kept.
#
# Net: 65 nodes -> 37, held-out chi median 3.30 -> 2.93 with the maximum over
# seeds falling 3.93 -> 3.03, and every single zone equal or better.
# The LST's radiometric reference: the final TWO campaigns, weighted equally,
# rather than the most recent one alone. The LST was used as the reference for a
# calibration transfer to the science radiometers in the interval BETWEEN those
# two campaigns, so they bookend the time range over which the LST response
# needs to be known best and neither end of it is privileged. Expressed as a
# rule over whatever campaigns are present rather than as hardcoded indices, so
# it survives a campaign being added or the indices being renumbered.
def lst_reference_campaigns(campaigns):
    c = sorted(np.unique(campaigns).tolist())
    return c[-2:] if len(c) >= 2 else c

LST_TAIL_NODES_UM = (0.234, 0.29)
LST_PLATEAU_RANGE_UM = (0.42, 4.5)
LST_PLATEAU_NODES = 18
LST_IR_RANGE_UM = (4.5, 15.5)
LST_IR_NODES = 5

# LW: replace everything below LW_EDGE_RANGE_UM[1] with the three blocks above,
# keeping the hand grid from 5.2um up. See the comment on
# LW_OOB_NOISE_RANGE_UM for the per-zone evidence.
SRF_NODE_GRIDS["lw"] = np.sort(np.unique(np.concatenate([
    SRF_NODE_GRIDS["lw"][SRF_NODE_GRIDS["lw"] > LW_EDGE_RANGE_UM[1]],
    _log_node_block(range_um=LW_OOB_NOISE_RANGE_UM, n_nodes=LW_OOB_NOISE_NODES),
    _log_node_block(range_um=LW_OOB_LEAK_RANGE_UM, n_nodes=LW_OOB_LEAK_NODES),
    _uniform_node_block(range_um=LW_EDGE_RANGE_UM, step_um=LW_EDGE_STEP_UM)])))

# SW: coarsen the flat 0.42-0.75um shoulder and the out-of-band IR. Applied
# after the etalon splice above, and neither block overlaps SW_ETALON_RANGE_UM.
SRF_NODE_GRIDS["sw"] = np.sort(np.unique(np.concatenate([
    SRF_NODE_GRIDS["sw"][(SRF_NODE_GRIDS["sw"] < SW_FLAT_RANGE_UM[0]) |
                         ((SRF_NODE_GRIDS["sw"] > SW_FLAT_RANGE_UM[1]) &
                          (SRF_NODE_GRIDS["sw"] < SW_OOB_IR_RANGE_UM[0]))],
    _wavenumber_node_block(range_um=SW_FLAT_RANGE_UM, step_cm1=SW_FLAT_STEP_CM1),
    _log_node_block(range_um=SW_OOB_IR_RANGE_UM, n_nodes=SW_OOB_IR_NODES)])))

# SSW: coarsen both out-of-band wings. Applied after the SS_FLAT splice above,
# and neither block overlaps SS_FLAT_RANGE_UM.
SRF_NODE_GRIDS["ss"] = np.sort(np.unique(np.concatenate([
    SRF_NODE_GRIDS["ss"][(SRF_NODE_GRIDS["ss"] > SS_OOB_UV_RANGE_UM[1]) &
                         (SRF_NODE_GRIDS["ss"] < SS_OOB_IR_RANGE_UM[0])],
    _log_node_block(range_um=SS_OOB_UV_RANGE_UM, n_nodes=SS_OOB_UV_NODES),
    _log_node_block(range_um=SS_OOB_IR_RANGE_UM, n_nodes=SS_OOB_IR_NODES)])))

# LST: keep the hand-placed shelf and turn-on between 0.30 and 0.42um, coarsen
# the tail below it, and replace the whole plateau above it with two log blocks.
# See the comment on LST_TAIL_NODES_UM for the per-zone evidence.
SRF_NODE_GRIDS["lst"] = np.sort(np.unique(np.concatenate([
    SRF_NODE_GRIDS["lst"][(SRF_NODE_GRIDS["lst"] >= 0.30) &
                          (SRF_NODE_GRIDS["lst"] <= LST_PLATEAU_RANGE_UM[0])],
    np.asarray(LST_TAIL_NODES_UM, dtype=float),
    _log_node_block(range_um=LST_PLATEAU_RANGE_UM, n_nodes=LST_PLATEAU_NODES),
    # The IR block shares its lower endpoint with the plateau block, so that
    # node is dropped here rather than being deduplicated by luck of rounding.
    _log_node_block(range_um=LST_IR_RANGE_UM, n_nodes=LST_IR_NODES + 1)[1:]])))

# Parametric fringe terms by channel, applied automatically by
# science_radiometer_srf_node_fit for any channel listed here. Registered
# centrally rather than passed by callers so that every path through the
# pipeline - the level_02 fit and the reference fits inside
# science_radiometer_srf_campaign_correct - uses the same fringe, which
# matters because a campaign offset estimated against a reference that lacks
# the fringe would absorb part of it.
#
# n_amplitude=2 (amplitude linear in scaled wavenumber) chosen by
# cross-validation with the 2/cycle envelope grid in place:
#
#   n_amplitude   held-out chi by zone                    pooled   coeff SNR
#             1    11.9   7.9   3.7   15.3                 11.54   130
#             2     6.5   7.0   3.0   15.3                 10.35   100, 38
#             3     6.8   7.0   3.1   15.3                 10.41    84, 34, 0
#
# A constant amplitude is clearly wrong - the fringe genuinely grows toward
# long wavelengths - but the quadratic term is not supported once the envelope
# nodes are present at 2/cycle, since they absorb that curvature themselves
# (SNR 0, and the pooled score gets marginally worse). With a sparser envelope
# the third term does earn its place, so this value is tied to the node
# density above and should be rechecked if that changes.
SRF_FRINGE_TERMS = {
    "sw": dict(two_nd_um=SW_ETALON_TWO_ND_UM,
               range_um=SW_FRINGE_RANGE_UM,
               taper_frac=0.15,
               n_amplitude=2),
    }

# Size of the fixed uniform mesh the smoothing penalty is normalized against,
# so that a given `smoothing` value means the same thing whatever node grid a
# channel uses. Its only role is normalization - it is not a fitting grid - so
# the exact value does not matter as long as it stays constant; changing it
# rescales every channel's `smoothing` by the same factor.
PENALTY_REFERENCE_NODES = 100

def _srf_node_design_matrix(*, x_nodes, x_eval):
    """
    Design matrix for an SRF built by natural-cubic-spline interpolation of node
    values.

    The whole point of the node fit is that interpolation from node values is a
    LINEAR operator: srf(x) = sum_j p_j * phi_j(x), where phi_j is the spline
    through node value 1 at node j and 0 at every other node. So the model is
    linear in the free parameters p even though the interpolant is cubic, and
    the fit is exact weighted least squares - no optimizer, no starting guess,
    no local minima, and an exact parameter covariance matrix. (The IDL version
    ran this same problem through mpfit, which solved it iteratively.)

    Returns an (len(x_eval), len(x_nodes)) matrix whose column j is phi_j
    sampled at x_eval.
    """

    from scipy.interpolate import CubicSpline

    n_nodes = len(x_nodes)
    b = np.empty((len(x_eval), n_nodes))

    for j in range(n_nodes):
        e = np.zeros(n_nodes)
        e[j] = 1.0

        # bc_type='natural' (zero second derivative at both ends) rather than
        # the default 'not-a-knot'. Beyond the outermost node the spline is
        # extrapolating, and a natural end condition extends it linearly
        # instead of continuing a cubic, which is the more conservative choice
        # where there is no data.
        b[:, j] = CubicSpline(x_nodes, e, bc_type='natural', extrapolate=True)(x_eval)

    return b

def _srf_node_design_matrix_convolved(*, x_nodes, x_kernel, w_kernel):
    """
    Design matrix whose row i is the laser-weighted average of each node basis
    function over measurement i's line profile, rather than that basis
    evaluated at a single wavelength.

    x_kernel and w_kernel are (n_meas, n_pad) as returned by
    load_laser_line_kernels, x_kernel in log10(wavelength) and each row of
    w_kernel summing to 1.

    One spline evaluation per node over the whole flattened kernel set, rather
    than per measurement, which is what keeps this cheap: ~0.09s for a
    600 x 143 matrix with 112-point kernels.
    """

    from scipy.interpolate import CubicSpline

    n_nodes = len(x_nodes)
    n_meas, n_pad = x_kernel.shape
    flat = x_kernel.ravel()

    b = np.empty((n_meas, n_nodes))
    for j in range(n_nodes):
        e = np.zeros(n_nodes)
        e[j] = 1.0
        phi = CubicSpline(x_nodes, e, bc_type='natural',
                          extrapolate=True)(flat).reshape(n_meas, n_pad)
        b[:, j] = (phi*w_kernel).sum(axis=1)

    return b

def _srf_node_penalty_matrix(x_nodes):
    """
    Second-difference penalty matrix P for node values on the (unevenly spaced)
    grid x_nodes, such that p @ P @ p is the sum of squared discrete second
    derivatives.

    This is what makes the node fit well-posed. With 173 SW nodes against ~600
    measurements the median node interval holds 2 points, 39 intervals hold a
    single point and 1 is empty, so an unpenalized spline is free to put
    whatever it likes in the under-constrained intervals - the same ringing
    failure the GPR had, in a different basis. Penalizing the second difference
    says "the SRF is smooth at the scale of the node spacing unless the data
    insists otherwise", which is a statement about the SRF rather than about
    the noise.

    It also fixes the extrapolation problem for free: the node grids run to
    15.6um but there are only 18 measurements above 13um, and driving the
    second difference to zero outside the data extends the curve linearly
    rather than letting an unconstrained node run away.
    """

    n = len(x_nodes)
    d = np.zeros((n - 2, n))

    for i in range(1, n - 1):
        h1 = x_nodes[i] - x_nodes[i - 1]
        h2 = x_nodes[i + 1] - x_nodes[i]

        # Standard three-point second derivative on an uneven grid:
        #   f'' ~ 2/(h1+h2) * [ (f_{i+1}-f_i)/h2 - (f_i-f_{i-1})/h1 ]
        #
        # Each row is then scaled by sqrt of the node's share of the abscissa,
        # so that p @ P @ p approximates the INTEGRAL of (f'')^2 dx rather than
        # a bare sum of (f'')^2 over nodes. This is what makes the penalty
        # mesh-invariant, and it matters more than it looks.
        #
        # Without the weighting, refining the grid leaves f'' unchanged for the
        # same underlying curve but adds terms to the sum, so the penalty grows
        # roughly in proportion to the node count. GCV then compensates by
        # raising the smoothing weight, and the fit comes out SMOOTHER for
        # having been given more resolution. Measured on the SW channel before
        # this change: going from 173 to 216 nodes made GCV raise the smoothing
        # from 2.6e-2 to 2.7e-1, a factor of 10, and the recovered amplitude of
        # the 604 cm-1 etalon fringe fell from 67-78% of the data's to 36-80%.
        # With the smoothing held fixed instead, the same 216-node grid
        # recovered 86-92% - i.e. the extra nodes were doing their job and the
        # penalty rescaling was undoing it.
        #
        # With the integral form, p @ P @ p converges to a fixed functional as
        # the mesh refines, so node density and smoothness become independent
        # choices: adding nodes where structure is believed to exist no longer
        # silently smooths everything else.
        h_share = 0.5*(h1 + h2)
        scale = math.sqrt(h_share)

        d[i - 1, i - 1] = scale*2.0/((h1 + h2)*h1)
        d[i - 1, i] = -scale*2.0/(h1*h2)
        d[i - 1, i + 1] = scale*2.0/((h1 + h2)*h2)

    return d.T @ d

def _srf_local_residual_scatter(*, wavelength_um, residual, u_residual,
                                wl_out, half_width_frac=0.10, min_n=3):
    """
    Local scatter of the fit residuals, evaluated on the wl_out grid. This is
    the second of the node fit's two uncertainty terms, and it is the one that
    carries the information the stated per-point uncertainties do not have.

    Ported from the IDL fit's boxcar (dwl = wl/5, i.e. +/-10% in wavelength),
    with one deliberate change. IDL took |weighted mean residual| plus the
    standard error of that weighted mean, sqrt(1/sum(w)), which is derived
    entirely from the stated u and therefore inherits the fact that u is far
    too small: measured against each channel's own residual scatter, u is low
    by 2.9x on the LST, 7.6x on Total, 21x on SW, 43x on SSW and 413x on LW.
    Here the standard error is taken as the LARGER of that stated-u value and
    the empirical one computed from the actual weighted spread of residuals in
    the window, so a region where the measurements disagree by more than they
    claim to reports the disagreement.

    The two parts mean different things and are added, not RSS'd, following
    IDL: the first is local bias (the fit is systematically off here, which the
    node grid may be too coarse to follow) and the second is local noise.

    Returns an array on wl_out, NaN where a window holds fewer than min_n
    points; the caller interpolates across those.
    """

    w = np.where(u_residual > 0, 1.0/np.maximum(u_residual, 1e-300)**2, 0.0)
    scatter = np.full(len(wl_out), np.nan)

    for m, wl in enumerate(wl_out):

        sel = ((wavelength_um > wl*(1.0 - half_width_frac)) &
               (wavelength_um < wl*(1.0 + half_width_frac)) &
               np.isfinite(residual) & (w > 0))

        n = int(sel.sum())
        if n < min_n:
            continue

        ww = w[sel]
        rr = residual[sel]
        sum_w = ww.sum()

        mean_res = float((rr*ww).sum()/sum_w)

        # Standard error from the stated uncertainties
        se_stated = math.sqrt(1.0/sum_w)

        # Standard error from the observed spread. Both factors here have to
        # use the EFFECTIVE sample size, not n: the stated uncertainties within
        # one window span up to four orders of magnitude (LW's u runs 1.8e-6 to
        # 1.7e-2), so a handful of high-weight points dominate the weighted
        # mean and the naive var/n understates its error by up to sqrt(600/85).
        # n_eff = (sum w)^2 / sum w^2 is the usual effective count for a
        # weighted mean, equal to n when the weights are equal.
        n_eff = float(sum_w**2/(ww**2).sum())

        if n_eff <= 1:
            se_emp = 0.0
        else:
            # n_eff/(n_eff-1) is the bias correction for estimating a variance
            # from the same points used to form the mean.
            var_emp = float((ww*(rr - mean_res)**2).sum()/sum_w)*n_eff/(n_eff - 1)
            se_emp = math.sqrt(var_emp/n_eff)

        scatter[m] = abs(mean_res) + max(se_stated, se_emp)

    return scatter

def load_laser_line_kernels(*, spectrum_filename, wavelength_um, wavelength_um_opa,
                            wavelength_source, half_width_factor=1.5):
    """
    Loads the measured laser line profile for each measurement, for use as the
    convolution kernel in science_radiometer_srf_node_fit.

    Why this exists. What a measurement actually reports is not the SRF at one
    wavelength. The radiometer signal is proportional to the integral of
    S(wl)*L(wl), and the PBR-R it is ratioed against is spectrally flat over a
    linewidth, so its signal is proportional to the integral of L(wl) alone.
    The measured ratio is therefore EXACTLY the laser-weighted average of S,
    <S>_L - not S at any single wavelength. Treating it as S(centroid) is the
    approximation, and it is a poor one wherever the SRF is steep:

        <S>_L = S(c) + (1/2)*S''(c)*Var(L) + higher moments

    Measured on the 31 OSA207 spectra between 4.15 and 4.90um, the difference
    |S(c) - <S>_L| has a median of 0.0086 on SSW and 0.0038 on LW, equivalent
    to apparent wavelength shifts of 14 and 23nm. An independent estimate -
    solving the cross-channel residuals for one wavelength error per
    measurement - gave 12.8nm at 4.2-4.5um and 17.5nm at 4.5-4.8um. The two
    agree, which is what identified the centroid approximation as the cause of
    that region's residuals rather than any error in the wavelength itself.

    Note the effect cannot be absorbed by correcting wavelengths. Where the
    SRF slope passes through a minimum (SSW near 4.43-4.56um) a wavelength
    shift produces no response change at all, yet the bias is still ~0.01;
    expressed as an equivalent shift it diverges there. Only carrying the full
    line profile handles it.

    Kernels are returned only for measurements whose recorded profile can be
    trusted as the light the radiometer saw:

      wavelength_source == 2   OSA207. An FTS, so its own instrument function
                               is negligible against the laser line and the
                               recorded profile is the laser itself.
      anything else            A DELTA kernel at the reported wavelength, i.e.
                               the centroid model unchanged. This covers the
                               CCS200 (168 of 600 measurements), whose grating
                               instrument function is NOT negligible - the
                               recorded width there is 113-163 cm-1 against
                               the 62-86 cm-1 the FTS sees, so convolving with
                               it would double-count the spectrometer's own
                               broadening - and the grating spectrometer (16).
                               The bias being corrected is small below 1um
                               anyway (~0.8nm equivalent).

    The path-length caveat: the OSA207's optical path is about 10% longer than
    the radiometer's, so its line is carved slightly more deeply by atmospheric
    absorption than the beam the radiometer measured. Treating the two as
    identical therefore over-corrects by roughly 10% of the bias, leaving ~1e-3
    of the ~1e-2 it removes.

    Returns a SimpleNamespace with, aligned row-for-row with the inputs:

      wavelength_um  (n_meas, n_pad) sample wavelengths
      weight         (n_meas, n_pad) weights, each row summing to 1, padded
                     with zeros
      n_real         how many rows got a measured profile rather than a delta

    Padded rather than ragged so the design matrix can be built with one spline
    evaluation per node over the whole flattened set - see the kernel branch in
    science_radiometer_srf_node_fit. Padding entries carry zero weight and
    repeat the centroid wavelength, so they contribute nothing but stay inside
    the interpolation range.
    """

    paths = load_config()

    spectrum_filename = np.asarray(spectrum_filename, dtype=object)
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    wavelength_um_opa = np.asarray(wavelength_um_opa, dtype=float)
    wavelength_source = np.asarray(wavelength_source)

    n_meas = len(wavelength_um)
    profiles = [None]*n_meas

    for i in range(n_meas):

        if wavelength_source[i] != 2:
            continue

        fn = spectrum_filename[i]
        if not isinstance(fn, str):
            continue

        path = paths.erf_osa_spectrum_dir / fn
        if not path.exists():
            continue

        sp = pd.read_csv(path)
        if "wavelength[um]" not in sp.columns or "signal" not in sp.columns:
            continue

        # Same window and background convention as analyze_osa207_spectrum, so
        # the kernel is the profile the reported centroid was computed from.
        # fwhm_est comes from the OPA setpoint via the same empirical scaling,
        # while the window is centred on the measured centroid rather than the
        # setpoint - the setpoint is off by -1.7 to +2.5 fwhm_est across this
        # dataset, which is why that function locates the line first.
        fwhm_est = (3.142*wavelength_um_opa[i] + 7.204*wavelength_um_opa[i]**2)/1000
        if not np.isfinite(fwhm_est) or fwhm_est <= 0:
            continue

        wl = sp["wavelength[um]"].values
        sig = sp["signal"].values
        centre = wavelength_um[i]

        half_width = half_width_factor*fwhm_est
        in_window = np.abs(wl - centre) < half_width
        in_annulus = ((np.abs(wl - centre) > half_width + 0.25*fwhm_est) &
                      (np.abs(wl - centre) < half_width + 1.50*fwhm_est))

        if in_window.sum() < 20 or in_annulus.sum() < 8:
            continue

        base_coeffs = np.polyfit(wl[in_annulus], sig[in_annulus], 1)

        # Clipped at zero for the same reason as in the centroid calculation:
        # background-level noise must not contribute negative weight.
        w = wl[in_window]
        y = np.clip(sig[in_window] - np.polyval(base_coeffs, w), 0.0, None)

        total = y.sum()
        if not np.isfinite(total) or total <= 0:
            continue

        # Sorted so the stored profile is monotonic in wavelength; the OSA
        # writes its trace in descending wavelength order.
        order = np.argsort(w)
        profiles[i] = (w[order], y[order]/total)

    n_real = sum(p is not None for p in profiles)
    n_pad = max((len(p[0]) for p in profiles if p is not None), default=1)

    kern_wl = np.repeat(wavelength_um[:, None], n_pad, axis=1)
    kern_w = np.zeros((n_meas, n_pad))

    for i, p in enumerate(profiles):
        if p is None:
            # Delta kernel: all the weight on the reported wavelength.
            kern_w[i, 0] = 1.0
            continue
        w, y = p
        kern_wl[i, :len(w)] = w
        kern_w[i, :len(w)] = y

    return SimpleNamespace(wavelength_um=kern_wl, weight=kern_w, n_real=n_real,
                           n_meas=n_meas, n_pad=n_pad)

def _srf_fringe_design_columns(*, wavelength_um, two_nd_um, range_um, taper_frac=0.15,
                               n_amplitude=2):
    """
    Design-matrix columns for a parametric etalon fringe, for use alongside the
    node columns in science_radiometer_srf_node_fit.

    An etalon of optical thickness n*d has transmission maxima where
    2*n*d*nu = m for integer m, so the fringe is

        A(nu) * cos(2*pi*nu*2nd)

    with NO free phase - a maximum necessarily sits at nu = 0. That was tested
    against the SW data rather than assumed: adding a free phase parameter
    gives F = 0.78, p = 0.38, i.e. it is not justified, and the locked
    one-parameter form fits as well (chi2 88868 vs 88536). Fitted optical
    thickness 2nd = 16.50 +/- 0.17um (n*d = 8.25um), stable across sub-bands
    (16.485 +/- 0.432, 16.514 +/- 0.153, 16.394 +/- 0.109), so no dispersion is
    detectable at the ~1% level.

    Because 2nd is fixed here from that measurement, cos(2*pi*nu*2nd) is a
    known function of wavelength and the fringe enters LINEARLY through its
    amplitude - so the whole fit stays exact weighted linear least squares,
    exactly as the node interpolation does. A(nu) is taken as a polynomial in
    nu of n_amplitude terms, which is what makes it linear: the returned
    columns are cos(.), cos(.)*nu', cos(.)*nu'^2, ... with nu' a centred and
    scaled wavenumber.

    The fringe is confined to range_um with a smoothstep taper over the outer
    taper_frac of each end (in log10 wavelength). Without a taper the model
    would step discontinuously at the range edges; without a range the linear
    amplitude term would be extrapolated across the whole channel and diverge.
    """

    nu = 1e4/np.asarray(wavelength_um, dtype=float)

    # Centred, scaled wavenumber so the amplitude polynomial is well
    # conditioned regardless of the band.
    nu_lo, nu_hi = 1e4/range_um[1], 1e4/range_um[0]
    nu_mid = 0.5*(nu_lo + nu_hi)
    nu_half = 0.5*(nu_hi - nu_lo)
    nu_scaled = (nu - nu_mid)/max(nu_half, 1e-300)

    # Smoothstep window in log10(wavelength), 1 inside and 0 outside, with the
    # 3t^2-2t^3 ramp whose derivative vanishes at both ends so the fitted SRF
    # picks up no kink where the fringe term starts and stops.
    x = np.log10(np.asarray(wavelength_um, dtype=float))
    x_lo, x_hi = math.log10(range_um[0]), math.log10(range_um[1])
    ramp = max(taper_frac*(x_hi - x_lo), 1e-300)

    t_up = np.clip((x - x_lo)/ramp, 0.0, 1.0)
    t_dn = np.clip((x_hi - x)/ramp, 0.0, 1.0)
    window = (t_up*t_up*(3.0 - 2.0*t_up))*(t_dn*t_dn*(3.0 - 2.0*t_dn))

    carrier = window*np.cos(2.0*np.pi*nu*two_nd_um*1e-4)

    return np.column_stack([carrier*nu_scaled**p for p in range(n_amplitude)])

def science_radiometer_srf_node_fit(*, wavelength_um, r_srf, u_srf_random, channel,
                                    smoothing=None, n_gcv=40, fringe="auto",
                                    kernels=None):
    """
    Fits an SRF as a natural cubic spline through a per-channel grid of free
    node values, by penalized weighted linear least squares, and returns the
    fit and its uncertainty on the same dense grid (and with the same column
    names) as the GPR fit this replaced, so callers needed no change.

    This exists because Gaussian process regression is the wrong prior for this
    measurement, in two ways that no kernel or bound choice fixes:

    1. A single learned noise level for a dataset whose real scatter varies by
       two orders of magnitude across the span. The optimizer sets WhiteKernel
       to a variance-weighted compromise over 0.26-15.5um, so the noisy IR
       drags it up and it is then applied to the clean visible. That is what
       washed out SW's genuine etalon oscillation from 0.4-2.2um: the fit was
       told that scatter was noise when it is signal.
    2. Reversion to the prior away from data. A GP's predictive variance tends
       to the prior signal variance where the data thins out, which is the
       opposite of the truth out of band - there the SRF is known to be ~0 and
       the measurements are MORE repeatable than in band, so the uncertainty
       should be small. That is why the SSW and LW out-of-band uncertainties
       came out far too large.

    The node fit inverts where the knowledge lives. Instead of asking a
    stationary kernel to discover the resolution from the data, the per-channel
    node grid (SRF_NODE_GRIDS, placed by hand over six years and sitting at
    about one node per laser linewidth) states it, and the fit only has to
    solve for levels. Because interpolation from node values is linear, that is
    exact weighted least squares - see _srf_node_design_matrix.

    Uncertainty is the sum in quadrature of:

      parameter  sqrt(diag(B Cov(p) B')), propagated from the stated per-point
                 u through the penalized-least-squares sandwich covariance.
                 This is what the stated uncertainties support, and it is
                 small.
      scatter    the local residual scatter, _srf_local_residual_scatter. This
                 is what the measurements actually disagree by, and given that
                 the stated u understates the real scatter by 3-413x depending
                 on channel it is the dominant term nearly everywhere.

    Keeping both matters: the parameter term knows about node density and data
    density (it grows where the grid is finer than the data can support), while
    the scatter term knows about real repeatability (it stays small out of band,
    where the GPR's did not).

    Arguments
      channel     key into SRF_NODE_GRIDS: 'sw', 'to', 'lw', 'ss' or 'lst'
      smoothing   penalty weight. None (default) selects it by generalized
                  cross-validation over n_gcv values; pass a float to fix it.
                  The penalty is pre-scaled so that smoothing=1 makes the
                  penalty term comparable in size to the data term, which
                  keeps the GCV search range meaningful across channels.
    """

    if channel not in SRF_NODE_GRIDS:
        raise ValueError(f"No node grid for channel {channel!r}; "
                         f"have {sorted(SRF_NODE_GRIDS)}.")

    # Fit in log10(wavelength), matching the coordinate the GPR fit used and the
    # coordinate the node grids are roughly uniform in. That last point matters
    # for the penalty: a second-difference penalty with one weight for the whole
    # span is only well scaled if the node spacing is roughly uniform in the
    # penalty coordinate, and the grids are close to log-spaced (SW spans 4.4
    # decades of node spacing in um but only ~1.5 in log10 um).
    # "auto" (the default) takes whatever parametric term is registered for
    # this channel, so callers do not each have to know about it and cannot
    # accidentally disagree. Pass fringe=None to fit without one, or an
    # explicit dict to override.
    if isinstance(fringe, str):
        if fringe != "auto":
            raise ValueError(f"fringe must be 'auto', None, or a dict; got {fringe!r}.")
        fringe = SRF_FRINGE_TERMS.get(channel)

    nodes_um = SRF_NODE_GRIDS[channel]
    a = np.log10(nodes_um)

    good = (np.isfinite(wavelength_um) & np.isfinite(r_srf) &
            np.isfinite(u_srf_random) & (u_srf_random > 0))
    x = np.log10(wavelength_um[good])
    y = r_srf[good]
    u = u_srf_random[good]

    if len(x) <= 2:
        raise ValueError(f"Only {len(x)} usable points for the {channel} node fit.")

    p_pen = _srf_node_penalty_matrix(a)
    n_nodes = len(nodes_um)

    # The design matrix. Two forms, and the difference is what the fit means.
    #
    # Without kernels, row i evaluates the basis at that measurement's single
    # centroid wavelength, so the fitted node values describe the SRF as the
    # centroid model sees it.
    #
    # With kernels, row i is the LASER-WEIGHTED AVERAGE of each basis function
    # over that measurement's measured line profile. Since the measured ratio
    # to the PBR-R is itself exactly <S>_L (see load_laser_line_kernels), this
    # makes the forward model exact rather than approximate, and the fitted
    # node values then describe the PRE-convolution SRF - the fit is doing a
    # deconvolution, constrained by the smoothing penalty.
    #
    # Convolution is linear and interpolation is linear, so either way the
    # problem stays exact weighted linear least squares. Building the convolved
    # matrix costs ~0.09s per channel against 0.009s for the centroid form.
    if kernels is None:
        x_conv = None
        w_conv = None
        b = _srf_node_design_matrix(x_nodes=a, x_eval=x)
    else:
        if kernels.n_meas != len(wavelength_um):
            raise ValueError(
                f"kernels cover {kernels.n_meas} measurements but "
                f"{len(wavelength_um)} were passed.")

        # Clipped into the node range so the natural-spline extrapolation is
        # not asked to reach beyond where it is meaningful; the outermost nodes
        # sit outside the data on every channel, so this bites only on the
        # extreme wings of a few kernels.
        x_conv = np.clip(np.log10(kernels.wavelength_um[good]), a.min(), a.max())
        w_conv = kernels.weight[good]

        # Renormalize after masking: nothing is dropped here, but a kernel that
        # arrived un-normalized would otherwise bias its row.
        row_sum = w_conv.sum(axis=1, keepdims=True)
        w_conv = np.divide(w_conv, row_sum, out=np.zeros_like(w_conv),
                           where=row_sum > 0)

        b = _srf_node_design_matrix_convolved(x_nodes=a, x_kernel=x_conv,
                                              w_kernel=w_conv)

    # Optional parametric etalon fringe, appended as extra design columns. It
    # is left OUT of the penalty (zero-padded below), which is deliberate and
    # is the mechanism that makes it useful: the smoothing penalty charges the
    # node spline for curvature but charges the fringe term nothing, so an
    # oscillation the fringe can explain gets attributed there rather than
    # being fought over by the nodes and then partly smoothed away.
    n_fringe = 0
    if fringe is not None:
        if kernels is None:
            b_fringe = _srf_fringe_design_columns(wavelength_um=10.0**x, **fringe)
        else:
            # The fringe columns must go through the kernel too, not just the
            # nodes. This is not bookkeeping: a 604 cm-1 fringe convolved with
            # an ~80 cm-1 laser line is attenuated by
            # exp(-2 pi^2 sigma^2 / P^2) ~ 0.85-0.95, so the pre-convolution
            # amplitude is 5-15% larger than the measured one. Convolving here
            # is what lets the fitted amplitude be the real one rather than the
            # smeared one.
            flat = _srf_fringe_design_columns(wavelength_um=10.0**x_conv.ravel(),
                                              **fringe)
            b_fringe = (flat.reshape(x_conv.shape[0], x_conv.shape[1], -1)
                        * w_conv[:, :, None]).sum(axis=1)

        n_fringe = b_fringe.shape[1]
        b = np.hstack([b, b_fringe])
        p_pen = np.pad(p_pen, ((0, n_fringe), (0, n_fringe)))

    # Weighted normal equations. w is 1/variance, so this is the standard
    # chi-square-minimizing weighting and each measurement enters with its own
    # uncertainty - no global noise scalar anywhere in this fit.
    w = 1.0/u**2
    btwb = b.T @ (w[:, None]*b)
    btwy = b.T @ (w*y)

    # Scale the penalty so that smoothing=1 puts the penalty term on the same
    # footing as the data term. Without this the meaningful range of smoothing
    # depends on the units of both the response and the penalty coordinate, and
    # differs between channels.
    #
    # The reference magnitude is taken from a FIXED uniform mesh across this
    # channel's span, not from the actual node grid. trace(P) is dominated by
    # the highest-frequency mode the mesh can represent and so grows like
    # h^-4 as the grid refines - normalizing by it would reintroduce exactly
    # the node-count coupling that the integral weighting in
    # _srf_node_penalty_matrix removes, and over-correct it. Referencing a
    # fixed mesh keeps smoothing=1 meaning the same thing regardless of how
    # many nodes the channel's grid happens to have.
    #
    # trace(btwb) is taken over the NODE block only, so that adding fringe
    # columns does not change what a given smoothing value means.
    p_ref = _srf_node_penalty_matrix(np.linspace(a.min(), a.max(), PENALTY_REFERENCE_NODES))
    scale = np.trace(btwb[:n_nodes, :n_nodes])/max(np.trace(p_ref), 1e-300)
    p_pen = p_pen*scale

    n = len(x)

    def _solve(lam):
        m = btwb + lam*p_pen
        p_hat = np.linalg.solve(m, btwy)
        return p_hat, m

    def _gcv(lam):
        p_hat, m = _solve(lam)
        resid = y - b @ p_hat
        wrss = float((w*resid**2).sum())

        # tr(H) for H = B (B'WB + lam P)^-1 B'W, the effective number of
        # parameters the fit is actually using. Computed as
        # tr((B'WB + lam P)^-1 B'WB) so it stays an (n_nodes x n_nodes)
        # operation rather than forming the n x n hat matrix.
        edof = float(np.trace(np.linalg.solve(m, btwb)))

        denom = 1.0 - edof/n
        if denom <= 0:
            return np.inf, p_hat, edof

        return (wrss/n)/denom**2, p_hat, edof

    if smoothing is None:

        # GCV rather than a hand-tuned value. Note GCV is safe here despite the
        # stated u being badly understated: its criterion is scale-free in u
        # (multiplying every u by a constant scales wrss by that constant and
        # leaves the minimizing lam unchanged), so it depends only on the
        # RELATIVE weighting across points, which is the part of u that is
        # trustworthy. This is a real advantage over the GPR, whose fitted
        # noise_level was in absolute units and so was sensitive to exactly the
        # thing that is wrong.
        lam_grid = np.logspace(-8, 2, n_gcv)
        best = (np.inf, None, None, None)
        for lam in lam_grid:
            score, p_hat, edof = _gcv(lam)
            if score < best[0]:
                best = (score, lam, p_hat, edof)
        gcv_score, lam, p_hat, edof = best

        if p_hat is None:
            raise RuntimeError(f"GCV found no usable smoothing for {channel}.")
    else:
        lam = float(smoothing)
        gcv_score, p_hat, edof = _gcv(lam)

    m = btwb + lam*p_pen
    m_inv = np.linalg.inv(m)

    # Sandwich covariance for the PENALIZED estimator,
    # Cov(p) = M^-1 (B'WB) M^-1 with M = B'WB + lam P. Not simply M^-1: that
    # is the Bayesian/ridge posterior covariance, which quietly includes the
    # penalty as if it were prior information about the SRF levels. The
    # sandwich form is the sampling covariance of this estimator given the
    # stated per-point u, which is what the uncertainty budget wants.
    cov_p = m_inv @ btwb @ m_inv

    # Prediction grid, identical to the GPR fit's so the two are directly
    # comparable and downstream code needs no change.
    wl_pred = np.logspace(math.log10(0.26), math.log10(14), 5000)
    x_pred = np.log10(wl_pred)

    b_pred = _srf_node_design_matrix(x_nodes=a, x_eval=x_pred)

    if n_fringe:
        b_pred = np.hstack([b_pred,
                            _srf_fringe_design_columns(wavelength_um=wl_pred, **fringe)])

    srf_pred = b_pred @ p_hat

    # diag(B Cov B') without forming the 5000x5000 matrix
    var_param = np.einsum('ij,jk,ik->i', b_pred, cov_p, b_pred)
    sd_param = np.sqrt(np.maximum(var_param, 0.0))

    # Local residual scatter. Computed on a coarse log grid (as in IDL, which
    # used 250 points over 0.3-14um) and interpolated up, because the boxcar
    # windows overlap heavily and evaluating it at all 5000 output points would
    # be both slow and no more informative.
    resid = y - b @ p_hat
    wl_scatter = np.logspace(math.log10(0.26), math.log10(14), 250)
    scatter = _srf_local_residual_scatter(
        wavelength_um=10.0**x, residual=resid, u_residual=u, wl_out=wl_scatter)

    # Fill any windows that held too few points by interpolating from the ones
    # that did, then carry the nearest good value out to the grid ends.
    ok = np.isfinite(scatter)
    if not ok.any():
        sd_scatter = np.zeros_like(wl_pred)
    else:
        sd_scatter = np.interp(wl_pred, wl_scatter[ok], scatter[ok])

    srf_pred_sd = np.sqrt(sd_param**2 + sd_scatter**2)

    result_df = pd.DataFrame({
        "wavelength_um": wl_pred,
        "srf_pred": srf_pred,
        "srf_pred_sd": srf_pred_sd,
        # Present only so this is column-compatible with the GPR fit, which
        # blended a separate UV fit into the main one. The node fit needs no
        # such split - the node grids are already dense through the UV turn-on,
        # which is the entire reason the GPR needed two fits - so this is
        # always 1 and no blending happens.
        "blend_weight": 1.0,
        "srf_pred_sd_param": sd_param,
        "srf_pred_sd_scatter": sd_scatter
    })

    # Fit diagnostics, attached to the frame rather than returned separately so
    # the drop-in signature is preserved.
    result_df.attrs.update({
        "channel": channel,
        "smoothing": lam,
        "gcv_score": gcv_score,
        "edof": edof,
        "n_nodes": len(nodes_um),
        "n_data": n,
        "chi2_reduced": float((w*resid**2).sum()/max(n - edof, 1.0)),
        "resid_rms": float(np.sqrt((resid**2).mean())),
        "node_wavelength_um": nodes_um,
        "node_value": p_hat[:n_nodes],
        "node_value_sd": np.sqrt(np.maximum(np.diag(cov_p)[:n_nodes], 0.0)),
        # Fitted fringe amplitude coefficients and their uncertainties, empty
        # when no fringe term was requested. The first is the amplitude at band
        # centre; the rest are its polynomial dependence on scaled wavenumber.
        "fringe": fringe,
        "fringe_coeff": p_hat[n_nodes:],
        "fringe_coeff_sd": np.sqrt(np.maximum(np.diag(cov_p)[n_nodes:], 0.0)),
        # Whether the forward model convolved with the measured laser line. When
        # it did, srf_pred is the PRE-convolution SRF, so it is not directly
        # comparable point-by-point with a centroid-model fit.
        "convolved": kernels is not None,
        "n_real_kernels": (kernels.n_real if kernels is not None else 0)
    })

    return result_df

def _polyval_variance(cov, x):
    """
    Variance of a fitted polynomial's value at each point in x, propagated
    from the coefficient covariance matrix cov returned by
    np.polyfit(..., cov=True/'unscaled'). For y(x) = phi(x) @ coeffs, the
    variance is phi(x) @ cov @ phi(x) - this evaluates that for every x at
    once instead of one point at a time.
    """

    order = cov.shape[0] - 1

    # phi(x) holds [x**order, x**(order-1), ..., x**0] per row, matching the
    # highest-degree-first coefficient order that np.polyfit/np.polyval use.
    phi = np.vstack([x**p for p in range(order, -1, -1)]).T

    # phi @ cov @ phi.T, but only the diagonal (one point's variance at a
    # time) is wanted, not the full cross-covariance between different x's.
    return np.einsum('ij,jk,ik->i', phi, cov, phi)

def science_radiometer_srf_campaign_correct(*, wavelength_um, r_srf, u_srf_random,
                                                  measurement_campaign, channel,
                                                  poly_order=2, min_n_for_poly=10,
                                                  response_floor=0.08, n_iter=8,
                                                  max_extrap_ratio=1.0, max_log_slope=10.0,
                                                  reference_campaigns=None,
                                                  wavelength_offset_frac=None,
                                                  wavelength_offset_applies=None,
                                                  estimate_wavelength_offset=True,
                                                  min_n_for_wavelength=5):
    """
    Iteratively fit a shared reference curve, describe each
    campaign's deviation from it with a low-order polynomial in
    log10(wavelength), and shift every campaign onto a reference - with three
    changes, of which the first two only work as a pair:

      1. The shared reference is science_radiometer_srf_node_fit, hence the
         required channel argument. (Historically this was a GPR fit, and the
         tables below quote that as the comparison; the GPR path has since been
         removed.)

      2. Points on a steep part of the reference curve are excluded from the
         offset polynomial fit (max_log_slope). They are still corrected, the
         same way sub-response_floor points already are.

      3. A per-campaign WAVELENGTH offset is applied to the wavelength
         assignment (wavelength_offset_frac), and a residual one is estimated
         from exactly the steep points that change 2 discards
         (estimate_wavelength_offset).

    Which campaign the others are moved onto is set by reference_campaigns.
    Default None means the most recent one alone, which is what the science
    radiometers want: May 2025 IS their radiometric scale. Passing a list makes
    the reference level the UNWEIGHTED MEAN of those campaigns' offset
    polynomials instead - the LST passes its final two, because it was the
    reference for a calibration transfer to the science radiometers in the
    interval between them, so they bookend the range over which its response
    must be right and neither end is privileged. One consequence worth knowing:
    with a single reference that campaign is corrected to itself and carries
    zero correction uncertainty, whereas with several none is, since each is
    moved to the mean of the set, so every campaign then gets a nonzero
    u_campaign_correction.

    Note the split of responsibility in change 3. This function APPLIES a
    wavelength offset it is given and ESTIMATES the residual, but does not
    apply what it estimates. That is deliberate: the laser wavelength belongs
    to the measurement, not to the channel, and this function only ever sees
    one channel. Four channels each applying their own fitted offset to the
    same measurement would be incoherent. The caller is responsible for
    combining the per-channel estimates into one offset per campaign - see
    science_radiometer_analyze_level_02, which inverse-variance weights
    them across channels and accumulates over its outer iterations.

    Why a wavelength offset, and why it is a measurement error rather than a
    real SRF change. The 4-5um SRF residuals are per-campaign and transform as
    a wavelength shift, not a gain: taking the eight measurements where SSW is
    falling and LW is rising, so the two channels respond with opposite sign,
    the independently fitted shifts correlate +0.923 with regression slope
    +0.975. A wavelength model explains 79-96% of the residual variance there
    against 0.0-1.5% for a gain model. Three different filters (SW, SSW and LW
    band edges) move together and then reverse sign between campaigns, which
    independent coating ageing would not do.

    The OSA207's own wavelength scale is not at fault, so this is not a
    correction for a broken instrument axis. Checked against the atmospheric
    absorption imprinted on the laser lines themselves - the CO2 nu3 band
    centre at 2349.14cm-1, a genuine P/R gap because the Q branch of a Sigma-
    Sigma transition is forbidden - the scale is right to ~1nm absolute at
    4.26um and stable campaign to campaign to <=0.5nm, over 20 variants of the
    estimator. Cross-correlating the H2O structure at 6.3 and 2.7um agrees:
    campaign differences of -0.02 to +0.08nm, where injected shifts recover
    with gain 1.00. The effect being corrected is 12-15nm, so 25-100x larger.
    The leading explanation is that the OSA samples a different part of a
    spatially chirped OPA beam than the radiometer does, and realignment
    changes which part - the OSA then measures its own light correctly while
    still disagreeing with what illuminated the radiometer.

    Referencing. Unlike the multiplicative correction, which goes onto the most
    recent campaign because May 2025 is the radiometric scale, the wavelength
    offsets are referenced to the UNWEIGHTED MEAN over campaigns. Writing
    delta_c = Delta + eps_c for a systematic sampling bias plus a per-campaign
    random part, only differences are observable, so one constraint must be
    assumed. Referencing one campaign leaves Delta + eps_ref, spread sigma;
    referencing the mean leaves Delta + eps_bar, spread sigma/sqrt(n). Same
    systematic, random part smaller by sqrt(3), and the relative structure
    between campaigns - the part that does the work - is identical either way.
    The mean is unweighted because under this model the randomness is in the
    beam sampling, drawn once per campaign, so campaign 2's 72 spectra do not
    make its draw better known than campaign 0's 253.

    What this does NOT remove is Delta itself, the fixed geometric part - if the
    OSA pickoff always takes the beam edge while the radiometer sees the centre,
    no averaging touches it. That belongs in the budget as a Type B term on the
    IR wavelength scale; sigma/sqrt(3) here is ~8nm with 2 dof.

    Parameterisation. All the steep-edge sensitivity inside the OSA range lies
    between 3.9 and 5.8um (SW 3.94-4.67, SSW 4.00-4.78, LW 4.70-5.05; Total has
    no steep edge above 1um at all), so the data constrain the offset's value
    but not its wavelength dependence. A constant offset in um is used, that
    being what the edge fit yields directly. The alternative of a constant
    fractional shift, pinned to agree at 4.26um, was checked and changes the
    SRF by at most 0.007 (LW, in band) against a 0.048 effect, and by <=0.004
    everywhere else - second order, and unconstrained either way.

    Why the slope guard. The offset is estimated from
    ratio_pct = 100*(r/ref - 1), a PERCENT deviation, and on a near-vertical
    band edge that normalization is treacherous: a sub-nanometre mismatch
    between the reference curve and the data becomes a tens-of-percent
    residual, and since those points carry a small u they enter the weighted
    polynomial fit with large weight and dominate it. Measured on this dataset,
    the median |deviation| for points with |dlnSRF/dlnwl| > 10 against those
    below it:

        channel   |slope|<=10   |slope|>10 (p95)     n above
        SW           0.13%        1.53% ( 9.0%)        48
        Total        0.07%        0.44% ( 1.3%)        21
        LW      0.56-0.74%        4.72% (29.6%)        26
        SSW     0.05-0.14%        1.80% (10.9%)        33

    This is what was breaking LW. It is not LW-specific physics but LW-specific
    geometry: response_floor is crossed at 4.74um, right on the steepest edge
    in the instrument (the SRF moves 0.449, 58% of its peak, across a single
    laser linewidth there), so LW's steep points sit at one end of a valid span
    only ~0.5 decades wide. SW and SSW have the same points but at the ends of
    a long flat plateau that outvotes them. LW's per-campaign deviation ran
    +19.7%/-9.1%/-8.3% in 4.7-5.0um while every other band was within +/-2.6%.

    Why the two changes are not separable. With the GPR reference the slope
    guard BACKFIRES on SSW - final offset spread 0.097% ungated, rising to
    0.177%/0.191%/0.210% at thresholds 10/5/3, with the total correction range
    blowing out to 0.9470-1.0029. Removing the edge points leaves only the
    plateau, and the GPR's over-smoothed plateau is not good enough to fit an
    offset on. With the node reference SSW holds 0.020-0.024% at every
    threshold. So the node fit's contribution here is making the plateau
    trustworthy enough for the guard to be safe.

    Final per-campaign offset spread after 8 iterations, current production
    config against this one:

        channel   GPR, no guard   node + slope<=10
        SW            0.1227%          0.0160%      7.7x
        Total         0.0143%          0.0074%      1.9x
        LW            0.4818%          0.1502%      3.2x
        SSW           0.0969%          0.0238%      4.1x

    and LW's total multiplicative correction tightens from 0.9362-1.0410 to
    0.9481-1.0000.

    Threshold 10 is where the deviation table breaks and is also LW's optimum
    under the node reference (0.150%, against 0.194% at 5 and 0.253% at 3 -
    tighter cuts start starving the fit of points).

    Note the order-reduction guard (max_extrap_ratio) is retained unchanged and
    still fires on LW. Disabling it was tested and is worse, not better: LW at
    order 2 degrades to a 1.69% spread with corrections reaching 0.74, and at
    order 3 to corrections of 1.45. The slope guard addresses a different
    failure and does not supersede it.

    Unlike science_radiometer_srf_campaign_correct, which computes the
    response_floor mask once from the raw r_srf, both masks here are recomputed
    each iteration from the current corrected values and the current reference,
    so the guard tracks the reference as it converges. Returns iteration
    diagnostics in `history` so that convergence can be checked - see the
    divergence warning below.
    """

    if channel not in SRF_NODE_GRIDS:
        raise ValueError(f"No node grid for channel {channel!r}; "
                         f"have {sorted(SRF_NODE_GRIDS)}.")

    campaigns = np.unique(measurement_campaign)

    # Which campaign(s) define the radiometric scale everything else is moved
    # onto. Default is the most recent one alone, which is right for the science
    # radiometers: May 2025 is the scale. The LST wants both of its final two
    # campaigns weighted equally instead, because the LST was used as the
    # reference for a calibration transfer to the science radiometers BETWEEN
    # those two campaigns, so they bookend the interval over which the LST
    # response needs to be known best. Passing several here makes the reference
    # level their unweighted mean rather than any one campaign's.
    if reference_campaigns is None:
        reference = [campaigns.max()]
    else:
        reference = [c for c in campaigns if c in set(reference_campaigns)]
        missing = sorted(set(reference_campaigns) - set(campaigns.tolist()))
        if missing:
            raise ValueError(f"reference_campaigns {missing} are not present in "
                             f"measurement_campaign (have {campaigns.tolist()}).")
        if not reference:
            raise ValueError("reference_campaigns selected no campaigns.")
    target_campaign = reference[-1] if len(reference) == 1 else tuple(reference)

    r_corrected = r_srf.copy()

    # Apply the wavelength offsets we were given, once, up front, as
    # wl -> wl*(1 + f). They are held fixed for the whole run - the reference
    # fit, the slope guard and the gain polynomial all then see a single
    # consistent wavelength assignment.
    wl_working = np.asarray(wavelength_um, dtype=float).copy()
    wl_offset = {int(s): 0.0 for s in campaigns}

    # Which measurements the wavelength offset is meaningful for. It describes
    # the OSA207 path, and the CCS200-measured wavelengths below 1um were shown
    # to be campaign-stable to <=1nm, so applying an OSA-derived offset to them
    # would be wrong. Everything else defaults to in-scope.
    wl_scope = (np.ones_like(wl_working, dtype=bool)
                if wavelength_offset_applies is None
                else np.asarray(wavelength_offset_applies, dtype=bool))

    if wavelength_offset_frac:
        for s in campaigns:
            f = float(wavelength_offset_frac.get(int(s), 0.0))
            wl_offset[int(s)] = f
            k = (measurement_campaign == s) & wl_scope
            wl_working[k] = wl_working[k]*(1.0 + f)

    history = []

    for it in range(n_iter):

        x = np.log10(wl_working)

        node_fit = science_radiometer_srf_node_fit(wavelength_um=wl_working,
                                                   r_srf=r_corrected,
                                                   u_srf_random=u_srf_random,
                                                   channel=channel)
        fit_x = np.log10(node_fit["wavelength_um"].values)
        fit_y = node_fit["srf_pred"].values
        ref = np.interp(x, fit_x, fit_y)

        # Points too close to the response floor give wildly noisy percent
        # ratios (dividing by a near-zero reference).
        above_floor = r_corrected > response_floor
        valid = above_floor.copy()

        # Points on a steep part of the reference, where the percent deviation
        # reports edge placement rather than calibration level. The slope is
        # the logarithmic derivative dlnSRF/dlnwl, which is the quantity that
        # actually converts a fractional wavelength error into a fractional
        # response error, and is dimensionless so one threshold works for every
        # channel. Taken from the reference curve rather than from the data so
        # it is a smooth function of wavelength rather than a noisy per-point
        # estimate.
        if max_log_slope is not None:
            d_ref = np.interp(x, fit_x, np.gradient(fit_y, fit_x))
            log_slope = d_ref/np.maximum(ref, 1e-12)
            steep = np.abs(log_slope) > max_log_slope
            valid = valid & ~steep
        else:
            steep = np.zeros_like(valid)

        # What the slope guard actually costs: points that cleared the response
        # floor and would have been used, but are on too steep a part of the
        # curve. Counting every steep point instead would be dominated by the
        # out-of-band region, where d_ref/ref blows up simply because ref is
        # near zero and which response_floor already excluded.
        n_steep_excluded = int((above_floor & steep).sum())

        ratio_pct = 100.0 * (r_corrected / ref - 1.0)
        sigma_pct = 100.0 * u_srf_random / ref

        campaign_coeffs = {}
        campaign_covs = {}
        campaign_x_range = {}
        for s in campaigns:
            mask = valid & (measurement_campaign == s)
            n = int(mask.sum())
            if n == 0:
                campaign_coeffs[s] = np.array([0.0])
                campaign_covs[s] = np.array([[0.0]])
                campaign_x_range[s] = (-np.inf, np.inf)
                continue
            order = poly_order if n >= min_n_for_poly else 0
            order = min(order, n - 1)

            x_fit_min, x_fit_max = x[mask].min(), x[mask].max()
            fit_span = x_fit_max - x_fit_min
            x_applied = x[measurement_campaign == s]
            reach = max(x_fit_min - x_applied.min(), x_applied.max() - x_fit_max, 0.0)
            if order > 1 and (fit_span <= 0 or reach/fit_span > max_extrap_ratio):
                order = 1

            w = 1.0 / np.clip(sigma_pct[mask], 1e-6, None)
            coeffs, cov = np.polyfit(x[mask], ratio_pct[mask], deg=order, w=w, cov='unscaled')
            campaign_coeffs[s] = coeffs
            campaign_covs[s] = cov
            campaign_x_range[s] = (x_fit_min, x_fit_max)

        r_new = r_corrected.copy()
        u_campaign_correction = np.zeros_like(r_corrected)
        n_ref = len(reference)

        for s in campaigns:
            mask = measurement_campaign == s
            lo, hi = campaign_x_range[s]

            # Hold each polynomial at its endpoint value outside its fitted
            # range. With the slope guard on, that range now also excludes the
            # steep edge, so the clamp is doing more work than before: the edge
            # points get their campaign's plateau offset rather than an
            # extrapolation toward the edge. That is the intended behaviour -
            # the offset is a gain, and the plateau is where it is measurable.
            x_s = np.clip(x[mask], lo, hi)

            # The reference level is the UNWEIGHTED MEAN of the reference
            # campaigns' offset polynomials, each clamped to its own fitted
            # range. With one reference this is the old behaviour exactly. The
            # campaigns are fitted on disjoint measurements, so their
            # coefficient covariances are independent and the variance of the
            # mean is the sum of variances over n_ref^2.
            target_offset_pct = 0.0
            var_target = 0.0
            for t in reference:
                t_lo, t_hi = campaign_x_range[t]
                x_t = np.clip(x[mask], t_lo, t_hi)
                target_offset_pct = target_offset_pct + np.polyval(campaign_coeffs[t], x_t)
                var_target = var_target + _polyval_variance(campaign_covs[t], x_t)
            target_offset_pct = target_offset_pct/n_ref
            var_target = var_target/n_ref**2

            session_offset_pct = np.polyval(campaign_coeffs[s], x_s)
            correction_pct = target_offset_pct - session_offset_pct
            r_new[mask] = r_corrected[mask] * (1.0 + correction_pct / 100.0)

            # A sole reference campaign is corrected to itself, so its
            # correction and that correction's uncertainty are identically zero.
            # With several references that is no longer true - each one is moved
            # to the mean of the set - so every campaign gets an uncertainty.
            if n_ref == 1 and s == reference[0]:
                continue

            var_campaign = _polyval_variance(campaign_covs[s], x_s)
            correction_pct_sd = np.sqrt(var_campaign + var_target)

            u_campaign_correction[mask] = r_new[mask] * correction_pct_sd / 100.0

        r_corrected = r_new

        # Residual per-campaign offset on the points the fit actually used, so
        # convergence can be checked. n_valid is recorded too because the
        # slope guard's mask moves with the reference: if it churns from one
        # iteration to the next the fit is chasing its own tail.
        offsets = {}
        for s in campaigns:
            m = valid & (measurement_campaign == s)
            if m.sum() == 0:
                continue
            offsets[int(s)] = float(np.median(100.0*(r_corrected[m]/ref[m] - 1.0)))

        history.append(SimpleNamespace(
            iteration=it,
            offset_spread=(max(offsets.values()) - min(offsets.values())) if offsets else np.nan,
            offsets=offsets,
            n_valid={int(s): int((valid & (measurement_campaign == s)).sum()) for s in campaigns},
            n_steep_excluded=n_steep_excluded,
            n_above_floor=int(above_floor.sum()),
            smoothing=node_fit.attrs["smoothing"]))

    # The condition for keeping the recomputed-each-iteration slope guard was
    # that it not diverge, so check it rather than assume it. Two things are
    # worth testing, and neither is "did the spread ever get smaller than it
    # ended up".
    #
    # An earlier version of this compared the best spread in the second half of
    # the iterations against the best in the first half, and it produced a
    # false alarm: every channel undershoots hard at iteration 1, because that
    # iteration's reference curve is still fitted on wholly uncorrected data,
    # and then climbs back to a plateau. Total runs
    # 0.110 0.0034 0.0091 0.0093 0.0111 0.0116 0.0118 0.0118 - a transient
    # minimum three times below a perfectly settled final value. That test was
    # measuring the depth of the undershoot, not convergence.
    #
    #   1. Mask stability, which is the actual worry with a guard recomputed
    #      against a moving reference. Measured directly: does the set of
    #      points the offset fit uses change from one iteration to the next?
    #      (In practice it does not - n_valid is constant across all eight
    #      iterations on all four channels.)
    #   2. Settling, measured as the size of the last step relative to the
    #      value it landed on. A channel still trending steadily downward is
    #      converging and is not flagged; one still moving by a large fraction
    #      of its own value has not settled.
    if len(history) >= 3:

        n_prev = sum(history[-2].n_valid.values())
        n_last = sum(history[-1].n_valid.values())
        if n_prev != n_last:
            warnings.warn(
                f"{channel}: the slope guard's point set is still moving at the "
                f"last iteration ({n_prev} -> {n_last} points). The guard and "
                f"the reference fit may be chasing each other; check "
                f"history[*].n_valid.",
                RuntimeWarning)

        last, prev = history[-1].offset_spread, history[-2].offset_spread
        if np.isfinite(last) and np.isfinite(prev) and last > 0:
            if abs(last - prev)/last > 0.25:
                warnings.warn(
                    f"{channel}: campaign-correction offset spread has not "
                    f"settled - it moved {abs(last - prev)/last:.0%} on the "
                    f"final iteration ({prev:.4g}% -> {last:.4g}%). Consider "
                    f"more iterations.",
                    RuntimeWarning)

    # ----------------------------------------------------------------------
    # Residual per-campaign wavelength offset, estimated but NOT applied.
    #
    # FRACTIONAL, not absolute: the offset is f in wl_true = wl*(1 + f), so on a
    # steep point r - ref(wl) ~= f * wl * dref/dwl, and f is a weighted
    # least-squares slope with design g = wl*dref/dwl and weights 1/u^2. ref is
    # interpolated against log10(wl), hence dref/dwl = (dref/dlog10wl)/(wl*ln10)
    # and the wl cancels: g = (dref/dlog10wl)/ln10. Sign convention: f is
    # positive when the true wavelength EXCEEDS the assigned one.
    #
    # Why fractional rather than a constant shift in um. The offset is measured
    # from steep band edges, which in the OSA range lie only between 3.9 and
    # 5.8um, so its wavelength dependence is not constrained by the edges at all
    # - but the SW etalon fringe constrains it independently over 0.4-3.0um.
    # The fringe is phase-locked (maximum at nu = 0, no free phase), and at
    # 1.15um one period is 606cm-1, only about 8nm in wavelength, so the phase
    # is a sensitive wavelength ruler.
    #
    # It is a ruler of unequal sensitivity to the two candidate models, and the
    # difference is exact algebra rather than a detail. A constant shift d gives
    # a wavenumber error d*nu^2/1e4, quadratic in nu, which no etalon parameter
    # can absorb. A fractional shift f gives f*nu, linear in nu, which is
    # identically a change of 2nd to 2nd/(1 + f) - so a fractional error is very
    # nearly invisible, and would be entirely invisible were the node positions
    # and the amplitude polynomial's argument not carried along with it.
    # Measured by injection, at fixed 2nd profiled over a fine grid: a 14nm
    # constant costs dchi2/chi2_red of 45 (+14nm) and 203 (-14nm), whereas
    # +/-2.8e-3 fractional costs -8 and +18. So the fringe emphatically excludes
    # a constant offset of the size the edges report, and is only weakly
    # sensitive to a fractional one.
    #
    # What it can still do is measure per-campaign DIFFERENCES in f, because
    # 2nd is one physical etalon common to every campaign and so cannot absorb
    # them. Fitting f per campaign over 0.4-3.0um with 2nd fixed and common:
    #
    #                   etalon, 0.4-3.0um     edges, 4-5um
    #     campaign 0     +1.64 +/- 1.25e-3      +2.78e-3     0.9 sigma
    #     campaign 1     -0.68 +/- 1.01e-3      -1.11e-3     0.4 sigma
    #     campaign 2     +1.04 +/- 1.38e-3      -1.68e-3     2.0 sigma
    #
    # chi2 = 5.0 on three campaigns, so the two anchors agree. The constant
    # model, by contrast, predicts the etalon should see 4.6/1.5 = 3.07 times
    # the edge value, i.e. +8.53/-3.41/-5.16e-3, and is excluded at chi2 = 57.6.
    # Campaign 2 is the persistent outlier at every exponent tried and has only
    # 9 OSA measurements in the fringe band, so it is the weakest anchor.
    #
    # The practical consequence is not just that the old constant model
    # over-stated the uncertainty below 3um - it was INJECTING error. Fitted in
    # the constant basis, the campaign spread over 0.4-3.0um is 5.8nm in the raw
    # wavelengths; the constant correction left it at 20.7nm, while the
    # fractional correction leaves it at 5.8nm, i.e. does no harm.
    #
    # Note the etalon says nothing about the COMMON part of f, only differences:
    # with 2nd free, a common fractional error is absorbed exactly. The common
    # part is what IR_WAVELENGTH_SCALE_U_FRAC covers.
    #
    # Note what fractional does NOT mean. A 2.7e-3 error in the OSA207's own
    # wavelength axis would be 11.5nm at 4.26um, which the CO2 nu3 band-centre
    # fiducial excludes at <=0.5nm campaign to campaign. This is an empirical
    # parameterisation of the difference between the part of the beam the OSA
    # samples and the part that illuminates the radiometer, which that fiducial
    # does not probe. Above the 4.6um anchor it is extrapolation: there is no
    # steep edge past 5.8um and no second ruler, so the fractional form is
    # carried on as the conservative choice rather than a measured one.
    #
    # The points used are exactly those the slope guard drops from the gain
    # fit, which is what makes the two estimates orthogonal rather than
    # competing - a plateau carries the gain and cannot see a wavelength shift,
    # an edge carries the wavelength and is useless for gain.
    #
    # The reference fit is redone here rather than reusing the last iteration's,
    # which was built before that iteration's gain correction was applied.
    wl_residual = {int(s): 0.0 for s in campaigns}
    wl_residual_sd = {int(s): np.inf for s in campaigns}
    n_wl_fit = {int(s): 0 for s in campaigns}

    if estimate_wavelength_offset and max_log_slope is not None:

        final_fit = science_radiometer_srf_node_fit(wavelength_um=wl_working,
                                                    r_srf=r_corrected,
                                                    u_srf_random=u_srf_random,
                                                    channel=channel)
        f_x = np.log10(final_fit["wavelength_um"].values)
        f_y = final_fit["srf_pred"].values
        ref_f = np.interp(np.log10(wl_working), f_x, f_y)
        d_ref_f = np.interp(np.log10(wl_working), f_x, np.gradient(f_y, f_x))

        # g = wl*dref/dwl = (dref/dlog10wl)/ln10 - the wl cancels, which is
        # what makes the fractional design simpler than the absolute one.
        g_all = d_ref_f/math.log(10.0)
        edge = (r_corrected > response_floor) & wl_scope \
               & (np.abs(d_ref_f/np.maximum(ref_f, 1e-12)) > max_log_slope)

        fitted = []
        for s in campaigns:
            m = edge & (measurement_campaign == s)
            n_wl_fit[int(s)] = int(m.sum())
            if m.sum() < min_n_for_wavelength:
                continue
            g = g_all[m]
            w = 1.0/np.clip(u_srf_random[m], 1e-12, None)**2
            denom = float((w*g*g).sum())
            if denom <= 0:
                continue
            d_hat = float((w*g*(r_corrected[m] - ref_f[m])).sum()/denom)
            wl_residual[int(s)] = d_hat

            # Inflate the formal standard error by the fit's own reduced
            # chi-square. The stated u is known to be understated on this
            # dataset - chi2_red runs 12 to 476 by channel - so 1/sqrt(denom)
            # alone is meaningless, and would hand the cross-channel
            # combination to whichever channel understates u the most. Scaling
            # by the scatter the edge points actually show makes the four
            # channels' sds comparable, which is all the combination needs.
            chi2 = float((w*(r_corrected[m] - ref_f[m] - d_hat*g)**2).sum())
            chi2_red = chi2/max(int(m.sum()) - 1, 1)
            wl_residual_sd[int(s)] = math.sqrt(max(chi2_red, 1.0)/denom)
            fitted.append(int(s))

        # Reference to the UNWEIGHTED mean over the campaigns actually fitted -
        # see the docstring. Unweighted because the randomness is in the beam
        # sampling, drawn once per campaign, not in the measurements.
        if fitted:
            n_f = len(fitted)
            mean_offset = sum(wl_residual[s] for s in fitted)/n_f
            base = {s: wl_residual_sd[s]**2 for s in fitted}
            for s in fitted:
                wl_residual[s] -= mean_offset
                # var(d_c - d_bar), d_bar built from the same estimates: the
                # campaign's own term scaled by (1 - 1/n)^2, each other by 1/n^2.
                wl_residual_sd[s] = math.sqrt(base[s]*(1.0 - 1.0/n_f)**2
                                              + sum(base[k] for k in fitted
                                                    if k != s)/n_f**2)
            # Partial coverage is expected here, not an anomaly, so nothing is
            # raised: LW has a single steep point in Feb 2025 against the five
            # this needs, and Total has none in any campaign - its band edges
            # are CCS200-measured below 1um and the offset applies only to
            # wavelength_source == 2. A campaign a channel cannot estimate is
            # returned as 0 with an infinite sd, so the caller's inverse-variance
            # combination ignores it rather than pulling the shared offset toward
            # zero. The per-channel table in science_radiometer_analyze_level_02
            # reports exactly this, showing such campaigns as "--" alongside the
            # steep-point counts, so a warning here would fire on every clean run
            # and say nothing that table does not.

    return SimpleNamespace(
        r_corrected=r_corrected,
        u_campaign_correction=u_campaign_correction,
        node_fit=node_fit,
        campaign_coeffs=campaign_coeffs,
        target_campaign=target_campaign,
        reference_campaigns=list(reference),
        # The wavelength assignment this run actually used, i.e. the input with
        # wavelength_offset_frac applied. Callers writing an SRF must use this,
        # not the raw input wavelengths, or the offset is silently discarded.
        wavelength_um_corrected=wl_working,
        campaign_wavelength_offset_frac=wl_offset,
        # Estimated but unapplied - the caller combines these across channels.
        # Fractional: wl_true = wl*(1 + f).
        campaign_wavelength_residual_frac=wl_residual,
        campaign_wavelength_residual_sd_frac=wl_residual_sd,
        n_wavelength_fit=n_wl_fit,
        history=history
    )

def science_radiometer_analyze_level_01():
    """
    Analyzes every ERF measurement taken with the science radiometers. For this 
    analysis we perform the global fit on the ERF dataset which assumes a global
    radiance field that is common to all radiometers.

    Summary output:
    Libera_ERF_analysis_summary_level_01.csv
    """

    # Analyze all the scirad data
    batch_file = 'libera_erf_sci_radiometers_cal_log.csv'
    paths = load_config()
    f = tidy_up_header(pd.read_csv(paths.erf_misc_file_dir / batch_file))

    # Throw out bad files
    f = f.loc[(f["good_file"] == 1)]

    # Keep only the files with these FPE/drivers:
    f = f.loc[f["driver"].isin(['emfpe', 'emfpeb', 'fmfpe'])]

    # Assign an alignment-session index. It increments whenever int_sphere or
    # driver changes from the previous (filtered, chronologically-ordered) row,
    # or whenever more than a week passes between consecutive files - each of
    # these means the test chamber was reconfigured, so the source/detector had
    # to be realigned. Files sharing a session number share one unknown
    # alignment offset, rather than each file drawing its own independent one.
    log_time = pd.to_datetime(f["log_file"].str.extract(r'(\d{8}_\d{4})')[0], format='%Y%m%d_%H%M')

    # True on the first row of each new session: a changed int_sphere/driver
    # (NaN != value is True, so the very first row always starts a session),
    # or a gap since the previous file of more than 7 days.
    new_session = ((f["int_sphere"] != f["int_sphere"].shift())
                    | (f["driver"] != f["driver"].shift())
                    | (log_time.diff() > pd.Timedelta(days=7)))

    # Define a new measurement campaign if there was a time
    # gap of one month. We'll use these to identify 
    # different time periods of the SRF data
    new_campaign = (log_time.diff() > pd.Timedelta(days=30))

    # Cumulative sum of the boolean flags turns "is this a new session" into
    # a running session index: 1 for all rows up to the first change, 2 from
    # there to the next change, and so on.
    f["alignment_session"] = new_session.cumsum()

    # Same with 
    f["measurement_campaign"] = new_campaign.cumsum()

    print("")
    print("Analysis summary:")

    summary_data = []
    for row in f.itertuples():
        summary_data = science_radiometer_cal_single(row=row,summary_data=summary_data)
        # Print out some details

        print(f"{row.Index:4}, {row.log_file}, "
            f"OPA Wl = {summary_data[-1]['wavelength_um_opa']:6.3f} um, "
            f"Wl = {summary_data[-1]['wavelength_um']:6.3f} um, "
            f"FWHM = {summary_data[-1]['wavelength_um_fwhm']:6.4f} um")
        
    # Change to a dataframe
    summary_data = pd.DataFrame(summary_data)

    # Throw data with no spectrometer data or a bad fit
    summary_data = summary_data.loc[(summary_data["wavelength_source"] != 10)]
    summary_data = summary_data.loc[(summary_data["wavelength_gauss_amp"] > 0)]

    # This is the lower limit threshold:
    wl_lmin = [0.25,  1.0,  2.5, 10.0, 16.0]
    lmin =    [4.00, 95.0, 13.0, 10.0,  2.0] 

    # Interpolate the lower radiance limit to the data wavelength grid
    lmin_interp = np.interp(summary_data["wavelength_um"], wl_lmin, lmin)

    # Threshold the data
    summary_data = summary_data.loc[summary_data["gf_l0"] > lmin_interp]

    # Save the data summary
    paths = load_config()
    filename = "Libera_ERF_analysis_summary_level_01.csv"
    summary_data.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    print("Done!")

def lst_analyze_level_01():
    """
    Analyzes every ERF measurement taken with the the LST. For this 
    analysis we perform the global fit on the ERF dataset which assumes a global
    radiance field that is common to all radiometers.

    Summary output:
    Libera_ERF_LST_analysis_summary_level_01.csv
    """

    # Analyze all the scirad data
    batch_file = 'libera_erf_lst_cal_log.csv'
    paths = load_config()
    f = tidy_up_header(pd.read_csv(paths.erf_misc_file_dir / batch_file))

    # Throw out bad files
    f = f.loc[(f["good_file"] == 1)]

    # Assign an alignment-session index. It increments whenever int_sphere or
    # driver changes from the previous (filtered, chronologically-ordered) row,
    # or whenever more than a week passes between consecutive files - each of
    # these means the test chamber was reconfigured, so the source/detector had
    # to be realigned. Files sharing a session number share one unknown
    # alignment offset, rather than each file drawing its own independent one.
    log_time = pd.to_datetime(f["log_file"].str.extract(r'(\d{8}_\d{4})')[0], format='%Y%m%d_%H%M')

    # True on the first row of each new session: a changed int_sphere/driver
    # (NaN != value is True, so the very first row always starts a session),
    # or a gap since the previous file of more than 7 days.
    new_session = ((f["int_sphere"] != f["int_sphere"].shift())
                    | (f["driver"] != f["driver"].shift())
                    | (log_time.diff() > pd.Timedelta(days=7)))

    # Define a new measurement campaign if there was a time
    # gap of one month. We'll use these to identify 
    # different time periods of the SRF data
    new_campaign = (log_time.diff() > pd.Timedelta(days=30))

    # Cumulative sum of the boolean flags turns "is this a new session" into
    # a running session index: 1 for all rows up to the first change, 2 from
    # there to the next change, and so on.
    f["alignment_session"] = new_session.cumsum()

    # Same with 
    f["measurement_campaign"] = new_campaign.cumsum()

    print("")
    print("Analysis summary:")

    summary_data = []
    for row in f.itertuples():
        summary_data = lst_cal_single(row=row,summary_data=summary_data)
        # Print out some details

        print(f"{row.Index:4}, {row.log_file}, "
            f"OPA Wl = {summary_data[-1]['wavelength_um_opa']:6.3f} um, "
            f"Wl = {summary_data[-1]['wavelength_um']:6.3f} um, "
            f"FWHM = {summary_data[-1]['wavelength_um_fwhm']:6.4f} um")
        
    # Change to a dataframe
    summary_data = pd.DataFrame(summary_data)

    # Throw data with no spectrometer data or a bad fit
    summary_data = summary_data.loc[(summary_data["wavelength_source"] != 10)]
    summary_data = summary_data.loc[(summary_data["wavelength_gauss_amp"] > 0)]

    # This is the lower limit threshold:
    wl_lmin = [0.25,  1.0,  2.5, 10.0, 16.0]
    lmin =    [4.00, 95.0, 13.0, 10.0,  2.0] 

    # Interpolate the lower radiance limit to the data wavelength grid
    lmin_interp = np.interp(summary_data["wavelength_um"], wl_lmin, lmin)

    # Threshold the data
    summary_data = summary_data.loc[summary_data["gf_l0"] > lmin_interp]

    # Save the data summary
    paths = load_config()
    filename = "Libera_ERF_LST_analysis_summary_level_01.csv"
    summary_data.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    print("Done!")

def science_radiometer_smooth_radiance_gradient(*, wavelength_um, value, value_sd, alignment_session,
                                                 min_n_for_gpr=10):
    """
    Smooths a radiance-field gradient term (e.g. gf_gp, the pitch gradient
    from the global fit) as a function of wavelength, fit independently
    within each alignment_session rather than across the whole dataset -
    the source/detector geometry resets at each realignment, but should vary
    smoothly with wavelength within one session.

    This exists because the raw per-measurement gradient and its fit
    uncertainty blow up wherever the science radiometers have very little
    signal to sense position/pointing with (e.g. ~300nm, where SW/TO SRF is
    near zero) - a session-local smooth fit borrows strength from the
    well-constrained nearby wavelengths within that session instead of
    taking that noise spike at face value.

    Sessions with fewer than min_n_for_gpr points fall back to a single
    sigma-weighted mean for that session, since a GPR fit to a handful of
    points would have essentially unconstrained hyperparameters.

    Returns the smoothed value and its 1-sigma uncertainty, evaluated at
    each input point's own wavelength (same order/length as the inputs).
    """

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.exceptions import ConvergenceWarning

    smoothed_value = np.full(len(value), np.nan)
    smoothed_sd = np.full(len(value), np.nan)

    for session in np.unique(alignment_session):
        mask = alignment_session == session
        n = int(mask.sum())

        wl = wavelength_um[mask]
        v = value[mask]
        v_sd = value_sd[mask]

        if n < min_n_for_gpr:
            wmean = weighted_mean_and_stddev(x=v, sd=v_sd)
            smoothed_value[mask] = wmean.x_mn
            smoothed_sd[mask] = wmean.x_sd
            continue

        x = np.log10(wl).reshape(-1, 1)

        # Note, I'm setting the lower length-scale bound to 0.5, this effectively limits how much
        # the gradients can vary over wavelength. There is no physical mechanism for them to 
        # show sharp variations with wavelength since the source is an integrating sphere and
        # all the optics are reflective, and so this lower bound of 0.5 prevent this.
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=0.8, length_scale_bounds=(0.5, 3.0))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=v_sd**2,            # per-point noise VARIANCE (not sd) — this is the key line
            n_restarts_optimizer=10,  # avoids getting stuck in a bad local optimum
            normalize_y=True          # centers y around 0 internally, generally improves fit stability
        )
        # The length_scale bound is intentional (keeps the fit from chasing
        # unphysical structure in a session with sparse coverage), so the
        # optimizer landing on it is expected - suppress the resulting
        # ConvergenceWarning instead of widening the bound.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gpr.fit(x, v)

        pred, pred_sd = gpr.predict(x, return_std=True)
        smoothed_value[mask] = pred
        smoothed_sd[mask] = pred_sd

    return SimpleNamespace(value=smoothed_value, value_sd=smoothed_sd)

# ---------------------------------------------------------------------------
# SSW filter-edge temperature correction
#
# The SSW passband's short-wavelength edge near 0.74um is set by the internal
# transmission of an uncoated AMTIR-1 (Ge33As12Se55) window - the SSW filter is
# one uncoated AMTIR-1 window plus one uncoated fused-silica window, and the SW
# filter is two uncoated fused-silica windows, so the SSW/SW response ratio
# isolates the AMTIR-1 transmission (fused silica is featureless from 0.4 to
# ~2um; above ~4um its OH/multiphonon absorption enters and the ratio stops
# being a clean isolation, but that is far from this edge). AMTIR-1 is a
# chalcogenide glass whose absorption edge is an Urbach tail, and that edge
# moves with temperature: measured here at 0.435 +/- 0.017nm/degC.
#
# This matters because that edge sits in the middle of a large spectral
# radiance signal, unlike the other band edges, so the ERF measurements taken
# at 29.1degC read systematically high there - up to 73% of the local SRF. That
# is why the level_02 analysis keeps only 32-34.5degC bench data. The residual
# spread inside that retained window still reaches 4.5% of the SRF at 0.72um,
# which this correction removes.
#
# The correction is the MEASURED profile, not a rigid shift of the edge. A
# rigid shift (constant dlambda through dlnT/dlambda) fits the lower 80% of the
# edge - the implied shift is 2.193 +/- 0.085nm across the six pairs from 0.68
# to 0.78um - but fails at 0.803um, where it predicts 0.808% against a measured
# 0.292% +/- 0.048%, a 10.8 sigma over-prediction. That is the expected behaviour of an Urbach edge,
# which BROADENS with temperature rather than translating rigidly, so the
# fractional change collapses faster than any translation predicts as the
# transmission saturates. Using the measured profile also makes the correction
# non-iterative (it needs no dSRF/dlambda from the SRF fit) and needs no
# arbitrary window, since the profile is zero outside its own support.
#
# SSW_EDGE_DRESP_DT_NODES holds -dSRF/dT in SRF units per degC, i.e. the amount
# to ADD per degC of (T - T_ref), measured from 7 matched hot/cold laser-line
# pairs in measurement campaign 0 / alignment session 2 (29.13 vs 34.17degC,
# dT = -5.04degC), each cold point differenced against the same laser line
# measured hot and slid to the cold wavelength with the local dlnSRF/dlambda
# (the wavelength offsets are <=0.23nm and this slide changes the result by
# <2%). Per-point significance runs from 6 sigma at the top of the edge to 700
# sigma at 0.72um.
#
# The support is bounded on both sides by measurement, not by choice:
#
#   low side   Ten pairs from 0.46 to 0.64um give -dSRF/dT = 0 to within 1-2
#              sigma, |value| <= 5e-6 SRF/degC. The response has dropped
#              cleanly to zero, so there is nothing left to shift.
#   high side  The fractional effect decays by a factor of 250 from 0.68 to
#              0.80um (72.9% to 0.292% per 5.04degC). The log-linear decay of
#              the top segment of -dSRF/dT reaches that same 5e-6 noise floor
#              at 0.859um. Independently, the SSW/SW ratio - i.e. the
#              AMTIR-1 transmission itself - is flat to <=0.5%/um from 0.9 to
#              3.5um against 249%/um at the edge peak, so above ~0.9um there is
#              no edge structure left for a shift to change. The two agree.
#
# Between nodes the profile is interpolated log-linearly in wavelength (it is
# close to exponential across the edge), and extrapolated at the same
# end-segment rates out to SSW_EDGE_SUPPORT_UM, beyond which it is zero.
#
# Known limitations, all pointing at the same follow-up measurement:
#
#   - Linearity in T is assumed, from a single dT of 5.04degC. Inside the
#     retained 32-34.5degC window this is at most a 1.7degC interpolation of a
#     5degC measurement, so the exposure is small, but it is untested.
#   - Only one measured node lies above 0.80um, so the high-side decay rate
#     rests on the 0.782/0.803um segment.
#   - The 32degC cut must stay in place. The coefficient is derived from the
#     29degC data, so re-admitting those data corrected by it would be circular.
SSW_EDGE_T_REF_C = 34.1716
SSW_EDGE_NODES_UM = np.array([0.681538, 0.700900, 0.720750, 0.741860,
                              0.761620, 0.781580, 0.802650])
SSW_EDGE_DRESP_DT_NODES = np.array([9.813e-05, 1.484e-03, 3.308e-03, 2.941e-03,
                                    2.428e-03, 1.312e-03, 2.860e-04])
SSW_EDGE_NOISE_FLOOR = 5.0e-6
SSW_EDGE_SUPPORT_UM = (0.6603, 0.8586)

# Fractional Type-B uncertainty on the applied correction. 20% for the
# untested linearity in T, 15% for the interpolated/extrapolated profile
# shape, added in quadrature; the statistical uncertainty on the nodes
# themselves is smaller than either except at the 0.803um node (16%).
SSW_EDGE_U_FRAC = 0.25


def ssw_edge_temperature_correction(*, wavelength_um, bench_temp_c,
                                    t_ref_c=SSW_EDGE_T_REF_C):
    """
    Corrects the SSW response for the temperature-driven motion of the AMTIR-1
    filter edge near 0.74um, referencing every measurement to t_ref_c. See the
    block comment above for the physics, the measurement and the bounds on the
    support.

    The default t_ref_c is the mean bench temperature of the hot cluster
    (34.1716degC), where the great majority of the retained data sit, so the
    correction is a small perturbation for most points and the profile is
    never extrapolated in temperature beyond its own anchor. Re-referencing to
    the flight operating temperature is a change to this one constant.

    The correction itself uses each measurement's own temperature, but its
    uncertainty is reported as a smooth envelope: the profile uncertainty
    evaluated at the LARGEST |T - t_ref_c| in the inputs, at every point.
    Two reasons. The correction's error is common to every point (it is one
    profile error, not per-measurement noise), so it does not average down
    the way a per-point term would. And the per-measurement value swings four
    orders of magnitude between adjacent wavelengths purely because adjacent
    laser lines were measured in different campaigns at different bench
    temperatures - interpolating that onto the level_03 wavelength grid, which
    is what the "Wavelength" uncertainty class does, notches the term to
    almost zero at wavelengths where a 34.17degC point happens to sit next to
    a 33.29degC one, understating it where the SRF there is in fact built
    partly from corrected measurements. The envelope is the conservative
    reading and is immune to which campaign happened to measure where.

    Returns the correction to be ADDED to the SSW SRF and its 1-sigma
    uncertainty, both in SRF units, same order/length as the inputs.
    """

    wl = np.asarray(wavelength_um, dtype=float)
    dt = np.asarray(bench_temp_c, dtype=float) - float(t_ref_c)

    # Log-linear interpolation across the edge, extended to the support edges
    # at the noise floor so np.interp reproduces the end-segment decay rates.
    x = np.concatenate(([SSW_EDGE_SUPPORT_UM[0]], SSW_EDGE_NODES_UM,
                        [SSW_EDGE_SUPPORT_UM[1]]))
    y = np.log(np.concatenate(([SSW_EDGE_NOISE_FLOOR], SSW_EDGE_DRESP_DT_NODES,
                               [SSW_EDGE_NOISE_FLOOR])))
    dresp_dt = np.exp(np.interp(wl, x, y))

    # Zero outside the measured support, where the response has either dropped
    # to zero (low side) or the edge has saturated (high side).
    in_support = (wl >= SSW_EDGE_SUPPORT_UM[0]) & (wl <= SSW_EDGE_SUPPORT_UM[1])
    dresp_dt = np.where(in_support, dresp_dt, 0.0)

    correction = dresp_dt*dt

    # See the docstring - the uncertainty is the envelope at the largest
    # temperature excursion present, not each point's own correction.
    dt_envelope = np.max(np.abs(dt)) if dt.size else 0.0
    u_correction = SSW_EDGE_U_FRAC*np.abs(dresp_dt)*dt_envelope

    return SimpleNamespace(correction=correction,
                           u_correction=u_correction,
                           dresp_dt=dresp_dt,
                           dt_envelope_c=dt_envelope)


def science_radiometer_analyze_level_02(*, fit_wavelength_offset=True):
    """
    Prototype alternative to science_radiometer_analyze_level_02 that replaces
    the Gaussian process regression with the penalized node spline. Two places
    change:

      the SRF fit itself     science_radiometer_srf_node_fit
      the campaign correct   science_radiometer_srf_campaign_correct,
                             which uses the node fit for its shared reference
                             curve AND excludes steep-edge points from the
                             per-campaign offset polynomials

    Everything else - the out of field correction, the non-linearity terms, the
    radiance-gradient smoothing and the wavelength-error iteration - is
    identical to what it replaced.

    This began as a prototype written alongside a GPR-based version so the two
    could be compared. The node fit won on every channel, so the GPR path and
    the "_nodes" suffix that distinguished the two have both been removed.
    Outputs:

      Libera_ERF_analysis_summary_level_02.csv
      Libera_erf_srfs_02.csv
      erf_srf_fitting_*_02.png
      erf_campaign_correction_02.png             (multiplicative, per campaign)
      erf_campaign_wavelength_correction_02.png  (wavelength, per campaign)
      erf_ssw_edge_temperature_correction_02.png (SSW AMTIR-1 edge, per measurement)

    The SRF output carries wavelength_um, {ch} and u_random_{ch}, which is
    what science_radiometer_analyze_level_03 reads.
    """

    # Load the data
    paths = load_config()
    filename = "Libera_ERF_analysis_summary_level_01.csv"
    df = tidy_up_header(pd.read_csv(paths.analysis_dir / filename))

    # Same filtering as science_radiometer_analyze_level_02 - see the comments
    # there for why wavelength_source alone is not a sufficient guard.
    df = df.loc[(df["wavelength_source"] < 5) & (df["wavelength_center_um_sd"].notna())]
    df.sort_values(by='wavelength_um', inplace=True)
    df = df.loc[(df["ap_dia_mm"] > 11)]
    df = df.loc[(df["sci_rad_bench_temp_c"] > 32) & (df["sci_rad_bench_temp_c"] < 34.5)]

    # Correct for out of field
    df["srf_sw"] = df["gf_resp_sw"]/df["c_scirad_out_of_field_sw"]
    df["srf_to"] = df["gf_resp_to"]/df["c_scirad_out_of_field_to"]
    df["srf_lw"] = df["gf_resp_lw"]/df["c_scirad_out_of_field_lw"]
    df["srf_ss"] = df["gf_resp_ss"]/df["c_scirad_out_of_field_ss"]

    # Correct the SSW response for the temperature-driven motion of the AMTIR-1
    # filter edge near 0.74um - see ssw_edge_temperature_correction. This is the
    # only band edge that both moves measurably with temperature and sits in the
    # middle of a large spectral radiance signal.
    ssw_edge = ssw_edge_temperature_correction(
        wavelength_um=df["wavelength_um"].values,
        bench_temp_c=df["sci_rad_bench_temp_c"].values)
    df["c_ssw_edge_temp_ss"] = ssw_edge.correction
    df["srf_ss"] = df["srf_ss"] + ssw_edge.correction

    # Carried as a u_..._{ch} term so erf_uncertainty_terms.csv can pick it up
    # with a single "{ch}" row; the other three channels have no AMTIR-1 in
    # their filter stack and so get an identically zero column rather than
    # being left without one, which would fail the level_03 column lookup.
    df["u_ssw_edge_temp_ss"] = ssw_edge.u_correction
    for ch in ["sw", "to", "lw"]:
        df[f"u_ssw_edge_temp_{ch}"] = 0.0
    print(f"SSW AMTIR-1 edge temperature correction: "
          f"{int((np.abs(ssw_edge.correction) > 1e-5).sum())} of {len(df)} points "
          f"corrected, max {np.abs(ssw_edge.correction).max():.2e} SRF; "
          f"uncertainty envelope at dT = {ssw_edge.dt_envelope_c:.2f}C")

    # Science radiometer electrical non-linearity, as in the GPR version:
    # SW = 0.015 W m-2 sr-1 + 0.012%, Total = 0.015 + 0.022%,
    # LW = 0.011 + 0.015%, SSW = 0.018 + 0.013%
    SW_NONLINEAR_ABS = 0.015
    TO_NONLINEAR_ABS = 0.015
    LW_NONLINEAR_ABS = 0.011
    SS_NONLINEAR_ABS = 0.018
    SW_NONLINEAR_REL = 0.012/100.0
    TO_NONLINEAR_REL = 0.022/100.0
    LW_NONLINEAR_REL = 0.015/100.0
    SS_NONLINEAR_REL = 0.013/100.0
    df["u_scirad_nonlinear_sw"] = np.sqrt((SW_NONLINEAR_ABS)**2 + (SW_NONLINEAR_REL*df["gf_l0"]*df["gf_resp_sw"])**2)/df["gf_l0"]
    df["u_scirad_nonlinear_to"] = np.sqrt((TO_NONLINEAR_ABS)**2 + (TO_NONLINEAR_REL*df["gf_l0"]*df["gf_resp_to"])**2)/df["gf_l0"]
    df["u_scirad_nonlinear_lw"] = np.sqrt((LW_NONLINEAR_ABS)**2 + (LW_NONLINEAR_REL*df["gf_l0"]*df["gf_resp_lw"])**2)/df["gf_l0"]
    df["u_scirad_nonlinear_ss"] = np.sqrt((SS_NONLINEAR_ABS)**2 + (SS_NONLINEAR_REL*df["gf_l0"]*df["gf_resp_ss"])**2)/df["gf_l0"]

    # Smooth the radiance-field gradients within each alignment_session - see
    # science_radiometer_smooth_radiance_gradient.
    for term in ["gf_gx", "gf_gy", "gf_gp", "gf_gw"]:
        smoothed = science_radiometer_smooth_radiance_gradient(
            wavelength_um=df["wavelength_um"].values,
            value=df[term].values,
            value_sd=df[f"{term}_sd"].values,
            alignment_session=df["alignment_session"].values)
        df[f"{term}_smooth"] = smoothed.value
        df[f"{term}_sd_smooth"] = smoothed.value_sd

    # Random errors from spatial and angular non-uniformity
    POSITION_ERROR_MM = 0.25
    ANGLE_ERROR_DEG = 0.1
    df["u_spatial_unif"] = POSITION_ERROR_MM*np.sqrt(df["gf_gx_smooth"]**2 + \
                                                     df["gf_gx_sd_smooth"]**2) + \
                           POSITION_ERROR_MM*np.sqrt(df["gf_gy_smooth"]**2 + \
                                                     df["gf_gy_sd_smooth"]**2)

    df["u_angular_unif"] = ANGLE_ERROR_DEG*np.sqrt(df["gf_gp_smooth"]**2 + \
                                                   df["gf_gp_sd_smooth"]**2) + \
                           ANGLE_ERROR_DEG*np.sqrt(df["gf_gw_smooth"]**2 + \
                                                   df["gf_gw_sd_smooth"]**2)

    df = df.rename(columns={'gf_resp_sd_sw': 'u_resp_noise_sw',
                            'gf_resp_sd_to': 'u_resp_noise_to',
                            'gf_resp_sd_lw': 'u_resp_noise_lw',
                            'gf_resp_sd_ss': 'u_resp_noise_ss'})

    # Placeholder for the wavelength errors, filled in by the iteration below
    df["u_wavelength_dsrf_dwl_sw"] = 0
    df["u_wavelength_dsrf_dwl_to"] = 0
    df["u_wavelength_dsrf_dwl_lw"] = 0
    df["u_wavelength_dsrf_dwl_ss"] = 0

    for ch in ["sw", "to", "lw", "ss"]:
        df[f"u_{ch}_campaign_correction"] = 0

    n_iter = 6
    srf = None
    node_fit_by_channel = {}
    campaign_corr_by_channel = {}

    # One shared wavelength offset per campaign, in um. The laser wavelength
    # belongs to the measurement, not the channel, so this is accumulated
    # across channels and across the outer iterations rather than being left to
    # each channel's own campaign correction - see
    # science_radiometer_srf_campaign_correct change 3.
    shared_wl_offset = {int(s): 0.0 for s in df["measurement_campaign"].unique()}
    wl_offset_history = []
    # Each channel's own cumulative view of the offset, kept separately. The
    # agreement between them is the discriminator between the two explanations:
    # a wavelength error of the beam is common to all channels, whereas filter
    # ageing is specific to one. Averaged into the shared offset it would be
    # invisible, so it is tracked and reported.


    for it in range(n_iter):

        # The wavelength assignment used by every channel this iteration. One
        # column, so the four channels cannot disagree about what wavelength a
        # measurement was taken at.
        df["wavelength_um_corrected"] = df["wavelength_um"]*(
            1.0 + np.where(df["wavelength_source"] == 2,
                           df["measurement_campaign"].map(shared_wl_offset), 0.0))

        # Recorded here, at the top, so history[i] is the offset iteration i
        # actually fitted with. Appending after the update instead would make
        # the last row a value no fit ever used.
        wl_offset_history.append({s: 1e3*v for s, v in shared_wl_offset.items()})   # in 1e-3

        df["u_random_sw"] = np.sqrt(df["u_resp_noise_sw"]**2 + \
                                    df["u_wavelength_dsrf_dwl_sw"]**2)

        df["u_random_to"] = np.sqrt(df["u_resp_noise_to"]**2 + \
                                    df["u_wavelength_dsrf_dwl_to"]**2)

        df["u_random_lw"] = np.sqrt(df["u_resp_noise_lw"]**2 + \
                                    df["u_wavelength_dsrf_dwl_lw"]**2)

        df["u_random_ss"] = np.sqrt(df["u_resp_noise_ss"]**2 + \
                                    df["u_wavelength_dsrf_dwl_ss"]**2)

        for ch in ["sw", "to", "lw", "ss"]:

            campaign_corr = science_radiometer_srf_campaign_correct(
                wavelength_um=df["wavelength_um"].values,
                r_srf=df[f"srf_{ch}"].values,
                u_srf_random=df[f"u_random_{ch}"].values,
                measurement_campaign=df["measurement_campaign"].values,
                channel=ch,
                wavelength_offset_frac=shared_wl_offset,
                wavelength_offset_applies=(df["wavelength_source"] == 2).values)

            campaign_corr_by_channel[ch] = campaign_corr

            df[f"srf_{ch}_corrected"] = campaign_corr.r_corrected
            df[f"srf_{ch}_campaign_correction"] = df[f"srf_{ch}_corrected"] - df[f"srf_{ch}"]
            df[f"u_{ch}_campaign_correction"] = np.sqrt(df[f"u_{ch}_campaign_correction"]**2 + campaign_corr.u_campaign_correction**2)

            u_srf_random = np.sqrt(df[f"u_random_{ch}"].values**2 + df[f"u_{ch}_campaign_correction"].values**2)

            # The shared SRF fit
            node_fit = science_radiometer_srf_node_fit(
                wavelength_um=df["wavelength_um_corrected"].values,
                r_srf=df[f"srf_{ch}_corrected"].values,
                u_srf_random=u_srf_random,
                channel=ch)

            node_fit_by_channel[ch] = node_fit

            if srf is None:
                srf = node_fit[["wavelength_um"]].copy()

            srf[f"{ch}"]          = node_fit["srf_pred"]
            srf[f"u_random_{ch}"] = node_fit["srf_pred_sd"]

            # The two uncertainty components kept separately as well, since the
            # split between them is the main diagnostic for whether the node
            # grid is right: sd_param growing is the grid asking for more
            # resolution than the data supports, sd_scatter growing is the
            # measurements genuinely disagreeing.
            srf[f"u_param_{ch}"]   = node_fit["srf_pred_sd_param"]
            srf[f"u_scatter_{ch}"] = node_fit["srf_pred_sd_scatter"]

            df[f"srf_fit_{ch}"] = np.interp(df["wavelength_um_corrected"], node_fit["wavelength_um"], node_fit["srf_pred"])
            df[f"srf_fit_{ch}_sd"] = np.interp(df["wavelength_um_corrected"], node_fit["wavelength_um"], node_fit["srf_pred_sd"])
            df[f"srf_fit_{ch}_res"] = df[f"srf_{ch}_corrected"] - df[f"srf_fit_{ch}"]

            # Two wavelength-error terms, both first-order |dSRF/dwl| times a
            # wavelength uncertainty, differing by 30-1000x. They are combined
            # in completely different ways, so keep them straight:
            #
            #   u_wavelength_dsrf_dwl_{ch}     RANDOM, class Random,
            #       low_level 1. The per-measurement uncertainty of the OSA's
            #       own centroid, ~1e-6 relative. An independent draw per
            #       measurement, so it averages down, and it IS summed into
            #       u_random_{ch} at the top of this loop. level_03 then skips
            #       it on low_level==1, because u_random_{ch} already carries it.
            #
            #   u_wavelength_dsrf_dwl_ir_{ch}  SYSTEMATIC, class Wavelength,
            #       low_level 0. One shared unknown FRACTIONAL error of the IR
            #       wavelength scale, IR_WAVELENGTH_SCALE_U_FRAC, so the shift
            #       in um is that times the wavelength - which is why this
            #       column carries a factor of wavelength_um_corrected that the
            #       random term above does not. It does NOT
            #       average down and must NEVER be summed into u_random_{ch} -
            #       that would both double count it in level_03 and let the node
            #       fit average it away as if it were measurement noise.
            dsrf_dwl = np.gradient(node_fit["srf_pred"], node_fit["wavelength_um"])
            dsrf_dwl_at_wl = np.interp(df["wavelength_um_corrected"], node_fit["wavelength_um"], dsrf_dwl)

            df[f"u_wavelength_dsrf_dwl_{ch}"] = np.abs(dsrf_dwl_at_wl) * df["wavelength_center_um_sd"]

            # Zero where the OSA207 did not set the wavelength: the CCS200 edges
            # were shown campaign-stable to <=1nm and the grating path is a
            # separate instrument, so neither inherits the OSA's beam-sampling
            # error.
            df[f"u_wavelength_dsrf_dwl_ir_{ch}"] = (
                np.abs(dsrf_dwl_at_wl)
                * df["wavelength_um_corrected"]
                * np.where(df["wavelength_source"] == 2,
                           IR_WAVELENGTH_SCALE_U_FRAC, 0.0))

        # ------------------------------------------------------------------
        # Combine the four channels' residual wavelength estimates into the one
        # shared offset per campaign, and accumulate it.
        #
        # Inverse-variance weighted across channels: they are measuring the same
        # quantity - the wavelength error of the beam - through different band
        # edges, so the edge that pins it down best should dominate. LW's
        # 4.70-5.05um edge is the steepest in the instrument (the SRF moves 58%
        # of its peak across one laser linewidth) and duly carries most of the
        # weight.
        #
        # Correlated across channels, since the four share the measurements, so
        # this understates the combined sd. It is reported rather than used - the
        # budget term is the campaign-to-campaign scatter, not this - so the
        # understatement does not propagate.
        step = {}
        for s in (shared_wl_offset if fit_wavelength_offset else {}):
            num = den = 0.0
            for ch in ["sw", "to", "lw", "ss"]:
                cc = campaign_corr_by_channel[ch]
                d = cc.campaign_wavelength_residual_frac.get(s, 0.0)
                sd = cc.campaign_wavelength_residual_sd_frac.get(s, np.inf)
                if np.isfinite(sd) and sd > 0:
                    num += d/sd**2
                    den += 1.0/sd**2
            step[s] = num/den if den > 0 else 0.0

        # Re-reference the accumulated offset to the unweighted campaign mean
        # every iteration. The per-channel estimates were each mean-referenced,
        # but an inverse-variance blend of them is not, and without this the
        # common mode would random-walk over the iterations.
        for s in step:
            shared_wl_offset[s] += step[s]
        mean_shared = sum(shared_wl_offset.values())/len(shared_wl_offset)
        for s in shared_wl_offset:
            shared_wl_offset[s] -= mean_shared

    # Per-campaign wavelength offset, referenced to the unweighted campaign
    # mean. Convergence is worth seeing: the step should collapse, and the
    # spread across campaigns is the quantity that carries into the budget.
    print(f"\nPer-campaign FRACTIONAL wavelength offset [x1e-3], by outer "
          f"iteration (referenced to the unweighted campaign mean; "
          f"x4.6 for nm at the 4-5um anchor):")
    camps = sorted(shared_wl_offset)
    print(f"{'iter':>5}" + "".join(f"{'camp '+str(c):>12}" for c in camps) + f"{'spread':>10}")
    for i, h in enumerate(wl_offset_history):
        print(f"{i:>5}" + "".join(f"{h[c]:>12.2f}" for c in camps)
              + f"{max(h.values()) - min(h.values()):>10.2f}")
    # Each channel's cumulative estimate, against the shared value that was
    # actually applied. Channels agreeing means one wavelength error common to
    # the beam; a channel persistently disagreeing means something specific to
    # its own filter, which a wavelength correction would be the wrong model
    # for. Total normally has no in-scope steep edge and contributes nothing.
    print("\nOffset by channel [x1e-3] - the cross-channel spread is the "
          "beam-error vs filter-ageing discriminator:")
    print(f"{'ch':>6}" + "".join(f"{'camp '+str(c):>12}" for c in camps)
          + f"{'n edge pts':>22}")
    # A channel's own estimate of the total is the shared offset that was
    # applied to it plus whatever residual it still reports: T_ch = S + r_ch.
    # (Summing r_ch over the iterations would NOT give this - r_ch is
    # T_ch - S_i, so the sum grows with the iteration count.) A campaign with
    # too few steep points in a channel is shown as "--", not 0, because the
    # channel has no information about it rather than an estimate of zero.
    per_channel = {}
    for ch in ["sw", "to", "lw", "ss"]:
        cc = campaign_corr_by_channel[ch]
        per_channel[ch] = {
            c: (1e-3*wl_offset_history[-1][c]
                + cc.campaign_wavelength_residual_frac.get(c, 0.0)
                if np.isfinite(cc.campaign_wavelength_residual_sd_frac.get(c, np.inf))
                else None)
            for c in camps}
        print(f"{ch:>6}" + "".join(
            f"{1e3*per_channel[ch][c]:>12.2f}" if per_channel[ch][c] is not None
            else f"{'--':>12}" for c in camps)
            + f"   { {c: cc.n_wavelength_fit.get(c, 0) for c in camps} }")
    print(f"{'APPLIED':>6}" + "".join(f"{wl_offset_history[-1][c]:>12.2f}" for c in camps))
    for c in camps:
        vals = [1e3*per_channel[ch][c] for ch in per_channel
                if per_channel[ch][c] is not None]
        if len(vals) > 1:
            print(f"  campaign {c}: channel spread {max(vals) - min(vals):6.2f}e-3 "
                  f"on an applied {wl_offset_history[-1][c]:+.2f}e-3 "
                  f"({4.6*(max(vals) - min(vals)):.1f}nm vs "
                  f"{4.6*wl_offset_history[-1][c]:+.1f}nm at 4.6um)")

    # Report the fit diagnostics for each channel
    print(f"{'ch':>4} {'smoothing':>10} {'edof':>7} {'nodes':>6} {'chi2_red':>9} {'resid_rms':>10}")
    for ch in ["sw", "to", "lw", "ss"]:
        a = node_fit_by_channel[ch].attrs
        print(f"{ch:>4} {a['smoothing']:>10.3e} {a['edof']:>7.1f} {a['n_nodes']:>6d} "
              f"{a['chi2_reduced']:>9.1f} {a['resid_rms']:>10.3e}")

    # Parametric fringe terms, where a channel has one. The signal-to-noise on
    # each amplitude coefficient is the check that the term is earning its
    # place; a coefficient that drifts toward SNR ~1 means the node grid has
    # started absorbing it and n_amplitude should be revisited.
    for ch in ["sw", "to", "lw", "ss"]:
        a = node_fit_by_channel[ch].attrs
        if a["fringe"] is None:
            continue
        coeff, coeff_sd = a["fringe_coeff"], a["fringe_coeff_sd"]
        snr = np.abs(coeff/np.maximum(coeff_sd, 1e-300))
        print(f"  {ch} fringe: 2nd = {a['fringe']['two_nd_um']:.3f} um over "
              f"{a['fringe']['range_um'][0]}-{a['fringe']['range_um'][1]} um, "
              f"amplitude " + ", ".join(f"{c:+.3e} (SNR {s:.0f})"
                                        for c, s in zip(coeff, snr)))

    # Campaign-correction convergence. The slope guard's mask is recomputed
    # against the reference fit every iteration, so both the offset spread and
    # the point counts are reported - a churning n_valid would mean the guard
    # and the reference are chasing each other rather than settling.
    print()
    print("campaign correction, offset spread by iteration (%):")
    for ch in ["sw", "to", "lw", "ss"]:
        h = campaign_corr_by_channel[ch].history
        spreads = " ".join(f"{s.offset_spread:7.4f}" for s in h)
        print(f"  {ch:>3}: {spreads}   dropped by slope guard "
              f"{h[-1].n_steep_excluded:3d} of {h[-1].n_above_floor:3d} above floor"
              f"   n_valid {h[-1].n_valid}")

    ch_list = ["sw", "to", "lw", "ss"]
    name_list = ["SW", "Total", "LW", "SSW"]
    colors = ['#1f77b4', '#2ca02c', '#d62728', "#dc8c14"]

    # Campaign corrections, one subplot per channel. The shaded vertical bands
    # mark where the slope guard excluded points from the offset fit.

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 9), sharex=True)
    for ch, ax, name in zip(ch_list, axes.flat, name_list):

        # Recover the guard's mask from the final reference fit, the same way
        # science_radiometer_srf_campaign_correct computes it.
        nf = node_fit_by_channel[ch]
        fx = np.log10(nf["wavelength_um"].values)
        fy = nf["srf_pred"].values
        with np.errstate(divide='ignore', invalid='ignore'):
            log_slope = np.gradient(fy, fx)/np.maximum(fy, 1e-12)
        steep = np.abs(log_slope) > 10.0
        above_floor = fy > 0.08

        # Contiguous runs of excluded-and-above-floor wavelength, drawn as spans
        flag = steep & above_floor
        edges = np.flatnonzero(np.diff(flag.astype(int)) != 0) + 1
        bounds = np.concatenate(([0], edges, [len(flag)]))
        for i0, i1 in zip(bounds[:-1], bounds[1:]):
            if flag[i0]:
                ax.axvspan(nf["wavelength_um"].values[i0], nf["wavelength_um"].values[i1-1],
                           color='grey', alpha=0.25, zorder=1)

        for camp, color, label in [(0, 'red', 'Fall 2024'), (1, 'blue', 'February 2025')]:
            sub = df.loc[df["measurement_campaign"] == camp].sort_values("wavelength_um")
            ax.fill_between(sub["wavelength_um"],
                            sub[f"srf_{ch}_campaign_correction"] - sub[f"u_{ch}_campaign_correction"],
                            sub[f"srf_{ch}_campaign_correction"] + sub[f"u_{ch}_campaign_correction"],
                            color=color, alpha=0.35, zorder=2)
            ax.plot(sub["wavelength_um"], sub[f"srf_{ch}_campaign_correction"],
                    color=color, zorder=3, linewidth=1.0, label=label)

        ax.axhline(0, color='black', linewidth=0.5, zorder=1)
        ax.set_title(f"{name} correction to May 2025  (grey = dropped by slope guard)")
        log_x_axis_decimal(ax)
        ax.set_xlim(0.25, 15)
        ax.set_ylim(-0.04,0.02)
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('Correction to May 2025')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.figure_dir / 'erf_campaign_correction_02.png', bbox_inches="tight", dpi=200)

    # Per-campaign wavelength correction. This is the companion to the
    # multiplicative campaign correction above: that one moves the response
    # axis, this one moves the wavelength axis. One shared offset per campaign
    # is fitted across all four channels (the laser wavelength belongs to the
    # measurement, not to the channel), referenced to the unweighted mean of
    # the three campaigns rather than to May 2025 - for the radiometric scale
    # May 2025 is the truth, but the three campaigns are taken to have sampled
    # the beam randomly, so no one of them is privileged for wavelength.
    #
    # The offset is fractional, not a constant shift, and it applies only to
    # the OSA-measured wavelengths (wavelength_source == 2, roughly >1um); the
    # CCS200-measured points below 1um are stable to <=1nm and are left alone.
    # Hence the step at the source boundary. A fractional shift is what the SW
    # etalon supports: a constant shift of the size needed in the IR is
    # excluded by the etalon fringe phase, while a fractional one is nearly
    # degenerate with the etalon's own 2nd and so is not excluded. See
    # IR_WAVELENGTH_SCALE_U_FRAC for the uncertainty this leaves behind.

    campaign_style = [(0, 'red', 'Fall 2024'), (1, 'blue', 'February 2025'),
                      (2, 'green', 'May 2025')]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4.5))
    for camp, color, label in campaign_style:
        sub = df.loc[df["measurement_campaign"] == camp].sort_values("wavelength_um")
        if sub.empty:
            continue
        frac = sub["wavelength_um_corrected"].values/sub["wavelength_um"].values - 1.0
        shift_nm = 1e3*(sub["wavelength_um_corrected"].values - sub["wavelength_um"].values)
        osa = (sub["wavelength_source"] == 2).values
        for ax, y in zip(axes, [1e3*frac, shift_nm]):
            ax.plot(sub["wavelength_um"].values[osa], y[osa], 'o', color=color,
                    markersize=2, zorder=3, label=label)
            ax.plot(sub["wavelength_um"].values[~osa], y[~osa], 'x', color=color,
                    markersize=3, markeredgewidth=0.6, zorder=3,
                    label='_nolegend_' if camp else 'not OSA-measured (no offset)')

    for ax, ylab, title in zip(
            axes,
            ['Applied wavelength offset [x1e-3, fractional]', 'Applied wavelength offset [nm]'],
            ['Fractional - flat by construction where it applies',
             'Same offset in nm - grows with wavelength']):
        ax.axhline(0, color='black', linewidth=0.5, zorder=1)
        ax.axvline(1.0, color='grey', linewidth=0.8, linestyle=':', zorder=1)
        log_x_axis_decimal(ax)
        ax.set_xlim(0.25, 15)
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle('Per-campaign wavelength correction, referenced to the three-campaign mean '
                 '(dotted line = CCS200/OSA boundary)', fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.figure_dir / 'erf_campaign_wavelength_correction_02.png',
                bbox_inches="tight", dpi=200)

    # SSW AMTIR-1 filter-edge temperature correction. Unlike the two campaign
    # corrections above, this one is per-measurement rather than per-campaign:
    # it depends on each measurement's own bench temperature. See
    # ssw_edge_temperature_correction for the physics and the measured profile.
    # Left panel is the profile itself, right panel is what it actually did to
    # this dataset.

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4.5))

    wl_fine = np.linspace(0.62, 0.92, 601)
    prof = ssw_edge_temperature_correction(wavelength_um=wl_fine,
                                           bench_temp_c=np.full_like(wl_fine, SSW_EDGE_T_REF_C))
    axes[0].plot(wl_fine, prof.dresp_dt, color="#dc8c14", linewidth=1.2, zorder=3,
                 label='interpolated profile')
    axes[0].plot(SSW_EDGE_NODES_UM, SSW_EDGE_DRESP_DT_NODES, 'o', color='black',
                 markersize=4, zorder=4, label='measured nodes (29.1 vs 34.2C pairs)')
    axes[0].axhline(SSW_EDGE_NOISE_FLOOR, color='grey', linewidth=0.8, linestyle='--',
                    zorder=1, label=f'noise floor {SSW_EDGE_NOISE_FLOOR:.0e}')
    for b in SSW_EDGE_SUPPORT_UM:
        axes[0].axvline(b, color='grey', linewidth=0.8, linestyle=':', zorder=1)
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-6, 1e-2)
    axes[0].set_ylabel('-dSRF/dT [SRF / C]')
    axes[0].set_title('Measured edge sensitivity; dotted lines bound the support', fontsize=9)

    for camp, color, label in campaign_style:
        sub = df.loc[(df["measurement_campaign"] == camp) &
                     (np.abs(df["c_ssw_edge_temp_ss"]) > 0)].sort_values("wavelength_um")
        if sub.empty:
            continue
        axes[1].plot(sub["wavelength_um"], sub["c_ssw_edge_temp_ss"], 'o', color=color,
                     markersize=4, zorder=3,
                     label=f'{label}  ({sub["sci_rad_bench_temp_c"].min():.1f}'
                           f'-{sub["sci_rad_bench_temp_c"].max():.1f}C)')
    # The dashed line is the correction the profile predicts for the coldest
    # retained measurement; the grey band is the 1-sigma UNCERTAINTY carried on
    # whatever correction is applied (25% of that same worst case, so it is a
    # quarter of the dashed line, not a bound on it).
    u_env = SSW_EDGE_U_FRAC*prof.dresp_dt*ssw_edge.dt_envelope_c
    axes[1].plot(wl_fine, -prof.dresp_dt*ssw_edge.dt_envelope_c, color='black',
                 linewidth=0.9, linestyle='--', zorder=2,
                 label=f'profile at the coldest retained point (dT = -{ssw_edge.dt_envelope_c:.2f}C)')
    axes[1].fill_between(wl_fine, -u_env, u_env, color='grey', alpha=0.3, zorder=1,
                         label='1-sigma uncertainty on the applied correction')
    axes[1].axhline(0, color='black', linewidth=0.5, zorder=2)
    axes[1].set_ylabel('Correction added to SRF [-]')
    axes[1].set_title('Correction actually applied, by measurement', fontsize=9)

    for ax in axes:
        ax.set_xlim(0.62, 0.92)
        ax.set_xlabel('Wavelength [um]')
        ax.legend(fontsize=7)
    fig.suptitle(f'SSW AMTIR-1 filter-edge temperature correction, referenced to '
                 f'{SSW_EDGE_T_REF_C:.2f}C', fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.figure_dir / 'erf_ssw_edge_temperature_correction_02.png',
                bbox_inches="tight", dpi=200)

    # Corrected fit with its uncertainty band, and the node positions marked
    fig, ax = plt.subplots(figsize=(10, 6))
    for ch, color in zip(ch_list, colors):
        ax.errorbar(df["wavelength_um"], df[f"srf_{ch}_corrected"], yerr=df[f"u_random_{ch}"],
                    fmt='o', color=color, markersize=1, capsize=1, zorder=1, alpha=0.6)
        ax.fill_between(srf["wavelength_um"],
                        srf[f"{ch}"] - srf[f"u_random_{ch}"],
                        srf[f"{ch}"] + srf[f"u_random_{ch}"],
                        alpha=0.3, color=color, zorder=2)
        ax.plot(srf["wavelength_um"], srf[f"{ch}"], color=color, zorder=3, linewidth=0.8)

        a = node_fit_by_channel[ch].attrs
        ax.plot(a["node_wavelength_um"], a["node_value"], 'o', color=color,
                markersize=2.5, markeredgecolor='black', markeredgewidth=0.3, zorder=4)
    ax.set_title('Node fit, corrected (markers on the curve are the fitted nodes)')
    log_x_axis_decimal(ax)
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('SRF [-]')
    ax.set_xlim(0.25, 15)
    ax.set_ylim(-0.02, 1.1)
    fig.savefig(paths.figure_dir / 'erf_srf_fitting_corr_02.png', bbox_inches="tight", dpi=200)

    # Residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    for ch, color in zip(ch_list, colors):
        df_tmp = df.loc[(df[f"srf_fit_{ch}"] > 0.02)].copy()
        ax.errorbar(df_tmp["wavelength_um"], df_tmp[f"srf_fit_{ch}_res"], yerr=df_tmp[f"u_random_{ch}"],
                    fmt='o', color=color, markersize=1, capsize=1, zorder=1, alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5, zorder=2)
    ax.set_title('Node fit residual errors')
    log_x_axis_decimal(ax)
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('Residual Error')
    ax.set_xlim(0.25, 15)
    ax.set_ylim(-0.02, 0.02)
    fig.savefig(paths.figure_dir / 'erf_srf_fitting_corr_residuals_02.png', bbox_inches="tight", dpi=200)


    # Save the data summary
    filename = "Libera_ERF_analysis_summary_level_02.csv"
    df.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    # Save the SRF data
    filename = "Libera_erf_srfs_02.csv"
    srf.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    print("Done!")


def lst_analyze_level_02():
    """
    Analyzes Libera_ERF_LST_analysis_summary_level_01.csv: applies the out of
    field correction, builds the uncertainty terms, corrects each measurement
    campaign for its own calibration offset, and fits the LST spectral response
    function by Gaussian process regression.

    The LST counterpart of science_radiometer_analyze_level_02. Structurally the
    same, with three differences that follow from the instrument rather than
    from choice:

      one channel     The LST is a single Total-channel witness radiometer, so
                      everything the science radiometer version does four times
                      over ["sw","to","lw","ss"] happens once here. The channel
                      key is "lst"; level_01 writes its response columns
                      unsuffixed (gf_resp, c_scirad_out_of_field), so they are
                      read under those names and carried forward as srf_lst,
                      u_random_lst and so on, which keeps this function line
                      for line comparable with the science radiometer one and
                      makes a level_03 adaptation a matter of ch="lst".

      no temperature  The science radiometer version keeps only its 32-34.5C
      filter          bench-temperature data. The LST's em_telescope_temp_c
                      spans 28.97-29.44C across the entire dataset - 0.47C end
                      to end - so there is no temperature spread to filter on,
                      and the science radiometer's window would keep no rows at
                      all. Worth remembering when comparing this SRF against
                      the flight Total channel: the two are characterized about
                      5C apart.

      non-linearity   The LST's own electrical non-linearity has not been
                      measured, so the Total channel's figures stand in - see
                      LST_NONLINEAR_ABS/REL below. The LST models the Total
                      channel but has its own detector board and FPE (driver
                      p02), so this is a placeholder to revisit.

    The campaign correction needs no special handling here: the LST responds
    across 0.35-13.8um, so its offset polynomial is fitted over 1.60 of the
    1.71 decades it is applied to (a reach of 0.07 fitted spans), which is the
    comfortable regime the science radiometer's Total channel sits in - not the
    LW channel's 2.4-2.7, which is what motivated the extrapolation guards in
    science_radiometer_srf_campaign_correct.

    Summary output:
    Libera_ERF_LST_analysis_summary_level_02.csv
    Libera_erf_lst_srfs_02.csv
    """

    # Load the data
    paths = load_config()
    filename = "Libera_ERF_LST_analysis_summary_level_01.csv"
    df = tidy_up_header(pd.read_csv(paths.analysis_dir / filename))

    # Keep data points with a valid spectrum. wavelength_source alone isn't a
    # reliable guard: analyze_ccs200_spectrum/analyze_osa207_spectrum/
    # analyze_grating_spectrum all set wavelength_source to a "real fit" code
    # even on their failure path, which only shows up as a NaN
    # wavelength_center_um_sd. Filtering on both keeps a failed fit from
    # silently turning into a NaN u_wavelength_dsrf_dwl_lst that would poison
    # the whole fit.
    df = df.loc[(df["wavelength_source"] < 5) & (df["wavelength_center_um_sd"].notna())]

    # Sort the data by wavelength
    df.sort_values(by='wavelength_um', inplace=True)

    # Throw out the 10 mm aperture data, will review this later. Same treatment
    # as the science radiometers; leaves 356 of 433 LST measurements.
    df = df.loc[(df["ap_dia_mm"] > 11)]

    # No bench-temperature filter here - see the docstring. The LST data is
    # effectively isothermal at ~29.1C.

    # Correct for out of field
    df["srf_lst"] = df["gf_resp"]/df["c_scirad_out_of_field"]

    # Error term for the LST electrical non-linearity. These are the Total
    # channel's measured values, standing in until the LST's own non-linearity
    # is characterized:
    # Total = 0.015 W m-2 sr-1 + 0.022%
    LST_NONLINEAR_ABS = 0.015
    LST_NONLINEAR_REL = 0.022/100.0
    # First an absolute error in radiance units, then scaled by the 100%
    # radiance level from PBR-R
    df["u_scirad_nonlinear_lst"] = np.sqrt((LST_NONLINEAR_ABS)**2 + \
                                           (LST_NONLINEAR_REL*df["gf_l0"]*df["gf_resp"])**2)/df["gf_l0"]

    # The raw per-measurement radiance-field gradients (and their fit
    # uncertainty) blow up wherever the channel has almost no signal to sense
    # position/pointing with - for the LST that is the deep UV, where its
    # response falls to a few tenths of a percent. Smoothing each gradient as a
    # function of wavelength within its own alignment_session (geometry resets
    # at realignment, but should vary smoothly with wavelength within one
    # session) borrows strength from the well-constrained nearby wavelengths
    # instead of taking that noise spike at face value. See
    # science_radiometer_smooth_radiance_gradient.
    for term in ["gf_gx", "gf_gy", "gf_gp", "gf_gw"]:
        smoothed = science_radiometer_smooth_radiance_gradient(
            wavelength_um=df["wavelength_um"].values,
            value=df[term].values,
            value_sd=df[f"{term}_sd"].values,
            alignment_session=df["alignment_session"].values)
        df[f"{term}_smooth"] = smoothed.value
        df[f"{term}_sd_smooth"] = smoothed.value_sd

    # Plot the raw per-measurement gradients against their smoothed fits, one
    # subplot per term, one color per alignment_session. The errorbars are the
    # raw per-measurement values, the line is the session-local smoothed fit.
    sessions = np.unique(df["alignment_session"])
    cmap = plt.get_cmap('tab20')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    for term, ax, rng, name, units in zip(["gf_gx", "gf_gy", "gf_gw", "gf_gp"],
                             axes.flat,
                             [0.006,0.006,0.1,0.1],
                             ["X","Y","Yaw","Pitch"],
                             ["mm","mm","deg","deg"]):
        for i, session in enumerate(sessions):
            color = cmap(i % 20)

            # Sorted by wavelength so the smoothed line doesn't zigzag.
            sub = df.loc[df["alignment_session"] == session].sort_values("wavelength_um")

            ax.errorbar(sub["wavelength_um"], sub[term], yerr=sub[f"{term}_sd"],
                        fmt='o', color=color, markersize=2, capsize=1, zorder=2, alpha=0.2)

            ax.plot(sub["wavelength_um"], sub[f"{term}_smooth"], color=color,
                    linewidth=1, zorder=3, label=f"session {session}")

        ax.axhline(0, color='black', linewidth=0.5, zorder=1)
        log_x_axis_decimal(ax)
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel(f"{name} Gradient [{units}-1]")
        ax.set_title(term)
        ax.set_ylim([-rng,rng])
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(paths.figure_dir / 'erf_lst_gradient_smoothing_02.png', bbox_inches="tight", dpi=200)

    # Now estimate the random errors from spatial and angular non-uniformity.
    # RSS the X and Y slope with the X and Y slope uncertainty, times the
    # position error; likewise with pitch and yaw.
    POSITION_ERROR_MM = 0.25
    ANGLE_ERROR_DEG = 0.1
    df["u_spatial_unif"] = POSITION_ERROR_MM*np.sqrt(df["gf_gx_smooth"]**2 + \
                                                     df["gf_gx_sd_smooth"]**2) + \
                           POSITION_ERROR_MM*np.sqrt(df["gf_gy_smooth"]**2 + \
                                                     df["gf_gy_sd_smooth"]**2)

    df["u_angular_unif"] = ANGLE_ERROR_DEG*np.sqrt(df["gf_gp_smooth"]**2 + \
                                                   df["gf_gp_sd_smooth"]**2) + \
                           ANGLE_ERROR_DEG*np.sqrt(df["gf_gw_smooth"]**2 + \
                                                   df["gf_gw_sd_smooth"]**2)

    # Rename the response uncertainty column to match the naming conventions
    df = df.rename(columns={'gf_resp_sd': 'u_resp_noise_lst'})

    # Placeholder for the wavelength error. The spectral response curve and its
    # slope are needed first, so this starts at zero and is filled in by the
    # iteration below.
    df["u_wavelength_dsrf_dwl_lst"] = 0

    # Column to hold the uncertainty in the campaign correction
    df["u_lst_campaign_correction"] = 0

    # Iterate the SRF fit, updating the wavelength error as the fit
    # improves.
    n_iter = 6
    srf = None
    campaign_corr = None
    for it in range(n_iter):

        # Combine the random uncertainties
        df["u_random_lst"] = np.sqrt(df["u_resp_noise_lst"]**2 + \
                                     df["u_wavelength_dsrf_dwl_lst"]**2)

        # Correct each measurement_campaign for its own multiplicative
        # calibration offset before fitting the shared SRF curve - see
        # science_radiometer_srf_campaign_correct. Unlike the science
        # radiometers, which are referenced to the most recent campaign alone
        # because May 2025 IS the radiometric scale, the LST is referenced to
        # the equally weighted mean of its final two campaigns - see
        # lst_reference_campaigns. The correction's own
        # uncertainty (from the covariance of the fitted offset polynomials) is
        # added in quadrature to the random uncertainty used downstream, so a
        # campaign whose offset was poorly constrained gets a bigger error bar
        # rather than being corrected "for free".
        campaign_corr = science_radiometer_srf_campaign_correct(
            wavelength_um=df["wavelength_um"].values,
            r_srf=df["srf_lst"].values,
            u_srf_random=df["u_random_lst"].values,
            measurement_campaign=df["measurement_campaign"].values,
            channel="lst",
            reference_campaigns=lst_reference_campaigns(
                df["measurement_campaign"].values))

        # The response corrected to the most recent campaign
        df["srf_lst_corrected"] = campaign_corr.r_corrected

        # The correction applied to each measurement
        df["srf_lst_campaign_correction"] = df["srf_lst_corrected"] - df["srf_lst"]

        # The uncertainty of the correction, accumulating this iteration's value
        df["u_lst_campaign_correction"] = np.sqrt(df["u_lst_campaign_correction"]**2 + \
                                                  campaign_corr.u_campaign_correction**2)

        # The propagated uncertainty, RSS'ing the random uncertainty with the
        # campaign correction
        u_srf_random = np.sqrt(df["u_random_lst"].values**2 + \
                               df["u_lst_campaign_correction"].values**2)

        # Fit the whole dataset with the penalized node fit, not the GPR -
        # see science_radiometer_srf_node_fit for why, and the comment on
        # LST_TAIL_NODES_UM for how this channel's grid was chosen.
        node_fit = science_radiometer_srf_node_fit(wavelength_um=df["wavelength_um"].values,
                                        r_srf=df["srf_lst_corrected"].values,
                                        u_srf_random=u_srf_random,
                                        channel="lst")

        if srf is None:
            srf = node_fit[["wavelength_um"]].copy()

        # Fill in the SRF data
        srf["lst"]          = node_fit["srf_pred"]
        srf["u_random_lst"] = node_fit["srf_pred_sd"]

        # Fill in the fit information
        df["srf_fit_lst"] = np.interp(df["wavelength_um"], node_fit["wavelength_um"], node_fit["srf_pred"])
        df["srf_fit_lst_sd"] = np.interp(df["wavelength_um"], node_fit["wavelength_um"], node_fit["srf_pred_sd"])
        df["srf_fit_lst_res"] = df["srf_lst_corrected"] - df["srf_fit_lst"]

        # Now use the fit to estimate the error due to wavelength
        # uncertainty: first-order propagation, |dSRF/dwavelength| times the
        # wavelength uncertainty for that measurement. np.gradient (not np.diff)
        # because it returns one derivative per grid point rather than n-1
        # midpoint values, and it correctly accounts for the prediction grid
        # being log-spaced rather than uniform.
        dsrf_dwl = np.gradient(node_fit["srf_pred"], node_fit["wavelength_um"])

        # Interpolate the derivative from the fit's dense grid onto each
        # measurement's own wavelength, the same way srf_fit_lst is.
        dsrf_dwl_at_wl = np.interp(df["wavelength_um"], node_fit["wavelength_um"], dsrf_dwl)

        # The response uncertainty due to wavelength errors
        df["u_wavelength_dsrf_dwl_lst"] = np.abs(dsrf_dwl_at_wl) * df["wavelength_center_um_sd"]

    # Label each campaign by the months its files were taken in, read from the
    # log filenames rather than hardcoded - the science radiometer version names
    # its campaigns in the plotting code, which only holds for that dataset.
    log_time = pd.to_datetime(df["filename"].str.extract(r'(\d{8}_\d{4})')[0], format='%Y%m%d_%H%M')
    campaign_label = {}
    for c in np.unique(df["measurement_campaign"]):
        t = log_time.loc[df["measurement_campaign"] == c]
        first, last = t.min().strftime("%b %Y"), t.max().strftime("%b %Y")
        campaign_label[c] = first if first == last else f"{first} - {last}"

    # Plot the campaign corrections, one line per campaign. With a single
    # reference campaign that one is corrected to itself, so its line sits at
    # zero by construction and is left out; with the LST's two references
    # neither is, since each is moved to the mean of the pair, so every campaign
    # is drawn.
    refs = campaign_corr.reference_campaigns
    target = refs[0] if len(refs) == 1 else None
    ref_label = " & ".join(campaign_label[c] for c in refs)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    for i, c in enumerate(np.unique(df["measurement_campaign"])):
        if c == target:
            continue
        sub = df.loc[df["measurement_campaign"] == c].sort_values("wavelength_um")
        color = cmap(i % 10)
        ax.fill_between(sub["wavelength_um"],
                        sub["srf_lst_campaign_correction"] - sub["u_lst_campaign_correction"],
                        sub["srf_lst_campaign_correction"] + sub["u_lst_campaign_correction"],
                        color=color, alpha=0.4, zorder=2)
        ax.plot(sub["wavelength_um"], sub["srf_lst_campaign_correction"], color=color,
                label=campaign_label[c], zorder=3)
    ax.axhline(0, color='black', linewidth=0.5, zorder=1)
    ax.set_title(f"LST Corrections to the {ref_label} Data")
    log_x_axis_decimal(ax)
    ax.set_xlim(0.25, 14)
    ax.set_ylim(-0.04,0.02)
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel(f'Correction to {ref_label}')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.figure_dir / 'erf_lst_campaign_correction_01.png', bbox_inches="tight", dpi=200)

    # Plot the uncorrected and corrected SRF against the fit
    color = '#2ca02c'
    for tag, col, title in [("uncorr", "srf_lst", "Uncorrected"),
                            ("corr", "srf_lst_corrected", "Corrected")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(df["wavelength_um"], df[col], yerr=df["u_random_lst"],
                fmt='o', color=color, markersize=1, capsize=1, zorder=1, alpha=0.6)
        ax.fill_between(srf["wavelength_um"],
                        srf["lst"] - srf["u_random_lst"],
                        srf["lst"] + srf["u_random_lst"],
                        alpha=0.3, color=color, zorder=2)
        ax.plot(srf["wavelength_um"], srf["lst"], color=color, zorder=3, linewidth=0.8)
        ax.set_title(f'LST {title}')
        log_x_axis_decimal(ax)
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('SRF [-]')
        ax.set_xlim(0.25,15)
        ax.set_ylim(-0.02,1.1)
        fig.savefig(paths.figure_dir / f'erf_lst_srf_fitting_{tag}_02.png', bbox_inches="tight", dpi=200)

    # Plot the residuals, over the region where the channel actually responds
    fig, ax = plt.subplots(figsize=(10, 6))
    df_tmp = df.loc[(df["srf_fit_lst"] > 0.02)].copy()
    ax.errorbar(df_tmp["wavelength_um"], df_tmp["srf_fit_lst_res"], yerr=df_tmp["u_random_lst"],
            fmt='o', color=color, markersize=1, capsize=1, zorder=1, alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5, zorder=1)
    ax.set_title('LST Residual Errors')
    log_x_axis_decimal(ax)
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('Residual Error')
    ax.set_xlim(0.25,15)
    ax.set_ylim(-0.02,0.02)
    fig.savefig(paths.figure_dir / 'erf_lst_srf_fitting_corr_residuals_02.png', bbox_inches="tight", dpi=200)

    # Save the data summary
    filename = "Libera_ERF_LST_analysis_summary_level_02.csv"
    df.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    # Save the SRF data
    filename = "Libera_erf_lst_srfs_02.csv"
    srf.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    print("Done!")

def interp_with_fallback(*, x_grid, x_primary, y_primary, x_fallback, y_fallback, max_gap=0.03):
    """
    Linearly interpolate y_primary(x_primary) onto x_grid, falling back to
    y_fallback(x_fallback) wherever x_grid isn't well-supported by the
    primary data (its nearest primary point is farther than max_gap away).

    This is for combining a preferred/"truth" dataset with a secondary one
    used only to fill coverage gaps - naively pooling both (e.g. sorting
    their union by x and interpolating across it) treats them as one
    homogeneous population, so the interpolation jumps between two
    genuinely different regimes at every point they happen to interleave.
    See science_radiometer_analyze_level_03's use of this: 13mm-aperture
    measurements are the truth (smallest out-of-field correction, matches
    the final campaign), 12mm-aperture measurements exist only to fill in
    wavelength coverage 13mm doesn't have, and mixing them directly caused
    several Wavelength-class uncertainty terms to visibly bounce every time
    the two apertures interleaved.

    x_grid/x_primary/x_fallback are compared directly, so pass whatever
    coordinate the gap should be measured in (e.g. log10(wavelength) for a
    roughly-log-spaced measurement grid, so max_gap has a consistent meaning
    across the whole range).
    """

    order_p = np.argsort(x_primary)
    xp, yp = x_primary[order_p], y_primary[order_p]

    order_f = np.argsort(x_fallback)
    xf, yf = x_fallback[order_f], y_fallback[order_f]

    value_primary = np.interp(x_grid, xp, yp)
    value_fallback = np.interp(x_grid, xf, yf)

    # Distance from each grid point to its nearest primary-data neighbor.
    idx = np.searchsorted(xp, x_grid)
    idx_lo = np.clip(idx - 1, 0, len(xp) - 1)
    idx_hi = np.clip(idx, 0, len(xp) - 1)
    gap = np.minimum(np.abs(x_grid - xp[idx_lo]), np.abs(x_grid - xp[idx_hi]))

    return np.where(gap > max_gap, value_fallback, value_primary)


def science_radiometer_analyze_level_03():
    """
    Combines every uncertainty term listed in erf_uncertainty_terms.csv into
    a single total SRF uncertainty curve per channel, as a function of
    wavelength.

    erf_uncertainty_terms.csv drives how each term is combined, via its
    class/low_level/session_variable columns:
      - low_level==1 terms (u_resp_noise_{ch}, u_wavelength_dsrf_dwl_{ch})
        are skipped - they're already combined into u_random_{ch} back in
        level_02 (that combination *is* the node fit's posterior uncertainty),
        so including them again here would double count them.
      - class=="Random" (only u_random_{ch} survives the low_level filter)
        is taken directly from Libera_erf_srfs_02.csv - it's already a
        function of wavelength on the same grid used here.
      - class=="Common" terms are fully correlated across every measurement,
        so their per-measurement values in level_02's summary are collapsed
        to one representative scalar (the mean) and applied flat across the
        whole curve, not evaluated as a function of wavelength.
      - class=="Wavelength" terms vary with wavelength but are correlated
        across measurements at a given wavelength, so their per-measurement
        values are interpolated onto the fit's wavelength grid instead of
        being averaged away.
      - session_variable==1 terms (u_spatial_unif, u_angular_unif) are also
        class=="Wavelength", but their true value is an independent draw per
        alignment_session rather than one fixed bias for the whole dataset -
        the instrument gets realigned every session (not just every
        measurement_campaign, which tracks the much slower timescale the SRF
        itself is expected to drift on), so pooling n_sessions independent
        realizations into the final curve averages them down by
        sqrt(n_sessions).

    Loads:
      Libera_ERF_analysis_summary_level_02.csv (per-measurement uncertainty terms)
      Libera_erf_srfs_02.csv                   (already-fit SRF curves)
      erf_uncertainty_terms.csv                (which terms exist and how to combine them)

    Summary output:
    Libera_erf_srfs_03.csv
    """

    # Load the data
    paths = load_config()
    filename = "Libera_ERF_analysis_summary_level_02.csv"
    df = tidy_up_header(pd.read_csv(paths.analysis_dir / filename))

    filename = "Libera_erf_srfs_02.csv"
    srf = tidy_up_header(pd.read_csv(paths.analysis_dir / filename))

    filename = "erf_uncertainty_terms.csv"
    terms = tidy_up_header(pd.read_csv(paths.analysis_dir / filename))

    # Already combined into u_random_{ch} in level_02 - re-including these
    # here would double count them.
    terms = terms.loc[terms["low_level"] == 0]

    channels = ["sw", "to", "lw", "ss"]

    # Independent realignment events pooled into the final curve - each
    # session draws its own independent spatial/angular offset rather than
    # sharing one fixed bias across the whole dataset, so session_variable
    # terms average down by sqrt of this.
    n_sessions = df["alignment_session"].nunique()

    wl_grid = srf["wavelength_um"].values
    x_grid = np.log10(wl_grid)

    # The 13mm aperture is the truth configuration (smallest out-of-field
    # correction, matches the final May 2025 campaign); the 12mm data is
    # included only to fill in wavelength coverage 13mm doesn't have. Split
    # once up front so every Wavelength-class term below can be interpolated
    # with 13mm preferred and 12mm as a gap-filling fallback (see
    # interp_with_fallback) rather than pooling both apertures into one
    # sorted-by-wavelength interpolation, which was jumping between the two
    # apertures' genuinely different values every time they interleaved.
    x_13 = np.log10(df.loc[df["ap_dia_mm"] == 13, "wavelength_um"].values)
    x_12 = np.log10(df.loc[df["ap_dia_mm"] == 12, "wavelength_um"].values)

    # Visualize the wavelength dependence of the individual uncertainty
    # terms that feed into the totals below (excluding the Random term,
    # which lives in srf as u_random_{ch} rather than as a raw column in
    # df). One subplot per channel, since a term with a "{ch}" placeholder
    # resolves to a different column per channel; shared terms (the PBR-R
    # terms, u_spatial_unif, u_angular_unif) show up identically in every
    # panel. Colors are assigned once so a given term is the same color in
    # every panel it appears in.
    plot_terms = [row["term"] for _, row in terms.iterrows() if row["class"] != "Random"]

    # Only the terms that clear this threshold in at least one channel are
    # ever actually drawn (see the loop below), so colors are assigned from
    # that shortened list rather than the full plot_terms - assigning colors
    # to every term first and then hiding some at plot time let two
    # unrelated (and simultaneously visible) terms land on the same color,
    # since cmap(i % 8) wraps every 8 terms regardless of which ones the
    # threshold actually keeps.
    visibility_threshold = 1e-4
    visible_terms = []
    for term in plot_terms:
        for ch in channels:
            col = term.format(ch=ch) if "{ch}" in term else term
            if col in df.columns and df[col].max() > visibility_threshold:
                visible_terms.append(term)
                break

    cmap = plt.get_cmap('tab10')
    term_color = {term: cmap(i) for i, term in enumerate(visible_terms)}

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    for ch, ax in zip(channels, axes.flat):
        for term in visible_terms:
            col = term.format(ch=ch) if "{ch}" in term else term
            if col not in df.columns:
                print(f"Warning: '{col}' not found in level_02 data - skipping {term} for {ch}.")
                continue
            ax.scatter(df["wavelength_um"], df[col], s=4, color=term_color[term], alpha=0.5, label=col)
        log_x_axis_decimal(ax)
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('Uncertainty')
        ax.set_ylim(1e-5,1e-2)
        ax.set_title(ch.upper())
        ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figure_dir / 'erf_uncertainty_terms_03.png', bbox_inches="tight", dpi=200)

    for ch in channels:

        # Running sums of variance (not sd) per category, so each category
        # can be RSS'd together at the end just by summing and taking one
        # final sqrt.
        var_common = np.zeros_like(wl_grid)
        var_wavelength = np.zeros_like(wl_grid)
        var_session_avg = np.zeros_like(wl_grid)

        for _, row in terms.iterrows():
            term = row["term"]
            cls = row["class"]

            if cls == "Random":
                # Handled separately below - already a function of
                # wavelength on this exact grid via u_random_{ch} in srf.
                continue

            # Per-channel terms have a literal "{ch}" placeholder in their
            # name; shared terms (e.g. the PBR-R terms, u_spatial_unif) use
            # the same column for every channel.
            col = term.format(ch=ch) if "{ch}" in term else term

            if col not in df.columns:
                print(f"Warning: '{col}' not found in level_02 data - skipping {term} for {ch}.")
                continue

            if cls == "Common":
                # A single value, the same at every wavelength. Collapsing
                # each measurement's value to its mean is a simplification -
                # worth revisiting if a term (e.g. u_scirad_nonlinear_{ch},
                # which depends on the radiance level of each measurement)
                # should instead be evaluated at a specific reference
                # radiance rather than averaged over historical conditions.
                value = np.full_like(wl_grid, df[col].mean(), dtype=float)
                var_common += value**2

            elif cls == "Wavelength":
                value = interp_with_fallback(
                    x_grid=x_grid,
                    x_primary=x_13, y_primary=df.loc[df["ap_dia_mm"] == 13, col].values,
                    x_fallback=x_12, y_fallback=df.loc[df["ap_dia_mm"] == 12, col].values,
                )

                if row["session_variable"] == 1:
                    value = value / np.sqrt(n_sessions)
                    var_session_avg += value**2
                else:
                    var_wavelength += value**2

            else:
                print(f"Warning: unrecognized class '{cls}' for term {term} - skipping.")

        u_common = np.sqrt(var_common)
        u_wavelength = np.sqrt(var_wavelength)
        u_session_avg = np.sqrt(var_session_avg)
        u_random = srf[f"u_random_{ch}"].values

        srf[f"u_common_{ch}"] = u_common
        srf[f"u_wavelength_{ch}"] = u_wavelength
        srf[f"u_session_avg_{ch}"] = u_session_avg
        srf[f"u_total_{ch}"] = np.sqrt(u_random**2 + u_common**2 +
                                        u_wavelength**2 + u_session_avg**2)

    # Make a plot of the results
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728', "#dc8c14"]
    for ch, color in zip(channels, colors):
        ax.plot(srf["wavelength_um"], srf[ch], color=color, zorder=3, linewidth=0.8, label=ch.upper())
        ax.fill_between(srf["wavelength_um"],
                         srf[ch] - srf[f"u_total_{ch}"],
                         srf[ch] + srf[f"u_total_{ch}"],
                         alpha=0.3, color=color, zorder=2)
    log_x_axis_decimal(ax)
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('SRF [-]')
    ax.set_title('Libera SW SRFs')
    ax.set_xlim(0.25, 14)
    ax.set_ylim(-0.02, 1.1)
    ax.legend()
    fig.savefig(paths.figure_dir / 'erf_srf_total_uncertainty_03.png', bbox_inches="tight", dpi=200)

    # Make a plots of the uncertainties for each channel
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), sharex=True, sharey=True)
    # for ch, ax, name in zip(ch_list, axes.flat, name_list):
    name_list = ['SW','Total','LW','SSW']
    for ch, color, ax, name in zip(channels, colors, axes.flat, name_list):        
        ax.plot(srf["wavelength_um"], srf[f"u_total_{ch}"], color="#004cff", zorder=3, linewidth=0.8, label='Total')
        ax.plot(srf["wavelength_um"], srf[f"u_wavelength_{ch}"], color="#ff8000", zorder=3, linewidth=0.8, label='Wavelength')
        ax.plot(srf["wavelength_um"], srf[f"u_common_{ch}"], color="#b41f1f", zorder=3, linewidth=0.8, label='Common')
        ax.plot(srf["wavelength_um"], srf[f"u_session_avg_{ch}"], color="#00ff11", zorder=3, linewidth=0.8, label='Alignment')
        ax.set_xlim(0.25, 16)
        ax.set_ylim(0.01/100.0,1.0/100.0)
        log_x_axis_decimal(ax)
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('SRF Uncertaintiy [-]')
        ax.set_title(f'{name} Channel')
        ax.legend(loc='upper right', frameon=False)
        print()
    fig.savefig(paths.figure_dir / f'erf_srf_total_uncertainty_summary_{ch}_03.png', bbox_inches="tight", dpi=200)

    # Save the SRF data with the combined uncertainty terms
    filename = "Libera_erf_srfs_03.csv"
    srf.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    # Save a simplified version with just the SRF and its total uncertainty
    # per channel (no wavelength/common/alignment breakdown, no session/fit
    # diagnostics) - for downstream users who only need the end result.
    srf4 = pd.DataFrame({"wavelength_um": srf["wavelength_um"]})
    for ch, name in zip(channels, name_list):
        srf4[name] = srf[ch]
        srf4[f"uc_{name}"] = srf[f"u_total_{ch}"]

    filename = "libera_erf_srfs_04.csv"
    srf4.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    print("Done!")


def read_modtran_tp7_scenes(*, file, max_cases=None):
    """
    Parses a MODTRAN tape7 (.tp7) CERES spectral-scene file into an xarray
    Dataset with one "scene" per MODTRAN case - a specific SZA/VZA/RAZ/CF
    viewing geometry (see Loeb et al. 2001, Table 1, for the angular bins
    these files were built around; a copy of that paper and a readme
    describing the file formats sit alongside these files).

    Each file is many concatenated cases, each ending in a "-9999." sentinel
    line, in one of two column layouts (confirmed against the actual files,
    not just the readme's format-string description, which turned out not
    to exactly match these files' real column layout):
      - "ocean_snow" family (oceclr/ocecld/sno files): wavelength is given
        directly in um as the first column ("wav").
      - "land_dcc" family (lnd/dc files): the first column, FREQ, is a
        wavenumber in cm-1, so wavelength_um = 10000/FREQ.
    The family is detected from the file's own header text, not the
    filename.

    For every case, SW/TOT/WN radiance are all computed from the readme's
    combination formulas, regardless of whether that case was a daytime
    ("F" flag, "sz*" files - SW+TOT populated) or nighttime ("T" flag,
    "*vz*" files - TOT+WN populated) MODTRAN run. Only one of SW or WN is
    physically meaningful for a given case (a nighttime run has no real
    solar term to give a meaningful SW, and vice versa) - is_daytime is
    kept per scene rather than silently dropping the other channel.

    VZA is currently a best guess for the land_dcc family: a header value
    that varies in exactly the pattern (3 repeats per SZA/RAZ pair) needed
    to be VZA, but isn't explicitly labeled the way SZA/RAZ/CF are (which
    come from an explicit "sza = ... raz = ... cf = ..." line). Flagged as
    low priority to verify - Libera is developing a new spectral scene set
    this preliminary analysis will move to once available.

    Cloud fraction (cf) is given directly per case for land_dcc. For
    ocean_snow, cf isn't in the per-case header at all - it's implied by
    the file (oceclr=clear, cf=0; ocecld=cloudy, cf=1). Loeb et al. 2001
    describes linearly blending clear/overcast radiances for intermediate
    cloud fractions (0.25/0.5/0.75); that blending is not done here.

    Radiances are converted from their native MODTRAN units
    (W cm-2 sr-1 (cm-1)-1) to this pipeline's convention of
    W m-2 sr-1 um-1, via a factor of freq_cm-1^2 (the wavenumber-to-
    wavelength Jacobian and the cm-2->m-2 area conversion's 1e4's cancel).
    Verified against the independently-converted *_CONVERTED.csv files that
    sit one directory up from these .tp7 files, for WN (both families) and
    TOT (land, daytime): matches to 5-6 significant figures.

    Parameters
    ----------
    file : Path or str
        Path to a .tp7 file.
    max_cases : int, optional
        Stop after this many cases - for a first look at a huge file
        without reading the whole thing.

    Returns
    -------
    xarray.Dataset with dims (scene, wavelength_um): sw_radiance,
    tot_radiance, wn_radiance, and per-scene sza_deg/vza_deg/raz_deg/cf/
    is_daytime.
    """

    import re
    import xarray as xr

    file = Path(file)

    # Peek at the header to detect which of the two column layouts this
    # file uses (both families share the same leading F=day/T=night flag
    # line, so the filename isn't relied on for this).
    with open(file) as f:
        head = [f.readline() for _ in range(20)]
    if any("SOLZEN" in line for line in head):
        family = "ocean_snow"
    elif any(line.lstrip().startswith("sza") for line in head):
        family = "land_dcc"
    else:
        raise ValueError(f"Could not detect MODTRAN tape7 column layout for {file}")

    sza_list, vza_list, raz_list, cf_list, day_list = [], [], [], [], []
    sw_rows, tot_rows, wn_rows = [], [], []
    wl_um = None

    with open(file) as f:
        n_cases = 0
        line = f.readline()
        while line:
            stripped = line.strip()

            # Every case starts with a single-character MODTRAN driver flag:
            # F for a daytime run, T for nighttime.
            if stripped[:1] in ("F", "T") and len(stripped) > 1 and stripped[1].isspace():
                is_daytime = stripped[0] == "F"

                if family == "land_dcc":
                    # Fixed 11-line preamble (this flag line plus 10 more),
                    # ending in the "sza = ... raz = ... cf = ..." line. The
                    # VZA guess is the 2nd value on the 3rd line of the
                    # preamble (see docstring).
                    header = [line] + [f.readline() for _ in range(10)]
                    vza = float(header[2].split()[1])
                    m = re.search(r"sza\s*=\s*([-\d.]+)\s*raz\s*=\s*([-\d.]+)\s*cf\s*=\s*([-\d.]+)", header[10])
                    sza, raz, cf = (float(x) for x in m.groups())
                else:
                    # Fixed 12-line preamble, ending in the SOLZEN/VZSFC/
                    # RELAZ values line.
                    header = [line] + [f.readline() for _ in range(11)]
                    sza, vza, raz = (float(x) for x in header[11].split())
                    cf = 1.0 if "cld" in file.name else 0.0

                f.readline()  # column-name header row, not otherwise used

                # Read data rows until the "-9999." end-of-case sentinel.
                freq_vals = []
                sw_a, sw_b, sw_c = [], [], []
                tot_vals = []
                wn_a, wn_b, wn_c = [], [], []
                data_line = f.readline()
                while data_line.strip() != "-9999.":
                    vals = data_line.split()
                    if family == "land_dcc":
                        # cols: 0=FREQ 2=PTH_THRML 4=SURF_EMIS 5=SOL_SCAT
                        # 7=GRND_RFLT 9=TOTAL_RAD 13=thmsfcref
                        freq_vals.append(float(vals[0]))
                        sw_a.append(float(vals[5])); sw_b.append(float(vals[7])); sw_c.append(float(vals[13]))
                        tot_vals.append(float(vals[9]))
                        wn_a.append(float(vals[2])); wn_b.append(float(vals[4])); wn_c.append(float(vals[13]))
                    else:
                        # cols: 0=wav(um) 4=totrad 5=pathscat 6=sfctot
                        # 8=atmemis 9=sfcemis 10=thmsfcref
                        freq_vals.append(float(vals[0]))
                        sw_a.append(float(vals[5])); sw_b.append(float(vals[6])); sw_c.append(float(vals[10]))
                        tot_vals.append(float(vals[4]))
                        wn_a.append(float(vals[8])); wn_b.append(float(vals[9])); wn_c.append(float(vals[10]))
                    data_line = f.readline()

                freq_vals = np.array(freq_vals)
                sw = np.array(sw_a) + np.array(sw_b) - np.array(sw_c)
                tot = np.array(tot_vals)
                wn = np.array(wn_a) + np.array(wn_b) + np.array(wn_c)

                this_wl_um = (10000.0 / freq_vals) if family == "land_dcc" else freq_vals

                # Native MODTRAN units here are W cm-2 sr-1 (cm-1)-1 (per
                # wavenumber, per cm^2). Converting to this pipeline's
                # W m-2 sr-1 um-1 convention needs the wavenumber-to-
                # wavelength Jacobian |dv/dwl| = (freq_cm-1)^2/1e4 together
                # with the cm-2->m-2 area factor of 1e4 - the two 1e4's
                # cancel, leaving a clean multiply by freq_cm-1^2. Verified
                # empirically against the independently-converted CSVs one
                # directory up (*_CONVERTED.csv) for WN (both families) and
                # TOT (land, daytime): matches to 5-6 significant figures.
                freq_cm1 = 10000.0 / this_wl_um
                sw, tot, wn = sw * freq_cm1**2, tot * freq_cm1**2, wn * freq_cm1**2

                order = np.argsort(this_wl_um)
                this_wl_um, sw, tot, wn = this_wl_um[order], sw[order], tot[order], wn[order]

                if wl_um is None:
                    wl_um = this_wl_um

                sza_list.append(sza); vza_list.append(vza); raz_list.append(raz)
                cf_list.append(cf); day_list.append(is_daytime)
                sw_rows.append(sw); tot_rows.append(tot); wn_rows.append(wn)

                n_cases += 1
                if max_cases and n_cases >= max_cases:
                    break

            line = f.readline()

    return xr.Dataset(
        data_vars=dict(
            sw_radiance=(("scene", "wavelength_um"), np.array(sw_rows)),
            tot_radiance=(("scene", "wavelength_um"), np.array(tot_rows)),
            wn_radiance=(("scene", "wavelength_um"), np.array(wn_rows)),
            sza_deg=("scene", np.array(sza_list)),
            vza_deg=("scene", np.array(vza_list)),
            raz_deg=("scene", np.array(raz_list)),
            cf=("scene", np.array(cf_list)),
            is_daytime=("scene", np.array(day_list)),
        ),
        coords=dict(wavelength_um=wl_um),
        attrs=dict(
            source_file=str(file),
            family=family,
            radiance_units="W m-2 sr-1 um-1",
        ),
    )


def science_radiometer_estimate_sw_uncertainty(*, wavelength_um, radiance_w_m2_sr_um,
                                                scene_label="scene", make_plot=True):
    """
    Estimates the impact of SRF uncertainty on the filtered (SRF-weighted)
    radiance for one Earth scene spectrum, over 0.25-5um.

    Rather than averaging the SRF uncertainty itself (which doesn't
    normalize by what the channel actually measures), this computes the
    filtered radiance the channel would see - L = integral(E(wl)*SRF(wl) dwl)
    - and how much that shifts if the SRF is coherently perturbed by its own
    uncertainty everywhere at once: dL = integral(E(wl)*u_SRF(wl) dwl). The
    relative impact is dL/L. Since the integral is linear in SRF, the +/-
    perturbation is exactly symmetric (E(wl)*(SRF+/-u_SRF) integrates to
    L +/- dL), so only dL/L needs to be computed, not both signs separately.

    This intentionally treats u_SRF(wl) as one coherent shift across the
    whole integration range - the right treatment for the fully
    correlated-in-wavelength parts of u_total (the Common terms, and the
    smoothly-varying Wavelength-class terms), but somewhat conservative for
    the more locally-correlated Random component, whose errors are only
    correlated over the node spacing, not the whole range.
    Absent a full wavelength covariance matrix, this is an acceptable,
    appropriately-conservative approximation for a first cut.

    Limited to 0.25-5um for every channel, including Total - the SRF above
    5um still needs the blackbody measurements folded in (future work), so
    this understates Total's full-band filtered radiance, but still gives a
    meaningful estimate of its SW-range contribution to the uncertainty.

    The instrument only ever observes the *total* (LW+SW) outgoing radiance
    - each channel's own SRF is what limits the wavelength range it's
    actually sensitive to - so a scene's total radiance (e.g. tot_radiance
    from read_modtran_tp7_scenes, which is physically valid for both day and
    night cases - see that function's docstring) should be passed in here
    for every channel, rather than hand-picking a SW-only variant.

    Parameters
    ----------
    wavelength_um : array
        The scene spectrum's own wavelength grid - does not need to match
        the SRF's grid or be pre-sorted, both are handled here.
    radiance_w_m2_sr_um : array
        The scene's radiance, same length as wavelength_um, already in
        W m-2 sr-1 um-1 (see read_modtran_tp7_scenes for how CERES/MODTRAN
        scenes are converted to this convention).
    scene_label : str, default "scene"
        Identifies this scene in the returned table and in the saved
        filenames, so results from multiple scenes don't overwrite
        each other.
    make_plot : bool, default True
        Save the per-channel integrand diagnostic plot. Off by default
        would make sense when calling this over many scenes in a loop.

    Loads:
      libera_erf_srfs_04.csv        (per-channel SRF and total uncertainty)

    Summary output:
    Libera_erf_sw_uncertainty_estimate_{scene_label}.csv
    """

    paths = load_config()

    # Load the SRF + total uncertainty curves, trimmed to the range where
    # every SRF term has been fully characterized - blackbody measurements
    # above 5um haven't been incorporated into the SRF fits yet.
    filename = "libera_erf_srfs_04.csv"
    # Not passed through tidy_up_header - that lowercases column names, and
    # this file's SW/Total/LW/SSW capitalization is already exactly what
    # gets used as column names here.
    srf = pd.read_csv(paths.analysis_dir / filename)
    srf = srf.loc[srf["wavelength_um"] <= 5.0].sort_values("wavelength_um")
    wl = srf["wavelength_um"].values

    # Interpolate the scene spectrum onto the SRF's own wavelength grid.
    # Sorted first since np.interp requires monotonic x and MODTRAN scenes
    # can come back in either wavelength order depending on family (see
    # read_modtran_tp7_scenes).
    scene_order = np.argsort(wavelength_um)
    scene_wl = np.asarray(wavelength_um)[scene_order]
    scene_radiance = np.asarray(radiance_w_m2_sr_um)[scene_order]
    earth_radiance_interp = np.interp(wl, scene_wl, scene_radiance)

    # Note, the SSW is hard coded here. This is something the project
    # needs to come to consensus on and document.
    channels = ["SW", "Total", "SSW"]
    band_start_um = [0.25,0.25,0.7432]
    results = []
    for ch, band_lower_um in zip(channels, band_start_um):
        mask = wl > band_lower_um
        l_unfiltered = np.trapz(earth_radiance_interp*mask, wl)    
        l_filtered = np.trapz(earth_radiance_interp * srf[ch].values, wl)
        dl = np.trapz(earth_radiance_interp * srf[f"uc_{ch}"].values, wl)

        results.append({
            "scene": scene_label,
            "channel": ch,
            "l_filtered_w_m2_sr": l_filtered,
            "l_unfiltered_w_m2_sr": l_unfiltered,
            "dl_w_m2_sr": dl,
            "relative_impact_pct": 100 * dl / l_filtered,
        })

    results = pd.DataFrame(results)
    print(results.to_string(index=False))

    if make_plot:
        # Plot each channel's integrand (scene radiance x SRF uncertainty),
        # to show where in wavelength the uncertainty impact is
        # concentrated.
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#2ca02c', "#dc8c14"]
        for ch, color in zip(channels, colors):
            ax.plot(wl, earth_radiance_interp * srf[f"uc_{ch}"].values, color=color, linewidth=0.8, label=ch)
        log_x_axis_decimal(ax)
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('Scene radiance x SRF uncertainty [W m-2 sr-1 um-1]')
        ax.set_title(f'SRF uncertainty impact on filtered radiance: {scene_label}')
        ax.legend()
        fig.savefig(paths.figure_dir / f'erf_sw_uncertainty_estimate_{scene_label}.png', bbox_inches="tight", dpi=200)
        plt.close(fig)

    filename = f"Libera_erf_sw_uncertainty_estimate_{scene_label}.csv"
    results.to_csv(paths.analysis_dir / filename, index=False, float_format='%.6g')

    print("Done!")

    return results


if __name__ == "__main__":
    # This block only runs when you execute this file directly —
    # it's skipped when the file is imported from a notebook or another script.

    #tmp = estimate_grating_spectrum_wavelength_error()
    #tmp = science_radiometer_analyze_level_01()
    #tmp = science_radiometer_analyze_level_02()
    #tmp = science_radiometer_analyze_level_03()
    #analyze_ccs200_calibration_spectrum()

    #tmp = lst_analyze_level_01()
    tmp = lst_analyze_level_02()
