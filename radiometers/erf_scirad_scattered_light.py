import erf_analysis as erf
import libera_telescope_pst as pst
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from types import SimpleNamespace
from lmfit import Model
from libera_config import load_config

# Additive radiance noise floor [W m-2 sr-1] on a radially averaged point.
#
# The uncertainty on the averaged signal was originally built entirely out of
# terms proportional to the signal, so a point that averaged near zero got a
# near-zero error bar and then dominated the fit. This is the missing additive
# detector-noise term. It is measured, not assumed: see
# scirad_estimate_noise_floor, which derives it from the disagreement between
# the +yaw and -yaw halves of the same scan. The value here comes from the SSW
# channel, the one verified radially symmetric, so its half-to-half differences
# contain nothing but noise.
#
# Total Channel Noise: 0.00542 W m-2 sr-1
# SSW Channel Noise:   0.00434 W m-2 sr-1
# SW Channel Noise:    0.00943 W m-2 sr-1
SCIRAD_L_TOTAL_NOISE_FLOOR = 0.00542
SCIRAD_L_SSW_NOISE_FLOOR   = 0.00434
SCIRAD_L_SW_NOISE_FLOOR    = 0.00943

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

def pst_two_component_fit_fn(angle_deg, wavelength_um, a0, a1, a2, lp0, lp1):
    """
    Wrapper for fitting the two-component scatter model with lmfit, returns the
    total PST. See pst.pst_two_components for the parameter definitions.

    It is fit globally across all angles and wavelengths at once, with the
    wavelength dependence coming from the physics (f = sin(angle)/lambda and
    the 1/lambda^4 prefactor) rather than from polynomials in wavenumber.

    Use as:

        model = Model(pst_two_component_fit_fn,
                      independent_vars=['angle_deg', 'wavelength_um'])

    """

    return pst.pst_two_components(angle_deg, wavelength_um,
                                  a0, a1, a2, lp0, lp1).pst

def pst_three_component_fit_fn(angle_deg, wavelength_um, a0, a1, a2, lp0, lp1,
                               gc0, gw0, gw1, ga0, ga1):
    """
    Wrapper for fitting the three-component model with lmfit, returns the total
    PST. See pst.pst_three_components for the parameter definitions.

    This is the two-component model plus the SW annulus, which appears to come
    from diffraction off the front surface of the secondary back-reflecting off
    the fused silica filter. Use as:

        model = Model(pst_three_component_fit_fn,
                      independent_vars=['angle_deg', 'wavelength_um'])

    """

    return pst.pst_three_components(angle_deg, wavelength_um,
                                    a0, a1, a2, lp0, lp1,
                                    gc0, gw0, gw1, ga0, ga1).pst

def scirad_pst_two_component_fitting(*,
                       meas_data,
                       fit_curvature=True,
                       fit_particulate=True):
    """
    Global fit of the two-component scatter model to a whole channel's dataset.

    This is called once with *all* wavelengths and angles rather than one
    wavelength at a time, so the returned coefficients describe the whole
    dataset. Fitting globally is also what makes the
    uncertainty meaningful: the model parameters are correlated, and only a
    single fit carries the full covariance matrix that eval_uncertainty needs.

    Parameters
    ----------
    meas_data : DataFrame
        Needs rad_angle_deg, wavelength_um, signal and signal_sem columns.
    fit_curvature : bool, default True
        Fit the PSD curvature term a2. Set False for a pure power-law PSD.
    fit_particulate : bool, default True
        Fit the particulate term. Set False to fit surface scatter only, i.e.
        to reproduce a pure 1/lambda^4 model for comparison.

    Returns
    -------
    SimpleNamespace holding the fit coefficients, their 1-sigma uncertainties,
    the fit statistics, and `result` - the lmfit ModelResult itself, which is
    what you need for eval() and eval_uncertainty() on new angles/wavelengths.

    """

    model = Model(pst_two_component_fit_fn,
                  independent_vars=['angle_deg', 'wavelength_um'])

    # Starting values from a fit to the Total channel. a0 is ln(PSD) at the
    # pivot, a1 is the slope there (about -c_scat), lp0/lp1 the particulate
    # amplitude and angular slope.
    # params = model.make_params(
    #     a0=dict(value=-14.6, min=-40.0, max=0.0),
    #     a1=dict(value=-2.2, min=-10.0, max=5.0),
    #     a2=dict(value=-0.84, min=-5.0, max=5.0),
    #     lp0=dict(value=-9.8, min=-60.0, max=0.0),
    #     lp1=dict(value=-3.3, min=-8.0, max=0.0))

    params = model.make_params(
        a0=dict(value=-14.39, min=-40.0, max=0.0),
        a1=dict(value=-2.53, min=-10.0, max=5.0),
        a2=dict(value=-0.37, min=-5.0, max=5.0),
        lp0=dict(value=-10.58, min=-60.0, max=0.0),
        lp1=dict(value=-3.14, min=-20.0, max=0.0))
    
    if not fit_curvature:
        params['a2'].set(value=0.0, vary=False)

    if not fit_particulate:
        # exp(-60) is ~1e-26, i.e. switched off but still finite
        params['lp0'].set(value=-60.0, vary=False)
        params['lp1'].set(value=-20.0, vary=False)

    result = model.fit(meas_data["signal"], params,
                       angle_deg=meas_data["rad_angle_deg"],
                       wavelength_um=meas_data["wavelength_um"],
                       weights=1.0/meas_data["signal_uc"])

    # Split the best-fit model into its components at the measured points, so
    # the surface/particulate balance can be inspected per wavelength.
    parts = pst.pst_two_components(meas_data["rad_angle_deg"].values,
                                   meas_data["wavelength_um"].values,
                                   **result.best_values)

    return SimpleNamespace(
        a0=result.params['a0'].value,
        a0_sd=result.params['a0'].stderr,
        a1=result.params['a1'].value,
        a1_sd=result.params['a1'].stderr,
        a2=result.params['a2'].value,
        a2_sd=result.params['a2'].stderr,
        lp0=result.params['lp0'].value,
        lp0_sd=result.params['lp0'].stderr,
        lp1=result.params['lp1'].value,
        lp1_sd=result.params['lp1'].stderr,
        chisqr=result.chisqr,
        redchi=result.redchi,
        aic=result.aic,
        nfree=result.nfree,
        surf=parts.surf,
        part=parts.part,
        f=parts.f,
        result=result)

def scirad_pst_three_component_fitting(*,
                       meas_data,
                       fit_curvature=True,
                       fit_particulate=True,
                       fit_gaussian=True):
    """
    Global fit of the three-component model to a whole channel's dataset.

    Same idea as scirad_pst_two_component_fitting - one fit across all angles
    and wavelengths so the covariance is meaningful - with the SW annulus added.
    The ring is what this exists for: in the SW the two-component model
    under-predicts by up to a factor of ~10 near 4-6 degrees at the shortest
    wavelengths, in a bump that moves outward with wavelength exactly as
    asin(wavelength/gc0) does.

    All three ring properties vary with wavelength: the centre follows
    first-order diffraction off the secondary, and the width and amplitude are
    log-linear in wavelength so neither can reach zero or go negative.

    Parameters
    ----------
    meas_data : DataFrame
        Needs rad_angle_deg, wavelength_um, signal and signal_uc columns.
    fit_curvature : bool, default True
        Fit the PSD curvature term a2. Set False for a pure power-law PSD.
    fit_particulate : bool, default True
        Fit the particulate term. Set False for surface scatter only.
    fit_gaussian : bool, default True
        Fit the SW annulus. Set False to switch the ring off entirely, which
        reduces this to the two-component model and gives the like-for-like
        comparison that shows how much the ring is worth (on the SW channel it
        is worth a drop in reduced chi-square from 8.29 to 3.03).

    Returns
    -------
    SimpleNamespace holding the fit coefficients, their 1-sigma uncertainties,
    the fit statistics, and `result` - the lmfit ModelResult itself, which is
    what you need for eval() and eval_uncertainty() on new angles/wavelengths.

    """

    model = Model(pst_three_component_fit_fn,
                  independent_vars=['angle_deg', 'wavelength_um'])

    # Scatter starting values from a two-component fit to the SW channel; ring
    # starting values from the residual of that fit (the excess sits at
    # asin(lam/5.41) and dies off by ~0.7 um).
    params = model.make_params(
        a0=dict(value=-14.51, min=-40.0, max=0.0),
        a1=dict(value=-2.37, min=-10.0, max=5.0),
        a2=dict(value=-0.57, min=-5.0, max=5.0),
        lp0=dict(value=-9.46, min=-60.0, max=0.0),
        lp1=dict(value=-2.45, min=-20.0, max=0.0),
        gc0=dict(value=5.410, min=1.0, max=50.0),
        gw0=dict(value=-1.06, min=-5.0, max=5.0),
        gw1=dict(value=1.33, min=-5.0, max=20.0),
        ga0=dict(value=-6.50, min=-30.0, max=5.0),
        ga1=dict(value=-15.90, min=-60.0, max=10.0))

    if not fit_curvature:
        params['a2'].set(value=0.0, vary=False)

    if not fit_particulate:
        # exp(-60) is ~1e-26, i.e. switched off but still finite
        params['lp0'].set(value=-60.0, vary=False)
        params['lp1'].set(value=-20.0, vary=False)

    if not fit_gaussian:
        # exp(-60) is ~1e-26, i.e. the ring is switched off but still finite.
        # The other four ring parameters have nothing to constrain them once the
        # amplitude is zero, so they are held too rather than left to wander.
        params['ga0'].set(value=-60.0, vary=False)
        params['ga1'].set(value=0.0, vary=False)
        params['gc0'].set(vary=False)
        params['gw0'].set(vary=False)
        params['gw1'].set(vary=False)

    result = model.fit(meas_data["signal"], params,
                       angle_deg=meas_data["rad_angle_deg"],
                       wavelength_um=meas_data["wavelength_um"],
                       weights=1.0/meas_data["signal_uc"])

    # Split the best-fit model into its components at the measured points, so
    # the surface / particulate / ring balance can be inspected per wavelength.
    parts = pst.pst_three_components(meas_data["rad_angle_deg"].values,
                                     meas_data["wavelength_um"].values,
                                     **result.best_values)

    out = SimpleNamespace(
        chisqr=result.chisqr,
        redchi=result.redchi,
        aic=result.aic,
        nfree=result.nfree,
        surf=parts.surf,
        part=parts.part,
        ring=parts.ring,
        f=parts.f,
        result=result)

    # Same value/stderr pairs as the two-component version, for all ten params.
    for name in ('a0', 'a1', 'a2', 'lp0', 'lp1',
                 'gc0', 'gw0', 'gw1', 'ga0', 'ga1'):
        setattr(out, name, result.params[name].value)
        setattr(out, name + '_sd', result.params[name].stderr)

    return out

# def single_scirad_erf_scattered_light_get_points(row):
#     """
#     Reduce one log file to individual measurement points, before averaging.

#     Returns every surviving point with its manipulator pitch and yaw intact, so
#     the azimuth of each measurement is recoverable - the radial average that
#     single_scirad_erf_scattered_light_get_measurement performs throws that away.
#     Useful for asking whether a feature is a true annulus (present at every
#     azimuth for a given radial angle) or concentrated at particular positions.

#     The peak normalisation and the 1.8 deg / yaw > 1.1 cuts are already applied,
#     so meas_data holds exactly the population that feeds the radial averages.

#     Returns
#     -------
#     SimpleNamespace with meas_data plus the scalars the averaging step needs
#     (center_l, center_l_sd, wavelength_um, detector_index, det_name, log_file).

#     """

#     from scipy.interpolate import interp1d
#     import statistics

#     paths = load_config()

#     # Initially just use the OPA wavelength, ovewrite if there is a spectrum file
#     wavelength_um = row.opa_wavelength_nm/1000
#     wavelength_um_fwhm = 0
#     wavelength_source = 10

#     # Make sure there's a spectrum filename present
#     if pd.notna(row.spectrum_file):

#         # Analyze a CCS spectrum
#         if "ccs200" in row.spectrum_file.lower():
#             #print(f"CCS200: {row.spectrum_file}")
#             wl_fit = erf.analyze_ccs200_spectrum(file=row.spectrum_file, wavelength_um_opa=wavelength_um)
#             wavelength_um = wl_fit.wavelength_center_um
#             wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
#             wavelength_source = 1

#         # Analyze a OSA spectrum
#         if "osa207" in row.spectrum_file.lower():
#             #print(f"OSA207: {row.spectrum_file}")
#             wl_fit = erf.analyze_osa207_spectrum(file=row.spectrum_file, wavelength_um_opa=wavelength_um)
#             wavelength_um = wl_fit.wavelength_center_um
#             wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
#             wavelength_source = 2
    
#     # Read in the data
#     df = erf.load_erf_logfile(file=row.log_file, wavelength_um=wavelength_um)

#     # Strip off the file extension so we can use the name to save the plot
#     log_file = row.log_file[0:-4]

#     # Loop through each active measurement
#     meas_data = []
#     for i in range(df["measurement_index"].max() + 1):

#         # Get the data with the current measurement index
#         df_tmp = df.loc[(df["measurement_index"] == i)].copy()

#         # Get the number of open and closed points
#         n_open   = ((df['measurement_index'] == i) & (df['shutter_a'] == 1)).sum()
#         n_closed = ((df['measurement_index'] == i) & (df['shutter_a'] == 0)).sum()

#         # Get the current detector index
#         detector_index = df_tmp["detector_index"].median()

#         # The the number of data points, analyze if there are more than 20
#         npts = len(df_tmp)
#         # print(f"Loop {i}, Npts {npts}, Npts Open {n_open}, Npts Closed {n_closed}")

#         # Case for the science radiometers
#         if (n_open >= 5) and (n_closed >= 5) and (detector_index <= 4):

#             match detector_index:
#                 case 0:
#                     # PBR-R
#                     l_col = "pbr_l"
#                     han_filter = 1
#                     det_name = 'PBR-R'
#                 case 1:
#                     # SSW
#                     l_col = "l3_ss"
#                     han_filter = 0
#                     det_name = 'SSW'
#                 case 2:
#                     # LW
#                     l_col = "l2_lw"
#                     han_filter = 0
#                     det_name = 'LW'
#                 case 3:
#                     # Total
#                     l_col = "l1_to"
#                     han_filter = 0
#                     det_name = 'Total'
#                 case 4:
#                     # SW
#                     l_col = "l0_sw"
#                     han_filter = 0
#                     det_name = 'SW'
#                 case _:  
#                     print('Bad Case!')
            
#             # Extract out as a plain numpy array
#             shutter = df_tmp["shutter_a"].values

#             # --- transitions: where shutter_a differs from the previous point (circular) ---
#             transitions = shutter != np.roll(shutter, 1)
#             itmp = np.where(transitions)[0]

#             # --- median spacing between transitions (also circular difference) ---
#             # half_cycle_pts = np.median(itmp - np.roll(itmp, 1))

#             # Number of points per half-cycle
#             half_cycle_pts = statistics.mode(itmp[1:] - itmp[:-1])

#             # --- round to nearest multiple of 4 ---
#             half_cycle_pts = 4 * round(half_cycle_pts / 4.0)

#             # --- cumulative count of transitions -> increments each half-cycle ---
#             df_tmp["measurement_index"] = np.cumsum(transitions)

#             # --- second half of the shutter cycle ---
#             navg_pts = int(np.floor(half_cycle_pts / 2.0))

#             # Create the hanning window
#             if han_filter:
#                 avg_kernel = np.hanning(navg_pts + 2)[1:-1]
#             else:
#                 avg_kernel = np.ones(navg_pts)

#             # Normalize avg_kernel
#             avg_kernel = avg_kernel/avg_kernel.sum()

#             # Loop through each active measurement
#             dc_sub_data = []
#             for j in range(df_tmp["measurement_index"].max() + 1):
#                 df_tmp2 = df_tmp.loc[(df_tmp["measurement_index"] == j)].copy()
#                 df_tmp2 = df_tmp2[-navg_pts:]

#                 dc_sub_data.append({
#                     "time":     (df_tmp2["gps_time_s"]*avg_kernel).sum(),
#                     "l":        (df_tmp2[l_col]*avg_kernel).sum(),
#                     "l_dev":     df_tmp2[l_col].std(),
#                     "shutter":   df_tmp2["shutter_a"].median(),
#                     "monitor":  (df_tmp2["monitor_a"]*avg_kernel).sum(),
#                 })

#             # Change to a dataframe
#             dc_sub_data = pd.DataFrame(dc_sub_data)

#             # Get the time array. Then throw out the first and last points so that
#             # every point is bracketed
#             time = dc_sub_data["time"].iloc[1:-1].values

#             # Subset the open and closed data
#             closed = dc_sub_data[dc_sub_data["shutter"] == 0]
#             open_  = dc_sub_data[dc_sub_data["shutter"] == 1]

#             # Make sure each open and closed cycle in "time" is bracketed
#             for name, subset in [("closed", closed), ("open", open_)]:
#                 subset_time = subset["time"].values
#                 if time.min() < subset_time.min() or time.max() > subset_time.max():
#                     raise ValueError(
#                         f"Interpolation times fall outside the {name} shutter time range: "
#                         f"time range [{time.min():.3f}, {time.max():.3f}] vs "
#                         f"{name} range [{subset_time.min():.3f}, {subset_time.max():.3f}]"
#             )

#             # Get the light and dark data interpolated to those times
#             est_dark_radiance     = np.interp(time, closed["time"].values, closed["l"].values)
#             est_dark_radiance_dev = np.interp(time, closed["time"].values, closed["l_dev"].values)

#             est_open_radiance     = np.interp(time, open_["time"].values, open_["l"].values)
#             est_open_radiance_dev = np.interp(time, open_["time"].values, open_["l_dev"].values)

#             monitor               = np.interp(time, dc_sub_data["time"].values, dc_sub_data["monitor"].values)

#             # Calculate the DC subtraction
#             dc_sub_radiance     = est_dark_radiance - est_open_radiance
#             dc_sub_radiance_dev = np.sqrt(est_dark_radiance_dev**2 + est_open_radiance_dev**2)

#             npts = len(dc_sub_radiance)
#             for j in range(npts):
#                 meas_data.append({
#                         "log_file":             row.log_file,
#                         "gps_time_s":           time[j],
#                         "type":                 row.type,
#                         "photodiode":           row.photodiode,
#                         "detector_index":       detector_index,
#                         "l":                    dc_sub_radiance[j],
#                         "l_sd":                 dc_sub_radiance_dev[j],
#                         "monitor":              monitor[j],
#                         "manip_x_mm":           df_tmp['manip_x_mm'].median(),
#                         "manip_y_mm":           df_tmp['manip_y_mm'].median(),
#                         "manip_pitch_deg":      df_tmp['manip_pitch_deg'].median(),
#                         "manip_yaw_deg":        df_tmp['manip_yaw_deg'].median(),
#                         "wavelength_um":        wavelength_um,
#                         "opa_wavelength_um":    df_tmp['opa_wavelength_um'].median(),
#                     })

#         # Case for the EM radiometer, some with the photodiode
#         if (n_open >= 7) and (n_closed >= 15) and (detector_index == 5):
    
#             det_name = 'EM'

#             closed = df_tmp.loc[(df_tmp["shutter_a"] == 0)].copy()
#             open_  = df_tmp.loc[(df_tmp["shutter_a"] == 1)].copy()

#             signal = open_["em_pwm_dn"].median() - closed["em_pwm_dn"].median() 

#             monitor = open_["monitor_a"].median()
                
#             meas_data.append({
#                 "log_file":             row.log_file,
#                 "gps_time_s":           open_["gps_time_s"].median(),
#                 "type":                 row.type,
#                 "photodiode":           row.photodiode,
#                 "detector_index":       detector_index,
#                 "l":                    signal,
#                 "l_sd":                 open_["em_pwm_dn"].std,
#                 "monitor":              monitor,
#                 "manip_x_mm":           df_tmp['manip_x_mm'].median(),
#                 "manip_y_mm":           df_tmp['manip_y_mm'].median(),
#                 "manip_pitch_deg":      df_tmp['manip_pitch_deg'].median(),
#                 "manip_yaw_deg":        df_tmp['manip_yaw_deg'].median(),
#                 "wavelength_um":        wavelength_um,
#                 "opa_wavelength_um":    df_tmp['opa_wavelength_um'].median(),
#             })

#     # Convert to a pandas dataframe
#     meas_data = pd.DataFrame(meas_data)

#     # Calculate the radial angle
#     meas_data['rad_angle_deg'] = np.sqrt(meas_data['manip_pitch_deg']**2 + meas_data['manip_yaw_deg']**2)
    
#     # Get the signal at the center, use this to peak-normalize the signal
#     center_data = meas_data.loc[(meas_data["rad_angle_deg"] < 0.25)].copy()
#     center_l    = center_data['l'].mean()
#     center_l_sd = center_data['l'].std()

#     # Peak normalize the radiance data
#     meas_data['signal'] = meas_data['l']/center_l

#     # Carry the centre radiance and its scatter on every row. They are constant
#     # per file, but keeping them as columns is what lets the averaging step -
#     # and anything reading the saved per-point CSV - reconstruct signal_uc
#     # without going back to the log file.
#     meas_data['l_center']    = center_l
#     meas_data['l_center_sd'] = center_l_sd

#     # Match the averaged output, which stores the log file without its
#     # extension so the name can be used directly for the per-file plot.
#     meas_data['log_file'] = log_file

#     # Detector name, carried so the averaging step can label its plots
#     meas_data['det_name'] = det_name

#     # The ERF aperture sizes
#     # OPA Focal Length = 165.43mm
#     #
#     #  9.920mm = 1.717˚ Half-Angle
#     # 11.947mm = 2.068˚ Half-Angle
#     # 13.625mm = 2.358˚ Half-Angle
#     #
#     # Fit only the data outside of 1.8 degrees
#     meas_data = meas_data.loc[(meas_data["rad_angle_deg"] >= 1.8)]

#     # Throw out the tips of the hexagon up and down in pitch. The cut is on the
#     # magnitude: the field of view is symmetric about zero yaw, so a one-sided
#     # yaw > 1.1 would also have discarded the whole negative-yaw half of every
#     # 2-D scan, which is real data and roughly doubles the azimuth coverage.
#     meas_data = meas_data.loc[(meas_data["manip_yaw_deg"].abs() > 1.1)]

#     # Everything above is the per-point reduction; everything below averages it
#     # down to one row per radial angle. Split here so the individual points -
#     # which still carry manip_pitch_deg / manip_yaw_deg, and so the azimuth the
#     # radial angle was measured at - can be had without redoing the work.
#     return SimpleNamespace(meas_data=meas_data,
#                            center_l=center_l,
#                            center_l_sd=center_l_sd,
#                            wavelength_um=wavelength_um,
#                            detector_index=detector_index,
#                            det_name=det_name,
#                            log_file=log_file,
#                            row=row)


# Geometric field of view: an elongated hexagon matching the CERES footprint,
# 1.3 deg across by 2.6 deg long, with the long axis along manipulator pitch.
# Its area, 2.535 deg^2, agrees with the solid_angle in the conversions table
# (2.527 deg^2) to 0.3%, which is what fixes the flat section at half the length.
FOV_WIDTH_DEG  = 1.3     # across the short axis, along yaw
FOV_LENGTH_DEG = 2.6     # along the long axis, along pitch
FOV_KEEPOUT_DEG = 1.0    # exclude points closer than this to the hexagon

def geometric_fov_hexagon(width_deg=FOV_WIDTH_DEG, length_deg=FOV_LENGTH_DEG):
    """
    Vertices of the field-of-view hexagon as (pitch, yaw) pairs [deg].

    Counter-clockwise, starting from the bottom of the flat central section. The
    flat section runs half the total length, which is what makes the area match
    the tabulated solid angle; beyond it the hexagon tapers to a point at each
    end of the long axis.

    """

    half_width  = width_deg/2.0
    half_length = length_deg/2.0
    flat        = half_length/2.0

    return np.array([[-flat,       -half_width],
                     [ flat,       -half_width],
                     [ half_length,  0.0],
                     [ flat,        half_width],
                     [-flat,        half_width],
                     [-half_length,  0.0]])


def geometric_fov_distance_deg(pitch_deg, yaw_deg,
                               width_deg=FOV_WIDTH_DEG,
                               length_deg=FOV_LENGTH_DEG):
    """
    Distance from each (pitch, yaw) to the field-of-view hexagon [deg].

    Zero for points inside the hexagon, otherwise the shortest distance to its
    boundary. Distance to the polygon rather than to each edge's infinite line,
    so that points off the ends are measured to the nearest vertex rather than
    to an edge extended out to meet them.

    """

    vertex = geometric_fov_hexagon(width_deg, length_deg)

    point = np.column_stack([np.asarray(pitch_deg, dtype=float).ravel(),
                             np.asarray(yaw_deg,   dtype=float).ravel()])

    inside   = np.ones(len(point), dtype=bool)
    distance = np.full(len(point), np.inf)

    for i in range(len(vertex)):
        a = vertex[i]
        b = vertex[(i + 1) % len(vertex)]
        edge = b - a

        # Closest point on this edge *segment*, clamped to its ends
        t = np.clip(((point - a) @ edge)/(edge @ edge), 0.0, 1.0)
        closest = a + t[:, None]*edge
        distance = np.minimum(distance, np.hypot(*(point - closest).T))

        # Convex polygon wound counter-clockwise: inside means left of every edge
        inside &= (edge[0]*(point[:, 1] - a[1]) - edge[1]*(point[:, 0] - a[0])) >= 0.0

    return np.where(inside, 0.0, distance).reshape(np.shape(pitch_deg))


def apply_geometric_fov_keepout(meas_data, keepout_deg=FOV_KEEPOUT_DEG,
                                width_deg=FOV_WIDTH_DEG,
                                length_deg=FOV_LENGTH_DEG):
    """
    Drop points lying within keepout_deg of the geometric field of view.

    Replaces the old pair of cuts - a radial angle floor plus a yaw magnitude cut
    - which together carved a circle and a stripe out of a field that is neither.
    Cutting on distance to the actual hexagon keeps considerably more data near
    the field of view, which is where the 2-D maps are most useful, while still
    excluding everything the source itself illuminates.

    The distance is measured to the hexagon boundary, so the keepout follows the
    field of view's shape.

    """

    distance = geometric_fov_distance_deg(meas_data["manip_pitch_deg"].values,
                                          meas_data["manip_yaw_deg"].values,
                                          width_deg=width_deg,
                                          length_deg=length_deg)

    return meas_data.loc[distance > keepout_deg]


def single_scirad_erf_scattered_light_get_and_average_points(row):
    """
    Reduce one log file to averaged measurement points per pitch/yaw position.
    The data was taken such that typically there were at least two shutter cycles
    per position.

    The peak normalisation and the 1.8 deg / yaw > 1.1 cuts are already applied,
    so meas_data holds exactly the population that feeds the radial averages.

    Returns
    -------
    SimpleNamespace with meas_data plus the scalars the averaging step needs
    (center_l, center_l_sd, wavelength_um, detector_index, det_name, log_file).

    """

    from scipy.interpolate import interp1d
    import statistics

    paths = load_config()

    # Initially just use the OPA wavelength, ovewrite if there is a spectrum file
    wavelength_um = row.opa_wavelength_nm/1000
    wavelength_um_fwhm = 0
    wavelength_source = 10

    # Make sure there's a spectrum filename present
    if pd.notna(row.spectrum_file):

        # Analyze a CCS spectrum
        if "ccs200" in row.spectrum_file.lower():
            #print(f"CCS200: {row.spectrum_file}")
            wl_fit = erf.analyze_ccs200_spectrum(file=row.spectrum_file, wavelength_um_opa=wavelength_um)
            wavelength_um = wl_fit.wavelength_center_um
            wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
            wavelength_source = 1

        # Analyze a OSA spectrum
        if "osa207" in row.spectrum_file.lower():
            #print(f"OSA207: {row.spectrum_file}")
            wl_fit = erf.analyze_osa207_spectrum(file=row.spectrum_file, wavelength_um_opa=wavelength_um)
            wavelength_um = wl_fit.wavelength_center_um
            wavelength_um_fwhm = wl_fit.wavelength_fwhm_um
            wavelength_source = 2
    
    # Read in the data
    df, pbr_conv = erf.load_erf_logfile(file=row.log_file, wavelength_um=wavelength_um)

    # Strip off the file extension so we can use the name to save the plot
    log_file = row.log_file[0:-4]

    # Loop through each active measurement
    meas_data = []
    for i in range(df["measurement_index"].max() + 1):

        # Get the data with the current measurement index
        df_tmp = df.loc[(df["measurement_index"] == i)].copy()

        # Get the number of open and closed points
        n_open   = ((df['measurement_index'] == i) & (df['shutter_a'] == 1)).sum()
        n_closed = ((df['measurement_index'] == i) & (df['shutter_a'] == 0)).sum()

        # Get the current detector index
        detector_index = df_tmp["detector_index"].median()

        # The the number of data points, analyze if there are more than 20
        npts = len(df_tmp)
        # print(f"Loop {i}, Npts {npts}, Npts Open {n_open}, Npts Closed {n_closed}")

        # Case for the science radiometers
        if (n_open >= 5) and (n_closed >= 5) and (detector_index <= 4):

            match detector_index:
                case 0:
                    # PBR-R
                    l_col = "pbr_l"
                    han_filter = 1
                    det_name = 'PBR-R'
                case 1:
                    # SSW
                    l_col = "l3_ss"
                    han_filter = 0
                    det_name = 'SSW'
                    noise_floor = SCIRAD_L_SSW_NOISE_FLOOR
                case 2:
                    # LW
                    l_col = "l2_lw"
                    han_filter = 0
                    det_name = 'LW'
                    noise_floor = SCIRAD_L_TOTAL_NOISE_FLOOR
                case 3:
                    # Total
                    l_col = "l1_to"
                    han_filter = 0
                    det_name = 'Total'
                    noise_floor = SCIRAD_L_TOTAL_NOISE_FLOOR
                case 4:
                    # SW
                    l_col = "l0_sw"
                    han_filter = 0
                    det_name = 'SW'
                    noise_floor = SCIRAD_L_SW_NOISE_FLOOR
                case _:  
                    print('Bad Case!')
            
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

            # Throw out the first 3 data points
            navg_pts = half_cycle_pts - 3

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
                    "l_dev":     df_tmp2[l_col].std(),
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
            est_dark_radiance     = np.interp(time, closed["time"].values, closed["l"].values)
            est_dark_radiance_dev = np.interp(time, closed["time"].values, closed["l_dev"].values)

            est_open_radiance     = np.interp(time, open_["time"].values, open_["l"].values)
            est_open_radiance_dev = np.interp(time, open_["time"].values, open_["l_dev"].values)

            monitor               = np.interp(time, dc_sub_data["time"].values, dc_sub_data["monitor"].values)

            # Calculate the DC subtraction
            dc_sub_radiance     = est_dark_radiance - est_open_radiance
            #dc_sub_radiance_dev = np.sqrt(est_dark_radiance_dev**2 + est_open_radiance_dev**2)

            meas_data.append({
                    "log_file":             row.log_file,
                    "gps_time_s":           np.mean(time),
                    "type":                 row.type,
                    "photodiode":           row.photodiode,
                    "detector_index":       detector_index,
                    "l":                    np.mean(dc_sub_radiance),
                    "l_sd":                 np.std(dc_sub_radiance),
                    "npts":                 np.floor(len(dc_sub_radiance)/2.0),
                    "monitor":              np.mean(monitor),
                    "manip_x_mm":           df_tmp['manip_x_mm'].median(),
                    "manip_y_mm":           df_tmp['manip_y_mm'].median(),
                    "manip_pitch_deg":      df_tmp['manip_pitch_deg'].median(),
                    "manip_yaw_deg":        df_tmp['manip_yaw_deg'].median(),
                    "wavelength_um":        wavelength_um,
                    "opa_wavelength_um":    df_tmp['opa_wavelength_um'].median(),
                })

        # Case for the EM radiometer, some with the photodiode
        if (n_open >= 7) and (n_closed >= 15) and (detector_index == 5):
    
            det_name = 'EM'

            closed = df_tmp.loc[(df_tmp["shutter_a"] == 0)].copy()
            open_  = df_tmp.loc[(df_tmp["shutter_a"] == 1)].copy()

            signal = open_["em_pwm_dn"].median() - closed["em_pwm_dn"].median() 

            monitor = open_["monitor_a"].median()
                
            meas_data.append({
                "log_file":             row.log_file,
                "gps_time_s":           open_["gps_time_s"].median(),
                "type":                 row.type,
                "photodiode":           row.photodiode,
                "detector_index":       detector_index,
                "l":                    signal,
                "l_sd":                 open_["em_pwm_dn"].std,
                "monitor":              monitor,
                "manip_x_mm":           df_tmp['manip_x_mm'].median(),
                "manip_y_mm":           df_tmp['manip_y_mm'].median(),
                "manip_pitch_deg":      df_tmp['manip_pitch_deg'].median(),
                "manip_yaw_deg":        df_tmp['manip_yaw_deg'].median(),
                "wavelength_um":        wavelength_um,
                "opa_wavelength_um":    df_tmp['opa_wavelength_um'].median(),
            })

    # Convert to a pandas dataframe
    meas_data = pd.DataFrame(meas_data)

    # Calculate the radial angle
    meas_data['rad_angle_deg'] = np.sqrt(meas_data['manip_pitch_deg']**2 + meas_data['manip_yaw_deg']**2)
    
    # Get the signal at the center, use this to peak-normalize the signal
    center_data = meas_data.loc[(meas_data["rad_angle_deg"] < 0.25)].copy()
    peak_l_data = erf.weighted_mean_and_stddev(x=center_data['l'], sd=center_data['l_sd'])
    center_l    = peak_l_data.x_mn
    center_l_sd = peak_l_data.x_sd

    # Peak normalize the radiance data
    meas_data['signal'] = meas_data['l']/center_l

    # Carry the center radiance and its scatter on every row. They are constant
    # per file, but keeping them as columns is what lets the averaging step -
    # and anything reading the saved per-point CSV - reconstruct signal_uc
    # without going back to the log file.
    meas_data['l_center']    = center_l
    meas_data['l_center_sd'] = center_l_sd

    # Match the averaged output, which stores the log file without its
    # extension so the name can be used directly for the per-file plot.
    meas_data['log_file'] = log_file

    # Detector name, carried so the averaging step can label its plots
    meas_data['det_name'] = det_name

    # Estimate the uncertainty in the data
    # 
    # Components:
    #   -Uncertainty in peak radiance
    #   -10% measurement uncertainty
    #   -The radiance noise floor of the channel
    meas_data['signal_uc'] = np.sqrt((meas_data['signal']*center_l_sd/center_l)**2 + \
                                     (0.10*meas_data['signal'])**2 + \
                                     (noise_floor/center_l)**2)

    # The ERF aperture sizes
    # OPA Focal Length = 165.43mm
    #
    #  9.920mm = 1.717˚ Half-Angle
    # 11.947mm = 2.068˚ Half-Angle
    # 13.625mm = 2.358˚ Half-Angle
    #
    # Exclude everything within FOV_KEEPOUT_DEG of the geometric field of view,
    # measured as a distance to the 1.3 x 2.6 deg hexagon itself rather than as a
    # radial angle floor plus a yaw stripe. Following the actual shape keeps the
    # data in as close as ~1.65 deg across the narrow axis, where the old cuts
    # kept nothing inside 1.8 deg, while still removing everything the source
    # directly illuminates.
    meas_data = apply_geometric_fov_keepout(meas_data)

    # Everything above is the per-point reduction; everything below averages it
    # down to one row per radial angle. Split here so the individual points -
    # which still carry manip_pitch_deg / manip_yaw_deg, and so the azimuth the
    # radial angle was measured at - can be had without redoing the work.
    return SimpleNamespace(meas_data=meas_data,
                           center_l=center_l,
                           center_l_sd=center_l_sd,
                           wavelength_um=wavelength_um,
                           detector_index=detector_index,
                           det_name=det_name,
                           log_file=log_file,
                           row=row)


def apply_pitch_cut(meas_data, pitch_max, wavelength_um=None):
    """
    Drop points above a maximum manipulator pitch.

    The SW channel carries a feature on the +pitch side of the field that the
    radially symmetric model cannot represent - see the radial symmetry section
    of the notebook. Because the radial average blends every azimuth at a given
    angle, that feature contaminates the averaged points rather than showing up
    as an obvious outlier, so it has to be cut before averaging.

    Parameters
    ----------
    pitch_max : float, callable, or None
        None keeps everything. A number keeps ``manip_pitch_deg <= pitch_max``.
        A callable is passed the wavelength [um] and returns the limit for that
        wavelength, which is what the SW cut needs: the feature is negligible
        below ~1 um, where the pitch = 0 1-D scans are still usable, but not
        above it. For example

            pitch_max=lambda wl: 0.0 if wl <= 1.0 else -0.5

        keeps pitch <= 0 in the short wave and strictly negative pitch above
        1 um. Note that the second case discards every 1-D scan, since those sit
        at pitch = 0.
    wavelength_um : float, optional
        Wavelength to hand a callable ``pitch_max``. Omit it when meas_data
        spans several files and the per-row ``wavelength_um`` column should be
        used instead, which is the case when cutting a whole channel at once.

    """

    if pitch_max is None:
        return meas_data

    if callable(pitch_max):
        if wavelength_um is None:
            wavelength_um = meas_data["wavelength_um"].values
        limit = np.array([pitch_max(w) for w in np.atleast_1d(wavelength_um)])
        if limit.size == 1:
            limit = limit[0]
    else:
        limit = pitch_max

    return meas_data.loc[meas_data["manip_pitch_deg"].values <= limit]

# Manipulator region occupied by the SW +yaw feature ("the blob"), and the
# wavelength above which it is detectable. Measured, not guessed: the +yaw/-yaw
# signal ratio at matched |yaw| is 1.0 outside these bounds and rises to ~2.9
# inside them by 2.175 um. See the blob section of the notebook.
SW_BLOB_YAW_MIN    = 2.0    # deg
SW_BLOB_YAW_MAX    = 6.0    # deg
SW_BLOB_WL_MIN_UM  = 0.5    # um

def apply_blob_mask(meas_data,
                    yaw_min=SW_BLOB_YAW_MIN,
                    yaw_max=SW_BLOB_YAW_MAX,
                    wavelength_um_min=SW_BLOB_WL_MIN_UM):
    """
    Drop the manipulator region occupied by the SW +yaw feature.

    The SW channel carries a fixed-position excess centred near pitch +2, yaw
    +4 deg, whose brightness grows as roughly lambda^1.4 and which is absent
    below ~0.5 um. Its origin could not be established - distinguishing an
    instrument ghost from a facility reflection would have needed a scan with
    the instrument rotated relative to the chamber, which is no longer possible
    - so it is treated as an artifact of the calibration setup and removed
    before fitting, and its integrated contribution is instead carried as an
    uncertainty on the ERF correction (see scirad_blob_excess_solid_angle).

    The cut is on +yaw at all pitch rather than on positive pitch. The feature
    is strongest at +pitch, reaching 2.9x the mirrored signal, but it is still
    1.3-1.4x at pitch 0 and -1, so a pitch-only cut leaves a residue. Cutting
    the +yaw band instead removes it fully while discarding 16.5% of the points
    rather than the 31% a pitch cut needed, and no radial angle is lost - the
    -yaw half still covers every one.

    Parameters
    ----------
    yaw_min, yaw_max : float
        Positive-yaw band to remove [deg].
    wavelength_um_min : float
        Only mask at or above this wavelength; the feature is not detectable
        below it, so the short-wave data is kept in full.

    """

    blob = (meas_data["manip_yaw_deg"].between(yaw_min, yaw_max)
            & (meas_data["wavelength_um"] > wavelength_um_min))

    return meas_data.loc[~blob]

# def single_scirad_erf_scattered_light_get_measurement(row,make_plot=False,
#                                                       pitch_max=None):
#     """
#     One row per radial angle, averaging the individual points at each angle.

#     This is the per-file entry point the batch analysis uses. It is a thin
#     wrapper over single_scirad_erf_scattered_light_get_points, which does all
#     the actual reduction; call that directly if you need the individual points
#     and their pitch/yaw rather than the radial averages.

#     Parameters
#     ----------
#     pitch_max : float, callable, or None
#         Optional cut on manipulator pitch applied before averaging, to exclude
#         the +pitch feature in the SW channel. See apply_pitch_cut.

#     """

#     pts = single_scirad_erf_scattered_light_get_points(row)

#     meas_data = apply_pitch_cut(pts.meas_data, pitch_max, pts.wavelength_um)

    # return scirad_average_measurements(meas_data, make_plot=make_plot)

# def scirad_average_measurements(meas_data, make_plot=False,
#                                 l_noise_floor=0.0):
#     """
#     Average individual measurement points down to one row per radial angle.

#     This is the second half of the reduction, split out from the collection so
#     that a cut can be applied to the individual points first - the SW +pitch
#     feature has to be removed before averaging, because the average blends
#     every azimuth at a given radial angle and would otherwise fold it in.

#     Accepts points from one file or from many; rows are grouped by log_file, so
#     a whole channel can be averaged in one call.

#     Parameters
#     ----------
#     meas_data : DataFrame
#         Individual points from single_scirad_erf_scattered_light_get_points or
#         batch_scirad_erf_scattered_light_store_all_measurements. Needs the
#         l_center / l_center_sd columns to reconstruct signal_uc.
#     make_plot : bool, default False
#         Write one PNG per log file showing the individual points behind the
#         averages. These are the images the notebook's slider cells display, so
#         they follow whatever cut was applied before averaging.
#     l_noise_floor : float, default SCIRAD_L_NOISE_FLOOR
#         Additive radiance noise floor [W m-2 sr-1] folded into signal_uc. Pass
#         0.0 to reproduce the old signal-proportional uncertainty.

#     Returns
#     -------
#     DataFrame with one row per (log_file, radial angle).

#     Notes
#     -----
#     Negative signals are kept. The DC subtraction is as likely to go negative as
#     positive once the true signal approaches the noise, so discarding them would
#     bias the large-angle data upward - by a median 34% beyond 7 deg on the Total
#     channel, and far more on individual points. With the noise floor in place
#     they carry honest error bars and the fit handles them correctly.

#     """

#     paths = load_config()

#     avg_all = []
#     for log_file, meas in meas_data.groupby('log_file', sort=False):

#         # Constant per file, carried on every row by the collection step
#         center_l    = meas['l_center'].iloc[0]
#         center_l_sd = meas['l_center_sd'].iloc[0]

#         # Average all the measurements at a given angle
#         avg_data = []
#         for angle in meas['rad_angle_deg'].unique():
#             data_tmp = meas.loc[(meas['rad_angle_deg'] == angle)]

#             avg_data.append({
#                             "log_file":             log_file,
#                             "type":                 data_tmp['type'].iloc[0],
#                             "photodiode":           data_tmp['photodiode'].iloc[0],
#                             "detector_index":       data_tmp['detector_index'].iloc[0],
#                             "rad_angle_deg":        data_tmp['rad_angle_deg'].median(),
#                             "wavelength_um":        data_tmp['wavelength_um'].iloc[0],
#                             "l_center":             center_l,
#                             "l":                    data_tmp['l'].mean(),
#                             "l_sd":                 data_tmp['l'].std(),
#                             "l_sem":                data_tmp['l'].sem(),
#                             "signal":               data_tmp['signal'].mean(),
#                             "signal_sd":            data_tmp['signal'].std(),
#                             "signal_sem":           data_tmp['signal'].sem()})

#         # Convert to a pandas dataframe
#         avg_data = pd.DataFrame(avg_data)

#         # RSS the standard error of the mean, the uncertainty in the center
#         # radiance, an additional 10%, and the additive detector noise floor.
#         #
#         # Written as a sum of squared absolute terms rather than |signal| times
#         # a sum of relative ones. Algebraically the first three are the same
#         # thing, but this form makes it clear that only signal_sem and the floor
#         # survive as the signal goes to zero - and it needs no abs(), which
#         # previously hid the fact that a near-zero point got a near-zero error
#         # bar. The floor is in radiance, so it is divided by the centre radiance
#         # to reach the peak-normalised units the rest of the row is in.
#         avg_data['signal_uc'] = np.sqrt(avg_data['signal_sem']**2 + \
#                                         (avg_data['signal']*center_l_sd/center_l)**2 + \
#                                         (0.10*avg_data['signal'])**2 + \
#                                         (l_noise_floor/center_l)**2)

#         if make_plot:
#             det_name      = meas['det_name'].iloc[0]
#             wavelength_um = meas['wavelength_um'].iloc[0]

#             fig, ax = plt.subplots()
#             ax.scatter(meas['rad_angle_deg'], meas['signal'],alpha=0.2,s=5)
#             ax.errorbar(avg_data["rad_angle_deg"], avg_data["signal"], yerr=avg_data["signal_uc"],
#                 fmt='o', color='red', markersize=2, capsize=2, zorder=1)
#             ax.set_title(f'{det_name}, {log_file}, {round(1000*wavelength_um)}nm')
#             ax.set_xlabel('Radial Angle [deg]')
#             ax.set_ylabel('PST [-]')
#             ax.set_yscale('log')
#             ax.set_xlim(1,10.5)
#             ax.set_ylim(1e-6,1)

#             plotname = paths.figure_dir / (log_file + ".png")
#             fig.savefig(plotname, dpi=150, bbox_inches='tight')
#             plt.close(fig)

#         avg_all.append(avg_data)

#     return pd.concat(avg_all, ignore_index=True)

def batch_scirad_erf_scattered_light_store_all_measurements(detector_index=None):
    """
    Every individual measurement point from every file, before averaging.

    Returns the points rather than the radial averages so that the manipulator
    pitch and yaw survive. That matters because the average blends all azimuths
    at a given radial angle: the SW +pitch feature has to be cut from the points
    first, which is only possible if the points are what get stored.

    Feed the result to scirad_average_measurements to get the one-row-per-angle
    table the fitting routines expect.

    Parameters
    ----------
    detector_index : int or iterable of int, optional
        Keep only these detectors (1=SSW, 2=LW, 3=Total, 4=SW, 5=EM). Default
        keeps everything.

    Returns
    -------
    DataFrame of individual points, including manip_pitch_deg / manip_yaw_deg,
    an azimuth_deg column, and the per-file l_center / l_center_sd needed to
    rebuild signal_uc.

    """

    paths = load_config()

    # Load the filenames
    batch_file = 'libera_erf_fiber_source_measurements_202502.csv'
    f = tidy_up_header(pd.read_csv(paths.analysis_dir / batch_file))

    # Only analyze the data with the forebaffle present
    f = f.loc[(f["forebaffle"] == 1)]

    if detector_index is not None and np.isscalar(detector_index):
        detector_index = [detector_index]

    results = []
    for row in f.itertuples():
        print(f"Row Number: {row.Index}, Filename: {row.log_file}")
        try:
            pts = single_scirad_erf_scattered_light_get_and_average_points(row)
        except Exception as ex:
            print(f"  skipped: {type(ex).__name__}: {ex}")
            continue

        if detector_index is not None and pts.detector_index not in detector_index:
            continue

        if len(pts.meas_data) > 5:
            results.append(pts.meas_data)

    all_points = pd.concat(results, ignore_index=True)

    # Azimuth around the boresight. The manipulator only reaches pitch +/-3 deg
    # with yaw positive, so this spans roughly 20-160 deg, not the full circle.
    all_points['azimuth_deg'] = np.degrees(np.arctan2(all_points['manip_yaw_deg'],
                                                      all_points['manip_pitch_deg']))

    return all_points


def flipbook(plots, width=400):
    """
    Slider-driven flipbook over a list of saved PNG paths, for stepping through
    the per-wavelength fit plots in a live notebook.

    This has to be a function rather than inline notebook code. The callback
    closes over the plot list, the slider and the output area, so three inline
    copies in one notebook would all close over the same three GLOBAL names -
    after a run-all, every slider would drive the last cell's plot list and
    write into the last cell's output area. Running the cells one at a time
    hides it, because each cell is interacted with before the next rebinds the
    globals. A function call gives each flipbook its own scope.

    ipywidgets renders nothing outside a live kernel - on GitHub in particular -
    so this is the interactive convenience, not the record. pst_fit_panels also
    writes a grid of the same panels, which stores as an ordinary image output
    and is what a reader sees.

    Imported locally so the module itself stays importable headlessly.
    """

    import ipywidgets as widgets
    from IPython.display import display, Image

    slider = widgets.IntSlider(min=0, max=len(plots) - 1, description='Plot')
    out = widgets.Output()

    def show(_change=None):
        with out:
            out.clear_output(wait=True)
            display(Image(plots[slider.value], width=width))

    slider.observe(show, names='value')
    display(slider, out)
    show()


def pst_fit_panels(*, panel_fn, wavelengths_um, figure_dir, name_fmt, grid_name,
                   suptitle=None, xlabel='Angle [deg]', ylabel='PST [-]',
                   ncol=5, panel_size=(3.0, 2.5), dpi_single=150, dpi_grid=110):
    """
    Draws one fit panel per wavelength twice: once as an individual figure (the
    flipbook frames) and once tiled onto a single grid figure.

    Both come from the same panel_fn(ax, wavelength_um) callable, so the two
    views cannot drift apart - which is the reason this exists rather than each
    notebook cell plotting twice. panel_fn draws onto the axis it is given and
    may return a dict of scalars, which are collected into the returned summary
    table (one row per wavelength).

    The grid is the version that survives outside a live kernel, so it is the
    one an external reader actually sees; see flipbook. Per-panel legends are
    kept only on the first grid panel, and axis labels only on the outer edge,
    since 23 copies of either is noise.

    Returns SimpleNamespace(plots, grid, summary).
    """

    figure_dir = Path(figure_dir)
    wavelengths_um = list(wavelengths_um)

    plots = []
    rows = []
    for wl in wavelengths_um:
        fig, ax = plt.subplots()
        info = panel_fn(ax, wl)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        path = figure_dir / name_fmt.format(nm=round(wl*1000))
        fig.savefig(path, dpi=dpi_single, bbox_inches='tight')
        plt.close(fig)
        plots.append(str(path))
        rows.append({"wavelength_um": wl, **(info or {})})

    n = len(wavelengths_um)
    nrow = math.ceil(n/ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(panel_size[0]*ncol, panel_size[1]*nrow),
                             sharex=True, sharey=True, squeeze=False)
    for i, (ax, wl) in enumerate(zip(axes.flat, wavelengths_um)):
        panel_fn(ax, wl)
        ax.set_title(ax.get_title(), fontsize=8)
        ax.tick_params(labelsize=7)
        legend = ax.get_legend()
        if legend is not None and i > 0:
            legend.remove()

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    # Label the outer edge only. The bottom row can be partly hidden when the
    # wavelength count does not fill the grid, so each column is labelled on
    # its lowest VISIBLE axis rather than on the last row blindly.
    for col in range(ncol):
        visible = [axes[r][col] for r in range(nrow) if axes[r][col].get_visible()]
        if visible:
            visible[-1].set_xlabel(xlabel, fontsize=8)
    for row in range(nrow):
        if axes[row][0].get_visible():
            axes[row][0].set_ylabel(ylabel, fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    grid_path = figure_dir / grid_name
    fig.savefig(grid_path, dpi=dpi_grid, bbox_inches='tight')

    return SimpleNamespace(plots=plots, grid=grid_path,
                           summary=pd.DataFrame(rows))

def cumulative_solid_angle(pst, solid_angle_sr):
    """
    Running integral of the PST over annular rings, by the trapezoid rule.

    ``solid_angle_sr`` is the per-ring element ``2*pi*sin(theta)*dtheta`` on a
    uniform theta grid, so dividing it back out recovers the integrand and the
    trapezoid can be taken in theta. Returns an array the same length as the
    input, starting at zero.

    """

    integrand = pst*solid_angle_sr
    cumulative = np.concatenate(([0.0],
                                 np.cumsum(0.5*(integrand[1:] + integrand[:-1]))))
    return cumulative

def scirad_erf_correction(*,
                          fit_result,
                          channel,
                          wavelength_um_min,
                          wavelength_um_max,
                          summary_filename=None,
                          plot_wavelength_um=None,
                          angle_deg=None,
                          half_angle_deg=(1.717, 2.068, 2.358),
                          fpe_name='fmfpe',
                          n_wavelengths=500,
                          extra_solid_angle=None,
                          make_plot=True,
                          show_plot=True):
    """
    Out-of-field loss versus wavelength for one channel, from a global PST fit.

    Integrates the fitted PST over annular rings to get the solid angle
    scattered beyond a given half angle, and expresses it against the signal the
    ERF calibration actually produced:

        correction = scattered outside the source
                     -------------------------------------------------
                     in-field solid angle + scattered inside the source

    That is the fraction by which the signal would rise if the calibration
    source were replaced by a uniform radiance scene larger than the whole
    integration range, which is what the correction is used for. Note the
    denominator excludes the scatter falling outside the source, since that
    light never reached the detector during the calibration.

    Assumptions worth knowing:

    * The angular grid starts at 1.5 deg by default, so the annulus between the
      edge of the field of view and 1.5 deg is counted in neither term. The fit
      is extrapolating below ~1.8 deg in any case, since that is where the
      measured data starts.
    * The field of view is an elongated hexagon, 1.3 by 2.6 deg, whose edge
      therefore sits anywhere from 0.65 to 1.3 deg out depending on azimuth.
      Only its total solid angle enters here, via ``geo_solid_angle``; treating
      it as an equal-area circle shifts the answer by about 1% of itself, well
      inside the fit uncertainty.
    * Truncating at 20 deg loses less than 0.1%, because the geometric
      illumination factor has vignetted to zero by then.

    The same calculation is wanted for every channel, so it lives here rather
    than being repeated per notebook cell.

    Parameters
    ----------
    fit_result : lmfit ModelResult
        The `.result` from one of the scirad_pst_*_fitting routines. Its eval()
        and eval_uncertainty() are what supply the PST and its 1-sigma band.
    channel : str
        Channel name as it appears in the conversions table: 'sw', 'ssw',
        'total', 'lw', 'swcr'. Used for the in-field solid angle and the labels.
    wavelength_um_min, wavelength_um_max : float
        Range to evaluate over, log spaced. Set this to the range the channel
        is actually used over - the fit is extrapolating outside the measured
        0.36-2.4 um either way.
    summary_filename : str, optional
        If given, the per-wavelength summary is written to this CSV in the
        analysis directory.
    plot_wavelength_um : float, optional
        Also plot the loss-versus-half-angle curve at the first wavelength at
        or above this value, which is the diagnostic view of a single fit.
    angle_deg : array_like, optional
        Angular grid to integrate over. Defaults to 1.5-20 deg in 1000 steps.
    half_angle_deg : sequence of float
        Aperture half angles to report [deg]. Defaults to the 10, 12 and 13.6 mm
        ERF apertures (OPA focal length 165.43 mm).
    extra_solid_angle : DataFrame, optional
        Additional scattered solid angle to carry as an uncertainty rather than
        as a correction, with columns wavelength_um and excess_solid_angle_sr -
        the output of scirad_blob_excess_solid_angle. It is interpolated in log
        space across wavelength, converted to a percentage the same way the
        correction itself is, and root-sum-squared into each loss_*_sd. Use it
        for a feature that has been masked out of the fit but whose real
        contribution is unknown.
    show_plot : bool, default True
        Leave the figures open so the inline backend renders them in the
        notebook. Set False to close them after saving, which is what you want
        when the cell also displays a widget - the inline backend draws every
        open figure at the end of the cell, so it would otherwise appear below
        the widget regardless of where it was created.

    Returns
    -------
    DataFrame with one row per wavelength: wavelength_um and loss_10 / loss_12 /
    loss_13 with their 1-sigma uncertainties, in percent.

    """

    paths = load_config()

    if angle_deg is None:
        angle_deg = np.linspace(1.5, 20, 1000)
    angle_deg = np.asarray(angle_deg, dtype=float)
    angle_rad = np.radians(angle_deg)

    # The point spacing in radians
    dangle_rad = (angle_rad[-1] - angle_rad[0]) / (angle_rad.size - 1)

    # Calculate the solid angle of each annular ring. The exact element is
    # 2*pi*sin(theta)*dtheta; using theta instead is the small-angle form, which
    # is 2.1% high by 20 deg.
    solid_angle_sr = 2*math.pi*np.sin(angle_rad)*dangle_rad

    # The wavelengths to calculate
    wavelength_um_array = np.logspace(math.log10(wavelength_um_min),
                                      math.log10(wavelength_um_max),
                                      n_wavelengths)

    # Load the conversion coefficients, this is to get the solid-angle of the channel
    conv = erf.get_scirad_conversions(fpe_name=fpe_name)
    geo_solid_angle = conv.loc[channel, "solid_angle"]

    half_angle = list(half_angle_deg)

    loss_summary = []
    plt_wl_um = plot_wavelength_um if plot_wavelength_um is not None else np.inf
    for wl_um in wavelength_um_array:

        fit    = fit_result.eval(angle_deg=angle_deg, wavelength_um=wl_um)
        fit_sd = fit_result.eval_uncertainty(angle_deg=angle_deg, wavelength_um=wl_um)

        # Scattered solid angle accumulated out to each angle [sr], and the
        # remainder lying beyond it. Trapezoid rather than a running sum: the
        # integrand falls steeply from the inner limit, and a rectangle rule
        # needs ~20000 points to match what the trapezoid reaches by 1000.
        scattered_inside       = cumulative_solid_angle(fit,           solid_angle_sr)
        scattered_inside_p_sd  = cumulative_solid_angle(fit + fit_sd,  solid_angle_sr)
        scattered_inside_m_sd  = cumulative_solid_angle(fit - fit_sd,  solid_angle_sr)

        scattered_outside      = scattered_inside[-1]      - scattered_inside
        scattered_outside_p_sd = scattered_inside_p_sd[-1] - scattered_inside_p_sd
        scattered_outside_m_sd = scattered_inside_m_sd[-1] - scattered_inside_m_sd

        # The correction is what the signal gains when the source grows from the
        # calibration size to a uniform scene. The denominator is therefore the
        # signal the calibration actually produced - in-field plus only the
        # scatter falling *inside* the source - not the whole integral.
        frac_outside      = 100*scattered_outside      / (geo_solid_angle + scattered_inside)
        frac_outside_p_sd = 100*scattered_outside_p_sd / (geo_solid_angle + scattered_inside_p_sd)
        frac_outside_m_sd = 100*scattered_outside_m_sd / (geo_solid_angle + scattered_inside_m_sd)

        if make_plot and wl_um >= plt_wl_um:
            plt_wl_um = np.inf   # only the first qualifying wavelength

            fig, ax = plt.subplots()
            ax.fill_between(angle_deg, frac_outside_m_sd, frac_outside_p_sd,
                            color='purple', alpha=0.3, zorder=1)
            ax.plot(angle_deg, frac_outside, color='purple', zorder=2, alpha=0.5)

            ax.set_title(f"{channel.upper()} Channel {round(1000*wl_um)} nm,")
            ax.set_xlabel('Half Angle [deg]')
            ax.set_ylabel('Fraction Outside Half Angle [%]')
            ax.set_yscale('log')
            ax.set_xlim(0, 10)
            ax.set_ylim(1e-2, 10)

            for ha, colour, label in zip(half_angle,
                                         ('red', 'green', 'blue'),
                                         ('10mm Aperture', '12mm Aperture',
                                          '13.6mm Aperture')):
                ax.plot([ha, ha], [1e-3, 100], color=colour, label=label)
            ax.legend()

        loss_vals      = np.interp(half_angle, angle_deg, frac_outside)
        loss_vals_p_sd = np.interp(half_angle, angle_deg, frac_outside_p_sd)
        loss_vals_m_sd = np.interp(half_angle, angle_deg, frac_outside_m_sd)

        # Average the positive and negative sd to get the estimated stddev
        loss_vals_sd = ((loss_vals_p_sd - loss_vals) + (loss_vals - loss_vals_m_sd))/2

        # Solid angle masked out of the fit, carried as an uncertainty. It sits
        # in the wings, outside every aperture, so it adds to the numerator of
        # the correction and not to the calibration signal in the denominator.
        if extra_solid_angle is not None and len(extra_solid_angle):
            extra_sr = np.exp(np.interp(np.log(wl_um),
                                        np.log(extra_solid_angle['wavelength_um'].values),
                                        np.log(extra_solid_angle['excess_solid_angle_sr'].values)))
            extra_pct = 100.0*extra_sr/(geo_solid_angle + np.interp(half_angle, angle_deg,
                                                                    scattered_inside))
        else:
            extra_pct = np.zeros(len(half_angle))

        loss_vals_sd = np.sqrt(loss_vals_sd**2 + extra_pct**2)

        loss_summary.append({
            "wavelength_um":  wl_um,
            "loss_10":        loss_vals[0],
            "loss_10_sd":     loss_vals_sd[0],
            "loss_12":        loss_vals[1],
            "loss_12_sd":     loss_vals_sd[1],
            "loss_13":        loss_vals[2],
            "loss_13_sd":     loss_vals_sd[2],
        })

    loss_summary = pd.DataFrame(loss_summary)

    if make_plot:
        fig, ax = plt.subplots()
        for key, colour, label in (('loss_10', 'red',   '10mm Aperture'),
                                   ('loss_12', 'green', '12mm Aperture'),
                                   ('loss_13', 'blue',  '13.7mm Aperture')):
            ax.fill_between(loss_summary['wavelength_um'],
                            loss_summary[key] - loss_summary[key + '_sd'],
                            loss_summary[key] + loss_summary[key + '_sd'],
                            color=colour, alpha=0.3, zorder=1)
            ax.plot(loss_summary['wavelength_um'], loss_summary[key],
                    color=colour, zorder=2, alpha=0.5, label=label)

        ax.set_title(f"{channel.upper()} Channel ERF Correction")
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('ERF Correction [%]')
        ax.set_xlim(wavelength_um_min, min(4, wavelength_um_max))
        ax.set_ylim(0, 3)
        plt.legend(fontsize=8)

        plotname = paths.figure_dir / f"scirad_{channel}_pst_erf_correction.png"
        fig.savefig(plotname, dpi=150, bbox_inches='tight')

        if not show_plot:
            plt.close('all')

    # Save the correction to a file
    if summary_filename is not None:
        loss_summary.to_csv(paths.analysis_dir / summary_filename,
                            index=False, float_format='%.6g')

    return loss_summary

def scirad_estimate_noise_floor(meas_data, r_min=1.8, r_max=None,
                                make_plot=False, show_plot=True,
                                channel_label=''):
    """
    Measure the additive radiance noise floor from the two halves of a scan.

    Every 2-D scan visits each radial angle at both +yaw and -yaw. If the field
    is radially symmetric those two halves measure the same physical quantity,
    so whatever they disagree by is noise - which makes this a self-calibrating
    estimate that needs no model of the scattered light at all.

    Averaging each half separately gives two independent estimates of the same
    radiance, l_pos and l_neg, with reported uncertainties l_sem_pos/l_sem_neg.
    If those uncertainties were right then

        z = (l_pos - l_neg) / sqrt(sem_pos**2 + sem_neg**2)

    would be a standard normal, whose median |z| is 0.6745. It is not: the
    measured value is around 1.7, because the points averaged at one angle are
    largely correlated and so the noise does not fall as 1/sqrt(n). Solving

        median |l_pos - l_neg| / sqrt(sem_pos**2 + sem_neg**2 + 2*L**2) = 0.6745

    for L gives the additive floor that restores consistency. The factor of two
    is because each half carries its own independent floor.

    The median is used rather than the standard deviation because real field
    structure - the SW +pitch feature, for instance - shows up as heavy tails
    that would inflate an RMS estimate. For the same reason this should be run
    on a channel known to be radially symmetric: on a channel with a genuine
    asymmetry it will absorb that asymmetry into the noise estimate and report a
    floor that is too large.

    Parameters
    ----------
    meas_data : DataFrame
        Individual points for one channel, from
        batch_scirad_erf_scattered_light_store_all_measurements.
    r_min, r_max : float, optional
        Restrict to a range of radial angle [deg].
    show_plot : bool, default True
        Leave the figure open for the notebook to render. Set False to close it
        after saving; see the same argument on scirad_erf_correction.
    channel_label : str
        Only used to title the plot.

    Returns
    -------
    SimpleNamespace with l_noise_floor [W m-2 sr-1], the median |z| before and
    after applying it, and the number of matched half-pairs.

    """

    from scipy.optimize import brentq

    # median of |z| for a standard normal
    target = 0.6744897501960817

    meas_data = meas_data.loc[meas_data['rad_angle_deg'] >= r_min]
    if r_max is not None:
        meas_data = meas_data.loc[meas_data['rad_angle_deg'] <= r_max]

    pos = meas_data.loc[meas_data['manip_yaw_deg'] > 0].copy()
    neg = meas_data.loc[meas_data['manip_yaw_deg'] < 0].copy()

    # pos = scirad_average_measurements(meas_data.loc[meas_data['manip_yaw_deg'] > 0],
    #                                   l_noise_floor=0.0)
    # neg = scirad_average_measurements(meas_data.loc[meas_data['manip_yaw_deg'] < 0],
    #                                   l_noise_floor=0.0)

    pair = pos.merge(neg, on=['log_file', 'rad_angle_deg'], suffixes=('_p', '_n'))
    pair = pair.loc[(pair['l_sem_p'] > 0) & (pair['l_sem_n'] > 0)]

    difference = np.abs(pair['l_p'] - pair['l_n'])
    reported   = pair['l_sem_p']**2 + pair['l_sem_n']**2

    def median_z(floor):
        return np.median(difference/np.sqrt(reported + 2*floor**2)) - target

    if median_z(0.0) <= 0:
        # already consistent - nothing to add
        l_noise_floor = 0.0
    else:
        l_noise_floor = brentq(median_z, 1e-8, 1.0)

    before = np.median(difference/np.sqrt(reported))
    after  = np.median(difference/np.sqrt(reported + 2*l_noise_floor**2))

    if make_plot:
        paths = load_config()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.linspace(0, 6, 40)
        ax.hist(difference/np.sqrt(reported), bins=bins, alpha=0.5,
                color='red', label=f'reported only (median {before:.2f})')
        ax.hist(difference/np.sqrt(reported + 2*l_noise_floor**2), bins=bins,
                alpha=0.5, color='blue', label=f'with floor (median {after:.2f})')
        ax.axvline(target, color='k', ls='--', lw=1,
                   label=f'expected median {target:.2f}')
        ax.set_xlabel('|z| between the +yaw and -yaw halves')
        ax.set_ylabel('count')
        ax.set_title(f'{channel_label} noise floor = {l_noise_floor:.3e} W m-2 sr-1')
        ax.legend(fontsize=8)

        plotname = paths.figure_dir / f"scirad_{channel_label.lower()}_noise_floor.png"
        fig.savefig(plotname, dpi=150, bbox_inches='tight')

        if not show_plot:
            plt.close(fig)

    return SimpleNamespace(l_noise_floor=l_noise_floor,
                           median_z_reported=before,
                           median_z_with_floor=after,
                           n_pairs=len(pair))

def scirad_blob_excess_solid_angle(meas_data, fit_result,
                                   yaw_min=SW_BLOB_YAW_MIN,
                                   yaw_max=SW_BLOB_YAW_MAX,
                                   d_pitch_deg=1.0, d_yaw_deg=0.625):
    """
    Integrated solid angle of the masked SW feature, against a baseline model.

    apply_blob_mask removes the feature so it cannot bias the fit. That leaves
    the question of what it would have contributed had it been real telescope
    scatter, which is the quantity to carry as an uncertainty on the ERF
    correction rather than as a correction in its own right - we do not know
    whether it is an instrument property or an artifact of the calibration
    setup.

    The excess (measurement minus baseline model) is integrated over the
    manipulator grid inside the masked band, using the annular element
    sin(theta)*dpitch*dyaw. Only the 2-D scans contribute, since a 1-D yaw scan
    does not sample the pitch extent of the feature.

    Returns
    -------
    DataFrame with one row per wavelength: wavelength_um, n_cells and
    excess_solid_angle_sr.

    Notes
    -----
    This is a lower bound. The feature is still rising at the +3 deg pitch
    travel limit, so an unknown part of it lies outside the region the
    manipulator can reach.

    """

    meas_data = meas_data.copy()
    meas_data['pitch'] = meas_data['manip_pitch_deg'].round(2)
    meas_data['yaw']   = meas_data['manip_yaw_deg'].round(2)

    # 2-D scans only: identify them by how many distinct pitch values they visit
    npitch = meas_data.groupby('log_file')['pitch'].nunique()
    meas_data = meas_data.loc[meas_data['log_file'].isin(npitch[npitch > 3].index)]

    cell_sr = np.radians(d_pitch_deg)*np.radians(d_yaw_deg)

    rows = []
    for wavelength_um, group in meas_data.groupby(np.round(meas_data['wavelength_um'], 3)):

        cell = group.groupby(['pitch', 'yaw']).agg(
                   signal=('signal', 'mean'),
                   rad_angle_deg=('rad_angle_deg', 'mean')).reset_index()

        model = fit_result.eval(angle_deg=cell['rad_angle_deg'].values,
                                wavelength_um=wavelength_um)
        cell['excess'] = cell['signal'] - model

        blob = cell.loc[cell['yaw'].between(yaw_min, yaw_max) & (cell['excess'] > 0)]
        if len(blob) < 4:
            continue

        solid_angle_sr = np.sin(np.radians(blob['rad_angle_deg']))*cell_sr

        rows.append({"wavelength_um":         wavelength_um,
                     "n_cells":               len(blob),
                     "excess_solid_angle_sr": (blob['excess']*solid_angle_sr).sum()})

    return pd.DataFrame(rows)

if __name__ == "__main__":

    # Analyze the radiometer ERF files
    tmp = batch_scirad_erf_scattered_light_store_all_measurements()

    #paths = pst.load_config()
    #dto_pts = tidy_up_header(pd.read_csv(paths.analysis_dir / "scirad_total_pst_data.csv"))

    # Average the data
    #dto = scirad_average_measurements(dto_pts)


