import pandas as pd
import numpy as np
import math
from types import SimpleNamespace

from libera_config import load_config

def dn_to_pow(*,pbr_dark_dn,
                pbr_sn,
                pbr_pcb_temp,
                pbr_vref=None,
                pbr_r_htr=None): 
    """
    Get the PBR-R DN to nW conversion
    """

    match pbr_sn:
        case "17":
            #-----------------------------------------------------------------------------
            #Cam measured the top resistance of the PBR-R active channel on 2024-02-09.
            #From an email from him on 2024-02-09
            #Top Resistor 1 = 9097.19
            #Top Resistor 2 = 9097.19
            #
            #
            #Calculated heater resistances, Cam's in-situ measurement from 2024-02-07:
            #Heater 1 = 8860.55
            #Heater 2 = 8832.54
            #
            #From PBR-R Z0 Measurement:
            #
            #libera_cal_pbrr_z0_analysis
            #PBR-R Tau =        33.5371492s, Z0 = 3087.2 K/W
            #
            #Appears that the detectors run at about 26.7 C
            #
            #2023-11-13 Measurement:
            #Active Heater = 8772.37 Ohms at 66.7F = 19.28 C
   
            #Fit from the TCR data: '20240426_15-55_PBRR_CHIP17_CH1.csv'
            #R(T) = 8764.38 + 12.18*(T - 20)
            pbr_htr_0 = 8764.38
            t0 = 20
            pbr_dr_dt = 12.18

            #The estimated Z0 value for PBR-R
            pbr_z0 = 3080

            #*************************************************************************
            #INITIAL RTOP MEASUREMENT
            #Measured PBR-R Top Resistor, 2024-02-09. 21.555 nW/DN
            #pbr_rtop = 9097.19d

            #*************************************************************************
            #UPDATED RTOP MEASUREMENT
            #Cam PBR-R Top R Measurement 2026-02-20 Email
            #The TCR is positive and very small, and all the measurements at a given
            #temperature are within about +/- 5 ppm of the mean.
            #
            #The 34465A meter dominates the uncertainty at +/- 45 ppm. I propose we
            #use the mean value of 9097.0 +/- 0.4 ohms corresponding to an amplifier
            #board temperature of ~30 °C.
            #9097.0 +/- 0.4
            pbr_rtop = 9096.958

            pbr_r_trace = 0.684
            pbr_r_harness = 0.084245
            pbr_lead = pbr_r_trace + pbr_r_harness

            #This is the approximate LTZ1000 value for the original board
            if pbr_vref is None:
                #pbr_vref = 7.14041    #Original value pre 2024-05-21
                #pbr_vref = 7.14408    #New value from 2024-05-21
                pbr_vref = 7.144122    #Best value using a 3458A from 2026-03

            # The relative uncertainty of the power due to each of the key elements
            # These values are straight from the PBR-R paper and are always k=1
            u_pbrr_vref = 4.04E-05
            u_pbrr_rhtr = 5.77E-06
            u_pbrr_rtop = 2.54E-05
            u_pbrr_rtrace = 9.24E-06
            u_pbrr_nonequiv = 6.18E-05
            u_pbrr_nonlinear = 2.89E-05

        case _:  
            print("Invalid PBR-R SN " + pbr_sn)
            breakpoint()

    if pbr_r_htr is None:
        #Iterate to determine the detector temperature, which is needed to determine
        #the power conversion
        t_det = 25

        for m in range(6):
            #Calculate the heater resistance at the estimate detector temperature
            pbr_rhtr = pbr_htr_0 + pbr_dr_dt*(t_det - t0)
      
            #Calculate the thermistor resisance at the estimated detector temperature
            #
            #PBR-R Thermistors:
            #Three Semitec 
            #Mouser PN: 954-103FT1005A5P1
            #Semitec PN: 103FT1005A5P1
            #Beta = 3370K
            #R at 25C = 10k
            r0 = 10e3
            b = 3370
            pbr_rtherm = r0*math.exp(b*(1/(t_det + 273.15) - 1/(25 + 273.15)))
      
            #Now calculate the resistance of three in parallel
            pbr_rtherm_eff = pbr_rtherm/3
      
            #Now calculate the power in the thermistors
            v_sine_amp = 3                          #Sine amplitude
            v_sine_rms = v_sine_amp/math.sqrt(2)    #Sine rms
            r_fixed = 3320                          #Fixed resistor in series with 
      
            #Calculate the sine current, +vrms applied to top of bridge, -vrms to bottom
            i_sine_rms = 2*v_sine_rms/(pbr_rtherm_eff + r_fixed)
      
            #Finally, the thermistor power in W
            therm_power_w = (i_sine_rms**2)*pbr_rtherm_eff
      
            #Calculate the 100% power level
            pbr_current = pbr_vref/(pbr_rtop + pbr_rhtr + pbr_lead)
            pbr_pow_max = (pbr_current**2)*pbr_rhtr
      
            #Now calculate the watts per dn
            pbr_w_per_dn = pbr_pow_max*(1/65000)
  
            #Calculate the "dark" heater power
            pbr_r_median_pow_w = pbr_w_per_dn*pbr_dark_dn
      
            #Calculate the total power on the active PBR-R in watts
            pbr_total_pow_w = therm_power_w + pbr_r_median_pow_w
      
            #Update the estimate of the detector temperature
            t_det = pbr_pcb_temp + pbr_total_pow_w*pbr_z0
    
    else:
            #Use the keyword value for the heater resistance
            pbr_rhtr = pbr_r_htr
            
            #Calculate the 100% power level
            pbr_current = pbr_vref/(pbr_rtop + pbr_rhtr + pbr_lead)
            pbr_pow_max = (pbr_current**2)*pbr_rhtr
            
            #Now calculate the watts per dn
            pbr_w_per_dn = pbr_pow_max*(1/65000)

    return SimpleNamespace(c_pbr_w_per_dn=pbr_w_per_dn,
                           u_pbrr_vref=u_pbrr_vref,
                           u_pbrr_rhtr=u_pbrr_rhtr,
                           u_pbrr_rtop=u_pbrr_rtop,
                           u_pbrr_rtrace=u_pbrr_rtrace,
                           u_pbrr_nonequiv=u_pbrr_nonequiv,
                           u_pbrr_nonlinear=u_pbrr_nonlinear)

def area_omega(baffle_temp_c=28.5):
    #Get the area-omega product for PBR-R

    #Initial values for the PBR-R
    a1 = 19.63327e-6    #A1 [m^2],         Front Aperture, 161658-SN01, r=2.49989mm, Area=19.63327mm2, 161658 SN-01_1107_0
    a2 = 19.63269e-6    #A2 [m^2], Rear/Detector Aperture, 161658-SN02, r=2.49986mm, Area=19.63269mm2, 161658 SN-02_1449_0

    # Original value
    # d = 199.909e-3      #d [m]

    # Final and best measurement
    #d [m] at 21.741 C, Measured on 2026-04-29 and in: 
    #Libera ERF Testing OneNote>PBR-R Testing>2026-04-29 Ap-Ap Distance Measurement
    d = 199.9035e-3     

    # Correct for thermal expansion of the baffle tube
    d = d*(1 + 23e-6*(baffle_temp_c - 21.741))
    ao = (math.pi/2)*(a1 + a2 + math.pi*(d**2) - math.sqrt((a1 + a2 + math.pi*(d**2))**2 - 4*a1*a2))

    # Relative uncertainty on the AO value due to each of the key measurements
    # These values are from the PBR-R paper and are always k=1
    u_pbrr_det_ap = 7.51E-06
    u_pbrr_ent_ap = 6.93E-06
    u_pbrr_ap_d = 8.31E-05
    u_pbrr_ap_align = 6.35E-06
    
    return SimpleNamespace(c_ao=ao,
                           u_pbrr_det_ap=u_pbrr_det_ap,
                           u_pbrr_ent_ap=u_pbrr_ent_ap,
                           u_pbrr_ap_d=u_pbrr_ap_d,
                           u_pbrr_ap_align=u_pbrr_ap_align)
                           

def diffraction_correction(*, wavelength_um, ap_dia_mm):
    #Returns the diffraction correction.
    #
    #Inputs
    #wavelength_um     Wavelength for the correction [um]
    #ap_dia            Integrating sphere aperture diameter [mm]
    #                   10, 12, or 13
    #
    #Returns
    #The diffraction loss for PBR-R for this wavelength
    
    match ap_dia_mm: 
        case 10:
            file = 'pbrr_sphere9_9195mm_OAP165_4mm_diffractionCorrection.csv'
        case 12:
            file = 'pbrr_sphere11_947mm_OAP165_4mm_diffractionCorrection.csv'
        case 13:
            file = 'pbrr_sphere13_625mm_OAP165_4mm_diffractionCorrection.csv'
        case _:  
            print("Invalid Sphere Diameter " + str(ap_dia_mm))
            breakpoint()

    # Read the correct file, based on the source sphere output diameter
    paths = load_config()
    loss = pd.read_csv(paths.conversions_and_calibrations_dir / file, skiprows=[1])

    # Linearly interpolate to the wavelength under test
    diff_corr = np.interp(wavelength_um*1e-6, loss["Wavelength"], loss["Normalized Flux"])
    diff_loss_rel_uncert = np.interp(wavelength_um*1e-6, loss["Wavelength"], loss["Relative Uncertainty"])

    # Calculate the relative uncertainty of the entrance aperture diffraction loss
    u_pbrr_ent_ap_diff = diff_loss_rel_uncert*(1 - diff_corr)

    # TODO Update this, it's not critical though because it's much smaller than u_pbrr_ent_ap_diff
    u_pbrr_det_ap_diff = 0
    
    return SimpleNamespace(c_diff_ent_ap=diff_corr, 
                           c_diff_det_ap=1, 
                           u_pbrr_ent_ap_diff=u_pbrr_ent_ap_diff,
                           u_pbrr_det_ap_diff=u_pbrr_det_ap_diff)

def reflectance(wavelength_um):
    #Get the reflectance of the PBR-R detector

    #Read the reflectance file
    #Detector: 23021pPBRrp17
    paths = load_config()

    file = "pbrr_reflectance_erf_active.csv"
    refl = pd.read_csv(paths.conversions_and_calibrations_dir / file, skiprows=[0])
    #Wavelength [nm]	Reflectance	Uncertainty

    # Linearly interpolate to the wavelength under test
    vacnt_abs = 1 - np.interp(wavelength_um*1e3, refl["Wavelength [nm]"], refl["Reflectance"])

    # This is the absolute uncertainty in the reflectance
    u_pbrr_vacnt  = np.interp(wavelength_um*1e3, refl["Wavelength [nm]"], refl["Uncertainty"])
    
    return SimpleNamespace(c_vacnt_abs=vacnt_abs, 
                           u_pbrr_vacnt=u_pbrr_vacnt)

def radiance_conversion(*, wavelength_um, ap_dia_mm, baffle_temp_c=28.5):
    """
    Returns the power to radiance conversion and associated uncertainties for PBR-R
    """

    # Get the Area-Solid Angle product and associated uncertaintes
    ao = area_omega(baffle_temp_c=baffle_temp_c)

    # Get the diffraction corrections and associated uncertaintes
    diff = diffraction_correction(wavelength_um=wavelength_um, ap_dia_mm=ap_dia_mm)

    # Get the PBR-R VACNT reflectivity and associated unceratinties
    cnt_refl = reflectance(wavelength_um)

    # The PBR-R conversion from power to radiance
    c_pbrr_pow_to_rad = 1/(ao.c_ao*diff.c_diff_ent_ap*cnt_refl.c_vacnt_abs)

    # Current PBR-R stray light uncertainty estimate
    u_pbrr_stray = 2.89E-04

    return SimpleNamespace(c_pbrr_ao=ao.c_ao,
                           c_pbrr_pow_to_rad=c_pbrr_pow_to_rad,
                           c_pbrr_diff_ent_ap=diff.c_diff_ent_ap,
                           c_pbrr_diff_det_ap=diff.c_diff_det_ap,                                                    
                           c_pbrr_vacnt_abs=cnt_refl.c_vacnt_abs,
                           u_pbrr_ap_align = ao.u_pbrr_ap_align,
                           u_pbrr_stray = u_pbrr_stray,
                           u_pbrr_ap_d = ao.u_pbrr_ap_d,
                           u_pbrr_det_ap = ao.u_pbrr_det_ap,
                           u_pbrr_ent_ap = ao.u_pbrr_ent_ap,
                           u_pbrr_det_ap_diff = diff.u_pbrr_det_ap_diff,
                           u_pbrr_ent_ap_diff = diff.u_pbrr_ent_ap_diff,
                           u_pbrr_vacnt=cnt_refl.u_pbrr_vacnt)