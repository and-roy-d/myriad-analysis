import os
import glob
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import re
import scipy.constants
import lmfit
import pandas as pd
import warnings
import pprint
from matplotlib import cm
import json

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
# List of channels to analyze. If empty, runs all active channels.
# Channel-to-pixel mapping: key = channel#, value = pixel# string
PIXEL_MAP = {
     2: "23",  3: "21",  4: "19",  5: "17",  6: "15",  7: "13",
     8: "11",  9:  "9", 10:  "7", 11:  "5", 12:  "3", 13:  "1",
    18: "24", 19: "22", 20: "20", 21: "18", 22: "16", 23: "14",
    24: "12", 25: "10", 26:  "8", 27:  "6", 28:  "4", 29:  "2",
}

PIXEL_MAP_B = {
    36: "23", 37: "21", 38: "19", 39: "17", 40: "15", 41: "13",
    42: "11", 43:  "9", 44:  "7", 45:  "5", 46:  "3", 47:  "1",
    52: "24", 53: "22", 54: "20", 55: "18", 56: "16", 57: "14",
    58: "12", 59: "10", 60:  "8", 61:  "6", 62:  "4", 63:  "2",
}

PIXEL_CELL_MAP = {
    1: '20um_3sq_SC',        2: '20um_3sq_SC',        3: '20um_3sq_SC',
    4: '20um_3sq_SC_E',      5: '20um_3sq_SC_E',      6: '20um_3sq_SC_E',
    7: '20um_3sq_SC_150',    8: '20um_3sq_SC_150',
    9: '20um_3sq_SC_150_E', 10: '20um_3sq_SC_150_E',
   11: '20um_2sq_SC',       12: '20um_2sq_SC',
   13: '20um_2sq_SC_E',     14: '20um_2sq_SC_E',
   15: '20um_3sq_barr_SC',  16: '20um_3sq_barr_SC',  17: '20um_3sq_barr_SC',
   18: '20um_3sq_barr_SC_E',19: '20um_3sq_barr_SC_E',20: '20um_3sq_barr_SC_E',
   21: '20um_2sq_barr_SC',  22: '20um_2sq_barr_SC',
   23: '20um_2sq_barr_SC_E',24: '20um_2sq_barr_SC_E'
}

CELL_COLOR_MAP = {
    '20um_3sq_SC':        '#1f77b4',  # Dark Blue
    '20um_3sq_SC_E':      '#72b7e0',  # Light Blue
    '20um_3sq_SC_150':    '#ff7f0e',  # Dark Orange
    '20um_3sq_SC_150_E':  '#ffbb78',  # Light Orange
    '20um_2sq_SC':        '#2ca02c',  # Dark Green
    '20um_2sq_SC_E':      '#98df8a',  # Light Green
    '20um_3sq_barr_SC':   '#d62728',  # Dark Red
    '20um_3sq_barr_SC_E': '#ff9896',  # Light Red / Pink
    '20um_2sq_barr_SC':   '#9467bd',  # Dark Purple
    '20um_2sq_barr_SC_E': '#c5b0d5',  # Light Purple
}

# Discrete list of channels to analyze. Edit as needed.
# CHANNELS_TO_ANALYZE = [ch for ch in PIXEL_MAP.keys()] + [ch for ch in PIXEL_MAP_B.keys()]
CHANNELS_TO_ANALYZE = [38]


# Rtes/Rn ratios for fitting (fractions)
R_OVER_RN_RATIOS = np.linspace(0.70, 0.99, 30)

# Save results (CSV files and summary CSV) to disk
SAVE_RESULTS = True

# Save generated plot figures to disk. If False, displays them interactively using plt.show()
SAVE_PLOTS = True

# Global hardware constants
R_BIAS = 1985.0     # Ohm
R_SHUNT = 250e-6    # Ohm
MIN_SI = 250e-12    # Mutual inductance in Henries
CHIP_MIN_SI = {
    'A': 250e-12,
    'B': 250e-12
}

# Temperature at which G = dP/dT is evaluated (K)
G_T_EVAL = 0.030  # 30 mK

# Hardcoded Rn override per channel (in Ohms). None = use dynamic estimation (last 100 pts).
# RN_OVERRIDES = {
#      3: 9e-3,  4: 15e-3,  5: 15e-3,  6: 15e-3,  7: None,
#      8: 12e-3,  9: 18e-3, 10: 18e-3, 11: None, 12: 18e-3, 13: 18e-3,
#     18: None, 19: 8e-3, 20: 14e-3, 21: 14e-3, 22: 14e-3, 23: None,
#     24: 11e-3, 25: 18e-3, 26: 18e-3, 27: None, 28: 18e-3, 29: None, 37: 10e-3, 38:7e-3, 39:15e-3, 40:None, 45: 18e-3
# }
RN_OVERRIDES = {}

# Exclude specific bath temperatures (in K) per channel. [] = no exclusions.
EXCLUDE_TEMPS = {
     3: [0.026],  4: [],  5: [],  6: [],  7: [],
     8: [0.020,0.021,0.022],  9: [0.028, 0.035], 10: [], 11: [], 12: [], 13: [],
    18: [], 19: [], 20: [], 21: [], 22: [], 23: [],
    24: [0.02, 0.021,0.022,0.023], 25: [0.021], 26: [], 27: [], 28: [0.02,0.021,0.022,0.023,0.024], 29: [],
    37: [0.02,0.021,0.022,0.023,0.024,0.025, 0.026, 0.027,0.028,0.029],
    39:[0.020, 0.021, 0.022, 0.023],
    45: [0.02,0.021,0.022,0.023,0.024,0.025, 0.026, 0.027,0.028,0.029]
}   

# Max bath temperature (K) to include in the P_TES fit, per channel. None = no upper limit.
TBASE_MAX = {
     3: 0.039,  4: 0.037,  5: 0.035,  6: 0.035,  7: 0.037,
     8: 0.037,  9: 0.037, 10: 0.034, 11: 0.037, 12: 0.033, 13: 0.035,
    18: 0.032, 19: 0.035, 20: 0.035, 21: 0.035, 22: 0.035, 23: 0.037,
    24: 0.035, 25: 0.035, 26: 0.035, 27: 0.037, 28: 0.032, 29: 0.037, 38:0.045, 39:0.039,
    45: 0.042, 47:0.038
}

# Custom fitting constraints & guesses per channel.
# Options: 'Tc_guess' (mK), 'Tc_min' (mK), 'Tc_max' (mK), 'n_guess', 'n_min', 'n_max', 'k_guess' (nW/K^n)
# Set to {} to use defaults.
FIT_OVERRIDES = {
     3: {},  4: {},  5: {},  6: {},  7: {},
     8: {},  9: {}, 10: {}, 11: {}, 12: {}, 13: {},
    18: {}, 19: {}, 20: {}, 21: {}, 22: {}, 23: {},
    24: {}, 25: {}, 26: {}, 27: {}, 28: {}, 29: {},
}

# Plot zoom limits per channel. {} = no zoom.
# Options: 'ibias_mA' (min, max tuple), 'ites_mA' (min, max tuple)
ZOOM_LIMITS = {
     3: {},  4: {},  5: {},  6: {},  7: {},
     8: {},  9: {}, 10: {}, 11: {}, 12: {}, 13: {},
    18: {}, 19: {}, 20: {}, 21: {}, 22: {}, 23: {},
    24: {}, 25: {}, 26: {}, 27: {}, 28: {}, 29: {},
}

# If SAVE_PLOTS is True, we use headless backend. Otherwise we allow default GUI backend for plt.show()
if SAVE_PLOTS:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
warnings.filterwarnings('ignore', message='invalid value encountered in divide')

phi0 = scipy.constants.value("mag. flux quantum")
savePath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Cooldown-B8", "G-analysis")


def find_npz_files(directory):
    npz_files = {}
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return npz_files

    pattern = os.path.join(directory, "*_iv_IV_*.npz")
    files = glob.glob(pattern)

    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.search(r'_IV_([\d\.]+)mK\.npz$', filename)
        if match:
            try:
                tbase_val = float(match.group(1))
                tbase = tbase_val / 1000.0
                npz_files[tbase] = file_path
            except ValueError:
                print(f"Warning: Could not extract valid Tbase from '{filename}'. Skipping.")
    return npz_files


def remove_offset(arr):
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return np.array([])
    if arr.size == 0 or arr.shape[1] == 0:
        return np.array([])

    last_elements = arr[:, -1]
    return arr - last_elements[:, np.newaxis]


def arb_to_amp(in_val, min_SI=250e-12):
    min_phi0_per_amp = min_SI / phi0
    amp_per_arb = 1 / min_phi0_per_amp
    return in_val * amp_per_arb


def Rtes(ibias, ites, Rshunt=250e-6):
    ites_safe = np.where(ites == 0, np.nan, ites)
    return Rshunt * (ibias - ites_safe) / ites_safe


def Ptes(ibias, ites, Rshunt=250e-6):
    return Rtes(ibias, ites, Rshunt) * ites ** 2


def convert_ang2_to_ites(ang2, channel_id, vb=None, min_SI=None):
    if min_SI is None:
        min_SI = CHIP_MIN_SI['A'] if channel_id < 32 else CHIP_MIN_SI['B']
    ites_uncorrected = arb_to_amp(ang2[:, channel_id], min_SI=min_SI)
    if vb is not None:
        zero_idx = np.argmin(np.abs(vb))
        return ites_uncorrected - ites_uncorrected[zero_idx]
    return ites_uncorrected - ites_uncorrected[-1]


def convert_vbias_to_ibias(vbias, rbias):
    return vbias / rbias


def calculate_rn(ibias, ites, Rshunt=250e-6):
    if len(ibias) < 100 or len(ites) < 100:
        return 10e-3
    rtes_values = Rtes(ibias[-100:], ites[-100:], Rshunt)
    rtes_values = rtes_values[np.isfinite(rtes_values)]
    if len(rtes_values) == 0:
        return 10e-3
    return np.nanmedian(rtes_values)


def estimate_rn_from_coldest_sweep(found_files, channel_id, rbias=1985.0, Rshunt=250e-6, skip_tbase=None):
    if skip_tbase is None:
        skip_tbase = []
    
    # 1. Find the coldest sweep file that is not skipped
    tbase_sorted = sorted(found_files.keys())
    coldest_fpath = None
    for tbase in tbase_sorted:
        # Check if rounded tbase is in skip_tbase (with tolerance)
        is_skipped = False
        for skip_t in skip_tbase:
            if np.abs(tbase - skip_t) < 0.001:
                is_skipped = True
                break
        if not is_skipped:
            coldest_fpath = found_files[tbase]
            break
            
    if coldest_fpath is None:
        return None
        
    try:
        # 2. Load the file
        data = np.load(coldest_fpath)
        vb = data['vb']
        ang2 = data['ang2']
        
        sort_idx = np.argsort(np.abs(vb))
        vb_sorted = vb[sort_idx]
        ibias = vb_sorted / rbias
        
        ites = convert_ang2_to_ites(ang2[sort_idx, :], channel_id, vb=vb_sorted)
        rtes = Rtes(ibias, ites, Rshunt)
        ptes = Ptes(ibias, ites, Rshunt)
        
        # 3. Filter valid finite points
        valid = np.isfinite(rtes) & np.isfinite(ptes) & (rtes > 0.1e-3)
        if np.sum(valid) < 50:
            return None
            
        rtes_v = rtes[valid]
        ptes_v = ptes[valid]
        
        # 4. Sort by Ptes
        sort_p = np.argsort(ptes_v)
        ptes_s = ptes_v[sort_p]
        rtes_s = rtes_v[sort_p]
        
        # 5. Fit spline to smooth the curve
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(ptes_s, rtes_s, s=1e-5)
        
        # Evaluate on a dense grid
        p_grid = np.linspace(ptes_s.min(), ptes_s.max(), 2000)
        r_fit = spl(p_grid)
        dr_dp = spl.derivative(1)(p_grid)
        
        # Limit to low power to find the transition inflection
        low_p_mask = p_grid < 4e-12
        if not np.any(low_p_mask):
            return None
            
        max_slope_idx = np.argmax(dr_dp[low_p_mask])
        p_inf = p_grid[low_p_mask][max_slope_idx]
        peak_slope = dr_dp[low_p_mask][max_slope_idx]
        
        # Find where slope drops to 8% of the peak slope after the inflection
        after_peak_mask = (p_grid > p_inf)
        below_thresh = dr_dp[after_peak_mask] < 0.08 * peak_slope
        if np.any(below_thresh):
            first_idx = np.where(below_thresh)[0][0]
            rn_estimate = r_fit[after_peak_mask][first_idx]
            return rn_estimate
            
    except Exception as e:
        print(f"Warning: Failed to estimate Rn using coldest sweep for channel {channel_id}: {e}")
        
    return None


def find_auto_tbase_max(npz_files, rbias, channel_id, Rn_fixed=None, Rshunt=250e-6, ratio=0.90):
    sorted_tbase = sorted(npz_files.keys())
    max_tbase = 0.020  # lowest possible default fallback
    for tbase in sorted_tbase:
        file_path = npz_files[tbase]
        try:
            with np.load(file_path) as data:
                if 'vb' not in data or 'ang2' not in data:
                    continue
                vbias, ang2 = data['vb'], data['ang2']
                if ang2.shape[1] <= channel_id:
                    continue
                ites = convert_ang2_to_ites(ang2, channel_id, vb=vbias)
                ibias = convert_vbias_to_ibias(vbias, rbias)
                
                rn = Rn_fixed if Rn_fixed is not None else calculate_rn(ibias, ites, Rshunt)
                rtes = Rtes(ibias, ites, Rshunt)
                
                # Check if transition is reachable (min rtes < rn * ratio)
                if np.nanmin(rtes) < rn * ratio:
                    max_tbase = max(max_tbase, tbase)
        except:
            continue
    return max_tbase


def find_bad_sweeps(npz_files, rbias, channel_id, Rn_fixed=None, Rshunt=250e-6):
    sorted_tbase = sorted(npz_files.keys())
    max_r_vals = {}
    for tbase in sorted_tbase:
        file_path = npz_files[tbase]
        try:
            with np.load(file_path) as data:
                if 'vb' not in data or 'ang2' not in data:
                    continue
                vbias, ang2 = data['vb'], data['ang2']
                if ang2.shape[1] <= channel_id:
                    continue
                ites = convert_ang2_to_ites(ang2, channel_id, vb=vbias)
                ibias = convert_vbias_to_ibias(vbias, rbias)
                rtes = Rtes(ibias, ites, Rshunt)
                
                valid = rtes[np.isfinite(rtes)]
                if len(valid) > 0:
                    max_r_vals[tbase] = np.nanmax(valid)
        except:
            continue
            
    if not max_r_vals:
        return []
        
    median_max_r = np.median(list(max_r_vals.values()))
    bad_tbases = []
    for tbase, max_r in max_r_vals.items():
        if np.abs(max_r - median_max_r) / median_max_r > 0.05:
            bad_tbases.append(tbase)
            
    return bad_tbases


def G_model(x, k, Tc, n):
    return k * (Tc**n - x**n)


def calculate_G(tbase_values, ptes_values, r_over_rn_ratio, channel_id, fit_opts=None):
    if len(tbase_values) < 3:
        return None, None, None

    if fit_opts is None:
        fit_opts = {}

    tc_guess_val = fit_opts.get('Tc_guess', 35.0) * 1e-3
    tc_min_val   = fit_opts.get('Tc_min',   30.0) * 1e-3
    tc_max_val   = fit_opts.get('Tc_max',   50.0) * 1e-3
    n_guess_val  = fit_opts.get('n_guess',   3.75)
    n_min_val    = fit_opts.get('n_min',     3.0)
    n_max_val    = fit_opts.get('n_max',     4)
    k_guess_val  = fit_opts.get('k_guess', 100.0) * 1e-9

    gmod = lmfit.Model(G_model)
    params = gmod.make_params(k=k_guess_val, Tc=tc_guess_val, n=n_guess_val)

    params['k'].min = 0
    params['Tc'].min = tc_min_val
    params['Tc'].max = tc_max_val
    params['n'].min = n_min_val
    params['n'].max = n_max_val

    try:
        result = gmod.fit(ptes_values, params, x=tbase_values)
        k = result.params['k'].value
        n = result.params['n'].value
        Tc = result.params['Tc'].value
        T_eval = G_T_EVAL

        G_eval = k * n * (T_eval ** (n - 1))

        k_err = result.params['k'].stderr
        n_err = result.params['n'].stderr
        Tc_err = result.params['Tc'].stderr

        if k_err is None or n_err is None or Tc_err is None:
            print("Warning: Could not estimate errors for all parameters.")
            return result, G_eval, None

        delta = 1e-14
        dG_dk = (G_model(T_eval, k + delta, Tc, n) - G_model(T_eval, k - delta, Tc, n)) / (2 * delta)
        dG_dTc = (G_model(T_eval, k, Tc + delta, n) - G_model(T_eval, k, Tc - delta, n)) / (2 * delta)
        dG_dn = (G_model(T_eval, k, Tc, n + delta) - G_model(T_eval, k, Tc, n - delta)) / (2 * delta)

        G_variance = (dG_dk * k_err) ** 2 + (dG_dTc * Tc_err) ** 2 + (dG_dn * n_err) ** 2

        if result.covar is not None:
            cov_kn = result.covar[0, 2]
            cov_kTc = result.covar[0, 1]
            cov_nTc = result.covar[2, 1]
            G_variance += 2 * (dG_dk * dG_dn * cov_kn + dG_dk * dG_dTc * cov_kTc + dG_dn * dG_dTc * cov_nTc)
        else:
            print("Warning: Covariance matrix not available. Error may be underestimated.")

        G_error = np.sqrt(G_variance)
        return result, G_eval, G_error

    except Exception as e:
        print(f"Error during fitting: {e}")
        return None, None, None


def calculate_G_all_ratios(npz_files, rbias, channel_id, r_over_rn_ratios, Rshunt=250e-6,
                           ptes_increase_threshold=0.05, tbase_max=None, Rn_fixed=None, skip_tbase=None, pixel_map=None, fit_opts=None):
    if tbase_max is None:
        tbase_max = TBASE_MAX.get(channel_id, "auto")
        
    if tbase_max == "auto":
        tbase_max = find_auto_tbase_max(npz_files, rbias, channel_id, Rn_fixed=Rn_fixed, Rshunt=Rshunt)
    results = {}

    for ratio in r_over_rn_ratios:
        tbase_values_filtered = []
        ptes_values_filtered = []
        sorted_tbase = sorted(npz_files.keys())

        for tbase in sorted_tbase:
            if skip_tbase is not None and np.round(tbase, 3) in np.round(skip_tbase, 3):
                continue
            file_path = npz_files[tbase]

            try:
                with np.load(file_path) as data:
                    if 'vb' not in data or 'ang2' not in data:
                        print(f"Error: 'vb'/'ang2' missing. Skipping.")
                        continue
                    vbias, ang2 = data['vb'], data['ang2']
                    if ang2.shape[1] <= channel_id:
                        print(f"Error: Channel {channel_id} out of range.")
                        continue
                    ites = convert_ang2_to_ites(ang2, channel_id, vb=vbias)
                    ibias = convert_vbias_to_ibias(vbias, rbias)

                    if Rn_fixed is None:
                        rn = calculate_rn(ibias, ites, Rshunt)
                    else:
                        rn = Rn_fixed
                    rtes = Rtes(ibias, ites, Rshunt)
                    target_rtes = rn * ratio
                    # Check if transition is reachable (min rtes < target_rtes)
                    if np.nanmin(rtes) < target_rtes:
                        idx = np.nanargmin(np.abs(rtes - target_rtes))
                        ptes = Ptes(ibias, ites, Rshunt)[idx]
                        if ptes > 0:
                            tbase_values_filtered.append(tbase)
                            ptes_values_filtered.append(ptes)
                        else:
                            break
                    else:
                        break

            except Exception as e:
                print(f"Error in file processing: {e}. Skipping.")
                continue

        if tbase_max is not None:
            tbase_fit = []
            ptes_fit = []
            for t, p in zip(tbase_values_filtered, ptes_values_filtered):
                if np.round(t,3) <= tbase_max:
                    tbase_fit.append(t)
                    ptes_fit.append(p)
            tbase_values_fit = np.array(tbase_fit)
            ptes_values_fit = np.array(ptes_fit)
        else:
            tbase_values_fit = np.array(tbase_values_filtered)
            ptes_values_fit = np.array(ptes_values_filtered)

        fit_result, G, G_error = calculate_G(tbase_values_fit, ptes_values_fit, ratio, channel_id, fit_opts=fit_opts)

        if fit_result:
            results[ratio] = {
                'G': G,
                'G_err': G_error,
                'k': fit_result.params['k'].value,
                'k_err': fit_result.params['k'].stderr,
                'Tc': fit_result.params['Tc'].value,
                'Tc_err': fit_result.params['Tc'].stderr,
                'n': fit_result.params['n'].value,
                'n_err': fit_result.params['n'].stderr,
                'fit_result': fit_result
            }
        else:
            results[ratio] = {}

    if SAVE_RESULTS and results:
        data_to_save = []
        for ratio, data in results.items():
            if 'G' in data and 'G_err' in data:
                pixel_number_for_channel = pixel_map[channel_id] if (pixel_map and channel_id in pixel_map) else str(channel_id)
                data_to_save.append({
                    'Pixel_Number': pixel_number_for_channel,
                    'Rtes/Rn': ratio,
                    'Rn': Rn_fixed if Rn_fixed is not None else rn,
                    'G': data['G'],
                    'G_err': data['G_err'],
                    'k': data['k'],
                    'k_err': data['k_err'],
                    'n': data['n'],
                    'n_err': data['n_err'],
                    'Tc': data['Tc'],
                    'Tc_err': data['Tc_err'],
                })

        if data_to_save:
            df = pd.DataFrame(data_to_save)
            df.to_csv(os.path.join(savePath, f'G_results_ch{channel_id}.csv'), index=False)
            print(f"Saved G results for channel {channel_id} to {os.path.join(savePath, f'G_results_ch{channel_id}.csv')}")
        else:
            print("No valid G results to save.")
    else:
        if not results:
            print("No fit results to save.")

    return results


def plot_fit_results(fit_results, channel_id):
    if not fit_results:
        print("Error: No fit results to plot.")
        return

    ratios = list(fit_results.keys())
    G_values = [fit_results[ratio].get('G', np.nan) for ratio in ratios]
    G_values = [v if v is not None else np.nan for v in G_values]
    G_errors = [fit_results[ratio].get('G_err', np.nan) for ratio in ratios]
    G_errors = [e if e is not None else np.nan for e in G_errors]
    k_values = [fit_results[ratio].get('k', np.nan) for ratio in ratios]
    k_values = [v if v is not None else np.nan for v in k_values]
    k_errors = [fit_results[ratio].get('k_err', np.nan) for ratio in ratios]
    k_errors = [e if e is not None else np.nan for e in k_errors]
    Tc_values = [fit_results[ratio].get('Tc', np.nan) for ratio in ratios]
    Tc_values = [v if v is not None else np.nan for v in Tc_values]
    Tc_errors = [fit_results[ratio].get('Tc_err', np.nan) for ratio in ratios]
    Tc_errors = [e if e is not None else np.nan for e in Tc_errors]
    n_values = [fit_results[ratio].get('n', np.nan) for ratio in ratios]
    n_values = [v if v is not None else np.nan for v in n_values]
    n_errors = [fit_results[ratio].get('n_err', np.nan) for ratio in ratios]
    n_errors = [e if e is not None else np.nan for e in n_errors]

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    axes[0].errorbar(ratios, np.array(G_values)*1e12, yerr=np.array(G_errors)*1e12, fmt='o-', capsize=5, label='G')
    axes[0].set_ylabel(f'G @ {G_T_EVAL*1e3:.0f} mK (pW/K)')
    axes[0].grid(True)

    axes[1].errorbar(ratios, np.array(k_values)*1e9, yerr=np.array(k_errors)*1e9, fmt='o-', capsize=5, label='k')
    axes[1].set_ylabel(r'k (nW/K$^\mathrm{n}$)')
    axes[1].grid(True)

    axes[2].errorbar(ratios, n_values, yerr=n_errors, fmt='o-', capsize=5, label='n')
    axes[2].set_ylabel('n')
    axes[2].grid(True)

    axes[3].errorbar(ratios, np.array(Tc_values)*1e3, yerr=np.array(Tc_errors)*1e3, fmt='o-', capsize=5, label='Tc')
    axes[3].set_ylabel('Tc (mK)')
    axes[3].set_xlabel('Rtes/Rn')
    axes[3].grid(True)

    plt.suptitle(f"Thermal fit parameters vs. Rtes/Rn (Channel {channel_id})")
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(savePath, f"G_fit_parameters_vs_ratio_ch{channel_id:02d}.png"), dpi=150)
        plt.close(fig)


def plot_ptes_vs_tbase_multiple_ratios(npz_files, rbias, channel_id, r_over_rn_ratios, Rshunt=250e-6, Rn_fixed=None,
                                       ptes_increase_threshold=0.05, tbase_max=None, skip_tbase=None, fit_opts=None):
    if tbase_max is None:
        tbase_max = TBASE_MAX.get(channel_id, "auto")
        
    if tbase_max == "auto":
        tbase_max = find_auto_tbase_max(npz_files, rbias, channel_id, Rn_fixed=Rn_fixed, Rshunt=Rshunt)
        print(f"Intelligently determined tbase_max for Channel {channel_id:02d}: {tbase_max*1e3:.1f} mK")
        
    if not npz_files:
        print("Error: No .npz files to plot.")
        return
    print(f"analyzing IV curves until tbase_max ={tbase_max:.3f} K")
    fit_results = calculate_G_all_ratios(npz_files, rbias, channel_id, r_over_rn_ratios,
                                         Rshunt, ptes_increase_threshold, tbase_max, Rn_fixed=Rn_fixed, skip_tbase=skip_tbase, fit_opts=fit_opts)

    fig = plt.figure(figsize=(6, 6), dpi=200)
    cmap = plt.get_cmap('coolwarm')
    num_ratios = len(r_over_rn_ratios)
    norm = mcolors.Normalize(vmin=min(r_over_rn_ratios) * 100, vmax=max(r_over_rn_ratios) * 100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i, ratio in enumerate(r_over_rn_ratios):
        tbase_values = []
        ptes_values = []
        sorted_tbase = sorted(npz_files.keys())
        color = cmap(i / (num_ratios - 1)) if num_ratios > 1 else cmap(0.5)

        for tbase in sorted_tbase:
            if skip_tbase is not None and np.round(tbase, 3) in np.round(skip_tbase, 3):
                continue
            if np.round(tbase, 3) > tbase_max:
                continue
            file_path = npz_files[tbase]
            try:
                with np.load(file_path) as data:
                    if 'vb' not in data or 'ang2' not in data:
                        continue
                    vbias, ang2 = data['vb'], data['ang2']
                    if ang2.shape[1] <= channel_id:
                        continue
                    ites = convert_ang2_to_ites(ang2, channel_id, vb=vbias)
                    ibias = convert_vbias_to_ibias(vbias, rbias)

                    if Rn_fixed is None:
                        rn = calculate_rn(ibias, ites, Rshunt)
                    else:
                        rn = Rn_fixed
                    rtes = Rtes(ibias, ites, Rshunt)
                    target_rtes = rn * ratio
                    # Check if transition is reachable (min rtes < target_rtes)
                    if np.nanmin(rtes) < target_rtes:
                        idx = np.nanargmin(np.abs(rtes - target_rtes))
                        ptes = Ptes(ibias, ites, Rshunt)[idx]
                        if ptes > 0:
                            tbase_values.append(tbase)
                            ptes_values.append(ptes)

            except Exception as e:
                continue

        plt.plot(np.array(tbase_values)*1e3, np.array(ptes_values)*1e12, marker='.', linestyle='',
                 label=f"{int(ratio*100)}", color=color, ms=3, lw=0.2)

        if ratio in fit_results and fit_results[ratio]:
            result = fit_results[ratio]['fit_result']
            tbase_values_all = np.array([t for t in sorted(npz_files.keys()) if np.round(t, 3) <= tbase_max])
            plt.plot(np.array(tbase_values_all)*1e3, G_model(np.array(tbase_values_all), **result.best_values)*1e12,
                     linestyle='-', color=color, lw=0.7)

    plt.xlabel(r"T$_\mathrm{bath}$ (mK)")
    plt.ylabel(r"P$_\mathrm{TES}$ (pW)")
    plt.grid(which="both", ls=":", lw=0.5, alpha=0.5)
    cax = plt.axes([0.82, 0.4, 0.03, 0.45])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('R/Rn (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.suptitle(f"Channel {channel_id:02d} P_TES vs T_bath")
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(savePath, f"G_fit_ptes_ch{channel_id:02d}.png"), dpi=150)
        plt.close(fig)

    plot_fit_results(fit_results, channel_id=channel_id)


def plot_channel_sweeps_3panel(npz_files, rbias, channel_id, Rshunt=250e-6, Rn_fixed=None, skip_tbase=None, zoom_limits=None):
    if not npz_files:
        print("Error: No .npz files to plot.")
        return

    if skip_tbase is None:
        tbase_values = sorted(npz_files.keys())
    else:
        tbase_values = sorted([t for t in npz_files.keys() if round(t, 3)
                               not in np.round(np.array(skip_tbase), 3)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    num_tbase_values = len(tbase_values)
    cmap = plt.get_cmap('coolwarm')

    _rn_line_val = None

    for i, tbase in enumerate(tbase_values):
        color = cmap(i / (num_tbase_values - 1)) if num_tbase_values > 1 else cmap(0.5)
        file_path = npz_files[tbase]
        try:
            with np.load(file_path) as data:
                if 'vb' not in data or 'ang2' not in data:
                    continue

                vbias = data['vb']
                ang2 = data['ang2']

                if ang2.shape[1] <= channel_id:
                    continue

                ites = convert_ang2_to_ites(ang2, channel_id, vb=vbias)
                ibias = convert_vbias_to_ibias(vbias, rbias)

                if _rn_line_val is None:
                    _rn_line_val = Rn_fixed if Rn_fixed is not None else calculate_rn(ibias, ites, Rshunt)

                axes[0].plot(ibias*1e3, ites*1e3, color=color, lw=1.2)
                if i == 0:
                    axes[0].plot(ibias * 1e3, ibias*1e3, color='k', linestyle='--', lw=0.75)

                rtes = Rtes(ibias, ites, Rshunt)
                axes[1].plot(ibias*1e3, rtes*1e3, color=color, lw=1.2)

                ptes = Ptes(ibias, ites, Rshunt)
                axes[2].plot(ptes*1e12, rtes*1e3, color=color, lw=1.2)

                # Add temperature label at the rightmost visual point of each curve
                idx_r_ib  = np.argmax(ibias)
                idx_r_pt  = np.argmax(ptes)
                axes[0].text(ibias[idx_r_ib]*1e3, ites[idx_r_ib]*1e3, f"{tbase*1e3:.0f}", color=color, fontsize=6, ha='left', va='center')
                axes[1].text(ibias[idx_r_ib]*1e3, rtes[idx_r_ib]*1e3, f"{tbase*1e3:.0f}", color=color, fontsize=6, ha='left', va='center')
                axes[2].text(ptes[idx_r_pt]*1e12,  rtes[idx_r_pt]*1e3, f"{tbase*1e3:.0f}", color=color, fontsize=6, ha='left', va='center')

        except Exception as e:
            print(f"Error loading/processing '{file_path}': {e}. Skipping.")

    if _rn_line_val is not None:
        axes[1].axhline(y=_rn_line_val*1e3, color='gray', linestyle='--', label=f"Rn ({_rn_line_val*1e3:.2f} mOhm)")
        axes[2].axhline(y=_rn_line_val*1e3, color='gray', linestyle='--', label=f"Rn ({_rn_line_val*1e3:.2f} mOhm)")
        axes[1].legend(loc='best', fontsize=8)
        axes[2].legend(loc='best', fontsize=8)

    # Zoom Limits
    if zoom_limits is not None:
        if 'ibias_mA' in zoom_limits:
            axes[0].set_xlim(zoom_limits['ibias_mA'])
            axes[1].set_xlim(zoom_limits['ibias_mA'])
        if 'ites_mA' in zoom_limits:
            axes[0].set_ylim(zoom_limits['ites_mA'])

    axes[0].set_xlabel("Ibias (mA)")
    axes[0].set_ylabel("Ites (mA)")
    axes[0].set_title(f"Ites vs. Ibias (Ch {channel_id:02d})")
    axes[0].grid(True, ls=':', alpha=0.5)

    axes[1].set_xlabel("Ibias (mA)")
    axes[1].set_ylabel("Rtes (mOhm)")
    axes[1].set_title(f"Rtes vs. Ibias (Ch {channel_id:02d})")
    axes[1].grid(True, ls=':', alpha=0.5)

    axes[2].set_xlabel("Ptes (pW)")
    axes[2].set_ylabel("Rtes (mOhm)")
    axes[2].set_title(f"Rtes vs. Ptes (Ch {channel_id:02d})")
    axes[2].grid(True, ls=':', alpha=0.5)

    norm = mcolors.Normalize(vmin=min(tbase_values)*1e3, vmax=max(tbase_values)*1e3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.subplots_adjust(top=0.9, bottom=0.15, right=0.91, left=0.06, wspace=0.25)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Bath Temperature (mK)")

    plt.suptitle(f"Channel {channel_id:02d} Sweeps vs T")

    if SAVE_PLOTS:
        plt.savefig(os.path.join(savePath, f"IV_and_Rtes_vs_Ibias_ch{channel_id:02d}.png"), dpi=150)
        plt.close(fig)


def _plot_param_heatmap(base_path, ratio, pixel_map,
                        val_col, err_col, unit, scale, title, cbar_label, filename, suffix=None):
    """Generic 4-row x 6-col heatmap with value ± 1σ error in each cell."""
    NROWS, NCOLS = 4, 6

    all_data = {}
    for fname in os.listdir(base_path):
        if fname.endswith(".csv") and fname.startswith("G_results_ch"):
            try:
                ch_id = int(fname.split("ch")[1].split(".")[0])
                df = pd.read_csv(os.path.join(base_path, fname), index_col='Rtes/Rn')
                if ratio in df.index:
                    all_data[ch_id] = df.loc[ratio]
            except (ValueError, IndexError) as e:
                print(f"Error processing {fname}: {e}")

    if not all_data:
        print("No CSV files found or no data at the specified ratio.")
        return

    heatmap_val = np.full((NROWS, NCOLS), np.nan)
    heatmap_err = np.full((NROWS, NCOLS), np.nan)
    heatmap_lbl = [["" for _ in range(NCOLS)] for _ in range(NROWS)]

    for ch, px_str in pixel_map.items():
        p = int(px_str)
        if p < 1 or p > NROWS * NCOLS:
            continue
        dcol    = (p - 1) // NROWS
        drow    = (p - 1) % NROWS
        arr_row = (NROWS - 1) - drow
        arr_col = dcol
        if ch in all_data:
            row = all_data[ch]
            heatmap_val[arr_row, arr_col] = row[val_col] * scale
            if err_col in row.index and not pd.isna(row[err_col]):
                heatmap_err[arr_row, arr_col] = row[err_col] * scale
        heatmap_lbl[arr_row][arr_col] = f"P{p} Ch{ch}"

    fig, ax = plt.subplots(figsize=(NCOLS * 2.2, NROWS * 2.0), dpi=150)
    im = ax.imshow(heatmap_val, cmap='viridis', origin='lower',
                   extent=[0, NCOLS, 0, NROWS], aspect='equal',
                   vmin=np.nanmin(heatmap_val), vmax=np.nanmax(heatmap_val))

    for r in range(NROWS):
        for c in range(NCOLS):
            val = heatmap_val[r, c]
            err = heatmap_err[r, c]
            lbl = heatmap_lbl[r][c]
            if not np.isnan(val):
                ax.text(c + 0.5, r + 0.5,
                        f"{lbl}\n{val:.1f} {unit}",
                        ha='center', va='center', color='white', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.45))
            elif lbl:
                ax.text(c + 0.5, r + 0.5, lbl,
                        ha='center', va='center', color='gray', fontsize=10)

    ax.set_xticks(np.arange(0.5, NCOLS + 0.5))
    ax.set_xticklabels([f"Col {i+1}" for i in range(NCOLS)])
    ax.set_yticks(np.arange(0.5, NROWS + 0.5))
    ax.set_yticklabels([f"Row {NROWS - i}" for i in range(NROWS)])
    ax.set_title(title, fontsize=13)
    chip_label = "chip B" if suffix == "chipB" else "chip A"
    fig.suptitle(f"dtest62 - {chip_label} (20 µm)", fontsize=14, fontweight='bold', y=1.02)
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    plt.tight_layout()

    outpath = os.path.join(base_path, filename)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {outpath}")


def plot_Tc_heatmap(base_path=savePath, ratio=0.9, pixel_map=None, suffix=None):
    filename = f'G_Tc_heatmap_ratio{ratio:.2f}.png'
    if suffix:
        filename = f'G_Tc_heatmap_{suffix}_ratio{ratio:.2f}.png'
    _plot_param_heatmap(
        base_path, ratio, pixel_map,
        val_col='Tc', err_col='Tc_err', unit='mK', scale=1e3,
        title=f'Tc Heatmap (mK) at R/Rn = {ratio*100:.0f}%',
        cbar_label='Tc (mK)',
        filename=filename,
        suffix=suffix
    )


def plot_G_heatmap(base_path=savePath, ratio=0.9, pixel_map=None, suffix=None):
    filename = f'G_G_heatmap_ratio{ratio:.2f}.png'
    if suffix:
        filename = f'G_G_heatmap_{suffix}_ratio{ratio:.2f}.png'
    _plot_param_heatmap(
        base_path, ratio, pixel_map,
        val_col='G', err_col='G_err', unit='pW/K', scale=1e12,
        title=f'G @ {G_T_EVAL*1e3:.0f} mK (pW/K) at R/Rn = {ratio*100:.0f}%',
        cbar_label='G (pW/K)',
        filename=filename,
        suffix=suffix
    )


def plot_Rn_heatmap(base_path=savePath, ratio=0.9, pixel_map=None, suffix=None):
    filename = f'G_Rn_heatmap_ratio{ratio:.2f}.png'
    if suffix:
        filename = f'G_Rn_heatmap_{suffix}_ratio{ratio:.2f}.png'
    _plot_param_heatmap(
        base_path, ratio, pixel_map,
        val_col='Rn', err_col='Rn_err', unit='mOhm', scale=1e3,
        title=f'Rn Heatmap (mOhm)',
        cbar_label='Rn (mOhm)',
        filename=filename,
        suffix=suffix
    )


def display_summary_plot(base_path=savePath, ratio=0.9, pixel_map=None, save_summary=False, suffix=None):
    all_data = {}
    for filename in os.listdir(base_path):
        if filename.endswith(".csv") and filename.startswith("G_results_ch"):
            try:
                channel_id = int(filename.split("ch")[1].split(".")[0])
                filepath = os.path.join(base_path, filename)
                df = pd.read_csv(filepath, index_col='Rtes/Rn')

                if ratio in df.index:
                    data_at_ratio = df.loc[ratio]
                    all_data[channel_id] = data_at_ratio

            except (ValueError, IndexError) as e:
                print(f"Error processing file {filename}: {e}")
                continue

    if not all_data:
        print("No CSV files found or no data at the specified ratio.")
        return

    if pixel_map:
        pixel_channel_pairs = [
            (int(pixel_map[ch]), ch) for ch in all_data.keys() if ch in pixel_map
        ]
        pixel_channel_pairs.sort(key=lambda x: x[0])
        x_labels = [p for p, _ in pixel_channel_pairs]
        sorted_channels = [ch for _, ch in pixel_channel_pairs]
    else:
        sorted_channels = sorted(all_data.keys())
        x_labels = sorted_channels

    if save_summary:
        rows = []
        for i, ch in enumerate(sorted_channels):
            pixel_num = x_labels[i]
            data = all_data[ch]
            g_val = (data['G'] * 1e12) if data.get('G') is not None else np.nan
            g_err = (data['G_err'] * 1e12) if data.get('G_err') is not None else np.nan
            n_val = data['n'] if data.get('n') is not None else np.nan
            n_err = data['n_err'] if data.get('n_err') is not None else np.nan
            k_val = (data['k'] * 1e9) if data.get('k') is not None else np.nan
            k_err = (data['k_err'] * 1e9) if data.get('k_err') is not None else np.nan
            tc_val = (data['Tc'] * 1e3) if data.get('Tc') is not None else np.nan
            tc_err = (data['Tc_err'] * 1e3) if data.get('Tc_err') is not None else np.nan
            cell_name = PIXEL_CELL_MAP.get(int(pixel_num), "Unknown")
            if suffix == "chipB":
                cell_name = cell_name.replace("20um_", "30um_")
            
            rows.append({
                "Pixel_Number": pixel_num,
                "CellName": cell_name,
                "Rtes/Rn Ratio": ratio,
                f"G_at_{G_T_EVAL*1e3:.0f}mK (pW/K)": g_val,
                f"G_err_at_{G_T_EVAL*1e3:.0f}mK (pW/K)": g_err,
                "n": n_val,
                "n_err": n_err,
                "k (nW/K^n)": k_val,
                "k_err (nW/K^n)": k_err,
                "Tc (mK)": tc_val,
                "Tc_err (mK)": tc_err,
            })
        summary_df = pd.DataFrame(rows)
        output_file = os.path.join(base_path, f"G_summary_{suffix}_ratio_{ratio:.2f}.csv" if suffix else f"G_summary_ratio_{ratio:.2f}.csv")
        summary_df.to_csv(output_file, index=False)
        print(f"Saved summary to {output_file}")

    fig, axes = plt.subplots(4, 1, figsize=(11, 16), sharex=True)
    fig_title = f"Thermal parameter summary plot ({suffix.upper() if suffix else 'ALL'}) at R/Rn = {ratio}"
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')

    items = ['G', 'n', 'k', 'Tc']
    units = ['(pW/K)', '', '(nW/K^n)', '(mK)']
    multipliers = [1e12, 1, 1e9, 1e3]

    for i, item in enumerate(items):
        values = [all_data[ch][item] for ch in sorted_channels]
        values = np.array([v if v is not None else np.nan for v in values]) * multipliers[i]
        errors = [all_data[ch][f"{item}_err"] for ch in sorted_channels]
        errors = np.array([e if e is not None else np.nan for e in errors]) * multipliers[i]
        pixels = np.array([int(p) for p in x_labels])

        # Plot grouped by CellName for distinct color coding
        legend_handles = {}
        for idx, p in enumerate(pixels):
            c_name = PIXEL_CELL_MAP.get(p, "Unknown")
            if suffix == "chipB":
                c_name = c_name.replace("20um_", "30um_")
            color = CELL_COLOR_MAP.get(c_name.replace("30um_", "20um_"), 'black')
            eb = axes[i].errorbar(
                p, values[idx], yerr=errors[idx],
                fmt='o', color=color, ecolor=color, capsize=4, ms=7, zorder=3
            )
            if c_name not in legend_handles:
                legend_handles[c_name] = eb

        axes[i].set_ylabel(f"{item} {units[i]}", fontsize=12, fontweight='bold')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].set_xticks(range(1, 25))

    axes[3].set_xlabel("Pixel Number", fontsize=13, fontweight='bold')

    # Add color-coded legend to top plot
    handles = []
    labels = []
    for c_name_base, color in CELL_COLOR_MAP.items():
        c_name = c_name_base.replace("20um_", "30um_") if suffix == "chipB" else c_name_base
        # Check if any active pixels have this cell name
        has_pixel = any((PIXEL_CELL_MAP.get(int(p), "").replace("20um_", "30um_") if suffix == "chipB" else PIXEL_CELL_MAP.get(int(p), "")) == c_name for p in x_labels)
        if has_pixel:
            h = matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8)
            handles.append(h)
            labels.append(c_name)

    axes[0].legend(handles, labels, bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=9, title='Pixel Geometry (CellName)')
    plt.tight_layout(rect=[0, 0.03, 0.82, 0.96])

    if SAVE_PLOTS:
        plot_filename = f"G_summary_plot_{suffix}_ratio_{ratio:.2f}.png" if suffix else f"G_summary_plot_ratio_{ratio:.2f}.png"
        plt.savefig(os.path.join(base_path, plot_filename), dpi=200)
        plt.close()
    else:
        plt.show()


def find_active_tes_channels(fpath, rbias=1980.0, Rshunt=250e-6):
    try:
        data = np.load(fpath)
        vb = data['vb']
        ang2 = data['ang2']
        n_steps, n_chan = ang2.shape
        
        sort_idx = np.argsort(np.abs(vb))
        vb_sorted = vb[sort_idx]
        ibias = convert_vbias_to_ibias(vb_sorted, rbias)
        
        active_channels = []
        for ch in range(n_chan):
            ites = convert_ang2_to_ites(ang2[sort_idx, :], ch, vb=vb_sorted)
            rtes = Rtes(ibias, ites, Rshunt)
            
            ites_range_uA = (np.max(ites) - np.min(ites)) * 1e6
            rtes_floor = np.nanmedian(rtes[5:30]) if len(rtes) > 30 else 0.0
            valid_rtes = rtes[~np.isnan(rtes)]
            
            has_swing = (10.0 < ites_range_uA < 2000.0)
            has_sc_state = np.any((rtes - rtes_floor) < 10e-6) if len(valid_rtes) > 0 else False
            
            if has_swing and has_sc_state:
                active_channels.append(ch)
        return np.array(active_channels)
    except Exception as e:
        print(f"Warning: Failed to auto-detect active TES channels: {e}")
        return np.arange(n_chan)


def main():
    data_directory = "C:/Users/anr29/Downloads/ravendata-dtest62/iv/"
    rbias_value = R_BIAS
    Rshunt_value = R_SHUNT
    
    found_files = find_npz_files(data_directory)
    if not found_files:
        print(f"Error: No sweep files found in directory {data_directory}")
        return
        
    tbase_sorted = sorted(found_files.keys())
    lowest_temp_fpath = found_files[tbase_sorted[0]]
    
    # Dynamic estimation of MIN_SI per SQUID chip once using base temp file
    try:
        data = np.load(lowest_temp_fpath)
        vb = data['vb']
        ang2 = data['ang2']
        sort_idx = np.argsort(np.abs(vb))
        vb_sorted = vb[sort_idx]
        ibias = vb_sorted / rbias_value
        sc_mask = np.abs(vb_sorted) < 0.2
        
        from scipy.constants import value as const_value
        phi0 = const_value("mag. flux quantum")
        
        chipA_vals = []
        chipB_vals = []
        for ch in range(ang2.shape[1]):
            ang2_ch = ang2[sort_idx, ch]
            if np.max(ang2_ch) - np.min(ang2_ch) > 10.0:
                try:
                    slope, _ = np.polyfit(ibias[sc_mask], ang2_ch[sc_mask], 1)
                    min_si = slope * phi0
                    if 100e-12 < min_si < 400e-12:
                        if ch < 32:
                            chipA_vals.append(min_si)
                        else:
                            chipB_vals.append(min_si)
                except:
                    continue
        if chipA_vals:
            CHIP_MIN_SI['A'] = np.median(chipA_vals)
        if chipB_vals:
            CHIP_MIN_SI['B'] = np.median(chipB_vals)
            
        print(f"Calculated Chip A MIN_SI: {CHIP_MIN_SI['A']*1e12:.2f} pH (from {len(chipA_vals)} channels)")
        print(f"Calculated Chip B MIN_SI: {CHIP_MIN_SI['B']*1e12:.2f} pH (from {len(chipB_vals)} channels)")
    except Exception as e:
        print(f"Warning: Failed to calculate chip MIN_SI: {e}. Using default 250 pH.")
    
    print(f"Auto-detecting active TES channels using base temp IV file: {lowest_temp_fpath} ...")
    active_channels = find_active_tes_channels(lowest_temp_fpath, rbias=rbias_value, Rshunt=Rshunt_value)
    print(f"Found {len(active_channels)} active channels: {active_channels}")
    
    # CHANNELS_TO_ANALYZE=[None] or [] both mean "run all active channels"
    filtered = [ch for ch in CHANNELS_TO_ANALYZE if ch is not None]
    if filtered:
        target_channels = [ch for ch in filtered if ch in active_channels]
        if not target_channels:
            print(f"Error: Specified CHANNELS_TO_ANALYZE {filtered} are not active (no IV curves found). Stopping execution.")
            return
    else:
        target_channels = list(active_channels)
        
    pixel_map = PIXEL_MAP
    r_over_rn_ratios = R_OVER_RN_RATIOS
    
    global savePath
    
    os.makedirs(savePath, exist_ok=True)
    
    for ch in target_channels:
        print(f"Running G fit for Channel {ch:02d} ...")
        
        skip_manual = EXCLUDE_TEMPS.get(ch, [])
        skip_auto = find_bad_sweeps(found_files, rbias_value, ch, Rn_fixed=RN_OVERRIDES.get(ch, None), Rshunt=Rshunt_value)
        skip = sorted(list(set(skip_manual + skip_auto)))
        if skip_auto:
            print(f"Intelligently detected bad sweeps (SQUID unlock) for Channel {ch:02d} at: " + 
                  ", ".join([f"{t*1000:.1f} mK" for t in sorted(skip_auto)]))
                  
        Rn_val = RN_OVERRIDES.get(ch, None)
        if Rn_val is None:
            # Dynamically estimate Rn using the coldest IV sweep inflection point algorithm
            Rn_val = estimate_rn_from_coldest_sweep(found_files, ch, rbias=rbias_value, Rshunt=Rshunt_value, skip_tbase=skip)
            if Rn_val is not None:
                print(f"Intelligently estimated Rn for Channel {ch:02d} using coldest sweep inflection point: {Rn_val*1e3:.2f} mOhm")
            else:
                print(f"Warning: Failed to estimate Rn for Channel {ch:02d}. Falling back to dynamic median estimation.")
        zoom = ZOOM_LIMITS.get(ch, {})
        fit_opts = FIT_OVERRIDES.get(ch, {})
        
        # 1. Ptes vs Tbase Fit Plot (internally calls plot_fit_results for parameters)
        plot_ptes_vs_tbase_multiple_ratios(
            found_files, 
            rbias=rbias_value, 
            channel_id=ch, 
            r_over_rn_ratios=r_over_rn_ratios, 
            Rshunt=Rshunt_value, 
            Rn_fixed=Rn_val,
            skip_tbase=skip,
            fit_opts=fit_opts,
            tbase_max=TBASE_MAX.get(ch, "auto")
        )
        
        # 2. 3-panel Sweeps Plot: Ites vs Ibias, Rtes vs Ibias, Rtes vs Ptes
        plot_channel_sweeps_3panel(
            found_files,
            rbias=rbias_value,
            channel_id=ch,
            Rn_fixed=Rn_val,
            skip_tbase=skip,
            zoom_limits=zoom,
            Rshunt=Rshunt_value
        )
        
    # Always generate array summary plots for Chip A and Chip B based on saved results
    print("Generating array summary plots and database records...")
    print("Generating summary for Chip A...")
    display_summary_plot(base_path=savePath, ratio=0.9, pixel_map=PIXEL_MAP, save_summary=SAVE_RESULTS, suffix="chipA")
    print("Generating summary for Chip B...")
    display_summary_plot(base_path=savePath, ratio=0.9, pixel_map=PIXEL_MAP_B, save_summary=SAVE_RESULTS, suffix="chipB")
    
    if not SAVE_PLOTS:
        plt.show()

if __name__ == "__main__":
    main()