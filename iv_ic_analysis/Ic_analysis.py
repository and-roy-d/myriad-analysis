#!/usr/bin/env python3
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import warnings
from scipy.optimize import curve_fit
import pandas as pd

def single_comp_model(T, Ic0, Tc0, n):
    """Single-component Ginzburg-Landau power-law model."""
    val = 1.0 - T / Tc0
    return np.where(val > 0, Ic0 * (val**n), 0.0)

def log_single_comp_model(T, log_Ic0, Tc0, n):
    """Single-component Ginzburg-Landau model in log space."""
    val = 1.0 - T / Tc0
    return np.where(val > 0, log_Ic0 + n * np.log(val), -20.0)

def two_comp_model(T, Ic1, Tc1, n1, Ic2, Tc2, n2):
    """Two-component Ginzburg-Landau model to capture bulk kink and proximity tail."""
    val1 = 1.0 - T / Tc1
    val2 = 1.0 - T / Tc2
    term1 = np.where(val1 > 0, Ic1 * (val1**n1), 0.0)
    term2 = np.where(val2 > 0, Ic2 * (val2**n2), 0.0)
    return term1 + term2

def log_two_comp_model(T, log_Ic1, Tc1, n1, log_Ic2, Tc2, n2):
    """Two-component Ginzburg-Landau model in log space."""
    val1 = 1.0 - T / Tc1
    val2 = 1.0 - T / Tc2
    term1 = np.where(val1 > 0, np.exp(log_Ic1) * (val1**n1), 0.0)
    term2 = np.where(val2 > 0, np.exp(log_Ic2) * (val2**n2), 0.0)
    return np.log(np.maximum(term1 + term2, 1e-10))

def find_noise_floor_cutoff(T_data_mK, Ic_data_uA, slope_threshold_frac=0.05):
    """
    Detect the last data point before Ic enters the noise-floor flatline.
    Returns a boolean mask (same shape as inputs) that is True for points
    in the active transition and False for the flatline tail.
    
    The flatline is defined as the region where each per-step |dIc|
    is less than slope_threshold_frac * max(|dIc|) across all steps.
    We find the last point (highest T) where the slope is still significant,
    and discard everything above it.
    """
    n = len(T_data_mK)
    if n < 4:
        return np.ones(n, dtype=bool)
    # Data must be sorted by ascending T for diffs to make sense
    dIc = np.diff(Ic_data_uA)            # negative during the transition
    abs_dIc = np.abs(dIc)
    max_drop = abs_dIc.max()
    if max_drop == 0:
        return np.ones(n, dtype=bool)
    threshold = slope_threshold_frac * max_drop
    # Scan backward to find the last step that is still "active"
    last_good = 0                          # index of last retained point
    for i in range(len(dIc) - 1, -1, -1):
        if abs_dIc[i] > threshold:
            last_good = i + 1              # include the endpoint of this step
            break
    mask = np.zeros(n, dtype=bool)
    mask[:last_good + 1] = True
    return mask

def load_tc_from_g_analysis(channel_id, base_path="C:/Users/anr29/Downloads/aroy/Cooldown-B8/G-analysis"):
    """Loads the Tc value for a channel from the G analysis files in Kelvin."""
    # First, try to load from individual channel CSV (which uses 'Tc' in Kelvin directly)
    indiv_path = os.path.join(base_path, f"G_results_ch{channel_id}.csv")
    if os.path.exists(indiv_path):
        try:
            df = pd.read_csv(indiv_path)
            row = df[np.round(df['Rtes/Rn'], 2) == 0.90]
            if not row.empty:
                return float(row['Tc'].values[0])
            return float(df['Tc'].mean())
        except:
            pass

    # Fallback to summary CSV (which uses Pixel_Number and 'Tc (mK)' in mK)
    suffix = "chipA" if channel_id < 32 else "chipB"
    summary_path = os.path.join(base_path, f"G_summary_{suffix}_ratio_0.90.csv")
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            pixel_map_dict = PIXEL_MAP if channel_id < 32 else PIXEL_MAP_B
            if channel_id in pixel_map_dict:
                pixel_num = int(pixel_map_dict[channel_id])
                row = df[df['Pixel_Number'] == pixel_num]
                if not row.empty:
                    col_name = 'Tc (mK)' if 'Tc (mK)' in df.columns else 'Tc'
                    val_mK = float(row[col_name].values[0])
                    # If value > 1.0, it's in mK, so convert to Kelvin
                    return val_mK / 1000.0 if val_mK > 1.0 else val_mK
        except:
            pass
            
    return None

# Suppress runtime optimization warnings cleanly
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')

# Initialize constants
phi0 = scipy.constants.value("mag. flux quantum")

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

CHIP_MIN_SI = {
    'A': 250e-12,
    'B': 250e-12
}

# Set up matplotlib configuration
import matplotlib
matplotlib.use('Agg')  # Headless generation via terminal
plt.rcParams['font.size'] = 12


def convert_ang2_to_ites(ang2, channel_id, correct_shift=True, vb=None):
    """Extracts phase angles and maps to calibrated TES current."""
    min_SI_val = CHIP_MIN_SI['A'] if channel_id < 32 else CHIP_MIN_SI['B']
    min_phi0_per_amp = min_SI_val / phi0
    amp_per_arb = 1.0 / min_phi0_per_amp
    ang2_channel = ang2[:, channel_id]
    ites_uncorrected = ang2_channel * amp_per_arb
    if correct_shift:
        if vb is not None:
            zero_idx = np.argmin(np.abs(vb))
            return ites_uncorrected - ites_uncorrected[zero_idx]
        else:
            return ites_uncorrected - ites_uncorrected[-1]
    return ites_uncorrected


def convert_vbias_to_ibias(vbias, rbias=1985.0):
    """Converts applied voltage drop to total circuit bias current."""
    return vbias / rbias


def Rtes(ibias, ites, Rshunt=250e-6):
    """Calculates active TES operating resistance."""
    ites_safe = np.where(ites == 0, np.nan, ites)
    return Rshunt * (ibias - ites_safe) / ites_safe


def extract_ic_by_rtes_threshold(vb, ang2, rbias=1985.0, Rshunt=250e-6, threshold=1e-6):
    """
    Finds Ic for each channel defined as the Ites current value where 
    Rtes crosses over the specified resistance threshold (e.g., 1 uOhm).
    """
    n_steps, n_chan = ang2.shape
    ic_currents = np.zeros(n_chan)
    ibias = convert_vbias_to_ibias(vb, rbias)
    
    # Sort arrays by increasing bias voltage magnitude
    sort_idx = np.argsort(np.abs(vb))
    vb_sorted = vb[sort_idx]
    ibias_sorted = ibias[sort_idx]
    
    for ch in range(n_chan):
        ites = convert_ang2_to_ites(ang2[sort_idx, :], ch, correct_shift=True, vb=vb_sorted)
        rtes = Rtes(ibias_sorted, ites, Rshunt)
        
        rtes_floor = np.nanmedian(rtes[5:30]) if len(rtes) > 30 else 0.0
        
        # Look for the transition point where rtes exceeds the floor + threshold
        normal_idx = np.where((rtes > rtes_floor + threshold) & (np.arange(len(rtes)) > 5))[0]
        if len(normal_idx) > 0:
            ic_currents[ch] = np.abs(ites[normal_idx[0]])
        else:
            # Fallback to absolute maximum derivative if superconducting state isn't broken cleanly
            max_deriv_idx = np.argmax(np.abs(np.diff(ites)))
            ic_currents[ch] = np.abs(ites[max_deriv_idx])
            
    return ic_currents


def find_active_tes_channels(fpath, rbias=1985.0, Rshunt=250e-6):
    """
    Identifies which channels have a functioning TES connected based on:
    1. The swing range of Ites current across the bias sweep (10 uA < range < 2000 uA).
    2. Min Rtes in superconducting state < 10 uOhm.
    """
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
            ites = convert_ang2_to_ites(ang2[sort_idx, :], ch, correct_shift=True, vb=vb_sorted)
            rtes = Rtes(ibias, ites, Rshunt)
            
            ites_range_uA = (np.max(ites) - np.min(ites)) * 1e6
            
            # Floor-subtracted valid_rtes
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





def plot_Ic_heatmap(output_dir, pixel_map, ic_values, suffix, base_temp_mK):
    """Generates 4-row x 6-col heatmap for base temp Ic."""
    NROWS, NCOLS = 4, 6
    
    heatmap_val = np.full((NROWS, NCOLS), np.nan)
    heatmap_lbl = [["" for _ in range(NCOLS)] for _ in range(NROWS)]
    
    for ch, px_str in pixel_map.items():
        p = int(px_str)
        if p < 1 or p > NROWS * NCOLS:
            continue
        dcol    = (p - 1) // NROWS
        drow    = (p - 1) % NROWS
        arr_row = (NROWS - 1) - drow
        arr_col = dcol
        if ch in ic_values:
            heatmap_val[arr_row, arr_col] = ic_values[ch] * 1e6  # convert to uA
        heatmap_lbl[arr_row][arr_col] = f"P{p} Ch{ch}"
        
    fig, ax = plt.subplots(figsize=(NCOLS * 2.2, NROWS * 2.0), dpi=150)
    im = ax.imshow(heatmap_val, cmap='viridis', origin='lower',
                   extent=[0, NCOLS, 0, NROWS], aspect='equal',
                   vmin=np.nanmin(heatmap_val), vmax=np.nanmax(heatmap_val))
                   
    for r in range(NROWS):
        for c in range(NCOLS):
            val = heatmap_val[r, c]
            lbl = heatmap_lbl[r][c]
            if not np.isnan(val):
                ax.text(c + 0.5, r + 0.5,
                        f"{lbl}\n{val:.1f} uA",
                        ha='center', va='center', color='white', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.45))
            elif lbl:
                ax.text(c + 0.5, r + 0.5, lbl,
                        ha='center', va='center', color='gray', fontsize=10)
                        
    ax.set_xticks(np.arange(0.5, NCOLS + 0.5))
    ax.set_xticklabels([f"Col {i+1}" for i in range(NCOLS)])
    ax.set_yticks(np.arange(0.5, NROWS + 0.5))
    ax.set_yticklabels([f"Row {NROWS - i}" for i in range(NROWS)])
    
    chip_label = "chip B" if suffix == "chipB" else "chip A"
    ax.set_title(f'Critical Current Ic Heatmap at {base_temp_mK:.1f} mK (uA)', fontsize=13)
    fig.suptitle(f"dtest62 - {chip_label} (20 µm)", fontsize=14, fontweight='bold', y=1.02)
    plt.colorbar(im, ax=ax, label=r'Ic ($\mu$A)', shrink=0.8)
    plt.tight_layout()
    
    outpath = os.path.join(output_dir, f"Ic_heatmap_{suffix}_{base_temp_mK:.1f}mK.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved Ic heatmap to {outpath}")


def plot_chip_results(output_dir, temperatures, ic_vs_t, all_iv_data, valid_indices, pixel_map, suffix, rbias, Rshunt):
    """Helper function to plot all results for a SQUID chip."""
    # Sort indices by ascending pixel number
    valid_indices = sorted(valid_indices, key=lambda ch: int(pixel_map[ch]))
    ic_base = ic_vs_t[0, :]
    # 1. Master Plot (loglog, color-coded by CellName label)
    plt.figure(figsize=(12, 7))
    added_labels = set()
    
    for ch in valid_indices:
        pixel_num = int(pixel_map[ch])
        cell_name = PIXEL_CELL_MAP.get(pixel_num, "Unknown")
        if suffix == "chipB":
            cell_name = cell_name.replace("20um_", "30um_")
        color = CELL_COLOR_MAP.get(cell_name.replace("30um_", "20um_"), 'gray')
        label = cell_name if cell_name not in added_labels else ""
        if label:
            added_labels.add(cell_name)
        plt.loglog(temperatures, ic_vs_t[:, ch] * 1e6, color=color, alpha=0.75, lw=1.5, label=label)
        
    plt.xlabel('Base Temperature (mK) [Log]', fontsize=12, fontweight='bold')
    plt.ylabel(r'Critical Current $I_c$ ($\mu$A) [Log]', fontsize=12, fontweight='bold')
    plt.title(f'Critical Current Temperature Dependence - {suffix.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=9, title='Pixel Geometry (CellName)')
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(os.path.join(output_dir, f"summary_Ic_vs_Temperature_{suffix}.png"), dpi=200)
    plt.close()
    # 2. IV vs T subplots — fixed 6-col x 4-row spatial pixel grid
    GRID_ROWS, GRID_COLS = 4, 6
    pixel_to_ch = {int(pixel_map[ch]): ch for ch in valid_indices}

    fig_iv, axes_iv = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(28, 18), sharex=True)
    for pixel_num in range(1, 25):
        row = (pixel_num - 1) % GRID_ROWS
        col = (pixel_num - 1) // GRID_ROWS
        ax = axes_iv[row, col]
        if pixel_num not in pixel_to_ch:
            ax.axis('off')
            continue
        ch = pixel_to_ch[pixel_num]
        temp_list = [item[0] for item in all_iv_data]
        t_min, t_max = min(temp_list), max(temp_list)
        for temp, vb, ang2 in all_iv_data:
            color_val = (temp - t_min) / (t_max - t_min) if t_max > t_min else 0.5
            color = plt.cm.coolwarm(color_val)
            ites = convert_ang2_to_ites(ang2, ch, correct_shift=True, vb=vb)
            ax.plot(vb, ites * 1e3, color=color, alpha=0.7)
        ax.set_title(f'P{pixel_num} (Ch{ch})', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.4)
        if col == 0:
            ax.set_ylabel('Ites (mA)', fontsize=11)
        if row == GRID_ROWS - 1:
            ax.set_xlabel('Vbias (V)', fontsize=11)
    plt.suptitle(f'{suffix.upper()} IV Temperature Evolution', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    page_png = os.path.join(output_dir, f"iv_vs_t_{suffix}_page_1.png")
    plt.savefig(page_png, dpi=200)
    plt.close(fig_iv)
    print(f"Saved 20-channel subplot grid sheet: {page_png}")

    # 3. Ic vs T subplots — same fixed 6-col x 4-row spatial pixel grid
    fig_ic, axes_ic = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(28, 18))
    for pixel_num in range(1, 25):
        row = (pixel_num - 1) % GRID_ROWS
        col = (pixel_num - 1) // GRID_ROWS
        ax = axes_ic[row, col]
        if pixel_num not in pixel_to_ch:
            ax.axis('off')
            continue
        ch = pixel_to_ch[pixel_num]

        T_data = temperatures
        Ic_data = ic_vs_t[:, ch]
        valid_mask = np.isfinite(Ic_data) & (Ic_data > 2.0e-6)
        T_data_clean = T_data[valid_mask]
        Ic_data_clean = Ic_data[valid_mask] * 1e6

        ax.loglog(T_data, ic_vs_t[:, ch] * 1e6, 'o', color='C0', ms=3)

        ic0_val, tc0_val, n_val, r2_val = np.nan, np.nan, np.nan, np.nan
        log_Ic_data_clean = np.log(Ic_data_clean) if len(Ic_data_clean) > 0 else np.array([])

        if len(T_data_clean) >= 6:
            tc_g = load_tc_from_g_analysis(ch)
            if tc_g is None:
                tc_g = 0.041
            try:
                idx_kink = np.argmin(np.abs(Ic_data_clean - 20.0))
                T_kink = T_data_clean[idx_kink]
                n_min, n_max = 0.2, 2
                tc1_lo = tc_g - 0.008
                tc1_hi = tc_g - 0.005
                popt, pcov = curve_fit(
                    log_two_comp_model, T_data_clean / 1000.0, log_Ic_data_clean,
                    p0=[np.log(Ic_data_clean.max()), tc_g - 0.006, 1.5, np.log(Ic_data_clean.max()*0.1), tc_g, 1.5],
                    bounds=(
                        [-20.0, tc1_lo, n_min, -20.0, tc_g * 0.95, 1.0],
                        [20.0,  tc1_hi, n_max, 20.0,  tc_g * 1.05, 2.0]
                    )
                )
                Ic1_fit = np.exp(popt[0]); Tc1_fit = popt[1] * 1000.0; n1_fit = popt[2]
                Ic2_fit = np.exp(popt[3]); Tc2_fit = popt[4] * 1000.0; n2_fit = popt[5]
                ic0_val = Ic1_fit + Ic2_fit; tc0_val = Tc2_fit; n_val = n2_fit
                residuals = Ic_data_clean - two_comp_model(T_data_clean / 1000.0, Ic1_fit, Tc1_fit/1000.0, n1_fit, Ic2_fit, Tc2_fit/1000.0, n2_fit)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((Ic_data_clean - np.mean(Ic_data_clean))**2)
                r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                t_grid = np.linspace(T_data_clean.min(), tc0_val * 0.999, 200)
                # C1=comp1 (red dashed), C2=comp2 (blue dashed), total (orange solid)
                ax.loglog(t_grid, single_comp_model(t_grid/1000.0, Ic1_fit, Tc1_fit/1000.0, n1_fit), color='tab:red', ls='--', lw=1.0)
                ax.loglog(t_grid, single_comp_model(t_grid/1000.0, Ic2_fit, Tc2_fit/1000.0, n2_fit), color='tab:blue', ls='--', lw=1.0)
                ax.loglog(t_grid, two_comp_model(t_grid/1000.0, Ic1_fit, Tc1_fit/1000.0, n1_fit, Ic2_fit, Tc2_fit/1000.0, n2_fit), color='tab:orange', lw=1.6)
                ax.axvline(x=tc_g * 1000.0, color='gray', ls=':', lw=0.8, alpha=0.7)
                ax.text(0.04, 0.22, f"comp1: {Ic1_fit:.0f}µA {Tc1_fit:.1f}mK n={n1_fit:.2f}",
                        color='tab:red', transform=ax.transAxes, fontsize=10, fontweight='bold')
                ax.text(0.04, 0.10, f"comp2: {Ic2_fit:.0f}µA {Tc2_fit:.1f}mK n={n2_fit:.2f}",
                        color='tab:blue', transform=ax.transAxes, fontsize=10, fontweight='bold')
            except:
                pass

        if np.isnan(ic0_val) and len(T_data_clean) >= 3:
            tc_g = load_tc_from_g_analysis(ch)
            if tc_g is None:
                tc_g = 0.041
            try:
                n_min, n_max = 0.3, 3.0
                popt, pcov = curve_fit(
                    log_single_comp_model, T_data_clean / 1000.0, log_Ic_data_clean,
                    p0=[np.log(Ic_data_clean.max()), tc_g, 1.5],
                    bounds=([-20.0, tc_g * 0.90, n_min], [20.0, tc_g * 1.10, n_max])
                )
                ic0_val = np.exp(popt[0]); tc0_val = popt[1] * 1000.0; n_val = popt[2]
                residuals = Ic_data_clean - single_comp_model(T_data_clean/1000.0, ic0_val, tc0_val/1000.0, n_val)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((Ic_data_clean - np.mean(Ic_data_clean))**2)
                r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                t_grid = np.linspace(T_data_clean.min(), tc0_val * 0.999, 200)
                ax.loglog(t_grid, single_comp_model(t_grid/1000.0, ic0_val, tc0_val/1000.0, n_val), color='tab:orange', lw=1.6)
                ax.axvline(x=tc_g * 1000.0, color='gray', ls=':', lw=0.8, alpha=0.7)
                ax.text(0.04, 0.10, f"main: {ic0_val:.0f}µA {tc0_val:.1f}mK n={n_val:.2f}",
                        color='tab:orange', transform=ax.transAxes, fontsize=10, fontweight='bold')
            except:
                pass

        title_str = f'P{pixel_num} (Ch{ch})'
        if not np.isnan(r2_val):
            title_str += f' R²={r2_val:.3f}'
        ax.set_title(title_str, fontsize=12)
        if col == 0:
            ax.set_ylabel(r'Ic (µA)', fontsize=11)
        if row == GRID_ROWS - 1:
            ax.set_xlabel('Tbase (mK)', fontsize=11)
        ax.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.suptitle(f'{suffix.upper()} Critical Current Ic vs Temperature', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    page_png = os.path.join(output_dir, f"ic_vs_t_{suffix}_page_1.png")
    plt.savefig(page_png, dpi=200)
    plt.close(fig_ic)
    print(f"Saved 20-channel Ic vs T subplot grid sheet: {page_png}")

    # 4. Heatmap
    base_temp_mK = temperatures[0]
    ic_base_dict = {ch: ic_base[ch] for ch in valid_indices if not np.isnan(ic_base[ch])}
    plot_Ic_heatmap(output_dir, pixel_map, ic_base_dict, suffix, base_temp_mK)


def analyze_run(data_dir, rbias=1985.0, Rshunt=250e-6, threshold=1e-6):
    file_pattern = os.path.join(data_dir, "*_*.npz")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No sweep arrays found inside target tree: {data_dir}")
        return

    parser_re = re.compile(r'_(IV|Ic)_([\d\.]+)mK\.npz')
    records = {'IV': [], 'Ic': []}
    
    for fpath in files:
        match = parser_re.search(os.path.basename(fpath))
        if match:
            sweep_type = match.group(1)
            temp_mK = float(match.group(2))
            records[sweep_type].append((temp_mK, fpath))

    # Sort files chronologically by temperature step
    for sweep_type in records:
        records[sweep_type].sort(key=lambda x: x[0])
        
    if not records['Ic']:
        print("No valid 'Ic' files found.")
        return

    # Dynamic calibration of MIN_SI per SQUID chip using the coldest IV sweep file
    iv_files = [f for f in files if "_IV_20.0mK.npz" in f]
    if iv_files:
        coldest_iv_fpath = iv_files[0]
        try:
            print(f"Calibrating SQUID mutual inductances from base temp IV sweep: {coldest_iv_fpath} ...")
            data_iv = np.load(coldest_iv_fpath)
            vb = data_iv['vb']
            ang2 = data_iv['ang2']
            sort_idx = np.argsort(np.abs(vb))
            vb_sorted = vb[sort_idx]
            ibias = vb_sorted / rbias
            sc_mask = np.abs(vb_sorted) < 0.2
            
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
            print(f"Warning: Failed to dynamically calibrate chip MIN_SI: {e}. Using default 250 pH.")
    else:
        print("Warning: Coldest IV sweep file not found. Using default 250 pH.")

    sample_data = np.load(records['Ic'][0][1])
    n_chan = sample_data['ang2'].shape[1]
    
    # Assume all channels are good as we'll find active ones dynamically
    good_channels = np.ones(n_chan, dtype=bool)
        
    # Auto-detect active TES channels at lowest temperature step
    active_indices = find_active_tes_channels(records['Ic'][0][1], rbias, Rshunt)
    valid_indices = np.intersect1d(np.where(good_channels)[0], active_indices)
    
    valid_indices_A = [ch for ch in valid_indices if ch in PIXEL_MAP]
    valid_indices_B = [ch for ch in valid_indices if ch in PIXEL_MAP_B]
    
    print(f"Analyzing {len(valid_indices)} valid signal channels out of {n_chan} total channels...")
    print(f"  Chip A: {len(valid_indices_A)} active channels")
    print(f"  Chip B: {len(valid_indices_B)} active channels")

    temperatures = []
    ic_vs_t = []
    all_iv_data = []

    for temp, fpath in records['Ic']:
        try:
            data = np.load(fpath)
            vb = data['vb']
            ang2 = data['ang2']
            
            # Extract Ic based on the Rtes criteria
            ic_array = extract_ic_by_rtes_threshold(vb, ang2, rbias, Rshunt, threshold)
            

            
            temperatures.append(temp)
            ic_vs_t.append(ic_array)
            all_iv_data.append((temp, vb, ang2))
            
            mean_valid_ic = np.nanmean(ic_array[valid_indices]) * 1e6
            print(f"Loaded Temp: {temp:5.1f} mK | Mean extracted Ic (valid): {mean_valid_ic:.3f} uA")
        except Exception as e:
            print(f"Skipping file {os.path.basename(fpath)}: {e}")

    temperatures = np.array(temperatures)
    ic_vs_t = np.array(ic_vs_t)

    # Set up output directory
    output_dir = "C:/Users/anr29/Downloads/aroy/Cooldown-B8/Ic-analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Generate Chip A plots
    if len(valid_indices_A) > 0:
        print("Generating summary plots and heatmaps for Chip A...")
        plot_chip_results(output_dir, temperatures, ic_vs_t, all_iv_data, valid_indices_A, PIXEL_MAP, "chipA", rbias, Rshunt)

    # Generate Chip B plots
    if len(valid_indices_B) > 0:
        print("Generating summary plots and heatmaps for Chip B...")
        plot_chip_results(output_dir, temperatures, ic_vs_t, all_iv_data, valid_indices_B, PIXEL_MAP_B, "chipB", rbias, Rshunt)

    # Collect fits for CSV export
    fit_records = []
    # Sort valid_indices by pixel number ascending
    valid_indices_sorted = sorted(valid_indices, key=lambda ch: int(PIXEL_MAP[ch] if ch in PIXEL_MAP else PIXEL_MAP_B[ch]))
    for ch in valid_indices_sorted:
        sub_suffix = "chipA" if ch in PIXEL_MAP else "chipB"
        sub_pixel_map = PIXEL_MAP if ch in PIXEL_MAP else PIXEL_MAP_B
        pixel_num = sub_pixel_map[ch]
        
        T_data = temperatures
        Ic_data = ic_vs_t[:, ch]
        valid_mask = np.isfinite(Ic_data) & (Ic_data > 2.0e-6)
        T_data_clean = T_data[valid_mask]
        Ic_data_clean = Ic_data[valid_mask] * 1e6

        ic0_val, tc0_val, n_val, r2_val = np.nan, np.nan, np.nan, np.nan
        ic1_val, tc1_val, n1_val = np.nan, np.nan, np.nan
        ic2_val, tc2_val, n2_val = np.nan, np.nan, np.nan
        ic1_err, tc1_err, n1_err = np.nan, np.nan, np.nan
        ic2_err, tc2_err, n2_err = np.nan, np.nan, np.nan
        log_Ic_data_clean = np.log(Ic_data_clean)
        fit_type = "None"
        tc_g_ch = load_tc_from_g_analysis(ch)

        if len(T_data_clean) >= 6:
            tc_g = tc_g_ch
            if tc_g is None:
                tc_g = 0.041
            try:
                idx_kink = np.argmin(np.abs(Ic_data_clean - 20.0))
                T_kink = T_data_clean[idx_kink]

                n_min, n_max = 0.3, 3.0
                tc1_lo = tc_g - 0.008
                tc1_hi = tc_g - 0.005
                popt, pcov = curve_fit(
                    log_two_comp_model, T_data_clean / 1000.0, log_Ic_data_clean,
                    p0=[np.log(Ic_data_clean.max()), tc_g - 0.006, 1.5, np.log(Ic_data_clean.max()*0.1), tc_g, 1.5],
                    bounds=(
                        [-20.0, tc1_lo, n_min, -20.0, tc_g * 0.95, 1.0],
                        [20.0,  tc1_hi, n_max, 20.0,  tc_g * 1.05, 2.0]
                    )
                )
                ic1_val = np.exp(popt[0])
                tc1_val = popt[1] * 1000.0
                n1_val = popt[2]
                ic2_val = np.exp(popt[3])
                tc2_val = popt[4] * 1000.0
                n2_val = popt[5]
                
                # Estimate 1-sigma errors from covariance matrix
                perr = np.sqrt(np.diag(pcov))
                ic1_err = ic1_val * perr[0]
                tc1_err = perr[1] * 1000.0
                n1_err = perr[2]
                ic2_err = ic2_val * perr[3]
                tc2_err = perr[4] * 1000.0
                n2_err = perr[5]
                
                ic0_val = ic1_val + ic2_val
                tc0_val = tc2_val
                n_val = n2_val
                
                residuals = Ic_data_clean - two_comp_model(T_data_clean / 1000.0, ic1_val, tc1_val / 1000.0, n1_val, ic2_val, tc2_val / 1000.0, n2_val)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((Ic_data_clean - np.mean(Ic_data_clean))**2)
                r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                fit_type = "Two-Component"
            except:
                pass
        
        if np.isnan(ic0_val) and len(T_data_clean) >= 3:
            tc_g = tc_g_ch
            if tc_g is None:
                tc_g = 0.041
            try:
                n_min, n_max = 0.3, 3.0
                popt, pcov = curve_fit(
                    log_single_comp_model, T_data_clean / 1000.0, log_Ic_data_clean, 
                    p0=[np.log(Ic_data_clean.max()), tc_g, 1.5], 
                    bounds=([-20.0, tc_g * 0.90, n_min], [20.0, tc_g * 1.10, n_max])
                )
                ic0_val = np.exp(popt[0])
                tc0_val = popt[1] * 1000.0
                n_val = popt[2]
                
                residuals = Ic_data_clean - single_comp_model(T_data_clean / 1000.0, ic0_val, tc0_val / 1000.0, n_val)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((Ic_data_clean - np.mean(Ic_data_clean))**2)
                r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                fit_type = "Single-Component"
            except:
                pass
                
        tc_g_ref = tc_g_ch * 1000.0 if tc_g_ch is not None else np.nan
        cell_name = PIXEL_CELL_MAP.get(int(pixel_num), "Unknown")
        if sub_suffix == "chipB":
            cell_name = cell_name.replace("20um_", "30um_")
        
        # Analyze LoPE weak-link behavior
        lope_res = analyze_lope(output_dir, T_data, Ic_data, pixel_num, sub_suffix)
        lope_a1 = lope_res['A_comp1'] if lope_res else np.nan
        lope_i0_1 = lope_res['I0_comp1_uA'] if lope_res else np.nan
        lope_a2 = lope_res['A_comp2'] if lope_res else np.nan
        lope_i0_2 = lope_res['I0_comp2_uA'] if lope_res else np.nan

        fit_records.append({
            'Pixel_Number': pixel_num,
            'CellName': cell_name,
            'Channel': ch,
            'Chip': sub_suffix.upper(),
            'Ic0_uA': round(ic0_val, 2) if not np.isnan(ic0_val) else np.nan,
            'Tc0_mK': round(tc0_val, 2) if not np.isnan(tc0_val) else np.nan,
            'n': round(n_val, 3) if not np.isnan(n_val) else np.nan,
            'Ic_comp1_uA': round(ic1_val, 2) if not np.isnan(ic1_val) else np.nan,
            'Ic_comp1_err_uA': round(ic1_err, 2) if not np.isnan(ic1_err) else np.nan,
            'Tc_comp1_mK': round(tc1_val, 2) if not np.isnan(tc1_val) else np.nan,
            'Tc_comp1_err_mK': round(tc1_err, 2) if not np.isnan(tc1_err) else np.nan,
            'n_comp1': round(n1_val, 3) if not np.isnan(n1_val) else np.nan,
            'n_comp1_err': round(n1_err, 3) if not np.isnan(n1_err) else np.nan,
            'Ic_comp2_uA': round(ic2_val, 2) if not np.isnan(ic2_val) else np.nan,
            'Ic_comp2_err_uA': round(ic2_err, 2) if not np.isnan(ic2_err) else np.nan,
            'Tc_comp2_mK': round(tc2_val, 2) if not np.isnan(tc2_val) else np.nan,
            'Tc_comp2_err_mK': round(tc2_err, 2) if not np.isnan(tc2_err) else np.nan,
            'n_comp2': round(n2_val, 3) if not np.isnan(n2_val) else np.nan,
            'n_comp2_err': round(n2_err, 3) if not np.isnan(n2_err) else np.nan,
            'Tc_G_mK': round(tc_g_ref, 2) if not np.isnan(tc_g_ref) else np.nan,
            'R2': round(r2_val, 4) if not np.isnan(r2_val) else np.nan,
            'Fit_Type': fit_type,
            'LoPE_A_comp1': round(lope_a1, 3) if not np.isnan(lope_a1) else np.nan,
            'LoPE_I0_comp1_uA': round(lope_i0_1, 2) if not np.isnan(lope_i0_1) else np.nan,
            'LoPE_A_comp2': round(lope_a2, 3) if not np.isnan(lope_a2) else np.nan,
            'LoPE_I0_comp2_uA': round(lope_i0_2, 2) if not np.isnan(lope_i0_2) else np.nan
        })
        
    df_fits = pd.DataFrame(fit_records)
    df_fits.to_csv(os.path.join(output_dir, "Ic_fit_parameters.csv"), index=False)
    print(f"Saved critical current fit parameters to {os.path.join(output_dir, 'Ic_fit_parameters.csv')}")

    # Generate parameter summary plots color-coded by CellName
    plot_ic_parameter_summary(output_dir, df_fits)


def plot_ic_parameter_summary(output_dir, df_fits):
    """Generates parameter summary plots color-coded by CellName for Chip A and Chip B."""
    for chip in ['CHIPA', 'CHIPB']:
        df_chip = df_fits[df_fits['Chip'] == chip].sort_values('Pixel_Number')
        if df_chip.empty:
            continue
            
        fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
        fig.suptitle(f"Critical Current Parameter Summary - {chip}", fontsize=16, fontweight='bold')
        
        pixels = df_chip['Pixel_Number'].values
        
        # 1. Tc panel (Tc_comp1, Tc_comp2, Tc_G)
        for _, row in df_chip.iterrows():
            p = int(row['Pixel_Number'])
            c_name = row['CellName']
            color = CELL_COLOR_MAP.get(c_name, 'black')
            
            # comp1 Tc (triangle down) & comp2 Tc (circle)
            if not np.isnan(row['Tc_comp1_mK']):
                axes[0].plot(p, row['Tc_comp1_mK'], 'v', color=color, ms=7, zorder=3)
            if not np.isnan(row['Tc_comp2_mK']):
                axes[0].plot(p, row['Tc_comp2_mK'], 'o', color=color, ms=8, zorder=3)
            if not np.isnan(row['Tc_G_mK']):
                axes[0].plot(p, row['Tc_G_mK'], 'x', color='black', ms=6, markeredgewidth=1.5, zorder=2)
                
        axes[0].set_ylabel("Tc (mK)", fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        
        # Add shape markers legend to panel 0
        m_comp1 = matplotlib.lines.Line2D([], [], color='gray', marker='v', linestyle='None', label='Tc (comp1)')
        m_comp2 = matplotlib.lines.Line2D([], [], color='gray', marker='o', linestyle='None', label='Tc (comp2)')
        m_g = matplotlib.lines.Line2D([], [], color='black', marker='x', linestyle='None', label='Tc (G-analysis)')
        axes[0].legend(handles=[m_comp1, m_comp2, m_g], loc='lower right', fontsize=9)

        # 2. Exponent n panel (n_comp1, n_comp2)
        for _, row in df_chip.iterrows():
            p = int(row['Pixel_Number'])
            c_name = row['CellName']
            color = CELL_COLOR_MAP.get(c_name, 'black')
            if not np.isnan(row['n_comp1']):
                axes[1].plot(p, row['n_comp1'], 'v', color=color, ms=7, zorder=3)
            if not np.isnan(row['n_comp2']):
                axes[1].plot(p, row['n_comp2'], 'o', color=color, ms=8, zorder=3)
                
        axes[1].set_ylabel("Exponent n", fontsize=12, fontweight='bold')
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # 3. Ic0 panel (Ic_comp1, Ic_comp2)
        for _, row in df_chip.iterrows():
            p = int(row['Pixel_Number'])
            c_name = row['CellName']
            color = CELL_COLOR_MAP.get(c_name, 'black')
            if not np.isnan(row['Ic_comp1_uA']):
                axes[2].plot(p, row['Ic_comp1_uA'], 'v', color=color, ms=7, zorder=3)
            if not np.isnan(row['Ic_comp2_uA']):
                axes[2].plot(p, row['Ic_comp2_uA'], 'o', color=color, ms=8, zorder=3)
                
        axes[2].set_ylabel(r"Ic0 ($\mu$A)", fontsize=12, fontweight='bold')
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].set_xlabel("Pixel Number", fontsize=13, fontweight='bold')
        axes[2].set_xticks(range(1, 25))

        # CellName Color Legend outside axes
        handles = []
        labels = []
        for c_name_base, color in CELL_COLOR_MAP.items():
            c_name = c_name_base.replace("20um_", "30um_") if chip == 'CHIPB' else c_name_base
            if c_name in df_chip['CellName'].values:
                h = matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8)
                handles.append(h)
                labels.append(c_name)

        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.90), loc='upper left', fontsize=9, title='Pixel Geometry (CellName)')
        plt.tight_layout(rect=[0, 0.02, 0.80, 0.95])
        out_path = os.path.join(output_dir, f"Ic_summary_plot_{chip.lower()}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved Ic parameter summary plot to {out_path}")


def analyze_lope(output_dir, temperatures, ic_data, pixel_num, suffix):
    """
    Calculates sqrt(T) and ln(Ic), performs piecewise linear fit (Component 1 = High-T, Component 2 = Low-T),
    extracts slopes (A1, A2), and plots results and residuals in a 1x2 figure.
    """
    valid_mask = np.isfinite(ic_data) & (ic_data > 2.0e-6)
    T_clean = temperatures[valid_mask]
    Ic_clean = ic_data[valid_mask] * 1e6  # in uA

    if len(Ic_clean) < 6:
        return None

    # T in Kelvin for physical unit compatibility (A in K^-1/2)
    T_K = T_clean / 1000.0
    x = np.sqrt(T_K)
    y = np.log(Ic_clean)

    # Separate into High-T (Component 1) and Low-T (Component 2) using 20 uA kink index
    idx_kink = np.argmin(np.abs(Ic_clean - 20.0))
    
    # Ensure both segments have at least 3 points
    if idx_kink < 3:
        idx_kink = 3
    if len(Ic_clean) - idx_kink < 3:
        idx_kink = len(Ic_clean) - 3

    x_low, y_low = x[:idx_kink+1], y[:idx_kink+1]
    x_high, y_high = x[idx_kink:], y[idx_kink:]

    try:
        p_low = np.polyfit(x_low, y_low, 1)
        p_high = np.polyfit(x_high, y_high, 1)
    except:
        return None

    A_comp2 = -p_low[0]      # Low-T slope (Component 2)
    I0_comp2 = np.exp(p_low[1])
    
    A_comp1 = -p_high[0]     # High-T slope (Component 1)
    I0_comp1 = np.exp(p_high[1])

    # Calculate residuals
    res_low = y_low - np.polyval(p_low, x_low)
    res_high = y_high - np.polyval(p_high, x_high)

    # Plot results in a 1x2 figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"LoPE Weak-Link Fit - Pixel {pixel_num} ({suffix})", fontsize=14, fontweight='bold')

    # Left subplot: transformed data and linear fit lines
    ax1.plot(x, y, 'o', color='gray', alpha=0.6, label='Data')
    ax1.plot(x_high, np.polyval(p_high, x_high), '-', color='tab:red', lw=2,
             label=f'High-T (Comp 1): A={A_comp1:.2f} K$^{{-1/2}}$, I0={I0_comp1:.1f} µA')
    ax1.plot(x_low, np.polyval(p_low, x_low), '-', color='tab:blue', lw=2,
             label=f'Low-T (Comp 2): A={A_comp2:.2f} K$^{{-1/2}}$, I0={I0_comp2:.1f} µA')
    ax1.set_xlabel(r'$\sqrt{T}$ (K$^{1/2}$)', fontsize=12)
    ax1.set_ylabel(r'$\ln(I_c)$ ($\ln(\mu\mathrm{A})$)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10, loc='best')

    # Right subplot: residuals
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.plot(x_high, res_high, 'v', color='tab:red', label='High-T (Comp 1) Residuals')
    ax2.plot(x_low, res_low, 'o', color='tab:blue', label='Low-T (Comp 2) Residuals')
    ax2.set_xlabel(r'$\sqrt{T}$ (K$^{1/2}$)', fontsize=12)
    ax2.set_ylabel(r'Residuals ($\ln(I_{c,\mathrm{meas}}) - \ln(I_{c,\mathrm{fit}})$)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plot_dir = os.path.join(output_dir, "lope-analysis")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"lope_fit_pixel_{pixel_num}_{suffix.lower()}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        'A_comp1': A_comp1,
        'I0_comp1_uA': I0_comp1,
        'A_comp2': A_comp2,
        'I0_comp2_uA': I0_comp2
    }


if __name__ == "__main__":
    # Point directly to your active data target directory
    target_dir = "C:/Users/anr29/Downloads/ravendata-dtest62/iv/"
    analyze_run(target_dir, rbias=1985.0, Rshunt=250e-6, threshold=1e-6)