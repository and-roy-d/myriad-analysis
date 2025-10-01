import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from past.translation import transform
from scipy.optimize import curve_fit, brentq
from scipy.special import gammaln
import time

# --- Matplotlib settings ---
# plt.rcParams['font.size'] = 14
ieee_single_col_width = 4 # inches
ieee_double_col_width = 7.2  # inches
plt.rcParams.update({
    'font.size': 8,                      # Base font size
    'axes.labelsize': 9,                 # X and Y axis labels
    'axes.titlesize': 0,                 # Subplot titles
    'xtick.labelsize': 8,                # X-axis tick labels
    'ytick.labelsize': 8,                # Y-axis tick labels
    'legend.fontsize': 'medium',          # Legend font size
    'figure.figsize': (ieee_single_col_width, 3), # Default figure size
    'figure.dpi': 200,
    'lines.linewidth': 1.0,
    'lines.markersize': 2,
    'font.family': 'sans-serif',
    'text.usetex': False,
})
cmap = plt.get_cmap('tab10')
state_colors = {}

# --- Constants ---
# !!! ADJUST THIS VALUE AS NEEDED !!!
Eph_global = 1239 / 515  # eV Example photon energy constant
CURRENT_DATE = time.strftime("%Y-%m-%d") # Use current date provided by context if relevant, else system time

import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, brentq
from scipy.special import gammaln
import time



# # --- Constants ---
# # !!! ADJUST THIS VALUE AS NEEDED !!!
# Eph_global = 1239 / 515  # eV Example photon energy constant
# # Get current date using system time as context doesn't always provide it
# # Use user context if available, otherwise use system time.
# CURRENT_DATE_CTX = "2024-12-16" # From user context provided previously
# try:
#     # Attempt to parse context date, fallback to system time if format is wrong or missing
#     time.strptime(CURRENT_DATE_CTX, "%Y-%m-%d")
#     CURRENT_DATE = CURRENT_DATE_CTX
#     print(f"Using date from context: {CURRENT_DATE}")
# except (ValueError, TypeError):
#     CURRENT_DATE = time.strftime("%Y-%m-%d")
#     print(f"Context date not usable, using system date: {CURRENT_DATE}")



def load_and_combine_data(path, states_to_skip=None, min_amplitude_threshold=0.0):
    """
    Loads and combines 'fit_results_state*.csv' files from a directory.
    Filters rows within each state based on a minimum amplitude value threshold.

    Args:
        path (str): The directory containing the CSV files.
        states_to_skip (list, optional): A list of state characters to ignore completely.
                                         Defaults to None (process all states).
        min_amplitude_threshold (float, optional): The minimum 'ampl' value required
                                                   for a row to be kept. Rows with
                                                   amplitude below this (or NaN/non-numeric)
                                                   will be dropped. Defaults to 0.0.

    Returns:
        pd.DataFrame: A combined DataFrame containing data from valid states AFTER
                      row filtering, or an empty DataFrame if no valid data remains.
    """
    if states_to_skip is None:
        states_to_skip = []

    csv_files = glob.glob(os.path.join(path, 'fit_results_state*.csv'))
    print(f"[{CURRENT_DATE}] Found {len(csv_files)} CSV files in '{path}'.")
    print(f"[{CURRENT_DATE}] Applying minimum amplitude value threshold: {min_amplitude_threshold}")
    global state_colors # Allow modification of the global variable

    # Filter files based on states_to_skip *before* processing
    valid_files = []
    skipped_states_from_arg = set()
    for f in csv_files:
        state_char = os.path.basename(f).replace('.csv', '')[-1]
        if state_char not in states_to_skip:
            valid_files.append(f)
        else:
            skipped_states_from_arg.add(state_char)

    if skipped_states_from_arg:
        print(f"[{CURRENT_DATE}] Skipping entire states based on 'states_to_skip' input: {sorted(list(skipped_states_from_arg))}")

    if not valid_files:
        print(f"[{CURRENT_DATE}] No CSV files remaining after initial skip filter.")
        return pd.DataFrame()

    # Determine states to process *after* initial filtering
    all_states_to_process = sorted([os.path.basename(f).replace('.csv', '')[-1] for f in valid_files])
    # Initialize colors for all potentially processable states first
    state_colors = {state: cmap(i % cmap.N) for i, state in enumerate(all_states_to_process)}
    print(f"[{CURRENT_DATE}] Attempting to process states: {all_states_to_process}")

    # Initialize plots
    fig_diff, ax_diff = plt.subplots(); fig_cent, ax_cent = plt.subplots()

    df_combined = pd.DataFrame()
    avg_n_dict = {}
    states_skipped_post_filter = set() # Track states skipped because they became empty

    for csv_file in valid_files: # Iterate through potentially valid files
        state = os.path.basename(csv_file).replace('.csv', '')[-1]
        if state in states_to_skip: continue # Redundant but safe

        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print(f"[{CURRENT_DATE}] Skipping empty file: {csv_file}")
                continue

            original_row_count = len(df)
            print(f"[{CURRENT_DATE}] Processing state '{state}' from {os.path.basename(csv_file)} (Initial rows: {original_row_count})...")

            # --- Amplitude Value Filtering ---
            if 'ampl' in df.columns:
                # Convert 'ampl' to numeric, coercing errors to NaN
                df['ampl_numeric'] = pd.to_numeric(df['ampl'], errors='coerce')

                # Filter rows: keep if 'ampl_numeric' is not NaN AND >= threshold
                rows_before_filter = len(df)
                filter_mask = (df['ampl_numeric'] >= min_amplitude_threshold) & pd.notna(df['ampl_numeric'])
                df = df[filter_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
                rows_after_filter = len(df)
                rows_dropped = rows_before_filter - rows_after_filter

                if rows_dropped > 0:
                    print(f"  State '{state}': Dropped {rows_dropped} rows due to amplitude < {min_amplitude_threshold} or non-numeric 'ampl'.")

                # Check if the DataFrame is empty *after* filtering
                if df.empty:
                    print(f"  State '{state}': Became empty after amplitude filtering. Skipping this state.")
                    states_skipped_post_filter.add(state)
                    if state in state_colors: del state_colors[state] # Remove color if state is skipped
                    continue # Skip to the next file
            else:
                 # If 'ampl' column is missing, we can't filter or calculate weighted avg_n reliably. Skip state.
                 print(f"[{CURRENT_DATE}] Warning: 'ampl' column not found in {csv_file}. Cannot apply threshold or calculate weighted avg_n. Skipping state '{state}'.")
                 states_skipped_post_filter.add(state)
                 if state in state_colors: del state_colors[state]
                 continue

            # --- If DataFrame still has rows after filtering ---
            df['state'] = state

            # Calculate avg_n using the *filtered* DataFrame
            # 'ampl_numeric' column already exists from the filtering step
            valid_ampl_filtered = df['ampl_numeric'] # These are already >= threshold and not NaN
            if not valid_ampl_filtered.empty and valid_ampl_filtered.sum() > 1e-9:
                 if 'n' in df.columns:
                     df['n'] = pd.to_numeric(df['n'], errors='coerce')
                     # We only need to check for non-NaN 'n' now, as 'ampl' is already filtered
                     valid_n_mask = pd.notna(df['n'])
                     n_values = df.loc[valid_n_mask, 'n']
                     ampl_values = valid_ampl_filtered[valid_n_mask] # Use corresponding filtered amplitudes
                     if not ampl_values.empty and ampl_values.sum() > 1e-9:
                         avg_n = np.sum(n_values * ampl_values) / ampl_values.sum()
                     elif 'n' in df.columns: # Fallback 1: simple mean of remaining 'n'
                         avg_n = np.mean(df['n'].dropna())
                     else: # Fallback 2: 'n' missing or all NaN
                         avg_n = np.nan
                 else: # 'n' column missing
                      avg_n = np.nan
            elif 'n' in df.columns: # Fallback if sum is too small or ampl empty (shouldn't happen often after filter)
                avg_n = np.mean(df['n'].dropna())
            else:
                avg_n = np.nan

            if np.isnan(avg_n):
                print(f"  State '{state}': Warning: Could not calculate 'avg_n' even after filtering. Check 'n' column.")

            avg_n_dict[state] = avg_n
            df['avg_n'] = avg_n
            color = state_colors.get(state, 'gray') # Get color for plotting

            # --- Raw data overview plotting (using filtered df) ---
            if 'n' in df.columns and 'mu' in df.columns:
                # df is already filtered by amplitude, just dropna for n/mu for plotting
                df_plot = df.dropna(subset=['n', 'mu'])
                if not df_plot.empty:
                    ax_cent.plot(df_plot['n'], df_plot['mu'], 'o-', color=color, label=state, markersize=4, alpha=0.7)
                    if len(df_plot) > 1:
                       diffs = np.diff(df_plot['mu'])/np.diff(df_plot['n'])
                       # n_mid = (df_plot['n'].iloc[:-1].values + df_plot['n'].iloc[1:].values) / 2
                       ax_diff.errorbar(df_plot['mu'][1::], diffs, xerr=df_plot['muStd'][1::], fmt = 'o', color=color, label=state, markersize=4, alpha=0.7)

            # --- Prepare columns for df_combined ---
            # Use the original 'ampl' column from the filtered df
            required_cols = ['n', 'mu', 'muStd', 'Es', 'ampl', 'state', 'avg_n', 'fwhm', 'fwhmStd']
            for col in required_cols:
                if col not in df.columns:
                    # This might happen if original CSV was missing cols, unlikely after filtering check
                    print(f"  State '{state}': Warning: Column '{col}' missing post-filter. Adding as NaN.")
                    df[col] = np.nan

            # Ensure essential columns are numeric (use original 'ampl' column)
            numeric_cols = ['Es', 'mu', 'muStd', 'avg_n', 'fwhm', 'fwhmStd', 'n', 'ampl']
            for col in numeric_cols:
                 if col in df.columns:
                     df[col] = pd.to_numeric(df[col], errors='coerce')

            # Select and append required columns from the filtered df
            df_to_append = df[required_cols].copy()
            df_combined = pd.concat([df_combined, df_to_append], ignore_index=True)

        except Exception as e:
            print(f"[{CURRENT_DATE}] Error processing {csv_file} for state '{state}': {e}")
            # Remove color if error occurs
            if state in state_colors: del state_colors[state]
            states_skipped_post_filter.add(state) # Treat errors as skipped states


    if states_skipped_post_filter:
        print(f"[{CURRENT_DATE}] States skipped because they became empty after amplitude filtering or had errors: {sorted(list(states_skipped_post_filter))}")

    # --- Final Data Cleanup and Plotting ---
    if not df_combined.empty:
        final_states_in_data = sorted(df_combined['state'].unique())
        # Update state_colors to only include states actually in the final dataframe
        state_colors = {state: cmap(i % cmap.N) for i, state in enumerate(final_states_in_data)}
        print(f"[{CURRENT_DATE}] States included in final combined DataFrame (after row filtering): {final_states_in_data}")

        # Drop rows missing essential data needed for fitting AFTER combining and row filtering
        essential_subset = ['Es', 'mu', 'muStd', 'avg_n', 'fwhm', 'fwhmStd']
        initial_rows = len(df_combined)
        df_combined.dropna(subset=essential_subset, inplace=True)
        rows_dropped_final = initial_rows - len(df_combined)
        if rows_dropped_final > 0:
            print(f"[{CURRENT_DATE}] Dropped {rows_dropped_final} additional rows due to NaNs in essential columns ({essential_subset}) before fitting.")

        # Finalize raw data overview plots only if data remains
        if not df_combined.empty:
            ax_diff.set(xlabel='PH (5lagy; arb. units)', ylabel=r'$\Delta$PH (a.u.)', title='Successive differences in pulse height')
            ax_cent.set(xlabel='n', ylabel='Pulse Height (arb. units)', title='Centroid vs n (Amp Filtered)')
            for ax in [ax_diff, ax_cent]:
                ax.grid(True, linestyle=':')
                handles, labels = ax.get_legend_handles_labels()
                # Filter legend to only include states present in the final data
                valid_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in state_colors]
                if valid_handles_labels:
                    handles, labels = zip(*valid_handles_labels)
                    by_label = dict(zip(labels, handles)) # Ensure unique labels
                    ax.legend(by_label.values(), by_label.keys(), fontsize='small')
                else: # Clear legend if no valid states remain
                     if ax.get_legend() is not None: ax.get_legend().remove()

            fig_diff.suptitle('Raw Data Overview (Filtered by Amplitude Value)'); fig_cent.suptitle('Raw Data Overview (Filtered by Amplitude Value)')
            fig_diff.tight_layout(rect=[0, 0.03, 1, 0.95]); fig_cent.tight_layout(rect=[0, 0.03, 1, 0.95])
            print(f"[{CURRENT_DATE}] Raw data overview plots generated.")
        else:
            print(f"[{CURRENT_DATE}] Combined DataFrame became empty after final dropping of NaNs. Closing raw data plots.")
            plt.close(fig_diff); plt.close(fig_cent)
    else:
        print(f"[{CURRENT_DATE}] Warning: Combined DataFrame is empty after processing all files (due to row filtering or initial skips). Closing raw data plots.")
        plt.close(fig_diff); plt.close(fig_cent)

    return df_combined

def sqrt_linear_func_fwhm(E, p0_sq, p1):
    """MODIFIED: FWHM fit is first order in energy inside the sqrt."""
    arg = p0_sq + p1 * E
    return np.sqrt(np.maximum(arg, 0))


def poly_func_f_new(x_composite, k, *coeffs):

    E_vals = x_composite[:, 0]; avg_n_vals = x_composite[:, 1]; order = len(coeffs)
    global Eph_global;
    if Eph_global is None: raise ValueError("Eph_global constant is not set!")
    argument = E_vals + k * Eph_global * avg_n_vals
    result = np.zeros_like(E_vals, dtype=float)
    for i in range(order): result += coeffs[i] * (argument**(order - i))
    return result


def poly_func_f_new_deriv(x_composite, k, *coeffs):

    E_vals = x_composite[:, 0]; avg_n_vals = x_composite[:, 1]; order = len(coeffs)
    global Eph_global;
    if Eph_global is None: raise ValueError("Eph_global constant is not set!")
    argument = E_vals + k * Eph_global * avg_n_vals
    deriv_result = np.zeros_like(E_vals, dtype=float)
    for i in range(order):
        power = order - i; coeff = coeffs[i]
        if power >= 1: deriv_result += coeff * power * (argument**(power - 1))
    deriv_result[np.isnan(deriv_result)] = 0
    return deriv_result


def poly_func_simple_g(E_vals, *coeffs):
    # (Definition same)
    order = len(coeffs); result = np.zeros_like(E_vals, dtype=float)
    for i in range(order): result += coeffs[i] * (E_vals**(order - i))
    return result


def poly_func_simple_g_deriv(E_vals, *coeffs):
    # (Definition same)
    order = len(coeffs); deriv_result = np.zeros_like(E_vals, dtype=float)
    for i in range(order):
        power = order - i; coeff = coeffs[i]
        if power >= 1: deriv_result += coeff * power * (E_vals**(power - 1))
    deriv_result[np.isnan(deriv_result)] = 0
    return deriv_result

def poly_func_avg_gain(E_vals, *coeffs):
    """Calculates (c_N*E^N + ... + c_1*E^1) / E = c_N*E^(N-1) + ... + c_1"""
    order = len(coeffs) # N = number of coeffs
    # Ensure E_vals is float to avoid potential type issues
    E_vals_float = np.array(E_vals, dtype=float)
    avg_gain = np.zeros_like(E_vals_float)

    # Calculate sum: c_N*E^(N-1) + ... + c_2*E^1 + c_1
    for i in range(order):
        power_in_g = order - i # N-i (power in g(E))
        coeff = coeffs[i]
        if power_in_g >= 1:
            power_in_avg_gain = power_in_g - 1 # N-i-1
            # Use np.power carefully to handle E=0
            # Calculate term only where E is not zero
            term = np.zeros_like(E_vals_float)
            non_zero_mask = E_vals_float != 0
            term[non_zero_mask] = coeff * np.power(E_vals_float[non_zero_mask], power_in_avg_gain)

            # For the c1 term (power_in_avg_gain=0), add coeff where E=0
            if power_in_avg_gain == 0:
                 term[E_vals_float == 0] = coeff # Set c1 where E=0

            avg_gain += term

    # Replace potential NaNs resulting from 0^negative_power if any snuck through
    avg_gain[np.isnan(avg_gain)] = 0 # Safeguard
    return avg_gain

if __name__ == '__main__':
    # --- Configuration ---
    data_path = '/home/pcuser/Runs/Cooldown_A12/' # !!! ADJUST THIS PATH !!!
    states_to_skip = ['M']
    amplitude_threshold = 4
    polynomial_order = 7
    initial_k_guess = 0.004
    energy_upper_threshold = 750.0
    E_highlight = np.array([277.0])
    print(f"[{CURRENT_DATE}] Using Eph_global = {Eph_global:.4f} eV")

    # --- Load Data ---
    df_combined = load_and_combine_data(data_path,
                                        states_to_skip=states_to_skip,
                                        min_amplitude_threshold=amplitude_threshold)
    if df_combined.empty:
        print(f"\n[{CURRENT_DATE}] No data available after loading/filtering. Exiting.")
        exit()

    # --- Prepare Data for Fitting ---
    df_fit_unfiltered = df_combined[['Es', 'avg_n', 'mu', 'muStd', 'fwhm', 'fwhmStd', 'state']].dropna().copy()
    df_fit = df_fit_unfiltered[df_fit_unfiltered['Es'] <= energy_upper_threshold].copy()
    if df_fit.empty:
        print(f"\n[{CURRENT_DATE}] No data available for fitting after energy cut. Exiting.")
        exit()

    print(f"[{CURRENT_DATE}] Using {len(df_fit)} data points for fitting.")
    x_data_fit = df_fit[['Es', 'avg_n']].to_numpy();
    x_data_simple = df_fit['Es'].to_numpy()
    y_data_fit = df_fit['mu'].to_numpy();
    sigma_fit = df_fit['muStd'].to_numpy()
    fwhm_ph = df_fit['fwhm'].to_numpy();
    fwhmStd_ph = df_fit['fwhmStd'].to_numpy()
    states_fit = df_fit['state'].to_numpy()
    sigma_fit[sigma_fit <= 0] = 1e-6
    fwhmStd_ph[fwhmStd_ph <= 0] = 1e-6

    # --- Create Figure 0: Sequential Fits in a Single Column ---
    # This figure will have four rows: (a) k=0 fit, (b) k=0 residual,
    # (c) k-fitted fit, (d) k-fitted residual.
    fig0, (ax_main_g0, ax_res_g0, ax_main_f, ax_res_f) = plt.subplots(
        4, 1, figsize=(ieee_single_col_width, 12),
        gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0})  # hspace=0 makes pairs touch

    # --- (a) & (b) Preliminary Fit (k=0 Forced) ---
    print(f"\n[{CURRENT_DATE}] Performing preliminary fit: mu = g0(E)...")
    try:
        popt_g0, _ = curve_fit(poly_func_simple_g, x_data_simple, y_data_fit, sigma=sigma_fit,
                               p0=[1e-3] * polynomial_order, absolute_sigma=True)
        coeffs_g0 = popt_g0
        print(f"[{CURRENT_DATE}] --- Preliminary Fit (k=0) Successful ---")

        # Plotting for k=0 panels
        E_plot_g0 = np.linspace(0, x_data_simple.max(), 300)
        y_curve_g0 = poly_func_simple_g(E_plot_g0, *coeffs_g0)
        unique_states_fit = sorted(df_fit['state'].unique())
        for state in unique_states_fit:
            mask = (states_fit == state)
            color = state_colors.get(state, 'gray')
            avg_n_val = df_fit.loc[mask, 'avg_n'].iloc[0]
            ax_main_g0.errorbar(x_data_simple[mask], y_data_fit[mask], yerr=sigma_fit[mask], fmt='o', color=color,
                                label=f"{state}: $\\langle n \\rangle={avg_n_val:.1f}$", capsize=2, alpha=0.8)
        ax_main_g0.plot(E_plot_g0, y_curve_g0, 'k--', label=f'Fit g(E)')
        ax_main_g0.set_ylabel("Pulse Height (arb. units)")
        ax_main_g0.text(0.3,0.1,"(a) No Crosstalk Correction (k=0)", transform=ax_main_g0.transAxes )
        ax_main_g0.legend(fontsize='small')
        ax_main_g0.tick_params(axis='x', labelbottom=False)  # Hide x-labels on top plot
        ax_main_g0.grid(which="both", ls=":", alpha=0.5, lw=1)

        residuals_mu_g0 = y_data_fit - poly_func_simple_g(x_data_simple, *coeffs_g0)
        slope_g0 = poly_func_simple_g_deriv(x_data_simple, *coeffs_g0)
        slope_g0[np.abs(slope_g0) < 1e-9] = np.nan
        residuals_eV_g0 = residuals_mu_g0 / slope_g0
        sigma_eV_g0 = sigma_fit / slope_g0
        for state in unique_states_fit:
            mask = (states_fit == state)
            color = state_colors.get(state, 'gray')
            valid_res_mask = mask & (~np.isnan(residuals_eV_g0))
            ax_res_g0.errorbar(x_data_simple[valid_res_mask], residuals_eV_g0[valid_res_mask],
                               yerr=sigma_eV_g0[valid_res_mask], fmt='o', color=color, alpha=0.8)
        ax_res_g0.axhline(0, color='k', linestyle='--', lw=1)
        ax_res_g0.set_ylabel("Residual (eV)")
        ax_res_g0.grid(which="both", ls=":", alpha=0.5, lw=1)

    except Exception as e:
        print(f"\n[{CURRENT_DATE}] Preliminary Fit Failed: {e}")

    # --- (c) & (d) Main Fit (k varies) ---
    print(f"\n[{CURRENT_DATE}] Performing main fit: mu = f(E + k*Eph*avg_n)...")
    try:
        popt_f, _ = curve_fit(poly_func_f_new, x_data_fit, y_data_fit, sigma=sigma_fit,
                              p0=[initial_k_guess] + [1e-3] * polynomial_order, absolute_sigma=True, maxfev=10000)
        k_fitted = popt_f[0];
        coeffs_f = popt_f[1:]
        print(f"[{CURRENT_DATE}] --- Main Fit (k fitted) Successful ---")
        print(f"Fitted k: {k_fitted:.6f}")

        # Plotting for k-fitted panels
        E_plot_f = E_plot_g0
        avg_n_mean_for_plot = np.mean(x_data_fit[:, 1])
        x_plot_composite_f = np.stack([E_plot_f, np.full_like(E_plot_f, avg_n_mean_for_plot)], axis=-1)
        y_curve_f = poly_func_f_new(x_plot_composite_f, k_fitted, *coeffs_f)
        for state in unique_states_fit:
            mask = (states_fit == state)
            color = state_colors.get(state, 'gray')
            ax_main_f.errorbar(x_data_simple[mask], y_data_fit[mask], yerr=sigma_fit[mask], fmt='o', color=color,
                               capsize=2, alpha=0.8)
        ax_main_f.plot(E_plot_f, y_curve_f, 'k--', label=f'Fit f(E,k), k={k_fitted:.4f}')
        ax_main_f.set_ylabel("Pulse Height (arb. units)")
        ax_main_f.text(0.3,0.1,"(b) With Crosstalk correction ", transform=ax_main_f.transAxes )
        ax_main_f.legend(fontsize='small')
        ax_main_f.tick_params(axis='x', labelbottom=False)  # Hide x-labels on this plot too
        ax_main_f.grid(which="both", ls=":", alpha=0.5, lw=1)

        residuals_mu_f = y_data_fit - poly_func_f_new(x_data_fit, k_fitted, *coeffs_f)
        slope_f = poly_func_f_new_deriv(x_data_fit, k_fitted, *coeffs_f)
        slope_f[np.abs(slope_f) < 1e-9] = np.nan
        residuals_eV_f = residuals_mu_f / slope_f
        sigma_eV_f = sigma_fit / slope_f
        for state in unique_states_fit:
            mask = (states_fit == state)
            color = state_colors.get(state, 'gray')
            valid_res_mask = mask & (~np.isnan(residuals_eV_f))
            ax_res_f.errorbar(x_data_simple[valid_res_mask], residuals_eV_f[valid_res_mask],
                              yerr=sigma_eV_f[valid_res_mask], fmt='o', color=color, alpha=0.8)
        ax_res_f.axhline(0, color='k', linestyle='--', lw=1)
        ax_res_f.set_xlabel("Energy (eV)")
        ax_res_f.set_ylabel("Residual (eV)")
        ax_res_f.grid(which="both", ls=":", alpha=0.5, lw=1)

        # Align y-axis limits for residual plots
        y_min_res = min(ax_res_g0.get_ylim()[0], ax_res_f.get_ylim()[0])
        y_max_res = max(ax_res_g0.get_ylim()[1], ax_res_f.get_ylim()[1])
        ax_res_g0.set_ylim(y_min_res, y_max_res)
        ax_res_f.set_ylim(-0.3,0.3)

        # --- Manually Adjust Subplot Positions to Create a Gap ---
        # Get the current positions of all axes
        pos_g0_main = ax_main_g0.get_position()
        pos_g0_res = ax_res_g0.get_position()
        pos_f_main = ax_main_f.get_position()
        pos_f_res = ax_res_f.get_position()

        # Define the vertical gap size (as a fraction of figure height)
        gap = 0.05

        # Shift the bottom two plots down by the gap amount
        ax_main_f.set_position([pos_f_main.x0, pos_f_main.y0 - gap, pos_f_main.width, pos_f_main.height])
        ax_res_f.set_position([pos_f_res.x0, pos_f_res.y0 - gap, pos_f_res.width, pos_f_res.height])

        # Adjust the figure's top margin to prevent the title from being cut off
        # fig0.subplots_adjust(top=0.95)

    except Exception as e:
        print(f"\n[{CURRENT_DATE}] Main Fit Failed: {e}")
        plt.close(fig0)


    # --- Process Results & Corrections using brentq ---
    print(f"\n[{CURRENT_DATE}] Finding correction factor 's' using brentq...")
    mu_pred_with_k = poly_func_f_new(x_data_fit, k_fitted, *coeffs_f);
    mu_pred_without_k = poly_func_f_new(x_data_fit, 0, *coeffs_f);
    delta_model_base = mu_pred_with_k - mu_pred_without_k
    def get_k_prime_for_brentq(s, x, y, sig, delta, p0_coeffs):
        try:
            popt, _ = curve_fit(poly_func_f_new, x, y - s * delta, sigma=sig, p0=[0.0] + list(p0_coeffs), absolute_sigma=True, maxfev=5000)
            return popt[0]
        except RuntimeError: return np.nan

    try:
        optimal_s = brentq(get_k_prime_for_brentq, a=0.5, b=1.5, args=(x_data_fit, y_data_fit, sigma_fit, delta_model_base, coeffs_f))
        print(f"Optimal 's': {optimal_s:.6f}")
        mu_prime_brentq_data = y_data_fit - optimal_s * delta_model_base
        popt_verify, _ = curve_fit(poly_func_f_new, x_data_fit, mu_prime_brentq_data, sigma=sigma_fit, p0=[0.0] + list(coeffs_f), absolute_sigma=True)
        coeffs_g = popt_verify[1:]
        print(f"Final Coefficients for g(E): {[f'{c:.3e}' for c in coeffs_g]}")

        # --- Plotting Figure 1: Corrected Data and Residuals ---
        print(f"\n[{CURRENT_DATE}] Plotting Corrected Data (Brentq)...")
        fig1, (ax_main_m2, ax_res_m2) = plt.subplots(2, 1, sharex=True, figsize=(ieee_single_col_width, 4),
                                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
        E_plot = np.linspace(0, x_data_simple.max(), 300)
        y_curve_m2 = poly_func_simple_g(E_plot, *coeffs_g)
        for state in unique_states_fit:
            mask = (states_fit == state); color = state_colors.get(state, 'gray')
            ax_main_m2.errorbar(x_data_simple[mask], mu_prime_brentq_data[mask], yerr=sigma_fit[mask],
                                fmt='o', color=color, alpha=0.8, label=state)
        ax_main_m2.plot(E_plot, y_curve_m2, 'k--', label=f'$g(E)$ fit')
        ax_main_m2.set_ylabel("Corrected PH (arb. units)", fontsize=8)
        ax_main_m2.legend(fontsize='x-small')

        y_fit_points_g = poly_func_simple_g(x_data_simple, *coeffs_g);
        residuals_mu_m2 = mu_prime_brentq_data - y_fit_points_g
        slope_g2 = poly_func_simple_g_deriv(x_data_simple, *coeffs_g); slope_g2[np.abs(slope_g2) < 1e-9] = np.nan
        residuals_eV_m2 = residuals_mu_m2 / slope_g2; sigma_eV_m2 = sigma_fit / slope_g2
        for state in unique_states_fit:
            mask = (states_fit == state); color = state_colors.get(state, 'gray')
            valid_slope_mask = mask & (~np.isnan(residuals_eV_m2));
            ax_res_m2.errorbar(x_data_simple[valid_slope_mask], residuals_eV_m2[valid_slope_mask], yerr=sigma_eV_m2[valid_slope_mask],
                               fmt='o', color=color, alpha=0.8)
        ax_res_m2.axhline(0, color='k', linestyle='--', lw=1)
        ax_res_m2.set_xlabel("Energy (eV)", fontsize=8); ax_res_m2.set_ylabel("Residual (eV)", fontsize=8)
        fig1.tight_layout(pad=0.5)



        # --- Plotting Figure 2: FWHM vs Energy ---
        print(f"\n[{CURRENT_DATE}] Plotting FWHM vs Energy...")
        fig2, ax_fwhm = plt.subplots(figsize=(ieee_single_col_width, 3))
        fwhm_eV = fwhm_ph / slope_g2; fwhmStd_eV = fwhmStd_ph / slope_g2
        valid_fwhm_mask = (~np.isnan(fwhm_eV)) & (~np.isnan(fwhmStd_eV)) & (fwhmStd_eV > 1e-9)
        if np.sum(valid_fwhm_mask) >= 2: # Need at least 2 points for 2 parameters
            E_fwhm_fit = x_data_simple[valid_fwhm_mask]
            fwhm_fit_data = fwhm_eV[valid_fwhm_mask]
            fwhm_err_fit_data = fwhmStd_eV[valid_fwhm_mask]

            popt_fwhm, _ = curve_fit(sqrt_linear_func_fwhm, E_fwhm_fit, fwhm_fit_data, sigma=fwhm_err_fit_data,
                                     p0=[max(fwhm_fit_data[0]**2, 1e-2), 1e-3], bounds=([0, -np.inf], [np.inf, np.inf]))
            p0_sq, p1 = popt_fwhm
            fwhm_fit_curve = sqrt_linear_func_fwhm(E_plot, p0_sq, p1)
            fit_label = fr"Fit: $\sqrt{{{p0_sq:.2f} {p1:+.2e}E}}$"
            # ax_fwhm.plot(E_plot, fwhm_fit_curve, 'k--', label=fit_label)
            print(f"FWHM Fit: A^2={p0_sq:.3f}, B={p1:.3e}")
        else:
            print("Skipping FWHM fit: not enough valid data points.")

        for state in unique_states_fit:
            mask = (states_fit == state); color = state_colors.get(state, 'gray')
            plot_mask = mask & valid_fwhm_mask
            if np.any(plot_mask):
                ax_fwhm.errorbar(x_data_simple[plot_mask], fwhm_eV[plot_mask], yerr=fwhmStd_eV[plot_mask],
                                 fmt='o', color=color,  alpha=0.8)
        ax_fwhm.set_xlabel("Energy (eV)"); ax_fwhm.set_ylabel("FWHM (eV)")
        ax_fwhm.grid(which='major', linestyle=':', linewidth=0.5, color='k', alpha=0.5)
        # ax_fwhm.legend(fontsize='x-small')
        fig2.tight_layout(pad=0.5)

    except Exception as e:
        print(f"Brentq or subsequent plotting failed: {e}")

    # Plot Gain curves (Fig 3)
    print(f"\n[{CURRENT_DATE}] Plotting Average and Differential Gain...")
    fig3, (ax_avg, ax_diff_gain) = plt.subplots(2, 1, figsize=(ieee_single_col_width, 8))
    c1_g = coeffs_g[-1];
    can_normalize = abs(c1_g) > 1e-9;
    normalization_factor = c1_g if can_normalize else 1.0;
    # Left Plot
    ax_avg.set_title("(a) Average Gain", fontsize=8)
    avg_gain_curve = poly_func_avg_gain(E_plot, *coeffs_g)
    avg_gain_curve_normalized = avg_gain_curve / normalization_factor
    ax_avg.plot(E_plot, avg_gain_curve_normalized, 'g-', label="$(g(E)/E) / g'(0)$")
    for E_h in E_highlight:
        avg_gain_h_norm = poly_func_avg_gain([E_h], *coeffs_g)[0] / normalization_factor
        ax_avg.axhline(y=avg_gain_h_norm, color='r', linestyle='--', lw=0.75, alpha=0.7)
        ax_avg.axvline(x=E_h, color='r', linestyle='--', lw=0.75, alpha=0.7)
        ax_avg.annotate(f"{avg_gain_h_norm:.4f}", xy=(E_h, avg_gain_h_norm), xytext=(3, -3),
                        textcoords='offset points', ha='left', va='top', fontsize='x-small', color='r')
    ax_avg.set_xlabel("Energy (eV)", fontsize=8);
    ax_avg.set_ylabel("Normalized Gain", fontsize=8);
    ax_avg.legend(fontsize='x-small')
    # Right Plot
    ax_diff_gain.set_title("(b) Differential Gain", fontsize=8)
    deriv_g_curve = poly_func_f_new_deriv(x_plot_composite_f, 0, *coeffs_g)
    deriv_g_curve_normalized = deriv_g_curve / normalization_factor
    ax_diff_gain.plot(E_plot, deriv_g_curve_normalized, 'b-', label="$g'(E) / g'(0)$")
    for E_h in E_highlight:
        deriv_h_norm = poly_func_f_new_deriv(np.array([[E_h, 0]]), 0, *coeffs_g)[0] / normalization_factor
        ax_diff_gain.axhline(y=deriv_h_norm, color='r', linestyle='--', lw=0.75, alpha=0.7)
        ax_diff_gain.axvline(x=E_h, color='r', linestyle='--', lw=0.75, alpha=0.7)
        ax_diff_gain.annotate(f"{deriv_h_norm:.4f}", xy=(E_h, deriv_h_norm), xytext=(3, 3),
                              textcoords='offset points', ha='left', va='bottom', fontsize='x-small', color='r')
    ax_diff_gain.set_xlabel("Energy (eV)", fontsize=8);
    fig3.tight_layout(pad=0.5)


    # --- Final Actions ---
    print(f"\n[{CURRENT_DATE}] Displaying generated plots ({len(plt.get_fignums())} figures)...")
    plt.show()
    print(f"\n[{CURRENT_DATE}] Script finished.")
