import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re  # Import the regular expression module
import scipy.constants
import lmfit
import pandas as pd
import warnings
import pprint
from matplotlib import cm
import matplotlib.colors as mcolors
import json

from analysis.plot_heater_vs_time import savePath

plt.rcParams['font.size'] = 14
# NaN ignored in processing step
warnings.filterwarnings('ignore', message='invalid value encountered in divide')


phi0 = scipy.constants.value(u"mag. flux quantum")

plt.rcParams['font.size'] = 9

# autotune_to_pixel_map = {2:}
def find_npz_files(directory):
    """
    Locates all .npz files in the specified directory.
    Uses regular expressions for robust filename parsing.

    Args:
        directory (str): The path to the directory containing the .npz files.

    Returns:
        dict: A dictionary where keys are Tbase values (in K) and values
              are the corresponding file paths.  Returns an empty dictionary
              if the directory doesn't exist or if no valid files are found.
    """
    npz_files = {}
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return npz_files

    for temp_suffix in range(80, 115):
        pattern = os.path.join(directory, f"*_{temp_suffix:.1f}mK.npz")
        files = glob.glob(pattern)
        # print(files)

        for file_path in files:
            filename = os.path.basename(file_path)
            match = re.search(r"_([\d\.]+)mK\.npz$", filename)
            if match:
                try:
                    tbase_val = float(match.group(1))  # handles "71" or "71.0"
                    tbase_mK = int(round(tbase_val))  # safe integer version
                    tbase = tbase_mK / 1000.0
                    npz_files[tbase] = file_path
                except ValueError:
                    print(f"Warning: Could not extract valid Tbase from '{filename}'. Skipping.")
            else:
                print(f"Warning: Filename '{filename}' does not match expected pattern. Skipping.")

    return npz_files


def remove_offset(arr):
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return np.array([])  # Handle invalid input (not a 2D NumPy array)
    if arr.size == 0 or arr.shape[1] == 0:
        return np.array([])  # Handle empty array or 0 columns

    last_elements = arr[:, -1]  # Get the last element of each row
    return arr - last_elements[:, np.newaxis]

def arb_to_amp(in_val):
    """
    Converts arbitrary units to Amperes.

    Args:
        in_val (float): The input value in arbitrary units.

    Returns:
        float: The equivalent value in Amperes.
    """
    min_SI = 180.5e-12 # 248e-12
    min_phi0_per_amp = min_SI / phi0
    arbs_per_phi0 = 1
    amp_per_arb = 1 / min_phi0_per_amp / arbs_per_phi0
    return in_val * amp_per_arb


def Rtes(ibias, ites, Rshunt=250e-6):
    """
    Calculates Rtes.

    Args:
        ibias (np.ndarray): Array of Ibias values (in Amperes).
        ites (np.ndarray): Array of Ites values (in Amperes).
        Rshunt (float): Shunt resistance (in Ohms).  Defaults to 250e-6.

    Returns:
        np.ndarray: Array of Rtes values (in Ohms).
    """
    Rtes = Rshunt * (ibias - ites) / ites
    return Rtes


def Ptes(ibias, ites, Rshunt=250e-6):
    """
    Calculates Ptes.

    Args:
        ibias (np.ndarray): Array of Ibias values (in Amperes).
        ites (np.ndarray): Array of Ites values (in Amperes).
        Rshunt (float): Shunt resistance (in Ohms).  Defaults to 250e-6.

    Returns:
        np.ndarray: Array of Ptes values (in Watts).
    """
    return Rtes(ibias, ites, Rshunt) * ites ** 2


def convert_ang2_to_ites(ang2, channel_id):
    """
    Converts ang2 values to Ites values.

    Args:
        ang2 (np.ndarray): Array of ang2 values (in arbitrary units).

    Returns:
        np.ndarray: Array of Ites values (in Amperes).
    """
    ites_uncorrected = arb_to_amp(ang2[:,channel_id])
    return ites_uncorrected- ites_uncorrected[-1]


def convert_vbias_to_ibias(vbias, rbias):
    """
    Converts Vbias to Ibias.

    Args:
        vbias (np.ndarray): Array of Vbias values (in Volts).
        rbias (float): Bias resistance (in Ohms).

    Returns:
        np.ndarray: Array of Ibias values (in Amperes).
    """
    return vbias / rbias


def calculate_rn(ibias, ites, Rshunt=250e-6):
    """
    Calculates Rn as the mean of the first 100 Rtes values, handling
    edge cases and NaN/inf values. This assumes that the Rnormal stays constant at higher bias current which we know is NOT true. Currently unused.

    Args:
        ibias (np.ndarray): Array of Ibias values (in Amperes).
        ites (np.ndarray): Array of Ites values (in Amperes).
        Rshunt (float): Shunt resistance (in Ohms). Defaults to 250e-6.

    Returns:
        float or None: The calculated Rn value (in Ohms), or None if Rn
                      cannot be calculated (not enough data points or all
                      Rtes values are invalid).
    """
    if len(ibias) < 100 or len(ites) < 100:
        print("Warning: Not enough data points to calculate Rn (need at least 100).")
        return None  # Or raise an exception

    rtes_values = Rtes(ibias[0:100], ites[0:100], Rshunt)
    rtes_values = rtes_values[np.isfinite(rtes_values)]  # Remove inf and NaN
    if len(rtes_values) == 0:
        print("Warning: All Rtes values are inf or NaN.  Cannot calculate Rn.")
        return None

    return np.mean(rtes_values)

def G_model(x, k, Tc, n):
    """The model function for fitting Ptes vs Tbase."""
    return k * (Tc**n - x**n)

# --- Fitting Function ---
def calculate_G(tbase_values, ptes_values, r_over_rn_ratio, channel_id):

    if len(tbase_values) < 3:  # Need at least 3 points to fit
        print("Warning: Not enough data points to perform fit.")
        return None, None

    gmod = lmfit.Model(G_model)
    params = gmod.make_params(k=1e-7, Tc=0.1, n=4)  # Initial guesses

    params['k'].min = 0
    params['Tc'].min = 0.105
    params['n'].min = 2
    params['n'].max = 4.5

    try:
        result = gmod.fit(ptes_values, params, x=tbase_values)

        # Calculate G at 30 mK (0.030 K)

        k = result.params['k'].value
        n = result.params['n'].value
        Tc = result.params['Tc'].value
        T_eval = 0.1

        G_eval = k * n * (T_eval ** (n - 1))

        # --- Error Propagation ---
        # Get standard errors from the fit result
        k_err = result.params['k'].stderr
        n_err = result.params['n'].stderr
        Tc_err = result.params['Tc'].stderr

        # Check if any errors are None (fit might not have converged)
        if k_err is None or n_err is None or Tc_err is None:
            print("Warning: Could not estimate errors for all parameters.")
            return result, G_eval, None  # Return G, but no error

        # Calculate partial derivatives (using finite differences for simplicity)
        delta = 1e-14  # A small change for numerical differentiation
        dG_dk = (G_model(T_eval, k + delta, Tc, n) - G_model(T_eval, k - delta, Tc, n)) / (2 * delta)
        dG_dTc = (G_model(T_eval, k, Tc + delta, n) - G_model(T_eval, k, Tc - delta, n)) / (2 * delta)
        dG_dn = (G_model(T_eval, k, Tc, n + delta) - G_model(T_eval, k, Tc, n - delta)) / (2 * delta)

        # dG_dk = n*T_eval**(n-1)
        # dG_dn = k*(T_eval**(n-1)+n(n-1)*T_eval**(n-1)*np.log(T_eval))
        # dG_dTc = n*(n-1)*T_eval**(n-2)

        # Calculate the variance of G
        G_variance = (dG_dk * k_err) ** 2 + (dG_dTc * Tc_err) ** 2 + (dG_dn * n_err) ** 2

        # Covariance
        if result.covar is not None:  # check if exists
            cov_kn = result.covar[0, 2]  # k is first param and n is third.
            cov_kTc = result.covar[0, 1]
            cov_nTc = result.covar[2, 1]
            G_variance += 2 * (dG_dk * dG_dn * cov_kn + dG_dk * dG_dTc * cov_kTc + dG_dn * dG_dTc * cov_nTc)
        else:
            print("Warning: Covariance matrix not available.  Error may be underestimated.")

        G_error = np.sqrt(G_variance)

        return result, G_eval, G_error

    except Exception as e:
        print(f"Error during fitting: {e}")
        return None, None, None


def calculate_G_all_ratios(npz_files, rbias, channel_id, r_over_rn_ratios, Rshunt=250e-6,
                           ptes_increase_threshold=0.05, tbase_max=None, Rn_fixed=None, skip_tbase=None):
    results = {}

    for ratio in r_over_rn_ratios:
        tbase_values_filtered = []  # Changed variable name to reflect filtering
        ptes_values_filtered = []  # Changed variable name to reflect filtering
        sorted_tbase = sorted(npz_files.keys())

        for tbase in sorted_tbase:
            if np.round(tbase, 3) in np.round(skip_tbase, 3):
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
                    ites = convert_ang2_to_ites(ang2, channel_id)
                    ibias = convert_vbias_to_ibias(vbias, rbias)

                    if Rn_fixed is None:
                        continue
                    else:
                        rn = Rn_fixed
                    rtes = Rtes(ibias, ites, Rshunt)
                    target_rtes = rn * ratio
                    idx = np.nanargmin(np.abs(rtes - target_rtes))
                    ptes = Ptes(ibias, ites, Rshunt)[idx]

                    # Filter for Ptes >= 0
                    if ptes > 0:
                        tbase_values_filtered.append(tbase)
                        ptes_values_filtered.append(ptes)
                    else:
                        # Optionally, you could print a message here if a point is skipped
                        # print(f"Skipping Tbase = {tbase:.3f} K for ratio {ratio:.2f} due to Ptes < 0.")
                        break

            except Exception as e:
                print(f"Error in file processing: {e}. Skipping.")
                continue

        # --- Apply Tbase range filtering for fitting ---
        if tbase_max is not None:
            tbase_fit = []
            ptes_fit = []
            for t, p in zip(tbase_values_filtered, ptes_values_filtered):  # Use filtered values here
                if t <= tbase_max:
                    tbase_fit.append(t)
                    ptes_fit.append(p)
            tbase_values_fit = np.array(tbase_fit)
            ptes_values_fit = np.array(ptes_fit)
        else:
            tbase_values_fit = np.array(tbase_values_filtered)  # Use filtered values here
            ptes_values_fit = np.array(ptes_values_filtered)

        # Perform the fit
        fit_result, G, G_error = calculate_G(tbase_values_fit, ptes_values_fit, ratio, channel_id)

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

    if saveResults and results:


        # Create a list of dictionaries for selected columns
        data_to_save = []
        for ratio, data in results.items():
            if 'G' in data and 'G_err' in data:  # Ensure G and G_err exist for this ratio

                pixel_number_for_channel = pixel_map[channel_id]  # Or look up in a pixel_map if available
                data_to_save.append({
                    'Pixel_Number': pixel_number_for_channel,
                    'Rtes/Rn': ratio,  # Include the ratio as a column
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
            print(f"Saved G results for channel {channel_id} to {savePath}G_results_ch{channel_id}.csv")
        else:
            print("No valid G results to save.")
    else:
        if not results:  # Only print if there are no results to save
            print("No fit results to save.")

    # print(f"Thermal parameters for channel {channel_id} at 7% Rn, where Rn = {Rn_fixed * 1000}" + " mOhm")
    return results

def plot_fit_results(fit_results, channel_id):
    """
    Plots G, k, n, and Tc with error bars as a function of Rtes/Rn ratio.

    Args:
        fit_results: Dictionary of fit results (output of calculate_G_all_ratios).
    """

    if not fit_results:
        print("Error: No fit results to plot.")
        return

    ratios = list(fit_results.keys())
    ratio=0.8
    # Extract the values and errors, handling missing values gracefully.
    G_values = [fit_results[ratio].get('G', np.nan) for ratio in ratios]
    G_errors = [fit_results[ratio].get('G_err', np.nan) for ratio in ratios]
    k_values = [fit_results[ratio].get('k', np.nan) for ratio in ratios]
    k_errors = [fit_results[ratio].get('k_err', np.nan) for ratio in ratios]
    Tc_values = [fit_results[ratio].get('Tc', np.nan) for ratio in ratios]
    Tc_errors = [fit_results[ratio].get('Tc_err', np.nan) for ratio in ratios]
    n_values = [fit_results[ratio].get('n', np.nan) for ratio in ratios]
    n_errors = [fit_results[ratio].get('n_err', np.nan) for ratio in ratios]

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    axes[0].errorbar(ratios, np.array(G_values)*1e12, yerr=np.array(G_errors)*1e12, fmt='o-', capsize=5, label='G')
    axes[0].set_ylabel('G @ 100 mK (pW/K)')
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


def plot_ptes_vs_tbase_multiple_ratios(npz_files, rbias, channel_id, r_over_rn_ratios, Rshunt=250e-6, Rn_fixed = None,
                                       ptes_increase_threshold=0.05, tbase_max=None, skip_tbase = None):
    """
    Plots Ptes vs. Tbase with fits and error bars, for different Rtes/Rn ratios.
    Plots fit lines over *all* data points and uses calculate_G_all_ratios.
    """

    if not npz_files:
        print("Error: No .npz files to plot.")
        return

    # --- Perform fits for all ratios ---
    fit_results = calculate_G_all_ratios(npz_files, rbias, channel_id, r_over_rn_ratios,
                                         Rshunt, ptes_increase_threshold, tbase_max, Rn_fixed = Rn_fixed, skip_tbase=skip_tbase)

    plt.figure(figsize=(4, 4), dpi=200)
    cmap = plt.get_cmap('coolwarm')
    num_ratios = len(r_over_rn_ratios)
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=min(r_over_rn_ratios) * 100, vmax=max(r_over_rn_ratios) * 100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    annotation_count = 0

    for i, ratio in enumerate(r_over_rn_ratios):
        tbase_values = []
        ptes_values = []
        sorted_tbase = sorted(npz_files.keys())
        last_valid_ptes = None
        color = cmap(i / (num_ratios - 1)) if num_ratios > 1 else cmap(0.5)  # Color for this ratio

        # --- Data Loading (remains the same) ---
        for tbase in sorted_tbase:
            if np.round(tbase,3) in np.round(skip_tbase,3):  # Skip this Tbase value

                # print(f"Skipping Tbase = {tbase:.3f} K (specified in skip_tbase).")
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
                    ites = convert_ang2_to_ites(ang2, channel_id)
                    ibias = convert_vbias_to_ibias(vbias, rbias)


                    if Rn_fixed is None:
                        rn = calculate_rn(ibias, ites, Rshunt)
                    else:
                        rn = Rn_fixed
                    rtes = Rtes(ibias, ites, Rshunt)
                    target_rtes = rn * ratio
                    idx = np.nanargmin(np.abs(rtes - target_rtes))
                    ptes = Ptes(ibias, ites, Rshunt)[idx]

                    # if last_valid_ptes is not None and ptes > (1 + ptes_increase_threshold) * last_valid_ptes:
                    #     print(f"Clipping at Tbase = {tbase:.3f} K.")
                    #     break  # Still break out of the inner loop

                    tbase_values.append(tbase)
                    ptes_values.append(ptes)
                    last_valid_ptes = ptes

            except Exception as e:
                print(f"Error in file processing: {e}. Skipping.")
                continue
        # --- (End Data Loading) ---


        # --- Plotting Data ---
        plt.plot(np.array(tbase_values)*1e3, np.array(ptes_values)*1e12, marker='.', linestyle ='',
                 label=f"{int(ratio*100)}", color=color, ms=3, lw=0.2)

        # --- Plotting Fit Results (if available) ---
        if ratio in fit_results and fit_results[ratio]:  # Check for the ratio
            result = fit_results[ratio]['fit_result']
            tbase_values_all = np.array(sorted(npz_files.keys()))
            plt.plot(np.array(tbase_values_all)*1e3, G_model(np.array(tbase_values_all), **result.best_values)*1e12,
                     linestyle='-', color=color, lw=0.7)

    plt.xlabel(r"T$_\mathrm{bath}$ (mK)")
    plt.ylabel("P$_\mathrm{TES}$ (pW)")
    # plt.title(f"Ptes vs. Tbase for Channel {channel_id} (Multiple Rtes/Rn Ratios)",
    #           )
    # plt.ylim(0,2)
    # plt.xlim(20,53)
    # plt.legend(title='R/Rn(%)', ncols=3, fontsize='small')
    plt.grid(which="both", ls=":", lw=0.5, alpha=0.5)
    cax = plt.axes([0.75, 0.4, 0.03, 0.45])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('R/Rn (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)




    # Plot fit parameters
    if plotFitResults:
        plot_fit_results(fit_results, channel_id = channel_id)



def plot_iv_and_rtes_vs_ibias(npz_files, rbias, channel_id, tbase_values=None,
                              Rshunt=250e-6, Rn_fixed = None, skip_tbase=None):
    """
    Plots Ites vs. Ibias, Rtes vs. Ibias, and Ptes vs Ibias for a
    specific channel and multiple Tbase values.

    Args:
        npz_files (dict): Dictionary of .npz file paths (keys are Tbase in K).
        rbias (float): Bias resistance (in Ohms).
        channel_id (int): The channel ID to plot.
        tbase_values (list, optional): A list of Tbase values (in K) to plot.
            If None, all available Tbase values are used. Defaults to None.
        Rshunt (float): Shunt resistance (in Ohms). Defaults to 250e-6.
    """
    if not npz_files:
        print("Error: No .npz files to plot.")
        return

    # Use all Tbase values if none are specified
    if tbase_values is None:
        if skip_tbase is None:
            tbase_values = sorted(npz_files.keys())
        else:
            tbase_values = sorted([t for t in npz_files.keys() if round(t,3)
                                   not in np.round(np.array(skip_tbase),3)])
    elif not all(tbase in npz_files for tbase in tbase_values):
        print("Error: One or more specified Tbase values are not found in the data.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # 3 rows, 1 column
    num_tbase_values = len(tbase_values)
    cmap = plt.get_cmap('coolwarm')  # Get the coolwarm colormap

    for i, tbase in enumerate(tbase_values):
        color = cmap(i / (num_tbase_values - 1)) if num_tbase_values > 1 else cmap(0.5)  # Handle single Tbase case
        file_path = npz_files[tbase]
        try:
            with np.load(file_path) as data:
                if 'vb' not in data or 'ang2' not in data:
                    print(f"Error: 'vb' or 'ang2' not found in '{file_path}'. Skipping.")
                    continue

                vbias = data['vb']
                ang2 = data['ang2']

                if ang2.shape[1] <= channel_id:
                    print(f"Error: Channel {channel_id} is out of range in '{file_path}'.")
                    continue

                ites = convert_ang2_to_ites(ang2, channel_id)
                ibias = convert_vbias_to_ibias(vbias, rbias)

                # Ites vs. Ibias Plot
                axes[0].plot(ibias*1e3, ites*1e3, label=f"{tbase*1e3:}", color=color)
                if i==0:
                    axes[0].plot(ibias * 1e3, ibias*1e3, color='k', linestyle = '--')


                # Rtes vs. Ibias Plot
                rtes = Rtes(ibias, ites, Rshunt)
                axes[1].plot(ibias*1e3, rtes*1e3, label=f"{tbase*1e3:}", color=color)
                axes[1].axhline(y=Rn_fixed*1e3, color='k', linestyle='--')

                # Ptes vs. Ibias Plot
                ptes = Ptes(ibias, ites, Rshunt)
                axes[2].plot(ibias*1e3, ptes*1e12, label=f"{tbase*1e3:}", color=color)

            if tbase == tbase_biasdict:
                percentdict = {}
                percentrns = [7]
                for percentrn in percentrns:
                    idx = np.nanargmin(np.abs(rtes-Rn_fixed*percentrn/100))
                    percentdict[percentrn] = {'Rtes (mOhm)':rtes[idx]*1000, 'Vbias (mV)':vbias[idx]*1000, 'Ibias (mA)':ibias[idx]*1000,
                                              'Ptes (pW)': ptes[idx]*1e12, r'Ites ($\mu$A)':ites[idx]*1e6}
                print(f"Bias parameters for channel {channel_id} at {tbase_biasdict*1000} mK bath, with Rn = {Rn_fixed*1e3} mOhm:\n")
                pprint.pp(percentdict, indent=4)
        except FileNotFoundError:
            print(f"Error: File not found: '{file_path}'. Skipping.")
        except Exception as e:
            print(f"Error loading/processing '{file_path}': {e}. Skipping.")

    # Ites vs. Ibias plot settings
    axes[0].set_xlabel("Ibias (mA)")
    axes[0].set_ylabel(r"Ites (mA)")
    axes[0].set_title(f"Ites vs. Ibias for Channel {channel_id}")
    # axes[0].legend(title = ' Tbase (mK)', ncols=4)
    axes[0].grid(True)

    # Rtes vs. Ibias plot settings
    axes[1].set_xlabel("Ibias (mA)")
    axes[1].set_ylabel("Rtes (mΩ)")
    axes[1].set_title(f"Rtes vs. Ibias for Channel {channel_id}")
    # axes[1].legend(title = ' Tbase (mK)', ncols=4)
    axes[1].grid(True)

    # Ptes vs. Ibias plot settings
    axes[2].set_xlabel("Ibias (mA)")
    axes[2].set_ylabel("Ptes (pW)")
    axes[2].set_title(f"Ptes vs. Ibias for Channel {channel_id}")
    # axes[2].legend(title = ' Tbase (mK)', ncols=4)
    axes[2].grid(True)

    plt.tight_layout()

def plot_iv_and_rtes_vs_ptes(npz_files, rbias, channel_id, tbase_values=None,
                              Rshunt=250e-6, Rn_fixed = None, skip_tbase=None,
                              tbase_biasdict=None):
    """
    Plots Ites vs. Ibias, Rtes vs. Ibias, and Ptes vs Ibias for a
    specific channel and multiple Tbase values, with a colorbar on the right.

    Args:
        npz_files (dict): Dictionary of .npz file paths (keys are Tbase in K).
        rbias (float): Bias resistance (in Ohms).
        channel_id (int): The channel ID to plot.
        tbase_values (list, optional): A list of Tbase values (in K) to plot.
            If None, all available Tbase values are used. Defaults to None.
        Rshunt (float): Shunt resistance (in Ohms). Defaults to 250e-6.
        Rn_fixed (float, optional): Fixed normal resistance value for plotting. Defaults to None.
        skip_tbase (list, optional): A list of Tbase values to skip. Defaults to None.
        tbase_biasdict (float, optional): Specific Tbase value to print bias parameters for. Defaults to None.
    """
    if not npz_files:
        print("Error: No .npz files to plot.")
        return

    # Use all Tbase values if none are specified
    if tbase_values is None:
        if skip_tbase is None:
            tbase_values = sorted(npz_files.keys())
        else:
            tbase_values = sorted([t for t in npz_files.keys() if round(t,3)
                                   not in np.round(np.array(skip_tbase),3)])
    elif not all(tbase in npz_files for tbase in tbase_values):
        print("Error: One or more specified Tbase values are not found in the data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 10), dpi=200)  # 2 subplots
    num_tbase_values = len(tbase_values)
    cmap = cm.coolwarm  # Get the coolwarm colormap
    norm = mcolors.Normalize(vmin=min(tbase_values) * 1e3, vmax=max(tbase_values) * 1e3) # Normalize Tbase values for colormap
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])  # For older matplotlib versions

    for i, tbase in enumerate(tbase_values):
        color = scalar_map.to_rgba(tbase * 1e3)
        file_path = npz_files[tbase]
        try:
            with np.load(file_path) as data:
                if 'vb' not in data or 'ang2' not in data:
                    print(f"Error: 'vb' or 'ang2' not found in '{file_path}'. Skipping.")
                    continue

                vbias = data['vb']
                # print(np.diff(vbias))
                # print(np.max(vbias))
                ang2 = data['ang2']

                if ang2.shape[1] <= channel_id:
                    print(f"Error: Channel {channel_id} is out of range in '{file_path}'.")
                    continue

                ites = convert_ang2_to_ites(ang2, channel_id)
                ibias = convert_vbias_to_ibias(vbias, rbias)

                # Ites vs. Ibias Plot
                axes[0].plot(ibias*1e3, ites*1e3, color=color, lw=0.75)
                if i==0:
                    axes[0].plot(ibias * 1e3, ibias*1e3, color='k', linestyle = '--', lw=0.5)


                # Rtes vs. Ibias Plot
                rtes = Rtes(ibias, ites, Rshunt)
                ptes = Ptes(ibias, ites, Rshunt)
                axes[1].plot(ptes*1e12, rtes*1e3, color=color, lw=0.75)

                label_text = str(round(tbase*1e3))

                axes[1].annotate(
                    label_text,
                    xy=(ptes[0] * 1e12, rtes[0] * 1e3),  # endpoint in plot units
                    xytext=(5, 0),  # small offset to the right
                    textcoords="offset points",
                    color=axes[1].lines[-1].get_color(),  # match line color
                    fontsize=4,
                    ha="left",
                    va="center"
                )



            if tbase_biasdict is not None and round(tbase, 3) == round(tbase_biasdict, 3):
                percentdict = {}
                percentrns = [7]
                for percentrn in percentrns:
                    idx = np.nanargmin(np.abs(rtes-Rn_fixed*percentrn/100))
                    percentdict[percentrn] = {'Rtes (mOhm)':rtes[idx]*1000, 'Vbias (mV)':vbias[idx]*1000, 'Ibias (mA)':ibias[idx]*1000,
                                              'Ptes (pW)': ptes[idx]*1e12, r'Ites ($\mu$A)':ites[idx]*1e6}
                print(f"Bias parameters for channel {channel_id} at {tbase_biasdict*1000} mK bath, with Rn = {Rn_fixed*1e3} mOhm:\n")
                pprint.pp(percentdict, indent=4)
        except FileNotFoundError:
            print(f"Error: File not found: '{file_path}'. Skipping.")
        except Exception as e:
            print(f"Error loading/processing '{file_path}': {e}. Skipping.")

    # Ites vs. Ibias plot settings
    axes[0].set_xlabel("Ibias (mA)")
    axes[0].set_ylabel("Ites (mA)")
    axes[0].set_title(f"Ites vs. Ibias for Channel {channel_id}")
    axes[0].set(xlim=(-0.01,0.8),ylim=(-0.01,0.4))
    axes[0].grid(True)

    # Rtes vs. Ibias plot settings
    if Rn_fixed is not None:
        axes[1].axhline(y=Rn_fixed * 1e3, color='k', linestyle='--', lw=0.5)
    axes[1].set_xlabel("Ptes (pW)")
    axes[1].set_ylabel(r"Rtes (m$\Omega$)")
    axes[1].set_title(f"Rtes vs. Ptes for Channel {channel_id}")
    # axes[1].set(xlim=(-0.1, 7), ylim=(-0.1, 8.5))
    axes[1].grid(True)

    # Add colorbar outside to the right of the right plot
    fig.colorbar(scalar_map, cax=fig.add_axes([0.92, 0.15, 0.03, 0.7]), label='Tbase (mK)')
    # [left, bottom, width, height] for the new axes

    # plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for the colorbar
    # plt.show()


def plot_iv_curves_subplots(npz_files, rbias, which_tbase_values = 'all', channels_to_ignore=None):
    """
    Plots IV curves (Ibias vs. Ites) with each channel in a separate subplot.
    """
    if not npz_files:
        print("Error: No .npz files to plot.")
        return

    if channels_to_ignore is None:
        channels_to_ignore = set()
    else:
        channels_to_ignore = set(channels_to_ignore)
    good_channels = np.array(list(set(np.arange(32)) - channels_to_ignore))
    num_channels = len(good_channels)
    num_cols = 4
    num_rows = (num_channels + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows), sharex=True, sharey=True)
    plt.subplots_adjust(top=0.98, bottom=0.065, right=0.98, left=0.05, hspace=0.2, wspace=0.1)
    axes = axes.flatten()
    fig.text(0.5, 0.01, 'Bias current (mA)', ha='center', va='bottom')  # x, y, text, alignment
    fig.text(0.01, 0.5, 'TES current (mA)', ha='left', va='center', rotation='vertical')
    if which_tbase_values == 'all':
        tbase_values = sorted(npz_files.keys())
    num_tbase_values = len(tbase_values)
    cmap = plt.get_cmap('coolwarm')  # Get the coolwarm colormap
    for i, tbase in enumerate(tbase_values):
        color = cmap(i / (num_tbase_values - 1)) if num_tbase_values > 1 else cmap(0.5)  # Handle single Tbase case
        file_path = npz_files[tbase]
        try:
            with np.load(file_path) as data:


                if 'vb' not in data or 'ang2' not in data:
                    print("Error: 'vb' or 'ang2' not found in the .npz file.")
                    return

                vbias = data['vb']
                ang2 = data['ang2']

                if ang2.shape != (3001, 32):
                    print(f"Warning: 'ang2' has unexpected shape: {ang2.shape}. Expected (3001, 32).")


                ibias = convert_vbias_to_ibias(vbias, rbias)


                for j, channel_id in enumerate(good_channels):

                        ites = convert_ang2_to_ites(ang2, channel_id)
                        ax = axes[j]
                        ax.plot(ibias * 1e3, ites * 1e3, color=color, )
                        if  i == 0:
                            ax.text(0.6, 0.6, f"Channel {channel_id}", transform=ax.transAxes, fontsize = 10)
                            ax.plot(ibias * 1e3, ibias * 1e3, ls='--', color='k', alpha=1, lw=0.75)

                        # ax.set_xlabel("Ibias (mA)")
                        # ax.set_ylabel(r"Ites ($\mu$A)")
                        ax.set(xlim = (0,1), ylim=(0,0.4))
                        ax.grid(True)
                    # else:
                    #     ax = axes[channel_id]
                    #     ax.set_title(f"Channel {channel_id} (Ignored)")
                    #     ax.set_xticks([])
                    #     ax.set_yticks([])

                for k in range(num_channels, num_rows * num_cols):
                    fig.delaxes(axes[k])

                # plt.suptitle(f"IV Curves for {os.path.basename(first_file_path)}", fontsize=16)
                # plt.tight_layout()


        except FileNotFoundError:
            print(f"Error: File not found: '{file_path}'")
        except Exception as e:
            print(f"Error loading or plotting data: {e}")


def plot_iv_curves(npz_files, rbias, channels_to_ignore=None):
    """
    Plots IV curves (Ibias vs. Ites), optionally ignoring channels
    """
    if not npz_files:
        print("Error: No .npz files to plot.")
        return

    first_tbase = sorted(npz_files.keys())[0]
    first_file_path = npz_files[first_tbase]

    try:
        with np.load(first_file_path) as data:
            if 'vb' not in data or 'ang2' not in data:
                print("Error: 'vb' or 'ang2' not found in the .npz file.")
                return

            vbias = data['vb']

            ang2 = data['ang2']

            if ang2.shape != (3001, 32):
                print(f"Warning: 'ang2' has unexpected shape: {ang2.shape}. Expected (3001, 32).")


            ibias = convert_vbias_to_ibias(vbias, rbias)

            plt.figure(figsize=(12, 8))

            if channels_to_ignore is None:
                channels_to_ignore = set()
            else:
                channels_to_ignore = set(channels_to_ignore)

            for channel_id in range(ang2.shape[1]):
                if channel_id not in channels_to_ignore:
                    ites = convert_ang2_to_ites(ang2, channel_id)
                    plt.plot(ibias, ites, label=f"Channel {channel_id}")

            plt.xlabel("Ibias (A)")
            plt.ylabel("Ites (A)")
            plt.title(f"IV Curves for {os.path.basename(first_file_path)}")
            plt.legend(loc='upper right', ncol=4)
            plt.grid(True)
            plt.show()

    except FileNotFoundError:
        print(f"Error: File not found: '{first_file_path}'")
    except Exception as e:
        print(f"Error loading or plotting data: {e}")



def plot_Tc_heatmap(base_path=savePath, ratio=0.9,
                    pixel_map=None):
    """
    Plots a heatmap of Tc values for different channels based on their pixel layout.

    Args:
        base_path (str): Directory containing the fit results CSV files.
        ratio (float): The Rtes/Rn ratio at which to extract Tc values.
        channel_id_to_pixel_map (dict, optional): A dictionary mapping channel IDs to pixel numbers.
            If None, pixel number is assumed to be the same as channel ID.
    """
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



    nrows = 4
    ncols = 6
    # Initialize the heatmap grid with NaN
    heatmap_data = np.full((nrows, ncols), np.nan)
    channel_labels = np.full((nrows, ncols), '', dtype=object)


    for channel_id, pixel_number in pixel_map.items():
        pixel_number = int(pixel_number)
        if channel_id in all_data:

            display_row = (pixel_number - 1) % nrows
            display_col = (pixel_number - 1) // nrows


            array_row = (nrows - 1) - display_row
            array_col = display_col

            if 0 <= array_row < nrows and 0 <= array_col < ncols:
                heatmap_data[array_row, array_col] = all_data[channel_id]['Tc'] * 1e3  # Convert to mK
            else:
                print(
                    f"Warning: Pixel number {pixel_number} for channel {channel_id} out of bounds for {nrows}x{ncols} grid. Skipping.")

    plt.figure(figsize=(ncols * 1.5, nrows * 1.5))
    cax = plt.imshow(heatmap_data, cmap='viridis', origin='lower',
                     extent=[0, ncols, 0, nrows], aspect='equal')

    # Add channel IDs and Tc values as text annotations
    for r_idx in range(nrows):
        for c_idx in range(ncols):
            if not np.isnan(heatmap_data[r_idx, c_idx]):
                # Convert array index back to display (pixel) row/col for text positioning
                display_r = (nrows - 1) - r_idx
                display_c = c_idx

                # Calculate the pixel number for this display position
                pixel_num = display_c * nrows + display_r + 1

                channel_id_for_pixel = None
                for ch, px in pixel_map.items():
                    if int(px) == pixel_num:
                        channel_id_for_pixel = ch
                        break

                if channel_id_for_pixel is not None:
                    plt.text(c_idx + 0.5, r_idx + 0.5, f"Pixel {pixel_num}\n{heatmap_data[r_idx, c_idx]:.2f}mK",
                             ha='center', va='center', color='white', fontsize=12,
                             bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="black", lw=0.5, alpha=0.5))

    plt.title(f'Tc Heatmap (mK) at R/Rn = {ratio * 100:.0f}% ')
    plt.xlabel('Column (Pixel Index)')
    plt.ylabel('Row (Pixel Index)')

    # Set custom ticks and labels
    plt.xticks(np.arange(0.5, ncols + 0.5), labels=np.arange(ncols) + 1)
    plt.yticks(np.arange(0.5, nrows + 0.5), labels=np.arange(nrows) + 1)
    # plt.grid(True, color='white', linestyle='-', linewidth=2)
    plt.colorbar(cax, label='Tc (mK)')
    plt.tight_layout()

def display_summary_plot(
    base_path=savePath,
    ratio=0.9,
    pixel_map=None,
    save_summary=False,  # NEW keyword
):
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

    # ---- Save summary if requested ----
    if save_summary:
        rows = []
        for i, ch in enumerate(sorted_channels):
            pixel_num = x_labels[i]
            data = all_data[ch]
            rows.append({
                "Pixel_Number": pixel_num,
                "Rtes/Rn Ratio": ratio,
                "G_at_100mK (pW/K)": data['G'] * 1e12,
                "G_err_at_100mK (pW/K)": data['G_err'] * 1e12,
                "n": data['n'],
                "n_err": data['n_err'],
                "k (nW/K^n)": data['k'] * 1e9,
                "k_err (nW/K^n)": data['k_err'] * 1e9,
                "Tc (mK)": data['Tc'] * 1e3,
                "Tc_err (mK)": data['Tc_err'] * 1e3,
            })
        summary_df = pd.DataFrame(rows)
        output_file = os.path.join(base_path, f"G_summary_ratio_{ratio:.2f}.csv")
        summary_df.to_csv(output_file, index=False)
        print(f"Saved summary to {output_file}")

    # ---- Plotting ----
    fig, axes = plt.subplots(4, 1, figsize=(8,16), sharex=True)
    fig.suptitle(f"Thermal parameter summary plot at R/Rn = {ratio}", fontsize=14)

    items = ['G', 'n', 'k', 'Tc']
    units = ['(pW/K)', '', '(nW/K^n)', '(mK)']
    multipliers = [1e12, 1, 1e9, 1e3]

    for i, item in enumerate(items):
        values = [all_data[ch][item] for ch in sorted_channels]
        errors = [all_data[ch][f"{item}_err"] for ch in sorted_channels]

        axes[i].errorbar(
            x_labels,
            np.array(values) * multipliers[i],
            yerr=np.array(errors) * multipliers[i],
            fmt='o',
            capsize=5
        )

        axes[i].set_ylabel(f"{item} {units[i]}")
        axes[i].grid(True)

    axes[3].set_xlabel("Pixel Number" if pixel_map else "Channel ID", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def main():
    """Example usage."""
    data_directory = "/data/20250213/iv/"  # Flat dtest
    data_directory = "/data/20250908/iv/"  # dtest60 chip A
    data_directory = "/data/20250915/iv/"  # dtest60 chip C
    # data_directory = "/data/20250514/iv/"
    rbias_value = 1965.4
    Rshunt_value = 250e-6

    channel_data = {
        1: {"tbase_skip": np.array([86,105])*1e-3, "Rn_fixed": 16.6e-3, "tbase_max": 0.12},
        2: {"tbase_skip": np.array([85,87, 90,93,97,104])*1e-3, "Rn_fixed": 17.7e-3, "tbase_max": 0.12},
        3: {"tbase_skip": np.array([93, 104])*1e-3, "Rn_fixed": 23e-3, "tbase_max": 0.12},
        4: {"tbase_skip": np.array([80,84,86,89,98,102,103,104,105,106,107,108,109])*1e-3, "Rn_fixed": 17.6e-3, "tbase_max": 0.12},
        5: {"tbase_skip": np.array([84,100,103])*1e-3, "Rn_fixed": 15e-3, "tbase_max": 0.12},
        7: {"tbase_skip": np.array([101])*1e-3, "Rn_fixed": 19.5e-3, "tbase_max": 0.12},
        8: {"tbase_skip": np.array([81])*1e-3, "Rn_fixed": 19e-3, "tbase_max": 0.12},
        11: {"tbase_skip": np.array([105])*1e-3, "Rn_fixed": 17e-3, "tbase_max": 0.12},
        12: {"tbase_skip": np.array([]), "Rn_fixed": 22e-3, "tbase_max": 0.11},
        17: {"tbase_skip": np.array([]), "Rn_fixed": 17e-3, "tbase_max": 0.12},
        18: {"tbase_skip": np.array([]), "Rn_fixed": 16.1e-3, "tbase_max": 0.12},
        21: {"tbase_skip": np.array([92])*1e-3, "Rn_fixed": 20e-3, "tbase_max": 0.12},
        22: {"tbase_skip": np.array([]), "Rn_fixed": 16e-3, "tbase_max": 0.12},
        23: {"tbase_skip": np.array([]), "Rn_fixed": 20e-3, "tbase_max": 0.12},
        24: {"tbase_skip": np.array([]), "Rn_fixed": 18.6e-3, "tbase_max": 0.12},
        25: {"tbase_skip": np.array([83,90,95])*1e-3, "Rn_fixed": 22e-3, "tbase_max": 0.12},
        26: {"tbase_skip": np.array([87])*1e-3, "Rn_fixed": 12.5e-3, "tbase_max": 0.111},
    }
    # channel_data = {                 # dtest60 chip A 7 um fil
    #     2: {"tbase_skip": np.array([]), "Rn_fixed": 11.2e-3, "tbase_max": 0.080},
    #     3: {"tbase_skip": np.array([]), "Rn_fixed": 10e-3, "tbase_max": 0.080},
    #     4: {"tbase_skip": np.array([]), "Rn_fixed": 17.6e-3, "tbase_max": 0.080},
    #     5: {"tbase_skip": np.array([]), "Rn_fixed": 15e-3, "tbase_max": 0.080},
    #     7: {"tbase_skip": np.array([]), "Rn_fixed": 8e-3, "tbase_max": 0.080},
    #     8: {"tbase_skip": np.array([0.051]), "Rn_fixed": 8e-3, "tbase_max": 0.080},
    #     9: {"tbase_skip": np.array([]), "Rn_fixed": 6.8e-3, "tbase_max": 0.080},
    #     12: {"tbase_skip": np.array([0.041, 0.046,0.052,0.053]), "Rn_fixed": 14.0e-3, "tbase_max": 0.080},
    #     14: {"tbase_skip": np.array([0.042,0.044,0.047,0.048, 0.052,0.054,]), "Rn_fixed": 6.85e-3, "tbase_max": 0.080},
    #     19: {"tbase_skip": np.array([]), "Rn_fixed": 6e-3,  "tbase_max": 0.080},
    #     20: {"tbase_skip": np.array([]), "Rn_fixed": 6e-3, "tbase_max": 0.080},
    #     21: {"tbase_skip": np.array([]), "Rn_fixed": 22.8e-3, "tbase_max": 0.080},
    #     22: {"tbase_skip": np.array([]), "Rn_fixed": 22e-3, "tbase_max": 0.080},
    #     23: {"tbase_skip": np.array([0.044, 0.063]), "Rn_fixed": 10.8e-3, "tbase_max": 0.080},
    #     24: {"tbase_skip": np.array([]), "Rn_fixed": 3.5e-3, "tbase_max": 0.080},
    #     25: {"tbase_skip": np.array([]), "Rn_fixed": 22e-3, "tbase_max": 0.080},
    #     26: {"tbase_skip": np.array([0.040, 0.044, 0.049,0.052,0.053,0.058]), "Rn_fixed": 6.9e-3, "tbase_max": 0.080},
    #     28: {"tbase_skip": np.array([]), "Rn_fixed": 12e-3, "tbase_max": 0.080},
    #     29: {"tbase_skip": np.linspace(31, 44, 14) * 1e-3, "Rn_fixed": 6.9e-3, "tbase_max": 0.080},
    # }
    # channel

    # channel_data = {
    #     2: {"tbase_skip": np.array([0.042, 0.043, 0.044, 0.045, 0.046, 0.047]), "Rn_fixed": 6.6e-3, "tbase_max": 0.052},
    #     4: {"tbase_skip": np.linspace(20, 45, 26) * 1e-3, "Rn_fixed": 9.3e-3, "tbase_max": 0.055},
    #     9: {"tbase_skip": np.linspace(29, 44, 16) * 1e-3, "Rn_fixed": 6.1e-3, "tbase_max": 0.051},
    #     11: {"tbase_skip": np.array([0.024, 0.025, 0.026, 0.028, 0.032]), "Rn_fixed": 6.0e-3, "tbase_max": 0.05},
    #     13: {"tbase_skip": np.linspace(35, 46, 12) * 1e-3, "Rn_fixed": 6.85e-3, "tbase_max": 0.051},
    #     18: {"tbase_skip": np.concatenate((np.linspace(20, 29, 10), np.linspace(42, 48, 7))) * 1e-3, "Rn_fixed": 6.3e-3,
    #          "tbase_max": 0.053},
    #     22: {"tbase_skip": np.array([0.02, 0.021]), "Rn_fixed": 6.6e-3, "tbase_max": 0.053},
    #     25: {"tbase_skip": np.linspace(24, 43, 20) * 1e-3, "Rn_fixed": 6.7e-3, "tbase_max": 0.052},
    #     26: {"tbase_skip": np.array([]), "Rn_fixed": 6.9e-3, "tbase_max": 0.052},
    #     28: {"tbase_skip": np.linspace(30, 43, 14) * 1e-3, "Rn_fixed": 6.9e-3, "tbase_max": 0.052},
    #     29: {"tbase_skip": np.linspace(31, 44, 14) * 1e-3, "Rn_fixed": 6.9e-3, "tbase_max": 0.052},
    # }
    # # channel_data = {
    #      2: {"tbase_skip": np.array([]), "Rn_fixed": 4.7e-3, "tbase_max": 0.035},
    #     3: {"tbase_skip": np.array([]),
    #         "Rn_fixed": 5e-3, "tbase_max": 0.037},
    #     4: {"tbase_skip": np.array([]),
    #         "Rn_fixed": 5e-3, "tbase_max": 0.04},
    #      5: {"tbase_skip": np.array([]),
    #          "Rn_fixed": 5e-3, "tbase_max": 0.045},
    #     6: {"tbase_skip": np.array([]),
    #         "Rn_fixed": 5e-3, "tbase_max": 0.040},
    #     7: {"tbase_skip": np.array([]),
    #         "Rn_fixed": 5e-3, "tbase_max": 0.042},
    #     8: {"tbase_skip": np.array([]),
    #         "Rn_fixed": 5e-3, "tbase_max": 0.044},
    #     10: {"tbase_skip": np.array([]),
    #          "Rn_fixed": 5e-3, "tbase_max": 0.042},
    #      21: {"tbase_skip": np.array([38,39,40,41,42])*1e-3,
    #           "Rn_fixed": 6e-3, "tbase_max": 0.05},
    #     22: {"tbase_skip": np.linspace(20,33,14)*1e-3,
    #          "Rn_fixed": 6e-3, "tbase_max": 0.05},
    #
    #     23: {"tbase_skip": np.array([]),
    #          "Rn_fixed": 4.5e-3, "tbase_max": 0.05},
    #      24: {"tbase_skip": np.array([0.055]),
    #           "Rn_fixed": 4.5e-3, "tbase_max": 0.05},
    #      25: {"tbase_skip": np.array([0.02,0.043]),
    #           "Rn_fixed": 4.5e-3, "tbase_max": 0.052},
    #     26: {"tbase_skip": np.array([20, 23,24,25,26, 28, 29,30,31,34,35,36])*1e-3,
    #          "Rn_fixed": 4.5e-3, "tbase_max": 0.053},
    #     # 29: {"tbase_skip": np.array([]), "Rn_fixed": 6.9e-3, "tbase_max": 0.052},
    # }

    # Now, retrieve the values:
    if channel_to_plot in channel_data:
        data = channel_data[channel_to_plot]
        tbase_skip = data["tbase_skip"]
        Rn_fixed = data["Rn_fixed"]
        tbase_max = data["tbase_max"]
    else:  # Default values if channel_to_plot is not in the dictionary
        tbase_skip = []
        Rn_fixed = 22e-3
        tbase_max = 0.11

    r_over_rn_ratios_to_plot = np.linspace(70,99,30)*1e-2  # Example ratios
    channels_to_skip = [0, 1,11,12,13,15,26, 27,28,29, 30, 31,35]

    # pixel_map = {2:1, 3:2 ,4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 14:13, 16:14, 17:15, 18:16, 19:17, 20:18,
    #                            21:19, 22:20, 23:21, 24:22, 25:23, 26:24}

    found_files = find_npz_files(data_directory)
    print(found_files)
    if found_files:
        # Plot Ites vs Ibias and Rtes vs Ibias, and Ptes vs Ibias
        plot_iv_and_rtes_vs_ptes(found_files, rbias_value, channel_to_plot, Rshunt=Rshunt_value, Rn_fixed = Rn_fixed, skip_tbase=tbase_skip)
        # Plot Ptes vs. Tbase for a single channel and multiple Rtes/Rn ratios:
        plot_ptes_vs_tbase_multiple_ratios(found_files, rbias_value, channel_to_plot, r_over_rn_ratios_to_plot, Rshunt=Rshunt_value, Rn_fixed = Rn_fixed,
                                            tbase_max=tbase_max, skip_tbase = tbase_skip)
        #
        #
        # # Plot IV curves (all channels on one plot):
        # plot_iv_curves(found_files, rbias_value, channels_to_ignore=channels_to_skip)
        #
        # # Plot IV curves (all channels on one plot):
        # # plot_iv_curves_subplots(found_files, rbias_value, channels_to_ignore=channels_to_skip)
        #
        display_summary_plot(ratio=0.9, pixel_map=pixel_map, save_summary=saveResults)
        # create_g_parameter_summary(pixel_map=pixel_map, target_ratio=0.9)
        plot_Tc_heatmap(ratio=0.9, pixel_map=pixel_map)

    plt.show()

if __name__ == "__main__":
    pixel_map = {0:"25", 1:"23", 2: "21", 3: "19", 4: "17", 5: "15", 6: "13", 7: "11", 8: "9", 9: "7", 10: "5", 11: "3", 12: "1",
                 17: "24", 18: "22", 19: "20", 20: "18", 21: "16", 22: "14", 23: "12", 24: "10", 25: "8", 26: "6", 27: "4", 28: "2"} #umux17a side 1 cooldown A27
    channels_to_skip = [0,6,9,10,13,14,15,16,19, 20,27, 28,29,30] # A27 chip C
    channel_to_plot = 1
    tbase_biasdict = 0.04
    plotFitResults = True
    savePath = "/home/pcuser/Runs/Cooldown_A27/Results/"
    saveResults = False
    main()