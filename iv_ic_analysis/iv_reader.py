import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re  # Import the regular expression module
import scipy.constants
# import lmfit # Not used in the provided snippet
import pandas as pd # Not used in the provided snippet
import warnings
# import pprint # Not used in the provided snippet
# import json # Not used in the provided snippet

# NaN ignored in processing step
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide') # Added for Rtes potential 0 division


phi0 = scipy.constants.value(u"mag. flux quantum")

plt.rcParams['font.size'] = 16

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
    '20um_3sq_SC_150':    '#2ca02c',  # Dark Green
    '20um_3sq_SC_150_E':  '#8ae08a',  # Light Green
    '20um_2sq_SC':        '#ff7f0e',  # Dark Orange
    '20um_2sq_SC_E':      '#ffbb78',  # Light Orange
    '20um_3sq_barr_SC':   '#d62728',  # Dark Red
    '20um_3sq_barr_SC_E': '#ff9896',  # Light Red
    '20um_2sq_barr_SC':   '#9467bd',  # Dark Purple
    '20um_2sq_barr_SC_E': '#c5b0d5'   # Light Purple
}




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

    # Simplified pattern to find all npz files first, then filter
    pattern = os.path.join(directory, f"*.npz")
    files = glob.glob(pattern)

    # Regex to extract Tbase from filenames like "..._XXXmK.npz"
    tbase_regex = re.compile(r"_(\d+)mK\.npz$")

    for file_path in files:
        filename = os.path.basename(file_path)
        match = tbase_regex.search(filename)
        if match:
            try:
                tbase_mK = int(match.group(1))
                tbase = tbase_mK / 1000.0
                npz_files[tbase] = file_path
            except ValueError:
                print(f"Warning: Could not extract valid integer Tbase from '{filename}'. Skipping.")
        # else: # Optional: Warn if a file doesn't match the Tbase pattern
        #     print(f"Info: Filename '{filename}' does not match Tbase pattern. Skipping Tbase association.")

    # Fallback if no Tbase files found, maybe list all npz found?
    if not npz_files and files:
        print("Warning: No files matched the Tbase pattern (_XXXmK.npz).")
        # You could return all found .npz files differently if needed, e.g., as a list
        # return {'all': files}
    elif not files:
        print(f"Warning: No .npz files found in '{directory}'.")


    return npz_files


def remove_offset(arr):
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        print("Warning: Input to remove_offset is not a 2D NumPy array.")
        return np.array([])  # Handle invalid input
    if arr.size == 0 or arr.shape[1] <= 1: # Need at least 2 points to have a meaningful offset
        # print("Warning: Input array to remove_offset is empty or has only one column.")
        return arr # Return original if too small to apply offset meaningfully

    last_elements = arr[:, -1]  # Get the last element of each row
    return arr - last_elements[:, np.newaxis]

def arb_to_amp(in_val):
    """
    Converts arbitrary units (assumed proportional to flux quanta) to Amperes.

    Args:
        in_val (float or np.ndarray): The input value(s) in arbitrary units (phi0).

    Returns:
        float or np.ndarray: The equivalent value(s) in Amperes.
    """
    # These values define the conversion factor

    min_phi0_per_amp = min_SI / phi0

    return in_val /min_phi0_per_amp


def Rtes(ibias, ites, Rshunt=250e-6):
    """
    Calculates TES resistance (Rtes).

    Args:
        ibias (np.ndarray): Array of Ibias values (in Amperes).
        ites (np.ndarray): Array of Ites values (in Amperes).
        Rshunt (float): Shunt resistance (in Ohms). Defaults to 250e-6.

    Returns:
        np.ndarray: Array of Rtes values (in Ohms). Handles division by zero
                    by returning NaN where Ites is zero.
    """
    # Avoid division by zero: replace ites=0 with NaN temporarily
    ites_safe = np.where(ites == 0, np.nan, ites)
    rtes_values = Rshunt * (ibias - ites_safe) / ites_safe
    return rtes_values


def Ptes(ibias, ites, Rshunt=250e-6):
    """
    Calculates power dissipated in the TES (Ptes).

    Args:
        ibias (np.ndarray): Array of Ibias values (in Amperes).
        ites (np.ndarray): Array of Ites values (in Amperes).
        Rshunt (float): Shunt resistance (in Ohms). Defaults to 250e-6.

    Returns:
        np.ndarray: Array of Ptes values (in Watts).
    """
    rtes_values = Rtes(ibias, ites, Rshunt)
    # Ptes is NaN if Rtes is NaN (i.e., if Ites was zero)
    ptes_values = rtes_values * ites ** 2
    return ptes_values


def convert_ang2_to_ites(ang2, channel_id, correct_shift = True):
    """
    Converts ang2 values (phase angle, assumed proportional to flux) for a
    specific channel to Ites values (TES current).

    Args:
        ang2 (np.ndarray): 2D array of ang2 values (rows=sweep points, cols=channels).
        channel_id (int): The index of the channel to convert.
        correct_shift (bool): If True, subtracts the last value to remove offset,
                              assuming the last point corresponds to zero TES current.
                              Defaults to True.

    Returns:
        np.ndarray: 1D array of Ites values (in Amperes) for the specified channel.
                    Returns an empty array if channel_id is invalid.
    """
    if not isinstance(ang2, np.ndarray) or ang2.ndim != 2:
        print("Error: 'ang2' must be a 2D NumPy array.")
        return np.array([])
    if not 0 <= channel_id < ang2.shape[1]:
        print(f"Error: channel_id {channel_id} is out of range (0-{ang2.shape[1]-1}).")
        return np.array([])
    if ang2.shape[0] == 0:
        print("Warning: 'ang2' array has zero rows.")
        return np.array([])

    # Select the specific channel
    ang2_channel = ang2[:, channel_id]

    # Convert arbitrary units (flux) to Amperes
    ites_uncorrected = arb_to_amp(ang2_channel)

    if correct_shift:
        if ites_uncorrected.size > 0:
            # Subtract the last value as the offset
            return ites_uncorrected - ites_uncorrected[-1]
        else:
            return ites_uncorrected # Return empty if input was empty
    else:
        return ites_uncorrected


def convert_vbias_to_ibias(vbias, rbias):
    """
    Converts Vbias (voltage across bias resistor) to Ibias (total bias current).

    Args:
        vbias (np.ndarray): Array of Vbias values (in Volts).
        rbias (float): Bias resistance (in Ohms).

    Returns:
        np.ndarray: Array of Ibias values (in Amperes). Returns empty array
                    if rbias is zero or non-numeric.
    """
    if not isinstance(rbias, (int, float)) or rbias == 0:
        print("Error: rbias must be a non-zero number.")
        return np.array([])
    if not isinstance(vbias, np.ndarray):
         print("Error: vbias must be a NumPy array.")
         return np.array([])

    return vbias / rbias


def plot_iv_and_rtes_vs_ibias(npz_file, rbias, channel_ids,
                              Rshunt=250e-6, Rn_fixed = None, xaxis = 'ibias'):
    """
    Plots Ites vs. X, Rtes vs. X, and Ptes vs. X for specific channels
    from an NPZ file, where X is either Ibias or Vbias.

    Args:
        npz_file (str): Path to the .npz file containing IV data.
        rbias (float): Bias resistance (in Ohms).
        channel_ids (int or list/tuple): The channel ID(s) to plot.
        Rshunt (float): Shunt resistance (in Ohms). Defaults to 250e-6.
        Rn_fixed (float, optional): Fixed normal resistance (Rn) value in Ohms
                                     to plot as a horizontal line on the Rtes plot.
                                     Defaults to None.
        xaxis (str): Specifies the x-axis quantity: 'ibias' or 'vbias'.
                     Defaults to 'ibias'.
    """
    if not npz_file or not os.path.exists(npz_file):
        print(f"Error: NPZ file not found or not specified: '{npz_file}'")
        return

    # --- Input Validation/Normalization for channel_ids ---
    if isinstance(channel_ids, int):
        channel_ids = [channel_ids] # Convert single int to list
    elif not isinstance(channel_ids, (list, tuple)) or not all(isinstance(cid, int) for cid in channel_ids):
         print("Error: channel_ids must be an integer or a list/tuple of integers.")
         return
    if not channel_ids:
         print("Warning: No channel IDs provided for plotting.")
         return # Nothing to plot

    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5))  # Taller aspect ratio for slide shows

    try:
        with np.load(npz_file) as data:
            if 'vb' not in data or 'ang2' not in data:
                print(f"Error: 'vb' or 'ang2' not found in '{npz_file}'. Cannot plot.")
                plt.close(fig) # Close the empty figure
                return

            vbias = data['vb']
            ang2 = data['ang2']
            num_channels_in_file = ang2.shape[1]

            # --- Determine X-axis data ---
            ibias = convert_vbias_to_ibias(vbias, rbias)
            if not ibias.size: # Check if conversion failed
                 plt.close(fig)
                 return

            if xaxis == 'ibias':
                xdata = ibias
                xlabel = 'Ibias (mA)'
                x_scale = 1e3
                y_scale = 1e3
            elif xaxis == 'vbias':
                xdata = vbias
                xlabel = 'Vbias (mV)'
                x_scale = 1e3
                y_scale = 1/rbias * 1e3
            else:
                print(f"Error: Invalid xaxis type '{xaxis}'. Choose 'ibias' or 'vbias'.")
                plt.close(fig)
                return

            # --- Loop through specified channels ---
            plotted_any = False
            max_x_val = 0.0
            max_ites_val = 0.0
            for channel_id in channel_ids:
                if not 0 <= channel_id < num_channels_in_file:
                    print(f"Warning: Channel {channel_id} is out of range (0-{num_channels_in_file - 1}) in '{os.path.basename(npz_file)}'. Skipping.")
                    continue

                # Calculate Ites, Rtes, Ptes for the current channel
                # Assuming offset correction is desired for these plots
                ites = convert_ang2_to_ites(ang2, channel_id, correct_shift=True)
                if not ites.size: # Check if conversion failed
                    print(f"Warning: Could not calculate Ites for channel {channel_id}. Skipping.")
                    continue

                rtes = Rtes(ibias, ites, Rshunt)
                ptes = Ptes(ibias, ites, Rshunt)
                vtes = ites*rtes

                # Track peak values
                max_ites_val = max(max_ites_val, np.max(ites) * 1e3)
                idx_peak = np.argmax(ites)
                x_peak = xdata[idx_peak] * x_scale
                if x_peak > max_x_val:
                    max_x_val = x_peak

                # Use pixel map if available, otherwise fall back to channel
                suffix_chip = "chipA" if channel_ids[0] < 32 else "chipB"
                try:
                    if channel_id in pixel_map:
                        pixel_num = int(pixel_map[channel_id])
                        cell_name = PIXEL_CELL_MAP.get(pixel_num, "Unknown")
                        label_name = cell_name
                        if suffix_chip == "chipB":
                            label_name = cell_name.replace("20um_", "30um_")
                        color = CELL_COLOR_MAP.get(cell_name, 'gray')
                    else:
                        label_name = f"Ch {channel_id}"
                        color = 'gray'
                except NameError:
                    label_name = f"Ch {channel_id}"
                    color = 'gray'

                # Only add unique cell geometries to the legend to avoid overfilling
                if 'added_labels_set' not in locals():
                    added_labels_set = set()
                if label_name not in added_labels_set:
                    plot_label = label_name
                    added_labels_set.add(label_name)
                else:
                    plot_label = ""

                # --- Plotting for the current channel ---
                # Ites vs. X Plot
                axes[0].plot(xdata * x_scale, ites * 1e3, color=color, label=plot_label, lw=2.0)

                # Rtes vs. X Plot
                axes[1].plot(xdata * x_scale, rtes * 1e3, color=color, label=plot_label, lw=2.0)

                # Ptes vs. X Plot
                axes[2].plot( ptes*1e12, rtes * 1e3, color=color, label=plot_label, lw=2.0)

                plotted_any = True # Mark that at least one channel was plotted

                # Annotate pixel number at end of R vs Ptes curve
                try:
                    if channel_id in pixel_map:
                        label_text = pixel_map[channel_id]
                    else:
                        label_text = str(channel_id)
                except NameError:
                    label_text = str(channel_id)
                axes[1].annotate(
                    label_text,
                    xy=(xdata[0]*x_scale, rtes[0] * 1e3),  # endpoint in plot units
                    xytext=(5, 0),  # small offset to the right
                    textcoords="offset points",
                    color=color,  # match line color
                    fontsize=9,
                    ha="left",
                    va="center"
                )
                axes[2].annotate(
                    label_text,
                    xy=(ptes[0] * 1e12, rtes[0] * 1e3),  # endpoint in plot units
                    xytext=(5, 0),  # small offset to the right
                    textcoords="offset points",
                    color=color,  # match line color
                    fontsize=9,
                    ha="left",
                    va="center"
                )

            if not plotted_any:
                print("Warning: None of the specified channels could be plotted.")
                plt.close(fig)
                return

            # --- Plot Settings (applied once after loop) ---
            # Ites plot settings
            axes[0].plot(xdata * x_scale, xdata * y_scale, 'k--', lw=1.5, label='MI Line') # line at x=y for MI check
            axes[0].set_xlabel(xlabel, fontweight='bold')
            axes[0].set_ylabel(r"Ites (mA)", fontweight='bold')
            axes[0].set_title(f"Ites vs. {xlabel.split(' ')[0]}", fontweight='bold')
            axes[0].legend(fontsize=10, loc='best')
            # Zoom in y-axis of leftmost plot to just above Ites peak, keeping x-limits same
            if max_ites_val > 0.0:
                axes[0].set_ylim(0, 1.15 * max_ites_val)
            axes[0].grid(True, ls=':', alpha=0.6)

            # Rtes plot settings
            axes[1].set_xlabel(xlabel, fontweight='bold')
            axes[1].set_ylabel(r"Rtes (m$\Omega$)", fontweight='bold')
            axes[1].set_title(f"Rtes vs. {xlabel.split(' ')[0]}", fontweight='bold')
            axes[1].legend(fontsize=10, loc='best')
            axes[1].grid(True, ls=':', alpha=0.6)

            # Ptes plot settings
            axes[2].set_xlabel("Ptes (pW)", fontweight='bold')
            axes[2].set_ylabel("Rtes (mΩ)", fontweight='bold')
            axes[2].set_title(f"Ptes vs. Rtes", fontweight='bold')
            axes[2].legend(fontsize=10, loc='best')
            axes[2].grid(True, ls=':', alpha=0.6)

            plt.suptitle(f"IV Characteristics from {os.path.basename(npz_file)}", fontsize=18, fontweight='bold')
            
            # Remove excessive whitespace: adjust layout and spacing
            plt.subplots_adjust(wspace=0.18)
            plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
            
            # Save the figure with dynamic suffix
            suffix = "chipA" if channel_ids[0] < 32 else "chipB"
            os.makedirs("C:/Users/anr29/Downloads/aroy/Cooldown-B8/iv-analysis", exist_ok=True)
            plt.savefig(f"C:/Users/anr29/Downloads/aroy/Cooldown-B8/iv-analysis/iv_and_rtes_vs_ibias_{suffix}.png", dpi=200)


    except FileNotFoundError:
        # This case is already handled at the beginning, but kept for robustness
        print(f"Error: File not found: '{npz_file}'.")
        plt.close(fig)
    except Exception as e:
        print(f"An error occurred loading/processing '{npz_file}': {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        plt.close(fig)


def plot_ptes_vs_rtes(npz_file, rbias, channel_ids,
                              Rshunt=250e-6,):
    """
    Plots Ites vs. X, Rtes vs. X, and Ptes vs. X for specific channels
    from an NPZ file, where X is either Ibias or Vbias.

    Args:
        npz_file (str): Path to the .npz file containing IV data.
        rbias (float): Bias resistance (in Ohms).
        channel_ids (int or list/tuple): The channel ID(s) to plot.
        Rshunt (float): Shunt resistance (in Ohms). Defaults to 250e-6.
        Rn_fixed (float, optional): Fixed normal resistance (Rn) value in Ohms
                                     to plot as a horizontal line on the Rtes plot.
                                     Defaults to None.
        xaxis (str): Specifies the x-axis quantity: 'ibias' or 'vbias'.
                     Defaults to 'ibias'.
    """
    if not npz_file or not os.path.exists(npz_file):
        print(f"Error: NPZ file not found or not specified: '{npz_file}'")
        return

    # --- Input Validation/Normalization for channel_ids ---
    if isinstance(channel_ids, int):
        channel_ids = [channel_ids] # Convert single int to list
    elif not isinstance(channel_ids, (list, tuple)) or not all(isinstance(cid, int) for cid in channel_ids):
         print("Error: channel_ids must be an integer or a list/tuple of integers.")
         return
    if not channel_ids:
         print("Warning: No channel IDs provided for plotting.")
         return # Nothing to plot

    fig, axes = plt.subplots(figsize=(10, 10))  # 1 row, 3 columns


    with np.load(npz_file) as data:
        if 'vb' not in data or 'ang2' not in data:
            print(f"Error: 'vb' or 'ang2' not found in '{npz_file}'. Cannot plot.")
            plt.close(fig) # Close the empty figure
            return

        vbias = data['vb']
        ang2 = data['ang2']
        num_channels_in_file = ang2.shape[1]

        # --- Determine X-axis data ---
        ibias = convert_vbias_to_ibias(vbias, rbias)
        if not ibias.size: # Check if conversion failed
             plt.close(fig)
             return

        # --- Loop through specified channels ---
        plotted_any = False
        for channel_id in channel_ids:
            if not 0 <= channel_id < num_channels_in_file:
                print(f"Warning: Channel {channel_id} is out of range (0-{num_channels_in_file - 1}) in '{os.path.basename(npz_file)}'. Skipping.")
                continue

            # Calculate Ites, Rtes, Ptes for the current channel
            # Assuming offset correction is desired for these plots
            ites = convert_ang2_to_ites(ang2, channel_id, correct_shift=True)
            if not ites.size: # Check if conversion failed
                print(f"Warning: Could not calculate Ites for channel {channel_id}. Skipping.")
                continue

            rtes = Rtes(ibias, ites, Rshunt)
            ptes = Ptes(ibias, ites, Rshunt)
            vtes = ites*rtes

            # Use pixel map if available, otherwise fall back to channel
            try:
                if channel_id in pixel_map:
                    plot_label = f"{pixel_map[channel_id]} (Ch {channel_id})"
                else:
                    plot_label = f"Ch {channel_id}"
            except NameError:
                plot_label = f"Ch {channel_id}"


            # --- Plotting for the current channel ---
            axes.plot(ptes*1e12, rtes * 1e3, label=plot_label, color= 'k', alpha=0.6)
            try:
                label_text = pixel_map[channel_id]
            except KeyError:
                label_text = f"Ch {channel_id}"
            axes.annotate(
                label_text,
                xy=(ptes[0] * 1e12, rtes[0] * 1e3),  # endpoint in plot units
                xytext=(5, 0),  # small offset to the right
                textcoords="offset points",
                color=axes.lines[-1].get_color(),  # match line color
                fontsize=10,
                ha="left",
                va="center"
            )

            plotted_any = True # Mark that at least one channel was plotted

            # Annotate pixel number at end of R vs Ptes curve
            try:
                if channel_id in pixel_map:
                    label_text = pixel_map[channel_id]
                    print(label_text)
                else:
                    label_text = str(channel_id)
            except NameError:
                label_text = f"Ch {channel_id}"



        if not plotted_any:
            print("Warning: None of the specified channels could be plotted.")
            plt.close(fig)
            return

        # --- Plot Settings (applied once after loop) ---
        # Ites plot settings
        axes.set_title(f"{filename}")
        axes.set_xlabel("Ptes (pW)")
        axes.set_ylabel(r"Rtes (m$\Omega$)")
        axes.set(xlim=(0,8),ylim=(0,30))
        # axes.legend(title = 'Pixel # (Channel ID)', ncol=2)
        axes.grid(True)
        
        # Save the figure with dynamic suffix
        suffix = "chipA" if channel_ids[0] < 32 else "chipB"
        os.makedirs("C:/Users/anr29/Downloads/aroy/Cooldown-B8/iv-analysis", exist_ok=True)
        plt.savefig(f"C:/Users/anr29/Downloads/aroy/Cooldown-B8/iv-analysis/ptes_vs_rtes_{suffix}.png", dpi=200)

def plot_iv_curves_subplots(npz_file, rbias, channels_to_ignore=None, correct_shift = True, y_unit = 'flux'):
    """
    Plots IV curves with each channel in a separate subplot.

    Args:
        npz_file (str): Path to the .npz file containing IV data.
        rbias (float): Bias resistance (in Ohms).
        channels_to_ignore (list or set, optional): Channel IDs to skip.
                                                    Defaults to None (plot all).
        correct_shift (bool): Whether to apply offset correction to Ites.
                              Defaults to True.
        y_unit (str): Unit for the y-axis: 'flux' (arbitrary units proportional
                      to phi0) or 'current' (mA). Defaults to 'flux'.
    """
    if not npz_file or not os.path.exists(npz_file):
        print(f"Error: NPZ file not found or not specified: '{npz_file}'")
        return

    try:
        with np.load(npz_file) as data:
            if 'vb' not in data or 'ang2' not in data:
                print(f"Error: 'vb' or 'ang2' not found in '{npz_file}'. Cannot plot.")
                return

            vbias = data['vb']
            ang2 = data['ang2']
            num_channels_total = ang2.shape[1]

            ibias = convert_vbias_to_ibias(vbias, rbias)
            if not ibias.size: # Check if conversion failed
                 return

            # Determine which channels to plot
            if channels_to_ignore is None:
                channels_to_ignore = set()
            else:
                # Ensure it's a set for efficient lookup
                channels_to_ignore = set(channels_to_ignore)

            # Create a sorted list of channels to actually plot
            good_channels = sorted([ch for ch in range(num_channels_total) if ch not in channels_to_ignore])

            if not good_channels:
                print("Warning: No channels left to plot after ignoring specified ones.")
                return

            num_channels_to_plot = len(good_channels)
            num_cols = 4 # Fixed number of columns
            num_rows = (num_channels_to_plot + num_cols - 1) // num_cols # Calculate needed rows

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows),
                                     sharex=True, sharey=True, squeeze=False) # squeeze=False ensures axes is always 2D array
            plt.subplots_adjust(top=0.95, bottom=0.08, right=0.98, left=0.07, hspace=0.25, wspace=0.1) # Adjusted spacing
            axes_flat = axes.flatten() # Flatten for easy indexing

            # --- Plotting Loop ---
            for i, channel_id in enumerate(good_channels):
                ax = axes_flat[i] # Get the current subplot axis

                # Determine Y data based on selected unit
                if y_unit == 'current':
                    ydata = convert_ang2_to_ites(ang2, channel_id, correct_shift=correct_shift)
                    if ydata.size: # Check conversion success
                        ydata *= 1e3 # Convert A to mA
                        ylabel = "TES current (mA)"
                    else:
                        print(f"Warning: Skipping plot for channel {channel_id} due to Ites calculation issue.")
                        ax.set_title(f"Ch {channel_id} (Error)") # Use SMuRF chan convention
                        ax.text(0.5, 0.5, 'Data Error', ha='center', va='center', transform=ax.transAxes)
                        continue # Skip to next channel
                elif y_unit == 'flux':
                    # Assuming ang2 is directly proportional to flux quanta
                    ydata = ang2[:, channel_id]
                    if correct_shift and ydata.size > 0:
                        ydata = ydata - ydata[-1] # Apply offset correction if requested for flux
                    ylabel = r"TES Flux ($\propto \phi_0$)" # More precise label
                else:
                    print(f"Error: Invalid y_unit '{y_unit}'. Choose 'flux' or 'current'.")
                    plt.close(fig)
                    return

                # Actual plotting
                ax.plot(ibias * 1e3, ydata, label=f"Ch {channel_id}") # Use SMuRF chan convention
                ax.grid(True)
                ax.legend(loc='best') # Add legend to each subplot

                # Add titles to subplots only if space allows (maybe only first row/col?)
                # ax.set_title(f"Ch {channel_id+4096}") # Optional: Title on each subplot

            # --- Clean up unused axes ---
            for k in range(num_channels_to_plot, num_rows * num_cols):
                fig.delaxes(axes_flat[k])

            # Add overall labels and title
            fig.suptitle(f"IV Curves from {os.path.basename(npz_file)} (Y-Unit: {y_unit})", fontsize=20, fontweight='bold')
            fig.text(0.5, 0.02, 'Bias Current (mA)', ha='center', va='bottom', fontsize=16, fontweight='bold')
            fig.text(0.02, 0.5, ylabel, ha='left', va='center', rotation='vertical', fontsize=16, fontweight='bold')
            
            # Save the figure with dynamic suffix
            active_channels = [c for c in range(67) if channels_to_ignore is None or c not in channels_to_ignore]
            suffix = "chipA" if len(active_channels) > 0 and active_channels[0] < 32 else "chipB"
            os.makedirs("C:/Users/anr29/Downloads/aroy/Cooldown-B8/iv-analysis", exist_ok=True)
            plt.savefig(f"C:/Users/anr29/Downloads/aroy/Cooldown-B8/iv-analysis/iv_curves_subplots_{suffix}.png", dpi=200)

    except FileNotFoundError:
        # Already handled at the start, but good practice
        print(f"Error: File not found: '{npz_file}'.")
    except Exception as e:
        print(f"An error occurred loading or plotting data from '{npz_file}': {e}")
        import traceback
        traceback.print_exc()
        # Ensure figure is closed if an error occurs mid-plotting
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)

def get_ites_from_iv_curve(iv_npz_file, rbias, channel_id, correct_shift=True):
    """
    Loads IV curve data and returns Vbias and Ites for a specific channel.

    Args:
        iv_npz_file (str): Path to the IV .npz file.
        rbias (float): Bias resistance (in Ohms).
        channel_id (int): The channel ID to extract data for.
        correct_shift (bool): Whether to apply offset correction to Ites.

    Returns:
        tuple: (vbias_array, ites_array) or (None, None) if data not found/error.
    """
    if not iv_npz_file or not os.path.exists(iv_npz_file):
        print(f"Error: IV NPZ file not found or not specified: '{iv_npz_file}'")
        return None, None
    try:
        with np.load(iv_npz_file) as data:
            if 'vb' not in data or 'ang2' not in data:
                print(f"Error: 'vb' or 'ang2' not found in '{iv_npz_file}'.")
                return None, None
            vbias = data['vb']
            ang2 = data['ang2']
            if not 0 <= channel_id < ang2.shape[1]:
                print(f"Error: channel_id {channel_id} out of range in '{iv_npz_file}'.")
                return None, None

            ites = convert_ang2_to_ites(ang2, channel_id, correct_shift=correct_shift)
            return vbias, ites
    except Exception as e:
        print(f"An error occurred loading IV data from '{iv_npz_file}': {e}")
        return None, None



if __name__ == "__main__":
    filename = "C:/Users/anr29/Downloads/ravendata-dtest62/iv/20260709_155350_iv_IV_20.0mK.npz"

    # Define pixel maps
    PIXEL_MAP_A = {
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
    
    try:
        if os.path.isabs(filename):
            npz_file = filename
        else:
            date_str = filename.split('_')[0]
            npz_file = f'/data/{date_str}/iv/{filename}'

        # Filter active channels based on signal variance
        data = np.load(npz_file)
        std_ang2 = np.std(data['ang2'], axis=0)
        good_channels = std_ang2 > 0.05

        rbias_value =   1980  
        Rshunt_value =  250e-6
        Rn_estimate = 0.01 
        global min_SI

        # ----------------------------------------------------
        # ANALYZE CHIP A
        # ----------------------------------------------------
        print("\n=== Analyzing Chip A ===")
        pixel_map = PIXEL_MAP_A
        min_SI = 240.72e-12  # Calibrated min_SI for Chip A
        channels_a = [ch for ch in PIXEL_MAP_A.keys() if good_channels[ch] and ch not in [11, 23, 27, 28]]
        exclude_a = [c for c in range(67) if c not in channels_a]

        print(f"Plotting Ites, Rtes, Ptes for Chip A channels {channels_a}...")
        plot_iv_and_rtes_vs_ibias(npz_file,
                                  rbias=rbias_value,
                                  channel_ids=channels_a, 
                                  Rshunt=Rshunt_value,
                                  Rn_fixed=Rn_estimate,
                                  xaxis='ibias') 

        print(f"Plotting individual IV curves for Chip A...")
        plot_iv_curves_subplots(npz_file,
                                rbias=rbias_value,
                                channels_to_ignore=exclude_a,
                                correct_shift=True,
                                y_unit='current') 
        
        plot_ptes_vs_rtes(npz_file, rbias=rbias_value, channel_ids=channels_a, Rshunt=Rshunt_value)

        # ----------------------------------------------------
        # ANALYZE CHIP B
        # ----------------------------------------------------
        print("\n=== Analyzing Chip B ===")
        pixel_map = PIXEL_MAP_B
        min_SI = 238.5e-12  # Calibrated min_SI for Chip B
        channels_b = [ch for ch in PIXEL_MAP_B.keys() if good_channels[ch] and ch != 42] # Drop pixel 11
        exclude_b = [c for c in range(67) if c not in channels_b]

        print(f"Plotting Ites, Rtes, Ptes for Chip B channels {channels_b}...")
        plot_iv_and_rtes_vs_ibias(npz_file,
                                  rbias=rbias_value,
                                  channel_ids=channels_b, 
                                  Rshunt=Rshunt_value,
                                  Rn_fixed=Rn_estimate,
                                  xaxis='ibias') 

        print(f"Plotting individual IV curves for Chip B...")
        plot_iv_curves_subplots(npz_file,
                                rbias=rbias_value,
                                channels_to_ignore=exclude_b,
                                correct_shift=True,
                                y_unit='current') 
        
        plot_ptes_vs_rtes(npz_file, rbias=rbias_value, channel_ids=channels_b, Rshunt=Rshunt_value)

        print("\nAll plots saved to Cooldown-B8/iv-analysis/!")

    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()