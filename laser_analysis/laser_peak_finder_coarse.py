import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
# Removed Sherpa imports as they are not used in the final provided code
# from sherpa.models import Gauss1D
# from sherpa.fit import Fit
# from sherpa.data import Data1D
# from sherpa.stats import CStat
# from sherpa.plot import DataPlot, ModelPlot, FitPlot

def load_and_plot_histogram(csv_path, binwidth, _find_peaks = False):
    """
    Loads a CSV, plots a histogram of '5lagy' colored by 'STATE',
    and finds/marks peaks.

    Args:
        csv_path (str): Path to the CSV file.
        binwidth (float): Width of the histogram bins.
        _find_peaks (bool): Although present, the main peak finding logic is always active.
                           The flag is kept for compatibility but doesn't switch logic off.

    Returns:
        tuple: (pandas.DataFrame, dict) or (None, None)
               - DataFrame containing the loaded and filtered data.
               - Dictionary containing peak positions for each state.
               Returns (None, None) if file/columns not found or errors occur.
    """

    try:
        df = pd.read_csv(csv_path)
        # Ensure '5lagy' is numeric, coercing errors to NaN, then drop rows with NaN in '5lagy'
        df['5lagy'] = pd.to_numeric(df['5lagy'], errors='coerce')
        df.dropna(subset=['5lagy'], inplace=True)
        # Filter out negative '5lagy' values AFTER ensuring it's numeric
        df = df[df['5lagy'] >= 0]
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error reading or processing CSV: {e}")
        return None, None

    if '5lagy' not in df.columns or 'state_label' not in df.columns:
        print("Error: '5lagy' or 'state_label' columns not found in CSV.")
        return None, None

    # Define state-specific parameters
    distance_dict = {'A': 15, 'E':15, 'G':14, 'H':14, 'J':12, 'K':11, 'L':12, 'M':8}
    # Threshold_dict seems unused currently, commenting out
    # threshold_dict = {'E':1 , 'G':1, 'H':1, 'J':1, 'K':1, 'L':1, 'M':1}
    range_dict = {'A':[0,10], 'E':[0,30], 'G':[5, 70], 'H':[48,146], 'J':[55,155], 'K':[231,394], 'L':[147,342], 'M':[350,608]}

    plt.figure(figsize=(12, 7))  # Adjusted figure size
    peak_positions = {}
    processed_states = [] # Keep track of states successfully processed

    # Define a colormap for potentially many states
    colors = plt.cm.viridis(np.linspace(0, 1, len(distance_dict)))

    # Iterate through the states defined in distance_dict
    for i, state in enumerate(distance_dict.keys()):
        print(f"Processing state: {state}")
        # Filter data for the current state and within its specified range
        state_data = df[df['state_label'] == state]['5lagy']
        if state in range_dict:
             state_data = state_data[(state_data > range_dict[state][0]) & (state_data < range_dict[state][1])]
        else:
             print(f"Warning: No range defined for state '{state}'. Using all data.")


        if state_data.empty:
            print(f"No data found for state '{state}' within the specified range.")
            peak_positions[state] = np.array([]) # Store empty array for this state
            continue # Skip to the next state

        min_val = state_data.min()
        max_val = state_data.max()

        # Ensure bin edges make sense even if min_val == max_val
        if max_val <= min_val:
             max_val = min_val + binwidth # Add binwidth if only one value or empty after range filter
        bins = np.arange(min_val, max_val + binwidth, binwidth)

        # Calculate histogram counts and bin edges
        counts, bin_edges = np.histogram(state_data, bins=bins)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin midpoints

        # Plot the histogram for this state
        plt.hist(state_data, bins=bins, alpha=0.6, label=f"{state} ({len(state_data)} pts)", color=colors[i]) # More informative label

        # Find peaks using scipy
        try:
            # Adjust prominence dynamically? Or use a fixed reasonable value. Prominence=5 is used here.
            peaks_indices, _ = find_peaks(counts, prominence=5, distance=distance_dict[state])
            if len(peaks_indices) > 0:
                 found_peaks = bin_midpoints[peaks_indices]
                 peak_counts = counts[peaks_indices]
                 # Mark peaks on the histogram plot
                 plt.plot(found_peaks, peak_counts, '*', markersize=8, color=colors[i], markeredgecolor='black')
                 peak_positions[state] = found_peaks
                 processed_states.append(state) # Mark state as processed
                 print(f"  Found {len(found_peaks)} peaks for state {state}.")
            else:
                 print(f"  No peaks found for state {state} with current settings.")
                 peak_positions[state] = np.array([]) # Store empty array
        except Exception as e:
            print(f"Error finding peaks for state {state}: {e}")
            peak_positions[state] = np.array([]) # Store empty array on error


    plt.xlabel('5lagy Value')
    plt.ylabel('Frequency (Counts)')
    plt.title('Histogram of 5lagy by State with Peak Finding')
    plt.legend(title='State (Points)', bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    print("\nPeak Positions Found:")
    for state, peaks in peak_positions.items():
        print(f"  {state}: {len(peaks)} peaks at {np.round(peaks, 2)}")

    # Only return df if it was loaded successfully
    return df if 'df' in locals() else None, peak_positions

def ns_to_Es(x):
    """Converts photon number (n) to Energy (eV)"""
    # Ensure input is treated as numpy array for vectorized operations
    # Using the provided conversion factor
    return 1239 / 515 * np.array(x)

def fitFunc4(x, a, b, c, d, e):
    """5th order polynomial for fitting."""
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x

def make_nonlinearity_plot(peak_positions, output_csv_path):
    """
    Analyzes peak positions, plots nonlinearity (n vs center and E vs center),
    fits a polynomial to E vs center, plots residuals, and saves peak data to a CSV file.

    Args:
        peak_positions (dict): Dictionary where keys are states and values are arrays of peak centers.
        output_csv_path (str): Path to save the resulting CSV file.

    Returns:
        tuple: (numpy.array, numpy.array) or (None, None)
               - Sorted energy values (eV) used in the fit.
               - Corresponding sorted peak center values (5lagy) used in the fit.
               Returns (None, None) if no valid data for fitting.
    """
    n0_guess_dict = {'A':0, 'E':0, 'G':3, 'H':18, 'J':20, 'K':103, 'L':60, 'M':174}
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax.set_title("Detector Nonlinearity Analysis")

    # Define the maximum number of peaks per state to store/analyze
    max_peaks = 200 # Adjust this value based on the maximum relevant peaks you expect

    data_dict = {} # To store data for the DataFrame (padded/truncated)
    all_ns_fit = []      # Collect all valid n values for fitting
    all_centers_fit = [] # Collect all valid center values for fitting

    # Define colors for plotting consistency if needed, or use defaults
    colors = plt.cm.viridis(np.linspace(0, 1, len(peak_positions)))

    plot_handles = [] # For combined legend later

    for i, state in enumerate(peak_positions.keys()):
        if state not in n0_guess_dict:
            print(f"Warning: State '{state}' not found in n0_guess_dict. Skipping analysis for this state.")
            # Still create empty columns in the DataFrame if desired
            data_dict[f'ns_{state}'] = np.full(max_peaks, np.nan)
            data_dict[f'centers_{state}'] = np.full(max_peaks, np.nan)
            continue

        n0 = n0_guess_dict[state]
        centers_state_raw = peak_positions[state] # Raw peak centers found for this state
        num_peaks_found = len(centers_state_raw)

        if num_peaks_found == 0:
            #print(f"Info: No peaks were input for state '{state}'.") # Already printed in previous func
            # Prepare empty arrays padded to max_peaks for the DataFrame
            ns_state_padded = np.full(max_peaks, np.nan)
            centers_state_padded = np.full(max_peaks, np.nan)
        else:
            # Generate photon numbers corresponding to the raw peaks
            ns_state_raw = n0 + np.arange(num_peaks_found)
            ns_state_raw = ns_state_raw.astype(float)

            # --- Truncation Logic ---
            if num_peaks_found > max_peaks:
                print(f"Warning: State '{state}' found {num_peaks_found} peaks. Truncating to {max_peaks} for analysis and CSV.")
                ns_state_truncated = ns_state_raw[:max_peaks]
                centers_state_truncated = centers_state_raw[:max_peaks]
            else:
                ns_state_truncated = ns_state_raw
                centers_state_truncated = centers_state_raw
            # --- End Truncation Logic ---

            # Add the valid (truncated, non-NaN) data to lists for fitting
            all_ns_fit.extend(ns_state_truncated)
            all_centers_fit.extend(centers_state_truncated)

            # Plotting only the valid (truncated) data points for n vs center
            line, = ax.plot(centers_state_truncated, ns_state_truncated, '.', label=state, color=colors[i], markersize=5)
            plot_handles.append(line) # Add handle for state legend

            # --- Padding Logic (for DataFrame export) ---
            pad_after = max_peaks - len(ns_state_truncated) # Calculate padding needed
            ns_state_padded = np.pad(ns_state_truncated, (0, pad_after), 'constant', constant_values=np.nan)
            centers_state_padded = np.pad(centers_state_truncated, (0, pad_after), 'constant', constant_values=np.nan)
            # --- End Padding Logic ---

        # Store the potentially truncated and padded data in the dictionary for the DataFrame
        data_dict[f'ns_{state}'] = ns_state_padded
        data_dict[f'centers_{state}'] = centers_state_padded

    # --- DataFrame Creation and Saving ---
    try:
        # Create DataFrame from the dictionary. All columns will have length max_peaks.
        df_results = pd.DataFrame(data_dict)
        # Reorder columns alphabetically by state pair (centers_X, ns_X) for better readability
        sorted_columns = sorted(df_results.columns, key=lambda c: (c.split('_')[-1], c.split('_')[0]))
        df_results = df_results[sorted_columns]
        df_results.index.name = 'Peak_Index (0 to max_peaks-1)'
        df_results.to_csv(output_csv_path, index=True) # Save with index
        print(f"\nPeak data successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
    # --- End DataFrame Section ---


    # --- Fitting Section (using only valid data from all_ns_fit, all_centers_fit) ---
    if not all_ns_fit or not all_centers_fit:
         print("\nError: No valid peak data available across all states for fitting.")
         ax.set_ylabel("Photon Number (n)")
         ax_res.set_xlabel("Peak centers (5lagy)")
         ax_res.set_ylabel("Fit Residual (eV)")
         ax.legend(handles=plot_handles, title="States", loc='upper left')
         plt.tight_layout()
         return None, None # Indicate fitting could not be performed

    # Convert combined lists to numpy arrays for processing
    all_ns_fit = np.array(all_ns_fit)
    all_centers_fit = np.array(all_centers_fit)

    # Sort data based on center position for plotting the fit nicely
    sort_indices = np.argsort(all_centers_fit)
    sorted_centers_fit = all_centers_fit[sort_indices]
    sorted_ns_fit = all_ns_fit[sort_indices]

    # Convert sorted photon number n to Energy E for fitting
    sorted_Es_fit = ns_to_Es(sorted_ns_fit)

    # Fit the polynomial function to E vs Center
    try:
        popt, pcov = curve_fit(fitFunc4, sorted_centers_fit, sorted_Es_fit)

        # Generate fitted Energy values across the range of centers
        fit_centers_dense = np.linspace(sorted_centers_fit.min(), sorted_centers_fit.max(), 500) # Denser points for smooth curve
        fit_y_E = fitFunc4(fit_centers_dense, *popt) # Calculate fitted Energy values on dense grid

        # Calculate residuals at the original data points
        residuals_E = sorted_Es_fit - fitFunc4(sorted_centers_fit, *popt)

        # Plot the fit curve using a twin axis for Energy (E)
        ax_E_twin = ax.twinx() # Create a second y-axis for Energy
        fit_line, = ax_E_twin.plot(fit_centers_dense, fit_y_E, 'r-', linewidth=2, label='Energy Fit (E vs Center)')
        ax_E_twin.set_ylabel("Fitted Energy (eV)", color='r')
        ax_E_twin.tick_params(axis='y', labelcolor='r')
        plot_handles.append(fit_line) # Add fit line to legend handles

        # Plot residuals on the lower subplot
        ax_res.plot(sorted_centers_fit, residuals_E, 'k.', markersize=3, label='Residuals (Data - Fit)')
        ax_res.axhline(0, color='grey', linestyle='--', linewidth=1) # Add y=0 line

        # Add fit equation text
        fit_equation = f"E Fit: {popt[0]:.2e}$x^5$ + {popt[1]:.2e}$x^4$ + {popt[2]:.2e}$x^3$ + {popt[3]:.2e}$x^2$ + {popt[4]:.2e}$x$"
        ax.text(0.05, 0.95, fit_equation, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    except Exception as e:
        print(f"Error during fitting or plotting fit: {e}")
        # Still configure the twin axis even if fit fails
        ax_E_twin = ax.twinx()
        ax_E_twin.set_ylabel("Energy (eV)", color='r')
        ax_E_twin.tick_params(axis='y', labelcolor='r')

    # --- Final Plot Configuration ---
    ax.set_ylabel("Photon Number (n)")
    ax.legend(handles=plot_handles, title="Legend", loc='center left', bbox_to_anchor=(1.15, 0.5)) # Legend outside plot
    ax.grid(True, linestyle='--', alpha=0.6)

    ax_res.set_xlabel("Peak centers (5lagy)")
    ax_res.set_ylabel("Fit Residual (eV)")
    ax_res.legend(loc='upper right')
    ax_res.grid(True, linestyle='--', alpha=0.6)

    fig.subplots_adjust(right=0.80) # Adjust plot area to make space for legend outside
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Refine layout further if needed


    # Return the data used for the final fit
    return sorted_Es_fit, sorted_centers_fit


def create_publication_histogram_plot(df, binwidth):
    """
    Creates a two-panel, publication-quality histogram plot for specific states
    in an IEEE single-column format.

    Args:
        df (pd.DataFrame): The dataframe containing '5lagy' and 'state_label' columns.
        binwidth (float): The width of the histogram bins.
    """
    print("\nGenerating publication-level histogram plot for selected states...")

    # --- Filter for specific states ---
    states_to_use = ['E', 'G', 'J', 'K', 'L']
    df_filtered = df[df['state_label'].isin(states_to_use)].copy()

    if df_filtered.empty:
        print(f"Warning: No data found for the specified states: {states_to_use}")
        return

    # --- Style and Figure Setup ---
    ieee_single_col_width = 4  # inches
    plt.rcParams.update({'font.size': 9})

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(ieee_single_col_width, 5),
        dpi=200,
    )
    fig.subplots_adjust(hspace=0)

    # --- Data and Color Preparation ---
    cmap = plt.get_cmap('tab10')
    # Use the predefined list to ensure consistent ordering and color assignment
    state_colors = {state: cmap(i % cmap.N) for i, state in enumerate(states_to_use)}

    # --- Top Panel: 0 to 150 ---
    bins_top = np.arange(0, 150 + binwidth, binwidth)
    for state in states_to_use:
        color = state_colors.get(state)
        # Filter from the already-filtered dataframe
        state_data = df_filtered[(df_filtered['state_label'] == state) & (df_filtered['5lagy'] <= 150)]['5lagy']
        if not state_data.empty:
            ax_top.hist(state_data, bins=bins_top, color=color,
                        label=state, histtype='stepfilled', alpha=0.7, ec='black', lw=0.5)

    # --- Bottom Panel: 150 to 300 ---
    bins_bottom = np.arange(150, 300 + binwidth, binwidth)
    for state in states_to_use:
        color = state_colors.get(state)
        state_data = df_filtered[
            (df_filtered['state_label'] == state) & (df_filtered['5lagy'] > 150) & (df_filtered['5lagy'] <= 300)][
            '5lagy']
        if not state_data.empty:
            ax_bottom.hist(state_data, bins=bins_bottom, color=color,
                           label=state, histtype='stepfilled', alpha=0.7, ec='black', lw=0.5)

    # --- Formatting and Labels ---
    ax_top.set_xlim(0, 150)
    # ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(axis='both', linestyle=':', alpha=0.6, lw=0.5)

    ax_bottom.set_xlim(150, 300)
    ax_bottom.set_xlabel("Pulse height (arb. units)")
    ax_bottom.grid(axis='both', linestyle=':', alpha=0.6, lw=0.5)

    fig.supylabel(f"Counts / {binwidth:.2f} PHU bin", x=0.02)

    handles, labels = ax_top.get_legend_handles_labels()
    # The handles from ax_bottom might be different if some states only appear there
    handles_b, labels_b = ax_bottom.get_legend_handles_labels()
    # Combine them ensuring no duplicates
    by_label = dict(zip(labels_b, handles_b))
    by_label.update(dict(zip(labels, handles)))

    if by_label:
        fig.legend(by_label.values(), by_label.keys(), title='State',
                   bbox_to_anchor=(0.99, 0.95), loc='upper left', fontsize='small')

    # fig.tight_layout(rect=[0.05, 0, 0.85, 1])

    print("Publication histogram plot for selected states generated successfully.")
# ========================
# Main Execution Block
# ========================
if __name__ == "__main__":

    # --- Configuration ---
    # Use one of the example paths or replace with your actual path
    csv_file_path = '/home/pcuser/Runs/Cooldown_A12/20250212_0004_pulsetable.csv'
    # csv_file_path = '/home/pcuser/Runs/Cooldown_A14/20250310_0008_pulsetable.csv' # Alternate file

    bin_width = 0.05 # Example bin width, adjust as needed
    # Define where you want to save the results CSV
    output_csv_path = '/home/pcuser/Runs/Cooldown_A12/peak_analysis_results_004.csv' # Example output path

    print(f"Starting analysis for: {csv_file_path}")
    print(f"Using bin width: {bin_width}")
    print(f"Output CSV will be saved to: {output_csv_path}")

    # --- Step 1: Load data and find peaks ---
    # The _find_peaks argument is technically redundant here as the function always finds peaks now
    df, peak_positions = load_and_plot_histogram(csv_file_path, bin_width, _find_peaks=True)

    # Check if data loading and peak finding were successful
    if df is not None and peak_positions is not None:
        print("\nData loaded and initial histogram plotted.")
        # --- Step 2: Analyze nonlinearity, plot, fit, and save CSV ---
        # This function now handles plotting, fitting, and saving the CSV
        create_publication_histogram_plot(df, binwidth=0.03)
        sorted_Es, sorted_centers = make_nonlinearity_plot(peak_positions, output_csv_path)

        if sorted_Es is not None:
            print("\nNonlinearity analysis and fitting complete.")
            # Optional: Further analysis with fit results if needed
            # results_df = fit_multigauss_sherpa(sorted_Es, sorted_centers, 10) # Example if you add Sherpa back
            # print("\nFit Results DataFrame:\n", results_df)
        else:
            print("\nNonlinearity analysis could not be performed due to lack of valid peak data.")

    else:
        print("\nExiting: Failed to load data or find peaks from the input CSV.")

    # --- Step 3: Show Plots ---
    # This will display the histogram from load_and_plot_histogram
    # and the nonlinearity plot from make_nonlinearity_plot
    print("\nDisplaying plots...")
    plt.show()

    print("\nScript finished.")