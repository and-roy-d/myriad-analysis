import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import erf
plt.rcParams['font.size'] = 14

base_path = '/home/pcuser/Runs/FilesforJonno/'
item = '5lagy'
csv_file_path = base_path + f'median_{item}_vs_y_coord_linescan_20250313.csv'
position_column_name = 'y_coord'
signal_column_name = f'median_{item}'


gaussian_sigma = 9e-3

# --- Load Data ---
try:
    data = pd.read_csv(csv_file_path)
    if position_column_name not in data.columns or signal_column_name not in data.columns:
        raise ValueError(
            f"Error: Columns '{position_column_name}' or '{signal_column_name}' not found in {csv_file_path}")

    data = data.sort_values(by=position_column_name)
    x_data = data[position_column_name].values
    y_data = np.abs(data[signal_column_name].values)
    print(f"Successfully loaded data from {csv_file_path}")
    print(f"Found {len(x_data)} data points.")
    # fig,ax = plt.subplots()
    # ax.plot(x_data, y_data, 'o-')
    # ax.set(xlabel=position_column_name, ylabel=signal_column_name, title = 'Raw data')
    # ax.grid()
    # plt.show()


except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    exit()
except ValueError as ve:
    print(ve)
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the CSV: {e}")
    exit()


def calculate_beamspot_fwhm_um(L, sigma, L_um = 250):
    sigma_um = L_um / L * sigma
    fwhm_um = sigma_um * np.sqrt(8*np.log(2))

    return fwhm_um


def centered_convolved_square_gaussian(x, A, L, sigma, x0):
    """

    Args:
        x (array-like): Position values.
        A (float): Total area of the original square (Area = Height * Width).
        L (float): ***Full width*** of the square function.
        sigma (float): Standard deviation of the Gaussian kernel.
        x0 (float): Center position of the square function.

    Returns:
        array-like: The calculated signal values for the given x and parameters.
    """


    x_shifted = x - x0  # Center the function relative to data

    term1 = erf((x_shifted + L/2) / (np.sqrt(2) * sigma))
    term2 = erf((x_shifted - L/2) / (np.sqrt(2) * sigma))


    amplitude_norm_factor = A / L

    return amplitude_norm_factor * (term1 - term2) * 1/2



dx = np.mean(np.diff(x_data)) if len(x_data) > 1 else 1.0
guess_A = np.sum(y_data) * dx

guess_x0 = np.median(x_data)

guess_L = 0.04

guess_sigma = 0.009

L_um = 250.0 + 48 # plus 48 from the y scan which includes TES + link


initial_guesses = [guess_A, guess_L, guess_sigma, guess_x0]


param_bounds = ([0, 0, 0, 0], [1, 1, 1, 1])

print(f"\n--- Starting Fit ---")
print(f"Initial Guesses (A, L, sigma, x0): [{guess_A:.3g}, {guess_L:.3g}, {guess_sigma:.3g}, {guess_x0:.3g}]")

try:
    # Perform the curve fit
    popt, pcov = curve_fit(
        centered_convolved_square_gaussian,  # Use the new centered function
        x_data,
        y_data,
        p0=initial_guesses,
        bounds=param_bounds,
        maxfev=5000
    )

    # Extract optimized parameters
    A_fit, L_fit, sigma_fit, x0_fit = popt

    # Calculate standard errors
    perr = np.sqrt(np.diag(pcov))
    A_err, L_err, sigma_err, x0_err = perr
    y_fit = centered_convolved_square_gaussian(x_data, *popt)
    residuals = y_data - y_fit



    # --- Results ---
    print("\n--- Fit Results ---")
    print(f"Fitted A (Area/Norm):              {A_fit:.4f} +/- {A_err:.4f}")
    print(f"Fitted L (Square Full Width):      {L_fit:.4f} +/- {L_err:.4f}")
    print(f"Fitted sigma (Gaussian Std Dev):   {sigma_fit:.4f} +/- {sigma_err:.4f}")
    print(f"Fitted x0 (Center Position):       {x0_fit:.4f} +/- {x0_err:.4f}")

    # Calculate derived Gaussian FWHM
    gaussian_fwhm = calculate_beamspot_fwhm_um(L_fit, sigma_fit, L_um)
    gaussian_fwhm_err = gaussian_fwhm * np.sqrt((L_err/L_fit)**2+(sigma_err/sigma_fit)**2)
    gain  = L_um/L_fit
    gain_err = L_um*L_err/L_fit**2

    print("\n--- Derived Gaussian Width ---")
    print(f"Gaussian FWHM:                     {gaussian_fwhm:.2f} +/- {gaussian_fwhm_err:.2f}")

    print("\n--- Derived Gain ---")
    print(f"Micron/MEMS unit:                     {gain:.2f} +/- {gain_err:.2f}")

    # --- Plot Results ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_data, y_data, 'o', label='Data', markersize=6, color='grey', alpha=0.7)
    # Plot fitted curve using a denser x-grid for smoothness
    x_fit_smooth = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit_smooth = centered_convolved_square_gaussian(x_fit_smooth, A_fit, L_fit, sigma_fit, x0_fit)
    ax.plot(x_fit_smooth, y_fit_smooth,
             label=f'Fit', color='red',
             linewidth=2)

    derived_text = ""


    if gaussian_fwhm is not None:
        derived_text += f"Gaussian FWHM = {gaussian_fwhm:.1f} $\pm$ {gaussian_fwhm_err:.1f}" +r" $\mu$m"
    else:
        derived_text += "Gaussian FWHM: calculation failed"
    if not np.isnan(gain):
        derived_text += f"\n         Gain scale = {round(gain)} $\pm$ {round(gain_err)}" + r" $\mu$m/MEMS unit"  # Using 'unit' generically
    else:
        derived_text += "Scale: calculation failed\n"

    ax.text(0.05, 0.95, derived_text,  # x, y position in axes fraction coords
            transform=ax.transAxes,  # Use axes fraction coordinates
            fontsize=12,
            verticalalignment='top',  # Anchor text box at its top edge
            horizontalalignment='left',  # Anchor text block at its left edge
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    ax.set_title('Median response during line scan (Y-axis)')
    ax.set_xlabel(position_column_name +" (MEMS units)")
    ax.set_ylabel(signal_column_name + " (arb. units)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

except RuntimeError as e:
    print(f"\nError: Curve fitting failed. Reason: {e}")
    print("Try adjusting the initial guesses (p0) or check data range and quality.")
    print(f"Current guesses (A, L, sigma, x0): {initial_guesses}")
except ValueError as e:
    print(f"\nError during fitting setup: {e}")
    print("Check bounds or initial guesses.")
except Exception as e:
    print(f"\nAn unexpected error occurred during fitting: {e}")