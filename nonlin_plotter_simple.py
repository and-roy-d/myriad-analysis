import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammaln

plt.rcParams['font.size'] = 14

Eph = 1239 / 520  # eV


def rough_crosstalk_correction():

    pass


def load_and_combine_data(path, states_to_skip=None, fit_polynomial=False):
    """
    Loads data from CSV files, combines it, and optionally fits a polynomial.

    Args:
        path (str): The path to the directory containing the CSV files.
        states_to_skip (list, optional): A list of states to skip. Defaults to None.
        fit_polynomial (bool, optional): Whether to fit a 5th order polynomial to the data. Defaults to False.

    Returns:
        tuple: A tuple containing the combined DataFrame and the polynomial fit parameters (if fit_polynomial is True).
    """

    if states_to_skip is None:
        states_to_skip = []

    # Find all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(path, 'fit_results_state*.csv'))

    # Combine data from all CSVs
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    df_combined = pd.DataFrame()
    corrs = {'E':0, 'G':0.14, 'J':0.34, 'L':0.84, 'K':1.34, 'M':1.75}
    for csv_file in csv_files:
        state = csv_file[-5]  # Extract the state letter from the filename
        if state in states_to_skip:
            continue

        df = pd.read_csv(csv_file)

        ax.errorbar(df.n[1::],
                    np.diff(df.mu.to_numpy()),
                    yerr=np.sqrt(df.muStd[0:-2]**2 + df.muStd[1:-1]**2),
                    fmt='o',
                    label=state)
        ax2.errorbar(df.n, df.mu, yerr=df.muStd, fmt='.', label=state)
        df['state'] = state  # Add a 'state' column
        df['mu'] = df['mu'].to_numpy() - corrs[state]
        df['Es'] = df['Es'].to_numpy() #+ corrs[state]
        df_combined = pd.concat([df_combined, df], ignore_index=True)


    ax.set(xlabel='n',
           ylabel=r'$\Delta$$\mu$ (5lagy)',
           title='Successive differences in peak separation')
    ax2.set(xlabel='n', ylabel='Centroid position (5lagy)')
    ax.grid()
    ax2.grid()
    ax.legend()
    ax2.legend()
    if fit_polynomial:
        # Fit a 5th order polynomial to mu vs E
        popt, _ = curve_fit(poly_func, df_combined['Es'], df_combined['mu'],
                            sigma=df_combined['muStd'])
        return df_combined, popt
    else:
        return df_combined, None


def poly_func(x, a, b, c, d, e, f, g, h, i, j, k):
    return  a*x**11 + b*x**10 + c*x**9 + d*x**8 + e*x**7 + f*x**6 + g*x**5 + h*x**4 + i*x**3 + j*x**2 + k*x


def poly_func_derivative(x, a, b, c, d, e, f, g, h, i, j, k):
    """Calculates the derivative of poly_func with respect to x."""
    term11 = 11 * a * x**10
    term10 = 10 * b * x**9
    term9  = 9  * c * x**8
    term8  = 8  * d * x**7
    term7  = 7  * e * x**6
    term6  = 6  * f * x**5
    term5  = 5  * g * x**4
    term4  = 4  * h * x**3
    term3  = 3  * i * x**2
    term2  = 2  * j * x**1
    term1  = k
    return (term11 + term10 + term9 + term8 + term7 + term6 +
            term5 + term4 + term3 + term2 + term1)


def create_plots(df_combined, poly_params=None):
    """
    Creates the mu vs E and FWHM vs E subplots.

    Args:
        df_combined (pd.DataFrame): The combined DataFrame.
        poly_params (list, optional): The polynomial fit parameters. Defaults to None.
    """
    if True:
        fig, (ax_0,ax_1,ax_2) = plt.subplots(3,
                                 1,
                                 figsize=(12, 12),
                                 sharex=True,
                                 gridspec_kw={'height_ratios':[4,1,2]})

    else:
        fig, axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.5]})
        fig.delaxes(axes[0, 1])
        fig.delaxes(axes[1, 1])
        axes_right = fig.add_subplot(1, 2, 2)
        axes[1, 0].get_shared_x_axes().joined(axes[1, 0], axes[0, 0])
        ax_0 = axes[0, 0]
        ax_1 = axes[1, 0]
        ax_2 = axes_right

    fig3, ax3 = plt.subplots()

    # Mu with error bars vs Es
    for state in df_combined['state'].unique():
        df_state = df_combined[df_combined['state'] == state]
        ax_0.errorbar(df_state['Es'],
                         df_state['mu'],
                         yerr=df_state['muStd'],
                         fmt='o',
                         label=state)

    ax_0.set_ylabel('Fit centroid (5lagy)')
    ax_0.grid(True)

    if poly_params is not None:
        # Generate points for the polynomial fit
        print(poly_params)
        x_fit = np.linspace(df_combined['Es'].min(), df_combined['Es'].max(),
                           len(df_combined['Es']))
        y_fit = np.polyval(poly_params, x_fit)
        ax_0.plot(x_fit,
                     poly_func(x_fit, *poly_params),
                     'k--', zorder= 6,
                     label=r'9$^\mathrm{th}$ order poly. fit')
        ax_0.legend()
        idx = df_combined["Es"].argsort()
        popt_PH_to_E, _ = curve_fit(poly_func, df_combined['mu'][idx], df_combined['Es'][idx], sigma=df_combined['muStd'][idx])
        popt_E_to_PH = poly_params
        for state in df_combined['state'].unique():

            df_state = df_combined[df_combined['state'] == state]
            ax_1.errorbar(df_state['Es'], df_state['Es'] - poly_func(df_state['mu'], *popt_PH_to_E,),
                             yerr=poly_func(df_state['muStd'], *popt_PH_to_E), fmt='o', label=state)
            # ax_1.errorbar(df_state['Es'], (df_state['mu'] - poly_func(df_state['Es'],  *poly_params)),
            #                  yerr=df_state['muStd'], fmt='o', label=state)

            ax_2.errorbar(
                df_state['Es'],
                poly_func_derivative(df_state['mu'], *popt_PH_to_E) * df_state['fwhm'],  # Multiply y by the derivative
                yerr=poly_func_derivative(df_state['mu'], *popt_PH_to_E) * df_state['fwhmStd'],
                # Multiply yerr by the derivative
                fmt='o',
                label=state
            )
            ax3.errorbar(df_state['mu'], df_state['Es'], xerr=df_state['muStd'], fmt='o', label=state)
        # ax_2.errorbar(df_combined['Es'], poly_func(df_combined['mu'], *popt_PH_to_E)*df_combined['fwhm']/df_combined['mu'],
        #               yerr=poly_func(df_combined['mu'], *popt_PH_to_E)*df_combined['fwhmStd']/df_combined['mu'],  # FWHM plot
        #               fmt='o', label='FWHM: E')
        # ax_2.errorbar(df_combined['Es'][idx], df_combined['fwhm'][idx],
        #               yerr=df_combined['fwhmStd'][idx],  # FWHM plot
        #               fmt='o', label='FWHM: PH')
        # ax_2.legend(loc='best')
        ax3.plot(np.sort(df_combined['mu'].to_numpy()), poly_func(np.sort(df_combined['mu'].to_numpy()), *popt_PH_to_E), 'k--', lw=0.8, zorder=10)

    ax_1.set(ylabel=r'Resid. (eV)', xlim=(0, 900))
    ax_1.grid(True)
    # axes[1].legend()



    ax3.legend()
    ax3.set(xlabel='PH (5lagy)', ylabel = 'Energy (eV)')
    ax3.grid(which='both', ls= ':', lw=0.5)
    ax_2.set(xlabel = 'Energy (eV)', ylabel = 'FWHM (eV)', ylim = (0.3,1.1),  xlim=(-0.5, 500))
    ax_2.set_ylabel('FWHM (eV)')
    # axes[2].legend()
    ax_2.grid(True)

    plt.tight_layout()
    return popt_PH_to_E, popt_E_to_PH
    # plt.subplots_adjust(hspace=0.05, wspace=0.01)

if __name__ == '__main__':
    data_path = '/home/pcuser/Runs/Cooldown_A12/'
    states_to_skip = ['M']
    fit_polynomial = True

    df_combined, poly_params = load_and_combine_data(data_path, states_to_skip,
                                                    fit_polynomial)
    PH_to_E, E_to_PH = create_plots(df_combined, poly_params)
    plt.show()