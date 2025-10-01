import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
from scipy.optimize import curve_fit as scipy_curve_fit

# --- Matplotlib Defaults ---
plt.rcParams['font.size'] = 9
plt.rcParams['axes.formatter.useoffset'] = False
# Set a thinner default line width for better visual appeal
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['errorbar.capsize'] = 3

# --- Pixel Group Definitions ---
pixel_groups = {
    1: {'pixels': [1, 2, 3], 'color': 'tab:red', 'label': r"250 $\mu m^2$, solid", 'alpha': 0.8},
    2: {'pixels': [4, 5, 6], 'color': 'tab:orange', 'label': r"250 $\mu m^2$, 80 $\mu$m leg", 'alpha': 0.8},
    3: {'pixels': [7, 8, 9], 'color': 'tab:green', 'label': r"250 $\mu m^2$, 15 $\mu$m leg", 'alpha': 0.8},
    4: {'pixels': [10, 11, 12], 'color': '#B22222', 'label': r"350 $\mu m^2$, solid", 'alpha': 0.8},
    5: {'pixels': [13, 14, 15], 'color': '#D2691E', 'label': r"350 $\mu m^2$, 80 $\mu$m leg", 'alpha': 0.8},
    6: {'pixels': [16, 17, 18], 'color': '#006400', 'label': r"350 $\mu m^2$, 15 $\mu$m leg", 'alpha': 0.8},
}

pixel_to_group_id_map = {px: gid for gid, g in pixel_groups.items() for px in g['pixels']}


def power_law_model(beta_data, A, n_power):
    """Power-law model definition: alpha = A * beta^n"""
    return A * beta_data ** n_power


def main():
    output_base_dir = Path("/home/pcuser/Runs/Cooldown_A18/Results/Complex_Z/")
    all_pixel_data_dfs = {}

    print(f"Scanning for CSV files in subdirectories of: {output_base_dir}")
    for item_path in output_base_dir.iterdir():
        if item_path.is_dir():
            match = re.match(r"Pixel_(\d+)", item_path.name)
            if match:
                pixel_num = int(match.group(1))
                if pixel_num not in pixel_to_group_id_map:
                    continue

                # Flexible path searching for CSV files
                csv_files = list(item_path.glob(f"*_Px{pixel_num}_FitParameters.csv")) \
                            or list(item_path.glob(f"FitParameters_Px{pixel_num}.csv")) \
                            or list(item_path.glob(f"*FitParameters.csv"))

                if csv_files:
                    try:
                        df = pd.read_csv(csv_files[0])
                        required_cols = ['R0/Rn', 'alpha', 'alpha_err', 'beta', 'beta_err']
                        if not all(col in df.columns for col in required_cols):
                            continue
                        all_pixel_data_dfs[pixel_num] = df
                        print(f"Loaded data for Pixel {pixel_num}")
                    except Exception as e:
                        print(f"Error reading CSV for Px {pixel_num}: {e}")

    if not all_pixel_data_dfs:
        print("No data loaded. Exiting.")
        return

    # --- Aggregated Plot: Alpha/Beta vs R/Rn ---
    fig_ab_vs_r, axs_ab_vs_r = plt.subplots(2, 1, figsize=(4, 6.5), dpi=200, sharex=True)
    r_bins = np.linspace(0.1, 0.9, 20)
    bin_centers = (r_bins[:-1] + r_bins[1:]) / 2

    # Plot group averages
    for group_id, group in pixel_groups.items():
        r_all, alpha_all, alpha_err_all, beta_all, beta_err_all = [], [], [], [], []

        for px in group['pixels']:
            if px not in all_pixel_data_dfs:
                continue
            df = all_pixel_data_dfs[px]
            mask = (
                    pd.notna(df['R0/Rn']) & pd.notna(df['alpha']) & pd.notna(df['beta']) &
                    pd.notna(df['alpha_err']) & pd.notna(df['beta_err']) &
                    (df['alpha'] > 1e-7) & (df['beta'] > 1e-7) &
                    (df['alpha_err'] > 0) & (df['beta_err'] > 0) &
                    (df['alpha_err'] <= 0.5 * np.abs(df['alpha'])) &
                    (df['beta_err'] <= 0.5 * np.abs(df['beta']))
            )
            df_filtered = df[mask]
            r_all.append(df_filtered['R0/Rn'].values)
            alpha_all.append(df_filtered['alpha'].values)
            alpha_err_all.append(df_filtered['alpha_err'].values)
            beta_all.append(df_filtered['beta'].values)
            beta_err_all.append(df_filtered['beta_err'].values)

        if not any(len(arr) > 0 for arr in r_all):
            continue

        r_all, alpha_all, alpha_err_all, beta_all, beta_err_all = (
            np.concatenate(r_all), np.concatenate(alpha_all), np.concatenate(alpha_err_all),
            np.concatenate(beta_all), np.concatenate(beta_err_all)
        )

        alpha_means, alpha_mean_errs, beta_means, beta_mean_errs = [], [], [], []
        bin_indices = np.digitize(r_all, r_bins)
        for i in range(1, len(r_bins)):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) >= 1:  # Original condition for groups
                alpha_vals, alpha_errors = alpha_all[bin_mask], alpha_err_all[bin_mask]
                alpha_weights = 1.0 / alpha_errors ** 2
                alpha_mean = np.sum(alpha_vals * alpha_weights) / np.sum(alpha_weights)
                alpha_mean_err = np.sqrt(1.0 / np.sum(alpha_weights))
                beta_vals, beta_errors = beta_all[bin_mask], beta_err_all[bin_mask]
                beta_weights = 1.0 / beta_errors ** 2
                beta_mean = np.sum(beta_vals * beta_weights) / np.sum(beta_weights)
                beta_mean_err = np.sqrt(1.0 / np.sum(beta_weights))
                alpha_means.append(alpha_mean);
                alpha_mean_errs.append(alpha_mean_err)
                beta_means.append(beta_mean);
                beta_mean_errs.append(beta_mean_err)
            else:
                alpha_means.append(np.nan);
                alpha_mean_errs.append(np.nan)
                beta_means.append(np.nan);
                beta_mean_errs.append(np.nan)

        # **NEW: Filter out NaN values to connect lines across gaps**
        valid_mask = ~np.isnan(alpha_means)
        if np.any(valid_mask):
            axs_ab_vs_r[0].errorbar(bin_centers[valid_mask], np.array(alpha_means)[valid_mask],
                                    yerr=np.array(alpha_mean_errs)[valid_mask],
                                    fmt='-o', markersize=2, color=group['color'],
                                    alpha=group['alpha'], label=group['label'])
            axs_ab_vs_r[1].errorbar(bin_centers[valid_mask], np.array(beta_means)[valid_mask],
                                    yerr=np.array(beta_mean_errs)[valid_mask],
                                    fmt='-o', markersize=2, color=group['color'],
                                    alpha=group['alpha'])

    # --- Highlight Pixel 6 ---
    pixel_to_highlight = 6
    if pixel_to_highlight in all_pixel_data_dfs:
        # ... (data loading and filtering for Pixel 6 is the same) ...
        df_px6 = all_pixel_data_dfs[pixel_to_highlight]
        mask_px6 = (
                pd.notna(df_px6['R0/Rn']) & pd.notna(df_px6['alpha']) & pd.notna(df_px6['beta']) &
                pd.notna(df_px6['alpha_err']) & pd.notna(df_px6['beta_err']) &
                (df_px6['alpha'] > 1e-7) & (df_px6['beta'] > 1e-7) &
                (df_px6['alpha_err'] > 0) & (df_px6['beta_err'] > 0) &
                (df_px6['alpha_err'] <= 0.5 * np.abs(df_px6['alpha'])) &
                (df_px6['beta_err'] <= 0.5 * np.abs(df_px6['beta']))
        )
        df_px6_filtered = df_px6[mask_px6]

        if not df_px6_filtered.empty:
            r_px6, alpha_px6, alpha_err_px6, beta_px6, beta_err_px6 = (
                df_px6_filtered['R0/Rn'].values, df_px6_filtered['alpha'].values,
                df_px6_filtered['alpha_err'].values, df_px6_filtered['beta'].values,
                df_px6_filtered['beta_err'].values
            )

            alpha_means_px6, alpha_mean_errs_px6, beta_means_px6, beta_mean_errs_px6 = [], [], [], []
            bin_indices_px6 = np.digitize(r_px6, r_bins)
            for i in range(1, len(r_bins)):
                bin_mask = (bin_indices_px6 == i)
                if np.sum(bin_mask) >= 1:  # Original condition for single pixel
                    alpha_vals, alpha_errors = alpha_px6[bin_mask], alpha_err_px6[bin_mask]
                    alpha_weights = 1.0 / alpha_errors ** 2
                    alpha_mean = np.sum(alpha_vals * alpha_weights) / np.sum(alpha_weights)
                    alpha_mean_err = np.sqrt(1.0 / np.sum(alpha_weights))
                    beta_vals, beta_errors = beta_px6[bin_mask], beta_err_px6[bin_mask]
                    beta_weights = 1.0 / beta_errors ** 2
                    beta_mean = np.sum(beta_vals * beta_weights) / np.sum(beta_weights)
                    beta_mean_err = np.sqrt(1.0 / np.sum(beta_weights))
                    alpha_means_px6.append(alpha_mean);
                    alpha_mean_errs_px6.append(alpha_mean_err)
                    beta_means_px6.append(beta_mean);
                    beta_mean_errs_px6.append(beta_mean_err)
                else:
                    alpha_means_px6.append(np.nan);
                    alpha_mean_errs_px6.append(np.nan)
                    beta_means_px6.append(np.nan);
                    beta_mean_errs_px6.append(np.nan)

            # **NEW: Filter out NaN values for highlighted pixel to connect lines**
            valid_mask_px6 = ~np.isnan(alpha_means_px6)
            group_id_px6 = pixel_to_group_id_map[pixel_to_highlight]
            group_color_px6 = pixel_groups[group_id_px6]['color']
            if np.any(valid_mask_px6):
                highlight_style = {'fmt': '--^', 'markersize': 2, 'color': group_color_px6, 'linewidth': 1,
                                   'label': f'Pixel {pixel_to_highlight}', 'zorder': 10}
                axs_ab_vs_r[0].errorbar(bin_centers[valid_mask_px6], np.array(alpha_means_px6)[valid_mask_px6],
                                        yerr=np.array(alpha_mean_errs_px6)[valid_mask_px6], **highlight_style)
                axs_ab_vs_r[1].errorbar(bin_centers[valid_mask_px6], np.array(beta_means_px6)[valid_mask_px6],
                                        yerr=np.array(beta_mean_errs_px6)[valid_mask_px6], **highlight_style)

    axs_ab_vs_r[0].set_ylabel(r"$\alpha_I$")
    axs_ab_vs_r[0].set_yscale('log')
    axs_ab_vs_r[0].legend(fontsize='small')
    axs_ab_vs_r[0].grid(True, which='both', linestyle=':', linewidth=0.5)

    axs_ab_vs_r[1].set_ylabel(r"$\beta_I$")
    axs_ab_vs_r[1].set_xlabel("R/R$_n$")
    axs_ab_vs_r[1].set_yscale('log')
    axs_ab_vs_r[1].grid(True, which='both', linestyle=':', linewidth=0.5)
    axs_ab_vs_r[1].set_xlim(0.1, 0.9)

    fig_ab_vs_r.tight_layout()
    plt.savefig(output_base_dir / "Alpha_Beta_vs_R_over_Rn_Averaged_Connected.png", dpi=300, bbox_inches='tight')

    # --- Aggregated Plot: Alpha vs Beta with Weighted Fits ---

    fig_alpha_beta, ax_alpha_beta = plt.subplots(figsize=(4, 4), dpi=200)

    text_y_offset = 0
    text_x_start = 0.04
    text_y_start = 0.96
    text_props = {
        'transform': ax_alpha_beta.transAxes, 'fontsize': 9,
        'verticalalignment': 'top',
        'bbox': dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.6)
    }

    for group_id, group in pixel_groups.items():
        if group_id!=1:
            alphas_raw, betas_raw, alpha_errs_raw, beta_errs_raw = [], [], [], []

            for px in group['pixels']:
                if px not in all_pixel_data_dfs:
                    continue
                df = all_pixel_data_dfs[px]
                mask = (
                        pd.notna(df['alpha']) & pd.notna(df['beta']) &
                        pd.notna(df['alpha_err']) & pd.notna(df['beta_err']) &
                        (df['alpha'] > 1e-7) & (df['beta'] > 1e-7) &
                        (df['alpha_err'] > 0) & (df['beta_err'] > 0) &
                        (df['alpha_err'] <= 0.5 * np.abs(df['alpha'])) &
                        (df['beta_err'] <= 0.5 * np.abs(df['beta']))
                )
                df_filtered = df[mask]
                alphas_raw.extend(df_filtered['alpha'].values)
                betas_raw.extend(df_filtered['beta'].values)
                alpha_errs_raw.extend(df_filtered['alpha_err'].values)
                beta_errs_raw.extend(df_filtered['beta_err'].values)

            alphas, betas, alpha_errs, beta_errs = (
                np.array(alphas_raw), np.array(betas_raw),
                np.array(alpha_errs_raw), np.array(beta_errs_raw)
            )

            if len(alphas) < 3:
                continue

            ax_alpha_beta.scatter(betas, alphas, s=15, alpha=0.3, label=group['label'], color=group['color'])

            try:
                popt_initial, _ = scipy_curve_fit(power_law_model, betas, alphas, p0=[100, 1.0])
                A_est, n_est = popt_initial
                d_alpha_d_beta = A_est * n_est * betas ** (n_est - 1)
                sigma_eff = np.sqrt(alpha_errs ** 2 + (d_alpha_d_beta * beta_errs) ** 2)
                popt, pcov = scipy_curve_fit(power_law_model, betas, alphas, p0=popt_initial,
                                             sigma=sigma_eff, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                A, n = popt;
                A_err, n_err = perr

                beta_fit = np.logspace(np.log10(min(betas) * 0.9), np.log10(max(betas) * 1.1), 100)
                alpha_fit = power_law_model(beta_fit, A, n)
                ax_alpha_beta.plot(beta_fit, alpha_fit, '-', color=group['color'], linewidth=2, alpha=0.95,
                                   label=f"{group['label'].split(',')[0]} fit")

                fit_label = (f"{group['label'].split(',')[0]}:\n"
                             f"  $\\alpha = ({A:.0f} \\pm {A_err:.0f}) \\cdot \\beta^{{{n:.2f} \\pm {n_err:.2f}}}$")
                ax_alpha_beta.text(text_x_start, text_y_start - text_y_offset, fit_label,
                                   color=group['color'], **text_props)
                text_y_offset += 0.1
                if text_y_start - text_y_offset < 0.15:
                    text_y_offset = 0;
                    text_x_start += 0.4
                    if text_x_start > 0.6: text_x_start = 0.04
            except Exception as e:
                print(f"Fit error for group {group_id}: {e}")

    ax_alpha_beta.set_xlabel(r"$\beta_I$")
    ax_alpha_beta.set_ylabel(r"$\alpha_I$")
    ax_alpha_beta.set_xscale('log')
    ax_alpha_beta.set_yscale('log')
    ax_alpha_beta.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax_alpha_beta.legend(fontsize='small', loc='lower right')
    fig_alpha_beta.tight_layout()
    plt.savefig(output_base_dir / "Alpha_vs_Beta_Averaged_Fits_WithText.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
    plt.show()