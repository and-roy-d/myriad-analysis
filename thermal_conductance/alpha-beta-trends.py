import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re  # For parsing pixel number
from scipy.optimize import curve_fit as scipy_curve_fit

# --- Matplotlib Defaults ---
plt.rcParams['font.size'] = 14
plt.rcParams['axes.formatter.useoffset'] = False  # Prevent scientific notation offset for axes

# 1. Define Pixel Groups and Styles (Refined for Aesthetics)
pixel_groups = {
    # Primary Groups - Distinct Colors, consistent marker style (e.g., circles)
    1: {'pixels': [1, 2, 3], 'color': 'tab:red', 'marker': 'o', 'markersize': 6,
        'label': r"250 $\mu m^2$, solid", 'alpha': 0.75},
    2: {'pixels': [4, 5, 6], 'color': 'tab:orange', 'marker': 'o', 'markersize': 6,
        'label': r"250 $\mu m^2$, 80 $\mu$m leg", 'alpha': 0.75},
    3: {'pixels': [7, 8, 9], 'color': 'tab:green', 'marker': 'o', 'markersize': 6,
        'label': r"250 $\mu m^2$, 15 $\mu$m leg", 'alpha': 0.75},

    # Secondary Groups - Darker shades of primary, different marker (e.g., squares)
    4: {'pixels': [10, 11, 12], 'color': '#B22222', 'marker': 's', 'markersize': 6,  # Firebrick (darker red)
        'label': r"350 $\mu m^2$, solid", 'alpha': 0.75},
    5: {'pixels': [13, 14, 15], 'color': '#D2691E', 'marker': 's', 'markersize': 6,  # Chocolate (darker orange)
        'label': r"350 $\mu m^2$, 80 $\mu$m leg", 'alpha': 0.75},
    6: {'pixels': [16, 17, 18], 'color': '#006400', 'marker': 's', 'markersize': 6,  # Darkgreen
        'label': r"350 $\mu m^2$, 15 $\mu$m leg", 'alpha': 0.75},

    # # Special Groups - Base color of G1, distinct markers for differentiation
    # 7: {'pixels': [19, 20], 'color': 'tab:red', 'marker': '^', 'markersize': 7,  # Triangle
    #     'label': r"G7: Px 19-20", 'alpha': 0.85},
    # 8: {'pixels': [21, 22], 'color': 'tab:red', 'marker': 'D', 'markersize': 6,  # Diamond
    #     'markeredgewidth': 1.5, 'label': r"G8: Px 21-22 (Bold)", 'alpha': 0.85},
    # 9: {'pixels': [23, 24], 'color': 'tab:red', 'marker': 'P', 'markersize': 7,  # Plus (filled)
    #     'label': r"G9: Px 23-24", 'alpha': 0.85}
}

# Invert for easy lookup: pixel_number -> group_id
pixel_to_group_id_map = {}
for group_id, group_info in pixel_groups.items():
    for px_num in group_info['pixels']:
        pixel_to_group_id_map[px_num] = group_id


# --- Power Law Model for Alpha vs Beta Fit ---
def power_law_model(beta_data, A, n_power):
    return A * beta_data ** n_power


def main():
    output_base_dir = Path("/home/pcuser/Runs/Cooldown_A18/Results/Complex_Z/")
    all_pixel_data_dfs = {}  # Store DFs by pixel_number for easier access

    print(f"Scanning for CSV files in subdirectories of: {output_base_dir}")

    for item_path in output_base_dir.iterdir():
        if item_path.is_dir():
            match = re.match(r"Pixel_(\d+)", item_path.name)
            if match:
                pixel_num = int(match.group(1))
                if pixel_num not in pixel_to_group_id_map:
                    continue

                csv_files = list(item_path.glob(f"*_Px{pixel_num}_FitParameters.csv"))
                if not csv_files:
                    csv_files = list(item_path.glob(f"FitParameters_Px{pixel_num}.csv"))
                    if not csv_files:
                        csv_files = list(item_path.glob(f"*FitParameters.csv"))

                if csv_files:
                    csv_path = csv_files[0]
                    try:
                        df = pd.read_csv(csv_path)
                        required_cols = ['R0/Rn', 'alpha', 'alpha_err', 'beta', 'beta_err']
                        if not all(col in df.columns for col in required_cols):
                            print(f"Warning: CSV {csv_path} for Px {pixel_num} missing required columns. Skipping.")
                            continue
                        all_pixel_data_dfs[pixel_num] = df
                        print(f"Loaded data for Pixel {pixel_num}")
                    except Exception as e:
                        print(f"Error reading/processing CSV {csv_path} for Px {pixel_num}: {e}")
                else:
                    print(f"Warning: No FitParameters CSV in {item_path} for Px {pixel_num}")

    if not all_pixel_data_dfs:
        print("No data loaded. Exiting.")
        return

    # --- Plot 1: Alpha and Beta vs R/Rn ---
    fig_ab_vs_r, axs_ab_vs_r = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    plotted_group_labels_r = set()
    sorted_group_ids = sorted(pixel_groups.keys())

    for group_id in sorted_group_ids:
        style = pixel_groups[group_id]

        for pixel_num in style['pixels']:
            if pixel_num in all_pixel_data_dfs:
                df = all_pixel_data_dfs[pixel_num]

                r_over_rn = df['R0/Rn'].values
                alpha_vals = df['alpha'].values
                alpha_err_vals = df['alpha_err'].fillna(0).values
                beta_vals = df['beta'].values
                beta_err_vals = df['beta_err'].fillna(0).values

                alpha_err_vals = np.array(
                    [0 if e is None or not np.isfinite(e) or e < 0 else e for e in alpha_err_vals])
                beta_err_vals = np.array([0 if e is None or not np.isfinite(e) or e < 0 else e for e in beta_err_vals])

                alpha_mask = (alpha_err_vals <= 0.5 * np.abs(alpha_vals)) & (alpha_vals > 1e-9) & np.isfinite(
                    alpha_vals) & np.isfinite(r_over_rn) & np.isfinite(alpha_err_vals)
                beta_mask = (beta_err_vals <= 0.5 * np.abs(beta_vals)) & (beta_vals > 1e-9) & np.isfinite(
                    beta_vals) & np.isfinite(r_over_rn) & np.isfinite(beta_err_vals)

                current_label = style['label'] if group_id not in plotted_group_labels_r else None

                if np.any(alpha_mask):
                    axs_ab_vs_r[0].errorbar(r_over_rn[alpha_mask], alpha_vals[alpha_mask],
                                            yerr=alpha_err_vals[alpha_mask],
                                            fmt=style['marker'], color=style['color'], markersize=style['markersize'],
                                            markeredgewidth=style.get('markeredgewidth', 1.0),
                                            capsize=3, label=current_label, alpha=style['alpha'], elinewidth=1,
                                            ecolor='gray')

                if np.any(beta_mask):
                    axs_ab_vs_r[1].errorbar(r_over_rn[beta_mask], beta_vals[beta_mask], yerr=beta_err_vals[beta_mask],
                                            fmt=style['marker'], color=style['color'], markersize=style['markersize'],
                                            markeredgewidth=style.get('markeredgewidth', 1.0),
                                            capsize=3, label=None, alpha=style['alpha'], elinewidth=1, ecolor='gray')

                if current_label:
                    plotted_group_labels_r.add(group_id)

    axs_ab_vs_r[0].set_ylabel(r"$\alpha_I$")
    axs_ab_vs_r[0].set_yscale('log')
    axs_ab_vs_r[0].grid(True, which="both", linestyle=':')
    axs_ab_vs_r[0].legend( fontsize='small', loc='best')

    axs_ab_vs_r[1].set_ylabel(r"$\beta_I$")
    axs_ab_vs_r[1].set_yscale('log')
    axs_ab_vs_r[1].set_xlabel(r"R/R$_n$")
    axs_ab_vs_r[1].grid(True, which="both", linestyle=':')
    axs_ab_vs_r[1].set_xlim(0.1,0.9)

    # fig_ab_vs_r.suptitle("Fitted Alpha and Beta vs. R/R$_n$ by Pixel Group", fontsize=16)
    fig_ab_vs_r.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_save_path_ab_r = output_base_dir / "Alpha_Beta_vs_R0Rn_Grouped.png"
    # try:
    #     fig_ab_vs_r.savefig(plot_save_path_ab_r)
    #     print(f"Plot saved to {plot_save_path_ab_r}")
    # except Exception as e:
    #     print(f"Error saving Alpha/Beta vs R0Rn plot: {e}")
    # plt.close(fig_ab_vs_r)

    # --- Plot 2: Alpha vs Beta (loglog) with Grouped Fits ---
    fig_alpha_beta_grouped, ax_alpha_beta = plt.subplots(figsize=(10, 8))
    text_y_offset = 0
    text_x_start = 0.03
    text_y_start = 0.97
    text_props = {'transform': ax_alpha_beta.transAxes, 'fontsize': 9,
                  'verticalalignment': 'top', 'bbox': dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)}

    for group_id in sorted_group_ids[1::5]:
        style = pixel_groups[group_id]

        group_alphas, group_betas, group_alpha_errors, group_beta_errors = [], [], [], []

        for pixel_num in style['pixels']:
            if pixel_num in all_pixel_data_dfs:
                df = all_pixel_data_dfs[pixel_num]
                alpha_vals, alpha_err_vals = df['alpha'].values, df['alpha_err'].fillna(0).values
                beta_vals, beta_err_vals = df['beta'].values, df['beta_err'].fillna(0).values

                alpha_err_vals = np.array(
                    [0 if e is None or not np.isfinite(e) or e < 0 else e for e in alpha_err_vals])
                beta_err_vals = np.array([0 if e is None or not np.isfinite(e) or e < 0 else e for e in beta_err_vals])

                mask = (alpha_err_vals <= 0.5 * np.abs(alpha_vals)) & \
                       (beta_err_vals <= 0.5 * np.abs(beta_vals)) & \
                       (alpha_vals > 1e-9) & (beta_vals > 1e-9) & \
                       np.isfinite(alpha_vals) & np.isfinite(beta_vals) & \
                       np.isfinite(alpha_err_vals) & np.isfinite(beta_err_vals)

                group_alphas.extend(alpha_vals[mask])
                group_betas.extend(beta_vals[mask])
                group_alpha_errors.extend(alpha_err_vals[mask])
                group_beta_errors.extend(beta_err_vals[mask])

        group_alphas, group_betas = np.array(group_alphas), np.array(group_betas)
        group_alpha_errors, group_beta_errors = np.array(group_alpha_errors), np.array(group_beta_errors)

        if len(group_alphas) > 1 and len(group_betas) > 1:
            ax_alpha_beta.errorbar(group_betas, group_alphas,
                                   yerr=group_alpha_errors, xerr=group_beta_errors,
                                   fmt=style['marker'], color=style['color'],
                                   markersize=style['markersize'], alpha=style['alpha'],
                                   markeredgewidth=style.get('markeredgewidth', 1.0),
                                   capsize=3, elinewidth=1, ecolor='gray', label=style['label'])

            sigma_for_group_fit = group_alpha_errors.copy()
            sigma_for_group_fit[sigma_for_group_fit <= 1e-9] = 1.0

            try:
                popt_g, pcov_g = scipy_curve_fit(power_law_model, group_betas, group_alphas,
                                                 p0=[np.nanmax(group_alphas) if len(group_alphas) > 0 else 100, 1.0],
                                                 sigma=sigma_for_group_fit, absolute_sigma=True,
                                                 maxfev=5000)
                A_g, n_g = popt_g

                if len(group_betas) > 0 and min(group_betas) > 0:
                    beta_line_g = np.logspace(np.log10(min(group_betas) * 0.8), np.log10(max(group_betas) * 1.2), 50)
                    alpha_line_g = power_law_model(beta_line_g, A_g, n_g)
                    ax_alpha_beta.plot(beta_line_g, alpha_line_g, '-', color=style['color'], lw=2,
                                       alpha=0.8)  # Thicker fit line

                    fit_text_g = f'{style["label"].split(":")[0]}: α={round(A_g)}β$^{{{n_g:.2f}}}$'  # Shortened label for text
                    current_text_props = text_props.copy()
                    current_text_props['bbox']['ec'] = style['color']
                    ax_alpha_beta.text(text_x_start, text_y_start - text_y_offset, fit_text_g,
                                       color=style['color'], **current_text_props)
                    text_y_offset += 0.06
                    if text_y_start - text_y_offset < 0.05:
                        text_y_offset = 0
                        text_x_start += 0.35
                        if text_x_start > 0.7: text_x_start = 0.03
            except Exception as e:
                print(f"Error during power law fit/plot for Group {group_id}: {e}")
        else:
            print(f"Not enough valid data points to fit/plot for Group {group_id}.")

    ax_alpha_beta.set_xlabel(r"$\beta_I$")
    ax_alpha_beta.set_ylabel(r"$\alpha_I$")
    ax_alpha_beta.set_xscale('log')
    ax_alpha_beta.set_yscale('log')
    # ax_alpha_beta.set_xlim(0,0.75)
    # ax_alpha_beta.set_title(r"$\alpha$ vs. $\beta$ by Pixel Group (with Individual Fits)")
    ax_alpha_beta.legend( fontsize='small', loc='lower right', ncol=2)
    ax_alpha_beta.grid(True, which="both", linestyle=':')
    fig_alpha_beta_grouped.tight_layout(rect=[0, 0, 1, 0.96])

    plot_save_path_ab_grouped = output_base_dir / "Alpha_vs_Beta_Grouped_Fits.png"
    # try:
    #     fig_alpha_beta_grouped.savefig(plot_save_path_ab_grouped)
    #     print(f"Plot saved to {plot_save_path_ab_grouped}")
    # except Exception as e:
    #     print(f"Error saving Alpha vs Beta grouped plot: {e}")
    # plt.close(fig_alpha_beta_grouped)

    print("\n--- Plotting Script Complete ---")


if __name__ == '__main__':
    main()
    plt.show()
