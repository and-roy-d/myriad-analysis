try:
    import umux_tune
except ImportError:
    umux_tune = None
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import configparser
import ast
from scipy.spatial.distance import cdist
import argparse

# Attempt to import qsdumux for the unpickler to use
try:
    import qsdumux
except ImportError:
    print("WARNING: Could not import 'qsdumux'. Loading older data that uses "
          "classes from 'qsghw' will fail unless 'qsdumux' is installed.")

# --- Custom Unpickler for Remapping ---
class LegacyRemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Catch exact matches or submodules starting with qsghw
        if module.startswith('qsghw'):
            remapped_module = 'qsdumux' + module[len('qsghw'):]
            print(f"Pickle Remap: Remapping '{module}.{name}' -> '{remapped_module}.{name}'")
            module = remapped_module
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError as e:
            print(f"Pickle Error: Module '{module}' not found while loading class '{name}'.")
            raise e
        except AttributeError as e:
            print(f"Pickle Error: Module '{module}' found, but class '{name}' is missing.")
            raise e

# --- Configuration and Setup ---
plt.rcParams.update({"font.size":12})
fset_global = "fset1"

def get_data_paths(datetime_str, current_fset, base_path="/data"):
    date_part = datetime_str.split("_")[0]
    data_dir = pathlib.Path(base_path) / date_part / "autotune" / datetime_str
    data_filename = data_dir / f"data_fineddc_all_tones_{current_fset}_{datetime_str}.pickle"
    cfg_filename = data_dir / f"tones_{datetime_str}"
    analysis_output_dir = data_dir / "arc_analysis_summary"
    return data_filename, cfg_filename, analysis_output_dir

def load_config(cfg_filename, current_fset):
    print(f"Loading config from: {cfg_filename} for fset: {current_fset}")
    cp = configparser.ConfigParser()
    read_ok = cp.read(cfg_filename)
    if not read_ok:
        raise FileNotFoundError(f"Configuration file not found or empty: {cfg_filename}")
    try:
        config_params = {
            'flux_ramp_points': int(cp["Global"]["flux_ramp_points"].split()[0]),
            'freq_lo_hz': float(cp[current_fset]["lo_freq_hz"].split()[0]),
            'tones_freq_mhz': ast.literal_eval(cp[current_fset]["tones_up_freq_mhz"]),
            'nominal_power_dbm': float(cp["Global"]["tone_power_dbm"].split()[0])
        }
    except KeyError as e:
        raise KeyError(f"Missing key {e} in config file {cfg_filename} for fset '{current_fset}'.") from e
    print("Config loaded successfully.")
    return config_params

def load_channel_data(pickle_filename):
    print(f"Loading data from: {pickle_filename}")
    with open(pickle_filename, "rb") as f:
        unpickler = LegacyRemappingUnpickler(f)
        data_cent_packet_list = unpickler.load()
    print(f"Data loaded: {len(data_cent_packet_list)} channels found.")
    return data_cent_packet_list

def calculate_end_idx(packet_list, flux_ramp_points):
    if not packet_list: return 0
    try:
        packet = packet_list[0]
        end_idx = flux_ramp_points * int(np.floor(packet.data['i'].shape[0] / flux_ramp_points))
        return end_idx
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        print(f"Error calculating end_idx from first packet: {e}")
        return 0

def analyze_single_channel(ch, raw_i, raw_q, config, end_idx):
    results = {'channel': ch, 'raw_i': raw_i, 'raw_q': raw_q, 'status': 'Failed'}
    flux_ramp_points = config['flux_ramp_points']
    freq_lo_hz = config['freq_lo_hz']
    tones_freq_mhz = config['tones_freq_mhz']
    nominal_power_dbm = config['nominal_power_dbm']

    try:
        if raw_i.ndim > 1: raw_i = raw_i[:,0]
        if raw_q.ndim > 1: raw_q = raw_q[:,0]

        n_ramps = len(raw_i[:end_idx]) // flux_ramp_points
        if n_ramps == 0:
            results['status'] = 'Avg Failed: Insufficient data'
            return results
        actual_end_idx = n_ramps * flux_ramp_points
        arc_i = np.mean(raw_i[:actual_end_idx].reshape((n_ramps, flux_ramp_points)), axis=0)
        arc_q = np.mean(raw_q[:actual_end_idx].reshape((n_ramps, flux_ramp_points)), axis=0)
        results['avg_i'], results['avg_q'] = arc_i, arc_q

        aas = umux_tune.ArcAquisitionSettings(
            lo_freq_hz=freq_lo_hz,
            freq_offset_hz=tones_freq_mhz[ch]*1e6 - freq_lo_hz,
            nominal_power_dbm=nominal_power_dbm
        )
        arc_data = umux_tune.ArcData(I=arc_i, Q=arc_q, aquisition_settings=aas)
        arc_summary = arc_data.analyze()
        results['arc_summary'] = arc_summary

        results['f0_fit'] = tones_freq_mhz[ch]*1e6
        results['radius'] = getattr(arc_summary, 'radius', np.nan)
        results['theta_span'] = getattr(arc_summary, 'theta_span_rad', np.nan)
        results['theta_center'] = getattr(arc_summary, 'theta_center_rad', np.nan)
        results['I0'] = getattr(arc_summary, 'I0', np.nan)
        results['Q0'] = getattr(arc_summary, 'Q0', np.nan)
        
        try:
            results['r_theta_text'] = fr"$r, \theta_s = {results['radius']:.2f}, {results['theta_span']:.2f}$"
        except TypeError: results['r_theta_text'] = r"$r, \theta_s$ N/A"
        try:
            results['off_resonance_distance_text'] = fr"$r+\sqrt{{I_0^2+Q_0^2}}$ = {(np.hypot(results['I0'], results['Q0'])+results['radius']):.2f}"
        except TypeError: results['off_resonance_distance_text'] = r"$r+\sqrt{I_0^2+Q_0^2}$ N/A"

        avg_i_fit = np.asarray(getattr(arc_summary.arc_data, 'I', np.array([])))
        avg_q_fit = np.asarray(getattr(arc_summary.arc_data, 'Q', np.array([])))
        results['avg_i_fit'], results['avg_q_fit'] = avg_i_fit, avg_q_fit

        raw_i_np, raw_q_np = np.asarray(raw_i), np.asarray(raw_q)
        if raw_i_np.ndim == 1 and avg_i_fit.ndim == 1 and avg_i_fit.size > 0 and raw_i_np.size > 0:
            raw_points = np.stack((raw_i_np[:end_idx], raw_q_np[:end_idx]), axis=-1)
            avg_points = np.stack((avg_i_fit, avg_q_fit), axis=-1)
            dists_sq = cdist(raw_points, avg_points, 'sqeuclidean')
            closest_avg_indices = np.argmin(dists_sq, axis=1)
            closest_avg_points = avg_points[closest_avg_indices]
            residual_vectors = raw_points - closest_avg_points
            results['residual_i'] = residual_vectors[:, 0]
            results['residual_q'] = residual_vectors[:, 1]
            results['rms_residual'] = np.sqrt(np.mean(results['residual_i']**2 + results['residual_q']**2))
        else:
            results['residual_i'], results['residual_q'] = np.array([]), np.array([])
            results['rms_residual'] = np.nan
            results['status'] = 'Resid Calc Failed'

        if results.get('status') == 'Failed':
            results['status'] = 'Success'

    except Exception as e:
        results['status'] = f'Analysis Error: {e}'
    
    for key in ['avg_i', 'avg_q', 'avg_i_fit', 'avg_q_fit', 'residual_i', 'residual_q']: results.setdefault(key, np.array([]))
    for key in ['rms_residual', 'f0_fit', 'radius', 'theta_span', 'theta_center', 'I0', 'Q0']: results.setdefault(key, np.nan)
    return results

def analyze_all_channels(data_cent_packet_list, config, end_idx):
    num_tones_in_config = len(config.get('tones_freq_mhz', []))
    num_data_packets = len(data_cent_packet_list)
    num_channels_to_process = min(num_data_packets, num_tones_in_config)

    all_results = []
    print(f"\nAnalyzing up to {num_channels_to_process} channels...")
    for ch in range(num_channels_to_process):
        packet_object = data_cent_packet_list[ch]
        
        # Simple, direct parsing block as requested
        raw_i = packet_object.data['i']
        raw_q = packet_object.data['q']

        channel_results = analyze_single_channel(ch, raw_i, raw_q, config, end_idx)
        all_results.append(channel_results)
    
    successful_count = sum(1 for r in all_results if r.get('status') == 'Success')
    print(f"Analysis complete. Processed: {successful_count} of {len(all_results)} channels successfully.")
    return all_results

def plot_multi_channel_summary(analysis_results, config_params, plot_title_prefix, rows=4, cols=8):
    num_grid_slots = rows * cols
    tones_freq_mhz = config_params.get('tones_freq_mhz', [])
    num_analyzed_channels = len(analysis_results)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), sharex= True, sharey=True)
    fig.suptitle(f"{plot_title_prefix}: Multi-Channel Arc Summary", fontsize=16)
    axs_flat = axs.flatten()

    print("\nGenerating multi-channel summary plot...")
    for i in range(num_grid_slots):
        ax = axs_flat[i]
        if i < num_analyzed_channels:
            result = analysis_results[i]
            ch_num = result.get('channel', i)
            status = result.get('status', 'Unknown Status')
            ax.set_title(f"Ch {ch_num}", fontsize=10)

            if status == 'Success':
                raw_i, raw_q = result.get('raw_i', np.array([])), result.get('raw_q', np.array([]))
                if raw_i.ndim > 1: raw_i = raw_i[:,0]
                if raw_q.ndim > 1: raw_q = raw_q[:,0]
                avg_i_fit, avg_q_fit = result.get('avg_i_fit', np.array([])), result.get('avg_q_fit', np.array([]))
                residual_i, residual_q = result.get('residual_i', np.array([])), result.get('residual_q', np.array([]))

                ax.scatter(raw_i, raw_q, s=5, alpha=0.3)
                if avg_i_fit.size > 0: ax.plot(avg_i_fit, avg_q_fit, color='red', linewidth=1.5)
                if residual_i.size > 0: ax.scatter(residual_i, residual_q, s=5, alpha=0.3, color='cornflowerblue', marker='.')

                rms_residual = result.get('rms_residual', np.nan)
                r_theta_text = result.get('r_theta_text', 'N/A')
                r_plus_r0_text = result.get('off_resonance_distance_text', 'N/A')
                current_tone_freq = tones_freq_mhz[ch_num] if ch_num < len(tones_freq_mhz) else np.nan
                bbox_props = dict(boxstyle='round,pad=0.15', fc='white', alpha=0.65)
                text_info = f"{r_theta_text}\nRMS Res: {rms_residual:.3f}\nf0: {current_tone_freq:.4f} MHz\n{r_plus_r0_text}"
                ax.text(0.03, 0.97, text_info, transform=ax.transAxes, fontsize=9, ha='left', va='top', bbox=bbox_props)
                ax.grid(which='both', ls='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, f"Ch {ch_num}\nFailed:\n{status}", ha='center', va='center', color='red', fontsize=8, wrap=True)
                ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(axis='both', which='major', labelsize=9)
        else:
            ax.set_visible(False)

    for r_idx in range(rows):
        for c_idx in range(cols):
            ax_current = axs[r_idx, c_idx]
            if not ax_current.get_visible(): continue
            if r_idx < rows - 1: ax_current.tick_params(labelbottom=False)
            if c_idx > 0: ax_current.tick_params(labelleft=False)

    fig.supxlabel("I (arb. units)", fontsize=12); fig.supylabel("Q (arb. units)", fontsize=12)
    try:
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.93, wspace=0.1, hspace=0.2)
    except ValueError: fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    return fig

def plot_parameter_summary(analysis_results, plot_title_prefix):
    print("\nExtracting parameters for summary plot...")
    all_f0_fit, all_r_param, all_theta_span, all_R_param = [], [], [], []
    for result in analysis_results:
        if result and result.get('status') == 'Success':
            f0, r_val, ts, i0, q0 = result.get('f0_fit'), result.get('radius'), result.get('theta_span'), result.get('I0'), result.get('Q0')
            if all(v is not None and not np.isnan(v) for v in [f0, r_val, ts, i0, q0]):
                all_f0_fit.append(f0); all_r_param.append(r_val); all_theta_span.append(ts)
                all_R_param.append(np.hypot(i0, q0) + r_val)
    if not all_f0_fit:
        print("Error: No valid parameters extracted for summary plot.")
        return None

    all_f0_fit, all_r_param, all_theta_span, all_R_param = np.array(all_f0_fit), np.array(all_r_param), np.array(all_theta_span), np.array(all_R_param)
    fig_frt, (ax_r, ax_R) = plt.subplots(2,1, figsize=(12, 8), sharex=True)
    color_r, color_theta = 'tab:red', 'tab:green'
    x_data = all_f0_fit / 1e6
    x_label = 'Fitted $f_0$ (MHz)'

    ax_r.set_ylabel(r"Radius $r_0$ (arb.)", color=color_r, fontsize=12)
    lns1 = ax_r.scatter(x_data, all_r_param, label='Radius', color=color_r, marker='o', s=30)
    ax_r.tick_params(axis='y', labelcolor=color_r, labelsize=12)
    ax_r.grid(True, axis='both', linestyle=':', alpha=0.6)

    ax_theta = ax_r.twinx()
    ax_theta.set_ylabel(r"$\theta_{span}$ (rad)", color=color_theta, fontsize=12)
    lns2 = ax_theta.scatter(x_data, all_theta_span, label='Theta Span', color=color_theta, marker='x', s=30)
    ax_theta.tick_params(axis='y', labelcolor=color_theta, labelsize=12)

    ax_R.set_xlabel(x_label, fontsize=12)
    ax_R.set_ylabel(r"$r+\sqrt{I_0^2+Q_0^2}$ (arb.)", fontsize=12)
    ax_R.scatter(x_data, all_R_param, label = r'$r+\sqrt{I_0^2+Q_0^2}$', color = 'k', marker = 'o', s= 30)
    ax_R.grid(True, axis='both', linestyle=':', alpha=0.6)
    ax_R.tick_params(axis='x', labelsize=12); ax_R.tick_params(axis='y', labelsize=12)

    ax_r.legend([lns1, lns2], [l.get_label() for l in [lns1, lns2]], loc='upper left', fontsize=10)
    ax_R.legend(loc='upper left', fontsize=10)
    fig_frt.suptitle(f"{plot_title_prefix}: Radius & Theta Span vs $f_0$", fontsize=14)
    try:
        fig_frt.tight_layout(rect=[0, 0.02, 1, 0.96])
    except ValueError: fig_frt.subplots_adjust(left=0.1, right=0.88, bottom=0.1, top=0.92, hspace=0.25)
    return fig_frt

def save_analysis_summary_npz(analysis_results, config_params, output_filepath):
    num_tones_in_config = len(config_params.get('tones_freq_mhz', []))
    if num_tones_in_config == 0: return

    channel_indices = np.arange(num_tones_in_config)
    f0_design_hz = np.array([f * 1e6 for f in config_params.get('tones_freq_mhz', [])])
    
    radii = np.full(num_tones_in_config, np.nan)
    theta_spans = np.full(num_tones_in_config, np.nan)
    off_resonance_distances = np.full(num_tones_in_config, np.nan)
    rms_residuals = np.full(num_tones_in_config, np.nan)
    f0_fit_hz = np.full(num_tones_in_config, np.nan)

    for result in analysis_results:
        if result and 'channel' in result:
            ch = result['channel']
            if 0 <= ch < num_tones_in_config and result.get('status') == 'Success':
                radii[ch] = result.get('radius', np.nan)
                theta_spans[ch] = result.get('theta_span', np.nan)
                i0, q0, r_val = result.get('I0', np.nan), result.get('Q0', np.nan), result.get('radius', np.nan)
                if not any(np.isnan(v) for v in [i0, q0, r_val]):
                     off_resonance_distances[ch] = np.hypot(i0, q0) + r_val
                rms_residuals[ch] = result.get('rms_residual', np.nan)
                f0_fit_hz[ch] = result.get('f0_fit', np.nan)
    try:
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_filepath, channel=channel_indices, f0_design_hz=f0_design_hz, f0_fit_hz=f0_fit_hz,
                 radius=radii, theta_span_rad=theta_spans, off_resonance_distance=off_resonance_distances,
                 rms_residual=rms_residuals)
        print(f"Saved NPZ summary to: {output_filepath}")
    except Exception as e: print(f"Error saving NPZ: {e}")

def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Analyze uMUX resonator arc data, generate and save plots and summary NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--folder", type=str, required=True, help="Datetime string for the data folder.")
    parser.add_argument("--fset", type=str, default="fset1", help="Frequency set identifier.")
    parser.add_argument("--basepath", type=str, default="/data", help="Base path for data folders.")
    parser.add_argument("--rows", type=int, default=4, help="Rows in multi-channel summary plot.")
    parser.add_argument("--cols", type=int, default=8, help="Columns in multi-channel summary plot.")
    args = parser.parse_args(args_list)

    datetime_str, current_fset, base_data_path = args.folder, args.fset, args.basepath
    plot_rows, plot_cols, plot_format = args.rows, args.cols, "png"
    plt.close("all")

    try:
        print(f"{'='*69}\nStarting analysis for folder: {datetime_str}, fset: {current_fset}\n{'='*69}")
        data_filename, cfg_filename, analysis_output_dir = get_data_paths(datetime_str, current_fset, base_path=base_data_path)
        analysis_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Analysis outputs will be saved to: {analysis_output_dir}")

        config_params = load_config(cfg_filename, current_fset)
        data_cent_packet_list = load_channel_data(data_filename)
        end_idx = calculate_end_idx(data_cent_packet_list, config_params['flux_ramp_points'])

        analysis_results = analyze_all_channels(data_cent_packet_list, config_params, end_idx)
        plot_title_prefix = f"{datetime_str} ({current_fset})"

        if analysis_results:
            npz_filename = f"analysis_summary_fset{current_fset}.npz"
            save_analysis_summary_npz(analysis_results, config_params, analysis_output_dir / npz_filename)

            print("\nPlotting results...")
            fig1 = plot_multi_channel_summary(analysis_results, config_params, plot_title_prefix, rows=plot_rows, cols=plot_cols)
            if fig1:
                fig1.savefig(analysis_output_dir / f"multi_channel_summary_fset{current_fset}.{plot_format}")
                print(f"Saved multi-channel summary plot.")
                plt.close(fig1)
            
            fig2 = plot_parameter_summary(analysis_results, plot_title_prefix)
            if fig2:
                fig2.savefig(analysis_output_dir / f"parameter_summary_fset{current_fset}.{plot_format}")
                print(f"Saved parameter summary plot.")
                plt.close(fig2)
        else:
            print("No analysis results generated.")
        print("\nScript finished successfully.")

    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"{'='*69}")

if __name__ == "__main__":
    main()