import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import savgol_filter
import scipy.constants

plt.rcParams.update({'font.size': 9})

PHI0 = scipy.constants.value(u"mag. flux quantum")

def convert_ang2_to_ites(phi0_data):
    min_SI = 248e-12  # in Amps
    min_phi0_per_amp = min_SI / PHI0
    amp_per_arb = 1 / min_phi0_per_amp
    return phi0_data * amp_per_arb * 1e6  # µA

def extract_laser_channel(filename):
    match = re.search(r'ch(\d+)', filename)
    return int(match.group(1)) if match else None


def find_nth_laser_edge(times, trace, n=1):
    """
    Finds the nth primary falling edge in a trace corresponding to the laser pulse.

    Args:
        times (np.array): The time array.
        trace (np.array): The data trace.
        n (int): The edge to find (e.g., n=1 for the first, n=2 for the second).

    Returns:
        float: The time of the nth edge, or None if not found.
    """
    # Use a threshold halfway between the min and max of the trace
    threshold = np.min(trace) + 0.5 * (np.max(trace) - np.min(trace))

    # Find where the trace crosses the threshold downwards
    v = trace > threshold
    falling_edges = np.where(np.diff(v.astype(int)) == -1)[0]

    # The index for the nth edge is n-1
    edge_index = n - 1

    if len(falling_edges) > edge_index:
        # Return the time of the nth edge found
        return times[falling_edges[edge_index]]
    else:
        print(f"Warning: Could not find edge number {n}. Only {len(falling_edges)} edges found.")
        return None

def load_channel_from_file(filepath, channel_id, start_id=0, stop_id=None, filter_data=False):
    t_sample = 8e-6
    data = np.load(filepath, allow_pickle=True)['data']
    times = np.arange(data.shape[0]) * t_sample

    if stop_id is None:
        stop_id = len(times)

    idx = channel_id - 4096
    if idx < 0 or idx >= data.shape[1]:
        print(f"Channel {channel_id} not in file {filepath}")
        return None, None

    trace = data[start_id:stop_id, idx]
    if filter_data:
        trace = savgol_filter(trace, window_length=21, polyorder=3)
        times = savgol_filter(times[start_id:stop_id], window_length=21, polyorder=3)
    else:
        times = times[start_id:stop_id]

    return times, trace

def find_bias_laser_sync_point(times_dark, trace_dark, times_laser, trace_laser, threshold=0.1):
    dark_positive = trace_dark > threshold
    dark_zero_crossings = np.where(np.diff(dark_positive.astype(int)) == -1)[0]  # bias turns off

    laser_zero = trace_laser < threshold
    laser_transitions = np.where(np.diff(laser_zero.astype(int)) == -1)[0]  # laser turns on

    for i in dark_zero_crossings:
        t_dark_off = times_dark[i+1]
        # Look for laser edge after this
        laser_after = [times_laser[j+1] for j in laser_transitions if times_laser[j+1] > t_dark_off]
        if laser_after:
            return laser_after[0]

    return None

def plot_unwrapped_raw_baseline_subtracted(
    folder_path,
    channels_to_plot,
    filter_data=False,
    dpi=200,
    plot_window_ms=100
):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    if len(files) == 0:
        raise ValueError("No .npz files found in the folder.")

    laser_file_map = {}
    for f in files:
        ch = extract_laser_channel(f)
        if ch is not None:
            laser_file_map[ch] = os.path.join(folder_path, f)

    missing = [ch for ch in channels_to_plot if ch not in laser_file_map]
    if missing:
        raise ValueError(f"Missing files for channels: {missing}")

    stop_id = int(0.5 / 8e-6)
    channel_color_map = {ch: plt.cm.tab10(i % 10) for i, ch in enumerate(sorted(set(channels_to_plot)))}

    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(4, 2.2 * len(channels_to_plot)), dpi=dpi, sharex=True)
    if len(channels_to_plot) == 1:
        axes = [axes]

    for ax, laser_ch in zip(axes, channels_to_plot):
        sign = sign_change[laser_ch]
        filepath = laser_file_map[laser_ch]
        times_laser, trace_laser = load_channel_from_file(filepath, laser_ch, 0, stop_id, filter_data)
        if times_laser is None:
            continue
        t_laser_on = find_nth_laser_edge(times_laser, trace_laser, n=2)

        if t_laser_on is None:
            print(f"Warning: Could not find the second laser edge for Ch {laser_ch}. Trying the first edge instead.")
            t_laser_on = find_nth_laser_edge(times_laser, trace_laser, n=1)  # Fallback to first edge
            if t_laser_on is None:
                print(f"Fatal: Could not find any laser edge for Ch {laser_ch}. Skipping plot.")
                continue

        # Define the plot's t=0 to be 25 ms before the laser pulse
        # This aligns the plot visually just like your example figure
        t_zero = t_laser_on - 0.025  # 25 ms

        # Determine the polarity using the pre-laser bias period
        mask_pre_laser = (times_laser >= t_zero + 0.025) & (times_laser < t_zero + 0.050)
        bias_sign = np.sign(np.mean(trace_laser[mask_pre_laser])) if np.any(mask_pre_laser) else 1.0
        if bias_sign == 0:
            bias_sign = 1.0

        channel_traces = {}
        for ch in channels_to_plot:
            times, trace = load_channel_from_file(filepath, ch, 0, stop_id, filter_data)
            if times is None or trace is None:
                continue

            mask_zero = (times >= t_zero) & (times < t_zero + 0.025)
            baseline = np.median(trace[mask_zero]) if np.any(mask_zero) else 0.0
            trace_zeroed = (trace - baseline) * bias_sign
            times_aligned = times - t_zero
            channel_traces[ch] = (times_aligned, trace_zeroed, trace)

        for i, ch in enumerate(channels_to_plot):
            times_aligned, trace_zeroed, _ = channel_traces[ch]

            mask = (times_aligned >= offset_ms) & (times_aligned < (plot_window_ms) / 1000.0)
            times_ms = times_aligned[mask] * 1000
            trace_unwrapped_phi0 = np.unwrap(trace_zeroed[mask] * 2 * np.pi, discont=np.pi) / (2 * np.pi)
            trace_uA = convert_ang2_to_ites(trace_unwrapped_phi0)

            color = channel_color_map[ch]
            lw = 1.0 if ch == laser_ch else 1.0
            ax.plot(times_ms, trace_uA*sign, label=f"Ch {ch}", color=color, linewidth=lw)
            ax.set_ylim(-6,6)

            if ch == laser_ch:
                ax.text(0.02, 0.15, f"Laser on {ch}", color=color, transform=ax.transAxes,
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.5, edgecolor='none'))

                # RMS values between 25-50ms and 75-100ms
                rmspos_mask = (times_ms >= 25+5) & (times_ms < 45-5)
                rmsneg_mask = (times_ms >= 75+5) & (times_ms < 100-5)
                if np.any(rmspos_mask):
                    rmspos = np.sqrt(np.mean(trace_uA[rmspos_mask]**2))
                    ax.text(0.5, 0.3, f"RMS$_+$ = {rmspos:.3f} µA", transform=ax.transAxes, ha='center', fontsize=6, color=color)
                if np.any(rmsneg_mask):
                    rmsneg = np.sqrt(np.mean(trace_uA[rmsneg_mask]**2))
                    ax.text(0.5, 0.2, f"RMS$_-$ = {rmsneg:.3f} µA", transform=ax.transAxes, ha='center', fontsize=6, color=color)

                percent_diff = np.abs(1- np.abs(rmsneg/rmspos))*100
                ax.text(0.5, 0.1, f"$\Delta$ = {percent_diff:.3f} %", transform=ax.transAxes, ha='center', fontsize=6, color=color)

        ax.set_ylabel(r"$I_\mathrm{TES}$ ($\mu$A)")
        ax.grid(True, linestyle=':', alpha=0.5, lw=0.75)

    axes[-1].set_xlabel("Time (ms)")
    fig.subplots_adjust(top=0.96)
    handles = [plt.Line2D([0], [0], color=channel_color_map[ch], lw=2) for ch in channels_to_plot]
    labels = [f"Ch {ch}" for ch in channels_to_plot]
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=8
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# === Example usage ===
if __name__ == "__main__":
    channels_to_plot = [4098, 4099, 4105, 4109, 4114, 4120]
    sign_change = {4098:-1,4099:-1,4105:1,4109:-1,4114:1,4120:1}
    folder = "/data/20250312/data"
    offset_ms = 1e-3 #ms
    plot_unwrapped_raw_baseline_subtracted(
        folder_path=folder,
        channels_to_plot=channels_to_plot,
        filter_data=False,
        plot_window_ms=105
    )
