#!/usr/bin/env python3
"""
Compare R (Resistance) vs P (Power) curves between two IV files for each channel,
calculate R_file1 - R_file2 difference, and perform Fourier Transform (FFT) analysis
to identify period and harmonics of ripples/wiggles.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from iv_ic_analysis.iv_reader import (
    convert_vbias_to_ibias,
    convert_ang2_to_ites,
    Rtes,
    Ptes
)

DEFAULT_DATA_DIR = r"C:\Users\anr29\Desktop\Data\ravendata-dtest62\iv"


def find_default_files(search_dir=DEFAULT_DATA_DIR):
    """Locate the two default IV npz files in the data directory."""
    if not os.path.exists(search_dir):
        alt_dirs = [
            os.path.expanduser("~/Desktop/Data/ravendata-dtest62/iv"),
            os.path.expanduser("~/Desktop/data/iv"),
            os.path.expanduser("~/Desktop/Data/iv"),
            os.path.join(repo_root, "data")
        ]
        for alt in alt_dirs:
            if os.path.exists(alt):
                search_dir = alt
                break

    if os.path.exists(search_dir):
        npz_files = sorted(glob.glob(os.path.join(search_dir, "*.npz")))
        if len(npz_files) >= 2:
            return npz_files[0], npz_files[1]
        elif len(npz_files) == 1:
            return npz_files[0], None
    return None, None


def load_and_compute_channel(filepath, channel_id, rbias=10e3, Rshunt=250e-6):
    """Load NPZ file and calculate Rtes (mOhm) and Ptes (pW) for a channel."""
    with np.load(filepath) as data:
        vbias = data['vb']
        ang2 = data['ang2']

    ibias = convert_vbias_to_ibias(vbias, rbias)
    ites = convert_ang2_to_ites(ang2, channel_id, correct_shift=True)

    if ites.size == 0:
        return None, None, None

    rtes_mOhm = Rtes(ibias, ites, Rshunt) * 1e3
    ptes_pW = Ptes(ibias, ites, Rshunt) * 1e12

    return ptes_pW, rtes_mOhm, ites


def get_active_channels(filepath, threshold=0.05, max_channel=32):
    """Identify channels with significant signal variation up to max_channel."""
    with np.load(filepath) as data:
        ang2 = data['ang2']
    stds = np.std(ang2, axis=0)
    channels = np.where(stds > threshold)[0]
    if max_channel is not None:
        channels = [ch for ch in channels if ch < max_channel]
    return channels


def plot_r_vs_p_comparison(file1, file2, channels=None, label1="sample rate = 244 kHz", label2="rate = 122 kHz",
                           rbias=10e3, Rshunt=250e-6, rows=4, cols=4, save_path=None, show_plot=True):
    """
    Plots R vs P subplots comparing file1 and file2 for each channel (up to channel 31).
    """
    if channels is None:
        ch1 = get_active_channels(file1, max_channel=32)
        ch2 = get_active_channels(file2, max_channel=32)
        channels = sorted(list(set(ch1).union(set(ch2))))
    else:
        channels = [ch for ch in channels if ch < 32]

    if not channels:
        print("No channels < 32 found to plot.")
        return

    num_channels = len(channels)
    plots_per_page = rows * cols
    num_pages = int(np.ceil(num_channels / plots_per_page))

    print(f"\n[1/3] Plotting R vs P curves for {num_channels} channels (< 32)...")

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        page_channels = channels[page * plots_per_page : (page + 1) * plots_per_page]

        for i, ch in enumerate(page_channels):
            ax = axes[i]
            p1, r1, _ = load_and_compute_channel(file1, ch, rbias=rbias, Rshunt=Rshunt)
            p2, r2, _ = load_and_compute_channel(file2, ch, rbias=rbias, Rshunt=Rshunt)

            if p1 is not None and len(p1) > 0:
                ax.plot(p1, r1, color='#1f77b4', lw=1.8, label=label1)

            if p2 is not None and len(p2) > 0:
                ax.plot(p2, r2, color='#ff7f0e', lw=1.8, linestyle='--', label=label2)

            ax.set_title(f"Channel {ch}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Ptes (pW)", fontsize=9)
            ax.set_ylabel("Rtes (mΩ)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax.legend(fontsize=8, loc='best')

        for j in range(len(page_channels), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"R vs P Comparison (Ch < 32): Page {page + 1}/{num_pages}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            out_file = f"{os.path.splitext(save_path)[0]}_rvsp_page{page+1}.png"
            plt.savefig(out_file, dpi=200)
            print(f"Saved R vs P comparison plot to: {out_file}")

    if show_plot:
        plt.show()


def plot_r_vs_p_difference(file1, file2, channels=None, label1="sample rate = 244 kHz", label2="rate = 122 kHz",
                           rbias=10e3, Rshunt=250e-6, rows=4, cols=4, save_path=None, show_plot=True):
    """
    Plots R_file1 - R_file2 vs P subplots for each channel (< 32).
    """
    if channels is None:
        ch1 = get_active_channels(file1, max_channel=32)
        ch2 = get_active_channels(file2, max_channel=32)
        channels = sorted(list(set(ch1).union(set(ch2))))
    else:
        channels = [ch for ch in channels if ch < 32]

    if not channels:
        print("No channels < 32 found to plot difference.")
        return

    num_channels = len(channels)
    plots_per_page = rows * cols
    num_pages = int(np.ceil(num_channels / plots_per_page))

    print(f"\n[2/3] Plotting Difference Delta R = R({label1}) - R({label2}) for {num_channels} channels (< 32)...")

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        page_channels = channels[page * plots_per_page : (page + 1) * plots_per_page]

        for i, ch in enumerate(page_channels):
            ax = axes[i]
            p1, r1, _ = load_and_compute_channel(file1, ch, rbias=rbias, Rshunt=Rshunt)
            p2, r2, _ = load_and_compute_channel(file2, ch, rbias=rbias, Rshunt=Rshunt)

            if (p1 is not None and len(p1) > 0) and (p2 is not None and len(p2) > 0):
                m1 = np.isfinite(p1) & np.isfinite(r1)
                m2 = np.isfinite(p2) & np.isfinite(r2)
                p1_v, r1_v = p1[m1], r1[m1]
                p2_v, r2_v = p2[m2], r2[m2]

                p_min = max(np.min(p1_v), np.min(p2_v))
                p_max = min(np.max(p1_v), np.max(p2_v))

                if p_max > p_min:
                    p_grid = np.linspace(p_min, p_max, 400)
                    r1_interp = interp1d(p1_v, r1_v, kind='linear', bounds_error=False)(p_grid)
                    r2_interp = interp1d(p2_v, r2_v, kind='linear', bounds_error=False)(p_grid)

                    delta_r = r1_interp - r2_interp
                    valid = np.isfinite(delta_r)

                    ax.plot(p_grid[valid], delta_r[valid], color='#d62728', lw=1.6, label='ΔR (File1 - File2)')
                    ax.axhline(0, color='black', linestyle=':', lw=1)

            ax.set_title(f"Channel {ch} ΔR", fontsize=11, fontweight='bold')
            ax.set_xlabel("Ptes (pW)", fontsize=9)
            ax.set_ylabel("ΔR (mΩ)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)

        for j in range(len(page_channels), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Resistance Difference ΔR (R_{label1[:12]} - R_{label2[:12]}): Page {page + 1}/{num_pages}", fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            out_file = f"{os.path.splitext(save_path)[0]}_diff_page{page+1}.png"
            plt.savefig(out_file, dpi=200)
            print(f"Saved Delta R plot to: {out_file}")

    if show_plot:
        plt.show()


def plot_r_vs_p_fft(file1, file2, channels=None, rbias=10e3, Rshunt=250e-6, rows=4, cols=4, save_path=None, show_plot=True):
    """
    Computes and plots Fourier Transform (FFT) power spectra of ΔR(P) to identify periods and harmonics of wiggles.
    """
    if channels is None:
        ch1 = get_active_channels(file1, max_channel=32)
        ch2 = get_active_channels(file2, max_channel=32)
        channels = sorted(list(set(ch1).union(set(ch2))))
    else:
        channels = [ch for ch in channels if ch < 32]

    if not channels:
        print("No channels < 32 found to analyze FFT.")
        return

    num_channels = len(channels)
    plots_per_page = rows * cols
    num_pages = int(np.ceil(num_channels / plots_per_page))

    print(f"\n[3/3] Computing Fourier Transform (FFT) spectra of Delta R(P) for {num_channels} channels (< 32)...")

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        page_channels = channels[page * plots_per_page : (page + 1) * plots_per_page]

        for i, ch in enumerate(page_channels):
            ax = axes[i]
            p1, r1, _ = load_and_compute_channel(file1, ch, rbias=rbias, Rshunt=Rshunt)
            p2, r2, _ = load_and_compute_channel(file2, ch, rbias=rbias, Rshunt=Rshunt)

            if (p1 is not None and len(p1) > 0) and (p2 is not None and len(p2) > 0):
                m1 = np.isfinite(p1) & np.isfinite(r1)
                m2 = np.isfinite(p2) & np.isfinite(r2)
                p1_v, r1_v = p1[m1], r1[m1]
                p2_v, r2_v = p2[m2], r2[m2]

                p_min = max(np.min(p1_v), np.min(p2_v))
                p_max = min(np.max(p1_v), np.max(p2_v))

                if p_max > p_min:
                    n_samples = 512
                    p_grid = np.linspace(p_min, p_max, n_samples)
                    dp = p_grid[1] - p_grid[0]

                    r1_interp = interp1d(p1_v, r1_v, kind='linear', bounds_error=False)(p_grid)
                    r2_interp = interp1d(p2_v, r2_v, kind='linear', bounds_error=False)(p_grid)

                    delta_r = r1_interp - r2_interp
                    valid = np.isfinite(delta_r)

                    if np.sum(valid) > 32:
                        p_sub = p_grid[valid]
                        d_sub = delta_r[valid] - np.mean(delta_r[valid])

                        window = np.hanning(len(d_sub))
                        d_win = d_sub * window

                        fft_mag = np.abs(np.fft.rfft(d_win))
                        freqs = np.fft.rfftfreq(len(d_sub), d=dp)

                        ax.plot(freqs, fft_mag, color='#9467bd', lw=1.6, label='FFT Amplitude')

                        if len(fft_mag) > 2:
                            peak_idx = np.argmax(fft_mag[1:]) + 1
                            if freqs[peak_idx] > 0:
                                peak_freq = freqs[peak_idx]
                                peak_period = 1.0 / peak_freq
                                ax.annotate(
                                    f"Period={peak_period:.2f} pW",
                                    xy=(peak_freq, fft_mag[peak_idx]),
                                    xytext=(peak_freq * 1.1, fft_mag[peak_idx] * 0.9),
                                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                                    fontsize=8,
                                    fontweight='bold',
                                    color='#2ca02c'
                                )

            ax.set_title(f"Channel {ch} FFT Spectrum", fontsize=11, fontweight='bold')
            ax.set_xlabel("Spatial Freq (1/pW)", fontsize=9)
            ax.set_ylabel("FFT Mag", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)

        for j in range(len(page_channels), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"FFT Spectrum of ΔR(P) Wiggles & Harmonics: Page {page + 1}/{num_pages}", fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            out_file = f"{os.path.splitext(save_path)[0]}_fft_page{page+1}.png"
            plt.savefig(out_file, dpi=200)
            print(f"Saved FFT spectrum plot to: {out_file}")

    if show_plot:
        plt.show()


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Compare R vs P, calculate ΔR = R1 - R2, and analyze wiggle periods/harmonics via FFT for channels < 32."
    )
    def_f1, def_f2 = find_default_files()

    parser.add_argument("--file1", type=str, default=def_f1, help="Path to first IV .npz file")
    parser.add_argument("--file2", type=str, default=def_f2, help="Path to second IV .npz file")
    parser.add_argument("--label1", type=str, default="sample rate = 244 kHz", help="Legend label for File 1")
    parser.add_argument("--label2", type=str, default="rate = 122 kHz", help="Legend label for File 2")
    parser.add_argument("--max-channel", type=int, default=32, help="Upper channel limit (default: 32)")
    parser.add_argument("--channels", type=int, nargs="+", default=None, help="Channel IDs to plot (< 32)")
    parser.add_argument("--rbias", type=float, default=10e3, help="Bias resistance in Ohms (default: 10000)")
    parser.add_argument("--rshunt", type=float, default=250e-6, help="Shunt resistance in Ohms (default: 250e-6)")
    parser.add_argument("--rows", type=int, default=4, help="Grid rows per page (default: 4)")
    parser.add_argument("--cols", type=int, default=4, help="Grid cols per page (default: 4)")
    parser.add_argument("--save", type=str, default=None, help="Optional base output PNG filepath")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive plot window")

    args = parser.parse_args(args_list)

    if not args.file1 or not os.path.exists(args.file1):
        print(f"Error: File 1 not found: '{args.file1}'")
        sys.exit(1)
    if not args.file2 or not os.path.exists(args.file2):
        print(f"Error: File 2 not found: '{args.file2}'")
        sys.exit(1)

    print(f"Comparing R vs P between:\n  File 1 ({args.label1}): {args.file1}\n  File 2 ({args.label2}): {args.file2}")

    # Determine channels < 32
    if args.channels is None:
        ch1 = get_active_channels(args.file1, max_channel=args.max_channel)
        ch2 = get_active_channels(args.file2, max_channel=args.max_channel)
        channels = sorted(list(set(ch1).union(set(ch2))))
    else:
        channels = [ch for ch in args.channels if ch < args.max_channel]

    # 1. R vs P comparison plot
    plot_r_vs_p_comparison(
        args.file1,
        args.file2,
        channels=channels,
        label1=args.label1,
        label2=args.label2,
        rbias=args.rbias,
        Rshunt=args.rshunt,
        rows=args.rows,
        cols=args.cols,
        save_path=args.save,
        show_plot=not args.no_show
    )

    # 2. R1 - R2 difference plot
    plot_r_vs_p_difference(
        args.file1,
        args.file2,
        channels=channels,
        label1=args.label1,
        label2=args.label2,
        rbias=args.rbias,
        Rshunt=args.rshunt,
        rows=args.rows,
        cols=args.cols,
        save_path=args.save,
        show_plot=not args.no_show
    )

    # 3. FFT spectrum of wiggles plot
    plot_r_vs_p_fft(
        args.file1,
        args.file2,
        channels=channels,
        rbias=args.rbias,
        Rshunt=args.rshunt,
        rows=args.rows,
        cols=args.cols,
        save_path=args.save,
        show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()
