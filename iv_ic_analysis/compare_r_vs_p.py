#!/usr/bin/env python3
"""
Compare Ites vs Ibias curves between two IV files for each channel (< 32),
calculate ΔItes = Ites_file1 - Ites_file2 difference, and perform Fourier Transform (FFT)
analysis on ΔItes(Ibias) to extract the SQUID mutual inductance periodic error (M_I * Phi_0).
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


def load_and_compute_channel_data(filepath, channel_id, rbias=10e3, Rshunt=250e-6):
    """
    Load NPZ file and calculate Ibias (mA), Ites (uA), Rtes (mOhm), and Ptes (pW) for a channel.
    """
    with np.load(filepath) as data:
        vbias = data['vb']
        ang2 = data['ang2']

    ibias_A = convert_vbias_to_ibias(vbias, rbias)
    ites_A = convert_ang2_to_ites(ang2, channel_id, correct_shift=True)

    if ites_A.size == 0:
        return None, None, None, None

    ibias_mA = ibias_A * 1e3
    ites_uA = ites_A * 1e6
    rtes_mOhm = Rtes(ibias_A, ites_A, Rshunt) * 1e3
    ptes_pW = Ptes(ibias_A, ites_A, Rshunt) * 1e12

    return ibias_mA, ites_uA, rtes_mOhm, ptes_pW


def get_active_channels(filepath, threshold=0.05, max_channel=32):
    """Identify channels with significant signal variation up to max_channel."""
    with np.load(filepath) as data:
        ang2 = data['ang2']
    stds = np.std(ang2, axis=0)
    channels = np.where(stds > threshold)[0]
    if max_channel is not None:
        channels = [ch for ch in channels if ch < max_channel]
    return channels


def plot_ites_vs_ibias_comparison(file1, file2, channels=None, label1="sample rate ~ 244 kHz", label2="sample rate ~ 122 kHz",
                                  rbias=10e3, Rshunt=250e-6, rows=4, cols=4, save_path=None):
    """
    Plots Ites (uA) vs Ibias (mA) subplots comparing file1 and file2 for each channel (< 32).
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

    print(f"\n[1/3] Plotting Ites vs Ibias curves for {num_channels} channels (< 32)...")

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        page_channels = channels[page * plots_per_page : (page + 1) * plots_per_page]

        for i, ch in enumerate(page_channels):
            ax = axes[i]
            ib1, it1, _, _ = load_and_compute_channel_data(file1, ch, rbias=rbias, Rshunt=Rshunt)
            ib2, it2, _, _ = load_and_compute_channel_data(file2, ch, rbias=rbias, Rshunt=Rshunt)

            if ib1 is not None and len(ib1) > 0:
                ax.plot(ib1, it1, color='#1f77b4', lw=1.8, label=label1)

            if ib2 is not None and len(ib2) > 0:
                ax.plot(ib2, it2, color='#ff7f0e', lw=1.8, linestyle='--', label=label2)

            ax.set_title(f"Channel {ch}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Ibias (mA)", fontsize=9)
            ax.set_ylabel("Ites (μA)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax.legend(fontsize=8, loc='best')

        for j in range(len(page_channels), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Ites vs Ibias Comparison (Ch < 32): Page {page + 1}/{num_pages}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            out_file = f"{os.path.splitext(save_path)[0]}_ites_vs_ibias_page{page+1}.png"
            plt.savefig(out_file, dpi=200)
            print(f"Saved Ites vs Ibias comparison plot to: {out_file}")


def plot_ites_vs_ibias_difference(file1, file2, channels=None, label1="sample rate ~ 244 kHz", label2="sample rate ~ 122 kHz",
                                  rbias=10e3, Rshunt=250e-6, rows=4, cols=4, save_path=None):
    """
    Plots ΔItes = Ites_file1 - Ites_file2 (uA) vs Ibias (mA) subplots for each channel (< 32).
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

    print(f"\n[2/3] Plotting Difference Delta Ites = Ites({label1}) - Ites({label2}) for {num_channels} channels (< 32)...")

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        page_channels = channels[page * plots_per_page : (page + 1) * plots_per_page]

        for i, ch in enumerate(page_channels):
            ax = axes[i]
            ib1, it1, _, _ = load_and_compute_channel_data(file1, ch, rbias=rbias, Rshunt=Rshunt)
            ib2, it2, _, _ = load_and_compute_channel_data(file2, ch, rbias=rbias, Rshunt=Rshunt)

            if (ib1 is not None and len(ib1) > 0) and (ib2 is not None and len(ib2) > 0):
                m1 = np.isfinite(ib1) & np.isfinite(it1)
                m2 = np.isfinite(ib2) & np.isfinite(it2)
                ib1_v, it1_v = ib1[m1], it1[m1]
                ib2_v, it2_v = ib2[m2], it2[m2]

                ib_min = max(np.min(ib1_v), np.min(ib2_v))
                ib_max = min(np.max(ib1_v), np.max(ib2_v))

                if ib_max > ib_min:
                    ib_grid = np.linspace(ib_min, ib_max, 500)
                    it1_interp = interp1d(ib1_v, it1_v, kind='linear', bounds_error=False)(ib_grid)
                    it2_interp = interp1d(ib2_v, it2_v, kind='linear', bounds_error=False)(ib_grid)

                    delta_it = it1_interp - it2_interp
                    valid = np.isfinite(delta_it)

                    ax.plot(ib_grid[valid], delta_it[valid], color='#d62728', lw=1.6, label='ΔItes (File1 - File2)')
                    ax.axhline(0, color='black', linestyle=':', lw=1)

            ax.set_title(f"Channel {ch} ΔItes", fontsize=11, fontweight='bold')
            ax.set_xlabel("Ibias (mA)", fontsize=9)
            ax.set_ylabel("ΔItes (μA)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)

        for j in range(len(page_channels), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Ites Difference ΔItes (Ites_{label1[:12]} - Ites_{label2[:12]}): Page {page + 1}/{num_pages}", fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            out_file = f"{os.path.splitext(save_path)[0]}_ites_diff_page{page+1}.png"
            plt.savefig(out_file, dpi=200)
            print(f"Saved Delta Ites plot to: {out_file}")


def plot_ites_vs_ibias_fft(file1, file2, channels=None, rbias=10e3, Rshunt=250e-6, rows=4, cols=4, save_path=None):
    """
    Computes and plots Fourier Transform (FFT) spectra of ΔItes(Ibias) to measure SQUID mutual inductance periodicity (M_I * Phi_0).
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

    print(f"\n[3/3] Computing Fourier Transform (FFT) spectra of Delta Ites(Ibias) for {num_channels} channels (< 32)...")

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        page_channels = channels[page * plots_per_page : (page + 1) * plots_per_page]

        for i, ch in enumerate(page_channels):
            ax = axes[i]
            ib1, it1, _, _ = load_and_compute_channel_data(file1, ch, rbias=rbias, Rshunt=Rshunt)
            ib2, it2, _, _ = load_and_compute_channel_data(file2, ch, rbias=rbias, Rshunt=Rshunt)

            if (ib1 is not None and len(ib1) > 0) and (ib2 is not None and len(ib2) > 0):
                m1 = np.isfinite(ib1) & np.isfinite(it1)
                m2 = np.isfinite(ib2) & np.isfinite(it2)
                ib1_v, it1_v = ib1[m1], it1[m1]
                ib2_v, it2_v = ib2[m2], it2[m2]

                ib_min = max(np.min(ib1_v), np.min(ib2_v))
                ib_max = min(np.max(ib1_v), np.max(ib2_v))

                if ib_max > ib_min:
                    n_samples = 512
                    ib_grid = np.linspace(ib_min, ib_max, n_samples)
                    dib = ib_grid[1] - ib_grid[0]

                    it1_interp = interp1d(ib1_v, it1_v, kind='linear', bounds_error=False)(ib_grid)
                    it2_interp = interp1d(ib2_v, it2_v, kind='linear', bounds_error=False)(ib_grid)

                    delta_it = it1_interp - it2_interp
                    valid = np.isfinite(delta_it)

                    if np.sum(valid) > 32:
                        d_sub = delta_it[valid] - np.mean(delta_it[valid])

                        window = np.hanning(len(d_sub))
                        d_win = d_sub * window

                        fft_mag = np.abs(np.fft.rfft(d_win))
                        freqs = np.fft.rfftfreq(len(d_sub), d=dib)  # in 1/mA

                        ax.plot(freqs, fft_mag, color='#9467bd', lw=1.6, label='FFT Amplitude')

                        if len(fft_mag) > 2:
                            peak_idx = np.argmax(fft_mag[1:]) + 1
                            if freqs[peak_idx] > 0:
                                peak_freq = freqs[peak_idx]
                                peak_period_mA = 1.0 / peak_freq
                                peak_period_uA = peak_period_mA * 1000
                                ax.annotate(
                                    f"T={peak_period_mA*1e3:.1f}μA",
                                    xy=(peak_freq, fft_mag[peak_idx]),
                                    xytext=(peak_freq * 1.05, fft_mag[peak_idx] * 0.85),
                                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                                    fontsize=8,
                                    fontweight='bold',
                                    color='#2ca02c'
                                )

            ax.set_title(f"Channel {ch} SQUID Wiggle FFT", fontsize=10, fontweight='bold')
            ax.set_xlabel("Spatial Freq (1/mA)", fontsize=9)
            ax.set_ylabel("FFT Mag", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)

        for j in range(len(page_channels), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"FFT Spectrum of ΔItes(Ibias) (SQUID M_I*Phi_0 Periodicity): Page {page + 1}/{num_pages}", fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            out_file = f"{os.path.splitext(save_path)[0]}_ites_fft_page{page+1}.png"
            plt.savefig(out_file, dpi=200)
            print(f"Saved FFT spectrum plot to: {out_file}")


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Compare Ites vs Ibias curves, calculate ΔItes = Ites1 - Ites2, and analyze SQUID M_I*Phi_0 wiggle periods via FFT for channels < 32."
    )
    def_f1, def_f2 = find_default_files()

    parser.add_argument("--file1", type=str, default=def_f1, help="Path to first IV .npz file")
    parser.add_argument("--file2", type=str, default=def_f2, help="Path to second IV .npz file")
    parser.add_argument("--label1", type=str, default="sample rate ~ 244 kHz", help="Legend label for File 1")
    parser.add_argument("--label2", type=str, default="sample rate ~ 122 kHz", help="Legend label for File 2")
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

    print(f"Comparing Ites vs Ibias between:\n  File 1 ({args.label1}): {args.file1}\n  File 2 ({args.label2}): {args.file2}")

    # Determine channels < 32
    if args.channels is None:
        ch1 = get_active_channels(args.file1, max_channel=args.max_channel)
        ch2 = get_active_channels(args.file2, max_channel=args.max_channel)
        channels = sorted(list(set(ch1).union(set(ch2))))
    else:
        channels = [ch for ch in args.channels if ch < args.max_channel]

    # 1. Ites vs Ibias comparison plot
    plot_ites_vs_ibias_comparison(
        args.file1,
        args.file2,
        channels=channels,
        label1=args.label1,
        label2=args.label2,
        rbias=args.rbias,
        Rshunt=args.rshunt,
        rows=args.rows,
        cols=args.cols,
        save_path=args.save
    )

    # 2. Ites1 - Ites2 difference plot
    plot_ites_vs_ibias_difference(
        args.file1,
        args.file2,
        channels=channels,
        label1=args.label1,
        label2=args.label2,
        rbias=args.rbias,
        Rshunt=args.rshunt,
        rows=args.rows,
        cols=args.cols,
        save_path=args.save
    )

    # 3. FFT spectrum of SQUID wiggles plot
    plot_ites_vs_ibias_fft(
        args.file1,
        args.file2,
        channels=channels,
        rbias=args.rbias,
        Rshunt=args.rshunt,
        rows=args.rows,
        cols=args.cols,
        save_path=args.save
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
