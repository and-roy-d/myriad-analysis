#!/usr/bin/env python3
"""
Compare R (Resistance) vs P (Power) curves between two IV files for each channel.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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


def get_active_channels(filepath, threshold=0.05):
    """Identify channels with significant signal variation."""
    with np.load(filepath) as data:
        ang2 = data['ang2']
    stds = np.std(ang2, axis=0)
    return np.where(stds > threshold)[0]


def plot_r_vs_p_comparison(file1, file2, channels=None, rbias=10e3, Rshunt=250e-6, rows=4, cols=4, save_path=None, show_plot=True):
    """
    Plots R vs P subplots comparing file1 and file2 for each channel.
    """
    label1 = os.path.basename(file1)
    label2 = os.path.basename(file2)

    # Detect active channels if not specified
    if channels is None:
        ch1 = get_active_channels(file1)
        ch2 = get_active_channels(file2)
        channels = sorted(list(set(ch1).union(set(ch2))))

    if not channels:
        print("No active channels found to plot.")
        return

    num_channels = len(channels)
    plots_per_page = rows * cols
    num_pages = int(np.ceil(num_channels / plots_per_page))

    print(f"Plotting R vs P for {num_channels} channels across {num_pages} page(s)...")

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), sharex=False, sharey=False)
        axes = np.atleast_2d(axes).flatten()

        page_channels = channels[page * plots_per_page : (page + 1) * plots_per_page]

        for i, ch in enumerate(page_channels):
            ax = axes[i]
            
            p1, r1, _ = load_and_compute_channel(file1, ch, rbias=rbias, Rshunt=Rshunt)
            p2, r2, _ = load_and_compute_channel(file2, ch, rbias=rbias, Rshunt=Rshunt)

            if p1 is not None and len(p1) > 0:
                ax.plot(p1, r1, color='#1f77b4', lw=1.8, label=f"File 1 ({label1[:15]}...)")

            if p2 is not None and len(p2) > 0:
                ax.plot(p2, r2, color='#ff7f0e', lw=1.8, linestyle='--', label=f"File 2 ({label2[:15]}...)")

            ax.set_title(f"Channel {ch}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Ptes (pW)", fontsize=9)
            ax.set_ylabel("Rtes (mΩ)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax.legend(fontsize=8, loc='best')

        # Hide unused subplots on final page
        for j in range(len(page_channels), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"R vs P Comparison: {label1} vs {label2} (Page {page + 1}/{num_pages})", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            out_file = save_path if num_pages == 1 else f"{os.path.splitext(save_path)[0]}_page{page+1}.png"
            plt.savefig(out_file, dpi=200)
            print(f"Saved summary plot to: {out_file}")

    if show_plot:
        plt.show()


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Compare Rtes vs Ptes curves between two IV files as subplots for each channel."
    )
    def_f1, def_f2 = find_default_files()

    parser.add_argument("--file1", type=str, default=def_f1, help="Path to first IV .npz file")
    parser.add_argument("--file2", type=str, default=def_f2, help="Path to second IV .npz file")
    parser.add_argument("--channels", type=int, nargs="+", default=None, help="Channel IDs to plot (default: auto-detect active)")
    parser.add_argument("--rbias", type=float, default=10e3, help="Bias resistance in Ohms (default: 10000)")
    parser.add_argument("--rshunt", type=float, default=250e-6, help="Shunt resistance in Ohms (default: 250e-6)")
    parser.add_argument("--rows", type=int, default=4, help="Grid rows per page (default: 4)")
    parser.add_argument("--cols", type=int, default=4, help="Grid cols per page (default: 4)")
    parser.add_argument("--save", type=str, default=None, help="Optional output PNG filepath")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive plot window")

    args = parser.parse_args(args_list)

    if not args.file1 or not os.path.exists(args.file1):
        print(f"Error: File 1 not found: '{args.file1}'")
        sys.exit(1)
    if not args.file2 or not os.path.exists(args.file2):
        print(f"Error: File 2 not found: '{args.file2}'")
        sys.exit(1)

    print(f"Comparing R vs P between:\n  File 1: {args.file1}\n  File 2: {args.file2}")
    plot_r_vs_p_comparison(
        args.file1,
        args.file2,
        channels=args.channels,
        rbias=args.rbias,
        Rshunt=args.rshunt,
        rows=args.rows,
        cols=args.cols,
        save_path=args.save,
        show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()
