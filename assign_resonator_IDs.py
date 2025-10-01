import os
import yaml
import numpy as np
import pandas as pd

def load_banddefs(base_dir, band):
    """Load YAML files for band (both a and b)."""
    bandname = f"band{band:02d}"
    files = [os.path.join(base_dir, f"{bandname}{suffix}.yml") for suffix in ["a", "b"]]
    res_list = []
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f, "r") as stream:
            yml = yaml.safe_load(stream)
        for r in yml["resonators"]:
            res_list.append({
                "bandname": yml["bandname"],
                "f0_design_hz": r["f0"],
                "wx_mm": r["wx"],
            })
    return pd.DataFrame(res_list)

def load_tones(tones_file):
    """Load tones file (ini-like structure)."""
    tones = {}
    with open(tones_file, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                tones[key.strip()] = val.strip()

    lo_freq = float(tones["lo_freq_hz"])  # Hz
    tones_up = np.fromstring(
        tones["tones_up_freq_mhz"].strip("[]"), sep=","
    ) * 1e6  # MHz → Hz
    tones_good = np.fromstring(
        tones["tones_good"].strip("[]")
        .replace("True", "1")
        .replace("False", "0"),
        sep=",",
        dtype=int,
    ).astype(bool)

    freqs = lo_freq + tones_up
    good_freqs = freqs[tones_good]

    df = pd.DataFrame({
        "tone_freq_hz": good_freqs,
        "autotune_id": np.arange(len(good_freqs)),
    })
    df["dastard_id"] = 4096 + df["autotune_id"]
    return df

def match_resonators(res_df, tone_df, tol_hz=2e6):
    """Match resonator f0 to closest tone within tolerance."""
    res_df = res_df.copy()
    res_df["tone_freq_hz"] = np.nan
    res_df["autotune_id"] = np.nan
    res_df["dastard_id"] = np.nan

    for i, row in res_df.iterrows():
        diffs = np.abs(tone_df["tone_freq_hz"] - row["f0_design_hz"])
        j = np.argmin(diffs)
        if diffs.iloc[j] <= tol_hz:
            res_df.loc[i, ["tone_freq_hz", "autotune_id", "dastard_id"]] = tone_df.iloc[j]

    return res_df

def add_pixel_map(res_df, pixel_map_df):
    """
    Attach pixel_number and umux_bondpad by ordering wx
    and matching to provided pixel map.
    """
    # sort by physical x position
    res_sorted = res_df.sort_values("wx_mm").reset_index(drop=True)

    # assign bondpads in order (descending is typical, adjust if needed)
    res_sorted["umux_bondpad"] = np.arange(len(res_sorted), 0, -1)

    # merge with pixel map (keeps NaNs where bondpad is missing in pixel map)
    merged = res_sorted.merge(pixel_map_df,
                              on="umux_bondpad",
                              how="left")
    return merged

def make_mapping(band, banddef_dir, tones_file, pixel_map_df, out_csv=None):
    res_df = load_banddefs(banddef_dir, band)
    tone_df = load_tones(tones_file)
    matched = match_resonators(res_df, tone_df)
    mapped = add_pixel_map(matched, pixel_map_df)

    if out_csv:
        mapped.to_csv(out_csv, index=False)
    return mapped

# Example usage
if __name__ == "__main__":
    band = 3
    banddef_dir = "/home/pcuser/Runs/Resonator banddef/umux2Mv1.0"
    tones_file = "/data/20250905/autotune/20250905_124927/tones_20250905_124927"

    # your pixel ↔ bondpad mapping table
    pixel_map = pd.DataFrame({
        "pixel_number": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
        "umux_bondpad": [27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,4,3]
    })

    df = make_mapping(band, banddef_dir, tones_file, pixel_map, out_csv=False)
    print(df.head())
