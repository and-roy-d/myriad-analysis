import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.tri as tri
from scipy.interpolate import griddata

def process_csv_files_polars(folder_path, channel_number):
    """
    Processes CSV files using Polars, creating a heatmap.

    Args:
        folder_path (str): The path to the folder containing CSV files.
        channel_number (str): The channel number (e.g., "0001").
    """
    filename = None
    for file in os.listdir(folder_path):
        if f"_ch{channel_number}_" in file and file.endswith(".csv"):
            filename = file
            break

    if filename is None:
        print(f"Error: CSV file for channel {channel_number} not found.")
        return

    filepath = os.path.join(folder_path, filename)

    try:
        df = pl.read_csv(filepath, columns=["state_label", "pulse_rms"])
        unique_labels = df["state_label"].unique().to_list()

        x_coords = []
        y_coords = []
        avg_pulses = []

        for label in unique_labels:
            if label not in ["MOVING", "IGNORE"]:
                signal = df.filter(pl.col("state_label") == label)["pulse_rms"].mean()
                match = re.search(r"x=([\d.-]+)\s+y=([\d.-]+)", label)
                if match is not None:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    x_coords.append(x)
                    y_coords.append(y)
                    avg_pulses.append(signal)

        n_points = int(np.sqrt(len(avg_pulses)))
        upscale_factor = 2

        if x_coords:
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)
            avg_pulses = np.array(avg_pulses)

            if False: #griddata interpolation
                xi = np.linspace(min(x_coords), max(x_coords), n_points * upscale_factor)
                yi = np.linspace(min(y_coords), max(y_coords), n_points * upscale_factor)
                xi, yi = np.meshgrid(xi, yi)
                zi = griddata((x_coords, y_coords), avg_pulses, (xi, yi), method='cubic')
                plt.contourf(xi, yi, zi, 20, cmap="coolwarm")
            else: #triangulation interpolation
                triang = tri.Triangulation(x_coords, y_coords)
                plt.tricontourf(triang, avg_pulses, cmap="coolwarm")

            plt.colorbar(label="Avg pulse height")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Heatmap of channel {channel_number}")
            plt.tight_layout()
            plt.show()

        else:
            print(f"Warning: No valid data found for channel {channel_number}.")

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except pl.exceptions.ComputeError:
        print(f"Error: Could not parse CSV file or compute: {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {filename}: {e}")

# Example usage:
folder_path = "/home/pcuser/Runs/Cooldown_A14/scan_maps"

channel_number = "4107" #replace with the channel you want to view.
process_csv_files_polars(folder_path, channel_number)