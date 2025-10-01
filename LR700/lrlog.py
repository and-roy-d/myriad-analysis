import bftc
import lr700
import datetime
import os
import time
import numpy as np
from npy_append_array import NpyAppendArray


# Get user-specified directory or default to "Data" folder with date subfolders
default_dir = os.path.join("Data", datetime.datetime.now().strftime("%Y%m%d"))
save_dir = input(f"Enter the directory to save data (or press Enter for default: {default_dir}): ")
if not save_dir:
    save_dir = default_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create a filename with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = os.path.join(save_dir, f"lr700log_{timestamp}.npy")

npaa = NpyAppendArray(filename)

# Data logging function
def log_data(logging_interval_s=1):
    while True:
        try:
            # Read data from the instrument
            r = lr700.read_ohm()
            t = bftc.read_mxc_temperature()

            # Get current time in Unix timestamp format
            current_time = time.time()

            if r is None:
                print("failed lr700 read")
                continue

            entry =  np.array([(r, t, current_time)], 
                 dtype=[('r_ohm', 'f8'), ('t_K', 'f8'), ('times_s', 'f8')])
            print(f"{r=:.6f} Ohm, {t=:.6f} K, {current_time=} s to {filename=}")
            npaa.append(entry)
        except KeyboardInterrupt:
            print("ctrl-c detected, exting")
            npaa.close()
            return

        # Sleep for the specified logging interval
        time.sleep(logging_interval_s)

if __name__ == "__main__":
    log_data()