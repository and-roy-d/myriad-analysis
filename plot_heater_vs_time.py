import json
import os
from datetime import datetime
from qsdumux.instruments import bftc  # Assuming this library is available
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import numpy as np
from scipy.signal import savgol_filter
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12


def get_and_save_data(filename_prefix="heater_data", start_minutes_ago=1000, stop_minutes_ago=250):
    """Retrieves heater power data, saves it to a JSON file, and returns the data."""
    try:
        data = bftc.get_heater_power(start_minutes_ago=start_minutes_ago, stop_minutes_ago=stop_minutes_ago)
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp_str}.json"

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {filename}")
        return data
    except Exception as e:
        print(f"Error retrieving and saving data: {e}")
        return None

def load_saved_data(filename):
    """Loads data from a saved JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            print(f"Data loaded from {filename}")
            return data
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            return None
    else:
        print(f"File {filename} does not exist.")
        return None

def load_log_data(log_file, timezone_str="America/Denver"):
    """Loads temperature log data from a .log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 3:
                    date_str, time_str, temp_str = parts
                    datetime_str = f"{date_str} {time_str}"
                    temp = float(temp_str)
                    data.append([datetime_str, temp])

        df = pd.DataFrame(data, columns=['datetime', 'temperature'])
        timezone = pytz.timezone(timezone_str)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%y %H:%M:%S').apply(timezone.localize)
        df = df.set_index('datetime')
        return df
    except Exception as e:
        print(f"Error loading log data: {e}")
        return None

def plot_power_and_temperature(data, log_df, timezone_str="America/Denver", filter_type=None, window=10, polyorder=2, start_time=None, stop_time=None, vlines_timestamps=None, vlines_annotations=None):

    if not data or 'measurements' not in data or 'timestamp' not in data['measurements'] or 'power' not in data['measurements']:
        print("Invalid data format for power data.")
        return

    timestamps = np.array(data['measurements']['timestamp'])
    power = np.array(data['measurements']['power'])

    utc_timestamps = [datetime.utcfromtimestamp(ts) for ts in timestamps]
    timezone = pytz.timezone(timezone_str)
    local_timestamps = [utc_ts.replace(tzinfo=pytz.utc).astimezone(timezone) for utc_ts in utc_timestamps]

    df_power = pd.DataFrame({'timestamp': local_timestamps, 'power': power})
    df_power = df_power.set_index('timestamp')

    if start_time:
        df_power = df_power[df_power.index >= start_time]
        if log_df is not None:
            log_df = log_df[log_df.index >= start_time]

    if stop_time:
        df_power = df_power[df_power.index <= stop_time]
        if log_df is not None:
            log_df = log_df[log_df.index <= stop_time]

    filtered_power = df_power['power'].copy()

    if filter_type == 'rolling':
        filtered_power = df_power['power'].rolling(window=window).mean()
    elif filter_type == 'savgol':
        try:
            filtered_power = savgol_filter(df_power['power'], window_length=window, polyorder=polyorder)
            filtered_power = pd.Series(filtered_power, index=df_power.index)
        except Exception as e:
            print(f"Error applying Savitzky-Golay filter: {e}")
            filtered_power = df_power['power']

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df_power.index, df_power['power']*1e6, color='k', alpha=0.25, marker='o', linestyle='-',)
    ax1.plot(df_power.index, filtered_power*1e6, marker='.', linestyle='-', label=f'Filtered Power ({filter_type})' if filter_type else 'Power', color='k')
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"MXC heater Power ($\mu$W)", color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(which='both', ls='--', alpha=0.5, zorder=0)
    ax1.set_ylim(0,4)
    if log_df is not None:
        ax2 = ax1.twinx()
        filtered_temp = log_df['temperature'].copy()

        if filter_type == 'rolling':
            filtered_temp = log_df['temperature'].rolling(window=window).mean()
        elif filter_type == 'savgol':
            try:
                filtered_temp = savgol_filter(log_df['temperature'], window_length=window, polyorder=polyorder)
                filtered_temp = pd.Series(filtered_temp, index=log_df.index)
            except Exception as e:
                print(f"Error applying Savitzky-Golay filter to temperature: {e}")
                filtered_temp = log_df['temperature']
        ax2.plot(log_df.index, log_df['temperature'] * 1e3, color='r', alpha=0.25, marker='o', linestyle='-', )
        ax2.plot(log_df.index, filtered_temp*1e3, marker='.', linestyle='-', label=f'Filtered Temp ({filter_type})' if filter_type else 'Temperature', color='red')
        ax2.set_ylabel("Scepter temperature (mK)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(20.9,21.2)

    if vlines_timestamps and vlines_annotations and len(vlines_timestamps) == len(vlines_annotations):
        for ts, annotation in zip(vlines_timestamps, vlines_annotations):
            if ts in df_power.index:
                ax1.axvline(x=ts, color='red', linestyle='--', linewidth=1)
                ax1.annotate(annotation, xy=(ts, max(df_power['power'])), xytext=(ts, max(df_power['power']) * 1.05),
                             arrowprops=dict(facecolor='black', shrink=0.05),
                             horizontalalignment='center', verticalalignment='bottom')

    fig.tight_layout()
    plt.show()




savePath = "/home/pcuser/Runs/Cooldown_A15/"
# data = get_and_save_data(start_minutes_ago=200, stop_minutes_ago=0)
# # data  = load_saved_data(savePath+"heater_data_20250317_151638.json")
timestamp_str = '20250318_154031'
filename_to_load = savePath+f"heater_data_{timestamp_str}.json"
loaded_data = load_saved_data(filename_to_load)


if loaded_data:
    timezone = pytz.timezone("America/Denver") #match the plotting function.
    start_plot = timezone.localize(datetime(2025, 3, 18, 14, 15, 0)) #example start time.
    stop_plot = timezone.localize(datetime(2025, 3, 18, 15, 5, 0)) #example stop time.

    # vlines_times = [timezone.localize(datetime(2025, 3, 18, 14, 40, 0)),
    #                 timezone.localize(datetime(2025, 3, 18, 14, 46, 0)),
    #                 timezone.localize(datetime(2025, 3, 18, 14, 56, 0))]
    # vlines_texts = ["MEMS enabled", "MEMS disabled", "MEMS enabled"]
    vlines_times = None; vlines_texts= None

    log_file = "CH7_T_25-03-18.log"
    log_data = load_log_data(savePath+log_file)

    if log_data is not None:
        plot_power_and_temperature(loaded_data, log_data, filter_type='rolling', window=20, start_time=start_plot,
                                   stop_time=stop_plot, vlines_timestamps=vlines_times, vlines_annotations=vlines_texts)
