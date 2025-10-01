import yaml
import matplotlib.pyplot as plt
import numpy as np

def create_indexed_f0_dict(file1_path, file2_path):
    """
    Reads two YAML files, combines resonator data, sorts by wx,
    and creates a dictionary of {index: f0}.

    Args:
        file1_path (str): Path to the first YAML file.
        file2_path (str): Path to the second YAML file.

    Returns:
        dict: A dictionary where keys are indices (sorted by wx) and values are f0.
    """

    all_resonators = []

    for file_path in [file1_path, file2_path]:
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

                if data and 'resonators' in data:
                    all_resonators.extend(data['resonators'])

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML format in {file_path}: {e}")
            return None
        except KeyError as e:
            print(f"Error: Missing key in {file_path}: {e}")
            return None
        except TypeError as e:
            print(f"Error: unexpected type in {file_path}: {e}")
            return None

    if not all_resonators:
        print("No resonator data found in the provided YAML files.")
        return {}

    # Sort resonators by wx
    sorted_resonators = sorted(all_resonators, key=lambda x: x['wx'])

    # Create the indexed f0 dictionary
    indexed_f0_dict = {index: resonator['f0'] for index, resonator in enumerate(sorted_resonators)}

    return indexed_f0_dict

def plot_f0_vs_wx_from_yaml(file1_path, file2_path):
    """
    Reads two YAML files, extracts f0 and wx values for each resonator,
    and plots f0 vs wx combining data from both files.

    Args:
        file1_path (str): Path to the first YAML file.
        file2_path (str): Path to the second YAML file.
    """

    f0_values = []
    wx_values = []

    for file_path in [file1_path, file2_path]:
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

                if data and 'resonators' in data:
                    for resonator in data['resonators']:
                        f0_values.append(resonator['f0'])
                        wx_values.append(resonator['wx'])

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML format in {file_path}: {e}")
            return
        except KeyError as e:
            print(f"Error: Missing key in {file_path}: {e}")
            return
        except TypeError as e:
            print(f"Error: unexpected type in {file_path}: {e}")
            return

    if f0_values and wx_values:
        plt.figure(figsize=(10, 6))
        plt.scatter(wx_values, f0_values)
        # plt.text()
        plt.xlabel('wx')
        plt.ylabel('f0')
        plt.title('f0 vs wx from Two YAML Files')
        plt.grid(True)
        if tones_up_dict:
            for i, tone in enumerate(tones_up_dict.values()):
                plt.axhline(y=tone * 1e6, color='r', linestyle='--', linewidth=1)
                freq_mhz = np.array(tones_up_dict[i])

                plt.text(x = -10, y=tone*1e6, s=f"{freq_mhz=} chan={4096+i}")
        plt.show()
    else:
        print("No resonator data found in the provided YAML files.")





def create_autotune_to_squid_dict(tones_up_dict, indexed_f0_dict):
    """
    Matches the closest value from tones_up_dict to the values in indexed_f0_dict.
    If the difference is less than 2e6, assigns NaN to the index.

    Args:
        tones_up_dict (dict): Dictionary with indices as keys and tones as values.
        indexed_f0_dict (dict): Dictionary with indices as keys and f0 values as values.

    Returns:
        dict: A dictionary mapping tones_up indices to indexed_f0 indices or NaN.
    """

    autotune_to_squid_dict = {}
    f0_values = list(indexed_f0_dict.values())

    for tones_up_index, tones_up_value in tones_up_dict.items():
        closest_index = min(range(len(f0_values)), key=lambda i: abs(f0_values[i] - tones_up_value * 1e6)) #convert tones_up_value to Hz
        difference = abs(f0_values[closest_index] - tones_up_value * 1e6)

        if difference < 7e6:
            autotune_to_squid_dict[tones_up_index] = closest_index
        else:
            autotune_to_squid_dict[tones_up_index] = np.nan

    return autotune_to_squid_dict

tones_up_freq_mhz = [
    5507.6875, 5521.0, 5533.46875, 5547.0625, 5561.03125, 5576.6875, 5594.21875, 5602.9375,
    5615.3125, 5634.34375, 5644.0, 5656.9375, 5671.28125, 5684.96875, 5699.59375, 5714.6875,
    5724.15625, 5737.5625, 5751.15625, 5760.25, 5777.6875, 5790.90625, 5807.21875, 5820.71875,
    5832.71875, 5846.78125, 5873.5, 5886.71875, 5901.8125, 5916.15625, 5930.125, 5943.25,
]

def create_indexed_dict_from_list(input_list):
    indexed_dict = {index: value for index, value in enumerate(input_list)}
    return indexed_dict

tones_up_dict = create_indexed_dict_from_list(tones_up_freq_mhz)


squid_channel_to_pixel_dict = {3:24, 4:23, 6:22, 7:21, 8:20, 9:19, 10:18, 11:17, 12:16, 13:15, 14:14, 15:13, 16:12, 17:11,
                               18:10, 19:9, 20:8, 21:7, 22:6, 23:5, 24:4, 25:3, 26:2, 27:1}

base_path = '/home/pcuser/Runs/Resonator banddef/umux2Mv1.0/'
file1_path = base_path+'band03a.yml'
file2_path = base_path+'band03b.yml'
indexed_f0_dict = create_indexed_f0_dict(file1_path, file2_path)
autotune_to_squid_dict = create_autotune_to_squid_dict(tones_up_dict, indexed_f0_dict)

print(autotune_to_squid_dict)
plot_f0_vs_wx_from_yaml(file1_path, file2_path)