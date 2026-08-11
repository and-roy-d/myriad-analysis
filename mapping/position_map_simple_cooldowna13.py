import matplotlib.pyplot as plt
import numpy as np

def create_channel_map_from_dict(channel_positions_volts, special_channels):

    x_values = []
    y_values = []
    channel_numbers = []
    colors = []
    for channel, (x, y) in channel_positions_volts.items():
        x_values.append(distance_to_array*2*np.array(x)*volt_to_angle)
        y_values.append(distance_to_array*2*np.array(y)*volt_to_angle)
        channel_numbers.append(channel)
        if channel in special_channels:
            colors.append('red')
        else:
            colors.append('blue')

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c=colors, marker='o')

    for i, channel in enumerate(channel_numbers):
        plt.annotate('+  '+str(channel), (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    plt.xlabel("MEMS X-coordinate")
    plt.ylabel("MEMS Y-coordinate")
    plt.title("Response Map")
    plt.grid(True)
    plt.show()

# Example Usage
channel_positions_volts = {
    4100: (0.08, -0.34),
    4117: (-0.18, -0.34),
    4101: (-0.1, -0.24),
    4103: (-0.1, -0.08),
    4107: (-0.1, -0.18),
    4119: (-0.2, -0.06),
    4118: (0.02, -0.06),
    4122: (-0.22, 0.18),
    4121: (0.02, 0.2),
    4120: (0, 0),
    4105: (-0.08, 0.06),
    4124: (-0.2, 0.42),
    4109: (-0.1, 0.5),
    4114: (0.1, -0.9),
    4098: (0, -0.9),
    4099: (-0.12, -0.9),
}
distance_to_array = 1# cm, approx
volt_to_angle = 1
special_channels = [4120, 4105, 4124, 4109, 4114, 4098, 4099]

create_channel_map_from_dict(channel_positions_volts, special_channels)

