import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({'font.size': 14})

def create_channel_map_from_dict(channel_positions_volts, special_channels):

    x_values = []
    y_values = []
    channel_numbers = []
    colors = []
    for channel, (x, y) in channel_positions_volts.items():
        x_values.append(x*scale_x)
        y_values.append(y*scale_y)
        channel_numbers.append(channel)
        if channel in special_channels:
            colors.append('red')
        else:
            colors.append('blue')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_values, y_values, c=colors, marker='o')

    for i, channel in enumerate(channel_numbers):
        ax.annotate(f"{channel_to_pix_map[channel]}\n({channel})", (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)

    ax.set_xlabel("MEMS X-coordinate")
    ax.set_ylabel("MEMS Y-coordinate")
    ax.set_title("Laser-derived response map \n Myriad cooldown A17, April 1 2025\n Pixel number (Dastard channel)")
    plt.grid(which = "both", ls= ':', zorder = 0, alpha=0.75)
    ax.set_aspect('equal')
    # ax.set(xlim=(-0.4, 0.4), ylim=(-0.8, 0.8))
    plt.show()

# Example Usage
channel_positions_volts = {
    4100: (0.18, -0.38),
    4117: (-0.26, -0.38),
    4101: (-0.115, -0.38),
    # 4103: (-0.1, -0.08),
    4104: (0.18, 0.13),
    4119: (-0.26, -0.12),
    4118: (0.04, -0.12),
    4122: (-0.26, 0.345),
    4121: (0.04, 0.345),
    4120: (0.04, 0.13),
    4105: (-0.115, 0.13),
    4124: (-0.26, 0.63),
    4114: (0.2, -0.63),
    4107: (-0.115, 0.345),
    4098: (0.04, -0.63),
    4123: (0.04, 0.63),
    4115: (-0.26, -0.63),
    4116: (0.04, -0.38),
    4109: (-0.115, 0.63),
    4106: (0.18, 0.345)
}

channel_to_pix_map = {4124:1, 4109:2, 4123:3, 4122:5, 4107:6, 4121:7, 4106:8, 4105:10, 4120:11, 4104:12, 4119:13,
4118:15, 4117:17, 4101:18, 4116:19, 4100:20, 4115:21, 4098:23, 4114:24}
distance_to_array = 1# cm, approx
volt_to_angle = 1/2
special_channels = [4107]
scale_x = 1#5462.2 # um/MEMS unit
scale_y = 1#4661
create_channel_map_from_dict(channel_positions_volts, special_channels)


data = []
for channel, (x, y) in channel_positions_volts.items():
    if channel in channel_to_pix_map:
        pixel = channel_to_pix_map[channel]
        data.append({
            "pixel number": pixel,
            "dastard channel": channel,
            "mems x": x,
            "mems y": y
        })

df = pd.DataFrame(data)
df.to_csv("/home/pcuser/Runs/Cooldown_A17/channel_pixel_mapping.csv", index=False)

