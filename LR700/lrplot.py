import numpy as np
import matplotlib.pyplot as plt
import pathlib
# plt.ion()
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

path = pathlib.Path(__file__)
dirs = list((path.parent / "Data").glob("*"))
print(f"{dirs=}")
dir_ = dirs[-1]
files = list(dir_.glob("*"))
print(f"{files=}")
file = files[-1]
file = 'Data/20241223/lr700log_20241223-122816.npy'
print(f"{file=}")
data = np.load(file)

device = 'F2'
filename = str(file).split('_')[-1].split('.')[0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(8,8))
ax1.plot(data["times_s"], data["r_ohm"]*1000,"r.")
ax2.plot(data["times_s"], data["t_K"]*1000,"r.")
ax1.set_xlabel("time_s")
ax1.set_ylabel(r"R (m$\Omega$)")
ax2.set_ylabel("MXC temp (mK)")
ax1.grid(which='major', ls='--', alpha=0.5)
ax2.grid(which='major', ls='--', alpha=0.5)
fig.suptitle(f'Device: {filename}:({device})')


fig, ax = plt.subplots(figsize=(8,8))
ax.plot(data["t_K"]*1000, data["r_ohm"]*1000,"r.")
ax.set_xlabel("MXC temp (mK)")
ax.set_ylabel(r"R (m$\Omega$)")
ax.set_title(f'Device: {device}')
ax.grid(which='major', ls='--', alpha=0.5)
plt.show()
plt.tight_layout()
# plt.pause(60)