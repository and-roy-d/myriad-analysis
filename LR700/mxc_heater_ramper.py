import bftc
import time
import tqdm


def ramp_up(P_init, target_temp, step=30e-6, timeout=3600, sleep_time=60):
    """Ramps up heater power in steps until target temperature is reached."""
    P_set = P_init
    start_time = time.time()
    while True:
        mxc_temp = bftc.read_mxc_temperature()
        with tqdm.tqdm(total=sleep_time, desc="Waiting for next step", unit="s", dynamic_ncols=True) as pbar:
            pbar.write(f'Heater power = {P_set * 1e6} uW, MXC_temp = {mxc_temp * 1000} mK')
            for _ in range(int(sleep_time)):
                time.sleep(1)
                pbar.update(1)

        if mxc_temp > target_temp:
            break
        if time.time() - start_time > timeout:
            print("Ramp up timed out!")
            return False

        P_set += step  # Add the step
        bftc.set_heaterpower(P_set)

    print("Ramp up complete.")
    return True


def ramp_down(P_init, target_temp, step=30e-6, timeout=3600, sleep_time=60):
    """Ramps down heater power in steps until target temperature is reached."""
    P_set = P_init
    start_time = time.time()
    while True:
        mxc_temp = bftc.read_mxc_temperature()
        with tqdm.tqdm(total=sleep_time, desc="Waiting for next step", unit="s", dynamic_ncols=True) as pbar:
            pbar.write(f'Heater power = {P_set * 1e6} uW, MXC_temp = {mxc_temp * 1000} mK')
            for _ in range(int(sleep_time)):
                time.sleep(1)
                pbar.update(1)

        if mxc_temp < target_temp:
            break
        if time.time() - start_time > timeout:
            print("Ramp down timed out!")
            return False

        if P_set < 1e-9:  # Check for minimum power
            print("Heater power reached minimum, stopping ramp down.")
            return True

        P_set -= step  # Subtract the step
        bftc.set_heaterpower(P_set)

    print("Ramp down complete.")
    return True


# --- User Interaction ---
while True:
    direction = input("Enter ramp direction ('up' or 'down'): ").lower()
    if direction in ('up', 'down'):
        break
    else:
        print("Invalid direction. Please enter 'up' or 'down'.")

P_init = float(input("Enter initial heater power (in uW, e.g., 20 for 20uW): ")) * 1e-6  # Convert to W
target_temp = float(input("Enter target MXC temperature (in mK, e.g., 130 for 130mK): ")) * 1e-3  # Convert to K

# Get step, sleep time, and timeout from user or use default
try:
    step = float(input("Enter power step (in uW, or press Enter for default 10uW): ")) * 1e-6  # Convert to W
except ValueError:
    step = 10e-6
    print("Using default step of 10 uW.")

try:
    sleep_time = int(input("Enter time between measurements (in seconds, or press Enter for default 10s): "))
except ValueError:
    sleep_time = 10
    print("Using default sleep time of 10 seconds.")

try:
    timeout = int(input("Enter timeout in seconds, or press Enter for default 1 hour(3600s): "))
except ValueError:
    timeout = 3600
    print("Using default timeout of 1 hour (3600s).")

if direction == 'up':
    success = ramp_up(P_init, target_temp, step, timeout, sleep_time)
elif direction == 'down':
    success = ramp_down(P_init, target_temp, step, timeout, sleep_time)

if success:
    print("Program finished successfully.")
else:
    print("Program finished with timeout.")