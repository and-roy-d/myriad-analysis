import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters

# Try import mdt3; if missing create safe dummies so script doesn't crash.
try:
    from mdt3 import tes_simple, tes_compound, tes_dangling, tes_intervening
    from mdt3 import mdt3_core
except Exception:
    print("Warning: mdt3 library not found. Using safe dummies - fitting will not be meaningful.")
    class DummyMDT3:
        def makeDefaultParamsDict(self, num_sets=1):
            # Return a minimal lmfit Parameters object for safe testing
            p = Parameters()
            # add a few potential parameter names used below so update_tes_params doesn't KeyError
            for name, val in [
                ('G_tes_bath_0', 0.093),
                ('G_abs_tes_0', 1.06),
                ('T_tes_0', 0.053),
                ('T_bath_0', 0.021),
                ('C_tes_0', 0.06),
                ('C_abs_0', 0.16),
                ('alpha_I_0', 1300),
                ('beta_I_0', 60),
                ('R_0_0', 470e-6),
                ('R_L_0', 250e-6),
                ('L_0', 120),
                ('n_mem_0', 3.8),
                ('initE_0', 2.4),
            ]:
                p.add(name, value=val, vary=False)
            return p
        def calc_pulse(self, params, tvals):
            # return tuple similar to real implementation
            return (np.zeros_like(tvals),)
        def calc_noise(self, params, fvals):
            return (np.zeros_like(fvals),)
        def calc_derived_params(self, params):
            self.simple_params = {"dummy": 0}
    class DummyTES:
        def __init__(self, core): pass
        def calc_pulse(self, params, tvals): return (np.zeros_like(tvals),)
        def calc_noise(self, params, fvals): return (np.zeros_like(fvals),)
        def calc_derived_params(self, params): pass

    mdt3_core = DummyMDT3()
    tes_intervening = DummyTES(mdt3_core)
    tes_simple = DummyTES(mdt3_core)

import functools
import cProfile
import pstats

# --- Constants ---
# Use physical_constants dict to get the magnetic flux quantum robustly:
phi0 = scipy.constants.physical_constants['mag. flux quantum'][0]  # [Wb]

def phi0_to_amp(inval):
    """Convert array in flux-quanta units to Amps using the 'min' sensitivity value.
       NOTE: min_SI is a magic constant in your original code (248e-12). Keep consistent units.
    """
    min_SI = 248e-12  # TODO: confirm this is the correct detector sensitivity (Wb/A)
    min_phi0_per_amp = min_SI / phi0
    return inval * (1.0 / min_phi0_per_amp)

def amp_to_phi0(inval):
    min_SI = 248e-12
    min_phi0_per_amp = min_SI / phi0
    return inval * min_phi0_per_amp

# --- Robust loaders ---
def load_noise_file(filename):
    """
    Attempts to be tolerant to a few npz shapes:
    - .npz with arrays 'Pxx' and 'f'
    - .npz with a single dict-like object inside
    Returns: psd (2D or 1D numpy), freqs (1D numpy) or (None, None) on error.
    """
    try:
        with np.load(filename, allow_pickle=True) as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError("NPZ file empty")
            # Common possible keys
            if 'Pxx' in f and 'f' in f:
                Pxx = f['Pxx']
                freqs = f['f']
                # If saved as object arrays with per-channel dicts, try to unwrap
                if Pxx.dtype == object or freqs.dtype == object:
                    # try to grab the first item that is array-like
                    Pxx = Pxx.item() if Pxx.size == 1 else Pxx
                    freqs = freqs.item() if freqs.size == 1 else freqs
                return Pxx, freqs
            # If there is a single array that contains a dict
            if len(keys) == 1:
                obj = f[keys[0]].item() if f[keys[0]].size == 1 else f[keys[0]]
                # Try to find 'Pxx' and 'f' inside the object/dict
                if isinstance(obj, dict):
                    if 'Pxx' in obj and 'f' in obj:
                        return obj['Pxx'], obj['f']
                    # try other likely names
                    for alt in ['psd', 'Sxx', 'S_I']:
                        if alt in obj:
                            return obj[alt], obj.get('f', obj.get('freqs', None))
                # fallback: try to interpret as 2-column arrays
                arr = np.asarray(obj)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    freqs = arr[:, 0]
                    psd = arr[:, 1:]
                    return psd, freqs
            # If multiple keys, try to pick likely ones
            if 'freqs' in keys:
                freqs = f['freqs']
                psd_key = next((k for k in keys if 'pxx' in k.lower() or 'psd' in k.lower()), None)
                if psd_key:
                    return f[psd_key], freqs
            raise ValueError("Could not find PSD and frequency arrays in npz.")
    except FileNotFoundError:
        print(f"Error: Noise file not found at {filename}")
        return None, None
    except Exception as e:
        print(f"Error loading noise file {filename}: {e}")
        return None, None

def load_avg_pulse(filename, pulse_arrival_sample=0):
    try:
        with np.load(filename, allow_pickle=True) as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError("NPZ file empty")
            # Try common names
            candidates = ['array1', 'avg_pulse', 'pulse_avg', 'pulse']
            for c in candidates:
                if c in f:
                    avg_pulse_raw = np.asarray(f[c]).astype(float) / 4096.0
                    break
            else:
                # fallback: take the first array-like key
                avg_pulse_raw = np.asarray(f[keys[0]]).astype(float) / 4096.0

            # baseline subtraction - be defensive on lengths
            n = len(avg_pulse_raw)
            if n <= 2:
                raise ValueError("Pulse array too short")
            start_idx = 1
            end_idx = min(400, n - 1)
            if start_idx >= end_idx:
                baseline = avg_pulse_raw[0]
            else:
                baseline = np.mean(avg_pulse_raw[start_idx:end_idx])
            avg_pulse_baselined = avg_pulse_raw - baseline
            times = (np.arange(len(avg_pulse_baselined)) - pulse_arrival_sample) * 8e-6
            return phi0_to_amp(avg_pulse_baselined), times
    except FileNotFoundError:
        print(f"Error: Pulse file not found at {filename}")
        return None, None
    except Exception as e:
        print(f"Error loading pulse file {filename}: {e}")
        return None, None

# --- Parameter initialization and updates ---
def initialize_model(tes_model_instance):
    # Call makeDefaultParamsDict on the TES model
    tes_params = tes_model_instance.makeDefaultInterParamsDict(num_sets=1)
    update_tes_params(tes_params)
    return tes_params


def update_tes_params(params):
    # Update parameters if present. Be careful to pick sensible min/max values.
    # NOTE: lmfit.Parameter.set expects min <= value <= max if provided.
    def safe_set(name, **kwargs):
        try:
            params[name].set(**kwargs)
        except KeyError:
            pass

    # Examples: check ranges and units
    safe_set('G_tes_bath_0', value=0.093, vary=False, min=0.0, max=1.0)  # fixed (min < max)
    safe_set('G_abs_tes_0', value=1.06, vary=True, min=0.2, max=5.0)
    safe_set('T_tes_0', value=0.053, vary=False, min=0.045, max=0.055)
    safe_set('T_bath_0', value=0.021, vary=False, min=0.019, max=0.025)
    safe_set('C_tes_0', value=0.06, vary=True, min=0.01, max=0.1)
    safe_set('C_abs_0', value=0.16, vary=True, min=0.01, max=1.0)
    safe_set('alpha_I_0', value=435, vary=True, min=20, max=1000)
    safe_set('beta_I_0', value=17, vary=True, min=5, max=25)
    safe_set('R_0_0', value=470e-6, vary=False, min=1e-9, max=1.0)  # sanity bounds
    safe_set('R_L_0', value=250e-6, vary=False, min=1e-9, max=1.0)
    safe_set('L_0', value=120e-9, vary=False, min=1e-12, max=1e-3)  # interpret as H -> 120 nH
    safe_set('n_mem_0', value=3.8, vary=False, min=1.0, max=10.0)
    safe_set('initE_0', value=2.4, vary=False, min=0.1, max=10.0)  # eV? check units

# --- Calculation wrappers ---
def calculate_avg_pulse(tes_model_instance, param_value_dict_or_params, tvals):
    # tes_model_instance.calc_pulse probably expects an lmfit.Parameters-like or dict-like object.
    pulse_tuple = tes_model_instance.calc_pulse(param_value_dict_or_params, tvals)
    # handle tuple or array
    if isinstance(pulse_tuple, tuple) or isinstance(pulse_tuple, list):
        return np.asarray(pulse_tuple[0])
    return np.asarray(pulse_tuple)

def calculate_noise(tes_model_instance, param_value_dict_or_params, fvals):
    noise_tuple = tes_model_instance.calc_noise(param_value_dict_or_params, fvals)
    if isinstance(noise_tuple, tuple) or isinstance(noise_tuple, list):
        return np.asarray(noise_tuple[0])
    return np.asarray(noise_tuple)

# --- Resampling ---
def logarithmic_resample(x, y, num_points):
    valid = np.logical_and(np.isfinite(x), np.isfinite(y), x > 0)
    if not np.any(valid):
        print("Warning: No positive/finite x values for logarithmic resampling.")
        return np.array([]), np.array([])
    x_valid = x[valid]
    y_valid = y[valid]
    if len(x_valid) < 2:
        print("Warning: Not enough valid points for logarithmic resampling.")
        return x_valid, y_valid
    log_start = np.log10(np.min(x_valid))
    log_end = np.log10(np.max(x_valid))
    if np.isclose(log_start, log_end):
        print("Warning: x range too small for logspace resampling.")
        return np.array([np.mean(x_valid)]), np.array([np.mean(y_valid)])
    x_resampled = np.logspace(log_start, log_end, num_points)
    interp = interp1d(x_valid, y_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
    y_resampled = interp(x_resampled)
    return x_resampled, y_resampled

# --- Residuals ---
def residual_pulse(params, tes_model_instance, tvals_pulse, pulse_data):
    Ites_vals = calculate_avg_pulse(tes_model_instance, params, tvals_pulse)
    return Ites_vals - pulse_data

def residual_noise(params, tes_model_instance, fvals_noise, noise_data):
    SI_total_s = calculate_noise(tes_model_instance, params, fvals_noise)
    return SI_total_s - noise_data

def residual_both(params, tes_model_instance, tvals_pulse, pulse_data, fvals_noise, noise_data):
    Ites_vals = calculate_avg_pulse(tes_model_instance, params, tvals_pulse)
    SI_total_s = calculate_noise(tes_model_instance, params, fvals_noise)
    pulse_resid = Ites_vals - pulse_data
    noise_resid = SI_total_s - noise_data
    # Optionally scale by data std to balance contributions:
    pulse_weight = 1.0 / np.std(pulse_data) if np.std(pulse_data) != 0 else 1.0
    noise_weight = 1.0 / np.std(noise_data) if np.std(noise_data) != 0 else 1.0
    return np.concatenate((pulse_resid * pulse_weight, noise_resid * noise_weight))

# === Main execution ===
if __name__ == "__main__":

    fit_mode = "noise"  # "pulse", "noise", "both"
    use_compound_model = False

    pulse_filename = '/data/20250226/0004/20250226_0004_chan4107_avgpulse_test.npz'
    noise_filename_bias = '/data/20250219/noise/noise_20250219_145355_20mK_bias0v25_.npz'
    noise_filename_zero = '/data/20250219/noise/noise_20250219_145647_20mK_bias0v0_.npz'
    noise_filename = noise_filename_bias
    squid_channel = 11

    tes = mdt3_core.MDT3_Core()
    tes_model = tes_intervening.TES_Intervening(tes)

    params = initialize_model(tes_model)

    # --- Load data ---
    pulse_data_raw, pulse_times_raw = load_avg_pulse(pulse_filename, pulse_arrival_sample=396)
    noise_psd_raw, noise_freqs_raw = load_noise_file(noise_filename)

    if pulse_data_raw is None and fit_mode in ["pulse", "both"]:
        raise ValueError(f"Pulse data could not be loaded from {pulse_filename}. Cannot perform {fit_mode} fit.")
    if noise_psd_raw is None and fit_mode in ["noise", "both"]:
        raise ValueError(f"Noise data could not be loaded from {noise_filename}. Cannot perform {fit_mode} fit.")

    # Prepare pulse
    tvals_pulse = None
    pulse_data = None
    if fit_mode in ["pulse", "both"] and pulse_data_raw is not None:
        tvals_pulse = np.linspace(0, pulse_times_raw[-1], 601)
        pulse_interp_func = interp1d(pulse_times_raw, pulse_data_raw, kind='linear', bounds_error=False, fill_value=0.0)
        pulse_data = pulse_interp_func(tvals_pulse)
        print(f"Pulse data prepared: {len(tvals_pulse)} points")

    # Prepare noise
    fvals_noise = None
    noise_data = None
    noise_data_resampled = None
    if fit_mode in ["noise", "both"] and noise_psd_raw is not None:
        # noise_psd_raw may be 2D (freq x channels) or dict-like; we attempt to pick a column.
        try:
            # If noise_psd_raw is dict-like mapping channel->array
            if isinstance(noise_psd_raw, dict):
                psd = noise_psd_raw.get(squid_channel, None)
                if psd is None:
                    # pick first available
                    first_key = next(iter(noise_psd_raw.keys()))
                    psd = noise_psd_raw[first_key]
            else:
                psd = np.asarray(noise_psd_raw)
            # handle 2D arrays (freq x channels)
            if psd.ndim == 2:
                if squid_channel >= psd.shape[1]:
                    raise IndexError(f"Noise PSD has {psd.shape[1]} columns; cannot access channel index {squid_channel}.")
                noise_data_raw_channel = phi0_to_amp(np.sqrt(psd[:, squid_channel]))
            else:
                # 1D PSD assumed single-channel
                noise_data_raw_channel = phi0_to_amp(np.sqrt(psd))
            # Frequencies: ensure shape matches

            # print(noise_freqs_raw)
            freqs = list(noise_freqs_raw.values())[0]
            positive = freqs > 0

            if not np.any(positive):
                raise ValueError("No positive frequencies in noise file.")
            fvals_noise, noise_data_resampled = logarithmic_resample(freqs[positive], noise_data_raw_channel[positive], 1200)
            noise_data = noise_data_resampled
            if len(fvals_noise) > 0:
                print(f"Noise data prepared: {len(fvals_noise)} points (log resampled)")
            else:
                print("Warning: Noise data preparation resulted in zero points.")
                if fit_mode == "noise":
                    raise ValueError("Cannot perform noise-only fit with no valid noise data points.")
                if fit_mode == "both":
                    print("Switching fit_mode to 'pulse' as noise data is empty.")
                    fit_mode = "pulse"
        except Exception as e:
            raise RuntimeError(f"Error preparing noise data: {e}")

    # --- Initialize parameters and model params ---
    params = initialize_model(tes)
    print("Initial Parameters:")
    try:
        params.pretty_print()
    except Exception:
        print("Parameters object not pretty-printable in this environment.")

    # --- Fit ---
    print(f"\n--- Starting Fit (mode: {fit_mode}) ---")
    out = None
    try:
        if fit_mode == "pulse" and tvals_pulse is not None:
            out = minimize(residual_pulse, params, args=(tes_model, tvals_pulse, pulse_data))
        elif fit_mode == "noise" and fvals_noise is not None:
            out = minimize(residual_noise, params, args=(tes_model, fvals_noise, noise_data))
        elif fit_mode == "both" and tvals_pulse is not None and fvals_noise is not None:
            out = minimize(residual_both, params, args=(tes_model, tvals_pulse, pulse_data, fvals_noise, noise_data))
        else:
            print("Warning: Could not perform fit. Check fit_mode and data availability.")
    except Exception as e:
        print(f"Exception while performing fit: {e}")

    # Check result robustly
    if out is not None and getattr(out, "success", False):
        print("\n--- Fit Successful ---")
        print(f"Fit Mode: {fit_mode}")
        print(f"Message: {getattr(out, 'message', '')}")
        print(f"Number of function evaluations: {getattr(out, 'nfev', 'N/A')}")
        print(f"Chi-squared: {getattr(out, 'chisqr', 'N/A')}")
        print(f"Reduced Chi-squared: {getattr(out, 'redchi', 'N/A')}")
        print("\nBest Fit Parameters:")
        try:
            out.params.pretty_print()
        except Exception:
            print("Couldn't pretty_print fitted params")

        # Attempt derived params
        try:
            tes_model.calc_derived_params(out.params)
            if hasattr(tes_model, 'simple_params'):
                print("\nDerived Parameters:")
                for k, v in tes_model.simple_params.items():
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"Could not calculate derived parameters: {e}")

        # Plotting
        num_plots = 0
        if fit_mode in ["pulse", "both"]: num_plots += 1
        if fit_mode in ["noise", "both"]: num_plots += 1

        if num_plots > 0:
            fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False)
            plot_idx = 0
            if fit_mode in ["pulse", "both"] and tvals_pulse is not None:
                ax = axes[0, plot_idx]
                ax.plot(tvals_pulse, pulse_data, '-', markersize=3, label='Pulse Data (Resampled)')
                fitted_pulse = calculate_avg_pulse(tes_model, out.params, tvals_pulse)
                ax.plot(tvals_pulse, fitted_pulse, '-', lw=2, label='Pulse Fit')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Current (A)')
                ax.set_title('Pulse Fit')
                ax.grid(True)
                ax.legend()
                plot_idx += 1
            if fit_mode in ["noise", "both"] and fvals_noise is not None:
                ax = axes[0, plot_idx]
                ax.loglog(fvals_noise, noise_data_resampled, '-', markersize=3, label='Noise Data (Resampled)')
                fitted_noise = calculate_noise(tes_model, out.params, fvals_noise)
                ax.loglog(fvals_noise, fitted_noise, '-', lw=2, label='Noise Fit')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel(r'Current Noise ($A/\sqrt{Hz}$)')
                ax.set_title('Noise Fit')
                ax.grid(True, which='both')
                ax.legend()
                plot_idx += 1
            plt.tight_layout()
            plt.show()
        else:
            print("No data available for plotting.")
    elif out is not None:
        print("\n--- Fit Completed but not successful ---")
        print(f"Success: {getattr(out, 'success', None)}")
        print(f"Message: {getattr(out, 'message', '')}")
    else:
        print("\n--- Fit Not Performed ---")
