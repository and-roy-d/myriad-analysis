# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
import threading
import datetime
import tkinter.font as tkFont # For font checking/setting
from ttkthemes import ThemedTk
from PIL import Image, ImageTk

# --- Scientific Imports ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import scipy.constants
from scipy.interpolate import interp1d
from lmfit import Parameters, minimize, fit_report # Only need Parameters from lmfit for this version

plt.rcParams._set("font.size", 14)
# --- MDT3 Imports ---
# Assuming mdt3 library is accessible (e.g., in the same directory or installed)
try:
    from mdt3 import mdt3_core
    from mdt3 import tes_simple
    # Import other models
    try:
        from mdt3 import tes_compound
        HAS_COMPOUND = True
    except ImportError:
        print("Warning: mdt3 tes_compound not found.")
        HAS_COMPOUND = False
    try:
        # Corrected import name based on user script
        from mdt3 import tes_intervening
        HAS_INTER = True
    except ImportError:
        print("Warning: mdt3 tes_intervening not found.")
        HAS_INTER = False
    try:
        from mdt3 import tes_dangling
        HAS_DANGLING = True
    except ImportError:
        print("Warning: mdt3 tes_dangling not found.")
        HAS_DANGLING = False

except ImportError as main_import_error:
    messagebox.showerror("Import Error", f"Could not import base mdt3 library: {main_import_error}\nPlease ensure mdt3_core.py and model files are accessible.")
    # Define dummy classes/functions if mdt3 is not available
    class DummyMDT3:
        def makeDefaultParamsDict(self, num_sets=1):
            return Parameters()
    class DummyTES:
        def __init__(self, core): pass
        def calc_pulse(self, params, tvals): return np.zeros_like(tvals),
        def calc_noise(self, params, fvals): return [np.zeros_like(fvals)]*6 # Return 6 noise arrays
        def calc_derived_params(self, params): self.simple_params = {}
    mdt3_core = DummyMDT3()

    tes_simple = DummyTES
    tes_compound = DummyTES
    tes_intervening = DummyTES
    tes_dangling = DummyTES

    HAS_COMPOUND = True
    HAS_INTER = True
    HAS_DANGLING = True


# === Constants and Configuration ===
phi0 = scipy.constants.value(u"mag. flux quantum")
MIN_SI_CONSTANT = 248e-12 # From user's PARAM_DEFS section
MIN_PHI0_PER_AMP = MIN_SI_CONSTANT / phi0 if phi0 != 0 else 1.0 # Avoid division by zero

PARAM_DEFS = {
    'G_tes_bath_0':   [ 0.093,         25e-3,   0.15,      False  ],
    'G_abs_tes_0':    [0.65,           0.1,     10,     True  ], #8,
    'G_tes_int_0':    [1.5,           0.1,     10.0,     False  ],
    'T_tes_0':        [ 0.053,        0.050,   0.055,    False ],
    'T_bath_0':       [ 0.021,        0.019,   0.023,    False ],
    'C_tes_0':        [ 0.16,         0.05,   0.5,      True  ],
    'C_abs_0':        [ .05,          0.02,    0.2,      False  ],
    'C_int_0':        [ 0.16,        0.01,    1,     False ],
    'alpha_I_0':      [ 1350,        100.0,    2500.0,   True  ],
    'beta_I_0':       [ 65.0,         1.0,     100.0,     True  ],
    'R_0_0':          [ 470e-6,       0.1e-6,   500e-6,  False ],
    'R_L_0':          [ 250e-6,       200e-6,  300e-6,   False ],
    'L_0':            [ 120,       80,   200,   False  ],
    'n_mem_0':        [ 3.8,          3.5,     4.0,      False ],
    'M_0':            [ 2,          0.0,     10,      True ],
    'initE_0':        [ 2.4,  0.0,     5.0, False ],
    'squid_noise_0':  [ 2.0e-11,      1.5e-11,   4e-11,    False ],
    'xi_0':           [ 0.0,          0.0,     1.0,      False ],
    'background_noise_0': [0.0, 0.0, 1e-9, False],
    'abs_background_noise_0': [0.0, 0.0, 1e-9, False], # Added based on noise map
    'n_int1_0':       [ 4.0,          2.0,     5.0,      False ], # Example value
    'n_int2_0':       [ 3.0,          2.0,     5.0,      False ], # Example value
    'k_int_ratio_0':  [ 2.0,          0.1,     10.0,     False ], # Example value
    'int_background_power_0': [0.0,   0.0,     1.0,      False ], # Example value

}
UNITS = {
    # Conductances
    'G_tes_bath_0': 'nW/K', 'G_abs_tes_0': 'nW/K', 'G_tes_int_0': 'nW/K',
    # Temperatures
    'T_tes_0': 'K', 'T_bath_0': 'K',
    # Heat Capacities
    'C_tes_0': 'pJ/K', 'C_abs_0': 'pJ/K', 'C_int_0': 'pJ/K',
    # Dimensionless Ratios / Exponents
    'alpha_I_0': '', 'beta_I_0': '', 'n_mem_0': '', 'n_int1_0': '', 'n_int2_0': '',
    'M_0': '', 'k_int_ratio_0': '', 'xi_0': '',
    # Resistances
    'R_0_0': 'Ohm', 'R_L_0': 'Ohm',
    # Inductance
    'L_0': 'nH',
    # Energy
    'initE_0': 'eV',
    'squid_noise_0': 'A/sqrt Hz',
    'background_noise_0': 'A/sqrt Hz',
    'abs_background_noise_0': 'A/sqrt Hz',
    # Other
    'int_background_power_0': 'N/A',
}

# Noise component mapping (adjust if Intervening model has different output)
NOISE_COMPONENT_MAP = {
    tes_simple.TES_Simple:   ['total', 'r0', 'rl', 'g1', 'squid', 'back'],
    tes_compound.TES_Compound: ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back'],
    tes_dangling.TES_Dangling: ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back'],
    tes_intervening.TES_Intervening: ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back'], # ASSUMING same as Compound/Dangling
}
# Define all possible keys that might exist across all models
ALL_POSSIBLE_NOISE_KEYS = ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back']
# Map keys to plot labels and colors/styles
NOISE_PLOT_CONFIG = {
    'total':    {'label': 'Total Model', 'color': 'red', 'linestyle': '-', 'linewidth': 2},
    'r0':       {'label': '$SI_{R_0}$', 'color': 'darkorange', 'linestyle': '--', 'linewidth': 1},
    'rl':       {'label': '$SI_{R_L}$', 'color': 'forestgreen', 'linestyle': '--', 'linewidth': 1},
    'g1':       {'label': '$SI_{G1}$', 'color': 'purple', 'linestyle': '--', 'linewidth': 1},
    'g2':       {'label': '$SI_{G2}$', 'color': 'brown', 'linestyle': '--', 'linewidth': 1},
    'squid':    {'label': '$SI_{SQUID}$', 'color': 'cyan', 'linestyle': '--', 'linewidth': 1},
    'back':     {'label': '$SI_{Back}$', 'color': 'magenta', 'linestyle': '--', 'linewidth': 1},
    'abs_back': {'label': '$SI_{AbsBack}$', 'color': 'gold', 'linestyle': '--', 'linewidth': 1},
}

# === Helper Functions ===
def phi0_to_amp(inval):
    """Converts flux quantum values to amplitude."""
    if MIN_PHI0_PER_AMP == 0: return np.zeros_like(inval)
    return inval * (1.0 / MIN_PHI0_PER_AMP)

def load_noise_file(filename):
    """Loads multi-channel noise data from NPZ file."""
    try:
        with np.load(filename, allow_pickle=True) as f:
            psd_key = 'Pxx'; freq_key = 'f'
            if psd_key not in f or freq_key not in f: messagebox.showerror("File Error", f"Keys '{psd_key}' or '{freq_key}' not found."); return None, None
            psd_data = f[psd_key]; freq_data = f[freq_key]
            psd_item = psd_data.item() if hasattr(psd_data, 'item') else psd_data; freq_item = freq_data.item() if hasattr(freq_data, 'item') else freq_data
            if isinstance(psd_item, dict): key = list(psd_item.keys())[0]; psd = psd_item[key]; freqs = freq_item[key]
            else: psd = psd_item; freqs = freq_item
            return freqs, psd
    except Exception as e: messagebox.showerror("File Error", f"Error loading noise file {filename}:\n{e}"); return None, None

def load_avg_pulse(filename, pulse_arrival_sample=0):
    """Loads average pulse data from NPZ file."""
    try:
        with np.load(filename, allow_pickle=True) as f:
            pulse_key = 'array1';
            if pulse_key not in f: messagebox.showerror("File Error", f"Key '{pulse_key}' not found."); return None, None
            avg_pulse_raw = f[pulse_key] / 4096.0; start_idx = 1; end_idx = min(400, len(avg_pulse_raw) - 1)
            if start_idx < end_idx: baseline = np.mean(avg_pulse_raw[start_idx:end_idx])
            elif len(avg_pulse_raw) > 0: baseline = avg_pulse_raw[0]
            else: baseline = 0
            avg_pulse_baselined = avg_pulse_raw - baseline; sample_time = 8e-6
            times = (np.arange(len(avg_pulse_baselined)) - pulse_arrival_sample) * sample_time
            pulse_amps = phi0_to_amp(avg_pulse_baselined); return times, pulse_amps
    except Exception as e: messagebox.showerror("File Error", f"Error loading pulse file {filename}:\n{e}"); return None, None

def logarithmic_resample(x, y, num_points=1200):
    """Resamples y(x) logarithmically in x."""
    valid_indices = (x > 0) & np.isfinite(x) & np.isfinite(y);
    if not np.any(valid_indices): return np.array([]), np.array([])
    x_valid, y_valid = x[valid_indices], y[valid_indices]
    if len(x_valid) < 2: return x_valid, y_valid
    try:
        log_start = np.log10(np.maximum(np.min(x_valid), 1e-12)); log_end = np.log10(np.max(x_valid))
        if np.isclose(log_start, log_end) or not np.isfinite(log_start) or not np.isfinite(log_end): return x_valid[:num_points], y_valid[:num_points]
        x_resampled = np.logspace(log_start, log_end, num_points); interpolation_function = interp1d(x_valid, y_valid, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_resampled = interpolation_function(x_resampled); return x_resampled, y_resampled
    except Exception as e: print(f"Warning: Log resampling failed - {e}"); return x_valid[:num_points], y_valid[:num_points]


# === Main GUI Application Class ===
class TESModelViewer(ThemedTk): # Inherit directly from tk.Tk
    def __init__(self):
        super().__init__() # Initialize the Tk part first

        # --- Apply Theme FIRST ---
        self.line_noise_data = None
        self.line_pulse_data = None
        self.set_theme("arc") # Specify theme name here ('arc', 'sv-ttk-light', 'sv-ttk-dark')

        # --- End Theme ---

        self.title("TES Model Viewer and Fitter (mdt3)")
        self.geometry("2540x1280")

        # --- Default File Paths ---
        self.default_pulse_path = '/data/20250226/0004/20250226_0004_chan4107_avgpulse_test.npz'
        self.default_noise_path = '/data/20250219/noise/noise_20250219_145355_20mK_bias0v25_.npz'

        # --- Data Storage ---
        initial_pulse_display = os.path.basename(self.default_pulse_path) if os.path.exists(self.default_pulse_path) else "..."
        initial_noise_display = os.path.basename(self.default_noise_path) if os.path.exists(self.default_noise_path) else "..."
        self.pulse_filename_var = tk.StringVar(value=initial_pulse_display); self.noise_filename_var = tk.StringVar(value=initial_noise_display)
        self.current_pulse_filepath = self.default_pulse_path if os.path.exists(self.default_pulse_path) else None
        self.current_noise_filepath = self.default_noise_path if os.path.exists(self.default_noise_path) else None
        self.pulse_times_raw, self.pulse_data_raw = None, None; self.raw_noise_freqs = None; self.raw_noise_psd = None
        self.pulse_times_interp, self.pulse_data_interp = None, None; self.noise_freqs_resampled, self.noise_data_resampled = None, None
        self.noise_model_arrays = {key: None for key in ALL_POSSIBLE_NOISE_KEYS}

        # --- Model Selection ---
        self.available_models = {"Simple": tes_simple.TES_Simple}
        if HAS_COMPOUND: self.available_models["Compound"] = tes_compound.TES_Compound
        if HAS_INTER: self.available_models["Inter"] = tes_intervening.TES_Intervening
        if HAS_DANGLING: self.available_models["Dangling"] = tes_dangling.TES_Dangling
        self.selected_model_name = tk.StringVar(value="Dangling")

        # --- Channel Selection ---
        self.selected_noise_channel = tk.IntVar(value=-1)

        # --- Fit Status ---
        self.fit_status_text = tk.StringVar(value="Fit status: Idle")
        if not isinstance(self.fit_status_text, tk.StringVar):
            print("CRITICAL ERROR: self.fit_status_text was not created correctly!")
        self.noise_weight_factor = tk.DoubleVar(value=1.0)
        self.selected_fit_method = tk.StringVar(value='least_squares')
        self.de_max_nfev_var = tk.IntVar(value=2000)
        self.fit_button = None
        self.noise_only_fit_button = None
        self.log_text = None
        self.log_scroll = None
        self.last_live_update_time = 0.0
        self.live_update_interval = 0.25
        self._terminate_event = threading.Event()
        self.last_fit_result = None
        self.fit_method_combo = None
        self.de_max_nfev_spinbox = None

        self.energy_res_text = tk.StringVar(value="")  # Variable for the label
        self.energy_res_label = None

        # --- Model and Parameter Initialization ---
        self.tes_model = None; self.params = None; self.mdt3_core = None
        try:
            self.mdt3_core = mdt3_core.MDT3_Core()
            InitialModelClass = self.available_models[self.selected_model_name.get()]
            self.tes_model = InitialModelClass(self.mdt3_core)
            self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
            self.initialize_params() # Apply PARAM_DEFS defaults
        except Exception as e: messagebox.showerror("Model Init Error", f"Failed to initialize mdt3 model:\n{e}"); self.quit()

        # self.selected_fit_method = 'diff_ev_then_leastsq'  # Store default method name
        # self.de_max_nfev = 2000  # Store default NFEV value
        # self.fit_method_combo = None  # Initialize widget reference to None


        # --- GUI Setup ---
        self.param_widgets = {}; self.noise_channel_combo = None; self.model_selector_combo = None
        self.noise_plot_lines = {}
        self.setup_gui() # Creates widgets that *use* fit_status_text

        # --- Attempt Auto-Load ---
        if self.current_pulse_filepath:
            self._load_pulse_data(self.current_pulse_filepath)
        if self.current_noise_filepath:
            self._load_noise_data(self.current_noise_filepath)
        self.update_model_and_plots()

    def initialize_params(self):
        """Sets GUI-defined overrides (value, min, max, vary) from PARAM_DEFS
           onto the existing self.params object created by makeDefaultParamsDict."""
        if self.params is None:
            print("Error: Parameters object not initialized.")
            return

        params_in_model_not_in_defs = []
        # Iterate through params created by makeDefault...
        for name, param in self.params.items():
            if name in PARAM_DEFS:
                # Apply overrides from GUI definition
                initial_gui, pmin_gui, pmax_gui, vary_gui = PARAM_DEFS[name]

                # Use GUI-defined bounds from PARAM_DEFS for sliders/GUI limits
                pmin_actual = pmin_gui
                pmax_actual = pmax_gui
                # Ensure GUI bounds are valid
                if pmin_actual is not None and pmax_actual is not None and pmin_actual > pmax_actual:
                    print(f"Warning: Invalid min/max in PARAM_DEFS for '{name}'. Using wide bounds for set().")
                    pmin_actual = -np.inf;
                    pmax_actual = np.inf

                # Clip the initial value specified in PARAM_DEFS to the GUI bounds
                initial_clipped = np.clip(initial_gui,
                                          pmin_actual if pmin_actual is not None else -np.inf,
                                          pmax_actual if pmax_actual is not None else np.inf)

                try:
                    # Set the parameter using values/settings from PARAM_DEFS
                    # This overrides the defaults set by makeDefaultParamsDict
                    param.set(value=initial_clipped, min=pmin_actual, max=pmax_actual, vary=vary_gui)
                    # print(f"  Set '{name}': val={param.value:.2e}, min={param.min:.2e}, max={param.max:.2e}, vary={param.vary}") # Optional debug
                except ValueError as ve:
                    print(f"Error setting parameter '{name}': {ve}. Check PARAM_DEFS value/min/max.")
                    # Fallback? Maybe just set value/vary?
                    try:
                        param.set(value=initial_clipped, vary=vary_gui)
                    except:
                        pass

            else:
                # Param exists in model but not GUI defs (e.g., t_0, f_0)
                # Ensure it doesn't vary by default for fitting
                param.set(vary=False)
                params_in_model_not_in_defs.append(name)

        if params_in_model_not_in_defs:
            print(f"Info: Params in model but not PARAM_DEFS (set vary=False): {params_in_model_not_in_defs}")

        # Sanity check (should be empty now): Params in PARAM_DEFS but missing from self.params
        missing_from_model = [name for name in PARAM_DEFS if name not in self.params]
        if missing_from_model:
            # This indicates makeDefaultParamsDict is still missing things defined in PARAM_DEFS
            print(
                f"ERROR: Params in PARAM_DEFS but still missing from self.params after makeDefault: {missing_from_model}")


    def restore_default_params(self):
        """Resets parameters to their initial default values defined in PARAM_DEFS."""
        self.energy_res_text.set("")
        if self.params is None or self.tes_model is None:
            messagebox.showwarning("Restore Defaults", "Model or parameters not initialized yet.")
            return

        current_model = self.selected_model_name.get()  # Get name for message
        print(f"Restoring default parameters for model '{current_model}' based on PARAM_DEFS...")
        self._update_log(f"--- Restoring PARAM_DEFS defaults for {current_model} ---")

        try:
            # 1. Re-apply the initial values, bounds, and vary status from PARAM_DEFS
            #    using the existing initialize_params method. This method iterates
            #    through PARAM_DEFS and calls param.set() on the existing self.params object.
            #    Make sure initialize_params is the version that *overrides* params
            #    based on PARAM_DEFS, assuming makeDefaultParamsDict already
            #    created all necessary parameter objects (as discussed previously).
            self.initialize_params()

            # 2. Update the GUI elements (sliders, checkboxes, value labels)
            #    to visually reflect the newly reset parameter values.
            self.update_sliders_from_params()

            # 3. Update the plots to show the model calculated with default parameters.
            self.update_model_and_plots()

            self.fit_status_text.set("Fit status: Defaults Restored.")  # Update status bar
            print("Default parameters restored successfully.")

        except Exception as e:
            print(f"Error restoring default parameters: {e}")
            import traceback;
            traceback.print_exc()  # Print full error for debugging
            messagebox.showerror("Restore Error", f"Could not restore default parameters:\n{e}")
            self.fit_status_text.set("Status: Error restoring defaults.")

    def setup_gui(self):
        """Creates the main GUI layout with sliders/diagram side-by-side."""
        # Configure root window resizing behavior (optional but good)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # Allow plot frame row to expand
        self.rowconfigure(2, weight=0)  # Mid frame takes natural height
        self.rowconfigure(3, weight=0)  # Bottom controls take natural height

        # --- Top Control Frame (Files, Selectors) ---
        control_frame = ttk.Frame(self, padding="5")
        control_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=(5, 0))  # Use grid for top-level frames
        control_frame.columnconfigure(1, weight=1)  # Allow labels to expand
        control_frame.columnconfigure(3, weight=1)

        # Row 1: File Loaders
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, columnspan=4, sticky='ew')
        ttk.Button(file_frame, text="Load Pulse File (.npz)", command=self.select_pulse_file).pack(side=tk.LEFT, padx=5,
                                                                                                   pady=2)
        # Ensure self.pulse_filename_var is initialized in __init__
        ttk.Label(file_frame, textvariable=self.pulse_filename_var, relief=tk.SUNKEN, width=30).pack(side=tk.LEFT,
                                                                                                     padx=5, pady=2,
                                                                                                     fill=tk.X,
                                                                                                     expand=True)
        ttk.Button(file_frame, text="Load Noise File (.npz)", command=self.select_noise_file).pack(side=tk.LEFT, padx=5,
                                                                                                   pady=2)
        # Ensure self.noise_filename_var is initialized in __init__
        ttk.Label(file_frame, textvariable=self.noise_filename_var, relief=tk.SUNKEN, width=30).pack(side=tk.LEFT,
                                                                                                     padx=5, pady=2,
                                                                                                     fill=tk.X,
                                                                                                     expand=True)

        # Row 2: Model and Channel Selectors
        selector_frame = ttk.Frame(control_frame)
        selector_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(5, 0))
        ttk.Label(selector_frame, text="TES Model:").pack(side=tk.LEFT, padx=(5, 2));
        # Ensure self.model_selector_combo and self.selected_model_name are initialized in __init__/setup
        self.model_selector_combo = ttk.Combobox(selector_frame, textvariable=self.selected_model_name,
                                                 values=list(self.available_models.keys()), state='readonly', width=12);
        self.model_selector_combo.pack(side=tk.LEFT, padx=(0, 15));
        self.model_selector_combo.bind('<<ComboboxSelected>>', self.on_model_selected)

        ttk.Label(selector_frame, text="Autotune Channel:").pack(side=tk.LEFT, padx=(5, 2))
        # Ensure self.noise_channel_combo and self.selected_noise_channel are initialized
        self.noise_channel_combo = ttk.Combobox(selector_frame, textvariable=self.selected_noise_channel,
                                                state='disabled', width=5);
        self.noise_channel_combo.pack(side=tk.LEFT, padx=(0, 5));
        self.noise_channel_combo.bind('<<ComboboxSelected>>', self.on_channel_selected)

        self.energy_res_label = ttk.Label(
            control_frame,
            textvariable=self.energy_res_text,
            foreground="red",  # Red text
            font=('TkDefaultFont', 10, 'bold'),  # Make it bold
            anchor='center'  # Center the text
        )
        # Place it in the grid below selectors
        self.energy_res_label.grid(row=2, column=0, columnspan=4, sticky='ew', pady=(5, 2))

        # --- Plot Frame (Takes up expanding middle space) ---
        plot_frame = ttk.Frame(self)
        # Make plot frame expand in the main grid layout
        plot_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(10, 4), dpi=100)  # Adjusted figsize slightly
        self.ax_pulse = self.fig.add_subplot(1, 2, 1)
        self.ax_noise = self.fig.add_subplot(1, 2, 2)

        # Create plot lines (ensure these attributes are initialized e.g., in __init__)
        self.line_pulse_data, = self.ax_pulse.plot([], [], 'b-', label='Pulse Data', markersize=3, alpha=0.7)
        self.line_pulse_model, = self.ax_pulse.plot([], [], 'r-', label='Pulse Model', linewidth=2)
        self.ax_pulse.set_xlabel("Time (ms)");
        self.ax_pulse.set_ylabel("Current (A)")
        self.ax_pulse.set_title("Average Pulse");
        self.ax_pulse.grid(True);
        self.ax_pulse.legend()

        self.line_noise_data, = self.ax_noise.plot([], [], 'b-', label='Noise Data', markersize=3, alpha=0.7)
        self.noise_plot_lines = {}  # Ensure initialized in __init__
        for key, config in NOISE_PLOT_CONFIG.items():
            line, = self.ax_noise.plot([], [], **config);
            self.noise_plot_lines[key] = line
        self.ax_noise.set_xlabel("Frequency (Hz)");
        self.ax_noise.set_ylabel(r"Current Noise (A/$\sqrt{Hz}$)")
        self.ax_noise.set_title("Noise Spectrum");
        self.ax_noise.set_yscale('log');
        self.ax_noise.set_xscale('log')
        self.ax_noise.grid(True, which='both');
        self.ax_noise.legend()

        # Canvas and Toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        # Use grid within plot_frame
        self.canvas_widget.grid(row=0, column=0, sticky='nsew')

        toolbar_frame = ttk.Frame(plot_frame)  # Frame for toolbar
        toolbar_frame.grid(row=1, column=0, sticky='ew')
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update();
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Middle Frame (Sliders LEFT, Diagram RIGHT) ---
        mid_frame = ttk.Frame(self, padding="5")
        # Use grid for better control with plot frame above
        mid_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=(0, 5))
        mid_frame.columnconfigure(0, weight=1)  # Allow slider area to expand horizontally
        mid_frame.columnconfigure(1, weight=0)  # Diagram fixed width (determined by content)

        # Slider Frame (Scrollable area on the LEFT of mid_frame)
        slider_outer_frame = ttk.Frame(mid_frame)
        # Use grid within mid_frame
        slider_outer_frame.grid(row=0, column=0, sticky='nsew')
        slider_outer_frame.rowconfigure(0, weight=1)  # Allow canvas to expand vertically
        slider_outer_frame.columnconfigure(0, weight=1)  # Allow canvas to expand horizontally

        # Set a height for the scrollable area (important!)
        slider_canvas = tk.Canvas(slider_outer_frame, height=200)  # Set fixed height here
        slider_scrollbar = ttk.Scrollbar(slider_outer_frame, orient="vertical", command=slider_canvas.yview)
        self.slider_frame = ttk.Frame(slider_canvas)  # Frame for grid goes inside canvas
        self.slider_frame.bind("<Configure>", lambda e: slider_canvas.configure(scrollregion=slider_canvas.bbox("all")))
        slider_canvas.create_window((0, 0), window=self.slider_frame, anchor="nw");
        slider_canvas.configure(yscrollcommand=slider_scrollbar.set)

        # Use grid for canvas/scrollbar within slider_outer_frame
        slider_canvas.grid(row=0, column=0, sticky='nsew')
        slider_scrollbar.grid(row=0, column=1, sticky='ns')

        self.create_sliders()  # Populates self.slider_frame with grid layout

        # Diagram Label (On the RIGHT of mid_frame)
        # Ensure self.diagram_label is initialized in __init__
        self.diagram_label = ttk.Label(mid_frame, text="[Model Diagram]", anchor='ne')
        # Use grid within mid_frame
        self.diagram_label.grid(row=0, column=1, sticky='ne', padx=(10, 0), pady=5)

        # --- Bottom Controls Frame (Fit, Weight, Log packed vertically at window bottom) ---
        bottom_controls_frame = ttk.Frame(self, padding="5")
        # Use grid for the final bottom section
        bottom_controls_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=(5, 5))
        bottom_controls_frame.columnconfigure(0, weight=1)  # Allow content to expand horizontally

        # Fit Control Frame (Inside bottom_controls_frame)
        fit_control_frame = ttk.Frame(bottom_controls_frame)
        # Use grid within bottom_controls_frame
        fit_control_frame.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        # Pack widgets inside this frame using pack(side=LEFT) as before
        self.fit_button = ttk.Button(fit_control_frame, text="Fit Current Model", command=self.start_fit_thread);
        self.fit_button.pack(side=tk.LEFT, padx=5)

        self.noise_only_fit_button = ttk.Button(fit_control_frame, text="Fit Noise Only",
                                                command=self.start_noise_only_fit_thread)
        self.noise_only_fit_button.pack(side=tk.LEFT, padx=5)

        self.terminate_button = ttk.Button(fit_control_frame, text="Terminate Fit",
                                           command=self.request_fit_termination, state='disabled');
        self.terminate_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(fit_control_frame, text="Save Fit Report", command=self.save_fit_report,
                                      state='disabled');
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.restore_button = ttk.Button(fit_control_frame, text="Restore Defaults",
                                         command=self.restore_default_params);
        self.restore_button.pack(side=tk.LEFT, padx=(10, 5))
        ttk.Label(fit_control_frame, text=" Method:").pack(side=tk.LEFT, padx=(10, 2))
        fit_methods = ['diff_ev_then_leastsq', 'least_squares', 'leastsq', 'nelder', 'lbfgsb', 'powell', 'cg', 'bfgs',
                       'slsqp', 'differential_evolution', 'basinhopping']
        self.fit_method_combo = ttk.Combobox(fit_control_frame, textvariable=self.selected_fit_method,
                                             values=fit_methods, state='readonly', width=18);
        self.fit_method_combo.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(fit_control_frame, text=" DE MaxNFEV:").pack(side=tk.LEFT, padx=(0, 2))
        self.de_max_nfev_spinbox = ttk.Spinbox(fit_control_frame, from_=100, to=100000, increment=100,
                                               textvariable=self.de_max_nfev_var, width=7);
        self.de_max_nfev_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(fit_control_frame, textvariable=self.fit_status_text, relief=tk.SUNKEN, anchor='w').pack(side=tk.LEFT,
                                                                                                           padx=5,
                                                                                                           fill=tk.X,
                                                                                                           expand=True)

        # Noise Weight Slider Frame (Inside bottom_controls_frame)
        weight_frame = ttk.Frame(bottom_controls_frame)
        weight_frame.grid(row=1, column=0, sticky='ew', pady=(0, 5))
        # Pack widgets inside this frame using pack(side=LEFT)
        ttk.Label(weight_frame, text="Noise Weight Factor:").pack(side=tk.LEFT, padx=(5, 2))
        self.noise_weight_label = ttk.Label(weight_frame, text=f"{self.noise_weight_factor.get():.2f}", width=5,
                                            anchor='w')
        weight_slider = ttk.Scale(weight_frame, from_=0.0, to=10.0, orient=tk.HORIZONTAL,
                                  variable=self.noise_weight_factor, command=self._update_noise_weight_label);
        weight_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.noise_weight_label.pack(side=tk.LEFT, padx=(0, 5))

        # Log Frame (Inside bottom_controls_frame)
        log_frame = ttk.LabelFrame(bottom_controls_frame, text="Fit Log", padding="5")
        log_frame.grid(row=2, column=0, sticky='ew')  # Span columns if needed later
        log_frame.columnconfigure(0, weight=1)  # Allow text widget to expand
        log_frame.rowconfigure(0, weight=1)  # Allow text widget to expand (optional height)
        # Pack widgets inside this frame using pack or grid
        self.log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        self.log_text = tk.Text(log_frame, height=8, wrap='word', yscrollcommand=self.log_scroll.set, relief=tk.SUNKEN,
                                borderwidth=1)  # Reduced height slightly
        self.log_scroll.config(command=self.log_text.yview)
        # Use pack or grid for log_text/scrollbar inside log_frame
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def load_model_diagram(self, model_name):
        """Loads and displays the corresponding diagram image for the selected model."""
        image_path = None # Initialize for error messages
        try:
            # Construct path relative to the script's location
            script_dir = os.path.dirname(__file__)
            # Assumes 'mdt3/examples/' is one level up from the script's directory
            # Adjust "../mdt3/examples/" if your structure is different
            base_path = os.path.abspath(os.path.join(script_dir, "/home/pcuser/caldaq/src/microcal_design_tools_3/mdt3/examples"))
            image_filename = f"{model_name}Microcal.png"
            image_path = os.path.join(base_path, image_filename)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at calculated path: {image_path}")

            img = Image.open(image_path)

            # Define desired size (adjust as needed)
            target_size = (200, 200)


            resample_filter = Image.Resampling.LANCZOS

            img = img.resize(target_size, resample=resample_filter)
            # --- END FIX ---

            img_tk = ImageTk.PhotoImage(img)

            self.diagram_label.config(image=img_tk, text="") # Update image, clear any placeholder text
            self.diagram_label.image = img_tk

        except FileNotFoundError:
            print(f"Warning: Diagram image not found at '{image_path}'")
            if self.diagram_label:
                 # Clear image and show placeholder text if file not found
                 self.diagram_label.config(image="", text="[Diagram N/A]") # Clear image
                 self.diagram_label.image = None # Clear reference
        except AttributeError as ae:
             if 'Resampling' in str(ae):
                 print("ERROR: Pillow version might be too old for Image.Resampling. Try using Image.LANCZOS instead of Image.Resampling.LANCZOS.")
                 if self.diagram_label: self.diagram_label.config(image="", text="[Pillow Error]"); self.diagram_label.image = None
             else:
                 print(f"ERROR loading/processing image '{image_path}': {ae}")
                 import traceback; traceback.print_exc()
                 if self.diagram_label: self.diagram_label.config(image="", text="[Img Attr Error]"); self.diagram_label.image = None
        except Exception as e:
            # Catch other potential errors (Pillow not installed, bad image file, etc.)
            print(f"ERROR loading/processing image '{image_path}': {e}")
            import traceback; traceback.print_exc() # Print full traceback
            if self.diagram_label:
                 # Clear image and show error text
                 self.diagram_label.config(image="", text="[Error Load Image]")
                 self.diagram_label.image = None # Clear reference

    # --- Internal Loading Methods ---
    def _load_pulse_data(self, filepath):
        if not filepath or not os.path.exists(filepath): self.current_pulse_filepath = None; self.pulse_filename_var.set("Pulse file not found"); self.pulse_times_raw, self.pulse_data_raw = None, None; self.pulse_times_interp, self.pulse_data_interp = None, None; return False
        pulse_arrival = 396;
        times, data = load_avg_pulse(filepath, pulse_arrival_sample=pulse_arrival)
        if times is None or data is None: self.current_pulse_filepath = None; self.pulse_filename_var.set("Failed to load pulse"); return False
        self.current_pulse_filepath = filepath; self.pulse_filename_var.set(os.path.basename(filepath)); self.pulse_times_raw, self.pulse_data_raw = times, data
        num_interp_points = 601
        if len(times) > 1:
            t_min, t_max = times[0], times[-1];
            if t_max <= t_min: print(f"Warning: Invalid time range ({t_min} to {t_max}). Skipping pulse interpolation."); self.pulse_times_interp, self.pulse_data_interp = None, None
            else:
                 self.pulse_times_interp = np.linspace(t_min, t_max, num_interp_points)
                 try: interp_func = interp1d(times, data, kind='linear', bounds_error=False, fill_value=0); self.pulse_data_interp = interp_func(self.pulse_times_interp)
                 except ValueError as e: print(f"Error during pulse interpolation: {e}"); self.pulse_times_interp, self.pulse_data_interp = None, None
        elif len(times) == 1: self.pulse_times_interp, self.pulse_data_interp = times, data
        else: self.pulse_times_interp, self.pulse_data_interp = None, None
        print(f"Pulse data loaded from {filepath}."); return True

    def _load_noise_data(self, filepath):
        if not filepath or not os.path.exists(filepath):
             self.current_noise_filepath = None; self.noise_filename_var.set("Noise file not found"); self.raw_noise_freqs, self.raw_noise_psd = None, None
             if self.noise_channel_combo: self.noise_channel_combo.config(values=[], state='disabled')
             self.selected_noise_channel.set(-1); self.noise_freqs_resampled, self.noise_data_resampled = None, None; return False
        freqs, psd_all_channels = load_noise_file(filepath)
        if freqs is None or psd_all_channels is None:
            self.current_noise_filepath = None; self.noise_filename_var.set("Failed to load noise"); self.raw_noise_freqs, self.raw_noise_psd = None, None
            if self.noise_channel_combo: self.noise_channel_combo.config(values=[], state='disabled')
            self.selected_noise_channel.set(-1); self.noise_freqs_resampled, self.noise_data_resampled = None, None; return False
        self.current_noise_filepath = filepath; self.noise_filename_var.set(os.path.basename(filepath))
        self.raw_noise_freqs = freqs; self.raw_noise_psd = psd_all_channels; print(f"Raw noise data loaded from {filepath}.")
        num_channels = 0
        if self.raw_noise_psd.ndim == 1: num_channels = 1
        elif self.raw_noise_psd.ndim > 1: num_channels = self.raw_noise_psd.shape[1]
        processed_ok = False
        if num_channels > 0 and self.noise_channel_combo:
            channel_indices = list(range(num_channels)); self.noise_channel_combo.config(values=channel_indices, state='readonly')
            initial_channel = 11 if 11 in channel_indices else 0
            self.noise_channel_combo.current(initial_channel); self.selected_noise_channel.set(initial_channel)
            print(f"Noise channels available: {num_channels}. Set to channel {initial_channel}.")
            processed_ok = self._process_noise_channel(initial_channel)
        elif self.noise_channel_combo:
             self.noise_channel_combo.config(values=[], state='disabled'); self.selected_noise_channel.set(-1)
             print("Warning: Could not determine noise channels."); self.noise_freqs_resampled, self.noise_data_resampled = None, None
        return processed_ok

    def _process_noise_channel(self, channel_index):
        if self.raw_noise_psd is None or self.raw_noise_freqs is None or channel_index < 0: print("Cannot process noise channel: Raw data missing or invalid channel index."); self.noise_freqs_resampled, self.noise_data_resampled = None, None; return False
        print(f"Processing noise data for channel {channel_index}...")
        psd_single_channel = None
        try:
            if self.raw_noise_psd.ndim == 1 and channel_index == 0: psd_single_channel = self.raw_noise_psd
            elif self.raw_noise_psd.ndim > 1 and channel_index < self.raw_noise_psd.shape[1]: psd_single_channel = self.raw_noise_psd[:, channel_index]
            else: raise IndexError
            with np.errstate(invalid='ignore'): amp_noise_density = phi0_to_amp(np.sqrt(psd_single_channel))
            amp_noise_density = np.nan_to_num(amp_noise_density, nan=0.0)
            num_resample_points = 1200;
            self.noise_freqs_resampled, self.noise_data_resampled = logarithmic_resample(self.raw_noise_freqs, amp_noise_density, num_points=num_resample_points)
            if self.noise_data_resampled is not None and not np.any(self.noise_data_resampled > 0): print(f"Warning: Processed noise data for channel {channel_index} contains no positive values.")
            print(f"Finished processing channel {channel_index}."); return True
        except IndexError: messagebox.showerror("Channel Error", f"Invalid channel index {channel_index}"); self.noise_freqs_resampled, self.noise_data_resampled = None, None; return False
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing noise channel {channel_index}:\n{e}");
            self.noise_freqs_resampled, self.noise_data_resampled = None, None;
            return False

    def _update_noise_weight_label(self, value_str):
        """Updates the label displaying the noise weight factor."""
        # The command passes the value as a string, get precise value from variable
        if self.noise_weight_label:
            try:
                current_value = self.noise_weight_factor.get()
                self.noise_weight_label.config(text=f"{current_value:.2f}")
            except tk.TclError:
                pass  # Ignore if GUI closing

    # --- Button Callbacks ---
    def select_pulse_file(self):
        filepath = filedialog.askopenfilename( title="Select Pulse NPZ File", filetypes=[("Numpy NPZ files", "*.npz"), ("All files", "*.*")], initialdir=os.path.dirname(self.current_pulse_filepath) if self.current_pulse_filepath else os.getcwd())
        if not filepath: return
        if self._load_pulse_data(filepath): self.update_model_and_plots()
    def select_noise_file(self):
        filepath = filedialog.askopenfilename( title="Select Noise NPZ File", filetypes=[("Numpy NPZ files", "*.npz"), ("All files", "*.*")], initialdir=os.path.dirname(self.current_noise_filepath) if self.current_noise_filepath else os.getcwd())
        if not filepath: return
        if self._load_noise_data(filepath): self.update_model_and_plots()

    # --- Combobox Callbacks ---
    def on_channel_selected(self, event=None):
        try:
            new_channel = self.selected_noise_channel.get()
            if new_channel >= 0:
                if self._process_noise_channel(new_channel): self.update_model_and_plots()
            else: print("Invalid channel selected.")
        except tk.TclError: print("Error reading selected channel value.")

    def on_model_selected(self, event=None):
        """Callback when a new TES model is selected."""
        self.energy_res_text.set("")
        # Read value directly from the Combobox widget
        if self.model_selector_combo is None: return  # Safety check
        new_model_name = self.model_selector_combo.get()

        self.selected_model_name.set(new_model_name)  # Keep StringVar consistent

        print(f"Switching to TES model: {new_model_name}")
        if new_model_name in self.available_models:
            NewModelClass = self.available_models[new_model_name]
            try:
                old_params_values = {};
                if self.params is not None:
                    try:
                        old_params_values = self.params.valuesdict()
                    except Exception as e:
                        print(f"Warning: Could not get values from old params: {e}")

                # <<< CHANGE: Always call the main makeDefaultParamsDict >>>
                self.tes_model = NewModelClass(self.mdt3_core)
                # print(f"DEBUG: Attempting to create default params for {NewModelClass.__name__}...")
                # self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
                if NewModelClass == tes_simple.TES_Simple:
                    if hasattr(self.mdt3_core, 'makeDefaultSimpleParamsDict'):
                        self.params = self.mdt3_core.makeDefaultSimpleParamsDict(num_sets=1)
                        print("DEBUG: Called makeDefaultSimpleParamsDict.")
                    else:
                        print("Warning: makeDefaultSimpleParamsDict not found, using generic.")
                        self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
                elif NewModelClass == tes_intervening.TES_Intervening:
                    if hasattr(self.mdt3_core, 'makeDefaultInterParamsDict'):
                        self.params = self.mdt3_core.makeDefaultInterParamsDict(num_sets=1)
                        print("DEBUG: Called makeDefaultInterParamsDict.")
                    else:
                        print("Warning: makeDefaultInterParamsDict not found, using generic.")
                        self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
                elif NewModelClass == tes_dangling.TES_Dangling:
                    # Assuming a method like makeDefaultDanglingParamsDict exists
                    if hasattr(self.mdt3_core, 'makeDefaultParamsDict'):
                        self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
                        print("DEBUG: Called makeDefaultParamsDict.")

                elif NewModelClass == tes_compound.TES_Compound:
                    # Assuming a method like makeDefaultCompoundParamsDict exists
                    if hasattr(self.mdt3_core, 'makeDefaultCompoundParamsDict'):
                        self.params = self.mdt3_core.makeDefaultCompoundParamsDict(num_sets=1)
                        print("DEBUG: Called makeDefaultCompoundParamsDict.")
                    else:
                        print("Warning: Method for Compound defaults not found, using generic.")
                        self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
                else:  # Fallback for unknown model types
                    print(f"Warning: No specific default param method for {NewModelClass.__name__}, using generic.")
                    self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
                self.initialize_params()

                # Restore previous VALUES where possible
                params_restored = 0
                for name, param in self.params.items():
                    if name in old_params_values:
                        old_value = old_params_values[name];
                        current_min = param.min if param.min is not None else -np.inf;
                        current_max = param.max if param.max is not None else np.inf
                        param.value = np.clip(old_value, current_min, current_max);
                        params_restored += 1
                print(f"  Restored values for {params_restored} parameters.")

                # Refresh GUI Sliders based on the new self.params
                self.create_sliders()

                # Trigger recalculation and plotting
                self.update_model_and_plots()
                self.load_model_diagram(new_model_name)

            except Exception as e:
                import traceback;
                traceback.print_exc()  # Print detailed error
                messagebox.showerror("Model Switch Error", f"Could not switch model or parameters:\n{e}")
        else:
            messagebox.showerror("Model Error", f"Selected model '{new_model_name}' is not available.")

    # --- Slider Creation / Handlers ---
    def create_sliders(self):
        """Recreate sliders/checkboxes using grid layout with headers."""
        # Clear previous widgets from the frame that's INSIDE the canvas
        for widget in self.slider_frame.winfo_children():
            widget.destroy()

        self.param_widgets = {} # Reset storage

        # --- Configure Grid Columns ---
        self.slider_frame.columnconfigure(0, weight=0) # Vary Checkbox
        self.slider_frame.columnconfigure(1, weight=1) # Parameter Name
        self.slider_frame.columnconfigure(2, weight=0) # Unit
        self.slider_frame.columnconfigure(3, weight=0) # Min Label
        self.slider_frame.columnconfigure(4, weight=3) # Slider (main expand)
        self.slider_frame.columnconfigure(5, weight=0) # Current Label
        self.slider_frame.columnconfigure(6, weight=0) # Max Label

        # --- Add Header Row (row 0) ---
        header_font = tkFont.nametofont("TkDefaultFont").copy()
        header_font.configure(weight="bold")

        vary_header = ttk.Label(self.slider_frame, text="Vary?", font=header_font, anchor="center")
        name_header = ttk.Label(self.slider_frame, text="Parameter", font=header_font, anchor="w")
        unit_header = ttk.Label(self.slider_frame, text="Unit", font=header_font, anchor="w")
        min_header = ttk.Label(self.slider_frame, text="Min", font=header_font, anchor="e")
        current_header = ttk.Label(self.slider_frame, text="Current", font=header_font, anchor="center")
        max_header = ttk.Label(self.slider_frame, text="Max", font=header_font, anchor="w")

        vary_header.grid(row=0, column=0, sticky='ew', padx=3)
        name_header.grid(row=0, column=1, sticky='w', padx=3)
        unit_header.grid(row=0, column=2, sticky='w', padx=3)
        min_header.grid(row=0, column=3, sticky='e', padx=3)
        current_header.grid(row=0, column=5, sticky='ew', padx=3) # Spans over slider visually
        max_header.grid(row=0, column=6, sticky='w', padx=3)
        # Add separator line
        ttk.Separator(self.slider_frame, orient='horizontal').grid(row=1, column=0, columnspan=7, sticky='ew', pady=(2, 4))
        # --- END HEADER ROW ---


        # --- START PARAMETER ROWS FROM Row 2 ---
        row = 2 # Start below header and separator
        for name, config in PARAM_DEFS.items():
            # Check if param exists for current model
            if name not in self.params: continue

            initial_gui, p_min, p_max, initial_vary = config
            # Get current values (may have been changed from default by restore etc.)
            current_val = self.params[name].value
            current_vary_status = self.params[name].vary
            # Clip value just for initial slider display consistency
            current_val_display = np.clip(current_val, p_min if p_min is not None else -np.inf, p_max if p_max is not None else np.inf)

            # --- Checkbox for Vary ---
            vary_var = tk.BooleanVar(master=self, value=current_vary_status)
            chk = ttk.Checkbutton(self.slider_frame, variable=vary_var, text="",
                                  command=lambda n=name: self.on_vary_checkbox_change(n))
            chk.grid(row=row, column=0, padx=(3,0))

            # --- Parameter Name ---
            name_label = ttk.Label(self.slider_frame, text=name, anchor='w')
            name_label.grid(row=row, column=1, sticky='w', padx=3)

            # --- Unit ---
            unit_str = UNITS.get(name, "") # Get unit from map
            unit_label = ttk.Label(self.slider_frame, text=unit_str, anchor='w')
            unit_label.grid(row=row, column=2, sticky='w', padx=3)

            # --- Min Label ---
            min_label = ttk.Label(self.slider_frame, text=f"{p_min:.2e}" if p_min is not None else "-inf", width=10, anchor='e')
            min_label.grid(row=row, column=3, sticky='e', padx=3)

            # --- Slider ---
            slider_var = tk.DoubleVar(master=self) # Still need var for scale
            slider = ttk.Scale(self.slider_frame, from_=0, to=1000, orient=tk.HORIZONTAL, variable=slider_var,
                               command=lambda val, p=name: self.on_slider_change(p, float(val)))
            initial_slider_pos = self.map_param_to_slider(current_val_display, p_min, p_max)
            slider_var.set(initial_slider_pos)
            slider.grid(row=row, column=4, sticky='ew', padx=5)
            # Bind keyboard events if needed
            slider.bind("<Left>", lambda event, s=slider, p_name=name, direction=-1: self.handle_arrow_key(event, s, p_name, direction))
            slider.bind("<Right>", lambda event, s=slider, p_name=name, direction=1: self.handle_arrow_key(event, s, p_name, direction))

            # --- Current Value Label ---
            value_label = ttk.Label(self.slider_frame, text=f"{current_val:.3e}", width=12, anchor='center')
            value_label.grid(row=row, column=5, sticky='ew', padx=3)

            # --- Max Label ---
            max_label = ttk.Label(self.slider_frame, text=f"{p_max:.2e}" if p_max is not None else "+inf", width=10, anchor='w')
            max_label.grid(row=row, column=6, sticky='w', padx=3)

            # Store widgets/variables needed later
            self.param_widgets[name] = {
                'slider_var': slider_var,
                'value_label': value_label,
                'min': p_min, # Store GUI min/max from PARAM_DEFS
                'max': p_max,
                'widget': slider,
                'vary_var': vary_var,
                'vary_checkbutton': chk
            }
            row += 1
        # <<< END PARAMETER ROWS >>>

        print(f"DEBUG: Created {row-2} parameter rows in sliders area.")

    def on_vary_checkbox_change(self, param_name):
        """Callback when a 'vary' checkbox is toggled."""
        if param_name in self.param_widgets and param_name in self.params:
            details = self.param_widgets[param_name]
            new_vary_state = details['vary_var'].get()  # Get boolean value from the checkbox's variable
            try:
                self.params[param_name].vary = new_vary_state
                print(f"Parameter '{param_name}' vary set to: {new_vary_state}")
            except Exception as e:
                print(f"Error setting vary for {param_name}: {e}")
        else:
            print(f"Warning: Could not find param details for '{param_name}' in on_vary_checkbox_change")


    def map_param_to_slider(self, param_val, p_min, p_max):
        if p_max <= p_min: return 500.0
        param_val_clamped = np.clip(param_val, p_min, p_max)
        if np.isclose(param_val_clamped, p_min): return 0.0
        if np.isclose(param_val_clamped, p_max): return 1000.0
        return 1000.0 * (param_val_clamped - p_min) / (p_max - p_min)

    def map_slider_to_param(self, slider_val, p_min, p_max):
        if p_max <= p_min: return p_min
        slider_val_clamped = np.clip(float(slider_val), 0.0, 1000.0) # Clip and ensure float
        return p_min + (slider_val_clamped / 1000.0) * (p_max - p_min)

    def handle_arrow_key(self, event, slider_widget, param_name, direction):
        step = 2.0; current_slider_val = slider_widget.get(); new_slider_val = current_slider_val + direction * step; new_slider_val_clamped = np.clip(new_slider_val, 0.0, 1000.0)
        if not np.isclose(current_slider_val, new_slider_val_clamped): slider_widget.set(new_slider_val_clamped); self.on_slider_change(param_name, new_slider_val_clamped)
        return "break"

    def on_slider_change(self, param_name, slider_value):
        if param_name in self.param_widgets and param_name in self.params:
            details = self.param_widgets[param_name]; p_min_gui, p_max_gui = details['min'], details['max']
            new_param_val = self.map_slider_to_param(slider_value, p_min_gui, p_max_gui)
            current_param_val = self.params[param_name].value
            new_param_val_clipped = np.clip(new_param_val, self.params[param_name].min, self.params[param_name].max) # Clip to lmfit bounds
            if not np.isclose(current_param_val, new_param_val_clipped):
                self.params[param_name].value = new_param_val_clipped; details['value_label'].config(text=f"{self.params[param_name].value:.3e}"); self.update_model_and_plots()
            else: details['value_label'].config(text=f"{current_param_val:.3e}") # Update label even if value didn't change

    # --- Fitting Methods ---
    def start_fit_thread(self):
        self._terminate_event.clear()
        self.energy_res_text.set("")
        if self.fit_button:
            self.fit_button.config(state='disabled')
        if self.terminate_button:
            self.terminate_button.config(state='normal')
        if self.save_button:
            self.save_button.config(state='disabled')  # <<< ADDED
        self.last_fit_result = None
        self.fit_status_text.set("Fit status: Preparing...");
        self._update_log(f"--- Starting Fit ({self.selected_model_name.get()}) ---")
        if self.pulse_times_interp is None or self.pulse_data_interp is None or len(self.pulse_times_interp)==0 or len(self.pulse_data_interp)==0:
            messagebox.showerror("Fit Error", "Valid pulse data must be loaded.");
            self.fit_status_text.set("Fit status: Error - Load pulse data");
            self._update_log("Error: Missing pulse data.");
            if self.fit_button:
                self.fit_button.config(state='normal');
                return
        if self.noise_freqs_resampled is None or self.noise_data_resampled is None or len(self.noise_freqs_resampled)==0 or len(self.noise_data_resampled)==0:
            messagebox.showerror("Fit Error", "Valid noise data must be loaded.");
            self.fit_status_text.set("Fit status: Error - Load noise data");
            self._update_log("Error: Missing noise data.");
            if self.fit_button:
                self.fit_button.config(state='normal');
                return
        if self.tes_model is None or self.params is None:
            messagebox.showerror("Fit Error", "Model/Params not initialized.");
            self.fit_status_text.set("Fit status: Error - Model/Params invalid");
            self._update_log("Error: Model/Params invalid.");
            if self.fit_button:
                self.fit_button.config(state='normal');
                return
        varying_params = [name for name, param in self.params.items() if param.vary];
        if not varying_params: messagebox.showwarning("Fit Warning", "No parameters set to vary."); self._update_log("Warning: No varying parameters.") # Optionally return
        else: self._update_log(f"Varying: {varying_params}")

        try:
            fit_method_to_use = self.selected_fit_method.get()
            de_max_nfev = self.de_max_nfev_var.get()
        except tk.TclError as e:
            messagebox.showerror("Input Error", f"Could not read fit settings: {e}")
            if self.fit_button: self.fit_button.config(state='normal')  # Re-enable fit button
            if self.terminate_button: self.terminate_button.config(state='disabled')
            return

        print(f"DEBUG: Using Fit Method: {fit_method_to_use}")
        print(f"DEBUG: Using DE MaxNFEV: {de_max_nfev}")
        self._update_log(f"--- Starting Fit ({self.selected_model_name.get()}) ---")
        self._update_log(f"Using Fit Method: {fit_method_to_use}")
        if fit_method_to_use == 'diff_ev_then_leastsq':
            self._update_log(f"DE Stage MaxNFEV: {de_max_nfev}")

        current_noise_weight = 1.0
        try:
            current_noise_weight = self.noise_weight_factor.get()
        except tk.TclError:
            print("Warning: Could not read noise weight factor, using 1.0")

        params_to_fit = self.params.copy()
        base_fit_args = (self.tes_model, self.pulse_times_interp, self.pulse_data_interp,
                         self.noise_freqs_resampled, self.noise_data_resampled,
                         current_noise_weight)

        self.last_live_update_time = time.time()
        self.fit_status_text.set("Fit status: Fitting...");
        self._update_log(f"Fitting with method '{fit_method_to_use}'...")

        thread_args = (params_to_fit, base_fit_args, fit_method_to_use, de_max_nfev)
        fit_thread = threading.Thread(target=self._fitting_thread_target,
                                      args=thread_args,  # Pass combined args tuple
                                      daemon=True)
        fit_thread.start()

    def _fitting_thread_target(self, params_copy, base_fit_args, fit_method, de_max_nfev):
        """Function executed by the fitting thread. Handles hybrid method."""
        print(f"Fitting thread started using method: {fit_method}")
        final_result = None
        try:
            if fit_method == 'diff_ev_then_leastsq':
                # --- Stage 1: Differential Evolution ---
                print("--- Starting Differential Evolution Stage ---")
                # Update status via print/log - direct GUI update unsafe from thread
                print(f"Status: DE Stage (max NFEV={de_max_nfev})...")
                # self.after(0, lambda: self.fit_status_text.set(f"Status: DE Stage (max NFEV={de_max_nfev})...")) # Safer update

                # Run DE - pass base_fit_args needed by residual_both_gui
                result_de = minimize(self.residual_both_gui, params_copy, args=base_fit_args,
                                     method='differential_evolution',
                                     iter_cb=self.iter_callback,  # Direct call to iter_callback
                                     max_nfev=de_max_nfev)  # Limit evaluations

                if self._terminate_event.is_set():  # Check termination
                    print("--- Fit terminated during Differential Evolution Stage ---")
                    final_result = result_de  # Return the DE result
                elif not result_de.success:
                    print(f"--- Differential Evolution Stage Failed: {result_de.message} ---")
                    final_result = result_de  # Return the failed DE result
                else:
                    print(f"--- Differential Evolution Stage Complete (NFEV={result_de.nfev}) ---")
                    print("Status: Refining (leastsq)...")  # Print status change
                    # self.after(0, lambda: self.fit_status_text.set("Status: Refining (leastsq)...")) # Safer update

                    # --- Stage 2: Least Squares Refinement ---
                    params_for_refine = result_de.params  # Use best params from DE
                    result_refine = minimize(self.residual_both_gui, params_for_refine, args=base_fit_args,
                                             method='leastsq',  # Using leastsq for refinement
                                             iter_cb=self.iter_callback)  # Can use same callback

                    if self._terminate_event.is_set():
                        print("--- Fit terminated during Refinement Stage ---")
                        final_result = result_refine
                    else:
                        print("--- Refinement Stage Complete ---")
                        final_result = result_refine  # Return the final refined result

            else:
                # --- Standard Single Method Fit ---
                print(f"--- Starting Standard Fit ({fit_method}) ---")
                print(f"Status: Fitting ({fit_method})...")
                # self.after(0, lambda: self.fit_status_text.set(f"Status: Fitting ({fit_method})...")) # Safer update

                result_single = minimize(self.residual_both_gui, params_copy, args=base_fit_args,
                                         method=fit_method,
                                         iter_cb=self.iter_callback)
                final_result = result_single

            # Schedule GUI update in main thread using self.after
            if final_result is not None:
                self.after(0, self._on_fit_complete, final_result)
            else:
                print("ERROR: Final fit result was None!")
                self.after(0, self._on_fit_error, "Fit returned None unexpectedly.")

        except Exception as e:
            print(f"Error during fitting thread ({fit_method}): {e}")
            import traceback;
            traceback.print_exc()
            self.after(0, self._on_fit_error, e)  # Schedule error handling
        print("Fitting thread finished.")

    def residual_both_gui(self, params, model_instance, tvals, pdata, fvals, ndata, noise_weight_factor):
        """Calculates residuals weighted by inverse variance and factor.
           Pulse weight uses std dev of first 25% of trace.
           Handles NaNs/Infs."""
        try:
            # --- Calculate Pulse Model & Residual ---
            pulse_resid = np.full_like(pdata, np.nan)  # Initialize with NaN
            try:
                pulse_result = model_instance.calc_pulse(params, tvals);
                if pulse_result is not None and len(pulse_result) > 0 and pulse_result[0] is not None:
                    pulse_model = pulse_result[0];
                    if np.any(~np.isfinite(pulse_model)):
                        print("Warning: NaNs/Infs in pulse_model output.")
                    else:
                        pulse_resid = pulse_model - pdata
                else:
                    print("Warning: calc_pulse returned None/invalid.")
            except Exception as e:
                pass
                # print(f"Warning: Error in calc_pulse: {e}")

            # --- Calculate Noise Model & Residual ---
            noise_resid = np.full_like(ndata, np.nan)  # Initialize with NaN
            try:
                noise_result = model_instance.calc_noise(params, fvals);
                if noise_result is not None and len(noise_result) > 0 and noise_result[0] is not None:
                    noise_model_total = noise_result[0];
                    if np.any(~np.isfinite(noise_model_total)):
                        print("Warning: NaNs/Infs in noise_model output.")
                    else:
                        noise_resid = noise_model_total - ndata
                else:
                    print("Warning: calc_noise returned None/invalid.")
            except Exception as e:
                print(f"Warning: Error in calc_noise: {e}")

            # --- Calculate Inverse Variance Weights ---
            pulse_var_weight = 1.0;
            noise_var_weight = 1.0
            try:  # Pulse variance weight calc
                n_pulse = len(pdata);
                idx_25 = max(2, int(0.25 * n_pulse))
                pulse_std_fq = np.std(pdata[:idx_25])
                if np.isfinite(pulse_std_fq) and pulse_std_fq > 1e-30: pulse_var_weight = 1.0 / (pulse_std_fq ** 2)
            except Exception:
                print("Warning: Error calculating pulse variance weight")
            try:  # Noise variance weight calc
                n_noise = len(ndata);
                noise_std = np.std(ndata)
                if np.isfinite(noise_std) and noise_std > 1e-30: noise_var_weight = 1.0 / (noise_std ** 2)
            except Exception:
                print("Warning: Error calculating noise variance weight")

            # --- Apply Weights (Variance + Manual Factor) ---
            weighted_pulse_resid = pulse_resid * pulse_var_weight
            # <<< MODIFIED: Apply noise_weight_factor >>>
            # Apply sqrt so factor scales residual magnitude linearly (approx) in least-squares
            # A factor of 4 means noise counts 4x as much in sum-of-squares
            weighted_noise_resid = noise_resid * noise_var_weight * noise_weight_factor
            # <<< MODIFICATION END >>>

            # Combine residuals
            combined_residuals = np.concatenate((np.ravel(weighted_pulse_resid),
                                                 np.ravel(weighted_noise_resid)))

            # --- Final NaN/Inf Check and Replacement ---
            bad_indices = ~np.isfinite(combined_residuals);
            if np.any(bad_indices): combined_residuals[bad_indices] = 1e18  # Replace NaN/Inf
            return combined_residuals.astype(np.float64)

        except Exception as e:
            print(f"Error inside residual_both_gui: {e}");
            import traceback;
            traceback.print_exc()
            total_len = len(pdata) + len(ndata);
            return np.full(total_len, 1e18); total_len = len(pdata) + len(ndata); return np.full(total_len, 1e18)

    def start_noise_only_fit_thread(self):
        """Starts the fitting process for noise data only."""
        self._terminate_event.clear()
        # Disable all fit buttons, enable terminate
        self._set_fit_buttons_state(tk.DISABLED)  # Use helper if available
        if self.terminate_button: self.terminate_button.config(state=tk.NORMAL)
        if self.save_button: self.save_button.config(state='disabled')
        self.last_fit_result = None
        self.energy_res_text.set("")  # Clear previous results display

        self.fit_status_text.set("Fit status: Preparing (Noise Only)...")
        self._update_log(f"--- Starting Noise Only Fit ({self.selected_model_name.get()}) ---")

        # --- Validate ONLY Noise Data ---
        if self.noise_freqs_resampled is None or self.noise_data_resampled is None or \
                len(self.noise_freqs_resampled) == 0 or len(self.noise_data_resampled) == 0:
            messagebox.showerror("Fit Error", "Valid noise data must be loaded for noise-only fit.")
            self.fit_status_text.set("Fit status: Error - Load noise data")
            self._update_log("Error: Missing noise data.")
            self._set_fit_buttons_state(tk.NORMAL)  # Re-enable fit buttons
            if self.terminate_button: self.terminate_button.config(state=tk.DISABLED)
            return
        # --- End Validation ---

        if self.tes_model is None or self.params is None:
            messagebox.showerror("Fit Error", "Model/Params not initialized.")
            self.fit_status_text.set("Fit status: Error - Model/Params invalid")
            self._update_log("Error: Model/Params invalid.")
            self._set_fit_buttons_state(tk.NORMAL)
            if self.terminate_button: self.terminate_button.config(state=tk.DISABLED)
            return

        varying_params = [name for name, param in self.params.items() if param.vary];
        if not varying_params:
            messagebox.showwarning("Fit Warning", "No parameters set to vary.")
            self._update_log("Warning: No varying parameters.")
        else:
            self._update_log(f"Varying: {varying_params}")

        # Get selected fit method settings
        fit_method_to_use = self.selected_fit_method.get()
        de_max_nfev = self.de_max_nfev_var.get()  # For hybrid method

        params_to_fit = self.params.copy()
        # Args needed by residual_noise_only_gui: model, freqs, data
        fit_args = (self.tes_model, self.noise_freqs_resampled, self.noise_data_resampled)

        self.last_live_update_time = time.time()
        self.fit_status_text.set(f"Fit status: Fitting Noise ({fit_method_to_use})...")
        self._update_log(f"Fitting noise with method '{fit_method_to_use}'...")
        if fit_method_to_use == 'diff_ev_then_leastsq':
            self._update_log(f"DE Stage MaxNFEV: {de_max_nfev}")

        # Pass needed args to the thread target
        thread_args = (params_to_fit, fit_args, fit_method_to_use, de_max_nfev)
        fit_thread = threading.Thread(target=self._noise_only_fitting_thread_target,  # Use new target
                                      args=thread_args,
                                      daemon=True)
        fit_thread.start()

    def _noise_only_fitting_thread_target(self, params_copy, fit_args, fit_method, de_max_nfev):
        """Function executed by the noise-only fitting thread."""
        print(f"Noise-only fitting thread started using method: {fit_method}")
        final_result = None
        try:
            if fit_method == 'diff_ev_then_leastsq':
                # --- Stage 1: DE ---
                print("--- Starting DE Stage (Noise Only) ---")
                # Note: Status updates need scheduling via self.after from thread
                result_de = minimize(self.residual_noise_only_gui, params_copy, args=fit_args,  # Use noise residual
                                     method='differential_evolution',
                                     iter_cb=self.iter_callback,
                                     max_nfev=de_max_nfev)

                if self._terminate_event.is_set() or not result_de.success:  # Check termination or DE failure
                    final_result = result_de
                else:
                    # --- Stage 2: Refinement ---
                    print(f"--- DE Stage Complete (NFEV={result_de.nfev}). Starting Refinement (Noise Only) ---")
                    params_for_refine = result_de.params
                    # Using leastsq (can change to least_squares if preferred/works)
                    result_refine = minimize(self.residual_noise_only_gui, params_for_refine, args=fit_args,
                                             # Use noise residual
                                             method='leastsq',
                                             iter_cb=self.iter_callback)
                    final_result = result_refine  # Store refinement result

            else:
                # --- Standard Single Method Fit ---
                print(f"--- Starting Standard Noise-Only Fit ({fit_method}) ---")
                result_single = minimize(self.residual_noise_only_gui, params_copy, args=fit_args,  # Use noise residual
                                         method=fit_method,
                                         iter_cb=self.iter_callback)
                final_result = result_single

            # Schedule GUI update in main thread using self.after
            if final_result is not None:
                self.after(0, self._on_fit_complete, final_result)
            else:
                print("ERROR: Final fit result was None!")
                self.after(0, self._on_fit_error, "Fit returned None unexpectedly.")

        except Exception as e:
            print(f"Error during noise-only fitting thread ({fit_method}): {e}")
            import traceback;
            traceback.print_exc()
            self.after(0, self._on_fit_error, e)  # Schedule error handling
        print("Noise-only fitting thread finished.")

    def residual_noise_only_gui(self, params, model_instance, fvals, ndata):
        """Calculates residuals for noise data only, weighted by inverse variance."""
        try:
            noise_resid = np.full_like(ndata, np.nan)  # Initialize with NaN
            try:
                # Calculate full noise model result (list/tuple of components)
                noise_result = model_instance.calc_noise(params, fvals)
                # Ensure result is valid and get the total noise (usually first element)
                if noise_result is not None and len(noise_result) > 0 and noise_result[0] is not None:
                    noise_model_total = noise_result[0]
                    if np.any(~np.isfinite(noise_model_total)):
                        print("Warning: NaNs/Infs in noise_model output during noise fit.")
                        # Leave noise_resid as NaN
                    else:
                        noise_resid = noise_model_total - ndata  # Calculate residual
                else:
                    print("Warning: calc_noise returned None/invalid during noise fit.")
                    # Leave noise_resid as NaN
            except Exception as e:
                print(f"Warning: Error in calc_noise during noise fit: {e}")
                # Leave noise_resid as NaN

            # --- Calculate Inverse Variance Weight ---
            noise_var_weight = 1.0  # Default weight
            try:
                n_noise = len(ndata)
                if n_noise > 1:  # Need >1 point for std dev
                    noise_std = np.std(ndata)
                    if np.isfinite(noise_std) and noise_std > 1e-30:  # Check std dev is valid
                        noise_var_weight = 1.0 / (noise_std ** 2)
                        # print(f"DEBUG Noise Only Weight: 1/({noise_std:.2e})^2 = {noise_var_weight:.2e}") # Optional
                    else:
                        # Don't print warning every iteration, too noisy
                        # print(f"Warning: Std dev of noise data ({noise_std:.2e}) invalid. Using default weight=1.")
                        pass
                else:
                    # print("Warning: Not enough noise data points for std dev. Using default weight=1.")
                    pass
            except Exception as e:
                # print(f"Warning: Error calculating noise weight: {e}. Using default weight=1.")
                pass  # Avoid excessive printing

            # Apply weight
            weighted_noise_resid = noise_resid * noise_var_weight

            # --- Final NaN/Inf Check ---
            bad_indices = ~np.isfinite(weighted_noise_resid);
            if np.any(bad_indices):
                weighted_noise_resid[bad_indices] = 1e18  # Replace NaN/Inf with large number

            return weighted_noise_resid.astype(np.float64)  # Return only noise residuals

        except Exception as e:
            print(f"Error inside residual_noise_only_gui: {e}");
            import traceback;
            traceback.print_exc()
            # Return array of large numbers on error
            return np.full_like(ndata, 1e18, dtype=np.float64)

    # --- Helper to manage button states (if not already present) ---
    def _set_fit_buttons_state(self, state):
        """Sets the state (tk.NORMAL or tk.DISABLED) for all fit-related buttons."""
        # Make sure to include the new noise_only_fit_button
        buttons = [self.fit_button, self.noise_only_fit_button,
                   self.terminate_button, self.save_button]
        for button in buttons:
            if button:  # Check if button exists
                try:
                    button.config(state=state)
                except tk.TclError:
                    pass  # Ignore errors if window closing



    def iter_callback(self, params, iteration, resid, *args, **kws):
        """Callback function called by lmfit after each iteration for live plot updates."""
        if self._terminate_event.is_set():
            print(f"Fit termination requested at iteration {iteration}. Stopping.")
            return True  # Return True to stop the minimizer
        current_time = time.time();
        if current_time - self.last_live_update_time > self.live_update_interval:
            self.last_live_update_time = current_time;
            params_copy_for_plot = params.copy()
            self.after_idle(self.update_plots_live, params_copy_for_plot)
        return None # Continue fitting

    def update_plots_live(self, live_params):
        """Updates only the model lines on the plots using intermediate params."""
        if self.tes_model is None or live_params is None: return
        live_pulse_model_y = None
        if self.pulse_times_interp is not None and len(self.pulse_times_interp) > 0:
            try:
                pulse_result = self.tes_model.calc_pulse(live_params, self.pulse_times_interp);
                if isinstance(pulse_result, (list, tuple)) and len(pulse_result) > 0 and pulse_result[0] is not None: live_pulse_model_y = pulse_result[0]
            except Exception: pass
        live_noise_arrays = {key: None for key in ALL_POSSIBLE_NOISE_KEYS}
        if self.noise_freqs_resampled is not None and len(self.noise_freqs_resampled) > 0:
            try:
                noise_results = self.tes_model.calc_noise(live_params, self.noise_freqs_resampled); model_type = type(self.tes_model); expected_names = NOISE_COMPONENT_MAP.get(model_type)
                if expected_names and isinstance(noise_results, (list, tuple)) and len(noise_results) >= len(expected_names):
                    for i, name in enumerate(expected_names):
                         if name in live_noise_arrays: live_noise_arrays[name] = noise_results[i]
            except Exception: pass
        if self.pulse_times_interp is not None and live_pulse_model_y is not None: valid_idx = ~np.isnan(live_pulse_model_y); self.line_pulse_model.set_data(self.pulse_times_interp[valid_idx], live_pulse_model_y[valid_idx])
        else: self.line_pulse_model.set_data([], [])
        if self.noise_freqs_resampled is not None:
            freqs = self.noise_freqs_resampled
            for key, line_obj in self.noise_plot_lines.items():
                 data_array = live_noise_arrays.get(key);
                 if data_array is not None:
                      if not isinstance(data_array, np.ndarray): data_array = np.array(data_array).astype(float);
                      else: data_array = data_array.astype(float)
                      valid_idx_model = np.isfinite(data_array) & (data_array > 0);
                      if np.any(valid_idx_model): x_data = freqs[valid_idx_model]; y_data = data_array[valid_idx_model]; line_obj.set_data(x_data, y_data)
                      else: line_obj.set_data([], [])
                 else: line_obj.set_data([], [])
        else:
            for line_obj in self.noise_plot_lines.values(): line_obj.set_data([], [])
        self.canvas.draw_idle()

    def _on_fit_complete(self, result):
        """Callback executed in the main thread after fitting finishes."""
        print("Fit complete. Updating GUI...")
        if self.terminate_button:
            try:
                self.terminate_button.config(state='disabled')
            except tk.TclError:
                pass  # Ignore error if window closed
        if self.log_text:
            try: self.log_text.config(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.config(state='disabled')
            except tk.TclError: pass
        if result and hasattr(result, 'success'):
            try: report = fit_report(result); self._update_log(report)
            except Exception as report_error: print(f"Error generating fit report: {report_error}"); self._update_log(f"--- Fit Report Generation Error ---\n{report_error}")
            if result.success:
                self.fit_status_text.set(f"Fit status: Success! (χ²={result.chisqr:.3e}, Nfev={result.nfev})"); print("\n--- Fit Successful ---")
                self.params = result.params;
                if self.save_button:
                    try:
                        self.save_button.config(state='normal')
                    except tk.TclError:
                        pass
                try:
                    if hasattr(self.tes_model, 'calc_energy_res'):
                        print("Calculating energy resolution...")  # Debug
                        # Assuming calc_energy_res returns value in eV
                        energy_res_ev = self.tes_model.calc_energy_res(self.params)
                        if energy_res_ev is not None and np.isfinite(energy_res_ev):
                            display_text = f"Est. Energy Res: {energy_res_ev:.2f} eV"
                            self.energy_res_text.set(display_text)
                            self._update_log(display_text)  # Add to log too
                            print(display_text)
                        else:
                            print("Warning: calc_energy_res returned invalid value.")
                            self.energy_res_text.set("Energy Res: Calculation Invalid")
                    else:
                        print(f"Warning: Model '{type(self.tes_model).__name__}' does not have calc_energy_res method.")
                        self.energy_res_text.set("Energy Res: N/A for model")
                except Exception as e_res:
                    print(f"ERROR calculating energy resolution: {e_res}")
                    import traceback;
                    traceback.print_exc()
                    self.energy_res_text.set("Energy Res: Error")
                self.update_sliders_from_params();
                self.update_model_and_plots()
                self.last_fit_result = result

            else:
                self.fit_status_text.set(f"Fit status: Failed - {result.message}"); self._update_log(f"\n--- Fit Failed ---\nMessage: {result.message}"); print(f"\n--- Fit Failed ---")
        else: status_msg = "Fit status: Error - Invalid result object received."; self.fit_status_text.set(status_msg); self._update_log(status_msg); print("Error: Invalid result object received from fitting thread.")
        if self.fit_button:
             try: self.fit_button.config(state='normal')
             except tk.TclError: pass
        print("GUI update complete.")

    def _on_fit_error(self, error):
        """Callback executed in the main thread if the fitting thread raised an exception."""
        if self.terminate_button:
            try:
                self.terminate_button.config(state='disabled')
            except tk.TclError:
                pass
        self.last_fit_result = None
        self.energy_res_text.set("")
        if self.save_button:
            try:
                self.save_button.config(state='disabled')
            except tk.TclError:
                pass
        messagebox.showerror("Fit Error", f"An unexpected error occurred during fitting:\n{error}")
        status_msg = f"Fit status: Thread Error - {type(error).__name__}"; self.fit_status_text.set(status_msg); self._update_log(f"\n--- Fit Error ---\n{error}")
        if self.fit_button:
             try: self.fit_button.config(state='normal')
             except tk.TclError: pass

    def request_fit_termination(self):
        """Signals the fitting thread to stop via the threading event."""
        if self._terminate_event.is_set():  # Already requested
            return
        print("Requesting fit termination...")
        self._update_log(">>> Requesting Fit Termination <<<")
        self.fit_status_text.set("Fit status: Terminating...")
        self._terminate_event.set()  # Set the event flag
        if self.terminate_button:
            self.terminate_button.config(state='disabled')

        # Inside class TESModelViewer:

    def save_fit_report(self):
        """Saves the last successful fit report to a text file."""
        if self.last_fit_result is None or not self.last_fit_result.success:
            messagebox.showinfo("Save Report", "No successful fit result available to save.")
            return
        # Check if filenames are available (optional, but good context)
        if not self.current_pulse_filepath or not self.current_noise_filepath:
            if not messagebox.askyesno("Save Report Warning",
                                       "Pulse and/or Noise file paths are missing in the GUI state.\nSave report without full file context?"):
                return

        # Suggest a filename based on model and channel
        model_name = self.selected_model_name.get()
        chan_num = self.selected_noise_channel.get()
        chan_str = f"chan{chan_num}" if chan_num >= 0 else "chanNA"
        default_filename = f"fit_report_{model_name}_{chan_str}.txt"

        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Fit Report As",
            initialfile=default_filename,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not save_path:  # User cancelled dialog
            self._update_log("Save report cancelled.")
            return

        print(f"Saving fit report to: {save_path}")
        try:
            # Generate the report string from lmfit
            report_str = fit_report(self.last_fit_result)

            # Create header information
            now = datetime.datetime.now()
            header = (
                f"==================================================\n"
                f" Fit Report\n"
                f"==================================================\n"
                f"Saved on:            {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"TES Model Used:      {model_name}\n"
                f"Pulse File Used:     {self.current_pulse_filepath or 'N/A'}\n"
                f"Noise File Used:     {self.current_noise_filepath or 'N/A'}\n"
                f"Noise Channel Used:  {chan_str}\n"
                f"--------------------------------------------------\n\n"
            )

            # Write header and report to file
            with open(save_path, 'w') as f:
                f.write(header)
                f.write(report_str)

            messagebox.showinfo("Save Report", f"Fit report successfully saved to:\n{os.path.basename(save_path)}")
            self._update_log(f"Fit report saved to {os.path.basename(save_path)}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save fit report:\n{e}")
            self._update_log(f"Error saving fit report: {e}")

    def update_sliders_from_params(self):
        """Updates the GUI sliders AND checkboxes to match the current self.params values."""
        print("Updating sliders and checkboxes from parameters...")
        if self.params is None: return
        for name, param in self.params.items():
            if name in self.param_widgets:
                details = self.param_widgets[name]
                p_min_gui, p_max_gui = details['min'], details['max'] # Use GUI bounds for mapping

                # Update Slider Position
                slider_val = self.map_param_to_slider(param.value, p_min_gui, p_max_gui)
                try:
                    details['slider_var'].set(slider_val)
                    details['value_label'].config(text=f"{param.value:.3e}")
                except tk.TclError as e: print(f"Warning: TclError updating slider/label for {name}: {e}")

                # --- ADD: Update Checkbox State ---
                if 'vary_var' in details:
                    try:
                        details['vary_var'].set(param.vary) # Set checkbox state from param.vary
                    except tk.TclError as e: print(f"Warning: TclError updating vary checkbox for {name}: {e}")
                # --- END ADD ---


    def _update_log(self, message):
        """Helper method to append messages to the log window."""
        if self.log_text:
            try: self.log_text.config(state='normal'); self.log_text.insert(tk.END, message + "\n"); self.log_text.see(tk.END); self.log_text.config(state='disabled')
            except tk.TclError as e: print(f"Warning: Could not update log window - {e}")

    # --- Plot Update ---
    def update_model_and_plots(self):
        """Recalculates model predictions using current model/params and updates plots."""
        if self.tes_model is None or self.params is None: print("Model or parameters not initialized..."); return
        # Parameter Pre-checks (Keep essential checks)
        calculation_valid = True; warning_messages = []; L_I = None
        try:
            p = self.params.valuesdict(); set_index = 0
            L = p.get(f'L_{set_index}', 0); R_0 = p.get(f'R_0_{set_index}', 0); T_tes = p.get(f'T_tes_{set_index}', 0); T_bath = p.get(f'T_bath_{set_index}', 0); alpha_I = p.get(f'alpha_I_{set_index}', 0); n_mem = p.get(f'n_mem_{set_index}', 1)
            if L <= 1e-12: warning_messages.append(f"L ({L:.2e}) <= 0"); calculation_valid = False
            if R_0 <= 1e-9: warning_messages.append(f"R_0 ({R_0:.2e}) <= 0"); calculation_valid = False
            if T_tes <= T_bath: warning_messages.append(f"T_tes ({T_tes:.3f}) <= T_bath ({T_bath:.3f})"); calculation_valid = False
            if calculation_valid and T_tes > 0 and n_mem != 0:
                 try:
                     L_I = (alpha_I / n_mem) * (1.0 - (T_bath / T_tes)**n_mem);
                     if np.isclose(L_I, 1.0, atol=1e-6): warning_messages.append(f"L_I ({L_I:.4f}) ~= 1"); calculation_valid = False
                     elif np.isclose(L_I, 0.0, atol=1e-9): warning_messages.append(f"L_I ({L_I:.4f}) ~= 0"); calculation_valid = False
                 except (ZeroDivisionError, ValueError, TypeError, OverflowError): warning_messages.append(f"L_I calc error."); calculation_valid = False
            elif calculation_valid: warning_messages.append(f"Cannot calc L_I (T_tes/n_mem invalid)."); calculation_valid = False
        except Exception as e: warning_messages.append(f"Pre-check Error: {e}"); calculation_valid = False

        # Calculations
        pulse_model_y = None; self.noise_model_arrays = {key: None for key in ALL_POSSIBLE_NOISE_KEYS}; noise_calculation_successful = False
        if not calculation_valid: print(f"Skipping model calculation: {warning_messages}")
        else:
            if self.pulse_times_interp is not None and len(self.pulse_times_interp) > 0:
                try:
                    pulse_result = self.tes_model.calc_pulse(self.params, self.pulse_times_interp);
                    if isinstance(pulse_result, (list, tuple)) and len(pulse_result) > 0 and pulse_result[0] is not None: pulse_model_y = pulse_result[0]
                    else: print(f"Warning: calc_pulse invalid result: {pulse_result}"); pulse_model_y = np.full_like(self.pulse_times_interp, np.nan)
                except Exception as e: print(f"Error (Pulse Calc): {e}"); pulse_model_y = np.full_like(self.pulse_times_interp, np.nan)
            if self.noise_freqs_resampled is not None and len(self.noise_freqs_resampled) > 0:
                try:
                    noise_results = self.tes_model.calc_noise(self.params, self.noise_freqs_resampled); model_type = type(self.tes_model); expected_names = NOISE_COMPONENT_MAP.get(model_type)
                    if expected_names and isinstance(noise_results, (list, tuple)) and len(noise_results) >= len(expected_names):
                        for i, name in enumerate(expected_names):
                             if name in self.noise_model_arrays: self.noise_model_arrays[name] = noise_results[i]
                        noise_calculation_successful = True
                    else: print(f"Warning: Unexpected noise result format for {model_type.__name__}.")
                except Exception as e: print(f"Error (Noise Calc): {e}")

        # Update Plot Lines
        if self.pulse_times_raw is not None: self.line_pulse_data.set_data(self.pulse_times_raw, self.pulse_data_raw)
        else: self.line_pulse_data.set_data([], [])
        if self.pulse_times_interp is not None and pulse_model_y is not None: valid_idx = ~np.isnan(pulse_model_y); self.line_pulse_model.set_data(self.pulse_times_interp[valid_idx], pulse_model_y[valid_idx])
        else: self.line_pulse_model.set_data([], [])
        valid_noise_data = False
        if self.noise_freqs_resampled is not None and self.noise_data_resampled is not None:
            valid_idx_data = self.noise_data_resampled > 0
            if np.any(valid_idx_data): self.line_noise_data.set_data(self.noise_freqs_resampled[valid_idx_data], self.noise_data_resampled[valid_idx_data]); valid_noise_data = True
            else: self.line_noise_data.set_data([], [])
        else: self.line_noise_data.set_data([], [])
        can_use_log_y = valid_noise_data
        if self.noise_freqs_resampled is not None:
            freqs = self.noise_freqs_resampled
            for key, line_obj in self.noise_plot_lines.items():
                 data_array = self.noise_model_arrays.get(key);
                 if data_array is not None:
                      if not isinstance(data_array, np.ndarray): data_array = np.array(data_array).astype(float);
                      else: data_array = data_array.astype(float)
                      valid_idx_model = np.isfinite(data_array) & (data_array > 0) & np.isfinite(freqs) & (freqs > 0);
                      if np.any(valid_idx_model):
                          x_data = freqs[valid_idx_model]; y_data = data_array[valid_idx_model];
                          if len(x_data)>0: line_obj.set_data(x_data, y_data); can_use_log_y = True # Check len > 0 after filtering
                          else: line_obj.set_data([], [])
                      else: line_obj.set_data([], [])
                 else: line_obj.set_data([], [])
        else:
            for line_obj in self.noise_plot_lines.values(): line_obj.set_data([], [])

        # Adjust Axes
        try:
            self.ax_pulse.relim(); self.ax_pulse.autoscale_view()
            if self.pulse_times_raw is not None and len(self.pulse_times_raw) > 1:
                 center_time_idx = np.argmin(self.pulse_data_raw) if self.pulse_data_raw is not None and len(self.pulse_data_raw) > 0 else len(self.pulse_times_raw)//2
                 if center_time_idx < len(self.pulse_times_raw):
                     center_time = self.pulse_times_raw[center_time_idx]; view_width = 2e-3; x_min_pulse, x_max_pulse = center_time - view_width, center_time + view_width
                     if x_max_pulse > x_min_pulse and np.isfinite(x_min_pulse) and np.isfinite(x_max_pulse): self.ax_pulse.set_xlim(x_min_pulse, x_max_pulse)
                 if self.pulse_data_raw is not None and len(self.pulse_data_raw) > 0:
                     try:
                         min_y, max_y = np.nanmin(self.pulse_data_raw), np.nanmax(self.pulse_data_raw); y_range = max_y - min_y
                         if y_range > 0 and np.isfinite(min_y) and np.isfinite(max_y):
                             self.ax_pulse.set_ylim(min_y - 0.1 * y_range, max_y + 0.1 * y_range)
                     except ValueError:
                         pass
        except Exception as e: (
            print(f"Warning: Error adjusting pulse axes limits: {e}"))
        try:
            current_yscale = self.ax_noise.get_yscale(); new_yscale = 'log' if can_use_log_y else 'linear';
            if new_yscale == 'log' and not can_use_log_y: new_yscale = 'linear'
            if current_yscale != new_yscale: self.ax_noise.set_yscale(new_yscale)
            current_xscale = self.ax_noise.get_xscale(); new_xscale = 'log' if (self.noise_freqs_resampled is not None and np.any(self.noise_freqs_resampled > 0)) else 'linear'
            if current_xscale != new_xscale: self.ax_noise.set_xscale(new_xscale)
            self.ax_noise.relim(); self.ax_noise.autoscale_view()
        except Exception as e: print(f"Warning: Could not autoscale/set scale for noise axes: {e}")

        # Final Draw
        try: self.fig.tight_layout(pad=2.0)
        except Exception as e: print(f"Warning: tight_layout failed: {e}")
        self.canvas.draw()


# === Run the Application ===
if __name__ == "__main__":
    if 'DummyMDT3' in str(mdt3_core.__class__): print("Exiting because base mdt3 library could not be loaded.")
    else:

        # --- Create App Instance ---
        app = TESModelViewer() # Inherits from tk.Tk
        app.mainloop()