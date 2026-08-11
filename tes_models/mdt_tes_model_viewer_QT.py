# --- GUI Imports ---
import sys
import os
import time
import datetime
import threading # Keep threading.Event for termination flag if preferred
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSlider, QCheckBox, QTextEdit,
    QFileDialog, QMessageBox, QScrollArea, QSizePolicy, QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIntValidator # Might be useful for channel entry later

# --- Styling Import ---
from qt_material import apply_stylesheet

# --- Scientific Imports ---
import numpy as np
import matplotlib.pyplot as plt
# Ensure using the Qt Agg backend for matplotlib
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import scipy.constants
from scipy.interpolate import interp1d
from lmfit import Parameters, minimize, fit_report

# --- MDT3 Imports ---
# (Same mdt3 imports and dummy classes as before)
try:
    from mdt3 import mdt3_core, tes_simple
    try: from mdt3 import tes_compound; HAS_COMPOUND = True
    except ImportError: print("Warning: mdt3 tes_compound not found."); HAS_COMPOUND = False
    try: from mdt3 import tes_intervening; HAS_INTER = True
    except ImportError: print("Warning: mdt3 tes_intervening not found."); HAS_INTER = False
    try: from mdt3 import tes_dangling; HAS_DANGLING = True
    except ImportError: print("Warning: mdt3 tes_dangling not found."); HAS_DANGLING = False
except ImportError as main_import_error:
    print(f"CRITICAL: Could not import base mdt3 library: {main_import_error}")
    # Dummy classes...
    class DummyMDT3:
        def makeDefaultParamsDict(self, num_sets=1):
            return Parameters()
    class DummyTES:
        def __init__(self, core): pass
        def calc_pulse(self, params, tvals): return np.zeros_like(tvals),
        def calc_noise(self, params, fvals): return [np.zeros_like(fvals)]*6
        def calc_derived_params(self, params): self.simple_params = {}
    mdt3_core = DummyMDT3(); tes_simple = DummyTES; tes_compound = DummyTES; tes_intervening = DummyTES; tes_dangling = DummyTES
    HAS_COMPOUND = True; HAS_INTER = True; HAS_DANGLING = True


# === Constants and Configuration ===
# (Keep phi0, MIN_SI_CONSTANT, PARAM_DEFS, NOISE_COMPONENT_MAP, etc.)
phi0 = scipy.constants.value(u"mag. flux quantum")
MIN_SI_CONSTANT = 248e-12
MIN_PHI0_PER_AMP = MIN_SI_CONSTANT / phi0 if phi0 != 0 else 1.0
PARAM_DEFS = {
    # ... (Existing parameters) ...
    'G_tes_bath_0':   [ 0.093,         25e-3,   1.0,      False  ],
    'G_abs_tes_0':    [8,           1.0,     20.0,     True  ],
    'G_tes_int_0':    [1.5,           0.1,     10.0,     False  ], # Keep this one
    'T_tes_0':        [ 0.053,        0.045,   0.055,    False ],
    'T_bath_0':       [ 0.021,        0.019,   0.025,    False ],
    'C_tes_0':        [ 0.06,         0.001,   1.0,      True  ],
    'C_abs_0':        [ .17,          0.01,    1.0,      True  ],
    'C_int_0':        [ 1.125,        0.01,    10.0,     False ], # Example: 0.075*15=1.125, adjusted max
    'alpha_I_0':      [ 435.0,        20.0,    2000.0,   True  ],
    'beta_I_0':       [ 17.0,         1.0,     30.0,     True  ],
    'R_0_0':          [ 470e-6,       450e-6,   500e-6,  False ],
    'R_L_0':          [ 250e-6,       200e-6,  300e-6,   False ],
    'L_0':            [ 120,       80,   150,   True  ],
    'n_mem_0':        [ 3.8,          3.5,     4.0,      False ], # Keep for Simple model? Or remove?
    'M_0':            [ 0.0,          0.0,     8.0,      False ],
    'initE_0':        [ 2.4,  0.0,     5.0, False ],
    'squid_noise_0':  [ 2.6e-11,      2e-11,   4e-11,    False ],
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
    'L_0': 'nH', # Based on values like 120e-9 H
    # Energy
    'initE_0': 'eV', # Based on values like 2.4 * 1.6e-19 J
    # Noise Terms (Using rtHz for sqrt(Hz))
    'squid_noise_0': r'A/$\sqrt{Hz}$',
    'background_noise_0': r'A/$\sqrt{Hz}$',
    'abs_background_noise_0': r'A/$\sqrt{Hz}$',
    # Other
    'int_background_power_0': 'N/A',
}

NOISE_COMPONENT_MAP = { tes_simple.TES_Simple: ['total', 'r0', 'rl', 'g1', 'squid', 'back'],
                        tes_compound.TES_Compound: ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back'],
                        tes_dangling.TES_Dangling: ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back'],
                        tes_intervening.TES_Intervening: ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back'], }
ALL_POSSIBLE_NOISE_KEYS = ['total', 'r0', 'rl', 'g1', 'g2', 'squid', 'back', 'abs_back']
NOISE_PLOT_CONFIG = { 'total': {'label': 'Total Model', 'color': 'red', 'linestyle': '-', 'linewidth': 2}, 'r0': {'label': '$SI_{R_0}$', 'color': 'darkorange', 'linestyle': '--', 'linewidth': 1}, 'rl': {'label': '$SI_{R_L}$', 'color': 'forestgreen', 'linestyle': '--', 'linewidth': 1}, 'g1': {'label': '$SI_{G1}$', 'color': 'purple', 'linestyle': '--', 'linewidth': 1}, 'g2': {'label': '$SI_{G2}$', 'color': 'brown', 'linestyle': '--', 'linewidth': 1}, 'squid': {'label': '$SI_{SQUID}$', 'color': 'cyan', 'linestyle': '--', 'linewidth': 1}, 'back': {'label': '$SI_{Back}$', 'color': 'magenta', 'linestyle': '--', 'linewidth': 1}, 'abs_back': {'label': '$SI_{AbsBack}$', 'color': 'gold', 'linestyle': '--', 'linewidth': 1}, }

# === Helper Functions ===
# (Keep phi0_to_amp, load_noise_file, load_avg_pulse, logarithmic_resample - unchanged)
def phi0_to_amp(inval):
    if MIN_PHI0_PER_AMP == 0: return np.zeros_like(inval)
    return inval * (1.0 / MIN_PHI0_PER_AMP)


def load_noise_file(filename):
    """Loads multi-channel noise data from NPZ file."""
    try:
        with np.load(filename, allow_pickle=True) as f:
            psd_key = 'Pxx'; freq_key = 'f';
            if psd_key not in f or freq_key not in f:
                # <<< MODIFIED >>>
                QMessageBox.critical(None, "File Error", f"Keys '{psd_key}' or '{freq_key}' not found in noise file: {filename}")
                return None, None
            psd_data = f[psd_key]; freq_data = f[freq_key];
            psd_item = psd_data.item() if hasattr(psd_data, 'item') else psd_data;
            freq_item = freq_data.item() if hasattr(freq_data, 'item') else freq_data
            if isinstance(psd_item, dict): key = list(psd_item.keys())[0]; psd = psd_item[key]; freqs = freq_item[key]
            else: psd = psd_item; freqs = freq_item
            return freqs, psd
    except FileNotFoundError:
        # <<< MODIFIED >>>
        QMessageBox.critical(None, "File Error", f"Noise file not found: {filename}")
        return None, None
    except Exception as e:
        # <<< MODIFIED >>>
        QMessageBox.critical(None, "File Error", f"Error loading noise file {filename}:\n{e}")
        import traceback; traceback.print_exc() # Also print traceback to console
        return None, None

def load_avg_pulse(filename, pulse_arrival_sample=0):
    """Loads average pulse data from NPZ file."""
    try:
        with np.load(filename, allow_pickle=True) as f:
            pulse_key = 'array1';
            if pulse_key not in f:
                 # <<< MODIFIED >>>
                 QMessageBox.critical(None, "File Error", f"Key '{pulse_key}' not found in pulse file: {filename}")
                 return None, None
            avg_pulse_raw = f[pulse_key] / 4096.0; start_idx = 1; end_idx = min(400, len(avg_pulse_raw) - 1)
            if start_idx < end_idx: baseline = np.mean(avg_pulse_raw[start_idx:end_idx])
            elif len(avg_pulse_raw) > 0: baseline = avg_pulse_raw[0]
            else: baseline = 0
            avg_pulse_baselined = avg_pulse_raw - baseline; sample_time = 8e-6
            times = (np.arange(len(avg_pulse_baselined)) - pulse_arrival_sample) * sample_time
            pulse_amps = phi0_to_amp(avg_pulse_baselined); return times, pulse_amps
    except FileNotFoundError:
        # <<< MODIFIED >>>
        QMessageBox.critical(None, "File Error", f"Pulse file not found: {filename}")
        return None, None
    except Exception as e:
        # <<< MODIFIED >>>
        QMessageBox.critical(None, "File Error", f"Error loading pulse file {filename}:\n{e}")
        import traceback; traceback.print_exc() # Also print traceback to console
        return None, None


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


# === PyQt6 Fitting Thread ===
class FitThread(QThread):
    # Define signals to communicate back to the main thread
    # Signal signature: result object (MinimizerResult), or error string/object
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(object) # To send intermediate params for live update

    def __init__(self, residual_func, params_copy, fit_args, fit_method, iter_callback_proxy):
        super().__init__()
        self.residual_func = residual_func
        self.params_copy = params_copy
        self.fit_args = fit_args
        self.fit_method = fit_method
        self.iter_callback_proxy = iter_callback_proxy # Function to call iter_cb

    def run(self):
        """Execute the fitting process."""
        print("Fitting thread started...")
        try:
            # The iter_callback_proxy will handle threading details (like using self.after_idle)
            result = minimize(self.residual_func, self.params_copy, args=self.fit_args,
                              method=self.fit_method,
                              iter_cb=self.iter_callback_proxy)
            self.finished_signal.emit(result) # Emit result object on success/failure
        except Exception as e:
            print(f"Error during fitting thread ({self.fit_method}): {e}")
            import traceback
            self.error_signal.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}") # Emit error string
        print("Fitting thread finished.")


# === Main GUI Application Class ===
class TESModelViewer(QMainWindow): # Use QMainWindow
    # Signal for live plot updates from iter_callback -> main thread
    live_update_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TES Model Viewer (PyQt6 + qt-material)")
        self.resize(2000, 1500) # Adjusted default size

        # --- Data Storage ---
        self.current_pulse_filepath = None; self.current_noise_filepath = None
        self.pulse_times_raw, self.pulse_data_raw = None, None; self.raw_noise_freqs = None; self.raw_noise_psd = None
        self.pulse_times_interp, self.pulse_data_interp = None, None; self.noise_freqs_resampled, self.noise_data_resampled = None, None
        self.noise_model_arrays = {key: None for key in ALL_POSSIBLE_NOISE_KEYS}
        self.selected_noise_channel = -1 # Store index directly

        # --- Model Selection ---
        self.available_models = {"Simple": tes_simple.TES_Simple}
        if HAS_COMPOUND: self.available_models["Compound"] = tes_compound.TES_Compound
        if HAS_INTER: self.available_models["Intervening"] = tes_intervening.TES_Intervening
        if HAS_DANGLING: self.available_models["Dangling"] = tes_dangling.TES_Dangling
        self.selected_model_name = "Dangling" # Store name directly

        # --- Fit Status ---
        self.last_fit_result = None
        self.selected_fit_method = 'least_squares' # Default fit method
        self.noise_weight_factor = 1.0 # Default weight
        self._terminate_event = threading.Event()
        self.last_live_update_time = 0.0
        self.live_update_interval = 0.25
        self.fit_thread = None # Reference to the running fit thread

        # --- Model and Parameter Initialization ---
        self.tes_model = None; self.params = None; self.mdt3_core = None
        try:
            self.mdt3_core = mdt3_core.MDT3_Core()
            InitialModelClass = self.available_models[self.selected_model_name]
            self.tes_model = InitialModelClass(self.mdt3_core)
            self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
            self.initialize_params()
        except Exception as e: QMessageBox.critical(self, "Model Init Error", f"Failed to initialize mdt3 model:\n{e}"); sys.exit()

        # --- GUI Setup ---
        self.param_widgets = {} # Store refs: {name: {'slider': QSlider, 'checkbox': QCheckBox, 'val_label': QLabel, 'min': min, 'max': max}}
        self.noise_plot_lines = {}
        self.initUI()

        # --- Connect live update signal ---
        self.live_update_signal.connect(self.update_plots_live)

        # --- Attempt Auto-Load ---
        # (Use paths defined above)
        self.default_pulse_path = '/data/20250226/0004/20250226_0004_chan4107_avgpulse_test.npz'
        self.default_noise_path = '/data/20250219/noise/noise_20250219_145355_20mK_bias0v25_.npz'
        self.current_pulse_filepath = self.default_pulse_path if os.path.exists(self.default_pulse_path) else None
        self.current_noise_filepath = self.default_noise_path if os.path.exists(self.default_noise_path) else None
        if self.current_pulse_filepath: self._load_pulse_data(self.current_pulse_filepath)
        if self.current_noise_filepath: self._load_noise_data(self.current_noise_filepath)
        self.update_model_and_plots() # Initial plot update

    def initUI(self):
        """Create all GUI elements and layouts."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget) # Main vertical layout

        # --- Top Control Area ---
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0,0,0,0) # Remove margins if needed
        main_layout.addWidget(control_widget,0)

        # File Loading Row
        file_layout = QHBoxLayout()
        control_layout.addLayout(file_layout)
        pulse_load_btn = QPushButton("Load Pulse")
        pulse_load_btn.clicked.connect(self.select_pulse_file)
        self.pulse_file_label = QLabel("No Pulse File Loaded")
        self.pulse_file_label.setFrameShape(QFrame.Shape.Panel)
        self.pulse_file_label.setFrameShadow(QFrame.Shadow.Sunken)
        noise_load_btn = QPushButton("Load Noise")
        noise_load_btn.clicked.connect(self.select_noise_file)
        self.noise_file_label = QLabel("No Noise File Loaded")
        self.noise_file_label.setFrameShape(QFrame.Shape.Panel)
        self.noise_file_label.setFrameShadow(QFrame.Shadow.Sunken)
        file_layout.addWidget(pulse_load_btn)
        file_layout.addWidget(self.pulse_file_label, 1) # Stretch label
        file_layout.addWidget(noise_load_btn)
        file_layout.addWidget(self.noise_file_label, 1) # Stretch label

        # Selectors Row
        selector_layout = QHBoxLayout()
        control_layout.addLayout(selector_layout)
        selector_layout.addWidget(QLabel("TES Model:"))
        self.model_selector_combo = QComboBox()
        self.model_selector_combo.addItems(list(self.available_models.keys()))
        self.model_selector_combo.setCurrentText(self.selected_model_name)
        self.model_selector_combo.currentTextChanged.connect(self.on_model_selected) # Use text signal
        selector_layout.addWidget(self.model_selector_combo)
        selector_layout.addSpacing(15)
        selector_layout.addWidget(QLabel("Autotune Channel:"))
        self.noise_channel_combo = QComboBox()
        self.noise_channel_combo.setEnabled(True)
        self.noise_channel_combo.setFixedWidth(60)
        self.noise_channel_combo.currentIndexChanged.connect(self.on_channel_selected) # Use index signal
        selector_layout.addWidget(self.noise_channel_combo)
        selector_layout.addStretch(1) # Push elements left
        print(f"DEBUG initUI: self.noise_channel_combo type = {type(self.noise_channel_combo)}, is None? {self.noise_channel_combo is None}")

        # --- Plot Area ---
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        # ... (Setup canvas, toolbar within plot_layout) ...
        # <<< Set higher stretch factor (e.g., 4 or 5) to make plots expand vertically >>>
        main_layout.addWidget(plot_widget, 4)

        self.fig = plt.Figure(figsize=(12, 4), dpi=100)
        self.ax_pulse = self.fig.add_subplot(1, 2, 1)
        self.ax_noise = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)

        # # Create plot lines (will be populated later)
        self.line_pulse_data, = self.ax_pulse.plot([], [], 'b-', label='Pulse Data', markersize=3, alpha=0.7)
        self.line_pulse_model, = self.ax_pulse.plot([], [], 'r-', label='Pulse Model', linewidth=2)
        self.ax_pulse.set_xlabel("Time (s)"); self.ax_pulse.set_ylabel("Current (A)")
        self.ax_pulse.set_title("Average Pulse"); self.ax_pulse.grid(True); self.ax_pulse.legend()
        self.line_noise_data, = self.ax_noise.plot([], [], 'b-', label='Noise Data', markersize=3, alpha=0.7)
        self.noise_plot_lines = {}
        for key, config in NOISE_PLOT_CONFIG.items():
            line, = self.ax_noise.plot([], [], **config); self.noise_plot_lines[key] = line
        self.ax_noise.set_xlabel("Frequency (Hz)"); self.ax_noise.set_ylabel(r"Current Noise (A/$\sqrt{Hz}$)")
        self.ax_noise.set_title("Noise Spectrum"); self.ax_noise.set_yscale('log'); self.ax_noise.set_xscale('log')
        self.ax_noise.grid(True, which='both'); self.ax_noise.legend()


        # --- Bottom UI Area (Sliders in Scroll Area, Fit Controls, Log) ---
        bottom_ui_widget = QWidget()
        bottom_ui_layout = QVBoxLayout(bottom_ui_widget)
        bottom_ui_layout.setContentsMargins(0, 0, 0, 0)
        # <<< Set stretch factor 0 (minimum vertical space) >>>
        main_layout.addWidget(bottom_ui_widget, 0)

        # Sliders in Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # <<< Set Maximum Height for sliders to make this area smaller >>>
        self.scroll_area.setMaximumHeight(180)  # Adjust pixel value (e.g., 150, 200) as needed
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Preferred)  # Horizontal expand is good
        bottom_ui_layout.addWidget(self.scroll_area)  # Add scroll area first in bottom section
        self.slider_widget = QWidget()  # Widget to hold the slider layout
        self.slider_layout = QVBoxLayout(self.slider_widget)  # Layout for sliders inside widget
        self.scroll_area.setWidget(self.slider_widget)
        self.create_sliders()  # Populate sliders

        # Fit Controls Area
        fit_controls_widget = QWidget()
        fit_controls_layout = QHBoxLayout(fit_controls_widget)
        fit_controls_layout.setContentsMargins(5, 5, 5, 5)
        bottom_ui_layout.addWidget(fit_controls_widget)  # Add below scroll area

        self.fit_button = QPushButton("Fit Current Model")
        self.fit_button.clicked.connect(self.start_fit_thread)
        self.terminate_button = QPushButton("Terminate Fit")
        self.terminate_button.clicked.connect(self.request_fit_termination)
        self.terminate_button.setEnabled(False)
        self.save_button = QPushButton("Save Fit Report")
        self.save_button.clicked.connect(self.save_fit_report)
        self.save_button.setEnabled(False)
        fit_controls_layout.addWidget(self.fit_button)
        fit_controls_layout.addWidget(self.terminate_button)
        fit_controls_layout.addWidget(self.save_button)

        fit_controls_layout.addWidget(QLabel("Method:"))
        self.fit_method_combo = QComboBox()
        fit_methods = ['least_squares', 'leastsq', 'nelder', 'lbfgsb', 'powell', 'cg', 'bfgs', 'slsqp', 'differential_evolution', 'basinhopping']
        self.fit_method_combo.addItems(fit_methods)
        self.fit_method_combo.setCurrentText(self.selected_fit_method)
        # Connect signal for changes *after* populating if needed
        self.fit_method_combo.currentTextChanged.connect(self.on_fit_method_changed)
        fit_controls_layout.addWidget(self.fit_method_combo)

        self.fit_status_label = QLabel("Fit status: Idle")
        self.fit_status_label.setFrameShape(QFrame.Shape.Panel)
        self.fit_status_label.setFrameShadow(QFrame.Shadow.Sunken)
        fit_controls_layout.addWidget(self.fit_status_label, 1) # Stretch status

        # Noise Weight Slider Area
        weight_widget = QWidget()
        weight_layout = QHBoxLayout(weight_widget)
        weight_layout.setContentsMargins(5, 0, 5, 5)
        bottom_ui_layout.addWidget(weight_widget)
        weight_layout.addWidget(QLabel("Noise Weight Factor:"))
        self.noise_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_weight_slider.setRange(0, 1000) # Use 0-1000 for mapping 0.0 to 10.0
        self.noise_weight_slider.setValue(int(self.noise_weight_factor * 100)) # Initial value
        self.noise_weight_slider.valueChanged.connect(self.on_noise_weight_slider_change)
        self.noise_weight_label = QLabel(f"{self.noise_weight_factor:.2f}")
        self.noise_weight_label.setFixedWidth(40)
        weight_layout.addWidget(self.noise_weight_slider)
        weight_layout.addWidget(self.noise_weight_label)


        # Log Area
        log_group = QWidget()
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 0, 5, 5)
        bottom_ui_layout.addWidget(log_group)
        log_label = QLabel("Fit Log") # Manual label
        log_layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(150) # Set fixed height
        log_layout.addWidget(self.log_text)

    # --- Placeholder / Adapt methods below ---

    def initialize_params(self):
        # (Keep logic using PARAM_DEFS and self.params.set)
        # Make sure it uses self.params and PARAM_DEFS correctly
        if self.params is None: print("Error: Parameters object not initialized."); return
        for name, config in PARAM_DEFS.items():
            initial_gui, pmin_gui, pmax_gui, is_variable_in_gui = config
            if name in self.params:
                param = self.params[name]; internal_min = param.min if param.min is not None else -np.inf; internal_max = param.max if param.max is not None else np.inf
                pmin_actual = max(pmin_gui, internal_min); pmax_actual = min(pmax_gui, internal_max)
                if pmin_actual > pmax_actual:
                    pmin_actual = internal_min; pmax_actual = internal_max; # Revert/warn if conflict
                    if pmin_actual > pmax_actual:
                        pmin_actual = -np.inf; pmax_actual = np.inf
                initial_clipped = np.clip(initial_gui, pmin_actual, pmax_actual)
                try: param.set(value=initial_clipped, min=pmin_actual, max=pmax_actual, vary=is_variable_in_gui)
                except ValueError as ve: print(f"Error setting param '{name}': {ve}.")
        for name, param in self.params.items():
             if name not in PARAM_DEFS: param.set(vary=False)
        print("DEBUG: Parameters initialized/updated.")


    def _load_pulse_data(self, filepath):
        """Internal logic to load and process pulse data."""
        print(f"Attempting to load pulse data from: {filepath}")  # DEBUG
        if not filepath or not os.path.exists(filepath):
            self.current_pulse_filepath = None
            # Update PyQt Label if it exists
            if hasattr(self, 'pulse_file_label'): self.pulse_file_label.setText("Pulse file not found")
            self.pulse_times_raw, self.pulse_data_raw = None, None
            self.pulse_times_interp, self.pulse_data_interp = None, None
            return False

        pulse_arrival = 396  # Keep consistent
        # Assuming load_avg_pulse helper function exists globally or is imported
        times, data = load_avg_pulse(filepath, pulse_arrival_sample=pulse_arrival)

        if times is None or data is None:
            # Error message handled within load_avg_pulse via messagebox
            self.current_pulse_filepath = None  # Clear path if loading failed
            if hasattr(self, 'pulse_file_label'): self.pulse_file_label.setText("Failed to load pulse")
            return False

        self.current_pulse_filepath = filepath  # Store the successfully loaded path
        base_name = os.path.basename(filepath)
        # Update PyQt Label if it exists
        if hasattr(self, 'pulse_file_label'): self.pulse_file_label.setText(base_name)
        self.pulse_times_raw, self.pulse_data_raw = times, data

        # Interpolate for model calculation (same logic as before)
        num_interp_points = 601
        if len(times) > 1:
            t_min, t_max = times[0], times[-1]
            if t_max <= t_min:
                print(f"Warning: Invalid time range ({t_min} to {t_max}). Skipping pulse interpolation.")
                self.pulse_times_interp, self.pulse_data_interp = None, None
            else:
                self.pulse_times_interp = np.linspace(t_min, t_max, num_interp_points)
                try:
                    interp_func = interp1d(times, data, kind='linear', bounds_error=False, fill_value=0)
                    self.pulse_data_interp = interp_func(self.pulse_times_interp)
                except ValueError as e:
                    print(f"Error during pulse interpolation: {e}")
                    self.pulse_times_interp, self.pulse_data_interp = None, None
        elif len(times) == 1:
            self.pulse_times_interp, self.pulse_data_interp = times, data
        else:
            self.pulse_times_interp, self.pulse_data_interp = None, None

        print(f"Pulse data loaded successfully from {filepath}.")
        return True

        # Add this method inside the TESModelViewer class:

        # Inside class TESModelViewer:

        # Inside class TESModelViewer:

    def _load_noise_data(self, filepath):
        """Loads raw noise data, populates channel selector, and processes initial channel."""
        # Check combo exists at start (though it should have been created in initUI)
        if self.noise_channel_combo is None:
            print("Error: Noise channel combobox not initialized before loading data.")
            # Attempt to continue without it, but channel selection won't work
            pass  # Or maybe return False? Depends if channel selection is critical path

        # Check filepath validity
        if not filepath or not os.path.exists(filepath):
            self.current_noise_filepath = None
            if hasattr(self, 'noise_file_label'): self.noise_file_label.setText("Noise file not found")
            self.raw_noise_freqs, self.raw_noise_psd = None, None
            if self.noise_channel_combo: self.noise_channel_combo.clear(); self.noise_channel_combo.setEnabled(
                False)
            self.selected_noise_channel = -1
            self.noise_freqs_resampled, self.noise_data_resampled = None, None
            return False

        # Load data using helper function
        freqs, psd_all_channels = load_noise_file(filepath)

        # Check if loading succeeded
        if freqs is None or psd_all_channels is None:
            # Error message handled within load_noise_file via QMessageBox
            self.current_noise_filepath = None
            if hasattr(self, 'noise_file_label'): self.noise_file_label.setText("Failed to load noise")
            self.raw_noise_freqs, self.raw_noise_psd = None, None
            if self.noise_channel_combo: self.noise_channel_combo.clear(); self.noise_channel_combo.setEnabled(
                False)
            self.selected_noise_channel = -1
            self.noise_freqs_resampled, self.noise_data_resampled = None, None
            return False

        # Store data and update label
        self.current_noise_filepath = filepath
        base_name = os.path.basename(filepath)
        if hasattr(self, 'noise_file_label'): self.noise_file_label.setText(base_name)
        self.raw_noise_freqs = freqs
        self.raw_noise_psd = psd_all_channels
        print(f"Raw noise data loaded successfully from {filepath}.")  # Keep informative print

        # Determine number of channels safely
        num_channels = 0
        if isinstance(psd_all_channels, np.ndarray):
            if psd_all_channels.ndim == 1:
                num_channels = 1
            elif psd_all_channels.ndim > 1:
                # Check shape length before accessing index 1
                if len(psd_all_channels.shape) > 1:
                    num_channels = psd_all_channels.shape[1]
                else:  # Treat shape (N,) as 1 channel
                    num_channels = 1

        processed_ok = False  # Initialize flag

        # Explicitly check conditions before the 'if' block
        cond1_num_channels_ok = (num_channels > 0)
        # Use more robust check for QComboBox existence and type
        cond2_combo_exists_and_valid = (
                    self.noise_channel_combo is not None and isinstance(self.noise_channel_combo, QComboBox))
        final_if_condition = cond1_num_channels_ok and cond2_combo_exists_and_valid

        # Use the pre-evaluated condition
        if final_if_condition:
            # This block executes if channels > 0 and combo widget is valid
            self.noise_channel_combo.blockSignals(True)  # Prevent callback while populating
            self.noise_channel_combo.clear()
            channel_indices_str = [str(i) for i in range(num_channels)]
            self.noise_channel_combo.addItems(channel_indices_str)
            self.noise_channel_combo.setEnabled(True)
            # Set initial selection (e.g., channel 11 if available, else 0)
            initial_channel_index = 11 if 11 < num_channels else 0
            self.noise_channel_combo.setCurrentIndex(initial_channel_index)
            self.selected_noise_channel = initial_channel_index  # Store index
            self.noise_channel_combo.blockSignals(False)
            print(f"Noise channels available: {num_channels}. Set to channel index {self.selected_noise_channel}.")

            # Process the initially selected channel
            processed_ok = self._process_noise_channel(initial_channel_index)

        else:
            # This block executes if num_channels is 0 OR if self.noise_channel_combo is None/invalid
            print(
                f"Warning: Cannot populate channel selector (Num channels: {num_channels}, Combo valid: {cond2_combo_exists_and_valid}).")
            if self.noise_channel_combo:  # Check again before disabling
                self.noise_channel_combo.clear()
                self.noise_channel_combo.setEnabled(False)
            self.selected_noise_channel = -1;
            self.noise_freqs_resampled, self.noise_data_resampled = None, None
            processed_ok = False  # Ensure it's False if this path is taken

        return processed_ok  # Return True only if processing succeeded in the IF block


    def _process_noise_channel(self, channel_index):
            if self.raw_noise_psd is None or self.raw_noise_freqs is None or channel_index < 0: print(
                "Cannot process noise channel: Raw data missing or invalid channel index."); self.noise_freqs_resampled, self.noise_data_resampled = None, None; return False
            print(f"Processing noise data for channel {channel_index}...")
            psd_single_channel = None
            try:
                if self.raw_noise_psd.ndim == 1 and channel_index == 0:
                    psd_single_channel = self.raw_noise_psd
                elif self.raw_noise_psd.ndim > 1 and channel_index < self.raw_noise_psd.shape[1]:
                    psd_single_channel = self.raw_noise_psd[:, channel_index]
                else:
                    raise IndexError
                with np.errstate(invalid='ignore'):
                    amp_noise_density = phi0_to_amp(np.sqrt(psd_single_channel))
                amp_noise_density = np.nan_to_num(amp_noise_density, nan=0.0)
                num_resample_points = 1200;
                self.noise_freqs_resampled, self.noise_data_resampled = logarithmic_resample(self.raw_noise_freqs,
                                                                                             amp_noise_density,
                                                                                             num_points=num_resample_points)
                if self.noise_data_resampled is not None and not np.any(self.noise_data_resampled > 0): print(
                    f"Warning: Processed noise data for channel {channel_index} contains no positive values.")
                print(f"Finished processing channel {channel_index}.");
                return True
            except IndexError:
                QMessageBox.critical("Channel Error",
                                     f"Invalid channel index {channel_index}"); self.noise_freqs_resampled, self.noise_data_resampled = None, None; return False
            except Exception as e:
                QMessageBox.critical("Processing Error", f"Error processing noise channel {channel_index}:\n{e}");
                self.noise_freqs_resampled, self.noise_data_resampled = None, None;
                return False

            # Inside class TESModelViewer:

            # Inside class TESModelViewer:

    def create_sliders(self):
        """Recreate sliders/checkboxes in scroll area with Parameter, Unit, Min, Current, Max headers."""
        # Clear previous layout content safely
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                layout_item = item.layout();
                if layout_item is not None: pass  # Assuming grid gets deleted okay

        self.param_widgets = {}
        param_grid_layout = QGridLayout()  # Recreate grid layout

        # --- Define Column Indices for Readability ---
        COL_VARY = 0;
        COL_NAME = 1;
        COL_UNIT = 2;
        COL_MIN_LABEL = 3
        COL_SLIDER = 4;
        COL_CUR_LABEL = 5;
        COL_MAX_LABEL = 6

        # --- Set Column Stretch Factors ---
        param_grid_layout.setColumnStretch(COL_VARY, 0)  # Checkbox width
        param_grid_layout.setColumnStretch(COL_NAME, 2)  # Parameter name (allow some stretch)
        param_grid_layout.setColumnStretch(COL_UNIT, 1)  # Unit (allow some stretch)
        param_grid_layout.setColumnStretch(COL_MIN_LABEL, 0)  # Min value width
        param_grid_layout.setColumnStretch(COL_SLIDER, 5)  # Slider (most stretch)
        param_grid_layout.setColumnStretch(COL_CUR_LABEL, 0)  # Current value width
        param_grid_layout.setColumnStretch(COL_MAX_LABEL, 0)  # Max value width
        self.slider_layout.addLayout(param_grid_layout)  # Add grid to the VBox layout

        # --- ADD HEADER ROW (Row 0) ---
        vary_header = QLabel("<b>Vary?</b>");
        vary_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_header = QLabel("<b>Parameter</b>");
        name_header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        unit_header = QLabel("<b>Unit</b>");
        unit_header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)  # <<< ADDED
        min_header = QLabel("<b>Min</b>");
        min_header.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        current_header = QLabel("<b>Current</b>");
        current_header.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        max_header = QLabel("<b>Max</b>");
        max_header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Add headers to the grid layout at row 0
        param_grid_layout.addWidget(vary_header, 0, COL_VARY)
        param_grid_layout.addWidget(name_header, 0, COL_NAME)
        param_grid_layout.addWidget(unit_header, 0, COL_UNIT)  # <<< ADDED
        param_grid_layout.addWidget(min_header, 0, COL_MIN_LABEL)
        param_grid_layout.addWidget(current_header, 0, COL_CUR_LABEL)
        param_grid_layout.addWidget(max_header, 0, COL_MAX_LABEL)
        # --- END HEADER ROW ---

        # --- START PARAMETER ROWS FROM Row 1 ---
        row = 1
        for name, config in PARAM_DEFS.items():
            if name not in self.params: continue  # Skip if param not in current model

            p_initial, p_min, p_max, initial_vary = config
            current_val = self.params[name].value;
            current_val = np.clip(current_val, p_min, p_max)
            current_vary_status = self.params[name].vary

            # Create widgets for this parameter row
            chk = QCheckBox();
            chk.setChecked(current_vary_status)
            chk.stateChanged.connect(lambda state, n=name: self.on_vary_checkbox_change(n, state))

            name_label = QLabel(name)  # Just the name, no colon needed

            # <<< ADD Unit Label >>>
            unit_str = UNITS.get(name, "")  # Get unit from map, default empty
            unit_label = QLabel(unit_str)
            unit_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            # <<< END ADD >>>

            min_label = QLabel(f"{p_min:.2e}")
            slider = QSlider(Qt.Orientation.Horizontal);
            slider.setRange(0, 1000)
            slider_val = self.map_param_to_slider(current_val, p_min, p_max)
            slider.setValue(int(slider_val))
            slider.valueChanged.connect(lambda val, n=name, s=slider: self.on_slider_value_change(n, s, val))

            val_label = QLabel(f"{current_val:.3e}");
            val_label.setMinimumWidth(80)
            max_label = QLabel(f"{p_max:.2e}")

            # Add widgets to the grid layout using the 'row' variable and column constants
            param_grid_layout.addWidget(chk, row, COL_VARY)
            param_grid_layout.addWidget(name_label, row, COL_NAME)
            param_grid_layout.addWidget(unit_label, row, COL_UNIT)  # <<< ADDED
            param_grid_layout.addWidget(min_label, row, COL_MIN_LABEL,
                                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            param_grid_layout.addWidget(slider, row, COL_SLIDER)
            param_grid_layout.addWidget(val_label, row, COL_CUR_LABEL,
                                        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            param_grid_layout.addWidget(max_label, row, COL_MAX_LABEL,
                                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            # Store widgets/variables for later access
            self.param_widgets[name] = {'checkbox': chk, 'slider': slider, 'val_label': val_label, 'min': p_min,
                                        'max': p_max}
            row += 1
        # <<< END PARAMETER ROWS >>>

        self.slider_layout.addStretch(1)  # Push sliders towards the top
        print(f"DEBUG: Created {row - 1} parameter rows in sliders area.")

    # --- Slots for GUI interactions ---
    @pyqtSlot()
    def select_pulse_file(self):
        """Opens file dialog for pulse file, starting in relevant directory."""
        start_dir = ""  # Default fallback
        if self.current_pulse_filepath and os.path.exists(os.path.dirname(self.current_pulse_filepath)):
            start_dir = os.path.dirname(self.current_pulse_filepath)
        elif self.default_pulse_path and os.path.exists(os.path.dirname(self.default_pulse_path)):
            start_dir = os.path.dirname(self.default_pulse_path)

        # <<< CHECK THIS CALL: Ensure first argument is 'self' >>>
        filepath, _ = QFileDialog.getOpenFileName(
            self,  # Should be self (the parent window)
            "Select Pulse NPZ File",  # Window title
            start_dir,  # Starting directory
            "Numpy NPZ (*.npz);;All Files (*)"  # Filter
        )
        # <<< END CHECK >>>
        if filepath:
            print(f"Pulse file selected: {filepath}")
            if self._load_pulse_data(filepath):
                self.update_model_and_plots()

    @pyqtSlot()
    def select_noise_file(self):
        """Opens file dialog for noise file, starting in relevant directory."""
        start_dir = ""  # Default fallback
        # Prioritize directory of currently loaded file
        if self.current_noise_filepath and os.path.exists(os.path.dirname(self.current_noise_filepath)):
            start_dir = os.path.dirname(self.current_noise_filepath)
        # Fallback to directory of default file path if no current file
        elif self.default_noise_path and os.path.exists(os.path.dirname(self.default_noise_path)):
            start_dir = os.path.dirname(self.default_noise_path)
        # else: start_dir remains ""

        filepath, _ = QFileDialog.getOpenFileName(
            self,  # Should be self (the parent window)
            "Select Noise NPZ File",  # Window title
            start_dir,  # Starting directory
            "Numpy NPZ (*.npz);;All Files (*)"  # Filter
        )
        if filepath:
            print(f"Noise file selected: {filepath}")
            load_success = self._load_noise_data(filepath)
            # <<< DEBUG: Check if update will be called >>>
            print(f"DEBUG: _load_noise_data success = {load_success}. Calling update_model_and_plots? {load_success}")
            # <<< END DEBUG >>>
            if load_success:
                self.update_model_and_plots()


    @pyqtSlot(str)
    def on_model_selected(self, model_name):
        """Slot when TES model combobox changes."""
        print(f"Switching to TES model: {model_name}")
        if model_name in self.available_models:
            NewModelClass = self.available_models[model_name]
            try:
                old_params_values = {};
                if self.params is not None: old_params_values = self.params.valuesdict()
                self.tes_model = NewModelClass(self.mdt3_core)
                self.params = self.mdt3_core.makeDefaultParamsDict(num_sets=1)
                self.initialize_params() # Apply PARAM_DEFS defaults/settings/vary
                params_restored = 0
                for name, param in self.params.items(): # Restore previous values
                    if name in old_params_values:
                        old_value = old_params_values[name]; current_min = param.min if param.min is not None else -np.inf; current_max = param.max if param.max is not None else np.inf
                        param.value = np.clip(old_value, current_min, current_max); params_restored += 1
                print(f"  Restored values for {params_restored} parameters.")
                self.create_sliders() # Refresh GUI sliders for new model/params
                self.update_model_and_plots()
            except Exception as e: QMessageBox.critical(self, "Model Switch Error", f"Could not switch model or parameters:\n{e}")
        else: QMessageBox.warning(self, "Model Error", f"Selected model '{model_name}' is not available.")
        self.selected_model_name = model_name # Update internal state AFTER potentially successful switch

    @pyqtSlot(int)
    def on_channel_selected(self, index):
        """Slot when noise channel combobox changes."""
        print(f"\nDEBUG: on_channel_selected triggered, received index: {index}")  # DEBUG
        if index >= 0 and self.noise_channel_combo:  # Check index is valid
            try:
                channel_text = self.noise_channel_combo.itemText(index)
                print(f"DEBUG: Selected item text: '{channel_text}'")  # DEBUG
                channel_num = int(channel_text)  # Convert text (which is the channel number as string) to integer index
                print(f"DEBUG: Converted channel index: {channel_num}")  # DEBUG
                self.selected_noise_channel = channel_num  # Update internal state variable

                print(f"DEBUG: Calling _process_noise_channel({channel_num})")  # DEBUG
                if self._process_noise_channel(channel_num):  # Re-process data for the selected channel
                    print(f"DEBUG: Calling update_model_and_plots after channel {channel_num} processing.")  # DEBUG
                    self.update_model_and_plots()  # Update plots if processing was successful
                else:
                    print(f"DEBUG: _process_noise_channel FAILED for channel {channel_num}.")  # DEBUG
            except ValueError:
                print(f"ERROR: Could not convert selected item text '{channel_text}' to an integer.")  # DEBUG
            except Exception as e:
                print(f"Error processing channel selection: {e}")  # DEBUG
                import traceback
                traceback.print_exc()  # Print full traceback if other error occurs
        else:
            print(f"DEBUG: Invalid index ({index}) received or combobox not ready in on_channel_selected.")

    @pyqtSlot(str)
    def on_fit_method_changed(self, method_name):
        """Slot when fit method combobox changes."""
        self.selected_fit_method = method_name
        print(f"Fit method set to: {self.selected_fit_method}")

    @pyqtSlot(str, int) # Slot receiving name and Qt.CheckState enum (0 or 2 usually)
    def on_vary_checkbox_change(self, param_name, state):
        """Slot when a vary checkbox is toggled."""
        if param_name in self.params:
            new_vary_state = (state == Qt.CheckState.Checked.value) # Convert enum to bool
            try:
                self.params[param_name].vary = new_vary_state
                print(f"Parameter '{param_name}' vary set to: {new_vary_state}")
            except Exception as e: print(f"Error setting vary for {param_name}: {e}")

    @pyqtSlot(str, object, int) # Slot receiving name, slider object, int value
    def on_slider_value_change(self, param_name, slider_widget, value):
        """Slot when a slider's value changes."""
        if param_name in self.param_widgets and param_name in self.params:
            details = self.param_widgets[param_name]
            p_min_gui, p_max_gui = details['min'], details['max']
            # Map 0-1000 slider value back to parameter value
            new_param_val = self.map_slider_to_param(value, p_min_gui, p_max_gui)
            current_param_val = self.params[param_name].value
            new_param_val_clipped = np.clip(new_param_val, self.params[param_name].min, self.params[param_name].max)

            # Update label immediately
            details['val_label'].setText(f"{new_param_val_clipped:.3e}")

            # Update underlying lmfit parameter only if changed significantly
            if not np.isclose(current_param_val, new_param_val_clipped):
                self.params[param_name].value = new_param_val_clipped
                # Update plots (might make this optional or delayed)
                self.update_model_and_plots()


    @pyqtSlot(int) # Slot receiving int value from slider
    def on_noise_weight_slider_change(self, value):
        """Slot when noise weight slider changes."""
        # Map slider value (0-1000) to weight factor (0-10)
        self.noise_weight_factor = value / 100.0
        if self.noise_weight_label:
            self.noise_weight_label.setText(f"{self.noise_weight_factor:.2f}")

    # --- Fitting Logic Slots & Methods ---
    @pyqtSlot()
    def start_fit_thread(self):
        """Starts the fitting process in a separate QThread."""
        self._terminate_event.clear()
        # Update GUI state (buttons, status)
        self.fit_button.setEnabled(False)
        self.terminate_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.fit_status_label.setText("Fit status: Preparing...")
        self._update_log(f"--- Starting Fit ({self.selected_model_name}) ---")
        self.last_fit_result = None

        # Data Validation
        if self.pulse_times_interp is None or self.pulse_data_interp is None or len(self.pulse_times_interp) == 0:
            QMessageBox.critical(self, "Fit Error", "Valid pulse data must be loaded."); self.fit_status_label.setText("Fit status: Error - Load pulse data"); self._update_log("Error: Missing pulse data."); self.fit_button.setEnabled(True); self.terminate_button.setEnabled(False); return
        if self.noise_freqs_resampled is None or self.noise_data_resampled is None or len(self.noise_freqs_resampled) == 0:
            QMessageBox.critical(self, "Fit Error", "Valid noise data must be loaded."); self.fit_status_label.setText("Fit status: Error - Load noise data"); self._update_log("Error: Missing noise data."); self.fit_button.setEnabled(True); self.terminate_button.setEnabled(False); return
        if self.tes_model is None or self.params is None:
            QMessageBox.critical(self, "Fit Error", "Model/Params not initialized."); self.fit_status_label.setText("Fit status: Error - Model/Params invalid"); self._update_log("Error: Model/Params invalid."); self.fit_button.setEnabled(True); self.terminate_button.setEnabled(False); return

        varying_params = [name for name, param in self.params.items() if param.vary];
        if not varying_params: QMessageBox.warning(self, "Fit Warning", "No parameters set to vary."); self._update_log("Warning: No varying parameters.")
        else: self._update_log(f"Varying: {varying_params}")

        # Get fit arguments
        params_to_fit = self.params.copy()
        fit_method_to_use = self.fit_method_combo.currentText() # Get from combobox
        current_noise_weight = self.noise_weight_slider.value() / 100.0 # Get from slider
        self._update_log(f"Using Fit Method: {fit_method_to_use}")
        self._update_log(f"Using Noise Weight Factor: {current_noise_weight:.3f}")

        fit_args = (self.tes_model, self.pulse_times_interp, self.pulse_data_interp,
                    self.noise_freqs_resampled, self.noise_data_resampled,
                    current_noise_weight)

        self.last_live_update_time = time.time()
        self.fit_status_label.setText("Fit status: Fitting...")
        self._update_log(f"Fitting with method '{fit_method_to_use}'...")

        # Create and start the thread
        self.fit_thread = FitThread(self.residual_both_gui, params_to_fit, fit_args,
                                    fit_method_to_use, self.iter_callback_proxy)
        self.fit_thread.finished_signal.connect(self._on_fit_complete)
        self.fit_thread.error_signal.connect(self._on_fit_error)
        # Live update signal connection is done in __init__
        self.fit_thread.start()


    def iter_callback_proxy(self, params, iteration, resid, *args, **kws):
        """Proxy for iter_callback that emits a signal for thread-safe GUI update."""
        if self._terminate_event.is_set():
             print(f"Fit termination requested at iteration {iteration}. Stopping.")
             return True # Signal lmfit to stop

        current_time = time.time()
        if current_time - self.last_live_update_time > self.live_update_interval:
            self.last_live_update_time = current_time
            # Emit signal with *copied* params object
            self.live_update_signal.emit(params.copy())

        return None # Continue fitting

    # Slot to receive signal from iter_callback_proxy
    @pyqtSlot(object)
    def update_plots_live(self, live_params):
        """Updates only the model lines on the plots using intermediate params. Runs in GUI thread."""
        # (Keep the logic from the previous tkinter version)
        if self.tes_model is None or live_params is None: return
        live_pulse_model_y = None
        if self.pulse_times_interp is not None and len(self.pulse_times_interp) > 0:
            try:
                pulse_result = self.tes_model.calc_pulse(live_params, self.pulse_times_interp);
                if isinstance(pulse_result, (list, tuple)) and len(pulse_result) > 0 and pulse_result[0] is not None:
                    live_pulse_model_y = pulse_result[0]
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
                      if not isinstance(data_array, np.ndarray):
                          data_array = np.array(data_array).astype(float);
                      else:
                          data_array = data_array.astype(float)
                      valid_idx_model = np.isfinite(data_array) & (data_array > 0) & np.isfinite(freqs) & (freqs > 0);
                      if np.any(valid_idx_model):
                          x_data = freqs[valid_idx_model]; y_data = data_array[valid_idx_model];
                          if len(x_data)>0: line_obj.set_data(x_data, y_data)
                          else: line_obj.set_data([], [])
                      else: line_obj.set_data([], [])
                 else: line_obj.set_data([], [])
        else:
            for line_obj in self.noise_plot_lines.values(): line_obj.set_data([], [])
        self.canvas.draw_idle()


    # Slot to receive signal from FitThread when finished
    @pyqtSlot(object)
    def _on_fit_complete(self, result):
        """Slot executed after fitting finishes successfully or otherwise."""
        print("Fit complete signal received. Updating GUI...")
        self.terminate_button.setEnabled(False)
        self.fit_button.setEnabled(True)
        self.log_text.clear() # Clear previous log

        if result and hasattr(result, 'success'):
            try: report = fit_report(result); self._update_log(report)
            except Exception as report_error: print(f"Error generating fit report: {report_error}"); self._update_log(f"--- Fit Report Error ---\n{report_error}")

            if result.success:
                self.fit_status_label.setText(f"Fit status: Success! (χ²={result.chisqr:.3e}, Nfev={result.nfev})"); print("\n--- Fit Successful ---")
                self.params = result.params # IMPORTANT: Update main params object
                self.last_fit_result = result # Store for saving
                self.save_button.setEnabled(True)
                self.update_sliders_from_params() # Sync GUI controls
                self.update_model_and_plots() # Update plots with final fit
            else: # Fit failed or terminated
                self.fit_status_label.setText(f"Fit status: {'Terminated' if self._terminate_event.is_set() else 'Failed'} - {result.message}")
                self._update_log(f"\n--- Fit {'Terminated' if self._terminate_event.is_set() else 'Failed'} ---\nMessage: {result.message}")
                print(f"\n--- Fit {'Terminated' if self._terminate_event.is_set() else 'Failed'} ---")
                self.last_fit_result = None # No valid result to save
                self.save_button.setEnabled(False)
                # Optionally update plots with the parameters result has (end state)
                # self.params = result.params
                # self.update_sliders_from_params()
                # self.update_model_and_plots()
        else:
             status_msg = "Fit status: Error - Invalid result object."; self.fit_status_label.setText(status_msg); self._update_log(status_msg); print("Error: Invalid result object received.")
             self.last_fit_result = None; self.save_button.setEnabled(False)

        self.fit_thread = None # Clear thread reference
        print("GUI update complete.")

    # Slot to receive error signal from FitThread
    @pyqtSlot(str)
    def _on_fit_error(self, error_message):
        """Slot executed if fitting thread raises an exception."""
        print(f"Fit error signal received: {error_message}")
        self.terminate_button.setEnabled(False)
        self.fit_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.last_fit_result = None
        QMessageBox.critical(self, "Fit Error", f"An unexpected error occurred during fitting:\n{error_message}")
        status_msg = f"Fit status: Thread Error"; self.fit_status_label.setText(status_msg); self._update_log(f"\n--- Fit Thread Error ---\n{error_message}")
        self.fit_thread = None # Clear thread reference

    @pyqtSlot()
    def request_fit_termination(self):
        """Sets the termination event when Terminate button clicked."""
        if self.fit_thread and self.fit_thread.isRunning():
             print("Requesting fit termination...")
             self._update_log(">>> Requesting Fit Termination <<<")
             self.fit_status_label.setText("Fit status: Terminating...")
             self._terminate_event.set() # Signal the event
             self.terminate_button.setEnabled(False) # Prevent multiple clicks
        else:
            print("No active fit thread to terminate.")


    @pyqtSlot()
    def save_fit_report(self):
        """Saves the last successful fit report to a text file."""
        if self.last_fit_result is None or not self.last_fit_result.success:
            QMessageBox.information(self, "Save Report", "No successful fit result available to save.")
            return
        if not self.current_pulse_filepath or not self.current_noise_filepath:
             if QMessageBox.question(self, "Save Report Warning", "Pulse/Noise file paths missing.\nSave report without full context?") == QMessageBox.StandardButton.No:
                 return

        model_name = self.model_selector_combo.currentText() # Read from combobox
        chan_num = self.noise_channel_combo.currentData() # Read from combobox (assuming index is stored or use currentText)
        # Correct way to get channel if just index is stored:
        try: chan_num = int(self.noise_channel_combo.currentText()) if self.noise_channel_combo.currentIndex() >= 0 else -1
        except: chan_num = -1
        chan_str = f"chan{chan_num}" if chan_num >= 0 else "chanNA"
        default_filename = f"fit_report_{model_name}_{chan_str}.txt"

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Fit Report As", default_filename, "Text Files (*.txt);;All Files (*)")

        if not save_path: self._update_log("Save report cancelled."); return

        print(f"Saving fit report to: {save_path}")
        try:
            report_str = fit_report(self.last_fit_result)
            now = datetime.datetime.now()
            header = (f"Fit Report - {now.strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n" +
                      f"Model:       {model_name}\n" + f"Pulse File:  {self.current_pulse_filepath or 'N/A'}\n" +
                      f"Noise File:  {self.current_noise_filepath or 'N/A'}\n" + f"Noise Chan:  {chan_str}\n" +
                      "-"*50 + "\n\n")
            with open(save_path, 'w') as f: f.write(header); f.write(report_str)
            QMessageBox.information(self, "Save Report", f"Fit report saved to:\n{os.path.basename(save_path)}")
            self._update_log(f"Fit report saved to {os.path.basename(save_path)}")
        except Exception as e: QMessageBox.critical(self, "Save Error", f"Failed to save fit report:\n{e}"); self._update_log(f"Error saving fit report: {e}")

    @pyqtSlot()
    def clear_log_window(self):
        """Clears the content of the fit log QTextEdit."""
        if self.log_text:
            self.log_text.clear()
            # Optionally add a confirmation message back to the log
            # self._update_log("Log cleared.")
            print("Fit log cleared.")

    # --- Helper methods for parameter mapping, log update ---
    def map_param_to_slider(self, param_val, p_min, p_max):
        if p_max <= p_min: return 500.0; param_val_clamped = np.clip(param_val, p_min, p_max)
        if np.isclose(param_val_clamped, p_min):
            return 0.0;
            if np.isclose(param_val_clamped, p_max): return 1000.0
        return 1000.0 * (param_val_clamped - p_min) / (p_max - p_min)

    def map_slider_to_param(self, slider_val, p_min, p_max):
        if p_max <= p_min: return p_min; slider_val_clamped = np.clip(float(slider_val), 0.0, 1000.0)
        return p_min + (slider_val_clamped / 1000.0) * (p_max - p_min)

    def update_sliders_from_params(self):
        """Updates GUI sliders/checkboxes from self.params."""
        print("Updating sliders/checkboxes from parameters...")
        if self.params is None: return
        for name, param in self.params.items():
            if name in self.param_widgets:
                details = self.param_widgets[name]; p_min_gui, p_max_gui = details['min'], details['max']
                # Update Slider Position
                slider_val_mapped = self.map_param_to_slider(param.value, p_min_gui, p_max_gui)
                # Prevent triggering valueChanged signal while setting programmatically
                details['slider'].blockSignals(True)
                details['slider'].setValue(int(slider_val_mapped))
                details['slider'].blockSignals(False)
                # Update Value Label
                details['val_label'].setText(f"{param.value:.3e}")
                # Update Checkbox State
                if 'checkbox' in details:
                     details['checkbox'].blockSignals(True)
                     details['checkbox'].setChecked(param.vary)
                     details['checkbox'].blockSignals(False)


    @pyqtSlot(str) # Use pyqtSlot decorator for clarity
    def _update_log(self, message):
        """Appends messages to the log window safely."""
        if self.log_text:
            self.log_text.append(message) # QTextEdit uses append
            self.log_text.ensureCursorVisible() # Scroll to end


    # --- Core Calculation/Plotting Update ---
    # Need to adapt residual_both_gui to be standalone or passed correctly
    # Keep update_model_and_plots logic similar to tkinter version, using Qt equivalents
    def residual_both_gui(self, params, model_instance, tvals, pdata, fvals, ndata, noise_weight_factor):
        # (Keep implementation from tkinter version - it's model/math logic)
        try:
            pulse_resid = np.full_like(pdata, np.nan);
            try:
                pulse_result = model_instance.calc_pulse(params, tvals);
                if pulse_result is not None and len(pulse_result) > 0 and pulse_result[0] is not None:
                    pulse_model = pulse_result[0];
                    if np.any(~np.isfinite(pulse_model)): pass # Keep NaN
                    else: pulse_resid = pulse_model - pdata
            except Exception: pass # Keep NaN
            noise_resid = np.full_like(ndata, np.nan);
            try:
                noise_result = model_instance.calc_noise(params, fvals);
                if noise_result is not None and len(noise_result) > 0 and noise_result[0] is not None:
                    noise_model_total = noise_result[0];
                    if np.any(~np.isfinite(noise_model_total)): pass # Keep NaN
                    else: noise_resid = noise_model_total - ndata
            except Exception: pass # Keep NaN
            pulse_std = np.std(pdata); noise_std = np.std(ndata); pulse_var_weight = 1.0 / pulse_std**2 if np.isfinite(pulse_std) and pulse_std > 1e-30 else 1.0; noise_var_weight = 1.0 / noise_std**2 if np.isfinite(noise_std) and noise_std > 1e-30 else 1.0
            weighted_pulse_resid = pulse_resid * pulse_var_weight; weighted_noise_resid = noise_resid * noise_var_weight * noise_weight_factor
            combined_residuals = np.concatenate((np.ravel(weighted_pulse_resid), np.ravel(weighted_noise_resid)))
            bad_indices = ~np.isfinite(combined_residuals);
            if np.any(bad_indices): combined_residuals[bad_indices] = 1e18
            return combined_residuals.astype(np.float64)
        except Exception as e: print(f"Error inside residual_both_gui: {e}"); total_len = len(pdata) + len(ndata); return np.full(total_len, 1e18)

    def update_model_and_plots(self):
        """Recalculates model predictions and updates the plots."""
        # (Keep logic from tkinter version, ensuring plot updates use self.canvas.draw_idle())
        if self.tes_model is None or self.params is None: return

        # Pre-checks
        calculation_valid = True; warning_messages = []; L_I = None
        try:
            p=self.params.valuesdict(); L=p.get('L_0',0); R_0=p.get('R_0_0',0); T_tes=p.get('T_tes_0',0); T_bath=p.get('T_bath_0',0); alpha_I=p.get('alpha_I_0',0); n_mem=p.get('n_mem_0',1)
            if L<=1e-12: calculation_valid=False; warning_messages.append("L<=0")
            if R_0<=1e-9: calculation_valid=False; warning_messages.append("R0<=0")
            if T_tes<=T_bath: calculation_valid=False; warning_messages.append("T_tes<=T_bath")
            if calculation_valid and T_tes>0 and n_mem!=0:
                try:
                    L_I=(alpha_I/n_mem)*(1.0-(T_bath/T_tes)**n_mem);
                    if np.isclose(L_I,1.0,atol=1e-6):
                        calculation_valid=False; warning_messages.append("L_I~=1")
                    elif np.isclose(L_I,0.0,atol=1e-9):
                        calculation_valid=False; warning_messages.append("L_I~=0")
                except: calculation_valid=False; warning_messages.append("L_I calc error")
            elif calculation_valid: calculation_valid=False; warning_messages.append("Cannot calc L_I")
        except Exception as e: calculation_valid = False; warning_messages.append(f"Pre-check Error: {e}")

        # Calculations
        pulse_model_y = None; self.noise_model_arrays = {key: None for key in ALL_POSSIBLE_NOISE_KEYS}; noise_calculation_successful = False
        if not calculation_valid: print(f"Skipping model calculation: {warning_messages}")
        else:
            if self.pulse_times_interp is not None and len(self.pulse_times_interp) > 0:
                try:
                    pulse_result = self.tes_model.calc_pulse(self.params, self.pulse_times_interp);
                    if isinstance(pulse_result, (list, tuple)) and len(pulse_result) > 0 and pulse_result[0] is not None:
                        pulse_model_y = pulse_result[0]
                    else:
                        pulse_model_y = np.full_like(self.pulse_times_interp, np.nan)
                except Exception as e:
                    print(f"Error(Pulse): {e}"); pulse_model_y = np.full_like(self.pulse_times_interp, np.nan)
            if self.noise_freqs_resampled is not None and len(self.noise_freqs_resampled) > 0:
                try:
                    noise_results = self.tes_model.calc_noise(self.params, self.noise_freqs_resampled); model_type = type(self.tes_model); expected_names = NOISE_COMPONENT_MAP.get(model_type)
                    if expected_names and isinstance(noise_results, (list, tuple)) and len(noise_results) >= len(expected_names):
                        for i, name in enumerate(expected_names):
                             if name in self.noise_model_arrays: self.noise_model_arrays[name] = noise_results[i]
                        noise_calculation_successful = True
                except Exception as e: print(f"Error(Noise): {e}")

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
                      if not isinstance(data_array, np.ndarray):
                          data_array = np.array(data_array).astype(float);
                      else:
                          data_array = data_array.astype(float)
                      valid_idx_model = np.isfinite(data_array) & (data_array > 0) & np.isfinite(freqs) & (freqs > 0);
                      if np.any(valid_idx_model):
                          x_data = freqs[valid_idx_model]; y_data = data_array[valid_idx_model];
                          if len(x_data)>0: line_obj.set_data(x_data, y_data); can_use_log_y = True
                          else: line_obj.set_data([], [])
                      else: line_obj.set_data([], [])
                 else: line_obj.set_data([], [])
        else:
            for line_obj in self.noise_plot_lines.values(): line_obj.set_data([], [])

        # Adjust Axes
        try: self.ax_pulse.relim(); self.ax_pulse.autoscale_view(); # Basic autoscale for pulse
        except Exception as e: print(f"Warning: Error adjusting pulse axes limits: {e}")
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
        self.canvas.draw_idle() # Use draw_idle here

    # --- Other methods (map_*, handle_arrow_key, etc. need PyQt equivalents if arrow key on slider is desired) ---
    # Placeholder for mapping logic, adjust as needed for QSlider
        # Inside class TESModelViewer(QMainWindow):

    def map_param_to_slider(self, param_val, p_min, p_max):
        """Maps parameter value to the slider's 0-1000 integer range."""
        # Handle invalid parameter range first
        if p_max <= p_min:
            # If the parameter range itself is invalid, return middle of slider
            return 500  # Return integer for QSlider

        # Clip the parameter value to the defined min/max range FIRST
        param_val_clamped = np.clip(param_val, p_min, p_max)

        # Check boundaries using the clamped value
        # Use isclose for floating point comparisons at the boundaries
        if np.isclose(param_val_clamped, p_min):
            return 0  # Return integer 0 for slider min
        if np.isclose(param_val_clamped, p_max):
            return 1000  # Return integer 1000 for slider max

        # Calculate slider position for values within the range
        # Denominator (p_max - p_min) is guaranteed to be positive here
        slider_float = 1000.0 * (param_val_clamped - p_min) / (p_max - p_min)

        # Return as integer for QSlider
        return int(round(slider_float))

    def map_slider_to_param(self, slider_val, p_min, p_max):
        """Maps slider 0-1000 integer value to parameter float range."""
        # Handle invalid parameter range first
        if p_max <= p_min:
            return p_min  # Return min if range invalid

        # Ensure slider value is within 0-1000 (it should be, but belt-and-braces)
        slider_val_clamped = np.clip(int(slider_val), 0, 1000)

        # Calculate the parameter value using the clamped slider value
        return p_min + (slider_val_clamped / 1000.0) * (p_max - p_min)


# === Run the Application ===
# === Run the Application ===
if __name__ == "__main__":
    if 'DummyMDT3' in str(mdt3_core.__class__):
        print("Exiting because base mdt3 library could not be loaded.")
        sys.exit()

    app = QApplication(sys.argv)

    # Apply qt-material stylesheet
    theme = 'dark_lightgreen.xml' # Or your preferred dark theme like dark_blue.xml etc.
    extra = {'density_scale': '-1'} # Optional: Adjust scaling (-2, -1, 0, 1, 2...)
    try:
        apply_stylesheet(app, theme=theme, extra=extra)
        print(f"Applied qt-material theme: {theme}")

        # <<< ADDED/REFINED: Custom QSS for ComboBox font color >>>
        # Get the current stylesheet to append rules, ensuring we don't overwrite theme
        current_stylesheet = app.styleSheet()

        # Define custom rules targeting combobox parts
        # Use {{ }} to escape curly braces within the f-string if needed,
        # but simple strings are fine here.
        custom_qss = """
            QComboBox {
                color: white; /* Sets text color of the currently selected item shown in the box */
                /* You could force background/border here if needed, e.g.: */
                /* background-color: #37474F; */
                /* border: 1px solid #78909C; */
            }
            /* Target the dropdown list view */
            QComboBox QAbstractItemView {
                color: white; /* Sets the default text color for items in the list */
                background-color: #263238; /* Set a dark background for the dropdown list */
                /* qt-material usually handles selection colors well, but you could force it: */
                /* selection-background-color: #546E7A; */
                /* selection-color: white; */ /* Text color for selected item in list */
            }
            /* More specific rule for items might sometimes be needed */
            QComboBox QAbstractItemView::item {
                color: white; /* Ensure individual items are white */
                /* Add padding or min-height if items feel cramped */
                 min-height: 20px;
                 padding: 2px 5px;
            }
        """
        # Append custom rules to the existing stylesheet
        app.setStyleSheet(current_stylesheet + custom_qss)
        print("Applied custom QSS for ComboBox text color.")
        # <<< END ADDITION >>>

    except Exception as e:
        print(f"Warning: Failed to apply qt-material theme or custom QSS. {e}")

    # --- Optional Font Modification ---
    # (Keep existing font modification block here - it might interact with QSS)
    # ...

    window = TESModelViewer()
    window.show()
    sys.exit(app.exec())