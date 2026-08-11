# Myriad Analysis

A modular python codebase for TES (Transition Edge Sensor) detector characterization, resonator Q-factor fitting, thermal conductance ($G$) analysis, complex impedance ($Z$) fitting, IV/Ic measurements, uMUX resonator arc analysis, and microwave readout mapping.

## Installation

Install the package in editable mode or as a library:

```bash
pip install -e .
```

After installation, the CLI commands `analyze_arcs` and `compare_r_vs_p` will be available globally from your terminal:

```bash
# Run resonator arc analysis
analyze_arcs --folder 20241216_123456 --fset fset1 --basepath /data

# Compare R vs P curves between two IV files
compare_r_vs_p
```

## Directory Structure

```text
myriad-analysis/
├── arc_analysis/           # Resonator Arc Analysis & CLI entry point (analyze_arcs)
├── complex_z/              # Complex Impedance & TES Transfer Function Analysis
│   ├── dev_tests/          # Development test scripts & iterations
│   └── analyze_complexZ*.py
├── tes_models/             # TES Thermal & Fitting Models (MDT, Sherpa fitters, viewers)
├── thermal_conductance/    # Thermal Conductance (G) & Alpha/Beta Sensitivity Analysis
├── iv_ic_analysis/         # IV Curves, Critical Current & R vs P Comparison (compare_r_vs_p.py)
├── laser_analysis/         # Laser Peak Fitting & Beam Profile Deconvolution
├── noise_and_pulses/       # Noise Analysis, Pulse Viewer & Resolution vs Bias
├── nonlinearity/           # Detector Nonlinearity Plotters
├── persistent_current/     # Persistent Current Analysis
├── mapping/                # Resonator ID, Channel, & Spatial Position Mapping
├── utils/                  # Software Lock-in, Sine Wave Generators & General Helpers
├── data/                   # Results CSVs & NPZ Data Storage
├── LR700/                  # LR700 Resistance Bridge Controls & Logging
└── Qmeasurement/           # Resonator Q Measurement & Fitting
```

## Module Overview

### 1. Arc Analysis (`arc_analysis/`)
Contains `analyze_arcs.py` for analyzing uMUX resonator arc data, fitting parameters, and producing summary plots/NPZ summaries. Installed as CLI entry point `analyze_arcs`.

### 2. Complex Impedance (`complex_z/`)
Contains scripts for calculating, fitting, and plotting complex impedance ($Z(\omega)$) transfer functions across superconducting and normal transition states.

### 3. TES Models (`tes_models/`)
Contains thermal model definitions (simple, compound, intervening, dangling thermal models) and interactive model viewer GUIs (Matplotlib / PyQt).

### 4. Thermal Conductance (`thermal_conductance/`)
Analyzes power vs. temperature curves to extract thermal conductance ($G = \frac{dP}{dT}$), $T_c$, and sensitivity parameters ($\alpha, \beta$).

### 5. IV & Ic Analysis (`iv_ic_analysis/`)
Tools for taking, parsing, and plotting current-voltage (IV) curves, extracting $I_c$, $R_n$, superconducting-to-normal transitions, and comparing $R$ vs $P$ curves (`compare_r_vs_p.py`). Installed as CLI entry point `compare_r_vs_p`.

### 6. Laser Analysis (`laser_analysis/`)
Peak finding, multi-Gaussian fitting of laser spectra, and deconvolution of laser beam spatial profiles.

### 7. Noise & Pulses (`noise_and_pulses/`)
Pulse shape inspection, noise power spectral density (PSD) calculation, and energy resolution vs bias point evaluation.

### 8. Mapping & Spatial Position (`mapping/`)
Maps readout channels to physical resonator locations on multiplexed detector chips.

## Usage & Development

Subdirectories are structured as Python packages. Modules can be imported directly within Python:

```python
from iv_ic_analysis.compare_r_vs_p import plot_r_vs_p_comparison
from iv_ic_analysis.iv_reader import get_ites_from_iv_curve
from arc_analysis.analyze_arcs import main as run_arc_analysis
```
