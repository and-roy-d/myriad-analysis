import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import glob
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import scipy.constants
    import lmfit
    import pandas as pd
    import marimo as mo
    return (
        os, glob, re, np, plt, mcolors, scipy, lmfit, pd, mo
    )


@app.cell
def _(mo):
    rbias_input = mo.ui.number(value=1980.0, step=1.0, label="R_bias (Ohm)")
    min_si_input = mo.ui.number(value=250.0, step=1.0, label="min_SI (pH)")
    rshunt_input = mo.ui.number(value=250.0, step=1.0, label="R_shunt (uOhm)")
    use_rn_override = mo.ui.checkbox(value=False, label="Use Override Rn")
    rn_override_input = mo.ui.number(value=10.0, step=0.1, label="Override Rn (mOhm)")
    return rbias_input, min_si_input, rshunt_input, use_rn_override, rn_override_input


@app.cell
def _(rbias_input, min_si_input, rshunt_input, use_rn_override, rn_override_input, mo):
    _sys_panel_1 = mo.hstack([rbias_input, min_si_input, rshunt_input], justify="start", gap=3)
    _sys_panel_2 = mo.hstack([use_rn_override, rn_override_input], justify="start", gap=3)
    mo.vstack([
        mo.md("## System & Resistance Configurations"),
        _sys_panel_1,
        _sys_panel_2
    ])
    return


@app.cell
def _(mo):
    ibias_min_input = mo.ui.number(value=0.0, step=0.05, label="Ibias Min (mA)")
    ibias_max_input = mo.ui.number(value=2.1, step=0.05, label="Ibias Max (mA)")
    ites_min_input = mo.ui.number(value=0.0, step=0.05, label="Ites Min (mA)")
    ites_max_input = mo.ui.number(value=1.2, step=0.05, label="Ites Max (mA)")
    return ibias_min_input, ibias_max_input, ites_min_input, ites_max_input


@app.cell
def _(ibias_min_input, ibias_max_input, ites_min_input, ites_max_input, mo):
    _zoom_panel = mo.hstack([ibias_min_input, ibias_max_input, ites_min_input, ites_max_input], justify="start", gap=3)
    mo.vstack([
        mo.md("## Zoom Controls (Plots 1 & 2)"),
        _zoom_panel
    ])
    return


@app.cell
def _(mo):
    tc_guess_input = mo.ui.number(value=100.0, step=1.0, label="Tc Guess (mK)")
    tc_min_input = mo.ui.number(value=40.0, step=1.0, label="Tc Min (mK)")
    tc_max_input = mo.ui.number(value=150.0, step=1.0, label="Tc Max (mK)")
    n_guess_input = mo.ui.number(value=4.0, step=0.1, label="n Guess")
    n_min_input = mo.ui.number(value=2.0, step=0.1, label="n Min")
    n_max_input = mo.ui.number(value=4.5, step=0.1, label="n Max")
    k_guess_input = mo.ui.number(value=100.0, step=1.0, label="k Guess (nW/K^n)")
    return (
        tc_guess_input,
        tc_min_input,
        tc_max_input,
        n_guess_input,
        n_min_input,
        n_max_input,
        k_guess_input,
    )


@app.cell
def _(
    tc_guess_input,
    tc_min_input,
    tc_max_input,
    n_guess_input,
    n_min_input,
    n_max_input,
    k_guess_input,
    mo,
):
    _fit_panel_1 = mo.hstack([tc_guess_input, tc_min_input, tc_max_input], justify="start", gap=3)
    _fit_panel_2 = mo.hstack([n_guess_input, n_min_input, n_max_input, k_guess_input], justify="start", gap=3)
    mo.vstack([
        mo.md("## Fit Controls & Guesses"),
        _fit_panel_1,
        _fit_panel_2
    ])
    return


@app.cell
def _(rbias_input, min_si_input, rshunt_input, scipy, np):
    _phi0 = scipy.constants.value("mag. flux quantum")
    _min_SI_val = min_si_input.value * 1e-12
    _amp_per_arb = 1 / (_min_SI_val / _phi0)
    _Rshunt = rshunt_input.value * 1e-6
    rbias = rbias_input.value

    def convert_ang2_to_ites(ang2, channel_id, vb=None):
        ites_uncorrected = ang2[:, channel_id] * _amp_per_arb
        if vb is not None:
            zero_idx = np.argmin(np.abs(vb))
            return ites_uncorrected - ites_uncorrected[zero_idx]
        return ites_uncorrected - ites_uncorrected[-1]

    def Rtes(ibias, ites):
        ites_safe = np.where(ites == 0, np.nan, ites)
        return _Rshunt * (ibias - ites_safe) / ites_safe

    def Ptes(ibias, ites):
        return Rtes(ibias, ites) * ites ** 2

    def calculate_rn(ibias, ites):
        if len(ibias) < 100 or len(ites) < 100:
            return 10e-3
        rtes_values = Rtes(ibias[-100:], ites[-100:])
        rtes_values = rtes_values[np.isfinite(rtes_values)]
        if len(rtes_values) == 0:
            return 10e-3
        return np.nanmedian(rtes_values)

    return (
        convert_ang2_to_ites, Rtes, Ptes, calculate_rn, rbias, _Rshunt, _min_SI_val
    )


@app.cell
def _(glob, os, re):
    _data_directory = "/data/20260709/iv/"
    npz_files = {}
    if os.path.isdir(_data_directory):
        _pattern = os.path.join(_data_directory, "*_iv_IV_*.npz")
        _files = glob.glob(_pattern)
        for _file_path in _files:
            _filename = os.path.basename(_file_path)
            _match = re.search(r'_IV_([\d\.]+)mK\.npz$', _filename)
            if _match:
                try:
                    _tbase_val = float(_match.group(1))
                    _tbase = _tbase_val / 1000.0
                    npz_files[_tbase] = _file_path
                except ValueError:
                    pass
    tbase_sorted = sorted(npz_files.keys())
    return npz_files, tbase_sorted


@app.cell
def _(Rtes, convert_ang2_to_ites, np, npz_files, tbase_sorted, rbias):
    active_channels = []
    if npz_files:
        _lowest_temp_fpath = npz_files[tbase_sorted[0]]
        _data = np.load(_lowest_temp_fpath)
        _vb = _data['vb']
        _ang2 = _data['ang2']
        _n_steps, _n_chan = _ang2.shape

        _sort_idx = np.argsort(np.abs(_vb))
        _vb_sorted = _vb[_sort_idx]
        _ibias = _vb_sorted / rbias

        for _ch in range(_n_chan):
            _ites = convert_ang2_to_ites(_ang2[_sort_idx, :], _ch, vb=_vb_sorted)
            _rtes = Rtes(_ibias, _ites)
            _ites_range_uA = (np.max(_ites) - np.min(_ites)) * 1e6
            _rtes_floor = np.nanmedian(_rtes[5:30]) if len(_rtes) > 30 else 0.0
            _valid_rtes = _rtes[~np.isnan(_rtes)]

            _has_swing = (10.0 < _ites_range_uA < 2000.0)
            _has_sc_state = np.any((_rtes - _rtes_floor) < 10e-6) if len(_valid_rtes) > 0 else False
            if _has_swing and _has_sc_state:
                active_channels.append(_ch)
    return (active_channels,)


@app.cell
def _(active_channels, mo, tbase_sorted):
    if not active_channels:
        mo.md("### Error: No active channels found!")
        channel_select = None
        exclude_select = None
    else:
        channel_select = mo.ui.dropdown(
            options=[str(ch) for ch in active_channels],
            value=str(active_channels[0]),
            label="Select TES Channel:"
        )
        exclude_select = mo.ui.multiselect(
            options=[f"{t*1000:.1f} mK" for t in tbase_sorted],
            label="Select bath temperatures to EXCLUDE from fitting:"
        )
    return channel_select, exclude_select


@app.cell
def _(channel_select, exclude_select, mo):
    if channel_select is None:
        layout = mo.md("No active TES channels.")
    else:
        layout = mo.hstack(
            [channel_select, exclude_select],
            justify="start",
            align="stretch",
            gap=4
        )
    layout
    return (layout,)


@app.cell
def _(
    Ptes,
    Rtes,
    calculate_rn,
    channel_select,
    convert_ang2_to_ites,
    exclude_select,
    use_rn_override,
    rn_override_input,
    ibias_min_input,
    ibias_max_input,
    ites_min_input,
    ites_max_input,
    lmfit,
    mcolors,
    mo,
    np,
    npz_files,
    pd,
    plt,
    tbase_sorted,
    rbias,
    tc_guess_input,
    tc_min_input,
    tc_max_input,
    n_guess_input,
    n_min_input,
    n_max_input,
    k_guess_input,
):
    if channel_select is None or not channel_select.value:
        display_layout = mo.md("")
    else:
        _ch = int(channel_select.value)
        _excluded_strs = exclude_select.value or []
        _excluded_temps_mK = [float(_s.split(" ")[0]) for _s in _excluded_strs]

        # 1. Three Panel Plot
        _fig, _axes = plt.subplots(1, 3, figsize=(18, 5.5))
        _cmap = plt.get_cmap('coolwarm')

        # Normalize for temperature steps
        _all_temps_mK = [_t*1000.0 for _t in tbase_sorted]
        _norm = mcolors.Normalize(vmin=min(_all_temps_mK), vmax=max(_all_temps_mK))

        _rn = None

        for _tbase in tbase_sorted:
            _temp_mK = _tbase * 1000.0
            _is_excluded = any(abs(_temp_mK - _et) < 0.05 for _et in _excluded_temps_mK)

            _data = np.load(npz_files[_tbase])
            _vb = _data['vb']
            _ang2 = _data['ang2']
            _sort_idx = np.argsort(np.abs(_vb))
            _vb_sorted = _vb[_sort_idx]
            _ibias = _vb_sorted / rbias
            _ites = convert_ang2_to_ites(_ang2[_sort_idx, :], _ch, vb=_vb_sorted)
            _rtes = Rtes(_ibias, _ites)
            _ptes = Ptes(_ibias, _ites)

            if _rn is None:
                if use_rn_override.value:
                    _rn = rn_override_input.value / 1e3
                else:
                    _rn = calculate_rn(_ibias, _ites)

            _color = _cmap(_norm(_temp_mK))
            _alpha_val = 0.2 if _is_excluded else 0.8
            _line_style = ':' if _is_excluded else '-'

            # Panel 0: Ites vs Ibias
            _axes[0].plot(_ibias * 1e3, _ites * 1e3, color=_color, alpha=_alpha_val, ls=_line_style, lw=1.2)
            # Panel 1: Rtes vs Ibias
            _axes[1].plot(_ibias * 1e3, _rtes * 1e3, color=_color, alpha=_alpha_val, ls=_line_style, lw=1.2)
            # Panel 2: Rtes vs Ptes
            _axes[2].plot(_ptes * 1e12, _rtes * 1e3, color=_color, alpha=_alpha_val, ls=_line_style, lw=1.2)

            # Add temperature label to the last/highest bias point of each curve
            _axes[0].text(_ibias[-1] * 1e3, _ites[-1] * 1e3, f"{_temp_mK:.0f}", color=_color, fontsize=6, ha='left', va='center')
            _axes[1].text(_ibias[-1] * 1e3, _rtes[-1] * 1e3, f"{_temp_mK:.0f}", color=_color, fontsize=6, ha='left', va='center')
            _axes[2].text(_ptes[-1] * 1e12, _rtes[-1] * 1e3, f"{_temp_mK:.0f}", color=_color, fontsize=6, ha='left', va='center')

        # Draw a horizontal dashed line at Rn for panels displaying Rtes
        if _rn is not None:
            _rn_mOhm = _rn * 1e3
            _axes[1].axhline(y=_rn_mOhm, color='gray', linestyle='--', label=f"Rn ({_rn_mOhm:.2f} mOhm)")
            _axes[2].axhline(y=_rn_mOhm, color='gray', linestyle='--', label=f"Rn ({_rn_mOhm:.2f} mOhm)")
            _axes[1].legend(loc='best', fontsize=8)
            _axes[2].legend(loc='best', fontsize=8)

        # Apply zoom limits to Panel 0 and Panel 1
        _axes[0].set_xlim(ibias_min_input.value, ibias_max_input.value)
        _axes[0].set_ylim(ites_min_input.value, ites_max_input.value)
        _axes[1].set_xlim(ibias_min_input.value, ibias_max_input.value)

        _axes[0].set_xlabel("Ibias (mA)")
        _axes[0].set_ylabel("Ites (mA)")
        _axes[0].set_title(f"Ites vs. Ibias (Ch {_ch:02d})")
        _axes[0].grid(True, ls=':', alpha=0.5)

        _axes[1].set_xlabel("Ibias (mA)")
        _axes[1].set_ylabel("Rtes (mOhm)")
        _axes[1].set_title(f"Rtes vs. Ibias (Ch {_ch:02d})")
        _axes[1].grid(True, ls=':', alpha=0.5)

        _axes[2].set_xlabel("Ptes (pW)")
        _axes[2].set_ylabel("Rtes (mOhm)")
        _axes[2].set_title(f"Rtes vs. Ptes (Ch {_ch:02d})")
        _axes[2].grid(True, ls=':', alpha=0.5)

        # Colorbar
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
        _sm.set_array([])
        _cbar_ax = _fig.add_axes([0.93, 0.15, 0.015, 0.7])
        _fig.colorbar(_sm, cax=_cbar_ax, label="Bath Temperature (mK)")
        plt.subplots_adjust(top=0.9, bottom=0.15, right=0.91, left=0.06, wspace=0.25)

        # 2. Ptes vs Tbath Plot
        _ratios = np.linspace(70, 99, 30) * 1e-2
        _fit_results = {}
        _tbase_fit_values = [_t for _t in tbase_sorted if not any(abs(_t*1000.0 - _et) < 0.05 for _et in _excluded_temps_mK)]

        _fig2, _ax2 = plt.subplots(figsize=(6, 5))
        _cmap_r = plt.get_cmap('viridis')
        _norm_r = mcolors.Normalize(vmin=min(_ratios)*100, vmax=max(_ratios)*100)

        for _ratio in _ratios:
            _ptes_ratio = []
            _tbase_ratio = []
            _color_r = _cmap_r(_norm_r(_ratio * 100))

            for _tbase in _tbase_fit_values:
                _data = np.load(npz_files[_tbase])
                _vb = _data['vb']
                _ang2 = _data['ang2']
                _sort_idx = np.argsort(np.abs(_vb))
                _vb_sorted = _vb[_sort_idx]
                _ibias = _vb_sorted / rbias
                _ites = convert_ang2_to_ites(_ang2[_sort_idx, :], _ch, vb=_vb_sorted)
                _rtes = Rtes(_ibias, _ites)
                _ptes = Ptes(_ibias, _ites)
                
                if use_rn_override.value:
                    _rn_val = rn_override_input.value / 1e3
                else:
                    _rn_val = calculate_rn(_ibias, _ites)

                _target_rtes = _rn_val * _ratio
                _idx = np.nanargmin(np.abs(_rtes - _target_rtes))
                _p = _ptes[_idx]
                if _p > 0:
                    _ptes_ratio.append(_p)
                    _tbase_ratio.append(_tbase)

            _ax2.plot(np.array(_tbase_ratio)*1e3, np.array(_ptes_ratio)*1e12, marker='.', ls='', color=_color_r, ms=4)

            if len(_tbase_ratio) >= 3:
                def _G_model(x, k, Tc, n):
                    return k * (Tc**n - x**n)
                _gmod = lmfit.Model(_G_model)
                _params = _gmod.make_params(
                    k=k_guess_input.value * 1e-9,
                    Tc=tc_guess_input.value * 1e-3,
                    n=n_guess_input.value
                )
                _params['k'].min = 0
                _params['Tc'].min = tc_min_input.value * 1e-3
                _params['Tc'].max = tc_max_input.value * 1e-3
                _params['n'].min = n_min_input.value
                _params['n'].max = n_max_input.value
                try:
                    _result = _gmod.fit(np.array(_ptes_ratio), _params, x=np.array(_tbase_ratio))
                    _t_evals = np.array(_tbase_ratio)
                    _ax2.plot(_t_evals*1e3, _result.eval(x=_t_evals)*1e12, color=_color_r, lw=0.8)

                    _k_val = _result.params['k'].value
                    _n_val = _result.params['n'].value
                    _Tc_val = _result.params['Tc'].value
                    _G_val = _k_val * _n_val * (0.1 ** (_n_val - 1))

                    _fit_results[_ratio] = {
                        'G (pW/K)': _G_val * 1e12,
                        'Tc (mK)': _Tc_val * 1000.0,
                        'n': _n_val,
                        'k (nW/K^n)': _k_val * 1e9
                    }
                except Exception:
                    pass

        _ax2.set_xlabel("Bath Temperature (mK)")
        _ax2.set_ylabel("P_TES (pW)")
        _ax2.set_title(f"P_TES vs. T_bath (Ch {_ch:02d})")
        _ax2.grid(True, ls=':', alpha=0.5)

        _sm_r = plt.cm.ScalarMappable(cmap=_cmap_r, norm=_norm_r)
        _sm_r.set_array([])
        _cbar_ax2 = _fig2.add_axes([0.91, 0.15, 0.02, 0.7])
        _fig2.colorbar(_sm_r, cax=_cbar_ax2, label="Rtes/Rn (%)")
        plt.subplots_adjust(top=0.9, bottom=0.15, right=0.88, left=0.12)

        _img1 = mo.as_html(_fig)
        _img2 = mo.as_html(_fig2)
        plt.close(_fig)
        plt.close(_fig2)

        if 0.90 in _fit_results:
            _tbl_df = pd.DataFrame([_fit_results[0.90]])
            _tbl = mo.ui.table(_tbl_df, label="Fit Parameters at R/Rn = 90%")
        else:
            _tbl = mo.md("*Not enough points to fit at R/Rn = 90%*")

        display_layout = mo.vstack([
            mo.md("### Channel Sweeps and Thermal Characteristics"),
            mo.hstack([_img1, _img2], gap=2),
            mo.md("#### Fitted Parameters at R/Rn = 90%"),
            _tbl
        ])

    display_layout
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Fit all active channels and Save results")
    run_button
    return (run_button,)


@app.cell
def _(
    Ptes,
    Rtes,
    active_channels,
    calculate_rn,
    convert_ang2_to_ites,
    exclude_select,
    use_rn_override,
    rn_override_input,
    lmfit,
    mo,
    np,
    npz_files,
    os,
    pd,
    plt,
    run_button,
    tbase_sorted,
    rbias,
    tc_guess_input,
    tc_min_input,
    tc_max_input,
    n_guess_input,
    n_min_input,
    n_max_input,
    k_guess_input,
):
    if not run_button.value:
        status_md = mo.md("*Click button to execute bulk fitting on all active channels and generate plots.*")
    else:
        _excluded_strs = exclude_select.value or []
        _excluded_temps_mK = [float(_s.split(" ")[0]) for _s in _excluded_strs]

        # Filter files
        _tbase_fit_values = [_t for _t in tbase_sorted if not any(abs(_t*1000.0 - _et) < 0.05 for _et in _excluded_temps_mK)]

        _savePath = "/home/pcuser/aroy/Cooldown-B8/"
        os.makedirs(_savePath, exist_ok=True)

        _r_over_rn_ratios = np.linspace(70, 99, 30) * 1e-2

        def _G_model(x, k, Tc, n):
            return k * (Tc**n - x**n)

        _summary_rows = []

        # Loop over active channels and fit
        for _ch in active_channels:
            _channel_ratios_data = []

            for _ratio in _r_over_rn_ratios:
                _ptes_ratio = []
                _tbase_ratio = []
                for _tbase in _tbase_fit_values:
                    _data = np.load(npz_files[_tbase])
                    _vb = _data['vb']
                    _ang2 = _data['ang2']
                    _sort_idx = np.argsort(np.abs(_vb))
                    _vb_sorted = _vb[_sort_idx]

                    _ibias = _vb_sorted / rbias
                    _ites = convert_ang2_to_ites(_ang2[_sort_idx, :], _ch, vb=_vb_sorted)
                    _rtes = Rtes(_ibias, _ites)
                    _ptes = Ptes(_ibias, _ites)
                    
                    if use_rn_override.value:
                        _rn = rn_override_input.value / 1e3
                    else:
                        _rn = calculate_rn(_ibias, _ites)

                    _target_rtes = _rn * _ratio
                    _idx = np.nanargmin(np.abs(_rtes - _target_rtes))
                    _p = _ptes[_idx]
                    if _p > 0:
                        _ptes_ratio.append(_p)
                        _tbase_ratio.append(_tbase)

                if len(_tbase_ratio) >= 3:
                    try:
                        _gmod = lmfit.Model(_G_model)
                        _params = _gmod.make_params(
                            k=k_guess_input.value * 1e-9,
                            Tc=tc_guess_input.value * 1e-3,
                            n=n_guess_input.value
                        )
                        _params['k'].min = 0
                        _params['Tc'].min = tc_min_input.value * 1e-3
                        _params['Tc'].max = tc_max_input.value * 1e-3
                        _params['n'].min = n_min_input.value
                        _params['n'].max = n_max_input.value
                        _res = _gmod.fit(np.array(_ptes_ratio), _params, x=np.array(_tbase_ratio))

                        _k_val = _res.params['k'].value
                        _k_err = _res.params['k'].stderr or 0.0
                        _n_val = _res.params['n'].value
                        _n_err = _res.params['n'].stderr or 0.0
                        _Tc_val = _res.params['Tc'].value
                        _Tc_err = _res.params['Tc'].stderr or 0.0
                        _G_val = _k_val * _n_val * (0.1 ** (_n_val - 1))

                        _dG_dk = _n_val * (0.1 ** (_n_val - 1))
                        _dG_dn = _k_val * (0.1 ** (_n_val - 1)) * (1.0 + _n_val * np.log(0.1))
                        _G_err = np.sqrt((_dG_dk * _k_err)**2 + (_dG_dn * _n_err)**2)

                        _channel_ratios_data.append({
                            'Pixel_Number': str(_ch),
                            'Rtes/Rn': _ratio,
                            'G': _G_val,
                            'G_err': _G_err,
                            'k': _k_val,
                            'k_err': _k_err,
                            'n': _n_val,
                            'n_err': _n_err,
                            'Tc': _Tc_val,
                            'Tc_err': _Tc_err
                        })
                    except Exception:
                        pass

            if _channel_ratios_data:
                _df = pd.DataFrame(_channel_ratios_data)
                _df.to_csv(os.path.join(_savePath, f"G_results_ch{_ch}.csv"), index=False)

                _df_90 = _df[np.abs(_df['Rtes/Rn'] - 0.90) < 1e-4]
                if not _df_90.empty:
                    _row = _df_90.iloc[0]
                    _summary_rows.append({
                        "Pixel_Number": str(_ch),
                        "Rtes/Rn Ratio": 0.90,
                        "G_at_100mK (pW/K)": _row['G'] * 1e12,
                        "G_err_at_100mK (pW/K)": _row['G_err'] * 1e12,
                        "n": _row['n'],
                        "n_err": _row['n_err'],
                        "k (nW/K^n)": _row['k'] * 1e9,
                        "k_err (nW/K^n)": _row['k_err'] * 1e9,
                        "Tc (mK)": _row['Tc'] * 1e3,
                        "Tc_err (mK)": _row['Tc_err'] * 1e3
                    })

        if _summary_rows:
            _summary_df = pd.DataFrame(_summary_rows)
            _summary_fpath = os.path.join(_savePath, "G_summary_ratio_0.90.csv")
            _summary_df.to_csv(_summary_fpath, index=False)

            # Save summary plots
            _fig_sum, _axes_sum = plt.subplots(4, 1, figsize=(8, 16), sharex=True)
            _fig_sum.suptitle("Thermal parameter summary plot at R/Rn = 0.90", fontsize=14)
            _items = ['G_at_100mK (pW/K)', 'n', 'k (nW/K^n)', 'Tc (mK)']
            _errors = ['G_err_at_100mK (pW/K)', 'n_err', 'k_err (nW/K^n)', 'Tc_err (mK)']

            _pixel_labels = [_r['Pixel_Number'] for _r in _summary_rows]
            _x_indices = np.arange(len(_pixel_labels))

            for _idx_itm, _itm in enumerate(_items):
                _values = [_r[_itm] for _r in _summary_rows]
                _errs = [_r[_errors[_idx_itm]] for _r in _summary_rows]
                _axes_sum[_idx_itm].errorbar(_x_indices, _values, yerr=_errs, fmt='o', capsize=5)
                _axes_sum[_idx_itm].set_ylabel(_itm)
                _axes_sum[_idx_itm].grid(True)

            plt.xticks(_x_indices, labels=_pixel_labels)
            _axes_sum[3].set_xlabel("Pixel Number (Channel ID)", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(_savePath, "G_summary_plot_ratio_0.90.png"), dpi=200)
            plt.close(_fig_sum)

            _max_pixel = max([int(_r['Pixel_Number']) for _r in _summary_rows]) if _summary_rows else 24
            _nrows = int(np.ceil(np.sqrt(_max_pixel)))
            _ncols = int(np.ceil(_max_pixel / _nrows))
            _heatmap_data = np.full((_nrows, _ncols), np.nan)

            for _r in _summary_rows:
                _pixel_num = int(_r['Pixel_Number'])
                _display_row = (_pixel_num - 1) % _nrows
                _display_col = (_pixel_num - 1) // _nrows
                _array_row = (_nrows - 1) - _display_row
                _array_col = _display_col

                if 0 <= _array_row < _nrows and 0 <= _array_col < _ncols:
                    _heatmap_data[_array_row, _array_col] = _r['Tc (mK)']

            _fig_map, _ax_map = plt.subplots(figsize=(_ncols * 1.5, _nrows * 1.5))
            _cax = _ax_map.imshow(_heatmap_data, cmap='viridis', origin='lower', extent=[0, _ncols, 0, _nrows], aspect='equal')

            for _r in _summary_rows:
                _pixel_num = int(_r['Pixel_Number'])
                _display_row = (_pixel_num - 1) % _nrows
                _display_col = (_pixel_num - 1) // _nrows
                _ax_map.text(_display_col + 0.5, (_nrows - 1 - _display_row) + 0.5, f"Pixel {_pixel_num}\n{_r['Tc (mK)']:.2f}mK",
                         ha='center', va='center', color='white', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="black", lw=0.5, alpha=0.5))

            _ax_map.set_xticks(np.arange(0.5, _ncols + 0.5))
            _ax_map.set_xticklabels(np.arange(_ncols) + 1)
            _ax_map.set_yticks(np.arange(0.5, _nrows + 0.5))
            _ax_map.set_yticklabels(np.arange(_nrows) + 1)
            plt.colorbar(_cax, label="Tc (mK)")
            plt.title("Tc Heatmap (mK) at R/Rn = 90%")
            plt.tight_layout()
            plt.savefig(os.path.join(_savePath, "G_Tc_heatmap.png"), dpi=200)
            plt.close(_fig_map)

            status_md = mo.md(f"### Success!\n G fitting run completed for {len(_summary_rows)} channels.\nFiles saved to `/home/pcuser/aroy/Cooldown-B8/`:\n- `G_summary_ratio_0.90.csv`\n- `G_summary_plot_ratio_0.90.png`\n- `G_Tc_heatmap.png`\n- `G_results_ch*.csv` for each channel.")
        else:
            status_md = mo.md("### Error! No successful G parameters were fitted.")

    status_md
    return


if __name__ == "__main__":
    app.run()
