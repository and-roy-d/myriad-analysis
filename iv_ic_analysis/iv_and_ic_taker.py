#!/opt/nist-qsghw-2023.12/bin/python

'''
written by Dan Becker, takes and plots iv curves. Plotting crashes at small step sizes, but still saves the data
11/15/2022: edited by Nathan Nakamura to include
    1) command line option for direct or differential bias. Default is differential bias
    2) sections of code in the qdac portion to actually do differential bias, and collect
       data on two differentially biased snouts simultaneously
    3) makes the qdac channels to bias a required command line input, rather than hard coded into the script
05/20/2026 Sophia Nowak was here to hard code in the correct USB for the Qdac instument.
2026 Headless Update: Structured back-to-back IV and Ic loops, isolated to functional execution 
                      and configured for terminal deployment.
'''

import matplotlib
matplotlib.use('Agg') # Force non-interactive headless rendering

import subprocess
import sys
import time
import argparse

import numpy as np
import scipy.interpolate
import pylab as plt
import IPython

from numpy import pi

import qsdumux.user.hardware as hardware
import qsdumux.user.util as util
import qsdumux.user.abaco_setup as abaco_setup

import qsdumux.instruments.bftc as bftc


def adjust_iv(vb, ang, vnorm_start):
    p = np.polyfit(vb[vb >= vnorm_start], ang[vb >= vnorm_start], 1)
    return np.sign(p[0])*(ang - p[1])


class BiasVoltageSource:
    def __init__(self, cfg):
        self.source_type = cfg.iv_source_type
        self.vboxes = []
        self.bias_type = cfg.bias_type

        if self.source_type == 'srs':
            import qsdumux.instruments.srs_sim900 as sim900
            import qsdumux.instruments.srs_sim928 as sim928

            mainframe = sim900.SRS_SIM900(url=cfg.iv_srs_device)
            for srs_id in cfg.iv_srs_ids:
                self.vboxes.append(sim928.SRS_SIM928(mainframe, srs_id))
            for vbox in self.vboxes:
                vbox.setOutputOn()

        elif self.source_type == 'qdac':
            from qcodes_contrib_drivers.drivers.QDevil import QDAC2
            qdac_addr = 'ASRL/dev/ttyUSB1::INSTR' 
            self.qdac = QDAC2.QDac2('QDACII', visalib='@py', address=f'{qdac_addr}')
            self.qdac.reset()
            time.sleep(2)
            print(self.qdac.errors())
            vbox1 = self.qdac.channel(cfg.chan1)
            vbox2 = self.qdac.channel(cfg.chan2)
            vbox3 = self.qdac.channel(cfg.chan3)
            vbox4 = self.qdac.channel(cfg.chan4)
            self.vboxes = [vbox1, vbox2, vbox3, vbox4]
            self.last_voltage = [None, None, None, None]
            for vbox in self.vboxes:
                vbox.output_mode(range='high', filter='med')
        else:
            raise Exception(f"Unknown iv_source type: {cfg.iv_source_type}")

    def setVoltage(self, v):
        if self.source_type == 'srs':
            for vbox in self.vboxes:
                vbox.setvolt(v)
        if self.source_type == 'qdac':
            if self.bias_type == 'direct':
                for (k, vbox) in enumerate(self.vboxes):
                    if self.last_voltage[k] is None:
                        vbox.dc_constant_V(v)
                    else:
                        dc_list = vbox.dc_list(repetitions=1, voltages=[self.last_voltage[k], v], dwell_s=100e-3)
                        dc_list.start()
                    self.last_voltage[k] = v
            elif self.bias_type == 'differential':
                for (k, vbox) in enumerate(self.vboxes):
                    val = v/2 if (k == 0 or k == 2) else -v/2
                    if self.last_voltage[k] is None:
                        vbox.dc_constant_V(val)
                    else:
                        dc_list = vbox.dc_list(repetitions=1, voltages=[self.last_voltage[k], val], dwell_s=100e-3)
                        dc_list.start()
                    self.last_voltage[k] = val
            elif self.bias_type == 'differential_direct':
                for (k, vbox) in enumerate(self.vboxes):
                    val = v if (k == 0 or k == 2) else 0
                    if self.last_voltage[k] is None:
                        vbox.dc_constant_V(val)
                    else:
                        if k == 0 or k == 2:
                            dc_list = vbox.dc_list(repetitions=1, voltages=[self.last_voltage[k], val], dwell_s=100e-3)
                            dc_list.start()
                        else:
                            vbox.dc_constant_V(0)
                    self.last_voltage[k] = val
            else:
                raise Exception("Unknown bias type, must be either 'direct' or 'differential'")


def run_sweeps(setpoint_array, cfg, base_fname, hw, srcids, n_chan, samp_per_pkt, phi0, n_pkts, n_samp_avg, fr_hz):
    """Executes the sequential back-to-back sweeps per temperature step."""
    
    # Initialize the voltage source ONCE to avoid QCoDeS name registry crashes
    vboxes = BiasVoltageSource(cfg)

    for target_K in setpoint_array:
        target_temp_mK = round(target_K * 1000, 1)
        print(f"\n========================================\nTargeting Temperature: {target_temp_mK} mK")
        
        # 1. COMMAND THE FRIDGE TO RAMP
        print(f"--> Sending setpoint command to Bluefors Controller (mxc): {target_temp_mK} mK")
        bftc.set_setpoint(target_K, thermometer='mxc')
        
        # 2. TEMPERATURE STABILIZATION WAIT LOOP
        # Adjust tolerance_mK and max_wait_time as needed for your specific fridge behavior
        max_wait_time = 900  # 15 minutes max wait per step
        wait_interval = 30   # Check every 30 seconds
        t_elapsed = 0
        tolerance_mK = 0.1   # Settling window around target
        
        print("--> Waiting for temperature to stabilize...")
        while t_elapsed < max_wait_time:
            current_temp_mK = round(bftc.read_mxc_temperature() * 1000, 3)
            diff = abs(current_temp_mK - target_temp_mK)
            print(f"    [Temp Sync] Elapsed: {t_elapsed}s | Current: {current_temp_mK:.3f} mK | Δ: {diff:.3f} mK")
            
            if diff <= tolerance_mK:
                print(f"--> Temperature stabilized within {tolerance_mK} mK. Proceeding to sweeps.")
                break
                
            time.sleep(wait_interval)
            t_elapsed += wait_interval
        else:
            print("--> [Warning] Stabilization timeout reached before hitting tolerance. Starting sweeps anyway.")

        # Capture the precise temperature at the start of data taking for the file label
        actual_start_temp_mK = round(bftc.read_mxc_temperature() * 1000, 1)

        sweeps = [
            {'name': 'IV', 'vstart': 4.0, 'vstop': 0.0, 'vstep': -np.abs(cfg.vstep), 'drive_normal': True},
            {'name': 'Ic', 'vstart': 0.0, 'vstop': 2.0, 'vstep': np.abs(cfg.vstep), 'drive_normal': False}
        ]

        for sweep in sweeps:
            print(f"\n--- Starting {sweep['name']} Sweep ({sweep['vstart']}V -> {sweep['vstop']}V) ---")
            num_steps = int(np.abs((sweep['vstart'] - sweep['vstop']) / cfg.vstep)) + 1
            vb = np.linspace(sweep['vstart'], sweep['vstop'], num_steps)

            if sweep['drive_normal']:
                vnorm_val = cfg.vnorm if cfg.vnorm is not None else vb[0]
                print(f"Driving normal to {vnorm_val} V...")
                vboxes.setVoltage(vnorm_val)
                time.sleep(cfg.t_norm)

            vboxes.setVoltage(vb[0])
            time.sleep(cfg.t_initial)

            full_data_dict = {ch: [] for ch in range(n_chan)}
            samp_nbrs = {srcid: np.zeros_like(vb, dtype='int64') for srcid in srcids}
            last_data = {srcid: None for srcid in srcids}
            last_unwrapped = {srcid: None for srcid in srcids}
            ang2 = np.zeros((len(vb), n_chan))
            ang2_std = np.zeros((len(vb), n_chan))

            for (k, v) in enumerate(vb):
                vboxes.setVoltage(v)

                if sweep['name'] == 'IV' and (cfg.vstart_slow >= v >= cfg.vstop_slow):
                    data = hw[cfg.fset_path[0]].take_new_data(srcids=srcids, n_pkts=cfg.slow_factor*n_pkts)
                else:
                    data = hw[cfg.fset_path[0]].take_new_data(srcids=srcids, n_pkts=n_pkts)
                
                print(f"[{sweep['name']}] Set Vb = {v:.3f} V | Samples = ", ", ".join([str(data[sid].data.shape[0]) for sid in srcids]))

                chan_base = 0
                for (s, srcid) in enumerate(srcids):
                    if k == 0:
                        samp_nbrs[srcid][k] = data[srcid].seqno * samp_per_pkt
                    if k < len(vb) - 1:
                        samp_nbrs[srcid][k+1] = data[srcid].seqno * samp_per_pkt + data[srcid].data['angle'].shape[0]

                    dcon = data[srcid].data['angle']
                    
                    available_chans = dcon.shape[1]
                    if chan_base + available_chans > n_chan:
                        slice_limit = n_chan - chan_base
                        dcon = dcon[:, :slice_limit]
                    
                    if dcon.shape[1] == 0:
                        continue

                    if k == 0:
                        unwrapped = np.unwrap(2*pi*dcon/phi0, axis=0)/(2*pi)
                    else:
                        data1 = np.concatenate((last_data[srcid], dcon), axis=0)
                        unwrapped1 = np.unwrap(2*pi*data1/phi0, axis=0)/(2*pi)
                        unwrapped1 += last_unwrapped[srcid] - last_data[srcid]/phi0
                        unwrapped = unwrapped1[1:, :]

                    for cch in range(unwrapped.shape[1]):
                        true_ch = chan_base + cch
                        if cfg.save_all_data or (s == 0 and true_ch == cfg.full_chan):
                            if k == 0:
                                full_data_dict[true_ch] = unwrapped[:, cch]
                            elif k < len(vb) - 1:
                                full_data_dict[true_ch] = np.append(full_data_dict[true_ch], np.zeros(samp_nbrs[srcid][k+1] - samp_nbrs[srcid][k]) + np.nan)
                            else:
                                full_data_dict[true_ch] = np.append(full_data_dict[true_ch], np.zeros(data[srcid].seqno * samp_per_pkt + data[srcid].data['angle'].shape[0] - samp_nbrs[srcid][k]) + np.nan)
                            full_data_dict[true_ch][-len(unwrapped[:, cch]):] = unwrapped[:, cch]

                    last_unwrapped[srcid] = unwrapped[-1, :]
                    last_data[srcid] = dcon[-1, :].reshape((1, dcon.shape[1]))

                    ang2[k, chan_base:(chan_base + unwrapped.shape[1])] = np.mean(unwrapped[-n_samp_avg:, :], axis=0)
                    ang2_std[k, chan_base:(chan_base + unwrapped.shape[1])] = np.std(unwrapped[-n_samp_avg:, :], axis=0)
                    chan_base += unwrapped.shape[1]

            for srcid in srcids:
                samp_nbrs[srcid] = samp_nbrs[srcid] - samp_nbrs[srcid][0]

            if cfg.save_all_data:
                full_data = np.zeros((len(full_data_dict[0]), len(full_data_dict)))
                for cch in range(n_chan):
                    full_data[:, cch] = full_data_dict[cch]

            tone_fname = subprocess.check_output('find /data -mindepth 4 -name "tones*"  | xargs ls -trd1 | tail -1', shell=True).decode()[:-1]
            tone_cfg = util.read_tone_file(tone_fname)
            good = np.array(tone_cfg['tones_good'], dtype=object) == 'True' if 'tones_good' in tone_cfg else np.ones(n_chan, dtype=bool)

            if good.shape[0] != n_chan:
                good = np.ones(n_chan, dtype=bool)

            # Updated dynamic name to use the actual stabilization readout temp
            sweep_fname = f"{base_fname}_{sweep['name']}_{actual_start_temp_mK:.1f}mK"
            print(f"--> Exporting: {sweep_fname}.npz")
            
            if cfg.save_all_data:
                np.savez(f"{sweep_fname}.npz", vb=vb, samp_nbrs=samp_nbrs, tsleep=0.0, t_data_sec=cfg.t_data_sec, ang2=ang2, ang2_std=ang2_std, full_data=full_data)
            else:
                np.savez(f"{sweep_fname}.npz", vb=vb, samp_nbrs=samp_nbrs, tsleep=0.0, t_data_sec=cfg.t_data_sec, ang2=ang2, ang2_std=ang2_std)

            # Plotting block
            plt.figure(300)
            plt.clf()
            vstart_norm = vb[0] - 1.0
            for ch in range(n_chan):
                if good[ch]:
                    try:
                        ang_adj = adjust_iv(vb, ang2[:, ch], vstart_norm)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            plt.plot(vb, (vb / ang_adj / np.mean(vb[vb >= vstart_norm] / ang_adj[vb >= vstart_norm])))
                    except Exception:
                        plt.plot(vb, ang2[:, ch])
            plt.xlabel("Applied V_bias (V)")
            plt.ylabel("Relative Slope")
            plt.savefig(f'{sweep_fname}_rel.png')

            nx, ny = 4, 4
            for ch in range(n_chan):
                if (ch % (nx*ny)) == 0:
                    if ch > 0:
                        plt.tight_layout(pad=True)
                        plt.savefig(f'{sweep_fname}.{ch//(nx*ny) - 1}.png')
                    plt.figure(10 + (ch//(nx*ny)))
                    plt.gcf().set_size_inches(10, 8, forward=True)
                    plt.clf()
                if good[ch]:
                    plt.subplot(nx, ny, ch % (nx*ny) + 1)
                    plt.plot(vb, ang2[:, ch])
                    plt.title(f'Ch: {ch:02d}')
            plt.tight_layout(pad=True)
            plt.savefig(f'{sweep_fname}.{ch//(nx*ny)}.png')

            if sweep['name'] == 'Ic':
                print("--> Sweep series finished. Grounding voltage source (Setting Bias to 0.0V)...")
                vboxes.setVoltage(0.0)
                time.sleep(2.0)


def main():
    parser = argparse.ArgumentParser(description='Acquire Coarse IV and Ic sweeps per temperature step')
    parser.add_argument("--bias_type", help="Biasing setup, either differential, direct, or differential_direct", type=str, default="direct")
    parser.add_argument("--chan1", help="First QDAC channel", type=int, default=1)
    parser.add_argument("--chan2", help="Second QDAC channel", type=int, default=2)
    parser.add_argument("--chan3", help="Third QDAC channel", type=int, default=3)
    parser.add_argument("--chan4", help="Fourth QDAC channel", type=int, default=4)
    parser.add_argument("--vstep", help="Step voltage (V)", type=float, default=0.005)
    parser.add_argument("--vnorm", help="Normalizing voltage (V)", type=float, default=None)
    parser.add_argument("--vstart_slow", help="Starting voltage for 'slow' portion (V)", type=float, default=3.0)
    parser.add_argument("--vstop_slow", help="Stopping voltage for 'slow' portion (V)", type=float, default=1.0)
    parser.add_argument("--slow_factor", help="Slow section packet multi-factor", type=int, default=1)
    parser.add_argument("--t_data_sec", help="Time per step to take data (seconds)", type=float, default=0.400)
    parser.add_argument("--t_avg_sec", help="Data integration time per step (seconds)", type=float, default=0.020)
    parser.add_argument("--t_norm", help="Time to spend at vnorm (seconds)", type=float, default=1.0)
    parser.add_argument("--t_initial", help="Time to spend at initial value (seconds)", type=float, default=1.0)
    parser.add_argument("--full_chan", help="Channel to plot full data", type=int, default=-1)
    parser.add_argument('--save_all_data', action='store_true', help='Store all data recorded', default=False)
    parser.add_argument('-i', dest='interactive', action='store_true', help='Interactive diagnostic fallback session')

    (cfg, argv_remaining) = parser.parse_known_args()
    cfg = abaco_setup.parse_config_file_and_command_line(argv_remaining, cfg=cfg)

    if cfg.interactive:
        plt.ion()
    else:
        plt.ioff()

    abaco_setup.config_dirs(cfg)
    base_fname = abaco_setup.config_dirs(cfg, "iv")

    hw = hardware.make_hardware(cfg)
    hw[cfg.fset_path[0]].flux_ramp_on()
    (fr_hz, _) = hw[cfg.fset_path[0]].get_flux_ramp()

    srcids = []
    for fset in cfg.fset_path:
        (srcids_fset, samp_per_pkt_fset, _, _) = hw[fset].get_backend_output_info(verbose=True)
        if len(srcids_fset) == 0:
            print(f"No backend pktdata tabs are enabled for {fset}")
            sys.exit(1)
        srcids += srcids_fset

    # Force channel restriction count to 67 channels exclusively
    n_chan = 67

    # Extract dynamic properties relative to first config set
    (_, samp_per_pkt, _, phi0) = hw[cfg.fset_path[0]].get_backend_output_info(verbose=False)

    n_pkts = len(srcids) * int(np.round((fr_hz * cfg.t_data_sec) / samp_per_pkt, -2))
    n_samp_avg = int(np.round(fr_hz * cfg.t_avg_sec))

    # Single Coarse 20mK validation run parameters
    setpoint_array = np.linspace(25, 50, 26) * 1e-3

    # Execute experiment sweeps loop
    run_sweeps(setpoint_array, cfg, base_fname, hw, srcids, n_chan, samp_per_pkt, phi0, n_pkts, n_samp_avg, fr_hz)

    if cfg.interactive:
        hw[cfg.fset_path[0]].dataif.close()
        IPython.start_ipython(argv=["-i", "-c", "import pylab as plt; plt.ion()"], user_ns=locals())


if __name__ == '__main__':
    main()