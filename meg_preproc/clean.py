import os
import numpy as np
import pandas as pd
import mne
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import argparse

from .util import stderr, info

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Apply basic signal processing to MEG data and safe output as FIF.
    ''')
    argparser.add_argument('config', help='Path to config file containing preprocessing instructions.')
    argparser.add_argument('-j', '--n_jobs', default='cuda', help='Number of jobs for parallel processing, or "cuda" to attempt GPU acceleration if available.')
    argparser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode on a subset of data for speed, saving plots but not data.')
    args = argparser.parse_args()

    # Parse config
    debug = args.debug
    n_jobs = args.n_jobs
    if n_jobs.lower() == 'cuda':
        try:
            import cupy as cp
            ndevice = cp.cuda.runtime.getDeviceCount()
            if ndevice:
                gpu_available = True
                mne.utils.set_config('MNE_USE_CUDA', 'true')
            else:
                gpu_available = False
        except ImportError:
            gpu_available = False
        if not gpu_available:
            stderr('GPU acceleration requested but no GPU configured. Falling back to CPU parallelism with njobs=8.\n')
            n_jobs = 8
    else:
        n_jobs = int(n_jobs)
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    subjects = config['subjects']
    remove_heartbeats = bool(config.get('remove_heartbeats', True))
    remove_blinks = bool(config.get('remove_blinks', True))
    low_freq = float(config.get('low_freq', 0.1))
    high_freq = float(config.get('high_freq', 35.))
    if not high_freq:
        high_freq = None

    reject = dict(mag=8000e-15, grad=8000e-13)  # T, T/m
    flat = dict(mag=1e-15, grad=1e-13)  # T/m

    for subject_dir in subjects:
        suffix = ''

        # Load data
        info('Processing subject directory: %s' % subject_dir, marker='*')
        fifs = [x[:-4] for x in os.listdir(subject_dir) if x.endswith('.fif')]
        if not fifs:
            stderr('No *.fif files found in directory %s. Skipping...\n' % subject_dir)
            continue
        prefix = os.path.commonprefix(fifs)
        fif_base = os.path.join(subject_dir, prefix + '.fif')
        if not os.path.exists(fif_base):
            stderr('The *.fif files in directory %s are not correctly named. Skipping...\n' % subject_dir)
            continue
        if debug:
            raw = mne.io.Raw(fif_base).crop(tmax=1000)
        else:
            raw = mne.io.Raw(fif_base)

        # Load events
        events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002)

        # Remove ECG (heartbeat) artifacts
        if remove_heartbeats:
            info('Finding and removing cardiac artifacts', marker='=')
            ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, reject=reject, flat=flat, n_jobs=n_jobs)
            fig = mne.viz.plot_projs_topomap(ecg_projs, info=raw.info, show=False)
            raw.add_proj(ecg_projs)
            ecg_fig_path = os.path.join(subject_dir, prefix + '_ecg.png')
            fig.savefig(ecg_fig_path)
            suffix += 'c'

        # Remove EOG (blink) artifacts
        if remove_blinks:
            info('Finding and removing ocular artifacts', marker='=')
            eog_projs, eog_events = mne.preprocessing.compute_proj_eog(
                raw, ch_name='MEG1411', reject=reject, flat=flat, n_jobs=n_jobs  # Channel 1411 is good for eye blink detection
            )
            fig = mne.viz.plot_projs_topomap(eog_projs, info=raw.info, show=False)
            raw.add_proj(eog_projs)
            eog_fig_path = os.path.join(subject_dir, prefix + '_eog.png')
            fig.savefig(eog_fig_path)
            suffix += 'o'

        # Low/high pass filter
        if low_freq or high_freq:
            info('Bandpass filtering', marker='=')
            raw.load_data()
            raw.filter(l_freq=low_freq, h_freq=high_freq, picks='meg', n_jobs=n_jobs)
            suffix += 'b'

        raw.apply_proj()

        if suffix:
            suffix = '_' + suffix

        suffix += '_cleaned_meg.fif'
        raw.save(os.path.join(subject_dir, prefix + suffix))







