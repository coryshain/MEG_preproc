import os
import numpy as np
import pandas as pd
import argparse
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from . import templates
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from matplotlib import pyplot as plt
import mne

from .util import stderr, info

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Apply basic signal processing to MEG data and safe output as FIF.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) either to subject directories to process (space-delimited) or to a single text file listing all paths to subject directories, one per line.')
    argparser.add_argument('-c', '--config', help='Path to YML config file defining cleaning parameters. If unspecified, use default parameters.')
    argparser.add_argument('-j', '--n_jobs', default='cuda', help='Number of jobs for parallel processing, or "cuda" to attempt GPU acceleration if available.')
    argparser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode on a subset of data for speed, saving plots but not data.')
    args = argparser.parse_args()

    # Collect default settings
    with pkg_resources.open_text(templates, 'cleaning_defaults.yml') as src_config:
        config_default = yaml.load(src_config, Loader=Loader)

    # Parse config
    subject_dirs = args.paths
    if len(subject_dirs) == 1 and not os.path.isdir(subject_dirs[0]):
        # Input is a text-based list of directories
        with open(subject_dirs[0], 'r') as f:
            subject_dirs = [x.strip() for x in f]
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
    if config_path:
        config_name = os.path.basename(config_path)[:-4]
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=Loader)
    else:
        config_name = 'default_meg'
        config = config_default
    config = {x: config.get(x, config_default[x]) for x in config_default}
    artifact_n_component = int(config['artifact_n_component'])
    remove_heartbeats = bool(config['remove_heartbeats'])
    remove_blinks = bool(config['remove_blinks'])
    low_freq = float(config['low_freq'])
    if not low_freq:
        low_freq = None
    high_freq = float(config['high_freq'])
    if not high_freq:
        high_freq = None
    reject = config['reject']
    reject = {x: float(reject[x]) for x in reject}
    flat = config['flat']
    flat = {x: float(flat[x]) for x in flat}

    for subject_dir in subject_dirs:
        with open(os.path.join(subject_dir, config_name + '.yml'), 'w') as f:
            yaml.dump(config, f, Dumper=Dumper)

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
        ecg_projs = None
        if remove_heartbeats:
            info('Finding and removing cardiac artifacts', marker='=')
            ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(
                raw,
                n_grad=artifact_n_component,
                n_mag=artifact_n_component,
                reject=reject,
                flat=flat,
                n_jobs=n_jobs
            )
            raw.add_proj(ecg_projs)

        # Remove EOG (blink) artifacts
        eog_projs = None
        if remove_blinks:
            info('Finding and removing ocular artifacts', marker='=')
            eog_projs, eog_events = mne.preprocessing.compute_proj_eog(
                raw,
                ch_name='MEG1411',  # Channel 1411 is good for eye blink detection
                n_grad=artifact_n_component,
                n_mag=artifact_n_component,
                reject=reject,
                flat=flat,
                n_jobs=n_jobs
            )
            raw.add_proj(eog_projs)

        # Low/high pass filter
        if low_freq or high_freq:
            info('Bandpass filtering', marker='=')
            raw.load_data()
            raw.filter(l_freq=low_freq, h_freq=high_freq, picks='meg', n_jobs=n_jobs)

        if suffix:
            suffix = '_' + suffix
        suffix += '_' + config_name

        print(raw)
        print(dir(raw))
        fig = raw.plot_projs_topomap(show=False)
        eog_fig_path = os.path.join(subject_dir, prefix + suffix + '_proj' + '.png')
        fig.savefig(eog_fig_path)
        plt.close('all')

        raw.apply_proj()

        suffix += '.fif'
        raw.save(os.path.join(subject_dir, prefix + suffix), overwrite=True)







