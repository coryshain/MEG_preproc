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
    Trim MEG data to a specific experiment, then dump it.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) either to subject directories to process (space-delimited) or to a single text file listing all paths to subject directories, one per line.')
    argparser.add_argument('expt_name', help='Name of experiment to extract.')
    argparser.add_argument('-o', '--outdir', default='trimmed_MEG', help='Path output directory.')
    argparser.add_argument('-j', '--n_jobs', default='cuda', help='Number of jobs for parallel processing, or "cuda" to attempt GPU acceleration if available.')
    args = argparser.parse_args()

    subject_dirs = args.paths
    if len(subject_dirs) == 1 and not os.path.isdir(subject_dirs[0]):
        # Input is a text-based list of directories
        with open(subject_dirs[0], 'r') as f:
            subject_dirs = [x.strip() for x in f]
    expt_name = args.expt_name
    outdir = args.outdir
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

    for subject_dir in subject_dirs:
        # Load data
        info('Processing subject directory: %s' % subject_dir, marker='*')
        protocol_path = os.path.join(subject_dir, 'protocol.txt')
        if not os.path.exists(protocol_path):
            stderr('No protocol file ("protocol.txt", comma-delimited list of expertiment_name,start_time) found in'
                ' directory %s. Skipping...\n' % subject_dir)
        with open(protocol_path) as f:
            expt_intervals = {}
            lines = list(f.readlines())
            for i, l in enumerate(lines):
                name, start = l.split(',')
                if i < len(lines) - 1:
                    end = float(lines[i+1].split(',')[1].strip())
                else:
                    end = None
                expt_intervals[name.strip()] = (float(start.strip()), end)
        if expt_name not in expt_intervals:
            stderr('Experiment name %s not found in protocol file for subject %s, which had experiment names %s. Skipping...\n' % (expt_name, subject_dir, ', '.join(expt_intervals.keys())))
        expt_start, expt_end = expt_intervals[expt_name]
        fifs = [x[:-4] for x in os.listdir(subject_dir) if x.endswith('.fif')]
        if not fifs:
            stderr('No *.fif files found in directory %s. Skipping...\n' % subject_dir)
            continue
        prefix = os.path.commonprefix(fifs)
        fif_base = os.path.join(subject_dir, prefix + '.fif')
        if not os.path.exists(fif_base):
            stderr('The *.fif files in directory %s are not correctly named. Skipping...\n' % subject_dir)
            continue
        raw = mne.io.Raw(fif_base)
        if expt_end:
            expt_end = min(expt_end, raw.times.max())
        else:
            expt_end = raw.times.max()
        raw.crop(tmin=expt_start, tmax=expt_end)

        # Save cleaned data
        suffix = '.fif'
        raw.save(os.path.join(outdir, prefix + '_%s.fif' % expt_name), overwrite=True)
