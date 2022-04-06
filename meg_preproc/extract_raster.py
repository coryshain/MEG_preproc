import os
import time
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
    Extract raster representation of cleaned MEG timecourses for use with Isik et al 14 decoding codebase.
    ''')
    argparser.add_argument('config', help='Path to config file containing extraction instructions.')
    argparser.add_argument('-j', '--n_jobs', default='cuda', help=('Number of jobs for parallel processing, or "cuda" to attempt GPU acceleration if available.'))
    argparser.add_argument('-d', '--debug', action='store_true', help=('Run in debug mode on a subset of data for speed, saving plots but not data.'))
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
    expt_name = config['expt_name']
    subjects = config['subjects']
    outdir = config.get('outdir', './rasters')
    sensor_type = config.get('sensor_type', 'all')
    clean_code = config.get('clean_code', 'default_meg')
    resample_to = config.get('resample_to', None)
    event_code_to_name = config.get('event_map', None)
    if event_code_to_name is None:
        use_default_event_map = True
        event_code_to_name = {}
    else:
        use_default_event_map = False
    if event_code_to_name:
        event_name_to_code = {event_code_to_name[x]: x for x in event_code_to_name}
    else:
        event_name_to_code = {}
    epoch_tmin = config.get('epoch_tmin', -0.5)
    epoch_tmax = config.get('epoch_tmax', 1.5)
    word_level_file = config['word_level_file']
    word_level_events = pd.read_csv(word_level_file)
    assert 'condition' in word_level_events and 'word_onset_time' in word_level_events, (
            'File %s was provided via field ``word_level_file``, but it is not correctly formatted.'
            ' It must contain columns called "condition" and "word_onset_time" to enable alignment'
            ' to the events file.'
    ) % word_level_file
    word_level_filter = config.get('word_level_filter', {})
    for k in word_level_filter:
        word_level_events = word_level_events[word_level_events[k].isin(word_level_filter[k])]
    label_col = config.get('label_col', 'condition')
    unique_labels = word_level_events[label_col].unique()
    label2id = {x: i + 1 for i, x in enumerate(unique_labels)}
    id2label = {i + 1: x for i, x in enumerate(unique_labels)}
    word_level_events['label_id'] = word_level_events[label_col].map(label2id)

    reject = dict(mag=8000e-15, grad=8000e-13)  # T, T/m
    flat = dict(mag=1e-15, grad=1e-13)  # T/m

    for subject_dir in subjects:
        subject_dir = os.path.normpath(subject_dir)

        # Load data
        info('Processing subject directory: %s' % subject_dir, marker='*')
        protocol_path = os.path.join(subject_dir, 'protocol.txt')
        assert os.path.exists(protocol_path), (
                'No protocol file ("protocol.txt", comma-delimited list of expertiment_name,start_time) found in'
                ' directory %s.'
        ) % subject_dir
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
        assert expt_name in expt_intervals, (
                'Experiment name %s not found in protocol file, which had experiment names %s.'
        ) % (expt_name, ', '.join(expt_intervals.keys()))
        expt_start, expt_end = expt_intervals[expt_name]
        fifs = [x for x in os.listdir(subject_dir) if x.endswith('%s.fif' % clean_code)]
        if not fifs:
            stderr('No *_cleaned_meg.fif files found in directory %s. Skipping...\n' % subject_dir)
            continue
        assert len(fifs) == 1, 'Found multiple matching files in directory %s: %s.' % (subject_dir, ', '.join(fifs))
        fif_base = os.path.join(subject_dir, fifs[0])
        if not os.path.exists(fif_base):
            stderr('The *.fif files in directory %s are not correctly named. Skipping...\n' % subject_dir)
            continue
        raw = mne.io.Raw(fif_base)
        if expt_end:
            expt_end = min(expt_end, raw.times.max())
        else:
            expt_end = raw.times.max()
        if debug:
            expt_end = min(expt_end, expt_start + 1000)
        raw.crop(tmin=expt_start, tmax=expt_end)
        max_time = raw.times.max()

        # Load events
        all_events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, consecutive=True)
        sfreq = raw.info['sfreq']
        time_scale = 1. / sfreq
        event_ids = all_events[:, 2]
        if use_default_event_map:
            for x in np.unique(event_ids):  # 3rd column contains event ids
                if x not in event_code_to_name:
                    event_code_to_name[x] = str(x)
                if str(x) not in event_name_to_code:
                    event_name_to_code[str(x)] = x

        event_mapper = np.vectorize(lambda x: event_code_to_name.get(x, str(x)))

        if word_level_events is None:
            epoch_events = all_events
            event_id = event_name_to_code
        else:
            epoch_events_src = pd.DataFrame(all_events, columns=['time', 'something', 'condition'])
            epoch_events_src['onset_time'] = epoch_events_src['time']
            epoch_events_src = epoch_events_src[epoch_events_src.condition.isin(event_code_to_name)]
            epoch_events_src.condition = event_mapper(epoch_events_src.condition)
            epoch_events_src = pd.merge(
                epoch_events_src, word_level_events,
                on='condition',
                how='inner'
            )
            _event_times = epoch_events_src['word_onset_time'] / time_scale + epoch_events_src['time']
            _event_other = epoch_events_src[['something', 'label_id']].values
            epoch_events = np.concatenate(
                [_event_times[..., None], _event_other],
                axis=1
            ).astype(int)

        data = mne.Epochs(
            raw,
            events=epoch_events,
            event_id=label2id,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            reject=reject,
            flat=flat,
            reject_by_annotation=False,
            preload=True
        )
        if resample_to:
            info('Resampling to %s Hz' % resample_to)
            data.resample(resample_to, n_jobs=n_jobs)

        out = data.get_data(picks='meg')
        print(out.shape)
        print(epoch_events)
        print(epoch_events.shape)
        print(data.events)
        print(data.events.shape)
        exit()

        _events = data.events
        _event_ids = _events[:, 2]
        if word_level_events is None:
            _event_names = event_mapper(_event_ids)
            _event_times = _events[:,0] * time_scale
            _events = pd.DataFrame({
                'epoch': np.arange(len(_event_names)),
                'condition': _event_names,
                'onset_time': _event_times
            })
        else:
            _events = pd.DataFrame({'index': _event_ids})
            _events = pd.merge(_events, epoch_events_src[['index', 'condition', 'onset_time']], on='index')
            _events['epoch'] = np.arange(len(_events))
            del _events['index']
        _events['subject'] = os.path.basename(subject_dir)
        events.append(_events)

        _responses = data.to_data_frame(picks='meg')
        _responses['subject'] = os.path.basename(subject_dir)
        _responses['time'] = _responses['time']
        responses.append(_responses)

    if responses:
        info('Saving response table')
        responses = pd.concat(responses, axis=0)
        responses = responses.reset_index(drop=True)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        grad_norms = {}
        if sensor_type[-1] == '+' or sensor_type.lower() == 'gradnorm':
            # Compute gradnorms
            grad_norms = {}
            sensor_locations = sorted(list(set([c[:-1] for c in responses.columns if c.startswith('MEG')])))
            for c in sensor_locations:
                g1 = responses[c + '2']
                g2 = responses[c + '3']
                n = np.sqrt(g1 ** 2 + g2 ** 2)
                grad_norms[c.replace('MEG', 'MEGGN')] = n
        if sensor_type.lower() == 'mag':
            # Delete gradiometers (sensor names that don't end in 1)
            for c in responses.columns:
                if c.startswith('MEG') and not c.endswith('1'):
                    del responses[c]
        elif sensor_type.lower() == 'grad':
            # Delete magentometers (sensor names that end in 1)
            for c in responses.columns:
                if c.startswith('MEG') and c.endswith('1'):
                    del responses[c]
        if grad_norms:
            grad_norms = pd.DataFrame(grad_norms)
            responses = pd.concat([responses, grad_norms], axis=1)

        responses.to_csv(
            os.path.join(outdir, os.path.basename(expt_name) + '_responses.csv'),
            index=False
        )
    else:
        info('No output response data.')

    info('End')
