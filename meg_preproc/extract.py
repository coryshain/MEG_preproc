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
    Extract tabular event-related representation of cleaned MEG timecourses
    ''')
    argparser.add_argument('config', help='Path to config file containing extraction instructions.')
    argparser.add_argument('-j', '--n_jobs', default='cuda', help=(
        'Number of jobs for parallel processing, or "cuda" to attempt GPU acceleration if available.'
    ))
    argparser.add_argument('-d', '--debug', action='store_true', help=(
        'Run in debug mode on a subset of data for speed, saving plots but not data.'
    ))
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
    outdir = config.get('outdir', './')
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
    epoch_tmin = config.get('epoch_tmin', -0.2)
    epoch_tmax = config.get('epoch_tmax', 0.8)
    end_code = config.get('end_code', None)
    end_key = config.get('end_key', None)
    if end_code is not None or end_key is not None:
        epoch_data = False
        if end_code is None: # end_key is defined
            if end_key not in event_name_to_code:
                event_name_to_code[end_key] = int(end_key)
            end_code = event_name_to_code[end_key]
        elif end_key is None: # end_code is defined
            if end_code not in event_code_to_name:
                event_code_to_name[end_code] = str(end_code)
            end_key = event_code_to_name[end_code]
    else:
        epoch_data = True
    word_level_file = config.get('word_level_file', None)
    if word_level_file:
        word_level_events = pd.read_csv(word_level_file)
        assert 'condition' in word_level_events and 'word_onset_time' in word_level_events, (
                'File %s was provided via field ``word_level_file``, but it is not correctly formatted.'
                ' It must contain columns called "condition" and "word_onset_time" to enable alignment'
                ' to the events file.'
        ) % word_level_file
    else:
        word_level_events = None

    reject = dict(mag=8000e-15, grad=8000e-13)  # T, T/m
    flat = dict(mag=1e-15, grad=1e-13)  # T/m

    # Container for output DataFrame
    responses = []
    events = []

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

        # Load events
        all_events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002)
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

        if epoch_data:
            if word_level_events is None:
                epoch_events = all_events
                event_id = event_name_to_code
            else:
                epoch_events_src = pd.DataFrame(all_events, columns=['time', 'something', 'condition'])
                epoch_events_src['onset_time'] = epoch_events_src['time']
                epoch_events_src = epoch_events_src[epoch_events_src.condition.isin(event_code_to_name)]
                epoch_events_src.condition = event_mapper(epoch_events_src.condition)
                epoch_events_src = pd.merge(
                    epoch_events_src, word_level_events.reset_index(),
                    on='condition',
                    how='inner'
                )
                _event_times = epoch_events_src['word_onset_time'] / time_scale + epoch_events_src['time']
                _event_other = epoch_events_src[['something', 'index']].values
                sel = np.logical_and(_event_times * time_scale >= expt_start, _event_times * time_scale <=  expt_end)
                _event_times = _event_times[sel]
                _event_other = _event_other[sel]
                epoch_events_src = epoch_events_src[sel]
                epoch_events = np.concatenate(
                    [_event_times[..., None], _event_other],
                    axis=1
                ).astype(int)
                del epoch_events_src['time']
                del epoch_events_src['something']
                event_id = None

            data = mne.Epochs(
                raw,
                events=epoch_events,
                event_id=event_id,
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
            responses.append(_responses)

        else:
            data = raw
            epoch_data = []
            epoch_codes = []
            epoch_starts = []
            epoch_ends = []
            seek_start = True
            _event_code = None
            _event_start = None
            _event_end = None
            epoch_ix = 0
            if resample_to:
                data.load_data()
            for t, _, code in all_events:
                if seek_start:
                    if code in event_code_to_name:
                        seek_start = False
                        _event_code = code
                        _event_start = t
                else:
                    if code == end_code:
                        assert _event_code is not None, "Finished with invalid event code at sample %s." % t
                        assert _event_start is not None, "Finished with invalid event start time at sample %s." % t

                        _event_end = t

                        start = min(_event_start * time_scale + epoch_tmin, data.times.max())
                        end = min(_event_end * time_scale + epoch_tmax, data.times.max())

                        _data = data.copy().crop(start, end)

                        if resample_to:
                            stderr('Resampling...\n')
                            _data = _data.resample(resample_to, n_jobs=n_jobs)

                        _responses = _data.to_data_frame(picks='meg')
                        _responses['subject'] = os.path.basename(subject_dir)
                        _responses['epoch'] = epoch_ix
                        _responses['condition'] = event_code_to_name[_event_code]
                        responses.append(_responses)

                        _events = pd.DataFrame({
                            'subject': [os.path.basename(subject_dir)],
                            'epoch': [epoch_ix],
                            'condition': [event_code_to_name[_event_code]],
                            'onset_time': [_event_start],
                            'offset_time': [_event_end]
                        })
                        events.append(_events)

                        seek_start = True
                        _event_code = None
                        _event_start = None
                        _event_end = None
                        epoch_ix += 1

    if events:
        info('Saving event table')
        events = pd.concat(events, axis=0).reset_index(drop=True)
        events['onset_time'] = events['onset_time'] * time_scale
        if 'offset_time' in events:
            events['offset_time'] = events['offset_time'] * time_scale
            events['duration'] = events['offset_time'] - events['onset_time']
        if word_level_events is not None:
            if epoch_data:
                events['word_pos'] = events.groupby('condition').cumcount() + 1
                on = ['condition', 'word_pos']
            else:
                on = ['condition']
            events = pd.merge(events, word_level_events, on=on, how='left')
            events.word_onset_time = events.word_onset_time + events.onset_time
            if 'word_offset_time' in events:
                events.word_offset_time = events.word_offset_time + events.onset_time
                events.word_duration = events.word_offset_time - events.word_onset_time

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        events.to_csv(
            os.path.join(outdir, os.path.basename(expt_name) + '_events.csv'),
            index=False
        )
    else:
        info('No output event data.')

    if responses:
        info('Saving response table')
        responses = pd.concat(responses, axis=0)
        responses = responses.reset_index()
        responses['time'] = responses['time'] * time_scale
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
