import sys
import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from . import templates
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Initialize a template config file for cleaning or data extraction
    ''')
    argparser.add_argument('-m', '--mode', default=None, help='Config type to create. Either ``clean`` or ``extract``.')
    argparser.add_argument('-s', '--subjects_dir', default=None, help='Path to subjects directory (used if ``mode`` is ``extract``).')
    argparser.add_argument('-n', '--expt_name', default=None, help='Experiment name (used if ``mode`` is ``extract``).')
    args = argparser.parse_args()

    mode = args.mode
    subjects_dir = args.subjects_dir
    expt_name = args.expt_name
    if not mode:
        while not mode:
            ans = input('Which kind of config file do you want, cleaning (c) or extraction (e)? [e]/c >>> ')
            if not ans.strip() or ans.strip().lower() == 'e':
                mode = 'extract'
            elif ans.strip().lower() == 'c':
                mode = 'clean'
            else:
                sys.stderr.write('Unrecognized answer.\n')
    assert mode in ('clean', 'extract'), 'Unrecognized config mode: %s. Exiting.' % mode

    # Read in template
    if mode == 'clean':
        cfg_name = 'clean_meg.yml'
    else: # mode == 'extract'
        cfg_name = 'extract_meg.yml'
    with pkg_resources.open_text(templates, cfg_name) as f:
        config_txt = list(f.readlines())

    if mode == 'extract':
        # Process config
        assert expt_name, 'If mode is extract, expt_name required.'
        assert subjects_dir, 'If mode is extract, subjects_dir required.'

        # Collect relevant subject directories
        subjects = []
        for x in os.listdir(subjects_dir):
            protocol_path = os.path.join(subjects_dir, x, 'protocol.txt')
            if os.path.exists(protocol_path):
                expts = set()
                with open(protocol_path, 'r') as f:
                    for _line in f:
                        _name = _line.strip().split(',')[0].strip()
                        expts.add(_name)
                if expt_name in expts:
                    subjects.append(os.path.realpath(os.path.normpath(os.path.join(subjects_dir, x))))

        in_subjects = False
        config_new = []
        for line in config_txt:
            if line.strip().startswith('expt_name'):
                line = line.replace('<Name>', expt_name)
            if line.strip().startswith('subjects:'):
                in_subjects = True
                config_new.append(line)
                for subject in subjects:
                    config_new.append('  - %s\n' % subject)
            else:
                if in_subjects and len(line.strip().split(':')) > 1:
                    in_subjects = False
                elif (in_subjects and not line.strip().startswith('-')) or not in_subjects:
                    config_new.append(line)

        config_txt = config_new

    for line in config_txt:
        sys.stdout.write(line)



