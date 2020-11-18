#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
from argparse import ArgumentParser
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'edlio'))
from tasks import TASK_MODS, load_task_module


'''
This script is intended to be run the on bwForCluster MLS&WISO HPC
to create jobs for various tasks semi-automatically.

@author: Matthias Klumpp
'''


def main():
    if len(sys.argv) < 2:
        print('You need to specify a task module to run as first parameter!')
        sys.exit(1)

    mod_name = sys.argv[1]

    parser = ArgumentParser()
    parser.add_argument('--dry', action='store_true',
                        help='Simulate a run without scheduling any jobs.')
    parser.add_argument('-d', '--data-location', type=str,
                        help='Location of the data (relative to SDS root).')
    parser.add_argument('--show-task-modules', action='store_true',
                        help='List all known task modules.')
    if mod_name == '--help' or mod_name == '--show-task-modules':
        options = parser.parse_args(sys.argv[1:])
    else:
        options, targs = parser.parse_known_args(sys.argv[2:])

    if options.show_task_modules:
        for tm in TASK_MODS.keys():
            print('* {}'.format(tm))
        return

    run = load_task_module(mod_name)
    run(dry_run=options.dry, data_location=options.data_location, args=targs)


if __name__ == '__main__':
    if not shutil.which('sbatch'):
        print('SLURM batch job submission tool not found. Are we running on the HPC?')
        sys.exit(1)
    main()
    sys.exit(0)
