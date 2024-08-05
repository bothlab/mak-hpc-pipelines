#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'edlio'))
from tasks import TASK_MODS, load_task_module
from taskscheduler import TaskScheduler
import gconst

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

    parser = ArgumentParser(description='Create & schedule jobs on bwForCluster')
    subparsers = parser.add_subparsers(dest='sp_name', title='subcommands')

    parser.add_argument('--show-task-modules', action='store_true', help='List all known task modules.')
    parser.add_argument(
        '--shell', action='store_true', help='Run an interactive shell with the Mamba env sourced.'
    )

    task_run_fns = {}
    for tm in TASK_MODS.keys():
        sp = subparsers.add_parser(tm)
        sp.add_argument('--dry', action='store_true', help='Simulate a run without scheduling any jobs.')
        sp.add_argument('--local', action='store_true', help='Run jobs locally without SLURM job scheduler.')
        sp.add_argument(
            '--root',
            type=str,
            help='Override the data root location.',
        )
        sp.add_argument(
            '-d',
            '--data-location',
            type=str,
            help='Location of the data (relative to SDS root), or task-specific data name.',
        )

        setup_fn, run_fn = load_task_module(tm)
        task_run_fns[tm] = run_fn
        if setup_fn:
            setup_fn(sp)
            sp.set_defaults(func=None)
        if not run_fn:
            raise Exception('Module "{}" is invalid: No "run" function is present!'.format(tm))

    # List task modules
    options = parser.parse_args(sys.argv[1:])
    if options.show_task_modules:
        for tm in TASK_MODS.keys():
            print('* {}'.format(tm))
        return

    if options.root:
        gconst.Globals.SDS_ROOT = os.path.realpath(options.root)
    print('Global root is: {}'.format(gconst.Globals.SDS_ROOT))

    if options.shell:
        script_helper = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'Setup', 'run-mamba-shell.sh'
        )
        os.execv(script_helper, [script_helper])
        return

    # sanity check
    scheduler = TaskScheduler()
    scheduler.dry_run = options.dry
    scheduler.run_local = options.local
    if not scheduler.check_prerequisites():
        sys.exit(1)

    # run submodule
    if mod_name not in task_run_fns:
        print('ERROR: Unable to find task module with name "{}"'.format(task_run_fns.keys()))
        sys.exit(3)
    run = task_run_fns[mod_name]
    run(scheduler, data_location=options.data_location, options=options)

    # print summary
    scheduler.print_jobs_summary()


if __name__ == '__main__':
    main()
    sys.exit(0)
