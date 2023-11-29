# -*- coding: utf-8 -*-
#
# Copyright (C) 2017-2022 Matthias Klumpp <matthias@tenstral.net>
#
# Licensed under the GNU Lesser General Public License Version 3

import sys
import importlib

'''
Determine which runner is responsible for which job type.
'''
TASK_MODS = {
    'dlc-analyze-videos': 'tasks.dlc_analyze_videos',
    'dlc-train-network': 'tasks.dlc_train_network',
    'min1pipe-analyze': 'tasks.min1pipe_analyze',
    'caiman-analyze': 'tasks.caiman_analyze',
    'minian-analyze': 'tasks.minian_analyze',
}


def load_task_module(what):
    if what not in TASK_MODS:
        print('ERROR: Unable to find task module with name "{}"'.format(what))
        sys.exit(3)
    path = TASK_MODS[what]
    mod = importlib.import_module(path)

    setup_fn = mod.setup_arguments if hasattr(mod, 'setup_arguments') else None
    run_fn = mod.run if hasattr(mod, 'run') else None
    return setup_fn, run_fn
