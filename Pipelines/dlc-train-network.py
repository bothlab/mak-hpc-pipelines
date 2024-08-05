#!/usr/bin/env python3

import os
import sys

import torch
import deeplabcut as dlc
from utils.messages import (
    print_info,
    print_task,
    print_warn,
    print_error,
    print_header,
    print_section,
)

if len(sys.argv) < 2:
    print_error('No config file given as first argument!')
    sys.exit(1)
config_fname = sys.argv[1]

NET_TYPE = 'hrnet_w32'

print_header('Training DLC network for: {}'.format(config_fname.replace('/mnt/sds-hd/', '', 1)))

print_info(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print_info(f'CUDA version: {torch.version.cuda}')
if torch.backends.mps.is_available():
    print_info(f'ROCm version: {torch.version.hip}')
print_info(f'DeepLabCut version: {dlc.__version__}')
sys.stdout.flush()

print_section('Creating Training Dataset')
dlc.create_training_dataset(config_fname, net_type=NET_TYPE)

print_section('Training Network')
dlc.train_network(
    config_fname,
    shuffle=1,
    trainingsetindex=0,
    max_snapshots_to_keep=5,
    autotune=False,
    displayiters=50,
    saveiters=15000,
    maxiters=30000,
    allow_growth=True,
    batch_size=4,
)

print_section('Evaluating Network')
dlc.evaluate_network(config_fname, Shuffles=[1], plotting=True)

print_info('Done.')
