#!/usr/bin/env python3

import os
import sys
import subprocess

# os.environ['DLClight'] = 'True'
# import deeplabcut as dlc
import deeplabcut as dlc
import tensorflow as tf
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

print_header('Training DLC network for: {}'.format(config_fname.replace('/mnt/sds-hd/', '', 1)))

print_info('TensorFlow Version: {}'.format(tf.__version__))
print_info('DeepLabCut Version: {}'.format('dlc-core alpha'))
# print_info('DeepLabCut Version: {}'.format(dlc.__version__))
sys.stdout.flush()
subprocess.check_call(['nvcc', '--version'])

print_section('Creating Training Dataset')
dlc.create_training_dataset(config_fname, Shuffles=[1])

print_section('Training Network')
dlc.train_network(config_fname, shuffle=1, saveiters=5000, displayiters=50, maxiters=500000)

print_section('Evaluating Network')
dlc.evaluate_network(config_fname)

print_info('Done.')
