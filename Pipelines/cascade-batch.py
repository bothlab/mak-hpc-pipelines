#!/usr/bin/env python

"""
Run spike inference on Minian-generated data using Cascade.

@author: Matthias Klumpp
"""

# ignore future warnings in this pipeline (useful for development, but just noise here)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import typing as T
import logging as log
import argparse

import yaml
import zarr
import edlio
import numpy as np
import xarray as xr
import tomlkit
import dask.array as darr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cascade2p import checks

checks.check_packages()

from cascade2p import cascade
from utils.messages import (
    print_info,
    print_task,
    print_warn,
    print_error,
    print_header,
    print_section,
)
from cascade2p.utils import (
    plot_dFF_traces,
    plot_noise_level_distribution,
    plot_noise_matched_ground_truth,
)
from cascade2p.utils_discrete_spikes import infer_discrete_spikes

# Set up logging level and format
log.basicConfig(
    format='%(asctime)s %(levelname).1s: %(message)s', level=log.INFO, datefmt='%Y-%m-%d %H:%M:%S'
)


def open_minian_dataset(dpath: T.Union[str, os.PathLike]):
    if os.path.isfile(dpath):
        ds = xr.open_dataset(dpath).chunk()
    elif os.path.isdir(dpath):
        dslist = []
        for d in os.listdir(dpath):
            arr_path = os.path.join(dpath, d)
            if os.path.isdir(arr_path):
                arr = list(xr.open_zarr(arr_path).values())[0]
                arr.data = darr.from_zarr(os.path.join(arr_path, arr.name), inline_array=True)
                dslist.append(arr)
        ds = xr.merge(dslist, compat="no_conflicts")
    else:
        raise Exception('Could not find {}!'.format(dpath))

    return ds


def load_minian_parameters(minian_dir):
    """
    Load Minian analysis parameter file.
    """

    def unknown_constructor(loader, node):
        return None

    class CustomSafeLoader(yaml.SafeLoader):
        pass

    # Add the constructor for unknown tags
    yaml.add_constructor(None, unknown_constructor, CustomSafeLoader)

    with open(os.path.join(minian_dir, 'minian-parameters.yaml')) as f:
        return yaml.load(f, Loader=CustomSafeLoader)


def find_edl_collection(start_dir):
    '''Find EDL collection name'''
    edl_root_dir = None
    current_dir = os.path.normpath(os.path.join(start_dir, '..'))
    while True:
        mf_fname = os.path.join(current_dir, 'manifest.toml')
        if os.path.isfile(mf_fname):
            with open(mf_fname, 'r') as f:
                md = tomlkit.load(f)
                if md.get('type') == 'collection':
                    edl_root_dir = current_dir
                    break
        else:
            if not os.path.isfile(os.path.join(current_dir, '..', 'manifest.toml')):
                break
        current_dir = os.path.normpath(os.path.join(current_dir, '..'))

    if not edl_root_dir:
        return None

    dcoll = edlio.load(edl_root_dir)
    return dcoll


def make_short_edl_id(dcoll):
    """Get a very short ID to identify this data collection."""
    if dcoll.collection_id:
        return str(dcoll.collection_id)[:6]
    return '00'


def run(options):
    minian_dir = options.minian_dir
    results_dir = options.dest_dir
    cascade_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Tools', 'Cascade')
    cascade_dir = os.path.abspath(cascade_dir)

    if not minian_dir:
        print_error('No Minian data source directory set.')
        sys.exit(1)
    if not results_dir:
        print_error('No destination to store analyzed data set.')
        sys.exit(1)

    print_header('Cascade Inference: {}'.format(os.path.dirname(results_dir).replace('/mnt/sds-hd/', '', 1)))
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)

    minian_params = load_minian_parameters(minian_dir)
    minian_ds = open_minian_dataset(os.path.join(minian_dir, 'data'))
    dcoll = find_edl_collection(minian_dir)
    exp_id = '00'
    if dcoll:
        exp_id = make_short_edl_id(dcoll)

    frame_rate = minian_params['framerate_fps']

    # we need to deep-copy the xarray into a numpy array,
    # as Cascade will fail with xarrays
    traces = np.array(minian_ds.C)
    print('Number of neurons in dataset:', traces.shape[0])
    print('Number of timepoints in dataset:', traces.shape[1])

    np.random.seed(42)
    noise_levels = plot_noise_level_distribution(traces, frame_rate)

    plt.savefig(os.path.join(results_dir, f'noise-level-histogram_{exp_id}.svgz'), bbox_inches='tight')
    plt.close('all')

    print_section('Select pretrained model and apply to dF/F data')

    os.chdir(cascade_dir)
    cascade.download_model(options.model_name, verbose=1)
    spike_prob = cascade.predict(options.model_name, traces)
    os.chdir(results_dir)

    # print_section('Plot noise-matched examples from the ground truth')
    # median_noise = np.round(np.median(noise_levels))
    # nb_traces = 4
    # duration = 50  # seconds
    # os.chdir(cascade_dir)
    # plot_noise_matched_ground_truth(options.model_name, median_noise, frame_rate, nb_traces, duration)
    # os.chdir(results_dir)

    print_section('Fill up probabilities (output of the network) with discrete spikes')

    os.chdir(cascade_dir)
    discrete_approximation, spike_time_estimates = infer_discrete_spikes(spike_prob, options.model_name)
    os.chdir(results_dir)

    print_section('Plot example predictions together with discrete spikes')

    neuron_indices = np.random.randint(spike_prob.shape[0], size=10)
    plot_dFF_traces(
        traces,
        neuron_indices,
        frame_rate,
        spiking=spike_prob,
        discrete_spikes=spike_time_estimates,
        y_range=(-2.0, 6.0),
    )
    plt.savefig(os.path.join(results_dir, f'dFF-traces-spike-predictions_{exp_id}.svgz'), bbox_inches='tight')
    plt.close('all')

    print_task('Saving predictions to disk')

    save_path = os.path.join(results_dir, f'discrete-spikes_{exp_id}')
    np.savez(
        save_path,
        spike_prob=spike_prob,
        discrete_approximation=discrete_approximation,
        spike_time_estimates=spike_time_estimates,
    )

    print_task('Writing parameter information file')
    info = {}
    attr_fname = os.path.join(results_dir, 'attributes.toml')

    info['minian_dir'] = minian_dir
    info['model_name'] = options.model_name
    info['n_neurons'] = traces.shape[0]

    with open(attr_fname, 'w') as f:
        tomlkit.dump(info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze arbitrary Miniscope videos using CaImAn CNMF-E')
    parser.add_argument(
        '-d', '--destdir', action='store', dest='dest_dir', help='Destination to store the analyzed data in.'
    )
    parser.add_argument(
        '--model-name',
        default='GCaMP6f_mouse_30Hz_smoothing200ms',
        dest='model_name',
        help='The name of the model to use.',
    )
    parser.add_argument(
        'minian_dir', action='store', nargs='?', help='Directory with the Miniscope video files to analyze.'
    )

    optn = parser.parse_args(sys.argv[1:])
    run(optn)
