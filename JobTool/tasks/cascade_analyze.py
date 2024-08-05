# -*- coding: utf-8 -*-

"""
Create CASCADE analysis jobs for data in a Syntalos-generated
data collection.

@author: Matthias Klumpp
"""

import os
import sys
from glob import glob
from argparse import ArgumentParser

import edlio
from utils import JobTemplateLoader
from gconst import Globals

"""
Create CASCADE spike-inference from calcium imaging jobs.
"""


def schedule_cascade_job_for(
    scheduler,
    tmpl_loader,
    collection_id,
    minian_dir,
    model_name,
    results_dir,
):
    job_fname = tmpl_loader.create_job_file(
        'cascade-analyze.tmpl',
        'Cascade_{}'.format(collection_id),
        MINIAN_DIR=minian_dir,
        MODEL_NAME=model_name,
        RESULTS_DIR=results_dir,
    )

    # submit the new job
    scheduler.schedule_job(job_fname, name=collection_id)


def setup_arguments(parser):
    '''Configure arguments for this task module'''

    parser.add_argument('--apattern', default='*', type=str, help='Animal glob pattern.')
    parser.add_argument('--dpattern', default='*', type=str, help='date glob pattern.')
    parser.add_argument('--epattern', default='*', type=str, help='Experiment glob pattern.')
    parser.add_argument(
        '--minian-dset',
        default='caimg-analysis/minian',
        type=str,
        help='Minian dataset name.',
    )
    parser.add_argument(
        '--model-name',
        default='GCaMP6f_mouse_30Hz_smoothing200ms',
        dest='model_name',
        help='The name of the model to use.',
    )
    parser.add_argument(
        '--out-dset', default='caimg-analysis/cascade', type=str, help='Name of the output dataset.'
    )


def run(scheduler, data_location, options):
    """Check which data we should analyze"""

    if not data_location:
        print('You need to specify a data location!')
        sys.exit(1)

    edl_root = os.path.join(Globals.SDS_ROOT, data_location)
    minian_dset_name = options.minian_dset
    output_dset_name = options.out_dset

    # fetch all EDL directories that match the given parameters
    edl_dirs = sorted(glob(os.path.join(edl_root, options.apattern, options.dpattern, options.epattern)))
    if not edl_dirs:
        print('No EDL data found in {}'.format(edl_root))
        return

    tmpl_loader = JobTemplateLoader()
    for edldir in edl_dirs:
        if not os.path.isdir(edldir):
            continue
        try:
            dcoll = edlio.load(edldir)
        except edlio.EDLError as e:
            print('SKIP: {} (reason: {})'.format(edldir, str(e)))
            continue
        print('Check: {}'.format(edldir))
        collection_id = dcoll.collection_idname
        minian_dset_path = os.path.join(dcoll.path, minian_dset_name)
        if not os.path.exists(os.path.join(minian_dset_path, 'data', 'C.zarr')):
            print('SKIP {}: No complete Minian dataset found.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id, 'no Minian calcium trace')
            continue

        results_dir = os.path.join(dcoll.path, output_dset_name)
        os.makedirs(results_dir, exist_ok=True)
        if os.path.isfile(os.path.join(results_dir, 'attributes.toml')):
            print('SKIP {}: Apparently already analyzed.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id)
            continue

        schedule_cascade_job_for(
            scheduler,
            tmpl_loader,
            collection_id,
            minian_dir=minian_dset_path,
            model_name=options.model_name,
            results_dir=results_dir,
        )
