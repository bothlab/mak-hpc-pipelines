# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

import edlio
from utils import JobTemplateLoader
from gconst import SDS_ROOT

'''
This script is intended to be run the on bwForCluster MLS&WISO HPC
to create jobs for automated animal tracking from prerecorded videos.
'''


def schedule_dlc_job_for(scheduler, tmpl_loader, dlc_config_fname, collection_id, video_fnames, results_dir):
    job_fname = tmpl_loader.create_job_file(
        'dlc-analyze-videos.tmpl',
        'DLCTracking_{}'.format(collection_id),
        CONFIG_FNAME=dlc_config_fname,
        RESULTS_DIR=results_dir,
        VIDEO_FILES=video_fnames,
    )
    # submit the new job
    scheduler.schedule_job(job_fname, name=collection_id)


def setup_arguments(parser):
    '''Configure arguments for this task module'''

    parser.add_argument('--dlc-project', type=str, help='DeepLabCut project name.')
    parser.add_argument('--apattern', default='*', type=str, help='Animal glob pattern.')
    parser.add_argument('--dpattern', default='*', type=str, help='Date glob pattern.')
    parser.add_argument('--epattern', default='*', type=str, help='Experiment glob pattern.')
    parser.add_argument(
        '--video-dset',
        default='tis-camera',
        type=str,
        help='Video dataset name (multiple options can be given as semicolon-separated list).',
    )
    parser.add_argument('--out-dset', default='dlc-tracking', type=str, help='Name of the output dataset.')


def run(scheduler, data_location, options):
    '''Check which data we should analyze'''

    if not data_location:
        print('You need to spcify a data location!')
        sys.exit(1)
    if not options.dlc_project:
        print('DLC project name was empty - did you specify `--dlc-project`?')
        sys.exit(1)

    edl_root = os.path.join(SDS_ROOT, data_location)
    # fetch all EDL directories that match the given parameters
    edl_dirs = sorted(glob(os.path.join(edl_root, options.apattern, options.dpattern, options.epattern)))

    video_dset_names = [s.strip() for s in options.video_dset.split(';')]
    output_dset_name = options.out_dset
    dlc_config_fname = os.path.join(SDS_ROOT, 'DLCProjects', options.dlc_project, 'config.yaml')

    tmpl_loader = JobTemplateLoader()

    for edldir in edl_dirs:
        try:
            dcoll = edlio.load(edldir)
        except Exception as e:
            print('ERROR: {}'.format(str(e)))
            continue
        collection_id = dcoll.collection_idname
        videos_group = dcoll.group_by_name('videos')

        if dcoll.name.lower().startswith('inv_'):
            print('SKIP {}: Invalid / ignored experiment.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id, 'invalid/ignored')
            continue
        if not videos_group:
            print('SKIP {}: No "videos" group!'.format(collection_id))
            scheduler.mark_skipped_job(collection_id, 'no videos')
            continue
        for vdset_name in video_dset_names:
            dset = videos_group.dataset_by_name(vdset_name)
            if dset:
                break
        if not dset:
            print(
                'SKIP {}: No dataset with name(s): {}!'.format(collection_id, ', or '.join(video_dset_names))
            )
            scheduler.mark_skipped_job(collection_id, 'no recognized video dset')
            continue

        aux_data = None
        for ad in dset.aux_data:
            if ad.file_type == 'tsync':
                aux_data = ad
                break

        valid_videos = []
        for data_fname, adata_fname in zip(dset.data.part_paths(), aux_data.part_paths()):
            # we only want to include videos which have a certain number of frames
            if os.stat(adata_fname).st_size < 500:
                print(
                    'DATA-SKIP: Video "{}" in "{}" was too short.'.format(
                        data_fname.replace(dset.path, '').strip('/'), collection_id
                    )
                )
                scheduler.mark_skipped_job(collection_id, 'video too short')
                continue
            valid_videos.append(data_fname)

        if not valid_videos:
            print('SKIP {}: No valid video files found.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id, 'no valid videos')
            continue

        results_dir = os.path.join(dcoll.path, output_dset_name)
        os.makedirs(results_dir, exist_ok=True)
        if list(glob(os.path.join(results_dir, '*.h5'))):
            print('SKIP {}: Apparently already analyzed.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id)
            continue

        schedule_dlc_job_for(
            scheduler, tmpl_loader, dlc_config_fname, collection_id, valid_videos, results_dir
        )
