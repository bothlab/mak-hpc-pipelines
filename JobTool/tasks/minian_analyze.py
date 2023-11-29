# -*- coding: utf-8 -*-

'''
Create Minian analysis jobs for data in a Syntalos-generated
data collection.

@author: Matthias Klumpp
'''

import os
import sys
from glob import glob
from argparse import ArgumentParser

import edlio
from utils import JobTemplateLoader
from gconst import SDS_ROOT

'''
Create Minian fluorescence activity analysis jobs.
'''


def schedule_minian_job_for(
    scheduler, tmpl_loader, collection_id, video_dir, write_video, with_deconvolution, results_dir
):
    job_fname = tmpl_loader.create_job_file(
        'minian-analyze.tmpl',
        'Minian_{}'.format(collection_id),
        WRITE_VIDEO=write_video,
        WITH_DECON=with_deconvolution,
        RESULTS_DIR=results_dir,
        RAW_VIDEOS_DIR=video_dir,
    )

    # submit the new job
    scheduler.schedule_job(job_fname, name=collection_id)


def setup_arguments(parser):
    '''Configure arguments for this task module'''

    parser.add_argument('--apattern', default='*', type=str, help='Animal glob pattern.')
    parser.add_argument('--dpattern', default='*', type=str, help='date glob pattern.')
    parser.add_argument('--epattern', default='*', type=str, help='Experiment glob pattern.')
    parser.add_argument(
        '--video-dset',
        default='miniscope',
        type=str,
        help='Video dataset name (multiple options can be given as semicolon-separated list).',
    )
    parser.add_argument('--out-dset', default='minian-analysis', type=str, help='Name of the output dataset.')
    parser.add_argument('--no-overview-video', action='store_true', help='Disable overview video generation.')
    parser.add_argument(
        '--with-decon',
        action='store_true',
        help='Deconvolve raw image data before processing it further (usually not needed).',
    )


def run(scheduler, data_location, options):
    '''Check which data we should analyze'''

    if not data_location:
        print('You need to specify a data location!')
        sys.exit(1)

    edl_root = os.path.join(SDS_ROOT, data_location)
    video_dset_names = [s.strip() for s in options.video_dset.split(';')]
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
        videos_group = dcoll.group_by_name('videos')

        if dcoll.name.lower().startswith('inv_'):
            print('SKIP {}: Invalid / ignored experiment.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id, 'invalid/ignored')
            continue
        if not videos_group:
            print('SKIP {}: No "videos" group!'.format(collection_id))
            scheduler.mark_skipped_job(collection_id, 'no videos group')
            continue
        for vdset_name in video_dset_names:
            dset = videos_group.dataset_by_name(vdset_name)
            if dset:
                break
        if not dset:
            print(
                'SKIP {}: No dataset with name(s): {}!'.format(collection_id, ', or '.join(video_dset_names))
            )
            scheduler.mark_skipped_job(collection_id, 'no miniscope videos')
            continue

        if len(dset.aux_data) != 1:
            raise Exception(
                (
                    'Unexpected amount of auxiliary data found (expected only '
                    'one tsync dataset, got {} sets'
                ).format(len(dset.aux_data))
            )

        valid_videos = []
        for data_fname, adata_fname in zip(dset.data.part_paths(), dset.aux_data[0].part_paths()):
            # TODO: Skip videos that are too short?
            valid_videos.append(data_fname)

        if not valid_videos:
            print('SKIP {}: No valid video files found.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id, 'no valid videos')
            continue

        results_dir = os.path.join(dcoll.path, output_dset_name)
        os.makedirs(results_dir, exist_ok=True)
        if list(glob(os.path.join(results_dir, 'data', 'A.zarr'))):
            print('SKIP {}: Apparently already analyzed.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id)
            continue

        schedule_minian_job_for(
            scheduler,
            tmpl_loader,
            collection_id,
            video_dir=dset.path,
            write_video=not options.no_overview_video,
            with_deconvolution=options.with_decon,
            results_dir=results_dir,
        )
