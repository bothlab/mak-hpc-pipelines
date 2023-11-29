# -*- coding: utf-8 -*-

'''
Create MIN1PIPE analysis jobs for data in a Syntalos-generated
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
Create MIN1PIPE fluorescence activity analysis jobs.
'''


def schedule_min1pipe_job_for(scheduler, tmpl_loader, collection_id, video_fnames, write_video, results_dir):
    job_fname = tmpl_loader.create_job_file(
        'min1pipe-analyze.tmpl',
        'MIN1PIPE_{}'.format(collection_id),
        WRITE_VIDEO=write_video,
        RESULTS_DIR=results_dir,
        RAW_VIDEO_FILES=video_fnames,
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
    parser.add_argument(
        '--out-dset', default='fluo-mp-analysis', type=str, help='Name of the output dataset.'
    )
    parser.add_argument('--no-overview-video', action='store_true', help='Disable overview video generation.')


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
            scheduler.mark_skipped_job(collection_id, 'no miniscope dset')
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
        if list(glob(os.path.join(results_dir, '*data_processed.mat'))):
            print('SKIP {}: Apparently already analyzed.'.format(collection_id))
            scheduler.mark_skipped_job(collection_id)
            continue

        schedule_min1pipe_job_for(
            scheduler,
            tmpl_loader,
            collection_id,
            video_fnames=valid_videos,
            write_video=not options.no_overview_video,
            results_dir=results_dir,
        )
