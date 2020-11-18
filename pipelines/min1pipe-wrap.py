#!/usr/bin/env python3
'''
MIN1PIPE video conversion & plotting wrapper.

@author: Matthias Klumpp
'''

import os
import sys
import subprocess
import argparse
import tempfile
import cv2 as cv
import numpy as np
import toml
import h5py
import logging as log
from datetime import datetime
from contextlib import contextmanager
from natsort import natsorted
from utils.messages import print_header, print_section, print_error
from min1pipe.mpipeplot import create_overview_video, plot_temporal_overview, plot_spatial_overview


M1P_NSIZE = 10    # structure element for background removal
                  #  half neuron size; 9 for Inscopix and 5
                  #  for UCLA, with 0.5 spatialr separately
M1P_SPATIALR = 1  # no spatial downsampling
M1P_ISMC = True   # run movement correction
M1P_FLAG = 1      # use auto seeds selection; 2 if manual

KEEP_REGMAT = False          # whether to keep the huge registration mapping when the run has finished


# Set up logging level and format
log.basicConfig(format='%(asctime)s %(levelname).1s - %(message)s',
                level=log.INFO,
                datefmt='%Y-%m-%d %H:%M:%S')


def find_edl_collection_root_dir(start_dir):
    ''' Find EDL collection root directory from any subdir '''
    edl_root_dir = None
    current_dir = os.path.normpath(os.path.join(start_dir, '..'))
    while True:
        mf_fname = os.path.join(current_dir, 'manifest.toml')
        if os.path.isfile(mf_fname):
            with open(mf_fname, 'r') as f:
                md = toml.load(f)
                if md.get('type') == 'collection':
                    edl_root_dir = current_dir
                    break
        else:
            break
        current_dir = os.path.normpath(os.path.join(current_dir, '..'))
    return edl_root_dir


def load_basic_metadata(edl_root):
    '''
    Load some basic metadata from the EDL tree which we may want to tag our
    analyzed data files with.
    '''
    if not edl_root:
        return {}

    res = {}
    mf_fname = os.path.join(edl_root, 'manifest.toml')
    if os.path.isfile(mf_fname):
        res['collection_name'] = os.path.basename(edl_root.rstrip(os.path.sep))  # collection name
        with open(mf_fname, 'r') as f:
            md = toml.load(f)
            res['collection_id'] = md.get('collection_id')  # collection UUID
            res['time_created'] = md.get('time_created')  # time when experiment was recorded
    else:
        log.error('No manifest file was found in (suspected) EDL directory {}'.format(edl_root))
    af_fname = os.path.join(edl_root, 'attributes.toml')
    if os.path.isfile(af_fname):
        with open(af_fname, 'r') as f:
            attrs = toml.load(f)
            res['subject_id'] = attrs.get('subject_id')  # animal ID
    return res


def write_parameters_file(results_dir, overview_video):
    '''
    Write information attributes file containing parameters used
    to process this file.
    '''
    log.info('Writing parameter information file.')
    info = {}
    attr_fname = os.path.join(results_dir, 'attributes.toml')
    if os.path.isfile(attr_fname):
        with open(attr_fname, 'r') as f:
            info = toml.load(f)

    mparams = dict(se=M1P_NSIZE,
                   spatialr=M1P_SPATIALR,
                   mc=M1P_ISMC,
                   seed_sel=M1P_FLAG,
                   keep_regmat=KEEP_REGMAT,
                   have_overview_video=overview_video)
    info['min1pipe'] = {'used_params': mparams}

    with open(attr_fname, 'w') as f:
        toml.dump(info, f)


@contextmanager
def open_hdf5_with_matlab_header(fname, **kwargs):
    now = datetime.now()
    # fake MATLAB HDF5 header string
    s = 'MATLAB 7.3 MAT-file, Platform: unknown-any ' \
        + '0.1' + ', Created on: ' \
        + now.strftime('%a %b %d %H:%M:%S %Y') \
        + ' HDF5 schema 1.00 .'

    # create HDF5 file template
    hf = h5py.File(fname, mode='w', userblock_size=512)
    hf.close()
    hf = None

    # Make the bytearray while padding with spaces up to
    # 128-12 (the minus 12 is there since the last 12 bytes
    # are special).
    b = bytearray(s + (128 - 12 - len(s)) * ' ', encoding='utf-8')
    # Add 8 nulls (0) and the magic number that MATLAB uses.
    b.extend(bytearray.fromhex('00000000 00000000 0002494D'))

    # write MATLAB userblock
    with open(fname, 'r+b') as f:
        f.write(b)
    # reopen the file in append-mode, skipping userblock
    hf = h5py.File(fname,
                   mode='a',
                   libver=('earliest', 'v110'),
                   **kwargs)
    yield hf
    hf.flush()
    hf.close()


def videos_to_mat(vid_fnames, mat_fname):
    ''' Convert list of videos into a single MATLAB-compatible HDF5 file. '''
    if not vid_fnames:
        raise Exception('No videos to analyze have been passed.')

    # get common video properties
    vreader = cv.VideoCapture(vid_fnames[0])
    if not vreader.isOpened():
        raise Exception('Unable to read from video file!')

    width = int(vreader.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vreader.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vreader.get(cv.CAP_PROP_FPS))
    frames_total = 0
    vreader.release()

    # validate all videos
    for v_fname in vid_fnames:
        vreader = cv.VideoCapture(vid_fnames[0])
        if not vreader.isOpened():
            raise Exception('Unable to read from video file!')

        if width != int(vreader.get(cv.CAP_PROP_FRAME_WIDTH)):
            print_error('Video {} has invalid width (expected {}px)'.format(v_fname, width))
            sys.exit(2)
        if height != int(vreader.get(cv.CAP_PROP_FRAME_HEIGHT)):
            print_error('Video {} has invalid height (expected {}px)'.format(v_fname, height))
            sys.exit(2)
        if fps != int(vreader.get(cv.CAP_PROP_FPS)):
            print_error('Video {} has invalid framerate (expected {}fps)'.format(v_fname, fps))
            sys.exit(2)
        frames_total += int(vreader.get(cv.CAP_PROP_FRAME_COUNT))
        vreader.release()

    # read all videos
    frames_all = np.zeros((frames_total, width, height), dtype=np.single)
    cur_total_frame_n = 0
    for v_fname in vid_fnames:
        vreader = cv.VideoCapture(v_fname)
        if not vreader.isOpened():
            raise Exception('Unable to read from video file!')

        fmt = int(vreader.get(cv.CAP_PROP_FORMAT))
        expected_frames_n = int(vreader.get(cv.CAP_PROP_FRAME_COUNT))

        log.info('Reading video: {}'.format(v_fname))
        frame_n = 0
        while True:
            ret, mat = vreader.read()
            if not ret:
                break
            cur_total_frame_n += 1
            frame_n += 1
            if frame_n > expected_frames_n:
                raise Exception('Read more frames than the expected numer ({})'.format(expected_frames_n))

            # we should already have an 8-bit grayscale image, but we convert it just in case
            gray_mat = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)
            frames_all[cur_total_frame_n - 1, :, :] = gray_mat.T.astype(np.single)
        vreader.release()

        if expected_frames_n != frame_n:
            raise Exception('Read an unexpected amount of frames (expected {}, got {}'.format(expected_frames_n, frame_n))
        log.info('Video file contained {} frames @ {}fps {}x{}px, fmt-{}; now converted to 8bit gray type:single matrices'.format(frame_n, fps, width, height, fmt))

    if frames_total != cur_total_frame_n:
        raise Exception('Read an invalid amount of frames: {} instead of {}'.format(cur_total_frame_n, frames_total))

    log.info('Total number of frames to process: {}'.format(frames_total))
    with open_hdf5_with_matlab_header(mat_fname) as f:
        log.info('Saving raw video data to MATLAB-compatible HDF5 file...')
        # direct assignment is *much* faster than writing to the HDF5 file in chunks
        f.create_dataset('frame_all', data=frames_all)
    log.info('Video conversion for mmap done.')

    return (width, height), fps


def run_min1pipe(fps, spatialr, se, ismc, flag, fname_video_raw, dir_results):
    ''' Run MIN1PIPE MATLAB script for HPC '''
    ml_helper_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'min1pipe')

    ismc = 'true' if ismc else 'false'
    if not dir_results.endswith('/'):
        dir_results = dir_results + '/'

    log.info('Running MIN1PIPE.')
    sys.stdout.flush()
    ret = subprocess.call(['matlab',
                           '-nodisplay',
                           '-batch',
                           'min1pipe_bwHPC({fps}, {spatialr}, {se}, {ismc}, {flag}, \'{fname_video_raw}\', \'{dir_results}\');'.format(
                               fps=fps, spatialr=spatialr, se=se, ismc=ismc, flag=flag,
                               fname_video_raw=fname_video_raw.replace('\'', '\'\''), dir_results=dir_results.replace('\'', '\'\''))],
                          cwd=ml_helper_dir)
    if ret != 0:
        log.error('MIN1PIPE run failed.')
        sys.exit(3)


def main(options):
    if not options.videos:
        print_error('No video files set.')
        sys.exit(1)
    if not options.dest_dir:
        print_error('No destination for analyzed data set.')
        sys.exit(1)
    results_dir = options.dest_dir
    write_overview_video = optn.write_video

    print_header('Analyzing Miniscope data: {}'.format(os.path.dirname(options.videos[0]).replace('/mnt/sds-hd/', '', 1)))
    os.makedirs(results_dir, exist_ok=True)

    edl_root_dir = find_edl_collection_root_dir(options.videos[0])
    if not edl_root_dir:
        edl_root_dir = find_edl_collection_root_dir(results_dir)
    edl_metadata = {}
    if edl_root_dir:
        log.info('Using EDL metadata from {}'.format(edl_root_dir))
        edl_metadata = load_basic_metadata(edl_root_dir)
    else:
        log.warning('No EDL metadata found.')

    log.info('Overview video {}'.format('enabled' if write_overview_video else 'disabled'))

    sys_tmpdir = os.environ.get('TMPDIR', '/scratch')
    if not os.path.isdir(sys_tmpdir):
        sys_tmpdir = '/var/tmp'
        if not os.path.isdir(sys_tmpdir):
            tempfile.gettempdir()

    dpmat_fname = os.path.join(results_dir, 'mp_data_processed.mat')
    supmat_fname = os.path.join(results_dir, 'mp_supporting.mat')
    regmat_fname = os.path.join(results_dir, 'mp_reg.mat')
    regpmat_fname = os.path.join(results_dir, 'mp_reg_post.mat')
    regprmat_fname = os.path.join(results_dir, 'mp_reg_post_res.mat')

    gen_fnames = [dpmat_fname, supmat_fname, regmat_fname, regpmat_fname, regprmat_fname]
    fnames_cleanup = []
    for fname_rm in gen_fnames:
        if os.path.isfile(fname_rm):
            fnames_cleanup.append(fname_rm)

    if fnames_cleanup:
        print_section('Housekeeping')
        log.info('Cleaning up data from a previous run.')
        for fname_rm in fnames_cleanup:
            log.info('Delete:  {}'.format(fname_rm))
            os.remove(fname_rm)

    write_parameters_file(results_dir, write_overview_video)
    with tempfile.TemporaryDirectory(prefix='mpipe_', dir=sys_tmpdir) as tdir:
        print_section('Preparing temporary data mmap file')
        frames_raw_fname = os.path.join(tdir, 'frames_raw.mat')
        _, fps = videos_to_mat(natsorted(options.videos),
                               frames_raw_fname)
        log.info('Data saved to {}'.format(frames_raw_fname))

        print_section('MIN1PIPE Pipeline')
        run_min1pipe(fps,
                     M1P_SPATIALR,
                     M1P_NSIZE,
                     M1P_ISMC,
                     M1P_FLAG,
                     frames_raw_fname,
                     results_dir)

        print_section('Creating Figures')

        # get experiment recording date as a string
        experiment_date_str = None
        if 'time_created' in edl_metadata:
            experiment_date_str = edl_metadata['time_created'].date().isoformat()

        log.info('Ploting temporal overview figure')
        plot_temporal_overview(dpmat_fname,
                               os.path.join(results_dir, 'temporal-overview.svg'),
                               subject_id=edl_metadata.get('subject_id'),
                               test_id=edl_metadata.get('collection_name'),
                               test_date=experiment_date_str)
        log.info('Ploting spatial overview figure')
        plot_spatial_overview(dpmat_fname,
                              os.path.join(results_dir, 'spatial-overview.svg'),
                              subject_id=edl_metadata.get('subject_id'),
                              test_id=edl_metadata.get('collection_name'),
                              test_date=experiment_date_str)
        log.info('Overview figures done')

        if write_overview_video:
            print_section('Creating Video')
            mkv_metadata = dict(analysis_time=datetime.now().isoformat())
            if edl_metadata:
                mkv_metadata = dict(title='{} [{}], {}: Calcium imaging signal analysis'.format(edl_metadata.get('subject_id'),
                                                                                                edl_metadata['collection_name'],
                                                                                                experiment_date_str),
                                    description=('Subject {} (experiment ID: {}, source data date: {}) ' +
                                                 'calcium imaging signal analysis using MIN1PIPE').format(edl_metadata.get('subject_id'),
                                                                                                          edl_metadata['collection_name'],
                                                                                                          experiment_date_str),
                                    collection_id=str(edl_metadata.get('collection_id')),
                                    recording_time=edl_metadata['time_created'].isoformat(),
                                    analysis_time=datetime.now().isoformat())
            movie_fname = os.path.join(results_dir, 'msig-overview.mkv')
            create_overview_video(dpmat_fname,
                                  regmat_fname,
                                  frames_raw_fname,
                                  movie_fname,
                                  fps,
                                  metadata=mkv_metadata,
                                  codec='h264')

        print_section('Cleanup')
        log.info('Deleting temporary files and cache data.')
        if not KEEP_REGMAT:
            for fname_rm in [regmat_fname, regpmat_fname, supmat_fname]:
                if os.path.isfile(fname_rm):
                    log.info('Delete: {}'.format(fname_rm))
                    os.remove(fname_rm)

    print()
    log.info('Success.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze arbitrary Miniscope videos using MIN1PIPE')
    parser.add_argument('-d', '--destdir', action='store', dest='dest_dir',
                        help='Destination to store the analyzed data in.')
    parser.add_argument('--write-video', action='store_true', dest='write_video',
                        help='Whether to render a video overview of the analyzed data.')
    parser.add_argument('videos', action='store', nargs='+',
                        help='The Miniscope video files to analyze.')

    optn = parser.parse_args(sys.argv[1:])
    main(optn)
