#!/usr/bin/env python

'''
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.

@author: Matthias Klumpp
'''

import os
import sys
import shutil
import logging as log
import argparse
import subprocess
from glob import glob

import cv2 as cv
import toml
import numpy as np
import caiman as cm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from utils.messages import (
    print_info,
    print_task,
    print_warn,
    print_error,
    print_header,
    print_section,
)
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params

# Set up logging level and format
log.basicConfig(
    format='%(asctime)s %(levelname).1s [%(filename)s:%(lineno)s] %(message)s',
    level=log.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)


def find_edl_collection_root_dir(start_dir):
    '''Find EDL collection root directory from any subdir'''
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


def write_parameters_file(results_dir, params, overview_video):
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

    mparams = params.copy()
    mparams['have_overview_video'] = overview_video
    info['caiman'] = {'params': mparams}

    with open(attr_fname, 'w') as f:
        toml.dump(info, f)


def estimates_write_movie(
    estm,
    imgs,
    q_max=99.75,
    q_min=2,
    gain_res=1,
    include_bck: bool = True,
    plot_text: bool = True,
    bpx=0,
    movie_name='results_movie.mkv',
    opencv_codec='VP90',
    use_color=False,
    gain_color=4,
    gain_bck=0.2,
):
    from caiman.source_extraction.cnmf.initialization import downscale

    log.info('Writing overview movie: {}'.format(movie_name.replace('/mnt/sds-hd/', '', 1)))
    fourcc = cv.VideoWriter_fourcc(*opencv_codec)
    vw_out = None  # for videowriter, once we can initialize it
    dims = imgs.shape[1:]
    maxmov = np.nan
    minmov = np.nan

    # write to temporary file initially, so we can later re-encode with different settings
    tmp_movie_name = '{}.tmp.mkv'.format(movie_name.replace('.mkv', '').replace('.mp4', ''))
    if os.path.lexists(tmp_movie_name):
        os.remove(tmp_movie_name)
    log.info('Temporary movie file: {}'.format(tmp_movie_name.replace('/mnt/sds-hd/', '', 1)))

    if 'movie' not in str(type(imgs)):
        imgs = cm.movie(imgs)

    if use_color:
        cols_c = np.random.rand(estm.C.shape[0], 1, 3) * gain_color

    chunk_size = 2 * 60 * 30  # frames in 2min @ 30fps
    for i in range(0, len(imgs), chunk_size):
        log.info('Processing video chunk {}'.format(i // chunk_size))
        frame_range = slice(i, i + chunk_size)
        imgs_chunk = imgs[frame_range]

        if use_color:
            Cs = np.expand_dims(estm.C[:, frame_range], -1) * cols_c
            # AC = np.tensordot(np.hstack((self.A.toarray(), self.b)), Cs, axes=(1, 0))
            Y_rec_color = np.tensordot(estm.A.toarray(), Cs, axes=(1, 0))
            Y_rec_color = Y_rec_color.reshape((dims) + (-1, 3), order='F').transpose(2, 0, 1, 3)

        AC = estm.A.dot(estm.C[:, frame_range])
        Y_rec = AC.reshape(dims + (-1,), order='F')
        Y_rec = Y_rec.transpose([2, 0, 1])
        if estm.W is not None:
            ssub_B = int(round(np.sqrt(np.prod(dims) / estm.W.shape[0])))
            B = imgs_chunk.reshape((-1, np.prod(dims)), order='F').T - AC
            if ssub_B == 1:
                B = estm.b0[:, None] + estm.W.dot(B - estm.b0[:, None])
            else:
                WB = estm.W.dot(
                    downscale(B.reshape(dims + (B.shape[-1],), order='F'), (ssub_B, ssub_B, 1)).reshape(
                        (-1, B.shape[-1]), order='F'
                    )
                )
                Wb0 = estm.W.dot(
                    downscale(estm.b0.reshape(dims, order='F'), (ssub_B, ssub_B)).reshape((-1, 1), order='F')
                )
                B = estm.b0.flatten('F')[:, None] + (
                    np.repeat(
                        np.repeat(
                            (WB - Wb0).reshape(
                                ((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'
                            ),
                            ssub_B,
                            0,
                        ),
                        ssub_B,
                        1,
                    )[: dims[0], : dims[1]].reshape((-1, B.shape[-1]), order='F')
                )
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        elif estm.b is not None and estm.f is not None:
            B = estm.b.dot(estm.f[:, frame_range])
            if 'matrix' in str(type(B)):
                B = B.toarray()
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        else:
            B = np.zeros_like(Y_rec)
        if bpx > 0:
            B = B[:, bpx:-bpx, bpx:-bpx]
            Y_rec = Y_rec[:, bpx:-bpx, bpx:-bpx]
            imgs_chunk = imgs_chunk[:, bpx:-bpx, bpx:-bpx]

        Y_res = imgs_chunk - Y_rec - B
        if use_color:
            if bpx > 0:
                Y_rec_color = Y_rec_color[:, bpx:-bpx, bpx:-bpx]
            mov = cm.concatenate(
                (
                    np.repeat(np.expand_dims(imgs_chunk - (not include_bck) * B, -1), 3, 3),
                    Y_rec_color + include_bck * np.expand_dims(B * gain_bck, -1),
                    np.repeat(np.expand_dims(Y_res * gain_res, -1), 3, 3),
                ),
                axis=2,
            )
        else:
            mov = cm.concatenate(
                (imgs_chunk - (not include_bck) * B, Y_rec + include_bck * B, Y_res * gain_res), axis=2
            )

        # initialize video writer, if that wasn't already done
        if vw_out == None:
            vw_out = cv.VideoWriter(
                tmp_movie_name, fourcc, 30.0, tuple([int(s) for s in mov[0].shape[1::-1]])
            )

            if q_max < 100:
                maxmov = np.nanpercentile(mov[0:10], q_max)
            else:
                maxmov = np.nanmax(mov)

            if q_min > 0:
                minmov = np.nanpercentile(mov[0:10], q_min)
            else:
                minmov = np.nanmin(mov)

        for iddxx, frame in enumerate(mov):
            frame = (frame - minmov) / (maxmov - minmov)

            if plot_text:
                text_width, text_height = cv.getTextSize(
                    'Frame = ' + str(iddxx), fontFace=5, fontScale=0.8, thickness=1
                )[0]
                cv.putText(
                    frame,
                    'Frame = ' + str(iddxx + i),
                    ((frame.shape[1] - text_width) - 20, text_height + 10),
                    fontFace=5,
                    fontScale=0.8,
                    color=(255, 255, 255),
                    thickness=1,
                )

            if frame.ndim < 3:
                frame = np.repeat(frame[:, :, None], 3, axis=-1)
            frame = np.minimum((frame * 255.0), 255).astype('u1')
            vw_out.write(frame)

    # FIXME: Since Anaconda's FFmpeg/OpenCV combination currently only really works with H.246,
    # we'll recompress the video with a different codec on a lower-quality setting and even scale
    # it down a bit to keep the file size reasonable
    # (white-noise-like input encodes especially poorly)
    log.info('Re-encoding movie for smaller file size.')
    subprocess.run(
        [
            'ffmpeg',
            '-nostats',
            '-nostdin',
            '-hide_banner',
            '-y',
            '-i',
            tmp_movie_name,
            '-vf',
            'scale=-1:512',
            '-c:v',
            'libsvtav1',
            '-b:v',
            '0',
            '-g',
            '300',
            '-la_depth',
            '120',
            '-qp',
            '49',  # quality setting, lower for higher quality but bigger file size
            '-preset',
            '7',
            '-c:a',
            'copy',
            movie_name,
        ],
        check=True,
    )
    os.remove(tmp_movie_name)

    log.info('Video written.')


def save_inferred_temporal_overview_plots(C, accepted_indices, fbasename):
    nr, T = C.shape
    sns.set_palette('bright')
    fbasename = fbasename.rstrip('/')

    # save accepted components overview
    fig = plt.figure(figsize=(24.0, 12.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.get_yaxis().set_visible(False)

    for i, ci in enumerate(accepted_indices):
        ax.plot(np.arange(T), C[ci] + (i * 100))
    sns.despine(trim=True, left=True)
    plt.savefig(fbasename + '_accepted.svg', transparent=True, bbox_inches='tight', dpi=200)
    plt.close(fig)

    # save rejected components overview
    fig = plt.figure(figsize=(24.0, 12.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.get_yaxis().set_visible(False)

    rejected_set = set(range(0, nr)) - set(accepted_indices)
    for i, ci in enumerate(rejected_set):
        ax.plot(np.arange(T), C[ci] + (i * 100))
    sns.despine(trim=True, left=True)
    plt.savefig(fbasename + '_rejected.svg', transparent=True, bbox_inches='tight', dpi=200)
    plt.close(fig)


def save_motioncorrection_shifts_plot(mc, fname):
    fig = plt.figure(figsize=(12.0, 6.0))
    plt.subplot(1, 2, 1)
    plt.imshow(mc.total_template_rig)  # plot template
    plt.subplot(1, 2, 2)
    plt.plot(mc.shifts_rig)  # plot rigid shifts
    plt.legend(['x shifts', 'y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')
    sns.despine(trim=True)
    plt.savefig(fname, transparent=True, bbox_inches='tight', dpi=200)
    plt.close(fig)


def run(options):
    if not options.videos:
        print_error('No video files set.')
        sys.exit(1)
    if not options.dest_dir:
        print_error('No destination for analyzed data set.')
        sys.exit(1)
    results_dir = options.dest_dir
    write_overview_video = optn.write_video

    print_header(
        'CaImAn CNMF-E Analysis: {}'.format(os.path.dirname(options.videos[0]).replace('/mnt/sds-hd/', '', 1))
    )
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

    # get temporary scratch location
    sys_tmpdir = os.environ.get('TMPDIR', '/scratch')
    if not os.path.isdir(sys_tmpdir):
        sys_tmpdir = '/var/tmp'
        if not os.path.isdir(sys_tmpdir):
            tempfile.gettempdir()
    os.chdir(sys_tmpdir)

    # copy videos to scratch workspace.
    # since CaImAn insist on creating its mmap files next to the videos,
    # this is faster as the videos are on a network drive with the scratch space is not
    print_section('Copying data to scratch space')
    fnames = []
    for i, vfname in enumerate(options.videos):
        target_fname = os.path.join(sys_tmpdir, 'msvideo_{}{}'.format(i, os.path.splitext(vfname)[1]))
        print_task('Copy: {} (as {})'.format(vfname, os.path.basename(target_fname)))
        shutil.copyfile(vfname, target_fname)
        fnames.append(target_fname)

    # configure plotting defaults
    matplotlib.rcParams['figure.autolayout'] = True
    sns.set()
    sns.set_style('ticks')

    print_section('Cluster setup')
    try:
        cm.stop_server()  # stop cluster if it was running
    except:
        pass
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=32, single_thread=False)

    # clean up data any potential previous run
    for vid_fname in options.videos:
        for tmp_fname in glob(os.path.join(os.path.dirname(vid_fname), '*.mmap')):
            print_task('Removing: {}'.format(tmp_fname))
            os.remove(tmp_fname)

    # Parameter setup
    fr = 30  # movie frame rate
    decay_time = 0.4  # length of a typical transient in seconds

    # motion correction parameters
    pw_rigid = False  # flag for pw-rigid motion correction

    gSig_filt = (3, 3)  # size of filter, in general gSig (see below),
    #                      change this one if algorithm does not work
    max_shifts = (4, 4)  # maximum allowed rigid shift
    strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    border_nan = 'copy'

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan,
    }
    opts = params.CNMFParams(params_dict=mc_dict)

    #  MOTION CORRECTION
    #  The pw_rigid flag set above, determines where to use rigid or pw-rigid
    #  motion correction
    print_section('Motion correction')

    # do motion correction rigid
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(
            np.maximum(np.max(np.abs(mc.x_shifts_els)), np.max(np.abs(mc.y_shifts_els)))
        ).astype(np.int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        save_motioncorrection_shifts_plot(mc, os.path.join(results_dir, 'mc-shifts.svg'))

    bord_px = 0 if border_nan == 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='mcmap', order='C', border_to_0=bord_px)

    # load memory mappable file
    print_section('Loading mmap data from motion correction')
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    #  Parameters for source extraction and deconvolution (CNMF-E algorithm)
    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None for 1p data
    gSig = (3, 3)  # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)  # average diameter of a neuron, in general 4*gSig+1
    Ain = None  # possibility to seed with predetermined binary masks
    merge_thr = 0.7  # merging threshold, max correlation allowed
    rf = 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20  # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 1  # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1  # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0  # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0  # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = 0.8  # min peak value from correlation image
    min_pnr = 9.4  # min peak to noise ration from PNR image
    ssub_B = 2  # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    opts.change_params(
        params_dict={
            'dims': dims,
            'method_init': 'corr_pnr',  # use this for 1 photon
            'K': K,
            'gSig': gSig,
            'gSiz': gSiz,
            'merge_thr': merge_thr,
            'p': p,
            'tsub': tsub,
            'ssub': ssub,
            'rf': rf,
            'stride': stride_cnmf,
            'only_init': True,  # set it to True to run CNMF-E
            'nb': gnb,
            'nb_patch': nb_patch,
            'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
            'low_rank_background': low_rank_background,
            'update_background_components': True,  # sometimes setting to False improve the results
            'min_corr': min_corr,
            'min_pnr': min_pnr,
            'normalize_init': False,  # just leave as is
            'center_psf': True,  # leave as is for 1 photon
            'ssub_B': ssub_B,
            'ring_size_factor': ring_size_factor,
            'del_duplicates': True,  # whether to remove duplicates from initialization
            'border_pix': bord_px,
        }
    )  # number of pixels to not consider in the borders)
    write_parameters_file(results_dir, opts.to_dict(), write_overview_video)

    # compute some summary images (correlation and peak to noise)
    # change swap dim if output looks weird, it is a problem with tiffile
    print_section('Computing correlation and peak to noise summary')
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::6], gSig=gSig[0], swap_dim=False)
    # if images file is too long this computation will take unnecessarily
    # long time and consume a lot of memory. Consider changing images[::1] to
    # images[::5] or something similar to compute on a subset of the data

    # save result, we may want to plot this later
    np.savez_compressed(os.path.join(results_dir, 'images_correlation_pnr.npz'), cn_filter=cn_filter, pnr=pnr)

    # print parameters set above, modify them if necessary based on summary images
    print()
    print_info(
        'Min correlation of peak: {}'.format(min_corr)
    )  # min correlation of peak (from correlation image)
    print_info('Min peak to noise ratio: {}'.format(min_pnr))  # min peak to noise ratio
    print()

    # RUN CNMF ON PATCHES
    print_section('CNMF on patches')
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)

    # DISCARD LOW QUALITY COMPONENTS
    print_section('Evaluating components')
    min_SNR = 2.5  # adaptive way to set threshold on the transient size
    r_values_min = 0.85  # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR, 'rval_thr': r_values_min, 'use_cnn': True})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print()
    print_info('Number of total components: ', len(cnm.estimates.C))
    print_info('Number of accepted components: ', len(cnm.estimates.idx_components))
    print()

    print_section('Saving result')
    cnm.dims = dims

    # HDF5
    cnm.save(os.path.join(results_dir, 'cnm_result.hdf5'))

    # Temporal component overview plot
    save_inferred_temporal_overview_plots(
        cnm.estimates.C, cnm.estimates.idx_components, os.path.join(results_dir, 'inferred-temporal-overview')
    )

    # NWB version
    # CURRENTLY BROKEN:
    # line 1624, in save_NWB
    # if self.idx_components:
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    # cnm.estimates.save_NWB(os.path.join(results_dir, 'cnm_estimates_result-nwb.nwb'), session_start_time=datetime.now())

    # MOVIES
    if write_overview_video:
        print_section('Creating movie')
        # movie without background
        estimates_write_movie(
            cnm.estimates,
            images,
            include_bck=False,
            q_max=99.9,
            gain_res=4,
            bpx=bord_px,
            use_color=True,
            opencv_codec='H264',
            movie_name=os.path.join(results_dir, 'results_nobg.mkv'),
        )

    print_section('Cleanup')
    for vid_fname in fnames:
        for tmp_fname in glob(os.path.join(os.path.dirname(vid_fname), '*.mmap')):
            print_task('Removing {}'.format(tmp_fname))
            os.remove(tmp_fname)

    # STOP SERVER
    print_section('Done')
    print_task('Stopping server')
    cm.stop_server(dview=dview)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze arbitrary Miniscope videos using CaImAn CNMF-E')
    parser.add_argument(
        '-d', '--destdir', action='store', dest='dest_dir', help='Destination to store the analyzed data in.'
    )
    parser.add_argument(
        '--write-video',
        action='store_true',
        dest='write_video',
        help='Whether to render a video overview of the analyzed data.',
    )
    parser.add_argument('videos', action='store', nargs='+', help='The Miniscope video files to analyze.')

    optn = parser.parse_args(sys.argv[1:])
    run(optn)
