#!/usr/bin/env python

'''
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the Minian package,
for running without Jupyter.

@author: Matthias Klumpp
'''

# ignore future warnings in this pipeline (useful for development, but just noise here)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import shutil
import logging as log
import argparse
import tempfile
import itertools as itt
import subprocess
from glob import glob

import dask
import pywt
import numpy as np
import pyfftw
import xarray as xr
import seaborn as sns
import tomlkit
import matplotlib
import matplotlib.pyplot as plt
from minian.cnmf import (
    smooth_sig,
    unit_merge,
    compute_AtC,
    compute_trace,
    get_noise_fft,
    update_spatial,
    update_temporal,
    update_background,
)
from utils.messages import (
    print_info,
    print_task,
    print_warn,
    print_error,
    print_header,
    print_section,
)
from dask.distributed import Client, LocalCluster
from minian.utilities import (
    TaskAnnotation,
    load_videos,
    open_minian,
    save_minian,
    get_optimal_chk,
)
from minian.preprocessing import denoise, remove_background
from minian.visualization import (
    CNMFViewer,
    VArrayViewer,
    write_video,
    generate_videos,
    visualize_seeds,
    visualize_motion,
    visualize_gmm_fit,
    visualize_preprocess,
    visualize_spatial_update,
    visualize_temporal_update,
)
from minian.initialization import (
    initA,
    initC,
    ks_refine,
    gmm_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
    intensity_refine,
)
from minian.motion_correction import apply_transform, estimate_motion

# Set up logging level and format
log.basicConfig(
    format='%(asctime)s %(levelname).1s: %(message)s', level=log.INFO, datefmt='%Y-%m-%d %H:%M:%S'
)


def find_edl_collection_root_dir(start_dir):
    '''Find EDL collection root directory from any subdir'''
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
            md = tomlkit.load(f)
            res['collection_id'] = md.get('collection_id')  # collection UUID
            res['time_created'] = md.get('time_created')  # time when experiment was recorded
    else:
        log.error('No manifest file was found in (suspected) EDL directory {}'.format(edl_root))
    af_fname = os.path.join(edl_root, 'attributes.toml')
    if os.path.isfile(af_fname):
        with open(af_fname, 'r') as f:
            attrs = tomlkit.load(f)
            res['subject_id'] = attrs.get('subject_id')  # animal ID
    return res


def write_parameters_file(results_dir, params, overview_video):
    '''
    Write information attributes file containing parameters used
    to process this file.
    '''
    import yaml

    log.info('Writing parameter information file.')
    info = {}
    attr_fname = os.path.join(results_dir, 'attributes.toml')
    mpara_fname = os.path.join(results_dir, 'minian-parameters.yaml')
    # if os.path.isfile(attr_fname):
    #    with open(attr_fname, 'r') as f:
    #        info = toml.load(f)

    info['minian'] = {}
    info['minian']['have_overview_video'] = overview_video
    info['minian']['parameters_file'] = os.path.basename(mpara_fname)

    with open(mpara_fname, 'w') as f:
        yaml.dump(params, f, indent=2)
    with open(attr_fname, 'w') as f:
        tomlkit.dump(info, f)


def write_array_as_video(
    arr: xr.DataArray,
    fname: str,
    norm=True,
    vcodec='ffv1',
    options={'g': '1', 'level': '3'},
    verbose=True,
) -> str:
    import functools as fct

    import ffmpeg
    from minian.utilities import custom_arr_optimize

    if not fname.endswith('.mkv'):
        fname = fname + '.mkv'
    if norm:
        arr_opt = fct.partial(custom_arr_optimize, rename_dict={"rechunk": "merge_restricted"})
        with dask.config.set(array_optimize=arr_opt):
            arr = arr.astype(np.float32)
            arr_max = arr.max().compute().values
            arr_min = arr.min().compute().values
        den = arr_max - arr_min
        arr -= arr_min
        arr /= den
        arr *= 255
    arr = arr.clip(0, 255).astype(np.uint8)
    w, h = arr.sizes["width"], arr.sizes["height"]
    verbosity_args = ["-hide_banner", "-nostats"]
    if verbose:
        verbosity_args = []
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="gray", s="{}x{}".format(w, h))
        .filter("pad", int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2))
        .output(fname, pix_fmt="yuv420p", vcodec=vcodec, r=30, **options)
        .overwrite_output()
        .global_args(*verbosity_args)
        .run_async(pipe_stdin=True)
    )
    for blk in arr.data.blocks:
        process.stdin.write(np.array(blk).tobytes())
    process.stdin.close()
    process.wait()
    return fname


def deconvolve_ssim(varr, framerate, n_iterations=100, max_workers=8, **kwargs):
    import gc

    from deconvolve_ssim.deblur import deblur_sparse_hessian

    gc.collect()
    log.info('Running SSIM deconvolution with a max worker count of %s', max_workers)
    _, varr_decon = deblur_sparse_hessian(
        varr, n_iterations, max_workers=max_workers, slice_size=10 * framerate, **kwargs
    )
    return xr.DataArray(
        varr_decon,
        dims=["frame", "height", "width"],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2]),
        ),
    )


def format_filesize(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def get_tree_size(path):
    '''
    Return total size of files in path and subdirs. If
    is_dir() or stat() fails, print an error message to stderr
    and assume zero size (for example, file has been deleted).
    '''
    total = 0
    for entry in os.scandir(path):
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError as error:
            print('Error calling is_dir():', error, file=sys.stderr)
            continue
        if is_dir:
            total += get_tree_size(entry.path)
        else:
            try:
                total += entry.stat(follow_symlinks=False).st_size
            except OSError as error:
                print('Error calling stat():', error, file=sys.stderr)
    return total


def run(options):
    if not options.videos_dir:
        print_error('No video files set.')
        sys.exit(1)
    if not options.dest_dir:
        print_error('No destination for analyzed data set.')
        sys.exit(1)
    results_dir = options.dest_dir
    write_overview_video = optn.write_video

    print_header(
        'Minian Analysis: {}'.format(os.path.dirname(options.videos_dir).replace('/mnt/sds-hd/', '', 1))
    )
    os.makedirs(results_dir, exist_ok=True)

    #
    # Set parameters and prepare run
    #
    print_section('Configuring')

    edl_root_dir = find_edl_collection_root_dir(options.videos_dir)
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
            sys_tmpdir = tempfile.gettempdir()
    os.chdir(sys_tmpdir)

    # configure plotting defaults
    matplotlib.rcParams['figure.autolayout'] = True
    sns.set()
    sns.set_style('ticks')

    # configure Minian defaults
    intpath = os.path.join(sys_tmpdir, "minian_intermediate")
    subset = dict(frame=slice(0, None))
    subset_mc = None
    n_workers = int(os.getenv("MINIAN_NWORKERS", 8))
    param_save_minian = {
        "dpath": os.path.join(results_dir, 'data'),
        "meta_dict": dict(
            session=-2,  # edl_metadata.get('collection_name', 'unknown-session'),
            animal=-4,  # edl_metadata.get('subject_id', 'unknown-animal')
        ),
        "overwrite": False,
    }

    # Pre-processing Parameters
    param_load_videos = {
        "pattern": r"^msslice.*\.mkv$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    }
    param_denoise = {"method": "median", "ksize": 7}
    param_background_removal = {"method": "tophat", "wnd": 15}

    # Motion Correction Parameters
    subset_mc = None
    param_estimate_motion = {"dim": "frame"}

    # Initialization Parameters
    param_seeds_init = {
        "wnd_size": 1000,
        "method": "rolling",
        "stp_size": 400,
        "max_wnd": 16,
        "diff_thres": 3.2,
    }

    param_pnr_refine = {"noise_freq": 0.06, "thres": 'auto'}  # could also be thres=1
    param_ks_refine = {"sig": 0.05}
    param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06}
    param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06}
    param_init_merge = {"thres_corr": 0.8}

    # CNMF Parameters
    param_get_noise = {"noise_range": (0.06, 0.5)}
    param_first_spatial = {
        "dl_wnd": 10,
        "sparse_penal": 0.05,
        "size_thres": (28, None),
    }
    param_first_temporal = {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.2,
    }
    param_first_merge = {"thres_corr": 0.8}
    param_second_spatial = {
        "dl_wnd": 10,
        "sparse_penal": 0.01,
        "size_thres": (25, None),
    }
    param_second_temporal = {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.4,
    }

    # SparseSIM Parameters
    framerate_fps = 30
    param_ssim = dict(background=1, fidelity=80, sparsity=0.8, tcontinuity=0.4, n_iterations=100)

    param_load_videos_nopattern = param_load_videos.copy()
    param_load_videos_nopattern.pop('pattern')
    write_parameters_file(
        results_dir,
        dict(
            framerate_fps=framerate_fps,
            load_videos=param_load_videos_nopattern,
            denoise=param_denoise,
            background_removal=param_background_removal,
            subset_mc=subset_mc,
            estimate_motion=param_estimate_motion,
            seeds_init=param_seeds_init,
            pnr_refine=param_pnr_refine,
            ks_refine=param_ks_refine,
            seeds_merge=param_seeds_merge,
            initialize=param_initialize,
            init_merge=param_init_merge,
            get_noise=param_get_noise,
            first_spatial=param_first_spatial,
            first_temporal=param_first_temporal,
            first_merge=param_first_merge,
            second_spatial=param_second_spatial,
            second_temporal=param_second_temporal,
            ssim=param_ssim,
        ),
        write_overview_video,
    )

    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    pyfftw.config.NUM_THREADS = 16

    #
    # Set up Dask cluster
    #
    print_section('Cluster Setup')
    cluster_memory_limit = "98GB"
    dask.config.set({'interface': 'lo'})
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit=cluster_memory_limit,
        resources={"MEM": 1},
        threads_per_worker=1,
        dashboard_address=":8787",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    log.info('Local cluster running (memlimit {}, worker count: {})'.format(cluster_memory_limit, n_workers))

    print_section('Loading Video Data')
    varr = load_videos(options.videos_dir, **param_load_videos)

    # save the raw data
    log.info('Saving intermediate raw video data')
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    log.info('Cached intermediate data space usage: {}'.format(format_filesize(get_tree_size(intpath))))

    # remove any interlacing artifacts (due to signal processing issues when e.g. the rotary joint turned or a cable was loose)
    log.info('Cleaning up any potential buffer corruption artifacts')
    _, (lh, _, _) = pywt.dwt2(varr, 'haar')
    lh_absmax = np.amax(np.abs(lh), axis=(1, 2))
    artifact_th = np.mean(lh_absmax) * 2.1
    artifact_mask = lh_absmax > artifact_th
    if artifact_mask.any():
        varr[artifact_mask] = np.zeros(varr.shape[1:], dtype=varr.dtype)
        log.info('Artifacts found and replaced with null frames')
    else:
        log.info('No artifacts found, taking data as-is.')

    #print_section('Modified SparseSIM Deconvolution')
    #dc_n_workers = n_workers // 2
    #if dc_n_workers < 2:
    #    dc_n_workers = 2
    #varr = deconvolve_ssim(varr, framerate=framerate_fps, max_workers=dc_n_workers, **param_ssim)
    #log.info('Done.')

    # save the data
    log.info('Saving intermediate video data')
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.astype(np.uint8).chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    varr_ref = varr.sel(subset)
    varr_ref_bg = save_minian(varr_ref.rename("varr_ref_bg"), dpath=intpath, overwrite=True)
    log.info('Cached intermediate data space usage: {}'.format(format_filesize(get_tree_size(intpath))))

    print_section('Glow Removal')
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    log.info('Done.')

    print_section('Denoise')
    varr_ref = denoise(varr_ref, **param_denoise)
    log.info('Done.')

    print_section('Intermediate Scratch Save')
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)
    log.info('Cached intermediate data space usage: {}'.format(format_filesize(get_tree_size(intpath))))
    log.info('Done.')

    print_section('Background Removal')
    varr_ref = remove_background(varr_ref, **param_background_removal)
    log.info('Done.')

    print_section('Motion Correction')
    log.info('Estimate motion')
    motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)
    log.info('Save motion data')
    motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian)
    log.info('Apply transform')
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_bg = apply_transform(varr_ref_bg, motion, fill=0)
    log.info('Store temporary result')
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )
    log.info('Done.')

    print_section('Saving deconvolved & motion corrected data')
    video_basename = 'decon-mc.mkv'
    if 'collection_id' in edl_metadata:
        video_basename = '{}_{}'.format(edl_metadata['collection_id'][0:6], video_basename)
    write_array_as_video(Y_bg, os.path.join(results_dir, video_basename))
    del Y_bg
    del varr_ref_bg
    log.info('Cached intermediate data space usage: {}'.format(format_filesize(get_tree_size(intpath))))

    print_section('CNMF Initialization')
    log.info('Compute max projection')
    max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian).compute()
    log.info('Generating over-complete set of seeds')
    seeds = seeds_init(Y_fm_chk, **param_seeds_init)
    log.info('Peak-noise-ratio refine')
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)
    log.info('KS refine')
    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)
    log.info('Merge seeds')
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)
    log.info('Initialize spatial matrix')
    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    log.info('Initialize temporal matrix')
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})
    log.info('Merge units')
    A, C = unit_merge(A_init, C_init, **param_init_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    log.info('Initialize background terms')
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    print_section('CNMF')
    log.info('Estimate spatial noise')
    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    log.info('First spatial update')
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_first_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    log.info('Store result')
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    log.info('First temporal update')
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_first_temporal)
    log.info('Store result')
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    log.info('Merge units')
    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    log.info('Second spatial update')
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_second_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    log.info('Store result')
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    log.info('Second temporal update')
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_second_temporal)

    log.info('Store result')
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    if write_overview_video:
        print_section('Visualization')
        video_basename = 'result-overview.mkv'
        if 'collection_id' in edl_metadata:
            video_basename = '{}_{}'.format(edl_metadata['collection_id'][0:6], video_basename)
        log.info('Writing overview video')
        generate_videos(
            varr.sel(subset),
            Y_fm_chk,
            A=A,
            C=C_chk,
            vpath=sys_tmpdir,
            vname=video_basename,
            vcodec='libsvtav1',
            options={'preset': '4', 'g': '240'},
            verbose=False,
        )
        log.info('Moving visualization result')
        shutil.move(os.path.join(sys_tmpdir, video_basename), results_dir)
        log.info('Done.')

    print_section('Save Final Result')
    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)
    log.info('Done.')

    print_section('Close Cluster')
    client.close()
    cluster.close()
    log.info('Done.')


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
    parser.add_argument(
        'videos_dir', action='store', nargs='?', help='Directory with the Miniscope video files to analyze.'
    )

    optn = parser.parse_args(sys.argv[1:])
    run(optn)
