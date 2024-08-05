#
# Based on code by Weisong Zhao from "Sparse deconvolution improves the resolution of live-cell
# super-resolution fluorescence microscopy, Nature Biotechnology (2021)", Licensed under ODbLv1
#
# With modifications by Matthias Klumpp to work faster with Miniscope data

import gc
import os
import math
import shutil
import tempfile
import multiprocessing as mp

import dask
import zarr
import numpy as np
import psutil
import dask.array as da
from tqdm import tqdm
from joblib import Parallel, delayed

from .sparse_hessian_recon import sparse_hessian
from .background_estimation import background_estimation


def sparse_hessian_slice(
    cache_fname,
    slice_idx,
    slice_count,
    iteration_num=100,
    fidelity=150,
    sparsity=10,
    contiz=0.5,
    mu=1,
):
    print(
        'Computing slice {}/{}, RAM utilization: {}%'.format(
            slice_idx + 1, slice_count, psutil.virtual_memory().percent
        ),
        flush=True,
    )
    slice_data = zarr.load(cache_fname)

    img_ssim = sparse_hessian(
        slice_data, iteration_num, fidelity, sparsity, contiz, mu, print_iter_info=False
    )

    # we cache the array to free up memory that future parallel tasks may need, while our result is
    # still waiting in the queue.
    zarr.save_array(cache_fname, img_ssim, compressor=zarr.Blosc(cname='zstd'))
    print('Completed & cached slice {}/{}'.format(slice_idx + 1, slice_count), flush=True)
    return cache_fname


def cache_chunk(tmp_dir, slice_idx, slice_count, chunk):
    cache_fname = os.path.join(tmp_dir, '{}_{}.zarr'.format(slice_idx, slice_count))
    zarr.save_array(cache_fname, chunk, compressor=zarr.Blosc(cname='zstd'))
    print('Prepared chunk {}/{}'.format(slice_idx, slice_count), flush=True)
    return slice_idx, cache_fname


def deblur_sparse_hessian(
    img,
    n_iterations,
    *,
    background=2,
    fidelity=150,
    sparsity=10,
    tcontinuity=0.5,
    slice_size=100,
    max_workers=8,
    sys_tmpdir=None
):
    img_sparse = np.asarray(img, dtype='float32')
    scaler = np.max(img_sparse)
    img_sparse = img_sparse / scaler

    print('Running: Background estimation')
    if background == 1:
        backgrounds = background_estimation(img_sparse / 2.5)
        img_sparse = img_sparse - backgrounds
    elif background == 2:
        backgrounds = background_estimation(img_sparse / 2)
        img_sparse = img_sparse - backgrounds
    elif background == 3:
        medVal = np.mean(img_sparse) / 2.5
        img_sparse[img_sparse > medVal] = medVal
        backgrounds = background_estimation(img_sparse)
        img_sparse = img_sparse - backgrounds
    elif background == 4:
        medVal = np.mean(img_sparse) / 2
        img_sparse[img_sparse > medVal] = medVal
        backgrounds = background_estimation(img_sparse)
        img_sparse = img_sparse - backgrounds
    elif background == 5:
        medVal = np.mean(img_sparse)
        img_sparse[img_sparse > medVal] = medVal
        backgrounds = background_estimation(img_sparse)
        img_sparse = img_sparse - backgrounds

    img_sparse = img_sparse / (img_sparse.max())
    img_sparse[img_sparse < 0] = 0

    img_sparse = img_sparse / (img_sparse.max())

    if img_sparse.ndim == 3:
        with tempfile.TemporaryDirectory(prefix='ssim_', dir=sys_tmpdir) as tmp_dir:
            print('NOTE: Keeping decon intermediates in temporary storage at: {}'.format(tmp_dir), flush=True)

            if img_sparse.shape[0] > slice_size:
                chunks = np.array_split(img_sparse, math.floor(img_sparse.shape[0] / slice_size))
            else:
                chunks = [img_sparse]

            chunk_tasks = []
            slice_count = len(chunks)
            for slice_idx, chunk in enumerate(chunks):
                chunk_tasks.append(delayed(cache_chunk)(tmp_dir, slice_idx, slice_count, chunk))

            frozen_chunks = tqdm(
                Parallel(n_jobs=max_workers, backend='loky', return_as='generator')(chunk_tasks),
                total=len(chunk_tasks),
                desc='Caching chunks',
            )

            # cleanup
            del chunks
            del chunk_tasks
            gc.collect()

            decon_tasks = []
            for slice_idx, chunk_fname in frozen_chunks:
                decon_tasks.append(
                    delayed(sparse_hessian_slice)(
                        chunk_fname, slice_idx, slice_count, n_iterations, fidelity, sparsity, tcontinuity
                    )
                )

            decon_slices = tqdm(
                Parallel(n_jobs=6, backend='loky', return_as='generator')(decon_tasks),
                total=len(decon_tasks),
                desc='Deconvolving slices',
            )

            print(
                'Combining result... (initial RAM % used:', psutil.virtual_memory().percent, ')', flush=True
            )
            np.concatenate([zarr.load(za_fname) for za_fname in decon_slices], axis=0, out=img_sparse)
    else:
        img_sparse = da.from_array(
            sparse_hessian(img, n_iterations, fidelity, sparsity, tcontinuity).astype(np.uint8)
        ).compute()

    print('Rescaling...', flush=True)
    img_sparse = img_sparse / (img_sparse.max())
    return scaler * img_sparse
