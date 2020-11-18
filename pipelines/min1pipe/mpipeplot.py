
import numpy as np
import h5py
import logging as log
from concurrent.futures import ThreadPoolExecutor
from numba import jit
from scipy.ndimage import gaussian_filter

import matplotlib
try:
    matplotlib.use('module://mplcairo.base')
    import matplotlib.pyplot as plt
    print('Using mplcairo as Matplotlib backend')
except ModuleNotFoundError:
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    print('Using agg as Matplotlib backend')
from matplotlib.animation import FFMpegWriter
import seaborn as sns
from utils.cmap_kindlmann import cmap_kindlmann_extended


__all__ = ['create_overview_video',
           'plot_temporal_overview',
           'plot_spatial_overview']


@jit(nopython=True, nogil=True)
def array_limits(a):
    return np.nanmin(a), np.nanmax(a)


@jit(nopython=True, nogil=True)
def array_scale_minmax_inplace(a):
    vmin, vmax = array_limits(a)
    a -= vmin
    a /= vmax - vmin


@jit(nopython=True, nogil=True)
def flat_normscale_inplace(a, center_median=False):
    astd = a.std()
    if center_median:
        a -= np.median(a)
    else:
        a -= np.mean(a)
    a /= astd
    array_scale_minmax_inplace(a)


def zscore_axis_inplace(a, axis=0, ddof=0):
    mns = a.mean(axis=axis, keepdims=True)
    sstd = a.std(axis=axis, ddof=ddof, keepdims=True)
    a -= mns
    a /= sstd


@jit(nopython=True, nogil=True)
def frame_array_from_sigs_rois(sigfn, roifn, width, height):
    return np.reshape(sigfn @ roifn, (-1, width, height))


def discrete_cmap_for_n(N, base_cmap=None):
    import random
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N * 2))
    cmap = base.from_list('{}_{}'.format(base.name, N), color_list, N)
    color_list = [np.array(cmap(i)) for i in range(0, N)]
    random.Random(42).shuffle(color_list)
    return color_list


def calculate_frames_res(fname, width, height):
    log.info('Loading processed data')
    f_dp = h5py.File(fname, 'r')
    sigs = np.asarray(f_dp['sigfn'])
    roifn = np.asarray(f_dp['roifn'])

    log.info('Calculating z-scores for result signal traces')
    zscore_axis_inplace(sigs)

    log.info('Generating results matrix from signal ROI and temporal z-scores')
    frames_res = np.reshape(sigs @ roifn, (-1, width, height))
    log.info('Scaling processed space/time z-score matrix.')
    array_scale_minmax_inplace(frames_res)
    return frames_res, sigs


def calculate_frames_reg(fname, brighten_factor=1):
    log.info('Loading motion-corrected frame data')
    f_reg = h5py.File(fname, 'r')
    log.info('Normalizing and scaling motion-corrected frames')
    frames_reg = np.asarray(f_reg['reg'])
    flat_normscale_inplace(frames_reg, center_median=True)
    # increase contrast
    frames_reg *= brighten_factor
    return frames_reg


def calculate_rawdata_range(fname):
    f_raw = h5py.File(fname, 'r')
    fraw = f_raw['frame_all']
    vmin, vmax = array_limits(np.asarray(fraw))
    log.info('Raw data value range is {} to {}'.format(vmin, vmax))
    return vmin, vmax


def create_overview_video(dpmat_fname, regmat_fname, rawmat_fname, movie_fname,
                          fps: int, metadata={}, mc_brighten_factor=1.2, codec='vp9'):
    ''' Write an overview video of the generated calcium traces, motion correction and
    temporal-spatial extracted cell activity. '''

    # configure plotting defaults
    matplotlib.rcParams['figure.autolayout'] = True
    plt.rc('axes.spines', top=False, right=False)
    sns.set()
    sns.set_style('white')

    f_dp = h5py.File(dpmat_fname, 'r')
    imax_mc = np.array(f_dp['imax'])
    pix_w = imax_mc.shape[0]
    pix_h = imax_mc.shape[1]
    log.info('Result image dimensions: {}x{}'.format(pix_w, pix_h))

    with ThreadPoolExecutor(max_workers=4) as e:
        frames_res_future = e.submit(calculate_frames_res, dpmat_fname, pix_w, pix_h)
        frames_reg_future = e.submit(calculate_frames_reg, regmat_fname, mc_brighten_factor)
        raw_range_future = e.submit(calculate_rawdata_range, rawmat_fname)

        raw_vmin, raw_vmax = raw_range_future.result()
        frames_reg = frames_reg_future.result()
        frames_res, sigs_zsc = frames_res_future.result()

    log.info('Parallel processing finished.')
    f_raw = h5py.File(rawmat_fname, 'r')
    frames_raw = f_raw['frame_all']

    trace_range = 15 * fps
    trace_hl_range = 8 * fps

    plt.close('all')
    plt.ioff()
    if not metadata:
        metadata = dict(title='Calcium trace analysis overview')

    sns.set_palette('deep')
    plt.style.use('dark_background')

    # prepare figure for animation
    log.info('Preparing video figure template')
    fig = plt.figure(figsize=(20, 10), dpi=60)
    gs = fig.add_gridspec(2, 3)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_reg = fig.add_subplot(gs[0, 1])
    ax_res = fig.add_subplot(gs[0, 2])
    ax_trace = fig.add_subplot(gs[1, :])

    # prepare axes
    ax_raw.axis('off')
    ax_reg.axis('off')
    ax_res.axis('off')

    ax_trace.get_yaxis().set_visible(False)
    ax_trace.get_xaxis().set_visible(False)
    sns.despine(ax=ax_trace, left=True, top=True, right=True, trim=True)

    # add title placeholder
    fig.suptitle('frame: ??????', fontsize=12, ha='right', x=0.98)

    # data placeholders
    spf_raw = ax_raw.imshow(np.zeros((pix_h, pix_w)),
                            cmap=cmap_kindlmann_extended,
                            vmin=raw_vmin, vmax=raw_vmax)
    ax_raw.set_title('Raw')

    spf_reg = ax_reg.imshow(np.zeros((pix_h, pix_w)),
                            cmap=cmap_kindlmann_extended,
                            vmin=0.0, vmax=1.0)
    ax_reg.set_title('After MC')

    spf_res = ax_res.imshow(np.zeros((pix_h, pix_w)),
                            cmap=cmap_kindlmann_extended,
                            vmin=0.0, vmax=1.0)
    ax_res.set_title('Processed')

    # create color mapping for our traces
    cmap_trace_base = discrete_cmap_for_n(len(sigs_zsc[0, :]), 'rainbow')
    cmap_trace_hl = list(map(lambda x: x * 0.90, cmap_trace_base))
    cmap_trace_bg = list(map(lambda x: x * 0.60, cmap_trace_hl))

    spf_traces = []
    sigs_zsc_plotadj = np.zeros(sigs_zsc.shape, dtype=sigs_zsc.dtype)
    plot_sigs_vismask = np.zeros(sigs_zsc.shape, dtype=bool)
    traces_ymin = 0
    traces_ymax = 0
    for i in range(0, len(sigs_zsc_plotadj[0, :])):
        plot_sigs_vismask[:, i] = np.ma.masked_inside(sigs_zsc[:, i], -1.5, 1.5)
        sigs_zsc_plotadj[:, i] = (sigs_zsc[:, i] * 1.5) + (i * 1.5)
        p, = ax_trace.plot(np.nan, np.nan, linewidth=1.5)
        p.set_color(cmap_trace_bg[i])
        spf_traces.append(p)
        if i == 0:
            traces_ymin = np.nanmin(sigs_zsc_plotadj[:, i])
        elif i == len(sigs_zsc_plotadj[0, :]) - 1:
            traces_ymax = np.nanmax(sigs_zsc_plotadj[:, i])

    # we intentionally do not set the actual limits here, as peaks that go way above the range
    # would otherwise make any other traces with smaller signals less visible. For a quick overview,
    # this display is good enough
    ax_trace.set_ylim(ymin=traces_ymin, ymax=traces_ymax)
    ax_trace.set_xlim(xmin=trace_range * - 1, xmax=(trace_range + 1))
    ax_trace_vline = ax_trace.axvline(x=0, linestyle=':', linewidth=1, zorder=100)
    log.info('Trace plot limits calculated: {} to {}'.format(traces_ymax, traces_ymin))

    render_proglog_interval = 60 * fps
    log.info('Writing video')
    frames_n = frames_res.shape[0]
    mwriter = FFMpegWriter(fps=fps, metadata=metadata, codec=codec)
    with mwriter.saving(fig, movie_fname, 60):
        for i in range(0, frames_n):
            fig.suptitle('frame: {:06d}'.format(i), fontsize=12, ha='right', x=0.98)

            # raw input frame
            spf_raw.set_data(np.transpose(frames_raw[i, :, :]))

            # motion-corrected data
            spf_reg.set_data(np.transpose(frames_reg[i, :, :]))

            # visulatization of resulting detected temporal-spatial units
            spf_res.set_data(frames_res[i, :, :].T)

            # plot trace overview
            trace_start = i - trace_range
            trace_end = trace_range + i
            ax_trace.set_xlim(xmin=trace_start, xmax=trace_end)
            if trace_start < 0:
                trace_start = 0
            if trace_end > len(sigs_zsc_plotadj[:, 0]):
                trace_end = len(sigs_zsc_plotadj[:, 0])
            trace_x = np.arange(trace_start, trace_end)
            ax_trace_vline.set_xdata(i)
            for j, spf_trace in enumerate(spf_traces):
                if np.count_nonzero(plot_sigs_vismask[:, j][i - trace_hl_range:i + trace_hl_range]) >= (trace_hl_range / 1.5):
                    spf_trace.set_color(cmap_trace_hl[j])
                else:
                    spf_trace.set_color(cmap_trace_bg[j])
                spf_trace.set_data(trace_x, sigs_zsc_plotadj[:, j][trace_start:trace_range + i])

            if i % render_proglog_interval == 0:
                log.info('Rendered {} of {} frames'.format(i, frames_n))

            mwriter.grab_frame()
    log.info('Video created successfully ({} frames @ {}fps)'.format(frames_n, fps))


def plot_temporal_overview(dpmat_fname, fig_fname,
                           subject_id=None, test_id=None, test_date=None):
    sns.set()
    sns.set_style('white')
    plt.style.use('seaborn-notebook')

    f_dp = h5py.File(dpmat_fname, 'r')
    sigs = np.asarray(f_dp['sigfn'])
    zscore_axis_inplace(sigs)
    sigs = sigs.T

    trace_spacing = np.amax(sigs) / 4
    if trace_spacing < 2:
        trace_spacing = 2

    _, T = sigs.shape
    sns.set_palette('muted')

    # accepted components overview
    fig = plt.figure(figsize=(24.0, 12.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.get_yaxis().set_visible(False)

    if subject_id:
        ax.set_title('{} ({} on {}): Temporal Components'.format(subject_id, test_id, test_date))
    else:
        ax.set_title('Temporal Components')

    for i, trace in enumerate(sigs):
        ax.plot(np.arange(T), trace + (i * trace_spacing))
    sns.despine(trim=True, left=True)
    fig.savefig(fig_fname)


def plot_spatial_overview(dpmat_fname, fig_fname, with_mc=True,
                          subject_id=None, test_id=None, test_date=None):
    sns.set()
    sns.set_style('white')
    plt.style.use('seaborn-notebook')

    f_dp = h5py.File(dpmat_fname, 'r')
    imax_raw = np.asarray(f_dp['imaxn'])
    imax_enh = np.asarray(f_dp['imaxy'])
    imax_mc = np.asarray(f_dp['imax'])
    roifn = np.asarray(f_dp['roifn'])

    fig = plt.figure(figsize=(24.0, 12.0))
    gs = fig.add_gridspec(2, 6)
    ax_raw = fig.add_subplot(gs[0, 0:2])
    ax_enh = fig.add_subplot(gs[0, 2:4])
    ax_mc = fig.add_subplot(gs[0, 4:6])
    ax_ctr = fig.add_subplot(gs[1, 0:3])
    ax_shifts = fig.add_subplot(gs[1, 3:6])

    ax_raw.axis('off')
    ax_enh.axis('off')
    ax_mc.axis('off')
    ax_ctr.axis('off')

    if subject_id:
        fig.suptitle('{} ({} on {}): Spatial Components'.format(subject_id, test_id, test_date))
    else:
        fig.suptitle('Spatial Components')

    # raw video
    ax_raw.imshow(imax_raw.T, cmap=cmap_kindlmann_extended)
    ax_raw.set_title('Raw')

    # neural enhanced video
    ax_enh.imshow(imax_enh.T, cmap=cmap_kindlmann_extended)
    ax_enh.set_title('Enhanced')

    # neural enhanced video
    ax_mc.imshow(imax_mc.T, cmap=cmap_kindlmann_extended)
    ax_mc.set_title('Enhanced + Motion Corrected')

    # contours
    sigfn = np.asarray(f_dp['sigfn'])
    seedsfn = np.asarray(f_dp['seedsfn'])
    pix_w, pix_h = imax_mc.shape

    x, y = np.unravel_index(seedsfn.T.astype(np.int), (pix_w, pix_h))
    contours_n = len(x)
    ax_ctr.imshow(imax_mc.T,
                  cmap=cmap_kindlmann_extended,
                  vmin=0.0, vmax=0.8)

    contour_threshold = 0.8
    for i in range(contours_n):
        a = np.reshape(roifn[i, :], (pix_w, pix_h)) * np.amax(sigfn[:, i])
        a = gaussian_filter(a, 3, mode='wrap')

        lvl = np.amax(a) * contour_threshold
        ax_ctr.contour(np.flipud(np.rot90(a)),
                       levels=[lvl],
                       colors='lightsteelblue',
                       linewidths=1.2)

    for i in range(contours_n):
        ax_ctr.text(x[i],
                    y[i],
                    str(i),
                    fontsize=10,
                    color='w')
    ax_ctr.set_title('Contours')

    # motion correction shifts
    if with_mc:
        ax_shifts.plot(f_dp['raw_score'], label='Raw Score')
        ax_shifts.plot(f_dp['corr_score'], label='Correlation Score')
        ax_shifts.set_title('MC Scores')
        ax_shifts.legend()
    else:
        ax_shifts.set_title('MC Skipped or not drawn')
        ax_shifts.axis('off')
    sns.despine(trim=True)
    fig.savefig(fig_fname)
