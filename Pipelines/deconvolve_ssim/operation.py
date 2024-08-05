import numpy as np
import pyfftw.interfaces.numpy_fft as nfftw


def operation_xx(gsize):
    delta_xx = np.array([[[1, -2, 1]]], dtype='float32')
    delta_xx_fft = nfftw.fftn(delta_xx, gsize)
    xxfft = delta_xx_fft * np.conj(delta_xx_fft)
    return xxfft


def operation_xy(gsize):
    delta_xy = np.array([[[1, -1], [-1, 1]]], dtype='float32')
    delta_xy_fft = nfftw.fftn(delta_xy, gsize)
    xyfft = delta_xy_fft * np.conj(delta_xy_fft)
    return xyfft


def operation_xz(gsize):
    delta_xz = np.array([[[1, -1]], [[-1, 1]]], dtype='float32')
    delta_xz_fft = nfftw.fftn(delta_xz, gsize)
    xzfft = delta_xz_fft * np.conj(delta_xz_fft)
    return xzfft


def operation_yy(gsize):
    delta_yy = np.array([[[1], [-2], [1]]], dtype='float32')
    delta_yy_fft = nfftw.fftn(delta_yy, gsize)
    yyfft = delta_yy_fft * np.conj(delta_yy_fft)
    return yyfft


def operation_yz(gsize):
    delta_yz = np.array([[[1], [-1]], [[-1], [1]]], dtype='float32')
    delta_yz_fft = nfftw.fftn(delta_yz, gsize)
    yzfft = delta_yz_fft * np.conj(delta_yz_fft)
    return yzfft


def operation_zz(gsize):
    delta_zz = np.array([[[1]], [[-2]], [[1]]], dtype='float32')
    delta_zz_fft = nfftw.fftn(delta_zz, gsize)
    zzfft = delta_zz_fft * np.conj(delta_zz_fft)
    return zzfft
