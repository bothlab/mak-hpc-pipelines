import gc

import numpy as np
import pyfftw.interfaces.numpy_fft as nfftw
from pebble import ThreadPool

from .operation import (
    operation_xx,
    operation_xy,
    operation_xz,
    operation_yy,
    operation_yz,
    operation_zz,
)
from .sparse_iteration import (
    iter_xx,
    iter_xy,
    iter_xz,
    iter_yy,
    iter_yz,
    iter_zz,
    iter_sparse,
)


def sparse_hessian(f, iteration_num=100, fidelity=150, sparsity=10, contiz=0.5, mu=1, print_iter_info=True):
    '''
    function g = SparseHessian_core(f,iteration_num,fidelity,sparsity,iteration,contiz,mu)
    -----------------------------------------------
    Source code for argmin_g { ||f-g ||_2^2 +||gxx||_1+||gxx||_1+||gyy||_1+lamdbaz*||gzz||_1+2*||gxy||_1
     +2*sqrt(lamdbaz)||gxz||_1+ 2*sqrt(lamdbaz)|||gyz||_1+2*sqrt(lamdbal1)|||g||_1}
     f : ndarray
       Input image (can be N dimensional).
     iteration_num:  int, optional
        the iteration of sparse hessian {default:100}
     fidelity : int, optional
       fidelity {default: 150}
     contiz  : int, optional
       continuity along z-axial {example:1}
     sparsity :  int, optional
        sparsity {example:15}
    ------------------------------------------------
    Output:
      g
    '''

    contiz = np.sqrt(contiz)
    f1 = f

    flage = 0
    # f = cp.divide(f,cp.max(f[:]))
    f_flag = f.ndim
    if f_flag == 2:
        contiz = 0
        flage = 1
        f = np.zeros((3, f.shape[0], f.shape[1]), dtype='float32')
        f = np.array(f)
        for i in range(0, 3):
            f[i, :, :] = f1

    elif f_flag > 2:
        if f1.shape[0] < 3:
            contiz = 0
            f = np.zeros((3, f.shape[1], f.shape[2]), dtype='float32')
            f[0 : f1.shape[0], :, :] = f1
            for i in range(f1.shape[0], 3):
                f[i, :, :] = f[1, :, :]
        else:
            f = f1
    imgsize = np.shape(f)

    ## calculate derivate
    xxfft = operation_xx(imgsize)
    yyfft = operation_yy(imgsize)
    zzfft = operation_zz(imgsize)
    xyfft = operation_xy(imgsize)
    xzfft = operation_xz(imgsize)
    yzfft = operation_yz(imgsize)

    operationfft = (
        xxfft + yyfft + (contiz**2) * zzfft + 2 * xyfft + 2 * (contiz) * xzfft + 2 * (contiz) * yzfft
    )
    normlize = (fidelity / mu) + (sparsity**2) + operationfft
    del xxfft, yyfft, zzfft, xyfft, xzfft, yzfft, operationfft

    #    np.clear_memo()
    ## initialize b
    bxx = np.zeros(imgsize, dtype='float32')
    byy = bxx
    bzz = bxx
    bxy = bxx
    bxz = bxx
    byz = bxx
    bl1 = bxx
    ## initialize g
    g_update = np.multiply(fidelity / mu, f)
    ## iteration
    with ThreadPool(max_workers=6) as pool:
        for iter in range(0, iteration_num):
            g_update = nfftw.fftn(g_update)

            if iter == 0:
                g = nfftw.ifftn(g_update / (fidelity / mu)).real

            else:
                g = nfftw.ifftn(np.divide(g_update, normlize)).real

            g_update = np.multiply((fidelity / mu), f)

            xx_future = pool.schedule(iter_xx, args=(g, bxx, 1, mu))
            yy_future = pool.schedule(iter_yy, args=(g, byy, 1, mu))
            zz_future = pool.schedule(iter_zz, args=(g, bzz, contiz**2, mu))
            xy_future = pool.schedule(iter_xy, args=(g, bxy, 2, mu))
            xz_future = pool.schedule(iter_xz, args=(g, bxz, 2 * contiz, mu))
            yz_future = pool.schedule(iter_yz, args=(g, byz, 2 * contiz, mu))
            itersparse_future = pool.schedule(iter_sparse, args=(g, bl1, sparsity, mu))

            Lxx, bxx = xx_future.result()
            Lyy, byy = yy_future.result()
            Lzz, bzz = zz_future.result()
            Lxy, bxy = xy_future.result()
            Lxz, bxz = xz_future.result()
            Lyz, byz = yz_future.result()

            Lsparse, bl1 = itersparse_future.result()
            g_update = g_update + Lxx + Lyy + Lzz + Lxy + Lxz + Lyz + Lsparse

            if print_iter_info:
                print('%d iterations done\r' % iter)

    g[g < 0] = 0

    del bxx, byy, bzz, bxy, byz, bl1, f, normlize, g_update
    gc.collect()

    return g[1, :, :] if flage else g
