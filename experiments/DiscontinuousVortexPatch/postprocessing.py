#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
import pickle
import sys


def analytic_sol(N):
    x = np.linspace(-0.5, 0.5, N, False)
    y = np.linspace(-0.5, 0.5, N, False)
    ym, xm = np.meshgrid(x, y)
    r = np.sqrt(xm**2 + ym**2)
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if r[i,j] < 0.25:
                u[i,j] = ym[i,j]
                v[i,j] = xm[i,j]
    return u, v


def read_sol(N, method):
    with nc.Dataset('discontinuous_vortex_patch_N{}_{}_T1.0/sample_0_time_1.000000.nc'.format(N, method)) as f:
        u = f['u'][:]
        v = f['v'][:]
    return u, v


def pad_fourier(u_hat, N_pad):
    N = u_hat.shape[0]
    Nf = N // 2 + 1
    u_pad_hat = np.zeros((N_pad, N_pad), dtype=u_hat.dtype)
    u_pad_hat[:Nf,:Nf] = u_hat[:Nf,:Nf]
    u_pad_hat[:Nf,-Nf:] = u_hat[:Nf,-Nf:]
    u_pad_hat[-Nf:,:Nf] = u_hat[-Nf:,:Nf]
    u_pad_hat[-Nf:,-Nf:] = u_hat[-Nf:,-Nf:]
    return (N_pad / N)**2 * u_pad_hat


def compute_diff(N, N_ref, method):
    u, v = read_sol(N, method)
    u_ref, v_ref = analytic_sol(N_ref)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    u_pad_hat = pad_fourier(u_hat, N_ref)
    v_pad_hat = pad_fourier(v_hat, N_ref)
    u_pad = np.real(np.fft.ifft2(u_pad_hat))
    v_pad = np.real(np.fft.ifft2(v_pad_hat))
    return u_ref-u_pad, v_ref-v_pad


def compute_err(N, N_ref, method):
    u_diff, v_diff = compute_diff(N, N_ref, method)
    err2_u = np.sum(np.abs(u_diff)**2)
    err2_v = np.sum(np.abs(v_diff)**2)
    err = np.sqrt(err2_u + err2_v) / N_ref**2
    return err


if __name__ == '__main__':
    Ns = [16, 32, 64, 128]
    N_ref = 1024
    to_dump = {}
    for method in sys.argv[1:-1]:
        errs = [compute_err(N, N_ref, method) for N in Ns]
        to_dump[method] = list(zip(Ns, errs))
        print('\n'.join([method]+['{}: {}'.format(N, err) for N, err in zip(Ns, errs)]))
    with open(sys.argv[-1], 'wb') as f:
        pickle.dump(to_dump, f)
