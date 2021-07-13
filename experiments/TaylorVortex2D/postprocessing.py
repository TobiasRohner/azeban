#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
import pickle
import sys


def analytic_sol(N, t):
    x = np.linspace(-8, 8, N, False)
    y = np.linspace(-8, 8, N, False)
    xm, ym = np.meshgrid(x, y)
    u = -ym * np.exp(0.5 * (1 - (xm - 4*t)**2 - ym**2)) + 8
    v = (xm - 4*t) * np.exp(0.5 * (1 - (xm - 4*t)**2 - ym**2))
    return u/16, v/16


def read_sol(N):
    with nc.Dataset('taylor_vortex_N{}_T1.0.nc'.format(N)) as f:
        u = f['sample_0_time_1.000000_u'][:]
        v = f['sample_0_time_1.000000_v'][:]
    return u, v


def pad_fourier(u_hat, N_pad):
    N = u_hat.shape[0]
    Nf = N // 2 + 1
    u_pad_hat = np.zeros((N_pad, N_pad), dtype=u_hat.dtype)
    u_pad_hat[:Nf,:Nf] = u_hat[:Nf,:Nf]
    u_pad_hat[:Nf,-Nf:] = u_hat[:Nf,-Nf:]
    u_pad_hat[-Nf:,:Nf] = u_hat[-Nf:,:Nf]
    u_pad_hat[-Nf:,-Nf:] = u_hat[-Nf:,-Nf:]
    return u_pad_hat


def compute_err(N, N_ref):
    u, v = read_sol(N)
    u_ref, v_ref = analytic_sol(N_ref, 1)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    u_pad_hat = pad_fourier(u_hat, N_ref)
    v_pad_hat = pad_fourier(v_hat, N_ref)
    u_pad = np.fft.ifft2(u_pad_hat)
    v_pad = np.fft.ifft2(v_pad_hat)
    err2_u = np.sum(np.abs(u_ref - u_pad)**2)
    err2_v = np.sum(np.abs(v_ref - v_pad)**2)
    err = np.sqrt(err2_u + err2_v)
    return err


if __name__ == '__main__':
    Ns = [16, 32, 64, 128]
    N_ref = 1024
    errs = [compute_err(N, N_ref) for N in Ns]
    print('\n'.join(['{}: {}'.format(N, err) for N, err in zip(Ns, errs)]))
    with open(sys.argv[1], 'wb') as f:
        pickle.dump(errs, f)
