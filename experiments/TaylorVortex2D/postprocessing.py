#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
import pickle
import sys


def analytic_sol(N, t):
    x = np.linspace(-8, 8, N, False)
    y = np.linspace(-8, 8, N, False)
    ym, xm = np.meshgrid(x, y)
    u = -ym * np.exp(0.5 * (1 - (xm - 8*t)**2 - ym**2)) + 8
    v = (xm - 8*t) * np.exp(0.5 * (1 - (xm - 8*t)**2 - ym**2))
    return u/16, v/16


def read_sol(N, method):
    with nc.Dataset('taylor_vortex_N{}_{}_T0.1/sample_0_time_0.100000.nc'.format(N, method)) as f:
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
    u_ref, v_ref = analytic_sol(N_ref, 0.1)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    u_pad_hat = pad_fourier(u_hat, N_ref)
    v_pad_hat = pad_fourier(v_hat, N_ref)
    u_pad = np.real(np.fft.ifft2(u_pad_hat))
    v_pad = np.real(np.fft.ifft2(v_pad_hat))
    return u_ref-u_pad, v_ref-v_pad


def compute_err(N, N_ref, method):
    u_diff, v_diff = compute_diff(N, N_ref, method)
    u_anal, v_anal = analytic_sol(N_ref, 0.1)
    err2_u = np.sum(np.abs(u_diff)**2)
    err2_v = np.sum(np.abs(v_diff)**2)
    norm2_u_anal = np.sum(np.abs(u_anal)**2)
    norm2_v_anal = np.sum(np.abs(v_anal)**2)
    err = np.sqrt(err2_u + err2_v) / N_ref**2
    norm = np.sqrt(norm2_u_anal + norm2_v_anal) / N_ref**2
    return err / norm


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
