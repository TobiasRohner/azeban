#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
import pickle
import sys


def analytic_sol(N, t):
    x = np.linspace(0, 2*np.pi, N, False)
    y = np.linspace(0, 2*np.pi, N, False)
    ym, xm = np.meshgrid(x, y)
    u = np.cos(xm) * np.sin(ym)
    v = -np.sin(xm) * np.cos(ym)
    return u, v


def read_sol(N):
    with nc.Dataset('taylor_green_N{}_T0.1/sample_0_time_0.100000.nc'.format(N)) as f:
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


def compute_diff(N, N_ref):
    u, v = read_sol(N)
    u_ref, v_ref = analytic_sol(N_ref, 0.1)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    u_pad_hat = pad_fourier(u_hat, N_ref)
    v_pad_hat = pad_fourier(v_hat, N_ref)
    u_pad = np.real(np.fft.ifft2(u_pad_hat))
    v_pad = np.real(np.fft.ifft2(v_pad_hat))
    return u_ref-u_pad, v_ref-v_pad


def compute_err(N, N_ref):
    u_diff, v_diff = compute_diff(N, N_ref)
    err2_u = np.sum(np.abs(u_diff)**2)
    err2_v = np.sum(np.abs(v_diff)**2)
    err = np.sqrt(err2_u + err2_v) / N_ref**2
    return err


if __name__ == '__main__':
    Ns = [16, 32, 64, 128]
    N_ref = 1024
    to_dump = {}
    errs = [compute_err(N, N_ref) for N in Ns]
    to_dump['No visc'] = list(zip(Ns, errs))
    print('\n'.join(['{}: {}'.format(N, err) for N, err in zip(Ns, errs)]))
    with open(sys.argv[1], 'wb') as f:
        pickle.dump(to_dump, f)
