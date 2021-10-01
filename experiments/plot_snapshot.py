#!/usr/bin/env python3

import sys
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def curl(x, y):
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[:,0] = x[:,1] - x[:,-1]
    dx[:,1:-1] = x[:,2:] - x[:,:-2]
    dx[:,-1] = x[:,0] - x[:,-2]
    dy[0,:] = y[1,:] - y[-1,:]
    dy[1:-1,:] = y[2:,:] - y[:-2,:]
    dy[-1,:] = y[0,:] - y[-2,:]
    return 0.5 * (dx - dy) * x.shape[0]


def plot(u, v, fname):
    plt.clf()
    min_u = np.min(u)
    min_v = np.min(v)
    max_u = np.max(u)
    max_v = np.max(v)
    fig = plt.figure(figsize=(5.5, 6), constrained_layout=True)
    ax = fig.add_gridspec(3, 2)
    ax_curl = fig.add_subplot(ax[0:2,0:2])
    ax_u = fig.add_subplot(ax[2:3,0:1])
    ax_v = fig.add_subplot(ax[2:3,1:2])
    pos = np.linspace(0, 1, num=u.shape[0]+1)
    X, Y = np.meshgrid(pos, pos)
    img_curl = ax_curl.pcolormesh(X, Y, np.transpose(curl(u, v)))
    img_u = ax_u.pcolormesh(X, Y, np.transpose(u))
    img_v = ax_v.pcolormesh(X, Y, np.transpose(v))
    img_u.set_clim(min(min_u, min_v), max(max_u, max_v))
    img_v.set_clim(min(min_u, min_v), max(max_u, max_v))
    plt.colorbar(img_curl, ax=ax_curl, location='right')
    plt.colorbar(img_v, ax=ax_v, location='right')
    plt.savefig(fname, dpi=300)


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print('Usage: ' + sys.argv[0] + ' output snapshots...')
        exit(1)

    # Plot first snapshot
    with nc.Dataset(sys.argv[2], 'r') as f:
        u = f['u'][:]
        v = f['v'][:]
    plot(u, v, sys.argv[1]+'_snapshot.png')

    # Plot mean
    u_mean = np.zeros_like(u)
    v_mean = np.zeros_like(v)
    for fname in sys.argv[2:]:
        with nc.Dataset(fname, 'r') as f:
            u = f['u'][:]
            v = f['v'][:]
        u_mean += u
        v_mean += v
    u_mean /= len(sys.argv[2:])
    v_mean /= len(sys.argv[2:])
    plot(u_mean, v_mean, sys.argv[1]+'_mean.png')

    # Plot variance
    u_var = np.zeros_like(u_mean)
    v_var = np.zeros_like(v_mean)
    for fname in sys.argv[2:]:
        with nc.Dataset(fname, 'r') as f:
            u = f['u'][:]
            v = f['v'][:]
        u_var += (u - u_mean)**2
        v_var += (v - v_mean)**2
    u_var /= len(sys.argv[2:])
    v_var /= len(sys.argv[2:])
    plot(u_var, v_var, sys.argv[1]+'_variance.png')
