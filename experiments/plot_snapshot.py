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


if __name__ == '__main__':
    with nc.Dataset(sys.argv[1], 'r') as f:
        u = f['u'][:]
        v = f['v'][:]
        min_u = np.min(u)
        min_v = np.min(v)
        max_u = np.max(u)
        max_v = np.max(v)
        fig = plt.figure(figsize=(5.5, 6), constrained_layout=True)
        ax = fig.add_gridspec(3, 2)
        ax_curl = fig.add_subplot(ax[0:2,0:2])
        ax_u = fig.add_subplot(ax[2:3,0:1])
        ax_v = fig.add_subplot(ax[2:3,1:2])
        #ax_curl.set_title('curl')
        #ax_u.set_title('u')
        #ax_v.set_title('v')
        #img_curl = ax_curl.imshow(curl(u, v), origin='lower')
        #img_u = ax_u.imshow(u, origin='lower')
        #img_v = ax_v.imshow(v, origin='lower')
        pos = np.linspace(0, 1, num=u.shape[0]+1)
        X, Y = np.meshgrid(pos, pos)
        img_curl = ax_curl.pcolormesh(X, Y, curl(u, v))
        img_u = ax_u.pcolormesh(X, Y, u)
        img_v = ax_v.pcolormesh(X, Y, v)
        img_u.set_clim(min(min_u, min_v), max(max_u, max_v))
        img_v.set_clim(min(min_u, min_v), max(max_u, max_v))
        plt.colorbar(img_curl, ax=ax_curl, location='right')
        plt.colorbar(img_v, ax=ax_v, location='right')
        if (len(sys.argv) > 2):
            plt.savefig(sys.argv[2], dpi=300)
        else:
            plt.show()
