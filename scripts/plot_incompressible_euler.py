#!/usr/bin/env python3


import sys
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DATA_FILE = sys.argv[1]


def curl(x, y):
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[:,0] = x[:,1] - x[:,-1]
    dx[:,1:-1] = x[:,2:] - x[:,:-2]
    dx[:,-1] = x[:,0] - x[:,-2]
    dy[0,:] = y[1,:] - y[-1,:]
    dy[1:-1,:] = y[2:,:] - y[:-2,:]
    dy[-1,:] = y[0,:] - y[-2,:]
    return (dx - dy) * x.shape[0]


with nc.Dataset(DATA_FILE, 'r', format='NETCDF4') as f:
    names = [v for v in f.variables]
    print(names)
    info = [(int(name.split('_')[1]), float(name.split('_')[3]), name[:-2]) for name in names if name != 'time' and name.endswith('_u')]
    info = sorted(info, key = lambda t: (t[0], t[1]))
    print(info)

    fig, ax = plt.subplots()
    img = curl(f[info[0][2]+'_u'], f[info[0][2]+'_v'])
    ln = plt.imshow(img)

    def init():
        plt.clim(-1, 1)
        return ln,

    def update(frame):
        print(frame[2])
        u = f[frame[2]+'_u'][:]
        v = f[frame[2]+'_v'][:]
        img = curl(u, v)
        print(img)
        ln.set_array(img)
        plt.clim(np.min(img), np.max(img))
        return ln,

    anim = FuncAnimation(fig, update, frames=info, init_func=init, blit=True, interval=1000./5, save_count=len(info))
    anim.save('euler.mp4')
    plt.show()
