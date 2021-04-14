#!/usr/bin/env python3


import sys
import h5py
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


with h5py.File(DATA_FILE, 'r') as f:
    keys = list(sorted(f.keys(), key = lambda n : float(n)))
    dim, Nx, Ny = f[keys[0]].shape

    fig, ax = plt.subplots()
    img = curl(f[keys[0]][0], f[keys[0]][1])
    ln = plt.imshow(img)

    def init():
        plt.clim(-1, 1)
        return ln,

    def update(frame):
        print(frame)
        img = curl(f[frame][0], f[frame][1])
        print(img)
        ln.set_array(img)
        plt.clim(np.min(img), np.max(img))
        return ln,

    anim = FuncAnimation(fig, update, frames=keys, init_func=init, blit=True, interval=1000./5, save_count=len(keys))
    anim.save('euler.mp4')
    plt.show()
