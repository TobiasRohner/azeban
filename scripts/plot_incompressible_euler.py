#!/usr/bin/env python3


import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DATA_FILE = sys.argv[1]


with h5py.File(DATA_FILE, 'r') as f:
    keys = list(sorted(f.keys(), key = lambda n : int(n)))
    dim, Nx, Ny = f[keys[0]].shape
    assert dim == 2

    fig, ax = plt.subplots()
    img = f[keys[0]][0]
    ln = plt.imshow(img)

    def init():
        return ln,

    def update(frame):
        print(frame)
        img = f[frame][0]
        ln.set_array(img)
        return ln,

    anim = FuncAnimation(fig, update, frames=keys, init_func=init, blit=True, interval=1000./1)
    plt.show()
