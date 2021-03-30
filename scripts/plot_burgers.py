#!/usr/bin/env python3


import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DATA_FILE = sys.argv[1]


with h5py.File(DATA_FILE, 'r') as f:
    keys = list(sorted(f.keys(), key = lambda n : float(n)))
    N = f[keys[0]].size

    fig, ax = plt.subplots()
    xdata = np.linspace(0, 1, N, endpoint=False)
    ydata = f[keys[0]]
    ln, = plt.plot([], [])
    ln1, = plt.plot([], [])

    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.2, 1.2)
        return ln, ln1

    def update(frame):
        plt.title(frame)
        ydata = f[frame]
        ln.set_data(xdata, ydata)
        ln1.set_data(xdata, np.abs(np.fft.fft(ydata))/N)
        return ln, ln1

    anim = FuncAnimation(fig, update, frames=keys, init_func=init, blit=True, interval=1000./1)
    plt.show()
