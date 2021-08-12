#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from postprocessing import compute_diff


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
    Ns = [128, 256, 512, 1024]
    N_ref = 8192

    for method in sys.argv[1:-1]:
        for N in Ns:
            u_diff, v_diff = compute_diff(N, N_ref, method)
            curl_diff = curl(u_diff, v_diff)
            pos = np.linspace(0, 1, num=u_diff.shape[0]+1)
            X, Y = np.meshgrid(pos, pos)
            plt.clf()
            img_diff = plt.pcolormesh(X, Y, np.transpose(curl_diff))
            plt.colorbar(img_diff)
            plt.savefig(sys.argv[-1].format(method, N), dpi=300)
