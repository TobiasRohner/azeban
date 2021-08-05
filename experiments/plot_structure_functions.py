#!/usr/bin/env python3

import sys
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot(experiment_name, Ns, t, p, output):
    filename = experiment_name + '_structure_cube_{}.000000_0_postprocess.nc'
    for N in Ns:
        with nc.Dataset(filename.format(N, t, p), 'r') as f:
            h = f['h'][:]
            s = f['s'][:]
            max_idx_plot = sum(h <= 0.1)
            plt.loglog(h[1:max_idx_plot], s[1:max_idx_plot], label='N={}'.format(N))
            z = np.polyfit(np.log(h[1:max_idx_plot]), np.log(s[1:max_idx_plot]), 1)
            z = z[0].round(2)
            C = s[max_idx_plot] * h[max_idx_plot]**(-z)
            plt.loglog(h[1:max_idx_plot], [C * x**z for x in h[1:max_idx_plot]], '--', label='h^'+str(z))
    plt.legend()
    plt.grid(which='both')
    if output is not None:
        plt.savefig(output, dpi=300)
    else:
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot Structure Functions from Postprocess script')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('-N', nargs='+', type=int, required=True)
    parser.add_argument('-t', type=str, required=True)
    parser.add_argument('-p', type=int, required=True)
    args = parser.parse_args(sys.argv[1:])

    plot(args.experiment_name, args.N, args.t, args.p, args.output)
