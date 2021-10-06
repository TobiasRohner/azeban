#!/usr/bin/env python3

import sys
import argparse
import json
import matplotlib.pyplot as plt


def read_kernel_timings(fname, kernels):
    with open(fname, 'r') as f:
        data = json.load(f)
    times = {}
    for stage in data['stages']:
        if stage['name'] in kernels:
            times[stage['name']] = stage['elapsed']
    return times


def plot_timings(fname, kernels, fname_out, kernel_map={}):
    plt.clf()
    kernels_mapped = [kernel_map.get(kernel, kernel) for kernel in kernels]
    data = read_kernel_timings(fname, kernels_mapped)
    X = kernels
    Y = [data[kernel]/1e6 for kernel in kernels_mapped]
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.bar(X, Y)
    plt.ylabel('t [ms]')
    ax.set_xticklabels(labels=kernels, rotation=45, horizontalalignment='right')
    plt.tight_layout()
    fig.savefig(fname_out, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot kernel timings')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--file_out', type=str, required=True)
    parser.add_argument('--kernels', nargs='+', type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    kernel_map = {'FFT::forward':'CUFFT_MPI::forward',
                  'FFT::backward':'CUFFT_MPI::backward',
                  'computeB':'IncompressibleEuler_MPI::computeB',
                  'computeDudt':'IncompressibleEuler_MPI::computeDudt',
                  'pad':'IncompressibleEuler_MPI::pad_u_hat',
                  'unpad':'IncompressibleEuler_MPI::unpad_B_hat'}
    plot_timings(args.file, args.kernels, args.file_out, kernel_map)
