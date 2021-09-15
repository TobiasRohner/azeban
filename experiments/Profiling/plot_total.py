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


def plot_timings(fname_cpu, fname_gpu, kernels, fname_out, kernel_map_cpu={}, kernel_map_gpu={}):
    plt.clf()
    kernels_cpu = [kernel_map_cpu.get(kernel, kernel) for kernel in kernels]
    kernels_gpu = [kernel_map_gpu.get(kernel, kernel) for kernel in kernels]
    data_cpu = read_kernel_timings(fname_cpu, kernels_cpu)
    data_gpu = read_kernel_timings(fname_gpu, kernels_gpu)
    X = kernels
    Y_cpu = [data_cpu[kernel]/1e6 for kernel in kernels_cpu]
    Y_gpu = [data_gpu[kernel]/1e6 for kernel in kernels_gpu]
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.bar(X, Y_cpu, width=-0.4, align='edge', label='CPU')
    ax.bar(X, Y_gpu, width=0.4, align='edge', label='GPU')
    plt.ylabel('t [ms]')
    ax.legend()
    fig.savefig(fname_out, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot kernel timings')
    parser.add_argument('--file_cpu', type=str, required=True)
    parser.add_argument('--file_gpu', type=str, required=True)
    parser.add_argument('--file_out', type=str, required=True)
    parser.add_argument('--kernels', nargs='+', type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    kernel_map_cpu = {'FFT::forward':'FFTWFFT::forward',
                      'FFT::backward':'FFTWFFT::backward',
                      'computeB':'IncompressibleEuler::computeB',
                      'computeDudt':'IncompressibleEuler::computeDudt',
                      'pad':'copy_to_padded 3d'}
    kernel_map_gpu = {'FFT::forward':'CUFFT::forward',
                      'FFT::backward':'CUFFT::backward',
                      'computeB':'IncompressibleEuler::computeB',
                      'computeDudt':'IncompressibleEuler::computeDudt',
                      'pad':'copy_to_padded 3d'}
    plot_timings(args.file_cpu, args.file_gpu, args.kernels, args.file_out, kernel_map_cpu, kernel_map_gpu)
