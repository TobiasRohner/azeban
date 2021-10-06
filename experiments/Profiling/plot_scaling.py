#!/usr/bin/env python3

import sys
import os
import argparse
import json
import matplotlib.pyplot as plt



def read_kernel_timings(experiment, Ns, kernel, kernel_map_cpu={}, kernel_map_gpu={}):
    times_cpu = []
    times_gpu = []
    for N in Ns:
        with open(os.path.join(experiment.format(N)+'_cpu', 'profiling_rank0.json'), 'r') as f:
            data_cpu = json.load(f)
        with open(os.path.join(experiment.format(N)+'_gpu', 'profiling_rank0.json'), 'r') as f:
            data_gpu = json.load(f)
        if kernel is not None:
            for stage in data_cpu['stages']:
                if stage['name'] == kernel_map_cpu.get(kernel, kernel):
                    times_cpu.append(stage['elapsed'] / stage['num_calls'])
                    break
            for stage in data_gpu['stages']:
                if stage['name'] == kernel_map_gpu.get(kernel, kernel):
                    times_gpu.append(stage['elapsed'] / stage['num_calls'])
                    break
        else:
            times_cpu.append(data_cpu['elapsed'])
            times_gpu.append(data_gpu['elapsed'])
    return times_cpu, times_gpu


def plot_timings(experiment, Ns, kernel, fname_out, optimal_cpu, optimal_gpu, kernel_map_cpu={}, kernel_map_gpu={}):
    plt.clf()
    data_cpu, data_gpu = read_kernel_timings(experiment, Ns, kernel, kernel_map_cpu, kernel_map_gpu)
    plt.loglog(Ns, [d/1e6 for d in data_cpu], '-o', color='C0', label='CPU')
    if optimal_cpu is not None:
        plt.loglog(Ns, optimal_cpu, '--', color='C0', label='Theoretical CPU')
    plt.loglog(Ns, [d/1e6 for d in data_gpu], '-o', color='C1', label='GPU')
    if optimal_gpu is not None:
        plt.loglog(Ns, optimal_gpu, '--', color='C1', label='Theoretical GPU')
    plt.xlabel('N')
    plt.ylabel('t [ms]')
    plt.grid(which='both')
    plt.legend()
    plt.savefig(fname_out, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot scaling of a single kernel')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('-N', nargs='+', type=int, required=True)
    parser.add_argument('--file_out', type=str, required=True)
    parser.add_argument('--kernel', type=str, required=False)
    parser.add_argument('--optimal_cpu', nargs='+', type=float, required=False)
    parser.add_argument('--optimal_gpu', nargs='+', type=float, required=False)
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
    plot_timings(args.experiment, args.N, args.kernel, args.file_out, args.optimal_cpu, args.optimal_gpu, kernel_map_cpu, kernel_map_gpu)
