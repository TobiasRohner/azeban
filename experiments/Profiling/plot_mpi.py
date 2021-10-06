#!/usr/bin/env python3

import sys
import argparse
import json
import os
import matplotlib.pyplot as plt



def read_kernel_timings(experiment, N, r, kernel):
    with open(os.path.join(experiment.format(N, r), 'profiling_rank0.json'), 'r') as f:
        data = json.load(f)
    if kernel is not None:
        for stage in data['stages']:
            if stage['name'] == kernel:
                return stage['elapsed'] / stage['num_calls']
    else:
        return data['elapsed']


def plot_vs_N(experiment, Ns, r, kernel, fname_out, optimal):
    plt.clf()
    data = [read_kernel_timings(experiment, N, r, kernel) for N in Ns]
    plt.loglog(Ns, [d/1e6 for d in data], '-o', color='C0', label='Measured')
    if optimal is not None:
        plt.loglog(Ns, optimal, '--', color='C0', label='Theoretical')
        plt.legend()
    plt.xlabel('N')
    plt.ylabel('t [ms]')
    plt.grid(which='both')
    plt.savefig(fname_out, dpi=300)


def plot_vs_r(experiment, N, rs, kernel, fname_out, optimal):
    plt.clf()
    data = [read_kernel_timings(experiment, N, r, kernel) for r in rs]
    plt.loglog(rs, [d/1e6 for d in data], '-o', color='C0', label='Measured')
    if optimal is not None:
        plt.loglog(rs, optimal, '--', color='C0', label='Theoretical')
        plt.legend()
    plt.xlabel('ranks')
    plt.ylabel('t [ms]')
    plt.grid(which='both')
    plt.savefig(fname_out, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot scaling of MPI implementation')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('-N', nargs='+', type=int, required=True)
    parser.add_argument('-r', nargs='+', type=int, required=True)
    parser.add_argument('--file_out', type=str, required=True)
    parser.add_argument('--kernel', type=str, required=False)
    parser.add_argument('--optimal', nargs='+', type=float, required=False)
    args = parser.parse_args(sys.argv[1:])

    if len(args.N) > 1 and len(args.r) == 1:
        plot_vs_N(args.experiment, args.N, args.r[0], args.kernel, args.file_out, args.optimal)
    if len(args.N) == 1 and len(args.r) > 1:
        plot_vs_r(args.experiment, args.N[0], args.r, args.kernel, args.file_out, args.optimal)
