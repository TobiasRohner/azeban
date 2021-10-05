#!/usr/bin/env python3

import sys
import json
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Profile Euler Spectral Solver')
    parser.add_argument('-b', '--benchmark_file', type=str, required=True)
    parser.add_argument('-d', '--dimensions', type=int, action='extend', nargs='+')
    parser.add_argument('-c', '--device', type=str, action='extend', nargs='+')
    args = parser.parse_args(sys.argv[1:])
    if args.dimensions is None:
        args.dimensions = [2, 3]
    if args.device is None:
        args.device = ['cpu', 'cuda']

    with open(args.benchmark_file, 'r') as f:
        data = json.load(f);
    benchmarks = data['benchmarks']

    data = {d: {c: {'N':[], 't':[]} for c in args.device} for d in args.dimensions}
    for bm in benchmarks:
        name = bm['name']
        if not name.startswith('bm_'):
            continue
        d = int(name.split('<')[1][0])
        c = 'cpu' if name.split('/')[-1] == '0' else 'cuda'
        N = int(name.split('/')[-2])
        t = float(bm['real_time']) / 1000
        if d in args.dimensions and c in args.device:
            data[d][c]['N'].append(N)
            data[d][c]['t'].append(t)

    for d in args.dimensions:
        for c in args.device:
            plt.loglog(data[d][c]['N'], data[d][c]['t'], 'o', label='{}d {}'.format(d, c))
    plt.xlabel('N')
    plt.ylabel('t [us]')
    plt.legend()
    plt.savefig('fft_benchmark.png', dpi=300)
