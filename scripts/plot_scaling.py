#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
import json


def load_time(folder, name):
    time_per_call = 0
    for info in filter(lambda f: f.startswith('profiling_rank'), os.listdir(folder)):
        data = None
        with open(os.path.join(folder, info), 'r') as f:
            data = json.load(f)
        for stage in data['stages']:
            if stage['name'] == name:
                tpc = stage['elapsed'] / stage['num_calls']
                if tpc > time_per_call:
                    time_per_call = tpc
    return time_per_call


def plot_strong(args, sims, name):
    n = []
    t = []
    for s in sorted(sims, key = lambda s: s[1][3]):
        n.append(s[1][3])
        t.append(load_time(os.path.join(args.tmp, s[0]), name))
    S = [t[0] / i for i in t]
    plt.close()
    plt.plot([n[0], n[-1]], [1, n[-1]/n[0]], '--', color='grey')
    plt.plot(n, S)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Speedup')
    plt.title('Strong Scaling of ' + name)
    filename = '_'.join([*sims[0][0].split('_')[:3], name]) + '.png'
    plt.savefig(os.path.join(args.tmp, filename))


def plot_weak(args, sims, name):
    n = []
    t = []
    for s in sorted(sims, key = lambda s: s[1][3]):
        n.append(s[1][3])
        t.append(load_time(os.path.join(args.tmp, s[0]), name) / 1000000)
    plt.close()
    plt.plot([n[0], n[-1]], [t[0], t[0]], '--', color='grey')
    plt.plot(n, t)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Time per Timestep [ms]')
    plt.title('Weak Scaling of ' + name)
    filename = '_'.join([*sims[0][0].split('_')[:3], name]) + '.png'
    plt.savefig(os.path.join(args.tmp, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot the strong/weak scaling data obtained with run_scaling_leonhard.py')
    parser.add_argument('--tmp', type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    prog = re.compile('^euler_(\d)d_((strong)|(weak))_N(\d+)_np(\d+)$')
    sim = []
    for d in os.listdir(args.tmp):
        m = prog.match(d)
        if m:
            dim = int(m.group(1))
            scaling = m.group(2)
            N = int(m.group(5))
            n = int(m.group(6))
            sim.append((d, (dim, scaling, N, n)))
    sim_2d_strong = list(filter(lambda s: s[1][0] == 2 and s[1][1] == 'strong', sim))
    sim_2d_weak   = list(filter(lambda s: s[1][0] == 2 and s[1][1] == 'weak'  , sim))
    sim_3d_strong = list(filter(lambda s: s[1][0] == 3 and s[1][1] == 'strong', sim))
    sim_3d_weak   = list(filter(lambda s: s[1][0] == 3 and s[1][1] == 'weak'  , sim))
    plot_strong(args, sim_2d_strong, 'SSP_RK3::integrate')
    plot_strong(args, sim_3d_strong, 'SSP_RK3::integrate')
    plot_weak(args, sim_2d_weak, 'SSP_RK3::integrate')
    plot_weak(args, sim_3d_weak, 'SSP_RK3::integrate')
