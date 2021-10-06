#!/usr/bin/env python3

import numpy as np
import re
import matplotlib.pyplot as plt
import os


PATH = os.path.dirname(os.path.abspath(__file__))



def parse_data(fname):
    with open(fname, 'r') as f:
        data = f.read()
    data = data.split('\n')
    data[0] = float(data[0])
    data[1] = [float(x) for x in data[1].split()]
    data[2] = [float(x) for x in data[2].split()]
    pattern_complex = r'(\([+-]?\d+(\.\d+)?(e[+-]?\d+)?, [+-]?\d+(\.\d+)?(e[+-]?\d+)?\))'
    data[3] = [[float(x) for x in m[0][1:-1].split(', ')] for m in re.findall(pattern_complex, data[3])]
    data[3] = [data[3][i::len(data[1])] for i in range(len(data[1]))]
    return {'dt':data[0], 'k':data[1], 'nu':data[2], 'data':data[3]}


def plot_data(fname):
    data = parse_data(fname)
    plt.figure(figsize=(5, 5))
    for i in range(len(data['k'])):
        N = len(data['data'][i])
        col = 'C{}'.format(i)
        t = np.asarray([t*data['dt'] for t in range(1, N+1)])
        plt.semilogy(t, [d[0] for d in data['data'][i]], label='k = ({0}, {0})'.format(data['k'][i]), color=col)
        #plt.semilogy(t, np.exp(data['nu'][i] * t), label='exp({}*t)'.format(data['nu'][i]), color=col, linestyle='--')
    plt.xlabel('t')
    plt.ylabel('\\rho')
    plt.title(os.path.splitext(os.path.basename(fname))[0])
    plt.legend()
    plt.grid()
    plt.savefig(os.path.splitext(fname)[0] + '.png', dpi=300)



if __name__ == '__main__':
    for method in ['Step1D_cutoff_0', 'Step1D_cutoff_sqrt(N)', 'Step1D_cutoff_N', 'SmoothCutoff1D_cutoff_1', 'SmoothCutoff1D_cutoff_sqrt(N)', 'SmoothCutoff1D_cutoff_N', 'Quadratic']:
        for N in [16, 32, 64, 128, 256]:
            fname = 'heat_{}_N{}.txt'.format(method, N)
            plot_data(fname)
