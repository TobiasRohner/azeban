#!/usr/bin/env python3

import sys
import os
import shutil
import subprocess
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


CONFIG_SINE = '''
{{
  "device": "cuda",
  "time": 0.1,
  "snapshots": [0.025, 0.05, 0.075],
  "output": "{output}",
  "grid": {{
    "N_phys": {N}
  }},
  "equation": {{
    "name": "Burgers",
    "visc": {{
      "type": "Smooth Cutoff",
      "eps": 0.05,
      "k0": 1
    }}
  }},
  "timestepper": {{
    "type": "SSP RK3",
    "C": 0.2
  }},
  "init": {{
    "name": "Sine 1D"
  }}
}}
'''

CONFIG_SHOCK = '''
{{
  "device": "cuda",
  "time": 1,
  "snapshots": [0.25, 0.5, 0.75],
  "output": "{output}",
  "grid": {{
    "N_phys": {N}
  }},
  "equation": {{
    "name": "Burgers",
    "visc": {{
      "type": "Smooth Cutoff",
      "eps": 0.05,
      "k0": 1
    }}
  }},
  "timestepper": {{
    "type": "SSP RK3",
    "C": 0.2
  }},
  "init": {{
    "name": "Shock",
    "x0": 0.25,
    "x1": 0.5
  }}
}}
'''

EXPERIMENTS = ['Sine 1D', 'Shock']



def plot_all_snapshots(args, result, name):
    folder = os.path.join(args.output, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with h5py.File(result, 'r') as f:
        N = f[list(f.keys())[0]].shape[1]
        for time in f.keys():
            save_to = os.path.join(folder, '%s_N%05d_T%s.svg' % (name, N, time))
            plt.close()
            plt.plot(np.linspace(0, 1, N, endpoint=False), f[time][0])
            plt.title('t = %s' % time)
            plt.savefig(save_to)



def plot_convergence(args, results, name):
    folder = os.path.join(args.output, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    N = []
    errL2 = []
    with h5py.File(results[-1], 'r') as f_ref:
        time = sorted(list(f_ref.keys()), key = lambda x : float(x))[-1]
        u_ref = f_ref[time][0]
        N_ref = u_ref.shape[0]
        for res in results[:-1]:
            with h5py.File(res, 'r') as f:
                u = f[time][0]
                N.append(u.shape[0])
                stride = N_ref // N[-1]
                errL2.append(np.linalg.norm(u - u_ref[::stride]) / N[-1])

    b, a = np.polyfit(np.log(N), np.log(errL2), 1)
    measured = np.exp(a + b * np.log(N))

    save_to = os.path.join(folder, '%s_convergence.svg' % name)
    plt.close()
    plt.loglog(N, errL2, '-o', label='L2 Error')
    plt.loglog(N, measured, '-', color='grey', label='$N^{%.2f}$' % b)
    plt.xlabel('N')
    plt.ylabel('L2 Error')
    plt.title('t = %s' % time)
    plt.legend()
    plt.savefig(save_to)



def generate_plots(args, config_template, name):
    tmp = os.path.join(args.tmp, name)
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    try:
        results = []
        for N in [2**i for i in range(6, round(np.log2(args.N)+1))]:
            result = os.path.join(tmp, 'result_%05d.h5' % N)
            results.append(result)
            config = config_template.format(output=result, N=N)
            config_fname = os.path.join(tmp, 'config.json')
            with open(config_fname, 'w') as f:
                f.write(config)

            cmd = args.executable + ' ' + config_fname
            print(cmd)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                print(str(line)[2:-3])
            retval = p.wait()
            assert retval == 0

            plot_all_snapshots(args, result, name)

        plot_convergence(args, results, name)

    finally:
        shutil.rmtree(tmp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate Plots for the Burgers Equation')
    parser.add_argument('--executable', type=str, required=True)
    parser.add_argument('--tmp', type=str, default='/tmp')
    parser.add_argument('--experiments', type=str, action='extend', nargs='+')
    parser.add_argument('-N', type=int, default=1024)
    parser.add_argument('-o', '--output', type=str, default=os.path.abspath(os.path.split(sys.argv[0])[0]))
    args = parser.parse_args(sys.argv[1:])
    if args.experiments is None:
        args.experiments = EXPERIMENTS


    if 'Sine 1D' in args.experiments:
        generate_plots(args, CONFIG_SINE, 'burgers_sine')
    if 'Shock' in args.experiments:
        generate_plots(args, CONFIG_SHOCK, 'burgers_shock')
