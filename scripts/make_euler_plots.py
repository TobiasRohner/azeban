#!/usr/bin/env python3

import sys
import os
import shutil
import subprocess
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


CONFIG_TAYLOR_VORTEX = '''
{{
  "device": "cuda",
  "dimension": 2,
  "time": 0.01,
  "snapshots": [],
  "output": "{output}",
  "grid": {{
    "N_phys": {N}
  }},
  "equation": {{
    "name": "Euler",
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
    "name": "Taylor Vortex"
  }}
}}
'''

CONFIG_DOUBLE_SHEAR_LAYER = '''
{{
  "device": "cuda",
  "dimension": 2,
  "time": 1,
  "snapshots": [0.5, 0.75, 0.9],
  "output": "{output}",
  "grid": {{
    "N_phys": {N}
  }},
  "equation": {{
    "name": "Euler",
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
    "name": "Double Shear Layer",
    "rho": 0.2,
    "delta": 0.05
  }}
}}
'''


CONFIG_DISCONTINUOUS_VORTEX_PATCH = '''
{{
  "device": "cuda",
  "dimension": 2,
  "time": 5,
  "snapshots": [2.5],
  "output": "{output}",
  "grid": {{
    "N_phys": {N}
  }},
  "equation": {{
    "name": "Euler",
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
    "name": "Discontinuous Vortex Patch"
  }}
}}
'''


EXPERIMENTS = ['TaylorVortex', 'DoubleShearLayer', 'DiscontinuousVortexPatch']



def curl(x, y):
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[:,0] = x[:,1] - x[:,-1]
    dx[:,1:-1] = x[:,2:] - x[:,:-2]
    dx[:,-1] = x[:,0] - x[:,-2]
    dy[0,:] = y[1,:] - y[-1,:]
    dy[1:-1,:] = y[2:,:] - y[:-2,:]
    dy[-1,:] = y[0,:] - y[-2,:]
    return (dx - dy) * x.shape[0]


def plot_all_snapshots(args, result, name):
    folder = os.path.join(args.output, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with h5py.File(result, 'r') as f:
        N = f[list(f.keys())[0]].shape[1]
        for time in f.keys():
            save_to = os.path.join(folder, '%s_N%05d_T%s.png' % (name, N, time))
            plt.close()
            img = curl(f[time][0], f[time][1])
            plt.imshow(img)
            plt.colorbar()
            plt.title('2D curl, t = %s' % time)
            plt.savefig(save_to)



def plot_convergence(args, results, name):
    folder = os.path.join(args.output, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    N = []
    errL2 = []
    with h5py.File(results[-1], 'r') as f_ref:
        time = sorted(list(f_ref.keys()), key = lambda x : float(x))[-1]
        u_ref = f_ref[time]
        N_ref = u_ref.shape[1]
        for res in results[:-1]:
            with h5py.File(res, 'r') as f:
                u = f[time]
                N.append(u.shape[1])
                stride = N_ref // N[-1]
                errL2.append(np.linalg.norm(u - u_ref[:,::stride,::stride]) / N[-1]**2)

    b, a = np.polyfit(np.log(N), np.log(errL2), 1)
    measured = np.exp(a + b * np.log(N))

    save_to = os.path.join(folder, '%s_convergence.png' % name)
    plt.close()
    plt.loglog(N, errL2, '-o', label='L2 Error')
    plt.loglog(N, measured, '-', color='grey', label='$N^{%.2f}$' % b)
    plt.xlabel('N')
    plt.ylabel('L2 Error')
    plt.title('t = %s' % time)
    plt.legend()
    plt.savefig(save_to)



def plot_convergence_taylor(args, results, reference, name):
    folder = os.path.join(args.output, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    N = []
    errL2 = []
    with h5py.File(reference, 'r') as f_ref:
        time = sorted(list(f_ref.keys()), key = lambda x : float(x))[-1]
        u_ref = np.array(f_ref[time]) / 16
        N_ref = u_ref.shape[1]
        for res in results:
            with h5py.File(res, 'r') as f:
                u = f[time]
                N.append(u.shape[1])
                ref_points = [int(round(i * N_ref / N[-1])) for i in range(N[-1])]
                u_diff = u - u_ref[:,ref_points,:][:,:,ref_points]
                errL2.append(np.linalg.norm(u_diff) / N[-1]**2)

                save_to = os.path.join(folder, '%s_N%05d_T%s_diff.png' % (name, N[-1], time))
                plt.close()
                img = np.linalg.norm(u_diff, axis=0)
                plt.imshow(img)
                plt.colorbar()
                plt.title('Pointwise Error, t = %s' % time)
                plt.savefig(save_to)

    b, a = np.polyfit(np.log(N), np.log(errL2), 1)
    measured = np.exp(a + b * np.log(N))

    save_to = os.path.join(folder, '%s_convergence.png' % name)
    plt.close()
    plt.loglog(N, errL2, '-o', label='L2 Error')
    plt.loglog(N, measured, '-', color='grey', label='$N^{%.2f}$' % b)
    plt.xlabel('N')
    plt.ylabel('L2 Error')
    plt.title('t = %s' % time)
    plt.legend()
    plt.savefig(save_to)



def run_simulations(args, config_template, name):
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

        if name == 'euler_taylor_vortex' and args.taylor_reference is not None:
            plot_convergence_taylor(args, results, args.taylor_reference, name)
        else:
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
    parser.add_argument('--taylor_reference', type=str)
    args = parser.parse_args(sys.argv[1:])
    if args.experiments is None:
        args.experiments = EXPERIMENTS


    if 'TaylorVortex' in args.experiments:
        results = run_simulations(args, CONFIG_TAYLOR_VORTEX, 'euler_taylor_vortex')
    if 'DoubleShearLayer' in args.experiments:
        results = run_simulations(args, CONFIG_DOUBLE_SHEAR_LAYER, 'euler_double_shear_layer')
    if 'DiscontinuousVortexPatch' in args.experiments:
        results = run_simulations(args, CONFIG_DISCONTINUOUS_VORTEX_PATCH, 'euler_discontinuous_vortex_patch')
