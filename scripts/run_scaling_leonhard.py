#!/usr/bin/env python3

import sys
import os
import subprocess
import argparse
import numpy as np


CONFIG_2D = '''{{
  "device": "cuda",
  "dimension": 2,
  "time": {time},
  "snapshots": [],
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
}}'''

CONFIG_3D = '''{{
  "device": "cuda",
  "dimension": 3,
  "time": {time},
  "snapshots": [],
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
    "name": "Shear Tube",
    "rho": 0.2,
    "delta": 0.05
  }}
}}'''


def run_strong(dim, args):
    Ns = []
    if dim == 2:
        Ns = [1024]
    if dim == 3:
        Ns = [128]
    for N in Ns:
        for n in [2, 3, 4, 5, 6, 7, 8]:
            jobname = 'euler_' + str(dim) + 'd_strong_N' + str(N) + '_np' + str(n)
            # Create custom config file
            tmpfolder = os.path.join(args.tmp, jobname)
            config = os.path.join(tmpfolder, 'config.json')
            if not args.dry:
                if not os.path.exists(tmpfolder):
                    os.makedirs(tmpfolder)
                with open(config, 'w') as f:
                    if dim == 2:
                        f.write(CONFIG_2D.format(time=str(102.4/N), N=str(N)))
                    if dim == 3:
                        f.write(CONFIG_3D.format(time=str(1.28/N), N=str(N)))
            cmd = ['bsub']
            cmd += ['-n', str(n)]
            cmd += ['-W', '00:10']
            cmd += ['-J', jobname]
            cmd += ['-cwd', tmpfolder]
            cmd += ['-R', '"rusage[ngpus_excl_p=' + str(n) + ']"']
            cmd += ['-R', '"select[gpu_model0==GeForceGTX1080]"']
            cmd += ['-R', '"rusage[mem=' + str(32 * 1024 // n) + ']"']
            cmd += ['mpirun', args.executable, config]
            print(' '.join(cmd))
            if not args.dry:
                retval = subprocess.call(' '.join(cmd), shell=True)


def run_weak(dim, args):
    Ns = []
    if dim == 2:
        Ns = [512, 1024, 1536]
    if dim == 3:
        Ns = [128, 256, 384]
    for N in Ns:
        n = 2 * (N // Ns[0])**dim
        n_nodes = int(np.ceil(n / 8))
        n_per_node = int(np.ceil(n / n_nodes))
        n_tot = n_nodes * n_per_node
        jobname = 'euler_' + str(dim) + 'd_weak_N' + str(N) + '_np' + str(n)
        # Create custom config file
        tmpfolder = os.path.join(args.tmp, jobname)
        config = os.path.join(tmpfolder, 'config.json')
        if not args.dry:
            if not os.path.exists(tmpfolder):
                os.makedirs(tmpfolder)
            with open(config, 'w') as f:
                if dim == 2:
                    f.write(CONFIG_2D.format(time=str(10.24/N), N=str(N)))
                if dim == 3:
                    f.write(CONFIG_3D.format(time=str(0.128/N), N=str(N)))
        cmd = ['bsub']
        cmd += ['-n', str(n_tot)]
        cmd += ['-W', '00:10']
        cmd += ['-J', jobname]
        cmd += ['-cwd', tmpfolder]
        cmd += ['-R', '"rusage[ngpus_excl_p=' + str(n_per_node) + ']"']
        cmd += ['-R', '"select[gpu_model0=GeForceGTX1080]"']
        cmd += ['-R', '"rusage[mem=' + str(32 * 1024 // n_per_node) + ']"']
        cmd += ['-R', '"span[ptile=' + str(n_per_node) + ']"']
        cmd += ['mpirun', '-n', str(n), args.executable, config]
        print(' '.join(cmd))
        if not args.dry:
            retval = subprocess.call(' '.join(cmd), shell=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Perform all job submissions to generate a scaling plot on leonhard')
    parser.add_argument('--executable', type=str, required=True)
    parser.add_argument('--tmp', type=str, default='/tmp')
    parser.add_argument('--dim', type=int, nargs='+')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    if args.dim is None:
        args.dim = [2, 3]
    args.executable = os.path.realpath(args.executable)
    args.tmp = os.path.realpath(args.tmp)

    if 2 in args.dim:
        run_strong(2, args)
        run_weak(2, args)
    if 3 in args.dim:
        run_strong(3, args)
        run_weak(3, args)
