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

SUBMISSION_SCRIPT = '''#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --account="s1069"
#SBATCH --time=00:10:00
#SBATCH --nodes={n}
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun {executable} {config}'''


def run_strong(dim, args):
    Ns = []
    if dim == 2:
        Ns = [1024]
    if dim == 3:
        Ns = [128]
    for N in Ns:
        for n in [2, 3, 4, 5, 6, 7, 8]:
            jobname = 'euler_' + str(dim) + 'd_strong_N' + str(N) + '_np' + str(n)
            tmpfolder = os.path.join(args.tmp, jobname)
            if not os.path.exists(tmpfolder):
                os.makedirs(tmpfolder)
            # Create custom config file
            config = os.path.join(tmpfolder, 'config.json')
            with open(config, 'w') as f:
                if dim == 2:
                    f.write(CONFIG_2D.format(time=str(102.4/N), N=str(N)))
                if dim == 3:
                    f.write(CONFIG_3D.format(time=str(1.28/N), N=str(N)))
            # Create custom job submission script
            job = os.path.join(tmpfolder, 'job.sh')
            with open(job, 'w') as f:
                f.write(SUBMISSION_SCRIPT.format(job_name=jobname, n=n, executable=args.executable, config=config))
            if not args.dry:
                retval = subprocess.call('cd ' + tmpfolder + ' && sbatch ' + job, shell=True)


def run_weak(dim, args):
    Ns = []
    if dim == 2:
        Ns = [512, 1024, 1536, 2048]
    if dim == 3:
        Ns = [128, 256, 384, 512]
    for N in Ns:
        n = 2 * (N // Ns[0])**dim
        jobname = 'euler_' + str(dim) + 'd_weak_N' + str(N) + '_np' + str(n)
        tmpfolder = os.path.join(args.tmp, jobname)
        if not os.path.exists(tmpfolder):
            os.makedirs(tmpfolder)
        # Create custom config file
        config = os.path.join(tmpfolder, 'config.json')
        with open(config, 'w') as f:
            if dim == 2:
                f.write(CONFIG_2D.format(time=str(10.24/N), N=str(N)))
            if dim == 3:
                f.write(CONFIG_3D.format(time=str(1.28/N), N=str(N)))
        # Create custom job submission script
        job = os.path.join(tmpfolder, 'job.sh')
        with open(job, 'w') as f:
            f.write(SUBMISSION_SCRIPT.format(job_name=jobname, n=n, executable=args.executable, config=config))
        if not args.dry:
            retval = subprocess.call('cd ' + tmpfolder + ' &&sbatch ' + job, shell=True)



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
