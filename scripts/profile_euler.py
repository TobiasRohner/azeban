#!/usr/bin/env python3

import sys
import os
import shutil
import subprocess
import argparse
import numpy as np
import json


CONFIG = '''
{{
  "device": "{device}",
  "dimension": {dimension},
  "time": 1,
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
    "delta": 0.05,
    "dimension": 0
  }}
}}
'''


def run_simulation(args, dimension, device, N):
    config = CONFIG.format(device=device, dimension=dimension, N=N)
    config_fname = os.path.join(args.tmp, 'config.json')
    with open(config_fname, 'w') as f:
        f.write(config)
    cmd = 'cd ' + args.tmp + ' && ' + os.path.abspath(args.executable) + ' config.json'
    print('{}d Euler, device={}, N={}:'.format(dimension, device, N), cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''.join([line.decode('UTF-8') for line in p.stdout.readlines()])
    retval = p.wait()
    assert retval == 0
    with open(os.path.join(args.tmp, 'profiling.json'), 'r') as f:
        profiling_info = json.load(f)
    return output, profiling_info


def profile_all_2d(args):
    outputs = []
    profiling_infos = []
    for device in ['cpu', 'cuda']:
        for N in [16, 32, 64, 128, 256] + ([512, 1024] if device=='cuda' else []):
            o, p = run_simulation(args, 2, device, N)
            outputs.append('--- 2d Euler, device={}, N={}---\n{}'.format(device, N, o))
            profiling_infos.append({'dimension':2, 'device':device, 'N':N, 'profiling_info':p})
    return outputs, profiling_infos

def profile_all_3d(args):
    outputs = []
    profiling_infos = []
    for device in ['cpu', 'cuda']:
        for N in [16, 32, 64] + ([128] if device=='cuda' else []):
            o, p = run_simulation(args, 3, device, N)
            outputs.append('--- 3d Euler, device={}, N={}---\n{}'.format(device, N, o))
            profiling_infos.append({'dimension':3, 'device':device, 'N':N, 'profiling_info':p})
    return outputs, profiling_infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Profile Euler Spectral Solver')
    parser.add_argument('--executable', type=str, required=True)
    parser.add_argument('--tmp', type=str, default='/tmp')
    parser.add_argument('-o', '--output', type=str, default=os.path.abspath(os.path.split(sys.argv[0])[0]))
    args = parser.parse_args(sys.argv[1:])

    o2d, p2d = profile_all_2d(args)
    o3d, p3d = profile_all_3d(args)

    with open(os.path.join(args.output, 'profiling_summary.txt'), 'w') as f:
        f.write('\n\n\n'.join(o2d + o3d))
    with open(os.path.join(args.output, 'profiling_info.json'), 'w') as f:
        json.dump({'data' : p2d+p3d}, f, indent=2)
