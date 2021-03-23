#!/usr/bin/env python3

import sys
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt




parser = argparse.ArgumentParser(description = 'Plot Profiling Results')
parser.add_argument('-p', '--profiling_file', type=str, required=True)
parser.add_argument('-d', '--dimension', type=int, required=True)
parser.add_argument('-c', '--device', type=str, required=True)
args = parser.parse_args(sys.argv[1:])

with open(args.profiling_file, 'r') as f:
    data = json.load(f)
data = data['data']
data = list(filter(lambda d: d['dimension'] == args.dimension and d['device'] == args.device, data))

N = sorted([d['N'] for d in data])
idxs_N = {N:i for i,N in enumerate(N)}
idxs_stages = {s['name']:i for i,s in enumerate(sorted(data[-1]['profiling_info']['stages'], key=lambda s: s['elapsed']))}
num_N = len(N)
num_stages = len(idxs_stages)
time_percentage = np.zeros((num_stages, num_N))
for d in data:
    for stage in d['profiling_info']['stages']:
        time_percentage[idxs_stages[stage['name']], idxs_N[d['N']]] = stage['elapsed'] / d['profiling_info']['elapsed']

for name, i in idxs_stages.items():
    plt.plot(N, time_percentage[i,:], label=name)
plt.xlabel('N')
plt.ylabel('time %')
plt.legend()
plt.show()
