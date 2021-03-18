#!/usr/bin/env python3

import sys
import json
import matplotlib.pyplot as plt


BM_FILE_NAME = sys.argv[1]

with open(BM_FILE_NAME, 'r') as f:
    data = json.load(f);

benchmarks = data['benchmarks']

N = []
time = []
for bm in benchmarks:
    if not bm['name'].startswith('bm_'):
        continue
    N.append(int(bm['name'].split('/')[-1]))
    time.append(float(bm['real_time']) / 1000)

plt.plot(N, time, 'o')
plt.xlabel('N')
plt.ylabel('t [us]')
plt.show()
