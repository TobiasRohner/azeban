#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import netCDF4 as nc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Post-Processing for output generated by the alsvinn structure standalone')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--nx', type=int, required=True)
    parser.add_argument('--ny', type=int, required=True)
    parser.add_argument('--nz', type=int, default=1)
    args = parser.parse_args(sys.argv[1:])

    with nc.Dataset(args.input, 'r', format='NETCDF4') as f:
        data = {}
        for v in f.variables:
            data[v] = {'h':[0], 's':[0]}
            s = f[v][:]
            for h in range(1, s.size):
                data[v]['h'].append((h) / args.nx)
                data[v]['s'].append(s[h])
                data[v]['s'][-1] += data[v]['s'][-2]
            for h in range(s.size):
                data[v]['s'][h] /= args.nx * args.ny * args.nz
        with nc.Dataset(args.output, 'w', format='NETCDF4') as o:
            for v, d in data.items():
                N = len(d['h'])
                o.createDimension('N_' + v, N)
                var_h = o.createVariable(v + '_h', float, ('N_'+v,))
                var = o.createVariable(v, float, ('N_'+v,))
                var_h[:] = d['h']
                var[:] = d['s']
