#!/usr/bin/env python3

import fbm
import netCDF4 as nc
import sys
import os
from tqdm import tqdm
import numpy as np




if __name__ == '__main__':
    N = int(sys.argv[1])
    H = float(sys.argv[2])
    output = sys.argv[3]

    if not os.path.exists(output):
        os.makedirs(output)

    u = np.empty((N, N))
    v = np.empty((N, N))
    for sample in tqdm(range(N)):
        fbm.generate_fourier_sample(u, H)
        fbm.generate_fourier_sample(v, H)
        fname = os.path.join(output, 'sample_{}_time_0.000000.nc'.format(sample))
        with nc.Dataset(fname, 'w', format='NETCDF4') as f:
            f.createDimension('N', N)
            var_u = f.createVariable('u', float, ('N', 'N'))
            var_v = f.createVariable('v', float, ('N', 'N'))
            var_u[:] = u
            var_v[:] = v
