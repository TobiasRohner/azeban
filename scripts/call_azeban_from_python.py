#!/usr/bin/env python3

import subprocess
import netCDF4 as nc
import os
import json
import numpy as np



def solve_NS(executable, u, visc, t_end, tmpdir='sol', device='cuda'):
    # Setting up simulation configuration file
    dim = len(u.shape) - 1
    N = u.shape[1]
    eps = N * visc
    config = {
                "output": tmpdir,
                "device": device,
                "dimension" : dim,
                "num_samlpes": 1,
                "time": t_end,
                "snapshots": {"start":0, "stop":t_end, "n":100},#[t_end],
                "grid": {
                  "N_phys": N,
                  "N_phys_pad": ""
                },
                "equation": {
                  "name": "Euler",
                  "visc": {
                      "type": "Step",
                      "k0": 1,
                      "eps": eps
                  }
                },
                "timestepper": {
                  "type": "SSP RK3",
                  "C": min(0.5 / (eps / 0.05), 0.5)
                },
                "init": {
                  "name": "Init From File",
                  "experiment": tmpdir,
                  "time": "0"
                }
              }
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    with open(tmpdir + "/config.json", "w") as cnf:
        json.dump(config, cnf, indent=2)
    # Storing initial conditions to be read by azeban
    with nc.Dataset(tmpdir + "/sample_0_time_0.nc", "w") as f:
        f.createDimension("N", N)
        var_u = f.createVariable("u", float, ("N",)*dim)
        var_v = f.createVariable("v", float, ("N",)*dim)
        var_u[:] = u[0,:]
        var_v[:] = u[1,:]
    # Running the simulation
    os.system("./" + executable + " " + tmpdir + "/config.json")
    # Reading in the simulation output
    u_new = np.empty_like(u)
    with nc.Dataset(tmpdir + "/sample_0_time_1.nc", "r") as f:
        u_new[0,:] = f["u"][:]
        u_new[1,:] = f["v"][:]
    return u_new



if __name__== '__main__':
    import sys
    import fbm

    N = 256
    u = np.empty((2, N, N))
    fbm.generate_fourier_efficient_sample(u[0,:], 0.5)
    fbm.generate_fourier_efficient_sample(u[1,:], 0.5)
    u_new = solve_NS(sys.argv[1], u, 1e-4, 1, tmpdir=sys.argv[2])
