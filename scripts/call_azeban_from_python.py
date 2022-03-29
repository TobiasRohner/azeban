#!/usr/bin/env python3

import netCDF4 as nc
import os
import json
import numpy as np



def solve_NS(executable, u, visc, t_end, tmpdir='/tmp', device='cuda'):
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
                "snapshots": [0, t_end],
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
                  "C": 0.5
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


def solve_Euler(executable, u, t_end, tmpdir='/tmp', device='cuda'):
    N = u.shape[1]
    return solve_NS(executable, u, 0.05 / N, t_end, tmpdir, device)



if __name__== '__main__':
    import sys
    import fbm
    import matplotlib.pyplot as plt

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

    N = 256
    t_end = 1
    u = np.empty((2, N, N))
    fbm.generate_fourier_efficient_sample(u[0,:], 0.5)
    fbm.generate_fourier_efficient_sample(u[1,:], 0.5)
    u_new = solve_Euler(sys.argv[1], u, t_end, tmpdir=sys.argv[2])

    curl_u = curl(u[0], u[1])
    curl_u_new = curl(u_new[0], u_new[1])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(curl_u)
    ax2.imshow(curl_u_new)
    ax1.set_title("curl(u(t=0))")
    ax2.set_title("curl(u(t={}))".format(t_end))
    plt.show()
