#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_midpoint_step(u, H, i0, i1, j0, j1):
    if i1-i0==1 or j1-j0==1:
        return
    N = u.shape[1]
    im = i0 + (i1 - i0) // 2
    jm = j0 + (j1 - j0) // 2
    ui0j0 = u[:,i0,j0]
    ui0j1 = u[:,i0,j1%N]
    ui1j0 = u[:,i1%N,j0]
    ui1j1 = u[:,i1%N,j1%N]
    sigma = np.sqrt(((i1-i0)/N)**(2*H) * (1 - 2**(2*H-2)))
    if i0 == 0:
        u[:,i0,jm] = 0.5 * (ui0j0 + ui0j1) + sigma * np.random.normal(0, 1, (u.shape[0],))
    if j0 == 0:
        u[:,im,j0] = 0.5 * (ui0j0 + ui1j0) + sigma * np.random.normal(0, 1, (u.shape[0],))
    if i1 < N:
        u[:,i1,jm] = 0.5 * (ui1j0 + ui1j1) + sigma * np.random.normal(0, 1, (u.shape[0],))
    if j1 < N:
        u[:,im,j1] = 0.5 * (ui0j1 + ui1j1) + sigma * np.random.normal(0, 1, (u.shape[0],))
    ui0jm = u[:,i0,jm]
    uimj0 = u[:,im,j0]
    ui1jm = u[:,i1%N,jm]
    uimj1 = u[:,im,j1%N]
    u[:,im,jm] = 0.25 * (ui0jm + uimj0 + ui1jm + uimj1) + sigma * np.random.normal(0, 1, (u.shape[0],))
    generate_step(u, H, i0, im, j0, jm)
    generate_step(u, H, i0, im, jm, j1)
    generate_step(u, H, im, i1, j0, jm)
    generate_step(u, H, im, i1, jm, j1)


def generate_midpoint(u, H):
    N = u.shape[1]
    u[:,0,0] = 0
    generate_midpoint_step(u, H, 0, N, 0, N)


def generate_fourier_sample(u, H):
    u[:,:] = 0
    N = u.shape[0]
    x = np.linspace(0, 1, N, endpoint=False)
    Y, X = np.meshgrid(x, x)
    for k1 in range(-N, N+1):
        cx = np.cos(2*np.pi*k1*X)
        sx = np.sin(2*np.pi*k1*X)
        for k2 in range(-N, N+1):
            if k1 == 0 and k2 == 0:
                continue
            cy = np.cos(2*np.pi*k2*Y)
            sy = np.sin(2*np.pi*k2*Y)
            cc = cx * cy
            cs = cx * sy
            sc = sx * cy
            ss = sx * sy
            alpha = np.random.uniform(-1, 1, (4,))
            fac = 1. / (k1*k1 + k2*k2)**((H+1) / 2)
            u += fac * (alpha[0]*cc + alpha[1]*cs + alpha[2]*sc + alpha[3]*ss)


def generate_fourier(u, H):
    for i in tqdm(range(u.shape[0])):
        generate_fourier_sample(u[i,:,:], H)


def generate_fourier_efficient_sample(u, H):
    N = u.shape[0]
    N_fourier = N // 2 + 1
    u_hat = np.empty((N, N_fourier), dtype=complex)
    u_hat[0,0] = 0
    for k1 in range(0, N_fourier):
        for k2 in range(0, N_fourier):
            if k1 == 0 and k2 == 0:
                continue
            alpha = np.random.uniform(-1, 1, (4,))
            cc = alpha[0]
            cs = alpha[1]
            sc = alpha[2]
            ss = alpha[3]
            fac = N*N / (4*np.pi**2 * (k1*k1 + k2*k2))**((H + 1) / 2)
            u_hat[k1,k2] = fac * complex(cc - ss, cs + sc)
            if k1 > 0:
                u_hat[N-k1, k2] = fac * complex(cc + ss, cs - sc)
    u[:,:] = np.fft.irfft2(u_hat)


def generate_fourier_efficient(u, H):
    for i in tqdm(range(u.shape[0])):
        generate_fourier_efficient_sample(u[i,:,:], H)




if __name__ == '__main__':
    N = int(sys.argv[1])
    H = float(sys.argv[2])

    u = np.empty((N, N, N))
    generate_fourier_efficient(u, H)

    m = np.mean(u, axis=0)
    v = np.var(u, axis=0)

    plt.imshow(u[0,:,:])
    plt.colorbar()
    plt.show()
    plt.imshow(m)
    plt.colorbar()
    plt.show()
    plt.imshow(v)
    plt.colorbar()
    plt.show()
