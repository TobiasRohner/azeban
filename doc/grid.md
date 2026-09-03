# Grid

The `grid` section of the configuration file controls the discretization. The
domain is always the unit torus $[0,1)^d$, discretized uniformly with $N$
grid points per direction.

```json
"grid": {
  "N_phys": 512,
  "N_phys_pad": ""
}
```

## Options

| Key | Type | Description |
| --- | --- | --- |
| `N_phys` | int | Number of grid points per spatial direction in physical space. Provide **either** `N_phys` **or** `N_fourier`, but not both. |
| `N_fourier` | int | Number of Fourier modes per direction kept in spectral space. The relation `N_phys = 2 * (N_fourier - 1)` holds, i.e. the real-to-complex FFT of an `N_phys`-point grid has `N_fourier` complex coefficients per direction. Provide **either** `N_phys` **or** `N_fourier`, but not both. |
| `N_phys_pad` | int *or* string | Number of grid points per direction of the *padded* physical grid used for dealiasing (the nonlinear term is evaluated on the padded grid following the 3/2 rule). If omitted, the default `3 * N_phys / 2` is used. Provide **either** `N_phys_pad` **or** `N_fourier_pad`, but not both. |
| `N_fourier_pad` | int *or* string | Number of Fourier modes of the padded grid (`N_phys_pad = 2 * (N_fourier_pad - 1)`). If omitted, the default `3 * N_phys / 2` padding is used. Provide **either** `N_phys_pad` **or** `N_fourier_pad`, but not both. |

## Automatic padding

FFT libraries perform best for certain "smooth" sizes (e.g. powers of two and
products of small primes). Instead of a number, `N_phys_pad` (or
`N_fourier_pad`) may be given as a *string*, e.g.

```json
"grid": {
  "N_phys": 1000,
  "N_phys_pad": ""
}
```

The empty string means: use the smallest fast FFT size that is at least
`3 * N_phys / 2`. azeban prints the chosen value to stderr, e.g.

```
Info: Minimal padding size given from "N_phys_pad" is 1500. Padded to 1536 for speed.
```

Any other string is passed to the FFT planning machinery as a hint for the set
of acceptable sizes. In practice, `""` is what you almost always want. The
content of the string is only interpreted as a switch between automatic
padding and an explicit integer size; a non-empty string has the same effect
as `""` in the current implementation.

## Notes

- The physical grid always spans `N_phys` points per direction; the padded
  grid is only used internally to evaluate the nonlinear term without
  aliasing errors. Output is always written on the unpadded `N_phys` grid.
- For MPI runs, the first dimension of the data is distributed across ranks
  (`N_phys` slab decomposition in physical space, `N_fourier` slab
  decomposition in spectral space). `N_phys` should therefore be divisible by
  the number of ranks in a group for an even load balance; a remainder is
  distributed over the first ranks.
