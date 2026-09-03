# Forcing

The optional `forcing` section of the `equation` object adds a source term to
the equations. If the section is omitted, no forcing is applied.

```json
"forcing": {
  "type": "White Noise High Freq",
  "b": 8000,
  "k_min": 10,
  "k_max": 16
}
```

All forcings act in Fourier space. The type is selected with the required key
`type`. The forcing is evaluated once per time step; stochastic forcings draw
fresh random numbers at every step (seeded by the top-level `seed`).

## `"No Forcing"`

No source term. This is the default when the `forcing` key is absent. It takes
no options.

## `"Sinusoidal"`

A deterministic, time-independent forcing concentrated on a single low mode.

| Key | Type | Description |
| --- | --- | --- |
| `amplitude` | double (required) | Amplitude of the forcing. |

In 2D the forcing acts only on the mode $(k_1, k_2) = (1, 1)$; in 3D on the
modes with $\lvert k_1 \rvert = \lvert k_2 \rvert = 1$ and $k_3 = 0$. The
resulting velocity components are divergence free. Not effective in 1D.

## `"White Noise"`

A divergence-free random forcing: at every time step an incompressible random
vector field is constructed from a fresh potential of Gaussian random numbers
and injected at low wavenumbers. The forcing scales like $1/\sqrt{\Delta t}$,
i.e. it is designed to be integrated with the `"Euler Maruyama"` time
integrator as a stochastic term of an SDE.

| Key | Type | Description |
| --- | --- | --- |
| `sigma` | double (required) | Standard deviation of the random potential; scales the overall strength of the forcing. |
| `N` | int (required) | Size of the random potential array (number of modes in each direction that receive random values). Modes outside this range are not forced. Choose `N` smaller than `N_phys` to force only large scales. |

Available in 2D and 3D (in 1D the implementation is a placeholder that
produces zero forcing). Works on CPU (std::mt19937) and GPU (cuRAND).

## `"White Noise High Freq"`

A divergence-free white-noise in time forcing that injects energy in a band of
*wavenumber shells* in Fourier space (typically near the dissipation range),
used to study forced turbulence with a controllable energy-injection scale.

| Key | Type | Description |
| --- | --- | --- |
| `b` | double (required) | Overall strength of the forcing. It is normalized internally so that the injected energy rate is controlled by `b` together with the `eps` of the spectral viscosity. |
| `k_min` | int (required) | Smallest wavenumber (per component) that is forced. Modes with both components below `k_min` are untouched. |
| `k_max` | int (required) | Modes with any component $\ge$ `k_max` are not forced. The forced band is therefore a square annulus in mode space between `k_min` and `k_max`. |
| `delta` | int | Stride between forced modes along the $k_1$ axis: only modes with $\lvert k_1 \rvert$ divisible by `delta` are forced (default `1`, i.e. all modes in the band). The amplitude of the remaining modes is rescaled to compensate. |
| `antisymmetric` | bool | If `true`, the forcing is antisymmetric under reflection $k_1 \to -k_1$ (the $k_1 = 0$ plane is not forced). This breaks statistical mirror symmetry of the driven flow (default `false`). |

Only implemented for 2D. Works on CPU and GPU. The strength of the noise is
additionally scaled with $\sqrt{\varepsilon / \Delta t}$, where $\varepsilon$
is the `eps` of the spectral viscosity, so it is designed for the
`"Euler Maruyama"` integrator and for simulations with a non-vanishing
spectral viscosity.

## `"Boussinesq"`

Boussinesq buoyancy coupling between the tracer and the velocity field: the
tracer density $\rho$ enters the momentum equation as a vertical body force.

| Key | Type | Description |
| --- | --- | --- |
| *(none)* | | The forcing has no parameters. |

This forcing requires a tracer: the `init` section must define a `tracer`
(see [initializer.md](initializer.md)), otherwise azeban aborts with an error
message. Not supported for 1D.
