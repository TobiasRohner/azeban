# Equation

The `equation` section selects the PDE that is solved, together with the
spectral viscosity regularization and an optional forcing term.

```json
"equation": {
  "name": "Euler",
  "visc": {
    "type": "Smooth Cutoff",
    "s": 1.5,
    "eps": 0.01
  },
  "forcing": {
    "type": "White Noise High Freq",
    "b": 8000,
    "k_min": 10,
    "k_max": 16
  }
}
```

## Options

| Key | Type | Description |
| --- | --- | --- |
| `name` | string (required) | The equation to integrate. One of `"Euler"` or `"Euler Naive"`. Both are only available for 2D and 3D simulations. |
| `visc` | object (required) | Spectral viscosity specification, see below. Must contain the key `type`. |
| `forcing` | object (optional) | Forcing term specification, see [forcing.md](forcing.md). Must contain the key `type`. If omitted, no forcing is applied. |

## Equations

### `"Euler"`

The incompressible Euler equations, solved pseudospectrally in Fourier space.
The nonlinear term is evaluated on the padded grid (3/2 rule, see
[grid.md](grid.md)), and the spectral viscosity defined by the `visc` section
is applied at every time step. This is the main solver of azeban and the only
one that supports forcing terms and a passively advected tracer (see
[initializer.md](initializer.md) for the `tracer` option of the `init`
section).

### `"Euler Naive"`

A straightforward re-implementation of the same equations that serves mainly
as a reference for testing. It supports neither forcing nor a tracer. The MPI
version of this equation is only available in CUDA builds.

## Spectral viscosity

All equations are regularized by a *spectral viscosity*: in Fourier space, a
damping term

$$
\nu(k) = -\varepsilon_N \, k^{2s} \, Q(k)
$$

is applied to every mode, where $k = 2\pi \lvert \mathbf{m} \rvert$ is the
physical wavenumber of a mode $\mathbf{m} \in \mathbb{Z}^d$ and $Q(k)$ is a
cutoff profile. The `visc` section selects the cutoff profile and its
parameters.

| Key | Type | Description |
| --- | --- | --- |
| `type` | string (required) | One of `"Smooth Cutoff"`, `"Step"`, `"Quadratic"`. |
| `eps` | double (required) | The viscosity strength $\varepsilon$ measured at resolution $N = 1$, i.e. the value used by the solver is rescaled as $\varepsilon_N = \varepsilon / N_{\text{phys}}^{2s-1}$. This makes the effective viscosity vanish as the grid is refined (vanishing viscosity), so that azeban converges to the inviscid solution. |
| `s` | double | Exponent of the viscosity. Default: `1`. Must be $> 1/2$ for the rescaling above to vanish. |
| `theta` | double | Exponent controlling how the cutoff wavenumber grows with the resolution. Default: $(2s-1)/(2s)$. |
| `m0` | double | Cutoff wavenumber at resolution $N = 1$. Default: $2\pi$ (i.e. the first nontrivial mode). The cutoff used by the solver is $m_N = m_0 \, N_{\text{phys}}^{\theta}$. |

`theta` and `m0` are accepted by the `"Smooth Cutoff"` and `"Step"` types.
`"Quadratic"` only accepts `eps`.

### Cutoff profiles

- **`"Smooth Cutoff"`** —
  $Q(k) = 1 - \exp\!\big(-(\lvert k \rvert / m_N)^{18}\big)$.
  Modes below the cutoff $m_N$ are essentially untouched, modes above it are
  smoothly and heavily damped. This is the standard choice in azeban.

- **`"Step"`** —
  $Q(k) = 1$ if $\lvert k \rvert > m_N$, and $0$ otherwise.
  A sharp version of the smooth cutoff: all modes above the cutoff are damped
  at full strength, all modes below it are untouched.

- **`"Quadratic"`** —
  $\nu(k) = -\dfrac{\varepsilon}{N_{\text{phys}}} \max\!\big(0, (k/2\pi)^2 - N_{\text{phys}}\big)$.
  A plain quadratic hyperdiffusion that only acts on modes with mode number
  larger than $\sqrt{N_{\text{phys}}}$. Here `eps` is *not* rescaled with the
  resolution; the viscosity simply weakens as $1/N_{\text{phys}}$ because of
  the explicit factor in the formula.

## Example

A 3D run with the smooth cutoff viscosity used throughout the azeban
publications:

```json
"equation": {
  "name": "Euler",
  "visc": {
    "type": "Smooth Cutoff",
    "s": 1.5,
    "eps": 0.01
  }
}
```
