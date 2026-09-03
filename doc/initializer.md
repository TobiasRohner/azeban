# Initializer

The `init` section defines the initial condition of the simulation.

```json
"init": {
  "name": "Taylor Green",
  "perturb": {
    "name": "Uniform",
    "min": -0.025,
    "max": 0.025
  }
}
```

The required key `name` selects the initializer. All other keys are
initializer-specific parameters. Unless stated otherwise, any parameter that
is documented as a "random variable" also accepts a plain number, which is
then treated as a deterministic value (see
[configuration.md](configuration.md)).

A passively advected tracer can be added with the optional key `tracer`. If
present, the simulation carries an additional scalar field $\rho$ that is
advected by the velocity field (and, with the `"Boussinesq"` forcing, feeds
back onto it). The tracer has its own initializer, see
[Tracer initializers](#tracer-initializers) below.

## Velocity initializers

### `"Sine 1D"`

A single sinusoidal velocity mode, $u(x) = \sin(2\pi x)$. Only available for
1D simulations. Takes no options.

### `"Shock"`

A square pulse. Only available for 1D.

| Key | Type | Description |
| --- | --- | --- |
| `x0` | random variable (required) | Left edge of the pulse. |
| `x1` | random variable (required) | Right edge of the pulse. |

The initial velocity is $u = 1$ on $[x_0, x_1)$ and $0$ elsewhere.

### `"Double Shear Layer"`

The classical smooth double shear layer (a tanh shear profile in $y$ with a
sinusoidal perturbation in $x$). Available for 2D and 3D.

| Key | Type | Description |
| --- | --- | --- |
| `rho` | random variable (required) | Thickness of the shear layers. `rho = 0` produces the discontinuous (piecewise constant) version of the profile. |
| `delta` | random variable (required) | Amplitude of the sinusoidal perturbation of the layers. |
| `dimension` | int | Only used for 3D simulations: the coordinate axis ($0 = x$, $1 = y$, $2 = z$) along which the 2D profile is constant. The velocity component along that axis is zero, i.e. the flow lives in a slab perpendicular to that axis. |

### `"Discontinuous Double Shear Layer"`

The discontinuous double shear layer with a randomly perturbed interface.

| Key | Type | Description |
| --- | --- | --- |
| `N` | int | Number of Fourier modes used to perturb the interface (default `1`). |
| `rho` | random variable (required) | Thickness of the shear layers. |
| `delta` | random variable (required) | Amplitude scale of the interface perturbation. |
| `perturb` | random variable | Random variable used to draw the mode amplitudes and phases (default: `Uniform(0, 1)`). |
| `dimension` | int | Only used for 3D simulations, same meaning as for `"Double Shear Layer"`. |

### `"Taylor Vortex"`

A single smooth vortex (a Gaussian-type Taylor vortex centered in the domain).
Available for 2D; for 3D the key `dimension` (int) must be given and has the
same meaning as for `"Double Shear Layer"`. Takes no further options.

### `"Discontinuous Vortex Patch"`

A circular patch of solid-body rotation (velocity pointing tangentially,
constant vorticity inside the disk of radius $1/4$ centered in the domain,
zero outside). Available for 2D; for 3D the key `dimension` (int) must be
given and has the same meaning as for `"Double Shear Layer"`. Takes no
options.

### `"Taylor Green"`

The classical Taylor-Green vortex, the standard benchmark initial condition of
azeban. Available for 2D and 3D.

| Key | Type | Description |
| --- | --- | --- |
| `perturb` | random variable | Amplitude of a random perturbation that is added on top of the Taylor-Green vortex. The perturbation is built from modes with wavenumber $2$ in every direction (default: `0`, i.e. the unperturbed vortex). Use e.g. `Uniform(-0.025, 0.025)` to break the exact symmetries of the benchmark. |

### `"Shear Tube"`

A smooth shear tube: a cylindrical shear layer (tanh profile in the radius)
running along the $x$-axis, with a sinusoidal perturbation along the tube.
Only available for 3D.

| Key | Type | Description |
| --- | --- | --- |
| `rho` | random variable (required) | Thickness of the tube. |
| `delta` | random variable (required) | Amplitude of the perturbation. |

### `"Discontinuous Shear Tube"`

The discontinuous shear tube with a randomly perturbed interface. Only
available for 3D.

| Key | Type | Description |
| --- | --- | --- |
| `N` | int | Number of Fourier modes used to perturb the interface (default `1`). |
| `rho` | random variable (required) | Thickness of the tube. |
| `delta` | random variable (required) | Amplitude scale of the interface perturbation. |
| `perturb` | random variable | Random variable used to draw the mode amplitudes and phases (default: `Uniform(0, 1)`). |

### `"Brownian Motion"`

A random velocity field given by a (fractional) Brownian motion.

| Key | Type | Description |
| --- | --- | --- |
| `hurst` | random variable | Hurst exponent $H$ of the fractional Brownian motion (default `0.5`, which corresponds to ordinary Brownian motion). Smaller values produce rougher fields. |

Available in 1D, 2D and 3D.

### `"Const Phys"`

A constant velocity field. Available for 2D and 3D.

| Key | Type | Description |
| --- | --- | --- |
| `u` | random variable (required) | Constant value of the $x$-velocity. |
| `v` | random variable (required) | Constant value of the $y$-velocity. |
| `w` | random variable (required, 3D only) | Constant value of the $z$-velocity. |

### `"Init From File"`

Reads the initial condition from the output of a previous azeban experiment.

| Key | Type | Description |
| --- | --- | --- |
| `experiment` | string (required) | Path to a NetCDF file, e.g. the file produced by the `NetCDF File` writer of a previous run. |
| `time` | int (required) | Index into the `time` dimension of the file (0-based), i.e. which stored snapshot to use. |
| `group` | string (required) | Name of the NetCDF group that contains the flow field, e.g. `flow_field_128` for a file written by the `Sample` content writer at resolution 128. |

The `sample_idx_start` top-level option selects the entry of the `member`
dimension: sample `i` of the new run reads member `sample_idx_start + i` of
the file. The variables `u`, `v` (and `w` in 3D) are read from the group; a
tracer is *not* read by this initializer.

### `"Python"`

Runs a Python script to fill the initial condition. Requires azeban to be
built with `-DENABLE_PYTHON=ON`. The script is executed with an embedded
Python interpreter; it receives a NumPy array `u` of shape
`(n_vars, N, N[, N])` (physical-space velocity components, plus the tracer
component if a tracer is configured) and must fill it *in place*. The domain
is the unit torus, i.e. `u[0, i, j] = u(x_i, y_j)` with
$x_i = i / N$, $y_j = j / N$.

| Key | Type | Description |
| --- | --- | --- |
| `script` | string (required) | Path to the Python script. The file is read at startup and also stored in the output file's `init_script` attribute. |
| `params` | object *or* array of objects | Named parameters passed to the script as NumPy arrays (see below). |

Each parameter object has the form

```json
{"name": "amplitudes", "value": {"name": "Normal", "mu": 0, "sigma": 1}, "N": 4}
```

| Key | Type | Description |
| --- | --- | --- |
| `name` | string (required) | Variable name under which the parameter is visible in the script. |
| `value` | random variable (required) | Distribution from which the values are drawn. |
| `N` | int | Number of values to draw (default `1`). The script receives them as a NumPy array of that length. |

## Tracer initializers

If the `init` section contains a `tracer` object, the simulation carries a
passively advected scalar field $\rho$ in addition to the velocity. The tracer
object requires a `name` key selecting one of the following initializers. The
tracer also increases the number of stored variables (output files then
contain an additional `rho` variable).

### `"Sphere"`

$\rho = 1$ inside a ball, $0$ outside. Available for 2D (a disk) and 3D.

| Key | Type | Description |
| --- | --- | --- |
| `center` | array of `dimension` numbers (required) | Center of the ball in the unit domain, e.g. `[0.5, 0.5, 0.5]`. |
| `radius` | number (required) | Radius of the ball. |

### `"Cube"`

$\rho = 1$ inside an axis-aligned box, $0$ outside. Available for 2D and 3D.

| Key | Type | Description |
| --- | --- | --- |
| `corner` | array of `dimension` numbers (required) | Lower corner of the box, e.g. `[0.25, 0.25, 0.25]`. |
| `size` | array of `dimension` numbers (required) | Extent of the box in each direction, e.g. `[0.5, 0.5, 0.5]`. |

### `"Const Fourier"`

Sets every Fourier coefficient of the tracer to `rho`, i.e. the density is
constant *in Fourier space*. In physical space this corresponds to a single
grid point of height `rho` at the domain origin.

| Key | Type | Description |
| --- | --- | --- |
| `rho` | number (required) | Value assigned to every Fourier coefficient of the tracer. |

## Example with tracer and Boussinesq forcing

```json
"equation": {
  "name": "Euler",
  "visc": {"type": "Smooth Cutoff", "eps": 0.05},
  "forcing": {"type": "Boussinesq"}
},
"init": {
  "name": "Const Phys",
  "u": 0,
  "v": 0,
  "tracer": {
    "name": "Sphere",
    "center": [0.5, 0.5],
    "radius": 0.2
  }
}
```
