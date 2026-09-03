# The configuration file

`azeban` is entirely configured through a single JSON file that is passed on
the command line. This page describes the structure of the file, the top-level
options and the two building blocks (snapshot *sequences* and *random
variables*) that are reused by many of the sections.

A minimal, complete example of a 3D Taylor-Green simulation on the GPU:

```json
{
  "device": "cuda",
  "dimension": 3,
  "num_samples": 1,
  "seed": 1,
  "time_offset": 0,
  "grid": {
    "N_phys": 512,
    "N_phys_pad": ""
  },
  "equation": {
    "name": "Euler",
    "visc": {
      "type": "Smooth Cutoff",
      "eps": 0.05
    }
  },
  "timestepper": {
    "type": "SSP RK3",
    "C": 0.125
  },
  "init": {
    "name": "Taylor Green",
    "perturb": {
      "name": "Uniform",
      "min": -0.025,
      "max": 0.025
    }
  },
  "writer": {
    "name": "NetCDF Snapshot",
    "path": "tg_N512",
    "snapshots": {
      "start": 0,
      "stop": 0.1,
      "n": 1
    }
  }
}
```

## Top-level options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dimension` | int | *required* | Spatial dimension of the simulation: `1`, `2` or `3`. Note that the distributed (MPI) code path currently only supports `2` and `3`, and that the only implemented equation (`Euler`) requires `2` or `3` as well. |
| `device` | string | `"cpu"` | Memory/compute device: `"cpu"` or `"cuda"`. `"cuda"` requires a CUDA build (`-DENABLE_CUDA=ON`) and selects device `0` of each process. If omitted, azeban prints a warning and falls back to the CPU. |
| `num_samples` | int | `1` | Number of independent simulations to run back to back. With MPI, this is the total number of samples over all rank groups (see [running.md](running.md)). Each sample is initialized independently (drawing fresh random numbers) and written under its own sample index. Note that stochastic *forcing* terms are seeded once per process, so samples running within the same process (or rank group) share the same forcing realization; only the initial conditions differ between those samples. |
| `sample_idx_start` | int | `0` | Index of the first sample. Sample indices are used in output file names and in the `member` dimension of collective NetCDF files. It is also the index into the `member` dimension when the `Init From File` initializer reads initial conditions from a previous experiment. Increment this when appending samples to an existing ensemble. |
| `time_offset` | double | `0` | Initial value of the simulation clock. The simulation itself always starts from the initial condition; this only shifts the time labels, e.g. to continue a time series started in a previous run. |
| `seed` | int | `1` | Seed for the random number generator. In serial runs the same seed drives the initial condition and the stochastic forcing. With MPI, rank group `g` uses `seed + g`, so different groups produce different random realizations. |
| `grid` | object | *required* | Discretization. See [grid.md](grid.md). |
| `equation` | object | *required* | The equation to solve, including the spectral viscosity and (optionally) a forcing term. See [equation.md](equation.md) and [forcing.md](forcing.md). |
| `timestepper` | object | *required* | Time integration scheme. See [timestepper.md](timestepper.md). |
| `init` | object | *required* | Initial condition. See [initializer.md](initializer.md). |
| `writer` | object *or* array of objects | *required* | Output specification. A single writer object, or a list of writer objects that are all evaluated. See [writers.md](writers.md). |

There is **no** top-level `time` key: the length of a simulation is determined
by the largest entry of the writers' `snapshots` sequences. The simulation
advances exactly to each requested snapshot time and stops once all snapshots
have been written. (Older configuration files sometimes contain a `time` key;
it is simply ignored by the current version.)

## Snapshot sequences

Writers store output at a fixed list of simulation times. Wherever a writer
expects a `snapshots` entry, one of the following JSON forms is accepted:

| Form | Meaning |
| --- | --- |
| `0.5` | A single snapshot at `t = 0.5`. |
| `[0, 0.5, 1.0]` | An explicit list of snapshot times. |
| `{"start": 0, "stop": 10, "n": 100}` | `n + 1` snapshots, evenly spaced between `start` and `stop` (both endpoints included). |
| `{"start": 0, "stop": 10, "step": 0.1}` | Snapshots every `0.1` time units from `start` up to and including `stop`. |

In the object form, `start` defaults to `0` and `stop` is required. The last
entry of a `step` sequence is always exactly `stop`, even if it is not an
integer multiple of `step` away from `start`.

The simulation is driven entirely by these snapshot times: azeban integrates
the equation up to the next requested snapshot, writes the output of every
writer whose next snapshot coincides with that time, and repeats until all
snapshot lists are exhausted. When `writer` is an array, each writer has its
own independent `snapshots` list.

## Random variables

Many parameters of the initial condition (and a few other places) accept a
*random variable* instead of a plain number. A random variable is either

- a plain number `x`, which is interpreted as the deterministic value `x`
  (equivalent to a point mass), or
- a JSON object of one of the following forms:

| Name | Parameters | Distribution |
| --- | --- | --- |
| `{"name": "Delta", "value": v}` | `value` (required) | Deterministic: always returns `v`. |
| `{"name": "Uniform", "min": a, "max": b}` | `min` (default `0`), `max` (default `1`) | Uniform distribution on `[a, b]`. |
| `{"name": "Normal", "mu": m, "sigma": s}` | `mu` (default `0`), `sigma` (default `1`) | Normal distribution with mean `m` and standard deviation `s`. |

Every time the simulation needs a value (once per sample for initial
conditions, once per time step for stochastic forcing terms), a fresh sample is
drawn from the distribution using the RNG seeded by the top-level `seed`.
