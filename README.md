# azeban

azeban is a GPU-accelerated pseudospectral solver for the incompressible Euler
equations with spectral viscosity, written in C++17 with CUDA and MPI support.
It integrates the equations in Fourier space on the torus, applies a spectral
viscosity regularization that vanishes under grid refinement, and supports
stochastic forcing, passively advected tracers, ensemble runs, and a rich set
of in-situ diagnostics (flow-field snapshots, energy/enstrophy spectra,
structure functions, and ParaView Catalyst visualization).

## Documentation

- [Running azeban](doc/running.md) — command line usage, MPI parallelization
  over samples, profiling output, the `postprocess` tool.
- [The configuration file](doc/configuration.md) — top-level options,
  snapshot sequences, random variables.
- [Grid](doc/grid.md) — resolutions and dealiasing padding.
- [Equation](doc/equation.md) — equations and spectral viscosity.
- [Forcing](doc/forcing.md) — available forcing terms.
- [Timestepper](doc/timestepper.md) — time integration schemes and the CFL
  constant.
- [Initializer](doc/initializer.md) — initial conditions and tracers.
- [Writers](doc/writers.md) — all output writers and their options.
- [Output formats](doc/output-formats.md) — file formats of the outputs.

## Building

### Requirements

- A C++17 compiler, CMake ≥ 3.24 and OpenMP.
- [FFTW3](https://www.fftw.org/) (found via the bundled
  `cmake/FindFFTW3.cmake`; on Debian/Ubuntu: `apt install libfftw3-dev`).
- For MPI builds (`ENABLE_MPI=ON`): MPI (C and CXX bindings).
- For CUDA builds (`ENABLE_CUDA=ON`): the CUDA toolkit (cuFFT and cuRAND are
  used).

Everything else — Boost.Program_options, fmt, nlohmann/json, Catch2, Google
Test, Google Benchmark, NetCDF-C, and the internal libraries
[ZisaCore](https://github.com/1uc/ZisaCore) and
[ZisaMemory](https://github.com/1uc/ZisaMemory) — is downloaded and built
automatically by CMake's `FetchContent` during configuration. An internet
connection is therefore required for the first configure step.

Optional dependencies:

- Python development headers, for the Python initializer
  (`ENABLE_PYTHON=ON`).
- [catalyst 2.x](https://gitlab.kitware.com/paraview/catalyst), for in-situ
  visualization (`ENABLE_INSITU=ON`).
- HDF5 (`HAVE_HDF5=ON`; rarely needed, NetCDF is the primary IO backend).

### Configuring and compiling

```bash
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=ON \
      -DENABLE_MPI=ON
cmake --build build -j$(nproc)
```

Available CMake options:

| Option | Default | Description |
| --- | --- | --- |
| `ENABLE_CUDA` | `ON` | Build the CUDA code paths and enable the `device: "cuda"` configuration option. |
| `ENABLE_MPI` | `ON` | Build with MPI support for distributed simulations and ensemble runs. **MPI requires CUDA**; configuring with `ENABLE_MPI=ON` and `ENABLE_CUDA=OFF` is rejected. |
| `ENABLE_CUDA_AWARE_MPI` | `ON` | Use CUDA-aware MPI if the installed MPI implementation supports it. |
| `ENABLE_PYTHON` | `OFF` | Embed a Python interpreter for the `"Python"` initializer. |
| `ENABLE_INSITU` | `OFF` | Enable the ParaView Catalyst in-situ visualization writer (needs an installed catalyst 2.x library). |
| `ENABLE_BENCHMARKS` | `ON` | Build the micro-benchmark executables. |
| `ENABLE_PROFILING` | `ON` | Instrument the code with the internal profiler and write `profiling*.out` files at the end of each run. Turn this off for production runs. |
| `SINGLE_PRECISION` | `OFF` | Compute in single instead of double precision. |
| `HAVE_HDF5` | `OFF` | Compile with HDF5 support. |
| `HAVE_NETCDF` | `ON` | Compile with NetCDF support (required by all standard writers). |

Valid build types are `Release`, `Debug`, and `FastDebug` (`-O3 -g`, a good
middle ground for development). For CUDA builds, `Debug` additionally enables
device-side debugging (`-g -G`).

The build produces the following executables in `build/`:

| Target | Description |
| --- | --- |
| `azeban` | The simulation driver (see [doc/running.md](doc/running.md)). |
| `postprocess` | Applies the output writers to existing samples (see [doc/running.md](doc/running.md)). |
| `unit_tests` | Unit tests (run with `ctest` or directly). |
| `micro_benchmarks` | Google-benchmark based micro benchmarks. |
| `system_properties` | Prints properties of the system (MPI, GPUs, ...). Run under `srun`/`mpirun`. |
| `benchmark_fft` | FFT benchmark: `srun ./build/benchmark_fft N` with `N` the grid size. |

### Docker

A self-contained image (CUDA 12.9, Ubuntu 24.04, OpenMPI 5, UCX, PMIx,
catalyst) can be built with:

```bash
docker build -t azeban .
```

The image builds with `-DENABLE_PYTHON=ON -DENABLE_INSITU=ON` and has the
`azeban` executable as its entrypoint, so a simulation can be run with

```bash
docker run --gpus all -v $PWD:/data azeban /data/config.json
```

Build arguments `ENABLE_PROFILING` (default `OFF`) and `SINGLE_PRECISION`
(default `ON`) are supported.

## Quick start

1. Create a configuration file `config.json`:

```json
{
  "device": "cuda",
  "dimension": 3,
  "num_samples": 1,
  "seed": 1,
  "grid": {
    "N_phys": 128,
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
    "C": 0.5
  },
  "init": {
    "name": "Taylor Green"
  },
  "writer": {
    "name": "NetCDF Snapshot",
    "path": "output",
    "snapshots": {"start": 0, "stop": 5, "n": 50}
  }
}
```

2. Run it:

```bash
./build/azeban config.json
```

3. Inspect the output in `output/` (see
   [doc/output-formats.md](doc/output-formats.md)).

All configuration options are documented in
[doc/configuration.md](doc/configuration.md) and the pages linked there.

## License

GPL-3.0-or-later, see [LICENSE.md](LICENSE.md).
