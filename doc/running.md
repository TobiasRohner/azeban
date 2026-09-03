# Running azeban

azeban ships two executables:

- `azeban` — the main simulation driver. It reads a JSON configuration file and
  runs one or more simulations, writing all output requested in the `writer`
  section of the configuration.
- `postprocess` — a small utility that applies the same output machinery to
  already existing samples (e.g. to compute spectra or structure functions of
  data that was stored without them).

Both executables are built into `<build_dir>` by the normal build
(see the main README).

## The `azeban` executable

```
azeban [OPTIONS] CONFIG
```

| Option | Description |
| --- | --- |
| `-h`, `--help` | Print a summary of the options and exit. |
| `CONFIG` (positional) | Path to the JSON configuration file. May also be given with `--config CONFIG`. This argument is mandatory. |
| `--ranks-per-sample R` | Only relevant for MPI builds (default: `1`). Splits the launched MPI ranks into independent groups of `R` ranks. Each group runs its own set of samples, i.e. the ranks inside a group collaborate on one simulation at a time, while different groups simulate different samples in parallel. |

Constraints for `--ranks-per-sample`:

- The total number of MPI ranks must be divisible by `R`.
- `num_samples` in the configuration file (see
  [configuration.md](configuration.md)) is interpreted as the *total* number of
  samples over all rank groups. Each group runs `num_samples / R_groups`
  samples (integer division), where `R_groups` is the number of rank groups.
  Choose `num_samples` to be a multiple of the number of rank groups, otherwise
  some samples are silently dropped.
- The `seed` is offset by the group index, so that samples running in different
  groups receive different random numbers (both for the initial condition and
  for stochastic forcing).

Example: launch a 3D simulation on 8 nodes with one GPU per node, where all 8
ranks collaborate on a single sample:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./build/azeban --ranks-per-sample=8 tg_N512.json
```

Example: run 16 independent samples, one per GPU (no collaboration between
ranks):

```bash
srun ./build/azeban --ranks-per-sample=1 ensemble.json   # launched with 16 ranks
```

Without MPI (single process), simply run

```bash
./build/azeban config.json
```

A single-process run always uses the CPU code paths unless `device` is set to
`cuda`, in which case one CUDA device is used (always device `0` of the
process).

Note that MPI builds require CUDA: it is currently not possible to build a
CPU-only MPI version (CMake will refuse to configure with
`-DENABLE_MPI=ON -DENABLE_CUDA=OFF`).

### Profiling output

If azeban was built with `-DENABLE_PROFILING=ON` (the default), the program
prints a summary of the time spent in the major code sections at the end of the
run and writes the raw profiling records to

- `profiling.out` for serial runs, or
- `profiling_rank<i>.out` for MPI runs (one file per rank, written in the
  working directory).

The scripts `scripts/plot_profiling_results.py` and
`scripts/plot_profiling_timeline.py` can be used to visualize these files.

## The `postprocess` executable

```
postprocess --sample SAMPLE --config CONFIG [--sample_idx IDX] [--time T]
```

| Option | Description |
| --- | --- |
| `-h`, `--help` | Print a summary of the options and exit. |
| `--sample SAMPLE` (required) | Path to a NetCDF file containing a stored sample with variables `u`, `v`, (`w`), (`rho`). This is the format written by the `NetCDF Snapshot` writer (see [writers.md](writers.md)). The dimensionality of the simulation is detected from this file. |
| `--config CONFIG` (required) | Path to a configuration file. Only the `writer` section (and the `dimension`/`device` keys, if present) of this file is used. The `snapshots` entries of the writers are ignored. |
| `--sample_idx IDX` | Sample index used when naming the output files (default: `0`). |
| `--time T` | Snapshot/time index used when naming the output files (default: `0`). |

The tool reads the sample, transforms it to Fourier space and then invokes the
configured writer(s) on both the physical and the spectral representation.
This is useful to compute derived quantities (energy spectra, enstrophy
spectra, structure functions) after the fact from stored flow fields, using
exactly the same code paths as the in-situ writers.

Example: compute the energy spectrum of a stored snapshot with
`postprocess`. Given a sample file `output/sample_0_time_9.nc`, create a
configuration file `spectrum_config.json` containing only the writer
specification:

```json
{
  "writer": {
    "name": "Energy Spectrum",
    "path": "spectrum_out",
    "snapshots": 0
  }
}
```

and run

```bash
./build/postprocess --sample output/sample_0_time_9.nc \
                    --config spectrum_config.json \
                    --sample_idx 0 --time 9
```

The `snapshots` entry is required by the writer factories but its value is
irrelevant here; the output is written immediately. The full range of writer
types (see [writers.md](writers.md)) can be used in the same way, e.g. to
compute structure functions of stored samples.
