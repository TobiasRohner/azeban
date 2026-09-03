# Output formats

This page describes the files produced by the various writers (see
[writers.md](writers.md) for how to configure them). All binary output is in
[NetCDF4](https://www.unidata.ucar.edu/software/netcdf/) format and can be
read, e.g., with `ncdump`, Python (`netCDF4`, `xarray`) or ParaView.

## NetCDF Snapshot files (`NetCDF Snapshot` writer)

One file per snapshot and sample, named

```
<path>/sample_<sample>_time_<idx>.nc
```

where `<sample>` is the sample index (starting at `sample_idx_start`) and
`<idx>` is the 0-based position in the snapshot list (not the time).

Physical-space mode (default, `fourier: false`):

| Variable | Shape | Description |
| --- | --- | --- |
| `t` | scalar | Simulation time of this snapshot. |
| `u` | `(N, N)` / `(N, N, N)` | $x$-velocity on the physical grid. |
| `v` | same as `u` | $y$-velocity. |
| `w` | same as `u` (3D only) | $z$-velocity. |
| `rho` | same as `u` | Tracer density (only if the simulation has a tracer). |

Fourier mode (`fourier: true`): the same information is stored as
`u_hat_abs`/`u_hat_arg` (etc.): modulus and phase of the complex spectral
coefficients on the Fourier grid of shape `(N, ..., N/2 + 1)`, exactly as the
solver keeps them internally (i.e. *without* the `1/N^dim` normalization that
is applied before transforming back to physical space).

## Collective NetCDF file (`NetCDF Collective` writer)

A single NetCDF4 file with:

| Dimension | Size | Description |
| --- | --- | --- |
| `member` | `num_samples` | Ensemble member (sample) index. |
| `time` | length of the snapshot sequence | Snapshot index. |
| `x`, `y`, `z` | `N_phys` | Physical grid coordinates. |

| Variable | Shape | Description |
| --- | --- | --- |
| `member` | `(member)` | The member indices. |
| `time` | `(time)` | The snapshot times. |
| `x`, `y`, `z` | `(x)` etc. | Grid coordinates $i/N$. |
| `u`, `v`, (`w`) | `(member, time, x[, y[, z]])` | Velocity components. |
| `rho` | `(member, time, x[, y[, z]])` | Tracer (only if present). |
| `p` | `(member, time, x[, y[, z]])` | Pressure (only with `save_pressure: true`). |

When azeban runs with MPI, the file is created with parallel NetCDF and all
ranks write cooperatively.

## NetCDF experiment file (`NetCDF File` writer)

A single NetCDF4 file that acts as a self-describing experiment archive. In
addition to the dimensions and variables of the collective writer it contains:

- Global attribute `config`: the full JSON configuration used for the run.
- Global attribute `init_script`: the Python initializer script, if one was
  used.
- One group per configured content writer:

| Content | Group name | Variables |
| --- | --- | --- |
| `Sample` | `flow_field_<N>` | `u`, `v`, (`w`), (`rho`) over `(member, time, x[, y[, z]])` at resolution `N`; with `save_statistics` also the subgroups `mean` and `variance` containing running mean/variance of each field over `(time, x[, ...])`. |
| `Energy Spectrum` | `energy_spectrum` | `time`, `k`, `Ek(member, time, k)`, `real_time(time)` (wall-clock time). |
| `Enstrophy Spectrum` | `enstrophy_spectrum` | same layout as the energy spectrum. |
| `Second Order Structure Function` | `second_order_structure_function` | `time`, `r`, `S2(member, time, r)`, `real_time(time)`. |
| `Structure Function Cube` | `structure_function_cube_<p>` / `third_order_structure_function` / `structure_function_longitudinal_<p>` / `structure_function_absolute_longitudinal_<p>` | `time`, `r`, `SF(member, time, r)`, `real_time(time)`. |

The `r` variable contains the separations (in grid-point units scaled to the
unit domain) at which the structure functions are sampled.

## Text outputs

### Energy / enstrophy spectra

- `Energy Spectrum` writes `<path>/energy_<sample>.txt`.
- `Enstrophy Spectrum` writes `<path>/enstrophy_<sample>.txt`.

Each snapshot appends one row; a row contains the spectrum values for
$k = 0, \dots, N_{\text{fourier}} - 1$ separated by tabs. Row `i` corresponds
to the `i`-th entry of the snapshot list.

### Structure functions

The `Structure Function` writer writes one file per snapshot:

```
<path>/<prefix>_<sample>_time_<idx>.txt
```

with `<prefix>` one of `S2`, `S3`, `SF_Cube_<p>`, `S_par`, `S_par_abs` (see
the table in [writers.md](writers.md)). The file contains one value per
tab-separated entry, sampled at the separations $h = 0$, `stride`, `2*stride`,
..., `maxH` grid points.

## Reading outputs in Python

Minimal example using `netCDF4` for a `NetCDF File` output:

```python
import netCDF4 as nc

f = nc.Dataset("N128.nc")
print(f.config)                     # the full configuration
grp = f.groups["flow_field_3"]      # downsampled flow fields
u = grp.variables["u"][:]           # shape (member, time, x, y)
spec = f.groups["energy_spectrum"]
E = spec.variables["Ek"][:]         # shape (member, time, k)
```

The scripts shipped in `scripts/` (`make_euler_plots.py`,
`plot_incompressible_euler.py`, `structure_postprocess.py`, ...) contain
further examples for loading and plotting azeban output.
