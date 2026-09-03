# Writers

The `writer` section of the configuration file specifies which output is
written, at which times, and where. It is either a single writer object or an
array of writer objects; in the latter case all writers are evaluated during
the same run.

```json
"writer": [
  {
    "name": "NetCDF Snapshot",
    "path": "output",
    "snapshots": {"start": 0, "stop": 1, "n": 100}
  },
  {
    "name": "Energy Spectrum",
    "path": "output",
    "snapshots": {"start": 0, "stop": 1, "n": 1000}
  }
]
```

## Common properties

- Every writer has a required `name` key that selects the writer type and a
  required `snapshots` key that specifies the simulation times at which it
  writes (see [Snapshot sequences](configuration.md#snapshot-sequences)). The
  simulation advances exactly from snapshot to snapshot, so the total run time
  is determined by the largest snapshot time over all writers.
- Each writer only writes at *its own* snapshot times. Different writers can
  use different (independent) snapshot lists.
- The sample index used in file names and NetCDF `member` dimensions comes
  from the top-level options `sample_idx_start` / `num_samples` (see
  [configuration.md](configuration.md)).
- Output in physical space is always written on the unpadded grid
  (`N_phys` points per direction).

## Writers

- [`"NetCDF Snapshot"`](#netcdf-snapshot) — flow fields, one NetCDF file per
  snapshot.
- [`"NetCDF Collective"`](#netcdf-collective) — flow fields, a single
  (MPI-parallel) NetCDF file.
- [`"NetCDF File"`](#netcdf-file) — a single NetCDF file with configurable
  contents: downsampled flow fields, spectra, structure functions.
- [`"Energy Spectrum"`](#energy-spectrum) — energy spectrum $E(k)$ as text.
- [`"Enstrophy Spectrum"`](#enstrophy-spectrum) — enstrophy spectrum as text.
- [`"Structure Function"`](#structure-function) — structure functions as text.
- [`"Catalyst"`](#catalyst) — in-situ visualization with ParaView Catalyst.

Details about the file formats are collected in
[output-formats.md](output-formats.md).

---

## `"NetCDF Snapshot"`

Writes one NetCDF file per snapshot and per sample. This is the writer to
use when the full flow fields are needed for visualization in ParaView.

| Key | Type | Description |
| --- | --- | --- |
| `path` | string (required) | Output directory. Created if it does not exist. One file `sample_<sample>_time_<idx>.nc` is written per snapshot, where `<idx>` is the 0-based index into the snapshot list. |
| `snapshots` | sequence (required) | Simulation times at which to store a snapshot. |
| `fourier` | bool | If `true`, the *Fourier coefficients* are stored instead of the physical fields (default `false`). |

The files contain the variables `u`, `v` (`w` in 3D) and, if the simulation
has a tracer, `rho`. In Fourier mode the modulus and phase of the coefficients
are stored as `u_hat_abs`/`u_hat_arg` etc.

Note for MPI runs: the files are gathered and written by rank 0 of each rank
group, and Fourier output is not supported in MPI runs.

## `"NetCDF Collective"`

Writes all snapshots and all samples into a single NetCDF4 file using parallel
NetCDF when running under MPI. This writer is optimized for filesystem performance.

| Key | Type | Description |
| --- | --- | --- |
| `path` | string (required) | Path of the output file. |
| `snapshots` | sequence (required) | Simulation times at which to store a snapshot. The file gets a `time` dimension of this length. |
| `save_pressure` | bool | If `true`, the pressure $p$ is reconstructed from the velocity field at every snapshot and stored in a variable `p` (default `false`). |

The file has the dimensions `member` (total number of samples), `time`
(number of snapshots) and `x`/`y`/`z`, and the variables `u`, `v`, `w`, `rho`
(if a tracer is present) and optionally `p`. Only physical-space fields are
written.

## `"NetCDF File"`

Writes a single NetCDF4 file whose contents are freely composed of several
*content writers*. This is the recommended writer for large ensembles, since
everything (including the full configuration) ends up in one self-describing
file.

| Key | Type | Description |
| --- | --- | --- |
| `path` | string (required) | Path of the output file. |
| `contents` | array of content objects | The content writers (see below). If omitted, the file only contains the coordinate variables and the global attributes. |

The file always contains:

- the dimensions `member` (total number of samples), `x`, `y`, `z`,
- the coordinate variables `member`, `x`, `y`, `z` (grid points $i/N$),
- the global attributes `config` (the full configuration as JSON) and, when a
  Python initializer was used, `init_script` (the script itself).

### Content: `"Sample"`

Stores (possibly downsampled) flow fields.

| Key | Type | Description |
| --- | --- | --- |
| `snapshots` | sequence (required) | Simulation times at which to store the fields. |
| `N` | int | Resolution at which the fields are stored (default: `N_phys` of the simulation). If smaller than `N_phys`, the fields are downsampled in situ. |
| `save_statistics` | bool | If `true`, running mean and variance over the samples are accumulated and stored in the subgroups `mean` and `variance` (default `false`). |

The data is stored in the group `flow_field_<N>` with the variables `u`, `v`,
`w`, `rho` over `(member, time, x[, y[, z]])`.

### Content: `"Energy Spectrum"`

| Key | Type | Description |
| --- | --- | --- |
| `snapshots` | sequence (required) | Simulation times at which to compute the spectrum. |

Stores the energy spectrum $E(k)$ in the group `energy_spectrum` with the
variables `time`, `k`, `Ek(member, time, k)` and `real_time` (wall-clock time
of each write).

### Content: `"Enstrophy Spectrum"`

Same options and layout as `"Energy Spectrum"`, but for the enstrophy
spectrum, stored in the group `enstrophy_spectrum`.

### Content: `"Second Order Structure Function"`

| Key | Type | Description |
| --- | --- | --- |
| `snapshots` | sequence (required) | Simulation times at which to compute the structure function. |
| `exact` | bool | If `true`, the structure function is computed exactly by summing over all pairs of grid points ($O(N^{2d})$). If `false` (default), a fast approximate algorithm based on the energy spectrum is used. |

Stored in the group `second_order_structure_function` with the variables
`time`, `r`, `S2(member, time, r)` and `real_time`.

### Content: `"Structure Function Cube"`

Computes structure functions by explicitly summing over displacement vectors
on a grid of separations.

| Key | Type | Description |
| --- | --- | --- |
| `type` | string (required) | One of `"SFCube"`, `"Third Order"`, `"Longitudinal"`, `"Absolute Longitudinal"`. |
| `snapshots` | sequence (required) | Simulation times at which to compute the structure function. |
| `p` | number (required, except for `"Third Order"`) | Order of the structure function: $S_p(r) = \langle |u(x+r) - u(x)|^p \rangle$. |
| `max_h` | int | Largest separation (in grid points) to consider (default: `(N_phys + 1) / 2`). |
| `stride` | int | Step size in grid points when enumerating separations and base points; values $> 1$ make the computation cheaper at the cost of sampling density (default `1`). |

Stored in the groups `structure_function_cube_<p>`,
`third_order_structure_function`, `structure_function_longitudinal_<p>` or
`structure_function_absolute_longitudinal_<p>` with the variables `time`, `r`,
`SF(member, time, r)` and `real_time`.

Note: in MPI runs, only the contents `"Sample"`, `"Energy Spectrum"`,
`"Enstrophy Spectrum"` and `"Second Order Structure Function"` are available;
the `"Structure Function Cube"` content requires a single rank per sample.

## `"Energy Spectrum"`

Appends the energy spectrum to a text file. One row per snapshot; each row
contains the values $E(k)$ for $k = 0, \dots, N_{\text{fourier}} - 1$,
tab-separated.

| Key | Type | Description |
| --- | --- | --- |
| `path` | string (required) | Output directory. The file is `path/energy_<sample>.txt`. |
| `snapshots` | sequence (required) | Simulation times at which to append a row. |

Works in serial and MPI runs (rank 0 writes).

## `"Enstrophy Spectrum"`

Same as `"Energy Spectrum"`, but for the enstrophy spectrum. The output file
is `path/enstrophy_<sample>.txt`.

## `"Structure Function"`

Computes structure functions of the velocity field and writes them as text.
The subtype is selected with the required key `type`.

| Key | Type | Description |
| --- | --- | --- |
| `type` | string (required) | One of `"Second Order"`, `"Third Order"`, `"Cube"`, `"Longitudinal"`, `"Absolute Longitudinal"`. |
| `path` | string (required) | Output directory. |
| `snapshots` | sequence (required) | Simulation times at which to write a structure function. |
| `exact` | bool | Only for `"Second Order"`: compute the exact pair-sum version instead of the spectral approximation (default `false`). |
| `p` | number | Order of the structure function; required for `"Cube"`, `"Longitudinal"` and `"Absolute Longitudinal"`. |
| `maxH` | int | Largest separation in grid points (default `N_phys / 2`; not used by `"Second Order"`). |
| `stride` | int | Step size when enumerating separations (default `1`; not used by `"Second Order"`). |

Each writer produces one text file per snapshot:

| Type | File name | Meaning |
| --- | --- | --- |
| `"Second Order"` | `S2_<sample>_time_<idx>.txt` | $S_2(r)$, either exact or spectrally approximated. |
| `"Third Order"` | `S3_<sample>_time_<idx>.txt` | Third-order structure function $S_3(r)$. |
| `"Cube"` | `SF_Cube_<p>_<sample>_time_<idx>.txt` | $S_p$ for all separation *vectors* up to `maxH`. |
| `"Longitudinal"` | `S_par_<sample>_time_<idx>.txt` | Longitudinal structure function of order `p`. |
| `"Absolute Longitudinal"` | `S_par_abs_<sample>_time_<idx>.txt` | Longitudinal structure function of the absolute velocity increment, order `p`. |

Under MPI only `"Second Order"` is supported; the other types require a serial
run (or the `"NetCDF File"` writer, where `"Third Order"` is also available).

## `"Catalyst"`

In-situ visualization with [ParaView Catalyst](https://www.paraview.org/).
Instead of writing files, the flow field is streamed to one or more Catalyst
Python pipelines, which typically render images. Requires azeban to be built
with `-DENABLE_INSITU=ON` and a Catalyst 2 library to be available.

| Key | Type | Description |
| --- | --- | --- |
| `scripts` | array of arrays of strings (required) | Each inner array is one Catalyst pipeline: the first element is the path to the Python script, the remaining elements are passed to the script as command-line arguments. |
| `snapshots` | sequence (required) | Simulation times at which the pipelines are executed. |

Example using the scripts shipped in the `catalyst/` folder:

```json
{
  "name": "Catalyst",
  "scripts": [["/path/to/azeban/catalyst/isosurface_curl.py",
               "--isosurfaces", "80",
               "--output", "dst_r0_N256/isosurface_%04d.png"]],
  "snapshots": {"start": 0, "stop": 1, "n": 300}
}
```

The shipped pipeline scripts are:

- `catalyst/isosurface_curl.py` — renders isosurfaces of the vorticity
  magnitude (`--isosurfaces v1 v2 ...`, `--output pattern` with `%04d`
  placeholder for the snapshot index).
- `catalyst/vorticity_magnitude.py` — volume rendering of the vorticity
  magnitude (`--output pattern`).
- `catalyst/q_criterion.py` — isosurfaces of the Q criterion.

Only physical-space fields are sent to Catalyst; a tracer is not visualized
by the shipped scripts.
