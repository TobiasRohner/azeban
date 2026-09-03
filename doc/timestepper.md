# Timestepper

The `timestepper` section selects the time integration scheme and the CFL
constant that controls the time step size.

```json
"timestepper": {
  "type": "SSP RK3",
  "C": 0.125
}
```

## Options

| Key | Type | Description |
| --- | --- | --- |
| `type` | string (required) | One of `"Forward Euler"`, `"SSP RK2"`, `"SSP RK3"`, `"Euler Maruyama"`. |
| `C` | double (required) | CFL constant. Every time step is bounded by `dt = C * min(dt_visc, dt_advect)`, where the advective bound is derived from the current maximum velocity and the viscous bound from the viscosity strength and the grid resolution. `C` must satisfy `0 < C <= 1`; it is a safety factor on the maximal stable time step. Smaller values make the run slower but more accurate/stable. |

## Integrators

- **`"Forward Euler"`** — Explicit first-order Euler step
  $u^{n+1} = u^n + \Delta t \, F(u^n)$. Cheap but only first-order accurate in
  time. Useful for quick tests.

- **`"SSP RK2"`** — Second-order strong-stability-preserving Runge-Kutta
  method.

- **`"SSP RK3"`** — Third-order strong-stability-preserving Runge-Kutta
  method. The default choice for deterministic simulations in azeban.

- **`"Euler Maruyama"`** — Euler-Maruyama method for stochastic differential
  equations. Use this whenever a stochastic forcing (`"White Noise"` or
  `"White Noise High Freq"`) is configured: those forcings scale with
  $1/\sqrt{\Delta t}$, which is only consistent with this integrator.

## Example

A typical configuration for a deterministic run:

```json
"timestepper": {
  "type": "SSP RK3",
  "C": 0.5
}
```

A typical configuration for a stochastically forced run:

```json
"timestepper": {
  "type": "Euler Maruyama",
  "C": 0.001
}
```

Note that stochastic runs usually require a much smaller `C`, because the
$1/\sqrt{\Delta t}$ scaling of the noise makes large steps unstable.
