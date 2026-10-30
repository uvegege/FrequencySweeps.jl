# FrequencySweeps.jl

A module for adaptive frequency sweeps using a black-box solver.

[English](README.md) | [Español](README-ES.md)

FrequencySweeps.jl provides an adaptive frequency sweep framework for black-box solvers. The package constructs either a `DescriptorModel` or a `BarycentricModel`. It supports SISO and MIMO problems and accepts any callable of the form `solver(f)` that returns either a scalar or a response matrix.

Because these are non intrusive sweep algorithms, they cannot guarantee that the returned model captures the true problem response without additional knowledge of the problem. Optional techniques have been added to help reduce premature stopping.

The configurable options for each sweep are described in their docstrings. You can inspect them from Julia with:

```julia
help?> SweepOptions
help?> BarycentricOptions
```

```julia
using FrequencySweeps
options = SweepOptions(; tol=1e-3, memory=3, distance_exponent=0.5)
result, data = sweep(solver, frequencies; options)

config = BarycentricOptions(; tol=1e-3, memory=3)
barycentric = barycentric_sweep(solver, frequencies; options=config)
```

Both functions accept precomputed initial samples:

```julia
result, data = sweep(solver, frequencies, initial_frequencies, initial_samples; options)
barycentric = barycentric_sweep(solver, frequencies, initial_frequencies, initial_samples; options=config)
```

`sweep` returns the result and the acquired data. `barycentric_sweep` returns a result whose model stores the support points and their responses. Both results expose `model`, `iterations`, and `termination`.

```julia
solver(f) = 1 / (1 + 2π * im * f)
result, data = sweep(solver, frequencies)
values = result.model(frequencies) # Vector{ComplexF64}
```

## References

- Loewner sweep with MFTI interpolation: [Sinha and T. N., *Fully-Adaptive and Semi-Adaptive Frequency Sweep Algorithm Exploiting Loewner-State Model for EM Simulation of Multiport Systems*](https://doi.org/10.1109/TMTT.2025.3557208).
- Barycentric sweep: [Pradovera, *Toward a certified greedy Loewner framework with minimal sampling*](https://doi.org/10.1007/s10444-023-10091-7).
