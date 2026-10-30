# FrequencySweeps.jl

[English](README.md) | [Español](README-ES.md)

FrequencySweepsjl es un módulo para realizar barridos de frecuencia adaptativos a usando solvers tipo caja negra. El paquete construye un modelo `DescriptorModel` o un `BarycentricModel`. Funciona para problemas SISO o MIMO y acepta cualquier función del tipo `solver(f)` que devuelva un valor escalar o en forma de matriz.

Al tratarse de un algoritmo de barrido no intrusivo no es posible garantizar que el modelo devuelto captura la respuesta real del problema sin conocimiento adicional del problema específico. Para tratar de evitar paradas tempranas se han implementado de manera opcional (pero recomendadas) algunas técnicas adicionales sobre los algoritmos usados como referencia.

Las opciones configurables de cada tipo de sweep se describen en sus docstrings. Puedes consultarlos desde Julia usando:

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

Ambas entradas admiten muestras iniciales ya calculadas:

```julia
result, data = sweep(solver, frequencies, initial_frequencies, initial_samples; options)
barycentric = barycentric_sweep(solver, frequencies, initial_frequencies, initial_samples; options=config)
```

`sweep` devuelve el resultado y los datos adquiridos. `barycentric_sweep` devuelve el resultado, cuyo modelo contiene los soportes y las respuestas adquiridas. Ambos resultados exponen `model`, `iterations` y `termination`.

```julia
solver(f) = 1 / (1 + 2π * im * f)
result, data = sweep(solver, frequencies)
values = result.model(frequencies) # Vector{ComplexF64}
```

## Referencias

- Sweep Loewner con interpolación MFTI: [Sinha y T. N., *Fully-Adaptive and Semi-Adaptive Frequency Sweep Algorithm Exploiting Loewner-State Model for EM Simulation of Multiport Systems*](https://doi.org/10.1109/TMTT.2025.3557208).
- Sweep baricéntrico: [Pradovera, *Toward a certified greedy Loewner framework with minimal sampling*](https://doi.org/10.1007/s10444-023-10091-7).
