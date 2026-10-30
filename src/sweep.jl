"""
    sweep(solver, candidates; D=nothing, options=SweepOptions(), convergence_tol=options.stability_tol)
    sweep(solver, candidates, f₀, H₀; kwargs...)

Build reduced descriptor models from the sampled solver responses using
matrix-format tangential interpolation (MFTI) with an even odd partition. 
At each iteration, compare two reduced Loewner models over the candidate 
frequencies and sample the frequency with the largest pseudoerror. 
The pseudoerror can optionally be weighted by distance from the existing samples.

At the selected frequency, compare the solver response with the model
prediction. Stop after `options.memory` consecutive comparisons meet `options.tol`.
If `options.relative_convergence` is enabled, also require successive model
predictions to be stable over the candidate grid.

# Arguments
- `solver`: callable `solver(f)` returning a scalar SISO response or a response matrix.
- `candidates`: candidate frequencies in Hz, also used as the temporal stability grid.
- `f₀`: initial frequencies in Hz; omitted to use the initialization selected by `options`.
- `H₀`: solver responses at `f₀`, in the same order.

# Keywords
- `D = nothing`: scalar or matrix direct term, subtracted before reduction and added back during evaluation; defaults to zero with the response shape.
- `options::SweepOptions = SweepOptions()`: initialization, reduction, selection, threading and stopping parameters.
- `convergence_tol = options.stability_tol`: temporal stability tolerance, used when `options.relative_convergence` is enabled.

# Returns
- `Tuple{SweepResult, LoewnerData}`: final model with validation history and stopping reason, followed by all sampled `frequencies` and original solver `samples`, including initialization.

# Reference

[Sinha y T. N., *Fully-Adaptive and Semi-Adaptive Frequency Sweep Algorithm Exploiting Loewner-State Model for EM Simulation of Multiport Systems*](https://doi.org/10.1109/TMTT.2025.3557208).

"""
function sweep(solver::F, candidates; D =nothing, options::SweepOptions = SweepOptions(), convergence_tol = options.stability_tol) where F
    fmin, fmax = extrema(candidates)
    f₀ = options.adaptive ? [fmin, fmax] : semi_adaptive_freqs(fmin, fmax, max(2, options.n0))
    H₀ = map(solver, f₀)
    return sweep(solver, candidates, f₀, H₀; D, options, convergence_tol)
end

function sweep(solver::F, candidates, f₀, H₀; D=nothing, options::SweepOptions=SweepOptions(), convergence_tol=options.stability_tol) where F
    
    length(f₀) == length(H₀) || throw(ArgumentError("f₀ and H₀ must have same length"))
    if options.relative_convergence
        isfinite(convergence_tol) && convergence_tol >= 0 || throw(ArgumentError("invalid stability tolerance"))
        isfinite(options.stability_floor) && options.stability_floor > 0 || throw(ArgumentError("stability_floor must be finite and positive"))
    end

    f₀ = collect(f₀)
    H₀ = collect(H₀)
    candidates = collect(candidates)
    sampled = Set(f₀)
    D = ComplexF64.(isnothing(D) ? zero(H₀[1]) : D)
    size(D) == size(H₀[1]) || throw(DimensionMismatch("D and solver responses"))
    errors = Vector{Float64}(undef, length(candidates))
    Hₖ = [zero(D) for _ in candidates]
    Hₖ₋₁ = [zero(D) for _ in candidates]

    sa, sb, Ha, Hb = loewner_partition_evenodd(f₀, [H - D for H in H₀])
    L, σL = loewner_mfti(Ha, Hb, sa, sb)
    V, W = vw_mfti(Ha, Hb)
    model1, model2 = loewner_models(L, σL, V, W, f₀, D, options)
    
    memory = 0
    history = SweepIteration[]
    stable = true

    while memory < options.memory && length(f₀) < options.max_samples || all(f -> f ∈ sampled, candidates)
        
        loewner_pseudo_errors!(errors, Hₖ, candidates, model1, model2, options.Δf, options.nthreads; blas_threads=options.blas_threads)

        if !iszero(options.distance_exponent)
            weighted_pseudoerrors!(errors, candidates, sampled, options.distance_exponent)
        end

        idx = argmax(Base.Fix1(getindex, errors), (i for i in eachindex(candidates) if candidates[i] ∉ sampled))
        indicator = errors[idx]
        f_new = candidates[idx]
        H_new = solver(f_new)
        error_act = relative_change(H_new, Hₖ[idx])
        push!(history, SweepIteration(f_new, indicator, error_act))

        if options.relative_convergence
            delta = approximation_convergence(Hₖ, Hₖ₋₁; floor=options.stability_floor)
            stable = delta < convergence_tol
            Hₖ, Hₖ₋₁ = Hₖ₋₁, Hₖ
        end

        memory = last(history).error < options.tol && stable ? memory + 1 : 0

        push!(f₀, f_new)
        push!(H₀, H_new)
        push!(sampled, f_new)
        sort_idx = sortperm(f₀)
        permute!(f₀, sort_idx)
        permute!(H₀, sort_idx)

        sa, sb, Ha, Hb = loewner_partition_evenodd(f₀, [H - D for H in H₀])
        L, σL = loewner_mfti(Ha, Hb, sa, sb)
        V, W = vw_mfti(Ha, Hb)
        model1, model2 = loewner_models(L, σL, V, W, f₀, D, options)
    end

    if memory >= options.memory
        termination = :tolerance
    elseif length(f₀) >= options.max_samples
        termination = :max_samples
    else
        termination = :max_iterations
    end

    return SweepResult(model1, history, termination), LoewnerData(f₀, H₀)
end

"""
    barycentric_sweep(solver, candidates; options=GreedyLoewnerConfig())
    barycentric_sweep(solver, candidates, f₀, H₀; options=GreedyLoewnerConfig())

Build a barycentric approximation by selecting the maximum inverse denominator,
validating against the solver, and incorporating each new sample. Stop after
`options.memory` consecutive successful validations, or when the sample budget
or candidates are exhausted. Optional temporal stability is checked after each
sample update on the original candidate grid.

# Arguments
- `solver`: callable `solver(f)` returning a scalar or matrix response.
- `candidates`: candidate frequencies in Hz, also used as the temporal stability grid.
- `f₀`: initial frequencies in Hz; omitted to use the initialization selected by `options`.
- `H₀`: solver responses at `f₀`, in the same order.

# Keywords
- `options::GreedyLoewnerConfig = GreedyLoewnerConfig()`: initialization, validation, threading and stopping parameters, including optional temporal stability.

# Returns
- `SweepResult`: final barycentric model, validation history and stopping reason. The model stores all support nodes and corresponding samples.

# Reference
[Pradovera, *Toward a certified greedy Loewner framework with minimal sampling*](https://doi.org/10.1007/s10444-023-10091-7).

"""
function barycentric_sweep(solver::F, candidates; options::GreedyLoewnerConfig=GreedyLoewnerConfig()) where F
    fmin, fmax = extrema(candidates)
    f₀ = options.adaptive ? [fmin, fmax] : semi_adaptive_freqs(fmin, fmax, max(2, options.n0))
    H₀ = map(solver, f₀)
    return barycentric_sweep(solver, candidates, f₀, H₀; options)
end

function barycentric_sweep(solver::F, candidates, f₀, H₀; options::GreedyLoewnerConfig=GreedyLoewnerConfig()) where F
    length(f₀) == length(H₀) || throw(ArgumentError("f₀ and H₀ must have same length"))
    if options.relative_convergence
        isfinite(options.stability_tol) && options.stability_tol >= 0 || throw(ArgumentError("invalid stability tolerance"))
        isfinite(options.stability_floor) && options.stability_floor > 0 || throw(ArgumentError("stability_floor must be finite and positive"))
    end
    
    f₀ = Float64.(collect(f₀))
    candidates = collect(candidates)
    sampled = Set(f₀)
    model = BarycentricModel(s_transform.(f₀), zeros(ComplexF64, length(f₀)), [ComplexF64.(H) for H in H₀], options.denom_tol)
    L = build_barycentric_loewner(f₀, model.values)
    model.weights .= barycentric_weights(L)
    indicators = zeros(length(candidates))
    history = SweepIteration[]
    memory = 0
    stable = true
    previous = options.relative_convergence ? map(model, candidates) : empty(model.values)

    while memory < options.memory && length(model.supports) < options.max_samples && any(f -> f ∉ sampled, candidates)
        compute_indicators!(indicators, model, candidates, options)
        idx = argmax(Base.Fix1(getindex, indicators), (i for i in eachindex(candidates) if candidates[i] ∉ sampled))
        f_new = candidates[idx]
        H_new = solver(f_new)
        error_act = norm(model(f_new) - H_new) / (norm(H_new) + options.delta)
        push!(history, SweepIteration(f_new, indicators[idx], error_act))

        L = append_loewner(L, model.supports, model.values, f_new, H_new)
        push!(model.supports, s_transform(f_new))
        push!(model.values, H_new)
        resize!(model.weights, length(model.supports))
        model.weights .= barycentric_weights(L)
        push!(sampled, f_new)

        if options.relative_convergence
            current = map(model, candidates)
            delta = approximation_convergence(current, previous; floor=options.stability_floor)
            stable = delta < options.stability_tol
            previous = current
        end
        memory = error_act < options.tol && stable ? memory + 1 : 0
    end

    if memory >= options.memory
        termination = :tolerance
    elseif length(model.supports) >= options.max_samples
        termination = :max_samples
    else
        termination = :max_iterations
    end

    return SweepResult(model, history, termination)
end
