"""
    SweepIteration

Record of one Sweep Iteration during a frequency sweep.

# Fields
- `frequency::Float64`: frequency evaluated by the solver, in Hz.
- `indicator::Float64`: selection indicator at the chosen frequency.
- `error::Float64`: relative prediction error against the solver response, before incorporating it.
"""
struct SweepIteration
    frequency::Float64
    indicator::Float64
    error::Float64
end

"""
    SweepResult{M}

Final approximation and validation history returned by a frequency sweep.
Evaluate the approximation with `result.model(f)` or `result.model(frequencies)`.

# Fields
- `model::M`: final descriptor or barycentric approximation.
- `iterations::Vector{SweepIteration}`: validation records, excluding initial samples.
- `termination::Symbol`: stopping reason: `:tolerance`, `:max_samples`, or `:max_iterations` (candidate exhaustion).
"""
struct SweepResult{M}
    model::M
    iterations::Vector{SweepIteration}
    termination::Symbol
end

"""
    semi_adaptive_freqs(fmin, fmax, n₀)

Generate initial frequencies using logarithmic spacing of the semi adaptive sweep.

# Arguments
- `fmin`: lower frequency bound in Hz.
- `fmax`: upper frequency bound in Hz.
- `n₀`: number of initial frequencies.

# Returns
- `AbstractVector`: `n₀` frequencies in Hz between `fmin` and `fmax`.
"""
semi_adaptive_freqs(fmin, fmax, n₀) = 2 * fmax .+ fmin .- 10.0 .^ range(log10(2 * fmax), log10(fmax + fmin), n₀)

magnitude(A::AbstractMatrix) = opnorm(A, 2)
magnitude(x) = norm(x)

function relative_error!(difference, reference; floor=0.0)
    change = difference isa AbstractMatrix ? maximum(svdvals!(difference)) : norm(difference)
    return iszero(change) ? zero(change) : change / max(magnitude(reference), floor)
end

relative_change(current, previous; floor=0.0) = relative_error!(current - previous, current; floor)

function approximation_convergence(current::F, previous::G, frequencies; floor=eps()) where {F, G}
    return maximum(relative_change(current(f), previous(f); floor) for f in frequencies)
end

function approximation_convergence(current::AbstractVector, previous::AbstractVector; floor=eps())
    return maximum(relative_change(current[i], previous[i]; floor) for i in eachindex(current, previous))
end

function approximation_errors(model::F, frequencies, samples; floor=eps()) where F
    absolute = Vector{Float64}(undef, length(frequencies))
    relative = similar(absolute)
    reference_max = error_sq = reference_sq = 0.0
    for i in eachindex(frequencies, samples)
        reference = samples[i]
        difference = model(frequencies[i]) - reference
        scale = magnitude(reference)
        absolute[i] = magnitude(difference)
        relative[i] = absolute[i] / max(scale, floor)
        reference_max = max(reference_max, scale)
        error_sq += sum(abs2, difference)
        reference_sq += sum(abs2, reference)
    end
    global_relative = maximum(absolute) / max(reference_max, floor)
    frobenius_relative = sqrt(error_sq / max(reference_sq, floor^2))
    return absolute, relative, global_relative, frobenius_relative
end

# frequency selection
distance(f, sampled) = minimum(abs(f - fi) for fi in sampled)

function weighted_pseudoerrors!(pseudoerrors, candidates, sampled, α)
    distances = [distance(f, sampled) for f in candidates]
    dmax = maximum(distances)
    @. pseudoerrors *= (distances / dmax)^α
    return pseudoerrors
end
