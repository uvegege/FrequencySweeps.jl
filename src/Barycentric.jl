"""
    BarycentricOptions(; kwargs...)

Options for the greedy barycentric frequency sweep.

# Fields
- `tol::Float64 = 1e-3`: relative solver validation tolerance.
- `delta::Float64 = 1e-8`: denominator regularization for solver validation errors.
- `denom_tol::Float64 = sqrt(eps(Float64))`: absolute distance in the complex frequency plane used to recognize an existing support.
- `max_samples::Int = 100`: total number of candidates, including initial samples.
- `memory::Int = 3`: consecutive successful validations required to stop.
- `relative_convergence::Bool = false`: additionally require stability after each model update on the candidate grid.
- `stability_tol::Float64 = tol`: tolerance for the maximum pointwise relative response change.
- `stability_floor::Float64 = eps(Float64)`: positive lower bound on the response norm used to normalize stability.
- `adaptive::Bool = true`: start from band endpoints when true. semiadaptive initial frequencies.
- `n0::Int = 10`: initial sample count for semiadaptive initialization, with a minimum of two.
- `nthreads::Int = 1`: maximum number of tasks for candidate indicator evaluation.
"""
Base.@kwdef struct BarycentricOptions
    tol::Float64 = 1e-3
    delta::Float64 = 1e-8
    denom_tol::Float64 = sqrt(eps(Float64))
    max_samples::Int = 100
    memory::Int = 3
    relative_convergence::Bool = false
    stability_tol::Float64 = tol
    stability_floor::Float64 = eps(Float64)
    adaptive::Bool = true
    n0::Int = 10
    nthreads::Int = 1
end

"""
    BarycentricModel{T, V}
    BarycentricModel(supports, weights, values, denom_tol)

Rational interpolant with barycentric weights shared by all response entries. 
Call `model(f)` or `model(frequencies)` with frequencies in Hz. Responses are
scalars or matrices according to the stored sample values.

# Fields
- `supports::Vector{Complex{T}}`: interpolation nodes in the complex frequency plane, `s = 2π * im * f`.
- `weights::Vector{Complex{T}}`: barycentric weights associated with the supports.
- `values::Vector{V}`: scalar or matrix responses at the supports.
- `denom_tol::T`: absolute complex-frequency distance within which evaluation returns the stored sample.
"""
struct BarycentricModel{T <: AbstractFloat, V}
    supports::Vector{Complex{T}}
    weights::Vector{Complex{T}}
    values::Vector{V}
    denom_tol::T
end

function (model::BarycentricModel{T})(f::Real) where T
    s = s_transform(f)
    support = findfirst(z -> abs(s - z) <= model.denom_tol, model.supports)
    if support !== nothing
        return copy(model.values[support])
    end
    num = zero(model.values[1])
    den = zero(Complex{T})
    for j in eachindex(model.supports)
        w = model.weights[j] / (s - model.supports[j])
        den += w
        if num isa Number #TODO: escribir de otra forma?
            num += w * model.values[j]
        else
            num .+= w .* model.values[j]
        end
    end
    return num / den
end

(model::BarycentricModel)(frequencies::AbstractVector) = map(model, frequencies)

# permitir SISO/MIMO
response_entries(H::Number) = H
response_entries(H::AbstractArray) = vec(H)

# pos/neg partition of sample data
function partition_loewner(frequencies::AbstractVector{T}, samples) where T

    n = length(frequencies)
    sa = Vector{Complex{T}}(undef, n)
    sb = Vector{Complex{T}}(undef, n)
    Ha = Vector{eltype(samples)}(undef, n)
    Hb = similar(Ha)

    for k in eachindex(frequencies)
        s = s_transform(frequencies[k])
        sb[k] = s
        Hb[k] = samples[k]
        sa[k] = conj(s)
        Ha[k] = conj(samples[k])
    end

    return sa, sb, Ha, Hb
end

function build_loewner(sa::AbstractVector{T}, sb, Ha, Hb) where T
    d = length(Hb[1])
    L = zeros(T, length(sa)*d, length(sb))
    for i in eachindex(sa)
        rows = (i-1)*d + 1:i*d
        for j in eachindex(sb)
            L[rows, j] .= response_entries((Ha[i] - Hb[j]) / (sa[i] - sb[j])) # vec for MIMO
        end
    end
    return L
end

function build_barycentric_loewner(frequencies::AbstractVector, samples)
    sa, sb, Ha, Hb = partition_loewner(frequencies, samples)
    return build_loewner(sa, sb, Ha, Hb)
end

function append_loewner(L, supports, values, f::Real, H)
    n = length(supports)
    d = length(H)
    s = s_transform(f)
    sa_new = conj(s)
    H_new = conj(H)

    Lnew = zeros(eltype(L), (n+1)*d, n+1)
    Lnew[1:n*d, 1:n] .= L

    for i in eachindex(supports)
        rows = (i-1)*d + 1:i*d
        Lnew[rows, n+1] .= response_entries((conj(values[i]) - H) / (conj(supports[i]) - s)) # vec for MIMO
    end

    rows = n*d + 1:(n+1)*d
    for j in eachindex(supports)
        Lnew[rows, j] .= response_entries((H_new - values[j]) / (sa_new - supports[j])) # vec for MIMO
    end
    Lnew[rows, n+1] .= response_entries((H_new - H) / (sa_new - s)) # vec for MIMO

    return Lnew
end

@inline s_transform(f) = im * 2 * pi * f
@inline f_transform(s) = imag(s)/(2* π)

function barycentric_weights(L::AbstractMatrix{Complex{T}}) where T
    return svd(qr(L).R).V[:, end]
end

# Q(z) = sum(q_j/(z-z_j)), Eq(5).
function barycentric_denominator(model::BarycentricModel{T}, f::Real) where T
    s = s_transform(f)
    support = findfirst(z -> abs(s - z) <= model.denom_tol, model.supports)
    if support !== nothing
        return Inf
    end
    den = zero(Complex{T})
    for j in eachindex(model.supports)
        den += model.weights[j] / (s - model.supports[j])
    end
    return den
end

function error_indicator(model::BarycentricModel, f::Real)
    return inv(abs(barycentric_denominator(model, f)))
end

function compute_indicators!(indicators, model, candidates, options::BarycentricOptions)
    foreach_chunk(candidates; nthreads=options.nthreads) do indices
        for i in indices
            indicators[i] = inv(abs(barycentric_denominator(model, candidates[i])))
        end
    end
    return indicators
end

