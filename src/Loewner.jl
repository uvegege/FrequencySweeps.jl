"""
    SweepOptions(; kwargs...)

Options for the Loewner frequency sweep.

# Fields
- `Δf::Float64 = 1e-5`: frequency shift in Hz applied to the second model when computing pseudoerrors.
- `q1::Int = 8`: singular value sum threshold exponent for the first reduced model.
- `q2::Int = 12`: singular value sum threshold exponent for the second reduced model; both exponents must lie in `6:12`.
- `adaptive::Bool = false`: start from band endpoints when true; otherwise use semi adaptive initial frequencies.
- `tol::Float64 = 10^(-60 / 20)`: relative solver validation tolerance.
- `n0::Int = 5`: initial sample count for semi adaptive initialization, with a minimum of two.
- `max_samples::Int = 100`: total sample candidates, including initial samples.
- `memory::Int = 3`: consecutive successful validations required to stop.
- `distance_exponent::Float64 = 0.0`: exponent of normalized nearest-sample distance used to weight pseudoerrors; zero disables weighting. Try 0.5 if you get premature stops.
- `relative_convergence::Bool = true`: additionally require stability between successive model responses on the candidate grid.
- `stability_tol::Float64 = tol`: tolerance for the maximum pointwise relative response change.
- `stability_floor::Float64 = eps(Float64)`: positive lower bound on the response norm used to normalize stability.
- `nthreads::Int = 1`: maximum number of tasks for candidate evaluation.
- `blas_threads::Int = 1`: temporary BLAS number of thread during pseudoerror evaluation; the previous number of threads is restored afterward.
"""
@kwdef struct SweepOptions
    Δf::Float64 = 1e-5
    q1::Int = 8
    q2::Int = 12
    adaptive::Bool = false
    tol::Float64 = 10^(-60 / 20)
    n0::Int = 5
    max_samples::Int = 100
    memory::Int = 3
    distance_exponent::Float64 = 0.0
    relative_convergence::Bool = true
    stability_tol::Float64 = tol
    stability_floor::Float64 = eps(Float64)
    nthreads::Int = 1
    blas_threads::Int = 1
end

struct LoewnerData{T1, T2}
    frequencies::Vector{T1}
    samples::Vector{T2}
end

"""
    DescriptorModel{T, TD}
    DescriptorModel(E, A, B, C, D)

Descriptor realization `H(f) = C * ((2π * im * f * E - A) \\ B) + D`.
Call `model(f)` for a frequency in Hz or `model(frequencies)` for a vector of
frequencies. A scalar direct term produces scalar SISO responses; a matrix
direct term produces matrix responses, including for a 1×1 system.

# Fields
- `E::Matrix{T}`: descriptor matrix.
- `A::Matrix{T}`: state matrix.
- `B::Matrix{T}`: input matrix.
- `C::Matrix{T}`: output matrix.
- `D::TD`: scalar or matrix direct term, independent of frequency.
"""
struct DescriptorModel{T, TD}
    E::Matrix{T}
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    D::TD
end


direct_term(H, D::Number) = only(H) + D
direct_term(H, D::AbstractMatrix) = H + D

function direct_term!(H, D::Number)
    H .+= D
    return only(H)
end

direct_term!(H, D::AbstractMatrix) = H .+= D

function (model::DescriptorModel)(f::Real)
    s = im * 2π * f
    H = model.C * ((s * model.E - model.A) \ model.B)
    return direct_term(H, model.D)
end

function (model::DescriptorModel)(f::AbstractVector)
    T = promote_type(eltype(model.E), ComplexF64)
    M = similar(model.E, T)
    X = similar(model.B, T)
    H = Matrix{T}(undef, size(model.C, 1), size(model.B, 2))
    return map(f) do fᵢ
        reduced_model!(H, M, X, model.E, model.A, model.B, model.C, im * 2π * fᵢ)
        direct_term(H, model.D)
    end
end

function loewner_partition_evenodd(f, S)
    n = length(f)
    n == length(S) || throw(ArgumentError("f and S must have same length"))
    any(iszero, f) && throw(ArgumentError("f contains 0. Use fmin > 0 to avoid s == conj(s)."))

    sa = Vector{ComplexF64}(undef, n)
    sb = Vector{ComplexF64}(undef, n)
    Ha = Vector{eltype(S)}(undef, n)
    Hb = similar(Ha)

    n_even = iseven(n) ? n : n - 1  

    for i in 1:2:n_even
        si = im * 2π * f[i]
        k = (i + 1) ÷ 2
        sb[2*k-1] = si
        sb[2*k] = conj(si)
        Hb[2*k-1] = S[i]
        Hb[2*k] = conj(S[i])
    end
    for i in 2:2:n_even
        si = im * 2π * f[i]
        k = i ÷ 2
        sa[2*k-1] = si
        sa[2*k] = conj(si)
        Ha[2*k-1] = S[i]
        Ha[2*k] = conj(S[i])
    end

    # positivo-negativo
    if isodd(n)
        sn = im * 2π * f[n]
        sb[n] = sn
        Hb[n] = S[n]
        sa[n] = conj(sn)
        Ha[n] = conj(S[n])
    end

    return sa, sb, Ha, Hb
end

# MFTI
function loewner_mfti(Ha, Hb, sa::AbstractVector{Complex{T}}, sb::AbstractVector{Complex{T}}) where T
    p = size(Ha[1], 1)
    m = size(Ha[1], 2)
    L = Matrix{Complex{T}}(undef, length(sb) * p, length(sa) * m)
    σL = similar(L)
    for i in eachindex(Hb)
        rows = (i-1)*p+1:i*p
        Hbi = Hb[i]
        sbi = sb[i]
        for j in eachindex(Ha)
            cols = (j-1)*m+1:j*m
            Haj = Ha[j]
            saj = sa[j]
            L[rows, cols] .= (Haj - Hbi) / (saj - sbi)
            σL[rows, cols] .= (saj * Haj - sbi * Hbi) / (saj - sbi)
        end
    end
    return L, σL
end

function vw_mfti(Ha, Hb)
    p = size(Ha[1], 1)
    m = size(Ha[1], 2)
    W = Matrix{ComplexF64}(undef, p, length(Ha) * m)
    V = Matrix{ComplexF64}(undef, length(Hb) * p, m)
    for j in eachindex(Ha, Hb)
        cols = (j-1)*m+1:j*m
        rows = (j-1)*p+1:j*p
        W[:, cols] .= Ha[j]
        V[rows, :] .= Hb[j]
    end
    return V, W
end

# Loewner matrix order
# Equation 19.
function loewner_order(σ::AbstractVector, q::Int=8)
    6 <= q <= 12 || throw(ArgumentError("loewner_order: q must satisfy 6 ≤ q ≤ 12"))
    total = sum(σ)
    threshold = total * (1 - 10.0^(-q))
    accum = zero(eltype(σ))
    for (r, σv) in enumerate(σ)
        accum += σv
        accum >= threshold && return r
    end
    return length(σ)
end

# Sin usar, debug
function loewner_rank(σ; rtol=1e-10, atol=0.0)
    rtol >= 0 && atol >= 0 || throw(ArgumentError("negative tolerance"))
    isempty(σ) && return 0
    threshold = max(atol, rtol * first(σ))
    return count(>(threshold), σ)
end

# Eq. 16
@inline function svd_matrix_pencil(x, L, σL)
    return svd(x * L - σL)
end

# Eq. 17
function loewner_reduce(L, σL, B, C, F, r)
    Xr = @view F.V[:, 1:r] # n*p × r
    Yr = @view F.U[:, 1:r] # n*p × r
    Er = -(Yr' * L * Xr)
    Ar = -(Yr' * σL * Xr)
    Br = Yr' * B
    Cr = C * Xr
    return Er, Ar, Br, Cr
end

function reduced_model!(H, M, X, E, A, B, C, s)
    M .= @. s * E - A
    F = lu!(M)
    ldiv!(X, F, B)
    mul!(H, C, X)
    return H
end

function loewner_pseudo_errors!(errors, candidates, model1::DescriptorModel, model2::DescriptorModel, Δf, options::SweepOptions)
    return loewner_pseudo_errors!(errors, candidates, model1, model2, Δf, options.nthreads; blas_threads=options.blas_threads)
end

function loewner_pseudo_errors!(errors, candidates, model1::DescriptorModel, model2::DescriptorModel, Δf, nthreads::Integer; blas_threads=1)
    return loewner_pseudo_errors!(errors, (), candidates, model1, model2, Δf, nthreads; blas_threads)
end

function loewner_pseudo_errors!(errors, results, candidates, model1::DescriptorModel, model2::DescriptorModel, Δf, nthreads::Integer; blas_threads=1)
    axes(errors) == axes(candidates) || throw(DimensionMismatch("errors and candidates"))
    with_BLAS_num_threads(blas_threads) do
        foreach_chunk(candidates; nthreads) do indices
            T1 = promote_type(eltype(model1.E), ComplexF64)
            T2 = promote_type(eltype(model2.E), ComplexF64)
            M1 = similar(model1.E, T1)
            M2 = similar(model2.E, T2)
            X1 = similar(model1.B, T1)
            X2 = similar(model2.B, T2)
            H1 = Matrix{ComplexF64}(undef, size(model1.C, 1), size(model1.B, 2))
            H2 = similar(H1)
            difference = similar(H1)
            for i in indices
                f = candidates[i]
                reduced_model!(H1, M1, X1, model1.E, model1.A, model1.B, model1.C, im * 2π * f)
                if !isempty(results)
                    results[i] = direct_term(H1, model1.D)
                end
                reduced_model!(H2, M2, X2, model2.E, model2.A, model2.B, model2.C, im * 2π * (f + Δf))
                @. difference = H2 - H1
                denominator = magnitude(H1)
                errors[i] = denominator == 0 ? Inf : maximum(svdvals!(difference)) / denominator
            end
        end
    end
    return errors
end

function loewner_response!(H, model, f)
    T = promote_type(eltype(model.E), ComplexF64)
    M = similar(model.E, T)
    X = similar(model.B, T)
    reduced_model!(H, M, X, model.E, model.A, model.B, model.C, im * 2π * f)
    return direct_term!(H, model.D)
end

function loewner_models(L, σL, V, W, frequencies, D, options)
    x = im * 2π * frequencies[cld(length(frequencies), 2)]
    F = svd_matrix_pencil(x, L, σL)

    r1 = loewner_order(F.S, options.q1)
    E1, A1, B1, C1 = loewner_reduce(L, σL, V, W, F, r1)
    first_model = DescriptorModel(E1, A1, B1, C1, D)

    r2 = loewner_order(F.S, options.q2)
    E2, A2, B2, C2 = loewner_reduce(L, σL, V, W, F, r2)
    second_model = DescriptorModel(E2, A2, B2, C2, D)

    return first_model, second_model
end

function loewner_model(L, σL, V, W, frequencies, D, options)
    x = im * 2π * frequencies[cld(length(frequencies), 2)]
    F = svd_matrix_pencil(x, L, σL)
    r = loewner_order(F.S, options.q1)
    E, A, B, C = loewner_reduce(L, σL, V, W, F, r)
    return DescriptorModel(E, A, B, C, D)
end

