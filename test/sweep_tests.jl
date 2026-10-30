@testset "Frequency sweeps" begin
    candidates = collect(range(0.1, 2.0; length=41))
    validation = collect(range(0.105, 1.995; length=47))
    initial = collect(extrema(candidates))

    for (p, m) in ((1, 1), (2, 2), (2, 3), (3, 2)), scalar in (false, true)
        scalar && (p, m) != (1, 1) && continue
        C = reshape(collect(1.0:2p), p, 2)
        B = reshape(collect(1.0:2m), 2, m)
        A = Diagonal([-1.0, -4.0])
        D = scalar ? 0.2 : fill(0.2, p, m)
        function solver(f)
            H = C * ((2π * im * f * I - A) \ B)
            return scalar ? only(H) + D : H + D
        end
        samples = map(solver, initial)
        saved_samples = deepcopy(samples)
        saved_D = copy(D)
        for guard in (false, true)
            options = SweepOptions(; adaptive=true, memory=2, max_samples=16,
                tol=1e-7, relative_convergence=guard)
            config = BarycentricOptions(; memory=2, max_samples=16,
                tol=1e-7, relative_convergence=guard)
            result, data = sweep(solver, candidates; D, options)
            barycentric = barycentric_sweep(solver, candidates; options=config)
            seeded, _ = sweep(solver, candidates, initial, samples; D, options)
            seeded_barycentric = barycentric_sweep(solver, candidates, initial, samples; options=config)
            @test samples == saved_samples
            @test D == saved_D
            @test data.samples ≈ map(solver, data.frequencies)
            @test length(unique(data.frequencies)) == length(data.frequencies)
            @test seeded.model(validation) ≈ result.model(validation)
            @test seeded_barycentric.model(validation) ≈ barycentric.model(validation)
            for approximation in (result, barycentric)
                @test approximation.termination == :tolerance
                @test length(approximation.iterations) >= 2
                @test approximation.model(0.7) isa (scalar ? Number : AbstractMatrix)
                @test size(approximation.model(0.7)) == size(D)
                @test approximation.model(validation) ≈ map(approximation.model, validation)
                @test maximum(norm(approximation.model(f) - solver(f)) / norm(solver(f)) for f in validation) < 1e-6
            end
        end
    end
end
