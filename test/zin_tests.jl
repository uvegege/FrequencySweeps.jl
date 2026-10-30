@testset "Rational input impedances Zin1 and Zin2" begin
    Zin1(s) = (2512 * s^2 + 1.866e4 * s + 1.305e5) / (s^3 + 17.42 * s^2 + 417.3 * s + 1305)
    Zin2(s) = (2486 * s^2 + 1.82e4 * s + 1.267e5) / (s^3 + 17.77 * s^2 + 405.8 * s + 1267)

    candidates = 10.0 .^ range(-1, 2; length=201) ./ (2π)
    validation = [(candidates[i] + candidates[i+1]) / 2 for i in 1:length(candidates)-1]
    options = SweepOptions(; adaptive=true, tol=1e-6, stability_tol=1e-6,
        memory=3, max_samples=30, distance_exponent=0.5)
    config = BarycentricOptions(; tol=1e-6, memory=3, max_samples=30)

    for Zin in (Zin1, Zin2)
        solver(f) = Zin(2π * im * f)
        loewner, data = sweep(solver, candidates; options)
        barycentric = barycentric_sweep(solver, candidates; options=config)
        @test data.samples ≈ solver.(data.frequencies)
        for result in (loewner, barycentric)
            @test result.termination == :tolerance
            @test maximum(abs(result.model(f) - solver(f)) / abs(solver(f)) for f in validation) < 1e-6
        end
    end
end

@testset "Quarter-wave transformer" begin
    function impedance(f)
        s = 2π * im * f
        return 70.7 * (170.7 * exp(s / 4) + 29.3 * exp(-s / 4)) / (170.7 * exp(s / 4) - 29.3 * exp(-s / 4))
    end
    reflection(Z) = (Z - 50) / (Z + 50)
    @test impedance(0.0) ≈ 100
    @test impedance(1.0) ≈ 70.7^2 / 100
    @test abs(reflection(impedance(1.0))) < 2e-4

    candidates = collect(range(0.05, 2.0; length=201))
    validation = [(candidates[i] + candidates[i+1]) / 2 for i in 1:length(candidates)-1]
    options = SweepOptions(; adaptive=true, tol=1e-6, stability_tol=1e-6,
        memory=3, max_samples=30, distance_exponent=0.5)
    config = BarycentricOptions(; tol=1e-6, memory=3, max_samples=30)
    loewner, data = sweep(impedance, candidates; options)
    barycentric = barycentric_sweep(impedance, candidates; options=config)
    @test data.samples ≈ impedance.(data.frequencies)
    for result in (loewner, barycentric)
        @test result.termination == :tolerance
        @test maximum(abs(result.model(f) - impedance(f)) / abs(impedance(f)) for f in validation) < 1e-3
        @test maximum(abs(reflection(result.model(f)) - reflection(impedance(f))) for f in validation) < 1e-3
    end
end
