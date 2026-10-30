@testset "Stopping and solver calls" begin
    candidates = collect(range(0.1, 2.0; length=31))
    truth(f) = 1 / (1 + 2π * im * f)
    for barycentric in (false, true), guard in (false, true)
        queries = Float64[]
        solver(f) = (push!(queries, f); truth(f))
        if barycentric
            options = BarycentricOptions(; memory=2, max_samples=8,
                relative_convergence=guard, stability_tol=0.0)
            result = barycentric_sweep(solver, candidates; options)
            sample_count = length(result.model.supports)
        else
            options = SweepOptions(; adaptive=true, memory=2, max_samples=8,
                relative_convergence=guard, stability_tol=0.0)
            result, data = sweep(solver, candidates; options)
            sample_count = length(data.frequencies)
        end
        @test result.termination == (guard ? :max_samples : :tolerance)
        @test length(queries) == sample_count == length(result.iterations) + 2
        @test length(unique(queries)) == length(queries)
        @test guard ? sample_count == 8 : all(step.error < options.tol for step in result.iterations[end-1:end])
    end
end

@testset "Distance weighting and parallel evaluation" begin
    candidates = collect(range(0.1, 2.0; length=41))
    solver(f) = 1 / (1 + 2π * im * f)
    frequencies = Float64[]
    previous_blas_threads = BLAS.get_num_threads()
    for α in (0.0, 0.5)
        serial_options = SweepOptions(; adaptive=true, relative_convergence=false,
            max_samples=3, memory=3, distance_exponent=α, nthreads=1)
        parallel_options = SweepOptions(; adaptive=true, relative_convergence=false,
            max_samples=3, memory=3, distance_exponent=α, nthreads=4)
        serial, serial_data = sweep(solver, candidates; options=serial_options)
        parallel, parallel_data = sweep(solver, candidates; options=parallel_options)
        @test serial.termination == parallel.termination == :max_samples
        @test serial_data.frequencies == parallel_data.frequencies
        @test serial.model(candidates) ≈ parallel.model(candidates)
        @test BLAS.get_num_threads() == previous_blas_threads
        push!(frequencies, only(serial.iterations).frequency)
    end
    @test frequencies[1] != frequencies[2]
    serial = barycentric_sweep(solver, candidates; options=BarycentricOptions(; nthreads=1))
    parallel = barycentric_sweep(solver, candidates; options=BarycentricOptions(; nthreads=4))
    @test serial.termination == parallel.termination
    @test serial.model.supports == parallel.model.supports
    @test serial.model(candidates) ≈ parallel.model(candidates)
    @test BLAS.get_num_threads() == previous_blas_threads
end
