module FrequencySweeps

    using LinearAlgebra
    using ChunkSplitters: index_chunks

    include("parallel.jl")
    include("common.jl")
    include("Loewner.jl")
    include("Barycentric.jl")
    include("sweep.jl")

    export sweep
    export barycentric_sweep
    export SweepOptions
    export SweepResult
    export BarycentricModel
    export DescriptorModel
    export BarycentricOptions
    export semi_adaptive_freqs

end
