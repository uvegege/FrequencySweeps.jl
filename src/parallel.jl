function with_BLAS_num_threads(f::F, n::Integer=1) where F
    n > 0 || throw(ArgumentError("BLAS number of threads must be positive"))
    previous = BLAS.get_num_threads()
    previous == n && return f()
    try
        BLAS.set_num_threads(n)
        return f()
    finally
        BLAS.set_num_threads(previous)
    end
end

function foreach_chunk(f::F, collection; nthreads::Integer) where F
    nthreads > 0 || throw(ArgumentError("nthreads must be positive"))
    isempty(collection) && return nothing
    if nthreads == 1
        f(eachindex(collection))
    else
        @sync for indices in index_chunks(collection; n=min(nthreads, length(collection)))
            Threads.@spawn f(indices)
        end
    end
    return nothing
end
