# Benchmark normalizing flow logpdf performance and allocations for vector and SVector inputs.
using NoLimits
using LinearAlgebra
using StaticArrays

function run_flow_microbench(; nsteps=10_000, dim=4, layers=4)
    flow = NormalizingPlanarFlow(dim, layers)
    x = randn(dim)
    xs = SVector{dim}(x)

    # Warmup
    for _ in 1:100
        logpdf(flow, x)
        logpdf(flow, xs)
    end
    GC.gc()

    alloc_vec = @allocated begin
        for _ in 1:nsteps
            logpdf(flow, x)
        end
    end
    elapsed_vec = @elapsed begin
        for _ in 1:nsteps
            logpdf(flow, x)
        end
    end

    alloc_svec = @allocated begin
        for _ in 1:nsteps
            logpdf(flow, xs)
        end
    end
    elapsed_svec = @elapsed begin
        for _ in 1:nsteps
            logpdf(flow, xs)
        end
    end

    @info "NPF logpdf microbench (Vector)" nsteps dim layers elapsed_sec=elapsed_vec alloc_bytes=alloc_vec alloc_per_call=alloc_vec / nsteps
    @info "NPF logpdf microbench (SVector)" nsteps dim layers elapsed_sec=elapsed_svec alloc_bytes=alloc_svec alloc_per_call=alloc_svec / nsteps
    return (elapsed_vec=elapsed_vec, alloc_vec=alloc_vec,
            elapsed_svec=elapsed_svec, alloc_svec=alloc_svec)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_flow_microbench()
end
