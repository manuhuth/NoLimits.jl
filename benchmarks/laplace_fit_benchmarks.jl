# Benchmark Laplace fit (serial vs threads).
using NoLimits
using DataFrames
using Random

function _laplace_fit_setup()
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    n_ids = 100
    df = DataFrame(
        ID = repeat(1:n_ids, inner=2),
        t = repeat([0.0, 1.0], n_ids),
        y = randn(2 * n_ids)
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    return dm
end

function run_laplace_fit_benchmarks(; n=3)
    dm = _laplace_fit_setup()
    meth = Laplace(; optim_kwargs=(maxiters=3,))

    alloc_serial = @allocated begin
        for _ in 1:n
            fit_model(dm, meth; serialization=EnsembleSerial())
        end
    end
    elapsed_serial = @elapsed begin
        for _ in 1:n
            fit_model(dm, meth; serialization=EnsembleSerial())
        end
    end

    alloc_threads = @allocated begin
        for _ in 1:n
            fit_model(dm, meth; serialization=EnsembleThreads())
        end
    end
    elapsed_threads = @elapsed begin
        for _ in 1:n
            fit_model(dm, meth; serialization=EnsembleThreads())
        end
    end

    @info "Laplace fit benchmark (serial)" n elapsed_sec=elapsed_serial alloc_bytes=alloc_serial alloc_per_call=alloc_serial / n
    @info "Laplace fit benchmark (threads)" n elapsed_sec=elapsed_threads alloc_bytes=alloc_threads alloc_per_call=alloc_threads / n
    return (serial=(elapsed=elapsed_serial, alloc=alloc_serial),
            threads=(elapsed=elapsed_threads, alloc=alloc_threads))
end

run_laplace_fit_benchmarks(n=50)
