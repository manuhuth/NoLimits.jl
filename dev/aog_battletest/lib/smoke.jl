# Quick compile/smoke check of the extension without fixtures (debug aid).
#   julia --project=envs/new lib/smoke.jl
using NoLimits, DataFrames, Distributions, CairoMakie, Random
import AlgebraOfGraphics
ext = Base.get_extension(NoLimits, :NoLimitsMakieExt)
println("ext loaded: ", ext !== nothing)
model = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.3)
        ω = RealNumber(0.3, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η + b * t, σ)
    end
end
rng = Xoshiro(1)
df = DataFrame(ID = repeat(1:5, inner = 4), t = repeat([0.0, 1.0, 2.0, 3.0], 5))
df.y = 1 .+ 0.3 .* df.t .+ repeat(randn(rng, 5) .* 0.3, inner = 4) .+ 0.1 .* randn(rng, 20)
dm = DataModel(model, df; primary_id = :ID, time_col = :t)
res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 5)))
out = joinpath(@__DIR__, "..", "out", "smoke"); mkpath(out)
for (name, f) in [
        "fits" => () -> plot_fits(res),
        "fits_density" => () -> plot_fits(res; plot_density = true, ncols = 2),
        "fits_unshared" => () -> plot_fits(res; shared_x_axis = false, shared_y_axis = false),
        "data" => () -> plot_data(res),
        "data_dm" => () -> plot_data(dm; ncols = 2),
        "fits_dm" => () -> plot_fits(dm),
        "dvpred_legacy" => () -> plot_dv_pred(res),
    ]
    try
        t = @elapsed fig = f()
        save(joinpath(out, name * ".png"), fig)
        println("OK $(name) $(round(t; digits = 2)) s")
    catch err
        println("ERR $(name): ")
        showerror(stdout, err, catch_backtrace()); println()
    end
end
