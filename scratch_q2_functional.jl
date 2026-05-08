using NoLimits, DataFrames, Distributions, Random, OrdinaryDiffEq

# Simple model: a, σ are Q1 (in @formulas), σ_η is Q2 (only in RE distribution)
model = @Model begin
    @fixedEffects begin
        a   = RealNumber(1.0)
        σ   = RealNumber(0.5, scale=:log)
        σ_η = RealNumber(1.0, scale=:log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, σ_η); column=:ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end

# Verify detection
free_names = NoLimits.get_names(model.fixed.fixed)
part = NoLimits._partition_q1_q2_names(model, free_names)
println("Q1 params: ", part.q1)
println("Q2 params: ", part.q2)
@assert Set(part.q1) == Set([:a, :σ])  "Q1 wrong"
@assert Set(part.q2) == Set([:σ_η])    "Q2 wrong"
println("[PASS] Detection correct")

# Build dataset
rng = MersenneTwister(42)
N = 20
df = DataFrame(
    ID = repeat(1:N, inner=5),
    t  = repeat(0.0:1.0:4.0, N),
    y  = randn(rng, 5*N)
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

# Fit with SAEM (Q2 separation should trigger automatically)
println("\n--- SAEM fit (Q2 separation active) ---")
res = fit_model(dm, NoLimits.SAEM(; maxiters=20,
    turing_kwargs=(n_samples=3, n_adapt=3, progress=false),
    progress=false))
params = NoLimits.get_params(res; scale=:untransformed)
println("σ_η = ", params.σ_η, "  (should be ~1.0)")
println("[PASS] SAEM with Q2 separation completed")

# Fit with MCEM (Q2 separation should trigger automatically)
println("\n--- MCEM fit (Q2 separation active) ---")
res_m = fit_model(dm, NoLimits.MCEM(; maxiters=10,
    turing_kwargs=(n_samples=5, n_adapt=3, progress=false),
    progress=false))
params_m = NoLimits.get_params(res_m; scale=:untransformed)
println("σ_η = ", params_m.σ_η, "  (should be ~1.0)")
println("[PASS] MCEM with Q2 separation completed")

println("\nAll checks passed!")
