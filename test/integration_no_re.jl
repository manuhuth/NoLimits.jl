using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using Random
using Turing
using MCMCChains
using OptimizationBBO
using OptimizationOptimisers
using OptimizationOptimJL

# ─────────────────────────────────────────────────────────────────────────────
# Integration file: tests for models WITHOUT random effects.
# Absorbed from: estimation_mle_tests.jl, estimation_map_tests.jl,
#   estimation_vi_tests.jl, accessors_tests.jl (no-RE parts),
#   serialization_tests.jl (no-RE parts).
#
# Shared fixtures at the top build the model and run fits ONCE; downstream
# testsets reuse the results instead of re-fitting the same model.
# ─────────────────────────────────────────────────────────────────────────────

# ── shared fixtures ───────────────────────────────────────────────────────────

# No-prior model: y ~ Normal(exp(a), σ)  —  used for MLE and as serialization base
const _NRE_DF = DataFrame(ID=[1,1,1], t=[0.0,1.0,2.0], y=[0.1,0.12,0.11])

const _NRE_DM = DataModel(
    @Model(begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale=:log)
        end
        @formulas begin; y ~ Normal(exp(a), σ); end
    end),
    _NRE_DF; primary_id=:ID, time_col=:t)

# With-priors model: same formula but a ~ N(0,1), σ ~ LogNormal
# Used for MAP, MCMC, VI, and serialization round-trips.
const _NRE_DM_P = DataModel(
    @Model(begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log; prior=LogNormal(0.0, 0.5))
        end
        @formulas begin; y ~ Normal(exp(a), σ); end
    end),
    _NRE_DF; primary_id=:ID, time_col=:t)

# Pre-fit once; all basic testsets below reuse these.
const _NRE_RES_MLE  = fit_model(_NRE_DM,   NoLimits.MLE())
const _NRE_RES_MAP  = fit_model(_NRE_DM_P, NoLimits.MAP())
const _NRE_RES_MCMC = fit_model(_NRE_DM_P,
    NoLimits.MCMC(; turing_kwargs=(n_samples=2, n_adapt=2, progress=false));
    rng=MersenneTwister(42))
const _NRE_RES_VI   = fit_model(_NRE_DM_P,
    NoLimits.VI(; turing_kwargs=(max_iter=15, progress=false));
    rng=Random.Xoshiro(1))

# ══════════════════════════════════════════════════════════════════════════════
# MLE
# ══════════════════════════════════════════════════════════════════════════════

@testset "MLE non-ODE" begin
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5, scale=:log)
        end
        @formulas begin; y ~ Normal(softplus(a), σ); end
    end
    df = DataFrame(ID=[1,1], t=[0.0,1.0], y=[1.0,1.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())
    @test res isa FitResult
    @test NoLimits.get_params(res; scale=:untransformed) isa ComponentArray
end

@testset "MLE ODE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
        end
        @covariates begin; t = Covariate(); end
        @DifferentialEquation begin; D(x1) ~ -a * x1^2; end
        @initialDE begin; x1 = 1.0; end
        @formulas begin; y ~ Normal(log1p(x1(t)^2), σ); end
    end
    df = DataFrame(ID=[1,1], t=[0.0,1.0], y=[1.0,1.05])
    dm = DataModel(set_solver_config(model; saveat_mode=:saveat), df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())
    @test res isa FitResult
end

@testset "MLE ODE with parameterized initial state" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            ka = RealNumber(1.0, prior=Normal(1.0, 0.5))
            ke = RealNumber(0.1, prior=Normal(0.1, 0.05))
            V  = RealNumber(20.0, prior=Normal(20.0, 5.0))
            D  = RealNumber(320.0, prior=Normal(320.0, 50.0))
            σ  = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @DifferentialEquation begin
            D(A) ~ -ka * A
            D(C) ~ (ka * A) / V - ke * C
        end
        @initialDE begin; A = D; C = 0.0; end
        @formulas begin; y ~ Normal(C(t), σ); end
    end
    df = DataFrame(ID=[1,1,1,1], t=[0.0,1.0,2.0,3.0], y=[1.0,1.05,0.98,1.02])
    dm = DataModel(set_solver_config(model; saveat_mode=:saveat), df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())
    @test res isa FitResult
end

@testset "MLE rejects random effects" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2); σ = RealNumber(0.3, scale=:log)
        end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(exp(a + η), σ); end
    end
    df = DataFrame(ID=[1,1], t=[0.0,1.0], y=[1.0,1.05])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.MLE())
end

@testset "MLE requires a free fixed effect" begin
    @test_throws ErrorException fit_model(_NRE_DM, NoLimits.MLE(); constants=(a=0.1, σ=0.5))
end

@testset "MLE fixed vector parameters" begin
    model = @Model begin
        @fixedEffects begin
            β = RealVector([0.2, -0.1])
            σ = RealNumber(0.3, scale=:log)
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin
            μ = exp(β[1] * z + β[2] * z^2)
            y ~ Normal(μ, σ)
        end
    end
    df = DataFrame(ID=[1,1,1,1], t=[0.0,1.0,2.0,3.0],
                   z=[0.1,0.2,0.15,0.3], y=[1.0,1.05,1.02,1.08])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())
    @test res isa FitResult
end

@testset "MLE respects bounds (σ lower bound)" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5; lower=0.3, scale=:identity)
        end
        @formulas begin; y ~ Normal(exp(a), σ); end
    end
    df = DataFrame(ID=[1,1,1], t=[0.0,1.0,2.0], y=[0.1,0.12,0.11])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())
    @test NoLimits.get_params(res; scale=:untransformed).σ >= 0.3
end

@testset "MLE constants" begin
    res = fit_model(_NRE_DM, NoLimits.MLE(); constants=(a=0.0,))
    @test NoLimits.get_params(res; scale=:untransformed).a == 0.0
end

@testset "MLE penalties" begin
    res_no_pen = fit_model(_NRE_DM, NoLimits.MLE())
    res        = fit_model(_NRE_DM, NoLimits.MLE(); penalty=(a=100.0,))
    a0 = NoLimits.get_params(res_no_pen; scale=:untransformed).a
    a  = NoLimits.get_params(res;        scale=:untransformed).a
    @test abs(a) ≤ abs(a0)
end

@testset "MLE penalty mimics Normal prior" begin
    model_prior = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.1; prior=Normal(0.0, 1.0)); end
        @formulas begin; y ~ Normal(exp(a), 2.0); end
    end
    model_penalty = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.1); end
        @formulas begin; y ~ Normal(exp(a), 2.0); end
    end
    df = DataFrame(ID=[1,1,1], t=[0.0,1.0,2.0], y=[0.1,0.12,0.11])
    res_map = fit_model(DataModel(model_prior,   df; primary_id=:ID, time_col=:t), NoLimits.MAP())
    res_pen = fit_model(DataModel(model_penalty, df; primary_id=:ID, time_col=:t), NoLimits.MLE(); penalty=(a=0.5,))
    a_map = NoLimits.get_params(res_map; scale=:untransformed).a
    a_pen = NoLimits.get_params(res_pen; scale=:untransformed).a
    @test isapprox(a_map, a_pen; rtol=1e-4, atol=1e-4)
end

@testset "MLE uses optim_kwargs" begin
    method = NoLimits.MLE(optim_kwargs=(; iterations=1))
    res = fit_model(_NRE_DM, method)
    @test res isa FitResult
    @test hasproperty(res.result.solution.stats, :iterations)
    @test res.result.solution.stats.iterations <= 1
end

@testset "MLE accepts lb-only user bounds" begin
    lb = ComponentArray((; a=-2.0, σ=-3.0))
    res = fit_model(_NRE_DM, NoLimits.MLE(lb=lb))
    @test res isa FitResult
end

@testset "MLE accepts ub-only user bounds" begin
    ub = ComponentArray((; a=2.0, σ=2.0))
    res = fit_model(_NRE_DM, NoLimits.MLE(ub=ub))
    @test res isa FitResult
end

@testset "MLE BBO requires finite bounds on both sides" begin
    lb = ComponentArray((; a=-2.0, σ=-3.0))
    method = NoLimits.MLE(optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
                          lb=lb, optim_kwargs=(; iterations=5))
    err = try fit_model(_NRE_DM, method); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("finite lower and upper bounds", sprint(showerror, err))
end

@testset "MLE optimizer BFGS (Optim)" begin
    res = fit_model(_NRE_DM, NoLimits.MLE(optimizer=BFGS(), optim_kwargs=(;)))
    @test res isa FitResult
end

@testset "MLE optimizer NelderMead (Optim)" begin
    res = fit_model(_NRE_DM, NoLimits.MLE(optimizer=Optim.NelderMead(), optim_kwargs=(;)))
    @test res isa FitResult
end

@testset "MLE optimizer Adam (OptimizationOptimisers)" begin
    res = fit_model(_NRE_DM, NoLimits.MLE(optimizer=OptimizationOptimisers.Adam(0.05),
                                           optim_kwargs=(; maxiters=2)))
    @test res isa FitResult
end

@testset "MLE optimizer BlackBoxOptim (OptimizationBBO)" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1; lower=-2.0, upper=2.0, scale=:identity)
            σ = RealNumber(0.5; lower=0.1, upper=2.0, scale=:identity)
        end
        @formulas begin; y ~ Normal(exp(a), σ); end
    end
    df = DataFrame(ID=[1,1,1], t=[0.0,1.0,2.0], y=[0.1,0.12,0.11])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(
        optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs=(; iterations=5)))
    @test res isa FitResult
end

@testset "MLE non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin; t = Covariate(); z = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.1); b = RealNumber(0.2); end
        @formulas begin; λ = exp(a + b * z); y ~ Poisson(λ); end
    end
    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0],
                   z=[0.0,0.5,1.0,1.5], y=[1,1,2,3])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())
    @test res isa FitResult
    θu = NoLimits.get_params(res; scale=:untransformed)
end

@testset "MLE handles +Inf objective in AD path" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.0); end
        @formulas begin; y ~ Poisson(a); end
    end
    df = DataFrame(ID=[1], t=[0.0], y=[1.0])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    @test res isa FitResult
    @test !isfinite(NoLimits.get_objective(res))
end

# ══════════════════════════════════════════════════════════════════════════════
# MAP
# ══════════════════════════════════════════════════════════════════════════════

@testset "MAP non-ODE" begin
    # Reuse the shared pre-fit result.
    @test _NRE_RES_MAP isa FitResult
end

@testset "MAP ODE" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3; prior=LogNormal(0.0, 0.5))
        end
        @DifferentialEquation begin; D(x1) ~ -a * x1^2; end
        @initialDE begin; x1 = 1.0; end
        @formulas begin; y ~ Normal(log1p(x1(t)^2), σ); end
    end
    df = DataFrame(ID=[1,1], t=[0.0,1.0], y=[1.0,1.05])
    dm = DataModel(set_solver_config(model; saveat_mode=:saveat), df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())
    @test res isa FitResult
end

@testset "MAP requires priors" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.2); σ = RealNumber(0.3); end
        @formulas begin; y ~ Normal(exp(a), σ); end
    end
    df = DataFrame(ID=[1,1], t=[0.0,1.0], y=[1.0,1.05])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.MAP())
end

@testset "MAP respects bounds (σ lower bound)" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5; lower=0.3, scale=:identity, prior=LogNormal(0.0, 0.5))
        end
        @formulas begin; y ~ Normal(exp(a), σ); end
    end
    df = DataFrame(ID=[1,1,1], t=[0.0,1.0,2.0], y=[0.1,0.12,0.11])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())
    @test NoLimits.get_params(res; scale=:untransformed).σ >= 0.3
end

@testset "MAP constants" begin
    res = fit_model(_NRE_DM_P, NoLimits.MAP(); constants=(a=0.0,))
    @test NoLimits.get_params(res; scale=:untransformed).a == 0.0
end

@testset "MAP requires a free fixed effect" begin
    @test_throws ErrorException fit_model(_NRE_DM_P, NoLimits.MAP(); constants=(a=0.1, σ=0.5))
end

@testset "MAP fixed vector parameters" begin
    model = @Model begin
        @fixedEffects begin
            β = RealVector([0.2, -0.1], prior=MvNormal(zeros(2), LinearAlgebra.I))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin
            μ = exp(β[1] * z + β[2] * z^2)
            y ~ Normal(μ, σ)
        end
    end
    df = DataFrame(ID=[1,1,1,1], t=[0.0,1.0,2.0,3.0],
                   z=[0.1,0.2,0.15,0.3], y=[1.0,1.05,1.02,1.08])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())
    @test res isa FitResult
end

@testset "MAP non-normal Bernoulli outcome" begin
    model = @Model begin
        @covariates begin; t = Covariate(); z = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
        end
        @formulas begin; p = logistic(a + b * z); y ~ Bernoulli(p); end
    end
    df = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   z=[0.1,0.2,-0.1,0.0,0.3,0.4], y=[0,1,0,0,1,1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())
    @test res isa FitResult
    θu = NoLimits.get_params(res; scale=:untransformed)
end

@testset "MAP handles +Inf objective in AD path" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.0, prior=Uniform(0.1, 1.0)); end
        @formulas begin; y ~ Poisson(a); end
    end
    df = DataFrame(ID=[1], t=[0.0], y=[1.0])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP(; optim_kwargs=(maxiters=2,)))
    @test res isa FitResult
    @test !isfinite(NoLimits.get_objective(res))
end

# ══════════════════════════════════════════════════════════════════════════════
# VI (fixed-effects only; VI rejects RE models — tested in integration_simple_re.jl)
# ══════════════════════════════════════════════════════════════════════════════

@testset "VI basic (no RE)" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            b = RealNumber(1.0, prior=Normal(0.0, 2.0))
            a = RealNumber(0.2, prior=Uniform(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=Uniform(0.001, 0.5))
        end
        @formulas begin; y ~ Normal(a*t + b, σ); end
    end
    df = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[1.0,1.05,0.98,1.02])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=20, progress=false)); rng=Random.Xoshiro(1))
    @test res isa FitResult
    @test res.result isa NoLimits.VIResult
    @test NoLimits.get_converged(res) isa Bool
    @test length(NoLimits.get_vi_trace(res)) > 0
    @test NoLimits.get_vi_state(res) isa NamedTuple
    @test rand(Random.Xoshiro(2), NoLimits.get_variational_posterior(res)) isa AbstractVector
    draws = NoLimits.sample_posterior(res; n_draws=7, rng=Random.Xoshiro(3))
    @test size(draws, 1) == 7
    @test size(draws, 2) >= 2
    draws_named = NoLimits.sample_posterior(res; n_draws=3, rng=Random.Xoshiro(4), return_names=true)
    @test haskey(draws_named, :draws) && haskey(draws_named, :names)
    @test size(draws_named.draws, 1) == 3
    @test_throws ErrorException NoLimits.get_chain(res)
end

@testset "VI requires priors" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log)          # no prior → error
        end
        @formulas begin; y ~ Normal(exp(a), σ); end
    end
    df = DataFrame(ID=[1,1,1], t=[0.0,1.0,2.0], y=[1.0,1.05,0.98])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.VI())
end

@testset "VI fixed-effects-only rejects all constants" begin
    err = try
        fit_model(_NRE_DM_P, NoLimits.VI(; turing_kwargs=(max_iter=5, progress=false));
                  constants=(a=0.2, σ=0.5))
        nothing
    catch e; e end
    @test err isa ErrorException
    @test occursin("at least one sampled parameter", sprint(showerror, err))
end

@testset "VI rejects penalty terms" begin
    @test_throws ErrorException fit_model(_NRE_DM_P, NoLimits.VI(); penalty=(a=1.0,))
end

# ══════════════════════════════════════════════════════════════════════════════
# Accessors (no-RE methods)
# ══════════════════════════════════════════════════════════════════════════════

@testset "Accessors (fixed effects)" begin
    res = _NRE_RES_MLE
    @test NoLimits.get_params(res; scale=:untransformed) isa ComponentArray
    @test NoLimits.get_converged(res) isa Bool
    @test_throws ErrorException NoLimits.get_chain(res)

    res_nostore = fit_model(_NRE_DM, NoLimits.MLE(; optim_kwargs=(maxiters=2,)); store_data_model=false)
    @test_throws ErrorException NoLimits.get_loglikelihood(res_nostore)

    # MAP accessor
    res_map = _NRE_RES_MAP
    @test_throws ErrorException NoLimits.get_chain(res_map)
end

@testset "Accessors (MCMC)" begin
    res = _NRE_RES_MCMC
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
    @test NoLimits.get_observed(res).y == _NRE_DF.y
    @test NoLimits.get_sampler(res) isa Any
    @test NoLimits.get_n_samples(res) == 2
    @test_throws ErrorException NoLimits.get_loglikelihood(res)
end

# ══════════════════════════════════════════════════════════════════════════════
# Serialization (no-RE methods)
# ══════════════════════════════════════════════════════════════════════════════

@testset "Serialization MLE" begin
    res = _NRE_RES_MLE
    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=_NRE_DM)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_iterations(res) == get_iterations(res2)
    @test get_converged(res) == get_converged(res2)
    @test get_raw(res2) === nothing
    @test get_data_model(res2) === _NRE_DM
    @test get_method(res2) isa NoLimits._SavedFittingMethod
    @test get_method(res2).kind == :mle
end

@testset "Serialization MLE include_data" begin
    res = _NRE_RES_MLE
    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    res2 = load_fit(path; model=_NRE_DM.model)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_data_model(res2) !== nothing
end

@testset "Serialization MLE no dm" begin
    res = _NRE_RES_MLE
    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_data_model(res2) === nothing
end

@testset "Serialization MAP" begin
    res = _NRE_RES_MAP
    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=_NRE_DM_P)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_method(res2) isa NoLimits._SavedFittingMethod
    @test get_method(res2).kind == :map
end

@testset "Serialization MCMC" begin
    # Uses shared result (5 samples) instead of original 50 — round-trip logic unchanged.
    res = _NRE_RES_MCMC
    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=_NRE_DM_P)
    @test get_chain(res).value == get_chain(res2).value
    @test get_n_samples(res) == get_n_samples(res2)
    @test get_sampler(res2) isa NoLimits._SavedSamplerStub
end

@testset "Serialization VI" begin
    res = _NRE_RES_VI
    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=_NRE_DM_P)
    @test get_objective(res) ≈ get_objective(res2)
    @test get_vi_state(res2) === nothing
    s1 = sample_posterior(res;  n_draws=20)
    s2 = sample_posterior(res2; n_draws=20)
    @test size(s1) == size(s2)
    @test get_vi_trace(res2) isa AbstractVector
end

@testset "Serialization Multistart" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 2.0))
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin; t = Covariate(); end
        @formulas begin; y ~ Normal(a, σ); end
    end
    df  = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[1.0,1.1,0.9,1.0])
    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms  = NoLimits.Multistart(dists=(; a=Normal(1.0, 0.2)), n_draws_requested=4, n_draws_used=3)
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)
    @test length(get_multistart_results(res)) == length(get_multistart_results(res2))
    @test get_multistart_best_index(res) == get_multistart_best_index(res2)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test eltype(get_multistart_errors(res2)) == String
end

@testset "Serialization format version check" begin
    import JLD2
    path = tempname() * ".jld2"
    JLD2.jldsave(path; saved=(; format_version=999))
    @test_throws ErrorException load_fit(path)
end

@testset "Cross-session MLE: params + LL + residuals" begin
    # Save the shared MLE result with include_data so a fresh process can load it.
    res = _NRE_RES_MLE
    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    expected_obj  = get_objective(res)
    expected_ll   = get_loglikelihood(_NRE_DM, res)
    expected_nres = nrow(get_residuals(res))

    script = """
    using NoLimits, Distributions, DataFrames
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(exp(a), σ) end
    end
    res = load_fit($(repr(path)); model=model)
    println(get_objective(res))
    println(get_method(res).kind)
    println(get_loglikelihood(res))
    println(nrow(get_residuals(res)))
    p = plot_fits(res)
    println(typeof(p))
    """
    script_path = tempname() * ".jl"
    write(script_path, script)
    out   = readchomp(`$(Base.julia_cmd()) --project=$(pkgdir(NoLimits)) $(script_path)`)
    lines = split(strip(out), '\n'; keepempty=false)
    @test length(lines) >= 5
    @test parse(Float64, strip(lines[end-4])) ≈ expected_obj   atol=1e-10
    @test strip(lines[end-3]) == "mle"
    @test parse(Float64, strip(lines[end-2])) ≈ expected_ll    atol=1e-8
    @test parse(Int,     strip(lines[end-1])) == expected_nres
    @test occursin("Plot", strip(lines[end]))
end
