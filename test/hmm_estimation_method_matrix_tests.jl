using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using MCMCChains

function _re_hmm_scalar_discrete_dm()
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p1_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
            p2_r = RealNumber(0.1, prior=Normal(0.0, 1.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            p1 = 0.8 / (1 + exp(-clamp(p1_r + η, -2.0, 2.0))) + 0.1
            p2 = 0.8 / (1 + exp(-clamp(p2_r, -2.0, 2.0))) + 0.1
            P = [0.85 0.15;
                 0.25 0.75]
            y ~ DiscreteTimeDiscreteStatesHMM(P,
                                              (Bernoulli(p1), Bernoulli(p2)),
                                              Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0, 1, 1, 0]
    )

    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function _re_hmm_scalar_continuous_dm()
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.0, prior=Normal(0.0, 0.5))
            λ21_r = RealNumber(-0.1, prior=Normal(0.0, 0.5))
            p1_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
            p2_r = RealNumber(0.1, prior=Normal(0.0, 1.0))
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            λ12 = exp(clamp(λ12_r, -1.0, 1.0))
            λ21 = exp(clamp(λ21_r, -1.0, 1.0))
            p1 = 0.8 / (1 + exp(-clamp(p1_r + η, -2.0, 2.0))) + 0.1
            p2 = 0.8 / (1 + exp(-clamp(p2_r, -2.0, 2.0))) + 0.1
            Q = [-λ12 λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        dt = fill(1.0, 4),
        y = [0, 1, 1, 0]
    )

    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function _re_hmm_mv_discrete_dm()
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p11_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
            p12_r = RealNumber(0.2, prior=Normal(0.0, 1.0))
            p21_r = RealNumber(-0.2, prior=Normal(0.0, 1.0))
            p22_r = RealNumber(0.3, prior=Normal(0.0, 1.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            p11 = 0.8 / (1 + exp(-clamp(p11_r + η, -2.0, 2.0))) + 0.1
            p12 = 0.8 / (1 + exp(-clamp(p12_r, -2.0, 2.0))) + 0.1
            p21 = 0.8 / (1 + exp(-clamp(p21_r, -2.0, 2.0))) + 0.1
            p22 = 0.8 / (1 + exp(-clamp(p22_r, -2.0, 2.0))) + 0.1
            P = [0.85 0.15;
                 0.25 0.75]
            e1 = (Bernoulli(p11), Bernoulli(p12))
            e2 = (Bernoulli(p21), Bernoulli(p22))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = Any[[0, 1], [1, 1], [1, 0], [0, 0]]
    )

    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function _re_hmm_mv_continuous_dm()
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.0, prior=Normal(0.0, 0.5))
            λ21_r = RealNumber(-0.1, prior=Normal(0.0, 0.5))
            p11_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
            p12_r = RealNumber(0.2, prior=Normal(0.0, 1.0))
            p21_r = RealNumber(-0.2, prior=Normal(0.0, 1.0))
            p22_r = RealNumber(0.3, prior=Normal(0.0, 1.0))
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            λ12 = exp(clamp(λ12_r, -1.0, 1.0))
            λ21 = exp(clamp(λ21_r, -1.0, 1.0))
            p11 = 0.8 / (1 + exp(-clamp(p11_r + η, -2.0, 2.0))) + 0.1
            p12 = 0.8 / (1 + exp(-clamp(p12_r, -2.0, 2.0))) + 0.1
            p21 = 0.8 / (1 + exp(-clamp(p21_r, -2.0, 2.0))) + 0.1
            p22 = 0.8 / (1 + exp(-clamp(p22_r, -2.0, 2.0))) + 0.1
            Q = [-λ12 λ12;
                  λ21 -λ21]
            e1 = (Bernoulli(p11), Bernoulli(p12))
            e2 = (Bernoulli(p21), Bernoulli(p22))
            y ~ MVContinuousTimeDiscreteStatesHMM(Q, (e1, e2), Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        dt = fill(1.0, 4),
        y = Any[[0, 1], [1, 1], [1, 0], [0, 0]]
    )

    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function _assert_hmm_method_smoke(dm, method_name::Symbol, method)
    res = fit_model(dm, method; rng=Random.Xoshiro(123), store_data_model=false)
    @test res isa FitResult
    if method_name === :MCMC
        @test NoLimits.get_chain(res) isa Chains
    else
        @test isfinite(NoLimits.get_objective(res))
    end
    return res
end

const _HMM_RE_SMOKE_MODELS = [
    ("scalar discrete", _re_hmm_scalar_discrete_dm),
    ("scalar continuous", _re_hmm_scalar_continuous_dm),
    ("mv discrete", _re_hmm_mv_discrete_dm),
    ("mv continuous", _re_hmm_mv_continuous_dm),
]

const _HMM_RE_SMOKE_METHODS = [
    (:Laplace, () -> NoLimits.Laplace(; optim_kwargs=(maxiters=1,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0)),
    (:LaplaceMAP, () -> NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=1,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0)),
    (:MCMC, () -> NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=0, progress=false))),
    (:VI, () -> NoLimits.VI(; turing_kwargs=(max_iter=3, progress=false))),
    (:MCEM, () -> NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=0, progress=false), maxiters=1, progress=false)),
    (:SAEM, () -> NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=0, progress=false), mcmc_steps=1, q_store_max=4, maxiters=1, progress=false, builtin_stats=:auto)),
]

for (model_name, dm_builder) in _HMM_RE_SMOKE_MODELS
    @testset "HMM RE smoke: $(model_name)" begin
        dm = dm_builder()
        for (method_name, method_builder) in _HMM_RE_SMOKE_METHODS
            @testset "$(method_name)" begin
                _assert_hmm_method_smoke(dm, method_name, method_builder())
            end
        end
    end
end
