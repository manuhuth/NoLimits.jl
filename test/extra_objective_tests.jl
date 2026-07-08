using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using Random
using Turing
using MCMCChains
using SciMLBase

# `extra_objective` is a user cost `θu -> Real` added to the negative log-likelihood.
# These tests pin the two fixes:
#   - SAEM/MCEM must fold a variance-only extra term into the RE covariance (ω), not drop it.
#   - MCMC/VI must accept extra_objective and add it to the target (via @addlogprob!).
# The `extra_objective === nothing` path must stay a no-op.

# RE model where ω appears ONLY in the RE distribution (a Q2/closed-form param).
function _eo_re_model()
    return @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale = :log, lower = 1e-8, upper = Inf)
            ω = RealNumber(0.6, scale = :log, lower = 1e-8, upper = Inf)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
end

# Data with a_true = 2.0, σ_true = 0.3, ω_true = 1.0.
function _eo_re_data(; n_id = 30, n_obs = 3, seed = 20260708)
    rng = MersenneTwister(seed)
    a_true, σ_true, ω_true = 2.0, 0.3, 1.0
    ids = Int[]
    ts = Float64[]
    ys = Float64[]
    for i in 1:n_id
        ηi = ω_true * randn(rng)
        for j in 1:n_obs
            push!(ids, i)
            push!(ts, float(j))
            push!(ys, a_true + ηi + σ_true * randn(rng))
        end
    end
    return DataFrame(ID = ids, t = ts, y = ys)
end

_eo_re_dm() = DataModel(_eo_re_model(), _eo_re_data(); primary_id = :ID, time_col = :t)

_ω(res) = NoLimits.get_params(res; scale = :untransformed).ω

# variance-only cost pulling ω toward a target away from its data-implied value (~1.0)
const _EO_ΩTARGET = 0.2
_eo_var_pull(λ) = θu -> λ * (θu.ω - _EO_ΩTARGET)^2

@testset "extra_objective ForwardDiff-compatible" begin
    dm = _eo_re_dm()
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    ax = getaxes(θ0)
    f = _eo_var_pull(200.0)
    g = ForwardDiff.gradient(v -> f(ComponentArray(v, ax)), collect(θ0))
    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

@testset "SAEM: variance-only extra_objective moves ω (bug fix)" begin
    dm = _eo_re_dm()
    saem = NoLimits.SAEM(; maxiters = 80, mcmc_steps = 10, n_chains = 1, progress = false)
    res0 = fit_model(dm, saem; rng = MersenneTwister(1),
        serialization = SciMLBase.EnsembleSerial())
    res1 = fit_model(dm, saem; rng = MersenneTwister(1),
        serialization = SciMLBase.EnsembleSerial(),
        extra_objective = _eo_var_pull(200.0))
    ω0, ω1 = _ω(res0), _ω(res1)
    @test ω0 > 0.7               # closed-form path recovers the data-implied ω (~1.0)
    @test ω1 < 0.65              # numeric joint M-step pulls ω toward the target 0.2 (~0.52)
    @test ω1 < ω0 - 0.3          # clearly moved (would be identical under the bug)
    @test isfinite(get_objective(res1))
end

@testset "MCEM: variance-only extra_objective moves ω (bug fix)" begin
    dm = _eo_re_dm()
    mcem = NoLimits.MCEM(; sampler = MH(),
        turing_kwargs = (n_samples = 30, n_adapt = 5, progress = false),
        maxiters = 40)
    res0 = fit_model(dm, mcem; rng = MersenneTwister(1),
        serialization = SciMLBase.EnsembleSerial())
    res1 = fit_model(dm, mcem; rng = MersenneTwister(1),
        serialization = SciMLBase.EnsembleSerial(),
        extra_objective = _eo_var_pull(200.0))
    ω0, ω1 = _ω(res0), _ω(res1)
    @test ω0 > 0.7
    @test ω1 < 0.65
    @test ω1 < ω0 - 0.3
end

@testset "SAEM ω with extra approaches Laplace/FOCEI" begin
    dm = _eo_re_dm()
    pull = _eo_var_pull(200.0)
    lap = fit_model(dm, NoLimits.Laplace(); extra_objective = pull,
        serialization = SciMLBase.EnsembleSerial())
    foc = fit_model(dm, NoLimits.FOCEI(); extra_objective = pull,
        serialization = SciMLBase.EnsembleSerial())
    saem = fit_model(dm,
        NoLimits.SAEM(; maxiters = 80, mcmc_steps = 10, n_chains = 1, progress = false);
        rng = MersenneTwister(2), extra_objective = pull,
        serialization = SciMLBase.EnsembleSerial())
    @test isapprox(_ω(lap), _ω(foc); atol = 0.05)   # both fold extra into D identically
    @test isapprox(_ω(saem), _ω(lap); atol = 0.08)  # SAEM now matches the faithful ω
end

@testset "extra_objective === nothing is a no-op (SAEM/MCEM unchanged)" begin
    dm = _eo_re_dm()
    saem = NoLimits.SAEM(; maxiters = 40, mcmc_steps = 8, n_chains = 1, progress = false)
    a = fit_model(dm, saem; rng = MersenneTwister(7),
        serialization = SciMLBase.EnsembleSerial())
    b = fit_model(dm, saem; rng = MersenneTwister(7),
        serialization = SciMLBase.EnsembleSerial(), extra_objective = nothing)
    @test get_objective(a) == get_objective(b)
    @test _ω(a) == _ω(b)
end

# Fixed-effects-only model with priors, for MCMC / VI.
function _eo_fe_dm()
    m = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0; prior = Normal(0.0, 10.0))
            σ = RealNumber(0.5, scale = :log, lower = 1e-8, upper = Inf;
                prior = LogNormal(0.0, 1.0))
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end
    return DataModel(m, _eo_re_data(); primary_id = :ID, time_col = :t)
end

@testset "MCMC honors extra_objective (bug fix)" begin
    dm = _eo_fe_dm()
    meth = NoLimits.MCMC(; sampler = NUTS(80, 0.8),
        turing_kwargs = (n_samples = 300, n_adapt = 120, progress = false))
    res0 = fit_model(dm, meth; rng = MersenneTwister(3))
    # strong pull of a toward 8.0 (cost = negative log-likelihood ⇒ Gaussian likelihood on a)
    res1 = fit_model(dm, meth; rng = MersenneTwister(3),
        extra_objective = θu -> 50.0 * (θu.a - 8.0)^2)
    amean(r) = mean(vec(Array(get_chain(r)[:, :a, :])))
    @test amean(res0) < 4.0             # data pulls a toward ~2
    @test amean(res1) > amean(res0) + 1.0  # extra term shifts the posterior toward 8
end

@testset "VI accepts extra_objective" begin
    dm = _eo_fe_dm()
    meth = NoLimits.VI(; turing_kwargs = (max_iter = 50,))
    # both the nothing path and a real term must run without error
    r0 = fit_model(dm, meth; rng = MersenneTwister(5), extra_objective = nothing)
    r1 = fit_model(dm, meth; rng = MersenneTwister(5),
        extra_objective = θu -> 10.0 * (θu.a - 5.0)^2)
    @test r0 isa FitResult
    @test r1 isa FitResult
end
