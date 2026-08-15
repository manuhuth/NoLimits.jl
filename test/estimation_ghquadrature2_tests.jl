using Test
using NoLimits
using OptimizationNLopt
using LikelihoodProfiler
using LinearAlgebra
using Distributions
using DataFrames
using ComponentArrays
using OptimizationOptimJL
using OptimizationBBO
using NoLimits.LineSearches
using FiniteDifferences
using Random
import Turing

# Access internal functions via module
const build_sparse_grid = NoLimits.build_sparse_grid

# ── Shared models (each @Model source block compiles once; data varies per use) ──

# Scalar Normal RE workhorse: y ~ Normal(a + η, σ), η ~ Normal(0, ω).
const _GHQ_SCALAR_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5, scale = :log)
        ω = RealNumber(1.0, scale = :log)
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

# Non-Gaussian RE families (transport-map coverage); one model per family,
# shared by the fit / get_random_effects / validation testsets.
const _GHQ_LOGN_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.5)
        σ = RealNumber(0.5, scale = :log)
        ω = RealNumber(0.6, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(LogNormal(0.0, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(a * η, σ)
    end
end

const _GHQ_BETA_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.4)
        b = RealNumber(2.5)
        σ = RealNumber(0.3, scale = :log)
        α = RealNumber(2.0, scale = :log)
        β = RealNumber(5.0, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Beta(α, β); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + b * η, σ)
    end
end

const _GHQ_GAMMA_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(2.0)
        σ = RealNumber(0.4, scale = :log)
        α = RealNumber(2.0, scale = :log)
        θ = RealNumber(0.5, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Gamma(α, θ); column = :ID)
    end
    @formulas begin
        y ~ Normal(a * η, σ)
    end
end

const _GHQ_EXP_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.5)
        σ = RealNumber(0.4, scale = :log)
        θ = RealNumber(0.5, scale = :log)   # scale = 1/rate
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Exponential(θ); column = :ID)
    end
    @formulas begin
        y ~ Normal(a * η, σ)
    end
end

const _GHQ_WEIB_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.3, scale = :log)
        α = RealNumber(2.0, scale = :log)
        θ = RealNumber(1.5, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Weibull(α, θ); column = :ID)
    end
    @formulas begin
        y ~ Normal(a * η, σ)
    end
end

const _GHQ_TDIST_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.3, scale = :log)
        ν = RealNumber(5.0, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(TDist(ν); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end

# Helper: signed quadrature sum for integration tests
function sg_integrate(f, sg::NoLimits.GHQuadratureNodes)
    total = 0.0
    for r in 1:size(sg.nodes, 2)
        z = sg.nodes[:, r]
        total += sg.signs[r] * exp(sg.logweights[r]) * f(z)
    end
    return total
end

# ============================================================
# STEP 1: nodes.jl — GH rule + Smolyak construction + cache
# ============================================================

# Second half of the GHQ tests (split from estimation_ghquadrature_tests.jl for
# CI shard balance; the halves run in different lanes and are self-contained —
# the shared model consts and grid helpers above are duplicated on purpose).
# ℝ-supported generic fallback (identity transport) — hits the generic branch.
const _GHQ_LAPL_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.2, scale = :log)
        b = RealNumber(0.5, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Distributions.Laplace(0.0, b); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end

# (0,∞)-supported generic fallback (exp transport) — hits the generic branch.
const _GHQ_INVG_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.5)
        σ = RealNumber(0.3, scale = :log)
        α = RealNumber(3.0, scale = :log)
        β = RealNumber(2.0, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(InverseGamma(α, β); column = :ID)
    end
    @formulas begin
        y ~ Normal(a * η, σ)
    end
end

# Assertion sets shared by several families.
function _ghq_family_fit_checks(res, dm)
    @test NoLimits.get_converged(res) isa Bool
    p = NoLimits.get_params(res; scale = :untransformed)
    return @test NoLimits.get_objective(res) < 1.0e6
end

function _ghq_family_re_checks(re, n_id)
    @test re isa NamedTuple && haskey(re, :η)
    return @test nrow(re.η) == n_id
end

# One entry per family. `fit`/`re` share the data-gen shell (per-id η via
# `eta`, iid N(0,1) noise mixed in by `y`); `nothing` skips that part.
const _GHQ_RE_FAMILY_CASES = [
    # LogNormal(0, ω): η = exp(ω z), z ~ N(0,1). The exp transport is nonlinear
    # (CompositeRE segment_fn), NOT the linear GaussianRE path; logcorrection = 0
    # only because the push-forward is exactly LogNormal (needs has_npf=true).
    (
        label = "LogNormal", model = _GHQ_LOGN_MODEL,
        fit = (
            seed = 42, n_id = 12, n_obs = 5,
            eta = (rng, n) -> exp.(0.5 .* randn(rng, n)),
            y = (η, ε) -> 2.0 .* η .+ 0.3 .* ε,
            check = (res, dm) -> begin
                _ghq_family_fit_checks(res, dm)
                p = NoLimits.get_params(res; scale = :untransformed)
                @test abs(p.a - 2.0) < 1.5  # a_true ballpark
            end
        ),
        re = (
            seed = 7, n_id = 8, n_obs = 4,
            eta = (rng, n) -> exp.(0.4 .* randn(rng, n)),
            y = (η, ε) -> 1.5 .* η .+ 0.2 .* ε,
            check = (re, n_id) -> begin
                @test re isa NamedTuple
                @test haskey(re, :η)
                @test nrow(re.η) == n_id
            end
        ),
        valid_y = [1.0, 1.1],
    ),
    # Beta(α, β): logistic transport of N(0,1).
    (
        label = "Beta", model = _GHQ_BETA_MODEL,
        fit = (
            seed = 99, n_id = 14, n_obs = 4,
            eta = (rng, n) -> rand(rng, Beta(2.0, 5.0), n),
            y = (η, ε) -> 0.5 .+ 2.0 .* η .+ 0.2 .* ε,
            check = _ghq_family_fit_checks,
        ),
        re = (
            seed = 55, n_id = 8, n_obs = 5,
            eta = (rng, n) -> rand(rng, Beta(2.0, 4.0), n),
            y = (η, ε) -> 1.0 .+ 2.0 .* η .+ 0.15 .* ε,
            check = (re, n_id) -> begin
                @test re isa NamedTuple
                @test haskey(re, :η)
                @test nrow(re.η) == n_id
                # EB modes in (0, 1); column :η_1 (flatten of 1-elem output)
                @test all(0.0 .< re.η.η_1 .< 1.0)
            end
        ),
        valid_y = [0.5, 0.6],
    ),
    (
        label = "Gamma", model = _GHQ_GAMMA_MODEL,
        fit = (
            seed = 301, n_id = 14, n_obs = 4,
            eta = (rng, n) -> rand(rng, Gamma(3.0, 0.5), n),
            y = (η, ε) -> 2.0 .* η .+ 0.3 .* ε,
            check = _ghq_family_fit_checks,
        ),
        re = (
            seed = 302, n_id = 8, n_obs = 4,
            eta = (rng, n) -> rand(rng, Gamma(2.0, 1.0), n),
            y = (η, ε) -> 1.5 .* η .+ 0.2 .* ε,
            check = _ghq_family_re_checks,
        ),
        valid_y = [1.0, 1.5],
    ),
    (
        label = "Exponential", model = _GHQ_EXP_MODEL,
        fit = (
            seed = 401, n_id = 12, n_obs = 4,
            eta = (rng, n) -> rand(rng, Exponential(1 / 2.0), n),
            y = (η, ε) -> 1.5 .* η .+ 0.3 .* ε,
            check = (res, dm) -> begin
                @test NoLimits.get_converged(res) isa Bool
                p = NoLimits.get_params(res; scale = :untransformed)
            end
        ),
        re = (
            seed = 402, n_id = 8, n_obs = 4,
            eta = (rng, n) -> rand(rng, Exponential(0.5), n),
            y = (η, ε) -> 2.0 .* η .+ 0.2 .* ε,
            check = nothing,
        ),
        valid_y = nothing,
    ),
    (
        label = "Weibull", model = _GHQ_WEIB_MODEL,
        fit = (
            seed = 501, n_id = 14, n_obs = 4,
            eta = (rng, n) -> rand(rng, Weibull(2.0, 1.5), n),
            y = (η, ε) -> 1.0 .* η .+ 0.3 .* ε,
            check = _ghq_family_fit_checks,
        ),
        re = (
            seed = 502, n_id = 8, n_obs = 4,
            eta = (rng, n) -> rand(rng, Weibull(2.0, 1.0), n),
            y = (η, ε) -> η .+ 0.2 .* ε,
            check = nothing,
        ),
        valid_y = nothing,
    ),
    # TDist(ν): heavy-tailed, ℝ-supported — identity transport.
    (
        label = "TDist", model = _GHQ_TDIST_MODEL,
        fit = (
            seed = 601, n_id = 14, n_obs = 4,
            eta = (rng, n) -> rand(rng, TDist(5.0), n),
            y = (η, ε) -> 1.0 .+ η .+ 0.3 .* ε,
            check = _ghq_family_fit_checks,
        ),
        re = (
            seed = 602, n_id = 8, n_obs = 5,
            eta = (rng, n) -> rand(rng, TDist(4.0), n),
            y = (η, ε) -> 0.5 .+ η .+ 0.2 .* ε,
            check = _ghq_family_re_checks,
        ),
        valid_y = nothing,
    ),
    (
        label = "Laplace", model = _GHQ_LAPL_MODEL,
        fit = (
            seed = 701, n_id = 12, n_obs = 4,
            eta = (rng, n) -> rand(rng, Distributions.Laplace(0.0, 0.5), n),
            y = (η, ε) -> 1.0 .+ η .+ 0.2 .* ε,
            check = (res, dm) -> begin
                @test NoLimits.get_converged(res) isa Bool
            end
        ),
        re = nothing,
        valid_y = nothing,
    ),
    # InverseGamma: fit + EB extraction smoke test (no assertions originally).
    (
        label = "InverseGamma", model = _GHQ_INVG_MODEL,
        fit = (
            seed = 702, n_id = 12, n_obs = 4,
            eta = (rng, n) -> rand(rng, InverseGamma(3.0, 2.0), n),
            y = (η, ε) -> 1.5 .* η .+ 0.3 .* ε,
            check = (res, dm) -> NoLimits.get_random_effects(dm, res),
        ),
        re = nothing,
        valid_y = nothing,
    ),
]

# Data-gen shell shared by the fit and EB runs of every family.
function _ghq_family_dm(model, spec)
    rng = MersenneTwister(spec.seed)
    ids = repeat(1:(spec.n_id), inner = spec.n_obs)
    η_i = spec.eta(rng, spec.n_id)
    yobs = spec.y(η_i[ids], randn(rng, spec.n_id * spec.n_obs))
    tobs = repeat(1:(spec.n_obs), spec.n_id) .* 1.0
    df = DataFrame(ID = ids, t = tobs, y = yobs)
    return DataModel(model, df; primary_id = :ID, time_col = :t)
end

for c in _GHQ_RE_FAMILY_CASES
    @testset "GHQuadrature $(c.label) RE" begin
        @testset "fit_model with $(c.label) RE" begin
            dm = _ghq_family_dm(c.model, c.fit)
            res = fit_model(
                dm, NoLimits.GHQuadrature(level = 2; optim_kwargs = (maxiters = 2,))
            )
            c.fit.check(res, dm)
        end

        if c.re !== nothing
            @testset "get_random_effects for $(c.label) RE" begin
                dm = _ghq_family_dm(c.model, c.re)
                res = fit_model(
                    dm,
                    NoLimits.GHQuadrature(level = 2; optim_kwargs = (maxiters = 2,))
                )
                re = NoLimits.get_random_effects(dm, res)
                c.re.check === nothing || c.re.check(re, c.re.n_id)
            end
        end

        if c.valid_y !== nothing
            @testset "validation allows $(c.label)" begin
                df = DataFrame(ID = [1, 1], t = [1.0, 2.0], y = c.valid_y)
                dm = DataModel(c.model, df; primary_id = :ID, time_col = :t)
                @test_nowarn NoLimits._ghq_validate_re_distributions(dm)
            end
        end
    end
end

# ============================================================
# Phase 3: Anisotropic grids
# ============================================================

@testset "Anisotropic sparse grids" begin
    build_tensor_product_grid = NoLimits.build_tensor_product_grid
    get_anisotropic_grid = NoLimits.get_anisotropic_grid

    @testset "tensor product d=1×1 gives d=2 grid" begin
        sg1 = build_sparse_grid(1, 2)  # 2 points
        sg2 = build_sparse_grid(1, 3)  # 3 points
        tp = build_tensor_product_grid([sg1, sg2])
        @test tp.dim == 2
        @test size(tp.nodes, 2) == 2 * 3
    end

    @testset "tensor product integration accuracy" begin
        sg1 = build_sparse_grid(1, 3)
        sg2 = build_sparse_grid(1, 3)
        tp = build_tensor_product_grid([sg1, sg2])
        # E_{N(0,I₂)}[z₁² + z₂²] = 2
        val = sg_integrate(z -> z[1]^2 + z[2]^2, tp)
        @test val ≈ 2.0 atol = 1.0e-10
        # E[z₁² * z₂²] = 1 (independence)
        val2 = sg_integrate(z -> z[1]^2 * z[2]^2, tp)
        @test val2 ≈ 1.0 atol = 1.0e-10
    end

    @testset "get_anisotropic_grid caches correctly" begin
        dims = [1, 1]
        levels = [2, 3]
        sg_a = get_anisotropic_grid(dims, levels)
        sg_b = get_anisotropic_grid(dims, levels)  # same key → same object
        @test sg_a === sg_b
        @test size(sg_a.nodes, 2) ==
            prod(NoLimits.n_ghq_points(d, l) for (d, l) in zip(dims, levels))
    end

    @testset "anisotropic fit with NamedTuple level" begin
        rng = MersenneTwister(12)
        n_id = 10
        ids = repeat(1:n_id, inner = 5)
        η_i = 0.5 .* randn(rng, n_id)
        yobs = 1.0 .+ η_i[ids] .+ 0.3 .* randn(rng, length(ids))
        tobs = repeat(1:5, n_id) .* 1.0

        df = DataFrame(ID = ids, t = tobs, y = yobs)
        dm = DataModel(_GHQ_SCALAR_MODEL, df; primary_id = :ID, time_col = :t)

        # Anisotropic level: η at level 2 (isotropic would use same level for all)
        res_iso = fit_model(dm, GHQuadrature(level = 2; optim_kwargs = (maxiters = 2,)))
        res_aniso = fit_model(
            dm, GHQuadrature(level = (η = 2,); optim_kwargs = (maxiters = 2,))
        )

        @test NoLimits.get_converged(res_iso) isa Bool
        @test NoLimits.get_converged(res_aniso) isa Bool

        # Objectives should be comparable (same level=2 for the only RE)
        obj_iso = NoLimits.get_objective(res_iso)
        obj_aniso = NoLimits.get_objective(res_aniso)
        @test abs(obj_iso - obj_aniso) / max(abs(obj_iso), 1.0) < 0.05

        # Parameters should be close
        p_iso = NoLimits.get_params(res_iso; scale = :untransformed)
        p_aniso = NoLimits.get_params(res_aniso; scale = :untransformed)
        @test abs(p_iso.a - p_aniso.a) < 0.1
    end

    @testset "anisotropic level rejects unknown RE names" begin
        # A misspelled RE name used to silently fall back to level 1 (#226).
        rng = MersenneTwister(7)
        n_id = 8
        ids = repeat(1:n_id, inner = 4)
        yobs = 1.0 .+ 0.3 .* randn(rng, length(ids))
        tobs = repeat(1:4, n_id) .* 1.0

        df = DataFrame(ID = ids, t = tobs, y = yobs)
        dm = DataModel(_GHQ_SCALAR_MODEL, df; primary_id = :ID, time_col = :t)

        @test_throws ErrorException fit_model(
            dm, GHQuadrature(level = (nonexistent = 5,); optim_kwargs = (maxiters = 2,))
        )
        # Scalar levels must be positive integers.
        @test_throws ErrorException GHQuadrature(level = 0)
        @test_throws ErrorException GHQuadrature(level = (η = 0,))
    end
end  # @testset "Anisotropic sparse grids"

# ============================================================
# Progressive refinement: level::Vector{Int}
# ============================================================

@testset "GHQuadrature progressive refinement (level::Vector{Int})" begin
    function _make_progressive_dm(; n_id = 10, n_obs = 5, rng = MersenneTwister(42))
        ids = repeat(1:n_id, inner = n_obs)
        yobs = 1.0 .+ 0.3 .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0
        df = DataFrame(ID = ids, t = tobs, y = yobs)
        DataModel(_GHQ_SCALAR_MODEL, df; primary_id = :ID, time_col = :t)
    end

    @testset "level=[1,2] converges and result is scalar-level" begin
        dm = _make_progressive_dm()
        res = fit_model(dm, GHQuadrature(level = [1, 2]; optim_kwargs = (maxiters = 2,)))
        @test NoLimits.get_converged(res) isa Bool
        # Returned method should carry the last scalar level (2)
        @test NoLimits.get_method(res).level == 2
    end

    @testset "level=[1] (single-element) behaves like level=1" begin
        rng = MersenneTwister(1)
        dm = _make_progressive_dm(; rng = rng)
        res_vec = fit_model(dm, GHQuadrature(level = [1]; optim_kwargs = (maxiters = 2,)))
        res_scalar = fit_model(dm, GHQuadrature(level = 1; optim_kwargs = (maxiters = 2,)))
        @test NoLimits.get_converged(res_vec) isa Bool
        @test abs(NoLimits.get_objective(res_vec) - NoLimits.get_objective(res_scalar)) <
            1.0e-4
    end

    @testset "level=[1,2,3] three-stage refinement" begin
        dm = _make_progressive_dm()
        res = fit_model(dm, GHQuadrature(level = [1, 2, 3]; optim_kwargs = (maxiters = 2,)))
        @test NoLimits.get_converged(res) isa Bool
        @test NoLimits.get_method(res).level == 3
        p = NoLimits.get_params(res; scale = :untransformed)
    end

    @testset "level=[1,2] result compatible with all accessors" begin
        dm = _make_progressive_dm()
        res = fit_model(dm, GHQuadrature(level = [1, 2]; optim_kwargs = (maxiters = 2,)))
        @test NoLimits.get_iterations(res) isa Integer
        re = NoLimits.get_random_effects(dm, res)
        @test re isa NamedTuple && haskey(re, :η)
        ll = NoLimits.get_loglikelihood(res)
    end

    @testset "empty level vector throws" begin
        dm = _make_progressive_dm()
        @test_throws ErrorException fit_model(dm, GHQuadrature(level = Int[]))
    end

    @testset "non-positive level entry throws" begin
        dm = _make_progressive_dm()
        @test_throws ErrorException fit_model(dm, GHQuadrature(level = [1, 0]))
    end
end  # @testset "GHQuadrature progressive refinement"

# ============================================================
# MCIntegrator: prior sampling and Turing MCMC sampling
# ============================================================

@testset "get_loglikelihood_quadrature MC sampling" begin

    # Shared fixture: simple Normal RE model with Laplace fit. Keeps its own
    # model (ω init 0.4): the MC-vs-quadrature ballpark tolerances below are
    # calibrated to the θ this short Laplace fit reaches from these inits.
    function _mc_test_dm(; n_id = 10, n_obs = 5, rng = MersenneTwister(900))
        ids = repeat(1:n_id, inner = n_obs)
        yobs = 1.0 .+ 0.4 .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.4, scale = :log)
                ω = RealNumber(0.4, scale = :log)
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
        df = DataFrame(ID = ids, t = tobs, y = yobs)
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
        # The MC-vs-quadrature tolerances below were calibrated against a 2-iteration
        # LBFGS fit. The default outer optimizer (NLopt.LN_BOBYQA) reads maxiters as a
        # function-evaluation cap, so 2 evals leave the fit at a degenerate point and the
        # comparisons blow past tolerance. Pin the gradient-based LBFGS here.
        res = fit_model(
            dm,
            NoLimits.Laplace(;
                optimizer = OptimizationOptimJL.LBFGS(
                    linesearch = LineSearches.BackTracking()
                ),
                optim_kwargs = (maxiters = 2,)
            )
        )
        return dm, res
    end

    @testset "MCIntegrator constructor" begin
        mc = MCIntegrator()
        @test mc.n_samples == 1000
        @test mc.mode === :turing
        @test mc.sampler === nothing
        @test mc.n_warmup == 500
        @test mc.rng === nothing

        mc2 = MCIntegrator(n_samples = 2, mode = :prior)
        @test mc2.n_samples == 2

        @test_throws ErrorException MCIntegrator(mode = :unknown)
        @test_throws ErrorException MCIntegrator(n_warmup = -1)
    end

    @testset "prior MC for all batches: finite and close to quadrature" begin
        dm, res = _mc_test_dm()
        ll_ghq = get_loglikelihood_quadrature(res; level = 2, seed = 1)
        ll_mc = get_loglikelihood_quadrature(
            res; seed = 1,
            mc_integrator = MCIntegrator(n_samples = 2, mode = :prior)
        )
        # Prior MC should be in the right ballpark (within 15 log units for n_id=10)
        @test abs(ll_mc - ll_ghq) < 15.0
    end

    @testset "seed makes result reproducible" begin
        dm, res = _mc_test_dm()
        ll1 = get_loglikelihood_quadrature(
            res; seed = 42,
            mc_integrator = MCIntegrator(n_samples = 2, mode = :prior)
        )
        ll2 = get_loglikelihood_quadrature(
            res; seed = 42,
            mc_integrator = MCIntegrator(n_samples = 2, mode = :prior)
        )
        @test ll1 == ll2
    end

    @testset "fallback=nothing is accepted and default AGHQ path works" begin
        dm, res = _mc_test_dm()
        ll = get_loglikelihood_quadrature(res; level = 2, seed = 1, fallback = nothing)
    end

    @testset "Turing MC for all batches: finite result" begin
        dm, res = _mc_test_dm(n_id = 8, n_obs = 5, rng = MersenneTwister(901))
        ll_mc_turing = get_loglikelihood_quadrature(
            res; seed = 2,
            mc_integrator = MCIntegrator(
                n_samples = 2, mode = :turing,
                sampler = Turing.MH(), n_warmup = 200
            )
        )
    end

    @testset "Turing MC close to quadrature" begin
        dm, res = _mc_test_dm(n_id = 10, n_obs = 5, rng = MersenneTwister(903))
        ll_ghq = get_loglikelihood_quadrature(res; level = 2, seed = 3)
        ll_turing = get_loglikelihood_quadrature(
            res; seed = 3,
            mc_integrator = MCIntegrator(
                n_samples = 2, mode = :turing,
                sampler = Turing.MH(), n_warmup = 500
            )
        )
        @test abs(ll_turing - ll_ghq) < 2.0
    end

    @testset "fallback MCIntegrator path works end-to-end" begin
        dm, res = _mc_test_dm(n_id = 8, n_obs = 4)
        ll = get_loglikelihood_quadrature(
            res; seed = 10, jitter = 1.0e-6,
            fallback = MCIntegrator(n_samples = 2, mode = :prior)
        )
    end
end  # @testset "get_loglikelihood_quadrature MC sampling"

# The adaptive-quadrature measure must whiten the integrand: with -H = L*L', the scaling
# matrix S has to satisfy S*S' = (-H)^{-1}, equivalently S'(-H)S = I. `inv(L)` is also
# lower triangular and has the same determinant, so getting this wrong left the Laplace
# term (level 1, single node at z=0) exact while every multi-node rule integrated a
# sheared integrand -- AGHQ then failed to converge in the quadrature level.
@testset "AGHQ measure whitens the integrand (S'(-H)S = I)" begin
    model = @Model begin
        @fixedEffects begin
            tcl = RealNumber(log(0.05))
            tv = RealNumber(log(1.0))
            Ω = RealPSDMatrix([0.2 0.05; 0.05 0.15], scale = :cholesky)
            σ = RealNumber(0.2, scale = :log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end

        @formulas begin
            cp = exp(tcl + η[1]) / exp(tv + η[2])
            y ~ Normal(cp, σ)
        end
    end

    rng = Xoshiro(4)
    df = DataFrame(
        ID = repeat(1:10; inner = 4), t = repeat([0.5, 1.0, 2.0, 4.0], 10),
        y = 0.05 .* exp.(0.3 .* randn(rng, 40))
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))
    θ_re = NoLimits.symmetrize_psd_parameters(θ, NoLimits.get_fixed(NoLimits.get_model(dm)))

    _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
    cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)
    bstars = NoLimits.empirical_bayes(dm, θ; rng = Xoshiro(2))

    for bi in 1:3
        rm = NoLimits.build_centered_re_measure(
            bstars[bi], infos[bi], bi, θ_re, cc, dm, cache
        )
        @test rm !== nothing
        H = NoLimits._laplace_hessian_b(
            dm, infos[bi], θ_re,
            Vector{Float64}(bstars[bi]), cc, cache, nothing, bi
        )
        A = rm.S' * (-H) * rm.S
        @test isapprox(A, I(size(A, 1)); rtol = 1.0e-6, atol = 1.0e-6)
        # the determinant was already right with the wrong factor, so assert the shape
        @test isapprox(rm.S * rm.S', inv(-H); rtol = 1.0e-6, atol = 1.0e-8)
    end
end

# Issue #98: the quadrature was centered on the RE prior, whose scale is far wider
# than the batch posterior. The signed Smolyak weights then drift away from the
# integral as the level rises and regularly flip the batch marginal negative,
# which the caller turns into -Inf (objective Inf, singular Wald Hessian, no
# recovery). The rule is now centered on the EB mode.
@testset "GHQuadrature is adaptive (issue #98)" begin
    ri_model = @Model begin
        @fixedEffects begin
            a = RealNumber(2.0)
            b = RealNumber(-0.5)
            σ = RealNumber(0.3, scale = :log)
            Ω = RealPSDMatrix([0.4 0.15; 0.15 0.25], scale = :cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end
        @formulas begin
            mu = (a + η[1]) + (b + η[2]) * t
            y ~ Normal(mu, σ)
        end
    end

    ts = collect(range(0.0, 2.0; length = 8))
    function _ri_df(nid)
        rng = Xoshiro(1)
        Ω = [0.4 0.15; 0.15 0.25]
        ID = String[]
        T = Float64[]
        Y = Float64[]
        for i in 1:nid
            e = rand(rng, MvNormal(zeros(2), Ω))
            for tt in ts
                push!(ID, "s$i")
                push!(T, tt)
                push!(Y, (2.0 + e[1]) + (-0.5 + e[2]) * tt + 0.3 * randn(rng))
            end
        end
        return DataFrame(ID = ID, t = T, y = Y)
    end

    @testset "batch marginal matches the exact linear-Gaussian value" begin
        dm = DataModel(ri_model, _ri_df(6); primary_id = :ID, time_col = :t)
        fe = NoLimits.get_fixed(NoLimits.get_model(dm))
        θ = NoLimits.get_θ0_untransformed(fe)
        θ_re = NoLimits._symmetrize_psd_params(θ, fe)
        _, infos, cc = NoLimits._build_re_batch_infos(dm, NamedTuple())
        cache = NoLimits.build_ll_cache(dm; force_saveat = true)
        cache = cache isa AbstractVector ? cache[1] : cache

        # y_i ~ N(a .+ b .* t, σ²I + Z Ω Z') for this model, so the marginal is exact.
        Z = hcat(ones(length(ts)), ts)
        V = Symmetric(
            0.3^2 * Matrix(I, length(ts), length(ts)) +
                Z * [0.4 0.15; 0.15 0.25] * Z'
        )
        rows = NoLimits.get_row_groups(dm).obs_rows
        for bi in 1:3
            inds = NoLimits.get_inds(infos[bi])
            @test length(inds) == 1
            y = NoLimits.get_df(dm).y[rows[only(inds)]]
            exact = logpdf(MvNormal(2.0 .- 0.5 .* ts, V), Vector{Float64}(y))
            for level in (3, 5)
                bll = NoLimits._ghq_batch_ll(dm, infos[bi], θ_re, cc, cache, level)
                @test isfinite(bll)
                @test isapprox(bll, exact; rtol = 1.0e-5)
            end
        end
    end

    @testset "fit recovers what Laplace recovers, with a finite objective" begin
        dm = DataModel(ri_model, _ri_df(30); primary_id = :ID, time_col = :t)
        θ0 = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(ri_model))
        θ0 = ComponentArray(copy(θ0), getaxes(θ0))
        θ0.a = 1.0
        θ0.b = 0.0
        θ0.σ = 1.0
        θ0.Ω = [1.0 0.0; 0.0 1.0]
        kw = (; theta_0_untransformed = θ0, serialization = NoLimits.EnsembleSerial())
        res_l = fit_model(dm, NoLimits.Laplace(); kw...)
        res_g = fit_model(dm, NoLimits.GHQuadrature(level = 5); kw...)

        @test isfinite(get_objective(res_g))
        # qualified: MCMCChains also exports get_params, ambiguous when batched
        p_l = NoLimits.get_params(res_l; scale = :untransformed)
        p_g = NoLimits.get_params(res_g; scale = :untransformed)
        # This model is linear-Gaussian, so Laplace is exact and adaptive
        # quadrature must reproduce it at any level.
        @test isapprox(p_g.σ, p_l.σ; rtol = 0.02)
        @test isapprox(p_g.Ω, p_l.Ω; rtol = 0.15)
        @test isapprox(get_objective(res_g), get_objective(res_l); rtol = 1.0e-4)
    end
end

# Issue #151: on ill-conditioned batches the adaptive rule was rejected by the
# *Laplace* admissibility test (λmin > 1e-8·λmax), which sent those batches back to
# the prior-centered rule of #98; its signed sum turns negative, the batch marginal
# becomes -Inf and the whole objective +Inf. That cliff amplified the 1e-16
# reduction-order noise of a threaded/permuted run into an O(1) objective gap.
@testset "GHQuadrature is order-invariant on ill-conditioned data (issue #151)" begin
    ill_model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            b = RealNumber(0.0)
            σ = RealNumber(0.5, scale = :log)
            Ω = RealPSDMatrix([0.3 0.0; 0.0 0.3], scale = :cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end
        @formulas begin
            mu = (a + η[1]) + (b + η[2]) * t
            y ~ Normal(mu, σ)
        end
    end

    # Mimics stress model M18: times over 1e-3..5e3, |y| over ~4 orders, 1-obs ids
    # alongside one heavily observed id, Ω eigenvalues 0.25 vs 1e-6.
    Ω_true = [0.25 2.0e-4; 2.0e-4 1.0e-6]
    function _ill_df()
        rng = Xoshiro(151)
        L = cholesky(Symmetric(Ω_true)).L
        rows = NamedTuple[]
        function add!(id, n)
            η = L * randn(rng, 2)
            for _ in 1:n
                t = 10.0^(rand(rng) * (log10(5000.0) + 3.0) - 3.0)
                μ = (5.0 + η[1]) + (-0.002 + η[2]) * t
                push!(rows, (; ID = id, t = t, y = μ + 0.3 * randn(rng)))
            end
        end
        add!("id_01", 40)
        for i in 2:9
            add!("id_" * lpad(string(i), 2, '0'), 1)
        end
        for i in 10:24
            add!("id_" * lpad(string(i), 2, '0'), 8)
        end
        df = DataFrame(rows)
        return df[randperm(Xoshiro(3), nrow(df)), :]
    end

    df = _ill_df()
    method = NoLimits.GHQuadrature(level = 3)

    @testset "ill-conditioned but definite -H keeps the adaptive rule" begin
        # One observation at a large time: -H = inv(Ω) + (1/σ²)[1 t; t t²] is positive
        # definite but has condition number ≫ 1e8, which the Laplace test rejects.
        df1 = DataFrame(ID = ["a"], t = [5000.0], y = [-5.0])
        dm1 = DataModel(ill_model, df1; primary_id = :ID, time_col = :t)
        θ0 = get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm1)))
        θ = ComponentArray(copy(θ0), getaxes(θ0))
        θ.σ = 0.3
        θ.Ω = [4.0 0.0; 0.0 4.0]
        θ_re = NoLimits.symmetrize_psd_parameters(
            θ, NoLimits.get_fixed(NoLimits.get_model(dm1))
        )
        _, infos, cc = NoLimits.build_re_batch_infos(dm1, NamedTuple())
        cache = NoLimits.build_likelihood_cache(dm1; force_saveat = true)
        bstars = NoLimits.empirical_bayes(dm1, θ; rng = Xoshiro(2))
        H = NoLimits._laplace_hessian_b(
            dm1, infos[1], θ_re,
            Vector{Float64}(bstars[1]), cc, cache, nothing, 1
        )
        # Admissible since #157 too: it factorizes without jitter. The old relative test
        # (λmin > 1e-8·λmax) rejected it, which is what forced the fallback below.
        @test NoLimits.negH_definite_without_jitter(H)
        rm = NoLimits.build_centered_re_measure(
            bstars[1], infos[1], 1, θ_re, cc, dm1, cache
        )
        @test rm !== nothing
        @test isapprox(rm.S' * (-H) * rm.S, I(2); rtol = 1.0e-6, atol = 1.0e-6)
        @test isfinite(NoLimits._ghq_batch_ll(dm1, infos[1], θ_re, cc, cache, 3, bstars[1]))
    end

    @testset "serial == threaded" begin
        if Threads.nthreads() < 2
            @info "skipped: needs Threads.nthreads() > 1"
            @test true
            return
        end
        dm_s = DataModel(
            ill_model, df; primary_id = :ID, time_col = :t,
            serialization = NoLimits.EnsembleSerial()
        )
        dm_t = DataModel(
            ill_model, df; primary_id = :ID, time_col = :t,
            serialization = NoLimits.EnsembleThreads()
        )
        o_s = get_objective(
            fit_model(
                dm_s, method; serialization = NoLimits.EnsembleSerial()
            )
        )
        o_t = get_objective(
            fit_model(
                dm_t, method; serialization = NoLimits.EnsembleThreads()
            )
        )
        @test isfinite(o_s)
        @test isapprox(o_s, o_t; rtol = 1.0e-6)
    end

    @testset "permuting and relabelling individuals" begin
        dm = DataModel(
            ill_model, df; primary_id = :ID, time_col = :t,
            serialization = NoLimits.EnsembleSerial()
        )
        ids = unique(df.ID)
        perm = randperm(Xoshiro(7), length(ids))
        relabel = Dict(
            ids[i] => "z" * lpad(string(perm[i]), 3, '0')
                for i in eachindex(ids)
        )
        df2 = copy(df)
        df2.ID = [relabel[i] for i in df.ID]
        df2 = df2[sortperm(df2.ID), :]
        dm2 = DataModel(
            ill_model, df2; primary_id = :ID, time_col = :t,
            serialization = NoLimits.EnsembleSerial()
        )
        kw = (; serialization = NoLimits.EnsembleSerial())
        res_a = fit_model(dm, method; kw...)
        res_b = fit_model(dm2, method; kw...)
        oa = get_objective(res_a)
        ob = get_objective(res_b)
        @test isfinite(oa)
        @test abs(oa - ob) <= 1.0e-8 * max(abs(oa), 1.0)
        # qualified: MCMCChains also exports get_params, ambiguous when batched
        pa = NoLimits.get_params(res_a; scale = :untransformed)
        pb = NoLimits.get_params(res_b; scale = :untransformed)
        @test maximum(abs.(collect(pa) .- collect(pb))) <= 1.0e-6
    end
end

# #176: adaptive centering is skipped when a crossed batch is not purely Gaussian, and the
# prior-centered Smolyak rule's signed weights used to turn the batch marginal negative at
# level >= 2 - reported as an Inf objective for the whole fit.
@testset "mixed Gaussian/non-Gaussian crossed batch stays finite" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.3, scale = :log)
            ω_id = RealNumber(0.5, scale = :log)
            α = RealNumber(2.0, scale = :log)
            β = RealNumber(2.0, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, ω_id); column = :ID)
            η_site = RandomEffect(Beta(α, β); column = :SITE)
        end
        @formulas begin
            y ~ Normal(a + η_id + (η_site - 0.5), σ)
        end
    end
    rng = Xoshiro(42)
    df = DataFrame(
        ID = repeat(1:6, inner = 2), SITE = repeat(1:6, inner = 2),
        t = repeat([0.0, 1.0], 6), y = rand(rng, 12) .+ 0.5
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm, GHQuadrature(level = 3, optim_kwargs = (maxiters = 5,));
        rng = Xoshiro(42)
    )
    @test isfinite(get_objective(res))
end
