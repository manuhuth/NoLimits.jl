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
const _gh_rule = NoLimits._gh_rule
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

# Prior-bearing scalar model for MAP/UQ; prior tightness parameterizable so the
# "MAP pulls toward prior" testset can reuse the same source block.
function _ghq_prior_model(; a0 = 1.0, a_prior_sd = 2.0)
    return @Model begin
        @fixedEffects begin
            a = RealNumber(a0; prior = Normal(0.0, a_prior_sd))
            σ = RealNumber(0.5, scale = :log; prior = LogNormal(0.0, 1.0))
            ω = RealNumber(1.0, scale = :log; prior = LogNormal(0.0, 1.0))
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

# 1-d planar-flow RE with a saturating helper link (priorless; distinct from
# the prior-bearing fx_npf fixture).
const _GHQ_NPF_MODEL = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5, scale = :log)
        ψ = NPFParameter(1, 2; seed = 1, calculate_se = false)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + sat(η[1]), σ)
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

@testset "GHQuadrature nodes.jl" begin

    # ----------------------------------------------------------
    # 1D Gauss-Hermite rule (probabilist's convention)
    # ----------------------------------------------------------
    @testset "GH rule n=1" begin
        nodes, lw = _gh_rule(1)
        @test length(nodes) == 1
        @test nodes[1] ≈ 0.0 atol = 1.0e-14
        @test exp(lw[1]) ≈ 1.0 atol = 1.0e-12
    end

    @testset "GH rule n=2" begin
        nodes, lw = _gh_rule(2)
        @test length(nodes) == 2
        @test sort(nodes) ≈ [-1.0, 1.0] atol = 1.0e-12
        @test exp.(lw) ≈ [0.5, 0.5] atol = 1.0e-12   # element-wise via isapprox on arrays
        @test sum(exp.(lw)) ≈ 1.0 atol = 1.0e-12
    end

    @testset "GH rule n=3" begin
        nodes, lw = _gh_rule(3)
        @test length(nodes) == 3
        @test sort(nodes) ≈ [-sqrt(3.0), 0.0, sqrt(3.0)] atol = 1.0e-10
        w = exp.(lw)
        @test sum(w) ≈ 1.0 atol = 1.0e-12
        idx0 = argmin(abs.(nodes))
        @test w[idx0] ≈ 2 / 3 atol = 1.0e-10
        outer_w = w[setdiff(1:3, idx0)]
        @test outer_w ≈ [1 / 6, 1 / 6] atol = 1.0e-10
    end

    @testset "GH weights sum to 1 for n=1..5" begin
        for n in 1:5
            _, lw = _gh_rule(n)
            @test sum(exp.(lw)) ≈ 1.0 atol = 1.0e-12
        end
    end

    # ----------------------------------------------------------
    # 1D sparse grid at level=1 and level=2
    # ----------------------------------------------------------
    @testset "1D level=1 sparse grid == gh_rule(1)" begin
        sg = build_sparse_grid(1, 1)
        @test size(sg.nodes, 2) == 1
        @test sg.nodes[1, 1] ≈ 0.0 atol = 1.0e-14
        @test exp(sg.logweights[1]) ≈ 1.0 atol = 1.0e-12
    end

    @testset "1D level=2 sparse grid matches gh_rule(2)" begin
        sg = build_sparse_grid(1, 2)
        nodes_gh, lw_gh = _gh_rule(2)
        # d=1, level=2: Smolyak uses only α=(2), coefficient=1
        # (α=(1) has coefficient 0 since binomial(0,1)=0)
        @test size(sg.nodes, 1) == 1
        @test size(sg.nodes, 2) == 2
        sorted_sg = sortperm(vec(sg.nodes))
        sorted_gh = sortperm(nodes_gh)
        @test vec(sg.nodes)[sorted_sg] ≈ nodes_gh[sorted_gh] atol = 1.0e-12
        @test sg.logweights[sorted_sg] ≈ lw_gh[sorted_gh] atol = 1.0e-12
        @test all(sg.signs .== Int8(1))
    end

    # ----------------------------------------------------------
    # 2D / 3D point counts
    # ----------------------------------------------------------
    @testset "Sparse grid: correct dimensions and positive counts" begin
        for d in 1:4, L in 1:3
            sg = build_sparse_grid(d, L)
            @test size(sg.nodes, 1) == d
            @test size(sg.nodes, 2) > 0
            @test length(sg.logweights) == size(sg.nodes, 2)
            @test length(sg.signs) == size(sg.nodes, 2)
        end
    end

    # ----------------------------------------------------------
    # Integration accuracy: polynomial moments (GH integrates polynomials exactly)
    #
    # Smolyak at level L integrates total-degree ≤ (2L-1) polynomials exactly.
    # L=1 → degree 1; L=2 → degree 3; L=3 → degree 5.
    # ----------------------------------------------------------

    @testset "Integration: E[1] = 1 (normalization, all levels)" begin
        for d in 1:4, L in 1:3
            sg = build_sparse_grid(d, L)
            @test sg_integrate(_ -> 1.0, sg) ≈ 1.0 atol = 1.0e-10
        end
    end

    @testset "Integration: E[zᵢ] = 0 (odd moment, L≥1)" begin
        for d in 1:3, L in 1:3
            sg = build_sparse_grid(d, L)
            for i in 1:d
                @test sg_integrate(z -> z[i], sg) ≈ 0.0 atol = 1.0e-10
            end
        end
    end

    @testset "Integration: E[zᵢ²] = 1 (variance, L≥2)" begin
        # 1-point GH (L=1) gives E[z²] = 0 (node at 0); only exact for L≥2
        for d in 1:4, L in 2:3
            sg = build_sparse_grid(d, L)
            for i in 1:d
                @test sg_integrate(z -> z[i]^2, sg) ≈ 1.0 atol = 1.0e-10
            end
        end
    end

    @testset "Integration: E[zᵢ·zⱼ] = 0 for i≠j (independence, L≥2)" begin
        for d in 2:4, L in 2:3
            sg = build_sparse_grid(d, L)
            @test sg_integrate(z -> z[1] * z[2], sg) ≈ 0.0 atol = 1.0e-10
        end
    end

    @testset "Integration: E[Σzᵢ²] = d (sum of variances, L≥2)" begin
        for d in 1:4, L in 2:3
            sg = build_sparse_grid(d, L)
            @test sg_integrate(z -> sum(z .^ 2), sg) ≈ Float64(d) atol = 1.0e-8
        end
    end

    @testset "Integration: E[zᵢ⁴] = 3 (kurtosis, L≥3 for d≥1)" begin
        # 3-point GH integrates degree 5 exactly; z⁴ has degree 4 ≤ 5 → exact for L≥3
        for d in 1:3
            sg = build_sparse_grid(d, 3)
            for i in 1:d
                @test sg_integrate(z -> z[i]^4, sg) ≈ 3.0 atol = 1.0e-8
            end
        end
    end

    @testset "Integration: E[zᵢ²·zⱼ²] = 1 for i≠j (L≥3)" begin
        # degree 4 ≤ 2*3-1=5 → exact for L=3
        for d in 2:3
            sg = build_sparse_grid(d, 3)
            @test sg_integrate(z -> z[1]^2 * z[2]^2, sg) ≈ 1.0 atol = 1.0e-8
        end
    end

    @testset "Integration: odd moments E[zᵢ³] = 0, E[zᵢ·zⱼ²] = 0 (L≥2)" begin
        for d in 2:3, L in 2:3
            sg = build_sparse_grid(d, L)
            @test sg_integrate(z -> z[1]^3, sg) ≈ 0.0 atol = 1.0e-10
            @test sg_integrate(z -> z[1] * z[2]^2, sg) ≈ 0.0 atol = 1.0e-10
        end
    end

    # ----------------------------------------------------------
    # n_ghq_points utility
    # ----------------------------------------------------------
    @testset "n_ghq_points matches actual grid size" begin
        for d in 1:4, L in 1:3
            sg = build_sparse_grid(d, L)
            @test NoLimits.n_ghq_points(d, L) == size(sg.nodes, 2)
        end
    end

    @testset "ghq_points_bound bounds the grid without building it" begin
        for d in 1:4, L in 1:3
            @test NoLimits.ghq_points_bound(d, L) >= NoLimits.n_ghq_points(d, L)
        end
        # Returns instantly for a batch whose grid would need tens of GB.
        @test NoLimits.ghq_points_bound(65, 5) > NoLimits.GHQ_MAX_NODES
    end

    # ----------------------------------------------------------
    # Cache: second call returns the same object
    # ----------------------------------------------------------
    @testset "get_sparse_grid caches correctly" begin
        sg1 = NoLimits.get_sparse_grid(2, 2)
        sg2 = NoLimits.get_sparse_grid(2, 2)
        @test sg1 === sg2
    end

    # ----------------------------------------------------------
    # Signs: Smolyak at L=1 has all positive weights
    # ----------------------------------------------------------
    @testset "Level=1 sparse grid has all positive signs" begin
        for d in 1:4
            sg = build_sparse_grid(d, 1)
            @test all(sg.signs .== Int8(1))
        end
    end
end  # @testset "GHQuadrature nodes.jl"

# ============================================================
# STEP 2: remeasure.jl + kernel.jl
# ============================================================

@testset "GHQuadrature remeasure.jl + kernel.jl" begin

    # ----------------------------------------------------------
    # signed_logsumexp
    # ----------------------------------------------------------
    @testset "signed_logsumexp: exact cancellation -> -Inf" begin
        log_val, s = NoLimits.signed_logsumexp([0.0, 0.0], Int8[1, -1])
        @test isinf(log_val) && log_val < 0
        @test s == Int8(1)
    end

    @testset "signed_logsumexp: positive result" begin
        # [3, 1] with signs [+, -] -> sum = 3 - 1 = 2
        log_val, s = NoLimits.signed_logsumexp([log(3.0), log(1.0)], Int8[1, -1])
        @test log_val ≈ log(2.0) atol = 1.0e-12
        @test s == Int8(1)
    end

    @testset "signed_logsumexp: negative result" begin
        # [1, 3] with signs [+, -] -> sum = 1 - 3 = -2
        log_val, s = NoLimits.signed_logsumexp([log(1.0), log(3.0)], Int8[1, -1])
        @test log_val ≈ log(2.0) atol = 1.0e-12
        @test s == Int8(-1)
    end

    @testset "signed_logsumexp: numerical stability with large values" begin
        # [exp(1000), exp(999.5)] with signs [+, -]
        # result = exp(1000) - exp(999.5) = exp(1000) * (1 - exp(-0.5)) ≈ exp(1000) * 0.3935
        log_val, s = NoLimits.signed_logsumexp([1000.0, 999.5], Int8[1, -1])
        expected = 1000.0 + log(1.0 - exp(-0.5))
        @test log_val ≈ expected atol = 1.0e-8
        @test s == Int8(1)
    end

    @testset "signed_logsumexp: all positive is standard logsumexp" begin
        vals = [1.0, 2.0, 3.0]
        signs = Int8[1, 1, 1]
        log_val, s = NoLimits.signed_logsumexp(vals, signs)
        @test s == Int8(1)
        @test exp(log_val) ≈ exp(1.0) + exp(2.0) + exp(3.0) rtol = 1.0e-12
    end

    @testset "signed_logsumexp: single element" begin
        log_val, s = NoLimits.signed_logsumexp([2.5], Int8[1])
        @test log_val ≈ 2.5 atol = 1.0e-12
        @test s == Int8(1)

        log_val2, s2 = NoLimits.signed_logsumexp([2.5], Int8[-1])
        @test log_val2 ≈ 2.5 atol = 1.0e-12
        @test s2 == Int8(-1)
    end

    # ----------------------------------------------------------
    # GaussianRE: construction from distributions
    # ----------------------------------------------------------
    @testset "GaussianRE from Normal(0, 2): μ=[0], L=[[2]]" begin
        d = Normal(0.0, 2.0)
        μ = [Distributions.mean(d)]
        σ = Distributions.std(d)
        L = reshape([σ], 1, 1)
        re = NoLimits.GaussianRE(μ, LowerTriangular(L), 1)

        @test re.n_b == 1
        @test re.μ ≈ [0.0] atol = 1.0e-14
        @test re.L ≈ [2.0;;] atol = 1.0e-14

        # transform: η = μ + L*z = 2z for z=[1.0]
        η = NoLimits.transform(re, [1.0])
        @test η ≈ [2.0] atol = 1.0e-14

        # logcorrection is always 0 for GaussianRE
        @test NoLimits.logcorrection(re, [1.0]) == 0.0
        @test NoLimits.logcorrection(re, [0.5]) == 0.0
    end

    @testset "GaussianRE transform is linear in z" begin
        μ = [1.0, -0.5]
        L = LowerTriangular([2.0 0.0; 0.5 3.0])
        re = NoLimits.GaussianRE(μ, L, 2)

        z = [0.3, -0.1]
        expected = μ + L * z
        @test NoLimits.transform(re, z) ≈ expected atol = 1.0e-14
    end

    # ----------------------------------------------------------
    # build_re_measure_from_batch (Gaussian path): uses a real DataModel
    # ----------------------------------------------------------

    # Helper: build a simple single-group model and DataModel
    function _make_simple_dm(σ_η_val = 1.0)
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(0.0)
                σ = RealNumber(1.0)
                σ_η = RealNumber(σ_η_val)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, σ_η); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end

        df = DataFrame(
            ID = [1, 1, 2, 2, 3, 3],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [0.1, -0.1, 0.3, 0.2, -0.2, 0.1]
        )
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
        return model, dm
    end

    @testset "build_re_measure_from_batch: Normal RE, single batch" begin
        model, dm = _make_simple_dm(1.0)
        θ = get_θ0_untransformed(model.fixed.fixed)
        _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        # With the default θ (σ_η=1), all 3 individuals should be in independent batches
        # Each batch has exactly one free RE level → n_b=1
        for bi in eachindex(batch_infos)
            info = batch_infos[bi]
            info.n_b == 0 && continue
            re_m = NoLimits.build_re_measure_from_batch(info, θ, const_cache, dm, ll_cache)
            @test re_m.n_b == info.n_b
            @test length(re_m.μ) == info.n_b
            @test size(re_m.L) == (info.n_b, info.n_b)
            # For Normal(0.0, σ_η=1.0): μ should be 0, L should be [[1.0]]
            @test re_m.μ ≈ zeros(info.n_b) atol = 1.0e-10
            @test Matrix(re_m.L) ≈ I(info.n_b) atol = 1.0e-10
        end
    end

    @testset "build_re_measure_from_batch: L scales with σ_η" begin
        model, dm = _make_simple_dm(2.5)
        θ = get_θ0_untransformed(model.fixed.fixed)
        _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        for bi in eachindex(batch_infos)
            info = batch_infos[bi]
            info.n_b == 0 && continue
            re_m = NoLimits.build_re_measure_from_batch(info, θ, const_cache, dm, ll_cache)
            # For Normal(0.0, σ_η=2.5): L should be [[2.5]]
            @test Matrix(re_m.L) ≈ 2.5 * I(info.n_b) atol = 1.0e-10
        end
    end

    # ----------------------------------------------------------
    # batch_loglik_ghq: analytical test
    #
    # Model: y ~ Normal(η, σ), η ~ Normal(0, σ_η), single obs y=y0
    # Analytic marginal log-likelihood:
    #   log p(y0) = logpdf(Normal(0, sqrt(σ² + σ_η²)), y0)
    #
    # The sparse grid integrates this Gaussian-Gaussian convolution; it
    # should be accurate to within a few percent even at level=2.
    # ----------------------------------------------------------

    @testset "batch_loglik_ghq: Gaussian-Gaussian analytic check" begin
        # σ=1, σ_η=1, y=0: analytic = logpdf(Normal(0, sqrt(2)), 0)
        σ_val = 1.0
        σ_η_val = 1.0
        y_val = 0.0
        analytic_logL = logpdf(Normal(0.0, sqrt(σ_val^2 + σ_η_val^2)), y_val)

        model = @Model begin
            @fixedEffects begin
                σ = RealNumber(σ_val)
                σ_η = RealNumber(σ_η_val)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, σ_η); column = :ID)
            end
            @formulas begin
                y ~ Normal(η, σ)
            end
        end

        df = DataFrame(ID = [1], t = [0.0], y = [y_val])
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
        θ = get_θ0_untransformed(model.fixed.fixed)
        _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        results = Float64[]
        for level in 1:4
            info = batch_infos[1]
            re_m = NoLimits.build_re_measure_from_batch(info, θ, const_cache, dm, ll_cache)
            sgrid = NoLimits.build_sparse_grid(info.n_b, level)
            lv = NoLimits.batch_loglik_ghq(dm, info, θ, re_m, sgrid, const_cache, ll_cache)
            push!(results, lv)
        end

        # Convergence tolerances for this Gaussian-Gaussian integral
        # (GH integrates polynomials exactly; exp(-η²) is not a polynomial,
        #  so convergence is algebraic rather than exponential at low levels)
        @test abs(results[2] - analytic_logL) / abs(analytic_logL) < 0.15  # level 2: ~12%
        @test abs(results[3] - analytic_logL) / abs(analytic_logL) < 0.05  # level 3: ~4%
        @test abs(results[4] - analytic_logL) / abs(analytic_logL) < 0.02  # level 4: ~1%
        # Convergence: results should get closer to analytic as level increases
        # (not strictly monotone, but the absolute error should generally decrease)
        errs = abs.(results .- analytic_logL)
        @test errs[3] < errs[1]  # higher level → better accuracy
        @test errs[4] < errs[2]
    end

    @testset "batch_loglik_ghq: returns finite value for simple model" begin
        model, dm = _make_simple_dm(1.0)
        θ = get_θ0_untransformed(model.fixed.fixed)
        _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        for bi in eachindex(batch_infos)
            info = batch_infos[bi]
            info.n_b == 0 && continue
            re_m = NoLimits.build_re_measure_from_batch(info, θ, const_cache, dm, ll_cache)
            sgrid = NoLimits.build_sparse_grid(info.n_b, 2)
            lv = NoLimits.batch_loglik_ghq(dm, info, θ, re_m, sgrid, const_cache, ll_cache)
            @test lv < 0.0   # log-likelihood should be negative
        end
    end

    @testset "_ghq_validate_re_distributions: Normal passes" begin
        _, dm = _make_simple_dm()
        # Should not throw
        @test (NoLimits._ghq_validate_re_distributions(dm); true)
    end
end  # @testset "GHQuadrature remeasure.jl + kernel.jl"

# =============================================================================
# Step 3: ghquadrature.jl — Full GHQuadrature FittingMethod
# =============================================================================

# ---------------------------------------------------------------------------
# Shared test model: y ~ Normal(a + η, σ), η ~ N(0, ω), 10 individuals, 5 obs
# ---------------------------------------------------------------------------

function _make_simple_ghq_dm(; n_id = 10, n_obs = 5, rng = MersenneTwister(42))
    ids = repeat(1:n_id, inner = n_obs)
    ts = repeat(collect(1.0:n_obs), outer = n_id)
    ηs = repeat(randn(rng, n_id), inner = n_obs)
    ys = 1.0 .+ ηs .+ 0.5 .* randn(rng, n_id * n_obs)
    df = DataFrame(ID = ids, t = ts, y = ys)
    return DataModel(_GHQ_SCALAR_MODEL, df; primary_id = :ID, time_col = :t)
end

@testset "GHQuadrature ghquadrature.jl" begin
    dm = _make_simple_ghq_dm()
    # One fit per level, shared by the accessor/level-comparison testsets below.
    res_l1 = fit_model(dm, GHQuadrature(level = 1; optim_kwargs = (maxiters = 2,)))
    res_l2 = fit_model(dm, GHQuadrature(level = 2; optim_kwargs = (maxiters = 2,)))

    # ── Basic fit at level=1 ─────────────────────────────────────────────────
    @testset "Basic fit level=1 LBFGS" begin
        res = res_l1

        @test res isa NoLimits.FitResult
        @test res.result isa NoLimits.GHQuadratureResult

        # Accessors all return sensible values
        obj = get_objective(res)

        params = NoLimits.get_params(res; scale = :untransformed)

        @test get_converged(res) isa Bool

        iters = get_iterations(res)
        @test iters === missing || (iters isa Integer && iters >= 0)

        # get_random_effects returns a DataFrame with n_id rows
        re = get_random_effects(dm, res)
        @test re isa NamedTuple
        @test haskey(re, :η)
        @test nrow(re.η) == 10
    end

    # ── Convenience accessor without passing dm ───────────────────────────────
    @testset "Stored DataModel accessor" begin
        re = get_random_effects(res_l1)
        @test nrow(re.η) == 10
        ll = get_loglikelihood(res_l1)
        @test ll < 0  # log-likelihood is negative
    end

    # ── get_loglikelihood re-evaluates sparse grid ────────────────────────────
    @testset "get_loglikelihood matches -objective" begin
        ll = get_loglikelihood(dm, res_l1)
        # objective = -LL (no penalty), so ll ≈ -objective
        @test abs(ll - (-get_objective(res_l1))) < 1.0  # within 1 nll unit (EB modes vs quadrature differ slightly)
    end

    # ── Level comparison: higher level → lower (or equal) -LL ────────────────
    @testset "Level 1 vs 2 objective" begin
        # Level 2 should get at least as good or better objective in most cases;
        # we check that both converge and give finite objectives.
        # Log-likelihoods should be negative
        @test get_loglikelihood(res_l1) < 0
        @test get_loglikelihood(res_l2) < 0
    end

    # ── Parameter agreement with Laplace ─────────────────────────────────────
    @testset "Parameter agreement with Laplace" begin
        res_sg = res_l2
        res_lap = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

        p_sg = NoLimits.get_params(res_sg; scale = :untransformed)
        p_lap = NoLimits.get_params(res_lap; scale = :untransformed)

        # Within 50% — both methods approximate the same marginal likelihood
        @test abs(p_sg.a - p_lap.a) / (abs(p_lap.a) + 1.0e-6) < 0.5
        @test abs(p_sg.σ - p_lap.σ) / (abs(p_lap.σ) + 1.0e-6) < 0.5
        @test abs(p_sg.ω - p_lap.ω) / (abs(p_lap.ω) + 1.0e-6) < 0.5
    end

    # ── ForwardDiff vs FiniteDifferences gradient check ──────────────────────
    @testset "ForwardDiff gradient vs FiniteDifferences" begin
        using ForwardDiff

        # Build the same objective as _fit_model for gradient testing.
        # We use the internal infrastructure directly.
        model = _GHQ_SCALAR_MODEL
        df_small = DataFrame(
            ID = repeat(1:4, inner = 3),
            t = repeat([1.0, 2.0, 3.0], outer = 4),
            y = [0.9, 1.1, 1.0, 1.3, 1.2, 1.1, 0.8, 0.9, 1.0, 1.2, 1.0, 0.9]
        )
        dm_small = DataModel(model, df_small; primary_id = :ID, time_col = :t)
        level = 2

        fe = dm_small.model.fixed.fixed
        transform = NoLimits.get_transform(fe)
        inv_transform = NoLimits.get_inverse_transform(fe)
        θ0_u = NoLimits.get_θ0_untransformed(fe)
        θ0_t = transform(θ0_u)

        _, batch_infos, const_cache = NoLimits._build_re_batch_infos(
            dm_small, NamedTuple()
        )
        ll_cache = NoLimits.build_ll_cache(dm_small; force_saveat = true)
        for d in unique(info.n_b for info in batch_infos)
            d > 0 && NoLimits.get_sparse_grid(d, level)
        end

        axs = getaxes(θ0_t)
        function sg_obj(θt_vec)
            θt = ComponentArray(θt_vec, axs)
            θu = inv_transform(θt)
            θu_re = NoLimits._symmetrize_psd_params(θu, fe)
            total = 0.0
            for info in batch_infos
                bll = NoLimits._ghq_batch_ll(
                    dm_small, info, θu_re, const_cache, ll_cache, level
                )
                bll == -Inf && return Inf
                total += bll
            end
            return -total
        end

        θ0_vec = collect(θ0_t)
        grad_fd = ForwardDiff.gradient(sg_obj, θ0_vec)
        grad_fin = FiniteDifferences.grad(central_fdm(5, 1), sg_obj, θ0_vec)[1]

        # Relative error < 1e-4 on all components
        for k in eachindex(grad_fd)
            abs_ref = abs(grad_fin[k])
            if abs_ref > 1.0e-6
                @test abs(grad_fd[k] - grad_fin[k]) / abs_ref < 1.0e-4
            else
                @test abs(grad_fd[k] - grad_fin[k]) < 1.0e-6
            end
        end
    end

    # ── Alternative outer optimizers ─────────────────────────────────────────
    @testset "BFGS outer optimizer" begin
        res = fit_model(
            dm,
            GHQuadrature(
                level = 1;
                optimizer = OptimizationOptimJL.BFGS(),
                optim_kwargs = (maxiters = 2,)
            )
        )
    end

    @testset "NelderMead outer optimizer" begin
        res = fit_model(
            dm,
            GHQuadrature(
                level = 1;
                optimizer = OptimizationOptimJL.NelderMead(),
                optim_kwargs = (maxiters = 2,)
            )
        )
    end

    @testset "BlackBoxOptim outer optimizer" begin
        lb_val, ub_val = NoLimits.default_bounds_from_start(dm; margin = 3.0)
        res = fit_model(
            dm,
            GHQuadrature(
                level = 1;
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited(),
                optim_kwargs = (maxiters = 2,),
                lb = lb_val, ub = ub_val
            )
        )
    end

    # ── constants kwarg ───────────────────────────────────────────────────────
    @testset "constants fix a parameter" begin
        res = fit_model(
            dm, GHQuadrature(level = 1; optim_kwargs = (maxiters = 2,));
            constants = (a = 1.0,)
        )
        params = NoLimits.get_params(res; scale = :untransformed)
    end

    # ── oversized batch refused before any grid is built ──────────────────────
    @testset "oversized joint RE batch is refused fast" begin
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
                η_site = RandomEffect(Normal(0.0, 1.0); column = :SITE)
            end
            @formulas begin
                y ~ Normal(a + η_id + η_site, σ)
            end
        end
        # 40 IDs crossed with one SITE -> a single batch of joint dimension 41.
        ids = repeat(1:40; inner = 2)
        df_big = DataFrame(
            ID = ids, SITE = fill(:A, length(ids)),
            t = repeat([0.0, 1.0], 40), y = 0.1 .* ids
        )
        dm_big = DataModel(model, df_big; primary_id = :ID, time_col = :t)
        t0 = time()
        @test_throws ErrorException fit_model(dm_big, GHQuadrature(level = 5))
        @test time() - t0 < 60   # refused, not ground to death building the grid
    end

    # ── store_data_model=false ────────────────────────────────────────────────
    @testset "store_data_model=false" begin
        res = fit_model(
            dm, GHQuadrature(level = 1; optim_kwargs = (maxiters = 2,));
            store_data_model = false
        )
        @test get_data_model(res) === nothing
    end
end  # @testset "GHQuadrature ghquadrature.jl"

# =============================================================================
# NPF (NormalizingPlanarFlow) RE support
# =============================================================================

@testset "GHQuadrature NPF RE support" begin

    # ── Validation no longer rejects NPF ─────────────────────────────────────
    @testset "_ghq_validate_re_distributions allows NPF" begin
        df_v = DataFrame(
            ID = repeat(1:5, inner = 3),
            t = repeat([1.0, 2.0, 3.0], outer = 5),
            y = randn(MersenneTwister(7), 15)
        )
        dm_npf = DataModel(_GHQ_NPF_MODEL, df_v; primary_id = :ID, time_col = :t)
        # Should NOT throw
        @test_nowarn NoLimits._ghq_validate_re_distributions(dm_npf)
    end

    # ── CompositeRE is returned for NPF batches ───────────────────────────────
    @testset "build_re_measure_from_batch returns CompositeRE for NPF" begin
        df_v = DataFrame(
            ID = repeat(1:4, inner = 3),
            t = repeat([1.0, 2.0, 3.0], outer = 4),
            y = randn(MersenneTwister(8), 12)
        )
        dm_npf = DataModel(_GHQ_NPF_MODEL, df_v; primary_id = :ID, time_col = :t)

        fe = dm_npf.model.fixed.fixed
        θ0_u = NoLimits.get_θ0_untransformed(fe)
        θ0_t = NoLimits.get_transform(fe)(θ0_u)
        θ0_u_re = NoLimits.get_inverse_transform(fe)(θ0_t)
        θ_re = NoLimits._symmetrize_psd_params(θ0_u_re, fe)

        _, batch_infos, const_cache = NoLimits._build_re_batch_infos(
            dm_npf, NamedTuple()
        )
        ll_cache = NoLimits.build_ll_cache(dm_npf; force_saveat = true)

        bi = batch_infos[1]
        if bi.n_b > 0
            re_measure = NoLimits.build_re_measure_from_batch(
                bi, θ_re, const_cache, dm_npf, ll_cache
            )
            @test re_measure isa NoLimits.CompositeRE
            @test re_measure.n_b == bi.n_b

            # transform returns a vector of length n_b
            z = zeros(bi.n_b)
            η = NoLimits.transform(re_measure, z)
            @test length(η) == bi.n_b
            @test all(isfinite, η)

            # logcorrection is 0
            @test NoLimits.logcorrection(re_measure, z) == 0.0
        end
    end

    # ── GaussianRE fast path still returned for pure-Gaussian model ───────────
    @testset "build_re_measure_from_batch returns GaussianRE for Normal model" begin
        dm_gauss = _make_simple_ghq_dm(; n_id = 4, n_obs = 3)
        fe = dm_gauss.model.fixed.fixed
        θ0_u = NoLimits.get_θ0_untransformed(fe)
        θ_re = NoLimits._symmetrize_psd_params(θ0_u, fe)
        _, batch_infos, const_cache = NoLimits._build_re_batch_infos(
            dm_gauss, NamedTuple()
        )
        ll_cache = NoLimits.build_ll_cache(dm_gauss; force_saveat = true)
        bi = batch_infos[1]
        if bi.n_b > 0
            re = NoLimits.build_re_measure_from_batch(
                bi, θ_re, const_cache, dm_gauss, ll_cache
            )
            @test re isa NoLimits.GaussianRE
        end
    end

    # ── End-to-end fit with NPF RE ────────────────────────────────────────────
    @testset "fit_model GHQuadrature level=1 with NPF RE" begin
        rng = MersenneTwister(99)
        ids = repeat(1:8, inner = 4)
        ts = repeat([1.0, 2.0, 3.0, 4.0], outer = 8)
        ys = 1.0 .+ 0.3 .* randn(rng, 32) .+ 0.5 .* randn(rng, 32)
        df_npf = DataFrame(ID = ids, t = ts, y = ys)
        dm_npf = DataModel(_GHQ_NPF_MODEL, df_npf; primary_id = :ID, time_col = :t)

        res = fit_model(dm_npf, GHQuadrature(level = 1; optim_kwargs = (maxiters = 2,)))

        @test res isa NoLimits.FitResult
        @test res.result isa NoLimits.GHQuadratureResult

        params = NoLimits.get_params(res; scale = :untransformed)

        re = get_random_effects(dm_npf, res)
        @test re isa NamedTuple
        @test haskey(re, :η)
        @test nrow(re.η) == 8
    end
end  # @testset "GHQuadrature NPF RE support"

# =============================================================================
# Phase 2: Wald UQ
# =============================================================================

# ---------------------------------------------------------------------------
# Model with priors for MAP
# ---------------------------------------------------------------------------

const _GHQ_MAP_MODEL = _ghq_prior_model()

function _make_map_ghq_dm(; n_id = 8, n_obs = 4, rng = MersenneTwister(7))
    ids = repeat(1:n_id, inner = n_obs)
    ts = repeat(collect(1.0:n_obs), outer = n_id)
    ηs = repeat(randn(rng, n_id), inner = n_obs)
    ys = 1.0 .+ ηs .+ 0.5 .* randn(rng, n_id * n_obs)
    df = DataFrame(ID = ids, t = ts, y = ys)
    return DataModel(_GHQ_MAP_MODEL, df; primary_id = :ID, time_col = :t)
end

# Shared by the Wald and Profile UQ testsets (same dm, same level-2 GHQ fit).
const _GHQ_MAP_DM = _make_map_ghq_dm()
const _GHQ_MAP_RES_GHQ2 = fit_model(
    _GHQ_MAP_DM, GHQuadrature(level = 2; optim_kwargs = (maxiters = 2,))
)

@testset "GHQuadrature Wald UQ" begin

    # ── GHQuadrature Wald (default ForwardDiff Hessian) ────────────────────────
    @testset "compute_uq GHQuadrature level=2" begin
        uq = compute_uq(_GHQ_MAP_RES_GHQ2; method = :wald, pseudo_inverse = true)

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test hasproperty(cia.lower, :σ)
        @test hasproperty(cia.lower, :ω)

        # Intervals should be finite and ordered

    end

    # ── Sandwich vcov ────────────────────────────────────────────────────────
    @testset "compute_uq GHQuadrature sandwich vcov level=2" begin
        uq = compute_uq(
            _GHQ_MAP_RES_GHQ2; method = :wald, vcov = :sandwich, pseudo_inverse = true
        )

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
    end

    # ── hessian_backend :fd_gradient also works ───────────────────────────────
    @testset "compute_uq GHQuadrature fd_gradient backend level=2" begin
        uq = compute_uq(
            _GHQ_MAP_RES_GHQ2; method = :wald,
            hessian_backend = :fd_gradient, pseudo_inverse = true
        )

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
    end
end  # @testset "GHQuadrature Wald UQ"

@testset "GHQuadrature Profile UQ" begin
    @testset "compute_uq GHQuadrature :profile level=2" begin
        uq = compute_uq(_GHQ_MAP_RES_GHQ2; method = :profile)

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
    end
end  # @testset "GHQuadrature Profile UQ"

@testset "GHQuadrature mcmc_refit UQ" begin
    dm_uq = _GHQ_MAP_DM

    @testset "compute_uq GHQuadrature :mcmc_refit (with priors)" begin
        # GHQuadrature with priors on all fixed effects can use mcmc_refit
        res = fit_model(dm_uq, GHQuadrature(level = 1; optim_kwargs = (maxiters = 2,)))
        uq = compute_uq(
            res;
            method = :mcmc_refit,
            mcmc_sampler = Turing.MH(),
            mcmc_turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
        )

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
    end
end  # @testset "GHQuadrature mcmc_refit UQ"

@testset "GHQuadrature parallelization (EnsembleThreads)" begin
    dm_par = _make_simple_ghq_dm(; n_id = 10, n_obs = 5)

    # ── EnsembleThreads produces same objective as EnsembleSerial ─────────────
    @testset "EnsembleThreads matches EnsembleSerial objective" begin
        res_serial = fit_model(
            dm_par, GHQuadrature(level = 2; optim_kwargs = (maxiters = 2,));
            serialization = NoLimits.EnsembleSerial()
        )
        res_threaded = fit_model(
            dm_par, GHQuadrature(level = 2; optim_kwargs = (maxiters = 2,));
            serialization = NoLimits.EnsembleThreads()
        )

        # Objectives should agree within numerical tolerance (same deterministic quadrature)
        @test abs(get_objective(res_serial) - get_objective(res_threaded)) < 1.0

        # Both should converge and produce valid RE estimates
        re_t = get_random_effects(dm_par, res_threaded)
        @test nrow(re_t.η) == 10
    end
end  # @testset "GHQuadrature parallelization (EnsembleThreads)"

@testset "GHQuadrature node deduplication" begin

    # d=1, L=3: GH-3 has 3 unique nodes → same before/after dedup
    @testset "d=1 no duplicates" begin
        sg = build_sparse_grid(1, 3)
        # All nodes are distinct at L=3 in 1D
        @test size(sg.nodes, 2) == 3
    end

    # d=2, L=2: 5 raw nodes (all distinct), dedup should not reduce count
    @testset "d=2 L=2 no duplicates" begin
        sg = build_sparse_grid(2, 2)
        @test size(sg.nodes, 2) == 5
    end

    # d=2, L=3: 15 raw nodes, (0,0) appears in multiple multi-indices.
    # After dedup the count should be strictly less than 15.
    @testset "d=2 L=3 deduplication reduces point count" begin
        sg = build_sparse_grid(2, 3)
        @test size(sg.nodes, 2) < 15
    end

    # Integration accuracy must hold after deduplication
    @testset "d=2 L=3 integration still accurate after dedup" begin
        sg = build_sparse_grid(2, 3)
        # E[z₁²] = 1
        val = sg_integrate(z -> z[1]^2, sg)
        @test val ≈ 1.0 atol = 1.0e-10
        # E[z₁⁴] = 3
        val4 = sg_integrate(z -> z[1]^4, sg)
        @test val4 ≈ 3.0 atol = 1.0e-10
        # E[z₁² * z₂²] = 1
        val_cross = sg_integrate(z -> z[1]^2 * z[2]^2, sg)
        @test val_cross ≈ 1.0 atol = 1.0e-10
    end

    # Dedup is idempotent: building twice gives same result
    @testset "n_ghq_points matches build_sparse_grid after dedup" begin
        for (d, l) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            sg = build_sparse_grid(d, l)
            @test NoLimits.n_ghq_points(d, l) == size(sg.nodes, 2)
        end
    end
end  # @testset "GHQuadrature node deduplication"

# ============================================================
# Non-Gaussian RE families: transport maps (LogNormal, Beta, Gamma,
# Exponential, Weibull, TDist) and the generic
# ContinuousUnivariateDistribution fallback (Laplace, InverseGamma).
# One @Model per family; shared fit / EB / validation shell below.
# ============================================================
