using Test
using NoLimits
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
const _gh_rule           = NoLimits._gh_rule
const build_sparse_grid  = NoLimits.build_sparse_grid

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
        @test nodes[1] ≈ 0.0   atol=1e-14
        @test exp(lw[1]) ≈ 1.0  atol=1e-12
    end

    @testset "GH rule n=2" begin
        nodes, lw = _gh_rule(2)
        @test length(nodes) == 2
        @test sort(nodes) ≈ [-1.0, 1.0]      atol=1e-12
        @test exp.(lw) ≈ [0.5, 0.5]          atol=1e-12   # element-wise via isapprox on arrays
        @test sum(exp.(lw)) ≈ 1.0             atol=1e-12
    end

    @testset "GH rule n=3" begin
        nodes, lw = _gh_rule(3)
        @test length(nodes) == 3
        @test sort(nodes) ≈ [-sqrt(3.0), 0.0, sqrt(3.0)]  atol=1e-10
        w = exp.(lw)
        @test sum(w) ≈ 1.0  atol=1e-12
        idx0 = argmin(abs.(nodes))
        @test w[idx0] ≈ 2/3  atol=1e-10
        outer_w = w[setdiff(1:3, idx0)]
        @test outer_w ≈ [1/6, 1/6]  atol=1e-10
    end

    @testset "GH weights sum to 1 for n=1..5" begin
        for n in 1:5
            _, lw = _gh_rule(n)
            @test sum(exp.(lw)) ≈ 1.0  atol=1e-12
        end
    end

    # ----------------------------------------------------------
    # 1D sparse grid at level=1 and level=2
    # ----------------------------------------------------------
    @testset "1D level=1 sparse grid == gh_rule(1)" begin
        sg = build_sparse_grid(1, 1)
        @test size(sg.nodes, 2) == 1
        @test sg.nodes[1, 1] ≈ 0.0  atol=1e-14
        @test exp(sg.logweights[1]) ≈ 1.0  atol=1e-12
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
        @test vec(sg.nodes)[sorted_sg] ≈ nodes_gh[sorted_gh]  atol=1e-12
        @test sg.logweights[sorted_sg]  ≈ lw_gh[sorted_gh]   atol=1e-12
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
            @test sg_integrate(_ -> 1.0, sg) ≈ 1.0  atol=1e-10
        end
    end

    @testset "Integration: E[zᵢ] = 0 (odd moment, L≥1)" begin
        for d in 1:3, L in 1:3
            sg = build_sparse_grid(d, L)
            for i in 1:d
                @test sg_integrate(z -> z[i], sg) ≈ 0.0  atol=1e-10
            end
        end
    end

    @testset "Integration: E[zᵢ²] = 1 (variance, L≥2)" begin
        # 1-point GH (L=1) gives E[z²] = 0 (node at 0); only exact for L≥2
        for d in 1:4, L in 2:3
            sg = build_sparse_grid(d, L)
            for i in 1:d
                @test sg_integrate(z -> z[i]^2, sg) ≈ 1.0  atol=1e-10
            end
        end
    end

    @testset "Integration: E[zᵢ·zⱼ] = 0 for i≠j (independence, L≥2)" begin
        for d in 2:4, L in 2:3
            sg = build_sparse_grid(d, L)
            @test sg_integrate(z -> z[1] * z[2], sg) ≈ 0.0  atol=1e-10
        end
    end

    @testset "Integration: E[Σzᵢ²] = d (sum of variances, L≥2)" begin
        for d in 1:4, L in 2:3
            sg = build_sparse_grid(d, L)
            @test sg_integrate(z -> sum(z .^ 2), sg) ≈ Float64(d)  atol=1e-8
        end
    end

    @testset "Integration: E[zᵢ⁴] = 3 (kurtosis, L≥3 for d≥1)" begin
        # 3-point GH integrates degree 5 exactly; z⁴ has degree 4 ≤ 5 → exact for L≥3
        for d in 1:3
            sg = build_sparse_grid(d, 3)
            for i in 1:d
                @test sg_integrate(z -> z[i]^4, sg) ≈ 3.0  atol=1e-8
            end
        end
    end

    @testset "Integration: E[zᵢ²·zⱼ²] = 1 for i≠j (L≥3)" begin
        # degree 4 ≤ 2*3-1=5 → exact for L=3
        for d in 2:3
            sg = build_sparse_grid(d, 3)
            @test sg_integrate(z -> z[1]^2 * z[2]^2, sg) ≈ 1.0  atol=1e-8
        end
    end

    @testset "Integration: odd moments E[zᵢ³] = 0, E[zᵢ·zⱼ²] = 0 (L≥2)" begin
        for d in 2:3, L in 2:3
            sg = build_sparse_grid(d, L)
            @test sg_integrate(z -> z[1]^3, sg)          ≈ 0.0  atol=1e-10
            @test sg_integrate(z -> z[1] * z[2]^2, sg)   ≈ 0.0  atol=1e-10
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
        @test log_val ≈ log(2.0)  atol=1e-12
        @test s == Int8(1)
    end

    @testset "signed_logsumexp: negative result" begin
        # [1, 3] with signs [+, -] -> sum = 1 - 3 = -2
        log_val, s = NoLimits.signed_logsumexp([log(1.0), log(3.0)], Int8[1, -1])
        @test log_val ≈ log(2.0)  atol=1e-12
        @test s == Int8(-1)
    end

    @testset "signed_logsumexp: numerical stability with large values" begin
        # [exp(1000), exp(999.5)] with signs [+, -]
        # result = exp(1000) - exp(999.5) = exp(1000) * (1 - exp(-0.5)) ≈ exp(1000) * 0.3935
        log_val, s = NoLimits.signed_logsumexp([1000.0, 999.5], Int8[1, -1])
        expected = 1000.0 + log(1.0 - exp(-0.5))
        @test log_val ≈ expected  atol=1e-8
        @test s == Int8(1)
    end

    @testset "signed_logsumexp: all positive is standard logsumexp" begin
        vals = [1.0, 2.0, 3.0]
        signs = Int8[1, 1, 1]
        log_val, s = NoLimits.signed_logsumexp(vals, signs)
        @test s == Int8(1)
        @test exp(log_val) ≈ exp(1.0) + exp(2.0) + exp(3.0)  rtol=1e-12
    end

    @testset "signed_logsumexp: single element" begin
        log_val, s = NoLimits.signed_logsumexp([2.5], Int8[1])
        @test log_val ≈ 2.5  atol=1e-12
        @test s == Int8(1)

        log_val2, s2 = NoLimits.signed_logsumexp([2.5], Int8[-1])
        @test log_val2 ≈ 2.5  atol=1e-12
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
        @test re.μ ≈ [0.0]   atol=1e-14
        @test re.L ≈ [2.0;;]  atol=1e-14

        # transform: η = μ + L*z = 2z for z=[1.0]
        η = NoLimits.transform(re, [1.0])
        @test η ≈ [2.0]  atol=1e-14

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
        @test NoLimits.transform(re, z) ≈ expected  atol=1e-14
    end

    # ----------------------------------------------------------
    # build_gaussian_re_from_batch: uses a real DataModel
    # ----------------------------------------------------------

    # Helper: build a simple single-group model and DataModel
    function _make_simple_dm(σ_η_val=1.0)
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
                η = RandomEffect(Normal(0.0, σ_η); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end

        df = DataFrame(
            ID = [1, 1, 2, 2, 3, 3],
            t  = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y  = [0.1, -0.1, 0.3, 0.2, -0.2, 0.1],
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        return model, dm
    end

    @testset "build_gaussian_re_from_batch: Normal RE, single batch" begin
        model, dm = _make_simple_dm(1.0)
        θ = get_θ0_untransformed(model.fixed.fixed)
        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        # With the default θ (σ_η=1), all 3 individuals should be in independent batches
        # Each batch has exactly one free RE level → n_b=1
        for bi in eachindex(batch_infos)
            info = batch_infos[bi]
            info.n_b == 0 && continue
            re_m = NoLimits.build_gaussian_re_from_batch(info, θ, const_cache, dm, ll_cache)
            @test re_m.n_b == info.n_b
            @test length(re_m.μ) == info.n_b
            @test size(re_m.L) == (info.n_b, info.n_b)
            # For Normal(0.0, σ_η=1.0): μ should be 0, L should be [[1.0]]
            @test re_m.μ ≈ zeros(info.n_b)   atol=1e-10
            @test Matrix(re_m.L) ≈ I(info.n_b)  atol=1e-10
        end
    end

    @testset "build_gaussian_re_from_batch: L scales with σ_η" begin
        model, dm = _make_simple_dm(2.5)
        θ = get_θ0_untransformed(model.fixed.fixed)
        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        for bi in eachindex(batch_infos)
            info = batch_infos[bi]
            info.n_b == 0 && continue
            re_m = NoLimits.build_gaussian_re_from_batch(info, θ, const_cache, dm, ll_cache)
            # For Normal(0.0, σ_η=2.5): L should be [[2.5]]
            @test Matrix(re_m.L) ≈ 2.5 * I(info.n_b)  atol=1e-10
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
        σ_val  = 1.0
        σ_η_val = 1.0
        y_val   = 0.0
        analytic_logL = logpdf(Normal(0.0, sqrt(σ_val^2 + σ_η_val^2)), y_val)

        model = @Model begin
            @fixedEffects begin
                σ   = RealNumber(σ_val)
                σ_η = RealNumber(σ_η_val)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, σ_η); column=:ID)
            end
            @formulas begin
                y ~ Normal(η, σ)
            end
        end

        df = DataFrame(ID=[1], t=[0.0], y=[y_val])
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        θ  = get_θ0_untransformed(model.fixed.fixed)
        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        results = Float64[]
        for level in 1:4
            info = batch_infos[1]
            re_m = NoLimits.build_gaussian_re_from_batch(info, θ, const_cache, dm, ll_cache)
            sgrid = NoLimits.build_sparse_grid(info.n_b, level)
            lv = NoLimits.batch_loglik_ghq(dm, info, θ, re_m, sgrid, const_cache, ll_cache)
            push!(results, lv)
            @test isfinite(lv)   # must be finite at all levels
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
        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)

        for bi in eachindex(batch_infos)
            info = batch_infos[bi]
            info.n_b == 0 && continue
            re_m  = NoLimits.build_gaussian_re_from_batch(info, θ, const_cache, dm, ll_cache)
            sgrid = NoLimits.build_sparse_grid(info.n_b, 2)
            lv = NoLimits.batch_loglik_ghq(dm, info, θ, re_m, sgrid, const_cache, ll_cache)
            @test isfinite(lv)
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

function _make_simple_ghq_dm(; n_id=10, n_obs=5, rng=MersenneTwister(42))
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
            ω = RealNumber(1.0, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    ids  = repeat(1:n_id, inner=n_obs)
    ts   = repeat(collect(1.0:n_obs), outer=n_id)
    ηs   = repeat(randn(rng, n_id), inner=n_obs)
    ys   = 1.0 .+ ηs .+ 0.5 .* randn(rng, n_id * n_obs)
    df   = DataFrame(ID=ids, t=ts, y=ys)
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "GHQuadrature ghquadrature.jl" begin

    dm = _make_simple_ghq_dm()

    # ── Basic fit at level=1 ─────────────────────────────────────────────────
    @testset "Basic fit level=1 LBFGS" begin
        res = fit_model(dm, GHQuadrature(level=1; optim_kwargs=(maxiters=15,)))

        @test res isa NoLimits.FitResult
        @test res.result isa NoLimits.GHQuadratureResult

        # Accessors all return sensible values
        obj = get_objective(res)
        @test isfinite(obj)
        @test obj > 0   # negative log-likelihood > 0

        params = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(params.a)
        @test isfinite(params.σ) && params.σ > 0
        @test isfinite(params.ω) && params.ω > 0

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
        res = fit_model(dm, GHQuadrature(level=1; optim_kwargs=(maxiters=100,)))
        re  = get_random_effects(res)
        @test nrow(re.η) == 10
        ll = get_loglikelihood(res)
        @test isfinite(ll)
        @test ll < 0  # log-likelihood is negative
    end

    # ── get_loglikelihood re-evaluates sparse grid ────────────────────────────
    @testset "get_loglikelihood matches -objective" begin
        res = fit_model(dm, GHQuadrature(level=1; optim_kwargs=(maxiters=15,)))
        ll  = get_loglikelihood(dm, res)
        @test isfinite(ll)
        # objective = -LL (no penalty), so ll ≈ -objective
        @test abs(ll - (-get_objective(res))) < 1.0  # within 1 nll unit (EB modes vs quadrature differ slightly)
    end

    # ── Level comparison: higher level → lower (or equal) -LL ────────────────
    @testset "Level 1 vs 2 objective" begin
        res1 = fit_model(dm, GHQuadrature(level=1; optim_kwargs=(maxiters=25,)))
        res2 = fit_model(dm, GHQuadrature(level=2; optim_kwargs=(maxiters=25,)))
        # Level 2 should get at least as good or better objective in most cases;
        # we check that both converge and give finite objectives.
        @test isfinite(get_objective(res1))
        @test isfinite(get_objective(res2))
        # Log-likelihoods should be negative
        @test get_loglikelihood(res1) < 0
        @test get_loglikelihood(res2) < 0
    end

    # ── Parameter agreement with Laplace ─────────────────────────────────────
    @testset "Parameter agreement with Laplace" begin
        res_sg  = fit_model(dm, GHQuadrature(level=2; optim_kwargs=(maxiters=25,)))
        res_lap = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=25,)))

        p_sg  = NoLimits.get_params(res_sg;  scale=:untransformed)
        p_lap = NoLimits.get_params(res_lap; scale=:untransformed)

        # Within 50% — both methods approximate the same marginal likelihood
        @test abs(p_sg.a  - p_lap.a)  / (abs(p_lap.a)  + 1e-6) < 0.5
        @test abs(p_sg.σ  - p_lap.σ)  / (abs(p_lap.σ)  + 1e-6) < 0.5
        @test abs(p_sg.ω  - p_lap.ω)  / (abs(p_lap.ω)  + 1e-6) < 0.5
    end

    # ── ForwardDiff vs FiniteDifferences gradient check ──────────────────────
    @testset "ForwardDiff gradient vs FiniteDifferences" begin
        using ForwardDiff

        # Build the same objective as _fit_model for gradient testing.
        # We use the internal infrastructure directly.
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.5, scale=:log)
                ω = RealNumber(1.0, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
        df_small = DataFrame(ID=repeat(1:4, inner=3),
                              t=repeat([1.0,2.0,3.0], outer=4),
                              y=[0.9, 1.1, 1.0, 1.3, 1.2, 1.1, 0.8, 0.9, 1.0, 1.2, 1.0, 0.9])
        dm_small = DataModel(model, df_small; primary_id=:ID, time_col=:t)
        level = 2

        fe = dm_small.model.fixed.fixed
        transform     = NoLimits.get_transform(fe)
        inv_transform = NoLimits.get_inverse_transform(fe)
        θ0_u = NoLimits.get_θ0_untransformed(fe)
        θ0_t = transform(θ0_u)

        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm_small, NamedTuple())
        ll_cache = NoLimits.build_ll_cache(dm_small; force_saveat=true)
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
                bll = NoLimits._ghq_batch_ll(dm_small, info, θu_re, const_cache, ll_cache, level)
                bll == -Inf && return Inf
                total += bll
            end
            return -total
        end

        θ0_vec = collect(θ0_t)
        grad_fd  = ForwardDiff.gradient(sg_obj, θ0_vec)
        grad_fin = FiniteDifferences.grad(central_fdm(5, 1), sg_obj, θ0_vec)[1]

        # Relative error < 1e-4 on all components
        for k in eachindex(grad_fd)
            abs_ref = abs(grad_fin[k])
            if abs_ref > 1e-6
                @test abs(grad_fd[k] - grad_fin[k]) / abs_ref < 1e-4
            else
                @test abs(grad_fd[k] - grad_fin[k]) < 1e-6
            end
        end
    end

    # ── Alternative outer optimizers ─────────────────────────────────────────
    @testset "BFGS outer optimizer" begin
        res = fit_model(dm, GHQuadrature(level=1;
                            optimizer=OptimizationOptimJL.BFGS(),
                            optim_kwargs=(maxiters=15,)))
        @test isfinite(get_objective(res))
        @test NoLimits.get_params(res; scale=:untransformed).σ > 0
    end

    @testset "NelderMead outer optimizer" begin
        res = fit_model(dm, GHQuadrature(level=1;
                            optimizer=OptimizationOptimJL.NelderMead(),
                            optim_kwargs=(maxiters=30,)))
        @test isfinite(get_objective(res))
        @test NoLimits.get_params(res; scale=:untransformed).σ > 0
    end

    @testset "BlackBoxOptim outer optimizer" begin
        lb_val, ub_val = NoLimits.default_bounds_from_start(dm; margin=3.0)
        res = fit_model(dm, GHQuadrature(level=1;
                            optimizer=BBO_adaptive_de_rand_1_bin_radiuslimited(),
                            optim_kwargs=(maxiters=30,),
                            lb=lb_val, ub=ub_val))
        @test isfinite(get_objective(res))
        @test NoLimits.get_params(res; scale=:untransformed).σ > 0
    end

    # ── constants kwarg ───────────────────────────────────────────────────────
    @testset "constants fix a parameter" begin
        res = fit_model(dm, GHQuadrature(level=1; optim_kwargs=(maxiters=15,));
                        constants=(a=1.0,))
        params = NoLimits.get_params(res; scale=:untransformed)
        @test params.a ≈ 1.0
        @test isfinite(params.σ)
    end

    # ── store_data_model=false ────────────────────────────────────────────────
    @testset "store_data_model=false" begin
        res = fit_model(dm, GHQuadrature(level=1; optim_kwargs=(maxiters=100,));
                        store_data_model=false)
        @test get_data_model(res) === nothing
    end

end  # @testset "GHQuadrature ghquadrature.jl"

# =============================================================================
# NPF (NormalizingPlanarFlow) RE support
# =============================================================================

@testset "GHQuadrature NPF RE support" begin

    # ── Validation no longer rejects NPF ─────────────────────────────────────
    @testset "_ghq_validate_re_distributions allows NPF" begin
        model_npf = @Model begin
            @helpers begin
                sat(u) = u / (1 + abs(u))
            end
            @fixedEffects begin
                a  = RealNumber(1.0)
                σ  = RealNumber(0.5, scale=:log)
                ψ  = NPFParameter(1, 2; seed=1, calculate_se=false)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + sat(η), σ)
            end
        end
        df_v = DataFrame(ID=repeat(1:5, inner=3),
                         t=repeat([1.0,2.0,3.0], outer=5),
                         y=randn(MersenneTwister(7), 15))
        dm_npf = DataModel(model_npf, df_v; primary_id=:ID, time_col=:t)
        # Should NOT throw
        @test_nowarn NoLimits._ghq_validate_re_distributions(dm_npf)
    end

    # ── CompositeRE is returned for NPF batches ───────────────────────────────
    @testset "build_re_measure_from_batch returns CompositeRE for NPF" begin
        model_npf = @Model begin
            @helpers begin
                sat(u) = u / (1 + abs(u))
            end
            @fixedEffects begin
                a  = RealNumber(1.0)
                σ  = RealNumber(0.5, scale=:log)
                ψ  = NPFParameter(1, 2; seed=1, calculate_se=false)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + sat(η), σ)
            end
        end
        df_v = DataFrame(ID=repeat(1:4, inner=3),
                         t=repeat([1.0,2.0,3.0], outer=4),
                         y=randn(MersenneTwister(8), 12))
        dm_npf = DataModel(model_npf, df_v; primary_id=:ID, time_col=:t)

        fe = dm_npf.model.fixed.fixed
        θ0_u = NoLimits.get_θ0_untransformed(fe)
        θ0_t = NoLimits.get_transform(fe)(θ0_u)
        θ0_u_re = NoLimits.get_inverse_transform(fe)(θ0_t)
        θ_re = NoLimits._symmetrize_psd_params(θ0_u_re, fe)

        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm_npf, NamedTuple())
        ll_cache = NoLimits.build_ll_cache(dm_npf; force_saveat=true)

        bi = batch_infos[1]
        if bi.n_b > 0
            re_measure = NoLimits.build_re_measure_from_batch(bi, θ_re, const_cache, dm_npf, ll_cache)
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
        dm_gauss = _make_simple_ghq_dm(; n_id=4, n_obs=3)
        fe = dm_gauss.model.fixed.fixed
        θ0_u = NoLimits.get_θ0_untransformed(fe)
        θ_re = NoLimits._symmetrize_psd_params(θ0_u, fe)
        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm_gauss, NamedTuple())
        ll_cache = NoLimits.build_ll_cache(dm_gauss; force_saveat=true)
        bi = batch_infos[1]
        if bi.n_b > 0
            re = NoLimits.build_re_measure_from_batch(bi, θ_re, const_cache, dm_gauss, ll_cache)
            @test re isa NoLimits.GaussianRE
        end
    end

    # ── End-to-end fit with NPF RE ────────────────────────────────────────────
    @testset "fit_model GHQuadrature level=1 with NPF RE" begin
        model_npf = @Model begin
            @helpers begin
                sat(u) = u / (1 + abs(u))
            end
            @fixedEffects begin
                a  = RealNumber(1.0)
                σ  = RealNumber(0.5, scale=:log)
                ψ  = NPFParameter(1, 2; seed=1, calculate_se=false)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + sat(η), σ)
            end
        end
        rng = MersenneTwister(99)
        ids = repeat(1:8, inner=4)
        ts  = repeat([1.0,2.0,3.0,4.0], outer=8)
        ys  = 1.0 .+ 0.3 .* randn(rng, 32) .+ 0.5 .* randn(rng, 32)
        df_npf = DataFrame(ID=ids, t=ts, y=ys)
        dm_npf = DataModel(model_npf, df_npf; primary_id=:ID, time_col=:t)

        res = fit_model(dm_npf, GHQuadrature(level=1; optim_kwargs=(maxiters=15,)))

        @test res isa NoLimits.FitResult
        @test res.result isa NoLimits.GHQuadratureResult
        @test isfinite(get_objective(res))
        @test get_objective(res) > 0

        params = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(params.a)
        @test isfinite(params.σ) && params.σ > 0

        re = get_random_effects(dm_npf, res)
        @test re isa NamedTuple
        @test haskey(re, :η)
        @test nrow(re.η) == 8
    end

end  # @testset "GHQuadrature NPF RE support"

# =============================================================================
# Phase 2: GHQuadratureMAP + Wald UQ
# =============================================================================

# ---------------------------------------------------------------------------
# Model with priors for MAP
# ---------------------------------------------------------------------------

function _make_map_ghq_dm(; n_id=8, n_obs=4, rng=MersenneTwister(7))
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0; prior=Normal(0.0, 2.0))
            σ = RealNumber(0.5, scale=:log; prior=LogNormal(0.0, 1.0))
            ω = RealNumber(1.0, scale=:log; prior=LogNormal(0.0, 1.0))
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    ids = repeat(1:n_id, inner=n_obs)
    ts  = repeat(collect(1.0:n_obs), outer=n_id)
    ηs  = repeat(randn(rng, n_id), inner=n_obs)
    ys  = 1.0 .+ ηs .+ 0.5 .* randn(rng, n_id * n_obs)
    df  = DataFrame(ID=ids, t=ts, y=ys)
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "GHQuadratureMAP" begin

    dm_map = _make_map_ghq_dm()

    # ── Basic fit ────────────────────────────────────────────────────────────
    @testset "Basic fit" begin
        res = fit_model(dm_map, GHQuadratureMAP(level=1; optim_kwargs=(maxiters=15,)))

        @test res isa NoLimits.FitResult
        @test res.result isa NoLimits.GHQuadratureMAPResult

        obj = get_objective(res)
        @test isfinite(obj)
        @test obj > 0   # negative MAP objective > 0

        params = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(params.a)
        @test isfinite(params.σ) && params.σ > 0
        @test isfinite(params.ω) && params.ω > 0

        @test get_converged(res) isa Bool

        iters = get_iterations(res)
        @test iters === missing || (iters isa Integer && iters >= 0)
    end

    # ── All accessors work ───────────────────────────────────────────────────
    @testset "Accessors" begin
        res = fit_model(dm_map, GHQuadratureMAP(level=1; optim_kwargs=(maxiters=15,)))

        re = get_random_effects(dm_map, res)
        @test re isa NamedTuple
        @test haskey(re, :η)
        @test nrow(re.η) == 8

        re2 = get_random_effects(res)
        @test nrow(re2.η) == 8

        ll = get_loglikelihood(res)
        @test isfinite(ll)
        @test ll < 0   # log-likelihood itself is negative
    end

    # ── GHQuadrature vs GHQuadratureMAP: MAP pulls parameters toward prior ───────
    @testset "MAP regularization pulls toward prior" begin
        # With a strong Normal(0, 0.5) prior on a, MAP estimate of a should
        # be closer to 0 than MLE estimate on same data.
        model_tight = @Model begin
            @fixedEffects begin
                a = RealNumber(2.0; prior=Normal(0.0, 0.3))  # strong prior toward 0
                σ = RealNumber(0.5, scale=:log; prior=LogNormal(0.0, 1.0))
                ω = RealNumber(1.0, scale=:log; prior=LogNormal(0.0, 1.0))
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
        rng_t = MersenneTwister(99)
        ids = repeat(1:6, inner=4)
        ts  = repeat(1.0:4.0, outer=6)
        ys  = 3.0 .+ 0.5 .* randn(rng_t, 24)   # true a ≈ 3, far from prior 0
        df_t = DataFrame(ID=ids, t=ts, y=ys)
        dm_t = DataModel(model_tight, df_t; primary_id=:ID, time_col=:t)

        res_mle = fit_model(dm_t, GHQuadrature(level=1; optim_kwargs=(maxiters=25,)))
        res_map = fit_model(dm_t, GHQuadratureMAP(level=1; optim_kwargs=(maxiters=25,)))

        a_mle = NoLimits.get_params(res_mle; scale=:untransformed).a
        a_map = NoLimits.get_params(res_map; scale=:untransformed).a

        # MAP should be pulled toward 0 compared to MLE
        @test abs(a_map) < abs(a_mle)
    end

    # ── Error: GHQuadratureMAP with no-prior model ─────────────────────────────
    @testset "Errors without priors" begin
        model_noprior = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)   # no prior
                σ = RealNumber(0.5, scale=:log)
                ω = RealNumber(1.0, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
        df_n = DataFrame(ID=repeat(1:3, inner=2), t=[1.0,2.0,1.0,2.0,1.0,2.0],
                         y=randn(MersenneTwister(1), 6))
        dm_n = DataModel(model_noprior, df_n; primary_id=:ID, time_col=:t)
        @test_throws ErrorException fit_model(dm_n, GHQuadratureMAP(level=1))
    end

end  # @testset "GHQuadratureMAP"

@testset "GHQuadrature Wald UQ" begin

    dm_uq = _make_map_ghq_dm()

    # ── GHQuadrature Wald (default ForwardDiff Hessian) ────────────────────────
    @testset "compute_uq GHQuadrature level=2" begin
        res = fit_model(dm_uq, GHQuadrature(level=2; optim_kwargs=(maxiters=30,)))
        uq  = compute_uq(res; method=:wald, pseudo_inverse=true)

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test hasproperty(cia.lower, :σ)
        @test hasproperty(cia.lower, :ω)

        # Intervals should be finite and ordered
        @test isfinite(cia.lower.a) && isfinite(cia.upper.a)
        @test cia.lower.a < cia.upper.a

        @test isfinite(cia.lower.σ) && isfinite(cia.upper.σ)
        @test cia.lower.σ < cia.upper.σ
        @test cia.lower.σ > 0.0   # σ on natural scale must be positive
    end

    # ── GHQuadratureMAP Wald (adds prior) ───────────────────────────────────────
    @testset "compute_uq GHQuadratureMAP level=2" begin
        res = fit_model(dm_uq, GHQuadratureMAP(level=2; optim_kwargs=(maxiters=30,)))
        uq  = compute_uq(res; method=:wald, pseudo_inverse=true)

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test isfinite(cia.lower.a) && isfinite(cia.upper.a)
        @test cia.lower.a < cia.upper.a
    end

    # ── Sandwich vcov ────────────────────────────────────────────────────────
    @testset "compute_uq GHQuadrature sandwich vcov level=2" begin
        res = fit_model(dm_uq, GHQuadrature(level=2; optim_kwargs=(maxiters=30,)))
        uq  = compute_uq(res; method=:wald, vcov=:sandwich, pseudo_inverse=true)

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test cia.lower.a < cia.upper.a
    end

    # ── hessian_backend :fd_gradient also works ───────────────────────────────
    @testset "compute_uq GHQuadrature fd_gradient backend level=2" begin
        res = fit_model(dm_uq, GHQuadrature(level=2; optim_kwargs=(maxiters=30,)))
        uq  = compute_uq(res; method=:wald, hessian_backend=:fd_gradient, pseudo_inverse=true)

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test isfinite(cia.lower.a) && isfinite(cia.upper.a)
    end

end  # @testset "GHQuadrature Wald UQ"

@testset "GHQuadrature Profile UQ" begin

    dm_uq = _make_map_ghq_dm()

    @testset "compute_uq GHQuadrature :profile level=2" begin
        res = fit_model(dm_uq, GHQuadrature(level=2; optim_kwargs=(maxiters=40,)))
        uq  = compute_uq(res; method=:profile)

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test isfinite(cia.lower.a) && isfinite(cia.upper.a)
        @test cia.lower.a < cia.upper.a
    end

end  # @testset "GHQuadrature Profile UQ"

@testset "GHQuadrature mcmc_refit UQ" begin

    dm_uq = _make_map_ghq_dm()

    @testset "compute_uq GHQuadratureMAP :mcmc_refit" begin
        res = fit_model(dm_uq, GHQuadratureMAP(level=1; optim_kwargs=(maxiters=15,)))
        uq  = compute_uq(res;
                         method=:mcmc_refit,
                         mcmc_sampler=Turing.MH(),
                         mcmc_turing_kwargs=(n_samples=50, n_adapt=20, progress=false))

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test isfinite(cia.lower.a) && isfinite(cia.upper.a)
    end

    @testset "compute_uq GHQuadrature :mcmc_refit (with priors)" begin
        # GHQuadrature with priors on all fixed effects can use mcmc_refit
        res = fit_model(dm_uq, GHQuadrature(level=1; optim_kwargs=(maxiters=15,)))
        uq  = compute_uq(res;
                         method=:mcmc_refit,
                         mcmc_sampler=Turing.MH(),
                         mcmc_turing_kwargs=(n_samples=50, n_adapt=20, progress=false))

        @test uq isa NoLimits.UQResult
        cia = get_uq_intervals(uq)
        @test hasproperty(cia.lower, :a)
        @test isfinite(cia.lower.a) && isfinite(cia.upper.a)
    end

end  # @testset "GHQuadrature mcmc_refit UQ"

@testset "GHQuadrature parallelization (EnsembleThreads)" begin

    dm_par = _make_simple_ghq_dm(; n_id=10, n_obs=5)

    # ── EnsembleThreads produces same objective as EnsembleSerial ─────────────
    @testset "EnsembleThreads matches EnsembleSerial objective" begin
        res_serial   = fit_model(dm_par, GHQuadrature(level=2; optim_kwargs=(maxiters=25,));
                                 serialization=EnsembleSerial())
        res_threaded = fit_model(dm_par, GHQuadrature(level=2; optim_kwargs=(maxiters=25,));
                                 serialization=EnsembleThreads())

        # Objectives should agree within numerical tolerance (same deterministic quadrature)
        @test abs(get_objective(res_serial) - get_objective(res_threaded)) < 1.0

        # Both should converge and produce valid RE estimates
        re_t = get_random_effects(dm_par, res_threaded)
        @test nrow(re_t.η) == 10
    end

    @testset "GHQuadratureMAP EnsembleThreads runs without error" begin
        dm_m = _make_map_ghq_dm()
        res = fit_model(dm_m, GHQuadratureMAP(level=2; optim_kwargs=(maxiters=25,));
                        serialization=EnsembleThreads())
        @test res.result isa NoLimits.GHQuadratureMAPResult
        @test isfinite(get_objective(res))
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
        @test val ≈ 1.0  atol=1e-10
        # E[z₁⁴] = 3
        val4 = sg_integrate(z -> z[1]^4, sg)
        @test val4 ≈ 3.0  atol=1e-10
        # E[z₁² * z₂²] = 1
        val_cross = sg_integrate(z -> z[1]^2 * z[2]^2, sg)
        @test val_cross ≈ 1.0  atol=1e-10
    end

    # Dedup is idempotent: building twice gives same result
    @testset "n_ghq_points matches build_sparse_grid after dedup" begin
        for (d, l) in [(1,1),(1,2),(2,1),(2,2),(2,3),(3,2)]
            sg = build_sparse_grid(d, l)
            @test NoLimits.n_ghq_points(d, l) == size(sg.nodes, 2)
        end
    end

end  # @testset "GHQuadrature node deduplication"

# ============================================================
# LogNormalRE and BoundedRE (Beta) transport maps
# ============================================================

@testset "GHQuadrature LogNormal RE" begin

    # LogNormal(μ_log, σ_log): η = exp(μ_log + σ_log * z), logcorrection = 0.
    # The push-forward of N(0,1) under this map is LogNormal(μ_log, σ_log),
    # so GH weights absorb the prior exactly — same as GaussianRE.

    @testset "fit_model with LogNormal RE — basic" begin
        rng = MersenneTwister(42)
        n_id = 12
        n_obs = 5

        # True parameters: η ~ LogNormal(0, 0.5), y ~ Normal(a * η, σ)
        a_true   = 2.0
        σ_true   = 0.3
        ω_true   = 0.5  # σ parameter of LogNormal (log-scale std)

        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = exp.(ω_true .* randn(rng, n_id))
        yobs = a_true .* η_i[ids] .+ σ_true .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.5)
                σ = RealNumber(0.5, scale=:log)
                ω = RealNumber(0.6, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(LogNormal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a * η, σ)
            end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)

        # Level 2 should work
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=100,)))

        @test NoLimits.get_converged(res)
        p = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(p.a)
        @test isfinite(p.σ)
        @test isfinite(p.ω)
        @test NoLimits.get_objective(res) < 1e6

        # a should be in reasonable ballpark
        @test abs(p.a - a_true) < 1.5
    end

    @testset "get_random_effects for LogNormal RE" begin
        rng = MersenneTwister(7)
        n_id = 8
        ids  = repeat(1:n_id, inner=4)
        η_i  = exp.(0.4 .* randn(rng, n_id))
        yobs = 1.5 .* η_i[ids] .+ 0.2 .* randn(rng, length(ids))
        tobs = repeat(1:4, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.5)
                σ = RealNumber(0.3, scale=:log)
                ω = RealNumber(0.4, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(LogNormal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a * η, σ)
            end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=80,)))

        re = NoLimits.get_random_effects(dm, res)
        @test re isa NamedTuple
        @test haskey(re, :η)
        @test nrow(re.η) == n_id
    end

    @testset "validation allows LogNormal" begin
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.3, scale=:log)
                ω = RealNumber(0.4, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(LogNormal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a * η, σ)
            end
        end
        df = DataFrame(ID=[1, 1], t=[1.0, 2.0], y=[1.0, 1.1])
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        # Should not throw
        @test_nowarn NoLimits._ghq_validate_re_distributions(dm)
    end

end  # @testset "GHQuadrature LogNormal RE"


@testset "GHQuadrature Beta RE" begin

    # Beta(α, β): η = logistic(z), z ~ N(0,1) reference.
    # logcorrection = logpdf(Beta(α,β), logistic(z)) + log(logistic(z))
    #               + log(1 - logistic(z)) + z²/2 + log(2π)/2

    @testset "fit_model with Beta RE — basic" begin
        rng = MersenneTwister(99)
        n_id = 14
        n_obs = 4

        # True parameters: η ~ Beta(2, 5), y ~ Normal(a + b*η, σ)
        a_true = 0.5
        b_true = 2.0
        σ_true = 0.2
        α_true = 2.0
        β_true = 5.0

        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = rand(rng, Beta(α_true, β_true), n_id)
        yobs = a_true .+ b_true .* η_i[ids] .+ σ_true .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a  = RealNumber(0.4)
                b  = RealNumber(2.5)
                σ  = RealNumber(0.3, scale=:log)
                α  = RealNumber(2.0, scale=:log)
                β  = RealNumber(5.0, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Beta(α, β); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * η, σ)
            end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)

        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=35,)))

        @test NoLimits.get_converged(res)
        p = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(p.a)
        @test isfinite(p.b)
        @test isfinite(p.σ)
        @test isfinite(p.α)
        @test isfinite(p.β)
        @test NoLimits.get_objective(res) < 1e6
    end

    @testset "get_random_effects for Beta RE" begin
        rng = MersenneTwister(55)
        n_id = 8
        ids  = repeat(1:n_id, inner=5)
        η_i  = rand(rng, Beta(2.0, 4.0), n_id)
        yobs = 1.0 .+ 2.0 .* η_i[ids] .+ 0.15 .* randn(rng, length(ids))
        tobs = repeat(1:5, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                b = RealNumber(2.0)
                σ = RealNumber(0.2, scale=:log)
                α = RealNumber(2.0, scale=:log)
                β = RealNumber(4.0, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Beta(α, β); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * η, σ)
            end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=100,)))

        re = NoLimits.get_random_effects(dm, res)
        @test re isa NamedTuple
        @test haskey(re, :η)
        @test nrow(re.η) == n_id
        # Beta RE EB modes should be in (0, 1)
        # Column is :η_1 (flatten of 1-element vector transform output)
        @test all(0.0 .< re.η.η_1 .< 1.0)
    end

    @testset "validation allows Beta" begin
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(0.5)
                σ = RealNumber(0.3, scale=:log)
                α = RealNumber(2.0, scale=:log)
                β = RealNumber(3.0, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Beta(α, β); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
        df = DataFrame(ID=[1, 1], t=[1.0, 2.0], y=[0.5, 0.6])
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        @test_nowarn NoLimits._ghq_validate_re_distributions(dm)
    end

end  # @testset "GHQuadrature Beta RE"

# ============================================================
# Phase 3: Additional 1D rules (GL, CC)
# ============================================================

@testset "Additional 1D quadrature rules" begin

    _gl_rule = NoLimits._gl_rule
    _cc_rule = NoLimits._cc_rule

    @testset "GL rule n=1" begin
        x, lw = _gl_rule(1)
        @test x ≈ [0.0]  atol=1e-14
        @test exp(lw[1]) ≈ 2.0  atol=1e-12  # weight = interval length
    end

    @testset "GL rule n=2" begin
        x, lw = _gl_rule(2)
        @test sort(x) ≈ [-1/sqrt(3), 1/sqrt(3)]  atol=1e-12
        @test exp.(lw) ≈ [1.0, 1.0]  atol=1e-12
        @test sum(exp.(lw)) ≈ 2.0  atol=1e-12
    end

    @testset "GL weights sum to 2 for n=1..5" begin
        for n in 1:5
            _, lw = _gl_rule(n)
            @test sum(exp.(lw)) ≈ 2.0  atol=1e-12
        end
    end

    @testset "GL integrates polynomials exactly" begin
        # GL with n points is exact for polynomials of degree ≤ 2n-1
        for n in 2:5
            x, lw = _gl_rule(n)
            w = exp.(lw)
            # ∫₋₁¹ x² dx = 2/3
            @test sum(w .* x.^2) ≈ 2/3  atol=1e-12
            # ∫₋₁¹ x^(2n-1) dx = 0 (odd → 0)
            @test abs(sum(w .* x.^(2n-1))) < 1e-12
        end
    end

    @testset "CC rule n=1" begin
        x, lw = _cc_rule(1)
        @test sort(x) ≈ [-1.0, 1.0]  atol=1e-14
        @test sum(exp.(lw)) ≈ 2.0  atol=1e-12  # weights sum to interval length
    end

    @testset "CC rule n=2 (3 nodes)" begin
        x, lw = _cc_rule(2)
        @test sort(x) ≈ [-1.0, 0.0, 1.0]  atol=1e-14
        w = exp.(lw)
        @test sum(w) ≈ 2.0  atol=1e-12
        # Endpoint weights 1/3, interior weight 4/3
        idx0 = argmin(abs.(x))
        @test w[idx0] ≈ 4/3  atol=1e-12
        @test all(abs.(w[setdiff(1:3, idx0)] .- 1/3) .< 1e-12)
    end

    @testset "CC weights sum to 2 for n=1..5" begin
        for n in 1:5
            _, lw = _cc_rule(n)
            @test sum(exp.(lw)) ≈ 2.0  atol=1e-10
        end
    end

    @testset "CC integrates polynomials exactly up to degree n" begin
        for n in 2:5
            x, lw = _cc_rule(n)
            w = exp.(lw)
            # ∫₋₁¹ x^k dx = 0 for k odd, 2/(k+1) for k even
            for k in 0:n
                exact = iseven(k) ? 2/(k+1) : 0.0
                @test sum(w .* x.^k) ≈ exact  atol=1e-10
            end
        end
    end

end  # @testset "Additional 1D quadrature rules"


# ============================================================
# Phase 3: Anisotropic grids
# ============================================================

@testset "Anisotropic sparse grids" begin

    build_tensor_product_grid = NoLimits.build_tensor_product_grid
    get_anisotropic_grid      = NoLimits.get_anisotropic_grid
    n_anisotropic_grid_points = NoLimits.n_anisotropic_grid_points

    @testset "tensor product d=1×1 gives d=2 grid" begin
        sg1 = build_sparse_grid(1, 2)  # 2 points
        sg2 = build_sparse_grid(1, 3)  # 3 points
        tp  = build_tensor_product_grid([sg1, sg2])
        @test tp.dim == 2
        @test size(tp.nodes, 2) == 2 * 3
    end

    @testset "tensor product integration accuracy" begin
        sg1 = build_sparse_grid(1, 3)
        sg2 = build_sparse_grid(1, 3)
        tp  = build_tensor_product_grid([sg1, sg2])
        # E_{N(0,I₂)}[z₁² + z₂²] = 2
        val = sg_integrate(z -> z[1]^2 + z[2]^2, tp)
        @test val ≈ 2.0  atol=1e-10
        # E[z₁² * z₂²] = 1 (independence)
        val2 = sg_integrate(z -> z[1]^2 * z[2]^2, tp)
        @test val2 ≈ 1.0  atol=1e-10
    end

    @testset "get_anisotropic_grid caches correctly" begin
        dims   = [1, 1]
        levels = [2, 3]
        sg_a = get_anisotropic_grid(dims, levels)
        sg_b = get_anisotropic_grid(dims, levels)  # same key → same object
        @test sg_a === sg_b
        @test n_anisotropic_grid_points(dims, levels) == size(sg_a.nodes, 2)
    end

    @testset "anisotropic fit with NamedTuple level" begin
        rng = MersenneTwister(12)
        n_id = 10
        ids  = repeat(1:n_id, inner=5)
        η_i  = 0.5 .* randn(rng, n_id)
        yobs = 1.0 .+ η_i[ids] .+ 0.3 .* randn(rng, length(ids))
        tobs = repeat(1:5, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.3, scale=:log)
                ω = RealNumber(0.5, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end

        df = DataFrame(ID=ids, t=tobs, y=yobs)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)

        # Anisotropic level: η at level 2 (isotropic would use same level for all)
        res_iso  = fit_model(dm, GHQuadrature(level=2; optim_kwargs=(maxiters=15,)))
        res_aniso = fit_model(dm, GHQuadrature(level=(η=2,); optim_kwargs=(maxiters=15,)))

        @test NoLimits.get_converged(res_iso)
        @test NoLimits.get_converged(res_aniso)

        # Objectives should be comparable (same level=2 for the only RE)
        obj_iso   = NoLimits.get_objective(res_iso)
        obj_aniso = NoLimits.get_objective(res_aniso)
        @test abs(obj_iso - obj_aniso) / max(abs(obj_iso), 1.0) < 0.05

        # Parameters should be close
        p_iso   = NoLimits.get_params(res_iso;  scale=:untransformed)
        p_aniso = NoLimits.get_params(res_aniso; scale=:untransformed)
        @test abs(p_iso.a - p_aniso.a) < 0.1
    end

    @testset "anisotropic level default=1 for unlisted RE" begin
        # Using level=(η=2,) on a model with only η should use level 2
        # Using level=(η_other=2,) should default to level 1 for η
        rng = MersenneTwister(7)
        n_id = 8
        ids  = repeat(1:n_id, inner=4)
        yobs = 1.0 .+ 0.3 .* randn(rng, length(ids))
        tobs = repeat(1:4, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.3, scale=:log)
                ω = RealNumber(0.5, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
        df = DataFrame(ID=ids, t=tobs, y=yobs)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)

        # (nonexistent=5,) → η defaults to level 1
        res = fit_model(dm, GHQuadrature(level=(nonexistent=5,); optim_kwargs=(maxiters=25,)))
        @test NoLimits.get_converged(res)
        @test isfinite(NoLimits.get_objective(res))
    end

end  # @testset "Anisotropic sparse grids"


# ============================================================
# Gamma, Exponential, Weibull, TDist RE distributions
# ============================================================

@testset "GHQuadrature Gamma RE" begin

    @testset "fit_model with Gamma RE" begin
        rng = MersenneTwister(301)
        n_id = 14; n_obs = 4
        α_true = 3.0; θ_true = 0.5; a_true = 2.0; σ_true = 0.3
        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = rand(rng, Gamma(α_true, θ_true), n_id)
        yobs = a_true .* η_i[ids] .+ σ_true .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a  = RealNumber(2.0)
                σ  = RealNumber(0.4, scale=:log)
                α  = RealNumber(2.0, scale=:log)
                θ  = RealNumber(0.5, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Gamma(α, θ); column=:ID)
            end
            @formulas begin
                y ~ Normal(a * η, σ)
            end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=35,)))

        @test NoLimits.get_converged(res)
        p = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(p.a) && isfinite(p.σ) && isfinite(p.α) && isfinite(p.θ)
        @test p.α > 0 && p.θ > 0
        @test NoLimits.get_objective(res) < 1e6
    end

    @testset "get_random_effects for Gamma RE" begin
        rng = MersenneTwister(302)
        n_id = 8; ids = repeat(1:n_id, inner=4)
        η_i  = rand(rng, Gamma(2.0, 1.0), n_id)
        yobs = 1.5 .* η_i[ids] .+ 0.2 .* randn(rng, length(ids))
        tobs = repeat(1:4, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.5);  σ = RealNumber(0.3, scale=:log)
                α = RealNumber(2.0, scale=:log);  θ = RealNumber(1.0, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(Gamma(α, θ); column=:ID)
            end
            @formulas begin; y ~ Normal(a * η, σ); end
        end
        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=80,)))

        re = NoLimits.get_random_effects(dm, res)
        @test re isa NamedTuple && haskey(re, :η)
        @test nrow(re.η) == n_id
        # Gamma RE should be positive
        @test all(re.η.η_1 .> 0)
    end

    @testset "validation allows Gamma" begin
        model = @Model begin
            @fixedEffects begin
                α = RealNumber(2.0, scale=:log); θ = RealNumber(1.0, scale=:log)
                σ = RealNumber(0.3, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(Gamma(α, θ); column=:ID)
            end
            @formulas begin; y ~ Normal(η, σ); end
        end
        df = DataFrame(ID=[1, 1], t=[1.0, 2.0], y=[1.0, 1.5])
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        @test_nowarn NoLimits._ghq_validate_re_distributions(dm)
    end

end  # @testset "GHQuadrature Gamma RE"


@testset "GHQuadrature Exponential RE" begin

    @testset "fit_model with Exponential RE" begin
        rng = MersenneTwister(401)
        n_id = 12; n_obs = 4
        λ_true = 2.0; a_true = 1.5; σ_true = 0.3
        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = rand(rng, Exponential(1/λ_true), n_id)
        yobs = a_true .* η_i[ids] .+ σ_true .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.5);  σ = RealNumber(0.4, scale=:log)
                θ = RealNumber(0.5, scale=:log)   # scale = 1/rate
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(Exponential(θ); column=:ID)
            end
            @formulas begin; y ~ Normal(a * η, σ); end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=35,)))

        @test NoLimits.get_converged(res)
        p = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(p.a) && isfinite(p.σ) && isfinite(p.θ)
        @test p.θ > 0
    end

    @testset "get_random_effects for Exponential RE are positive" begin
        rng = MersenneTwister(402)
        n_id = 8; ids = repeat(1:n_id, inner=4)
        η_i  = rand(rng, Exponential(0.5), n_id)
        yobs = 2.0 .* η_i[ids] .+ 0.2 .* randn(rng, length(ids))
        tobs = repeat(1:4, n_id) .* 1.0
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(2.0); σ = RealNumber(0.2, scale=:log)
                θ = RealNumber(0.5, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin; η = RandomEffect(Exponential(θ); column=:ID); end
            @formulas begin; y ~ Normal(a * η, σ); end
        end
        df = DataFrame(ID=ids, t=tobs, y=yobs)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=80,)))
        re = NoLimits.get_random_effects(dm, res)
        @test all(re.η.η_1 .> 0)
    end

end  # @testset "GHQuadrature Exponential RE"


@testset "GHQuadrature Weibull RE" begin

    @testset "fit_model with Weibull RE" begin
        rng = MersenneTwister(501)
        n_id = 14; n_obs = 4
        α_true = 2.0; θ_true = 1.5; a_true = 1.0; σ_true = 0.3
        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = rand(rng, Weibull(α_true, θ_true), n_id)
        yobs = a_true .* η_i[ids] .+ σ_true .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a  = RealNumber(1.0);  σ = RealNumber(0.3, scale=:log)
                α  = RealNumber(2.0, scale=:log);  θ = RealNumber(1.5, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(Weibull(α, θ); column=:ID)
            end
            @formulas begin; y ~ Normal(a * η, σ); end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=35,)))

        @test NoLimits.get_converged(res)
        p = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(p.a) && isfinite(p.σ) && isfinite(p.α) && isfinite(p.θ)
        @test p.α > 0 && p.θ > 0
        @test NoLimits.get_objective(res) < 1e6
    end

    @testset "get_random_effects for Weibull RE are positive" begin
        rng = MersenneTwister(502)
        n_id = 8; ids = repeat(1:n_id, inner=4)
        η_i  = rand(rng, Weibull(2.0, 1.0), n_id)
        yobs = η_i[ids] .+ 0.2 .* randn(rng, length(ids))
        tobs = repeat(1:4, n_id) .* 1.0
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0); σ = RealNumber(0.2, scale=:log)
                α = RealNumber(2.0, scale=:log); θ = RealNumber(1.0, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin; η = RandomEffect(Weibull(α, θ); column=:ID); end
            @formulas begin; y ~ Normal(a * η, σ); end
        end
        df = DataFrame(ID=ids, t=tobs, y=yobs)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=80,)))
        re = NoLimits.get_random_effects(dm, res)
        @test all(re.η.η_1 .> 0)
    end

end  # @testset "GHQuadrature Weibull RE"


@testset "GHQuadrature TDist RE" begin

    @testset "fit_model with TDist RE" begin
        # TDist(ν): heavy-tailed, ℝ-supported.  Identity transport.
        rng = MersenneTwister(601)
        n_id = 14; n_obs = 4
        ν_true = 5.0; a_true = 1.0; σ_true = 0.3
        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = rand(rng, TDist(ν_true), n_id)
        yobs = a_true .+ η_i[ids] .+ σ_true .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a  = RealNumber(1.0);  σ = RealNumber(0.3, scale=:log)
                ν  = RealNumber(5.0, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(TDist(ν); column=:ID)
            end
            @formulas begin; y ~ Normal(a + η, σ); end
        end

        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=70,)))

        @test NoLimits.get_converged(res)
        p = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(p.a) && isfinite(p.σ) && isfinite(p.ν)
        @test p.ν > 0
        @test NoLimits.get_objective(res) < 1e6
    end

    @testset "get_random_effects for TDist RE" begin
        rng = MersenneTwister(602)
        n_id = 8; ids = repeat(1:n_id, inner=5)
        η_i  = rand(rng, TDist(4.0), n_id)
        yobs = 0.5 .+ η_i[ids] .+ 0.2 .* randn(rng, length(ids))
        tobs = repeat(1:5, n_id) .* 1.0
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(0.5); σ = RealNumber(0.2, scale=:log)
                ν = RealNumber(4.0, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin; η = RandomEffect(TDist(ν); column=:ID); end
            @formulas begin; y ~ Normal(a + η, σ); end
        end
        df = DataFrame(ID=ids, t=tobs, y=yobs)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=80,)))
        re = NoLimits.get_random_effects(dm, res)
        @test re isa NamedTuple && haskey(re, :η)
        @test nrow(re.η) == n_id
    end

end  # @testset "GHQuadrature TDist RE"


@testset "GHQuadrature generic ContinuousUnivariateDistribution fallback" begin

    # Laplace(μ, b): ℝ-supported, identity transport — hits generic branch
    @testset "Laplace RE (ℝ-supported generic)" begin
        rng = MersenneTwister(701)
        n_id = 12; n_obs = 4
        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = rand(rng, Distributions.Laplace(0.0, 0.5), n_id)
        yobs = 1.0 .+ η_i[ids] .+ 0.2 .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0); σ = RealNumber(0.2, scale=:log)
                b = RealNumber(0.5, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(Distributions.Laplace(0.0, b); column=:ID)
            end
            @formulas begin; y ~ Normal(a + η, σ); end
        end
        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=100,)))
        @test NoLimits.get_converged(res)
        @test isfinite(NoLimits.get_objective(res))
    end

    # InverseGamma(α, β): (0,∞)-supported, exp transport — hits generic branch
    @testset "InverseGamma RE (ℝ⁺-supported generic)" begin
        rng = MersenneTwister(702)
        n_id = 12; n_obs = 4
        ids  = repeat(1:n_id, inner=n_obs)
        η_i  = rand(rng, InverseGamma(3.0, 2.0), n_id)
        yobs = 1.5 .* η_i[ids] .+ 0.3 .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0

        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.5); σ = RealNumber(0.3, scale=:log)
                α = RealNumber(3.0, scale=:log); β = RealNumber(2.0, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(InverseGamma(α, β); column=:ID)
            end
            @formulas begin; y ~ Normal(a * η, σ); end
        end
        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.GHQuadrature(level=2; optim_kwargs=(maxiters=100,)))
        @test isfinite(NoLimits.get_objective(res))
        re = NoLimits.get_random_effects(dm, res)
        @test all(re.η.η_1 .> 0)
    end

end  # @testset "GHQuadrature generic ContinuousUnivariateDistribution fallback"


# ============================================================
# Progressive refinement: level::Vector{Int}
# ============================================================

@testset "GHQuadrature progressive refinement (level::Vector{Int})" begin

    function _make_progressive_dm(; n_id=10, n_obs=5, rng=MersenneTwister(42))
        ids  = repeat(1:n_id, inner=n_obs)
        yobs = 1.0 .+ 0.3 .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.3, scale=:log)
                ω = RealNumber(0.3, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column=:ID)
            end
            @formulas begin; y ~ Normal(a + η, σ); end
        end
        df = DataFrame(ID=ids, t=tobs, y=yobs)
        DataModel(model, df; primary_id=:ID, time_col=:t)
    end

    @testset "level=[1,2] converges and result is scalar-level" begin
        dm  = _make_progressive_dm()
        res = fit_model(dm, GHQuadrature(level=[1, 2]; optim_kwargs=(maxiters=25,)))
        @test NoLimits.get_converged(res)
        @test isfinite(NoLimits.get_objective(res))
        # Returned method should carry the last scalar level (2)
        @test NoLimits.get_method(res).level == 2
    end

    @testset "level=[1] (single-element) behaves like level=1" begin
        rng = MersenneTwister(1)
        dm  = _make_progressive_dm(; rng=rng)
        res_vec    = fit_model(dm, GHQuadrature(level=[1];    optim_kwargs=(maxiters=25,)))
        res_scalar = fit_model(dm, GHQuadrature(level=1;      optim_kwargs=(maxiters=25,)))
        @test NoLimits.get_converged(res_vec)
        @test abs(NoLimits.get_objective(res_vec) - NoLimits.get_objective(res_scalar)) < 1e-4
    end

    @testset "level=[1,2,3] three-stage refinement" begin
        dm  = _make_progressive_dm()
        res = fit_model(dm, GHQuadrature(level=[1, 2, 3]; optim_kwargs=(maxiters=25,)))
        @test NoLimits.get_converged(res)
        @test NoLimits.get_method(res).level == 3
        p = NoLimits.get_params(res; scale=:untransformed)
        @test isfinite(p.a) && isfinite(p.σ) && isfinite(p.ω)
    end

    @testset "level=[1,2] result compatible with all accessors" begin
        dm  = _make_progressive_dm()
        res = fit_model(dm, GHQuadrature(level=[1, 2]; optim_kwargs=(maxiters=12,)))
        @test isfinite(NoLimits.get_objective(res))
        @test NoLimits.get_iterations(res) isa Integer
        re = NoLimits.get_random_effects(dm, res)
        @test re isa NamedTuple && haskey(re, :η)
        ll = NoLimits.get_loglikelihood(res)
        @test isfinite(ll)
    end

    @testset "GHQuadratureMAP level=[1,2] works" begin
        rng  = MersenneTwister(7)
        n_id = 10; n_obs = 5
        ids  = repeat(1:n_id, inner=n_obs)
        yobs = 1.0 .+ 0.3 .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0; prior=Normal(0.0, 2.0))
                σ = RealNumber(0.3, scale=:log; prior=LogNormal(0.0, 1.0))
                ω = RealNumber(0.3, scale=:log; prior=LogNormal(0.0, 1.0))
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column=:ID)
            end
            @formulas begin; y ~ Normal(a + η, σ); end
        end
        df  = DataFrame(ID=ids, t=tobs, y=yobs)
        dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, GHQuadratureMAP(level=[1, 2]; optim_kwargs=(maxiters=15,)))
        @test NoLimits.get_converged(res)
        @test NoLimits.get_method(res).level == 2
    end

    @testset "empty level vector throws" begin
        dm = _make_progressive_dm()
        @test_throws ErrorException fit_model(dm, GHQuadrature(level=Int[]))
    end

    @testset "non-positive level entry throws" begin
        dm = _make_progressive_dm()
        @test_throws ErrorException fit_model(dm, GHQuadrature(level=[1, 0]))
    end

end  # @testset "GHQuadrature progressive refinement"


# ============================================================
# MCIntegrator: prior sampling and Turing MCMC sampling
# ============================================================

@testset "get_loglikelihood_quadrature MC sampling" begin

    # Shared fixture: simple Normal RE model with Laplace fit
    function _mc_test_dm(; n_id=10, n_obs=5, rng=MersenneTwister(900))
        ids  = repeat(1:n_id, inner=n_obs)
        yobs = 1.0 .+ 0.4 .* randn(rng, n_id * n_obs)
        tobs = repeat(1:n_obs, n_id) .* 1.0
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.4, scale=:log)
                ω = RealNumber(0.4, scale=:log)
            end
            @covariates begin; t = Covariate(); end
            @randomEffects begin; η = RandomEffect(Normal(0.0, ω); column=:ID); end
            @formulas begin; y ~ Normal(a + η, σ); end
        end
        df = DataFrame(ID=ids, t=tobs, y=yobs)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=15,)))
        return dm, res
    end

    @testset "MCIntegrator constructor" begin
        mc = MCIntegrator()
        @test mc.n_samples == 1000
        @test mc.mode === :turing
        @test mc.sampler === nothing
        @test mc.n_warmup == 500
        @test mc.rng === nothing

        mc2 = MCIntegrator(n_samples=500, mode=:prior)
        @test mc2.n_samples == 500

        @test_throws ErrorException MCIntegrator(mode=:unknown)
        @test_throws ErrorException MCIntegrator(n_samples=0)
        @test_throws ErrorException MCIntegrator(n_warmup=-1)
    end

    @testset "prior MC for all batches: finite and close to quadrature" begin
        dm, res = _mc_test_dm()
        ll_ghq = get_loglikelihood_quadrature(res; level=2, seed=1)
        ll_mc  = get_loglikelihood_quadrature(res; seed=1,
                     mc_integrator=MCIntegrator(n_samples=3000, mode=:prior))
        @test isfinite(ll_ghq)
        @test isfinite(ll_mc)
        # Prior MC should be in the right ballpark (within 15 log units for n_id=10)
        @test abs(ll_mc - ll_ghq) < 15.0
    end

    @testset "seed makes result reproducible" begin
        dm, res = _mc_test_dm()
        ll1 = get_loglikelihood_quadrature(res; seed=42,
                  mc_integrator=MCIntegrator(n_samples=200, mode=:prior))
        ll2 = get_loglikelihood_quadrature(res; seed=42,
                  mc_integrator=MCIntegrator(n_samples=200, mode=:prior))
        @test ll1 == ll2
    end

    @testset "fallback=nothing is accepted and default AGHQ path works" begin
        dm, res = _mc_test_dm()
        ll = get_loglikelihood_quadrature(res; level=2, seed=1, fallback=nothing)
        @test isfinite(ll)
    end

    @testset "Turing MC for all batches: finite result" begin
        dm, res = _mc_test_dm(n_id=8, n_obs=5, rng=MersenneTwister(901))
        ll_mc_turing = get_loglikelihood_quadrature(res; seed=2,
            mc_integrator=MCIntegrator(n_samples=500, mode=:turing,
                                       sampler=Turing.MH(), n_warmup=200))
        @test isfinite(ll_mc_turing)
    end

    @testset "Turing MC close to quadrature" begin
        dm, res = _mc_test_dm(n_id=10, n_obs=5, rng=MersenneTwister(903))
        ll_ghq    = get_loglikelihood_quadrature(res; level=2, seed=3)
        ll_turing = get_loglikelihood_quadrature(res; seed=3,
            mc_integrator=MCIntegrator(n_samples=1000, mode=:turing,
                                       sampler=Turing.MH(), n_warmup=500))
        @test isfinite(ll_turing)
        @test abs(ll_turing - ll_ghq) < 2.0
    end

    @testset "fallback MCIntegrator path works end-to-end" begin
        dm, res = _mc_test_dm(n_id=8, n_obs=4)
        ll = get_loglikelihood_quadrature(res; seed=10, jitter=1e-6,
            fallback=MCIntegrator(n_samples=500, mode=:prior))
        @test isfinite(ll)
    end

end  # @testset "get_loglikelihood_quadrature MC sampling"
