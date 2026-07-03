using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using Random
using LinearAlgebra
using Lux
using OrdinaryDiffEq
import Optimisers

# ─────────────────────────────────────────────────────────────────────────────
# Enzyme-compatibility PROXY tests (no Enzyme dependency, fast).
#
# Enzyme.jl support rests on structural invariants of the code (established
# 2026-06-06/07): concrete containers, Restructure-free positional parameter
# rebuilds, axes-equipped transforms, flat-vector ODE solve parameters with
# options baked into the problem, and index-based solution access. Breaking any
# of these would NOT fail the numerical test suite but would silently re-break
# Enzyme. These tests pin the invariants and the numerical equivalence of each
# Enzyme-motivated rewrite, so regressions are flagged without paying for
# Enzyme compiles. Real Enzyme gradients: see enzyme_smoke_tests.jl (opt-in).
# ─────────────────────────────────────────────────────────────────────────────

@testset "individuals container is concrete (reverse-mode getfield)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5, scale = :log)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + x.Age + η, σ)
        end
    end
    df = DataFrame(ID = repeat(1:3, inner = 2), t = repeat([0.0, 1.0], outer = 3),
        Age = repeat([0.3, -0.2, 0.1], inner = 2), y = randn(Xoshiro(1), 6))
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test isconcretetype(eltype(dm.individuals))
end

@testset "transforms carry output axes; stable path ≡ legacy path" begin
    fe = @fixedEffects begin
        a = RealNumber(0.5)
        b = RealNumber(0.2, scale = :log, lower = 1e-12, upper = Inf)
        c = RealNumber(0.3, scale = :logit, lower = 0.0, upper = 1.0)
        v = RealVector([0.4, 2.0], scale = [:identity, :log],
            lower = [-Inf, 1e-12], upper = [Inf, Inf])
        Ω = RealPSDMatrix([1.0 0.2; 0.2 1.0], scale = :cholesky)
        E = RealPSDMatrix([1.5 0.1; 0.1 0.8], scale = :expm)
        G = RealLiePSDMatrix([1.5 0.1; 0.1 0.8])
        p = ProbabilityVector([0.2, 0.3, 0.5])
        P = DiscreteTransitionMatrix([0.9 0.1; 0.2 0.8])
    end
    tf = get_transform(fe)
    itf = get_inverse_transform(fe)
    # axes-equipped (type-stable assembly path — required for Enzyme)
    @test tf.out_axes isa Tuple
    @test itf.out_axes isa Tuple
    # exact equivalence with the legacy dynamic path
    tf0 = ForwardTransform(tf.names, tf.specs)
    itf0 = InverseTransform(itf.names, itf.specs)
    θu = get_θ0_untransformed(fe)
    θt = tf(θu)
    @test collect(θt) == collect(tf0(θu))
    @test getaxes(θt) == getaxes(tf0(θu))
    @test collect(itf(θt)) == collect(itf0(θt))
    g_new = ForwardDiff.gradient(
        v -> sum(abs2, collect(itf(ComponentArray(v, getaxes(θt))))), collect(θt))
    g_old = ForwardDiff.gradient(
        v -> sum(abs2, collect(itf0(ComponentArray(v, getaxes(θt))))), collect(θt))
    @test isapprox(g_new, g_old; rtol = 1e-12)
end

@testset "SoftTree: positional flat rebuild ≡ Restructure; BLAS-free eval ≡ reference" begin
    tree = SoftTree(2, 2, 1)
    p0 = init_params(tree, Xoshiro(0))
    flat, recon = destructure_params(p0)
    p_pos = NoLimits.softtree_params_from_flat(flat, tree)
    p_ref = recon(flat)
    @test p_pos.node_weights == p_ref.node_weights
    @test p_pos.node_biases == p_ref.node_biases
    @test p_pos.leaf_values == p_ref.leaf_values
    # round-trip pins the flat layout
    @test destructure_params(p_pos)[1] == flat
    # BLAS-free eval matches the dot/matvec reference
    x = [0.3, -0.2]
    y_ref = let params = p0
        probs = [1.0]
        for level in 0:1
            idx = (2^level):(2^(level + 1) - 1)
            pv = [1 / (1 +
                   exp(-(dot(view(params.node_weights, i, :), x) + params.node_biases[i])))
                  for i in idx]
            probs = vcat(probs .* pv, probs .* (1 .- pv))
        end
        params.leaf_values * probs
    end
    @test isapprox(tree(x, p0), y_ref; rtol = 1e-14)
end

@testset "NN model_fun: CA-axes rebuild ≡ Restructure path" begin
    chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
    fe = @fixedEffects begin
        σ = RealNumber(0.4)
        ζ = NNParameters(chain; function_name = :NN1, seed = 1)
    end
    mf = get_model_funs(fe)
    θ0 = get_θ0_untransformed(fe)
    ζp = NoLimits.get_params(fe).ζ
    xin = [0.3, -0.2]
    st0 = Lux.initialstates(Xoshiro(0), chain)
    nn_direct = first(Lux.apply(chain, xin, ζp.reconstructor(collect(θ0.ζ)), st0))
    @test isapprox(mf.NN1(xin, collect(θ0.ζ)), nn_direct; rtol = 1e-6)
    g = ForwardDiff.gradient(v -> mf.NN1(xin, v)[1], collect(θ0.ζ))
    @test all(isfinite, g)
end

@testset "NPF: positional planar rebuild round-trips; base dist structure preserved" begin
    p = NPFParameter(1, 2; seed = 1)
    chain = NoLimits._planar_chain_from_flat(p.value, p.n_input)
    @test Optimisers.destructure(chain)[1] == p.value
    q0 = MvNormal(zeros(1), I)
    d = NormalizingPlanarFlow(p.value, p.reconstructor, q0)
    @test isfinite(logpdf(d, [0.3]))
    # densifying ScalMat → PDMat would route MvNormal logpdf through LAPACK trtrs,
    # which Enzyme forward cannot handle under runtime activity
    q64 = NoLimits._adapt_base_dist(MvNormal(zeros(2), I), Float64)
    @test q64.Σ isa Distributions.PDMats.ScalMat
end

@testset "ODE solve params: flat Vector via DERHSFlat; options baked into prob.kwargs" begin
    model = @Model begin
        @fixedEffects begin
            k = RealNumber(0.8, scale = :log)
            a = RealNumber(1.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x1) ~ -k * x1
        end
        @initialDE begin
            x1 = a
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
    df = DataFrame(ID = repeat(1:2, inner = 3), t = repeat([0.0, 0.5, 1.0], outer = 2),
        y = abs.(randn(Xoshiro(1), 6)) .+ 0.5)
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θu = get_θ0_untransformed(dm.model.fixed.fixed)
    # fitting-path cache (force_saveat=true, as in _fit_no_re/laplace): saveat baked
    cache = NoLimits.build_ll_cache(dm; force_saveat = true)
    ll = NoLimits.loglikelihood(dm, θu, ComponentArray(); cache = cache,
        serialization = NoLimits.EnsembleSerial())
    @test isfinite(ll)
    prob = cache.prob_templates[1]
    @test prob.f.f isa NoLimits.DERHSFlat
    @test prob.p isa Vector{Float64}
    kw = keys(prob.kwargs)
    @test :saveat in kw
    @test :save_everystep in kw
    @test :dense in kw
    # dense fallback branch: plain cache for this model has no saveat → dense=true baked
    cache_d = NoLimits.build_ll_cache(dm)
    NoLimits.loglikelihood(dm, θu, ComponentArray(); cache = cache_d,
        serialization = NoLimits.EnsembleSerial())
    prob_d = cache_d.prob_templates[1]
    @test keys(prob_d.kwargs) == (:dense,)
    @test prob_d.kwargs[:dense] === true
    # FD gradient through the flat-p path stays finite and matches finite reality
    axs = getaxes(θu)
    g = ForwardDiff.gradient(
        v -> NoLimits.loglikelihood(dm, ComponentArray(v, axs), ComponentArray();
            cache = cache, serialization = NoLimits.EnsembleSerial()),
        collect(θu))
    @test all(isfinite, g)
end

@testset "_de_state_at: exact-time index access (no interpolation at saved points)" begin
    decay!(du, u, p, t) = (du[1] = -p[1] * u[1]; nothing)
    ts = [0.0, 0.5, 1.0]
    sol = solve(ODEProblem(decay!, [1.0], (0.0, 1.0), [0.8]), Tsit5();
        saveat = ts, abstol = 1e-10, reltol = 1e-10)
    @test NoLimits._de_state_at(sol, 1, 0.5) == sol.u[2][1]
    @test NoLimits._de_state_at(sol, 1, 0.0) == sol.u[1][1]
    # off-grid falls back to interpolation
    @test isapprox(NoLimits._de_state_at(sol, 1, 0.25), sol(0.25; idxs = 1); rtol = 1e-12)
end

@testset "_logpdf_re_static ≡ dynamic logpdf dispatch" begin
    dists = (b = Normal(0.0, 0.7), m = MvNormal(zeros(2), I))
    @test NoLimits._logpdf_re_static(keys(dists), values(dists), :b, 0.3, Float64) ==
          logpdf(dists.b, 0.3)
    @test NoLimits._logpdf_re_static(
        keys(dists), values(dists), :m, [0.1, -0.2], Float64) ==
          logpdf(dists.m, [0.1, -0.2])
    @test_throws ArgumentError NoLimits._logpdf_re_static(
        keys(dists), values(dists), :nope, 0.0, Float64)
end
