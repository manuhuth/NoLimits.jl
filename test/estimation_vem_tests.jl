using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using Random
using OptimizationOptimJL
using Statistics
using ForwardDiff
using ComponentArrays
import Optimisers

# VEM (variational EM, Tarek & Afonso 2026). Linear-Gaussian anchor: the dense Gaussian
# family is exact, so VEM must reproduce the (exact) Laplace MLE up to the sample-average
# error; a bounded-support (LogNormal) RE exercises the transport-space family.
function _vem_lin_dm(rng)
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            b = RealNumber(0.0)
            σ = RealNumber(0.5; scale = :log)
            Ω = RealPSDMatrix([0.3 0.0; 0.0 0.3]; scale = :cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
        end
        @formulas begin
            y ~ Normal((a + η[1]) + (b + η[2]) * t, σ)
        end
    end
    L = cholesky([0.4 0.15; 0.15 0.25]).L
    rows = NamedTuple[]
    for i in 1:60
        η = L * randn(rng, 2)
        for t in 0.0:1.0:5.0
            push!(rows, (; ID = i, t, y = 2.0 + η[1] + (-0.5 + η[2]) * t + 0.3 * randn(rng)))
        end
    end
    return DataModel(model, DataFrame(rows); primary_id = :ID, time_col = :t)
end

@testset "VEM: dense family reproduces the exact MLE (linear Gaussian)" begin
    dm = _vem_lin_dm(Xoshiro(11))
    rl = fit_model(dm, NoLimits.Laplace())
    rv = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(3))
    θl = get_params(rl; scale = :untransformed)
    θv = get_params(rv; scale = :untransformed)
    @test isapprox(θv.a, θl.a; atol = 0.02)
    @test isapprox(θv.b, θl.b; atol = 0.02)
    @test isapprox(θv.σ, θl.σ; rtol = 0.02)
    @test isapprox(θv.Ω, θl.Ω; atol = 0.03)
    # IS objective close to the exact marginal likelihood
    @test abs(get_objective(rv) - get_objective(rl)) < 1.0
    @test get_converged(rv)
    @test get_result(rv).notes.elbo isa Real
    # default Optimization.jl LBFGS reproduces the pre-interface Optim.optimize fit
    @test collect(get_params(rv; scale = :transformed)) ≈ [
        1.9644147957675464, -0.5044155641405206, -1.350884846083775,
        -0.4893695877661404, 0.25538065373340313, -0.8110340978938972,
    ] rtol = 1.0e-6
    @test get_iterations(rv) == 25
    # random effects for prediction
    p = NoLimits.predict(rv, get_df(dm); re_mode = :ebe)
    @test all(isfinite, p.prediction)
end

@testset "VEM: options and validation" begin
    @test_throws ErrorException NoLimits.VEM(; family = :full)
    @test_throws ErrorException NoLimits.VEM(; n_samples = 0)
    @test_throws ErrorException NoLimits.VEM(; init = :zero)
    dm = _vem_lin_dm(Xoshiro(12))
    rd = fit_model(dm, NoLimits.VEM(; family = :diagonal, n_samples = 8, n_is = 0); rng = Xoshiro(1))
    @test isfinite(get_objective(rd))
    @test get_result(rd).notes.objective_kind === :neg_elbo
    rp = fit_model(dm, NoLimits.VEM(; init = :prior, n_samples = 8); rng = Xoshiro(1))
    @test isfinite(get_objective(rp))
end

@testset "VEM: Optimization.jl optimizer, optim_kwargs and bounds" begin
    dm = _vem_lin_dm(Xoshiro(11))
    θd = collect(get_params(fit_model(dm, NoLimits.VEM(; refine = false, n_is = 0); rng = Xoshiro(3)); scale = :transformed))
    rb = fit_model(
        dm, NoLimits.VEM(; optimizer = OptimizationOptimJL.BFGS(), refine = false, n_is = 0);
        rng = Xoshiro(3)
    )
    @test get_method(rb).optimizer isa OptimizationOptimJL.BFGS
    @test collect(get_params(rb; scale = :transformed)) ≈ θd atol = 1.0e-3
    # optim_kwargs replace the defaults and reach both stages
    rm = fit_model(dm, NoLimits.VEM(; optim_kwargs = (; maxiters = 3)); rng = Xoshiro(3))
    @test get_iterations(rm) <= 3
    @test get_result(rm).notes.refine_iterations <= 3
    @test !get_converged(rm)
    # transformed-scale upper bound on `a` (MLE ≈ 1.96) is active at the fit
    lo, hi = default_bounds_from_start(dm; margin = 5.0)
    hi.a = 1.5
    rub = fit_model(dm, NoLimits.VEM(; lb = lo, ub = hi, n_is = 0); rng = Xoshiro(3))
    @test get_params(rub; scale = :transformed).a <= 1.5 + 1.0e-8
    @test get_params(rub; scale = :transformed).a > 1.4
    # all randomness (MC draws, IS draws) comes from the fit `rng`
    mc = NoLimits.VEM(; expectation = :mc, n_samples = 8, refine = false, n_is = 50)
    r1 = fit_model(dm, mc; rng = Xoshiro(7))
    r2 = fit_model(dm, mc; rng = Xoshiro(7))
    r3 = fit_model(dm, mc; rng = Xoshiro(8))
    @test collect(get_params(r1; scale = :transformed)) == collect(get_params(r2; scale = :transformed))
    @test get_objective(r1) === get_objective(r2)
    @test collect(get_params(r3; scale = :transformed)) != collect(get_params(r1; scale = :transformed))
end

@testset "VEM: planar-flow variational family" begin
    @test_throws ErrorException PlanarFlowFamily(0)
    @test_throws ErrorException PlanarFlowFamily(2; base = :full)
    @test NoLimits.VEM(; family = PlanarFlowFamily(2)).flow == 2
    # identity at the init (û = 0) and exact log-Jacobian of the layer stack
    n, K = 3, 2
    rng = Xoshiro(4)
    θ = zeros(K * (2n + 1))
    for l in 1:K
        w = 0.1 .* randn(rng, n)
        o = (l - 1) * (2n + 1)
        θ[(o + 1):(o + n)] .= w .* (log(ℯ - 1) / sum(abs2, w))
        θ[(o + n + 1):(o + 2n)] .= w
    end
    ξ = randn(rng, n)
    z, ld = NoLimits._vem_flow(θ, 0, ξ, n, K)
    @test z ≈ ξ atol = 1.0e-12
    @test abs(ld) < 1.0e-12
    θr = randn(Xoshiro(5), K * (2n + 1))
    _, ldr = NoLimits._vem_flow(θr, 0, ξ, n, K)
    J = ForwardDiff.jacobian(x -> NoLimits._vem_flow(θr, 0, x, n, K)[1], ξ)
    @test ldr ≈ logabsdet(J)[1] rtol = 1.0e-10
    # Gaussian posterior: the flow family contains the exact q, so the fit matches Laplace
    dm = _vem_lin_dm(Xoshiro(11))
    rl = fit_model(dm, NoLimits.Laplace())
    fam = NoLimits.VEM(; family = PlanarFlowFamily(1))
    rf = fit_model(dm, fam; rng = Xoshiro(3))
    θl = get_params(rl; scale = :untransformed)
    θf = get_params(rf; scale = :untransformed)
    @test isapprox(θf.a, θl.a; atol = 0.03)
    @test isapprox(θf.b, θl.b; atol = 0.03)
    @test isapprox(θf.σ, θl.σ; rtol = 0.03)
    @test isapprox(θf.Ω, θl.Ω; atol = 0.04)
    @test abs(get_objective(rf) - get_objective(rl)) < 1.0
    @test all(q -> all(isfinite, q.mean) && all(isfinite, q.scale), get_notes(rf).variational_params)
    rf2 = fit_model(dm, fam; rng = Xoshiro(3))
    @test collect(get_params(rf2; scale = :transformed)) == collect(get_params(rf; scale = :transformed))
end

@testset "VEM: MiniBatch (stochastic VEM)" begin
    import Optimisers
    dm = _vem_lin_dm(Xoshiro(11))
    K = length(NoLimits.get_individuals(dm))
    base = NoLimits.VEM(; refine = false, n_is = 0)
    r0 = fit_model(dm, base; rng = Xoshiro(3))
    # every batch selected + the same LBFGS reproduces the unwrapped fit
    rf = @test_logs (:warn, r"stochastic") fit_model(
        dm, NoLimits.MiniBatch(base; schedule = K, optimizer = base.optimizer); rng = Xoshiro(3)
    )
    @test collect(get_params(rf; scale = :transformed)) ≈
        collect(get_params(r0; scale = :transformed)) rtol = 1.0e-8
    @test get_objective(rf) ≈ get_objective(r0) rtol = 1.0e-10
    # the optimizer sees (K/|B|) Σ_B (-ELBO_i): two halves average to the full objective
    obj(sched) = get_result(
        fit_model(
            dm,
            NoLimits.MiniBatch(
                base; schedule = sched, optimizer = Optimisers.Adam(0.0),
                optim_kwargs = (; maxiters = 1)
            );
            rng = Xoshiro(3)
        )
    ).solution.objective
    half = K ÷ 2
    @test (obj((n, it, r) -> 1:half) + obj((n, it, r) -> (half + 1):n)) / 2 ≈ obj(:all) rtol = 1.0e-10
    # default Adam(0.01), 1000 iterations on minibatches of 10 lands near the full fit
    # (measured: a 1.953 vs 1.964, b -0.498 vs -0.504, σ 0.264 vs 0.259)
    mb = NoLimits.MiniBatch(NoLimits.VEM(; n_is = 0); schedule = 10)
    ra = fit_model(dm, mb; rng = Xoshiro(5))
    θa, θ0 = get_params(ra; scale = :untransformed), get_params(r0; scale = :untransformed)
    @test isapprox(θa.a, θ0.a; atol = 0.05)
    @test isapprox(θa.b, θ0.b; atol = 0.03)
    @test isapprox(θa.σ, θ0.σ; rtol = 0.05)
    @test get_result(ra).notes.refine_iterations == 0
    @test collect(get_params(fit_model(dm, mb; rng = Xoshiro(5)); scale = :transformed)) ==
        collect(get_params(ra; scale = :transformed))
end

# Holds θ (and q) at the start: Adam with learning rate 0 for one iteration (both stages).
_vem_frozen(; kw...) = NoLimits.VEM(;
    optimizer = Optimisers.Adam(0.0), optim_kwargs = (; maxiters = 1), kw...
)

@testset "VEM: penalty and extra_objective" begin
    dm = _vem_lin_dm(Xoshiro(11))
    fe = NoLimits.get_fixed(NoLimits.get_model(dm))
    θ0 = NoLimits.get_θ0_untransformed(fe)
    pen = (; σ = 2.0)
    extra = θ -> (θ.a - 1.0)^2 + sum(abs2, θ.Ω)
    term(θ) = 2.0 * θ.σ^2 + extra(θ)

    # objective = unpenalized objective + penalty + extra_objective at the same θ; the
    # notes keep the plain ELBO
    r0 = fit_model(dm, _vem_frozen(); rng = Xoshiro(3))
    rp = fit_model(dm, _vem_frozen(); penalty = pen, extra_objective = extra, rng = Xoshiro(3))
    θh = get_params(r0; scale = :untransformed)
    @test collect(θh) ≈ collect(θ0) rtol = 1.0e-12
    @test collect(get_params(rp; scale = :untransformed)) == collect(θh)
    @test get_objective(rp) ≈ get_objective(r0) + term(θh) rtol = 1.0e-12
    @test get_result(rp).notes.elbo ≈ get_result(r0).notes.elbo rtol = 1.0e-12
    @test get_result(rp).solution.objective ≈ get_result(r0).solution.objective + term(θh) rtol = 1.0e-12
    rn = fit_model(dm, _vem_frozen(; n_is = 0); penalty = pen, extra_objective = extra, rng = Xoshiro(3))
    @test get_objective(rn) ≈ -get_result(rn).notes.elbo + term(θh) rtol = 1.0e-12

    # gradient: one plain gradient step (MiniBatch keeps the fixed-effect scaling at 1), so
    # θ1 = θ0 - η ∇f and the difference of the penalized and plain steps is η ∇term;
    # the term is not scaled by K/|B| when only half of the batches are used
    K = length(NoLimits.get_individuals(dm))
    η = 1.0e-2
    tr, itr = NoLimits.get_transform(fe), NoLimits.get_inverse_transform(fe)
    θt0 = tr(θ0)
    tt(v) = term(itr(ComponentArray(v, getaxes(θt0))))
    gref = ForwardDiff.gradient(tt, collect(θt0))
    h = 1.0e-6
    gfd = map(eachindex(θt0)) do i
        e = zeros(length(θt0))
        e[i] = h
        (tt(collect(θt0) .+ e) - tt(collect(θt0) .- e)) / (2h)
    end
    @test gref ≈ gfd rtol = 1.0e-6
    for sched in (K, K ÷ 2)
        mb = NoLimits.MiniBatch(
            NoLimits.VEM(; refine = false, n_is = 0); schedule = sched,
            optimizer = Optimisers.Descent(η), optim_kwargs = (; maxiters = 1)
        )
        s0 = fit_model(dm, mb; rng = Xoshiro(3))
        sp = fit_model(dm, mb; penalty = pen, extra_objective = extra, rng = Xoshiro(3))
        Δ = (collect(get_params(s0; scale = :transformed)) .- collect(get_params(sp; scale = :transformed))) ./ η
        @test Δ ≈ gref rtol = 1.0e-7
        mb0 = NoLimits.MiniBatch(
            NoLimits.VEM(; refine = false, n_is = 0); schedule = sched,
            optimizer = Optimisers.Adam(0.0), optim_kwargs = (; maxiters = 1)
        )
        o0 = get_result(fit_model(dm, mb0; rng = Xoshiro(3))).solution.objective
        op = get_result(
            fit_model(dm, mb0; penalty = pen, extra_objective = extra, rng = Xoshiro(3))
        ).solution.objective
        @test op ≈ o0 + term(θh) rtol = 1.0e-12
    end

    # a penalty moves the estimates as for Laplace (exact for this linear Gaussian model)
    pa = (; a = 50.0)
    eb = θ -> 30.0 * (θ.b + 1.0)^2
    rv0 = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(3))
    rv = fit_model(dm, NoLimits.VEM(); penalty = pa, extra_objective = eb, rng = Xoshiro(3))
    rl = fit_model(dm, NoLimits.Laplace(); penalty = pa, extra_objective = eb)
    θv0, θv, θl = (get_params(r; scale = :untransformed) for r in (rv0, rv, rl))
    @test get_converged(rv)
    @test θv.a < θv0.a - 0.5
    @test θv.b < θv0.b - 0.05
    @test isapprox(θv.a, θl.a; atol = 0.02)
    @test isapprox(θv.b, θl.b; atol = 0.02)
    @test isapprox(θv.σ, θl.σ; rtol = 0.02)
    @test isapprox(θv.Ω, θl.Ω; atol = 0.03)
    @test abs(get_objective(rv) - get_objective(rl)) < 1.0
    # Wald UQ includes the terms (as for Laplace)
    iv = get_uq_intervals(compute_uq(rv; method = :wald))
    @test iv.lower.a < θv.a < iv.upper.a
end

@testset "VEM: DataModel weights" begin
    dm = _vem_lin_dm(Xoshiro(11))
    model = NoLimits.get_model(dm)
    df = get_df(dm)
    mk(d; kw...) = DataModel(model, d; primary_id = :ID, time_col = :t, kw...)
    j = 7
    df1 = copy(df)
    df1.w = ones(nrow(df1))
    df1.w2 = [id == j ? 2.0 : 1.0 for id in df1.ID]
    dup = df[df.ID .== j, :]
    dup.ID .= 1000
    dm1, dmw, dmd = mk(df1; weights = :w), mk(df1; weights = :w2), mk(vcat(df, dup))

    # unit weights reproduce the unweighted fit exactly
    r0 = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(3))
    r1 = fit_model(dm1, NoLimits.VEM(); rng = Xoshiro(3))
    @test collect(get_params(r1; scale = :transformed)) == collect(get_params(r0; scale = :transformed))
    @test get_objective(r1) == get_objective(r0)
    @test get_result(r1).notes.elbo == get_result(r0).notes.elbo

    # weight 2 equals a duplicated individual: ELBO and objective at the start, one gradient
    # step, and the fit
    fw = fit_model(dmw, _vem_frozen(); rng = Xoshiro(3))
    fd = fit_model(dmd, _vem_frozen(); rng = Xoshiro(3))
    @test get_result(fw).notes.elbo ≈ get_result(fd).notes.elbo rtol = 1.0e-10
    @test get_objective(fw) ≈ get_objective(fd) rtol = 1.0e-10
    f0 = fit_model(dm, _vem_frozen(); rng = Xoshiro(3))
    @test !isapprox(get_objective(fw), get_objective(f0); rtol = 1.0e-3)
    step(d, K) = fit_model(
        d,
        NoLimits.MiniBatch(
            NoLimits.VEM(; refine = false, n_is = 0); schedule = K,
            optimizer = Optimisers.Descent(1.0e-2), optim_kwargs = (; maxiters = 1)
        );
        rng = Xoshiro(3)
    )
    @test collect(get_params(step(dmw, 60); scale = :transformed)) ≈
        collect(get_params(step(dmd, 61); scale = :transformed)) rtol = 1.0e-10
    rw = fit_model(dmw, NoLimits.VEM(); rng = Xoshiro(3))
    rd = fit_model(dmd, NoLimits.VEM(); rng = Xoshiro(3))
    @test get_converged(rw) && get_converged(rd)
    @test collect(get_params(rw; scale = :untransformed)) ≈
        collect(get_params(rd; scale = :untransformed)) atol = 2.0e-3
    @test get_objective(rw) ≈ get_objective(rd) atol = 1.0e-3

    # MiniBatch: weighted batch terms scale by K/|B| like unweighted ones
    obj(sched) = get_result(
        fit_model(
            dmw,
            NoLimits.MiniBatch(
                NoLimits.VEM(; refine = false, n_is = 0); schedule = sched,
                optimizer = Optimisers.Adam(0.0), optim_kwargs = (; maxiters = 1)
            );
            rng = Xoshiro(3)
        )
    ).solution.objective
    @test (obj((n, it, r) -> 1:30) + obj((n, it, r) -> 31:n)) / 2 ≈ obj(:all) rtol = 1.0e-10

    # levels fixed by constants_re enter as for Laplace: a batch without free levels
    # contributes log p(y_b, b_fixed | θ) (at the start: a = b = 0, η = 0)
    cre = (; η = Dict(1 => [0.0, 0.0]))
    rc = fit_model(dm, _vem_frozen(); constants_re = cre, rng = Xoshiro(3))
    rr = fit_model(mk(df[df.ID .!= 1, :]), _vem_frozen(); rng = Xoshiro(3))
    θc = get_params(rc; scale = :untransformed)
    ll1 = sum(logpdf.(Normal(0.0, θc.σ), df.y[df.ID .== 1])) +
        logpdf(MvNormal(zeros(2), Matrix(θc.Ω)), zeros(2))
    @test get_objective(rc) ≈ get_objective(rr) - ll1 rtol = 1.0e-10
    @test get_result(rc).notes.elbo ≈ get_result(rr).notes.elbo + ll1 rtol = 1.0e-10
    # at the start q is the exact posterior of this linear Gaussian model, so the reported
    # objective is the exact marginal likelihood, as Laplace's; also with a fixed level in
    # a batch with free levels (prior density of the fixed site level)
    lfrozen = NoLimits.Laplace(; optimizer = Optimisers.Adam(0.0), optim_kwargs = (; maxiters = 1))
    @test get_objective(rc) ≈ get_objective(fit_model(dm, lfrozen; constants_re = cre)) rtol = 1.0e-6
    for c in (NamedTuple(), (; η_site = Dict(:A => 0.3)))
        ov = get_objective(fit_model(fx_mg_dm(), _vem_frozen(); constants_re = c, rng = Xoshiro(3)))
        ol = get_objective(fit_model(fx_mg_dm(), lfrozen; constants_re = c))
        @test ov ≈ ol rtol = 1.0e-6
    end
    rcf = fit_model(dm, NoLimits.VEM(); constants_re = cre, rng = Xoshiro(3))
    rcl = fit_model(dm, NoLimits.Laplace(); constants_re = cre)
    @test collect(get_params(rcf; scale = :untransformed)) ≈
        collect(get_params(rcl; scale = :untransformed)) atol = 0.03
end

@testset "VEM: model bounds, ignore_model_bounds, optimizers without bounds support" begin
    dm = _vem_lin_dm(Xoshiro(11))
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0; upper = 1.5)
            b = RealNumber(0.0)
            σ = RealNumber(0.5; scale = :log)
            Ω = RealPSDMatrix([0.3 0.0; 0.0 0.3]; scale = :cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
        end
        @formulas begin
            y ~ Normal((a + η[1]) + (b + η[2]) * t, σ)
        end
    end
    dmb = DataModel(model, get_df(dm); primary_id = :ID, time_col = :t)
    # the declared bound binds in the joint fit and in the refinement (MLE a ≈ 1.96)
    # (refinement without bounds would move a back above 1.5)
    rb = fit_model(dmb, NoLimits.VEM(); rng = Xoshiro(3))
    @test 1.4 < get_params(rb; scale = :untransformed).a <= 1.5 + 1.0e-8
    ri = fit_model(dmb, NoLimits.VEM(; ignore_model_bounds = true); rng = Xoshiro(3))
    @test get_params(ri; scale = :untransformed).a > 1.8
    # the variational parameters are unbounded: optimizers that need a finite box are rejected
    @test_throws "needs finite bounds" fit_model(
        dmb, NoLimits.VEM(; optimizer = OptimizationOptimJL.ParticleSwarm())
    )
end

@testset "VEM: get_variational_distributions maps q to individuals" begin
    dm0 = _vem_lin_dm(Xoshiro(11))
    df = copy(get_df(dm0))
    # a distinctive individual: its q mean of η[1] is far above the others
    df.y[df.ID .== 17] .+= 5.0
    dm = DataModel(NoLimits.get_model(dm0), df; primary_id = :ID, time_col = :t)
    rv = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(3))
    notes = get_notes(rv)
    _, infos, _ = NoLimits.build_re_batch_infos(dm, NamedTuple())
    # the batch order differs from the individual order here
    batch_first = [first(NoLimits.get_batch_individuals(infos[bi])) for bi in notes.variational_batches]
    @test batch_first != 1:60
    @test argmax([q.mean[1] for q in notes.variational_params]) != 17

    vd = get_variational_distributions(rv)
    @test vd.ID == unique(df.ID)
    @test names(vd) == ["ID", "batch", "mean", "scale", "space", "coordinates"]
    @test argmax([m[1] for m in vd.mean]) == 17
    for i in 1:60
        k = vd.batch[i]
        @test batch_first[k] == i
        @test vd.mean[i] == notes.variational_params[k].mean
        @test vd.scale[i] == notes.variational_params[k].scale
        @test vd.space[i] === :natural
        @test vd.coordinates[i] == [(re = :η, level = i, component = 1), (re = :η, level = i, component = 2)]
    end
    @test istril(vd.scale[17])
    @test get_variational_distributions(dm, rv) == vd

    # shared levels: individuals of one site share one joint q over all its levels
    r = fit_model(fx_mg_dm(), NoLimits.VEM(; n_is = 50, optim_kwargs = (; maxiters = 5)); rng = Xoshiro(1))
    v = get_variational_distributions(r)
    @test v.batch[1] == v.batch[2] != v.batch[3] == v.batch[4]
    @test v.mean[1] == v.mean[2]
    for i in 1:4
        @test length(v.coordinates[i]) == length(v.mean[i]) == 3
        @test (re = :η_id, level = v.ID[i], component = 1) in v.coordinates[i]
    end
    @test (re = :η_site, level = :A, component = 1) in v.coordinates[1]
    @test (re = :η_site, level = :B, component = 1) in v.coordinates[3]

    # levels fixed by constants_re: no q for that individual
    rc = fit_model(
        dm0, NoLimits.VEM(; n_is = 0, optim_kwargs = (; maxiters = 5));
        constants_re = (; η = Dict(1 => [0.0, 0.0])), rng = Xoshiro(3)
    )
    vc = get_variational_distributions(rc)
    @test vc.batch[1] == 0 && isempty(vc.mean[1]) && vc.space[1] === :none
    @test all(>(0), vc.batch[2:end])
    @test_throws "requires a VEM fit" get_variational_distributions(fit_model(dm0, NoLimits.Laplace()))
end

# Opt-in like enzyme_compat_proxy_tests.jl: Enzyme is not a test dependency.
if get(ENV, "NOLIMITS_TEST_ENZYME", "false") == "true" && VERSION >= v"1.12.5" &&
        Base.find_package("Enzyme") !== nothing
    using Enzyme
    @testset "VEM: Enzyme reverse mode matches ForwardDiff" begin
        dm = _vem_lin_dm(Xoshiro(11))
        ad = NoLimits.Optimization.AutoEnzyme(;
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Duplicated
        )
        rf = fit_model(dm, NoLimits.VEM(; n_is = 0); rng = Xoshiro(3))
        re = fit_model(dm, NoLimits.VEM(; n_is = 0, adtype = ad); rng = Xoshiro(3))
        @test get_method(re).adtype === ad
        @test collect(get_params(re; scale = :transformed)) ≈
            collect(get_params(rf; scale = :transformed)) rtol = 1.0e-6
        # per-batch reverse passes are threaded and reduced in batch order
        rs = fit_model(
            dm, NoLimits.VEM(; n_is = 0, adtype = ad); rng = Xoshiro(3),
            serialization = NoLimits.EnsembleSerial()
        )
        @test collect(get_params(rs; scale = :transformed)) ==
            collect(get_params(re; scale = :transformed))
        @test get_objective(rs) == get_objective(re)
        # MiniBatch: the minibatch enters the reverse pass as constant contexts
        import Optimisers
        mbv(adt) = NoLimits.MiniBatch(
            NoLimits.VEM(; n_is = 0, refine = false, adtype = adt); schedule = 10,
            optimizer = Optimisers.Adam(0.05), optim_kwargs = (; maxiters = 20)
        )
        rmf = fit_model(dm, mbv(NoLimits.Optimization.AutoForwardDiff()); rng = Xoshiro(3))
        rme = fit_model(dm, mbv(ad); rng = Xoshiro(3))
        @test collect(get_params(rme; scale = :transformed)) ≈
            collect(get_params(rmf; scale = :transformed)) rtol = 1.0e-6
    end
    @testset "VEM: Enzyme reverse mode with a planar-flow family" begin
        dm = _vem_lin_dm(Xoshiro(11))
        ad = NoLimits.Optimization.AutoEnzyme(;
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Duplicated
        )
        fam = PlanarFlowFamily(1)
        rf = fit_model(dm, NoLimits.VEM(; family = fam, n_is = 0); rng = Xoshiro(3))
        re = fit_model(dm, NoLimits.VEM(; family = fam, n_is = 0, adtype = ad); rng = Xoshiro(3))
        @test collect(get_params(re; scale = :transformed)) ≈
            collect(get_params(rf; scale = :transformed)) rtol = 1.0e-4
    end
end

@testset "VEM: LogNormal random effect (transport-space family)" begin
    model = @Model begin
        @fixedEffects begin
            k = RealNumber(1.0; scale = :log)
            ω = RealNumber(0.3; scale = :log)
            σ = RealNumber(0.3; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(10 * exp(-k * η * t), σ)
        end
    end
    rng = Xoshiro(5)
    rows = NamedTuple[]
    for i in 1:40
        η = exp(0.4 * randn(rng))
        for t in (0.25, 0.5, 1.0, 2.0)
            push!(rows, (; ID = i, t, y = 10 * exp(-0.7 * η * t) + 0.2 * randn(rng)))
        end
    end
    dm = DataModel(model, DataFrame(rows); primary_id = :ID, time_col = :t)
    rl = fit_model(dm, NoLimits.Laplace())
    rv = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(2))
    θl = get_params(rl; scale = :untransformed)
    θv = get_params(rv; scale = :untransformed)
    @test isapprox(θv.k, θl.k; rtol = 0.05)
    @test isapprox(θv.ω, θl.ω; rtol = 0.15)
    @test isapprox(θv.σ, θl.σ; rtol = 0.05)
end

@testset "VEM: threaded fit equals serial bitwise" begin
    dm = _vem_lin_dm(Xoshiro(11))
    for m in (NoLimits.VEM(), NoLimits.VEM(; expectation = :mc, n_samples = 8, n_is = 50, optim_kwargs = (; maxiters = 10)))
        rt = fit_model(dm, m; rng = Xoshiro(3))
        rs = fit_model(dm, m; rng = Xoshiro(3), serialization = NoLimits.EnsembleSerial())
        @test collect(get_params(rt; scale = :transformed)) == collect(get_params(rs; scale = :transformed))
        @test get_objective(rt) === get_objective(rs)
        @test get_result(rt).notes.elbo === get_result(rs).notes.elbo
        @test get_result(rt).eb_modes == get_result(rs).eb_modes
    end
end

@testset "VEM: Wald UQ, conditional draws from q, CV with constants_re" begin
    dm = _vem_lin_dm(Xoshiro(11))
    rv = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(3))
    @test uq_family(get_method(rv)) === :wald_re
    uq = compute_uq(rv; method = :wald)
    iv = get_uq_intervals(uq)
    θv = get_params(rv; scale = :untransformed)
    @test iv.lower.a < θv.a < iv.upper.a
    @test 0 < iv.upper.a - iv.lower.a < 1

    # draws come from the fitted variational q (b = mean + scale ξ), not a Laplace Gaussian
    bstars, bis, θu, cc, llc, _ = NoLimits._resolve_bstars_for_re(dm, rv, NamedTuple())
    draws = NoLimits._sample_conditional_bstars(dm, bis, bstars, θu, cc, llc, rv, 4000, Xoshiro(1))
    q = get_result(rv).notes.variational_params[1]
    B = reduce(hcat, [d[1] for d in draws])
    @test vec(mean(B; dims = 2)) ≈ q.mean atol = 0.01
    @test cov(B; dims = 2) ≈ q.scale * q.scale' rtol = 0.1
    sre = sample_random_effects(rv; n_samples = 3, rng = Xoshiro(1))
    @test nrow(sre.η) == 3 * 60

    @test NoLimits._cv_method_accepts_constants_re(NoLimits.VEM())
    cv = cross_validate(dm, 2; kind = :observation, rng = Xoshiro(4))
    cres = fit_cv(
        cv, NoLimits.VEM(; n_is = 0, refine = false, optim_kwargs = (; maxiters = 20));
        constants_re = (; η = Dict(1 => [0.0, 0.0])), seen_re_mode = :conditional,
        n_mc_samples = 10, store_results = true, rng = Xoshiro(4)
    )
    @test isfinite(cres.mean_test_loglikelihood)
    for fr in cres.fold_results
        @test haskey(NoLimits.get_fit_kwargs(fr.fit_result).constants_re, :η)
    end
end

@testset "VEM: model errors propagate, discrete random effects are rejected" begin
    model = @Model begin
        @helpers begin
            vem_guard(x) = x > 1.5 ? error("vem_guard: user error") : x
        end
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5; scale = :log)
            ω = RealNumber(0.5; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(vem_guard(a) + η, σ)
        end
    end
    df = DataFrame(ID = repeat(1:8, inner = 3), t = repeat(0.0:2.0, 8), y = 3.0 .+ 0.1 .* randn(Xoshiro(1), 24))
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    # used to be scored -Inf, so the fit silently stalled at a = 1.5
    @test_throws "vem_guard: user error" fit_model(dm, NoLimits.VEM(; n_is = 0); rng = Xoshiro(1))

    dmodel = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Poisson(2.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    ddm = DataModel(dmodel, df; primary_id = :ID, time_col = :t)
    @test_throws "VEM does not support discrete" fit_model(ddm, NoLimits.VEM())
end

# Smoke coverage on shared fixtures (few iterations; not accuracy checks).
@testset "VEM: ODE, Poisson outcome, two RE groupings, Multistart" begin
    m = NoLimits.VEM(; n_is = 50, optim_kwargs = (; maxiters = 5))
    for dm in (fx_ode_dm(), fx_pois_dm(), fx_mg_dm())
        r = fit_model(dm, m; rng = Xoshiro(1))
        @test isfinite(get_objective(r))
        @test all(isfinite, collect(get_params(r; scale = :untransformed)))
        @test length(get_result(r).eb_modes) == length(get_result(r).notes.variational_params)
    end
    ms = NoLimits.Multistart(; dists = (; a = Normal(0.2, 0.3)), n_draws_requested = 3, n_draws_used = 2)
    rm = fit_model(ms, fx_pois_dm(), m; rng = Xoshiro(2))
    @test rm isa NoLimits.MultistartFitResult
    @test length(NoLimits.get_multistart_results(rm)) == 2
    @test isfinite(get_objective(rm))
end
