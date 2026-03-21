#!/usr/bin/env julia
# Diagnostic script: when/why do Laplace outer gradients become NaN?
# Usage: julia --project scripts/laplace_nan_diagnostic.jl

using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using OrdinaryDiffEq
using Random
using ForwardDiff

# ============================================================
# Infrastructure helpers
# ============================================================

function make_fresh_ebe_cache(dm, Tθ, n_batches)
    bstar_cache = NoLimits._LaplaceBStarCache(
        [Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = NoLimits._LaplaceGradCache(
        [Vector{Tθ}() for _ in 1:n_batches],
        fill(Tθ(NaN), n_batches),
        [Vector{Tθ}() for _ in 1:n_batches],
        falses(n_batches))
    ad_cache  = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(Tθ, n_batches)
    return NoLimits._LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
end

function setup_scenario(dm; method=NoLimits.Laplace(), ode_args=(), ode_kwargs=NamedTuple())
    fe = dm.model.fixed.fixed
    θ0_t  = NoLimits.get_θ0_transformed(fe)
    Tθ    = eltype(θ0_t)
    pairing, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NoLimits.build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)
    n_batches = length(batch_infos)
    inner_opts = NoLimits._resolve_inner_options(method.inner, dm)
    ms_opts = NoLimits._resolve_multistart_options(method.multistart, inner_opts)
    return (; batch_infos, const_cache, ll_cache, inner_opts, ms_opts,
              n_batches, Tθ, method)
end

function eval_grad(dm, θu, sc; fresh=true)
    ebe_cache = make_fresh_ebe_cache(dm, sc.Tθ, sc.n_batches)
    obj, grad, bstars = NoLimits._laplace_objective_and_grad(
        dm, sc.batch_infos, θu, sc.const_cache, sc.ll_cache, ebe_cache;
        inner=sc.inner_opts,
        hessian=sc.method.hessian,
        cache_opts=sc.method.cache,
        multistart=sc.ms_opts,
        fastpath=nothing,
        rng=Xoshiro(0))
    return obj, grad, bstars
end

# ============================================================
# Sub-component diagnosis for a single batch with NaN gradient
# ============================================================

function diagnose_batch(dm, info, θu, b_star, const_cache, ll_cache, bi, method)
    d = Dict{String, Any}()

    # --- logf at b* -------------------------------------------------------
    try
        logf = NoLimits._laplace_logf_batch(dm, info, θu, b_star, const_cache, ll_cache)
        d["logf"] = logf
        d["logf_ok"] = isfinite(logf)
    catch e
        d["logf"] = nothing
        d["logf_ok"] = false
        d["logf_err"] = sprint(showerror, e)
    end

    # --- Hessian H at b* --------------------------------------------------
    try
        H = NoLimits._laplace_hessian_b(dm, info, θu, b_star, const_cache, ll_cache, nothing, bi)
        d["H_nan"] = any(isnan, H)
        d["H_inf"] = any(isinf, H)
        if !d["H_nan"] && !d["H_inf"]
            ev = eigvals(Symmetric(-H))
            d["H_min_eig"] = minimum(ev)
            d["H_neg_def"] = d["H_min_eig"] > 0
        else
            d["H_min_eig"] = NaN
            d["H_neg_def"] = false
        end

        # Cholesky
        chol, jit = NoLimits._laplace_cholesky_negH(H;
            jitter=method.hessian.jitter,
            max_tries=method.hessian.max_tries,
            growth=method.hessian.growth)
        d["chol_ok"]  = chol !== nothing && chol.info == 0
        d["jitter"]   = jit

        if d["chol_ok"]
            n = length(b_star)
            # --- grad_logf (envelope term) --------------------------------
            try
                gl = ForwardDiff.gradient(
                    θv -> NoLimits._laplace_logf_batch(dm, info, θv, b_star, const_cache, ll_cache),
                    collect(θu))
                d["grad_logf_nan"] = any(isnan, gl)
                d["grad_logf_inf"] = any(isinf, gl)
            catch e
                d["grad_logf_nan"] = true
                d["grad_logf_err"] = sprint(showerror, e)
            end

            # --- Jacobian Gθ (∂²logf / ∂b ∂θ) ----------------------------
            try
                tmp = zeros(n)
                Gθ = ForwardDiff.jacobian(
                    (out, θv) -> begin
                        fb = bv -> NoLimits._laplace_logf_batch(dm, info, θv, bv, const_cache, ll_cache)
                        ForwardDiff.gradient!(out, fb, b_star)
                    end,
                    tmp, collect(θu))
                d["Gθ_nan"] = any(isnan, Gθ)
                d["Gθ_inf"] = any(isinf, Gθ)
            catch e
                d["Gθ_nan"] = true
                d["Gθ_err"] = sprint(showerror, e)
            end

            # --- logdet gradient (trace method: ∂vech(H)/∂θ) --------------
            try
                Ainv = chol \ Matrix{Float64}(I, n, n)
                weights = Vector{Float64}(undef, NoLimits._ntri(n))
                NoLimits._vech_weights!(weights, Ainv)
                Jθ = ForwardDiff.jacobian(
                    θv -> begin
                        Hθ = NoLimits._laplace_hessian_b(dm, info, θv, b_star, const_cache, ll_cache, nothing, bi)
                        NoLimits._vech(Hθ)
                    end,
                    collect(θu))
                gld = -(Jθ' * weights)
                d["grad_logdet_θ_nan"] = any(isnan, gld)
                d["grad_logdet_θ_inf"] = any(isinf, gld)
            catch e
                d["grad_logdet_θ_nan"] = true
                d["grad_logdet_θ_err"] = sprint(showerror, e)
            end

            # --- corr term (grad_logdet_b' * dbdθ) ------------------------
            try
                Ainv = chol \ Matrix{Float64}(I, n, n)
                weights = Vector{Float64}(undef, NoLimits._ntri(n))
                NoLimits._vech_weights!(weights, Ainv)
                Jb = ForwardDiff.jacobian(
                    bv -> begin
                        Hb = NoLimits._laplace_hessian_b(dm, info, θu, bv, const_cache, ll_cache, nothing, bi)
                        NoLimits._vech(Hb)
                    end,
                    b_star)
                gld_b = -(Jb' * weights)
                tmp2 = zeros(n)
                Gθ2 = ForwardDiff.jacobian(
                    (out, θv) -> begin
                        fb = bv -> NoLimits._laplace_logf_batch(dm, info, θv, bv, const_cache, ll_cache)
                        ForwardDiff.gradient!(out, fb, b_star)
                    end,
                    tmp2, collect(θu))
                if !any(isnan, Gθ2) && !any(isnan, gld_b)
                    dbdθ = chol \ Gθ2
                    corr = vec(gld_b' * dbdθ)
                    d["corr_nan"] = any(isnan, corr)
                    d["corr_inf"] = any(isinf, corr)
                end
            catch e
                d["corr_nan"] = true
                d["corr_err"] = sprint(showerror, e)
            end
        end
    catch e
        d["H_nan"] = true
        d["H_err"] = sprint(showerror, e)
    end

    return d
end

function summarise_cause(d)
    !get(d, "logf_ok", true)       && return "logf non-finite at b*"
    get(d, "H_nan", false)         && return "Hessian has NaN"
    get(d, "H_inf", false)         && return "Hessian has Inf"
    !get(d, "H_neg_def", true)     && return "H not neg-def (min_eig=$(round(get(d,"H_min_eig",NaN),sigdigits=2)))"
    !get(d, "chol_ok", true)       && return "Cholesky of −H failed"
    get(d, "grad_logf_nan", false) && return "grad_logf (envelope term) is NaN"
    get(d, "grad_logf_inf", false) && return "grad_logf (envelope term) is Inf"
    get(d, "Gθ_nan", false)        && return "Jacobian Gθ (∂²logf/∂b∂θ) is NaN"
    get(d, "Gθ_inf", false)        && return "Jacobian Gθ is Inf"
    get(d, "grad_logdet_θ_nan", false) && return "grad_logdet_θ (trace method) is NaN"
    get(d, "grad_logdet_θ_inf", false) && return "grad_logdet_θ is Inf"
    get(d, "corr_nan", false)      && return "correction term (corr) is NaN"
    for (k, v) in d
        endswith(k, "_err") && return "Error in $(replace(k, "_err"=>"")): $v"
    end
    return "cause unknown"
end

# ============================================================
# Run a scenario: scan θ grid, report NaN frequency + cause
# ============================================================

function run_scenario(name, dm, θ_grid; method=NoLimits.Laplace())
    println("\n" * "="^72)
    println("SCENARIO: $name")
    println("="^72)

    sc = setup_scenario(dm; method=method)
    n_total = length(θ_grid)
    n_nan   = 0
    causes  = Dict{String, Int}()

    for θu in θ_grid
        obj, grad, bstars = eval_grad(dm, θu, sc)
        has_nan = any(isnan, grad)
        has_nan || continue
        n_nan += 1

        # Diagnose the first offending batch
        for (bi, info) in enumerate(sc.batch_infos)
            bst = bstars[bi]
            if isempty(bst)
                k = "b* empty (inner opt failed)"
                causes[k] = get(causes, k, 0) + 1
                break
            end
            bg = NoLimits._laplace_grad_batch(
                dm, info, θu, bst, sc.const_cache, sc.ll_cache, nothing, bi;
                jitter=method.hessian.jitter,
                max_tries=method.hessian.max_tries,
                growth=method.hessian.growth,
                use_trace_logdet_grad=method.hessian.use_trace_logdet_grad)
            if any(isnan, bg.grad)
                d = diagnose_batch(dm, info, θu, bst, sc.const_cache, sc.ll_cache, bi, method)
                c = summarise_cause(d)
                causes[c] = get(causes, c, 0) + 1
                break
            end
        end
    end

    pct = n_total > 0 ? round(100.0 * n_nan / n_total; digits=1) : 0.0
    println("  θ points tested : $n_total")
    println("  NaN gradients   : $n_nan  ($pct %)")
    if isempty(causes)
        println("  No NaN gradients detected.")
    else
        println("  Breakdown by cause:")
        for (c, cnt) in sort(collect(causes); by=x -> -x[2])
            println("    [$cnt]  $c")
        end
    end
    return (; n_total, n_nan, causes)
end

# ============================================================
# Helpers to build θ grids
# ============================================================

"""Set named parameter in a copy of θ0 (untransformed scale)."""
function make_θu(model, overrides::Pair...)
    fe = model.fixed.fixed
    θ  = copy(NoLimits.get_θ0_untransformed(fe))
    for (k, v) in overrides
        setproperty!(θ, k, v)
    end
    return θ
end

"""Grid of 1-D scans over a single parameter, others held at nominal."""
function param_scan(model, param::Symbol, values)
    [make_θu(model, param => v) for v in values]
end

"""2-D grid over two parameters."""
function param_grid_2d(model, p1::Symbol, vals1, p2::Symbol, vals2)
    θs = ComponentArray[]
    for v1 in vals1, v2 in vals2
        push!(θs, make_θu(model, p1 => v1, p2 => v2))
    end
    return θs
end

# ============================================================
# ===================== SCENARIOS ============================
# ============================================================

println("Laplace NaN Gradient Diagnostic")
println("Julia version: $(VERSION)")
println()

# -----------------------------------------------------------
# Non-ODE model 1: simple additive RE, vary a and σ
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = repeat(1:10, inner=5),
        t  = repeat(0.0:1.0:4.0, outer=10),
        y  = 1.0 .+ randn(Xoshiro(1), 50) .* 0.3)

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # σ scan (log scale → untransformed σ from 0.01 to 20)
    σ_vals = exp.(range(log(0.01), log(20.0); length=40))
    grid_σ = param_scan(model, :σ, σ_vals)
    run_scenario("Non-ODE simple: vary σ (0.01..20)", dm, grid_σ)

    # a scan (mean far from data)
    a_vals = range(-10.0, 10.0; length=40)
    grid_a = param_scan(model, :a, a_vals)
    run_scenario("Non-ODE simple: vary a (−10..10)", dm, grid_a)

    # 2-D grid: a × σ
    a_vals2  = range(-5.0, 5.0; length=10)
    σ_vals2  = exp.(range(log(0.05), log(5.0); length=10))
    grid_2d  = param_grid_2d(model, :a, a_vals2, :σ, σ_vals2)
    run_scenario("Non-ODE simple: 2-D grid a × σ", dm, grid_2d)
end

# -----------------------------------------------------------
# Non-ODE model 2: RE variance as fixed effect, vary it
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            a    = RealNumber(1.0)
            σ    = RealNumber(0.5, scale=:log)
            σ_η  = RealNumber(1.0, scale=:log)
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
        ID = repeat(1:8, inner=6),
        t  = repeat(0.0:1.0:5.0, outer=8),
        y  = 1.0 .+ randn(Xoshiro(2), 48) .* 0.5)

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # Vary σ_η (RE std)
    ση_vals = exp.(range(log(0.001), log(10.0); length=40))
    grid = param_scan(model, :σ_η, ση_vals)
    run_scenario("Non-ODE RE-variance: vary σ_η (0.001..10)", dm, grid)

    # Vary σ (residual std) with fixed σ_η
    σ_vals = exp.(range(log(0.01), log(5.0); length=40))
    grid2 = param_scan(model, :σ, σ_vals)
    run_scenario("Non-ODE RE-variance: vary σ (0.01..5) with σ_η=1", dm, grid2)

    # 2-D grid: σ × σ_η
    grid_2d = param_grid_2d(model, :σ, exp.(range(log(0.05), log(3.0); length=8)),
                             :σ_η, exp.(range(log(0.01), log(5.0); length=8)))
    run_scenario("Non-ODE RE-variance: 2-D σ × σ_η", dm, grid_2d)
end

# -----------------------------------------------------------
# Non-ODE model 3: 2 RE groups (batching)
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.3, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η_id   = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end
        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    n = 12
    df = DataFrame(
        ID   = repeat(1:n, inner=4),
        SITE = repeat([:A, :B, :C], inner=16, outer=1)[1:n*4],
        t    = repeat(0.0:1.0:3.0, outer=n),
        y    = 0.5 .+ randn(Xoshiro(3), n*4) .* 0.3)

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    σ_vals = exp.(range(log(0.01), log(5.0); length=30))
    grid = param_scan(model, :σ, σ_vals)
    run_scenario("Non-ODE 2-RE-groups: vary σ", dm, grid)

    a_vals = range(-5.0, 5.0; length=30)
    grid2 = param_scan(model, :a, a_vals)
    run_scenario("Non-ODE 2-RE-groups: vary a (−5..5)", dm, grid2)
end

# -----------------------------------------------------------
# Non-ODE model 4: heavy shrinkage regime (few obs per individual)
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            a   = RealNumber(0.0)
            σ   = RealNumber(1.0, scale=:log)
            σ_η = RealNumber(2.0, scale=:log)
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

    # Only 2 obs per individual → posterior of η is wide, inner opt harder
    df = DataFrame(
        ID = repeat(1:20, inner=2),
        t  = repeat([0.0, 1.0], outer=20),
        y  = randn(Xoshiro(4), 40))

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    ση_vals = exp.(range(log(0.001), log(20.0); length=40))
    grid = param_scan(model, :σ_η, ση_vals)
    run_scenario("Non-ODE heavy-shrinkage (2 obs/ID): vary σ_η", dm, grid)
end

# -----------------------------------------------------------
# ODE model 1: 1-compartment PK, no events, vary k
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            k   = RealNumber(0.5, scale=:log)
            C0  = RealNumber(10.0, scale=:log)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.3, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @DifferentialEquation begin
            D(C) ~ -(k + η) * C
        end
        @initialDE begin
            C = C0
        end
        @formulas begin
            y ~ Normal(C(t), σ)
        end
    end

    t_obs = [0.5, 1.0, 2.0, 4.0, 8.0]
    n_ids = 8
    rng = Xoshiro(5)
    ids = repeat(1:n_ids, inner=length(t_obs))
    ts  = repeat(t_obs, outer=n_ids)
    ys  = 10.0 .* exp.(-0.5 .* ts) .+ randn(rng, length(ts)) .* 0.5

    df = DataFrame(ID=ids, t=ts, y=ys)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # Vary k (elimination rate)
    k_vals = exp.(range(log(0.01), log(5.0); length=30))
    grid_k = param_scan(model, :k, k_vals)
    run_scenario("ODE 1-compartment: vary k (0.01..5)", dm, grid_k)

    # Vary σ (residual)
    σ_vals = exp.(range(log(0.05), log(5.0); length=30))
    grid_σ = param_scan(model, :σ, σ_vals)
    run_scenario("ODE 1-compartment: vary σ (0.05..5)", dm, grid_σ)

    # Vary σ_η (RE std)
    ση_vals = exp.(range(log(0.001), log(2.0); length=30))
    grid_ση = param_scan(model, :σ_η, ση_vals)
    run_scenario("ODE 1-compartment: vary σ_η (0.001..2)", dm, grid_ση)

    # 2-D: k × σ
    grid_2d = param_grid_2d(model, :k, exp.(range(log(0.05), log(3.0); length=7)),
                             :σ, exp.(range(log(0.05), log(3.0); length=7)))
    run_scenario("ODE 1-compartment: 2-D k × σ", dm, grid_2d)
end

# -----------------------------------------------------------
# ODE model 2: 2-compartment PK (more complex dynamics)
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            k12  = RealNumber(0.3, scale=:log)
            k21  = RealNumber(0.1, scale=:log)
            ke   = RealNumber(0.4, scale=:log)
            C0   = RealNumber(10.0, scale=:log)
            σ    = RealNumber(0.5, scale=:log)
            σ_η  = RealNumber(0.2, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @DifferentialEquation begin
            D(C1) ~ -(k12 + ke + η) * C1 + k21 * C2
            D(C2) ~ k12 * C1 - k21 * C2
        end
        @initialDE begin
            C1 = C0
            C2 = 0.0
        end
        @formulas begin
            y ~ Normal(C1(t), σ)
        end
    end

    t_obs = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    n_ids = 6
    rng = Xoshiro(6)
    ids = repeat(1:n_ids, inner=length(t_obs))
    ts  = repeat(t_obs, outer=n_ids)
    ys  = 10.0 .* exp.(-0.4 .* ts) .+ randn(rng, length(ts)) .* 0.3

    df = DataFrame(ID=ids, t=ts, y=ys)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # Vary ke
    ke_vals = exp.(range(log(0.01), log(3.0); length=25))
    grid_ke = param_scan(model, :ke, ke_vals)
    run_scenario("ODE 2-compartment: vary ke (0.01..3)", dm, grid_ke)

    # Vary σ_η
    ση_vals = exp.(range(log(0.001), log(2.0); length=25))
    grid_ση = param_scan(model, :σ_η, ση_vals)
    run_scenario("ODE 2-compartment: vary σ_η (0.001..2)", dm, grid_ση)

    # Vary k12 (inter-compartmental)
    k12_vals = exp.(range(log(0.01), log(5.0); length=25))
    grid_k12 = param_scan(model, :k12, k12_vals)
    run_scenario("ODE 2-compartment: vary k12 (0.01..5)", dm, grid_k12)
end

# -----------------------------------------------------------
# ODE model 3: 1-compartment with events (dose bolus)
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            k   = RealNumber(0.5, scale=:log)
            V   = RealNumber(5.0, scale=:log)
            σ   = RealNumber(0.3, scale=:log)
            σ_η = RealNumber(0.2, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @DifferentialEquation begin
            D(C) ~ -(k + η) * C
        end
        @initialDE begin
            C = 0.0
        end
        @formulas begin
            y ~ Normal(C(t) / V, σ)
        end
    end

    n_ids = 8
    rng = Xoshiro(7)
    dose = 100.0

    # Build PKPD-style DataFrame with EVID
    rows = []
    for id in 1:n_ids
        push!(rows, (ID=id, t=0.0, EVID=1, AMT=dose, RATE=0.0, CMT=1, y=missing))
        for t_obs in [0.5, 1.0, 2.0, 4.0, 8.0]
            C_true = dose / 5.0 * exp(-0.5 * t_obs)
            push!(rows, (ID=id, t=t_obs, EVID=0, AMT=0.0, RATE=0.0, CMT=1,
                        y=C_true + randn(rng) * 0.05))
        end
    end
    df = DataFrame(rows)

    dm = DataModel(model, df; primary_id=:ID, time_col=:t, evid_col=:EVID,
                  amt_col=:AMT, rate_col=:RATE, cmt_col=:CMT)

    # Vary k
    k_vals = exp.(range(log(0.05), log(3.0); length=25))
    grid_k = param_scan(model, :k, k_vals)
    run_scenario("ODE bolus-events: vary k (0.05..3)", dm, grid_k)

    # Vary σ_η
    ση_vals = exp.(range(log(0.001), log(1.0); length=25))
    grid_ση = param_scan(model, :σ_η, ση_vals)
    run_scenario("ODE bolus-events: vary σ_η (0.001..1)", dm, grid_ση)

    # Vary V
    V_vals = exp.(range(log(0.5), log(20.0); length=25))
    grid_V = param_scan(model, :V, V_vals)
    run_scenario("ODE bolus-events: vary V (0.5..20)", dm, grid_V)
end

# -----------------------------------------------------------
# Extended: extreme parameter values (non-ODE)
# -----------------------------------------------------------
let
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

    df = DataFrame(
        ID = repeat(1:10, inner=5),
        t  = repeat(0.0:1.0:4.0, outer=10),
        y  = 1.0 .+ randn(Xoshiro(1), 50) .* 0.3)

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # Very wide σ range including near-zero and very large
    σ_vals = exp.(range(log(1e-5), log(100.0); length=50))
    run_scenario("Extreme σ (1e-5..100)", dm, param_scan(model, :σ, σ_vals))

    # Very wide σ_η range
    ση_vals = exp.(range(log(1e-5), log(50.0); length=50))
    run_scenario("Extreme σ_η (1e-5..50)", dm, param_scan(model, :σ_η, ση_vals))

    # Very extreme a values
    a_vals = [v for v in [-100.0, -50.0, -20.0, 20.0, 50.0, 100.0]]
    run_scenario("Extreme a values (±100)", dm, param_scan(model, :a, a_vals))
end

# -----------------------------------------------------------
# Extended: helpers with non-smooth functions (relu, abs, clamp)
# -----------------------------------------------------------
let
    model = @Model begin
        @helpers begin
            relu(u) = max(0.0, u)
            softclamp(u) = max(0.01, min(10.0, u))
        end
        @fixedEffects begin
            a   = RealNumber(0.5)
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
            y ~ Normal(relu(a + η), σ)
        end
    end

    df = DataFrame(
        ID = repeat(1:8, inner=5),
        t  = repeat(0.0:1.0:4.0, outer=8),
        y  = abs.(0.5 .+ randn(Xoshiro(8), 40) .* 0.3))

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # Vary a around the relu kink at 0
    a_vals = range(-3.0, 3.0; length=40)
    run_scenario("Non-smooth (relu): vary a around kink", dm, param_scan(model, :a, a_vals))

    # Vary σ
    σ_vals = exp.(range(log(0.05), log(5.0); length=30))
    run_scenario("Non-smooth (relu): vary σ", dm, param_scan(model, :σ, σ_vals))

    # Vary σ_η
    ση_vals = exp.(range(log(0.01), log(5.0); length=30))
    run_scenario("Non-smooth (relu): vary σ_η", dm, param_scan(model, :σ_η, ση_vals))
end

# -----------------------------------------------------------
# Extended: very few observations per individual (under-determined inner problem)
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            a   = RealNumber(0.0)
            σ   = RealNumber(1.0, scale=:log)
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

    # Only 1 observation per individual
    df1 = DataFrame(
        ID = 1:20,
        t  = zeros(20),
        y  = randn(Xoshiro(9), 20))
    dm1 = DataModel(model, df1; primary_id=:ID, time_col=:t)

    ση_vals = exp.(range(log(0.01), log(10.0); length=30))
    run_scenario("1 obs/ID (under-determined): vary σ_η", dm1, param_scan(model, :σ_η, ση_vals))

    σ_vals = exp.(range(log(0.05), log(5.0); length=30))
    run_scenario("1 obs/ID (under-determined): vary σ", dm1, param_scan(model, :σ, σ_vals))
end

# -----------------------------------------------------------
# Extended: ODE with very extreme dynamics
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            k   = RealNumber(0.5, scale=:log)
            C0  = RealNumber(10.0, scale=:log)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.3, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @DifferentialEquation begin
            D(C) ~ -(k + η) * C
        end
        @initialDE begin
            C = C0
        end
        @formulas begin
            y ~ Normal(C(t), σ)
        end
    end

    t_obs = [0.5, 1.0, 2.0, 4.0, 8.0]
    n_ids = 8
    ids = repeat(1:n_ids, inner=length(t_obs))
    ts  = repeat(t_obs, outer=n_ids)
    ys  = 10.0 .* exp.(-0.5 .* ts) .+ randn(Xoshiro(10), length(ts)) .* 0.5
    df = DataFrame(ID=ids, t=ts, y=ys)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # Very extreme k values (near-zero and very large)
    k_vals = exp.(range(log(1e-4), log(50.0); length=40))
    run_scenario("ODE extreme k (1e-4..50)", dm, param_scan(model, :k, k_vals))

    # Very extreme σ_η (near-zero RE to very large RE)
    ση_vals = exp.(range(log(1e-4), log(5.0); length=40))
    run_scenario("ODE extreme σ_η (1e-4..5)", dm, param_scan(model, :σ_η, ση_vals))

    # Very small σ (sharp likelihood) for ODE
    σ_vals = exp.(range(log(1e-3), log(0.5); length=20))
    run_scenario("ODE extreme σ small (1e-3..0.5)", dm, param_scan(model, :σ, σ_vals))

    # Very large C0 (mismatch with data)
    C0_vals = exp.(range(log(0.1), log(1000.0); length=30))
    run_scenario("ODE extreme C0 (0.1..1000)", dm, param_scan(model, :C0, C0_vals))
end

# -----------------------------------------------------------
# Extended: ODE with fastpath disabled (generic inner opt path)
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            k   = RealNumber(0.5, scale=:log)
            C0  = RealNumber(10.0)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.3, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @DifferentialEquation begin
            D(C) ~ -(k + η) * C
        end
        @initialDE begin
            C = C0
        end
        @formulas begin
            y ~ Normal(C(t), σ)
        end
    end

    t_obs = [0.5, 1.0, 2.0, 4.0]
    n_ids = 6
    ids = repeat(1:n_ids, inner=length(t_obs))
    ts  = repeat(t_obs, outer=n_ids)
    ys  = 10.0 .* exp.(-0.5 .* ts) .+ randn(Xoshiro(11), length(ts)) .* 0.3
    df = DataFrame(ID=ids, t=ts, y=ys)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    method_off = NoLimits.Laplace(fastpath_mode=:off)

    k_vals = exp.(range(log(0.05), log(5.0); length=30))
    run_scenario("ODE fastpath=:off: vary k", dm, param_scan(model, :k, k_vals);
                 method=method_off)

    ση_vals = exp.(range(log(0.001), log(2.0); length=30))
    run_scenario("ODE fastpath=:off: vary σ_η", dm, param_scan(model, :σ_η, ση_vals);
                 method=method_off)
end

# -----------------------------------------------------------
# Extended: non-ODE with log-scale logpdf (Lognormal observation)
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            μ   = RealNumber(0.0)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @formulas begin
            y ~ LogNormal(μ + η, σ)
        end
    end

    df = DataFrame(
        ID = repeat(1:10, inner=5),
        t  = repeat(0.0:1.0:4.0, outer=10),
        y  = exp.(randn(Xoshiro(12), 50) .* 0.5))

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    σ_vals = exp.(range(log(0.001), log(5.0); length=40))
    run_scenario("LogNormal obs: vary σ (0.001..5)", dm, param_scan(model, :σ, σ_vals))

    μ_vals = range(-5.0, 5.0; length=30)
    run_scenario("LogNormal obs: vary μ (−5..5)", dm, param_scan(model, :μ, μ_vals))
end

# -----------------------------------------------------------
# Extended: Multivariate Normal random effect
# -----------------------------------------------------------
let
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
            Ω = RealPSDMatrix([1.0 0.3; 0.3 1.0]; scale=:cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2], σ)
        end
    end

    df = DataFrame(
        ID = repeat(1:8, inner=5),
        t  = repeat(0.0:1.0:4.0, outer=8),
        y  = 1.0 .+ randn(Xoshiro(13), 40) .* 0.5)

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    σ_vals = exp.(range(log(0.05), log(5.0); length=25))
    run_scenario("MvNormal RE: vary σ (0.05..5)", dm, param_scan(model, :σ, σ_vals))

    # Vary Ω entries (keeping PSD by varying diagonal elements)
    θ_grid = [make_θu(model, :a => 1.0, :Ω => [d 0.0; 0.0 d]) for d in exp.(range(log(0.01), log(5.0); length=20))]
    run_scenario("MvNormal RE: vary Ω diagonal (0.01I..5I)", dm, θ_grid)
end

# -----------------------------------------------------------
# Test: stickbreak / ProbabilityVector parameters
# The stickbreak Jacobian has a potential 0/0 when
# the last component probability → 0 (extreme t values).
# -----------------------------------------------------------
let
    # We need a distribution that takes a ProbabilityVector as parameter.
    # Use Categorical with mixture components.
    model = @Model begin
        @fixedEffects begin
            μ1  = RealNumber(0.0)
            μ2  = RealNumber(2.0)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.5, scale=:log)
            # Mixture weight (2-simplex → ProbabilityVector)
            p   = ProbabilityVector([0.5, 0.5])
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @formulas begin
            # mixture: p[1]*Normal(μ1+η,σ) + p[2]*Normal(μ2+η,σ)
            y ~ MixtureModel([Normal(μ1 + η, σ), Normal(μ2 + η, σ)], p)
        end
    end

    rng = Xoshiro(20)
    n = 15
    df = DataFrame(
        ID = repeat(1:n, inner=4),
        t  = repeat(0.0:1.0:3.0, outer=n),
        y  = vcat([rand(rng, rand(rng) < 0.6 ? Normal(0.0, 0.5) : Normal(2.0, 0.5), 4) for _ in 1:n]...))

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    # Vary p toward extreme values (on untransformed probability scale)
    p_vals = range(0.001, 0.999; length=40)
    θ_grid = [make_θu(model, :p => [pv, 1.0 - pv]) for pv in p_vals]
    run_scenario("ProbabilityVector: sweep p[1] from 0.001 to 0.999", dm, θ_grid)
end

# -----------------------------------------------------------
# Test: stickbreak — check apply_inv_jacobian_T directly
# for extreme transformed (logit) values
# -----------------------------------------------------------
println("\n" * "-"^72)
println("Direct stickbreak Jacobian NaN test:")
println("-"^72)
let
    for t_extreme in [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 709.0]
        t = [t_extreme]            # k-1 = 1 element (k=2 probabilities)
        g = [1.0, 1.0]             # gradient on untransformed scale
        result = NoLimits._stickbreak_inv_jacobian_T(t, g)
        println("  t=$t_extreme  g_t=$(result)  has_nan=$(any(isnan, result))")
    end
    println()
    for t_extreme in [0.0, 5.0, 10.0, 20.0, 50.0]
        t = [t_extreme, t_extreme]   # k-1 = 2, k=3
        g = [1.0, 1.0, 0.0]
        result = NoLimits._stickbreak_inv_jacobian_T(t, g)
        println("  t=[$t_extreme,$t_extreme]  g=[1,1,0]  g_t=$(result)  has_nan=$(any(isnan, result))")
    end
    println()
    # Vary g[k] toward 0 with extreme t
    for gk in [1.0, 0.1, 0.01, 0.0]
        t = [20.0]
        g = [1.0, gk]
        result = NoLimits._stickbreak_inv_jacobian_T(t, g)
        println("  t=[20.0]  g=[1,$gk]  g_t=$(result)  has_nan=$(any(isnan, result))")
    end
end

# -----------------------------------------------------------
# Test: apply_inv_jacobian_T for log-scale parameters
# at extreme θt values (checking 0 * Inf scenario)
# -----------------------------------------------------------
println("\n" * "-"^72)
println("apply_inv_jacobian_T: log-scale at extreme θt values:")
println("-"^72)
let
    model = @Model begin
        @fixedEffects begin
            a  = RealNumber(1.0)
            σ  = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    fe = model.fixed.fixed
    inv_transform = NoLimits.get_inverse_transform(fe)

    # For each extreme θt[σ], compute a dummy grad_u and check if g_t is NaN
    for θt_σ in [0.0, 5.0, 10.0, 50.0, 100.0, 300.0, 500.0, 709.0, 710.0, 730.0]
        for g_u_σ in [1.0, 0.001, 1e-10, 1e-20, 0.0]
            θt_full = ComponentArray(a=0.0, σ=θt_σ)
            grad_u  = ComponentArray(a=1.0, σ=g_u_σ)
            grad_t = NoLimits.apply_inv_jacobian_T(inv_transform, θt_full, grad_u)
            if any(isnan, grad_t)
                println("  NaN! θt[σ]=$θt_σ  g_u[σ]=$g_u_σ  → g_t[σ]=$(grad_t.σ)")
            end
        end
    end
    println("  (no output above = no NaN found)")
end

# -----------------------------------------------------------
# Test: full obj_grad trace during actual optimization
# Run fits from very bad starting points and check convergence
# -----------------------------------------------------------
println("\n" * "-"^72)
println("Full optimization convergence test (bad initializations):")
println("-"^72)
let
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

    df = DataFrame(
        ID = repeat(1:10, inner=5),
        t  = repeat(0.0:1.0:4.0, outer=10),
        y  = 1.0 .+ randn(Xoshiro(1), 50) .* 0.3)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    bad_starts = [
        (a=10.0,  σ=10.0,  σ_η=0.001),
        (a=-10.0, σ=0.001, σ_η=10.0),
        (a=0.0,   σ=0.001, σ_η=0.001),
        (a=0.0,   σ=5.0,   σ_η=5.0),
        (a=50.0,  σ=50.0,  σ_η=50.0),
    ]

    method = NoLimits.Laplace(optim_kwargs=(; iterations=30, show_trace=false))
    for start in bad_starts
        θ0 = make_θu(model, :a => start.a, :σ => start.σ, :σ_η => start.σ_η)
        try
            res = fit_model(dm, method; theta_0_untransformed=θ0)
            conv = NoLimits.get_converged(res)
            obj  = NoLimits.get_objective(res)
            println("  start=(a=$(start.a),σ=$(start.σ),σ_η=$(start.σ_η)): converged=$conv  obj=$(round(obj,digits=2))")
        catch e
            println("  start=(a=$(start.a),σ=$(start.σ),σ_η=$(start.σ_η)): ERROR $(typeof(e)): $(sprint(showerror, e)[1:min(80,end)])")
        end
    end
end

# -----------------------------------------------------------
# Test: ODE with very bad initialization
# -----------------------------------------------------------
println()
let
    model = @Model begin
        @fixedEffects begin
            k   = RealNumber(0.5, scale=:log)
            C0  = RealNumber(10.0, scale=:log)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.3, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @DifferentialEquation begin
            D(C) ~ -(k + η) * C
        end
        @initialDE begin
            C = C0
        end
        @formulas begin
            y ~ Normal(C(t), σ)
        end
    end

    t_obs = [0.5, 1.0, 2.0, 4.0, 8.0]
    n_ids = 8
    ids = repeat(1:n_ids, inner=length(t_obs))
    ts  = repeat(t_obs, outer=n_ids)
    ys  = 10.0 .* exp.(-0.5 .* ts) .+ randn(Xoshiro(5), length(ts)) .* 0.5
    df = DataFrame(ID=ids, t=ts, y=ys)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    bad_starts = [
        (k=0.001, C0=0.001, σ=10.0,  σ_η=0.001),
        (k=50.0,  C0=1000.0, σ=0.001, σ_η=5.0),
        (k=0.1,   C0=5.0,   σ=0.1,   σ_η=0.001),
    ]

    println("ODE optimization with bad initializations:")
    method = NoLimits.Laplace(optim_kwargs=(; iterations=30, show_trace=false))
    for start in bad_starts
        θ0 = make_θu(model, :k => start.k, :C0 => start.C0, :σ => start.σ, :σ_η => start.σ_η)
        try
            res = fit_model(dm, method; theta_0_untransformed=θ0)
            conv = NoLimits.get_converged(res)
            obj  = NoLimits.get_objective(res)
            println("  start=(k=$(start.k),C0=$(start.C0),σ=$(start.σ),σ_η=$(start.σ_η)): converged=$conv  obj=$(round(obj,digits=2))")
        catch e
            println("  start=(k=$(start.k),C0=$(start.C0),σ=$(start.σ),σ_η=$(start.σ_η)): ERROR $(typeof(e)): $(sprint(showerror, e)[1:min(80,end)])")
        end
    end
end

println("\n" * "="^72)
println("DIAGNOSTIC COMPLETE")
println("="^72)
