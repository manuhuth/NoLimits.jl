#!/usr/bin/env julia
# Focused reproducer for the NaN gradient in the user's model
# (joint antibody / infection model) when logk_pop = 40 (extreme k_i)
#
# Key structure:
#   preDifferentialEquation: A0_i = exp(logA0 + eta[1]); k_i = exp(logk + eta[2])
#   DifferentialEquation:    Ab(t) = A0_i * exp(-k_i * t)   (signal, not state)
#                            h(t)  = ...; D(H) ~ h(t)
#   formulas:                y ~ LogNormal(safe_log(Ab(t)), sigma)
#
# When logk is large, Ab(t) underflows to 0. safe_log adds 1e-12 to avoid
# log(0). The gradient of safe_log(0) w.r.t. b (via k_i) is 1/1e-12 * 0 = 0.
# But in the nested ForwardDiff used for the trace logdet gradient, the
# product "large_number * underflowed_zero" can be computed in a different order,
# producing NaN via Inf * 0 or Inf - Inf in higher-order dual arithmetic.

using Printf
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using OrdinaryDiffEq
using Random
using ForwardDiff

# ============================================================
# Simplified version of the user's model
# ============================================================

function make_simplified_model()
    @Model begin
        @helpers begin
            safe_log(x) = log(x + 1e-12)
        end

        @fixedEffects begin
            logA0_pop    = RealNumber(log(300.0))
            logk_pop     = RealNumber(2.0)        # will be varied in tests
            sigma_log_ab = RealNumber(0.45, scale=:log)
            Omega        = RealDiagonalMatrix([1.0, 1.0], scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            eta = RandomEffect(MvNormal([0.0, 0.0], Omega); column=:ID)
        end

        @preDifferentialEquation begin
            A0_i = exp(logA0_pop + eta[1])
            k_i  = exp(logk_pop  + eta[2])
        end

        @DifferentialEquation begin
            Ab(t) = A0_i * exp(-k_i * t)
            h(t)  = Ab(t)   # simplified: no incidence covariate
            D(H) ~ h(t)
        end

        @initialDE begin
            H = 0.0
        end

        @formulas begin
            y ~ LogNormal(safe_log(Ab(t)), sigma_log_ab)
        end
    end
end

# ============================================================
# Build synthetic data and DataModel
# ============================================================

function make_dm(model)
    t_obs = [0.1, 0.5, 1.0, 2.0, 4.0]
    n_ids = 6
    rng = Xoshiro(42)
    ids = repeat(1:n_ids, inner=length(t_obs))
    ts  = repeat(t_obs, outer=n_ids)
    # Generate y from the "true" model at logk_pop = 2.0
    A0_true = 300.0; k_true = exp(2.0)
    Ab_true(t) = A0_true * exp(-k_true * t)
    ys = [exp(randn(rng) * 0.45) * Ab_true(t) for t in ts]
    df = DataFrame(ID=ids, t=ts, y=ys)
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

# ============================================================
# Infrastructure helpers (same as main diagnostic)
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

function setup_sc(dm; method=NoLimits.Laplace())
    fe = dm.model.fixed.fixed
    θ0_t  = NoLimits.get_θ0_transformed(fe)
    Tθ    = eltype(θ0_t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NoLimits.build_ll_cache(dm; force_saveat=true)
    n_batches = length(batch_infos)
    inner_opts = NoLimits._resolve_inner_options(method.inner, dm)
    ms_opts    = NoLimits._resolve_multistart_options(method.multistart, inner_opts)
    return (; batch_infos, const_cache, ll_cache, inner_opts, ms_opts, n_batches, Tθ, method)
end

function eval_full_grad(dm, θu, sc)
    fe = dm.model.fixed.fixed
    transform    = NoLimits.get_transform(fe)
    inv_transform = NoLimits.get_inverse_transform(fe)
    θt = transform(θu)
    ebe_cache = make_fresh_ebe_cache(dm, sc.Tθ, sc.n_batches)

    obj, grad_u, bstars = NoLimits._laplace_objective_and_grad(
        dm, sc.batch_infos, θu, sc.const_cache, sc.ll_cache, ebe_cache;
        inner=sc.inner_opts,
        hessian=sc.method.hessian,
        cache_opts=sc.method.cache,
        multistart=sc.ms_opts,
        fastpath=nothing,
        rng=Xoshiro(0))

    if obj == Inf
        return (; obj, grad_u, grad_t=similar(grad_u) .* NaN,
                  grad_u_nan=false, grad_t_nan=true,
                  note="obj==Inf, early exit before Jacobian transform")
    end

    # Apply Jacobian transform (what obj_grad does after _laplace_objective_and_grad)
    grad_t = NoLimits.apply_inv_jacobian_T(inv_transform, θt, grad_u)

    return (; obj,
              grad_u,
              grad_t,
              grad_u_nan = any(isnan, grad_u),
              grad_t_nan = any(isnan, grad_t),
              note = "")
end

function make_θu(model, overrides::Pair...)
    fe  = model.fixed.fixed
    θ   = copy(NoLimits.get_θ0_untransformed(fe))
    for (k, v) in overrides
        setproperty!(θ, k, v)
    end
    return θ
end

# ============================================================
# TEST 1: scan logk_pop — identify the NaN threshold
# ============================================================

println("="^72)
println("TEST 1: scan logk_pop (identity scale) — where does NaN appear?")
println("="^72)

let
    model = make_simplified_model()
    model = set_solver_config(model; alg=Tsit5(), kwargs=(abstol=1e-4, reltol=1e-4))
    dm    = make_dm(model)
    sc    = setup_sc(dm)

    logk_vals = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 12.0,
                 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

    println("\n  logk_pop  │ Ab(1.0)         │ obj_u_nan │ grad_u_nan │ grad_t_nan │ obj")
    println("  " * "-"^78)
    for logk in logk_vals
        θu = make_θu(model, :logk_pop => logk)
        Ab_check = 300.0 * exp(-exp(logk) * 1.0)
        r = eval_full_grad(dm, θu, sc)
        @printf("  %-9.1f │ %-15.4g │ %-9s │ %-10s │ %-10s │ %s\n",
            logk, Ab_check,
            string(r.obj == Inf || isnan(r.obj)),
            string(r.grad_u_nan),
            string(r.grad_t_nan),
            isfinite(r.obj) ? string(round(r.obj; digits=1)) : string(r.obj))
        if !isempty(r.note)
            println("           note: $(r.note)")
        end
    end
end

# ============================================================
# TEST 2: diagnose which gradient component produces NaN at logk=40
# ============================================================

println("\n" * "="^72)
println("TEST 2: component-level diagnosis at logk_pop = 40 (first batch)")
println("="^72)

let
    model = make_simplified_model()
    model = set_solver_config(model; alg=Tsit5(), kwargs=(abstol=1e-4, reltol=1e-4))
    dm    = make_dm(model)
    method = NoLimits.Laplace()

    for logk in [5.0, 10.0, 20.0, 30.0, 40.0]
        println("\n--- logk_pop = $logk ---")

        sc = setup_sc(dm; method=method)
        θu = make_θu(model, :logk_pop => logk)
        ebe_cache = make_fresh_ebe_cache(dm, sc.Tθ, sc.n_batches)

        # Get b* by running inner optimization
        bstars = NoLimits._laplace_get_bstar!(
            ebe_cache, dm, sc.batch_infos, θu, sc.const_cache, sc.ll_cache;
            optimizer=sc.inner_opts.optimizer,
            optim_kwargs=sc.inner_opts.kwargs,
            adtype=sc.inner_opts.adtype,
            grad_tol=sc.inner_opts.grad_tol,
            theta_tol=method.cache.theta_tol,
            fastpath=nothing,
            multistart=sc.ms_opts,
            rng=Xoshiro(0))

        bi   = 1
        info = sc.batch_infos[bi]
        b    = bstars[bi]

        if isempty(b)
            println("  b* empty (inner opt failed)")
            continue
        end

        println("  b* = $(round.(b; sigdigits=3))")

        # logf at b*
        logf = NoLimits._laplace_logf_batch(dm, info, θu, b, sc.const_cache, sc.ll_cache)
        println("  logf at b* = $logf  (finite=$(isfinite(logf)))")

        # Hessian H
        H = NoLimits._laplace_hessian_b(dm, info, θu, b, sc.const_cache, sc.ll_cache, nothing, bi)
        println("  H has NaN=$(any(isnan,H))  Inf=$(any(isinf,H))")
        if !any(isnan,H) && !any(isinf,H)
            ev = eigvals(Symmetric(-H))
            println("  -H eigenvalues: $(round.(ev; sigdigits=4))  (should all be > 0)")
        end

        # Cholesky
        chol, jit = NoLimits._laplace_cholesky_negH(H;
            jitter=method.hessian.jitter, max_tries=method.hessian.max_tries,
            growth=method.hessian.growth)
        println("  Cholesky ok=$(chol !== nothing && chol.info == 0)  jitter=$(jit)")

        if chol !== nothing && chol.info == 0
            # --- Envelope term: grad_logf w.r.t. θ ---
            try
                gl = ForwardDiff.gradient(
                    θv -> NoLimits._laplace_logf_batch(dm, info, θv, b, sc.const_cache, sc.ll_cache),
                    collect(θu))
                has_nan = any(isnan, gl); has_inf = any(isinf, gl)
                println("  grad_logf (envelope): nan=$has_nan  inf=$has_inf  norm=$(round(norm(gl),sigdigits=3))")
            catch e
                println("  grad_logf: ERROR $(typeof(e))")
            end

            # --- Jacobian Gθ: ∂²logf/∂b∂θ ---
            try
                nb  = length(b)
                tmp = zeros(nb)
                Gθ  = ForwardDiff.jacobian(
                    (out, θv) -> begin
                        fb = bv -> NoLimits._laplace_logf_batch(dm, info, θv, bv, sc.const_cache, sc.ll_cache)
                        ForwardDiff.gradient!(out, fb, b)
                    end,
                    tmp, collect(θu))
                has_nan = any(isnan, Gθ); has_inf = any(isinf, Gθ)
                println("  Gθ (∂²logf/∂b∂θ):   nan=$has_nan  inf=$has_inf  norm=$(round(norm(Gθ),sigdigits=3))")
            catch e
                println("  Gθ: ERROR $(typeof(e))")
            end

            # --- Trace logdet gradient w.r.t. θ ---
            try
                n      = length(b)
                Ainv   = chol \ Matrix{Float64}(I, n, n)
                weights = Vector{Float64}(undef, NoLimits._ntri(n))
                NoLimits._vech_weights!(weights, Ainv)
                Jθ = ForwardDiff.jacobian(
                    θv -> begin
                        Hθ = NoLimits._laplace_hessian_b(dm, info, θv, b, sc.const_cache, sc.ll_cache, nothing, bi)
                        NoLimits._vech(Hθ)
                    end,
                    collect(θu))
                g_ld = -(Jθ' * weights)
                has_nan = any(isnan, g_ld); has_inf = any(isinf, g_ld)
                println("  grad_logdet_θ (trace): nan=$has_nan  inf=$has_inf  norm=$(round(norm(g_ld),sigdigits=3))")
                if has_nan || has_inf
                    # Find which columns of Jθ are bad
                    fe = dm.model.fixed.fixed
                    names = NoLimits.get_names(fe)
                    bad_params = [names[i] for i in 1:size(Jθ, 2) if any(!isfinite, Jθ[:, i])]
                    println("    bad columns: $bad_params")
                end
            catch e
                println("  grad_logdet_θ: ERROR $(typeof(e))")
            end

            # --- Trace logdet gradient w.r.t. b ---
            try
                n      = length(b)
                Ainv   = chol \ Matrix{Float64}(I, n, n)
                weights = Vector{Float64}(undef, NoLimits._ntri(n))
                NoLimits._vech_weights!(weights, Ainv)
                Jb = ForwardDiff.jacobian(
                    bv -> begin
                        Hb = NoLimits._laplace_hessian_b(dm, info, θu, bv, sc.const_cache, sc.ll_cache, nothing, bi)
                        NoLimits._vech(Hb)
                    end,
                    b)
                g_ldb = -(Jb' * weights)
                has_nan = any(isnan, g_ldb); has_inf = any(isinf, g_ldb)
                println("  grad_logdet_b (trace): nan=$has_nan  inf=$has_inf")
            catch e
                println("  grad_logdet_b: ERROR $(typeof(e))")
            end
        end
    end
end

# ============================================================
# TEST 3: isolate safe_log interaction with nested ForwardDiff
# ============================================================

println("\n" * "="^72)
println("TEST 3: isolate safe_log(Ab(t)) under nested ForwardDiff at large logk")
println("="^72)

let
    # Manual computation of what happens in the Hessian Jacobian
    # when Ab(t) underflows and safe_log is used

    safe_log(x) = log(x + 1e-12)

    for logk in [5.0, 10.0, 20.0, 30.0, 40.0]
        k   = exp(logk)
        A0  = 300.0
        t1  = 1.0
        Ab  = A0 * exp(-k * t1)
        σ   = 0.45
        y   = A0 * exp(-exp(2.0) * t1)  # observation from logk=2

        # logpdf of LogNormal(safe_log(Ab), σ) for observation y
        logpdf_val = logpdf(LogNormal(safe_log(Ab), σ), y)

        # Grad of logpdf w.r.t. logk (via chain rule manually):
        dAb_dlogk = A0 * exp(-k * t1) * (-k * t1)  # = Ab * (-k*t)
        dsafelog_dAb = 1.0 / (Ab + 1e-12)
        dsafelog_dlogk = dsafelog_dAb * dAb_dlogk

        @printf("  logk=%-4.1f: Ab(1)=%.2e  dAb/dlogk=%.2e  d(safelog)/dlogk=%.2e  logpdf=%.3f\n",
            logk, Ab, dAb_dlogk, dsafelog_dlogk, logpdf_val)
    end

    println()
    println("  ForwardDiff gradient of logpdf(LogNormal(safe_log(A0*exp(-exp(logk)*t)), σ), y) w.r.t. logk:")
    for logk in [5.0, 10.0, 20.0, 30.0, 40.0]
        σ   = 0.45
        t1  = 1.0
        A0  = 300.0
        y   = A0 * exp(-exp(2.0) * t1)

        f(lk) = logpdf(LogNormal(safe_log(A0 * exp(-exp(lk[1]) * t1)), σ), y)
        g = ForwardDiff.gradient(f, [logk])
        println("  logk=$logk: grad=$g  has_nan=$(any(isnan, g))")
    end

    println()
    println("  ForwardDiff HESSIAN of logpdf w.r.t. logk:")
    for logk in [5.0, 10.0, 20.0, 30.0, 40.0]
        σ   = 0.45
        t1  = 1.0
        A0  = 300.0
        y   = A0 * exp(-exp(2.0) * t1)

        f(lk) = logpdf(LogNormal(safe_log(A0 * exp(-exp(lk[1]) * t1)), σ), y)
        try
            H = ForwardDiff.hessian(f, [logk])
            println("  logk=$logk: hessian=$(H[1])  has_nan=$(isnan(H[1]))")
        catch e
            println("  logk=$logk: ERROR $(typeof(e))")
        end
    end

    println()
    println("  Jacobian of Hessian (∂H/∂logk) — the trace logdet term:")
    for logk in [5.0, 10.0, 20.0, 30.0, 40.0]
        σ   = 0.45
        t1  = 1.0
        A0  = 300.0
        y   = A0 * exp(-exp(2.0) * t1)

        # This is what the trace logdet gradient computes:
        # ForwardDiff.jacobian(θv -> ForwardDiff.hessian(b -> logf(θv, b), b_star), θ_vec)
        # For our 1-D simplification (b = eta2, affects logk via k_i = exp(logk + eta)):
        hess_of_logf(b, lk) = begin
            k_eff = exp(lk[1] + b[1])
            Ab_eff = A0 * exp(-k_eff * t1)
            logpdf(LogNormal(safe_log(Ab_eff), σ), y)
        end

        try
            jac_of_hess = ForwardDiff.jacobian(
                lk -> ForwardDiff.hessian(b -> hess_of_logf(b, lk), [0.0]),
                [logk])
            has_nan = any(isnan, jac_of_hess)
            println("  logk=$logk: ∂H/∂logk=$(round(jac_of_hess[1]; sigdigits=4))  has_nan=$has_nan")
        catch e
            println("  logk=$logk: ERROR $(typeof(e))")
        end
    end
end

# ============================================================
# TEST 4: apply_inv_jacobian_T for sigma_log_ab at extreme θt
# ============================================================

println("\n" * "="^72)
println("TEST 4: apply_inv_jacobian_T for sigma_log_ab at extreme transformed values")
println("="^72)

let
    model = make_simplified_model()
    fe = model.fixed.fixed
    inv_transform = NoLimits.get_inverse_transform(fe)

    println("  θt[sigma_log_ab] | σ = exp(θt) | g_u | g_t | NaN?")
    println("  " * "-"^60)
    for θt_σ in [-0.799, 10.0, 50.0, 100.0, 300.0, 600.0, 650.0, 709.0, 710.0, 730.0]
        for g_u_σ in [1.0, 0.01, 0.0]
            θt_full = ComponentArray(logA0_pop=0.0, logk_pop=0.0, sigma_log_ab=θt_σ, Omega=[0.0, 0.0])
            grad_u  = ComponentArray(logA0_pop=0.0, logk_pop=0.0, sigma_log_ab=g_u_σ, Omega=[0.0, 0.0])
            grad_t  = NoLimits.apply_inv_jacobian_T(inv_transform, θt_full, grad_u)
            has_nan = any(isnan, grad_t)
            if has_nan || θt_σ ∈ [-0.799, 709.0, 710.0]
                @printf("  θt_σ=%-7.3f  exp(θt)=%-12.3e  g_u=%-6.4f  g_t[σ]=%-12.4e  NaN=%s\n",
                    θt_σ, exp(θt_σ), g_u_σ, grad_t.sigma_log_ab, string(has_nan))
            end
        end
    end
end

println("\n" * "="^72)
println("FOCUSED DIAGNOSTIC COMPLETE")
println("="^72)
