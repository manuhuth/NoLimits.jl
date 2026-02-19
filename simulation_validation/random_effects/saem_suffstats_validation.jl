include("_setup.jl")

function _simulate_fit(model, n_ids, n_obs; rng::AbstractRNG)
    df = _make_panel(n_ids, n_obs; rng=rng)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=rng)
    dm_sim = DataModel(model, sim; primary_id=:ID, time_col=:t)
    fe = model.fixed.fixed
    θ_init = _jitter_params(fe, rng, scale=2.0)
    res = fit_model(dm_sim,
                    SAEM(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false, verbose=false),
                        verbose=true,
                         max_store=20,
                         maxiters=50,
                         t0=10, kappa=0.7,
                         consecutive_params=10,
                         builtin_stats=:gaussian_re,
                         resid_var_param=:σ,
                         re_cov_params=(; η=:τ)),
                    theta_0_untransformed=θ_init,
                    rng=rng)
    return res, θ_init
end

function run_saem_suffstats_validation(; n_reps=400, n_ids=100, n_obs=5, rel_tol=0.1, abs_tol=0.1, seed=1)
    model = _build_model()
    θ_true = get_θ0_untransformed(model.fixed.fixed)

    mean_ok = Vector{Float64}(undef, n_reps)
    mean_abs = Vector{Float64}(undef, n_reps)
    mean_rel = Vector{Float64}(undef, n_reps)
    start_ok = Vector{Float64}(undef, n_reps)
    start_abs = Vector{Float64}(undef, n_reps)
    start_rel = Vector{Float64}(undef, n_reps)
    fill!(mean_ok, NaN)
    fill!(mean_abs, NaN)
    fill!(mean_rel, NaN)
    fill!(start_ok, NaN)
    fill!(start_abs, NaN)
    fill!(start_rel, NaN)
    rngs = _rep_rngs(seed, n_reps)
    use_progress = Threads.nthreads() == 1
    p = use_progress ? Progress(n_reps, desc="SAEM suffstats validation") : nothing
    failed = Threads.Atomic{Int}(0)
    if use_progress
        for r in 1:n_reps
            try
                res, θ_init = _simulate_fit(model, n_ids, n_obs; rng=rngs[r])
                θ_hat = NoLimits.get_params(res; scale=:untransformed)
                s = _error_stats(θ_hat, θ_true; rel_tol=rel_tol, abs_tol=abs_tol)
                s0 = _error_stats(θ_init, θ_true; rel_tol=rel_tol, abs_tol=abs_tol)
                mean_ok[r] = s.mean_ok
                mean_abs[r] = s.mean_abs
                mean_rel[r] = s.mean_rel
                start_ok[r] = s0.mean_ok
                start_abs[r] = s0.mean_abs
                start_rel[r] = s0.mean_rel
            catch
                failed[] += 1
            end
            next!(p)
        end
    else
        Threads.@threads for r in 1:n_reps
            try
                res, θ_init = _simulate_fit(model, n_ids, n_obs; rng=rngs[r])
                θ_hat = get_params(res; scale=:untransformed)
                s = _error_stats(θ_hat, θ_true; rel_tol=rel_tol, abs_tol=abs_tol)
                s0 = _error_stats(θ_init, θ_true; rel_tol=rel_tol, abs_tol=abs_tol)
                mean_ok[r] = s.mean_ok
                mean_abs[r] = s.mean_abs
                mean_rel[r] = s.mean_rel
                start_ok[r] = s0.mean_ok
                start_abs[r] = s0.mean_abs
                start_rel[r] = s0.mean_rel
            catch
                failed[] += 1
            end
        end
    end

    valid = isfinite.(mean_ok)
    n_valid = count(valid)
    stats = n_valid == 0 ? (mean_ok=NaN, mean_abs=NaN, mean_rel=NaN) :
            (mean_ok=mean(mean_ok[valid]), mean_abs=mean(mean_abs[valid]), mean_rel=mean(mean_rel[valid]))
    start_stats = n_valid == 0 ? (mean_ok=NaN, mean_abs=NaN, mean_rel=NaN) :
                 (mean_ok=mean(start_ok[valid]), mean_abs=mean(start_abs[valid]), mean_rel=mean(start_rel[valid]))
    @info "SAEM suffstats validation summary" n_reps n_ids n_obs rel_tol abs_tol stats start_stats n_valid n_failed=failed[]
    return stats
end

stats = run_saem_suffstats_validation(; n_reps=20, n_ids=100, n_obs=5, rel_tol=0.1, abs_tol=0.1, seed=1);
