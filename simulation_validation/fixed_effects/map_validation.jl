include("_setup.jl")

function _simulate_fit(model, n_ids, n_obs; rng::AbstractRNG)
    df = _make_panel(n_ids, n_obs; rng=rng)
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=rng)
    dm_sim = DataModel(model, sim; primary_id=:ID, time_col=:t)
    fe = model.fixed.fixed
    θ_init = _jitter_params(fe, rng)
    res = fit_model(dm_sim, MAP(); theta_0_untransformed=θ_init, rng=rng)
    return res, θ_init
end

function run_map_validation(; n_reps=400, n_ids=100, n_obs=5, rel_tol=0.1, abs_tol=0.1, seed=1)
    models = _build_models()
    stats = Vector{Any}(undef, length(models))

    for (mi, model) in enumerate(models)
        θ_true = get_θ0_untransformed(model.fixed.fixed)
        mean_ok = Vector{Float64}(undef, n_reps)
        mean_abs = Vector{Float64}(undef, n_reps)
        mean_rel = Vector{Float64}(undef, n_reps)
        start_ok = Vector{Float64}(undef, n_reps)
        start_abs = Vector{Float64}(undef, n_reps)
        start_rel = Vector{Float64}(undef, n_reps)
        rngs = _rep_rngs(seed, n_reps; offset=mi * 1_000_000)
        use_progress = Threads.nthreads() == 1
        p = use_progress ? Progress(n_reps, desc="MAP validation model $(mi)") : nothing
        if use_progress
            for r in 1:n_reps
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
                next!(p)
            end
        else
            Threads.@threads for r in 1:n_reps
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
            end
        end
        stats[mi] = (mean_ok=mean(mean_ok), mean_abs=mean(mean_abs), mean_rel=mean(mean_rel))
        start_stats = (mean_ok=mean(start_ok), mean_abs=mean(start_abs), mean_rel=mean(start_rel))
        @info "MAP validation start summary model $(mi)" n_reps n_ids n_obs rel_tol abs_tol start_stats
    end

    @info "MAP validation summary" n_reps n_ids n_obs rel_tol abs_tol stats
    return stats
end

stats = run_map_validation(; n_reps=500, n_ids=100, n_obs=5, rel_tol=0.1, abs_tol=0.1, seed=1);
