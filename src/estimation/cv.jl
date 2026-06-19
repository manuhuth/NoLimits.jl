export cross_validate, fit_cv
export CVSpec, CVResult, CVFoldResult
export get_fold_results, get_obs_scores, get_spec

# ── Structs ───────────────────────────────────────────────────────────────────

"""
    CVSpec

Stores the fold split configuration for cross-validation. Row indices into the
original `DataModel`'s DataFrame are stored rather than full `DataModel` copies
to keep memory use low.

Created by [`cross_validate`](@ref).
"""
struct CVSpec
    dm::DataModel
    train_rows::Vector{Vector{Int}}
    test_rows::Vector{Vector{Int}}
    kind::Symbol
    n_folds::Int
end

"""
    CVFoldResult{R}

Results for a single cross-validation fold, including per-observation scores on
the held-out set and optionally the fitted `FitResult`.
"""
struct CVFoldResult{R}
    fold::Int
    test_loglikelihood::Float64
    obs_scores::DataFrame
    fit_result::R
end

"""
    CVResult

Aggregate cross-validation results. `obs_scores` combines all folds with a
`:fold` column. `mean_test_loglikelihood` and `std_test_loglikelihood` summarize
predictive performance across folds.
"""
struct CVResult
    spec::CVSpec
    fold_results::Vector{<:CVFoldResult}
    obs_scores::DataFrame
    mean_test_loglikelihood::Float64
    std_test_loglikelihood::Float64
end

# ── Accessors ─────────────────────────────────────────────────────────────────

"""
    get_fold_results(cv_res::CVResult) -> Vector{CVFoldResult}

Return the per-fold results stored in `cv_res`.
"""
get_fold_results(r::CVResult) = r.fold_results

"""
    get_obs_scores(cv_res::CVResult) -> DataFrame

Return the combined per-observation score table from all folds. Contains columns
`:fold`, `:individual`, `:time`, `:outcome`, `:obs`, `:loglikelihood`,
`:predicted_mean`, and optionally `:loss` when a loss function was supplied.
"""
get_obs_scores(r::CVResult) = r.obs_scores

"""
    get_spec(cv_res::CVResult) -> CVSpec

Return the [`CVSpec`](@ref) that describes the fold split used to produce `cv_res`.
"""
get_spec(r::CVResult) = r.spec

# ── Internal helpers ──────────────────────────────────────────────────────────

function _rebuild_dm(dm_ref::DataModel, rows::Vector{Int})
    df_sub = dm_ref.df[rows, :]
    cfg = dm_ref.config
    return DataModel(dm_ref.model, df_sub;
        primary_id = cfg.primary_id, time_col = cfg.time_col,
        evid_col = cfg.evid_col, amt_col = cfg.amt_col,
        rate_col = cfg.rate_col, cmt_col = cfg.cmt_col,
        serialization = cfg.serialization)
end

function _cv_empty_obs_df(has_loss::Bool)
    df = DataFrame(fold = Int[], individual = [], time = Float64[], outcome = Symbol[],
        obs = Float64[], loglikelihood = Float64[], predicted_mean = Float64[])
    has_loss && insertcols!(df, :loss => [])
    return df
end

function _cv_has_re_support(res::FitResult)
    r = res.result
    return r isa LaplaceResult || r isa GHQuadratureResult ||
           r isa MCEMResult || r isa SAEMResult
end

# Only RE-aware fitting methods accept the constants_re kwarg.
_cv_method_accepts_constants_re(::FittingMethod) = false
_cv_method_accepts_constants_re(::Laplace) = true
_cv_method_accepts_constants_re(::MCEM) = true
_cv_method_accepts_constants_re(::SAEM) = true
_cv_method_accepts_constants_re(::MCMC) = true
_cv_method_accepts_constants_re(::VI) = true

# ── cross_validate ────────────────────────────────────────────────────────────

"""
    cross_validate(dm::DataModel, n_folds::Int; kind=:id, rng=Random.default_rng())

Partition `dm` into `n_folds` train/test splits for cross-validation.

- `kind=:id` — whole individuals are assigned to folds; test individuals are
  entirely absent from training.
- `kind=:observation` — observations from each individual are distributed
  across folds (floor/ceiling split); training includes all event rows for
  individuals with any training observations.

Returns a [`CVSpec`](@ref) storing row indices only (not full `DataModel`s).
"""
function cross_validate(dm::DataModel, n_folds::Int;
        kind::Symbol = :id,
        rng::AbstractRNG = Random.default_rng())
    n_folds >= 2 || error("n_folds must be ≥ 2, got $n_folds")
    kind ∈ (:id, :observation) || error("kind must be :id or :observation, got $kind")
    n = length(dm.individuals)
    n >= n_folds || error("n_folds ($n_folds) exceeds number of individuals ($n)")

    train_rows = Vector{Vector{Int}}(undef, n_folds)
    test_rows = Vector{Vector{Int}}(undef, n_folds)

    if kind == :id
        perm = shuffle(rng, 1:n)
        fold_of = Vector{Int}(undef, n)
        for k in 1:n
            fold_of[perm[k]] = ((k - 1) % n_folds) + 1
        end
        for f in 1:n_folds
            test_inds = findall(==(f), fold_of)
            train_inds = findall(!=(f), fold_of)
            test_rows[f] = sort(vcat((dm.row_groups.rows[i] for i in test_inds)...))
            train_rows[f] = sort(vcat((dm.row_groups.rows[i] for i in train_inds)...))
        end
    else  # :observation
        test_sets = [Int[] for _ in 1:n_folds]
        train_sets = [Int[] for _ in 1:n_folds]
        for i in 1:n
            all_i = dm.row_groups.rows[i]
            obs_i = dm.row_groups.obs_rows[i]
            event_i = setdiff(all_i, obs_i)
            perm = shuffle(rng, 1:length(obs_i))
            obs_shuffled = obs_i[perm]
            fold_obs = [Int[] for _ in 1:n_folds]
            for (k, row) in enumerate(obs_shuffled)
                push!(fold_obs[((k - 1) % n_folds) + 1], row)
            end
            for f in 1:n_folds
                test_f = fold_obs[f]
                train_f = vcat((fold_obs[g] for g in 1:n_folds if g != f)...)
                if !isempty(test_f)
                    append!(test_sets[f], event_i)
                    append!(test_sets[f], test_f)
                end
                if !isempty(train_f)
                    append!(train_sets[f], event_i)
                    append!(train_sets[f], train_f)
                end
            end
        end
        for f in 1:n_folds
            test_rows[f] = sort(unique(test_sets[f]))
            train_rows[f] = sort(unique(train_sets[f]))
        end
    end

    return CVSpec(dm, train_rows, test_rows, kind, n_folds)
end

# ── Per-observation evaluation ────────────────────────────────────────────────

# Mirrors _loglikelihood_individual but collects per-obs rows instead of
# accumulating. HMM filter state (hmm_priors) is maintained sequentially.
# Returns empty DataFrame on ODE failure; records NaN for non-finite logpdf.
function _eval_individual_obs(dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache,
        loss::Union{Nothing, Function})
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    isempty(obs_rows) && return DataFrame()
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    η_ind isa NamedTuple && (η_ind = ComponentArray(η_ind))

    # ODE solving — shared scaffolding with _loglikelihood_individual (the preDE
    # NamedTuple is computed once and reused for the compile context and u0).
    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        sol_accessors = _ll_solve_de(dm, idx, θ, η_ind, cache, pre)
        sol_accessors === nothing && return DataFrame()
    end

    obs_cols = dm.config.obs_cols
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only = true)
    T_el = promote_type(eltype(θ), eltype(η_ind))
    T_hmm = T_el
    hmm_priors = nothing
    hmm_seen = nothing
    hmm_init = nothing
    time_vec = _get_col(dm.df, dm.config.time_col)[obs_rows]
    id_val = dm.df[dm.row_groups.rows[idx][1], dm.config.primary_id]

    rows_out = NamedTuple[]

    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, time_vec) : vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only = true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        t_i = Float64(time_vec[i])

        for (j, col) in pairs(obs_cols)
            y_raw = getfield(obs_series, col)[i]
            dist = getproperty(obs, col)

            is_hmm = dist isa ContinuousTimeDiscreteStatesHMM ||
                     dist isa DiscreteTimeDiscreteStatesHMM ||
                     dist isa MVContinuousTimeDiscreteStatesHMM ||
                     dist isa MVDiscreteTimeDiscreteStatesHMM ||
                     dist isa DiscreteTimeObservedStatesMarkovModel ||
                     dist isa ContinuousTimeObservedStatesMarkovModel ||
                     dist isa CoarsedObservedStatesMarkovModel

            if is_hmm
                # HMM filter state must be maintained across obs — mirrors
                # _loglikelihood_individual exactly, but records lp instead of accumulating.
                if hmm_seen === nothing
                    hmm_init = Vector{Vector{T_hmm}}(undef, length(obs_cols))
                    hmm_seen = falses(length(obs_cols))
                end
                hs = hmm_seen::BitVector
                hi = hmm_init::Vector{Vector{T_hmm}}
                if !hs[j]
                    init_probs = dist isa CoarsedObservedStatesMarkovModel ?
                                 dist.base_dist.initial_dist.p : dist.initial_dist.p
                    buf = Vector{T_hmm}(undef, length(init_probs))
                    copyto!(buf, init_probs)
                    hi[j] = buf
                    hs[j] = true
                end
                init_p = hi[j]
                dist_up = if dist isa ContinuousTimeDiscreteStatesHMM
                    ContinuousTimeDiscreteStatesHMM(
                        dist.transition_matrix, dist.emission_dists,
                        Distributions.Categorical(init_p; check_args = false), dist.Δt;
                        propagation_mode = dist.propagation_mode)
                elseif dist isa MVContinuousTimeDiscreteStatesHMM
                    MVContinuousTimeDiscreteStatesHMM(
                        dist.transition_matrix, dist.emission_dists,
                        Distributions.Categorical(init_p; check_args = false), dist.Δt;
                        propagation_mode = dist.propagation_mode)
                elseif dist isa MVDiscreteTimeDiscreteStatesHMM
                    MVDiscreteTimeDiscreteStatesHMM(
                        dist.transition_matrix, dist.emission_dists,
                        Distributions.Categorical(init_p; check_args = false))
                elseif dist isa DiscreteTimeObservedStatesMarkovModel
                    DiscreteTimeObservedStatesMarkovModel(dist.transition_matrix,
                        Distributions.Categorical(init_p; check_args = false), dist.state_labels)
                elseif dist isa ContinuousTimeObservedStatesMarkovModel
                    ContinuousTimeObservedStatesMarkovModel(dist.transition_matrix,
                        Distributions.Categorical(init_p; check_args = false), dist.Δt,
                        dist.state_labels; propagation_mode = dist.propagation_mode)
                elseif dist isa CoarsedObservedStatesMarkovModel &&
                       dist.base_dist isa DiscreteTimeObservedStatesMarkovModel
                    bd = dist.base_dist
                    coarsed(DiscreteTimeObservedStatesMarkovModel(bd.transition_matrix,
                        Distributions.Categorical(init_p; check_args = false), bd.state_labels))
                elseif dist isa CoarsedObservedStatesMarkovModel &&
                       dist.base_dist isa ContinuousTimeObservedStatesMarkovModel
                    bd = dist.base_dist
                    coarsed(ContinuousTimeObservedStatesMarkovModel(bd.transition_matrix,
                        Distributions.Categorical(init_p; check_args = false), bd.Δt,
                        bd.state_labels; propagation_mode = bd.propagation_mode))
                else
                    DiscreteTimeDiscreteStatesHMM(
                        dist.transition_matrix, dist.emission_dists,
                        Distributions.Categorical(init_p; check_args = false))
                end
                hmm_priors === nothing && (hmm_priors = Dict{Symbol, Any}())
                prior = get(hmm_priors::Dict{Symbol, Any}, col, nothing)
                dist_use = try
                    _hmm_with_prior(dist_up, prior)
                catch e
                    (e isa DomainError || e isa ArgumentError) ? dist_up : rethrow(e)
                end
                if y_raw === missing
                    (hmm_priors::Dict{Symbol, Any})[col] = try
                        probabilities_hidden_states(dist_use)
                    catch
                        prior
                    end
                    continue
                end
                lp = try
                    Float64(logpdf(dist_use, y_raw))
                catch e
                    (e isa DomainError || e isa ArgumentError) ? NaN : rethrow(e)
                end
                (hmm_priors::Dict{Symbol, Any})[col] = try
                    posterior_hidden_states(dist_use, y_raw)
                catch
                    prior
                end
                row = if loss !== nothing
                    lv = try
                        loss(dist_use, y_raw)
                    catch
                        NaN
                    end
                    (individual = id_val, time = t_i, outcome = col,
                        obs = Float64(y_raw), loglikelihood = lp, predicted_mean = NaN, loss = lv)
                else
                    (individual = id_val, time = t_i, outcome = col,
                        obs = Float64(y_raw), loglikelihood = lp, predicted_mean = NaN)
                end
                push!(rows_out, row)
            else
                y_raw === missing && continue
                v = _fast_logpdf(dist, y_raw)
                v === nothing && (v = logpdf(dist, y_raw))
                lp = isfinite(v) ? Float64(v) : NaN
                pm = try
                    Float64(mean(dist))
                catch
                    NaN
                end
                row = if loss !== nothing
                    lv = try
                        loss(dist, y_raw)
                    catch
                        NaN
                    end
                    (individual = id_val, time = t_i, outcome = col,
                        obs = Float64(y_raw), loglikelihood = lp, predicted_mean = pm, loss = lv)
                else
                    (individual = id_val, time = t_i, outcome = col,
                        obs = Float64(y_raw), loglikelihood = lp, predicted_mean = pm)
                end
                push!(rows_out, row)
            end
        end
    end

    return isempty(rows_out) ? DataFrame() : DataFrame(rows_out)
end

# ── Fold-level evaluation helpers ─────────────────────────────────────────────

function _cv_collect_obs(dm_test, θu, η_vec, ll_cache_test, loss)
    dfs = DataFrame[]
    empty_eta = ComponentArray()
    for j in 1:length(dm_test.individuals)
        η_j = η_vec === nothing ? empty_eta : η_vec[j]
        df = _eval_individual_obs(dm_test, j, θu, η_j, ll_cache_test, loss)
        isempty(df) || push!(dfs, df)
    end
    return isempty(dfs) ? DataFrame() : vcat(dfs...)
end

# Prior-mean RE value for an unseen test individual, shaped to match `ref` (a
# scalar or vector component of a reference η). Tries the distribution mean, then
# the median, then zero — mirroring `_re_start_value`, so non-zero-mean RE priors
# (Beta, Gumbel, LogNormal, …) are honored rather than collapsed to zero.
function _re_prior_mean_or_zero(dist, ref)
    v = try
        Distributions.mean(dist)
    catch
        nothing
    end
    v === nothing && (v = try
        Distributions.median(dist)
    catch
        nothing
    end)
    if ref isa AbstractVector
        v === nothing && return zeros(Float64, length(ref))
        return v isa AbstractVector ? collect(Float64.(v)) : fill(Float64(v), length(ref))
    else
        v === nothing && return 0.0
        return v isa AbstractVector ? Float64(first(v)) : Float64(v)
    end
end

# Build a prior-mean η ComponentArray for unseen test individual `j`, evaluating
# each RE distribution at that individual's constant covariates.
function _cv_prior_mean_eta(
        dm, j, θu, dists_builder, model_funs, helpers, re_names, ref_eta)
    const_cov = dm.individuals[j].const_cov
    dists = dists_builder(θu, const_cov, model_funs, helpers)
    nt = NamedTuple((re => _re_prior_mean_or_zero(getproperty(dists, re),
                         getproperty(ref_eta, re))
    for re in re_names))
    return ComponentArray(nt)
end

# Pooled path: every test individual — seen or unseen — gets the deterministic
# plug-in η evaluated from their own covariates with the strategies resolved by
# the training fit (replays demotions and fixed Monte-Carlo draws exactly).
function _cv_evaluate_pooled(dm_test, res_train, θu, ll_cache_test, loss)
    η_test = _compute_pooled_etas(dm_test, θu, res_train.result.strategies)
    return _cv_collect_obs(dm_test, θu, η_test, ll_cache_test, loss)
end

# EBE path: pre-build η for each test individual (seen → training EBE,
# unseen → RE prior mean).
function _cv_evaluate_ebe(dm_train, dm_test, res_train, θu, ll_cache_test, loss,
        constants_re)
    bstars, batch_infos, _, const_cache, _, _ = _resolve_bstars_for_re(
        dm_train, res_train, constants_re)
    η_train_vec = _eta_from_eb(dm_train, batch_infos, bstars, const_cache, θu)
    re_to_eta = Dict{Any, ComponentArray}(
        dm_train.individuals[i].re_groups => η_train_vec[i]
    for i in 1:length(dm_train.individuals))
    ref_eta = η_train_vec[1]
    dists_builder = get_create_random_effect_distribution(dm_test.model.random.random)
    model_funs_test = get_model_funs(dm_test.model)
    helpers_test = get_helper_funs(dm_test.model)
    re_names = get_re_names(dm_test.model.random.random)
    mean_eta_cache = Dict{Int, ComponentArray}()
    η_test = [haskey(re_to_eta, dm_test.individuals[j].re_groups) ?
              re_to_eta[dm_test.individuals[j].re_groups] :
              get!(
                  () -> _cv_prior_mean_eta(dm_test, j, θu, dists_builder,
                      model_funs_test, helpers_test,
                      re_names, ref_eta),
                  mean_eta_cache, j)
              for j in 1:length(dm_test.individuals)]
    return _cv_collect_obs(dm_test, θu, η_test, ll_cache_test, loss)
end

# MC path: marginalize over the conditional (seen) or prior (unseen) using S MC draws.
# Aggregates per-obs log-likelihoods via logsumexp and predicted means via arithmetic mean.
function _cv_evaluate_mc(dm_train, dm_test, res_train, θu, ll_cache_test, loss,
        seen_re_mode, unseen_re_mode, n_mc_samples, rng, re_names,
        constants_re)
    bstars, batch_infos, _, const_cache, ll_cache_train, _ = _resolve_bstars_for_re(
        dm_train, res_train, constants_re)
    η_train_vec = _eta_from_eb(dm_train, batch_infos, bstars, const_cache, θu)

    # Lookup: test individual re_groups → (batch_idx, train_ind_idx)
    re_to_train = Dict{Any, Tuple{Int, Int}}()
    for (bi, info) in enumerate(batch_infos)
        for i in info.inds
            re_to_train[dm_train.individuals[i].re_groups] = (bi, i)
        end
    end
    ref_eta = η_train_vec[1]

    # Conditional samples for seen individuals (Laplace or MCMC path)
    bstars_per_sample = nothing
    if seen_re_mode == :conditional
        if res_train.result isa LaplaceResult ||
           res_train.result isa GHQuadratureResult
            lcl = ll_cache_train isa Vector ? ll_cache_train[1] : ll_cache_train
            bstars_per_sample = _sample_laplace_bstars_raw(
                dm_train, batch_infos, bstars, θu, const_cache, lcl;
                n_samples = n_mc_samples, rng = rng)
        elseif res_train.result isa MCEMResult || res_train.result isa SAEMResult
            lcl = ll_cache_train isa Vector ? ll_cache_train[1] : ll_cache_train
            method_sampler, method_tkwargs = if res_train.result isa MCEMResult
                es = _mcmc_e_step(res_train.method.e_step)
                es === nothing ? (SaemixMH(), NamedTuple()) : (es.sampler, es.turing_kwargs)
            else
                (res_train.method.saem.sampler, res_train.method.saem.turing_kwargs)
            end
            bstars_per_sample = _sample_mcmc_bstars_raw(
                dm_train, batch_infos, bstars, θu, const_cache, lcl,
                get_re_names(dm_train.model.random.random), method_sampler, method_tkwargs;
                n_samples = n_mc_samples, n_adapt = 200, rng = rng, warm_start = true)
        else
            error("seen_re_mode=:conditional requires Laplace, MCEM, or SAEM. " *
                  "Got: $(typeof(res_train.result))")
        end
    end

    # RE distribution builder — used for unseen :montecarlo draws and :mean plug-in.
    dists_builder = get_create_random_effect_distribution(dm_test.model.random.random)
    model_funs_test = get_model_funs(dm_test.model)
    helpers_test = get_helper_funs(dm_test.model)

    n_test = length(dm_test.individuals)
    sample_rngs = _spawn_child_rngs(rng, n_mc_samples)

    # Prior-mean η for unseen individuals under :mean — identical across samples.
    mean_eta_cache = Dict{Int, ComponentArray}()
    # Unseen-individual RE distributions are θ-fixed — build once per individual
    # instead of once per (sample, individual). The rand stream is unchanged.
    unseen_dists_cache = Dict{Int, Any}()

    # Collect per-sample DataFrames: all_dfs[s] = vector of DataFrames, one per test individual
    all_dfs = [Vector{DataFrame}(undef, n_test) for _ in 1:n_mc_samples]
    for s in 1:n_mc_samples
        srng = sample_rngs[s]
        for j in 1:n_test
            ind_j = dm_test.individuals[j]
            key = ind_j.re_groups
            tinfo = get(re_to_train, key, nothing)
            is_seen = tinfo !== nothing

            η_j = if is_seen && seen_re_mode == :conditional
                bi, ti = tinfo
                b_s = bstars_per_sample[s][bi]
                ComponentArray(_build_eta_ind(
                    dm_train, ti, batch_infos[bi], b_s, const_cache, θu))
            elseif is_seen   # :ebe — same for every sample
                η_train_vec[tinfo[2]]
            elseif unseen_re_mode == :montecarlo
                dists = get!(
                    () -> dists_builder(θu, ind_j.const_cov, model_funs_test, helpers_test),
                    unseen_dists_cache, j)
                ComponentArray(NamedTuple((re => rand(srng, getproperty(dists, re))
                for re in re_names)))
            else             # :mean — RE prior mean, same for every sample
                get!(
                    () -> _cv_prior_mean_eta(dm_test, j, θu, dists_builder,
                        model_funs_test, helpers_test,
                        re_names, ref_eta),
                    mean_eta_cache, j)
            end

            all_dfs[s][j] = _eval_individual_obs(dm_test, j, θu, η_j, ll_cache_test, loss)
        end
    end

    # Aggregate across samples: logsumexp for loglikelihood, mean for predicted_mean/loss
    result_dfs = DataFrame[]
    for j in 1:n_test
        base_df = all_dfs[1][j]
        isempty(base_df) && continue
        n_rows = nrow(base_df)

        ll_acc = fill(-Inf, n_rows)
        mean_acc = fill(0.0, n_rows)
        mean_cnt = fill(0, n_rows)
        loss_acc = :loss ∈ names(base_df) ? fill(0.0, n_rows) : nothing
        loss_cnt = loss_acc !== nothing ? fill(0, n_rows) : nothing

        for s in 1:n_mc_samples
            df_s = all_dfs[s][j]
            nrow(df_s) == n_rows || continue   # ODE failure → contributes 0 probability
            for r in 1:n_rows
                lp = df_s[r, :loglikelihood]
                isnan(lp) || (ll_acc[r] = logaddexp(ll_acc[r], lp))
                pm = df_s[r, :predicted_mean]
                if !isnan(pm)
                    mean_acc[r] += pm
                    mean_cnt[r] += 1
                end
                if loss_acc !== nothing && :loss ∈ names(df_s)
                    lv = Float64(df_s[r, :loss])
                    if !isnan(lv)
                        loss_acc[r] += lv
                        loss_cnt[r] += 1
                    end
                end
            end
        end

        df_out = copy(base_df)
        df_out[!, :loglikelihood] = ll_acc .- log(n_mc_samples)
        df_out[!, :predicted_mean] = [mean_cnt[r] > 0 ? mean_acc[r] / mean_cnt[r] : NaN
                                      for r in 1:n_rows]
        if loss_acc !== nothing
            df_out[!, :loss] = [loss_cnt[r] > 0 ? loss_acc[r] / loss_cnt[r] : NaN
                                for r in 1:n_rows]
        end
        push!(result_dfs, df_out)
    end

    return isempty(result_dfs) ? DataFrame() : vcat(result_dfs...)
end

# ── fit_cv ────────────────────────────────────────────────────────────────────

"""
    fit_cv(cv_spec, method, args...;
           seen_re_mode=:ebe, unseen_re_mode=:mean,
           n_mc_samples=100, store_results=false, loss=nothing,
           fold_serialization=EnsembleSerial(), rng=Random.default_rng(),
           constants_re=NamedTuple(), ode_args=(), ode_kwargs=NamedTuple(),
           kwargs...)

Fit `method` on each training fold defined by `cv_spec` and evaluate predictive
performance on the held-out test set. All `kwargs` are forwarded to
[`fit_model`](@ref).

# Keyword Arguments
- `seen_re_mode`: prediction strategy for individuals present in the training
  set.  `:ebe` uses the empirical Bayes estimate (MAP of posterior); `:conditional`
  integrates over `n_mc_samples` draws from `p(b|y_train, θ̂)`.
- `unseen_re_mode`: prediction strategy for individuals absent from training.
  `:mean` plugs in the RE prior mean (zero for zero-mean priors); `:montecarlo`
  integrates over `n_mc_samples` draws from the RE prior `p(b|θ̂)`.
- `n_mc_samples`: number of MC draws when either mode is `:conditional` or
  `:montecarlo`.
- `store_results`: if `true`, each [`CVFoldResult`](@ref) stores the full
  `FitResult` from that fold.
- `loss`: optional `(dist, y) -> scalar` function. When provided, a `:loss`
  column is added to `obs_scores`.
- `fold_serialization`: controls fold-level parallelism. Use `EnsembleThreads()`
  to evaluate folds concurrently.
- `constants_re`: fix specific RE levels on the natural scale.

[`Pooled`](@ref)/[`PooledMap`](@ref) fits evaluate every test individual — seen or
unseen — at the deterministic plug-in η computed from that individual's covariates
with the strategies resolved by the training fit; `seen_re_mode`/`unseen_re_mode`
do not apply and must be left at their defaults.

Returns a [`CVResult`](@ref).
"""
function fit_cv(cv_spec::CVSpec, method::FittingMethod, args...;
        seen_re_mode::Symbol = :ebe,
        unseen_re_mode::Symbol = :mean,
        n_mc_samples::Int = 100,
        store_results::Bool = false,
        loss::Union{Nothing, Function} = nothing,
        fold_serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial(),
        rng::AbstractRNG = Random.default_rng(),
        constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        kwargs...)
    seen_re_mode ∈ (:ebe, :conditional) ||
        error("seen_re_mode must be :ebe or :conditional, got $seen_re_mode")
    unseen_re_mode ∈ (:mean, :montecarlo) ||
        error("unseen_re_mode must be :mean or :montecarlo, got $unseen_re_mode")
    if method isa Pooled || method isa PooledMap
        (seen_re_mode == :ebe && unseen_re_mode == :mean) ||
            error("Pooled/PooledMap cross-validation evaluates the deterministic plug-in " *
                  "η for every test individual; seen_re_mode/unseen_re_mode do not apply.")
    end

    n_folds = cv_spec.n_folds
    dm_ref = cv_spec.dm
    fold_rngs = _spawn_child_rngs(rng, n_folds)

    function _run_fold(f)
        dm_train = _rebuild_dm(dm_ref, cv_spec.train_rows[f])
        dm_test = _rebuild_dm(dm_ref, cv_spec.test_rows[f])

        base_kw = (store_data_model = false, ode_args = ode_args, ode_kwargs = ode_kwargs)
        fit_kw = _cv_method_accepts_constants_re(method) ?
                 merge(base_kw, (constants_re = constants_re,)) : base_kw
        res_train = fit_model(dm_train, method, args...; fit_kw..., kwargs...)
        θu = get_params(res_train; scale = :untransformed)

        ll_cache_test = build_ll_cache(
            dm_test; ode_args = ode_args, ode_kwargs = ode_kwargs,
            serialization = EnsembleSerial(), force_saveat = true)

        re_names = get_re_names(dm_train.model.random.random)
        has_re = !isempty(re_names)

        cr = _res_constants_re(res_train, constants_re)

        obs_df = if !has_re
            _cv_collect_obs(dm_test, θu, nothing, ll_cache_test, loss)
        elseif res_train.result isa PooledResult
            # Pooled/PooledMap: plug-in η from each test individual's covariates
            _cv_evaluate_pooled(dm_test, res_train, θu, ll_cache_test, loss)
        elseif !_cv_has_re_support(res_train)
            # Method without EBE support (e.g. MCMC) → evaluate at zero RE
            _cv_collect_obs(dm_test, θu, nothing, ll_cache_test, loss)
        elseif seen_re_mode == :ebe && unseen_re_mode == :mean
            _cv_evaluate_ebe(dm_train, dm_test, res_train, θu, ll_cache_test, loss, cr)
        else
            _cv_evaluate_mc(dm_train, dm_test, res_train, θu, ll_cache_test, loss,
                seen_re_mode, unseen_re_mode, n_mc_samples,
                fold_rngs[f], re_names, cr)
        end

        insertcols!(obs_df, 1, :fold => fill(f, nrow(obs_df)))

        ll_finite = filter(!isnan, obs_df[!, :loglikelihood])
        test_ll = isempty(ll_finite) ? NaN : sum(ll_finite)

        fit_res = store_results ? res_train : nothing
        return CVFoldResult{typeof(fit_res)}(f, test_ll, obs_df, fit_res)
    end

    fold_results = if fold_serialization isa SciMLBase.EnsembleSerial
        [_run_fold(f) for f in 1:n_folds]
    else
        buf = Vector{CVFoldResult}(undef, n_folds)
        Threads.@threads for f in 1:n_folds
            buf[f] = _run_fold(f)
        end
        buf
    end

    all_scores = vcat([fr.obs_scores for fr in fold_results]...)
    ll_vec = [fr.test_loglikelihood for fr in fold_results]
    ll_valid = filter(!isnan, ll_vec)

    return CVResult(cv_spec, fold_results, all_scores,
        isempty(ll_valid) ? NaN : mean(ll_valid),
        length(ll_valid) >= 2 ? std(ll_valid) : NaN)
end
