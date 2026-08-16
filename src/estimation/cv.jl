export cross_validate, fit_cv
export CVSpec, CVResult, CVFoldResult
export get_fold_results, get_obs_scores, get_spec
export get_mean_obs_loglikelihood, get_n_scored_obs

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
`:fold` column. `mean_test_loglikelihood` and `std_test_loglikelihood` are the
mean and standard deviation of the per-fold *total* test log-likelihoods; they
weight every fold equally, which only compares folds fairly when the folds hold
equally many scored observations. `mean_obs_loglikelihood` is the aggregate test
log-likelihood divided by the total number of scored observations
(`n_scored_obs`) and is the fold-size-independent summary.
"""
struct CVResult
    spec::CVSpec
    fold_results::Vector{<:CVFoldResult}
    obs_scores::DataFrame
    mean_test_loglikelihood::Float64
    std_test_loglikelihood::Float64
    mean_obs_loglikelihood::Float64
    n_scored_obs::Int
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

"""
    get_mean_obs_loglikelihood(cv_res::CVResult) -> Float64

Return the aggregate test log-likelihood divided by the number of scored held-out
observations. Unlike `mean_test_loglikelihood` (a plain mean of per-fold totals) this
does not depend on how the observations are distributed over folds.
"""
get_mean_obs_loglikelihood(r::CVResult) = r.mean_obs_loglikelihood

"""
    get_n_scored_obs(cv_res::CVResult) -> Int

Return the number of held-out observations with a finite log-likelihood across all folds.
"""
get_n_scored_obs(r::CVResult) = r.n_scored_obs

# ── Internal helpers ──────────────────────────────────────────────────────────

function _rebuild_dm(dm_ref::DataModel, rows::Vector{Int})
    df_sub = get_df(dm_ref)[rows, :]
    cfg = dm_ref.config
    return DataModel(
        get_model(dm_ref), df_sub;
        primary_id = cfg.primary_id, time_col = cfg.time_col,
        evid_col = cfg.evid_col, amt_col = cfg.amt_col,
        rate_col = cfg.rate_col, cmt_col = cfg.cmt_col, t0 = cfg.t0,
        serialization = cfg.serialization
    )
end

function _cv_has_re_support(res::FitResult)
    r = get_result(res)
    return r isa FrequentistREResult || r isa GHQuadratureResult ||
        r isa MCEMResult || r isa SAEMResult
end

# Only RE-aware fitting methods accept the constants_re kwarg.
_cv_method_accepts_constants_re(::FittingMethod) = false
_cv_method_accepts_constants_re(::Laplace) = true
_cv_method_accepts_constants_re(::FOCEI) = true
_cv_method_accepts_constants_re(::GHQuadrature) = true
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
function cross_validate(
        dm::DataModel, n_folds::Int;
        kind::Symbol = :id,
        rng::AbstractRNG = Random.default_rng()
    )
    n_folds >= 2 || error("n_folds must be ≥ 2, got $n_folds")
    kind ∈ (:id, :observation) || error("kind must be :id or :observation, got $kind")
    n = length(get_individuals(dm))
    if kind == :id
        n >= n_folds || error("n_folds ($n_folds) exceeds number of individuals ($n)")
    else
        # Observation folds split each individual's observations round-robin, so the
        # binding constraint is observations per individual, not the individual count
        # (#229): every fold needs at least one observation to score.
        max_obs = maximum(length.(get_obs_rows(get_row_groups(dm))); init = 0)
        max_obs >= n_folds ||
            error("n_folds ($n_folds) exceeds the largest number of observations for one individual ($max_obs); some folds would be empty.")
    end

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
            test_rows[f] = sort(
                vcat(
                    (
                        get_rows(get_row_groups(dm))[i]
                            for i in test_inds
                    )...
                )
            )
            train_rows[f] = sort(
                vcat(
                    (
                        get_rows(get_row_groups(dm))[i]
                            for i in train_inds
                    )...
                )
            )
        end
    else  # :observation
        test_sets = [Int[] for _ in 1:n_folds]
        train_sets = [Int[] for _ in 1:n_folds]
        for i in 1:n
            all_i = get_rows(get_row_groups(dm))[i]
            obs_i = get_obs_rows(get_row_groups(dm))[i]
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
function _eval_individual_obs(
        dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache,
        loss::Union{Nothing, Function};
        score_rows::Union{Nothing, BitVector} = nothing
    )
    model = get_model(dm)
    ind = get_individuals(dm)[idx]
    obs_rows = get_obs_rows(get_row_groups(dm))[idx]
    isempty(obs_rows) && return DataFrame()
    const_cov = get_const_cov(ind)
    obs_series = get_obs(get_series(ind))
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    η_ind isa NamedTuple && (η_ind = ComponentArray(η_ind))

    # ODE solving — shared scaffolding with _loglikelihood_individual (the preDE
    # NamedTuple is computed once and reused for the compile context and u0).
    sol_accessors = nothing
    if get_de(model) !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        sol_accessors = _ll_solve_de(dm, idx, θ, η_ind, cache, pre)
        sol_accessors === nothing && return DataFrame()
    end

    obs_cols = get_obs_cols(dm)
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only = true)
    T_el = promote_type(eltype(θ), eltype(η_ind))
    T_hmm = T_el
    hmm_priors = nothing
    hmm_seen = nothing
    hmm_init = nothing
    time_vec = _get_col(get_df(dm), get_time_col(dm))[obs_rows]
    id_val = get_df(dm)[get_rows(get_row_groups(dm))[idx][1], get_primary_id(dm)]

    rows_out = NamedTuple[]

    for i in eachindex(obs_rows)
        # Rows outside the scored set are still evaluated (the HMM filter has to run
        # through them) but are not reported.
        keep_row = score_rows === nothing || score_rows[obs_rows[i]]
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, time_vec) : vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only = true)
        obs = sol_accessors === nothing ?
            calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
            calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        t_i = Float64(time_vec[i])

        for (j, col) in pairs(obs_cols)
            y_raw = getfield(obs_series, col)[i]
            dist = getproperty(obs, col)

            is_hmm = _is_hmm_dist(dist)

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
                dist_up = _hmm_pin_initial_probs(dist, init_p)
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
                    (
                        individual = id_val, time = t_i, outcome = col,
                        obs = _cv_obs_float(y_raw), loglikelihood = lp, predicted_mean = NaN, loss = lv,
                    )
                else
                    (
                        individual = id_val, time = t_i, outcome = col,
                        obs = _cv_obs_float(y_raw), loglikelihood = lp, predicted_mean = NaN,
                    )
                end
                keep_row && push!(rows_out, row)
            else
                _obs_is_missing(y_raw) && continue
                # `-Inf` is a valid score (the model rules the held-out value out); only
                # a failed evaluation may become `NaN`, which aggregation drops (#249).
                yv = _narrow_obs_eltype(y_raw)
                lp = try
                    v = _fast_logpdf(dist, yv)
                    v === nothing && (v = logpdf(dist, yv))
                    Float64(v)
                catch e
                    (e isa DomainError || e isa ArgumentError) ? NaN : rethrow(e)
                end
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
                    (
                        individual = id_val, time = t_i, outcome = col,
                        obs = _cv_obs_float(y_raw), loglikelihood = lp, predicted_mean = pm, loss = lv,
                    )
                else
                    (
                        individual = id_val, time = t_i, outcome = col,
                        obs = _cv_obs_float(y_raw), loglikelihood = lp, predicted_mean = pm,
                    )
                end
                keep_row && push!(rows_out, row)
            end
        end
    end

    return isempty(rows_out) ? DataFrame() : DataFrame(rows_out)
end

# ── Fold-level evaluation helpers ─────────────────────────────────────────────

function _cv_collect_obs(dm_test, θu, η_vec, ll_cache_test, loss, score_rows = nothing)
    dfs = DataFrame[]
    empty_eta = ComponentArray()
    for j in 1:length(get_individuals(dm_test))
        η_j = η_vec === nothing ? empty_eta : η_vec[j]
        df = _eval_individual_obs(
            dm_test, j, θu, η_j, ll_cache_test, loss; score_rows = score_rows
        )
        isempty(df) || push!(dfs, df)
    end
    return isempty(dfs) ? DataFrame() : vcat(dfs...)
end

# Prior-mean RE value for an unseen test individual, shaped to match `ref` (a
# scalar or vector component of a reference η). Tries the distribution mean, then
# the median, then zero — mirroring `_re_start_value`, so non-zero-mean RE priors
# (Beta, Gumbel, LogNormal, …) are honored rather than collapsed to zero.
# Vector-valued observations (multivariate outcomes) have no scalar `obs` column
# entry; the loglikelihood/loss columns still carry their information.
_cv_obs_float(y) = y isa Number ? Float64(y) : NaN

function _re_prior_mean_or_zero(dist, ref)
    v = try
        _re_mean(dist)
    catch
        nothing
    end
    v === nothing && (
        v = try
            Distributions.median(dist)
        catch
            nothing
        end
    )
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
        dm, j, θu, dists_builder, model_funs, helpers, re_names, ref_eta
    )
    const_cov = get_const_cov(get_individuals(dm)[j])
    dists = dists_builder(θu, const_cov, model_funs, helpers)
    nt = NamedTuple(
        (
            re => _re_prior_mean_or_zero(
                    getproperty(dists, re),
                    getproperty(ref_eta, re)
                )
                for re in re_names
        )
    )
    return ComponentArray(nt)
end

# Pooled path: every test individual — seen or unseen — gets the deterministic
# plug-in η evaluated from their own covariates with the strategies resolved by
# the training fit (replays demotions and fixed Monte-Carlo draws exactly).
function _cv_evaluate_pooled(dm_test, res_train, θu, ll_cache_test, loss, score_rows)
    η_test = _compute_pooled_etas(dm_test, θu, get_result(res_train).strategies)
    return _cv_collect_obs(dm_test, θu, η_test, ll_cache_test, loss, score_rows)
end

# EBE path: pre-build η for each test individual (seen → training EBE,
# unseen → RE prior mean).
function _cv_evaluate_ebe(
        dm_train, dm_test, res_train, θu, ll_cache_test, loss,
        constants_re, score_rows
    )
    bstars, batch_infos, _, const_cache, _, _ = _resolve_bstars_for_re(
        dm_train, res_train, constants_re
    )
    η_train_vec = _eta_from_eb(dm_train, batch_infos, bstars, const_cache, θu)
    ref_eta = η_train_vec[1]
    dists_builder = create_random_effect_distribution(get_random(get_model(dm_test)))
    model_funs_test = get_model_funs(get_model(dm_test))
    helpers_test = get_helper_funs(get_model(dm_test))
    re_names = get_re_names(get_random(get_model(dm_test)))
    # Levels are resolved one random effect at a time: in a crossed design a test
    # individual can share a trained level for one effect (say SITE) while its level for
    # another (say ID) is new, and only the genuinely new one needs the prior (#226).
    level_eta = Dict{Tuple{Symbol, Any}, Any}()
    for i in 1:length(get_individuals(dm_train))
        g = get_re_groups(get_individuals(dm_train)[i])
        for re in re_names
            level_eta[(re, getproperty(g, re))] = getproperty(η_train_vec[i], re)
        end
    end
    mean_eta_cache = Dict{Int, ComponentArray}()
    η_test = map(1:length(get_individuals(dm_test))) do j
        g = get_re_groups(get_individuals(dm_test)[j])
        prior_eta() = get!(
            () -> _cv_prior_mean_eta(
                dm_test, j, θu, dists_builder,
                model_funs_test, helpers_test, re_names, ref_eta
            ),
            mean_eta_cache, j
        )
        ComponentArray(
            NamedTuple(
                (
                    re => get(
                            () -> getproperty(prior_eta(), re),
                            level_eta, (re, getproperty(g, re))
                        ) for re in re_names
                )
            )
        )
    end
    return _cv_collect_obs(dm_test, θu, η_test, ll_cache_test, loss, score_rows)
end

# MC path: marginalize over the conditional (seen) or prior (unseen) using S MC draws.
# Aggregates per-obs log-likelihoods via logsumexp and predicted means via arithmetic mean.
function _cv_evaluate_mc(
        dm_train, dm_test, res_train, θu, ll_cache_test, loss,
        seen_re_mode, unseen_re_mode, n_mc_samples, rng, re_names,
        constants_re, score_rows
    )
    bstars, batch_infos, _, const_cache, ll_cache_train, _ = _resolve_bstars_for_re(
        dm_train, res_train, constants_re
    )
    η_train_vec = _eta_from_eb(dm_train, batch_infos, bstars, const_cache, θu)

    # Lookup per random effect and level, not per full level tuple: a crossed test
    # individual can share one trained level while another of its levels is new, and only
    # the new one needs the prior (#229). For a non-crossed design this is the same
    # mapping the whole-tuple lookup produced.
    level_train = Dict{Tuple{Symbol, Any}, Tuple{Int, Int}}()
    for (bi, info) in enumerate(batch_infos)
        for i in get_inds(info)
            g = get_re_groups(get_individuals(dm_train)[i])
            for re in re_names
                level_train[(re, getproperty(g, re))] = (bi, i)
            end
        end
    end
    ref_eta = η_train_vec[1]

    # Conditional samples for seen individuals (Laplace or MCMC path)
    bstars_per_sample = seen_re_mode == :conditional ?
        _sample_conditional_bstars(
            dm_train, batch_infos, bstars, θu,
            const_cache, ll_cache_train, res_train, n_mc_samples, rng
        ) : nothing

    # RE distribution builder — used for unseen :montecarlo draws and :mean plug-in.
    dists_builder = create_random_effect_distribution(get_random(get_model(dm_test)))
    model_funs_test = get_model_funs(get_model(dm_test))
    helpers_test = get_helper_funs(get_model(dm_test))

    n_test = length(get_individuals(dm_test))
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
        # Conditional draws are built per training individual, once per sample.
        cond_cache = Dict{Int, ComponentArray}()
        for j in 1:n_test
            ind_j = get_individuals(dm_test)[j]
            g = get_re_groups(ind_j)
            unseen_dists() = get!(
                () -> dists_builder(
                    θu, get_const_cov(ind_j), model_funs_test, helpers_test
                ),
                unseen_dists_cache, j
            )
            prior_eta() = get!(
                () -> _cv_prior_mean_eta(
                    dm_test, j, θu, dists_builder,
                    model_funs_test, helpers_test,
                    re_names, ref_eta
                ),
                mean_eta_cache, j
            )
            η_j = ComponentArray(
                NamedTuple(
                    map(re_names) do re
                        tinfo = get(level_train, (re, getproperty(g, re)), nothing)
                        v = if tinfo !== nothing && seen_re_mode == :conditional
                            bi, ti = tinfo
                            η_ti = get!(cond_cache, ti) do
                                ComponentArray(
                                    _build_eta_ind(
                                        dm_train, ti, batch_infos[bi],
                                        bstars_per_sample[s][bi], const_cache, θu
                                    )
                                )
                            end
                            getproperty(η_ti, re)
                        elseif tinfo !== nothing   # :ebe — same for every sample
                            getproperty(η_train_vec[tinfo[2]], re)
                        elseif unseen_re_mode == :montecarlo
                            rand(srng, getproperty(unseen_dists(), re))
                        else                       # :mean — same for every sample
                            getproperty(prior_eta(), re)
                        end
                        return re => v
                    end
                )
            )

            all_dfs[s][j] = _eval_individual_obs(
                dm_test, j, θu, η_j, ll_cache_test, loss; score_rows = score_rows
            )
        end
    end

    # Aggregate across samples: logsumexp for loglikelihood, mean for predicted_mean/loss
    result_dfs = DataFrame[]
    for j in 1:n_test
        s0 = findfirst(s -> !isempty(all_dfs[s][j]), 1:n_mc_samples)
        s0 === nothing && continue
        base_df = all_dfs[s0][j]
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
        df_out[!, :predicted_mean] = [
            mean_cnt[r] > 0 ? mean_acc[r] / mean_cnt[r] : NaN
                for r in 1:n_rows
        ]
        if loss_acc !== nothing
            df_out[!, :loss] = [
                loss_cnt[r] > 0 ? loss_acc[r] / loss_cnt[r] : NaN
                    for r in 1:n_rows
            ]
        end
        push!(result_dfs, df_out)
    end

    return isempty(result_dfs) ? DataFrame() : vcat(result_dfs...)
end

# ── fit_cv ────────────────────────────────────────────────────────────────────

# Per-observation loss failures degrade to NaN by design (one bad row should not abort a
# fold), which meant a wrong-arity or non-numeric callback silently produced an all-NaN
# loss column. Check the contract once, up front, instead (#220).
function _validate_cv_loss(cv_spec::CVSpec, loss)
    applicable(loss, Normal(0.0, 1.0), 0.0) ||
        error("The cv `loss` callback must take two arguments, `(distribution, observation)`, and return a real number; got $(repr(loss)).")
    probe = try
        loss(Normal(0.0, 1.0), 0.0)
    catch e
        error("The cv `loss` callback threw on a probe call `loss(Normal(0, 1), 0.0)`: $(sprint(showerror, e))")
    end
    probe isa Real ||
        error("The cv `loss` callback must return a real number; a probe call returned $(repr(probe))::$(typeof(probe)).")
    return nothing
end

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

With `seen_re_mode=:ebe, unseen_re_mode=:mean` each random effect's level is resolved
separately, so a crossed test individual keeps the trained value for every level it shares
with the training fold and falls back to the prior only for genuinely new levels. The
Monte-Carlo modes (`:conditional`/`:montecarlo`) resolve levels the same way.

`MCMC`/`VI` fits carry no empirical-Bayes random effects; their test predictions are made at
zero/population random effects and `seen_re_mode`/`unseen_re_mode` do not apply (a warning is
emitted). Refit with an EBE-based estimator for random-effect-aware cross-validation.

[`Pooled`](@ref)/[`PooledMap`](@ref) fits evaluate every test individual — seen or
unseen — at the deterministic plug-in η computed from that individual's covariates
with the strategies resolved by the training fit; `seen_re_mode`/`unseen_re_mode`
do not apply and must be left at their defaults.

Returns a [`CVResult`](@ref).
"""
function fit_cv(
        cv_spec::CVSpec, method::FittingMethod, args...;
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
        kwargs...
    )
    seen_re_mode ∈ (:ebe, :conditional) ||
        error("seen_re_mode must be :ebe or :conditional, got $seen_re_mode")
    unseen_re_mode ∈ (:mean, :montecarlo) ||
        error("unseen_re_mode must be :mean or :montecarlo, got $unseen_re_mode")
    n_mc_samples >= 1 ||
        error("n_mc_samples must be >= 1; got $(n_mc_samples).")
    loss === nothing || _validate_cv_loss(cv_spec, loss)
    if method isa Pooled || method isa PooledMap
        (seen_re_mode == :ebe && unseen_re_mode == :mean) ||
            error(
            "Pooled/PooledMap cross-validation evaluates the deterministic plug-in " *
                "η for every test individual; seen_re_mode/unseen_re_mode do not apply."
        )
    end

    n_folds = cv_spec.n_folds
    dm_ref = cv_spec.dm
    fold_rngs = _spawn_child_rngs(rng, n_folds)

    function _run_fold(f)
        dm_train = _rebuild_dm(dm_ref, cv_spec.train_rows[f])
        # Observation folds: evaluate on train ∪ test rows and score only the test rows.
        # Held-out observations then see the same history the training observations
        # provide (the HMM filter is carried through them) instead of restarting the
        # sequence at the test rows (#226).
        eval_rows, score_rows = if cv_spec.kind == :observation
            rows = sort(unique(vcat(cv_spec.train_rows[f], cv_spec.test_rows[f])))
            test_set = Set(cv_spec.test_rows[f])
            (rows, BitVector(r ∈ test_set for r in rows))
        else
            (cv_spec.test_rows[f], nothing)
        end
        dm_test = _rebuild_dm(dm_ref, eval_rows)

        base_kw = (store_data_model = false, ode_args = ode_args, ode_kwargs = ode_kwargs)
        fit_kw = _cv_method_accepts_constants_re(method) ?
            merge(base_kw, (constants_re = constants_re,)) : base_kw
        res_train = fit_model(dm_train, method, args...; fit_kw..., kwargs...)
        θu = get_params(res_train; scale = :untransformed)

        ll_cache_test = build_ll_cache(
            dm_test; ode_args = ode_args, ode_kwargs = ode_kwargs,
            serialization = EnsembleSerial(), force_saveat = true
        )

        re_names = get_re_names(get_random(get_model(dm_train)))
        has_re = !isempty(re_names)

        cr = _res_constants_re(res_train, constants_re)

        obs_df = if !has_re
            _cv_collect_obs(dm_test, θu, nothing, ll_cache_test, loss, score_rows)
        elseif get_result(res_train) isa PooledResult
            # Pooled/PooledMap: plug-in η from each test individual's covariates
            _cv_evaluate_pooled(dm_test, res_train, θu, ll_cache_test, loss, score_rows)
        elseif !_cv_has_re_support(res_train)
            # MCMC/VI results carry no empirical-Bayes modes, so predictions are made at
            # the RE prior location. Say so rather than letting seen_re_mode look honored
            # (#229).
            @warn "fit_cv: $(nameof(typeof(get_result(res_train)))) fits provide no empirical-Bayes random effects; test predictions use zero/population random effects and seen_re_mode/unseen_re_mode do not apply. Refit with Laplace/FOCEI/GHQuadrature/SAEM/MCEM for random-effect-aware cross-validation." maxlog = 1
            _cv_collect_obs(dm_test, θu, nothing, ll_cache_test, loss, score_rows)
        elseif seen_re_mode == :ebe && unseen_re_mode == :mean
            _cv_evaluate_ebe(
                dm_train, dm_test, res_train, θu, ll_cache_test, loss, cr, score_rows
            )
        else
            _cv_evaluate_mc(
                dm_train, dm_test, res_train, θu, ll_cache_test, loss,
                seen_re_mode, unseen_re_mode, n_mc_samples,
                fold_rngs[f], re_names, cr, score_rows
            )
        end

        insertcols!(obs_df, 1, :fold => fill(f, nrow(obs_df)))

        # A fold can score nothing at all (every outcome missing, every ODE solve
        # failed): keep the empty frame instead of indexing a column that is not there.
        ll_all = "loglikelihood" ∈ names(obs_df) ? obs_df[!, :loglikelihood] : Float64[]
        ll_finite = filter(!isnan, ll_all)
        n_dropped = length(ll_all) - length(ll_finite)
        n_dropped == 0 ||
            @warn "fit_cv: fold $f scored $(length(ll_finite)) of $(length(ll_all)) held-out observations; $(n_dropped) could not be evaluated (NaN) and were dropped from the fold total. A model that rules an observation out scores -Inf and is kept."
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
    # Folds hold unequal numbers of observations, so the mean of fold totals depends on
    # the split; report the per-observation score alongside it (#226).
    scored = nrow(all_scores) == 0 ? Float64[] :
        filter(!isnan, all_scores[!, :loglikelihood])

    return CVResult(
        cv_spec, fold_results, all_scores,
        isempty(ll_valid) ? NaN : mean(ll_valid),
        length(ll_valid) >= 2 ? std(ll_valid) : NaN,
        isempty(scored) ? NaN : sum(scored) / length(scored),
        length(scored)
    )
end
