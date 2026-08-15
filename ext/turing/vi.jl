function _vi_unpack_output(out)
    # Turing >=0.45 returns a `VIResult` struct (fields `q`, `info`, `state`, `ldf`);
    # older versions returned a 3-tuple `(q, info, state)`. Handle both.
    if !(out isa Tuple)
        (hasproperty(out, :q) && hasproperty(out, :info) && hasproperty(out, :state)) ||
            error("Unexpected VI output type $(typeof(out)); expected a VIResult struct or a 3-tuple from Turing.vi.")
        return (getproperty(out, :q), getproperty(out, :info), getproperty(out, :state))
    end
    length(out) == 3 ||
        error("Unexpected VI output tuple length $(length(out)); expected 3.")
    q, b, c = out
    b_is_info = b isa AbstractVector && (isempty(b) || first(b) isa NamedTuple)
    c_is_info = c isa AbstractVector && (isempty(c) || first(c) isa NamedTuple)
    b_is_state = b isa NamedTuple
    c_is_state = c isa NamedTuple
    if b_is_info && c_is_state
        return q, b, c
    elseif c_is_info && b_is_state
        return q, c, b
    elseif b_is_info
        return q, b, c
    elseif c_is_info
        return q, c, b
    end
    error("Could not infer VI output ordering from Turing.vi return values.")
end

function _vi_info_elbos(info)
    isempty(info) && return Float64[]
    out = Vector{Float64}(undef, length(info))
    for i in eachindex(info)
        row = _as_namedtuple(info[i])
        haskey(row, :elbo) || error("VI trace entry $(i) is missing :elbo.")
        out[i] = Float64(getfield(row, :elbo))
    end
    return out
end

function _vi_converged(
        info, max_iter::Int; window::Int = 20, rtol::Float64 = 1.0e-3, atol::Float64 = 1.0e-6
    )
    isempty(info) && return false
    if length(info) < max_iter
        return true
    end
    elbos = _vi_info_elbos(info)
    all(isfinite, elbos) || return false
    length(elbos) >= 2 || return false
    w = min(window, length(elbos) - 1)
    w >= 1 || return false
    tail = @view elbos[(end - w):end]
    deltas = abs.(diff(tail))
    scale = max(1.0, abs(elbos[end]))
    return maximum(deltas) <= atol + rtol * scale
end

function _vi_coord_names(varinfo, θ0_u::ComponentArray)
    # DynamicPPL >=0.41 removed `syms`, the `varinfo.metadata` layout, and `varinfo[vn]`.
    # Build per-coordinate names from `keys(varinfo)` (the VarNames, in VarInfo order)
    # expanded by each variable's internal (flattened) length — matching the flat order
    # AdvancedVI uses for the variational parameters. Array-valued blocks are spelled
    # with their Cartesian index ("Ω[1,1]"), the coordinate-key convention every chain
    # consumer looks up; `getindex_internal` flattens column-major, as does
    # `CartesianIndices`.
    names = Symbol[]
    for vn in keys(varinfo)
        base = string(vn)
        n = length(DynamicPPL.getindex_internal(varinfo, vn))
        sym = Symbol(base)
        val = hasproperty(θ0_u, sym) ? getproperty(θ0_u, sym) : nothing
        if n == 1
            push!(names, sym)
        elseif val isa AbstractArray && length(val) == n
            for idx in CartesianIndices(val)
                push!(names, Symbol(base, "[", join(Tuple(idx), ","), "]"))
            end
        else
            for i in 1:n
                push!(names, Symbol(base, "[", i, "]"))
            end
        end
    end
    return names
end

# Map variational draws (rows) from the model's unconstrained/linked space — which is what
# `rand(q)` produces under Turing >=0.45 (vi runs with `unconstrained=true`) — back to the
# natural (constrained) parameter space the consumers expect. Requires the stored model;
# after deserialization (model === nothing) the linked draws are returned unchanged.
function NoLimits._vi_unlink_draws(res::VIResult, linked::AbstractMatrix)
    res.model === nothing && return linked
    model = res.model
    vil = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    ks = collect(keys(vil))
    out = nothing
    for r in axes(linked, 1)
        z = collect(@view linked[r, :])
        vn = DynamicPPL.invlink!!(DynamicPPL.unflatten!!(deepcopy(vil), z), model)
        vals = Float64[]
        for k in ks
            append!(vals, DynamicPPL.getindex_internal(vn, k))
        end
        # The constrained space can be wider than the linked one (a PSD block links to
        # its n(n+1)/2 free coordinates), so size the output from the unlinked row.
        out === nothing && (out = Matrix{Float64}(undef, size(linked, 1), length(vals)))
        out[r, :] .= vals
    end
    return out === nothing ? linked : out
end

function NoLimits._vi_fit_impl(
        dm::DataModel, method::VI, args...;
        constants::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        extra_objective = nothing,
        store_data_model::Bool = true
    )
    fit_kwargs = (
        constants = constants,
        constants_re = constants_re,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
    )
    re_names = get_re_names(get_random(get_model(dm)))
    if !isempty(re_names)
        error(
            "VI is not supported for models with random effects. " *
                "Use MCMC for full Bayesian inference on mixed-effects models, or " *
                "use Laplace/MCEM/SAEM for likelihood-based mixed-effects estimation."
        )
    end
    isempty(keys(penalty)) ||
        error("VI does not support penalty terms. Use priors and MAP instead.")

    fe = get_fixed(get_model(dm))
    _warn_if_scaled_params(fe; method_name = "VI")
    priors = get_priors(fe)
    fixed_names = get_names(fe)
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    free_names = [n for n in fixed_names if !(n in keys(constants))]
    if isempty(free_names)
        error("VI requires at least one sampled parameter. Leave at least one fixed effect free.")
    end
    for name in free_names
        haskey(priors, name) ||
            error("VI requires priors on all free fixed effects. Missing prior for $(name).")
        getfield(priors, name) isa Priorless &&
            error("VI requires priors on all free fixed effects. Priorless for $(name).")
    end
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) ||
                error("theta_0_untransformed is missing parameter $(n).")
        end
        @debug "theta_0_untransformed is currently not used by VI unless turing_kwargs provides q_init."
    end

    cache = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true
    )

    free_names_t = Tuple(free_names)
    priors_nt = NamedTuple{free_names_t}(
        Tuple(
            _turing_prior(getfield(priors, n), n)
                for n in free_names
        )
    )
    fname = _build_turing_model(fixed_names, free_names)
    model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
    model = Base.invokelatest(
        model_fn, dm, cache, serialization, priors_nt, constants, extra_objective
    )
    model = _invokelatest_model(model)

    max_iter = Int(get(method.turing_kwargs, :max_iter, 1000))
    max_iter >= 1 || error("VI requires max_iter >= 1.")
    family = get(method.turing_kwargs, :family, :meanfield)
    family in (:meanfield, :fullrank) || error("VI family must be :meanfield or :fullrank.")
    q_init = get(method.turing_kwargs, :q_init, nothing)
    adtype = get(method.turing_kwargs, :adtype, Turing.AutoForwardDiff())
    show_progress = Bool(
        get(
            method.turing_kwargs, :show_progress, get(method.turing_kwargs, :progress, false)
        )
    )
    algorithm = get(method.turing_kwargs, :algorithm, nothing)
    conv_window = Int(get(method.turing_kwargs, :convergence_window, 20))
    conv_rtol = Float64(get(method.turing_kwargs, :convergence_rtol, 1.0e-3))
    conv_atol = Float64(get(method.turing_kwargs, :convergence_atol, 1.0e-6))
    conv_window >= 1 || error("VI convergence_window must be >= 1.")
    conv_rtol >= 0 || error("VI convergence_rtol must be >= 0.")
    conv_atol >= 0 || error("VI convergence_atol must be >= 0.")

    vi_kwargs = Base.structdiff(
        method.turing_kwargs,
        (
            max_iter = 0,
            family = :meanfield,
            q_init = nothing,
            adtype = nothing,
            progress = false,
            show_progress = false,
            algorithm = nothing,
            convergence_window = 0,
            convergence_rtol = 0.0,
            convergence_atol = 0.0,
        )
    )
    # Turing >=0.45: `vi(rng, model, family, max_iter)` takes the variational FAMILY as the
    # third argument — a function `(rng, ldf) -> q` (e.g. `q_meanfield_gaussian`). `vi` builds
    # a correctly linked `LogDensityFunction` internally and calls the family on it; the old
    # API of pre-constructing `q` from the model was removed. Pass the family function (a
    # user-supplied `q_init` is honored as-is).
    if q_init === nothing
        q_init = family == :meanfield ? Turing.q_meanfield_gaussian :
            Turing.q_fullrank_gaussian
    end

    _set_turing_adbackend!(adtype)
    out = if algorithm === nothing
        Turing.vi(
            rng, model, q_init, max_iter; adtype = adtype,
            show_progress = show_progress, vi_kwargs...
        )
    else
        Turing.vi(
            rng, model, q_init, max_iter; adtype = adtype,
            algorithm = algorithm, show_progress = show_progress, vi_kwargs...
        )
    end
    posterior, trace, state = _vi_unpack_output(out)
    n_iter = length(trace)
    elbos = _vi_info_elbos(trace)
    final_elbo = isempty(elbos) ? NaN : elbos[end]
    converged = _vi_converged(
        trace, max_iter; window = conv_window, rtol = conv_rtol, atol = conv_atol
    )

    varinfo = DynamicPPL.VarInfo(model)
    coord_names = _vi_coord_names(varinfo, get_θ0_untransformed(fe))
    obs = get_df(dm)[:, get_obs_cols(dm)]
    summary = FitSummary(
        final_elbo, converged,
        FitParameters(ComponentArray(), ComponentArray()),
        NamedTuple()
    )
    diagnostics = FitDiagnostics(
        (;),
        (
            family = family, algorithm = algorithm === nothing ? :default : algorithm,
            adtype = adtype,
        ),
        (n_iter = n_iter, max_iter = max_iter),
        (
            final_elbo = final_elbo,
            convergence_window = conv_window,
            convergence_rtol = conv_rtol,
            convergence_atol = conv_atol,
        )
    )
    result = VIResult(
        posterior, trace, state, n_iter, max_iter, final_elbo, converged,
        NamedTuple(), obs, coord_names, model
    )
    res = FitResult(
        method, result, summary, diagnostics,
        store_data_model ? dm : nothing, args, fit_kwargs
    )
    return _with_posterior_params(res, dm; rng = rng)
end
