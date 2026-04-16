export save_fit, load_fit

import JLD2
using Logging: with_logger, NullLogger
using SciMLBase: EnsembleSerial, EnsembleThreads, EnsembleDistributed
using Random: default_rng

# ─── Format version ──────────────────────────────────────────────────────────

const _SERIALIZATION_FORMAT_VERSION = 1

# ─── Sampler stub ────────────────────────────────────────────────────────────

"""
Placeholder for an MCMC sampler in a loaded `FitResult`. Records the original
sampler kind (`:nuts`, `:hmc`, `:mh`, etc.) but cannot be used to run sampling.
"""
struct _SavedSamplerStub
    kind::Symbol
end
Base.show(io::IO, s::_SavedSamplerStub) =
    print(io, "_SavedSamplerStub(:$(s.kind)) [not a live sampler; reconstructed from disk]")

# ─── Saved DataModel config ──────────────────────────────────────────────────

struct _SavedDataModelConfig
    primary_id::Symbol
    time_col::Symbol
    evid_col::Union{Nothing, Symbol}
    amt_col::Symbol
    rate_col::Symbol
    cmt_col::Symbol
    serialization_kind::Symbol   # :serial / :threads / :distributed
end

# ─── Per-method saved result structs ─────────────────────────────────────────
# The `raw` field (SciMLBase.OptimizationSolution) is always dropped because its
# `cache` field contains closures.  After loading, `get_raw(res)` returns `nothing`.

struct SavedMLEResult{S, O, I, N}
    solution::S
    objective::O
    iterations::I
    notes::N
end

struct SavedMAPResult{S, O, I, N}
    solution::S
    objective::O
    iterations::I
    notes::N
end

struct SavedLaplaceResult{S, O, I, N, B}
    solution::S
    objective::O
    iterations::I
    notes::N
    eb_modes::B
end

struct SavedLaplaceMAPResult{S, O, I, N, B}
    solution::S
    objective::O
    iterations::I
    notes::N
    eb_modes::B
end

struct SavedMCEMResult{S, O, I, N, B}
    solution::S
    objective::O
    iterations::I
    notes::N
    eb_modes::B
end

struct SavedSAEMResult{S, O, I, N, B}
    solution::S
    objective::O
    iterations::I
    notes::N
    eb_modes::B
end

struct SavedGHQuadratureResult{S, O, I, N, B}
    solution::S
    objective::O
    iterations::I
    notes::N
    eb_modes::B
end

struct SavedGHQuadratureMAPResult{S, O, I, N, B}
    solution::S
    objective::O
    iterations::I
    notes::N
    eb_modes::B
end

# MCMCChains.Chains is plain array data and serializes directly with JLD2.
# The sampler is replaced with a _SavedSamplerStub.
struct SavedMCMCResult{C, N, O}
    chain::C
    sampler_kind::Symbol
    n_samples::Int
    notes::N
    observed::O
end

# The VI posterior (AdvancedVI distribution) is plain data and saved directly.
# `state` and `varinfo` are dropped; `get_vi_state` returns `nothing` after loading.
struct SavedVIResult{Q, T, N, O, C}
    posterior::Q
    trace::T
    n_iter::Int
    max_iter::Int
    final_elbo::Float64
    converged::Bool
    notes::N
    observed::O
    coord_names::C
end

# ─── Envelope structs ────────────────────────────────────────────────────────

struct SavedFitResult{M, R, S, D, K, DF}
    format_version::Int
    method::M
    result::R
    summary::S
    diagnostics::D
    fit_kwargs_saved::K
    df::DF                           # DataFrame or nothing
    data_model_config::Union{Nothing, _SavedDataModelConfig}
end

struct SavedMultistartFitResult{M, R, RE, SK, EK, B, DF}
    format_version::Int
    method::M                        # inner FittingMethod (MLE, MAP, etc.)
    saved_results_ok::R
    saved_results_err::RE
    starts_ok::SK
    starts_err::EK
    errors_err_strings::Vector{String}
    best_idx::Int
    scores_ok::B
    df::DF
    data_model_config::Union{Nothing, _SavedDataModelConfig}
end

# ─── Method stripping ─────────────────────────────────────────────────────────
# FittingMethod structs contain optimizer/sampler objects with bare Function fields
# that JLD2 cannot serialise cleanly.  Replace them with a lightweight stub.

_strip_fitting_method(::MLE)                  = _SavedFittingMethod(:mle)
_strip_fitting_method(::MAP)                  = _SavedFittingMethod(:map)
_strip_fitting_method(::Laplace)              = _SavedFittingMethod(:laplace)
_strip_fitting_method(::LaplaceMAP)           = _SavedFittingMethod(:laplacemap)
_strip_fitting_method(::MCEM)                 = _SavedFittingMethod(:mcem)
_strip_fitting_method(::SAEM)                 = _SavedFittingMethod(:saem)
_strip_fitting_method(::MCMC)                 = _SavedFittingMethod(:mcmc)
_strip_fitting_method(::GHQuadrature)         = _SavedFittingMethod(:ghquadrature)
_strip_fitting_method(::GHQuadratureMAP)      = _SavedFittingMethod(:ghquadraturemap)
_strip_fitting_method(::VI)                   = _SavedFittingMethod(:vi)
_strip_fitting_method(m::_SavedFittingMethod) = m  # idempotent

# ─── Strip helpers ────────────────────────────────────────────────────────────

# OptimizationSolution holds a `cache` field that contains the problem closure
# (which captures the DataModel / model functions).  Extract only the parameter
# vector `sol.u` so nothing session-specific leaks into the saved file.
_strip_solution(sol) = hasproperty(sol, :u) ? sol.u : sol

function _ensemble_to_symbol(s)
    s isa EnsembleSerial      && return :serial
    s isa EnsembleThreads     && return :threads
    s isa EnsembleDistributed && return :distributed
    return :serial
end

function _build_saved_config(dm::DataModel)
    cfg = dm.config
    return _SavedDataModelConfig(
        cfg.primary_id,
        cfg.time_col,
        cfg.evid_col,
        cfg.amt_col,
        cfg.rate_col,
        cfg.cmt_col,
        _ensemble_to_symbol(cfg.serialization)
    )
end

function _strip_fit_kwargs(kw::NamedTuple)
    keep = (:constants, :constants_re, :penalty, :ode_args, :ode_kwargs, :theta_0_untransformed)
    kept = NamedTuple(k => getfield(kw, k) for k in keep if haskey(kw, k))
    ser_kind = haskey(kw, :serialization) ? _ensemble_to_symbol(kw.serialization) : :serial
    return merge(kept, (; serialization_kind=ser_kind))
end

_strip_method_result(r::MLEResult) =
    SavedMLEResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes)

_strip_method_result(r::MAPResult) =
    SavedMAPResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes)

_strip_method_result(r::LaplaceResult) =
    SavedLaplaceResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes, r.eb_modes)

_strip_method_result(r::LaplaceMAPResult) =
    SavedLaplaceMAPResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes, r.eb_modes)

_strip_method_result(r::MCEMResult) =
    SavedMCEMResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes, r.eb_modes)

_strip_method_result(r::SAEMResult) =
    SavedSAEMResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes, r.eb_modes)

_strip_method_result(r::GHQuadratureResult) =
    SavedGHQuadratureResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes, r.eb_modes)

_strip_method_result(r::GHQuadratureMAPResult) =
    SavedGHQuadratureMAPResult(_strip_solution(r.solution), r.objective, r.iterations, r.notes, r.eb_modes)

_strip_method_result(r::MCMCResult) =
    SavedMCMCResult(r.chain, _mcmc_sampler_kind(r.sampler), r.n_samples, r.notes, r.observed)

_strip_method_result(r::VIResult) =
    SavedVIResult(r.posterior, r.trace, r.n_iter, r.max_iter, r.final_elbo,
                  r.converged, r.notes, r.observed, r.coord_names)

function _strip_fit_result(res::FitResult; include_data::Bool=false)
    # Strip optimizer from diagnostics (it may hold Optim.jl internal state)
    diag = FitDiagnostics(res.diagnostics.timing, nothing,
                          res.diagnostics.convergence, res.diagnostics.notes)
    fkw  = _strip_fit_kwargs(res.fit_kwargs)
    dm   = res.data_model
    df     = (include_data && dm !== nothing) ? dm.df     : nothing
    config = (include_data && dm !== nothing) ? _build_saved_config(dm) : nothing
    return SavedFitResult(
        _SERIALIZATION_FORMAT_VERSION,
        _strip_fitting_method(res.method),
        _strip_method_result(res.result),
        res.summary,
        diag,
        fkw,
        df,
        config,
    )
end

_strip_partial_result(r::FitResult) = _strip_fit_result(r; include_data=false)
_strip_partial_result(::Nothing)    = nothing
_strip_partial_result(r)            = nothing  # unknown type; drop silently

function _strip_fit_result(res::MultistartFitResult; include_data::Bool=false)
    saved_ok  = SavedFitResult[_strip_fit_result(r; include_data=false) for r in res.results_ok]
    saved_err = [_strip_partial_result(r) for r in res.results_err]
    err_strs  = String[sprint(showerror, e) for e in res.errors_err]
    # Take df/config from the best result's data_model
    best_dm = isempty(res.results_ok) ? nothing : res.results_ok[res.best_idx].data_model
    df      = (include_data && best_dm !== nothing) ? best_dm.df                   : nothing
    config  = (include_data && best_dm !== nothing) ? _build_saved_config(best_dm) : nothing
    return SavedMultistartFitResult(
        _SERIALIZATION_FORMAT_VERSION,
        _strip_fitting_method(res.method),   # store stub, not full method
        saved_ok,
        saved_err,
        res.starts_ok,
        res.starts_err,
        err_strs,
        res.best_idx,
        res.scores_ok,
        df,
        config,
    )
end

# ─── Reconstruct helpers ──────────────────────────────────────────────────────

function _symbol_to_serialization(sym::Symbol)
    sym === :threads     && return EnsembleThreads()
    sym === :distributed && return EnsembleDistributed()
    return EnsembleSerial()
end

_reconstruct_method_result(s::SavedMLEResult) =
    MLEResult(s.solution, s.objective, s.iterations, nothing, s.notes)

_reconstruct_method_result(s::SavedMAPResult) =
    MAPResult(s.solution, s.objective, s.iterations, nothing, s.notes)

_reconstruct_method_result(s::SavedLaplaceResult) =
    LaplaceResult(s.solution, s.objective, s.iterations, nothing, s.notes, s.eb_modes)

_reconstruct_method_result(s::SavedLaplaceMAPResult) =
    LaplaceMAPResult(s.solution, s.objective, s.iterations, nothing, s.notes, s.eb_modes)

_reconstruct_method_result(s::SavedMCEMResult) =
    MCEMResult(s.solution, s.objective, s.iterations, nothing, s.notes, s.eb_modes)

_reconstruct_method_result(s::SavedSAEMResult) =
    SAEMResult(s.solution, s.objective, s.iterations, nothing, s.notes, s.eb_modes)

_reconstruct_method_result(s::SavedGHQuadratureResult) =
    GHQuadratureResult(s.solution, s.objective, s.iterations, nothing, s.notes, s.eb_modes)

_reconstruct_method_result(s::SavedGHQuadratureMAPResult) =
    GHQuadratureMAPResult(s.solution, s.objective, s.iterations, nothing, s.notes, s.eb_modes)

_reconstruct_method_result(s::SavedMCMCResult) =
    MCMCResult(s.chain, _SavedSamplerStub(s.sampler_kind), s.n_samples, s.notes, s.observed)

# VIResult: state=nothing, varinfo=nothing — both are not reconstructable from disk.
_reconstruct_method_result(s::SavedVIResult) =
    VIResult(s.posterior, s.trace, nothing, s.n_iter, s.max_iter,
             s.final_elbo, s.converged, s.notes, s.observed, nothing, s.coord_names)

function _reconstruct_fit_kwargs(kw::NamedTuple)
    keep = (:constants, :constants_re, :penalty, :ode_args, :ode_kwargs, :theta_0_untransformed)
    kept = NamedTuple(k => getfield(kw, k) for k in keep if haskey(kw, k))
    ser  = _symbol_to_serialization(get(kw, :serialization_kind, :serial))
    return merge(kept, (; serialization=ser))
end

function _reconstruct_data_model(df, config::_SavedDataModelConfig, model)
    return DataModel(model, df;
                     primary_id=config.primary_id,
                     time_col=config.time_col,
                     evid_col=config.evid_col,
                     amt_col=config.amt_col,
                     rate_col=config.rate_col,
                     cmt_col=config.cmt_col,
                     serialization=_symbol_to_serialization(config.serialization_kind))
end

function _resolve_dm(saved_df, saved_config, model, dm)
    if dm !== nothing
        saved_df !== nothing &&
            @warn "load_fit: both `dm` kwarg and a saved DataFrame are present; using provided `dm`."
        return dm
    end
    if model !== nothing && saved_df !== nothing && saved_config !== nothing
        return _reconstruct_data_model(saved_df, saved_config, model)
    end
    return nothing
end

function _reconstruct_fit_result(saved::SavedFitResult, model, dm)
    dm_r   = _resolve_dm(saved.df, saved.data_model_config, model, dm)
    result = _reconstruct_method_result(saved.result)
    fkw    = _reconstruct_fit_kwargs(saved.fit_kwargs_saved)
    return FitResult(saved.method, result, saved.summary, saved.diagnostics,
                     dm_r, (dm_r, saved.method), fkw)
end

_reconstruct_partial_result(::Nothing, model, dm) = nothing
_reconstruct_partial_result(r::SavedFitResult, model, dm) = _reconstruct_fit_result(r, model, dm)
_reconstruct_partial_result(_, model, dm) = nothing

function _reconstruct_multistart(saved::SavedMultistartFitResult, model, dm)
    dm_r        = _resolve_dm(saved.df, saved.data_model_config, model, dm)
    results_ok  = FitResult[_reconstruct_fit_result(r, model, dm_r)
                             for r in saved.saved_results_ok]
    results_err = [_reconstruct_partial_result(r, model, dm_r)
                   for r in saved.saved_results_err]
    return MultistartFitResult(saved.method, results_ok, results_err,
                               saved.starts_ok, saved.starts_err,
                               saved.errors_err_strings,
                               saved.best_idx, saved.scores_ok)
end

# ─── Public API ──────────────────────────────────────────────────────────────

"""
    save_fit(path, res; include_data=false)

Save a [`FitResult`](@ref) or [`MultistartFitResult`](@ref) to a JLD2 file at `path`.

Non-serializable fields are stripped automatically:
- Runtime-generated functions in `DataModel` (closures from the `@Model` macro)
- Optimization solver cache in `raw` (`get_raw` returns `nothing` after loading)
- MCMC `sampler` (replaced with a `_SavedSamplerStub` on load)
- VI `state` and `varinfo` (`get_vi_state` returns `nothing` after loading)
- `rng` from `fit_kwargs` (defaults to `Random.default_rng()` on load)
- Diagnostic `optimizer` field

# Keyword Arguments
- `include_data::Bool = false`: if `true`, also save the original `DataFrame` and
  `DataModelConfig`. When loading, only a `model` argument is then required (not `dm`).

# Returns
The path string.

# See also
[`load_fit`](@ref)
"""
function save_fit(path::AbstractString, res::Union{FitResult, MultistartFitResult};
                  include_data::Bool=false)
    stripped = _strip_fit_result(res; include_data=include_data)
    with_logger(NullLogger()) do
        JLD2.jldsave(path; saved=stripped)
    end
    return path
end

"""
    load_fit(path; model=nothing, dm=nothing)

Load a fit result saved with [`save_fit`](@ref) from a JLD2 file at `path`.

Returns a [`FitResult`](@ref) or [`MultistartFitResult`](@ref) depending on what was saved.

Provide `model` and/or `dm` to reconstruct the `DataModel`, which is required for
accessors that need it: `get_random_effects`, `get_loglikelihood`, and all plotting
functions.

# Keyword Arguments
- `model`: the [`Model`](@ref) used during fitting. Required when the file was not
  saved with `include_data=true` and `dm` is not provided.
- `dm`: the [`DataModel`](@ref) used during fitting. If provided, takes precedence
  over any saved `DataFrame`.

# Limitations after loading
| Accessor | After load |
|---|---|
| `get_params`, `get_objective`, `get_converged`, `get_iterations` | Full |
| `get_raw` | Returns `nothing` (solver cache not serializable) |
| `get_sampler` (MCMC) | Returns `_SavedSamplerStub` (not a live sampler) |
| `get_vi_state` (VI) | Returns `nothing` (state not serializable) |
| `get_random_effects`, `get_loglikelihood`, plotting | Require `dm` or `include_data=true` |
| `get_multistart_errors` | Returns `Vector{String}` instead of `Vector{Exception}` |

# See also
[`save_fit`](@ref)
"""
function load_fit(path::AbstractString; model=nothing, dm=nothing)
    saved = with_logger(NullLogger()) do
        JLD2.load(path, "saved")
    end
    saved.format_version == _SERIALIZATION_FORMAT_VERSION ||
        error("Unsupported SavedFitResult format version $(saved.format_version). " *
              "Expected version $(_SERIALIZATION_FORMAT_VERSION). " *
              "This file may have been written by a different version of NoLimits.jl.")
    if saved isa SavedMultistartFitResult
        return _reconstruct_multistart(saved, model, dm)
    end
    return _reconstruct_fit_result(saved, model, dm)
end
