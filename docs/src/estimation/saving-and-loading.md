# Saving and Loading Fit Results

Fit results can be persisted to disk using `save_fit` and reloaded with `load_fit`. Results are stored as [JLD2](https://github.com/JuliaIO/JLD2.jl) files. Because the `DataModel` contains runtime-generated closures (from the `@Model` macro) that cannot be serialized, these are stripped on save and must be reconstructed on load. All numerical content — parameters, objectives, chains, empirical Bayes modes — is preserved exactly.

## Basic Usage

```julia
using NoLimits

res = fit_model(dm, NoLimits.Laplace())

# Save to disk
save_fit("my_fit.jld2", res)

# Load in the same session — supply dm so RE/plotting accessors work
res2 = load_fit("my_fit.jld2"; dm=dm)

get_params(res2; scale=:untransformed)
get_random_effects(res2)
```

If `dm` is not provided, the result still loads successfully but accessors that require the data model — `get_random_effects`, `get_loglikelihood`, and all plotting functions — will raise an informative error.

## Reloading in a New Session

Pass `include_data=true` to embed the original `DataFrame` and `DataModelConfig` in the file. When loading, only the `model` is then required — the `DataModel` is reconstructed automatically.

```julia
# Save with data embedded
save_fit("my_fit.jld2", res; include_data=true)

# --- new Julia session ---
include("model.jl")          # defines build_model(...)
model = build_model(...)

res2 = load_fit("my_fit.jld2"; model=model)
dm2  = get_data_model(res2)  # reconstructed from saved DataFrame

get_params(res2; scale=:untransformed)
get_random_effects(res2)
```

If both `dm` and `model` are provided to `load_fit`, `dm` takes precedence over the saved `DataFrame`.

## Multistart Results

`save_fit` and `load_fit` work identically for `MultistartFitResult`. All individual run results are preserved.

```julia
res_ms = fit_model(dm, ms, NoLimits.Laplace())
save_fit("multistart.jld2", res_ms; include_data=true)

# --- new Julia session ---
model   = build_model(...)
res_ms2 = load_fit("multistart.jld2"; model=model)

best = get_multistart_best(res_ms2)
get_params(best; scale=:untransformed)
plot_multistart_waterfall(res_ms2)
```

Note: `get_multistart_errors` returns `Vector{String}` after loading rather than `Vector{Exception}`, since exception objects are not serializable.

## What Is Preserved

| Accessor | After loading |
| --- | --- |
| `get_params`, `get_objective`, `get_converged`, `get_iterations` | Fully preserved |
| `get_chain` (MCMC) | Fully preserved |
| `get_random_effects`, `get_loglikelihood`, plotting | Require `dm` (pass directly or use `include_data=true`) |
| `get_raw` | Returns `nothing` — the optimization solver cache is not serializable |
| `get_sampler` (MCMC) | Returns a `_SavedSamplerStub` recording the sampler kind (`:nuts`, `:mh`, …) |
| `get_multistart_errors` | `Vector{String}` instead of `Vector{Exception}` |

## API

```julia
save_fit(path, res; include_data=false) -> path::String
load_fit(path; model=nothing, dm=nothing) -> FitResult or MultistartFitResult
```
