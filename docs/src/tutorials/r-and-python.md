# Using NoLimits from R and Python

NoLimits.jl is fully usable from R and Python. Two thin wrapper packages expose the
Julia package to those languages:

- [NoLimitsR](https://github.com/manuhuth/NoLimitsR) for R
- [NoLimitsPy](https://github.com/manuhuth/NoLimitsPy) for Python

This tutorial shows how to install them, how the quickstart looks in each language, and
how Julia concepts map onto native R and Python ones. All code blocks on this page are
shown for reference and are not executed when the documentation is built, because they
require an R or Python session.

The wrappers target NoLimits 0.2.5 and newer.

## What the wrappers are

Both wrappers are *dynamic*: every name exported by NoLimits.jl is reachable through the
wrapper without any per-function glue code. In R the functions live in the environment
returned by `nolimits()`, in Python they are attributes of the `NoLimitsPy` module. The
wrappers add only session management and a handful of conversion helpers, so a feature
that lands in NoLimits.jl is available from R and Python as soon as the Julia package is
updated, with no wrapper release in between.

The consequence is that this documentation is the API reference for the wrappers too.
Function names, arguments, and semantics are the Julia ones described throughout these
pages; only the syntax around them changes.

Julia macros are not part of the dynamic surface. Models are therefore written as
strings and handed to `nl_model` (R) or `nl.model` (Python), which wraps the block in
`@Model begin ... end` and evaluates it in a session module.

## Installation

### R

Install Julia 1.10 or newer, for example with [juliaup](https://github.com/JuliaLang/juliaup),
then install the R package and the Julia dependencies:

```r
install.packages("remotes")
remotes::install_github("manuhuth/NoLimitsR")

NoLimitsR::install_nolimits()
```

`install_nolimits()` adds NoLimits.jl and CairoMakie to a shared Julia environment named
`NoLimitsR`. It downloads and precompiles the Julia packages, which takes several minutes
on the first run. Optional integrations go through the same call:

```r
NoLimitsR::install_nolimits(extras = c("SimpleChains", "Lux", "CSV", "JLD2"))
```

Julia boots lazily on the first `nolimits()` call, not at `library()` time.

### Python

```bash
pip install git+https://github.com/manuhuth/NoLimitsPy.git
```

Julia itself and the NoLimits.jl packages are resolved automatically on first import by
[juliapkg](https://github.com/JuliaPy/pyjuliapkg), so no separate Julia installation step
is needed. The first import downloads Julia if necessary and precompiles NoLimits.jl,
which takes several minutes; later imports are fast. Install `pandas` as well, since
`NoLimitsPy.collect` returns a pandas `DataFrame`.

Optional Julia integrations are installed from Python:

```python
import NoLimitsPy as nl

nl.install_julia_packages("SimpleChains", "JLD2")
```

## Quickstart side by side

The same one-compartment exponential-decay model with a subject-level random effect, in
both languages. Note the shape in each case: model as a string, a native data frame
straight into `DataModel`, `fit_model`, then `predict`.

### R

```r
library(NoLimitsR)

nl <- nolimits()

model <- nl_model('
@fixedEffects begin
    A0    = RealNumber(10.0, scale=:log)
    k     = RealNumber(0.5,  scale=:log)
    omega = RealNumber(0.3,  scale=:log)
    sigma = RealNumber(0.5,  scale=:log)
end

@covariates begin
    time = Covariate()
end

@randomEffects begin
    eta = RandomEffect(Normal(0.0, omega); column=:ID)
end

@formulas begin
    pred = A0 * exp(eta) * exp(-k * time)
    y ~ Normal(pred, sigma)
end
')

df <- data.frame(
  ID = rep(c("s1", "s2", "s3", "s4"), each = 4),
  time = rep(c(0, 1, 2, 4), times = 4),
  y = c(10.2, 6.1, 3.6, 1.4, 12.5, 7.8, 4.9, 1.9,
        8.1, 4.9, 3.0, 1.1, 11.0, 6.5, 4.1, 1.6)
)

jdf <- nl_data(df)
dm <- nl$DataModel(model, jdf, primary_id = jl_sym("ID"), time_col = jl_sym("time"))
res <- nl$fit_model(dm, nl$Laplace())

nl$get_objective(res)
head(nl_collect(nl$predict(res, jdf)))

nl_plot("plot_fits", res, file = "fits.png")
```

`nl_attach()` puts every NoLimits function on the search path, so `fit_model(dm, Laplace())`
also works without the `nl$` prefix. Names that would mask an existing R function, such as
`stats::predict`, are skipped and reported; call those as `nl$predict(...)`.

### Python

```python
import pandas as pd
import NoLimitsPy as nl

model = nl.model("""
@fixedEffects begin
    A0    = RealNumber(10.0, scale=:log)
    k     = RealNumber(0.5,  scale=:log)
    omega = RealNumber(0.3,  scale=:log)
    sigma = RealNumber(0.5,  scale=:log)
end

@covariates begin
    time = Covariate()
end

@randomEffects begin
    eta = RandomEffect(Normal(0.0, omega); column=:ID)
end

@formulas begin
    pred = A0 * exp(eta) * exp(-k * time)
    y ~ Normal(pred, sigma)
end
""")

df = pd.DataFrame({
    "ID": [s for s in ["s1", "s2", "s3", "s4"] for _ in range(4)],
    "time": [0.0, 1.0, 2.0, 4.0] * 4,
    "y": [10.2, 6.1, 3.6, 1.4, 12.5, 7.8, 4.9, 1.9,
          8.1, 4.9, 3.0, 1.1, 11.0, 6.5, 4.1, 1.6],
})

dm = nl.DataModel(model, df, primary_id=nl.sym("ID"), time_col=nl.sym("time"))
res = nl.fit_model(dm, nl.Laplace())

print(nl.get_objective(res))
print(nl.collect(nl.predict(res, df)).head())

nl.plot("plot_fits", res, file="fits.png")
```

## Symbols, NamedTuples, and other option types

NoLimits uses a few Julia-specific argument types. Each has a direct native equivalent.

Symbol arguments such as `primary_id`, `time_col`, and `scale` are written as strings
passed through a helper:

```r
dm <- nl$DataModel(model, jdf, primary_id = jl_sym("ID"), time_col = jl_sym("time"))
```

```python
dm = nl.DataModel(model, df, primary_id=nl.sym("ID"), time_col=nl.sym("time"))
```

Inside a model string, Julia syntax applies as usual, so `scale=:log` is written exactly
as in Julia.

NamedTuple options such as `optim_kwargs` or `theta_0_untransformed` are built with
`nl_nt()` or a named `list()` in R, and with a plain dict in Python:

```r
res <- nl$fit_model(dm, nl$Laplace(optim_kwargs = nl_nt(show_trace = TRUE)),
                    theta_0_untransformed = nl_nt(k = 0.7))
```

```python
res = nl.fit_model(dm, nl.Laplace(optim_kwargs={"show_trace": True}),
                   theta_0_untransformed={"k": 0.7})
```

Options that NoLimits keys by value rather than by name, such as `constants_re`, are
Julia dictionaries. Build those with `nl_eval()` in R; in Python a dict with non-string
keys is passed through as a Julia dictionary unchanged.

## Data in, results out

`DataModel` expects a Julia `DataFrame`. In R, convert with `nl_data(df)`; in Python a
pandas `DataFrame` is converted automatically at the call boundary, and `nl.to_julia(df)`
converts once up front when the same frame is used repeatedly.

Result tables come back with `nl_collect(x)` (R) and `nl.collect(x)` (Python). Scalars
convert automatically. Julia `missing` becomes `NA` in R and `None` in Python.

Two caveats are worth remembering in both languages: duplicate column names are renamed,
because DataFrames.jl requires unique names, and wide tables are the expensive direction
of the round trip, since collection cost tracks the number of columns rather than rows.
Reshape a result with one column per parameter, subject, or simulation draw to long form
in Julia before collecting it. The wrapper READMEs list the remaining type-specific
caveats, including R factor columns and pandas nullable dtypes.

Objects that hold one table per random effect, such as `get_random_effects(res)`, are
collected field by field, as in `nl_collect(re$eta)` or `nl.collect(re.eta)`.

## Plotting

Any NoLimits `plot_*` function is called through a single helper that forwards
`save_path`:

```r
nl_plot("plot_fits", res, file = "fits.png")
nl_plot("plot_vpc", res, file = "vpc.png")
```

```python
nl.plot("plot_fits", res, file="fits.png")
nl.plot("plot_vpc", res, file="vpc.png")
```

CairoMakie is loaded on the first plot call only. In R, omitting `file` writes a
temporary PNG and draws it in the active graphics device in interactive sessions. Makie
leaves some figure memory behind on every call; the helpers run a Julia garbage
collection after saving, but a very long plotting loop still grows, so restart the
session or run the loop in a subprocess if memory matters.

## How it maps

| Julia concept | R | Python |
|---|---|---|
| Exported function `f` | `nl$f(...)`, or `f(...)` after `nl_attach()` | `nl.f(...)` |
| `@Model begin ... end` | `nl_model("...")` | `nl.model("...")` |
| Symbol `:ID` | `jl_sym("ID")` | `nl.sym("ID")` |
| NamedTuple `(a = 1,)` | `nl_nt(a = 1)` or `list(a = 1)` | `{"a": 1}` |
| Dictionary keyed by values | `nl_eval("Dict(...)")` | dict with non-string keys |
| `DataFrame` argument | `nl_data(df)` | pandas frame passed directly |
| Table result | `nl_collect(x)` | `nl.collect(x)` |
| `print(x)` | `nl_string(x)` | `str(x)` |
| Arbitrary Julia code | `nl_eval(code)` | `nl.seval(code)` |
| Saving a fit | `nl$save_fit(res, "fit.jld2")` | `nl.save_fit(res, "fit.jld2")` |
| Loading a fit | `nl$load_fit("fit.jld2")` | `nl.load_fit("fit.jld2")` |

## Environment management

### R

Everything lives in a shared Julia environment named `NoLimitsR` by default, so all R
sessions on the machine use the same NoLimits.jl version. A per-project environment takes
one extra call, where `env` is either a shared environment name or a directory path:

```r
NoLimitsR::install_nolimits(env = "myproject")   # once, creates the environment
nl_use_env("myproject")                          # before the first nolimits() call

NoLimitsR::nolimits_status()                     # what Julia and NoLimits versions are in use
NoLimitsR::update_nolimits()                     # Pkg.update() on the target environment
```

Pre-release versions can be tracked with `install_nolimits(rev = "main")`, which installs
NoLimits.jl from that branch, tag, or commit instead of the registry; a plain
`install_nolimits()` returns the environment to registry releases.

`nolimits_status()` is the first thing to run when something looks wrong. It reports
whether a suitable Julia was found, the target and active environment, the Julia and
NoLimits.jl versions, and the packages installed in the environment.

### Python

juliapkg puts the Julia environment inside the active virtualenv, so a new venv already
gives a fresh, isolated set of Julia packages, and a venv per project is the natural unit
of isolation. To place the environment somewhere else, point juliapkg at a project
directory before the first NoLimits use:

```python
nl.use_env("julia-envs/myproject")   # created and resolved on first use

nl.status()                          # Julia and NoLimits versions, active project, packages
nl.update()                          # Pkg.update() on the active environment
```

### Both

The environment is a boot-time choice, because Julia cannot unload a package. Select it
before the first NoLimits call. After Julia is running, an update or a switch takes effect
in the next session; in R, `nl_use_env(env, restart = TRUE)` stops the running Julia
process and switches immediately, which discards all existing Julia objects.

## Honest notes

- **Objects are session-bound.** `Model`, `DataModel`, and fit-result objects are live
  references into the Julia session and do not survive the process. In R, `save.image()`,
  `.RData`, and `saveRDS()` store the reference rather than the data, so the object
  reappears and then fails on first use. In Python, `pickle.dump` appears to succeed and
  then fails when loaded in a new process. Persist fits with `save_fit` and `load_fit`
  after installing JLD2. A reloaded fit needs its model string re-evaluated, because
  models themselves are not serializable.
- **For a readable record**, save the text of `nl_string(x)` in R or `str(x)` in Python;
  that is Julia's own printout of the object.
- **Julia console output streams through.** Optimizer traces from `optim_kwargs` with
  `show_trace` and progress logs appear in the R or Python console automatically. In
  Jupyter or the VS Code Interactive Window, raw Julia output may show up in the terminal
  running the kernel instead of in the cell.
- **Julia errors arrive as Julia's own message** without the Julia stack trace, so the
  first failing call is the place to look, not a traceback.
- **Threads (Python).** Julia must boot on the main thread. Call something once from the
  main thread, for example `nl.seval("1+1")`, before using NoLimitsPy from worker threads.
- **First run is slow.** Installation precompiles Julia packages, and the first fit in a
  session pays Julia's compilation cost for the generated model code. Both are one-time
  per environment and per session respectively.

## Models with a Julia prelude

`nl_model` and `nl.model` accept arbitrary Julia code before the `@Model` block, which is
how neural-network parameters, custom functions, and weak-dependency imports enter a
model. Install the optional Julia package once (`install_nolimits(extras = "SimpleChains")`
in R, `nl.install_julia_packages("SimpleChains")` in Python), then write the prelude into
the model string:

!!! tip "No install needed for a plain network"
    `FFNNParameters` builds the same architecture from its layer sizes without any optional
    dependency, so the prelude reduces to the `@Model` block:

    ```julia
    z = FFNNParameters((1, 2, 1); activation=:tanh, output_activation=:identity,
        function_name=:NN1, calculate_se=false)
    ```

    See [function approximators](../model-building/universal-function-approximators.md).

```r
model <- nl_model('
using SimpleChains
chain = SimpleChain(static(1), TurboDense(tanh, 2), TurboDense(identity, 1))

@Model begin
    @fixedEffects begin
        z = NNParameters(chain; function_name=:NN1, calculate_se=false)
        sigma = RealNumber(0.5, scale=:log)
    end
    @covariates begin
        time = Covariate()
    end
    @formulas begin
        pred = NN1([time], z)[1]
        y ~ Normal(pred, sigma)
    end
end
')
```

```python
model = nl.model("""
using SimpleChains
chain = SimpleChain(static(1), TurboDense(tanh, 2), TurboDense(identity, 1))

@Model begin
    @fixedEffects begin
        z = NNParameters(chain; function_name=:NN1, calculate_se=false)
        sigma = RealNumber(0.5, scale=:log)
    end
    @covariates begin
        time = Covariate()
    end
    @formulas begin
        pred = NN1([time], z)[1]
        y ~ Normal(pred, sigma)
    end
end
""")
```

Note that the whole `@Model begin ... end` block is written out explicitly once a prelude
is present. The same mechanism covers Lux, CSV, JLD2, Copulas, and Turing.

## Where to go next

Everything else in this documentation applies unchanged. Pick the estimator on the
[Estimation](../estimation/index.md) pages, build the model with the
[Model Building](../model-building/index.md) blocks, and translate the Julia calls with
the mapping table above. The wrapper READMEs at
[NoLimitsR](https://github.com/manuhuth/NoLimitsR) and
[NoLimitsPy](https://github.com/manuhuth/NoLimitsPy) document the remaining
language-specific details and conversion caveats.
