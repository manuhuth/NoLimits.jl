
# Developers Guide {#Developers-Guide}

This page is for contributors and maintainers working on NoLimits.jl itself. For using the package, see [Quickstart](quickstart.md) and the [Tutorials](tutorials/mixed-effects-multiple-methods.md).

## Scope {#Scope}

NoLimits.jl is a framework for nonlinear mixed-effects modeling of longitudinal data. The codebase is organized around three concerns: a macro-based modeling language (`@Model` and its block macros), data binding and validation (`DataModel`), and a family of estimation methods that all share the `fit_model` interface. Contributions should preserve this separation and the conventions described below.

## Codebase Layout {#Codebase-Layout}

```
src/
├── NoLimits.jl          # Module definition and includes
├── Constants.jl         # Global constants (scales, epsilon)
├── model/               # Modeling language: @Model and its blocks
│   ├── Model.jl                 # @Model macro and the Model struct/bundles
│   ├── Parameters.jl            # Fixed-effect parameter blocks and scales
│   ├── FixedEffects.jl          # @fixedEffects
│   ├── RandomEffects.jl         # @randomEffects and RE distributions
│   ├── Covariates.jl            # @covariates (constant / varying / dynamic)
│   ├── Helpers.jl               # @helpers
│   ├── PreDE.jl                 # @preDifferentialEquation
│   ├── DifferentialEquation.jl  # @DifferentialEquation
│   ├── InitialDE.jl             # @initialDE
│   └── Formulas.jl              # @formulas (deterministic nodes + observations)
├── data_model/          # DataModel construction, validation, batching, events
├── data_simulation/     # simulate_data / simulate_data_model
├── estimation/          # MLE, MAP, MCMC, VI, Laplace, FOCEI, GHQuadrature,
│                        #   MCEM, SAEM, Pooled, Multistart, cross-validation, UQ
├── plotting/            # Fits, VPC, residual and random-effect diagnostics
├── distributions/       # Custom outcome (HMM/Markov) and RE (flow) distributions
├── soft_trees/          # Soft decision trees
└── utils/               # Transforms, splines, general utilities
```


The repository also contains the test suite under `test/`, this documentation under `docs/`, and standard project files. Local-only scratch directories (e.g. `benchmarks/`, `examples/`, `tmp/`) and editor/build artifacts are listed in `.gitignore` and must not be committed.

## Local Development Setup {#Local-Development-Setup}

Clone the repository and instantiate the project environment:

```bash
git clone https://github.com/manuhuth/NoLimits.jl
cd NoLimits.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```


`Manifest.toml` is intentionally not tracked, so `Pkg.instantiate()` resolves dependencies against the `[compat]` bounds in `Project.toml`.

## Coding Conventions {#Coding-Conventions}

A few conventions are load-bearing and enforced throughout the codebase:
- **ForwardDiff compatibility.** All model and likelihood code must be differentiable with ForwardDiff; this requires non-mutating implementations on the differentiated paths.
  
- **Accessor functions.** Struct fields are accessed through `get_*` accessor functions rather than direct field access, both internally and in the public API.
  
- **`ComponentArray` construction.** Reuse existing axes with `ComponentArray(values, getaxes(existing))` rather than reconstructing axes.
  

See the source of an existing estimation method (for example `src/estimation/laplace.jl`) for the expected style when adding a new method.

## Testing Strategy {#Testing-Strategy}

The suite lives in `test/`, with files grouped by component (`estimation_*_tests.jl`, `data_model_*_tests.jl`, `ad_*.jl`, and so on). Run the whole suite with:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```


`test/runtests.jl` runs the suite as a sequence of `julia` **subprocess batches** rather than a single process. Each distinct `@Model` emits type-specialized native code that Julia does not free within a process, so running all files in one process accumulates compiled code and exhausts memory; exiting between batches caps per-process memory. The batch count can be overridden with the `NL_BATCHES` environment variable.

To iterate on a single area, run an individual test file directly:

```bash
julia --project test/estimation_laplace_tests.jl
```


Automatic-differentiation correctness is covered by the `ad_*.jl` files, which check gradients and Hessians of the likelihood and random-effects terms. When adding a feature, add or extend the corresponding test file and keep it wired into the `TEST_FILES` list in `test/runtests.jl`.

## Documentation Workflow {#Documentation-Workflow}

The documentation is built with [Documenter.jl](https://documenter.juliadocs.org) and the [DocumenterVitepress.jl](https://luxdl.github.io/DocumenterVitepress.jl) theme, with a bibliography managed by [DocumenterCitations.jl](https://juliadocs.org/DocumenterCitations.jl). It uses its own environment in `docs/Project.toml`.

Build and preview locally:

```bash
# one-time: resolve the docs environment and dev-install the package into it
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# build the site (set GKSwstype=100 for the headless GR backend used by Plots.jl)
GKSwstype=100 julia --project=docs docs/make.jl

# preview the built site (a local build writes the rendered site to docs/build/1)
julia --project=docs -e 'using LiveServer; LiveServer.serve(dir="docs/build/1")'
```


Tutorials are **precomputed offline** rather than executed during the build: the scripts in `docs/scripts/` (notably `precompute_tutorials.jl`) run the expensive fits once and inject their text output and figures into the tutorial pages via `<!-- injected:... -->` markers and the `docs/src/tutorials/figures/` directory. This keeps the Documenter build fast and deterministic. When changing a tutorial's code, re-run the relevant precompute script so the injected outputs stay in sync. New entries in the bibliography go in `docs/src/references.bib` and are cited inline with the `[key](@cite)` syntax.

## Release Process {#Release-Process}

Releases follow the standard Julia workflow:
1. Bump the `version` field in `Project.toml` (semantic versioning) and update any `[compat]` bounds as needed.
  
2. Merge to `main`; continuous integration runs the test suite and builds the documentation.
  
3. Register the new version in the Julia General registry (e.g. via the Registrator bot). `TagBot` then creates the matching Git tag and GitHub release automatically, and the documentation for the tagged version is deployed to GitHub Pages.
  

## How to Contribute {#How-to-Contribute}

See [How to Contribute](how-to-contribute.md) for the issue and pull-request workflow.
