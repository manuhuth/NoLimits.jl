using Documenter
using DocumenterVitepress
using DocumenterCitations

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using NoLimits
# Load the Makie extension so its plot_* method docstrings resolve for the @docs
# blocks in api.md (the drawing functions live in ext/NoLimitsMakieExt.jl).
using CairoMakie
# Makie also exports `RealVector`; the explicit import disambiguates the bare name so
# the `@docs` block and `@ref` links in api.md resolve to the NoLimits binding.
using NoLimits: RealVector
const NoLimitsMakieExt = Base.get_extension(NoLimits, :NoLimitsMakieExt)
NoLimitsMakieExt === nothing &&
    error("NoLimitsMakieExt failed to load; ensure CairoMakie.jl is available")

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style = :numeric
)

makedocs(;
    sitename = "NoLimits.jl",
    authors = "Manuel Huth, Jonas Arruda, Clemens Peiter, Roy Gusinow, Nina Schmid, Jan Hasenauer",
    modules = [NoLimits, NoLimitsMakieExt],
    checkdocs = :none,
    plugins = [bib],
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/manuhuth/NoLimits.jl",
        devbranch = "main",
        devurl = "dev",
        description = "Nonlinear mixed-effects modeling for longitudinal data in Julia: mechanistic ODEs, Markov models, machine-learning components, and frequentist and Bayesian estimation through one interface."
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "installation.md",
            "Quickstart" => "quickstart.md",
            "Coming from NONMEM / Monolix / nlmixr2" => "coming-from-nonmem.md",
        ],
        "Model Building" => [
            "Overview" => "model-building/index.md",
            "Model structure" => [
                "@Model" => "model-building/model-macro.md",
                "@helpers" => "model-building/helpers.md",
                "@formulas" => "model-building/formulas.md",
            ],
            "Parameters and covariates" => [
                "@fixedEffects" => "model-building/fixed-effects.md",
                "@covariates" => "model-building/covariates.md",
                "@randomEffects" => "model-building/random-effects.md",
                "Copula Distributions" => "model-building/copulas.md",
            ],
            "Differential equations" => [
                "@preDifferentialEquation" => "model-building/pre-differential-equation.md",
                "@DifferentialEquation" => "model-building/differential-equation.md",
                "@initialDE" => "model-building/initial-de.md",
            ],
            "Function Approximators (NNs + SoftTrees)" => "model-building/universal-function-approximators.md",
        ],
        "Data Model Construction" => "data-model-construction.md",
        "Estimation" => [
            "Overview" => "estimation/index.md",
            "Approximation-based" => [
                "Laplace" => "estimation/laplace.md",
                "FOCEI" => "estimation/focei.md",
                "GH Quadrature" => "estimation/ghquadrature.md",
            ],
            "Stochastic EM" => [
                "MCEM" => "estimation/mcem.md",
                "SAEM" => "estimation/saem.md",
                "SAEM: advanced" => "estimation/saem-advanced.md",
            ],
            "Bayesian" => [
                "MCMC" => "estimation/mcmc.md",
                "VI" => "estimation/vi.md",
            ],
            "Without random effects" => [
                "MLE / MAP" => "estimation/mle.md",
                "Pooled / PooledMap" => "estimation/pooled.md",
            ],
            "Workflow" => [
                "Multistart" => "estimation/multistart.md",
                "Cross-Validation" => "estimation/cv.md",
                "Saving & Loading" => "estimation/saving-and-loading.md",
                "Reproducibility" => "estimation/reproducibility.md",
            ],
        ],
        "Uncertainty Quantification" => [
            "Overview" => "uncertainty-quantification/index.md",
            "Wald" => "uncertainty-quantification/wald.md",
            "Profile likelihood" => "uncertainty-quantification/profile-likelihood.md",
            "MCMC-based uncertainty" => "uncertainty-quantification/mcmc-based-uncertainty.md",
        ],
        "Plotting" => "plotting/index.md",
        "Troubleshooting" => "troubleshooting.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "Mixed effects" => [
                "Multi-Method Comparison" => "tutorials/mixed-effects-multiple-methods.md",
                "ODE Model with Dosing (MCEM)" => "tutorials/mixed-effects-ode-mcem.md",
                "Count Outcomes: Poisson & NegativeBinomial (MCEM)" => "tutorials/mixed-effects-seizure-counts-poisson-nb-mcem.md",
                "Left-Censored Nonlinear Model (Laplace)" => "tutorials/mixed-effects-left-censored-virload50-laplace.md",
                "Interval-Censored Outcomes (Laplace)" => "tutorials/mixed-effects-interval-censored-binned-laplace.md",
                "Copula Random Effects (Laplace)" => "tutorials/mixed-effects-copula-random-effects-laplace.md",
                "Hidden & Observed Markov Models" => "tutorials/markov-models-observed-hidden-coarsed.md",
            ],
            "Machine-learning components" => [
                "Neural Differential Equations (SAEM)" => "tutorials/mixed-effects-nn-saem.md",
                "Soft-Tree Differential Equations (SAEM)" => "tutorials/mixed-effects-softtree-saem.md",
            ],
            "Fixed effects" => [
                "MLE & MAP" => "tutorials/fixed-effects-nonlinear-mle-map.md",
                "Variational Inference" => "tutorials/fixed-effects-vi.md",
            ],
            "Using NoLimits from R and Python" => "tutorials/r-and-python.md",
            "Building Custom Estimators" => "tutorials/building-custom-estimators.md",
        ],
        "API" => [
            "Overview" => "api.md",
            "Model Building" => "api/model-building.md",
            "Data Binding" => "api/data.md",
            "Estimation" => "api/estimation.md",
            "Uncertainty & Simulation" => "api/uncertainty.md",
            "Plotting & Diagnostics" => "api/plotting.md",
            "Distributions & Utilities" => "api/distributions.md",
        ],
        "Background & Reference" => [
            "Capabilities" => "capabilities.md",
            "NLME Methodology" => "nlme-methodology.md",
            "References" => "references.md",
            "Migrating to v0.2 (Makie)" => "migration-v0.2-makie.md",
        ],
        "Contributing" => [
            "Method-Developer API" => "method-developer-api.md",
            "Developers Guide" => "developers-guide.md",
            "How to Contribute" => "how-to-contribute.md",
        ],
    ]
)

# Must be DocumenterVitepress.deploydocs (NOT Documenter.deploydocs): it reads bases.txt
# and deploys the built site from build/<i> into the correct version subfolder. Plain
# Documenter.deploydocs would deploy build/ wholesale and leave the site under /dev/1/.
DocumenterVitepress.deploydocs(;
    repo = "github.com/manuhuth/NoLimits.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true
)
