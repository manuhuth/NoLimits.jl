using Documenter
using DocumenterVitepress
using DocumenterCitations

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using NoLimits

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style = :numeric
)

makedocs(;
    sitename = "NoLimits.jl",
    authors = "Manuel Huth, Jonas Arruda, Clemens Peiter, Roy Gusinow, Nina Schmid, Jan Hasenauer",
    modules = [NoLimits],
    checkdocs = :none,
    plugins = [bib],
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/manuhuth/NoLimits.jl",
        devbranch = "main",
        devurl = "dev"
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "installation.md",
            "Quickstart" => "quickstart.md",
            "Capabilities" => "capabilities.md",
            "NLME Methodology" => "nlme-methodology.md"
        ],
        "Model Building" => [
            "Overview" => "model-building/index.md",
            "@Model" => "model-building/model-macro.md",
            "@helpers" => "model-building/helpers.md",
            "@fixedEffects" => "model-building/fixed-effects.md",
            "@covariates" => "model-building/covariates.md",
            "@randomEffects" => "model-building/random-effects.md",
            "@preDifferentialEquation" => "model-building/pre-differential-equation.md",
            "@DifferentialEquation" => "model-building/differential-equation.md",
            "@initialDE" => "model-building/initial-de.md",
            "@formulas" => "model-building/formulas.md",
            "Function Approximators (NNs + SoftTrees)" => "model-building/universal-function-approximators.md"
        ],
        "Data Model Construction" => "data-model-construction.md",
        "Estimation" => [
            "Overview" => "estimation/index.md",
            "MLE / MAP" => "estimation/mle.md",
            "Laplace" => "estimation/laplace.md",
            "FOCEI" => "estimation/focei.md",
            "GH Quadrature" => "estimation/ghquadrature.md",
            "MCEM" => "estimation/mcem.md",
            "SAEM" => "estimation/saem.md",
            "Pooled / PooledMap" => "estimation/pooled.md",
            "MCMC" => "estimation/mcmc.md",
            "VI" => "estimation/vi.md",
            "Multistart" => "estimation/multistart.md",
            "Cross-Validation" => "estimation/cv.md",
            "Saving & Loading" => "estimation/saving-and-loading.md"
        ],
        "Uncertainty Quantification" => [
            "Overview" => "uncertainty-quantification/index.md",
            "Wald" => "uncertainty-quantification/wald.md",
            "Profile likelihood" => "uncertainty-quantification/profile-likelihood.md",
            "MCMC-based uncertainty" => "uncertainty-quantification/mcmc-based-uncertainty.md"
        ],
        "Plotting" => "plotting/index.md",
        "Tutorials" => [
            "Multi-Method Comparison" => "tutorials/mixed-effects-multiple-methods.md",
            "ODE Model with Dosing (MCEM)" => "tutorials/mixed-effects-ode-mcem.md",
            "Neural Differential Equations (SAEM)" => "tutorials/mixed-effects-nn-saem.md",
            "Soft-Tree Differential Equations (SAEM)" => "tutorials/mixed-effects-softtree-saem.md",
            "Count Outcomes: Poisson & NegativeBinomial (MCEM)" => "tutorials/mixed-effects-seizure-counts-poisson-nb-mcem.md",
            "Left-Censored Nonlinear Model (Laplace)" => "tutorials/mixed-effects-left-censored-virload50-laplace.md",
            "Fixed-Effects: MLE & MAP" => "tutorials/fixed-effects-nonlinear-mle-map.md",
            "Fixed-Effects: Variational Inference" => "tutorials/fixed-effects-vi.md",
            "Hidden & Observed Markov Models" => "tutorials/markov-models-observed-hidden-coarsed.md"
        ],
        "Developers Guide" => "developers-guide.md",
        "How to Contribute" => "how-to-contribute.md",
        "References" => "references.md",
        "API" => "api.md"
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
