using Documenter

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using NoLimits

makedocs(;
    sitename="NoLimits.jl",
    modules=[NoLimits],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true", collapselevel=1, sidebar_sitename=false, size_threshold=400*1024),
    repo=Remotes.GitHub("manuhuth", "NoLimits.jl"),
    checkdocs=:none,
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Capabilities" => "capabilities.md",
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
            "Function Approximators (NNs + SoftTrees)" => "model-building/universal-function-approximators.md",
        ],
        "Data Model Construction" => "data-model-construction.md",
        "Estimation" => [
            "Overview" => "estimation/index.md",
            "Laplace" => "estimation/laplace.md",
            "Laplace MAP" => "estimation/laplace-map.md",
            "MCEM" => "estimation/mcem.md",
            "SAEM" => "estimation/saem.md",
            "MCMC" => "estimation/mcmc.md",
            "VI" => "estimation/vi.md",
            "MLE" => "estimation/mle.md",
            "MAP" => "estimation/mle-map.md",
            "Multistart" => "estimation/multistart.md",
        ],
        "Uncertainty Quantification" => [
            "Overview" => "uncertainty-quantification/index.md",
            "Wald" => "uncertainty-quantification/wald.md",
            "Profile likelihood" => "uncertainty-quantification/profile-likelihood.md",
            "MCMC-based uncertainty" => "uncertainty-quantification/mcmc-based-uncertainty.md",
        ],
        "Plotting" => "plotting/index.md",
        "Tutorials" => [
            "Mixed-Effects Tutorial 1: Nonlinear Random-Effects Model Across Multiple Estimation Methods" => "tutorials/mixed-effects-multiple-methods.md",
            "Mixed-Effects Tutorial 2: ODE Model with Dosing Events (MCEM)" => "tutorials/mixed-effects-ode-mcem.md",
            "Mixed-Effects Tutorial 3: Neural Differential-Equation Components (SAEM)" => "tutorials/mixed-effects-nn-saem.md",
            "Mixed-Effects Tutorial 4: SoftTree Differential-Equation Components (SAEM)" => "tutorials/mixed-effects-softtree-saem.md",
            "Mixed-Effects Tutorial 5: Seizure Counts with Poisson and NegativeBinomial Outcomes (MCEM)" => "tutorials/mixed-effects-seizure-counts-poisson-nb-mcem.md",
            "Mixed-Effects Tutorial 6: Left-Censored Nonlinear Model (Laplace)" => "tutorials/mixed-effects-left-censored-virload50-laplace.md",
            "Mixed-Effects Tutorial 7: Variational Inference (VI)" => "tutorials/mixed-effects-vi.md",
            "Fixed-Effects Tutorial 1: Nonlinear Longitudinal Model (MLE + MAP)" => "tutorials/fixed-effects-nonlinear-mle-map.md",
            "Fixed-Effects Tutorial 2: Variational Inference (VI)" => "tutorials/fixed-effects-vi.md",
        ],
        "How to Contribute" => "how-to-contribute.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/manuhuth/NoLimits.jl",
    devbranch = "main",
)
