# Precompilation of the model-independent outer-optimizer glue (#408).
#
# The outer objectives of Laplace/FOCEI, the SAEM/MCEM numeric M-step and VEM are handed to
# Optimization.jl through the type-erased `_ErasedObjective`/`_ErasedGradient`/
# `_ErasedValueGradient` wrappers (common.jl), so the `OptimizationFunction`,
# `OptimizationProblem`, `OptimizationCache`, the OptimizationOptimJL callback closure,
# `Optim.optimize` and `build_solution` have the same types for every model. The workload
# below runs that glue once on a trivial erased objective while the package image is
# generated, so the compiled code is cached in it instead of being compiled on the first
# fit. The optimizers and `solve` keywords are taken from the default-constructed methods,
# so their types are the ones `fit_model` uses with default settings. It runs only during
# precompilation and never makes precompilation fail; `strict = true` rethrows (tests).

using PrecompileTools: @compile_workload, @setup_workload

function _pc_erased_optf(adtype)
    return OptimizationFunction(
        _ErasedObjective((x, p) -> sum(abs2, x .- 0.5)), adtype;
        grad = _ErasedGradient((G, x, p) -> (G .= 2 .* (x .- 0.5); nothing)),
        fg = _ErasedValueGradient(
            (G, x, p) -> begin
                G .= 2 .* (x .- 0.5)
                return sum(abs2, x .- 0.5)
            end
        )
    )
end

# One outer solve per (optimizer, keywords, bounds) combination, built as at the fit sites:
# `OptimizationProblem(optf, z0)` or `OptimizationProblem(optf, z0; lb, ub)` with
# `Vector{Float64}` start and bounds, then `Optimization.solve(prob, opt; kwargs...)`.
function _pc_outer_solve(optf, opt, kw::NamedTuple, bounded::Bool, remake_too::Bool)
    z0 = [1.0, -1.0]
    prob = bounded ? OptimizationProblem(optf, z0; lb = [-2.0, -2.0], ub = [2.0, 2.0]) :
        OptimizationProblem(optf, z0)
    sol = Optimization.solve(prob, opt; kw...)
    if remake_too   # VEM refinement solves `remake(rprob; u0 = xs)`
        sol = Optimization.solve(remake(prob; u0 = Vector{Float64}(sol.u)), opt; kw...)
    end
    # Post-solve accesses made by the fit functions.
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
        sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    ok = sol.retcode == SciMLBase.ReturnCode.Success
    return (sol.u, sol.objective, niter, raw, ok)
end

function _pc_try(f, strict::Bool)
    try
        f()
    catch
        strict && rethrow()
    end
    return nothing
end

function _precompile_workload(; strict::Bool = false)
    # Laplace and FOCEI: default outer LBFGS, `(; maxiters, reltol)`, with and without bounds
    # (with bounds OptimizationOptimJL wraps the optimizer in `Fminbox`).
    for M in (Laplace, FOCEI)
        _pc_try(strict) do
            m = M()
            optf = _pc_erased_optf(m.adtype)
            _pc_outer_solve(optf, m.optimizer, m.optim_kwargs, false, false)
            _pc_outer_solve(optf, m.optimizer, m.optim_kwargs, true, false)
        end
    end
    # VEM: its own default LBFGS and keywords; main solve and `remake`d refinement solve.
    _pc_try(strict) do
        m = VEM()
        optf = _pc_erased_optf(m.adtype)
        _pc_outer_solve(optf, m.optimizer, m.optim_kwargs, false, true)
        _pc_outer_solve(optf, m.optimizer, m.optim_kwargs, true, true)
    end
    # SAEM and MCEM numeric M-step: plain LBFGS (first M-step, or bounded -> Fminbox) and
    # the LBFGS carrying the Cholesky preconditioner from `_saem_mstep_optimizer`.
    for M in (SAEM, MCEM)
        _pc_try(strict) do
            m = M()
            optf = _pc_erased_optf(m.adtype)
            _pc_outer_solve(optf, m.optimizer, m.optim_kwargs, false, false)
            _pc_outer_solve(optf, m.optimizer, m.optim_kwargs, true, false)
            curv = _SAEMMStepCurvature(
                [:a, :b], [2.0 0.0; 0.0 2.0], 1, Vector{Float64}[], Vector{Float64}[]
            )
            opt_pc = _saem_mstep_optimizer(m.optimizer, curv, [1.0, 1.0])
            _pc_outer_solve(optf, opt_pc, m.optim_kwargs, false, false)
        end
    end
    # Symbolic closed-form ODE detection (argument types are model-independent):
    # a diagonal, a Bateman (1-cmt oral) and a nonlinear (ineligible) system.
    _pc_try(strict) do
        @lock _CLOSED_FORM_LOCK begin
            _cf_plan([:x], Expr[:(D(x) ~ -k * x + r)], Symbol[])
            _cf_plan(
                [:depot, :center],
                Expr[:(D(depot) ~ -ka * depot), :(D(center) ~ ka * depot - (CL / V) * center)],
                Symbol[]
            )
            _cf_plan(
                [:x], Expr[:(inp(t) = a * exp(-b * t)), :(D(x) ~ r * x * (1 - x / K) + inp(t))],
                [:inp]
            )
        end
    end
    return nothing
end

precompile(_cf_plan, (Vector{Symbol}, Vector{Expr}, Vector{Symbol}))

@setup_workload begin
    @compile_workload begin
        _precompile_workload()
    end
end
