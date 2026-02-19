export compute_uq

using Random

"""
    compute_uq(res::FitResult; method, interval, vcov, re_approx, re_approx_method,
               level, pseudo_inverse, hessian_backend, fd_abs_step, fd_rel_step,
               fd_max_tries, n_draws, mcmc_warmup, mcmc_draws, constants,
               constants_re, penalty, ode_args, ode_kwargs, serialization,
               profile_method, profile_scan_width, profile_scan_tol, profile_loss_tol,
               profile_local_alg, profile_max_iter, profile_ftol_abs, profile_kwargs,
               mcmc_method, mcmc_sampler, mcmc_turing_kwargs, mcmc_adtype,
               mcmc_fit_kwargs, rng) -> UQResult

Compute uncertainty quantification for the fixed-effect parameters of a fitted model.

Three backends are supported:
- **`:wald`** – Wald intervals derived from the inverse Hessian of the objective.
- **`:chain`** – Posterior intervals from an MCMC chain (when the source fit used MCMC
  or when `method=:mcmc`).
- **`:profile`** – Profile-likelihood intervals computed by NLopt.

# Keyword Arguments
- `method::Symbol = :auto`: UQ backend. `:auto` selects `:chain` for MCMC fits and
  `:wald` otherwise; can also be `:wald`, `:profile`, or `:mcmc`.
- `interval::Symbol = :auto`: interval type. `:auto` picks a sensible default per backend.
  For Wald: `:wald` or `:normal`; for chain: `:equaltail` or `:chain`; for profile:
  `:profile`.
- `vcov::Symbol = :hessian`: covariance source for Wald UQ (`:hessian` only).
- `re_approx::Symbol = :auto`: random-effects approximation for Laplace/FOCEI Hessians.
- `re_approx_method`: fitting method used for the RE approximation, or `nothing`.
- `level::Real = 0.95`: nominal coverage level for the intervals.
- `pseudo_inverse::Bool = false`: use the Moore-Penrose pseudo-inverse for singular
  Hessians (Wald only).
- `hessian_backend::Symbol = :auto`: Hessian computation backend.
- `fd_abs_step`, `fd_rel_step`, `fd_max_tries`: finite-difference Hessian settings.
- `n_draws::Int = 2000`: number of draws to generate (for the chain and MCMC backends).
- `mcmc_warmup`, `mcmc_draws`: MCMC warm-up and draw counts (overrides defaults).
- `constants`, `constants_re`, `penalty`, `ode_args`, `ode_kwargs`, `serialization`:
  forwarded to objective evaluations (default: inherit from the source fit result).
- `profile_method`, `profile_scan_width`, `profile_scan_tol`, `profile_loss_tol`,
  `profile_local_alg`, `profile_max_iter`, `profile_ftol_abs`, `profile_kwargs`:
  NLopt profile-likelihood settings.
- `mcmc_method`, `mcmc_sampler`, `mcmc_turing_kwargs`, `mcmc_adtype`, `mcmc_fit_kwargs`:
  MCMC backend settings.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.

# Returns
A [`UQResult`](@ref) with point estimates, intervals, covariance matrices, and draws.
"""
function compute_uq(res::FitResult;
                    method::Symbol=:auto,
                    interval::Symbol=:auto,
                    vcov::Symbol=:hessian,
                    re_approx::Symbol=:auto,
                    re_approx_method::Union{Nothing, FittingMethod}=nothing,
                    level::Real=0.95,
                    pseudo_inverse::Bool=false,
                    hessian_backend::Symbol=:auto,
                    fd_abs_step::Real=1e-4,
                    fd_rel_step::Real=1e-3,
                    fd_max_tries::Int=8,
                    n_draws::Int=2000,
                    mcmc_warmup::Union{Nothing, Int}=nothing,
                    mcmc_draws::Union{Nothing, Int}=nothing,
                    constants::Union{Nothing, NamedTuple}=nothing,
                    constants_re::Union{Nothing, NamedTuple}=nothing,
                    penalty::Union{Nothing, NamedTuple}=nothing,
                    ode_args::Union{Nothing, Tuple}=nothing,
                    ode_kwargs::Union{Nothing, NamedTuple}=nothing,
                    serialization::Union{Nothing, SciMLBase.EnsembleAlgorithm}=nothing,
                    profile_method::Symbol=:LIN_EXTRAPOL,
                    profile_scan_width::Real=3.0,
                    profile_scan_tol::Real=1e-3,
                    profile_loss_tol::Real=1e-3,
                    profile_local_alg::Symbol=:LN_NELDERMEAD,
                    profile_max_iter::Int=10_000,
                    profile_ftol_abs::Real=1e-3,
                    profile_kwargs::NamedTuple=NamedTuple(),
                    mcmc_method::Union{Nothing, MCMC}=nothing,
                    mcmc_sampler=Turing.NUTS(0.75),
                    mcmc_turing_kwargs::NamedTuple=NamedTuple(),
                    mcmc_adtype=Turing.AutoForwardDiff(),
                    mcmc_fit_kwargs::NamedTuple=NamedTuple(),
                    rng::AbstractRNG=Random.default_rng())
    level_use = _validate_level(level)
    n_draws >= 1 || error("n_draws must be >= 1.")

    backend = if method == :auto
        if interval == :profile
            :profile
        else
            res.result isa MCMCResult ? :chain : :wald
        end
    else
        method
    end

    if interval != :auto
        if backend == :chain && !(interval in (:equaltail, :chain))
            error("For chain-based UQ, interval must be :auto, :equaltail, or :chain.")
        elseif backend == :wald && !(interval in (:wald, :normal, :auto))
            error("For Wald UQ, interval must be :auto, :wald, or :normal.")
        elseif backend == :profile && !(interval in (:profile, :auto))
            error("For profile UQ, interval must be :auto or :profile.")
        end
    end
    if backend == :wald
        vcov in (:hessian, :sandwich) || error("For Wald UQ, vcov must be :hessian or :sandwich.")
    end

    if backend == :chain
        return _compute_uq_chain(res;
                                 level=level_use,
                                 constants=constants,
                                 mcmc_warmup=mcmc_warmup,
                                 mcmc_draws=mcmc_draws,
                                 rng=rng)
    elseif backend == :wald
        src_method = get_method(res)
        if src_method isa MLE || src_method isa MAP
            return _compute_uq_wald_no_re(res;
                                          level=level_use,
                                          vcov=vcov,
                                          pseudo_inverse=pseudo_inverse,
                                          hessian_backend=hessian_backend,
                                          fd_abs_step=fd_abs_step,
                                          fd_rel_step=fd_rel_step,
                                          fd_max_tries=fd_max_tries,
                                          n_draws=n_draws,
                                          constants=constants,
                                          penalty=penalty,
                                          ode_args=ode_args,
                                          ode_kwargs=ode_kwargs,
                                          serialization=serialization,
                                          rng=rng)
        elseif src_method isa Laplace || src_method isa LaplaceMAP || src_method isa FOCEI || src_method isa FOCEIMAP ||
               src_method isa MCEM || src_method isa SAEM
            return _compute_uq_wald_re(res;
                                       level=level_use,
                                       vcov=vcov,
                                       re_approx=re_approx,
                                       re_approx_method=re_approx_method,
                                       pseudo_inverse=pseudo_inverse,
                                       hessian_backend=hessian_backend,
                                       fd_abs_step=fd_abs_step,
                                       fd_rel_step=fd_rel_step,
                                       fd_max_tries=fd_max_tries,
                                       n_draws=n_draws,
                                       constants=constants,
                                       constants_re=constants_re,
                                       penalty=penalty,
                                       ode_args=ode_args,
                                       ode_kwargs=ode_kwargs,
                                       serialization=serialization,
                                       rng=rng)
        else
            error("Wald UQ is currently supported for MLE, MAP, Laplace, LaplaceMAP, FOCEI, FOCEIMAP, MCEM, and SAEM fit results.")
        end
    elseif backend == :profile
        return _compute_uq_profile(res;
                                   level=level_use,
                                   constants=constants,
                                   constants_re=constants_re,
                                   penalty=penalty,
                                   ode_args=ode_args,
                                   ode_kwargs=ode_kwargs,
                                   serialization=serialization,
                                   profile_method=profile_method,
                                   profile_scan_width=profile_scan_width,
                                   profile_scan_tol=profile_scan_tol,
                                   profile_loss_tol=profile_loss_tol,
                                   profile_local_alg=profile_local_alg,
                                   profile_max_iter=profile_max_iter,
                                   profile_ftol_abs=profile_ftol_abs,
                                   profile_kwargs=profile_kwargs,
                                   rng=rng)
    elseif backend == :mcmc_refit
        return _compute_uq_mcmc_refit(res;
                                      level=level_use,
                                      constants=constants,
                                      constants_re=constants_re,
                                      ode_args=ode_args,
                                      ode_kwargs=ode_kwargs,
                                      serialization=serialization,
                                      mcmc_warmup=mcmc_warmup,
                                      mcmc_draws=mcmc_draws,
                                      mcmc_method=mcmc_method,
                                      mcmc_sampler=mcmc_sampler,
                                      mcmc_turing_kwargs=mcmc_turing_kwargs,
                                      mcmc_adtype=mcmc_adtype,
                                      mcmc_fit_kwargs=mcmc_fit_kwargs,
                                      rng=rng)
    else
        error("Unsupported UQ method $(backend). Use :auto, :wald, :chain, :profile, or :mcmc_refit.")
    end
end
