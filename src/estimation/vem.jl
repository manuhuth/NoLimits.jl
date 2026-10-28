# Variational Expectation Maximization (VEM) for nonlinear mixed-effects models.
#
# Tarek & Afonso (2026), "Fitting Large Nonlinear Mixed Effects Models Using Variational
# Expectation Maximization" (arXiv:2604.26160): the marginal likelihood is replaced by the
# evidence lower bound
#
#     Σ_i ELBO_i(θ, κ_i),   ELBO_i = E_{q(u; κ_i)}[log p(y_i, u | θ)] + H(q(·; κ_i)),
#
# with one Gaussian variational distribution q(u; κ_i) = N(μ_i, L_i L_i') per RE batch,
# and (θ, κ) are optimized JOINTLY (L-BFGS by default). The expectation is a sample average over
# draws ξ_ij ~ N(0, I) fixed once per fit (u_ij = μ_i + L_i ξ_ij; "deterministic VEM",
# the paper's default: a deterministic objective that admits quasi-Newton steps and line
# searches). The variational family lives in the standardized RE space u of the prior
# transport b = T_θ(u) (`build_re_measure_from_batch`: Gaussian REs b = m_θ + C_θ u, other
# supports through their quantile transport with its log-Jacobian), so bounded-support
# random effects are handled exactly and κ stays well scaled while θ moves.
# The final objective is an importance-sampling estimate of -log p(y | θ̂) with a
# multivariate-t proposal around each fitted q (the ELBO itself is reported in the notes).

export VEM, PlanarFlowFamily, get_variational_distributions

"""
    PlanarFlowFamily(K::Integer; base = :dense, seed = 0)

Variational family for [`VEM`](@ref) (`family = PlanarFlowFamily(K)`): `K` planar layers
F(z) = z + û tanh(wᵀz + b) on the base draw of a Gaussian family, so each batch's q is
the law of m₀ + S₀(μ + L F(ξ)), ξ ~ N(0, I) (`base = :dense` or `:diagonal` for L).
Planar layers can represent skewed random-effect posteriors (few binary or count
observations per subject). û is the invertible reparameterization of u (wᵀû > -1).
Layers start at the identity (û = 0) with random nonzero w (|w| ≈ 1), drawn from the fit `rng`
(`seed = 0`) or from `Xoshiro(seed)`. Flow fits use the Monte Carlo expectation with
max(64, 8 n_b) draws per batch (`n_samples = :auto`) and a joint ForwardDiff gradient;
the refinement and the importance-sampling objective use a t proposal around the
moment-matched q.
"""
struct PlanarFlowFamily
    K::Int
    base::Symbol
    seed::Int
end

function PlanarFlowFamily(K::Integer; base = :dense, seed::Integer = 0)
    K >= 1 || error("PlanarFlowFamily: K must be ≥ 1. Got: $K")
    base = _as_symbol(base)
    base in (:diagonal, :dense) ||
        error("PlanarFlowFamily: base must be :diagonal or :dense. Got: $base")
    return PlanarFlowFamily(Int(K), base, Int(seed))
end

"""
    VEM(; optimizer = nothing, optim_kwargs = nothing, adtype = AutoForwardDiff(), lb = nothing,
        ub = nothing, ignore_model_bounds = false, n_samples = :auto, expectation = :auto,
        family = :dense, init = :laplace, n_is = 1000, is_df = 5.0, refine = :auto,
        refine_nodes = 64, ebe_optimizer, ebe_optim_kwargs, ebe_adtype, ebe_grad_tol,
        ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds, ebe_multistart_sampling)

Variational Expectation Maximization (Tarek & Afonso 2026). Maximizes the evidence lower
bound jointly over the fixed effects and one Gaussian variational distribution per
random-effect batch with an Optimization.jl optimizer (L-BFGS by default), using
`n_samples` fixed standard-normal draws per batch (sample-average approximation of the
ELBO). Scales to models with many population parameters (e.g. neural networks) and random
effects: no inner mode finding or Hessian per objective evaluation, unlike Laplace/FOCEI.

# Keyword Arguments
- `optimizer`: Optimization.jl-compatible optimizer for the joint ELBO fit and the `refine`
  step. `nothing` selects `LBFGS(m = 20, linesearch = BackTracking(maxstep = 30.0))`. The optimizer works
  on a diagonally preconditioned offset of the fixed effects and receives the analytic
  value and gradient in one call (`fg`).
- `optim_kwargs = nothing`: keyword arguments forwarded to `Optimization.solve` (both
  stages). `nothing` uses `(; maxiters = 1000)`, plus `g_abstol = 1e-5`, `reltol = 1e-8` and
  `allow_f_increases = false` for an Optim.jl optimizer (the sample-average objective's
  gradient norm stalls around 1e-2 near the optimum, so in practice the relative objective
  change stops the fit). A user-supplied NamedTuple replaces these defaults.
- `adtype = AutoForwardDiff()`: automatic differentiation of the joint objective. The default
  differentiates per batch in forward mode (cheap for few fixed effects). Any other ADTypes
  backend - typically `AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
  function_annotation = Enzyme.Duplicated)` with Enzyme loaded (without the `Duplicated`
  annotation Enzyme rejects the objective, which captures mutable caches) - differentiates
  each batch's ELBO over the fixed effects and that batch's variational parameters in one
  DifferentiationInterface pass, batches threaded like the default: reverse mode for large
  networks (paper: reverse mode is what makes VEM scale). Rule of thumb: reverse mode pays off from
  roughly 100 fixed-effect coordinates (FFNN with 116 weights: 69 s vs 258 s with forward
  mode; 27 coordinates with 13-dimensional random effects: 418 s vs 295 s). Requires
  Gaussian random effects (centered variational space, evaluated through the
  Enzyme-compatible batch log-joint). The `refine` step always uses ForwardDiff.
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing` to use the
  model-declared bounds (as for [`Laplace`](@ref)). Variational parameters are unbounded,
  so optimizers that need finite bounds on every coordinate (BlackBoxOptim methods,
  `ParticleSwarm`) are rejected.
- `ignore_model_bounds::Bool = false`: ignore the bounds declared in `@fixedEffects`
  (explicit `lb`/`ub` still apply).
- `n_samples = :auto`: draws per batch in the sample-average ELBO (fixed per fit). `:auto`
  uses max(16, 4 n_b) for an n_b-dimensional batch: 16 draws biased the 13-dimensional
  network-weight random effects of a test model (objective -59.5 vs -93.4 with 64 draws).
- `expectation = :auto`: how E_q is computed. `:quadrature` uses a Smolyak Gauss-Hermite
  rule in the base space (7 nodes for scalar random effects, level 5 / 4 for 2 / 3
  dimensions), deterministic and far more accurate than a few draws for small batches;
  `:mc` uses `n_samples` fixed draws (the paper's sample-average approximation). `:auto`
  picks quadrature for batches with ≤ 3 random effects.
- `family = :dense`: `:dense` (full lower-triangular scale) or `:diagonal` (mean-field)
  Gaussian family in the standardized random-effect space, or a
  [`PlanarFlowFamily`](@ref) (normalizing planar flow on the Gaussian base draw). Mean-field q cannot represent
  posterior correlations between random effects and biases correlated RE covariances
  (M03 anchor: Ω₁₂ = -0.08 vs 0.15); use it only for very high-dimensional REs.
- `init::Symbol = :laplace`: initialize each q at the Laplace approximation of the batch
  posterior at the starting θ (`:prior` starts from the prior, μ = 0, L = I).
- `n_is::Int = 1000`, `is_df = 5.0`: importance-sampling draws and t degrees of freedom for
  the final marginal-likelihood estimate (the reported objective) of batches with more than
  3 random effects; batches with 1-3 random effects use adaptive Gauss-Hermite quadrature
  around q (21 / 9² / 7³ nodes, as `refine`). `n_is = 0` reports -ELBO instead (at θ̂ and
  the fitted q).
- `refine = :auto`, `refine_nodes = 64`: after the joint VEM optimization, maximize the
  marginal likelihood over the fixed effects with fixed base nodes: a tensor Gauss-Hermite
  rule for batches with 1-3 random effects (adaptive quadrature; 21 / 9² / 7³ nodes),
  `refine_nodes` multivariate-t draws (ν = `is_df`) otherwise. At every θ the rule of each
  batch is centered at the mode of its integrand and scaled by the curvature there (Newton
  from the fitted q's mean; q where the curvature is not negative definite), and the
  reported objective uses the rule at θ̂. This removes the
  variational bias of the Gaussian family (binary toenail data: VEM ω = 3.52, MLE 4.00)
  and is consistent as the number of nodes grows. `:auto` refines when the model has at most
  50 free fixed-effect coordinates and at most 3 random effects per batch (each refinement
  iteration costs one forward-mode pass over the fixed effects; network models skip it).
  The refinement uses the same `optimizer`, `optim_kwargs` and bounds as the joint fit,
  followed by a damped Newton finish (at most 50 fixed-effect coordinates; Hessian by
  central differences of the gradient) within the remaining `maxiters`, whose
  `g_abstol` / `reltol` test decides the refinement's convergence.
- `ebe_*`: options of the final empirical-Bayes modes (as for `MCEM`).

The `penalty` and `extra_objective` keywords of [`fit_model`](@ref) act as for `Laplace`:
both are functions of the natural-scale fixed effects, added (with ForwardDiff gradients
for every `adtype`, as Laplace does) to the negative ELBO, to the `refine` objective and to
the reported objective (`get_result(res).notes.elbo` stays the plain ELBO). Likelihood
weights of the `DataModel` multiply the ELBO term of each random-effect batch, and its
marginal-likelihood term in `refine` and in the reported objective. Random-effect levels
fixed through `constants_re` enter as for `Laplace`: with the prior density of the fixed
values, and a batch without free levels (no q) with its log-likelihood at the fixed values.

For mini-batching (stochastic VEM) wrap the method in [`MiniBatch`](@ref); the penalty and
`extra_objective` are then added once per iteration, not scaled by the number of batches.

Randomness (the fixed Monte Carlo draws, multivariate-t refinement nodes, importance-sampling
draws, start repair and EBE multistart) comes only from the `rng` keyword of
[`fit_model`](@ref), as for `SAEM` and `MCEM`: the same seeded `rng` gives identical fits.
The `serialization` keyword of `fit_model` (default `EnsembleThreads()`) runs the per-batch
work on threads, for every `adtype`; threaded and serial (`EnsembleSerial()`) fits are
bitwise equal.

Fits support Wald UQ ([`compute_uq`](@ref)), and [`sample_random_effects`](@ref) and
conditional cross-validation draw from the fitted variational distributions: for a
[`PlanarFlowFamily`](@ref) from its moment-matched Gaussian, and for batches of weight zero
(which do not enter the objective) from the Laplace approximation at ``\\hat\\theta``.
[`get_variational_distributions`](@ref) returns these distributions per individual
(`get_result(res).notes.variational_params` holds them in random-effect batch order).
Discrete random effects are not supported.
"""
struct VEM{O, K, A, L, U, EB, US} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
    n_samples::Union{Int, Symbol}
    expectation::Symbol
    family::Symbol
    flow::Int
    flow_seed::Int
    init::Symbol
    n_is::Int
    is_df::Float64
    refine::Symbol
    refine_nodes::Int
    ebe::EB
    update_schedule::US
end

_update_schedule(m::VEM) = m.update_schedule

function VEM(;
        optimizer = nothing,
        optim_kwargs = nothing,
        adtype = Optimization.AutoForwardDiff(),
        lb = nothing,
        ub = nothing,
        ignore_model_bounds::Bool = false,
        n_samples::Union{Int, Symbol} = :auto,
        expectation = :auto,
        family = :dense,
        init = :laplace,
        n_is::Int = 1000,
        is_df::Real = 5.0,
        refine = :auto,
        refine_nodes::Int = 64,
        ebe_optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        ebe_optim_kwargs = NamedTuple(),
        ebe_adtype = Optimization.AutoForwardDiff(),
        ebe_grad_tol = :auto,
        ebe_multistart_n = 50,
        ebe_multistart_k = 1,
        ebe_multistart_max_rounds = 5,
        ebe_multistart_sampling = :lhs
    )
    flow, flow_seed = 0, 0
    if family isa PlanarFlowFamily
        flow, flow_seed, family = family.K, family.seed, family.base
    end
    family = _as_symbol(family)
    init = _as_symbol(init)
    family in (:diagonal, :dense) ||
        error("VEM: family must be :diagonal or :dense. Got: $family")
    init in (:laplace, :prior) || error("VEM: init must be :laplace or :prior. Got: $init")
    n_samples = n_samples isa Symbol ? _as_symbol(n_samples) : n_samples
    (n_samples === :auto || (n_samples isa Int && n_samples >= 1)) ||
        error("VEM: n_samples must be :auto or an integer ≥ 1. Got: $n_samples")
    expectation = _as_symbol(expectation)
    expectation in (:auto, :mc, :quadrature) ||
        error("VEM: expectation must be :auto, :mc or :quadrature. Got: $expectation")
    n_is >= 0 || error("VEM: n_is must be ≥ 0. Got: $n_is")
    is_df > 0 || error("VEM: is_df must be > 0. Got: $is_df")
    refine_nodes >= 1 || error("VEM: refine_nodes must be ≥ 1. Got: $refine_nodes")
    # memory 20; BackTracking step capped at 30 in the preconditioned coordinates (|.|∞): the
    # outer default's maxstep = 1 shortens early steps of the joint problem, while an
    # uncapped first step sent a CT-Markov fit (M14) to rates whose likelihood evaluation
    # did not return for 20 minutes
    optimizer === nothing &&
        (optimizer = OptimizationOptimJL.LBFGS(m = 20, linesearch = LineSearches.BackTracking(maxstep = 30.0)))
    optim_kwargs === nothing && (
        optim_kwargs = optimizer isa Optim.AbstractOptimizer ?
            (; maxiters = 1000, g_abstol = 1.0e-5, reltol = 1.0e-8, allow_f_increases = false) :
            (; maxiters = 1000)
    )
    ebe = EBEOptions(
        ebe_optimizer, _as_namedtuple(ebe_optim_kwargs), ebe_adtype, ebe_grad_tol,
        ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds,
        ebe_multistart_sampling
    )
    return VEM(
        optimizer, _as_namedtuple(optim_kwargs), adtype, lb, ub, ignore_model_bounds,
        n_samples, expectation, family, flow, flow_seed, init, n_is, Float64(is_df),
        refine === :auto ? :auto : refine === true ? :on : refine === false ? :off :
            error("VEM: refine must be :auto, true or false. Got: $refine"),
        refine_nodes, ebe, :all
    )
end

# ---------------------------------------------------------------------------------------
# variational parameters of one batch (dimension n):
#   κ = [μ (n); log diag(L) (n); strictly-lower entries of L, column-major (dense only)]
# ---------------------------------------------------------------------------------------
_vem_nkappa(n::Int, family::Symbol) = family === :dense ? 2n + n * (n - 1) ÷ 2 : 2n
# planar-flow parameters of one batch, after the Gaussian part: [u (n); w (n); b] per layer
_vem_nflow(n::Int, K::Int) = K * (2n + 1)

@inline _vem_softplus(x) = x > 30 ? x : log1p(exp(x))

# K planar layers on the base draw ξ (θ = κ, flow block after offset `off`):
# z ← z + û tanh(wᵀz + b), û = u + (m(wᵀu) - wᵀu) w / |w|², m(x) = softplus(x) - 1, so
# wᵀû = m(wᵀu) > -1 and log|det ∂z'/∂z| = log1p((1 - tanh²) m(wᵀu)). Plain loops (Enzyme).
function _vem_flow(θ::AbstractVector, off::Int, ξ::AbstractVector, n::Int, K::Int)
    T = promote_type(eltype(θ), eltype(ξ))
    z = Vector{T}(undef, n)
    for i in 1:n
        z[i] = ξ[i]
    end
    ld = zero(T)
    for k in 1:K
        o = off + (k - 1) * (2n + 1)
        wu = zero(T)
        ww = zero(T)
        a = zero(T)
        for i in 1:n
            wu += θ[o + n + i] * θ[o + i]
            ww += θ[o + n + i]^2
            a += θ[o + n + i] * z[i]
        end
        m = _vem_softplus(wu) - 1
        c = (m - wu) / ww
        h = tanh(a + θ[o + 2n + 1])
        for i in 1:n
            z[i] += (θ[o + i] + c * θ[o + n + i]) * h
        end
        ld += log1p((1 - h^2) * m)
    end
    return z, ld
end

function _vem_scale(κ::AbstractVector, n::Int, family::Symbol)
    T = eltype(κ)
    L = zeros(T, n, n)
    for k in 1:n
        L[k, k] = exp(κ[n + k])
    end
    if family === :dense
        idx = 2n
        for j in 1:n, i in (j + 1):n
            idx += 1
            L[i, j] = κ[idx]
        end
    end
    return L
end

# u-space draw j of a batch: μ + L ξ_j (diagonal: μ + s .* ξ_j)
@inline function _vem_draw(μ, L, ξj, family::Symbol)
    return family === :dense ? μ .+ L * ξj : μ .+ diag(L) .* ξj
end

# Expectation rule of one batch under q: nodes ξ (n, R) in the base space and weights w
# (R): Monte Carlo (w = 1/M) or a Smolyak Gauss-Hermite rule (signed weights summing to 1).
struct _VEMDraws
    ξ::Matrix{Float64}
    w::Vector{Float64}
    # fixed affine preconditioner from the initialization: q's draws are
    # m0 + S0 (μ̃ + L̃ ξ) with (μ̃, L̃) optimized from (0, I). The raw variational
    # parameters of badly scaled REs (M18: intercept sd 0.5, slope sd 1e-3) made the joint
    # L-BFGS ill-conditioned (stuck at a = 1.6 vs 5); S0 is fixed, so - unlike the
    # θ-dependent standardization - it cannot trap a collapsing RE variance.
    m0::Vector{Float64}
    S0::Matrix{Float64}
    logdetS0::Float64
    # automatic centering (as for SAEM/MCEM): for each pair `fe + η` (`fe * exp(η)`: log fe)
    # the q mean of the paired RE coordinates moves by -(fe - fe0), so changing fe does not
    # require all batches' q means to follow along the a-η ridge (M18 crawled along it for
    # 1000 iterations). Entries (fe index, RE indices, fe0 or log fe0, log scale).
    center::Vector{Tuple{Int, Vector{Int}, Float64, Bool}}
    K::Int   # planar-flow layers (0: Gaussian family)
end

# effective mean and scale of q: m0 + S0 μ̃, S0 L̃
function _vem_effective(κ::AbstractVector, dr::_VEMDraws, n::Int, family::Symbol, θu)
    μ = dr.m0 .+ dr.S0 * κ[1:n]
    if !isempty(dr.center)
        T = promote_type(eltype(μ), eltype(θu))
        μ = Vector{T}(μ)
        for (j, idx, c0, lg) in dr.center
            δ = (lg ? log(θu[j]) : θu[j]) - c0
            for i in idx
                μ[i] -= δ
            end
        end
    end
    L = dr.S0 * _vem_scale(κ, n, family)
    return μ, L
end

function _vem_draws(
        n::Int, method, rng::AbstractRNG, κinit::Vector{Float64}, family::Symbol,
        center::Vector{Tuple{Int, Vector{Int}, Float64, Bool}} = Tuple{Int, Vector{Int}, Float64, Bool}[]
    )
    m0 = κinit[1:n]
    S0 = _vem_scale(κinit, n, family)
    logdetS0 = sum(log(abs(S0[k, k])) for k in 1:n)
    K = method.flow
    mode = method.expectation === :auto ? (n <= 3 && K == 0 ? :quadrature : :mc) :
        method.expectation
    if mode === :quadrature
        g = get_sparse_grid(n, n == 1 ? 7 : n == 2 ? 5 : 4)
        return _VEMDraws(
            Matrix{Float64}(g.nodes), Float64.(g.signs) .* exp.(g.logweights),
            m0, S0, logdetS0, center, K
        )
    end
    M = method.n_samples === :auto ? (K > 0 ? max(64, 8n) : max(16, 4n)) : method.n_samples
    return _VEMDraws(randn(rng, n, M), fill(1.0 / M, M), m0, S0, logdetS0, center, K)
end

# Log integrand in the variational space. Gaussian random effects are CENTERED: q lives
# on b itself, value = logint(L⁻¹(b - μ)) - log|det L| (the standardized u-space of the
# prior transport, used for other supports, traps VEM at a collapsed RE variance: with a
# small L the likelihood barely depends on u and the gradient that would grow L vanishes -
# M03 from a perturbed start converged to Ω₁₁ = 0, objective 992 vs 702). Numerical errors
# at trial points count as -Inf for the line search; anything else is rethrown.
function _vem_logjoint(v, base, dm::DataModel, info::REBatchInfo, θ_re, const_cache, ll_cache::_LLCache)
    T = promote_type(eltype(v), eltype(base), eltype(θ_re))
    try
        if base isa GaussianRE
            L = base.L
            u = L \ (v .- base.μ)
            return convert(T, _transport_logintegrand(u, base, dm, info, θ_re, const_cache, ll_cache)) -
                sum(x -> log(abs(x)), diag(L))
        end
        return convert(T, _transport_logintegrand(v, base, dm, info, θ_re, const_cache, ll_cache))
    catch err
        _is_numeric_error(err) || rethrow()
        return convert(T, -Inf)
    end
end

# RE measure of a batch, `nothing` at a numerically degenerate θ
function _vem_measure(info::REBatchInfo, θu, const_cache, dm::DataModel, ll_cache::_LLCache)
    try
        return build_re_measure_from_batch(info, θu, const_cache, dm, ll_cache)
    catch err
        _is_numeric_error(err) || rethrow()
        return nothing
    end
end

# Sample-average ELBO of one batch (exact constants: log p(y, u) = logint - n/2 log 2π and
# H(q) = n/2 (1 + log 2π) + log|det L|).
function _vem_batch_elbo(
        θu::ComponentArray, κ::AbstractVector, dr::_VEMDraws, family::Symbol,
        dm::DataModel, info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    ξ, w = dr.ξ, dr.w
    n = size(ξ, 1)
    T = promote_type(eltype(θu), eltype(κ))
    base = _vem_measure(info, θu, const_cache, dm, ll_cache)
    base === nothing && return T(-Inf)
    θ_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))
    μ, L = _vem_effective(κ, dr, n, family, θu)
    acc = zero(T)
    M = size(ξ, 2)
    nG = _vem_nkappa(n, family)
    for j in 1:M
        if dr.K == 0
            u = _vem_draw(μ, L, view(ξ, :, j), family)
            v = _vem_logjoint(u, base, dm, info, θ_re, const_cache, ll_cache)
        else
            z, ld = _vem_flow(κ, nG, view(ξ, :, j), n, dr.K)
            v = _vem_logjoint(_vem_draw(μ, L, z, family), base, dm, info, θ_re, const_cache, ll_cache) + ld
        end
        isfinite(v) || return T(-Inf)
        acc += w[j] * v
    end
    logdet = T(dr.logdetS0)
    for k in 1:n
        logdet += κ[n + k]
    end
    return acc + logdet + T(n) / 2
end

# Batch ELBO for the reverse-mode path (`adtype` other than ForwardDiff): Gaussian REs in the
# centered space, joint density from the Enzyme-compatible `_laplace_logf_batch_impl` (no
# try/catch, no transport measure). Same value as `_vem_batch_elbo` for Gaussian REs:
# E_q[log p(y, b | θ)] + n/2 (1 + log 2π) + log|det L|, without the prior of levels fixed by
# `constants_re` (added once per batch by the θ-only terms, as on the ForwardDiff path).
function _vem_batch_elbo_rev(
        θu::ComponentArray, κ::AbstractVector, dr::_VEMDraws, family::Symbol,
        dm::DataModel, info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    ξ, w = dr.ξ, dr.w
    n = size(ξ, 1)
    μ, L = _vem_effective(κ, dr, n, family, θu)
    acc = zero(promote_type(eltype(θu), eltype(κ)))
    nG = _vem_nkappa(n, family)
    for j in 1:size(ξ, 2)
        if dr.K == 0
            v = _vem_draw(μ, L, view(ξ, :, j), family)
            acc += w[j] * _laplace_logf_batch_impl(
                dm, info, θu, v, const_cache, ll_cache; const_prior = false
            )
        else
            z, ld = _vem_flow(κ, nG, view(ξ, :, j), n, dr.K)
            v = _vem_draw(μ, L, z, family)
            lf = _laplace_logf_batch_impl(
                dm, info, θu, v, const_cache, ll_cache; const_prior = false
            )
            acc += w[j] * (lf + ld)
        end
    end
    logdet = dr.logdetS0
    for k in 1:n
        logdet += κ[n + k]
    end
    return acc + logdet + n / 2 * (1 + log(2π))
end

# value-only reverse-path ELBO: numerical failures at line-search trial points count as -Inf
function _vem_batch_elbo_rev_value(args...)
    try
        return _vem_batch_elbo_rev(args...)
    catch err
        _is_numeric_error(err) || rethrow()
        return -Inf
    end
end

# Value and gradient of one batch's ELBO. θ-part: one ForwardDiff pass over the fixed
# effects with κ held at Float64. κ-part by the chain rule from per-draw u-gradients of the
# integrand (ForwardDiff over n_b only): ∂/∂μ = mean ∇u, ∂/∂L = mean ∇u ξ' (+ 1 on log-diag
# from the entropy). Cost (p + n_b) dual directions instead of p + dim(κ).
function _vem_batch_grad(
        θu_of, xθ::Vector{Float64}, κ::Vector{Float64}, dr::_VEMDraws, family::Symbol,
        dm::DataModel, info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    DR = ForwardDiff.DiffResults
    fθ = z -> _vem_batch_elbo(θu_of(z), κ, dr, family, dm, info, const_cache, ll_cache)
    res = DR.GradientResult(xθ)
    ForwardDiff.gradient!(res, fθ, xθ)
    v = DR.value(res)
    gθ = copy(DR.gradient(res))
    ξ, w = dr.ξ, dr.w
    n, M = size(ξ)
    gκ = zeros(Float64, length(κ))
    isfinite(v) || return v, gθ, gκ
    θu = θu_of(xθ)
    base = build_re_measure_from_batch(info, θu, const_cache, dm, ll_cache)
    θ_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))
    f = u -> _vem_logjoint(u, base, dm, info, θ_re, const_cache, ll_cache)
    μ, L = _vem_effective(κ, dr, n, family, θu)
    L̃ = _vem_scale(κ, n, family)
    gμ = zeros(n)
    gL = zeros(n, n)
    for j in 1:M
        ξj = view(ξ, :, j)
        u = _vem_draw(μ, L, ξj, family)
        gu = ForwardDiff.gradient(f, u)
        gμ .+= w[j] .* gu
        gL .+= w[j] .* (gu * ξj')
    end
    # chain rule through the fixed preconditioner: v = m0 + S0 (μ̃ + L̃ ξ)
    gκ[1:n] .= dr.S0' * gμ
    gL = dr.S0' * gL
    for k in 1:n
        gκ[n + k] = gL[k, k] * L̃[k, k] + 1.0
    end
    if family === :dense
        idx = 2n
        for jj in 1:n, i in (jj + 1):n
            idx += 1
            gκ[idx] = gL[i, jj]
        end
    end
    return v, gθ, gκ
end

# Joint ForwardDiff over (θ, κ): cheaper when both fit in few dual chunks (small models).
function _vem_batch_grad_joint(
        θu_of, xθ::Vector{Float64}, κ::Vector{Float64}, dr::_VEMDraws, family::Symbol,
        dm::DataModel, info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    DR = ForwardDiff.DiffResults
    p = length(xθ)
    z0 = vcat(xθ, κ)
    fz = z -> _vem_batch_elbo(
        θu_of(view(z, 1:p)), view(z, (p + 1):length(z)), dr, family, dm, info, const_cache,
        ll_cache
    )
    res = DR.GradientResult(z0)
    ForwardDiff.gradient!(res, fz, z0)
    g = DR.gradient(res)
    return DR.value(res), g[1:p], g[(p + 1):end]
end

# Dual-chunk count (ForwardDiff's default chunk is ≤ 12): joint needs ⌈(p+v)/12⌉ passes over
# the draws; split needs ⌈p/12⌉ passes plus per-draw u-gradients worth ⌈n_b/12⌉ passes.
_vem_split_cheaper(p::Int, v::Int, n_b::Int) = cld(p + v, 12) > cld(p, 12) + cld(n_b, 12)

# IS-refinement nodes of one batch: base-space nodes x (n, K) and log weights such that
# log p(y_b | θ) ≈ logsumexp_k [lw_k + logint(μ + L x_k)] (+ the u-space Jacobian log|L|).
# 1-3 REs: tensor Gauss-Hermite (lw = log w + |x|²/2; adaptive quadrature around q);
# otherwise multivariate-t draws (lw = -log K - log t_ν(x) - n/2 log 2π; `refine = true` only:
# optimizing θ over a fixed draw set fits its noise - 64 draws moved a 2-D Gaussian fit).
function _vem_refine_nodes(n::Int, K::Int, ν::Float64, rng::AbstractRNG)
    if n <= 3
        # tensor Gauss-Hermite (positive weights): 21 (n = 1), 9² (n = 2), 7³ (n = 3) nodes;
        # 3 REs: the Gaussian-q bias of a skewed binary posterior (Ω₃₃ 12% low) is refined away
        g = get_sparse_grid(1, n == 1 ? 21 : n == 2 ? 9 : 7)
        x1, w1 = vec(g.nodes), g.logweights
        idx = collect(Iterators.product(ntuple(_ -> eachindex(x1), n)...))
        x = reduce(hcat, [[x1[i] for i in I] for I in vec(idx)])
        lw = [sum(w1[i] for i in I) for I in vec(idx)] .+ vec(sum(abs2, x; dims = 1)) ./ 2
        return Matrix{Float64}(x), lw
    end
    t = Distributions.MvTDist(ν, zeros(n), Distributions.PDMats.PDMat(Matrix(1.0I, n, n)))
    x = rand(rng, t, K)
    lw = [-log(K) - Distributions.logpdf(t, view(x, :, k)) - n / 2 * log(2π) for k in 1:K]
    return x, lw
end

function _vem_refine_batch_ll(
        θu::ComponentArray, μ::AbstractVector, L::AbstractMatrix, x::Matrix{Float64},
        lw::Vector{Float64}, dm::DataModel, info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    T = eltype(θu)
    base = _vem_measure(info, θu, const_cache, dm, ll_cache)
    base === nothing && return T(-Inf)
    θ_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))
    K = size(x, 2)
    v = Vector{T}(undef, K)
    for k in 1:K
        u = μ .+ L * view(x, :, k)
        v[k] = _vem_logjoint(u, base, dm, info, θ_re, const_cache, ll_cache) + lw[k]
    end
    m = maximum(v)
    isfinite(m) || return T(-Inf)
    return m + log(sum(exp.(v .- m))) + logabsdet(L)[1]
end

# Laplace initialization of one batch: damped Newton for the mode of the u-space
# integrand, q = N(û, (-H)⁻¹) (diagonal family: marginal sds), sds capped at the prior's 1.
function _vem_init_batch(
        θu::ComponentArray, n::Int, family::Symbol, dm::DataModel, info::REBatchInfo,
        const_cache, ll_cache::_LLCache
    )
    base = _vem_measure(info, θu, const_cache, dm, ll_cache)
    base === nothing && return zeros(Float64, _vem_nkappa(n, family))
    θ_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))
    # mode and curvature in the standardized u-space (well conditioned), then mapped to the
    # variational space (centered b for Gaussian REs)
    f = u -> try
        _transport_logintegrand(u, base, dm, info, θ_re, const_cache, ll_cache)
    catch err
        _is_numeric_error(err) || rethrow()
        -Inf
    end
    û, Σu = _vem_u_laplace(f, n)
    if base isa GaussianRE
        L = Matrix{Float64}(base.L)
        m = Vector{Float64}(base.μ) .+ L * û
        Σ = Symmetric(L * Σu * L')
    else
        m, Σ = û, Symmetric(Σu)
    end
    return _vem_kappa_from(m, Σ, n, family)
end

# q = the RE prior at θ (centered space for Gaussian REs, N(0, I) in u-space otherwise)
function _vem_init_prior(
        θu::ComponentArray, n::Int, family::Symbol, dm::DataModel, info::REBatchInfo,
        const_cache, ll_cache::_LLCache
    )
    base = _vem_measure(info, θu, const_cache, dm, ll_cache)
    base isa GaussianRE || return _vem_kappa_from(zeros(n), Matrix(1.0I, n, n), n, family)
    L = Matrix{Float64}(base.L)
    return _vem_kappa_from(Vector{Float64}(base.μ), L * L', n, family)
end

# damped Newton for the mode of f, curvature covariance (-H)⁻¹ capped at the prior's I
function _vem_u_laplace(f, n::Int)
    u, Σ = _vem_newton_mode(f, zeros(Float64, n))
    Σ === nothing && return u, Matrix(1.0I, n, n)
    # posterior no wider than the prior in u-space
    for k in 1:n
        Σ[k, k] > 1.0 && (r = 1 / sqrt(Σ[k, k]); Σ[k, :] .*= r; Σ[:, k] .*= r)
    end
    return u, Σ
end

# damped Newton for the mode of f from u0: (mode, (-H)⁻¹), or (point, nothing) when f is
# not finite at u0 or -H is not positive definite at the end point
function _vem_newton_mode(f, u0::Vector{Float64})
    n = length(u0)
    u = copy(u0)
    fu = f(u)
    isfinite(fu) || return u, nothing
    λ = 1.0e-3
    for _ in 1:100
        g = ForwardDiff.gradient(f, u)
        H = ForwardDiff.hessian(f, u)
        (all(isfinite, g) && all(isfinite, H)) || break
        C = cholesky(Symmetric(-H + λ * I); check = false)
        if !issuccess(C)
            λ *= 10
            λ > 1.0e8 && break
            continue
        end
        Δ = C \ g
        unew = u .+ Δ
        fnew = f(unew)
        if isfinite(fnew) && fnew >= fu
            done = maximum(abs, Δ) < 1.0e-8 || fnew - fu < 1.0e-12
            u, fu = unew, fnew
            λ = max(λ / 10, 1.0e-10)
            done && break
        else
            λ *= 10
            λ > 1.0e8 && break
        end
    end
    H = try
        ForwardDiff.hessian(f, u)
    catch err
        _is_numeric_error(err) || rethrow()
        return u, nothing
    end
    all(isfinite, H) || return u, nothing
    C = cholesky(Symmetric(-H); check = false)
    issuccess(C) || return u, nothing
    return u, Matrix(inv(C))
end

# Center and scale of the adaptive rule of one batch at θu (Float64): the mode of the log
# integrand in the variational space and the Cholesky factor of (-H)⁻¹ there (Newton from
# the fitted q's mean), or q itself where that curvature is not negative definite.
function _vem_adapt_rule(
        θu::ComponentArray, μq::Vector{Float64}, Lq::Matrix{Float64}, dm::DataModel,
        info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    μ, L, _ = _vem_adapt_rule_cov(θu, μq, Lq, dm, info, const_cache, ll_cache)
    return μ, L
end

# `_vem_adapt_rule` and (-H)⁻¹ at the mode (nothing for the fallback q)
function _vem_adapt_rule_cov(
        θu::ComponentArray, μq::Vector{Float64}, Lq::Matrix{Float64}, dm::DataModel,
        info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    base = _vem_measure(info, θu, const_cache, dm, ll_cache)
    base === nothing && return μq, Lq, nothing
    θ_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))
    f = v -> _vem_logjoint(v, base, dm, info, θ_re, const_cache, ll_cache)
    v, Σ = _vem_newton_mode(f, μq)
    Σ === nothing && return μq, Lq, nothing
    C = cholesky(Symmetric(Σ); check = false)
    issuccess(C) || return μq, Lq, nothing
    return v, Matrix(C.L), Σ
end

# The rule of `_vem_adapt_rule` as a function of θu of any element type (the refinement
# objective under ForwardDiff). The Float64 rule at θv = value(θu) alone drops dμ/dθ and
# dL/dθ from the gradient while the value depends on them (M05: AD gradient
# (0.04, -0.07, 0.04, ...) vs central differences (0.23, 0.91, -0.11, ...) at the end of the
# refinement). The first derivatives of the mode follow from the implicit function theorem,
# dμ/dθ = (-H)⁻¹ ∂g/∂θ with the Float64 (-H)⁻¹ of the Newton mode and the gradient g of the
# log integrand at the mode in dual arithmetic (one nested-dual Hessian per batch and
# evaluation instead of two); L is the Cholesky factor of (-H)⁻¹ at that dual mode. Values
# are reset to the Float64 rule (μ, L, Σ) = `_vem_adapt_rule_cov` at value(θu), so that the
# objective's value does not depend on the element type.
function _vem_refine_rule(
        θu::ComponentArray, μ::Vector{Float64}, L::Matrix{Float64}, Σ, dm::DataModel,
        info::REBatchInfo, const_cache, ll_cache::_LLCache
    )
    T = eltype(θu)
    # the fallback rule (q itself) does not depend on θ
    (T === Float64 || Σ === nothing) && return μ, L
    base = _vem_measure(info, θu, const_cache, dm, ll_cache)
    base === nothing && return μ, L
    θ_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))
    f = v -> _vem_logjoint(v, base, dm, info, θ_re, const_cache, ll_cache)
    g = ForwardDiff.gradient(f, Vector{T}(μ))
    all(isfinite, g) || return μ, L
    μd = μ .+ Σ * (g .- ForwardDiff.value.(g))
    Hd = ForwardDiff.hessian(f, μd)
    all(isfinite, Hd) || return μd, L
    Cd = cholesky(Symmetric(-Hd); check = false)
    issuccess(Cd) || return μd, L
    Cs = cholesky(Symmetric(Cd \ Matrix{T}(I, length(μ), length(μ))); check = false)
    issuccess(Cs) || return μd, L
    Ld = Matrix(Cs.L)
    return μd, L .+ (Ld .- ForwardDiff.value.(Ld))
end

# Damped Newton finish of the refinement from x (in place) within [lb, ub]: curvature by
# forward differences of the gradient `grad!(G, x)`, eigenvalues clipped at 1e-8 of the
# largest, Armijo backtracking, coordinates at a bound with the gradient pointing outward
# held fixed. The curvature is kept while full steps reduce |g|∞ at least fourfold, and
# renewed when a kept one fails its line search (a new one costs p gradients; M14: 5 steps
# with central differences, 2p gradients each, took 33 of 50 s of the fit). Converged when
# |g|∞ ≤ gtol on the free coordinates, or when the predicted or the achieved decrease of a
# Newton step is at most rtol max(|f|, 1).
# Returns (f(x), converged, iterations).
function _vem_newton_finish!(
        f, grad!, x::Vector{Float64}, lb, ub, nmax::Int, gtol::Float64, rtol::Float64
    )
    p = length(x)
    fx = f(x)
    isfinite(fx) || return fx, false, 0
    g, gp = zeros(p), zeros(p)
    # curvature of the last step: free set, eigen decomposition, clipped eigenvalues
    E, λ, free_E = nothing, Float64[], Int[]
    gmax_prev, keep = Inf, false
    for it in 1:nmax
        grad!(g, x)
        all(isfinite, g) || return fx, false, it - 1
        free = findall(j -> !((x[j] <= lb[j] && g[j] > 0) || (x[j] >= ub[j] && g[j] < 0)), 1:p)
        gmax = maximum(j -> abs(g[j]), free; init = 0.0)
        gmax <= gtol && return fx, true, it - 1
        reused = keep && free == free_E && gmax <= gmax_prev / 4
        if !reused
            H = zeros(length(free), length(free))
            for (c, j) in enumerate(free)
                h = 1.0e-6 * max(1.0, abs(x[j]))
                xj = copy(x)
                xj[j] = x[j] + h
                grad!(gp, xj)
                @views H[:, c] .= (gp[free] .- g[free]) ./ h
            end
            all(isfinite, H) || return fx, false, it - 1
            E = eigen(Symmetric((H .+ H') ./ 2))
            λmax = maximum(abs, E.values)
            λmax > 0 || return fx, false, it - 1
            λ = max.(E.values, 1.0e-8 * λmax)
            free_E = free
        end
        gmax_prev = gmax
        Δ = -(E.vectors * ((E.vectors' * g[free]) ./ λ))
        pred = -dot(g[free], Δ)
        tol = rtol * max(abs(fx), 1.0)
        α, xn, fn = 1.0, x, Inf
        for _ in 1:40
            xn = copy(x)
            xn[free] .= clamp.(x[free] .+ α .* Δ, lb[free], ub[free])
            fn = f(xn)
            isfinite(fn) && fn <= fx - 1.0e-4 * α * pred && break
            # converged: backtracking along a step of predicted decrease ≤ tol only costs
            # evaluations (M19: near-singular curvature)
            pred / 2 <= tol && return fx, true, it - 1
            α /= 2
        end
        if !(isfinite(fn) && fn <= fx - 1.0e-4 * α * pred)
            # a kept curvature failed: retry from x with a new one
            reused && (keep = false; gmax_prev = Inf; continue)
            return fx, pred / 2 <= tol, it - 1
        end
        keep = α == 1.0
        df = fx - fn
        x .= xn
        fx = fn
        (pred / 2 <= tol || df <= tol) && return fx, true, it
    end
    return fx, false, nmax
end

function _vem_kappa_from(m::AbstractVector, Σ::AbstractMatrix, n::Int, family::Symbol)
    κ = zeros(Float64, _vem_nkappa(n, family))
    κ[1:n] .= m
    if family === :dense
        C = cholesky(Symmetric(Matrix(Σ)) + 1.0e-12 * I; check = false)
        if issuccess(C)
            Lm = Matrix(C.L)
            for k in 1:n
                κ[n + k] = log(Lm[k, k])
            end
            idx = 2n
            for j in 1:n, i in (j + 1):n
                idx += 1
                κ[idx] = Lm[i, j]
            end
        end
    else
        for k in 1:n
            κ[n + k] = log(sqrt(max(Σ[k, k], 1.0e-16)))
        end
    end
    return κ
end

# log p̂(y_batch | θ) by importance sampling from a multivariate t around the fitted q.
# Mean and scale factor of q: (μ, L) of the Gaussian family; for a planar flow the
# moment-matched Gaussian over the fit's draws.
function _vem_q_moments(κ::Vector{Float64}, dr::_VEMDraws, n::Int, family::Symbol, θu)
    μ, L = _vem_effective(κ, dr, n, family, θu)
    family === :dense || (L = Matrix(Diagonal(diag(L))))
    dr.K == 0 && return μ, L
    M = size(dr.ξ, 2)
    B = zeros(Float64, n, M)
    for j in 1:M
        z, _ = _vem_flow(κ, _vem_nkappa(n, family), view(dr.ξ, :, j), n, dr.K)
        B[:, j] .= _vem_draw(μ, L, z, family)
    end
    m = vec(sum(B; dims = 2)) ./ M
    D = B .- m
    C = cholesky(Symmetric(D * D' ./ max(M - 1, 1)) + 1.0e-10 * I; check = false)
    issuccess(C) || return μ, L
    return m, Matrix(C.L)
end

function _vem_is_loglik(
        θu::ComponentArray, μ::Vector{Float64}, L::Matrix{Float64}, n::Int, K::Int,
        ν::Float64, rng::AbstractRNG, dm::DataModel, info::REBatchInfo, const_cache,
        ll_cache::_LLCache
    )
    base = _vem_measure(info, θu, const_cache, dm, ll_cache)
    base === nothing && return -Inf
    θ_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))
    # PDMat from the factor: L L' formed explicitly is not numerically PD for badly scaled REs
    q = Distributions.MvTDist(ν, μ, Distributions.PDMats.PDMat(LinearAlgebra.Cholesky(LowerTriangular(Matrix(L)))))
    lw = Vector{Float64}(undef, K)
    c = -n / 2 * log(2π)
    for k in 1:K
        u = rand(rng, q)
        v = _vem_logjoint(u, base, dm, info, θ_re, const_cache, ll_cache)
        lw[k] = isfinite(v) ? v + c - Distributions.logpdf(q, u) : -Inf
    end
    lmax = maximum(lw)
    isfinite(lmax) || return -Inf
    return lmax + log(sum(exp(w - lmax) for w in lw)) - log(K)
end

# f(k, ll_cache) over batches 1:n; threaded with one cache per task when `caches` has more
# than one entry. Callers write results by k and reduce in index order.
function _vem_foreach(f, n::Int, caches::Vector)
    if length(caches) == 1
        for k in 1:n
            f(k, caches[1])
        end
    else
        _greedy_foreach((k, c) -> f(k, caches[c]), n, length(caches))
    end
    return nothing
end

# whether a batch has random-effect levels fixed by `constants_re`
function _vem_has_const_levels(dm::DataModel, info::REBatchInfo, const_cache)
    isempty(const_cache.is_const) && return false
    ids = get_ind_level_ids(get_laplace_cache(get_re_group_info(dm)))
    return any(
        const_cache.is_const[ri][id] for i in get_inds(info)
            for ri in eachindex(const_cache.is_const) for id in ids[i][ri]
    )
end

# penalty and extra_objective on the natural scale (both optional), as for Laplace
function _vem_penalty(θu, penalty::NamedTuple, extra_objective)
    T = eltype(θu)
    v = convert(T, _penalty_value(θu, penalty))
    extra_objective === nothing && return v
    return v + convert(T, extra_objective(θu))
end

# The variational parameters are unbounded, so optimizers that need a finite box on every
# coordinate cannot run the joint problem.
function _vem_check_optimizer(opt)
    (
        nameof(parentmodule(typeof(opt))) === :OptimizationBBO ||
            opt isa OptimizationOptimJL.ParticleSwarm
    ) || return nothing
    return error(
        "VEM: $(nameof(typeof(opt))) needs finite bounds on every coordinate, but VEM " *
            "optimizes the unbounded variational parameters jointly with the fixed effects. " *
            "Use a gradient-based optimizer (e.g. the default LBFGS)."
    )
end

_sol_iterations(sol) = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
    sol.stats.iterations : missing

function _vem_theta_u(
        xθ::AbstractVector, θ_const_t::ComponentArray, axs_full, axs_free, free_names,
        inv_transform
    )
    T = eltype(xθ)
    θt = ComponentArray(T.(collect(θ_const_t)), axs_full)
    θf = ComponentArray(Vector(xθ), axs_free)
    for name in free_names
        setproperty!(θt, name, getproperty(θf, name))
    end
    return inv_transform(θt)
end

function _fit_model(
        dm::DataModel, method::VEM, args...;
        constants::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = _default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_eb_modes::Bool = true,
        store_data_model::Bool = true
    )
    fit_kwargs = (
        constants = constants, constants_re = constants_re, penalty = penalty,
        ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization,
        rng = rng, theta_0_untransformed = theta_0_untransformed,
        store_eb_modes = store_eb_modes, store_data_model = store_data_model,
        extra_objective = extra_objective,
    )
    _vem_check_optimizer(method.optimizer)
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) &&
        error("VEM requires random effects. Use MLE/MAP for fixed-effects models.")
    _ghq_validate_re_distributions(dm, "VEM")
    fe = get_fixed(get_model(dm))
    priors = get_priors(fe)
    if any(!(getfield(priors, k) isa Priorless) for k in keys(priors))
        @info "VEM ignores fixed-effect priors (marginal maximum likelihood)."
    end
    fixed_names = get_names(fe)
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    free_names = [n for n in fixed_names if !(n in keys(constants))]
    isempty(free_names) && error("VEM requires at least one free fixed effect.")
    θ0_u = theta_0_untransformed === nothing ? get_θ0_untransformed(fe) :
        theta_0_untransformed
    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)

    constants_re = _normalize_constants_re(dm, constants_re)
    const_cache = _build_constants_cache(dm, constants_re)
    _, batch_infos, _ = _build_re_batch_infos(dm, constants_re)
    ll_caches = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true
    )
    caches = ll_caches isa Vector ? ll_caches : [ll_caches]
    ll_cache = caches[1]
    # start repair shared with SAEM/MCEM: no finite log-joint for any initial RE draw
    θ0_r = _em_repair_start(dm, batch_infos, θ0_u, const_cache, ll_cache, rng, re_names, "VEM")
    θ0_u = θ0_r
    θ_const_u = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)
    θt_free = ComponentArray(
        NamedTuple{Tuple(free_names)}(Tuple(getproperty(θ_const_t, n) for n in free_names))
    )
    axs_free = getaxes(θt_free)
    axs_full = getaxes(θ_const_t)
    # type-stable free -> full assembly (index merge, no runtime-Symbol property access: the
    # reverse-mode path needs Enzyme to infer every type)
    θ_const_t_vec = Vector{Float64}(ComponentArrays.getdata(θ_const_t))
    idx_full = ComponentArray(collect(1:length(θ_const_t)), axs_full)
    free_idx = reduce(vcat, [vec(collect(getproperty(idx_full, n))) for n in free_names])
    θu_of(xθ) = inv_transform(_merge_free_into_full(θ_const_t_vec, free_idx, xθ, axs_full))

    # ---- variational layout, fixed draws, initialization ----
    family = method.family
    p = length(θt_free)
    nb = [get_n_b(info) for info in batch_infos]
    keep = findall(>(0), nb)
    # DataModel weights multiply each batch's ELBO term (as Laplace weights its
    # log-likelihood term); zero-weight batches are skipped
    wk = [_batch_weight(dm, get_inds(batch_infos[bi])) for bi in keep]
    # levels fixed by `constants_re` enter as for Laplace/GHQuadrature: a batch without
    # free levels contributes log p(y_b, b_fixed | θ) (it has no q), other batches the
    # prior density of their fixed levels, log p(b_fixed | θ)
    fixed_b = [
        bi for bi in eachindex(batch_infos)
            if !iszero(_batch_weight(dm, get_inds(batch_infos[bi]))) &&
            (nb[bi] == 0 || _vem_has_const_levels(dm, batch_infos[bi], const_cache))
    ]
    ranges = UnitRange{Int}[]
    off = p
    for bi in keep
        k = _vem_nkappa(nb[bi], family) + _vem_nflow(nb[bi], method.flow)
        push!(ranges, (off + 1):(off + k))
        off += k
    end
    x0 = zeros(Float64, off)
    lb, ub, use_bounds, θ0_init = _resolve_optim_bounds(
        fe, free_names, θt_free, method.optimizer, method.lb, method.ub, constants;
        ignore_model_bounds = method.ignore_model_bounds, method_label = "VEM"
    )
    x0[1:p] .= collect(Float64, θ0_init)
    θu0 = θu_of(view(x0, 1:p))
    ξs = _VEMDraws[]
    cpairs = _saem_detect_centering_pairs(dm, free_names)
    idx_u = ComponentArray(collect(1:length(θu0)), getaxes(θu0))
    nk = length(keep)
    κinits = Vector{Vector{Float64}}(undef, nk)
    _vem_foreach(nk, caches) do k, lc
        bi = keep[k]
        κinits[k] = method.init === :laplace ?
            _vem_init_batch(θu0, nb[bi], family, dm, batch_infos[bi], const_cache, lc) :
            _vem_init_prior(θu0, nb[bi], family, dm, batch_infos[bi], const_cache, lc)
    end
    # draws serially, in batch order: the rng stream does not depend on the threading
    for (k, bi) in enumerate(keep)
        # only where q lives on b (Gaussian batches): in the standardized u-space the ridge
        # shift of a paired coordinate would be δ / sd(θ)
        gauss = _vem_measure(batch_infos[bi], θu0, const_cache, dm, ll_cache) isa GaussianRE
        center = Tuple{Int, Vector{Int}, Float64, Bool}[
            (
                    Int(getproperty(idx_u, cp.fe)), _cs_batch_indices(batch_infos[bi], re_names, cp.re, cp.k),
                    cp.mode === :mul ? log(Float64(getproperty(θu0, cp.fe))) :
                    Float64(getproperty(θu0, cp.fe)), cp.mode === :mul,
                ) for cp in cpairs if gauss
        ]
        push!(ξs, _vem_draws(nb[bi], method, rng, κinits[k], family, center))
        # preconditioned start: μ̃ = 0, L̃ = I  (κ zeros)
    end
    # planar layers start at the identity: û = 0 for u = w log(e - 1) / |w|², b = 0
    if method.flow > 0
        rngf = method.flow_seed == 0 ? rng : Random.Xoshiro(method.flow_seed)
        for (k, bi) in enumerate(keep)
            n = nb[bi]
            o = first(ranges[k]) - 1 + _vem_nkappa(n, family)
            for l in 1:method.flow
                # |w| ≈ 1: with |w| = 0.1 the û reparameterization was ill-conditioned
                # (1000 L-BFGS iterations, Ω off by 0.3 on a 3-RE binary model)
                w = randn(rngf, n) ./ sqrt(n)
                ol = o + (l - 1) * (2n + 1)
                x0[(ol + 1):(ol + n)] .= w .* (log(ℯ - 1) / sum(abs2, w))
                x0[(ol + n + 1):(ol + 2n)] .= w
            end
        end
    end

    # ---- θ-only terms: penalty + extra_objective - Σ_b w_b (fixed-level terms of b) ----
    # penalty and extra_objective act on the natural-scale θu, as for Laplace; these terms
    # are added once per evaluation and not scaled under MiniBatch. Their gradients use
    # ForwardDiff for every adtype, as in Laplace (`obj_grad` there).
    has_pen = !isempty(keys(penalty)) || extra_objective !== nothing
    nf = length(fixed_b)
    has_glob = has_pen || nf > 0
    pen_of(θu) = _vem_penalty(θu, penalty, extra_objective)
    function fixed_ll(θu, j, lc)
        info = batch_infos[fixed_b[j]]
        w = _batch_weight(dm, get_inds(info))
        get_n_b(info) == 0 &&
            return w * _laplace_logf_batch(dm, info, θu, Float64[], const_cache, lc)
        lp = try
            _const_re_prior_logf(dm, info, _symmetrize_psd_params(θu, fe), const_cache, lc)
        catch err
            _is_numeric_error(err) || rethrow()
            convert(eltype(θu), -Inf)
        end
        return w * lp
    end
    fvals = Vector{Float64}(undef, nf)
    fgs = Vector{Vector{Float64}}(undef, nf)
    DRg = ForwardDiff.DiffResults
    # value of the θ-only terms at xθ; with a vector `gθ`, also adds their gradient to it
    function glob(xθ::Vector{Float64}, gθ)
        total = 0.0
        if gθ === nothing
            θu = θu_of(xθ)
            has_pen && (total += pen_of(θu))
            _vem_foreach(nf, caches) do j, lc
                fvals[j] = fixed_ll(θu, j, lc)
            end
        else
            if has_pen
                r = DRg.GradientResult(xθ)
                ForwardDiff.gradient!(r, z -> pen_of(θu_of(z)), xθ)
                total += DRg.value(r)
                gθ .+= DRg.gradient(r)
            end
            _vem_foreach(nf, caches) do j, lc
                r = DRg.GradientResult(xθ)
                ForwardDiff.gradient!(r, z -> fixed_ll(θu_of(z), j, lc), xθ)
                fvals[j] = DRg.value(r)
                fgs[j] = copy(DRg.gradient(r))
            end
        end
        for j in 1:nf
            total -= fvals[j]
            gθ === nothing || (gθ .-= fgs[j])
        end
        return total
    end

    # ---- objective: -Σ w_i ELBO_i (+ θ-only terms), value and gradient per batch ----
    # per-batch results by index, reduced in batch order: threaded == serial bitwise.
    # Under MiniBatch: -(K/|B|) Σ_{i∈B} w_i ELBO_i, the unbiased estimate of the full
    # objective and its (θ, κ) gradient (zero for the κ blocks outside B). Kept `nothing`
    # outside solve.
    all_k = [k for k in eachindex(keep) if !iszero(wk[k])]
    mb = _minibatch_state(_update_schedule(method), length(keep), rng)
    mb_ref = Ref{Union{Nothing, typeof(mb)}}(nothing)
    vals = Vector{Float64}(undef, nk)
    gθs = Vector{Vector{Float64}}(undef, nk)
    gκs = Vector{Vector{Float64}}(undef, nk)

    # ForwardDiff (default): per batch over (θ, κ_i). Other backends (reverse mode): one
    # DifferentiationInterface pass per batch over z = [xθ; κ_k] with the batch index a
    # constant context; one function per task cache, since the Duplicated function shadow
    # is mutable state of the preparation, and one preparation per κ-block length (batches
    # differ in size, e.g. nested or partly fixed levels).
    use_fd = nameof(typeof(method.adtype)) === :AutoForwardDiff
    if !use_fd
        for bi in keep
            build_re_measure_from_batch(batch_infos[bi], θu0, const_cache, dm, ll_cache) isa
                GaussianRE || error("VEM: adtype = $(nameof(typeof(method.adtype))) requires Gaussian (Normal/MvNormal) random effects; use the default AutoForwardDiff().")
        end
    end
    elbo_b = use_fd ? _vem_batch_elbo : _vem_batch_elbo_rev_value
    fbs = [
        (z, k) -> _vem_batch_elbo_rev(
                θu_of(view(z, 1:p)), view(z, (p + 1):length(z)), ξs[k], family, dm,
                batch_infos[keep[k]], const_cache, lc
            ) for lc in caches
    ]
    DI = Optimization.OptimizationBase.DifferentiationInterface
    preps = use_fd ? nothing : [
            Dict(
                begin
                    kl = findfirst(r -> length(r) == l, ranges)
                    l => DI.prepare_gradient(fb, method.adtype, vcat(x0[1:p], x0[ranges[kl]]), DI.Constant(kl))
                end for l in unique(length.(ranges))
            ) for fb in fbs
        ]
    tids = collect(eachindex(caches))
    function fg!(F, G, x)
        st = _minibatch_current!(mb_ref[])
        c = _minibatch_scale(st)
        ks = st === nothing ? all_k : st.selected
        nsel = length(ks)
        if G === nothing
            θu = θu_of(collect(view(x, 1:p)))
            _vem_foreach(nsel, caches) do j, lc
                k = ks[j]
                vals[j] = elbo_b(
                    θu, view(x, ranges[k]), ξs[k], family, dm, batch_infos[keep[k]],
                    const_cache, lc
                )
            end
            total = 0.0
            for j in 1:nsel
                w = c * wk[ks[j]]
                iszero(w) && continue
                isfinite(vals[j]) || return Inf
                total -= w * vals[j]
            end
            has_glob && (total += glob(collect(view(x, 1:p)), nothing))
            return isfinite(total) ? total : Inf
        end
        if use_fd
            xθ = collect(view(x, 1:p))
            _vem_foreach(nsel, caches) do j, lc
                k = ks[j]
                r, ξ = ranges[k], ξs[k]
                grad = ξ.K == 0 && _vem_split_cheaper(p, length(r), size(ξ.ξ, 1)) ?
                    _vem_batch_grad : _vem_batch_grad_joint
                vals[j], gθs[j], gκs[j] = grad(
                    θu_of, xθ, Vector{Float64}(x[r]), ξ, family, dm, batch_infos[keep[k]],
                    const_cache, lc
                )
            end
        else
            _vem_foreach(nsel, tids) do j, t
                k = ks[j]
                z = vcat(x[1:p], x[ranges[k]])
                g = similar(z)
                vals[j] = try
                    first(DI.value_and_gradient!(fbs[t], g, preps[t][length(ranges[k])], method.adtype, z, DI.Constant(k)))
                catch err
                    _is_numeric_error(err) || rethrow()
                    -Inf
                end
                gθs[j] = g[1:p]
                gκs[j] = g[(p + 1):end]
            end
        end
        fill!(G, 0.0)
        total = 0.0
        for j in 1:nsel
            w = c * wk[ks[j]]
            iszero(w) && continue
            v, gθ, gκ = vals[j], gθs[j], gκs[j]
            if !(isfinite(v) && all(isfinite, gθ) && all(isfinite, gκ))
                fill!(G, NaN)
                return Inf
            end
            total -= w * v
            @views G[1:p] .-= w .* gθ
            @views G[ranges[ks[j]]] .-= w .* gκ
        end
        if has_glob
            gg = zeros(Float64, p)
            total += glob(collect(view(x, 1:p)), gg)
            if !(isfinite(total) && all(isfinite, gg))
                fill!(G, NaN)
                return Inf
            end
            @views G[1:p] .+= gg
        end
        return F === nothing ? nothing : total
    end

    # Diagonal preconditioning of the fixed effects (p <= 50): optimize y with
    # x[1:p] = x0[1:p] + s .* y[1:p], s_j = 1/sqrt(curvature_jj) from a finite difference of
    # the gradient at the start (the variational block is already O(1) through the fixed
    # preconditioner of each q). Badly scaled fixed effects (M18: slope 1e-3 x times up to
    # 5e3) otherwise stalled L-BFGS (5000 iterations, |g| = 3e3, a = 1.5 vs 5).
    # Where that curvature is not positive, the scale comes from the BHHH information
    # Σ_b w_b (∂ELBO_b/∂θ_j)² (the per-batch score outer product) instead of s_j = 1: unit scale
    # was ~100x the posterior-sd scale of the other coordinates, and on a Beta RE (M05)
    # L-BFGS then drove log α₁ onto the flat asymptote α₁ → ∞ (Beta mass at 1).
    s = ones(Float64, p)
    # Under MiniBatch an Optimisers rule takes fixed-size steps in these units; posterior-sd
    # units would make them tiny for well-determined effects, so s stays 1 there.
    if p <= 50 && !(mb !== nothing && method.optimizer isa Optimisers.AbstractRule)
        G0 = zeros(Float64, off)
        f0 = fg!(0.0, G0, x0)
        if isfinite(f0) && all(isfinite, G0)
            # BHHH = Σ_b w_b s_b², s_b the unweighted score of batch b (its ELBO plus the
            # prior of its fixed levels, or the log-likelihood of a batch without q): a
            # weight-w batch counts as w copies, as in the objective. The penalty and
            # extra_objective have no per-batch score; their own curvature is added.
            sc = Vector{Union{Nothing, Vector{Float64}}}(nothing, length(batch_infos))
            for (j, k) in enumerate(all_k)
                sc[keep[k]] = copy(gθs[j])
            end
            for j in 1:nf
                g = fgs[j] ./ _batch_weight(dm, get_inds(batch_infos[fixed_b[j]]))
                bi = fixed_b[j]
                sc[bi] === nothing ? (sc[bi] = g) : (sc[bi] .+= g)
            end
            bhhh = zeros(Float64, p)
            for bi in eachindex(sc)
                sc[bi] === nothing && continue
                bhhh .+= _batch_weight(dm, get_inds(batch_infos[bi])) .* abs2.(sc[bi])
            end
            gpen = z -> has_pen ? ForwardDiff.gradient(y -> pen_of(θu_of(y)), z) : zeros(p)
            gp0 = nothing
            Gj = similar(G0)
            for j in 1:p
                h = 1.0e-4 * max(1.0, abs(x0[j]))
                xj = copy(x0)
                xj[j] += h
                fj = fg!(0.0, Gj, xj)
                Hjj = (Gj[j] - G0[j]) / h
                if !(isfinite(fj) && isfinite(Hjj) && Hjj > 0)
                    gp0 === nothing && (gp0 = gpen(x0[1:p]))
                    hp = (gpen(xj[1:p])[j] - gp0[j]) / h
                    Hjj = bhhh[j] + (isfinite(hp) ? max(hp, 0.0) : 0.0)
                end
                (isfinite(Hjj) && Hjj > 0) && (s[j] = clamp(1 / sqrt(Hjj), 1.0e-6, 1.0e2))
            end
        end
    end
    xmap(y) = (x = copy(y); @views x[1:p] .= x0[1:p] .+ s .* y[1:p]; x)
    function fgy!(F, G, y)
        x = xmap(y)
        v = fg!(F, G, x)
        G === nothing || (@views G[1:p] .*= s)
        return v
    end
    y0 = copy(x0)
    y0[1:p] .= 0.0
    fy(y, _) = fgy!(0.0, nothing, y)
    # `fg` makes Optim's line search and Optimisers rules use the fused value + gradient
    optf = OptimizationFunction(
        fy, method.adtype;
        grad = (G, y, _) -> (fgy!(nothing, G, y); nothing),
        fg = (G, y, _) -> fgy!(0.0, G, y)
    )
    # bounds act on the fixed effects only, mapped into y; κ stays unbounded
    lb_y = vcat((lb .- x0[1:p]) ./ s, fill(-Inf, off - p))
    ub_y = vcat((ub .- x0[1:p]) ./ s, fill(Inf, off - p))
    opt_use, use_bounds_y = _bounded_optimizer(method.optimizer, use_bounds, lb_y, ub_y)
    prob = use_bounds_y ? OptimizationProblem(optf, y0; lb = lb_y, ub = ub_y) :
        OptimizationProblem(optf, y0)
    t0 = time()
    mb_ref[] = mb
    sol = Optimization.solve(
        prob, opt_use;
        _minibatch_solve_kwargs(method.optim_kwargs, opt_use, mb, Returns(nothing))...
    )
    mb_ref[] = nothing
    x̂ = xmap(sol.u)
    xθ_hat = Vector{Float64}(x̂[1:p])
    θ_hat_u = θu_of(xθ_hat)
    refine_iters = 0
    refine_ok = true
    refine_retcode = nothing
    refined = false
    # a full-data pass per iteration: off by default under MiniBatch
    do_refine = method.refine === :on || (
        method.refine === :auto && mb === nothing && p <= 50 &&
            maximum(nb[keep]; init = 0) <= 3
    )
    if do_refine
        # ---- refinement: maximize the adaptive quadrature / IS marginal likelihood ----
        # (removes the variational bias). The rule of each batch is re-centered at every θ
        # (mode and Laplace scale, Newton from q's mean): nodes frozen at the VEM fit lose
        # accuracy as θ moves, and the optimizer exploited that error (3 correlated binary
        # REs: objective 20 below the MLE, a variance 3.7 vs 1.52).
        qs = [
            begin
                    μq, Lq = _vem_q_moments(Vector{Float64}(x̂[ranges[k]]), ξs[k], nb[bi], family, θ_hat_u)
                    (μq, Lq, _vem_refine_nodes(nb[bi], method.refine_nodes, method.is_df, rng))
                end for (k, bi) in enumerate(keep)
        ]
        DRr = ForwardDiff.DiffResults
        # Float64 rule of each batch at the last value(xθ): ForwardDiff's chunks (p > 12) and
        # the evaluations of f and its gradient at one point share it
        RuleT = Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}, Union{Nothing, Matrix{Float64}}}
        rules = Vector{Union{Nothing, RuleT}}(nothing, nk)
        fr = xθ -> begin
            θu = θu_of(xθ)
            xv = eltype(xθ) === Float64 ? xθ : ForwardDiff.value.(xθ)
            θv = eltype(xθ) === Float64 ? θu : θu_of(xv)
            T = eltype(θu)
            lls = Vector{T}(undef, nk)
            _vem_foreach(nk, caches) do k, lc
                iszero(wk[k]) && (lls[k] = zero(T); return)
                μq, Lq, (xn, lw) = qs[k]
                info = batch_infos[keep[k]]
                rule = rules[k]
                if rule === nothing || rule[1] != xv
                    rule = (Vector{Float64}(xv), _vem_adapt_rule_cov(θv, μq, Lq, dm, info, const_cache, lc)...)
                    rules[k] = rule
                end
                μ, L = _vem_refine_rule(θu, rule[2], rule[3], rule[4], dm, info, const_cache, lc)
                lls[k] = _vem_refine_batch_ll(θu, μ, L, xn, lw, dm, info, const_cache, lc)
            end
            flls = Vector{T}(undef, nf)
            _vem_foreach(nf, caches) do j, lc
                flls[j] = fixed_ll(θu, j, lc)
            end
            (all(isfinite, lls) && all(isfinite, flls)) || return convert(eltype(xθ), Inf)
            tot = zero(eltype(xθ))
            for k in 1:nk
                tot -= wk[k] * lls[k]
            end
            for j in 1:nf
                tot -= flls[j]
            end
            has_pen && (tot += pen_of(θu))
            isfinite(tot) ? tot : convert(eltype(xθ), Inf)
        end
        function fgr!(F, G, xθ)
            if G === nothing
                return fr(xθ)
            end
            r = DRr.GradientResult(xθ)
            ForwardDiff.gradient!(r, fr, xθ)
            v = DRr.value(r)
            G .= DRr.gradient(r)
            isfinite(v) || (G .= NaN)
            return F === nothing ? nothing : v
        end
        xθ0 = Vector{Float64}(x̂[1:p])
        rf = OptimizationFunction(
            (xθ, _) -> fr(xθ), method.adtype;
            grad = (G, xθ, _) -> (fgr!(nothing, G, xθ); nothing),
            fg = (G, xθ, _) -> fgr!(0.0, G, xθ)
        )
        ropt, rbounds = _bounded_optimizer(method.optimizer, use_bounds, lb, ub)
        rprob = rbounds ? OptimizationProblem(rf, xθ0; lb = lb, ub = ub) :
            OptimizationProblem(rf, xθ0)
        rsol = Optimization.solve(rprob, ropt; method.optim_kwargs...)
        refine_retcode = rsol.retcode
        refine_iters = _sol_iterations(rsol)
        # damped Newton finish (p ≤ 50), which also decides convergence: L-BFGS in these
        # coordinates stalls when the problem is ill-conditioned (M18: cond(H) ≈ 6e5, stopped
        # by reltol 0.004 above the optimum, which Newton reaches in 2 steps) or fails its
        # first line search (M19: fr = Inf at α = 1 along -g). Within the remaining maxiters.
        xr = Vector{Float64}(rsol.u)
        okw = method.optim_kwargs
        nmax = p <= 50 ? min(50, get(okw, :maxiters, 1000) - refine_iters) : 0
        newton_ok = rsol.retcode == SciMLBase.ReturnCode.Success
        if nmax > 0
            fθ, newton_ok, nit = _vem_newton_finish!(
                fr, (G, x) -> fgr!(nothing, G, x), xr, lb, ub, nmax,
                Float64(get(okw, :g_abstol, 1.0e-5)), Float64(get(okw, :reltol, 1.0e-8))
            )
            refine_iters += nit
        else
            fθ = fr(xr)
        end
        if isfinite(fθ) && fθ <= fr(xθ0)
            xθ_hat = xr
            θ_hat_u = θu_of(xθ_hat)
            refine_ok = newton_ok
            refined = true
        end
    end
    θ_hat_t = transform(θ_hat_u)
    # -ELBO at the returned θ̂ and the fitted κ̂ (after an accepted refinement θ̂ is not the
    # joint optimum). notes.elbo is the bound itself; the penalty and extra_objective are
    # reported only in the objective (as for Laplace)
    obj_joint = fg!(0.0, nothing, vcat(xθ_hat, x̂[(p + 1):end]))
    elbo = has_pen ? -(obj_joint - pen_of(θ_hat_u)) : -obj_joint

    # ---- objective: importance-sampling marginal likelihood around the fitted q ----
    # batches with 1-3 REs: adaptive tensor Gauss-Hermite around q (the refinement rule;
    # deterministic and more accurate than n_is draws), otherwise t importance sampling.
    # After a refinement the rule is re-centered at θ̂ (q was fitted at the VEM estimate).
    objective = if method.n_is > 0
        is_rngs = _spawn_child_rngs(rng, nk)
        lls = Vector{Float64}(undef, nk)
        _vem_foreach(nk, caches) do k, lc
            iszero(wk[k]) && (lls[k] = 0.0; return)
            bi = keep[k]
            # after a refinement: Newton from the refinement's start, so that 1-3 RE batches
            # report the value it minimized
            μq, Lq = do_refine ? (qs[k][1], qs[k][2]) :
                _vem_q_moments(Vector{Float64}(x̂[ranges[k]]), ξs[k], nb[bi], family, θ_hat_u)
            do_refine &&
                ((μq, Lq) = _vem_adapt_rule(θ_hat_u, μq, Lq, dm, batch_infos[bi], const_cache, lc))
            lls[k] = nb[bi] <= 3 ?
                _vem_refine_batch_ll(
                    θ_hat_u, μq, Lq, _vem_refine_nodes(nb[bi], 0, method.is_df, is_rngs[k])...,
                    dm, batch_infos[bi], const_cache, lc
                ) :
                _vem_is_loglik(
                    θ_hat_u, μq, Lq, nb[bi], method.n_is, method.is_df,
                    is_rngs[k], dm, batch_infos[bi], const_cache, lc
                )
        end
        ll = 0.0
        for k in 1:nk
            ll += wk[k] * lls[k]
        end
        # fixed batches, penalty and extra_objective at θ̂ (glob is their negative sum)
        -ll + (has_glob ? glob(xθ_hat, nothing) : 0.0)
    else
        obj_joint
    end
    niter = _sol_iterations(sol)
    # after an accepted refinement θ̂ is the refinement's optimum and the objective is
    # re-centered there, so the joint fit's retcode does not bear on it (M18: the joint fit
    # stops at MaxIters on an ELBO plateau in 10/10 starts although the refinement converged)
    converged = (refined ? refine_ok : sol.retcode == SciMLBase.ReturnCode.Success) &&
        isfinite(objective)
    q_params = [
        begin
                bi = keep[k]
                # q lives on b for Gaussian random effects, else on the standardized u of
                # the prior transport b = T_θ(u) (a type property, so θ0 serves as fallback)
                base = _vem_measure(batch_infos[bi], θ_hat_u, const_cache, dm, ll_cache)
                μq, Lq = if iszero(wk[k]) && base !== nothing
                    # zero-weight batches do not enter the objective, so their κ is the start
                    # at θ0: report the Laplace approximation at θ̂ (as sampled)
                    κ0 = _vem_init_batch(
                        θ_hat_u, nb[bi], family, dm, batch_infos[bi], const_cache, ll_cache
                    )
                    κ0[1:nb[bi]], _vem_scale(κ0, nb[bi], family)
                else
                    κk = Vector{Float64}(x̂[ranges[k]])
                    ξs[k].K == 0 ?
                        _vem_effective(κk, ξs[k], nb[bi], family, θ_hat_u) :
                        _vem_q_moments(κk, ξs[k], nb[bi], family, θ_hat_u)
                end
                base === nothing &&
                (base = _vem_measure(batch_infos[bi], θu0, const_cache, dm, ll_cache))
                (mean = μq, scale = Lq, space = base isa GaussianRE ? :natural : :standardized)
            end for k in eachindex(ranges)
    ]

    summary = FitSummary(objective, converged, FitParameters(θ_hat_t, θ_hat_u), NamedTuple())
    diagnostics = FitDiagnostics(
        (; fit_seconds = time() - t0), (optimizer = method.optimizer,),
        (
            retcode = sol.retcode, iterations = niter, elbo = elbo,
            refine_retcode = refine_retcode, refine_iterations = refine_iters,
            refine_accepted = refined,
        ),
        NamedTuple()
    )
    eb_modes = store_eb_modes ?
        _compute_bstars(dm, θ_hat_u, constants_re, ll_caches, method.ebe, rng)[1] : nothing
    notes = (
        elbo = elbo, objective_kind = method.n_is > 0 ? :importance_sampling : :neg_elbo,
        refine_iterations = refine_iters, family = family, n_nodes = [length(ξ.w) for ξ in ξs], variational_params = q_params,
        variational_batches = keep,
    )
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = FrequentistREResult(sol, objective, niter, raw, notes, eb_modes)
    return FitResult(
        method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs
    )
end

"""
    get_variational_distributions(res::FitResult) -> DataFrame
    get_variational_distributions(dm::DataModel, res::FitResult) -> DataFrame

Fitted variational distributions ``q`` of a [`VEM`](@ref) fit, one row per individual in
the order of the individuals of the `DataModel` (the order that
[`get_random_effects`](@ref) uses for random effects grouped by the primary id).
Individuals linked by a shared random-effect level (e.g. a site effect next to a subject
effect) form one random-effect batch and share one joint ``q``.

Columns:
- the primary-id column: the id of the individual;
- `batch`: index of the batch of the individual in `get_notes(res).variational_params`, or
  `0` when all its random-effect levels are fixed through `constants_re` (no ``q``);
- `mean`, `scale`: ``q = N(\\mathrm{mean}, \\mathrm{scale}\\,\\mathrm{scale}^\\top)`` of the
  batch, with `scale` lower triangular (diagonal for `family = :diagonal`). For a
  [`PlanarFlowFamily`](@ref) they are the mean and Cholesky factor of the moment-matched
  Gaussian;
- `space`: `:natural` when all random effects of the batch are Gaussian (`Normal`,
  `MvNormal`), so that ``q`` is a distribution of the random effects ``b``;
  `:standardized` otherwise, so that ``q`` is a distribution of the standardized
  ``u`` and ``b = T_{\\hat\\theta}(u)`` with the prior transport ``T_{\\hat\\theta}`` (as
  in [`sample_random_effects`](@ref));
- `coordinates`: one `(re, level, component)` named tuple per entry of `mean`: the random
  effect, its grouping level and the component index (`1` for a scalar random effect).
"""
function get_variational_distributions(dm::DataModel, res::FitResult)
    notes = get_notes(res)
    (notes isa NamedTuple && haskey(notes, :variational_params)) ||
        error("get_variational_distributions requires a VEM fit result.")
    _, batch_infos, _ = _build_re_batch_infos(dm, _res_constants_re(res, NamedTuple(), dm))
    keep = notes.variational_batches
    qs = notes.variational_params
    (
        keep == findall(info -> get_n_b(info) > 0, batch_infos) &&
            all(length(qs[k].mean) == get_n_b(batch_infos[bi]) for (k, bi) in enumerate(keep))
    ) || error("The batch layout of this DataModel differs from the VEM fit; pass the DataModel of the fit.")
    cache = get_laplace_cache(get_re_group_info(dm))
    re_names = get_re_names(cache)
    n = length(get_individuals(dm))
    batch_of = zeros(Int, n)
    coords = Vector{Vector{NamedTuple}}(undef, length(keep))
    for (k, bi) in enumerate(keep)
        info = batch_infos[bi]
        batch_of[collect(get_inds(info))] .= k
        c = Vector{NamedTuple}(undef, get_n_b(info))
        for (ri, re) in enumerate(re_names)
            rei = get_re_info(info)[ri]
            levels_all = get_re_index(cache)[ri].levels
            for (li, lvl_id) in enumerate(get_levels(get_re_map(rei)))
                for (j, pos) in enumerate(get_ranges(rei)[li])
                    c[pos] = (re = re, level = levels_all[lvl_id], component = j)
                end
            end
        end
        coords[k] = c
    end
    q_of(i, f, none) = batch_of[i] == 0 ? none : copy(getfield(qs[batch_of[i]], f))
    return DataFrame(
        get_primary_id(dm) => [_individual_id(dm, i) for i in 1:n],
        :batch => batch_of,
        :mean => [q_of(i, :mean, Float64[]) for i in 1:n],
        :scale => [q_of(i, :scale, zeros(0, 0)) for i in 1:n],
        :space => [batch_of[i] == 0 ? :none : qs[batch_of[i]].space for i in 1:n],
        :coordinates => [batch_of[i] == 0 ? NamedTuple[] : copy(coords[batch_of[i]]) for i in 1:n],
    )
end

function get_variational_distributions(res::FitResult)
    dm = get_data_model(res)
    dm === nothing &&
        error("This fit result does not store a DataModel; call get_variational_distributions(dm, res) instead.")
    return get_variational_distributions(dm, res)
end
