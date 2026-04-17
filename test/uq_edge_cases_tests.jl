using Test
using NoLimits
using DataFrames
using Distributions
using Lux
using LinearAlgebra
using Random

_uq_edge_mvprior(n::Int) = MvNormal(zeros(n), Matrix{Float64}(I, n, n))

function _uq_edge_fixed_blocks_model(; priors::Bool)
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=4))
    if priors
        ζ0 = NNParameters(chain; function_name=:NN1, calculate_se=false)
        Γ0 = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
        sp0 = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
        return @Model begin
            @fixedEffects begin
                β  = RealVector([0.1, 0.2], prior=_uq_edge_mvprior(2), calculate_se=true)
                a  = RealNumber(0.05, prior=Normal(0.0, 1.0), calculate_se=true)
                σ  = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
                ζ  = NNParameters(chain; function_name=:NN1, calculate_se=false, prior=_uq_edge_mvprior(length(ζ0.value)))
                Γ  = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false, prior=_uq_edge_mvprior(length(Γ0.value)))
                sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false, prior=_uq_edge_mvprior(length(sp0.value)))
            end
            @covariates begin; t = Covariate(); z = Covariate()
                x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID); end
            @formulas begin
                μ = a + β[1]*t + β[2]*z + NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age/100, sp)
                y ~ Normal(μ, σ)
            end
        end
    end
    return @Model begin
        @fixedEffects begin
            β  = RealVector([0.1, 0.2], calculate_se=true)
            a  = RealNumber(0.05, calculate_se=true)
            σ  = RealNumber(0.3, scale=:log, calculate_se=true)
            ζ  = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ  = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
        end
        @covariates begin; t = Covariate(); z = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID); end
        @formulas begin
            μ = a + β[1]*t + β[2]*z + NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age/100, sp)
            y ~ Normal(μ, σ)
        end
    end
end

function _uq_edge_re_blocks_flow_model()
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=4))
    return @Model begin
        @fixedEffects begin
            β  = RealVector([0.1, 0.2], calculate_se=true)
            a  = RealNumber(0.05, calculate_se=true)
            σ  = RealNumber(0.3, scale=:log, calculate_se=true)
            ζ  = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ  = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
            ψ  = NPFParameter(1, 2, seed=1, calculate_se=false)
        end
        @covariates begin; t = Covariate(); z = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID); end
        @randomEffects begin
            η_mv   = RandomEffect(MvNormal(zeros(2), LinearAlgebra.I); column=:ID)
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:SITE)
        end
        @formulas begin
            μ = a + β[1]*t + β[2]*z + η_mv[1] + 0.2*η_mv[2] + η_flow[1] +
                NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age/100, sp)
            y ~ Normal(μ, σ)
        end
    end
end

function _uq_edge_re_flow_map_model()
    ψ0 = NPFParameter(1, 2, seed=1, calculate_se=false)
    return @Model begin
        @fixedEffects begin
            β = RealVector([0.1, 0.2], prior=_uq_edge_mvprior(2), calculate_se=true)
            a = RealNumber(0.05, prior=Normal(0.0, 1.0), calculate_se=true)
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
            ψ = NPFParameter(1, 2, seed=1, calculate_se=false, prior=_uq_edge_mvprior(length(ψ0.value)))
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @randomEffects begin
            η_mv   = RandomEffect(MvNormal(zeros(2), LinearAlgebra.I); column=:ID)
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:SITE)
        end
        @formulas begin
            μ = a + η_mv[1] + 0.2*η_mv[2] + η_flow[1]
            y ~ Normal(μ, σ)
        end
    end
end

# ── module-level fixtures (built once, reused across testsets) ─────────────
const _UQE_FIXED_DF = DataFrame(
    ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0], z=[0.2,0.1,0.15,0.2,0.05,0.1],
    Age=[30.0,30.0,35.0,35.0,32.0,32.0], BMI=[20.0,20.0,22.0,22.0,21.0,21.0],
    y=[1.0,1.1,0.9,1.0,1.05,1.08])

const _UQE_RE_DF = DataFrame(
    ID=[1,1,2,2,3,3], SITE=[:A,:A,:B,:B,:A,:A], t=[0.0,1.0,0.0,1.0,0.0,1.0],
    z=[0.2,0.1,0.15,0.2,0.05,0.1], Age=[30.0,30.0,35.0,35.0,32.0,32.0],
    BMI=[20.0,20.0,22.0,22.0,21.0,21.0], y=[1.0,1.1,0.9,1.0,1.05,1.08])

# Shared DataModels — each expensive model built ONCE
const _UQE_FIXED_DM_P    = DataModel(_uq_edge_fixed_blocks_model(priors=true), _UQE_FIXED_DF; primary_id=:ID, time_col=:t)
const _UQE_RE_BLOCKS_DM  = DataModel(_uq_edge_re_blocks_flow_model(),           _UQE_RE_DF;    primary_id=:ID, time_col=:t)
const _UQE_RE_FLOW_MAP_DM = DataModel(_uq_edge_re_flow_map_model(),             _UQE_RE_DF;    primary_id=:ID, time_col=:t)

# Pre-compute θ0 and constants (fast once dm exists)
const _UQE_FIXED_θ0         = get_θ0_untransformed(_UQE_FIXED_DM_P.model.fixed.fixed)
const _UQE_FIXED_CONSTANTS   = (ζ=_UQE_FIXED_θ0.ζ, Γ=_UQE_FIXED_θ0.Γ, sp=_UQE_FIXED_θ0.sp)
const _UQE_BLOCKS_θ0         = get_θ0_untransformed(_UQE_RE_BLOCKS_DM.model.fixed.fixed)
const _UQE_BLOCKS_CONSTANTS   = (ζ=_UQE_BLOCKS_θ0.ζ, Γ=_UQE_BLOCKS_θ0.Γ, sp=_UQE_BLOCKS_θ0.sp, ψ=_UQE_BLOCKS_θ0.ψ)
const _UQE_FLOW_MAP_θ0        = get_θ0_untransformed(_UQE_RE_FLOW_MAP_DM.model.fixed.fixed)
const _UQE_FLOW_MAP_CONSTANTS  = (ψ=_UQE_FLOW_MAP_θ0.ψ,)

# ══════════════════════════════════════════════════════════════════════════════

@testset "UQ edge: MLE/MAP/profile/mcmc_refit with vector FE + NN/SoftTree/Spline" begin
    res_mle = fit_model(_UQE_FIXED_DM_P, NoLimits.MLE(; optim_kwargs=(maxiters=8,));
                        constants=_UQE_FIXED_CONSTANTS)
    uq_mle = compute_uq(res_mle; method=:wald, n_draws=8, rng=Random.Xoshiro(201))
    @test get_uq_source_method(uq_mle) == :mle
    @test get_uq_parameter_names(uq_mle) == [:β_1, :β_2, :a, :σ]

    uq_profile = compute_uq(res_mle;
                            method=:profile,
                            profile_method=:LIN_EXTRAPOL,
                            profile_scan_width=0.8,
                            profile_max_iter=12,
                            profile_scan_tol=1e-2,
                            profile_loss_tol=1e-2,
                            rng=Random.Xoshiro(202))
    @test get_uq_source_method(uq_profile) == :mle
    @test get_uq_parameter_names(uq_profile) == [:β_1, :β_2, :a, :σ]

    uq_refit = compute_uq(res_mle;
                          method=:mcmc_refit,
                          mcmc_turing_kwargs=(n_samples=6, n_adapt=2, progress=false),
                          mcmc_draws=5,
                          rng=Random.Xoshiro(203))
    @test get_uq_backend(uq_refit) == :mcmc_refit
    @test get_uq_parameter_names(uq_refit) == [:β_1, :β_2, :a, :σ]

    res_map = fit_model(_UQE_FIXED_DM_P, NoLimits.MAP(; optim_kwargs=(maxiters=8,));
                        constants=_UQE_FIXED_CONSTANTS)
    uq_map = compute_uq(res_map; method=:wald, n_draws=8, rng=Random.Xoshiro(204))
    @test get_uq_source_method(uq_map) == :map
    @test get_uq_parameter_names(uq_map) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: MCMC chain with multivariate + planar-flow REs and vector FE" begin
    res = fit_model(_UQE_RE_FLOW_MAP_DM,
                    NoLimits.MCMC(; turing_kwargs=(n_samples=15, n_adapt=5, progress=false));
                    constants=_UQE_FLOW_MAP_CONSTANTS)
    uq = compute_uq(res; method=:chain, mcmc_draws=5, rng=Random.Xoshiro(205))
    @test get_uq_source_method(uq) == :mcmc
    @test get_uq_parameter_names(uq) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: Laplace with multivariate + planar-flow REs, vector FE, NN/SoftTree/Spline" begin
    res_laplace = fit_model(_UQE_RE_BLOCKS_DM,
                            NoLimits.Laplace(;
                                optim_kwargs=(maxiters=5,),
                                inner_kwargs=(maxiters=6,),
                                multistart_n=0, multistart_k=0);
                            constants=_UQE_BLOCKS_CONSTANTS)
    uq_laplace = compute_uq(res_laplace;
                            method=:wald,
                            pseudo_inverse=true,
                            n_draws=8,
                            fd_abs_step=1e-4, fd_rel_step=1e-4, fd_max_tries=50,
                            rng=Random.Xoshiro(206))
    @test get_uq_source_method(uq_laplace) == :laplace
    @test get_uq_parameter_names(uq_laplace) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: LaplaceMAP with multivariate + planar-flow REs and vector FE" begin
    res_lmap = fit_model(_UQE_RE_FLOW_MAP_DM,
                         NoLimits.LaplaceMAP(;
                             optim_kwargs=(maxiters=4,),
                             inner_kwargs=(maxiters=6,),
                             multistart_n=0, multistart_k=0);
                         constants=_UQE_FLOW_MAP_CONSTANTS)
    uq_lmap = compute_uq(res_lmap;
                         method=:wald,
                         pseudo_inverse=true,
                         n_draws=8,
                         fd_abs_step=1e-4, fd_rel_step=1e-4, fd_max_tries=50,
                         rng=Random.Xoshiro(208))
    @test get_uq_source_method(uq_lmap) == :laplace_map
    @test get_uq_parameter_names(uq_lmap) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: MCEM/SAEM with multivariate + planar-flow REs, vector FE, NN/SoftTree/Spline" begin
    res_mcem = fit_model(_UQE_RE_BLOCKS_DM,
                         NoLimits.MCEM(;
                             maxiters=1,
                             sample_schedule=2,
                             turing_kwargs=(n_samples=4, n_adapt=1, progress=false),
                             optim_kwargs=(maxiters=4,));
                         constants=_UQE_BLOCKS_CONSTANTS)
    uq_mcem = compute_uq(res_mcem;
                         method=:wald, re_approx=:laplace, pseudo_inverse=true,
                         n_draws=8, rng=Random.Xoshiro(210))
    @test get_uq_source_method(uq_mcem) == :mcem
    @test get_uq_parameter_names(uq_mcem) == [:β_1, :β_2, :a, :σ]
    @test get_uq_diagnostics(uq_mcem).approximation_method == :laplace

    res_saem = fit_model(_UQE_RE_BLOCKS_DM,
                         NoLimits.SAEM(;
                             maxiters=1,
                             mcmc_steps=1,
                             update_schedule=:all,
                             turing_kwargs=(n_adapt=1, progress=false),
                             optim_kwargs=(maxiters=4,));
                         constants=_UQE_BLOCKS_CONSTANTS)
    uq_saem = compute_uq(res_saem;
                         method=:wald, re_approx=:laplace, pseudo_inverse=true,
                         n_draws=8, rng=Random.Xoshiro(211))
    @test get_uq_source_method(uq_saem) == :saem
    @test get_uq_parameter_names(uq_saem) == [:β_1, :β_2, :a, :σ]
    @test get_uq_diagnostics(uq_saem).approximation_method == :laplace
end
