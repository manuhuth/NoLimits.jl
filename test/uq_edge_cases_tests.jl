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
                β = RealVector([0.1, 0.2], prior=_uq_edge_mvprior(2), calculate_se=true)
                a = RealNumber(0.05, prior=Normal(0.0, 1.0), calculate_se=true)
                σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
                ζ = NNParameters(chain; function_name=:NN1, calculate_se=false, prior=_uq_edge_mvprior(length(ζ0.value)))
                Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false, prior=_uq_edge_mvprior(length(Γ0.value)))
                sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false, prior=_uq_edge_mvprior(length(sp0.value)))
            end
            @covariates begin
                t = Covariate()
                z = Covariate()
                x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
            end
            @formulas begin
                μ = a + β[1] * t + β[2] * z + NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age / 100, sp)
                y ~ Normal(μ, σ)
            end
        end
    end

    return @Model begin
        @fixedEffects begin
            β = RealVector([0.1, 0.2], calculate_se=true)
            a = RealNumber(0.05, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=true)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
        end
        @formulas begin
            μ = a + β[1] * t + β[2] * z + NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age / 100, sp)
            y ~ Normal(μ, σ)
        end
    end
end

function _uq_edge_re_blocks_flow_model()
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=4))
    return @Model begin
        @fixedEffects begin
            β = RealVector([0.1, 0.2], calculate_se=true)
            a = RealNumber(0.05, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=true)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
            ψ = NPFParameter(1, 2, seed=1, calculate_se=false)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
        end
        @randomEffects begin
            η_mv = RandomEffect(MvNormal(zeros(2), LinearAlgebra.I); column=:ID)
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:SITE)
        end
        @formulas begin
            μ = a + β[1] * t + β[2] * z + η_mv[1] + 0.2 * η_mv[2] + η_flow[1] +
                NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age / 100, sp)
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
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @randomEffects begin
            η_mv = RandomEffect(MvNormal(zeros(2), LinearAlgebra.I); column=:ID)
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:SITE)
        end
        @formulas begin
            μ = a + β[1] * t + β[2] * z + η_mv[1] + 0.2 * η_mv[2] + η_flow[1]
            y ~ Normal(μ, σ)
        end
    end
end

function _uq_edge_fixed_df()
    return DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z=[0.2, 0.1, 0.15, 0.2, 0.05, 0.1],
        Age=[30.0, 30.0, 35.0, 35.0, 32.0, 32.0],
        BMI=[20.0, 20.0, 22.0, 22.0, 21.0, 21.0],
        y=[1.0, 1.1, 0.9, 1.0, 1.05, 1.08],
    )
end

function _uq_edge_re_df()
    return DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        SITE=[:A, :A, :B, :B, :A, :A],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z=[0.2, 0.1, 0.15, 0.2, 0.05, 0.1],
        Age=[30.0, 30.0, 35.0, 35.0, 32.0, 32.0],
        BMI=[20.0, 20.0, 22.0, 22.0, 21.0, 21.0],
        y=[1.0, 1.1, 0.9, 1.0, 1.05, 1.08],
    )
end

@testset "UQ edge: MLE/MAP/profile/mcmc_refit with vector FE + NN/SoftTree/Spline" begin
    model = _uq_edge_fixed_blocks_model(priors=true)
    dm = DataModel(model, _uq_edge_fixed_df(); primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    constants = (ζ=θ0.ζ, Γ=θ0.Γ, sp=θ0.sp)

    res_mle = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=8,)); constants=constants)
    uq_mle = compute_uq(res_mle; method=:wald, n_draws=40, rng=Random.Xoshiro(201))
    @test get_uq_source_method(uq_mle) == :mle
    @test get_uq_parameter_names(uq_mle) == [:β_1, :β_2, :a, :σ]

    uq_profile = compute_uq(res_mle;
                            method=:profile,
                            profile_method=:LIN_EXTRAPOL,
                            profile_scan_width=0.8,
                            profile_max_iter=250,
                            profile_scan_tol=1e-2,
                            profile_loss_tol=1e-2,
                            rng=Random.Xoshiro(202))
    @test get_uq_source_method(uq_profile) == :mle
    @test get_uq_parameter_names(uq_profile) == [:β_1, :β_2, :a, :σ]

    uq_refit = compute_uq(res_mle;
                          method=:mcmc_refit,
                          mcmc_turing_kwargs=(n_samples=25, n_adapt=5, progress=false),
                          mcmc_draws=15,
                          rng=Random.Xoshiro(203))
    @test get_uq_backend(uq_refit) == :mcmc_refit
    @test get_uq_parameter_names(uq_refit) == [:β_1, :β_2, :a, :σ]

    res_map = fit_model(dm, NoLimits.MAP(; optim_kwargs=(maxiters=8,)); constants=constants)
    uq_map = compute_uq(res_map; method=:wald, n_draws=40, rng=Random.Xoshiro(204))
    @test get_uq_source_method(uq_map) == :map
    @test get_uq_parameter_names(uq_map) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: MCMC chain with multivariate + planar-flow REs and vector FE" begin
    model = _uq_edge_re_flow_map_model()
    dm = DataModel(model, _uq_edge_re_df(); primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    constants = (ψ=θ0.ψ,)

    res = fit_model(dm,
                    NoLimits.MCMC(; turing_kwargs=(n_samples=25, n_adapt=5, progress=false));
                    constants=constants)
    uq = compute_uq(res; method=:chain, mcmc_draws=12, rng=Random.Xoshiro(205))
    @test get_uq_source_method(uq) == :mcmc
    @test get_uq_parameter_names(uq) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: Laplace/FOCEI with multivariate + planar-flow REs, vector FE, NN/SoftTree/Spline" begin
    model = _uq_edge_re_blocks_flow_model()
    dm = DataModel(model, _uq_edge_re_df(); primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    constants = (ζ=θ0.ζ, Γ=θ0.Γ, sp=θ0.sp, ψ=θ0.ψ)

    res_laplace = fit_model(dm,
                            NoLimits.Laplace(;
                                                     optim_kwargs=(maxiters=5,),
                                                     inner_kwargs=(maxiters=15,),
                                                     multistart_n=0,
                                                     multistart_k=0);
                            constants=constants)
    uq_laplace = compute_uq(res_laplace;
                            method=:wald,
                            n_draws=30,
                            fd_abs_step=1e-4,
                            fd_rel_step=1e-4,
                            fd_max_tries=50,
                            rng=Random.Xoshiro(206))
    @test get_uq_source_method(uq_laplace) == :laplace
    @test get_uq_parameter_names(uq_laplace) == [:β_1, :β_2, :a, :σ]

    res_focei = fit_model(dm,
                          NoLimits.FOCEI(;
                                                 optim_kwargs=(maxiters=5,),
                                                 inner_kwargs=(maxiters=15,),
                                                 info_mode=:custom,
                                                 info_custom=NoLimits.focei_information_opg,
                                                 multistart_n=0,
                                                 multistart_k=0);
                          constants=constants)
    uq_focei = compute_uq(res_focei;
                          method=:wald,
                          n_draws=30,
                          fd_abs_step=1e-6,
                          fd_rel_step=1e-6,
                          fd_max_tries=12,
                          rng=Random.Xoshiro(207))
    @test get_uq_source_method(uq_focei) == :focei
    @test get_uq_parameter_names(uq_focei) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: LaplaceMAP/FOCEIMAP with multivariate + planar-flow REs and vector FE" begin
    model = _uq_edge_re_flow_map_model()
    dm = DataModel(model, _uq_edge_re_df(); primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    constants = (ψ=θ0.ψ,)

    res_lmap = fit_model(dm,
                         NoLimits.LaplaceMAP(;
                                                     optim_kwargs=(maxiters=4,),
                                                     inner_kwargs=(maxiters=10,),
                                                     multistart_n=0,
                                                     multistart_k=0);
                         constants=constants)
    uq_lmap = compute_uq(res_lmap; method=:wald, n_draws=20, rng=Random.Xoshiro(208))
    @test get_uq_source_method(uq_lmap) == :laplace_map
    @test get_uq_parameter_names(uq_lmap) == [:β_1, :β_2, :a, :σ]

    res_fmap = fit_model(dm,
                         NoLimits.FOCEIMAP(;
                                                   optim_kwargs=(maxiters=4,),
                                                   inner_kwargs=(maxiters=10,),
                                                   info_mode=:custom,
                                                   info_custom=NoLimits.focei_information_opg,
                                                   multistart_n=0,
                                                   multistart_k=0);
                         constants=constants)
    uq_fmap = compute_uq(res_fmap; method=:wald, n_draws=20, rng=Random.Xoshiro(209))
    @test get_uq_source_method(uq_fmap) == :focei_map
    @test get_uq_parameter_names(uq_fmap) == [:β_1, :β_2, :a, :σ]
end

@testset "UQ edge: MCEM/SAEM with multivariate + planar-flow REs, vector FE, NN/SoftTree/Spline" begin
    model = _uq_edge_re_blocks_flow_model()
    dm = DataModel(model, _uq_edge_re_df(); primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    constants = (ζ=θ0.ζ, Γ=θ0.Γ, sp=θ0.sp, ψ=θ0.ψ)

    res_mcem = fit_model(dm,
                         NoLimits.MCEM(;
                                               maxiters=1,
                                               sample_schedule=2,
                                               turing_kwargs=(n_adapt=1, progress=false),
                                               optim_kwargs=(maxiters=4,));
                         constants=constants)
    uq_mcem = compute_uq(res_mcem;
                         method=:wald,
                         re_approx=:laplace,
                         pseudo_inverse=true,
                         n_draws=20,
                         rng=Random.Xoshiro(210))
    @test get_uq_source_method(uq_mcem) == :mcem
    @test get_uq_parameter_names(uq_mcem) == [:β_1, :β_2, :a, :σ]
    @test get_uq_diagnostics(uq_mcem).approximation_method == :laplace

    res_saem = fit_model(dm,
                         NoLimits.SAEM(;
                                               maxiters=1,
                                               mcmc_steps=1,
                                               update_schedule=:all,
                                               turing_kwargs=(n_adapt=1, progress=false),
                                               optim_kwargs=(maxiters=4,));
                         constants=constants)
    uq_saem = compute_uq(res_saem;
                         method=:wald,
                         re_approx=:focei,
                         re_approx_method=NoLimits.FOCEI(; info_mode=:custom, info_custom=NoLimits.focei_information_opg, multistart_n=0, multistart_k=0),
                         pseudo_inverse=true,
                         n_draws=20,
                         rng=Random.Xoshiro(211))
    @test get_uq_source_method(uq_saem) == :saem
    @test get_uq_parameter_names(uq_saem) == [:β_1, :β_2, :a, :σ]
    @test get_uq_diagnostics(uq_saem).approximation_method == :focei
end
