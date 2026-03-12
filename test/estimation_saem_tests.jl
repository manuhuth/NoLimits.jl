using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using SciMLBase
using OptimizationOptimisers
using OptimizationBBO
using LinearAlgebra

const SAEM_FAST = (maxiters=3, t0=1, kappa=0.6, mcmc_steps=1, max_store=3)

function _saem_test_create_trans_pi0(eta_hmm, eta_initial)
    n_states = length(eta_initial) + 1
    @assert length(eta_hmm) == n_states * (n_states - 1)

    T = promote_type(eltype(eta_hmm), eltype(eta_initial))
    A = Matrix{T}(undef, n_states, n_states)
    pi0 = Vector{T}(undef, n_states)

    m0 = zero(T)
    @inbounds for j in 1:(n_states - 1)
        eta = eta_initial[j]
        m0 = ifelse(eta > m0, eta, m0)
    end

    denom0 = exp(-m0)
    @inbounds for s in 2:n_states
        val = exp(eta_initial[s - 1] - m0)
        pi0[s] = val
        denom0 += val
    end
    invden0 = inv(denom0)
    pi0[1] = exp(-m0) * invden0
    @inbounds for s in 2:n_states
        pi0[s] *= invden0
    end

    row_logits = Vector{T}(undef, n_states - 1)
    idx = 1
    @inbounds for i in 1:n_states
        m = zero(T)
        for k in 1:(n_states - 1)
            eta = eta_hmm[idx]
            row_logits[k] = eta
            m = ifelse(eta > m, eta, m)
            idx += 1
        end

        denom = exp(-m)
        k = 1
        for s in 1:n_states
            if s == i
                continue
            end
            val = exp(row_logits[k] - m)
            A[i, s] = val
            denom += val
            k += 1
        end

        invden = inv(denom)
        A[i, i] = exp(-m) * invden
        for s in 1:n_states
            if s == i
                continue
            end
            A[i, s] *= invden
        end
    end

    return A, pi0
end

@testset "SAEM default sampler" begin
    method = NoLimits.SAEM()
    @test method.saem.sampler isa NUTS
    @test method.saem.ebe_multistart_n == 50
    @test method.saem.ebe_multistart_k == 10
    @test method.saem.ebe_multistart_sampling == :lhs
    @test method.saem.ebe_rescue.sampling == :lhs
end

@testset "SAEM closed-form M-step flag metadata" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res_numeric = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
        builtin_stats=:none,
        maxiters=2,
        progress=false,
        max_store=4,
    ))
    @test !NoLimits.get_closed_form_mstep_used(res_numeric)
    notes_numeric = NoLimits.get_notes(res_numeric)
    @test notes_numeric.closed_form_mstep_used === false
    @test notes_numeric.closed_form_mstep_mode == :numeric_only
    @test isempty(notes_numeric.closed_form_mstep_sources)

    res_builtin = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
        builtin_stats=:closed_form,
        maxiters=2,
        progress=false,
        max_store=4,
    ))
    @test NoLimits.get_closed_form_mstep_used(res_builtin)
    notes_builtin = NoLimits.get_notes(res_builtin)
    @test notes_builtin.closed_form_mstep_used === true
    @test notes_builtin.closed_form_mstep_mode == :hybrid
    @test :builtin_stats in notes_builtin.closed_form_mstep_sources

    suffstats = (dm, batch_infos, b_current, θ, constants_re) -> begin
        s = 0.0
        for b in b_current
            s += sum(b)
        end
        return (; s)
    end
    mstep_closed_form = (s, dm) -> get_θ0_untransformed(dm.model.fixed.fixed)
    res_custom = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
        builtin_stats=:none,
        suffstats=suffstats,
        mstep_closed_form=mstep_closed_form,
        maxiters=2,
        progress=false,
        max_store=4,
    ))
    @test NoLimits.get_closed_form_mstep_used(res_custom)
    notes_custom = NoLimits.get_notes(res_custom)
    @test notes_custom.closed_form_mstep_used === true
    @test notes_custom.closed_form_mstep_mode == :closed_form_only
    @test :custom_mstep_closed_form in notes_custom.closed_form_mstep_sources

    @test_logs match_mode=:any (:info, r"SAEM closed-form M-step plan") begin
        fit_model(dm, NoLimits.SAEM(;
            sampler=MH(),
            turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
            builtin_stats=:auto,
            maxiters=1,
            progress=false,
            max_store=2,
        ))
    end
end

@testset "SAEM basic (random effects)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4))
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

@testset "SAEM serial vs threaded is reproducible" begin
    Threads.nthreads() < 2 && return

    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.SAEM(; sampler=MH(),
                                   turing_kwargs=(n_samples=1, n_adapt=0, progress=false, verbose=false),
                                   maxiters=2,
                                   mcmc_steps=1,
                                   max_store=4,
                                   progress=false)
    res_serial = fit_model(dm, method; serialization=EnsembleSerial(), rng=MersenneTwister(123))
    res_threads = fit_model(dm, method; serialization=EnsembleThreads(), rng=MersenneTwister(123))
    @test res_serial.summary.objective == res_threads.summary.objective
    @test collect(NoLimits.get_params(res_serial, scale=:untransformed)) ==
          collect(NoLimits.get_params(res_threads, scale=:untransformed))
end

@testset "SAEM basic with NUTS" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=5, n_adapt=3, progress=false),
                             max_store=4))
    @test res isa FitResult
end

@testset "SAEM convergence requires both parameter and Q stabilization" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
        maxiters=2,
        t0=0,
        consecutive_params=1,
        atol_theta=Inf,
        rtol_theta=Inf,
        atol_Q=0.0,
        rtol_Q=0.0,
        max_store=4
    ))
    # If stopping used only parameter tolerance, this would stop after 1 iteration.
    @test res.result.iterations == 2
end

@testset "SAEM multiple RE groups" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        SITE = [:X, :X, :X, :X, :Y, :Y, :Y, :Y],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4))
    @test res isa FitResult
    re = NoLimits.get_random_effects(dm, res)
    @test !isempty(re)
end

@testset "SAEM constants_re" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4);
                    constants_re=(; η=(; A=0.0,)))
    @test res isa FitResult
end

@testset "SAEM threaded updates" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4);
                    serialization=EnsembleThreads())
    @test res isa FitResult
end

@testset "SAEM minibatch updates" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             update_schedule=1, max_store=4))
    @test res isa FitResult
end

@testset "SAEM optimizer Adam (OptimizationOptimisers)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.SAEM(optimizer=OptimizationOptimisers.Adam(0.05),
                  optim_kwargs=(; maxiters=3),
                  sampler=MH(),
                  turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                  max_store=4)
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "SAEM optimizer BlackBoxOptim (OptimizationBBO)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    lb, ub = default_bounds_from_start(dm; margin=1.0)
    method = NoLimits.SAEM(optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
                  optim_kwargs=(; iterations=3),
                  sampler=MH(),
                  turing_kwargs=(n_samples=4, n_adapt=0, progress=false),
                  max_store=4,
                  lb=lb, ub=ub)
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "SAEM constants for fixed effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4);
                    constants=(a=0.2,))
    @test res isa FitResult
end

@testset "SAEM RE distribution with constant covariates" begin
    model = @Model begin
        @fixedEffects begin
            μ0 = RealNumber(0.0)
            β = RealNumber(0.5)
            σ = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariate(constant_on=:ID)
        end

        @randomEffects begin
            η = RandomEffect(Normal(μ0 + β * x, τ); column=:ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        x = [1.0, 1.0, 2.0, 2.0],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (scalar RE)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_stats=:closed_form,
                             resid_var_param=:σ,
                             re_cov_params=(; η=:τ)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multivariate RE)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale=:cholesky)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_stats=:closed_form,
                             resid_var_param=:σ,
                             re_cov_params=(; η=:Ω)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multivariate diagonal + means)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ1 = RealNumber(0.1)
            μ2 = RealNumber(0.2)
            ω1 = RealNumber(0.5, scale=:log)
            ω2 = RealNumber(0.4, scale=:log)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([μ1, μ2], LinearAlgebra.Diagonal([ω1, ω2])); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.10, 0.15, 0.20, 0.25, 0.05, 0.10]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             maxiters=2,
                             builtin_stats=:closed_form,
                             resid_var_param=:σ,
                             re_cov_params=(; η=(:ω1, :ω2)),
                             re_mean_params=(; η=(:μ1, :μ2))))
    @test res isa FitResult
    θ = NoLimits.get_params(res; scale=:untransformed)
    @test isfinite(θ.μ1) && isfinite(θ.μ2)
    @test θ.ω1 > 0 && θ.ω2 > 0 && θ.σ > 0
end

@testset "SAEM builtin_stats uses variance for MvNormal diagonal targets" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ1 = RealNumber(0.1)
            μ2 = RealNumber(0.2)
            ω1 = RealNumber(0.5, scale=:log)
            ω2 = RealNumber(0.4, scale=:log)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([μ1, μ2], LinearAlgebra.Diagonal([ω1, ω2])); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.10, 0.15, 0.20, 0.25]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = NoLimits.get_θ0_untransformed(model.fixed.fixed)
    stats = (;
        re=(; η=(family=:mvnormal, mean=[0.0, 0.0], second=[0.25 0.0; 0.0 0.04], n=10)),
        outcome=NamedTuple(),
        hmm=NamedTuple()
    )

    updates = NoLimits._saem_builtin_updates_from_smoothed_stats(
        dm,
        θ,
        stats,
        NamedTuple(),
        NamedTuple(),
        (; η=(:ω1, :ω2)),
        NamedTuple()
    )
    @test isapprox(updates.ω1, 0.25; atol=1e-12)
    @test isapprox(updates.ω2, 0.04; atol=1e-12)
end

@testset "SAEM builtin_stats gaussian_re respects fixed-effect lower bounds for RE means" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ = RealNumber(0.2, lower=0.0)
            ω = RealNumber(0.5, scale=:log)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(μ, ω); column=:ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [-2.1, -1.8, -2.0, -1.9, -2.2, -1.7]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             maxiters=3,
                             builtin_stats=:closed_form,
                             resid_var_param=:σ,
                             re_cov_params=(; η=:ω),
                             re_mean_params=(; η=:μ)))
    @test res isa FitResult
    θ = NoLimits.get_params(res; scale=:untransformed)
    @test θ.μ >= -1e-10
    @test θ.ω > 0 && θ.σ > 0
end

@testset "SAEM builtin_stats gaussian_re (multiple RE dists)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
            τ_id = RealNumber(0.3, scale=:log)
            τ_site = RealNumber(0.2, scale=:log)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, τ_id); column=:ID)
            η_site = RandomEffect(Normal(0.0, τ_site); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        SITE = [:X, :X, :Y, :Y],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_stats=:closed_form,
                             resid_var_param=:σ,
                             re_cov_params=(; η_id=:τ_id, η_site=:τ_site)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multiple normal outcomes)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ)
            y2 ~ Normal(b + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_stats=:closed_form,
                             resid_var_param=:σ,
                             re_cov_params=(; η=:τ)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multiple normal outcomes, separate σ)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ1 = RealNumber(0.4, scale=:log)
            σ2 = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ1)
            y2 ~ Normal(b + η, σ2)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_stats=:closed_form,
                             resid_var_param=(; y1=:σ1, y2=:σ2),
                             re_cov_params=(; η=:τ)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats skips missing normal outcomes (regression)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ1 = RealNumber(0.4, scale=:log)
            σ2 = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ1)
            y2 ~ Normal(b + η, σ2)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = Union{Missing, Float64}[0.1, missing, 0.0, -0.1],
        y2 = Union{Missing, Float64}[missing, 0.25, 0.05, missing]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             maxiters=2,
                             builtin_stats=:closed_form,
                             resid_var_param=(; y1=:σ1, y2=:σ2),
                             re_cov_params=(; η=:τ)))
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    @test notes.builtin_stats_mode_effective == :closed_form
    @test :builtin_stats in notes.closed_form_mstep_sources
    θ = NoLimits.get_params(res; scale=:untransformed)
    @test θ.σ1 > 0 && θ.σ2 > 0 && θ.τ > 0
end

@testset "SAEM builtin_stats gaussian_re falls back for non-Normal outcomes" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0, 1, 1, 0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_stats=:closed_form,
                             re_cov_params=(; η=:τ)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects gaussian_re (scalar RE)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(a, τ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η=:τ)
    @test auto_cfg.re_mean_params == (; η=:a)
    @test auto_cfg.resid_var_param == :σ

    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             maxiters=2,
                             builtin_stats=:auto))
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects gaussian_re (MvNormal diagonal + means)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ1 = RealNumber(0.1)
            μ2 = RealNumber(-0.2)
            ω1 = RealNumber(0.5, scale=:log)
            ω2 = RealNumber(0.7, scale=:log)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([μ1, μ2], Diagonal([ω1, ω2])); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.15, -0.1, -0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η=(:ω1, :ω2))
    @test auto_cfg.re_mean_params == (; η=(:μ1, :μ2))
    @test auto_cfg.resid_var_param == :σ

    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             maxiters=2,
                             builtin_stats=:auto))
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects MvNormal symbol mean with fixed diagonal expression" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            z = RealVector([0.0, 0.0])
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(z, Diagonal(ones(length(z)))); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.15, -0.1, -0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == NamedTuple()
    @test auto_cfg.re_mean_params == (; η=:z)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM builtin_stats auto detects LogNormal/Exponential RE + outcomes" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μη = RealNumber(0.1)
            ση = RealNumber(0.5, scale=:log)
            θη = RealNumber(1.2, scale=:log)
            μy = RealNumber(0.0)
            σy = RealNumber(0.4, scale=:log)
            θy = RealNumber(1.1, scale=:log)
        end

        @randomEffects begin
            η_ln = RandomEffect(LogNormal(μη, ση); column=:ID)
            η_exp = RandomEffect(Exponential(θη); column=:SITE)
        end

        @formulas begin
            y_ln ~ LogNormal(μy, σy)
            y_exp ~ Exponential(θy)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        SITE = [:S1, :S1, :S2, :S2],
        t = [0.0, 1.0, 0.0, 1.0],
        y_ln = [1.2, 1.1, 0.9, 1.0],
        y_exp = [0.8, 1.0, 0.7, 1.3]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η_ln=:ση, η_exp=:θη)
    @test auto_cfg.re_mean_params == (; η_ln=:μη)
    @test auto_cfg.resid_var_param == (; y_ln=(; μ=:μy, σ=:σy), y_exp=:θy)

    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             maxiters=2,
                             builtin_stats=:auto))
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects Bernoulli/Poisson outcome params when direct symbols" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p = RealNumber(0.6)
            λ = RealNumber(1.5)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            yb ~ Bernoulli(p)
            yp ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        yb = [0, 1, 1, 0],
        yp = [1, 2, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.resid_var_param == (; yb=:p, yp=:λ)
end

@testset "SAEM builtin_stats classifies HMM closed-form eligibility" begin
    model_partial = @Model begin
        @helpers begin
            create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(eta_hmm, eta_initial)
        end

        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            mean_transitions = RealVector([0.0, 0.0])
            mean_initial = RealNumber(0.0)
            omega_hmm = RealVector([0.3, 0.3], scale=fill(:log, 2))
            omega_initial = RealNumber(0.3, scale=:log)
            eta_theta = RealVector(fill(0.6, 4), scale=fill(:logit, 4))
        end

        @randomEffects begin
            eta_hmm = RandomEffect(MvNormal(mean_transitions, Diagonal(omega_hmm)); column=:ID)
            eta_initial = RandomEffect(Normal(mean_initial, omega_initial); column=:ID)
        end

        @formulas begin
            trans_pi0 = create_trans_pi0(eta_hmm, [eta_initial])
            theta_mat = reshape(eta_theta, 2, 2)
            emissions = ntuple(s -> ntuple(j -> Bernoulli(theta_mat[s, j]), 2), 2)
            y ~ MVDiscreteTimeDiscreteStatesHMM(trans_pi0[1], emissions, Categorical(trans_pi0[2]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [[1, 0], [0, 1], [1, 1], [0, 0]]
    )

    dm_partial = DataModel(model_partial, df; primary_id=:ID, time_col=:t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm_partial, NoLimits.get_names(model_partial.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; eta_hmm=:omega_hmm, eta_initial=:omega_initial)
    @test auto_cfg.re_mean_params == (; eta_hmm=:mean_transitions, eta_initial=:mean_initial)
    @test auto_cfg.resid_var_param == NamedTuple()
    @test auto_cfg.hmm_emission_params.y.target == :eta_theta
    @test auto_cfg.hmm_outcomes == (:y,)
    @test auto_cfg.hmm_transition_closed_form == :covered_by_re_or_external
    @test auto_cfg.hmm_emission_closed_form == :eligible

    elig_partial = NoLimits._saem_builtin_closed_form_eligibility(
        dm_partial,
        auto_cfg.re_cov_params,
        auto_cfg.re_mean_params,
        auto_cfg.resid_var_param,
        auto_cfg.hmm_emission_params
    )
    @test elig_partial.re_block_eligible
    @test !elig_partial.outcome_block_eligible
    @test elig_partial.hmm_emission_block_eligible
    @test elig_partial.has_any_closed_form_block
    @test elig_partial.hmm_outcomes == (:y,)
    @test elig_partial.hmm_transition_closed_form == :covered_by_re_or_external
    @test elig_partial.hmm_emission_closed_form == :eligible
    @test !(:hmm_latent_state_suffstats_not_available_builtin in elig_partial.reasons)

    res_partial = fit_model(dm_partial, NoLimits.SAEM(;
                                  sampler=MH(),
                                  turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
                                  mcmc_steps=1,
                                  max_store=4,
                                  maxiters=2,
                                  progress=false,
                                  builtin_stats=:auto);
                            store_eb_modes=false)
    notes_partial = NoLimits.get_notes(res_partial)
    @test notes_partial.builtin_stats_mode_effective == :closed_form
    @test notes_partial.builtin_stats_closed_form_eligibility.re_block_eligible
    @test !notes_partial.builtin_stats_closed_form_eligibility.outcome_block_eligible
    @test notes_partial.builtin_stats_closed_form_eligibility.hmm_emission_block_eligible
    @test notes_partial.builtin_stats_closed_form_eligibility.hmm_outcomes == (:y,)
    @test NoLimits.get_closed_form_mstep_used(res_partial)
    @test notes_partial.closed_form_mstep_mode == :closed_form_only
    @test :builtin_stats in notes_partial.closed_form_mstep_sources

    model_ineligible = @Model begin
        @helpers begin
            create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(eta_hmm, eta_initial)
        end

        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            eta_theta = RealVector(fill(0.6, 4), scale=fill(:logit, 4))
        end

        @randomEffects begin
            eta_hmm = RandomEffect(MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])); column=:ID)
            eta_initial = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            trans_pi0 = create_trans_pi0(eta_hmm, [eta_initial])
            theta_mat = reshape(eta_theta, 2, 2)
            emissions = ntuple(s -> ntuple(j -> Bernoulli(clamp(0.8 * theta_mat[s, j] + 0.1, 0.0, 1.0)), 2), 2)
            y ~ MVDiscreteTimeDiscreteStatesHMM(trans_pi0[1], emissions, Categorical(trans_pi0[2]))
        end
    end

    dm_ineligible = DataModel(model_ineligible, df; primary_id=:ID, time_col=:t)
    auto_cfg_none = NoLimits._saem_autodetect_gaussian_re(dm_ineligible, NoLimits.get_names(model_ineligible.fixed.fixed))
    @test auto_cfg_none === nothing

    res_ineligible = fit_model(dm_ineligible, NoLimits.SAEM(;
                                    sampler=MH(),
                                    turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
                                    mcmc_steps=1,
                                    max_store=4,
                                    maxiters=2,
                                    progress=false,
                                    builtin_stats=:auto);
                              store_eb_modes=false)
    notes_ineligible = NoLimits.get_notes(res_ineligible)
    @test notes_ineligible.builtin_stats_mode_effective == :none
    @test !notes_ineligible.builtin_stats_closed_form_eligibility.has_any_closed_form_block
    @test notes_ineligible.builtin_stats_closed_form_eligibility.hmm_outcomes == (:y,)
    @test !NoLimits.get_closed_form_mstep_used(res_ineligible)
    @test notes_ineligible.closed_form_mstep_mode == :numeric_only
end

@testset "SAEM discrete HMM emission stats use rowwise posteriors" begin
    P = [0.85 0.15; 0.25 0.75]
    init = Categorical([0.6, 0.4])
    dists = [
        DiscreteTimeDiscreteStatesHMM(P, (Bernoulli(0.9), Bernoulli(0.2)), init),
        DiscreteTimeDiscreteStatesHMM(P, (Bernoulli(0.9), Bernoulli(0.2)), init),
        DiscreteTimeDiscreteStatesHMM(P, (Bernoulli(0.9), Bernoulli(0.2)), init),
    ]
    ys = Union{Missing, Int}[1, missing, 0]

    gamma, ok = NoLimits._saem_hmm_smoothed_gamma(dists, ys, Float64)

    @test ok
    @test isapprox(gamma[:, 1], posterior_hidden_states(dists[1], 1); atol=1e-12)
    @test isapprox(gamma[:, 2], probabilities_hidden_states(dists[2]); atol=1e-12)
    @test isapprox(gamma[:, 3], posterior_hidden_states(dists[3], 0); atol=1e-12)
end

@testset "SAEM builtin_stats HMM emission handles fully missing rows (regression)" begin
    model = @Model begin
        @helpers begin
            create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(eta_hmm, eta_initial)
        end

        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            mean_transitions = RealVector([0.0, 0.0])
            mean_initial = RealNumber(0.0)
            omega_hmm = RealVector([0.3, 0.3], scale=fill(:log, 2))
            omega_initial = RealNumber(0.3, scale=:log)
            eta_theta = RealVector(fill(0.6, 4), scale=fill(:logit, 4))
        end

        @randomEffects begin
            eta_hmm = RandomEffect(MvNormal(mean_transitions, Diagonal(omega_hmm)); column=:ID)
            eta_initial = RandomEffect(Normal(mean_initial, omega_initial); column=:ID)
        end

        @formulas begin
            trans_pi0 = create_trans_pi0(eta_hmm, [eta_initial])
            theta_mat = reshape(eta_theta, 2, 2)
            emissions = ntuple(s -> ntuple(j -> Bernoulli(theta_mat[s, j]), 2), 2)
            y ~ MVDiscreteTimeDiscreteStatesHMM(trans_pi0[1], emissions, Categorical(trans_pi0[2]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = Any[[1, 0], missing, [1, 1], [0, 0]]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
                              sampler=MH(),
                              turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
                              mcmc_steps=1,
                              max_store=4,
                              maxiters=2,
                              progress=false,
                              builtin_stats=:auto);
                        store_eb_modes=false)
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    @test notes.builtin_stats_mode_effective == :closed_form
    @test :builtin_stats in notes.closed_form_mstep_sources
end

@testset "SAEM builtin_stats auto detects lts_random_no_cv full closed-form coverage" begin
    n_states = 7
    n_outcomes = 9
    n_transition_random_effects = n_states * (n_states - 1)

    model = @Model begin
        @helpers begin
            create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(eta_hmm, eta_initial)
        end

        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            mean_initial = RealVector(fill(0.0, n_states - 1))
            mean_transitions = RealVector(fill(0.0, n_transition_random_effects))
            omega_hmm = RealVector(fill(0.3, n_transition_random_effects), scale=fill(:log, n_transition_random_effects))
            omega_initial = RealVector(fill(0.3, n_states - 1), scale=fill(:log, n_states - 1))
            eta_theta = RealVector(fill(0.5, n_states * n_outcomes), scale=fill(:logit, n_states * n_outcomes))
        end

        @randomEffects begin
            eta_hmm = RandomEffect(MvNormal(mean_transitions, Diagonal(omega_hmm)); column=:ID)
            eta_initial = RandomEffect(MvNormal(mean_initial, Diagonal(omega_initial)); column=:ID)
        end

        @formulas begin
            trans_pi0 = create_trans_pi0(eta_hmm, eta_initial)
            theta_mat = reshape(eta_theta, 7, 9)
            emissions = ntuple(s -> ntuple(j -> Bernoulli(theta_mat[s, j]), 9), 7)
            y ~ MVDiscreteTimeDiscreteStatesHMM(trans_pi0[1], emissions, Categorical(trans_pi0[2]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
        ]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    fixed_names = NoLimits.get_names(model.fixed.fixed)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm, fixed_names)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; eta_hmm=:omega_hmm, eta_initial=:omega_initial)
    @test auto_cfg.re_mean_params == (; eta_hmm=:mean_transitions, eta_initial=:mean_initial)
    @test auto_cfg.resid_var_param == NamedTuple()
    @test auto_cfg.hmm_emission_params.y.target == :eta_theta
    @test auto_cfg.hmm_outcomes == (:y,)
    @test auto_cfg.hmm_transition_closed_form == :covered_by_re_or_external
    @test auto_cfg.hmm_emission_closed_form == :eligible

    elig = NoLimits._saem_builtin_closed_form_eligibility(
        dm,
        auto_cfg.re_cov_params,
        auto_cfg.re_mean_params,
        auto_cfg.resid_var_param,
        auto_cfg.hmm_emission_params,
    )
    @test elig.re_block_eligible
    @test !elig.outcome_block_eligible
    @test elig.hmm_emission_block_eligible
    @test elig.has_any_closed_form_block
    @test elig.hmm_outcomes == (:y,)
    @test elig.hmm_transition_closed_form == :covered_by_re_or_external
    @test elig.hmm_emission_closed_form == :eligible

    closed_form_targets = Symbol[]
    for v in values(auto_cfg.re_cov_params)
        NoLimits._saem_collect_target_symbols!(closed_form_targets, v)
    end
    for v in values(auto_cfg.re_mean_params)
        NoLimits._saem_collect_target_symbols!(closed_form_targets, v)
    end
    for col in keys(auto_cfg.hmm_emission_params)
        info = getfield(auto_cfg.hmm_emission_params, col)
        if hasproperty(info, :target) && info.target isa Symbol
            push!(closed_form_targets, info.target)
        end
    end
    @test Set(closed_form_targets) == Set(fixed_names)
end

@testset "SAEM builtin_stats runs lts_random_no_cv style model in closed-form-only mode" begin
    n_states = 7
    n_outcomes = 9
    n_transition_random_effects = n_states * (n_states - 1)

    model = @Model begin
        @helpers begin
            create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(eta_hmm, eta_initial)
        end

        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            mean_initial = RealVector(fill(0.0, n_states - 1))
            mean_transitions = RealVector(fill(0.0, n_transition_random_effects))
            omega_hmm = RealVector(fill(0.3, n_transition_random_effects), scale=fill(:log, n_transition_random_effects))
            omega_initial = RealVector(fill(0.3, n_states - 1), scale=fill(:log, n_states - 1))
            eta_theta = RealVector(fill(0.5, n_states * n_outcomes), scale=fill(:logit, n_states * n_outcomes))
        end

        @randomEffects begin
            eta_hmm = RandomEffect(MvNormal(mean_transitions, Diagonal(omega_hmm)); column=:ID)
            eta_initial = RandomEffect(MvNormal(mean_initial, Diagonal(omega_initial)); column=:ID)
        end

        @formulas begin
            trans_pi0 = create_trans_pi0(eta_hmm, eta_initial)
            theta_mat = reshape(eta_theta, 7, 9)
            emissions = ntuple(s -> ntuple(j -> Bernoulli(theta_mat[s, j]), 9), 7)
            y ~ MVDiscreteTimeDiscreteStatesHMM(trans_pi0[1], emissions, Categorical(trans_pi0[2]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
        ]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
            sampler=MH(),
            turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
            mcmc_steps=1,
            max_store=4,
            maxiters=2,
            progress=false,
            builtin_stats=:auto);
        store_eb_modes=false)

    notes = NoLimits.get_notes(res)
    @test notes.builtin_stats_mode_effective == :closed_form
    @test NoLimits.get_closed_form_mstep_used(res)
    @test notes.closed_form_mstep_mode == :closed_form_only
    @test :builtin_stats in notes.closed_form_mstep_sources
end

@testset "SAEM builtin_mean glm (Bernoulli + Poisson)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y1 ~ Bernoulli(p)
            λ = exp(b + η)
            y2 ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0, 1, 1, 0],
        y2 = [1, 2, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_mean=:glm))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm (Normal + Exponential)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ)
            y2 ~ Exponential(exp(b + η))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.8, 1.2, 0.5, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_mean=:glm))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm (ODE Normal)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            k = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @initialDE begin
            x1 = exp(a + η)
        end

        @DifferentialEquation begin
            D(x1) ~ -exp(k) * x1
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 0.8, 1.05, 0.85]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             maxiters=2,
                             builtin_mean=:glm))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm + builtin_stats (Normal)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_mean=:glm,
                             builtin_stats=:closed_form,
                             resid_var_param=:σ,
                             re_cov_params=(; η=:τ)))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm + builtin_stats (Normal outcomes, separate σ)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ1 = RealNumber(0.4, scale=:log)
            σ2 = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ1)
            y2 ~ Normal(b + η, σ2)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                             max_store=4,
                             builtin_mean=:glm,
                             builtin_stats=:closed_form,
                             resid_var_param=(; y1=:σ1, y2=:σ2),
                             re_cov_params=(; η=:τ)))
    @test res isa FitResult
end

@testset "SAEM threaded helper cache preserves ODE options" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + η
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.9, 0.7]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    ll_cache = build_ll_cache(dm; ode_kwargs=(abstol=1e-8, reltol=1e-7))
    threaded = NoLimits._saem_thread_caches(dm, ll_cache, 2)
    @test length(threaded) == 2
    @test all(c -> c.ode_args == ll_cache.ode_args, threaded)
    @test all(c -> c.ode_kwargs == ll_cache.ode_kwargs, threaded)
end

@testset "SAEM thread RNGs are reproducible from passed rng" begin
    r1 = NoLimits._saem_thread_rngs(MersenneTwister(42), 3)
    r2 = NoLimits._saem_thread_rngs(MersenneTwister(42), 3)
    r3 = NoLimits._saem_thread_rngs(MersenneTwister(99), 3)
    s1 = [rand(r, Float64) for r in r1]
    s2 = [rand(r, Float64) for r in r2]
    s3 = [rand(r, Float64) for r in r3]
    @test s1 == s2
    @test s1 != s3
end

@testset "SAEM final EBE rescue options are configurable" begin
    method = NoLimits.SAEM(;
        ebe_rescue_on_high_grad=false,
        ebe_rescue_multistart_n=91,
        ebe_rescue_multistart_k=13,
        ebe_rescue_max_rounds=7,
        ebe_rescue_grad_tol=1e-5
    )
    @test method.saem.ebe_rescue.enabled == false
    @test method.saem.ebe_rescue.multistart_n == 91
    @test method.saem.ebe_rescue.multistart_k == 13
    @test method.saem.ebe_rescue.max_rounds == 7
    @test method.saem.ebe_rescue.grad_tol == 1e-5
end

@testset "SAEM get_random_effects recomputes EB modes with rescue options" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0.1, 0.2, 0.0, -0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
        max_store=4,
        maxiters=2,
        mcmc_steps=1,
        t0=1,
        ebe_rescue_on_high_grad=true,
        ebe_rescue_multistart_n=12,
        ebe_rescue_multistart_k=4,
        ebe_rescue_max_rounds=2,
        ebe_rescue_grad_tol=1e-7
    )
    res = fit_model(dm, method; store_eb_modes=false)
    re = NoLimits.get_random_effects(dm, res)
    @test haskey(re, :η)
    @test nrow(re.η) == length(unique(df.ID))
end
