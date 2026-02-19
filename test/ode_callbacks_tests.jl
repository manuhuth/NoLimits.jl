using Test
using NoLimits
using DataFrames
using ComponentArrays
using OrdinaryDiffEq

@testset "ODE callbacks: bolus, infusion, reset" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            σ = RealNumber(0.01)
        end

        @DifferentialEquation begin
            D(x1) ~ 0.0
        end

        @initialDE begin
            x1 = 0.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    function _ll(df; use_events::Bool)
        dm = DataModel(model, df;
                       primary_id=:ID,
                       time_col=:t,
                       evid_col=use_events ? :EVID : nothing,
                       amt_col=:AMT,
                       rate_col=:RATE,
                       cmt_col=:CMT)
        θ = get_θ0_untransformed(model.fixed.fixed)
        return NoLimits.loglikelihood(dm, θ, ComponentArray())
    end

    function _solve_with_callbacks(df)
        dm = DataModel(model, df;
                       primary_id=:ID,
                       time_col=:t,
                       evid_col=:EVID,
                       amt_col=:AMT,
                       rate_col=:RATE,
                       cmt_col=:CMT)
        θ = get_θ0_untransformed(model.fixed.fixed)
        ind = get_individuals(dm)[1]
        η = ComponentArray()
        const_cov = ind.const_cov
        pre = calculate_prede(model, θ, η, const_cov)
        pc = (;
            fixed_effects = θ,
            random_effects = η,
            constant_covariates = const_cov,
            varying_covariates = (t = ind.series.vary.t[1],),
            helpers = get_helper_funs(model),
            model_funs = get_model_funs(model),
            preDE = pre
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θ, η, const_cov)
        f! = get_de_f!(model.de.de)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            NoLimits._apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
            if infusion_rates !== nothing
                f! = function (du, u, p, t)
                    get_de_f!(model.de.de)(du, u, p, t)
                    @inbounds for i in eachindex(infusion_rates)
                        du[i] += infusion_rates[i]
                    end
                    return nothing
                end
            end
        end
        prob = ODEProblem(f!, u0, ind.tspan, compiled)
        sol = cb === nothing ?
              solve(prob, Tsit5(); dense=true) :
              solve(prob, Tsit5(); dense=true, callback=cb)
        return sol
    end

    @testset "Bolus" begin
        df_evt = DataFrame(
            ID = [1, 1, 1],
            t = [0.0, 1.0, 2.0],
            EVID = [0, 1, 0],
            AMT = [0.0, 3.0, 0.0],
            RATE = [0.0, 0.0, 0.0],
            CMT = [1, 1, 1],
            y = [0.0, missing, 3.0]
        )
        df_no = DataFrame(
            ID = [1, 1],
            t = [0.0, 2.0],
            y = [0.0, 3.0]
        )
        ll_evt = _ll(df_evt; use_events=true)
        ll_no = _ll(df_no; use_events=false)
        @test ll_evt > ll_no

        sol = _solve_with_callbacks(df_evt)
        @test isapprox(sol(0.0)[1], 0.0)
        @test isapprox(sol(2.0)[1], 3.0)
        @test isapprox(sol(1.1)[1], 3.0)
    end

    @testset "Infusion" begin
        df_evt = DataFrame(
            ID = [1, 1, 1],
            t = [0.0, 0.0, 2.0],
            EVID = [0, 1, 0],
            AMT = [0.0, 1.0, 0.0],
            RATE = [0.0, 0.5, 0.0],
            CMT = [1, 1, 1],
            y = [0.0, missing, 1.0]
        )
        df_no = DataFrame(
            ID = [1, 1],
            t = [0.0, 2.0],
            y = [0.0, 1.0]
        )
        ll_evt = _ll(df_evt; use_events=true)
        ll_no = _ll(df_no; use_events=false)
        @test ll_evt > ll_no

        sol = _solve_with_callbacks(df_evt)
        @test isapprox(sol(0.0)[1], 0.0)
        @test isapprox(sol(1.0)[1], 0.5)
        @test isapprox(sol(2.0)[1], 1.0)
    end

    @testset "Reset" begin
        df_evt = DataFrame(
            ID = [1, 1, 1],
            t = [0.0, 1.0, 2.0],
            EVID = [0, 2, 0],
            AMT = [0.0, 2.5, 0.0],
            RATE = [0.0, 0.0, 0.0],
            CMT = [1, 1, 1],
            y = [0.0, missing, 2.5]
        )
        df_no = DataFrame(
            ID = [1, 1],
            t = [0.0, 2.0],
            y = [0.0, 2.5]
        )
        ll_evt = _ll(df_evt; use_events=true)
        ll_no = _ll(df_no; use_events=false)
        @test ll_evt > ll_no

        sol = _solve_with_callbacks(df_evt)
        @test isapprox(sol(0.0)[1], 0.0)
        @test isapprox(sol(1.1)[1], 2.5)
        @test isapprox(sol(2.0)[1], 2.5)
    end
end

@testset "ODE callbacks: DataModel stores callbacks for ODE models" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            σ = RealNumber(0.01)
        end

        @DifferentialEquation begin
            D(x1) ~ 0.0
        end

        @initialDE begin
            x1 = 0.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        EVID = [1, 0, 0],
        AMT = [100.0, 0.0, 0.0],
        RATE = [0.0, 0.0, 0.0],
        CMT = [1, 1, 1],
        y = [missing, 1.0, 1.1]
    )

    dm = DataModel(model, df;
                   primary_id=:ID,
                   time_col=:t,
                   evid_col=:EVID,
                   amt_col=:AMT,
                   rate_col=:RATE,
                   cmt_col=:CMT)

    ind1 = get_individual(dm, 1)
    @test ind1.callbacks !== nothing
end
@testset "ODE callbacks: CMT name mapping" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            σ = RealNumber(0.01)
        end

        @DifferentialEquation begin
            D(x1) ~ 0.0
            D(central) ~ 0.0
        end

        @initialDE begin
            x1 = 0.0
            central = 0.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    @testset "CMT name mapping works" begin
        df = DataFrame(
            ID = [1, 1],
            t = [0.0, 0.0],
            EVID = [1, 1],
            AMT = [2.0, 3.0],
            RATE = [0.0, 0.0],
            CMT = ["x1", "central"],
            y = [missing, missing]
        )
        dm = DataModel(model, df;
                       primary_id=:ID,
                       time_col=:t,
                       evid_col=:EVID,
                       amt_col=:AMT,
                       rate_col=:RATE,
                       cmt_col=:CMT)
        ind = get_individuals(dm)[1]
        @test ind.callbacks !== nothing
    end

    @testset "CMT mixed styles error" begin
        df = DataFrame(
            ID = [1, 1],
            t = [0.0, 0.0],
            EVID = [1, 1],
            AMT = [2.0, 3.0],
            RATE = [0.0, 0.0],
            CMT = Any["x1", 2],
            y = [missing, missing]
        )
        @test_throws ErrorException DataModel(model, df;
                                              primary_id=:ID,
                                              time_col=:t,
                                              evid_col=:EVID,
                                              amt_col=:AMT,
                                              rate_col=:RATE,
                                              cmt_col=:CMT)
    end

    @testset "CMT closest-match suggestion" begin
        df = DataFrame(
            ID = [1],
            t = [0.0],
            EVID = [1],
            AMT = [2.0],
            RATE = [0.0],
            CMT = ["x2"],
            y = [missing]
        )
        err = try
            DataModel(model, df;
                      primary_id=:ID,
                      time_col=:t,
                      evid_col=:EVID,
                      amt_col=:AMT,
                      rate_col=:RATE,
                      cmt_col=:CMT)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("Closest matches", sprint(showerror, err))
    end
end

@testset "ODE callbacks: t=0 dose affects parameter-sensitive dynamics" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            ka = RealNumber(0.4)
            σ = RealNumber(0.1)
        end

        @DifferentialEquation begin
            D(depot) ~ -ka * depot
            D(center) ~ ka * depot
        end

        @initialDE begin
            depot = 0.0
            center = 0.0
        end

        @formulas begin
            y ~ Normal(center(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        EVID = [1, 0],
        AMT = [10.0, 0.0],
        RATE = [0.0, 0.0],
        CMT = ["depot", missing],
        y = [missing, 3.0]
    )

    dm = DataModel(model, df;
                   primary_id=:ID,
                   time_col=:t,
                   evid_col=:EVID,
                   amt_col=:AMT,
                   rate_col=:RATE,
                   cmt_col=:CMT)

    θ1 = get_θ0_untransformed(model.fixed.fixed)
    θ2 = copy(θ1)
    θ2.ka = θ1.ka + 1.0

    ll1 = NoLimits.loglikelihood(dm, θ1, ComponentArray())
    ll2 = NoLimits.loglikelihood(dm, θ2, ComponentArray())
    @test isfinite(ll1)
    @test isfinite(ll2)
    @test abs(ll2 - ll1) > 1e-6
end
