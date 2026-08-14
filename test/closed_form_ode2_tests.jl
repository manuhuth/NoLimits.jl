using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using OrdinaryDiffEq
using Random
using ForwardDiff
using FiniteDifferences
using LinearAlgebra

# Two-state decoupled diagonal-linear PK-flavored model (constant forcing on x2),
# no random effects → fit with MLE. Built once, reused across the oracle tests.
function _cf_diag2_model(cf::Symbol = :auto)
    m = @Model begin
        @fixedEffects begin
            k1 = RealNumber(0.5, scale = :log)
            k2 = RealNumber(0.3, scale = :log)
            Input = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x1) ~ -k1 * x1
            D(x2) ~ -k2 * x2 + Input
        end
        @initialDE begin
            x1 = 1.0
            x2 = 0.5
        end
        @formulas begin
            y ~ Normal(x1(t) + x2(t), σ)
        end
    end
    return set_solver_config(m; saveat_mode = :saveat, closed_form = cf)
end

function _cf_diag2_df()
    DataFrame(ID = [1, 1, 1, 2, 2, 2], t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [1.5, 1.0, 0.8, 1.4, 0.95, 0.75])
end

# Second half of the closed-form ODE tests (split from closed_form_ode_tests.jl
# for CI shard balance; the halves run in different shards and are
# self-contained — the _cf_diag2_* helpers above are shared by both).
@testset "closed-form fit matches numerical (single state, Laplace)" begin
    res_cf = fit_model(fx_ode_dm(), NoLimits.Laplace())
    dm_off = DataModel(
        set_solver_config(fx_ode_model(); saveat_mode = :saveat,
            closed_form = :off),
        fx_ode_df(); primary_id = :ID, time_col = :t)
    res_off = fit_model(dm_off, NoLimits.Laplace())
    @test NoLimits.get_objective(res_cf)≈NoLimits.get_objective(res_off) rtol=1e-5
    p_cf = NoLimits.get_params(res_cf; scale = :untransformed)
    p_off = NoLimits.get_params(res_off; scale = :untransformed)
    @test collect(p_cf)≈collect(p_off) rtol=1e-4
end

@testset "closed-form fit matches numerical (two-state, MLE)" begin
    df = _cf_diag2_df()
    dm_cf = DataModel(_cf_diag2_model(:auto), df; primary_id = :ID, time_col = :t)
    dm_off = DataModel(_cf_diag2_model(:off), df; primary_id = :ID, time_col = :t)
    res_cf = fit_model(dm_cf, NoLimits.MLE())
    res_off = fit_model(dm_off, NoLimits.MLE())
    @test NoLimits.get_objective(res_cf)≈NoLimits.get_objective(res_off) rtol=1e-5
    @test collect(NoLimits.get_params(res_cf;
        scale = :untransformed))≈
    collect(NoLimits.get_params(res_off; scale = :untransformed)) rtol=1e-4
end

@testset "closed-form fit matches numerical (Bateman chain, MLE)" begin
    # Sequential two-compartment chain (:bateman mode — scalar Bateman closed form).
    # Exercises the scalar chain path and its ForwardDiff gradient (fit-driven).
    function chain(cf)
        m = @Model begin
            @fixedEffects begin
                k1 = RealNumber(0.7, scale = :log)
                k2 = RealNumber(0.4, scale = :log)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x1) ~ -k1 * x1
                D(x2) ~ k1 * x1 - k2 * x2
            end
            @initialDE begin
                x1 = 1.0
                x2 = 0.0
            end
            @formulas begin
                y ~ Normal(x2(t), σ)
            end
        end
        set_solver_config(m; saveat_mode = :saveat, closed_form = cf)
    end
    df = DataFrame(ID = [1, 1, 1, 2, 2, 2], t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [0.2, 0.35, 0.3, 0.25, 0.4, 0.28])
    dm_cf = DataModel(chain(:auto), df; primary_id = :ID, time_col = :t)
    @test get_closed_form_plan(dm_cf).mode === :bateman
    dm_off = DataModel(chain(:off), df; primary_id = :ID, time_col = :t)
    res_cf = fit_model(dm_cf, NoLimits.MLE())
    res_off = fit_model(dm_off, NoLimits.MLE())
    @test NoLimits.get_objective(res_cf)≈NoLimits.get_objective(res_off) rtol=1e-5
    @test collect(NoLimits.get_params(res_cf;
        scale = :untransformed))≈
    collect(NoLimits.get_params(res_off; scale = :untransformed)) rtol=1e-4
end

@testset "closed-form matches numerical (PKPD events)" begin
    # Mid-trajectory doses/infusions: closed form splits the trajectory at event
    # breakpoints. Compare the marginal log-likelihood at FIXED params (isolates the
    # event solve from optimizer sensitivity on sparse dosing data).
    function pk(cf)
        m = @Model begin
            @fixedEffects begin
                k = RealNumber(0.2, scale = :log)
                σ = RealNumber(5.0, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x1) ~ -k * x1
            end
            @initialDE begin
                x1 = 0.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end
        set_solver_config(m; saveat_mode = :saveat, closed_form = cf)
    end
    kw = (; primary_id = :ID, time_col = :t, evid_col = :EVID, amt_col = :AMT,
        rate_col = :RATE, cmt_col = :CMT)
    ll_match(df) = begin
        dm_off = DataModel(pk(:off), df; kw...)
        dm_cf = DataModel(pk(:auto), df; kw...)
        @test get_closed_form_plan(dm_cf).eligible
        r = fit_model(dm_off, NoLimits.MLE())
        NoLimits.get_loglikelihood(dm_cf, r), NoLimits.get_loglikelihood(dm_off, r)
    end
    # initial (t0) + mid-trajectory bolus
    a, b = ll_match(DataFrame(ID = [1, 1, 1, 1, 2, 2, 2, 2],
        t = [0.0, 2.0, 5.0, 8.0, 0.0, 2.0, 5.0, 8.0], EVID = [1, 0, 1, 0, 1, 0, 1, 0],
        AMT = [100.0, 0, 50.0, 0, 100.0, 0, 50.0, 0], RATE = zeros(8), CMT = fill(1, 8),
        y = [missing, 60.0, missing, 25.0, missing, 65.0, missing, 28.0]))
    @test a≈b rtol=1e-4
    # zero-order infusion over [0, 3]
    a, b = ll_match(DataFrame(ID = [1, 1, 1, 1, 2, 2, 2, 2],
        t = [0.0, 2.0, 4.0, 6.0, 0.0, 2.0, 4.0, 6.0], EVID = [1, 0, 0, 0, 1, 0, 0, 0],
        AMT = [30.0, 0, 0, 0, 30.0, 0, 0, 0], RATE = [10.0, 0, 0, 0, 10.0, 0, 0, 0],
        CMT = fill(1, 8), y = [missing, 15.0, 12.0, 7.0, missing, 16.0, 13.0, 8.0]))
    @test a≈b rtol=1e-4
end

@testset "closed-form Bateman with a forced upstream compartment (#112)" begin
    # An infusion event dosing the DEPOT adds constant forcing to the upstream state,
    # which the Bateman kernel (initial-value propagation only) cannot represent — it
    # must fall back to the matrix-exp form. Tight solver tolerances so the comparison
    # measures the closed form, not solver error.
    function oral(cf)
        m = @Model begin
            @fixedEffects begin
                ka = RealNumber(0.7, scale = :log)
                ke = RealNumber(0.2, scale = :log)
                σ = RealNumber(5.0, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(depot) ~ -ka * depot
                D(central) ~ ka * depot - ke * central
            end
            @initialDE begin
                depot = 0.0
                central = 0.0
            end
            @formulas begin
                y ~ Normal(central(t), σ)
            end
        end
        set_solver_config(m; alg = Vern9(), saveat_mode = :saveat, closed_form = cf,
            kwargs = (; reltol = 1e-12, abstol = 1e-12))
    end
    # bolus at t0 (depot), infusion into the depot at t=6, reset of central at t=12
    df = DataFrame(ID = fill(1, 7), t = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
        EVID = [1, 0, 0, 1, 0, 2, 0], AMT = [100.0, 0, 0, 30.0, 0, 5.0, 0],
        RATE = [0.0, 0, 0, 10.0, 0, 0, 0], CMT = [1, 1, 1, 1, 1, 2, 1],
        y = [missing, 60.0, 40.0, missing, 25.0, missing, 20.0])
    kw = (; primary_id = :ID, time_col = :t, evid_col = :EVID, amt_col = :AMT,
        rate_col = :RATE, cmt_col = :CMT)
    dm_cf = DataModel(oral(:auto), df; kw...)
    dm_off = DataModel(oral(:off), df; kw...)
    @test get_closed_form_plan(dm_cf).mode === :bateman
    θ = get_θ0_untransformed(get_fixed(get_model(dm_cf)))
    η = ComponentArray(NamedTuple())
    ll_cf = NoLimits.conditional_loglikelihood(dm_cf, 1, θ, η)
    ll_off = NoLimits.conditional_loglikelihood(dm_off, 1, θ, η)
    @test ll_cf≈ll_off atol=1e-8
    # #153: an observation at the same time as an event (here the EVID=2 reset at t=12,
    # plus a RATE dose on the DOWNSTREAM compartment) must see the PRE-event state, the
    # numerical path's convention.
    df2 = DataFrame(ID = fill(1, 8), t = [0.0, 2.0, 6.0, 6.0, 8.0, 12.0, 12.0, 16.0],
        EVID = [1, 0, 0, 1, 0, 0, 2, 0], AMT = [100.0, 0, 0, 30.0, 0, 0, 0.0, 0],
        RATE = [0.0, 0, 0, 15.0, 0, 0, 0, 0], CMT = [1, 1, 1, 2, 1, 1, 2, 1],
        y = [missing, 60.0, 40.0, missing, 25.0, 22.0, missing, 20.0])
    dm_cf2 = DataModel(oral(:auto), df2; kw...)
    dm_off2 = DataModel(oral(:off), df2; kw...)
    @test get_closed_form_plan(dm_cf2).mode === :bateman
    @test NoLimits.conditional_loglikelihood(
        dm_cf2, 1, θ, η)≈
    NoLimits.conditional_loglikelihood(dm_off2, 1, θ, η) atol=1e-8
end

@testset "closed-form/numerical split (partial)" begin
    # Linear PK compartment x1 (closed-form) driving a nonlinear PD state x2
    # (numerical, reads x1(t) from the closed form). Compare marginal loglik at
    # fixed params to the full numerical solve.
    function pkpd(cf; alg = nothing)
        m = @Model begin
            @fixedEffects begin
                k = RealNumber(0.5, scale = :log)
                kin = RealNumber(1.0)
                kout = RealNumber(0.3, scale = :log)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x1) ~ -k * x1
                D(x2) ~ kin * x1 - kout * x2^2
            end
            @initialDE begin
                x1 = 1.0
                x2 = 0.0
            end
            @formulas begin
                y ~ Normal(x2(t), σ)
            end
        end
        set_solver_config(m; alg = alg, saveat_mode = :saveat, closed_form = cf)
    end
    df = DataFrame(ID = [1, 1, 1, 1, 2, 2, 2, 2],
        t = [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
        y = [0.4, 0.6, 0.7, 0.5, 0.45, 0.62, 0.68, 0.52])
    # partial (linear/nonlinear split) is opt-in via :all (not in the fast :auto set)
    dm_cf = DataModel(pkpd(:all), df; primary_id = :ID, time_col = :t)
    plan = get_closed_form_plan(dm_cf)
    @test plan.eligible
    @test plan.cf_states == [1]          # only the linear PK state is closed-form
    @test length(plan.cf_states) < plan.n # partial (hybrid) solve
    @test !get_closed_form_plan(DataModel(pkpd(:auto), df; primary_id = :ID,
        time_col = :t)).eligible          # :auto keeps the split numerical
    dm_off = DataModel(pkpd(:off), df; primary_id = :ID, time_col = :t)
    r = fit_model(dm_off, NoLimits.MLE())
    @test NoLimits.get_loglikelihood(dm_cf, r)≈NoLimits.get_loglikelihood(dm_off, r) rtol=1e-5

    # Stiff solver: the hybrid's reduced problem uses a clock state so it stays
    # autonomous and the ForwardDiff objective gradient does not nest Duals through
    # the implicit solver's time-gradient. The fit must run (not error) and agree.
    dm_cf_s = DataModel(pkpd(:all; alg = Rodas5P()), df; primary_id = :ID, time_col = :t)
    dm_off_s = DataModel(pkpd(:off; alg = Rodas5P()), df; primary_id = :ID, time_col = :t)
    r_s = fit_model(dm_off_s, NoLimits.MLE())
    @test isfinite(NoLimits.get_objective(fit_model(dm_cf_s, NoLimits.MLE())))
    @test NoLimits.get_loglikelihood(dm_cf_s, r_s)≈NoLimits.get_loglikelihood(dm_off_s, r_s) rtol=1e-4
end

# Bidirectional two-compartment (genuinely `:linear`, so opt-in via `closed_form = :all`)
# with a random effect on clearance. The Laplace inner Hessian drives nested Duals
# through `_cf_matexp`.
function _cf_twocmt_re_model(cf::Symbol)
    m = @Model begin
        @fixedEffects begin
            k10 = RealNumber(0.5, scale = :log)
            k12 = RealNumber(0.3, scale = :log)
            k21 = RealNumber(0.2, scale = :log)
            ω = RealNumber(0.3, scale = :log)
            σ = RealNumber(0.05, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @DifferentialEquation begin
            D(x1) ~ -(k10 * exp(η) + k12) * x1 + k21 * x2
            D(x2) ~ k12 * x1 - k21 * x2
        end
        @initialDE begin
            x1 = 1.0
            x2 = 0.0
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
    return set_solver_config(m; saveat_mode = :saveat, closed_form = cf)
end

function _cf_twocmt_re_df()
    ts = [0.5, 1.0, 2.0]
    rates = [0.45, 0.75, 0.6, 0.95, 0.55]   # between-ID spread keeps ω identified
    noise = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.005, 0.01, -0.02, 0.012,
        -0.008, 0.018, -0.011, 0.007, -0.016]
    return DataFrame(ID = repeat(1:length(rates), inner = length(ts)),
        t = repeat(ts, length(rates)),
        y = vec([0.9 * exp(-r * t) for t in ts, r in rates]) .+ noise)
end

@testset "closed-form :linear under nested Duals" begin
    # Derivative correctness of `_cf_matexp` at 1st and 2nd ForwardDiff order.
    A(p) = [-(p[1] + p[2]) p[3] p[4]; p[2] -p[3] 0.0; 0.0 0.0 0.0]
    w = [0.3, -0.7, 1.1, 0.2, 0.9, -0.4, 0.5, 0.15, -0.25]
    f(p) = sum(vec(NoLimits._cf_matexp(A(p) .* 1.7)) .* w)
    p0 = [0.5, 0.3, 0.2, 0.4]
    fd = central_fdm(5, 1)
    @test ForwardDiff.gradient(f, p0)≈FiniteDifferences.grad(fd, f, p0)[1] rtol=1e-6 atol=1e-9
    H = ForwardDiff.hessian(f, p0)
    H_fd = FiniteDifferences.jacobian(fd, x -> ForwardDiff.gradient(f, x), p0)[1]
    @test H≈H_fd rtol=1e-6 atol=1e-9
    @test H ≈ H'

    # Nested-Dual value and first partials match the Float64 and single-Dual paths.
    Mv = A(p0) .* 1.7
    d1 = ForwardDiff.Dual{:t1}.(Mv, 1.0)
    E2 = NoLimits._cf_matexp(ForwardDiff.Dual{:t2}.(d1, 1.0))
    @test ForwardDiff.value.(ForwardDiff.value.(E2)) ≈ exp(Mv)
    @test ForwardDiff.partials.(ForwardDiff.value.(E2), 1) ≈
          ForwardDiff.partials.(NoLimits._cf_matexp(d1), 1)

    # Laplace fit (nested Duals in the inner Hessian) and Wald UQ on that fit.
    df = _cf_twocmt_re_df()
    dm_all = DataModel(_cf_twocmt_re_model(:all), df; primary_id = :ID, time_col = :t)
    dm_off = DataModel(_cf_twocmt_re_model(:off), df; primary_id = :ID, time_col = :t)
    @test get_closed_form_plan(dm_all).mode === :linear
    # k12/k21 held fixed: with only x1 observed they sit on a flat ridge, which would
    # make the two optimizer paths land far apart for no path-related reason.
    cst = (k12 = 0.3, k21 = 0.2)
    res_all = fit_model(dm_all, NoLimits.Laplace(); constants = cst)
    res_off = fit_model(dm_off, NoLimits.Laplace(); constants = cst)
    @test NoLimits.get_objective(res_all)≈NoLimits.get_objective(res_off) rtol=1e-4
    @test collect(NoLimits.get_params(res_all;
        scale = :untransformed))≈
    collect(NoLimits.get_params(res_off; scale = :untransformed)) rtol=1e-2
    # Same params, both paths: isolates the closed-form solve from optimizer noise.
    @test NoLimits.get_loglikelihood(dm_all,
        res_off)≈NoLimits.get_loglikelihood(dm_off, res_off) rtol=1e-5
    @test compute_uq(res_all; method = :wald, n_draws = 20,
        rng = Random.Xoshiro(11)) !== nothing
end

@testset "closed-form and numerical simulate agree" begin
    # Exercises the simulation consumer (and, transitively, the shared solve helper
    # on a dense/off-grid path). Same RNG → identical noise draws, so simulated
    # outcomes agree up to the closed-form vs numerical state difference.
    df = _cf_diag2_df()
    dm_cf = DataModel(_cf_diag2_model(:auto), df; primary_id = :ID, time_col = :t)
    dm_off = DataModel(_cf_diag2_model(:off), df; primary_id = :ID, time_col = :t)
    s_cf = simulate_data(dm_cf; rng = MersenneTwister(1), replace_missings = true)
    s_off = simulate_data(dm_off; rng = MersenneTwister(1), replace_missings = true)
    @test s_cf.y≈s_off.y rtol=1e-5
end
