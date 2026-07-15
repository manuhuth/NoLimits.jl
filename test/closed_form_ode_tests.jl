using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using OrdinaryDiffEq
using Random

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

@testset "closed-form scalar arithmetic" begin
    # x(Δ) = exp(aΔ)x0 + (b/a)(exp(aΔ)-1) for a ≠ 0
    a, b, x0, Δ = -0.7, 0.4, 1.3, 2.0
    ref = exp(a * Δ) * x0 + (b / a) * (exp(a * Δ) - 1)
    @test NoLimits._cf_state_value(a, b, x0, Δ)≈ref rtol=1e-12
    # a → 0 limit is x0 + b·Δ (exactly, and continuously)
    @test NoLimits._cf_state_value(0.0, 0.4, 1.3, 2.0)≈1.3 + 0.4 * 2.0 rtol=1e-12
    @test NoLimits._cf_state_value(1e-10, 0.4, 1.3, 2.0)≈1.3 + 0.4 * 2.0 rtol=1e-6
    @test NoLimits._phi_expm1(0.0) ≈ 1.0
    @test NoLimits._phi_expm1(1e-6)≈expm1(1e-6) / 1e-6 rtol=1e-9
    @test NoLimits._phi_expm1(0.5)≈expm1(0.5) / 0.5 rtol=1e-12
end

@testset "closed-form eligibility detection" begin
    # single-state linear + additive RE (fx_ode_model) → eligible
    dm = fx_ode_dm()
    plan = get_closed_form_plan(dm)
    @test plan.eligible
    @test plan.mode === :diagonal
    @test plan.n == 1

    # two-state decoupled diagonal with constant forcing → eligible, n = 2
    dm2 = DataModel(_cf_diag2_model(:auto), _cf_diag2_df(); primary_id = :ID, time_col = :t)
    @test get_closed_form_plan(dm2).eligible
    @test get_closed_form_plan(dm2).n == 2

    df = _cf_diag2_df()
    cf_plan(m) = get_closed_form_plan(DataModel(m, df; primary_id = :ID, time_col = :t))
    ineligible(m) = !cf_plan(m).eligible

    # coupled (off-diagonal, constant A) → eligible as a general :linear system
    coupled = set_solver_config(
        (@Model begin
            @fixedEffects begin
                k10 = RealNumber(0.5, scale = :log)
                k12 = RealNumber(0.2, scale = :log)
                k20 = RealNumber(0.3, scale = :log)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x1) ~ -k10 * x1
                D(x2) ~ k12 * x1 - k20 * x2
            end
            @initialDE begin
                x1 = 1.0
                x2 = 0.0
            end
            @formulas begin
                y ~ Normal(x1(t) + x2(t), σ)
            end
        end); saveat_mode = :saveat)
    @test cf_plan(coupled).eligible
    @test cf_plan(coupled).mode === :linear

    # nonlinear in state
    @test ineligible(set_solver_config(
        (@Model begin
            @fixedEffects begin
                a = RealNumber(0.3)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x1) ~ -a * x1^2
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end); saveat_mode = :saveat))

    # time-varying forcing via a signal → constant-forcing check fails
    @test ineligible(set_solver_config(
        (@Model begin
            @fixedEffects begin
                a = RealNumber(0.3)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                s(t) = sin(t)
                D(x1) ~ -a * x1 + s(t)
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end); saveat_mode = :saveat))

    # a linear state that feeds FROM a nonlinear state cannot form a self-contained
    # block → no closed-form subset (fixpoint excludes it)
    @test ineligible(set_solver_config(
        (@Model begin
            @fixedEffects begin
                a = RealNumber(0.3)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x1) ~ -a * x1 + x2
                D(x2) ~ -a * x2^2
            end
            @initialDE begin
                x1 = 1.0
                x2 = 0.5
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end); saveat_mode = :saveat))
end

@testset "closed_form = :diagonal errors on ineligible model" begin
    mnl = set_solver_config(
        (@Model begin
            @fixedEffects begin
                a = RealNumber(0.3)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x1) ~ -a * x1^2
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end); saveat_mode = :saveat, closed_form = :diagonal)
    dm = DataModel(mnl, _cf_diag2_df(); primary_id = :ID, time_col = :t)
    @test_throws ErrorException get_closed_form_plan(dm)
end

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

@testset "closed-form fit matches numerical (linear chain, MLE)" begin
    # Coupled two-compartment chain (:linear mode via matrix-exp action). Exercises
    # the general-linear path and its ForwardDiff gradient (fit is gradient-driven).
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
    @test get_closed_form_plan(dm_cf).mode === :linear
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
    dm_cf = DataModel(pkpd(:auto), df; primary_id = :ID, time_col = :t)
    plan = get_closed_form_plan(dm_cf)
    @test plan.eligible
    @test plan.cf_states == [1]          # only the linear PK state is closed-form
    @test length(plan.cf_states) < plan.n # partial (hybrid) solve
    dm_off = DataModel(pkpd(:off), df; primary_id = :ID, time_col = :t)
    r = fit_model(dm_off, NoLimits.MLE())
    @test NoLimits.get_loglikelihood(dm_cf, r)≈NoLimits.get_loglikelihood(dm_off, r) rtol=1e-5

    # Stiff solver: the hybrid's reduced problem uses a clock state so it stays
    # autonomous and the ForwardDiff objective gradient does not nest Duals through
    # the implicit solver's time-gradient. The fit must run (not error) and agree.
    dm_cf_s = DataModel(pkpd(:auto; alg = Rodas5P()), df; primary_id = :ID, time_col = :t)
    dm_off_s = DataModel(pkpd(:off; alg = Rodas5P()), df; primary_id = :ID, time_col = :t)
    r_s = fit_model(dm_off_s, NoLimits.MLE())
    @test isfinite(NoLimits.get_objective(fit_model(dm_cf_s, NoLimits.MLE())))
    @test NoLimits.get_loglikelihood(dm_cf_s, r_s)≈NoLimits.get_loglikelihood(dm_off_s, r_s) rtol=1e-4
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
