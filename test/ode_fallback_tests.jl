using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using OrdinaryDiffEq
using ForwardDiff

# x tracks t with lag 1/k: stiff for k = 1e4. Tsit5 needs thousands of steps on [0, 2]
# (stability-limited), so a budget of 300 steps fails; Rodas5P needs a few dozen.
# The crossing x = level fires early (t ≈ level + 1/k), i.e. inside the failed Tsit5
# attempt, so the fallback must start from a reset crossing record.
function _fb_model(; crossing::Bool)
    return crossing ? @Model(
            begin
                @fixedEffects begin
                    k = RealNumber(1.0e4)
                    σ = RealNumber(0.2, scale = :log)
                    level = RealNumber(0.01)
                end
                @covariates begin
                    t = Covariate()
                end
                @preDifferentialEquation begin
                    thr = level
                end
                @DifferentialEquation begin
                    D(x) ~ -k * (x - t)
                end
                @initialDE begin
                    x = 0.0
                end
                @formulas begin
                    tc = crossing_time(:x, :thr)
                    y ~ Normal(x(t), σ)
                    zt ~ Normal(tc, 0.5)
                end
            end
        ) : @Model(
            begin
                @fixedEffects begin
                    k = RealNumber(1.0e4)
                    σ = RealNumber(0.2, scale = :log)
                end
                @covariates begin
                    t = Covariate()
                end
                @DifferentialEquation begin
                    D(x) ~ -k * (x - t)
                end
                @initialDE begin
                    x = 0.0
                end
                @formulas begin
                    y ~ Normal(x(t), σ)
                end
            end
        )
end

function _fb_df(; crossing::Bool)
    df = DataFrame(
        ID = [1, 1, 1, 1, 1], t = [0.0, 0.5, 1.0, 1.5, 2.0],
        y = [0.0, 0.45, 1.05, 1.5, 1.95]
    )
    crossing && (df.zt = [missing, missing, missing, missing, 0.02])
    return df
end

function _fb_dm(alg, fallback_alg; crossing::Bool, stiffness_check = 2.0e4)
    m = set_solver_config(
        _fb_model(; crossing = crossing); alg = alg, fallback_alg = fallback_alg,
        stiffness_check = stiffness_check,
        kwargs = (; maxiters = 300, abstol = 1.0e-8, reltol = 1.0e-8), closed_form = :off
    )
    return DataModel(m, _fb_df(; crossing = crossing); primary_id = :ID, time_col = :t)
end

function _fb_ll_grad(dm)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    η = NoLimits._default_random_effects_from_dm(dm, NamedTuple(), θ0)
    f = v -> NoLimits.loglikelihood(dm, ComponentArray(v, getaxes(θ0)), η)
    v0 = collect(θ0)
    return f(v0), ForwardDiff.gradient(f, v0)
end

@testset "ODE fallback solver" begin
    @test get_solver_config(_fb_model(; crossing = false)).fallback_alg === nothing
    @test_throws ErrorException set_solver_config(
        _fb_model(; crossing = false); fallback_alg = :Rodas5P
    )
    # ρ·T = k·2 = 2e4: the default threshold does not trigger (Float64 solves attempt
    # Tsit5, dual solves are sent to the fallback by the probe), 1e3 does, `nothing`
    # restores the plain retry.
    for crossing in (false, true), sc in (2.0e4, 1.0e3, nothing)
        ll_fail, _ = _fb_ll_grad(_fb_dm(Tsit5(), nothing; crossing, stiffness_check = sc))
        @test ll_fail == -Inf                      # no fallback: no second attempt
        ll_ref, g_ref = _fb_ll_grad(_fb_dm(Rodas5P(), nothing; crossing = crossing))
        ll_fb, g_fb = _fb_ll_grad(_fb_dm(Tsit5(), Rodas5P(); crossing, stiffness_check = sc))
        @test isfinite(ll_ref)
        @test ll_fb == ll_ref                       # same problem, same algorithm
        @test g_fb == g_ref
        @test all(isfinite, g_fb)
        # primary succeeds -> fallback never used (unless the check routes to it)
        sc == 1.0e3 && continue
        ll_ok, _ = _fb_ll_grad(_fb_dm(Rodas5P(), Tsit5(); crossing, stiffness_check = sc))
        @test ll_ok == ll_ref
    end
    # simulation path
    dm = _fb_dm(Tsit5(), Rodas5P(); crossing = false)
    sim = simulate_data(dm)
    @test all(isfinite, sim.y)
    dm = _fb_dm(Tsit5(), Rodas5P(); crossing = false, stiffness_check = 1.0e3)
    @test all(isfinite, simulate_data(dm).y)
end

# Counts RHS calls by element type: Float64 calls and dual calls separately.
mutable struct _FbCount
    f64::Int
    dual::Int
end
function _fb_raw_solve(k, u0, sc; alg = Tsit5(), fallback = Rodas5P(), cnt = _FbCount(0, 0))
    f!(du, u, p, t) = (
        eltype(u) <: ForwardDiff.Dual ? (cnt.dual += 1) : (cnt.f64 += 1);
        du[1] = -p[1] * (u[1] - t); nothing
    )
    prob = ODEProblem{true, NoLimits.SciMLBase.FullSpecialize}(f!, u0, (0.0, 2.0), eltype(u0).([k]))
    sol = NoLimits._solve_with_fallback(
        prob, alg, fallback, (); stiffness_check = sc,
        NoLimits._ode_solve_kwargs((; maxiters = 300, abstol = 1.0e-8, reltol = 1.0e-8))...
    )
    return sol, cnt
end

@testset "ODE stiffness routing" begin
    @test get_solver_config(_fb_model(; crossing = false)).stiffness_check == 2.0e4
    m = set_solver_config(_fb_model(; crossing = false); stiffness_check = nothing)
    @test get_solver_config(m).stiffness_check === nothing
    @test_throws ErrorException set_solver_config(
        _fb_model(; crossing = false); stiffness_check = -1.0
    )
    # stiff at the start (ρ·T = 2e5): one Jacobian call, then exactly the calls of the
    # fallback solved directly; the primary is never run.
    _, c_direct = _fb_raw_solve(1.0e5, [0.0], nothing; alg = Rodas5P(), fallback = nothing)
    sol, c = _fb_raw_solve(1.0e5, [0.0], 2.0e4)
    @test sol.retcode == ReturnCode.Success
    @test c.f64 == c_direct.f64 && c.dual == c_direct.dual + 1
    _, c_off = _fb_raw_solve(1.0e5, [0.0], nothing)
    @test c_off.f64 > c_direct.f64 + 1000       # old behaviour: failed Tsit5 attempt first
    # non-stiff (ρ·T = 2): the primary solves, the check costs one Jacobian call
    _, c_direct = _fb_raw_solve(1.0, [0.0], nothing; fallback = nothing)
    sol, c = _fb_raw_solve(1.0, [0.0], 2.0e4)
    @test sol.retcode == ReturnCode.Success
    @test c.f64 == c_direct.f64 && c.dual == 1
    # dual probe: ρ·T = 2e4 passes the check, the Float64 Tsit5 probe fails, so the dual
    # problem goes to the fallback without a dual Tsit5 attempt.
    function dx(sc, alg, fb)
        cnt = _FbCount(0, 0)
        d = ForwardDiff.derivative(
            k -> _fb_raw_solve(k, [zero(k)], sc; alg, fallback = fb, cnt)[1].u[end][1], 1.0e4
        )
        return d, cnt
    end
    d_direct, c_direct = dx(nothing, Rodas5P(), nothing)
    d, c = dx(2.0e4, Tsit5(), Rodas5P())
    @test d == d_direct
    @test c.dual == c_direct.dual + 1          # + the Jacobian call
    @test c.f64 > 300                          # the Float64 probe ran and failed
    _, c_off = dx(nothing, Tsit5(), Rodas5P())
    @test c_off.dual > c_direct.dual + 1000    # old behaviour: failed dual Tsit5 attempt
end

# Infusion into x (rate 1 for t in [0, 0.5]); with k = 1e3 Tsit5 runs out of its step
# budget after the infusion stop, so the retry and the dual probe must restore the
# infusion rate the callback changed.
@testset "ODE fallback restores infusion rates" begin
    model = @Model(
        begin
            @fixedEffects begin
                k = RealNumber(1.0e3)
                σ = RealNumber(0.2, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @DifferentialEquation begin
                D(x) ~ -k * (x - t)
            end
            @initialDE begin
                x = 0.0
            end
            @formulas begin
                y ~ Normal(x(t), σ)
            end
        end
    )
    df = DataFrame(
        ID = 1, t = [0.0, 0.0, 1.0, 2.0], EVID = [0, 1, 0, 0],
        AMT = [0.0, 0.5, 0.0, 0.0], RATE = [0.0, 1.0, 0.0, 0.0], CMT = 1,
        y = [0.0, missing, 1.0, 2.0]
    )
    dmf(alg, fb, sc) = DataModel(
        set_solver_config(
            model; alg = alg, fallback_alg = fb, stiffness_check = sc,
            kwargs = (; maxiters = 300, abstol = 1.0e-8, reltol = 1.0e-8), closed_form = :off
        ),
        df; primary_id = :ID, time_col = :t, evid_col = :EVID, amt_col = :AMT,
        rate_col = :RATE, cmt_col = :CMT
    )
    @test _fb_ll_grad(dmf(Tsit5(), nothing, nothing))[1] == -Inf
    ll_ref, g_ref = _fb_ll_grad(dmf(Rodas5P(), nothing, nothing))
    for sc in (nothing, 2.0e4)
        ll, g = _fb_ll_grad(dmf(Tsit5(), Rodas5P(), sc))
        @test ll == ll_ref
        @test g == g_ref
    end
end
