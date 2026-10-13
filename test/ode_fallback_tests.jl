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

function _fb_dm(alg, fallback_alg; crossing::Bool)
    m = set_solver_config(
        _fb_model(; crossing = crossing); alg = alg, fallback_alg = fallback_alg,
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
    for crossing in (false, true)
        ll_fail, _ = _fb_ll_grad(_fb_dm(Tsit5(), nothing; crossing = crossing))
        @test ll_fail == -Inf                      # default: no second attempt
        ll_ref, g_ref = _fb_ll_grad(_fb_dm(Rodas5P(), nothing; crossing = crossing))
        ll_fb, g_fb = _fb_ll_grad(_fb_dm(Tsit5(), Rodas5P(); crossing = crossing))
        @test isfinite(ll_ref)
        @test ll_fb == ll_ref                       # same problem, same algorithm
        @test g_fb == g_ref
        @test all(isfinite, g_fb)
        # primary succeeds -> fallback never used
        ll_ok, _ = _fb_ll_grad(_fb_dm(Rodas5P(), Tsit5(); crossing = crossing))
        @test ll_ok == ll_ref
    end
    # simulation path
    dm = _fb_dm(Tsit5(), Rodas5P(); crossing = false)
    sim = simulate_data(dm)
    @test all(isfinite, sim.y)
end
