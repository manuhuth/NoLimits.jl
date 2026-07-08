using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using OrdinaryDiffEq
using ForwardDiff

# x(t) = t  (D(x) = k = 1, x0 = 0); crosses `level` at t = level.
function _crossing_model()
    @Model begin
        @fixedEffects begin
            k = RealNumber(1.0)
            σ = RealNumber(0.2, scale = :log)
            level = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
        end
        @preDifferentialEquation begin
            thr = level
        end
        @DifferentialEquation begin
            D(x) ~ k
        end
        @initialDE begin
            x = 0.0
        end
        @formulas begin
            tc = crossing_time(:x, :thr)
            rv = crossing_rootval(:x, :thr)
            y ~ Normal(x(t), σ)
            zt ~ Normal(tc, 0.5)
            rz ~ Normal(rv, 0.5)
        end
    end
end

# ind 1 crosses within its horizon (t up to 2 > 1); ind 2 does not (t up to 0.5 < 1).
function _crossing_dm()
    df = DataFrame(
        ID = [1, 1, 1, 1, 1, 2, 2, 2],
        t = [0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.25, 0.5],
        y = [0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.25, 0.5],
        zt = [missing, missing, missing, missing, 1.0, missing, missing, 0.5],
        rz = [missing, missing, missing, missing, 0.0, missing, missing, 0.0])
    return DataModel(_crossing_model(), df; primary_id = :ID, time_col = :t)
end

@testset "crossing_rootval values (fired vs non-fired)" begin
    dm = _crossing_dm()
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ηz = NoLimits._default_random_effects_from_dm(dm, NamedTuple(), θ)

    accs = map(1:2) do i
        ind = dm.individuals[i]
        sol, comp = NoLimits._solve_dense_individual(dm, ind, θ, ηz[i])
        NoLimits._sol_accessors_with_crossings(dm.model, sol, comp, θ, ηz[i], ind.const_cov)
    end

    # ind 1 crosses at t = level = 1 → rootvalue 0
    @test isapprox(accs[1].tc, 1.0; atol = 1e-4)
    @test isapprox(accs[1].rv, 0.0; atol = 1e-8)

    # ind 2 never crosses within [0, 0.5] → tc falls back to the horizon (0.5),
    # rootvalue = x(0.5) - level = 0.5 - 1.0 = -0.5
    @test isapprox(accs[2].tc, 0.5; atol = 1e-4)
    @test isapprox(accs[2].rv, -0.5; atol = 1e-6)
end

@testset "crossing_rootval likelihood is finite and AD-differentiable" begin
    dm = _crossing_dm()
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ηz = NoLimits._default_random_effects_from_dm(dm, NamedTuple(), θ)

    ll = NoLimits.loglikelihood(dm, θ, ηz; serialization = NoLimits.EnsembleSerial())
    @test isfinite(ll)

    tr = get_transform(dm.model.fixed.fixed)
    itr = get_inverse_transform(dm.model.fixed.fixed)
    f = θt -> begin
        θu = itr(θt)
        η = NoLimits._default_random_effects_from_dm(dm, NamedTuple(), θu)
        NoLimits.loglikelihood(dm, θu, η; serialization = NoLimits.EnsembleSerial())
    end
    g = ForwardDiff.gradient(f, tr(θ))
    @test all(isfinite, g)
    # the rootvalue term keeps a gradient w.r.t. the threshold for the non-firing cell
    @test abs(g.level) > 1e-8
end

@testset "estimation and plotting crossing values agree" begin
    dm = _crossing_dm()
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ηz = NoLimits._default_random_effects_from_dm(dm, NamedTuple(), θ)
    cache = NoLimits._build_ll_cache_single(dm)
    for i in 1:2
        ind = dm.individuals[i]
        pre = NoLimits.calculate_prede(dm.model, θ, ηz[i], ind.const_cov)
        acc_ll = NoLimits._ll_solve_de(dm, i, θ, ηz[i], cache, pre)
        sol, comp = NoLimits._solve_dense_individual(dm, ind, θ, ηz[i])
        acc_pl = NoLimits._sol_accessors_with_crossings(
            dm.model, sol, comp, θ, ηz[i], ind.const_cov)
        @test isapprox(acc_ll.tc, acc_pl.tc; atol = 1e-6)
        @test isapprox(acc_ll.rv, acc_pl.rv; atol = 1e-6)
    end
end
