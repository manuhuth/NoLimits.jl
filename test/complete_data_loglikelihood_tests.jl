using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using NoLimits: complete_data_loglikelihood, loglikelihood, get_random_effects,
                get_params, get_θ0_untransformed

# η ~ Normal(0, σ_η) per :ID; joint ln p(y, η | θ) has a closed form we can check.
function _cdll_model()
    @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale = :log)
            σ_η = RealNumber(0.7, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
end

const _CDLL_DF = DataFrame(ID = [:s1, :s1, :s2, :s2],
    t = [0.0, 1.0, 0.0, 1.0],
    y = [1.2, 0.9, 1.5, 1.1])

# ln p(y, η | θ) = Σ_i logpdf(Normal(a+η_i, σ), y_i) + Σ_lvl logpdf(Normal(0, σ_η), η_lvl)
function _cdll_manual(df, a, σ, ση, ηmap; ids = [:s1, :s2])
    ll = 0.0
    for r in eachrow(df)
        r.ID in ids || continue
        ll += logpdf(Normal(a + ηmap[r.ID], σ), r.y)
    end
    for lvl in ids
        ll += logpdf(Normal(0.0, ση), ηmap[lvl])
    end
    return ll
end

@testset "complete_data_loglikelihood" begin
    model = _cdll_model()
    df = _CDLL_DF
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    a, σ, ση = θ.a, θ.σ, θ.σ_η

    @testset "supplied per-level η" begin
        got = complete_data_loglikelihood(dm, θ; eta = (; η = (; s1 = 0.2, s2 = -0.1)))
        exp = _cdll_manual(df, a, σ, ση, Dict(:s1 => 0.2, :s2 => -0.1))
        @test isapprox(got, exp; rtol = 1e-10)
    end

    @testset ":mean plug-in (Normal mean = 0)" begin
        got = complete_data_loglikelihood(dm, θ; eta = :mean)
        exp = _cdll_manual(df, a, σ, ση, Dict(:s1 => 0.0, :s2 => 0.0))
        @test isapprox(got, exp; rtol = 1e-10)
    end

    @testset "individual subset" begin
        eta = (; η = (; s1 = 0.2, s2 = -0.1))
        got1 = complete_data_loglikelihood(dm, θ; eta = eta, individuals = :s1)
        exp1 = _cdll_manual(df, a, σ, ση, Dict(:s1 => 0.2, :s2 => -0.1); ids = [:s1])
        @test isapprox(got1, exp1; rtol = 1e-10)
        # subset over both ids == whole model
        got_both = complete_data_loglikelihood(dm, θ; eta = eta, individuals = [:s1, :s2])
        got_all = complete_data_loglikelihood(dm, θ; eta = eta)
        @test isapprox(got_both, got_all; rtol = 1e-12)
    end

    @testset "= data-loglik + RE-prior" begin
        got = complete_data_loglikelihood(dm, θ; eta = (; η = (; s1 = 0.2, s2 = -0.1)))
        ηvec = [ComponentArray(η = 0.2), ComponentArray(η = -0.1)]
        data_ll = loglikelihood(dm, θ, ηvec; serialization = NoLimits.EnsembleSerial())
        prior_ll = logpdf(Normal(0.0, ση), 0.2) + logpdf(Normal(0.0, ση), -0.1)
        @test isapprox(got, data_ll + prior_ll; rtol = 1e-10)
    end

    @testset "ForwardDiff gradient in θ" begin
        f = t -> complete_data_loglikelihood(dm, ComponentArray(t, getaxes(θ));
            eta = (; η = (; s1 = 0.2, s2 = -0.1)))
        g = ForwardDiff.gradient(f, collect(θ))
        @test all(isfinite, g)
        @test length(g) == length(θ)
    end

    @testset ":ebe matches joint at fitted modes" begin
        res = fit_model(dm, NoLimits.Laplace())
        re = get_random_effects(dm, res; flatten = false)
        ebe = Dict(row.ID => row.value for row in eachrow(re.η))
        θhat = get_params(res; scale = :untransformed)
        got = complete_data_loglikelihood(dm, res; eta = :ebe)
        exp = _cdll_manual(df, θhat.a, θhat.σ, θhat.σ_η, ebe)
        @test isapprox(got, exp; rtol = 1e-6)
        # stored-dm single-arg overload agrees
        @test isapprox(complete_data_loglikelihood(res; eta = :ebe), got; rtol = 1e-10)
    end

    @testset ":ebe computed from dm + θ (no fit result)" begin
        res = fit_model(dm, NoLimits.Laplace())
        θhat = get_params(res; scale = :untransformed)
        from_res = complete_data_loglikelihood(dm, res; eta = :ebe)
        # No res: EBEs are recomputed at θhat and must match the fit's modes.
        computed = complete_data_loglikelihood(dm, θhat; eta = :ebe)
        @test isapprox(computed, from_res; rtol = 1e-4)
        # optimizer options are accepted and customizable
        opts = NoLimits._default_ebe_options()
        @test isapprox(
            complete_data_loglikelihood(dm, θhat; eta = :ebe, ebe_options = opts),
            computed; rtol = 1e-8)
        # pre-fit sanity check: at the starting values, moving RE from the prior mean
        # to the per-individual EBEs cannot lower the joint density.
        θ0 = get_θ0_untransformed(dm)
        ebe0 = complete_data_loglikelihood(dm, θ0; eta = :ebe)
        @test isfinite(ebe0)
        @test ebe0 >= complete_data_loglikelihood(dm, θ0; eta = :mean)
    end

    @testset "per-individual decomposition sums to the scalar" begin
        eta = (; η = (; s1 = 0.2, s2 = -0.1))
        per = complete_data_loglikelihood_per_individual(dm, θ; eta = eta)
        @test nrow(per) == 2
        @test Set(per.ID) == Set([:s1, :s2])
        total = complete_data_loglikelihood(dm, θ; eta = eta)
        @test isapprox(sum(per.complete_data_loglikelihood), total; rtol = 1e-12)
        # one row per selected individual; matches the manual per-subject joint
        per1 = complete_data_loglikelihood_per_individual(
            dm, θ; eta = eta, individuals = :s1)
        @test nrow(per1) == 1
        @test per1.ID[1] == :s1
        exp1 = _cdll_manual(df, a, σ, ση, Dict(:s1 => 0.2, :s2 => -0.1); ids = [:s1])
        @test isapprox(per1.complete_data_loglikelihood[1], exp1; rtol = 1e-10)
        # FitResult overload agrees with the (dm, θ) form at fitted params
        res = fit_model(dm, NoLimits.Laplace())
        perr = complete_data_loglikelihood_per_individual(res; eta = :ebe)
        @test isapprox(sum(perr.complete_data_loglikelihood),
            complete_data_loglikelihood(res; eta = :ebe); rtol = 1e-10)
    end

    @testset "DataModel parameter accessors" begin
        fe = get_model(dm).fixed.fixed
        @test get_θ0_untransformed(dm) == get_θ0_untransformed(fe)
        @test get_θ0_transformed(dm) == get_θ0_transformed(fe)
        @test get_params(dm; scale = :untransformed) == get_θ0_untransformed(fe)
        @test get_params(dm; scale = :transformed) == get_θ0_transformed(fe)
        p = get_params(dm)  # :both
        @test p.untransformed == get_θ0_untransformed(fe)
        @test p.transformed == get_θ0_transformed(fe)
    end

    @testset "invalid eta is rejected" begin
        # top-level key must be an RE name, not a fixed-effect / mean name
        @test_throws ErrorException complete_data_loglikelihood(
            dm, θ; eta = (; η_mean = (; s1 = 0.2)))
        # bare symbol must be :mean or :ebe
        @test_throws ErrorException complete_data_loglikelihood(dm, θ; eta = :foo)
    end

    @testset "no random effects -> population loglik" begin
        m2 = @Model begin
            @fixedEffects begin
                a = RealNumber(0.3)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(a, σ)
            end
        end
        dm2 = DataModel(m2, df; primary_id = :ID, time_col = :t)
        θ2 = get_θ0_untransformed(m2.fixed.fixed)
        got = complete_data_loglikelihood(dm2, θ2)
        exp = loglikelihood(
            dm2, θ2, ComponentArray(); serialization = NoLimits.EnsembleSerial())
        @test isapprox(got, exp; rtol = 1e-10)
    end
end

# A normalizing-flow random effect exercises the :ebe-without-fit path on a non-Gaussian
# RE distribution (the EBE optimizer is the same machinery Laplace uses). Reuses the
# shared fx_npf fixture (a known-good NPF model).
@testset "complete_data_loglikelihood :ebe without a fit — NPF random effect" begin
    npf_dm = fx_npf_dm()
    θ0 = get_θ0_untransformed(npf_dm)
    ebe = complete_data_loglikelihood(npf_dm, θ0; eta = :ebe)
    @test isfinite(ebe)
    @test ebe >= complete_data_loglikelihood(npf_dm, θ0; eta = :mean)
    # per-individual breakdown sums to the scalar
    per = complete_data_loglikelihood_per_individual(npf_dm, θ0; eta = :ebe)
    @test isapprox(sum(per.complete_data_loglikelihood), ebe; rtol = 1e-8)
end
