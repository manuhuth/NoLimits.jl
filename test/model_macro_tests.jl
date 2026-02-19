using Test
using NoLimits
using ComponentArrays
using Distributions

@testset "Model macro wiring (no DE)" begin
    model = @Model begin
        @helpers begin
            add1(x) = x + 1.0
        end

        @fixedEffects begin
            a = RealNumber(1.0, prior=Normal(0.0, 10.0))
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:id)
        end

        @formulas begin
            lin = add1(a) + x.Age + η
            obs ~ Normal(lin, σ)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray((η = 0.1,))
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0,)

    obs = calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates)
    all = calculate_formulas_all(model, θ, η, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 4.1; rtol=1e-6, atol=1e-8)
    @test isapprox(all.lin, 4.1; rtol=1e-6, atol=1e-8)
end

@testset "Model macro wiring (with DE + initialDE)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            b = RealNumber(0.1)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end

        @preDifferentialEquation begin
            pre = a + b + x.Age
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + pre
        end

        @initialDE begin
            x1 = pre
        end

        @formulas begin
            y = x1(t)
            obs ~ Normal(y, 1.0)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = (x = (Age = 2.0,),)

    u0 = calculate_initial_state(model, θ, η, const_covariates_i)
    @test u0 == [θ.a + θ.b + 2.0]
end

@testset "Model macro validation" begin
    @test_throws ErrorException @eval @Model begin
        @formulas begin
            obs ~ Normal(0.0, 1.0)
        end
    end

    @test_throws LoadError @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @formulas begin
            y = a + x.Age
            obs ~ Normal(y, 1.0)
        end
        @initialDE begin
            x1 = 1.0
        end
    end

    @test_throws LoadError @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @formulas begin
            y = a + x.Age
            obs ~ Normal(y, 1.0)
        end
        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end
    end

    @test_throws LoadError @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @formulas begin
            y = a + x.Age
            obs ~ Normal(y, 1.0)
        end
        @helpers begin
            h(u) = u
        end
        @helpers begin
            h2(u) = u
        end
    end

    @test_throws LoadError @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @fixedEffects begin
            b = RealNumber(2.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @formulas begin
            y = a + x.Age
            obs ~ Normal(y, 1.0)
        end
    end

    @test_throws LoadError @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @formulas begin
            y = a + x.Age
            obs ~ Normal(y, 1.0)
        end
        @formulas begin
            y2 = a
            obs2 ~ Normal(y2, 1.0)
        end
    end

    @test_throws LoadError @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @formulas begin
            y = a + x.Age
            obs ~ Normal(y, 1.0)
        end
        @unknownBlock begin
            z = 1.0
        end
    end

    @test_throws LoadError @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @formulas begin
            y = a + x.Age
            obs ~ Normal(y, 1.0)
        end
        foo = 1 + 2
    end
end

@testset "Model macro constant_on defaults" begin
    model_single = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c = ConstantCovariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(c, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    @test model_single.covariates.covariates.params.c.constant_on == [:ID]
end

@testset "Model macro constant_on requires explicit when multiple groups" begin
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c = ConstantCovariate()
        end

        @randomEffects begin
            η1 = RandomEffect(Normal(c, 1.0); column=:ID)
            η2 = RandomEffect(Normal(c, 1.0); column=:YEAR)
        end

        @formulas begin
            y ~ Normal(η1 + η2, σ)
        end
    end
end

@testset "Model macro constant_on must include RE group" begin
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c = ConstantCovariate(; constant_on=:ID)
        end

        @randomEffects begin
            η = RandomEffect(Normal(c, 1.0); column=:YEAR)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end
end

@testset "Model runtime checks" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end
        @initialDE begin
            x1 = 1.0
        end
        @formulas begin
            y = x1(t)
            obs ~ Normal(y, 1.0)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0,)

    @test_throws ErrorException calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates)
end

@testset "Model macro hygiene" begin
    fixed = :outer_fixed
    random = :outer_random
    covariates = :outer_covariates
    helpers = :outer_helpers
    de = :outer_de
    initial = :outer_initial
    formulas = :outer_formulas

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    @test model isa Model
    @test fixed === :outer_fixed
    @test random === :outer_random
    @test covariates === :outer_covariates
    @test helpers === :outer_helpers
    @test de === :outer_de
    @test initial === :outer_initial
    @test formulas === :outer_formulas
end

@testset "Model macro auto-initializes RuntimeGeneratedFunctions in caller module" begin
    mod_name = gensym(:LDRGFInit)
    mod = Core.eval(Main, :(module $mod_name end))
    Core.eval(mod, :(using NoLimits))
    Core.eval(mod, :(using Distributions))
    ok = Core.eval(mod, quote
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(0.1)
                σ = RealNumber(0.2, scale=:log)
            end

            @covariates begin
                t = Covariate()
            end

            @formulas begin
                y ~ Normal(a, σ)
            end
        end

        model isa NoLimits.Model
    end)

    @test ok === true
end

@testset "Component macros auto-initialize RuntimeGeneratedFunctions in caller module" begin
    mod_name = gensym(:LDRGFInitParts)
    mod = Core.eval(Main, :(module $mod_name end))
    Core.eval(mod, :(using NoLimits))
    Core.eval(mod, :(using Distributions))
    ok = Core.eval(mod, quote
        re = @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        pre = @preDifferentialEquation begin
            x = 1.0
        end
        de = @DifferentialEquation begin
            D(u) ~ -u
        end

        re isa NoLimits.RandomEffects &&
        pre isa NoLimits.PreDifferentialEquation &&
        de isa NoLimits.DifferentialEquation
    end)

    @test ok === true
end
