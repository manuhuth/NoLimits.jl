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
            a = RealNumber(1.0, prior = Normal(0.0, 10.0))
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :id)
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
    @test isapprox(mean(obs.obs), 4.1; rtol = 1.0e-6, atol = 1.0e-8)
    @test isapprox(all.lin, 4.1; rtol = 1.0e-6, atol = 1.0e-8)
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

    # Unknown DE symbols used to surface only on the first ODE solve (#314).
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x1) ~ -aa * x1
        end
        @initialDE begin
            x1 = 1.0
        end
        @formulas begin
            obs ~ Normal(x1(t), 1.0)
        end
    end

    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x1) ~ -sqr(a) * x1
        end
        @initialDE begin
            x1 = 1.0
        end
        @formulas begin
            obs ~ Normal(x1(t), 1.0)
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

    # `eta`/`beta`/`gamma`/`zeta`/`digamma` are SpecialFunctions re-exports and used to
    # pass the undefined-symbol check (#312).
    @test_throws "undefined symbol(s) eta" @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
        end
        @formulas begin
            y ~ Normal(a + eta, 1.0)
        end
    end

    # A misspelled random-effect distribution used to fail only at fit time (#312).
    @test_throws "calls undefined function(s) Normall" @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            om = RealNumber(0.3, scale = :log)
        end
        @randomEffects begin
            b = RandomEffect(Normall(0.0, om); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + b, 1.0)
        end
    end

    # An outcome name colliding with a fixed effect used to resolve to the fixed effect (#312).
    @test_throws "ambiguous in @formulas" @Model begin
        @fixedEffects begin
            y = RealNumber(0.7)
        end
        @formulas begin
            y ~ Normal(y, 1.0)
        end
    end

    # A pre-DE output used to silently shadow a same-named fixed effect (#312).
    @test_throws "collide with the fixed effect" @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            sig = RealNumber(0.4, scale = :log)
        end
        @preDifferentialEquation begin
            a = 99.0
        end
        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end
        @initialDE begin
            x1 = 1.0
        end
        @formulas begin
            y ~ Normal(x1(t), sig)
        end
    end

    # A length-1 network called with a bare scalar used to escape the checker (#316).
    @test_throws "got the scalar `t`" @Model begin
        @fixedEffects begin
            sig = RealNumber(0.4, scale = :log)
            zeta = FFNNParameters((1, 4, 1); function_name = :NN1)
        end
        @formulas begin
            lin = NN1(t, zeta)[1]
            obs ~ Normal(lin, sig)
        end
    end

    # Output-index bounds were only checked for soft trees (#316).
    @test_throws "index [2] is out of range" @Model begin
        @fixedEffects begin
            sig = RealNumber(0.4, scale = :log)
            zeta = FFNNParameters((1, 4, 1); function_name = :NN1)
        end
        @formulas begin
            lin = NN1([t], zeta)[2]
            obs ~ Normal(lin, sig)
        end
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
            η = RandomEffect(Normal(c, 1.0); column = :ID)
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
            η1 = RandomEffect(Normal(c, 1.0); column = :ID)
            η2 = RandomEffect(Normal(c, 1.0); column = :YEAR)
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
            c = ConstantCovariate(; constant_on = :ID)
        end

        @randomEffects begin
            η = RandomEffect(Normal(c, 1.0); column = :YEAR)
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

    @test_throws ErrorException calculate_formulas_obs(
        model, θ, η, const_covariates_i, varying_covariates
    )
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
    ok = Core.eval(
        mod, quote
            model = @Model begin
                @fixedEffects begin
                    a = RealNumber(0.1)
                    σ = RealNumber(0.2, scale = :log)
                end

                @covariates begin
                    t = Covariate()
                end

                @formulas begin
                    y ~ Normal(a, σ)
                end
            end

            model isa NoLimits.Model
        end
    )

    @test ok === true
end

@testset "Component macros auto-initialize RuntimeGeneratedFunctions in caller module" begin
    mod_name = gensym(:LDRGFInitParts)
    mod = Core.eval(Main, :(module $mod_name end))
    Core.eval(mod, :(using NoLimits))
    Core.eval(mod, :(using Distributions))
    ok = Core.eval(
        mod,
        quote
            re = @randomEffects begin
                η = RandomEffect(Normal(0.0, 1.0); column = :ID)
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
        end
    )

    @test ok === true
end

@testset "Model type is identical across expansion modules" begin
    body = quote
        @Model begin
            @fixedEffects begin
                a = RealNumber(0.3)
                σ = RealNumber(0.4, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, 1.0); column = :ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -a * exp(η) * x1
                sig(t) = 2.0 * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t) + 0.0 * sig(t), σ)
            end
        end
    end
    types = map(1:2) do _
        mod = Core.eval(Main, :(module $(gensym(:LDTypeStable)) end))
        Core.eval(mod, :(using NoLimits))
        Core.eval(mod, :(using Distributions))
        typeof(Core.eval(mod, body))
    end
    @test types[1] === types[2]
end
