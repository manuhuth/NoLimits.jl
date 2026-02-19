using Test
using NoLimits
using ComponentArrays
using Distributions

struct FakeSol end

@inline (s::FakeSol)(t; idxs) = t + idxs

@testset "Formulas basic" begin
    formulas = @formulas begin
        lin = a + b + x.Age
        obs ~ Normal(lin, σ)
    end
    ir = get_formulas_ir(formulas)
    @test ir.det_names == [:lin]
    @test ir.obs_names == [:obs]

    (form_all, form_obs, req_states, req_signals) = get_formulas_builders(
        formulas;
        fixed_names = [:a, :b, :σ],
        const_cov_names = [:x]
    )
    @test req_states == Symbol[]
    @test req_signals == Symbol[]

    ctx = (;
        fixed_effects = (a = 1.0, b = 2.0, σ = 0.5),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = (x = (Age = 3.0,),)
    varying_covariates = (t = 0.0,)
    sol_accessors = NamedTuple()

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 6.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.lin, 6.0; rtol=1e-6, atol=1e-8)
end

@testset "Formulas helpers and model_funs" begin
    formulas = @formulas begin
        y = helper(x.Age) + NN1([x.Age], ζ)[1]
        obs ~ Normal(y, σ)
    end
    ir = get_formulas_ir(formulas)
    @test ir.det_names == [:y]
    @test ir.obs_names == [:obs]

    (form_all, form_obs, _, _) = get_formulas_builders(
        formulas;
        fixed_names = [:σ, :ζ],
        const_cov_names = [:x],
        helper_names = [:helper],
        model_fun_names = [:NN1]
    )

    NN1(x, ζ) = [x[1] + ζ]
    helpers = @helpers begin
        helper(x) = x + 1.0
    end

    ctx = (;
        fixed_effects = (σ = 0.2, ζ = 1.0),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = helpers,
        model_funs = (NN1 = NN1,)
    )
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0,)
    sol_accessors = NamedTuple()

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 6.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.y, 6.0; rtol=1e-6, atol=1e-8)
end

@testset "Formulas varying covariates" begin
    formulas = @formulas begin
        z = w1 + w2(t) + x.Age
        obs ~ Normal(z, 1.0)
    end
    ir = get_formulas_ir(formulas)
    @test ir.det_names == [:z]
    @test ir.obs_names == [:obs]

    (form_all, form_obs, _, _) = get_formulas_builders(
        formulas;
        const_cov_names = [:x],
        varying_cov_names = [:w1, :w2]
    )

    ctx = (;
        fixed_effects = NamedTuple(),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 3.0, w1 = 1.0, w2 = (t -> 2.0 * t))
    sol_accessors = NamedTuple()

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 9.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.z, 9.0; rtol=1e-6, atol=1e-8)
end

@testset "Formulas with DE accessors" begin
    formulas = @formulas begin
        y = x1(t) + s(t)
        obs ~ Normal(y, 1.0)
    end
    ir = get_formulas_ir(formulas)
    @test ir.det_names == [:y]
    @test ir.obs_names == [:obs]

    (form_all, form_obs, req_states, req_signals) = get_formulas_builders(
        formulas;
        state_names = [:x1],
        signal_names = [:s]
    )
    @test req_states == [:x1]
    @test req_signals == [:s]

    sol_accessors = (x1 = (t -> t + 1.0), s = (t -> 2.0 * t))
    ctx = (;
        fixed_effects = NamedTuple(),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = NamedTuple()
    varying_covariates = (t = 1.0,)

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 4.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.y, 4.0; rtol=1e-6, atol=1e-8)
end

@testset "Formulas multiple deterministics and observations" begin
    formulas = @formulas begin
        d1 = a + x.Age
        d2 = d1 + b + w1
        y ~ Normal(d2, σ)
        y2 ~ Normal(d1, 1.0)
    end

    (form_all, form_obs, _, _) = get_formulas_builders(
        formulas;
        fixed_names = [:a, :b, :σ],
        const_cov_names = [:x],
        varying_cov_names = [:w1]
    )

    ctx = (;
        fixed_effects = (a = 1.0, b = 2.0, σ = 0.5),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = (x = (Age = 3.0,),)
    varying_covariates = (t = 0.0, w1 = 4.0)
    sol_accessors = NamedTuple()

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.y isa Normal
    @test obs.y2 isa Normal
    @test isapprox(mean(obs.y), 10.0; rtol=1e-6, atol=1e-8)
    @test isapprox(mean(obs.y2), 4.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.d1, 4.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.d2, 10.0; rtol=1e-6, atol=1e-8)
end

@testset "Formulas mixed constant and varying covariates" begin
    formulas = @formulas begin
        u = x.Age + w1 + w2(t) + w3(t)
        obs ~ Normal(u, 1.0)
    end

    (form_all, form_obs, _, _) = get_formulas_builders(
        formulas;
        const_cov_names = [:x],
        varying_cov_names = [:w1, :w2, :w3]
    )

    ctx = (;
        fixed_effects = NamedTuple(),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 2.0, w1 = 1.0, w2 = (t -> 2.0 * t), w3 = (t -> t + 1.0))
    sol_accessors = NamedTuple()

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 10.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.u, 10.0; rtol=1e-6, atol=1e-8)
end

@testset "Formulas helpers, model_funs, and random effects" begin
    formulas = @formulas begin
        y = helper(a) + NN1([x.Age], ζ)[1] + η
        obs ~ Normal(y, σ)
    end

    (form_all, form_obs, _, _) = get_formulas_builders(
        formulas;
        fixed_names = [:a, :σ, :ζ],
        random_names = [:η],
        const_cov_names = [:x],
        helper_names = [:helper],
        model_fun_names = [:NN1]
    )

    NN1(x, ζ) = [x[1] + ζ]
    helpers = @helpers begin
        helper(x) = x + 1.0
    end

    ctx = (;
        fixed_effects = (a = 1.0, σ = 0.5, ζ = 2.0),
        random_effects = (η = 0.25,),
        prede = NamedTuple(),
        helpers = helpers,
        model_funs = (NN1 = NN1,)
    )
    const_covariates_i = (x = (Age = 3.0,),)
    varying_covariates = (t = 0.0,)
    sol_accessors = NamedTuple()

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 7.25; rtol=1e-6, atol=1e-8)
    @test isapprox(all.y, 7.25; rtol=1e-6, atol=1e-8)
end

@testset "Formulas with DE accessors and covariates" begin
    formulas = @formulas begin
        y = x1(t) + s(t) + x.Age + w1
        obs ~ Normal(y, σ)
    end

    (form_all, form_obs, req_states, req_signals) = get_formulas_builders(
        formulas;
        fixed_names = [:σ],
        const_cov_names = [:x],
        varying_cov_names = [:w1],
        state_names = [:x1],
        signal_names = [:s]
    )
    @test req_states == [:x1]
    @test req_signals == [:s]

    sol_accessors = (x1 = (t -> t + 1.0), s = (t -> 2.0 * t))
    ctx = (;
        fixed_effects = (σ = 0.5,),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 1.0, w1 = 3.0)

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isapprox(mean(obs.obs), 9.0; rtol=1e-6, atol=1e-8)
    @test isapprox(all.y, 9.0; rtol=1e-6, atol=1e-8)
end

@testset "Formulas with HMM outcome" begin
    formulas = @formulas begin
        outcome_1_current ~ ContinuousTimeDiscreteStatesHMM(
            [-λ12  λ12;
              λ21 -λ21],
            (Bernoulli(p1), Bernoulli(p2)),
            Categorical([0.6, 0.4]),
            delta_t
        )
    end

    (form_all, form_obs, _, _) = get_formulas_builders(
        formulas;
        fixed_names = [:λ12, :λ21, :p1, :p2],
        varying_cov_names = [:delta_t]
    )

    ctx = (;
        fixed_effects = (λ12 = 0.2, λ21 = 0.3, p1 = 0.25, p2 = 0.75),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = NamedTuple()
    varying_covariates = (t = 0.0, delta_t = 1.0)
    sol_accessors = NamedTuple()

    obs = form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
    all = form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    @test obs.outcome_1_current isa ContinuousTimeDiscreteStatesHMM
    @test all.outcome_1_current isa ContinuousTimeDiscreteStatesHMM
    @test isfinite(logpdf(obs.outcome_1_current, 1.0))
end
