using Test
using OrdinaryDiffEq
using SciMLSensitivity
using ForwardDiff
using FiniteDifferences
using Lux
using NoLimits

@testset "ODE forward sensitivity" begin
    # Simple linear ODE with parameter sensitivity check.
    function rhs!(du, u, p, t)
        du[1] = -p[1] * u[1]
    end
    u0 = [1.0]
    p = [0.7]
    tspan = (0.0, 0.5)

    prob = ODEForwardSensitivityProblem(rhs!, u0, tspan, p)
    sol = solve(prob, Tsit5(); abstol=1e-9, reltol=1e-9)
    x, dp = extract_local_sensitivities(sol, length(sol.u), Val(true))
    @test length(x) == 1
    @test size(dp) == (1, 1)
    @test isfinite(dp[1, 1])

    # Compare to ForwardDiff gradient of solution at final time.
    g(pv) = begin
        probp = ODEProblem(rhs!, u0, tspan, pv)
        solp = solve(probp, Tsit5(); abstol=1e-9, reltol=1e-9)
        solp.u[end][1]
    end
    grad_fd = ForwardDiff.gradient(g, p)
    @test isapprox(dp[1, 1], grad_fd[1]; rtol=1e-4, atol=1e-6)

    # Hessian via ForwardDiff Jacobian of forward sensitivities.
    function sens_final(pv)
        probp = ODEForwardSensitivityProblem(rhs!, u0, tspan, pv)
        solp = solve(probp, Tsit5();
                     sensealg=ForwardSensitivity(autodiff=false, autojacvec=true),
                     abstol=1e-9, reltol=1e-9)
        _, dp_local = extract_local_sensitivities(solp, length(solp.u), Val(true))
        vec(dp_local)
    end
    hess_fd = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1), sens_final, p)[1]
    @test size(hess_fd) == (length(p), length(p))
    @test all(isfinite, hess_fd)
end

@testset "ODE forward sensitivity with model functions" begin
    # Forward sensitivity with a DSL-generated RHS using model_funs and preDE.
    de = @DifferentialEquation begin
        D(x1) ~ -a * x1 + NN([x1], ζ)[1] + pre
    end
    fe = @fixedEffects begin
        a = RealNumber(0.5)
        ζ = NNParameters(Chain(Dense(1, 2, tanh), Dense(2, 1));
                         function_name=:NN, calculate_se=false)
    end
    prede = @preDifferentialEquation begin
        pre = a
    end
    fe0 = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)
    pre = get_prede_builder(prede)(fe0, ComponentArray(), NamedTuple(), model_funs, NamedTuple())
    p = (; fixed_effects = fe0,
          random_effects = ComponentArray(),
          constant_covariates = NamedTuple(),
          varying_covariates = NamedTuple(),
          helpers = NamedTuple(),
          model_funs = model_funs,
          preDE = pre)
    pc = get_de_compiler(de)(p)
    de_rhs! = get_de_f!(de)

    u0 = [0.2]
    tspan = (0.0, 0.5)
    pvec = [fe0.a]  # only a is tunable here

    function rhs_p!(du, u, pθ, t)
        fe_local = ComponentArray(a=pθ[1], ζ=fe0.ζ)
        pre_local = get_prede_builder(prede)(fe_local, ComponentArray(), NamedTuple(), model_funs, NamedTuple())
        p_local = (; fixed_effects = fe_local,
                   random_effects = ComponentArray(),
                   constant_covariates = NamedTuple(),
                   varying_covariates = NamedTuple(),
                   helpers = NamedTuple(),
                   model_funs = model_funs,
                   preDE = pre_local)
        pc_local = get_de_compiler(de)(p_local)
        de_rhs!(du, u, pc_local, t)
    end

    prob = ODEForwardSensitivityProblem(rhs_p!, u0, tspan, pvec)
    sol = solve(prob, Tsit5(); abstol=1e-9, reltol=1e-9)
    x, dp = extract_local_sensitivities(sol, length(sol.u), Val(true))
    @test length(x) == 1
    @test size(dp) == (1, 1)
    @test isfinite(dp[1, 1])
end

@testset "ODE forward sensitivity with spline and softtree" begin
    # Forward sensitivity through DSL RHS using spline + softtree model functions.
    knots = collect(range(0.0, 1.0; length=5))
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + SP1(t, sp) + ST([x1, cons], Γ)[1]
    end
    fe = @fixedEffects begin
        a = RealNumber(0.3)
        Γ = SoftTreeParameters(2, 2; function_name=:ST, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
    end
    prede = @preDifferentialEquation begin
        cons = a
    end
    fe0 = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)

    u0 = [0.15]
    tspan = (0.0, 0.4)
    pvec = collect(fe0.a)
    de_rhs! = get_de_f!(de)

    function rhs_p!(du, u, pθ, t)
        fe_local = ComponentArray(a=pθ[1], Γ=fe0.Γ, sp=fe0.sp)
        pre_local = get_prede_builder(prede)(
            fe_local, ComponentArray(), NamedTuple(), model_funs, NamedTuple()
        )
        p_local = (; fixed_effects = fe_local,
                   random_effects = ComponentArray(),
                   constant_covariates = NamedTuple(),
                   varying_covariates = NamedTuple(),
                   helpers = NamedTuple(),
                   model_funs = model_funs,
                   preDE = pre_local)
        pc_local = get_de_compiler(de)(p_local)
        de_rhs!(du, u, pc_local, t)
    end

    prob = ODEForwardSensitivityProblem(rhs_p!, u0, tspan, pvec)
    sol = solve(prob, Tsit5(); abstol=1e-9, reltol=1e-9)
    x, dp = extract_local_sensitivities(sol, length(sol.u), Val(true))
    @test length(x) == 1
    @test size(dp) == (1, 1)
    @test isfinite(dp[1, 1])
end
