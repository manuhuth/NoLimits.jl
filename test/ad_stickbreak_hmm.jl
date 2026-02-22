using Test
using NoLimits
using ComponentArrays
using DataFrames
using Distributions
using ForwardDiff
using FiniteDifferences
using Random

# ---------------------------------------------------------------------------
# Gradient tests: ProbabilityVector and DiscreteTransitionMatrix
# with DiscreteTimeDiscreteStatesHMM / MVDiscreteTimeDiscreteStatesHMM
# ---------------------------------------------------------------------------
@testset "AD gradients – stickbreak params + HMM" begin

    # -----------------------------------------------------------------------
    # Helper: finite-difference gradient via central differences
    # -----------------------------------------------------------------------
    function fd_grad(f, x; h=1e-5)
        g = similar(x, Float64)
        for i in eachindex(x)
            xp, xm = copy(x), copy(x)
            xp[i] += h; xm[i] -= h
            g[i] = (f(xp) - f(xm)) / (2h)
        end
        return g
    end

    # -----------------------------------------------------------------------
    # 1. ProbabilityVector (k=3) → gradient non-zero via loglikelihood
    # -----------------------------------------------------------------------
    @testset "ProbabilityVector k=3: ForwardDiff gradient non-zero" begin
        model = @Model begin
            @fixedEffects begin
                pi  = ProbabilityVector([0.5, 0.3, 0.2]; calculate_se=true)
                mu1 = RealNumber(0.0; calculate_se=true)
                mu2 = RealNumber(1.0; calculate_se=true)
                mu3 = RealNumber(-1.0; calculate_se=true)
                sig = RealNumber(0.5; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ DiscreteTimeDiscreteStatesHMM(
                    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],  # identity (fixed) transition
                    (Normal(mu1, sig), Normal(mu2, sig), Normal(mu3, sig)),
                    Categorical(pi)
                )
            end
        end

        df = DataFrame(
            ID=vcat(fill(1, 6), fill(2, 6)),
            t=vcat(1:6, 1:6) .* 1.0,
            y=[0.1, 0.9, -1.1, 0.2, 1.0, -0.9,
               0.05, 0.85, -1.05, 0.15, 0.95, -0.95]
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)

        fe = dm.model.fixed.fixed
        inv_transform = get_inverse_transform(fe)
        θ0_t = get_θ0_transformed(fe)
        axs = getaxes(θ0_t)

        function ll_obj(θt_vec)
            θt = ComponentArray(θt_vec, axs)
            θu = inv_transform(θt)
            return NoLimits.loglikelihood(dm, θu, ComponentArray())
        end

        θt_vec = collect(θ0_t)
        # ForwardDiff gradient
        grad_fd = ForwardDiff.gradient(ll_obj, θt_vec)

        # Finite-difference gradient for comparison
        grad_num = fd_grad(ll_obj, θt_vec)

        # The pi parameters are at indices 1,2 (k-1=2 stickbreak params)
        pi_idx = 1:2
        other_idx = 3:length(θt_vec)

        @test !all(iszero, grad_fd)
        # pi gradient components should be non-zero (data distinguishes states)
        @test any(!iszero, grad_fd[pi_idx])
        # ForwardDiff and finite differences should agree
        @test isapprox(grad_fd, grad_num; atol=1e-4, rtol=1e-3)
    end

    # -----------------------------------------------------------------------
    # 2. DiscreteTransitionMatrix (n=2) → gradient non-zero via loglikelihood
    # -----------------------------------------------------------------------
    @testset "DiscreteTransitionMatrix n=2: ForwardDiff gradient non-zero" begin
        model = @Model begin
            @fixedEffects begin
                T   = DiscreteTransitionMatrix([0.8 0.2; 0.3 0.7]; calculate_se=true)
                mu1 = RealNumber(0.0; calculate_se=true)
                mu2 = RealNumber(2.0; calculate_se=true)
                sig = RealNumber(0.3; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ DiscreteTimeDiscreteStatesHMM(
                    T,
                    (Normal(mu1, sig), Normal(mu2, sig)),
                    Categorical([0.5, 0.5])
                )
            end
        end

        rng = MersenneTwister(42)
        # Generate data with clear switching: [0,0,0,2,2,2] pattern
        df = DataFrame(
            ID=vcat(fill(1, 6), fill(2, 6)),
            t=vcat(1:6, 1:6) .* 1.0,
            y=[0.1, -0.1, 0.2, 1.9, 2.1, 1.8,
               -0.2, 0.1, 0.0, 2.0, 1.95, 2.05]
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)

        fe = dm.model.fixed.fixed
        inv_transform = get_inverse_transform(fe)
        θ0_t = get_θ0_transformed(fe)
        axs = getaxes(θ0_t)

        function ll_obj(θt_vec)
            θt = ComponentArray(θt_vec, axs)
            θu = inv_transform(θt)
            return NoLimits.loglikelihood(dm, θu, ComponentArray())
        end

        θt_vec = collect(θ0_t)
        grad_fd = ForwardDiff.gradient(ll_obj, θt_vec)
        grad_num = fd_grad(ll_obj, θt_vec)

        # T has n*(n-1)=2 transformed params (indices 1,2)
        T_idx = 1:2
        @test !all(iszero, grad_fd)
        @test any(!iszero, grad_fd[T_idx])
        @test isapprox(grad_fd, grad_num; atol=1e-4, rtol=1e-3)
    end

    # -----------------------------------------------------------------------
    # 3. ProbabilityVector AND DiscreteTransitionMatrix together
    # -----------------------------------------------------------------------
    @testset "ProbabilityVector + DiscreteTransitionMatrix: both grads non-zero" begin
        model = @Model begin
            @fixedEffects begin
                pi  = ProbabilityVector([0.6, 0.4]; calculate_se=true)
                T   = DiscreteTransitionMatrix([0.9 0.1; 0.2 0.8]; calculate_se=true)
                mu1 = RealNumber(0.0; calculate_se=true)
                mu2 = RealNumber(3.0; calculate_se=true)
                sig = RealNumber(0.5; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ DiscreteTimeDiscreteStatesHMM(
                    T,
                    (Normal(mu1, sig), Normal(mu2, sig)),
                    Categorical(pi)
                )
            end
        end

        df = DataFrame(
            ID=vcat(fill(1, 8), fill(2, 8)),
            t=vcat(1:8, 1:8) .* 1.0,
            y=[0.1, 0.2, -0.1, 2.9, 3.1, 2.8, 0.0, 0.1,
               -0.1, 0.0, 0.2, 3.0, 2.95, 3.05, 0.1, -0.1]
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)

        fe = dm.model.fixed.fixed
        inv_transform = get_inverse_transform(fe)
        θ0_t = get_θ0_transformed(fe)
        axs = getaxes(θ0_t)

        function ll_obj(θt_vec)
            θt = ComponentArray(θt_vec, axs)
            θu = inv_transform(θt)
            return NoLimits.loglikelihood(dm, θu, ComponentArray())
        end

        θt_vec = collect(θ0_t)
        grad_fd = ForwardDiff.gradient(ll_obj, θt_vec)
        grad_num = fd_grad(ll_obj, θt_vec)

        # pi has k-1=1 param (index 1), T has n*(n-1)=2 params (indices 2,3)
        pi_idx = 1:1
        T_idx  = 2:3

        @test !all(iszero, grad_fd)
        @test any(!iszero, grad_fd[pi_idx])
        @test any(!iszero, grad_fd[T_idx])
        @test isapprox(grad_fd, grad_num; atol=1e-4, rtol=1e-3)
    end

    # -----------------------------------------------------------------------
    # 4. Verify MLE convergence with HMM + ProbabilityVector
    # -----------------------------------------------------------------------
    @testset "MLE converges: ProbabilityVector + HMM" begin
        model = @Model begin
            @fixedEffects begin
                pi  = ProbabilityVector([0.5, 0.5]; calculate_se=false)
                T   = DiscreteTransitionMatrix([0.8 0.2; 0.3 0.7]; calculate_se=false)
                mu1 = RealNumber(0.0; calculate_se=false)
                mu2 = RealNumber(2.0; calculate_se=false)
                sig = RealNumber(0.5; scale=:log, calculate_se=false)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ DiscreteTimeDiscreteStatesHMM(
                    T,
                    (Normal(mu1, sig), Normal(mu2, sig)),
                    Categorical(pi)
                )
            end
        end

        df = DataFrame(
            ID=vcat(fill(1, 8), fill(2, 8)),
            t=vcat(1:8, 1:8) .* 1.0,
            y=[0.1, 0.2, -0.1, 1.9, 2.1, 1.8, 0.0, 0.15,
               -0.2, 0.1, 2.0, 1.95, 2.05, 0.05, -0.1, 0.2]
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)

        res = fit_model(dm, NoLimits.MLE())
        @test NoLimits.get_converged(res)

        params = NoLimits.get_params(res; scale=:untransformed)
        # pi should still be a valid probability vector
        @test isapprox(sum(params.pi), 1.0; atol=1e-4)
        @test all(params.pi .>= 0)
        # T should still be row-stochastic
        @test isapprox(sum(params.T[1, :]), 1.0; atol=1e-4)
        @test isapprox(sum(params.T[2, :]), 1.0; atol=1e-4)
    end

end
