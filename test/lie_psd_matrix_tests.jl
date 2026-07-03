using Test
using NoLimits
using ComponentArrays
using LinearAlgebra
using Distributions
using ForwardDiff
using Random

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
function _rand_spd(n, rng)
    M = randn(rng, n, n)
    return Matrix(Symmetric(M * M' + 0.5I))
end

# ---------------------------------------------------------------------------
# 1. Constructor tests
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix construction" begin
    @testset "from full matrix" begin
        Σ = [4.0 -1.3; -1.3 1.75]
        p = RealLiePSDMatrix(Σ; name = :Ω, eigenvalue_lower = 1e-3,
            eigenvalue_upper = 1e3, calculate_se = true)
        @test p.name == :Ω
        @test p.scale == :lie
        @test p.prior isa Priorless
        @test p.calculate_se == true
        @test p.value == Σ
        @test p.eigenvalue_lower == [1e-3, 1e-3]
        @test p.eigenvalue_upper == [1e3, 1e3]
    end

    @testset "scalar bounds broadcast; vector bounds honored" begin
        p1 = RealLiePSDMatrix([2.0 0.0; 0.0 3.0]; eigenvalue_lower = 0.1)
        @test p1.eigenvalue_lower == [0.1, 0.1]
        @test p1.eigenvalue_upper == [Inf, Inf]
        p2 = RealLiePSDMatrix([2.0 0.0; 0.0 3.0]; eigenvalue_lower = [0.1, 0.2],
            eigenvalue_upper = [10.0, 20.0])
        @test p2.eigenvalue_lower == [0.1, 0.2]
        @test p2.eigenvalue_upper == [10.0, 20.0]
    end

    @testset "from (log_eigenvalues, angles)" begin
        p = RealLiePSDMatrix(; log_eigenvalues = [0.0, 0.7], angles = [0.3], name = :Ω)
        @test size(p.value) == (2, 2)
        @test issymmetric(p.value)
        @test minimum(eigen(Symmetric(p.value)).values) > 0
        # Recovering (λ, α) and re-expanding reproduces the same matrix.
        t = liepsd_forward(p.value)
        @test isapprox(liepsd_inverse(t), p.value; atol = 1e-10)
    end

    @testset "n = 1 edge (no angles)" begin
        p = RealLiePSDMatrix(reshape([2.5], 1, 1))
        @test size(p.value) == (1, 1)
        p2 = RealLiePSDMatrix(; log_eigenvalues = [log(2.5)], angles = Float64[])
        @test isapprox(p2.value[1, 1], 2.5; atol = 1e-12)
    end

    @testset "invalid inputs error" begin
        @test_throws ErrorException RealLiePSDMatrix([1.0 2.0; 0.0 1.0])   # not PSD
        @test_throws ErrorException RealLiePSDMatrix([1.0 0.0 0.0; 0.0 1.0 0.0])  # non-square
        @test_throws ErrorException RealLiePSDMatrix([1.0 0.0; 0.0 1.0];
            scale = :cholesky)                                            # wrong scale
        @test_throws ErrorException RealLiePSDMatrix([1.0 0.0; 0.0 1.0];
            eigenvalue_lower = -1.0)                                      # negative lower
        @test_throws ErrorException RealLiePSDMatrix([1.0 0.0; 0.0 1.0];
            eigenvalue_lower = 5.0, eigenvalue_upper = 1.0)               # lower > upper
        @test_throws ErrorException RealLiePSDMatrix(; log_eigenvalues = [0.0, 0.0],
            angles = [0.1, 0.2])                                         # wrong angle count
    end
end

# ---------------------------------------------------------------------------
# 2. Transform round-trip: liepsd_inverse ∘ liepsd_forward ≈ id
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix transform round-trip" begin
    rng = MersenneTwister(20200525)

    @testset "diagonal (α = 0)" begin
        Σ = [2.0 0.0; 0.0 3.0]
        t = liepsd_forward(Σ)
        @test t[3] == 0.0                       # single angle is zero
        @test isapprox(t[1:2], log.([2.0, 3.0]); atol = 1e-14)
        @test isapprox(liepsd_inverse(t), Σ; atol = 1e-14)
    end

    @testset "thesis example (Fig. 5.6)" begin
        Σ = [3.25 -1.30; -1.30 1.75]
        t = liepsd_forward(Σ)
        @test isapprox(liepsd_inverse(t), Σ; atol = 1e-12)
        # Eigenvalues are ≈ 1 and 4 (thesis Fig. 5.6 rounds them), so λ ≈ log([1, 4]).
        @test isapprox(sort(exp.(t[1:2])), sort(eigen(Symmetric(Σ)).values); atol = 1e-10)
        @test isapprox(sort(exp.(t[1:2])), [1.0, 4.0]; atol = 1e-2)
    end

    @testset "random SPD, n = 2..5 (non-diagonal, det-fix branch)" begin
        for n in 2:5
            for _ in 1:5
                Σ = _rand_spd(n, rng)
                t = liepsd_forward(Σ)
                @test length(t) == n * (n + 1) ÷ 2
                @test isapprox(liepsd_inverse(t), Σ; atol = 1e-8)
                @test issymmetric(liepsd_inverse(t))
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 3. ForwardDiff through the inverse map (incl. repeated eigenvalues at Λ = I)
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix ForwardDiff correctness" begin
    rng = MersenneTwister(1)
    n = 3
    L = n * (n + 1) ÷ 2

    @testset "gradient finite & matches finite differences" begin
        for _ in 1:3
            t0 = vcat(0.5 .* randn(rng, n), 0.4 .* randn(rng, n * (n - 1) ÷ 2))
            W = Symmetric(randn(rng, n, n)) |> Matrix
            f = t -> sum(liepsd_inverse(t) .* W)
            g = ForwardDiff.gradient(f, t0)
            @test all(isfinite, g)
            gfd = similar(g)
            ε = 1e-6
            for k in 1:L
                tp = copy(t0)
                tp[k] += ε
                tm = copy(t0)
                tm[k] -= ε
                gfd[k] = (f(tp) - f(tm)) / (2ε)
            end
            @test isapprox(g, gfd; atol = 1e-6)
        end
    end

    @testset "finite & symmetric at Λ = I (repeated eigenvalues)" begin
        t0 = vcat(zeros(n), 0.3 .* randn(rng, n * (n - 1) ÷ 2))
        J = ForwardDiff.jacobian(liepsd_inverse, t0)
        @test all(isfinite, J)
        # Dual output must be exactly symmetric in value and partials.
        td = [ForwardDiff.Dual{:t}(t0[i], ntuple(k -> k == i ? 1.0 : 0.0, L)...)
              for i in 1:L]
        Σd = liepsd_inverse(td)
        for i in 1:n, j in 1:n
            @test ForwardDiff.value(Σd[i, j]) == ForwardDiff.value(Σd[j, i])
            @test ForwardDiff.partials(Σd[i, j]) == ForwardDiff.partials(Σd[j, i])
        end
    end
end

# ---------------------------------------------------------------------------
# 4. Jacobianᵀ (`_inv_jac_spec_val`) matches the transform Jacobian
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix Jacobianᵀ" begin
    using NoLimits: _inv_jac_spec_val, TransformSpec
    rng = MersenneTwister(7)
    n = 3
    spec = TransformSpec(:Ω, :lie, (n, n), nothing)
    for _ in 1:4
        t = vcat(0.5 .* randn(rng, n), 0.4 .* randn(rng, n * (n - 1) ÷ 2))
        G = randn(rng, n, n)                       # raw natural gradient (any G)
        gt = _inv_jac_spec_val(spec, t, G)
        J = ForwardDiff.jacobian(liepsd_inverse, collect(t))
        @test isapprox(gt, J' * vec(G); atol = 1e-9)
    end
end

# ---------------------------------------------------------------------------
# 5. @fixedEffects integration: spec, bounds, flat names
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix @fixedEffects integration" begin
    fe = @fixedEffects begin
        Ω = RealLiePSDMatrix([1.0 0.0; 0.0 1.0]; eigenvalue_lower = 1e-3,
            eigenvalue_upper = 1e3, calculate_se = true)
    end

    specs = get_transforms(fe).forward.specs
    @test specs[1].kind == :lie
    @test specs[1].size == (2, 2)

    # Identity matrix → log-eigenvalues 0, angle 0.
    θ0t = get_θ0_transformed(fe)
    @test isapprox(collect(θ0t[:Ω]), [0.0, 0.0, 0.0]; atol = 1e-14)

    # Untransformed round-trip is the identity matrix.
    θ0u = get_θ0_untransformed(fe)
    @test isapprox(Matrix(θ0u[:Ω]), [1.0 0.0; 0.0 1.0]; atol = 1e-14)

    # Transformed bounds: eigenvalue box on the λ block, unbounded α.
    lo, up = get_bounds_transformed(fe)
    @test isapprox(collect(lo[:Ω]), [log(1e-3), log(1e-3), -Inf]; atol = 1e-12)
    @test isapprox(collect(up[:Ω]), [log(1e3), log(1e3), Inf]; atol = 1e-12)

    # SE mask covers all n(n+1)/2 transformed entries.
    @test count(get_se_mask(fe)) == 3
end

# ---------------------------------------------------------------------------
# 6. UQ natural-scale coordinates (upper triangle, same count as transformed)
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix UQ coords (natural scale)" begin
    using NoLimits: _coords_for_param

    fe = @fixedEffects begin
        Ω = RealLiePSDMatrix([2.0 0.5; 0.5 1.0]; calculate_se = true)
    end
    θ0u = get_θ0_untransformed(fe)
    spec = get_transforms(fe).forward.specs[1]

    coords_nat = _coords_for_param(Matrix(θ0u[:Ω]), spec; natural = true)
    @test length(coords_nat) == 3               # upper triangle of a 2×2
    @test isapprox(coords_nat, [2.0, 0.5, 1.0]; atol = 1e-12)

    # Count matches transformed coords (Wald delta-method stays square).
    θ0t = get_θ0_transformed(fe)
    coords_trans = _coords_for_param(collect(θ0t[:Ω]), spec; natural = false)
    @test length(coords_trans) == length(coords_nat)
end

# ---------------------------------------------------------------------------
# 6b. Structured covariances: block-diagonal + fixed eigenvalues
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix block-diagonal structure" begin
    using NoLimits: _inv_jac_spec_val

    Σ0 = [2.0 0.5 0.0; 0.5 1.0 0.0; 0.0 0.0 3.0]

    @testset "construction validates block-diagonal input" begin
        @test_throws ErrorException RealLiePSDMatrix(
            [2.0 0.5 0.1; 0.5 1.0 0.0;
             0.1 0.0 3.0]; blocks = [1, 1, 2])           # nonzero cross-block entry
        @test_throws ErrorException RealLiePSDMatrix(Σ0; blocks = [1, 1])   # wrong length
        p = RealLiePSDMatrix(Σ0; blocks = [1, 1, 2])
        @test p.blocks == [1, 1, 2]
    end

    fe = @fixedEffects begin
        Ω = RealLiePSDMatrix([2.0 0.5 0.0; 0.5 1.0 0.0; 0.0 0.0 3.0];
            blocks = [1, 1, 2], eigenvalue_lower = 1e-3, eigenvalue_upper = 1e3,
            calculate_se = true)
    end
    θ0t = get_θ0_transformed(fe)
    θ0u = get_θ0_untransformed(fe)
    spec = get_transforms(fe).forward.specs[1]

    @testset "drops cross-block coefficients from the optimizer" begin
        # 3×3 full = 6 params; block-diag [1,1,2] drops the 2 cross-block angles → 4 free.
        @test length(θ0t.Ω) == 4
        @test spec.lie !== nothing
        @test spec.lie.fixed_idx == [5, 6]           # cross-block angle positions
        @test spec.lie.fixed_val == [0.0, 0.0]
        @test isapprox(Matrix(θ0u.Ω), Σ0; atol = 1e-10)
    end

    @testset "block-diagonal structure preserved under any free params" begin
        it = get_inverse_transform(fe)
        rng = MersenneTwister(3)
        for _ in 1:5
            tp = copy(θ0t)
            tp.Ω .= θ0t.Ω .+ 0.8 .* randn(rng, 4)
            Σp = Matrix(it(tp).Ω)
            @test abs(Σp[1, 3]) < 1e-12
            @test abs(Σp[2, 3]) < 1e-12
            @test minimum(eigen(Symmetric(Σp)).values) > 0
        end
    end

    @testset "Jacobianᵀ matches structured inverse Jacobian" begin
        rng = MersenneTwister(9)
        for _ in 1:3
            t = θ0t.Ω .+ 0.5 .* randn(rng, 4)
            G = randn(rng, 3, 3)
            gt = _inv_jac_spec_val(spec, collect(t), G)
            J = ForwardDiff.jacobian(
                x -> NoLimits._liepsd_inverse_layout(x, spec.lie), collect(t))
            @test isapprox(gt, J' * vec(G); atol = 1e-9)
        end
    end

    @testset "UQ natural-coord count equals transformed count" begin
        using NoLimits: _coords_for_param
        coords_t = _coords_for_param(collect(θ0t.Ω), spec; natural = false)
        coords_n = _coords_for_param(Matrix(θ0u.Ω), spec; natural = true)
        @test length(coords_t) == 4
        @test length(coords_n) == 4
    end
end

@testset "RealLiePSDMatrix fixed eigenvalues" begin
    fe = @fixedEffects begin
        Ω = RealLiePSDMatrix([2.0 0.0; 0.0 3.0]; fix_eigenvalues = [1],
            calculate_se = true)
    end
    θ0t = get_θ0_transformed(fe)
    spec = get_transforms(fe).forward.specs[1]
    @test length(θ0t.Ω) == 2                         # 3 full − 1 fixed eigenvalue
    @test spec.lie.fixed_idx == [1]
    @test isapprox(spec.lie.fixed_val, [log(2.0)]; atol = 1e-12)

    it = get_inverse_transform(fe)
    tp = copy(θ0t)
    tp.Ω .= θ0t.Ω .+ [0.4, 0.3]
    Σp = Matrix(it(tp).Ω)
    # The fixed eigenvalue stays pinned at 2.0 regardless of the free params.
    @test minimum(abs.(sort(eigen(Symmetric(Σp)).values) .- 2.0)) < 1e-10
end

# ---------------------------------------------------------------------------
# 7. Model integration + end-to-end estimation (RE covariance)
# ---------------------------------------------------------------------------
@testset "RealLiePSDMatrix as random-effect covariance" begin
    using DataFrames

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale = :log)
            Ω = RealLiePSDMatrix([1.0 0.0; 0.0 1.0]; eigenvalue_lower = 1e-3,
                eigenvalue_upper = 1e3)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2] * t, σ)
        end
    end

    rng = MersenneTwister(2020)
    ids = repeat(1:12, inner = 4)
    tt = repeat([0.0, 1.0, 2.0, 3.0], 12)
    y = 1.0 .+ 0.2 .* tt .+ 0.3 .* randn(rng, length(ids))
    df = DataFrame(ID = ids, t = tt, y = y)
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_lap = fit_model(dm, NoLimits.Laplace())
    @test NoLimits.get_converged(res_lap)
    Ω_est = NoLimits.get_params(res_lap; scale = :untransformed).Ω
    @test issymmetric(Matrix(Ω_est))
    @test minimum(eigen(Symmetric(Matrix(Ω_est))).values) > 0

    # FOCEI exercises the analytic Jacobianᵀ path and should agree with Laplace.
    res_foc = fit_model(dm, NoLimits.FOCEI())
    @test NoLimits.get_converged(res_foc)
    @test isapprox(NoLimits.get_objective(res_lap), NoLimits.get_objective(res_foc);
        atol = 1e-2)
end

@testset "RealLiePSDMatrix block-diagonal RE covariance end-to-end" begin
    using DataFrames

    # Three random effects; a block-diagonal covariance couples RE 1&2 but not RE 3.
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale = :log)
            Ω = RealLiePSDMatrix(Matrix{Float64}(I, 3, 3); blocks = [1, 1, 2],
                eigenvalue_lower = 1e-3, eigenvalue_upper = 1e3)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(3), Ω); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2] * t + η[3] * t^2, σ)
        end
    end

    rng = MersenneTwister(11)
    ids = repeat(1:15, inner = 4)
    tt = repeat([0.0, 1.0, 2.0, 3.0], 15)
    y = 1.0 .+ 0.2 .* tt .+ 0.3 .* randn(rng, length(ids))
    df = DataFrame(ID = ids, t = tt, y = y)
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res = fit_model(dm, NoLimits.Laplace())
    @test NoLimits.get_converged(res)
    Ω_est = Matrix(NoLimits.get_params(res; scale = :untransformed).Ω)
    # The estimated covariance must retain the enforced block-diagonal zeros.
    @test abs(Ω_est[1, 3]) < 1e-8
    @test abs(Ω_est[2, 3]) < 1e-8
    @test minimum(eigen(Symmetric(Ω_est)).values) > 0
end
