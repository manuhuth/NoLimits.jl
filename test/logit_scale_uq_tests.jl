using Test
using NoLimits
using ComponentArrays
using Distributions

# ─── _flat_transform_kinds_for_free ─────────────────────────────────────────

@testset "_flat_transform_kinds_for_free — :logit uniform" begin
    fe = @fixedEffects begin
        p = RealNumber(0.3; scale=:logit, calculate_se=true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:p])
    @test kinds == [:logit]
end

@testset "_flat_transform_kinds_for_free — :logit vector" begin
    fe = @fixedEffects begin
        v = RealVector([0.2, 0.5, 0.8]; scale=[:logit, :logit, :logit], calculate_se=true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:v])
    @test kinds == [:logit, :logit, :logit]
end

@testset "_flat_transform_kinds_for_free — :elementwise mixed" begin
    fe = @fixedEffects begin
        v = RealVector([0.4, 2.0, -1.0]; scale=[:logit, :log, :identity], calculate_se=true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:v])
    @test kinds == [:logit, :log, :identity]
end

@testset "_flat_transform_kinds_for_free — mixed params" begin
    fe = @fixedEffects begin
        p = RealNumber(0.3; scale=:logit, calculate_se=true)
        σ = RealNumber(1.0; scale=:log, calculate_se=true)
        a = RealNumber(0.5; scale=:identity, calculate_se=true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:p, :σ, :a])
    @test kinds == [:logit, :log, :identity]
end

@testset "_flat_transform_kinds_for_free — :elementwise two params" begin
    fe = @fixedEffects begin
        v = RealVector([0.3, 2.0, -1.0]; scale=[:logit, :log, :identity], calculate_se=true)
        σ = RealNumber(0.5; scale=:log, calculate_se=true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:v, :σ])
    @test kinds == [:logit, :log, :identity, :log]
end

# ─── _wald_closed_form_kind ──────────────────────────────────────────────────

@testset "_wald_closed_form_kind — :logit on natural scale" begin
    # coord_transforms[j] == :logit → :logitnormal
    coord_transforms = [:logit]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :logitnormal
end

@testset "_wald_closed_form_kind — :log on natural scale" begin
    coord_transforms = [:log]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :lognormal
end

@testset "_wald_closed_form_kind — :identity on natural scale" begin
    coord_transforms = [:identity]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :normal
end

@testset "_wald_closed_form_kind — any on transformed scale" begin
    # On transformed scale, always :normal regardless of coord transform kind
    coord_transforms = [:logit]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :transformed, 1, vcov_t, coord_transforms)
    @test kind == :normal
end

@testset "_wald_closed_form_kind — :cholesky on natural scale → :none" begin
    coord_transforms = [:cholesky]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :none
end

@testset "_wald_closed_form_kind — not wald backend → :none" begin
    coord_transforms = [:logit]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:chain, :natural, 1, vcov_t, coord_transforms)
    @test kind == :none
end

# ─── _wald_density_xy ────────────────────────────────────────────────────────

# Note: _wald_density_xy takes v = variance (not sigma). σ = sqrt(v) inside.

@testset "_wald_density_xy — :logitnormal basic properties" begin
    # μ=0 (logit(0.5)=0), v=1.0 → σ=1: LogitNormal(0, 1)
    # Density should be on (0,1), symmetric around 0.5
    μ = 0.0
    v = 1.0  # variance → σ = 1.0
    result = NoLimits._wald_density_xy(:logitnormal, μ, v)
    @test result !== nothing
    x, y = result
    @test length(x) == length(y)
    # All x in (0, 1)
    @test all(0.0 .< x .< 1.0)
    # All density values non-negative
    @test all(y .>= 0.0)
    # Density integrates close to 1 (trapezoid rule check)
    area = sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1]) ./ 2)
    @test isapprox(area, 1.0; atol=0.01)
end

@testset "_wald_density_xy — :logitnormal matches Distributions.LogitNormal" begin
    μ = 1.0
    σ = 0.5
    v = σ^2  # variance → pass v to _wald_density_xy
    result = NoLimits._wald_density_xy(:logitnormal, μ, v)
    @test result !== nothing
    x, y = result
    dist = LogitNormal(μ, σ)  # σ = sqrt(v)
    # Check density at a few interior points
    for (xi, yi) in zip(x[50:10:end-50], y[50:10:end-50])
        @test isapprox(yi, pdf(dist, xi); rtol=1e-8)
    end
end

@testset "_wald_density_xy — :logitnormal zero variance returns nothing" begin
    result = NoLimits._wald_density_xy(:logitnormal, 0.0, 0.0)
    @test result === nothing
end

@testset "_wald_density_xy — :lognormal still works (regression)" begin
    result = NoLimits._wald_density_xy(:lognormal, 0.0, 1.0)  # v=1.0 → σ=1.0
    @test result !== nothing
    x, y = result
    @test all(x .> 0.0)
    @test all(y .>= 0.0)
end

@testset "_wald_density_xy — :normal still works (regression)" begin
    result = NoLimits._wald_density_xy(:normal, 0.0, 1.0)  # v=1.0 → σ=1.0
    @test result !== nothing
    x, y = result
    @test all(isfinite.(x))
    @test all(y .>= 0.0)
end
