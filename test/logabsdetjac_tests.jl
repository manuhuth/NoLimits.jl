using NoLimits
using ComponentArrays
using Distributions
using LinearAlgebra
using ForwardDiff
using FiniteDifferences
using NoLimits: _block_logabsdetjac, cholesky_inverse, expm_inverse, _sym_from_upper,
                _upper_tri_vec, stickbreak_inverse, lograterows_inverse, liepsd_inverse,
                _lie_dim
using Test

# Each golden compares logabsdetjac / _block_logabsdetjac to an INDEPENDENT central
# finite-difference logabsdet of the minimal (square) natural map built from the public
# inverse primitives (FD vs ForwardDiff are independent numerical routes), plus a
# nested-AD finiteness probe.

fdm = central_fdm(5, 1)
vechL(M) = [M[i, j] for j in axes(M, 2) for i in j:size(M, 1)]
function lower_from_free(z, n)
    T = zeros(eltype(z), n, n)
    k = 1
    for j in 1:n, i in j:n
        T[i, j] = z[k]
        k += 1
    end
    return T
end
fd_ref(gmin, z) = logabsdet(FiniteDifferences.jacobian(fdm, gmin, collect(z))[1])[1]

function _it_theta(fe)
    it = NoLimits.get_inverse_transform(fe)
    θt = NoLimits.get_transform(fe)(NoLimits.get_θ0_untransformed(fe))
    return it, θt
end
_spec_for(it, name) = it.specs[findfirst(s -> s.name == name, it.specs)]

@testset "logabsdetjac structured scales" begin
    @testset "scalar closed forms still exact" begin
        m = @Model begin
            @fixedEffects begin
                a = RealNumber(1.5, scale = :log)
                p = RealNumber(0.3, scale = :logit, lower = 0.0, upper = 1.0)
                c = RealNumber(0.7)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(c, a)
            end
        end
        it, θt = _it_theta(NoLimits.get_fixed(m))
        @test logabsdetjac(it, θt)≈log(1.5) + log(0.3 * 0.7) atol=1e-10
    end

    @testset "cholesky" begin
        m = @Model begin
            @fixedEffects begin
                Ω = RealPSDMatrix([1.3 0.2; 0.2 0.9], scale = :cholesky)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
            end
            @formulas begin
                y ~ Normal(η[1], σ)
            end
        end
        it, θt = _it_theta(NoLimits.get_fixed(m))
        spec = _spec_for(it, :Ω)
        blk = collect(getproperty(θt, :Ω))  # flat n^2 log-Cholesky factor
        n = 2
        Tmat = reshape(blk, n, n)
        gmin = z -> vechL(cholesky_inverse(lower_from_free(z, n)))
        z0 = [Tmat[i, j] for j in 1:n for i in j:n]  # lower-tri free coords
        @test _block_logabsdetjac(spec, blk)≈fd_ref(gmin, z0) atol=1e-6
        @test logabsdetjac(it, θt)≈_block_logabsdetjac(spec, blk) + log(0.5) atol=1e-6
        @test all(isfinite, ForwardDiff.gradient(z -> _block_logabsdetjac(spec, z), blk))
    end

    @testset "expm" begin
        m = @Model begin
            @fixedEffects begin
                Ω = RealPSDMatrix([1.4 0.3; 0.3 1.1], scale = :expm)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
            end
            @formulas begin
                y ~ Normal(η[1], σ)
            end
        end
        it, θt = _it_theta(NoLimits.get_fixed(m))
        spec = _spec_for(it, :Ω)
        blk = collect(getproperty(θt, :Ω))
        n = _lie_dim(length(blk))
        gmin = z -> _upper_tri_vec(expm_inverse(_sym_from_upper(z, n)))
        @test _block_logabsdetjac(spec, blk)≈fd_ref(gmin, blk) atol=1e-6
        @test all(isfinite, ForwardDiff.gradient(z -> _block_logabsdetjac(spec, z), blk))
    end

    @testset "stickbreak" begin
        m = @Model begin
            @fixedEffects begin
                w = ProbabilityVector([0.2, 0.5, 0.3])
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(w[1], σ)
            end
        end
        it, θt = _it_theta(NoLimits.get_fixed(m))
        spec = _spec_for(it, :w)
        blk = collect(getproperty(θt, :w))
        k1 = length(blk)
        gmin = z -> stickbreak_inverse(z)[1:k1]
        @test _block_logabsdetjac(spec, blk)≈fd_ref(gmin, blk) atol=1e-6
        @test all(isfinite, ForwardDiff.gradient(z -> _block_logabsdetjac(spec, z), blk))
    end

    @testset "stickbreakrows" begin
        P = [0.7 0.2 0.1; 0.1 0.8 0.1; 0.2 0.3 0.5]
        m = @Model begin
            @fixedEffects begin
                Tm = DiscreteTransitionMatrix(P)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(σ, σ)
            end
        end
        it, θt = _it_theta(NoLimits.get_fixed(m))
        spec = _spec_for(it, :Tm)
        blk = collect(getproperty(θt, :Tm))
        n = spec.size[1]
        k1 = n - 1
        gmin = z -> vcat((stickbreak_inverse(z[((i - 1) * k1 + 1):(i * k1)])[1:k1]
        for i in 1:n)...)
        @test _block_logabsdetjac(spec, blk)≈fd_ref(gmin, blk) atol=1e-6
        @test all(isfinite, ForwardDiff.gradient(z -> _block_logabsdetjac(spec, z), blk))
    end

    @testset "lograterows" begin
        Q = [-0.5 0.3 0.2; 0.1 -0.4 0.3; 0.2 0.2 -0.4]
        m = @Model begin
            @fixedEffects begin
                Qm = ContinuousTransitionMatrix(Q)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(σ, σ)
            end
        end
        it, θt = _it_theta(NoLimits.get_fixed(m))
        spec = _spec_for(it, :Qm)
        blk = collect(getproperty(θt, :Qm))
        n = spec.size[1]
        offdiag(M) = [M[i, j] for i in 1:n for j in 1:n if i != j]
        gmin = z -> offdiag(lograterows_inverse(z, n))
        @test _block_logabsdetjac(spec, blk)≈fd_ref(gmin, blk) atol=1e-6
        @test _block_logabsdetjac(spec, blk)≈sum(blk) atol=1e-10  # closed form
    end

    @testset "lie (unstructured)" begin
        m = @Model begin
            @fixedEffects begin
                Ω = RealLiePSDMatrix([1.3 0.2; 0.2 0.9], scale = :lie)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
            end
            @formulas begin
                y ~ Normal(η[1], σ)
            end
        end
        it, θt = _it_theta(NoLimits.get_fixed(m))
        spec = _spec_for(it, :Ω)
        blk = collect(getproperty(θt, :Ω))
        gmin = z -> vechL(liepsd_inverse(z))
        @test _block_logabsdetjac(spec, blk)≈fd_ref(gmin, blk) atol=1e-6
        @test all(isfinite, ForwardDiff.gradient(z -> _block_logabsdetjac(spec, z), blk))
    end
end
