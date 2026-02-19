using Test
using NoLimits
using ComponentArrays
using LinearAlgebra
using ForwardDiff
using ReverseDiff
using Zygote
using DataFrames

@testset "ParameterTransformations" begin
    # Log transform round-trip on scalar and vector.
    names = [:a, :b, :c]
    θ = ComponentArray((a=1.5, b=2.0, c=3.0))
    specs = [TransformSpec(:a, :log, (1, 1), nothing),
             TransformSpec(:b, :identity, (1, 1), nothing),
             TransformSpec(:c, :identity, (1, 1), nothing)]
    ft = ForwardTransform(collect(names), specs)
    it = InverseTransform(collect(names), specs)
    θt = ft(θ)
    @test isapprox(θt.a, log(θ.a); rtol=1e-8, atol=1e-10)
    @test isapprox(it(θt).a, θ.a; rtol=1e-8, atol=1e-10)

    # Cholesky transform round-trip on a PSD matrix block.
    A = [2.0 0.3; 0.3 1.5]
    θA = ComponentArray((A=A,))
    specsA = [TransformSpec(:A, :cholesky, (2, 2), nothing)]
    ftA = ForwardTransform([:A], specsA)
    itA = InverseTransform([:A], specsA)
    θAt = ftA(θA)
    Arec = itA(θAt).A
    @test isapprox(Arec, A; rtol=1e-8, atol=1e-8)

    # Matrix-exponential transform round-trip on a PSD matrix block.
    Aexp = [1.5 0.2; 0.2 1.2]
    θE = ComponentArray((E=Aexp,))
    specsE = [TransformSpec(:E, :expm, (2, 2), nothing)]
    ftE = ForwardTransform([:E], specsE)
    itE = InverseTransform([:E], specsE)
    θEt = ftE(θE)
    Erec = itE(θEt).E
    @test isapprox(Erec, Aexp; rtol=1e-8, atol=1e-8)
    @test length(θEt.E) == 3

    # AD check for log transform (ForwardDiff + Zygote).
    f_log(x) = log_forward(x)
    @test isapprox(ForwardDiff.derivative(f_log, 2.0), 1 / 2.0; rtol=1e-8, atol=1e-10)
    f_log_vec(v) = log_forward(v[1])
    @test isapprox(ReverseDiff.gradient(f_log_vec, [2.0])[1], 1 / 2.0; rtol=1e-8, atol=1e-10)
    @test isapprox(Zygote.gradient(f_log, 2.0)[1], 1 / 2.0; rtol=1e-8, atol=1e-10)

    # AD check for cholesky transform on a PSD matrix built from parameters.
    function f_chol(v)
        L = reshape(v, 2, 2)
        A = L * L' + 1e-3I
        T = cholesky_forward(A)
        return sum(T)
    end
    v0 = [1.0, 0.0, 0.2, 0.8]
    g = ForwardDiff.gradient(f_chol, v0)
    @test length(g) == length(v0)

    g_rev = ReverseDiff.gradient(f_chol, v0)
    @test length(g_rev) == length(v0)

    g_zyg = Zygote.gradient(f_chol, v0)[1]
    @test length(g_zyg) == length(v0)
    @test isapprox(g_rev, g; rtol=1e-6, atol=1e-8)
    @test isapprox(g_zyg, g; rtol=1e-6, atol=1e-8)

    # AD check for expm-based transform on a PSD matrix built from parameters.
    function f_expm(v)
        L = reshape(v, 2, 2)
        A = L * L' + 1e-3I
        T = expm_forward(A)
        return sum(T)
    end
    g_expm = ForwardDiff.gradient(f_expm, v0)
    @test length(g_expm) == length(v0)
    @test_skip ReverseDiff.gradient(f_expm, v0)
    g_expm_zyg = Zygote.gradient(f_expm, v0)[1]
    @test length(g_expm_zyg) == length(v0)
    @test isapprox(g_expm_zyg, g_expm; rtol=1e-6, atol=1e-8)

    # PSD params are symmetrized for log-likelihood paths.
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
            Ω = RealPSDMatrix([1.0 0.0; 0.0 1.0], scale=:cholesky)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    fe = dm.model.fixed.fixed
    θu = deepcopy(get_θ0_untransformed(fe))
    Ω_bad = [1.0 2.0; 0.0 1.0]
    setproperty!(θu, :Ω, Ω_bad)
    θsym = NoLimits._symmetrize_psd_params(θu, fe)
    @test θsym.Ω ≈ (Ω_bad + Ω_bad') ./ 2
    @test issymmetric(θsym.Ω)

end
