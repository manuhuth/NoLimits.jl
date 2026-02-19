using Test
using NoLimits
using Distributions
using LinearAlgebra

@testset "Equation display ordering and rendering" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            sigma = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            eta = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @preDifferentialEquation begin
            k = exp(a + eta^2)
        end

        @DifferentialEquation begin
            s(t) = k * t
            D(x1) ~ -k * x1 + s(t)
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            mu = x1(t) + k
            y ~ Normal(mu, sigma)
        end
    end

    eqs = NoLimits.get_equation_lines(model)
    @test length(eqs) == 5

    @test eqs[1].head == :(=)
    @test eqs[1].args[1] == :k

    @test eqs[2].head == :(=)
    @test eqs[2].args[1] isa Expr
    @test eqs[2].args[1].head == :call
    @test eqs[2].args[1].args[1] == :s

    @test eqs[3].head == :call
    @test eqs[3].args[1] == :~
    @test eqs[3].args[2] isa Expr
    @test eqs[3].args[2].head == :call
    @test eqs[3].args[2].args[1] == :D

    @test eqs[4].head == :(=)
    @test eqs[4].args[1] == :mu

    @test eqs[5].head == :call
    @test eqs[5].args[1] == :~
    @test eqs[5].args[2] == :y

    txt_plain = sprint(io -> NoLimits.show_equations(io, model; latex=false))
    @test occursin("k", txt_plain)
    @test occursin("x1̇(t) =", txt_plain)
    @test occursin("y = Normal", txt_plain)

    txt_num = sprint(io -> NoLimits.show_equations(io, model; latex=false, numbered=true))
    @test startswith(txt_num, "1.")
    @test occursin("\n2.", txt_num)

    txt_latex = sprint(io -> NoLimits.show_equations(io, model; latex=true))
    @test !isempty(strip(txt_latex))
    @test !occursin("~\\left", txt_latex)
    @test occursin("\\dot{", txt_latex)
    @test occursin("&=", txt_latex)

    model_vec = @Model begin
        @fixedEffects begin
            omega1 = RealNumber(1.0, scale=:log)
            omega2 = RealNumber(1.0, scale=:log)
            sigma = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            eta = RandomEffect(
                MvNormal([0.0, 0.0], Diagonal([omega1, omega2]));
                column=:ID,
            )
        end

        @formulas begin
            mu = eta[1] + 2 * eta[2] + t
            y ~ Normal(mu, sigma)
        end
    end

    txt_vec_latex = sprint(io -> NoLimits.show_equations(io, model_vec; latex=true))
    @test occursin(raw"\eta^{\left(1\right)}", txt_vec_latex)
    @test occursin(raw"\eta^{\left(2\right)}", txt_vec_latex)
    @test !occursin(raw"\eta\left( 1 \right)", txt_vec_latex)
end
