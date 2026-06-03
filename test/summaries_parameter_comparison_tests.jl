using Test
using NoLimits
using DataFrames
using Distributions
using Random

@testset "compare_parameters" begin
    model1 = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3; calculate_se=true)
            b = RealNumber(0.1; calculate_se=false)
            s = RealNumber(0.5; scale=:log, calculate_se=true)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            mu = a + b * t
            y ~ Normal(mu, s)
        end
    end

    model2 = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3; calculate_se=true)
            s = RealNumber(0.5; scale=:log, calculate_se=true)
            d = RealNumber(0.2; calculate_se=true)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            mu = a + d * t
            y ~ Normal(mu, s)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.4, 0.1, 0.5, 0.0, 0.3],
    )

    dm1 = DataModel(model1, df; primary_id=:ID, time_col=:t)
    dm2 = DataModel(model2, df; primary_id=:ID, time_col=:t)
    fit1 = fit_model(dm1, MLE(; optim_kwargs=(maxiters=2,)))
    fit2 = fit_model(dm2, MLE(; optim_kwargs=(maxiters=2,)))

    # Default: union of SE-eligible parameters, in declaration order.
    c = compare_parameters(fit1, fit2)
    @test c isa ParameterComparison
    @test c.labels == ["model 1", "model 2"]
    @test c.scale == :natural
    @test c.parameters == [:a, :s, :d]
    @test size(c.estimates) == (3, 2)

    i_a = findfirst(==(:a), c.parameters)
    i_d = findfirst(==(:d), c.parameters)
    @test all(c.estimates[i_a, :] .!== nothing)      # shared parameter
    @test c.estimates[i_d, 1] === nothing            # d absent from model1
    @test c.estimates[i_d, 2] !== nothing            # d present in model2

    # Custom labels and the label => fit pair form agree on contents.
    c_lbl = compare_parameters(fit1, fit2; labels = ["A", "B"])
    @test c_lbl.labels == ["A", "B"]
    c_pair = compare_parameters("A" => fit1, "B" => fit2)
    @test c_pair.labels == ["A", "B"]
    @test c_pair.parameters == c.parameters

    # common_only keeps only parameters shared by every model.
    c_common = compare_parameters(fit1, fit2; common_only = true)
    @test c_common.parameters == [:a, :s]
    @test all(c_common.estimates .!== nothing)

    # include_non_se brings in b (model1 only).
    c_all = compare_parameters(fit1, fit2; include_non_se = true)
    @test :b in c_all.parameters
    i_b = findfirst(==(:b), c_all.parameters)
    @test c_all.estimates[i_b, 1] !== nothing
    @test c_all.estimates[i_b, 2] === nothing

    # scale is validated and recorded.
    @test compare_parameters(fit1, fit2; scale = :transformed).scale == :transformed
    @test_throws ErrorException compare_parameters(fit1, fit2; scale = :bogus)

    # Mismatched label count is an error.
    @test_throws ErrorException compare_parameters(fit1, fit2; labels = ["only one"])

    # Rendered table shows the title, labels, parameter names, and "-" for absences.
    txt = sprint(show, MIME"text/plain"(), c)
    @test occursin("ParameterComparison", txt)
    @test occursin("model 1", txt)
    @test occursin("model 2", txt)
    @test occursin("a", txt)
    @test occursin("d", txt)
    @test occursin("-", txt)
end
