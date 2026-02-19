# Data Model Construction

The `DataModel` constructor binds a `@Model` specification to a tabular dataset, producing the per-individual data structures required by estimation, simulation, and plotting workflows.

At construction time, `DataModel` validates that the dataset schema is consistent with the model definition, splits rows by individual identifier, builds observation and covariate series, prepares random-effect group mappings, and -- for ODE models with event columns -- assembles event callback metadata.

## Constructor

```julia
using NoLimits

dm = DataModel(
    model,
    df;
    primary_id=:ID,
    time_col=:t,
)
```

The key constructor arguments are:

- `primary_id`: the column that identifies individual trajectories (e.g., subject ID).
- `time_col`: the column used for longitudinal time indexing.
- `evid_col`: enables event-table parsing for ODE models (see below).
- `amt_col`, `rate_col`, `cmt_col`: event-related columns, used when `evid_col` is set.
- `serialization`: controls parallel execution mode (default `EnsembleSerial()`; use `EnsembleThreads()` for multi-threaded evaluation).

If `primary_id` is omitted, it is inferred automatically when the model uses exactly one random-effect grouping column. When multiple grouping columns are present, `primary_id` must be specified explicitly.

## Example: Non-ODE DataModel

This example builds a `DataModel` for a model without differential equations, featuring two random-effect grouping levels and a non-Gaussian observation distribution:

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        sigma = RealNumber(0.4, scale=:log)
    end

    @covariates begin
        t = Covariate()
        z = Covariate()
    end

    @randomEffects begin
        eta_id = RandomEffect(TDist(6.0); column=:ID)
        eta_site = RandomEffect(Gamma(2.0, 1.0); column=:SITE)
    end

    @formulas begin
        mu = a + z + tanh(eta_id) + eta_site^2
        y ~ Laplace(mu, sigma)
    end
end

df = DataFrame(
    ID   = [1, 1, 2, 2],
    SITE = [:A, :A, :B, :B],
    t    = [0.0, 1.0, 0.0, 1.0],
    z    = [0.2, 0.3, -0.1, 0.0],
    y    = [0.5, 0.6, 0.1, 0.2],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
```

## Example: ODE DataModel (No Events)

For models with differential equations but no dosing or reset events, the constructor requires only the standard arguments:

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        k = RealNumber(0.3)
        sigma = RealNumber(0.2, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(TDist(5.0); column=:ID)
    end

    @DifferentialEquation begin
        D(x1) ~ -k * x1 + eta^2
    end

    @initialDE begin
        x1 = 1.0
    end

    @formulas begin
        y ~ Normal(x1(t), sigma)
    end
end

df = DataFrame(
    ID = [1, 1, 1],
    t  = [0.0, 0.5, 1.0],
    y  = [1.0, 0.9, 0.8],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
```

## Example: ODE DataModel with Event Columns

For models where external inputs (e.g., doses, stimuli, or resets) modify the ODE state during integration, use the `evid_col` argument to activate event parsing. The supported event types are:

- `EVID = 1`: input event (instantaneous bolus if `RATE = 0`, or constant-rate infusion if `RATE > 0`).
- `EVID = 2`: reset event (sets the specified compartment to the `AMT` value).

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        ka = RealNumber(0.5)
        kel = RealNumber(0.2)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(TDist(6.0); column=:ID)
    end

    @DifferentialEquation begin
        D(depot) ~ -ka * depot
        D(center) ~ ka * depot - kel * center + eta^2
    end

    @initialDE begin
        depot = 0.0
        center = 0.0
    end

    @formulas begin
        y ~ LogNormal(center(t), sigma)
    end
end

df = DataFrame(
    ID   = [1, 1, 1, 1],
    t    = [0.0, 0.0, 1.0, 2.0],
    EVID = [1, 0, 0, 0],
    AMT  = [10.0, 0.0, 0.0, 0.0],
    RATE = [0.0, 0.0, 0.0, 0.0],
    CMT  = ["depot", "depot", "center", "center"],
    y    = [missing, 0.0, 2.1, 1.8],
)

dm = DataModel(
    model,
    df;
    primary_id=:ID,
    time_col=:t,
    evid_col=:EVID,
    amt_col=:AMT,
    rate_col=:RATE,
    cmt_col=:CMT,
)
```

The `CMT` column can contain either integer indices or state names (`String`/`Symbol`), but the format must be consistent within the DataFrame.

## Dynamic Covariates and Interpolation

When using `DynamicCovariate` or `DynamicCovariateVector`, the `DataInterpolations` package must be loaded in the user environment:

```julia
using DataInterpolations
```

`DataModel` requires that time values are sorted within each individual and enforces minimum observation counts per interpolation type:

| Interpolation | Minimum observations per individual |
| --- | --- |
| `ConstantInterpolation` | 1 |
| `SmoothedConstantInterpolation` | 2 |
| `LinearInterpolation` | 2 |
| `LagrangeInterpolation` | 2 |
| `AkimaInterpolation` | 2 |
| `QuadraticInterpolation` | 3 |
| `QuadraticSpline` | 3 |
| `CubicSpline` | 3 |

## Validation Rules

`DataModel` performs comprehensive validation at construction time to catch specification and data inconsistencies early. The checks include:

- **Schema validation**: `primary_id`, `time_col`, and outcome columns referenced in `@formulas` must exist in the DataFrame. Neither `primary_id` nor `time_col` may contain missing values.
- **Time column declaration**: `time_col` must be declared in `@covariates` as `Covariate()` or `DynamicCovariate()`.
- **Event columns**: when `evid_col` is set, event columns (`AMT`, `RATE`, `CMT`) are required and validated for completeness on event rows.
- **Missing values**: observation rows (`EVID == 0`) cannot have missing values in outcome or covariate columns used by `@formulas`.
- **Grouping consistency**: random-effect grouping columns cannot contain missing values. Grouping columns other than `primary_id` must be constant within each `primary_id`.
- **Numeric random-effect grouping levels**: if any random-effect grouping column uses numeric levels, `DataModel` emits:
  `DataModel detected numeric random-effect grouping levels in column(s) $(cols_str). You wwill not be able to use constant random-effects. If you want to use constantr andom effects, consider ralabeling your random efefcts to strings or symbols.`
  In this case, constant random-effects are unavailable unless grouping levels are relabeled to strings or symbols.
- **Constant covariate consistency**: constant covariates must be constant within `primary_id` and within all declared `constant_on` groups.
- **Random-effect covariate dependencies**: random-effect distributions that use covariates require those covariates to be `ConstantCovariate` or `ConstantCovariateVector`.
- **Formula time offsets**: constant time offsets in formulas (e.g., `x1(t - 0.5)`) extend the integration window automatically. Offsets that push evaluation before `t = 0` are rejected. Non-constant offsets require `saveat_mode = :dense`.
- **PreDE random-effect constraint**: random effects used in `@preDifferentialEquation` must be grouped by `primary_id`.

## DataModel Summary

Use `NoLimits.summarize(dm)` for a compact structural overview:

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```

## Accessors

After construction, several accessor functions are available for diagnostics, custom tooling, and low-level inspection:

```julia
individuals = get_individuals(dm)
ind1 = get_individual(dm, 1)
batches = get_batches(dm)
batch_ids = get_batch_ids(dm)
row_groups = get_row_groups(dm)
re_info = get_re_group_info(dm)
re_idx_obs = get_re_indices(dm, 1)               # observation rows
re_idx_all = get_re_indices(dm, 1; obs_only=false)
```
