# Public plotting API. The drawing implementations live in the Plots extension
# (ext/NoLimitsPlotsExt.jl) and load automatically when Plots.jl is imported
# alongside NoLimits. Without Plots these are method-less stubs; calling one raises a
# MethodError carrying a hint to load Plots (registered in `__init__` below). The
# Plots-free pieces (PlotStyle, PlotCache, get_residuals, build_plot_cache and their
# helpers) live in the core module, in the other plotting/*.jl files.

const _PLOT_API_FUNCTIONS = (:plot_data, :plot_observed_profiles, :plot_fits,
    :plot_fits_comparison, :plot_multistart_waterfall,
    :plot_multistart_fixed_effect_variability, :plot_em_trajectories, :plot_hidden_states,
    :plot_emission_distributions, :plot_dv_pred, :plot_dv_ipred, :plot_wres_pred,
    :plot_shrinkage, :plot_vpc, :plot_observation_distributions, :plot_residuals,
    :plot_residual_distribution, :plot_residual_qq, :plot_residual_pit,
    :plot_residual_acf, :plot_uq_distributions, :plot_random_effects_pdf,
    :plot_random_effects_scatter, :plot_random_effect_pairplot,
    :plot_random_effect_distributions, :plot_random_effect_pit,
    :plot_random_effect_standardized, :plot_random_effect_standardized_scatter)

function plot_data end
function plot_observed_profiles end
function plot_fits end
function plot_fits_comparison end
function plot_multistart_waterfall end
function plot_multistart_fixed_effect_variability end
function plot_em_trajectories end
function plot_hidden_states end
function plot_emission_distributions end
function plot_dv_pred end
function plot_dv_ipred end
function plot_wres_pred end
function plot_shrinkage end
function plot_vpc end
function plot_observation_distributions end
function plot_residuals end
function plot_residual_distribution end
function plot_residual_qq end
function plot_residual_pit end
function plot_residual_acf end
function plot_uq_distributions end
function plot_random_effects_pdf end
function plot_random_effects_scatter end
function plot_random_effect_pairplot end
function plot_random_effect_distributions end
function plot_random_effect_pit end
function plot_random_effect_standardized end
function plot_random_effect_standardized_scatter end

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, _argtypes, _kwargs
        f = exc.f
        if f isa Function && nameof(f) in _PLOT_API_FUNCTIONS &&
           Base.get_extension(@__MODULE__, :NoLimitsPlotsExt) === nothing
            print(io,
                "\n\nNoLimits plotting functions live in a package extension that loads ",
                "when Plots.jl is available. Run `using Plots` alongside `using NoLimits` ",
                "to enable `", nameof(f), "` and the other `plot_*` functions.")
        end
    end
    return nothing
end
