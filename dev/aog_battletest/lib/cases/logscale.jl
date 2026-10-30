# Log-scale axes (review of #383): shared facet limits must stay positive.
function cases_logscale()
    cs = Case[]
    ylog = "kwargs_subplot = (; yscale = log10)"
    for fn in (:plot_fits, :plot_data, :plot_fits_comparison)
        append!(cs, cross(fn, fixtures_for(fn), ["ylog" => ylog]))
    end
    append!(
        cs, cross(
            :plot_residuals, fixtures_for(:plot_residuals),
            ["res_pit_ylog" => "residual = :pit, $(ylog)"]
        )
    )
    return cs
end
