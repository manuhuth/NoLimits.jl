module NoLimitsCopulasExt

# Copulas.jl interop. Loads when NoLimits and Copulas are both present. A
# `SklarDist` exposes its marginals, which is all the generic RE machinery
# needs: marginal-quantile transports (GHQuadrature), linked MCMC sampling,
# and exact plug-in means (a copula never shifts its marginals).

using NoLimits: NoLimits
using Copulas: SklarDist

NoLimits._re_marginals(d::SklarDist) = d.m

end
