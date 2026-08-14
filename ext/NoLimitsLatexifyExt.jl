module NoLimitsLatexifyExt

# LaTeX rendering for `show_equations(...; latex = true)`. The plain-text path needs
# nothing, so Latexify (and the Ghostscript/JpegTurbo artifacts behind it) stays optional.

import Latexify
import NoLimits: _eq_latexraw, _eq_latexify

_eq_latexraw(x; kwargs...) = Latexify.latexraw(x; kwargs...)
_eq_latexify(x; kwargs...) = Latexify.latexify(x; kwargs...)

end
