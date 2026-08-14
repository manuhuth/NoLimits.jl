# Optional dependencies. Features that pull in heavy or rarely-needed packages live in
# package extensions, so a plain `using NoLimits` does not pay for them. Each such feature
# calls `_require_ext` at its public entry point, which turns "the extension is not loaded"
# into one actionable sentence instead of a MethodError from somewhere inside the call.

"""
    _require_ext(ext::Symbol, pkgs, what::AbstractString)

Return `nothing` when the `ext` extension is loaded; otherwise throw an error naming the
package(s) the user has to add and load. `pkgs` is a `Symbol` or a tuple of `Symbol`s.
"""
@noinline function _require_ext(ext::Symbol, pkgs, what::AbstractString)
    Base.get_extension(@__MODULE__, ext) === nothing || return nothing
    names = pkgs isa Symbol ? (pkgs,) : pkgs
    adds = join(("\"$(p)\"" for p in names), ", ")
    uses = join(names, ", ")
    plural = length(names) == 1 ? "an optional dependency" : "optional dependencies"
    error("""
          $what requires $(join(string.(names) .* ".jl", ", ", " and ")), \
          which NoLimits declares as $plural and therefore does not install or load for you.

              using Pkg; Pkg.add([$adds])
              using $uses

          Load $(length(names) == 1 ? "it" : "them") alongside NoLimits and retry.""")
end
