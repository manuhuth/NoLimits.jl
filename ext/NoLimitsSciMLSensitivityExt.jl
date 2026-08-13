module NoLimitsSciMLSensitivityExt

# The Enzyme → DiffEqBase → SciMLSensitivity adjoint route needs these five operations
# on the parameter object it carries through the reverse pass. Nothing else in NoLimits
# calls them, so SciMLSensitivity (and with it Enzyme, ReverseDiff, Tracker, GPUCompiler)
# only loads for users who actually differentiate an ODE in reverse mode.

using NoLimits: DEParams
import SciMLSensitivity: recursive_copyto!, recursive_add!, recursive_sub!, recursive_neg!,
                         allocate_vjp

function recursive_copyto!(y::AbstractArray, x::DEParams)
    copyto!(y, vcat(collect(x.θ), collect(x.η)))
end
recursive_copyto!(y::DEParams, x::DEParams) = (copyto!(y.θ, x.θ);
copyto!(y.η, x.η);
y)
recursive_neg!(x::DEParams) = (x.θ .*= -1; x.η .*= -1; x)
recursive_add!(y::DEParams, x::DEParams) = (y.θ .+= x.θ;
y.η .+= x.η;
y)
recursive_sub!(y::DEParams, x::DEParams) = (y.θ .-= x.θ;
y.η .-= x.η;
y)
allocate_vjp(λ::AbstractArray, x::DEParams) = fill!(similar(λ, length(x)), zero(eltype(λ)))
allocate_vjp(x::DEParams) = zero(vcat(collect(x.θ), collect(x.η)))

end
