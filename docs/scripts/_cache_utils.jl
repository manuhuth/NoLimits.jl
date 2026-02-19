using Serialization

@inline function _docs_tutorial_env_is_true(name::AbstractString, default::Bool=false)
    value = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return value in ("1", "true", "yes", "on")
end

@inline function _docs_tutorial_env_is_false(name::AbstractString, default::Bool=false)
    value = lowercase(strip(get(ENV, name, default ? "false" : "true")))
    return value in ("0", "false", "no", "off")
end

@inline function docs_tutorials_use_cache()
    return !_docs_tutorial_env_is_false("DOCS_TUTORIALS_USE_CACHE", false)
end

@inline function docs_tutorials_force_recompute()
    return _docs_tutorial_env_is_true("DOCS_RECOMPUTE_TUTORIALS", false)
end

@inline function docs_tutorials_debug_cache()
    return _docs_tutorial_env_is_true("DOCS_TUTORIALS_DEBUG_CACHE", false)
end

@inline function docs_tutorials_root()
    return normpath(joinpath(dirname(pathof(NoLimits)), "..", "docs"))
end

@inline function docs_tutorial_cache_dir()
    if haskey(ENV, "DOCS_TUTORIAL_CACHE_DIR")
        return abspath(ENV["DOCS_TUTORIAL_CACHE_DIR"])
    end
    return joinpath(docs_tutorials_root(), "cache")
end

@inline function docs_tutorial_cache_file(cache_key::AbstractString)
    return joinpath(docs_tutorial_cache_dir(), cache_key * ".jls")
end

function write_tutorial_cache(cache_key::AbstractString, payload)
    cache_file = docs_tutorial_cache_file(cache_key)
    mkpath(dirname(cache_file))
    serialize(cache_file, payload)
    return cache_file
end

function load_or_compute_tutorial_cache(cache_key::AbstractString, compute_fn::Function)
    cache_file = docs_tutorial_cache_file(cache_key)
    use_cache = docs_tutorials_use_cache()
    force_recompute = docs_tutorials_force_recompute()

    if use_cache && !force_recompute && isfile(cache_file)
        try
            return deserialize(cache_file)
        catch err
            if docs_tutorials_debug_cache()
                @warn "Failed to deserialize tutorial cache. Recomputing." cache_file exception=(err, catch_backtrace())
            end
            try
                rm(cache_file; force=true)
            catch
            end
        end
    end

    payload = compute_fn()

    if use_cache
        try
            write_tutorial_cache(cache_key, payload)
        catch err
            if docs_tutorials_debug_cache()
                @warn "Failed to write tutorial cache." cache_file exception=(err, catch_backtrace())
            end
        end
    end

    return payload
end

function load_or_compute_tutorial_cache(compute_fn::Function, cache_key::AbstractString)
    return load_or_compute_tutorial_cache(cache_key, compute_fn)
end
