export get_equation_lines
export show_equations

using Latexify

@inline _eq_clean_expr(ex::Expr) = Base.remove_linenums!(deepcopy(ex))

@inline function _eq_assignment_expr(ex::Expr)
    if ex.head == :call && length(ex.args) == 3 && ex.args[1] == :~
        return Expr(:(=), ex.args[2], ex.args[3])
    end
    return ex
end

@inline function _eq_derivative_state(lhs)
    if lhs isa Expr && lhs.head == :call && length(lhs.args) == 2 && lhs.args[1] == :D
        st = lhs.args[2]
        st isa Symbol && return st
    end
    return nothing
end

@inline function _eq_show_plain_piece(x)
    io = IOBuffer()
    if x isa Expr
        Base.show_unquoted(io, x)
    else
        show(io, x)
    end
    return String(take!(io))
end

@inline function _eq_show_latex_piece(x)
    try
        return String(Latexify.latexraw(x))
    catch
        return _eq_show_plain_piece(x)
    end
end

@inline function _eq_collect_vector_component_pairs!(
    pairs::Vector{Tuple{String, String}},
    x,
)
    if x isa Expr
        if x.head == :ref && length(x.args) == 2
            base = x.args[1]
            idx = x.args[2]
            if idx isa Integer
                push!(pairs, (_eq_show_latex_piece(base), string(idx)))
            end
        end
        for a in x.args
            _eq_collect_vector_component_pairs!(pairs, a)
        end
    end
    return pairs
end

@inline function _eq_regex_escape(s::String)
    return replace(s, r"([\\.^$|?*+(){}\[\]])" => s"\\\1")
end

function _eq_apply_vector_component_notation(line::String, ex::Expr)
    pairs = Tuple{String, String}[]
    _eq_collect_vector_component_pairs!(pairs, ex)
    isempty(pairs) && return line

    seen = Set{Tuple{String, String}}()
    out = line
    for pair in pairs
        pair in seen && continue
        push!(seen, pair)
        base_tex, idx_tex = pair
        base_pat = _eq_regex_escape(base_tex)
        idx_pat = _eq_regex_escape(idx_tex)
        pat = Regex(base_pat * raw"\\left\(\s*" * idx_pat * raw"\s*\\right\)")
        repl = base_tex * raw"^{\left(" * idx_tex * raw"\right)}"
        out = replace(out, pat => repl)
    end
    return out
end

function _eq_plain_string(ex::Expr)
    ex_clean = _eq_assignment_expr(_eq_clean_expr(ex))
    if ex_clean.head == :(=)
        lhs = ex_clean.args[1]
        rhs = ex_clean.args[2]
        st = _eq_derivative_state(lhs)
        if st !== nothing
            return string(st, "̇(t) = ", _eq_show_plain_piece(rhs))
        end
    end
    return _eq_show_plain_piece(ex_clean)
end

function _eq_latex_string(ex::Expr)
    ex_clean = _eq_assignment_expr(_eq_clean_expr(ex))
    if ex_clean.head == :(=)
        lhs = ex_clean.args[1]
        rhs = ex_clean.args[2]
        st = _eq_derivative_state(lhs)
        if st !== nothing
            lhs_tex = _eq_show_latex_piece(st)
            rhs_tex = _eq_show_latex_piece(rhs)
            line = "\\dot{" * lhs_tex * "}(t) = " * rhs_tex
            return _eq_apply_vector_component_notation(line, ex_clean)
        end
    end
    try
        line = String(Latexify.latexify(ex_clean))
        return _eq_apply_vector_component_notation(line, ex_clean)
    catch
        return _eq_plain_string(ex_clean)
    end
end

function _eq_latex_line(ex::Expr)
    ex_clean = _eq_assignment_expr(_eq_clean_expr(ex))
    if ex_clean.head == :(=)
        lhs = ex_clean.args[1]
        rhs = ex_clean.args[2]
        st = _eq_derivative_state(lhs)
        if st !== nothing
            lhs_tex = _eq_show_latex_piece(st)
            rhs_tex = _eq_show_latex_piece(rhs)
            line = "\\dot{" * lhs_tex * "}(t) = " * rhs_tex
            return _eq_apply_vector_component_notation(line, ex_clean)
        end
    end
    try
        line = String(Latexify.latexraw(ex_clean))
        return _eq_apply_vector_component_notation(line, ex_clean)
    catch
        s = _eq_latex_string(ex_clean)
        return strip(s, ['$', ' '])
    end
end

function _eq_align_at_equals(line::String)
    occursin("&=", line) && return line
    i = findfirst(==('='), line)
    i === nothing && return line
    left = rstrip(line[1:(i - 1)])
    right = lstrip(line[(i + 1):end])
    return left * " &= " * right
end

function _eq_latex_block(lines::Vector{Expr}; numbered::Bool=false)
    tex_lines = String[]
    for (i, ex) in enumerate(lines)
        line = _eq_align_at_equals(_eq_latex_line(ex))
        if numbered
            line = "\\text{$(i). } " * line
        end
        push!(tex_lines, line)
    end
    body = join(tex_lines, " \\\\\n")
    return Latexify.latexraw("\$\\begin{aligned}\n" * body * "\n\\end{aligned}\$"; parse=false)
end

function _eq_try_display(block)
    try
        display(block)
        return true
    catch
        return false
    end
end

function _eq_try_render(block)
    if displayable(MIME("juliavscode/html"))
        try
            Latexify.render(block, MIME("juliavscode/html"))
            return true
        catch
        end
    end

    if displayable(MIME("image/svg+xml"))
        try
            Latexify.render(block, MIME("image/svg"); callshow=true, open=false)
            return true
        catch
        end
    end

    if Sys.which("lualatex") !== nothing
        try
            Latexify.render(block)
            return true
        catch
        end
    end

    return false
end

function get_equation_lines(m::Model)
    lines = Expr[]

    if m.de.prede !== nothing
        append!(lines, get_prede_lines(m.de.prede))
    end

    if m.de.de !== nothing
        append!(lines, get_de_lines(m.de.de))
    end

    f = m.formulas.formulas
    f_lines = get_formulas_lines(f)
    if !isempty(f_lines)
        append!(lines, f_lines)
    else
        ir = get_formulas_ir(f)
        for i in eachindex(ir.det_names)
            push!(lines, Expr(:(=), ir.det_names[i], ir.det_exprs[i]))
        end
        for i in eachindex(ir.obs_names)
            push!(lines, Expr(:call, :~, ir.obs_names[i], ir.obs_exprs[i]))
        end
    end

    return [_eq_clean_expr(ex) for ex in lines]
end

function show_equations(io::IO, m::Model; latex::Bool=true, numbered::Bool=false)
    lines = get_equation_lines(m)
    isempty(lines) && return nothing

    for (i, ex) in enumerate(lines)
        line = latex ? _eq_align_at_equals(_eq_latex_line(ex)) : _eq_plain_string(ex)
        if numbered
            print(io, i, ". ", line)
        else
            print(io, line)
        end
        i < length(lines) && print(io, '\n')
    end
    return nothing
end

function show_equations(m::Model; latex::Bool=true, numbered::Bool=false)
    lines = get_equation_lines(m)
    isempty(lines) && return nothing

    if !latex
        return sprint(io -> show_equations(io, m; latex=false, numbered=numbered))
    end

    block = _eq_latex_block(lines; numbered=numbered)
    return block
end
