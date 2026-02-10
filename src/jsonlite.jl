module JSONLite

export parse_json, write_json

mutable struct Parser
    data::Vector{UInt8}
    i::Int
    n::Int
end

Parser(s::AbstractString) = Parser(Vector{UInt8}(codeunits(s)), 1, ncodeunits(s))

@inline function _peek(p::Parser)
    p.i > p.n && return UInt8(0)
    return p.data[p.i]
end

@inline function _advance!(p::Parser, k::Int = 1)
    p.i += k
    return nothing
end

@inline function _skip_ws!(p::Parser)
    while p.i <= p.n
        b = p.data[p.i]
        if b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09
            p.i += 1
        else
            break
        end
    end
    return nothing
end

function _error(p::Parser, msg::AbstractString)
    error("JSON parse error at byte $(p.i): $msg")
end

function _parse_hex4(p::Parser)
    p.i + 3 > p.n && _error(p, "invalid unicode escape")
    v = 0
    for _ in 1:4
        b = p.data[p.i]
        d =
            (b >= 0x30 && b <= 0x39) ? (b - 0x30) :
            (b >= 0x41 && b <= 0x46) ? (b - 0x41 + 10) :
            (b >= 0x61 && b <= 0x66) ? (b - 0x61 + 10) : UInt8(255)
        d == 0xff && _error(p, "invalid hex digit in unicode escape")
        v = (v << 4) + Int(d)
        p.i += 1
    end
    return v
end

function _parse_string!(p::Parser)
    _peek(p) == 0x22 || _error(p, "expected string opening quote")
    _advance!(p)
    io = IOBuffer()
    while p.i <= p.n
        b = p.data[p.i]
        if b == 0x22
            _advance!(p)
            return String(take!(io))
        elseif b == 0x5c
            _advance!(p)
            p.i > p.n && _error(p, "dangling escape")
            esc = p.data[p.i]
            if esc == 0x22
                write(io, UInt8('"'))
                _advance!(p)
            elseif esc == 0x5c
                write(io, UInt8('\\'))
                _advance!(p)
            elseif esc == 0x2f
                write(io, UInt8('/'))
                _advance!(p)
            elseif esc == 0x62
                write(io, UInt8('\b'))
                _advance!(p)
            elseif esc == 0x66
                write(io, UInt8('\f'))
                _advance!(p)
            elseif esc == 0x6e
                write(io, UInt8('\n'))
                _advance!(p)
            elseif esc == 0x72
                write(io, UInt8('\r'))
                _advance!(p)
            elseif esc == 0x74
                write(io, UInt8('\t'))
                _advance!(p)
            elseif esc == 0x75
                _advance!(p)
                cp = _parse_hex4(p)
                print(io, Char(cp))
            else
                _error(p, "unsupported escape sequence")
            end
        else
            write(io, b)
            _advance!(p)
        end
    end
    _error(p, "unterminated string")
end

function _parse_literal!(p::Parser, literal::AbstractString, value)
    bytes = codeunits(literal)
    p.i + length(bytes) - 1 <= p.n || _error(p, "unexpected EOF")
    for k in eachindex(bytes)
        p.data[p.i + k - 1] == bytes[k] || _error(p, "invalid token")
    end
    _advance!(p, length(bytes))
    return value
end

function _parse_number!(p::Parser)
    start = p.i
    _peek(p) == 0x2d && _advance!(p)

    if _peek(p) == 0x30
        _advance!(p)
    else
        (_peek(p) >= 0x31 && _peek(p) <= 0x39) || _error(p, "invalid number")
        while true
            b = _peek(p)
            if b >= 0x30 && b <= 0x39
                _advance!(p)
            else
                break
            end
        end
    end

    is_float = false
    if _peek(p) == 0x2e
        is_float = true
        _advance!(p)
        (_peek(p) >= 0x30 && _peek(p) <= 0x39) || _error(p, "invalid fraction")
        while true
            b = _peek(p)
            if b >= 0x30 && b <= 0x39
                _advance!(p)
            else
                break
            end
        end
    end

    if _peek(p) == 0x65 || _peek(p) == 0x45
        is_float = true
        _advance!(p)
        if _peek(p) == 0x2b || _peek(p) == 0x2d
            _advance!(p)
        end
        (_peek(p) >= 0x30 && _peek(p) <= 0x39) || _error(p, "invalid exponent")
        while true
            b = _peek(p)
            if b >= 0x30 && b <= 0x39
                _advance!(p)
            else
                break
            end
        end
    end

    s = String(p.data[start:(p.i - 1)])
    if is_float
        v = tryparse(Float64, s)
        v === nothing && _error(p, "invalid float")
        return v
    end
    v = tryparse(Int64, s)
    v === nothing && _error(p, "invalid integer")
    return v
end

function _parse_array!(p::Parser)
    _peek(p) == 0x5b || _error(p, "expected '['")
    _advance!(p)
    _skip_ws!(p)
    out = Any[]
    if _peek(p) == 0x5d
        _advance!(p)
        return out
    end
    while true
        push!(out, _parse_value!(p))
        _skip_ws!(p)
        b = _peek(p)
        if b == 0x2c
            _advance!(p)
            _skip_ws!(p)
        elseif b == 0x5d
            _advance!(p)
            break
        else
            _error(p, "expected ',' or ']'")
        end
    end
    return out
end

function _parse_object!(p::Parser)
    _peek(p) == 0x7b || _error(p, "expected '{'")
    _advance!(p)
    _skip_ws!(p)
    out = Dict{String, Any}()
    if _peek(p) == 0x7d
        _advance!(p)
        return out
    end
    while true
        _peek(p) == 0x22 || _error(p, "expected object key")
        k = _parse_string!(p)
        _skip_ws!(p)
        _peek(p) == 0x3a || _error(p, "expected ':'")
        _advance!(p)
        _skip_ws!(p)
        out[k] = _parse_value!(p)
        _skip_ws!(p)
        b = _peek(p)
        if b == 0x2c
            _advance!(p)
            _skip_ws!(p)
        elseif b == 0x7d
            _advance!(p)
            break
        else
            _error(p, "expected ',' or '}'")
        end
    end
    return out
end

function _parse_value!(p::Parser)
    _skip_ws!(p)
    b = _peek(p)
    if b == 0x7b
        return _parse_object!(p)
    elseif b == 0x5b
        return _parse_array!(p)
    elseif b == 0x22
        return _parse_string!(p)
    elseif b == 0x74
        return _parse_literal!(p, "true", true)
    elseif b == 0x66
        return _parse_literal!(p, "false", false)
    elseif b == 0x6e
        return _parse_literal!(p, "null", nothing)
    elseif b == 0x2d || (b >= 0x30 && b <= 0x39)
        return _parse_number!(p)
    end
    _error(p, "unexpected token")
end

function parse_json(s::AbstractString)
    p = Parser(s)
    value = _parse_value!(p)
    _skip_ws!(p)
    p.i <= p.n && _error(p, "trailing content")
    return value
end

function _write_escaped_string(io::IO, s::AbstractString)
    write(io, '"')
    for c in s
        if c == '"'
            write(io, "\\\"")
        elseif c == '\\'
            write(io, "\\\\")
        elseif c == '\b'
            write(io, "\\b")
        elseif c == '\f'
            write(io, "\\f")
        elseif c == '\n'
            write(io, "\\n")
        elseif c == '\r'
            write(io, "\\r")
        elseif c == '\t'
            write(io, "\\t")
        else
            print(io, c)
        end
    end
    write(io, '"')
end

function _write_value(io::IO, x)
    if x === nothing
        write(io, "null")
    elseif x isa Bool
        write(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        if !isfinite(x)
            error("Cannot JSON-encode non-finite float: $x")
        end
        print(io, x)
    elseif x isa AbstractString
        _write_escaped_string(io, x)
    elseif x isa AbstractVector
        write(io, '[')
        for (i, item) in enumerate(x)
            i > 1 && write(io, ',')
            _write_value(io, item)
        end
        write(io, ']')
    elseif x isa AbstractDict
        write(io, '{')
        first = true
        for (k, v) in x
            first || write(io, ',')
            first = false
            _write_escaped_string(io, String(k))
            write(io, ':')
            _write_value(io, v)
        end
        write(io, '}')
    else
        error("Unsupported JSON type: $(typeof(x))")
    end
end

function write_json(io::IO, x)
    _write_value(io, x)
    return nothing
end

function write_json(path::AbstractString, x)
    open(path, "w") do io
        write_json(io, x)
        write(io, '\n')
    end
    return path
end

end
