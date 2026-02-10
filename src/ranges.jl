module Ranges

export parse_ranges, format_ranges

"""
Parse range expression like "1-30,40-50,66" into (start, stop) tuples.
"""
function parse_ranges(range_str::AbstractString)
    parts = split(range_str, ',')
    ranges = Tuple{Int, Int}[]
    for raw in parts
        p = strip(raw)
        isempty(p) && continue
        if occursin('-', p)
            ab = split(p, '-')
            length(ab) == 2 || error("Invalid range token: $p")
            a = parse(Int, strip(ab[1]))
            b = parse(Int, strip(ab[2]))
            push!(ranges, (a, b))
        else
            x = parse(Int, p)
            push!(ranges, (x, x))
        end
    end
    return ranges
end

"""
Compress integer positions into compact ranges like "1-3,5,7-9".
"""
function format_ranges(values::AbstractVector{<:Integer})
    isempty(values) && return ""
    xs = sort!(unique(Int.(values)))

    io = IOBuffer()
    start = xs[1]
    prev = xs[1]
    first_out = true

    function emit_seg(a::Int, b::Int)
        if !first_out
            write(io, ',')
        end
        if a == b
            print(io, a)
        else
            print(io, a, "-", b)
        end
    end

    for x in xs[2:end]
        if x == prev + 1
            prev = x
            continue
        end
        emit_seg(start, prev)
        first_out = false
        start = x
        prev = x
    end
    emit_seg(start, prev)

    return String(take!(io))
end

end
