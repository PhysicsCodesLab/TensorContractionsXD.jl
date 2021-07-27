# Auxiliary tools to manipulate tuples
"""
    tsetdiff(a::Tuple, b::Tuple)

Assume `b` is completely contained in `a`, and return the tuple with elements of `a` that are
not in `b`.
"""
tsetdiff(a::Tuple, b::Tuple{}) = a
tsetdiff(a::Tuple{Any}, b::Tuple{Any}) = ()
tsetdiff(a::Tuple, b::Tuple{Any}) = a[1] == b[1] ? tail(a) : (a[1], tsetdiff(tail(a), b)...)
tsetdiff(a::Tuple, b::Tuple) = tsetdiff(tsetdiff(a, (b[1],)), tail(b))

@noinline tseterror() = throw(ArgumentError("tuples did not meet requirements"))

"""
    tunique(src::Tuple, dst::Tuple)

Assume that every element appears exactly twice in the combination of `src` and `dst`, and
each elements in `dst` only appears once in `dst`. Ruturn the list which contains every
element only once.
"""
tunique(src::Tuple) = tunique(src, ())
tunique(src::NTuple{N,Any}, dst::NTuple{N,Any}) where {N} = dst
tunique(src::Tuple, dst::Tuple) = src[1] in dst ? tunique((tail(src)..., src[1]), dst) : tunique(tail(src), (dst..., src[1]))

"""
    _findfirst(args...)

Reload the methods `findfirst(args...)` by returning `0` if nothing is found to make it
type stable.
"""
_findfirst(args...) = (i = findfirst(args...); i === nothing ? 0 : i)

"""
    _findnext(args...)

Reload the methods `findnext(args...)` by returning `0` if nothing is found to make it
type stable.
"""
_findnext(args...) = (i = findnext(args...); i === nothing ? 0 : i)

"""
    _findlast(args...)

Reload the methods `findlast(args...)` by returning `0` if nothing is found to make it
type stable.
"""
_findlast(args...) = (i = findlast(args...); i === nothing ? 0 : i)

# Auxiliary method to analyze trace indices
"""
    unique2(itr)

Returns an array containing only those elements that appear exactly once in itr,
and without any elements that appear more than once.
"""
function unique2(itr)
    out = reshape(collect(itr),length(itr))
    i = 1
    while i < length(out)
        inext = _findnext(isequal(out[i]), out, i+1)
        if inext == nothing || inext == 0
            i += 1
            continue
        end
        while !(inext == nothing || inext == 0)
            deleteat!(out,inext)
            inext = _findnext(isequal(out[i]), out, i+1)
        end
        deleteat!(out,i)
    end
    out
end

# Extract index information
"""
    add_indices(IA::NTuple{NA,Any}, IC::NTuple{NC,Any}) where {NA,NC}

Assume `IA` and `IC` has the same length, and `IA` can be obtained from `IC` by permutations.
Return the indices of `IA` where the elements of `IC` appears, i.e., the permutation.
"""
function add_indices(IA::NTuple{NA,Any}, IC::NTuple{NC,Any}) where {NA,NC}
    indCinA = map(l->_findfirst(isequal(l), IA), IC)
    (NA == NC && isperm(indCinA)) || throw(IndexError("invalid index specification: $IA to $IC"))
    return indCinA
end

"""
    trace_indices(IA::NTuple{NA,Any}, IC::NTuple{NC,Any}) where {NA,NC}

`IC` is obtained by tracing out the indices that appear twice in `IA` and then do a
permutation. Return the permutation and two lists which corresponds to the positions of the
first and second traced indices.
"""
function trace_indices(IA::NTuple{NA,Any}, IC::NTuple{NC,Any}) where {NA,NC}
    isodd(length(IA)-length(IC)) && throw(IndexError("invalid trace specification: $IA to $IC"))
    Itrace = tunique(tsetdiff(IA, IC)) # give the indices list that needed to be traced

    cindA1 = map(l->_findfirst(isequal(l), IA), Itrace)
    cindA2 = map(l->_findnext(isequal(l), IA, _findfirst(isequal(l), IA)+1), Itrace)
    indCinA = map(l->_findfirst(isequal(l), IA), IC)

    pA = (indCinA..., cindA1..., cindA2...)
    (isperm(pA) && length(pA) == NA) || throw(IndexError("invalid trace specification: $IA to $IC"))
    return indCinA, cindA1, cindA2
end

"""
    contract_indices(IA::NTuple{NA,Any}, IB::NTuple{NB,Any}, IC::NTuple{NC,Any}) where {NA,NB,NC}

`IC` is obtained by contracting the indices that are shared by `IA` and `IB`, and then do
a permutation. Return `oindA, cindA, oindB, cindB, indCinoAB`, where `oindA`, `oindB` are
positions of indices in `IA` and `IB` that are left open, `cindA`, `cindB` are positions of
indices in `IA` and `IB` that are contracted, `indCinoAB` is the permutation of the indices
`IC` to `(IopenA..., IopenB...)`.
"""
function contract_indices(IA::NTuple{NA,Any}, IB::NTuple{NB,Any}, IC::NTuple{NC,Any}) where {NA,NB,NC}
    # labels
    IAB = (IA..., IB...)
    isodd(length(IAB)-length(IC)) && throw(IndexError("invalid contraction pattern: $IA and $IB to $IC"))
    Icontract = tunique(tsetdiff(IAB, IC))
    IopenA = tsetdiff(IA, Icontract)
    IopenB = tsetdiff(IB, Icontract)

    # to indices
    cindA = map(l->_findfirst(isequal(l), IA), Icontract)
    cindB = map(l->_findfirst(isequal(l), IB), Icontract)
    oindA = map(l->_findfirst(isequal(l), IA), IopenA)
    oindB = map(l->_findfirst(isequal(l), IB), IopenB)
    indCinoAB = map(l->_findfirst(isequal(l), (IopenA..., IopenB...)), IC)

    if !isperm((oindA..., cindA...)) || !isperm((oindB..., cindB...)) || !isperm(indCinoAB)
        throw(IndexError("invalid contraction pattern: $IA and $IB to $IC"))
    end

    return oindA, cindA, oindB, cindB, indCinoAB
end
