"""
    similarstructure_from_indices

Return the structure of that similar to the input and should always be specified for
custom array/tensor types.
"""
function similarstructure_from_indices end

"""
    memsize(a::Any)

Return the size of the `a` in memory. This is a generic definition, not very efficient.
Provide more efficient version if possible
"""
memsize(a::Any) = Base.summarysize(a)

"""
    similar_from_structure(A, T, structure)

Return an object that is similar to `A` with `eltype = T` and the structure specified by
`structure`.

Generic definitions, should be overwritten if your array/tensor type does not support
`Base.similar(object, eltype, structure)`.
"""
function similar_from_structure(A, T, structure)
    if isbits(T)
        similar(A, T, structure)
    else
        fill!(similar(A, T, structure), zero(T)) # this fixes BigFloat issues
    end
end

"""
    similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol)

Return the object that is similar to `A` which has an `eltype` given by `T` and whose
left indices correspond to the indices `p1` from `op(A)`, and its right indices correspond
to the indices `p2` from `op(A)`, where `op` is `conj` if `CA == :C` or does nothing if
`CA == :N` (default).
"""
function similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol)
    structure = similarstructure_from_indices(T, p1, p2, A, CA)
    similar_from_structure(A, T, structure)
end

"""
    similar_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
                p1::IndexTuple, p2::IndexTuple, A, B, CA::Symbol, CB::Symbol)

Return an object that is similar to the tensor map `A`, i.e., the HomSpace from
the domain to the codomain, where the codomain is the tensor product of spaces selected by
`p1` from the list of spaces that are selected by `poA` from `op(A)` and `poB` from `op(B)`
in sequence, and the domain is that by `p2`. Th `opA` is `conj` if `CA == :C` or does
nothing if `CA == :N` (default), and similarly for `opB`.
"""
function similar_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
                p1::IndexTuple, p2::IndexTuple, A, B, CA::Symbol, CB::Symbol)
    structure = similarstructure_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
    similar_from_structure(A, T, structure)
end

"""
    similartype_from_indices(T::Type, p1, p2, A, CA)

Return the type of the object that obtained by `similar_from_indices`. This should work
generically but can be overwritten.
"""
function similartype_from_indices(T::Type, p1, p2, A, CA)
    Core.Compiler.return_type(similar_from_indices,
                                Tuple{Type{T}, typeof(p1), typeof(p2), typeof(A), Symbol})
end

"""
    similartype_from_indices(T::Type, poA, poB, p1, p2, A, B, CA, CB)

Return the type of the object that obtained by `similar_from_indices`. This should work
generically but can be overwritten.
"""
function similartype_from_indices(T::Type, poA, poB, p1, p2, A, B, CA, CB)
    Core.Compiler.return_type(similar_from_indices, Tuple{Type{T}, typeof(poA), typeof(poB),
                typeof(p1), typeof(p2), typeof(A), typeof(B), Symbol, Symbol})
end

"""
    cached_similar_from_indices(sym::Symbol, T::Type, p1::IndexTuple, p2::IndexTuple,
                                    A, CA::Symbol)

The version of `similar_from_indices` that uses cache is `use_cache() == true`. This is
generic, should probably not be overwritten.
"""
function cached_similar_from_indices(sym::Symbol, T::Type, p1::IndexTuple, p2::IndexTuple,
                                        A, CA::Symbol)
    if use_cache()
        structure = similarstructure_from_indices(T, p1, p2, A, CA)
        typ = similartype_from_indices(T, p1, p2, A, CA)
        key = (sym, taskid(), typ, structure)
        C::typ = get!(cache, key) do
            similar_from_indices(T, p1, p2, A, CA)
        end
        return C
    else
        return similar_from_indices(T, p1, p2, A, CA)
    end
end

"""
    cached_similar_from_indices(sym::Symbol, T::Type, poA::IndexTuple, poB::IndexTuple,
                        p1::IndexTuple, p2::IndexTuple, A, B, CA::Symbol, CB::Symbol)

The version of `similar_from_indices` that uses cache is `use_cache() == true`. This is
generic, should probably not be overwritten.
"""
function cached_similar_from_indices(sym::Symbol, T::Type, poA::IndexTuple, poB::IndexTuple,
                        p1::IndexTuple, p2::IndexTuple, A, B, CA::Symbol, CB::Symbol)

    if use_cache()
        structure = similarstructure_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
        typ = similartype_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
        key = (sym, taskid(), typ, structure)
        C::typ = get!(cache, key) do
            similar_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
        end
        return C
    else
        return similar_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
    end
end
