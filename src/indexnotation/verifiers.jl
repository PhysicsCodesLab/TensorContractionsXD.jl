const prime = Symbol("'")

"""
    isindex(ex)

Check if the input `ex` is a valid index for a tensor. It should be a `Symbol` or an `Int`,
and prime of them are also allowed.
"""
function isindex(ex)
    if isa(ex, Symbol) || isa(ex, Int)
        return true
    elseif isa(ex, Expr) && ex.head == prime && length(ex.args) == 1
        return isindex(ex.args[1])
    else
        return false
    end
end

"""
    istensor(ex::Expr)

Check if the input `ex` is a tensor object with valid indices. The possible forms of the
tensor could be:
1. `A[]` with head `:ref`
2. `A[a, b; c, d]` with head `:ref`
3. `A[a, b, c]` with head `:ref`
4. `A[a b c]` with head `:typed_hcat`
5. `A[a b; c d]` with head `:typed_vcat`
6. `A[(a,b);(c,d)]` with head `:typed_vcat`
where `A` can itself be an expression, such as `A[1:2:end, :][a, b, c]`.
"""
istensor(ex) = false
function istensor(ex::Expr)
    if ex.head == :ref || ex.head == :typed_hcat
        if length(ex.args) == 1
            # A[]
            return true
        elseif isa(ex.args[2], Expr) && ex.args[2].head == :parameters
            # A[a,b,...;c,d,...]
            return all(isindex, ex.args[2].args) || all(isindex, ex.args[3:end])
        else
            # A[a,b,c,...] or A[a b c ...]
            return all(isindex, ex.args[2:end])
        end
    elseif ex.head == :typed_vcat && length(ex.args) == 3
        length(ex.args) == 3 || return false
        if isa(ex.args[2], Expr) && (ex.args[2].head == :row || ex.args[2].head == :tuple)
            # row: A[a b ...; ...]; or tuple: A[(a,b,...); ...]
            all(isindex, ex.args[2].args) || return false
        else
            # A[a; ...]
            isindex(ex.args[2]) || return false
        end
        if isa(ex.args[3], Expr) && (ex.args[3].head == :row || ex.args[3].head == :tuple)
            # row: A[...; b c ...]; or tuple: A[...;(b,c,...)]
            all(isindex, ex.args[3].args) || return false
        else
            # A[...; c]
            isindex(ex.args[3]) || return false
        end
        return true
    end
    return false
end

"""
    isgeneraltensor(ex::Expr)

Check if the input `ex` is a general tensor in one of the following forms:
1. `ex` itself is a tensor
2. `+ tensor` or `- tensor`
3. `conj(tensor)`, or `adjoint(tensor)`, or `tensor'`, or `transpose(tensor)`
4. `α * tensor` or `tensor * α`, where `α` is a scalar
5. `tensor/α` or `α\tensor`, where `α` is a scalar
"""
isgeneraltensor(ex) = false
function isgeneraltensor(ex::Expr)
    if istensor(ex)
        return true
    elseif ex.head == :call && ex.args[1] == :+ && length(ex.args) == 2
        # unary plus
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :- && length(ex.args) == 2
        # unary minus
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        # conjugation
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :adjoint && length(ex.args) == 2
        # adjoint
        return isgeneraltensor(ex.args[2])
    elseif ex.head == prime && length(ex.args) == 1
        # adjoint
        return isgeneraltensor(ex.args[1])
    elseif ex.head == :call && ex.args[1] == :transpose && length(ex.args) == 2
        # conjugation
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :*
        # scalar multiplication
        count = 0
        for i = 2:length(ex.args)
            if isgeneraltensor(ex.args[i])
                count += 1
            elseif !isscalarexpr(ex.args[i])
                return false
            end
        end
        return count == 1
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        # scalar multiplication
        return (isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2]))
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        # scalar multiplication
        return (isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3]))
    end
    return false
end

"""
    isscalarexpr(ex::Expr)

Check if the input `ex` is a scalar expression with no indices.
"""
isscalarexpr(ex::Symbol) = true
isscalarexpr(ex::Number) = true
isscalarexpr(ex) = true
function isscalarexpr(ex::Expr)
    if ex.head == :call && ex.args[1] == :scalar
        return istensorexpr(ex.args[2])
    elseif ex.head in (:ref, :typed_vcat, :typed_hcat)
        return false
    else
        return all(isscalarexpr, ex.args)
    end
end

"""
    istensorexpr(ex)

Check if the input `ex` is a tensor expression, i.e. something that can be evaluated to a
tensor.
"""
function istensorexpr(ex)
    if isgeneraltensor(ex)
        return true
    elseif isa(ex, Expr) && ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        # linear combination of general tensors
        return all(istensorexpr, ex.args[2:end])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :*
        # multiplications of several general tensors
        count = 0
        for i = 2:length(ex.args)
            if istensorexpr(ex.args[i])
                count += 1
            elseif !isscalarexpr(ex.args[i])
                return false
            end
        end
        return count > 0
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        return istensorexpr(ex.args[2]) && isscalarexpr(ex.args[3])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        return istensorexpr(ex.args[3]) && isscalarexpr(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        # conj all general tensors
        return istensorexpr(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :adjoint && length(ex.args) == 2
        # adjoint all general tensors
        return istensorexpr(ex.args[2])
    elseif isa(ex, Expr) && ex.head == prime
        # adjoint all general tensors
        return istensorexpr(ex.args[1])
    end
    return false
end

"""
    hastraceindices(ex)

Check if the input tensor `ex` has indices that needed to be contracted within the tensor,
i.e., whether any trace operation is applied on the tensor.
"""
function hastraceindices(ex)
    obj, leftind, rightind, = decomposegeneraltensor(ex)
    allind = vcat(leftind, rightind)
    return length(allind) != length(unique(allind))
end

"""
    istensorcontraction(ex)

Check if the input `ex` is an expression for a tensor contraction operation in the form such
as `A[a b]*B[c a]*...`.
"""
function istensorcontraction(ex)
    if isa(ex, Expr) && ex.head == :call && ex.args[1] == :*
        return count(istensorexpr, ex.args[2:end]) >= 2
    end
    return false
end

"""
    isassignment(ex)

Check if the input `ex` is an assignment to an existing tensor.
"""
isassignment(ex) = false
isassignment(ex::Expr) = (ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=))

"""
    isdefinition(ex)

Check if the input `ex` is an expression that create a new tensor by definition.
"""
isdefinition(ex) = false
isdefinition(ex::Expr) = (ex.head == :(:=) || ex.head == :(≔))
