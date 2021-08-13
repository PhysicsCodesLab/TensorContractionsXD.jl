# Tools to analyze parts of tensor expressions and extract information
"""
    decomposetensor(ex::Expr)

Decompose a tensor expression into the tensor object, left indices, and right indices, where
the left indices correspond to the codomain and right indices correspond to the domain if we
treat the tensor as a tensor map.
"""
function decomposetensor(ex::Expr)
    istensor(ex) || throw(ArgumentError("not a valid tensor: $ex"))
    if ex.head == :ref || ex.head == :typed_hcat # ref: A[a,b,c] or A[a,b;c,d] or typed_hcat: A[a b c], A can be replaced by any `Expr`, e.g. A[1:3]
        if length(ex.args) == 1 # A[]
            return ex.args[1], Any[], Any[]
        elseif isa(ex.args[2], Expr) && ex.args[2].head == :parameters # A[a,b;c,d]
            return ex.args[1], ex.args[3:end], ex.args[2].args
        else # A[a,b,c] or A[a b c]
            return ex.args[1], ex.args[2:end], Any[]
        end
    else #if ex.head == :typed_vcat A[a b;c d] or A[(a,b);(c,d)]
        if isa(ex.args[2], Expr) && (ex.args[2].head == :row || ex.args[2].head == :tuple) # row: A[a b; c]; or tuple: A[(a,b);c]
            leftind = ex.args[2].args
        else
            leftind = ex.args[2:2] # A[a; b c]
        end
        if isa(ex.args[3], Expr) && (ex.args[3].head == :row || ex.args[3].head == :tuple) # row: A[a; b c]; or tuple: A[a;(b,c)]
            rightind = ex.args[3].args
        else
            rightind = ex.args[3:3] # A[a b; c]
        end
        return ex.args[1], leftind, rightind
    end
end

"""
    gettensorobject(ex)

Get the name or object of the tensor.
"""
gettensorobject(ex) = decomposetensor(ex)[1]

"""
    getleftindices(ex)

Get the left indices of a tensor that correspond to the codomain.
"""
getleftindices(ex) = decomposetensor(ex)[2]

"""
    getrightindices(ex)

Get the right indices of a tensor that correspond to the domain.
"""
getrightindices(ex) = decomposetensor(ex)[3]


"""
    decomposegeneraltensor(ex)

Decompose a general tensor and return `(object, leftind, rightind, α, conj)`, where `α` is
the possible scalar that multiplied with the tensor, and `conj` is a symbol that denote
whether we need to apply complex conjugation on that tensor.

??? Why there is no cases with `:adjoint` and `:transpose`.
"""
function decomposegeneraltensor(ex)
    if istensor(ex)
        object, leftind, rightind = decomposetensor(ex)
        return (object, leftind, rightind, true, false)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :+ && length(ex.args) == 2 # unary plus: pass on
        return decomposegeneraltensor(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :- && length(ex.args) == 2 # unary minus: flip scalar factor
        (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[2])
        return (object, leftind, rightind, Expr(:call, :-, α), conj)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2 # conjugation: flip conjugation flag and conjugate scalar factor
        (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[2])
        return (object, leftind, rightind, Expr(:call, :conj, α), !conj)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # scalar multiplication: multiply scalar factors
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[3])
            return (object, leftind, rightind, Expr(:call, :*, ex.args[2], α), conj)
        elseif isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[2])
            return (object, leftind, rightind, Expr(:call, :*, α, ex.args[3]), conj)
        end
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
        if isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[2])
            return (object, leftind, rightind, Expr(:call, :/, α, ex.args[3]), conj)
        end
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            (object, leftind, rightind, α, conj) = decomposegeneraltensor(ex.args[3])
            return (object, leftind, rightind, Expr(:call, :\, ex.args[2], α), conj)
        end
    end
    throw(ArgumentError("not a valid generalized tensor expression $ex"))
end

"""
    getlhs(ex::Expr)

Get the left hand side of an assignment or definition expression.
"""
function getlhs(ex::Expr)
    if ex.head in (:(=), :(+=), :(-=), :(:=), :(≔))
        return ex.args[1]
    else
        throw(ArgumentError("invalid assignment or definition $ex"))
    end
end

"""
    getrhs(ex::Expr)

Get the right hand side of an assignment or definition expression.
"""
function getrhs(ex::Expr)
    if ex.head in (:(=), :(+=), :(-=), :(:=), :(≔))
        return ex.args[2]
    else
        throw(ArgumentError("invalid assignment or definition $ex"))
    end
end

"""
    gettensors(ex)

Return a list of all the tensors in a tensor expression (not a definition or assignment).
"""
function gettensors(ex)
    if istensor(ex)
        Any[ex]
    elseif istensorexpr(ex)
        list = Any[]
        for e in ex.args
            append!(list, gettensors(e))
        end
        return list
    else
        return Any[]
    end
end

"""
    gettensorobjects(ex)

Return a list of all the tensor objects in a tensor expression (not a definition or
assignment).
"""
gettensorobjects(ex) = gettensorobject.(gettensors(ex))

"""
    getinputtensorobjects(ex)

Return a list of all the existing tensor objects which are inputs (i.e. appear in the rhs
of assignments and definitions).
"""
function getinputtensorobjects(ex)
    list = Any[]
    if istensorexpr(ex)
        append!(list, gettensorobjects(ex))
    elseif isdefinition(ex)
        append!(list, gettensorobjects(getrhs(ex)))
    elseif isassignment(ex)
        if ex.head == :(+=) || ex.head == :(-=)
            lhs = getlhs(ex)
            if istensor(lhs)
                push!(list, gettensorobject(lhs))
            end
        end
        append!(list, gettensorobjects(getrhs(ex)))
    elseif isa(ex, Expr) && ex.head == :block
        for i = 1:length(ex.args)
            list2 = getinputtensorobjects(ex.args[i])
            for j = 1:i-1
                # if objects have previously been defined or assigned to, they are not inputs
                list2 = setdiff(list2, getnewtensorobjects(ex.args[j]))
                list2 = setdiff(list2, getoutputtensorobjects(ex.args[j]))
            end
            append!(list, list2)
        end
    elseif isa(ex, Expr) && ex.head in (:for, :while)
        append!(list, getinputtensorobjects(ex.args[2]))
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :scalar
        append!(list, gettensorobjects(ex.args[2]))
    end
    return unique!(list)
end

"""
    getoutputtensorobjects(ex)

Return a list of all the existing tensor objects which are outputs (i.e. appear in the lhs
of assignments).
"""
function getoutputtensorobjects(ex)
    list = Any[]
    if isassignment(ex)
        lhs = getlhs(ex)
        if istensor(lhs)
            push!(list, gettensorobject(lhs))
        end
    elseif isa(ex, Expr) && ex.head == :block
        for i = 1:length(ex.args)
            list2 = getoutputtensorobjects(ex.args[i])
            for j = 1:i-1
                # if objects have previously been defined, they are not existing outputs
                list2 = setdiff(list2, getnewtensorobjects(ex.args[j]))
            end
            append!(list, list2)
        end
    elseif isa(ex, Expr) && ex.head in (:for, :while)
        append!(list, getoutputtensorobjects(ex.args[2]))
    end
    return unique!(list)
end

"""
    getnewtensorobjects(ex)

Return a list of all the existing tensor objects which are newly created (i.e. appear in
the lhs of definition).
"""
function getnewtensorobjects(ex)
    list = Any[]
    if isdefinition(ex)
        lhs = getlhs(ex)
        if istensor(lhs)
            push!(list, gettensorobject(lhs))
        end
    elseif isa(ex, Expr) && ex.head == :block
        for e in ex.args
            append!(list, getnewtensorobjects(e))
        end
    elseif isa(ex, Expr) && ex.head in (:for, :while)
        append!(list, getnewtensorobjects(ex.args[2]))
    end
    return unique!(list)
end

"""
    getindices(ex::Expr)

For any tensor expression, get the list of uncontracted indices that would remain after
evaluating that expression.
"""
function getindices(ex::Expr)
    if istensor(ex)
        _,leftind,rightind = decomposetensor(ex)
        return unique2(vcat(leftind, rightind))
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        return getindices(ex.args[2]) # getindices on any of the args[2:end] should yield the same result
    elseif ex.head == :call && ex.args[1] == :*
        indices = getindices(ex.args[2])
        for k = 3:length(ex.args)
            append!(indices, getindices(ex.args[k]))
        end
        return unique2(indices)
    elseif ex.head == :call && ex.args[1] == :/
        return getindices(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :\
        return getindices(ex.args[3])
    elseif ex.head == :call && length(ex.args) == 2
        return getindices(ex.args[2])
    else
        return Any[]
    end
end
getindices(ex) = Any[]

"""
    getallindices(ex::Expr)

Return a list of all indices appearing in a tensor expression and only once for each.
"""
function getallindices(ex::Expr)
    if istensor(ex)
        _,leftind,rightind = decomposetensor(ex)
        return unique!(vcat(leftind, rightind))
    elseif isassignment(ex) || isdefinition(ex)
        return getallindices(getrhs(ex))
    else
        return unique!(mapreduce(getallindices, vcat, ex.args))
    end
end
getallindices(ex) = Any[]
