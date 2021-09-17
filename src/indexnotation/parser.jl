"""
    mutable struct TensorParser

Define the `TensorParser` object which include the fields:
- `preprocessors::Vector{Any}`: a list of all preprocessing steps
- `contractiontreebuilder::Any`: determine a contraction tree for a contraction involving
                                    multiple tensors
- `contractiontreesorter::Any`: transforms the contraction expression into an expression of
                                    nested binary contractions using the tree output from
                                    the contractiontreebuilder
- `postprocessors::Vector{Any}`: a list of all postprocessing steps
"""
mutable struct TensorParser
    preprocessors::Vector{Any}
    contractiontreebuilder::Any
    contractiontreesorter::Any
    postprocessors::Vector{Any}
    function TensorParser()
        preprocessors = [normalizeindices, expandconj, nconindexcompletion,
                            extracttensorobjects]
        contractiontreebuilder = defaulttreebuilder
        contractiontreesorter = defaulttreesorter
        postprocessors = [_flatten, removelinenumbernode, addtensoroperations]
        return new(preprocessors, contractiontreebuilder, contractiontreesorter,
                    postprocessors)
    end
end

"""
    (parser::TensorParser)(ex::Expr)

Do the tensor contractions with the steps defined in `TensorParser`.
"""
function (parser::TensorParser)(ex::Expr)
    if ex.head == :function
        return Expr(:function, ex.args[1], parser(ex.args[2]))
    end
    for p in parser.preprocessors
        ex = p(ex)::Expr
    end
    treebuilder = parser.contractiontreebuilder
    treesorter = parser.contractiontreesorter
    ex = processcontractions(ex, treebuilder, treesorter)::Expr
    ex = tensorify(ex)::Expr
    for p in parser.postprocessors
        ex = p(ex)::Expr
    end
    return ex
end

"""
    defaulttreebuilder(network)

The input `network` is a list of the untraced indices of each tensorexpression in a tensor
contraction expression, and the `length(network)` is the number of tensorexpressions to be
contracted. Return `tree` in the form e.g. `Any[Any[Any[Any[1, 2], 3], 4], 5]` if
`length(network) == 5`.
"""
function defaulttreebuilder(network)
    if isnconstyle(network)
        tree = ncontree(network)
    else
        tree = Any[1,2]
        for k = 3:length(network)
            tree = Any[tree, k]
        end
    end
    return tree
end

"""
    defaulttreesorter(args, tree)

Sort the sequencee of the contractions as given by the input `tree` (default in the form
`Any[Any[Any[Any[1, 2], 3], 4], 5]` if `length(network) == 5`) by changing the
positions of the tensor expressions to be contracted. The input `args` is a list of the
tensor expressions to be contracted. The returned expression has the form e.g. (default)
`(((A[a,b]*B[c,d])*C[a,c]))*D[b,f])`. This default tree contracts the tensors in the
expression two-by-two from left to right.
"""
function defaulttreesorter(args, tree)
    if isa(tree, Int)
        return args[tree]
    else
        return Expr(:call, :*,
                    defaulttreesorter(args, tree[1]), defaulttreesorter(args, tree[2]))
    end
end

"""
    processcontractions(ex::Expr, treebuilder, treesorter)

Sort the contractions based on the `treebuilder` and `treesorter` if the number of tensor
expressions to be contracted is larger than two.
"""
function processcontractions(ex::Expr, treebuilder, treesorter)
    if ex.head == :macrocall && ex.args[1] == Symbol("@notensor")
        return ex
    end
    ex = Expr(ex.head, map(e->processcontractions(e, treebuilder, treesorter), ex.args)...)
    if istensorcontraction(ex) && length(ex.args) > 3
        args = ex.args[2:end]
        network = map(getindices, args)
        for a in getallindices(ex)
            count(a in n for n in network) <= 2 ||
                throw(ArgumentError("invalid tensor contraction: $ex"))
        end
        tree = treebuilder(network)
        ex = treesorter(args, tree)
    end
    return ex
end
processcontractions(ex, treebuilder, treesorter) = ex # if `ex` is not an Expr do nothing

"""
    tensorify(ex::Expr)

Functions for parsing and processing tensor expressions. Change the tensor expressions to
the actual functions like `add!`, `trace!` and `contract!` they represent.
"""
function tensorify(ex::Expr)
    if ex.head == :macrocall && ex.args[1] == Symbol("@notensor")
        return ex.args[3]
        # > dump(:(@notensor A))
        # > Expr
        #     head: Symbol macrocall
        #     args: Array{Any}((3,))
        #       1: Symbol @notensor
        #       2: LineNumberNode
        #         line: Int64 1
        #         file: Symbol REPL[29]
        #       3: Symbol A
    end
    # assignment case
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhs(ex), getrhs(ex)
        if isa(rhs, Expr) && rhs.head == :call && rhs.args[1] == :throw
            return rhs
        end
        if istensor(lhs) && istensorexpr(rhs)
            indices = getindices(rhs)
            if hastraceindices(lhs)
                err = "left hand side of an assignment should have unique indices: $lhs"
                return :(throw(IndexError($err)))
            end
            dst, leftind, rightind = decomposetensor(lhs)
            if Set(vcat(leftind,rightind)) != Set(indices)
                err = "non-matching indices between left and right hand side: $ex"
                return :(throw(IndexError($err)))
            end
            if isassignment(ex)
                if ex.head == :(=)
                    return instantiate(dst, false, rhs, true, leftind, rightind)
                elseif ex.head == :(+=)
                    return instantiate(dst, true, rhs, 1, leftind, rightind)
                else # if ex.head == :(-=)
                    return instantiate(dst, true, rhs, -1, leftind, rightind)
                end
            else # if isdefinition(ex)
                return Expr(:(=), dst, instantiate(nothing, false, rhs, true, leftind,
                                                    rightind, false))
            end
        elseif isassignment(ex) && isscalarexpr(lhs)
            if istensorexpr(rhs) && isempty(getindices(rhs))
                return Expr(ex.head, instantiate_scalar(lhs), Expr(:call, :scalar,
                                    instantiate(nothing, false, rhs, true, [], [], true)))
            elseif isscalarexpr(rhs)
                return Expr(ex.head, instantiate_scalar(lhs), instantiate_scalar(rhs))
            end
        else
            return ex # likely an error
        end
    end
    if ex.head == :block
        return Expr(ex.head, map(tensorify, ex.args)...)
    end
    if ex.head == :for
        return Expr(ex.head, ex.args[1], tensorify(ex.args[2]))
    end
    if ex.head == :function
        return Expr(ex.head, ex.args[1], tensorify(ex.args[2]))
    end
    # constructions of the form: a = @tensor ..., where a is a scalar
    if isscalarexpr(ex)
        return instantiate_scalar(ex)
    end
    if istensorexpr(ex)
        if !isempty(getindices(ex))
            err = "cannot evaluate $ex to a scalar: uncontracted indices"
            return :(throw(IndexError($err)))
        end
        return Expr(:call, :scalar, instantiate(nothing, false, ex, true, [], [], true))
    end
    error("invalid syntax in @tensor macro: $ex")
end
tensorify(ex) = ex
