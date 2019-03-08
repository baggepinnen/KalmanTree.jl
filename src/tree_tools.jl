using Parameters
abstract type AbstractNode end

@with_kw mutable struct GridNode <: AbstractNode
    parent = nothing
    left = nothing
    right = nothing
    dim = 0
    split = 0.0
end
function GridNode(p=nothing)
    g = GridNode(parent=p,left=LeafNode(),right=LeafNode(),dim=0,split=0.0)
    g.left.parent = g
    g.right.parent = g
    g
end

@with_kw mutable struct LeafNode <: AbstractNode
    parent = nothing
    model = nothing
end

isleaf(g) = g.dim == 0
isleaf(g::LeafNode) = true
isleaf(g::GridNode) = false

function walk_down(g, x)
    isleaf(g) && (return g) # Reached the end
    walk_down(active_node(g, x), x)
end

function walk_down(g, x, u)
    isleaf(g) && (return g) # Reached the end
    walk_down(active_node(g, x, u), x, u)
end

function walk_up(g)
    g.parent === nothing && (return g) # Reached the root
    walk_up(g.parent)
end

function depthfirst(f,g)
    if isleaf(g)
        f(g)
        return g
    end
    depthfirst(f, g.left)
    depthfirst(f, g.right)
end

function breadthfirst(f,g)
    q = AbstractNode[g.left, g.right]
    while !isempty(q)
        g = popfirst!(q)
        if isleaf(g)
            f(g)
            continue
        end
        push!(q, g.left)
        push!(q, g.right)
    end
    g
end

breadthfirst(f,g::LeafNode) = (f(g);g)

function countnodes(root)
    counter = 0
    breadthfirst(root) do node
        counter += 1
    end
    counter
end

function Grid(ndims, model)
    g = split(LeafNode(model=model), 1, 0.)
    for d = 2:ndims
        depthfirst(g) do c
            split(c, d, 0.)
        end
    end
    g
end

function xu_val(g,x,u)
    dim = g.dim
    if dim > length(x)
        return u[dim - length(x)]
    else
        return x[dim]
    end
end

function active_node(g::GridNode, x)
    x[g.dim] > g.split ? g.right : g.left
end
function active_node(g::GridNode, x, u)
    xu_val(g,x,u) > g.split ? g.right : g.left
end

function find_split(g::GridNode)
end

function Base.split(g::AbstractNode, dim, split)
    @assert isleaf(g) "Can only split leaf nodes"
    model = g.model
    node  = GridNode(parent = g.parent)
    if g.parent === nothing
        parent = node
    else
        parent = g.parent
        node.parent  = parent
        if parent.left === g
            parent.left = node
        else
            parent.right = node
        end
    end
    node.dim   = dim
    node.split = split
    node.left = LeafNode(node, g.model)
    node.right = LeafNode(node, deepcopy(g.model))
    node
end

function update!(g::GridNode, x, u, y)
    active = walk_down(g,x,u)
    update!(active.model, x, u, y)
end

function predict(g::GridNode, x, u)
    active = walk_down(g,x,u)
    predict(active.model, x, u)
end
