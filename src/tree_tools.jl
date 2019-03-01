abstract type AbstractNode end

mutable struct GridNode <: AbstractNode
    parent
    left
    right
    dim
    split
    function GridNode(p::GridNode=nothing)
        g = new(p,LeafNode(),LeafNode(),0,0.0)
        g.left.parent = g
        g.right.parent = g
        g
    end
end

mutable struct LeafNode <: AbstractNode
    parent
    model
    function LeafNode(p=nothing)
        g = new(p, nothing)
    end
end

isleaf(g) = g.dim == 0
isleaf(g::LeafNode) = true
isleaf(g::GridNode) = false

function walk_down(g, x)
    isleaf(g) && (return g) # Reached the end
    walk_down(active_node(g, x), x)
end

function walk_up(g)
    g.parent === g && (return g) # Reached the root
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
    q = [g.left, g.right]
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
