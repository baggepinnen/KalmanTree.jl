
abstract type AbstractNode end

@with_kw mutable struct RootNode <: AbstractNode
    left   = LeafNode()
    right  = LeafNode()
    dim    = 0
    split  = 0.0
    domain = [(-1.,1.)]
end
function RootNode(domain)
    g = RootNode(domain = domain)
    g.left.parent = g
    g.right.parent = g
    g
end

@with_kw mutable struct GridNode <: AbstractNode
    parent = nothing
    left   = nothing
    right  = nothing
    dim    = 0
    split  = 0.0
    domain = [(-1.,1.)]
end
function GridNode(p=nothing)
    g = GridNode(parent=p,left=LeafNode(),right=LeafNode(),dim=0,split=0.0)
    g.left.parent  = g
    g.right.parent = g
    g
end

@with_kw mutable struct LeafNode <: AbstractNode
    parent = nothing
    model = nothing
    domain = [(-1.,1.)]
    visited = false
end

visited(l::LeafNode) = l.visited
visited(n) = any(visited, Leaves(n))

isleaf(g::LeafNode) = true
isleaf(g::GridNode) = false
isleaf(g::RootNode) = false
# isroot(g::LeafNode) = false
# isroot(g::GridNode) = false
# isroot(g::RootNode) = true

@inline function xu_val(g,x,u,dim=g.dim)
    if dim > length(u)
        return x[dim - length(u)]
    else
        return u[dim]
    end
end

@inline function active_node(g::AbstractNode, xu)
    xu[g.dim] > g.split ? g.right : g.left
end
@inline function active_node(g::AbstractNode, x, u)
    xu_val(g,x,u) > g.split ? g.right : g.left
end
@inline function active_node_x(g::AbstractNode, x)
    x[g.dim-length(g.domain)+length(x)] > g.split ? g.right : g.left
end

walk_down(g::LeafNode, args...) = g
walk_down_x(g::LeafNode, args...) = g

function walk_down(g, args...)
    walk_down(active_node(g, args...), args...)
end

function walk_down_x(g, args...)
    walk_down(active_node_x(g, args...), args...)
end

walk_up(g::RootNode, d) = g,0
function walk_up(g, d)
    g.parent === nothing && (return g, d) # Reached the root
    walk_up(g.parent, d+1)
end

function depth(node)
    root, d = walk_up(g, 0)
    d
end

depthfirst(f,g::LeafNode) = (f(g);g)
function depthfirst(f,g)
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

function Grid(domain::AbstractVector{<:Tuple}, model, splitter; initial_split=allowed_dims(splitter))
    initial_split == 0 && (return LeafNode(model=model, domain=domain))
    nsplits = length(initial_split)
    seed = split(LeafNode(model=model, domain=domain), initial_split[1])
    nsplits == 1 && (return seed)
    for d = initial_split[2:end]
        depthfirst(seed) do c
            split(c, d)
        end
    end
    seed
end

function predict(g::AbstractNode, args...)
    active = walk_down(g,args...)
    predict(active.model, args...)
end

function update!(g::AbstractNode, x, u, y)
    active = walk_down(g,x,u)
    active.visited = true
    update!(active.model, x, u, y)
end

function update!(g::AbstractNode, x, y)
    active = walk_down(g,x)
    active.visited = true
    update!(active.model, x, y)
end



AbstractTrees.children(g::AbstractNode) = (g.left, g.right)
AbstractTrees.children(g::LeafNode) = ()
AbstractTrees.printnode(io::IO,g::AbstractNode) = print(io,"Node, split: dim: $(g.dim), split: $(g.split)")
AbstractTrees.printnode(io::IO,g::LeafNode) = print(io,"Leaf, domain: $(g.domain)")
