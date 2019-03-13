using Parameters, AbstractTrees
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
end

isleaf(g) = g.dim == 0
isleaf(g::LeafNode) = true
isleaf(g::GridNode) = false
isleaf(g::RootNode) = false
isroot(g::LeafNode) = false
isroot(g::GridNode) = false
isroot(g::RootNode) = true

function walk_down(g, x)
    isleaf(g) && (return g) # Reached the end
    walk_down(active_node(g, x), x)
end

function walk_down(g, x, u)
    isleaf(g) && (return g) # Reached the end
    walk_down(active_node(g, x, u), x, u)
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

function Grid(domain::AbstractVector{<:Tuple}, model)
    ndims = length(domain)
    g = split(LeafNode(model=model, domain=domain), 1)
    for d = 2:ndims
        depthfirst(g) do c
            split(c, d)
        end
    end
    g
end

function xu_val(g,x,u,dim=g.dim)
    if dim > length(u)
        return x[dim - length(u)]
    else
        return u[dim]
    end
end

function active_node(g::AbstractNode, x)
    x[g.dim] > g.split ? g.right : g.left
end
function active_node(g::AbstractNode, x, u)
    xu_val(g,x,u) > g.split ? g.right : g.left
end

function predict(g::AbstractNode, args...)
    active = walk_down(g,args...)
    predict(active.model, args...)
end

function update!(g::AbstractNode, x, u, y)
    active = walk_down(g,x,u)
    update!(active.model, x, u, y)
end

function update!(g::AbstractNode, x, y)
    active = walk_down(g,x)
    update!(active.model, x, y)
end



AbstractTrees.children(g::AbstractNode) = (g.left, g.right)
AbstractTrees.children(g::LeafNode) = ()
AbstractTrees.printnode(io::IO,g::AbstractNode) = print(io,"Node, split: dim: $(g.dim), split: $(g.split)")
AbstractTrees.printnode(io::IO,g::LeafNode) = print(io,"Leaf, domain: $(g.domain)")


@recipe function plot_tree(g::AbstractNode, indicate = :cov)
    rect(d) = Shape([d[1][1],d[1][1],d[1][2],d[1][2],d[1][1]],  [d[2][1],d[2][2],d[2][2],d[2][1],d[2][1]])
    covs = [tr(l.model.updater.P) for l in Leaves(g)]
    mc = maximum(covs)
    colorbar := true
    label := ""
    legend := false
    cg = cgrad(:inferno)
    for l in Leaves(g)
        if indicate == :cov
            c = tr(l.model.updater.P)/mc
        else
            c = (predict(l, centroid(l))+2)/4
        end
        @series begin
            color := cg[c]
            rect(l.domain)
        end
    end
    nothing
end
