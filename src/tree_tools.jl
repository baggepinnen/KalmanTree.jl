using Parameters, AbstractTrees, RecipesBase, LinearAlgebra
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

function Grid(domain::AbstractVector{<:Tuple}, model, splitter; initial_split=splitter.allowed_dims)
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


@recipe function plot_tree(g::AbstractNode, indicate = :cov; dims=1:2)
    rect(d) = Shape([d[dims[1]][1],d[dims[1]][1],d[dims[1]][2],d[dims[1]][2],d[dims[1]][1]],  [d[dims[2]][1],d[dims[2]][2],d[dims[2]][2],d[dims[2]][1],d[dims[2]][1]])
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

@userplot Gridmat
@recipe function gridmat(gm::Gridmat)
    g = gm.args[1]
    d = g.domain
    dims = length(domain)
    colorbar := false
    label := ""
    legend := false
    cg = cgrad(:inferno)
    layout := (dims,dims)
    seriestype := :surface
    indmat = LinearIndices((dims,dims))'
    for d1 = 1:dims
        for d2 = 1:d1
            subplot := indmat[d1,d2]
            for l in Leaves(g)
                c = centroid(l)
                @series begin
                    x = LinRange(d[d1]...,20)
                    y = LinRange(d[d2]...,20)
                    pf = (x,y) -> begin
                    c[d1] = x
                    c[d2] = y
                    predict(g, c)
                    end
                    x,y,pf
                end
            end
        end
    end
    nothing
end
