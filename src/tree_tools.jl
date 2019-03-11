using Parameters, AbstractTrees
abstract type AbstractNode end

@with_kw mutable struct RootNode <: AbstractNode
    left = LeafNode()
    right = LeafNode()
    dim = 0
    split = 0.0
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
    left = nothing
    right = nothing
    dim = 0
    split = 0.0
    domain = [(-1.,1.)]
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

volume(g::AbstractNode) = volume(g.domain)
function volume(d::AbstractVector)
    width(d) = d[2]-d[1]
    prod(width(d) for d in d)
end

abstract type AbstractSplitter end

function find_split(g::GridNode, splitter::AbstractSplitter)
    maxscore = -Inf
    maxleaf = g
    breadthfirst(g) do g
        s = score(g, splitter)
        if s > maxscore
            maxscore = s
            maxleaf = g
        end
    end
    maxleaf
end

struct TraceSplitter <: AbstractSplitter end
struct NormalizedTraceSplitter <: AbstractSplitter end

score(node, splitter::TraceSplitter) = node.model.updater.P |> tr

score(node, splitter::NormalizedTraceSplitter) = (node.model.updater.P |> tr)* volume(node)

function Base.split(node::AbstractNode, splitter::AbstractSplitter)
    dim = findmax(collect(d[2]-d[1] for d in node.domain))[2]
    split(node, dim)
    node
end

function (splitter::TraceSplitter)(g)
    g = find_split(g, splitter)
    split(g, splitter)
end
function (splitter::NormalizedTraceSplitter)(g)
    g = find_split(g, splitter)
    split(g, splitter)
end
function Base.split(g::AbstractNode, dim::Integer, split = :half)
    @assert isleaf(g) "Can only split leaf nodes"
    g.domain
    if split == :half
        split = (g.domain[dim][1]+g.domain[dim][2])/2
    end
    model = g.model
    node  = GridNode(parent = g.parent, domain=g.domain, dim=dim, split=split)
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
    ldomain = copy(node.domain)
    rdomain = copy(node.domain)
    ldomain[dim] = (ldomain[dim][1], split)
    rdomain[dim] = (split, rdomain[dim][2])
    node.left = LeafNode(node, g.model, ldomain)
    node.right = LeafNode(node, deepcopy(g.model), rdomain)
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


AbstractTrees.children(g::AbstractNode) = (g.left, g.right)
AbstractTrees.children(g::LeafNode) = ()
AbstractTrees.printnode(io::IO,g::AbstractNode) = print(io,"Node, split: dim: $(g.dim), split: $(g.split)")
AbstractTrees.printnode(io::IO,g::LeafNode) = print(io,"Leaf, domain: $(g.domain)")



function plot_tree(g)
    p = plot()
    # rect(d) = plot!([d[1][1],d[1][1],d[1][2],d[1][2],d[1][1]], [d[2][1],d[2][2],d[2][2],d[2][1],d[2][1]], l=(:black,))#, fill=:orage)
    rect(d) = Shape([d[1][1],d[1][1],d[1][2],d[1][2],d[1][1]],  [d[2][1],d[2][2],d[2][2],d[2][1],d[2][1]])
    covs = [tr(l.model.updater.P) for l in Leaves(g)]
    mc = maximum(covs)

    cg = cgrad(:inferno)
    for l in Leaves(g)
        c = tr(l.model.updater.P)
        plot!(rect(l.domain), color=cg[c/mc])
    end
    p
end
