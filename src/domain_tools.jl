
volume(g::AbstractNode) = volume(g.domain)
volume(d::Tuple) = d[2]-d[1]

function volume(d::AbstractVector)
    prod(volume(d) for d in d)
end

centroid(g::AbstractNode) = centroid(g.domain)
centroid(d::AbstractVector) = centroid.(d)
centroid(d::Tuple) = (d[2]+d[1])/2

abstract type AbstractSplitter end
struct TraceSplitter <: AbstractSplitter
    allowed_dims
end
struct NormalizedTraceSplitter <: AbstractSplitter
    allowed_dims
end
struct CountSplitter <: AbstractSplitter
    allowed_dims
end
struct QuadformSplitter <: AbstractSplitter
    allowed_dims
end

function (splitter::TraceSplitter)(g)
    g = find_split(g, splitter)
    split(g, splitter)
end
function (splitter::NormalizedTraceSplitter)(g)
    g = find_split(g, splitter)
    split(g, splitter)
end

function (splitter::QuadformSplitter)(g)
    g = find_split(g, splitter)
    split(g, splitter)
end

score(node, splitter::TraceSplitter) = node.model.updater.P |> tr
score(node, splitter::NormalizedTraceSplitter) = (node.model.updater.P |> tr)* volume(node)
function score(node, splitter::QuadformSplitter)
    c = centroid(node)
    x = feature!(node.model, c)
    x'*(node.model.updater.P\x) # TODO: not sure about inverse
end

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

"split(node::AbstractNode, splitter::AbstractSplitter)
Split node with highest score."
function Base.split(node::AbstractNode, splitter::AbstractSplitter)
    # dim = findmax(collect(d[2]-d[1] for d in node.domain[splitter.allowed_dims]))[2]
    dim = findmax(collect(d[2]-d[1] for d in node.domain))[2]
    split(node, dim)
    node
end

# "split(node::AbstractNode, splitter::AllBelowSplitter)
# Split node with highest score."
# function Base.split(node::AbstractNode, splitter::AllBelowSplitter)
#     dim = findmax(volume.(node.domain))[2]
#     split(node, dim)
#     node
# end


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