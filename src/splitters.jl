
abstract type AbstractSplitter end
struct TraceSplitter <: AbstractSplitter
    allowed_dims
end
struct CountSplitter <: AbstractSplitter
    allowed_dims
end
struct QuadformSplitter <: AbstractSplitter
    allowed_dims
end
struct RandomSplitter <: AbstractSplitter
    allowed_dims
end

struct InnovationSplitter <: AbstractSplitter
    allowed_dims
end

function find_and_apply_split(g, splitter)
    g = find_split(g, splitter)
    split(g, splitter)
end


score(node, splitter::TraceSplitter) = tr(parameter_cov(node))*volume(node)

function score(node, splitter::QuadformSplitter)
    c = centroid(node)
    x = feature!(node.model, c)
    x'*(node.model.updater.P\x) # TODO: not sure about inverse
end
score(node, splitter::InnovationSplitter) = innovation_var(node.model)*volume(node)

function find_split(g::AbstractNode, splitter::AbstractSplitter)
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


function find_split(g::AbstractNode, splitter::RandomSplitter)
    c = countnodes(g)
    n = rand(1:c)
    for (i,l) in enumerate(Leaves(g))
        if i == n
            return l
        end
    end
end

@inline allowed_dim(splitter,i) = i ∈ splitter.allowed_dims

"split(node::AbstractNode, splitter::AbstractSplitter)
Split node with highest score."
function Base.split(node::AbstractNode, splitter::AbstractSplitter)
    scores = Vector{Float64}(undef,length(node.domain))
    for i in eachindex(scores)
        scores[i] = allowed_dim(splitter,i) ? volume(node.domain[i]) : -Inf
    end
    dim = findmax(scores)[2]
    # dim = findmax(collect(d[2]-d[1] for d in node.domain))[2]
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
