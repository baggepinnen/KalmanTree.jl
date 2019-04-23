
abstract type AbstractSplitter end
abstract type AbstractWrapper <: AbstractSplitter end
Base.@kwdef struct TraceSplitter <: AbstractSplitter
    allowed_dims
end
Base.@kwdef struct CountSplitter <: AbstractSplitter
    allowed_dims
end
Base.@kwdef struct QuadformSplitter <: AbstractSplitter
    allowed_dims
end
Base.@kwdef struct RandomSplitter <: AbstractSplitter
    allowed_dims
end

Base.@kwdef struct InnovationSplitter <: AbstractSplitter
    allowed_dims
end

struct VisitedWrapper <: AbstractWrapper
    inner
end

struct VolumeWrapper <: AbstractWrapper
    inner
end

function find_and_apply_split(g, splitter)
    g = find_split(g, splitter)
    split(g, splitter)
end

score(node, splitter::VisitedWrapper) =
        score(node, splitter.inner)*node.visited

score(node, splitter::VolumeWrapper) =
        score(node, splitter.inner)*volume(node)

score(node, splitter::TraceSplitter) = tr(parameter_cov(node))


function score(node, splitter::QuadformSplitter)
    c = centroid(node)
    x = feature!(node.model, c)
    x'*(cov(node.model)\x) # TODO: not sure about inverse
end
score(node, splitter::InnovationSplitter) = sqrt(abs(innovation_var(node.model)))

function find_split(g::AbstractNode, splitter::AbstractSplitter)
    maxscore = -Inf
    maxleaf = g
    breadthfirst(g) do g
        s = score(g, splitter)
        if s > maxscore # TODO: when nodes are split, several consequtive nodes have the same score and the first one will always be selected. Inflate parameter cov to somewhat avoid this?
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

@inline allowed_dim(splitter,i) = i ∈ allowed_dims(splitter)
@inline allowed_dim(splitter::AbstractWrapper,i) = allowed_dim(splitter.inner, i)

@inline allowed_dims(splitter) = splitter.allowed_dims
@inline allowed_dims(splitter::AbstractWrapper) = allowed_dims(splitter.inner)

"split(node::AbstractNode, splitter::AbstractSplitter)
Split node with highest score."
function Base.split(node::LeafNode, splitter::AbstractSplitter)
    cell_dims = map(eachindex(node.domain)) do i
        allowed_dim(splitter,i) ? volume(node.domain[i]) : -Inf
    end
    widest_dim = findmax(cell_dims)[2]
    split(node, widest_dim)
    node
end

Base.split(::Nothing, args...) = nothing

# "split(node::AbstractNode, splitter::AllBelowSplitter)
# Split node with highest score."
# function Base.split(node::AbstractNode, splitter::AllBelowSplitter)
#     dim = findmax(volume.(node.domain))[2]
#     split(node, dim)
#     node
# end


function Base.split(g::LeafNode, dim::Integer, split = :half)
    g.domain
    if split === :half
        split = centroid(g.domain[dim])
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
    node.left = LeafNode(node, g.model, ldomain, false)
    node.right = LeafNode(node, deepcopy(g.model), rdomain, false)
    node
end
