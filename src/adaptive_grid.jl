include("tree_tools.jl")

abstract type AbstractModel end

struct LinearModel <: AbstractModel
    x
end
LinearModel() = LinearModel(0)

struct QuadraticModel <: AbstractModel
end



function Grid(ndims)
    g = LeafNode()
    for d = 1:ndims
        g = breadthfirst(g) do g
            split(g, d, 0.)
        end
    end
    walk_up(g)
end

active_node(g::GridNode, x) = x > g.split ? g.right : g.left

function find_split(g::GridNode)
end

function split(g, dim, split)
    @assert isleaf(g) "Can only split leaf nodes"
    node         = GridNode()
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
    parent.dim   = dim
    parent.split = split
    g
end

function update!(g::GridNode, x, y)
    active = active_node(g,x)
    update!(active.model, x, y)
end

function update!(m::AbstractModel, x, y)
    m.x += 1
end

##
grid = Grid(1)
for i = 1:100
    x = randn()
    y = 2x + 0.1randn()
    update!(grid,x,y)
