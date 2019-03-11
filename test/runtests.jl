
using Test

root = LeafNode()
@test root.parent === nothing
root = split(root, 1, 0.0)
@test root.dim == 1
@test root.left isa LeafNode
@test root.right isa LeafNode
@test root.left.parent === root
@test root.right.parent === root
@test root.left.domain[1] == (-1,0)
@test root.right.domain[1] == (0,1)

split(root.left, 1, -0.5)
split(root.right, 1)
@test root.left.dim == 1
@test root.right.dim == 1
@test root.left.left.domain[1] == (-1,-0.5)
@test root.left.right.domain[1] == (-0.5,0)
@test root.right.domain[1] == (0,1)



@test root.left.left isa LeafNode
@test walk_up(root.left.left,0) === (root,2)
@test walk_down(root, -2) === root.left.left
@test walk_down(root, 2) === root.right.right
@test walk_down(root, 0.4) === root.right.left
@test walk_down(root, -0.4) === root.left.right


@test countnodes(root) == 4

counter = 0
depthfirst(root) do node
    global counter += 1
end
@test counter == 4

domain = [(-1,1),(-1,1),(-1,1)]
grid = Grid(domain, nothing)
@test countnodes(grid) == 2^3
@test grid.domain == domain
@test grid.left.domain == [(-1,0),(-1,1),(-1,1)]
@test grid.right.domain == [(0,1),(-1,1),(-1,1)]
@test grid.left.left.domain == [(-1,0),(-1,0),(-1,1)]

domain = [(-1,1),(-1,1),(-1,1),(-2,2)]
grid = Grid(domain, nothing)
@test countnodes(grid) == 2^4

@test grid.dim == 1
@test grid.left.dim == 2
@test grid.left.left.dim == 3
@test grid.left.left.right.dim == 4
