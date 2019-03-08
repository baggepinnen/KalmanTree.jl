
using Test

root = LeafNode()
@test root.parent === nothing
root = split(root, 1, 0.0)
@test root.dim == 1
@test root.left isa LeafNode
@test root.right isa LeafNode
@test root.left.parent === root
@test root.right.parent === root

split(root.left, 1, -1.0)
split(root.right, 1, 1.0)
@test root.left.dim == 1
@test root.right.dim == 1

@test root.left.left isa LeafNode
@test walk_up(root.left.left) === root
@test walk_down(root, -2) === root.left.left
@test walk_down(root, 2) === root.right.right
@test walk_down(root, 0.5) === root.right.left
@test walk_down(root, -0.5) === root.left.right


@test countnodes(root) == 4

counter = 0
depthfirst(root) do node
    global counter += 1
end
@test counter == 4


grid = Grid(3, nothing)
@test countnodes(grid) == 2^3
grid = Grid(4, nothing)
@test countnodes(grid) == 2^4

@test grid.dim == 1
@test grid.left.dim == 2
@test grid.left.left.dim == 3
@test grid.left.left.right.dim == 4
