
using Test, LinearAlgebra, Random

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

# Models and updaters
include("/local/home/fredrikb/.julia/dev/KalmanTree/src/tree_tools.jl")
include("/local/home/fredrikb/.julia/dev/KalmanTree/src/domain_tools.jl")
include("/local/home/fredrikb/.julia/dev/KalmanTree/src/models.jl")
using Test, LinearAlgebra, Random
N = 200
x = [randn(2) for i = 1:N]
u = [randn(3) for i = 1:N]
Q = randn(5,5)
Q = (Q + Q')/2 + 5I
@test isposdef(Q)
q = randn(5)
c = randn()
fq(x,u) = fq([u;x])
fq(x) = x'Q*x + q'x + c
y = map(fq,x,u)

m = QuadraticModel(5, actiondims=1:3)
P0 = det(cov(m))
update!(m,x[1],u[1],y[1])
foreach(x,u,y) do x,u,y
    update!(m,x,u,y)
end

@test all(zip(x,u,y)) do (x,u,y)
    abs(y - predict(m,x,u)) < 0.0001
end

@test det(cov(m)) < P0
@test cond(cov(m)) < 100
Quu,Qux, qu = Qmats(m,x[1])

@test isposdef(-Quu)
@test sum(abs, -Quu \ Q[1:3,1:3] - I) < 5
@test sum(abs, -qu - q[1:3]) < 1e-5
@test sum(abs, -Qux - Q[1:3,4:5]) < 1e-5
@test abs(m.w[end] - c) < 1e-5




updater = NewtonUpdater(0.5, 0.999)
m = QuadraticModel(5, actiondims=1:3, updater = updater)
for i = 1:5
    foreach(x,u,y) do x,u,y
        update!(m,x,u,y)
    end
end

@test mean(zip(x,u,y)) do (x,u,y)
    abs2(y - predict(m,x,u))
end  < 1e-6

updater = GradientUpdater(0.01, 0.999)
m = QuadraticModel(5, actiondims=1:3, updater = updater)
for i = 1:50
    foreach(x,u,y) do x,u,y
        update!(m,x,u,y)
    end
end

@test mean(zip(x,u,y)) do (x,u,y)
    abs2(y - predict(m,x,u))
end  < 1e-6
