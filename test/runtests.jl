
using Test, LinearAlgebra, Random
using KalmanTree
using KalmanTree: depthfirst, breadthfirst

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
splitter = RandomSplitter(1:3)
g = Grid(domain, nothing, splitter)
@test countnodes(g) == 2^3
@test g.domain == domain
@test g.left.domain == [(-1,0),(-1,1),(-1,1)]
@test g.right.domain == [(0,1),(-1,1),(-1,1)]
@test g.left.left.domain == [(-1,0),(-1,0),(-1,1)]

domain = [(-1,1),(-1,1),(-1,1),(-2,2)]
splitter = RandomSplitter(1:4)
g = Grid(domain, nothing, splitter)
@test countnodes(g) == 2^4

@test g.dim == 1
@test g.left.dim == 2
@test g.left.left.dim == 3
@test g.left.left.right.dim == 4

# Models and updaters

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
Quu,Qux, qu = KalmanTree.Qmats(m,x[1])

@test isposdef(-Quu)
@test sum(abs, -Quu \ Q[1:3,1:3] - I) < 5
@test sum(abs, -2qu - q[1:3]) < 1e-5
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




# Integration tests

#= Notes:
Should splitting along action dimensions be allowed? Then how does one find argmaxₐ(Q)? since this max can be outside the domain of the model.
If splts along action dimensions are kept at the bottom of the tree, one could operate on a subtree when finding argmaxₐ(Q). Each argmax is a box-constrained QP where the domain in a-dimensions determine the bounds. With this strategy, argmaxₐ must be carried out as many times as there are leaves in the subtree corresponding to the s-coordinate.
First, the unconstraind argmax can be calculated for each cell. If the highest is inside its bounds, no box-constrained QP has to be solved
=#

##
nx,nu = 2,2
domain = [(-1.,1.),(-1.,1.),(-1.,1.),(-1.,1.)]#,(-1.,1.)]
model = QuadraticModel(nx+nu; actiondims=1:2)
splitter = RandomSplitter(3:4)
g = Grid(domain, model, splitter)
# splitter = TraceSplitter(1:2)
# splitter = NormalizedTraceSplitter(1:2)
# splitter = QuadformSplitter(1:2)
X,U,Y = [],[],[]
f(x,u) = sin(3sum(x)) + sum(-(u-x).^2)
for i = 1:10000
    if i % 100 == 0
        splitter(g)
        @show countnodes(g)
    end
    @show i
    x = 2 .*rand(nx) .-1
    u = 2 .*rand(nu) .-1
    y = f(x,u) + 0.1*(sum(x)+sum(u))*randn()
    push!(X,x)
    push!(U,u)
    push!(Y,y)
    yh = predict(g, x, u)
    @show y-yh
    update!(g,x,u,y)
    yh = predict(g, x, u)
    @show y-yh
end
plot(g, :value)

##
# plotly()
# gr()
po = (zlims=(-1.5,1.5), clims=(-2,2))
xu = LinRange(-1,1,30),LinRange(-1,1,30)
surface(xu..., f; title="True fun", layout=4, po...)
surface!(xu..., (x,u)->predict(g,x,u); title="Approximation", subplot=2, po...)
surface!(xu..., (x,u)->predict(g,x,u)-f(x,u); title="Error", subplot=3, po...)
plot!(g, :value, title="Grid cells", subplot=4)

##

# E = @benchmark begin
x,u = (X[rand(1:length(X))],U[rand(1:length(X))])
n = walk_down(g,x,u)
um = argmax_u(n, x)
# argmax_u(n, x)
# end
# display(E)
plot(u->predict(n.model, x, u), -2,2, title="Q(a)", legend=false)
vline!([um])
vline!([n.domain[n.model.actiondims][]...], l=(:dash,:black))
# pf = (u1,u2)->predict(n.model, x, [u1;u2])
# surface(xu...,pf, title="Q(a)", legend=false)
# scatter3d!(um[1:1],um[2:2], [pf(um...)], m=(10,:cyan))
