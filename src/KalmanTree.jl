include("tree_tools.jl")
include("domain_tools.jl")
include("models.jl")
using Plots
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
grid = Grid(domain, model, splitter)
# splitter = TraceSplitter(1:2)
# splitter = NormalizedTraceSplitter(1:2)
# splitter = QuadformSplitter(1:2)
X,U,Y = [],[],[]
f(x,u) = sin(3sum(x)) + sum(-(u-x).^2)
for i = 1:10000
    if i % 100 == 0
        splitter(grid)
        @show countnodes(grid)
    end
    @show i
    x = 2 .*rand(nx) .-1
    u = 2 .*rand(nu) .-1
    y = f(x,u) + 0.1*(sum(x)+sum(u))*randn()
    push!(X,x)
    push!(U,u)
    push!(Y,y)
    yh = predict(grid, x, u)
    @show y-yh
    update!(grid,x,u,y)
    yh = predict(grid, x, u)
    @show y-yh
end
plot(grid, :value)

##
# plotly()
# gr()
po = (zlims=(-1.5,1.5), clims=(-2,2))
xu = LinRange(-1,1,30),LinRange(-1,1,30)
surface(xu..., f; title="True fun", layout=4, po...)
# surface!(xu..., (x,u)->predict(grid,x,u); title="Approximation", subplot=2, po...)
# surface!(xu..., (x,u)->predict(grid,x,u)-f(x,u); title="Error", subplot=3, po...)
plot!(grid, :value, title="Grid cells", subplot=4)

##

x,u = (X[rand(1:length(X))],U[rand(1:length(X))])
n = walk_down(grid,x,u)
um = argmax_u(n, x)
# plot(u->predict(n.model, x, u), -2,2, title="Q(a)", legend=false)
# vline!(um)
pf = (u1,u2)->predict(n.model, x, [u1;u2])
surface(xu...,pf, title="Q(a)", legend=false)
scatter3d!(um[1:1],um[2:2], [pf(um...)], m=(10,:cyan))
