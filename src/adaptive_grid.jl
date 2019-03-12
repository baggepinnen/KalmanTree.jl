include("tree_tools.jl")
include("domain_tools.jl")
include("models.jl")

#= Notes:
Should splitting along action dimensions be allowed? Then how does one find argmaxₐ(Q)? since this max can be outside the domain of the model.
If splts along action dimensions are kept at the bottom of the tree, one could operate on a subtree when finding argmaxₐ(Q). Each argmax is a box-constrained QP where the domain in a-dimensions determine the bounds. With this strategy, argmaxₐ must be carried out as many times as there are leaves in the subtree corresponding to the s-coordinate.
First, the unconstraind argmax can be calculated for each cell. If the highest is inside its bounds, no box-constrained QP has to be solved
=#

##
nx,nu = 1,1
np = 2nx+2nu+nx*nu+1
w = zeros(np)
λ = 1
domain = [(-1.,1.),(-1.,1.)]#,(-1.,1.)]
updater = RLSUpdater(Matrix{Float64}(100I,np,np), λ)
model = QuadraticModel(nx+nu,updater)
grid = Grid(domain, model)
# splitter = TraceSplitter(1:1)
# splitter = NormalizedTraceSplitter(1:1)
splitter = QuadformSplitter(1:1)
X,U,Y = [],[],[]
f(x,u) = sin(3sum(x)) + sin(3sum(u))
for i = 1:10000
    if i % 100 == 0
        splitter(grid)
        @show countnodes(grid)
    end
    @show i
    x = 2 .*rand(nx) .-1
    u = 2 .*rand(nu) .-1
    y = f(x,u) + 0.1*(sum(x)+sum(u))*randn()
    push!(X,x[])
    push!(U,u[])
    push!(Y,y[])
    yh = predict(grid, x, u)
    @show y-yh
    update!(grid,x,u,y)
    yh = predict(grid, x, u)
    @show y-yh
end
plot_tree(grid)

##
# surface(X,U,f.(X,U))
