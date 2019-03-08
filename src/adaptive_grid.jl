include("tree_tools.jl")

abstract type AbstractModel end

struct LinearModel <: AbstractModel
    x
end
LinearModel() = LinearModel(0)

@with_kw struct QuadraticModel <: AbstractModel
    w
    updater = RLSUpdater(Matrix{Float64}(I,length(w),length(w)), 1.0)
    ϕ = similar(w)
end

# QuadraticModel(w, updater=RLSUpdater(Matrix{Float64}(I,length(w),length(w)), 1.0), ϕ=similar(w)) = QuadraticModel(w=w, updater=updater, ϕ=ϕ)

function feature!(m::QuadraticModel, x, u)
    ϕ = m.ϕ
    i = 1
    lf = length(x)
    ϕ[i:lf] .= x
    i += lf
    ϕ[i:i+lf-1] .= x.^2
    i += lf
    lf = length(u)
    ϕ[i:i+lf-1] .= u
    i += lf
    ϕ[i:i+lf-1] .= u.^2
    i += lf
    for j = 1:length(x), k = 1:length(u)
        ϕ[i] = x[j]*u[k]
        i += 1
    end
    ϕ[i] = 1
    ϕ
end

function update!(m::QuadraticModel, x, u, y)
    feature!(m, x, u)
    update!(m.updater, m.w, m.ϕ, y)
end

function predict(m::QuadraticModel, x, u)
    feature!(m, x, u)
    m.ϕ'm.w
end

abstract type AbstractUpdater end

struct RLSUpdater <: AbstractUpdater
    P
    λ
end

function update!(m::RLSUpdater, w, ϕ, y)
    P,λ = m.P, m.λ
    ϕᵀP = ϕ'*P
    P .-= (P*ϕ*ϕᵀP)/(λ + ϕᵀP*ϕ)
    e = y - ϕ'w
    w .+= P*ϕ .* e
end

function update!(m::AbstractModel, x, y)
    m.x += 1
end


nx,nu = 2,1
np = 2nx+2nu+nx*nu+1
w = zeros(np)
λ = 1
updater = RLSUpdater(Matrix{Float64}(I,np,np), λ)
##
model = QuadraticModel(w,updater)
grid = Grid(nx+nu, model)
for i = 1:1000
    if i % 10 == 0
        g = find_split(grid)
        split(g, rand(1:nx+nu), randn())
    end
    @show i
    x = randn(nx)
    u = randn(nu)
    y = 2sum(x.*u) + 0.1randn()
    yh = predict(grid, x, u)
    @show y-yh
    update!(grid,x,u,y)
    yh = predict(grid, x, u)
    @show y-yh
end
