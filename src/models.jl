
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

QuadraticModel(n::Int) = (n+=1;QuadraticModel(w = zeros(n*(n+1)÷2)))
QuadraticModel(n::Int,u::AbstractUpdater) = (n+=1;QuadraticModel(w = zeros(n*(n+1)÷2),updater=u))

# QuadraticModel(w, updater=RLSUpdater(Matrix{Float64}(I,length(w),length(w)), 1.0), ϕ=similar(w)) = QuadraticModel(w=w, updater=updater, ϕ=ϕ)

function feature!(m::QuadraticModel, x, u)
    feature!(m::QuadraticModel, [x;u])
end

function feature!(m::QuadraticModel, x)
    ϕ = m.ϕ
    k = 1
    for i in eachindex(x)
        for j in i:length(x)
            ϕ[k] = x[i]*x[j]
            k += 1
        end
        ϕ[k] = x[i]
        k += 1
    end
    ϕ[k] = 1
    ϕ
end

function update!(m::QuadraticModel, x, u, y)
    feature!(m, x, u)
    update!(m.updater, m.w, m.ϕ, y)
end

function predict(m::QuadraticModel, args...)
    feature!(m, args...)
    m.ϕ'm.w
end


function update!(m::AbstractModel, x, y)
    m.x += 1
end
