
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
    actiondims = 1:length(w)
end

QuadraticModel(n::Int) = (n+=1;QuadraticModel(w = zeros(n*(n+1)÷2)))
QuadraticModel(n::Int,u::AbstractUpdater, actiondims) = (n+=1;QuadraticModel(w = zeros(n*(n+1)÷2),updater=u, actiondims=actiondims))

function feature!(m::QuadraticModel, x, u)
    feature!(m::QuadraticModel, [u;x])
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

function argmax_u(m::QuadraticModel, x)
    # TODO: move actions to beginning to improve cache locality, store Q in m to avoid allocating new
    nu = length(m.actiondims)
    nx = length(x)
    np = nx+nu+1
    Q  = zeros(np,np)
    k  = 0
    for i = 1:np, j = i:np
         Q[i,j] = Q[j,i] = model.w[k+=1]
    end
    uinds = 1:nu
    Qux   = Q[uinds, nu+1:nu+nx]
    Quu   = Q[uinds, uinds]
    qu    = Q[uinds, nu+nx+1]
    if isposdef(Quu)
        # TODO: check in domain
        # TODO: solve QP if this solution is not feasible
        return -(Quu\(Qux*x + qu)) ./2
    else
        return Inf
    end
end


# Q = [Qxx Qxu qx;
#      Qxu' Quu qu;
#      qx'  qu' q]
# X = [x;u;1]
