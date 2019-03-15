using DifferentialDynamicProgramming, PositiveFactorizations

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
    Q = w2Q(w)
end

w2Q(w) = zeros(p2n(length(w)),p2n(length(w)))
p2n(p) = Int((-1 + sqrt(1+8p))/2)

QuadraticModel(n::Int;kwargs...) = (n+=1;QuadraticModel(;w = zeros(n*(n+1)÷2),kwargs...))

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

argmax_u(g::AbstractNode, x) = argmax_u(g.model, x, g.domain)

function argmax_u(m::QuadraticModel, x, domain)
    nu = length(m.actiondims)
    nx = length(x)
    np = nx+nu+1
    Q  = m.Q
    k  = 0
    for i = 1:np
        for j = i:np
            Q[i,j] = Q[j,i] = -m.w[k+=1] # TODO: OBS! This negates Quu AND Qux and qu
        end
    end
    uinds = 1:nu
    Qux   = Q[uinds, nu+1:nu+nx]
    Quu   = Symmetric(Q[uinds, uinds])
    qu    = Q[uinds, nu+nx+1]
    RHS = (Qux*x + qu)
    as = vec((Quu\(-RHS)) ./2) # Negated both Q and RHS to be able to check posdef

    posdef = isposdef(Quu)
    if as ∈ domain && posdef
        return as
    elseif posdef # but argmax not in domain
        actiondomain = domain[m.actiondims]
        lb = getindex.(actiondomain, 1)
        ub = getindex.(actiondomain, 2)
        res = boxQP(Quu,vec(RHS), lb, ub, 0as)
        return vec(res[1])
    else # This is tricky
        actiondomain = domain[m.actiondims]
        return gradient_ascent(Quu, RHS, actiondomain)

    end
end

function gradient_ascent(Quu, RHS, domain)
    @show u = centroid(domain) # start point
    @show α = 1maximum(volume(d) for d in domain)
    grad(u) = -α*(2Quu*u + RHS)
    @show g = grad(u)
    @show u .+= g # Take enormous gradient step
    @show project!(u, domain)
    for iter = 1:5
        @show g = grad(u)
        @show project!(u,domain,g)
        @show u .+= grad(u)
        @show project!(u, domain)
    end
    u
end
function project!(u, domain)
    @inbounds for i = eachindex(u) # Project onto domain
        u[i] = clamp(u[i], domain[i][1], domain[i][2])
    end
    u
end

function project!(u, domain, g)
    for i in eachindex(u)
        if (g[i] > 0 && u[i] == domain[i][2]) || (g[i] < 0 && u[i] == domain[i][1])
            g[i] = 0
        end
    end
    g
end

Base.:∈(x::AbstractArray, dom::Vector{<:Tuple}) = all(x ∈ d for (x,d) in zip(x,dom))
Base.:∈(x::Number, dom::Tuple{<:Number,<:Number}) = dom[1] ≤ x ≤ dom[2]

# Q = [Qxx Qxu qx;
#      Qxu' Quu qu;
#      qx'  qu' q]
# X = [x;u;1]
