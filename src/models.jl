using DifferentialDynamicProgramming
using PositiveFactorizations
using OnlineStats

abstract type AbstractUpdater end
@with_kw struct RLSUpdater{TP,Tl} <: AbstractUpdater
    P::TP
    λ::Tl
    σ2::OnlineStats.Variance{Float64,ExponentialWeight} = Variance(weight=ExponentialWeight(λ))
end

function update!(m::RLSUpdater, w, ϕ, y)
    P,λ = m.P, m.λ
    ϕᵀP = ϕ'*P
    P .= (P .- (P*ϕ*ϕᵀP) ./(λ + ϕᵀP*ϕ))./λ
    e = y - ϕ'w
    fit!(m.σ2, e)
    w .+= P*ϕ .* e
end

@with_kw struct KalmanUpdater{TP,Tl} <: AbstractUpdater
    P::TP
    λ::Tl
    σ2::OnlineStats.Variance{Float64,ExponentialWeight} = Variance(weight=ExponentialWeight(λ))
end

function update!(m::KalmanUpdater, w, ϕ, y)
    # TODO: this can be made more efficient
    P,λ  = m.P, m.λ
    ϕᵀP  = ϕ'*P
    Pϕ   = P*ϕ
    ϕᵀPϕ = ϕᵀP*ϕ
    K    = Pϕ ./(1+ϕᵀPϕ)
    P   .= P .- K*ϕᵀP + λ*I
    P   .= (P .+ P') ./ 2
    e    = y - ϕ'w
    fit!(m.σ2, e)
    w  .+= K .* e
end

abstract type AbstractModel end

struct LinearModel <: AbstractModel
    x
end
LinearModel() = LinearModel(0)

@with_kw struct QuadraticModel <: AbstractModel
    w
    updater = KalmanUpdater(P=Matrix{Float64}(I,length(w),length(w)), λ=1.0)
    ϕ = similar(w)
    actiondims = 1:length(w)
    Q = w2Q(w)
end

w2Q(w) = zeros(p2n(length(w)),p2n(length(w)))
p2n(p) = Int((-1 + sqrt(1+8p))/2)

function QuadraticModel(n::Int;λ=1.0,P0=1000,kwargs...)
    n+=1
    w = zeros(n*(n+1)÷2)
    updater = KalmanUpdater(P=Matrix{Float64}(P0*I,length(w),length(w)), λ=λ)
    QuadraticModel(;w = w, updater=updater, kwargs...)
end

function feature!(m::QuadraticModel, x, u)
    ux = @view m.Q[1:length(x)+length(u),1]
    ux[1:length(x)] .= x
    ux[length(x)+1:end] .= u
    feature!(m::QuadraticModel, ux)
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


argmax_u(g::AbstractNode, x) = argmax_u(g.model, x, g.domain)

function Qmats(m::QuadraticModel,x)
    nu = length(m.actiondims)
    nx = length(x)
    np = nx+nu+1
    Q  = m.Q
    k  = 0
    for i = 1:np
        Q[i,i] = Q[i,i] = -m.w[k+=1]
        for j = i+1:np
            Q[i,j] = Q[j,i] = -0.5m.w[k+=1]# TODO: OBS! This negates Quu AND Qux and qu
        end
    end
    uinds = 1:nu
    Qux   = Q[uinds, nu+1:nu+nx]
    Quu   = Symmetric(Q[uinds, uinds])
    qu    = Q[uinds, nu+nx+1]
    Quu, Qux, qu
end

function argmax_u(m::QuadraticModel, x, domain)
    actiondomain = domain[m.actiondims]
    Quu, Qux, qu = Qmats(m,x)
    RHS = (Qux*x + qu)
    as = vec((Quu\(-RHS)) ./2) # Negated both Q and RHS to be able to check posdef
    posdef = isposdef(Quu)
    if as ∈ actiondomain && posdef
        return as
    elseif posdef # but argmax not in domain
        lb = getindex.(actiondomain, 1)
        ub = getindex.(actiondomain, 2)
        u0 = centroid(actiondomain)
        res = boxQP(Quu,RHS, lb, ub, u0;
                                    minGrad = 1e-10,
                                    maxIter=200)
        if res[2] >= 6
            return vec(res[1])
        end
    end
    # This is tricky
    return newton_ascent(Quu, RHS, actiondomain)

end


function newton_ascent(Quu, RHS, domain)
    u = centroid(domain) # start point
    α = 1maximum(volume(d) for d in domain)
    Quuf = cholesky(Positive, Quu)
    grad(u) = -α*Quuf\(2Quu*u + RHS)
    g = grad(u)
    u .+= g # Take enormous gradient step
    project!(u, domain)
    for iter = 1:5
        g = grad(u)
        project!(u,domain,g)
        u .+= grad(u)
        project!(u, domain)
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

Statistics.cov(m::AbstractModel) = cov(m.updater)
Statistics.cov(u::AbstractUpdater) = u.P
innovation_var(m::AbstractModel) = innovation_var(m.updater)
innovation_var(u::AbstractUpdater) = value(u.σ2)
parameter_cov(m::AbstractModel) = parameter_cov(m.updater)
parameter_cov(u::AbstractUpdater) = value(u.σ2)*cov(u)

# Q = [Qxx Qxu qx;
#      Qxu' Quu qu;
#      qx'  qu' q]
# X = [x;u;1]
