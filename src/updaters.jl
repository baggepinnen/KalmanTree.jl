abstract type AbstractUpdater end
@with_kw struct RLSUpdater{TP,Tl,Tσ} <: AbstractUpdater
    P::TP
    λ::Tl = 0.999
    σ2::Tσ = Variance(weight=McclainWeight(0.01))
end

RLSUpdater(n::Int; kwargs...) = RLSUpdater(;P=100000Matrix(Eye(n2p(n,QuadraticModel))), kwargs...)

function update!(m::RLSUpdater, w, ϕ, y, x)
    P,λ = m.P, m.λ
    ϕᵀP = ϕ'*P
    P .= (P .- (P*ϕ*ϕᵀP) ./(λ + ϕᵀP*ϕ))./λ
    e = y - ϕ'w
    fit!(m.σ2, e, x)
    w .+= P*ϕ .* e
end

@with_kw struct KalmanUpdater{TP,Tl,Tσ} <: AbstractUpdater
    P::TP
    λ::Tl = 0.001
    σ2::Tσ = Variance(weight=McclainWeight(0.01)) # TODO: it seems OnlineStats are using weight as opposed to forgetting factor, does w = 1-λ hold?
end

KalmanUpdater(n::Int, modeltype=QuadraticModel; kwargs...) = KalmanUpdater(;P=100000Matrix(Eye(n2p(n,modeltype))), kwargs...)

function update!(m::KalmanUpdater, w, ϕ, y, x)
    # TODO: this can be made more efficient
    P,λ,σ²  = m.P, m.λ, value(m.σ2,x) # TODO: must shift x by centroid
    ϕᵀP  = ϕ'*P
    Pϕ   = P*ϕ
    ϕᵀPϕ = ϕᵀP*ϕ
    K    = Pϕ ./(σ²+ϕᵀPϕ)
    P   .= P .- K*ϕᵀP + λ^2*I
    P   .= (P .+ P') ./ 2
    e    = y - ϕ'w
    fit!(m.σ2, e, x)
    w  .+= K .* e
end


@with_kw struct NewtonUpdater{Tσ} <: AbstractUpdater
    α::Float64
    σ2::Tσ = Variance(;weight=McclainWeight(0.01))
end
NewtonUpdater(α) = NewtonUpdater(α=α)

function update!(m::NewtonUpdater, w, ϕ, y, x)
    n,e = newton(m, w, ϕ, y)
    fit!(m.σ2, e, x)
    w .-= m.α*n
end



function newton(m::NewtonUpdater, w, ϕ, y)
    e = (y - ϕ'w)
    (ϕ*ϕ' + 1e-8I)\(ϕ.*(-e)), e
    # -(ϕ.*e), e
end

@with_kw struct GradientUpdater{Tσ} <: AbstractUpdater
    α::Float64
    σ2::Tσ = Variance(;weight=McclainWeight(0.01))
end
GradientUpdater(α) = GradientUpdater(α=α)

function update!(m::GradientUpdater, w, ϕ, y, x)
    e = (y - ϕ'w)
    fit!(m.σ2, e, x)
    w .+= ϕ .* (m.α*e)
end



Statistics.cov(u::AbstractUpdater) = u.P
Statistics.cov(m::AbstractModel) = cov(m.updater)
innovation_var(u::AbstractUpdater) = value(u.σ2)
innovation_var(m::AbstractModel) = innovation_var(m.updater)
innovation_var(n::LeafNode) = innovation_var(n.model)
parameter_cov(u::AbstractUpdater) = value(u.σ2)*cov(u)
parameter_cov(m::AbstractModel) = parameter_cov(m.updater)
parameter_cov(n::LeafNode) = parameter_cov(n.model)
# Q = [Qxx Qxu qx;
#      Qxu' Quu qu;
#      qx'  qu' q]
# X = [x;u;1]
