
abstract type AbstractModel end
abstract type LinearInParamsModel <: AbstractModel end

function update!(m::LinearInParamsModel, x, u, y)
    feature!(m, x, u)
    update!(m.updater, m.w, m.ϕ, y, x)
end

function update!(m::LinearInParamsModel, x, y)
    feature!(m, x)
    update!(m.updater, m.w, m.ϕ, y, x)
end

function predict(m::LinearInParamsModel, args...)
    feature!(m, args...)
    m.ϕ'm.w
end

# OnlineStats interface
OnlineStats.fit!(m,e,x) = fit!(m,e) # Fallback to discarding state x
OnlineStats.fit!(m::AbstractModel,e,x) = update!(m,x,e^2) # If using one of our models
OnlineStats.value(m,x) = value(m) # Fallback to discarding state x
OnlineStats.value(m::AbstractModel,x) = predict(m,x) # If using one of our models


@with_kw struct QuadraticModel <: LinearInParamsModel
    w
    updater
    ϕ = similar(w)
    actiondims = 1:p2n(length(w),QuadraticModel)
    Q = w2Q(w,QuadraticModel)
end

w2Q(w, ::Type{QuadraticModel}) = zeros(p2n(length(w),QuadraticModel)+1,p2n(length(w),QuadraticModel)+1)
p2n(p, ::Type{QuadraticModel}) = Int((-1 + sqrt(1+8p))/2)-1
n2p(n, ::Type{QuadraticModel}) = (n+=1;n*(n+1)÷2)

function QuadraticModel(n::Int; kwargs...)
    QuadraticModel(;w = zeros(n2p(n,QuadraticModel)), kwargs...)
end


function feature!(m::QuadraticModel, x, u)
    ux = @view m.Q[1:length(x)+length(u),1]
    ux[1:length(u)] .= u
    ux[length(u)+1:end] .= x
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

# QuadraticOnlyModels


@with_kw struct QuadraticOnlyModel <: LinearInParamsModel
    w
    updater
    ϕ = similar(w)
    Q = w2Q(w,QuadraticOnlyModel)
end

w2Q(w, ::Type{QuadraticOnlyModel}) = zeros(p2n(length(w),QuadraticOnlyModel),p2n(length(w),QuadraticOnlyModel))
p2n(p, ::Type{QuadraticOnlyModel}) = Int((sqrt(8p + 1) - 1)/2)
n2p(n, ::Type{QuadraticOnlyModel}) = n*(n+1)÷2
@assert n2p(3,QuadraticOnlyModel) == 6
@assert p2n(6,QuadraticOnlyModel) == 3

function QuadraticOnlyModel(n::Int; kwargs...)
    QuadraticOnlyModel(;w = zeros(n2p(n, QuadraticOnlyModel)), kwargs...)
end


function feature!(m::QuadraticOnlyModel, x)
    ϕ = m.ϕ
    k = 1
    for i in eachindex(x)
        for j in i:length(x)
            ϕ[k] = x[i]*x[j]
            k += 1
        end
    end
    @assert k == length(ϕ)+1
    ϕ
end

function Qmats(m::QuadraticOnlyModel,x)
    nx = length(x)
    np = nx
    Q  = m.Q
    k  = 0
    for i = 1:np
        Q[i,i] = Q[i,i] = m.w[k+=1]
        for j = i+1:np
            Q[i,j] = Q[j,i] = 0.5m.w[k+=1]
        end
    end
    Symmetric(Q)
end




# Quadratic plus constant


@with_kw struct QuadraticConstantModel <: LinearInParamsModel
    w
    updater
    ϕ = similar(w)
    Q = w2Q(w,QuadraticConstantModel)
end

w2Q(w, ::Type{QuadraticConstantModel}) = zeros(p2n(length(w),QuadraticConstantModel),p2n(length(w),QuadraticConstantModel))
p2n(p, ::Type{QuadraticConstantModel}) = Int((sqrt(8(p-1) + 1) - 1)/2)
n2p(n, ::Type{QuadraticConstantModel}) = n*(n+1)÷2+1
@assert n2p(3,QuadraticConstantModel) == 7
@assert p2n(7,QuadraticConstantModel) == 3

function QuadraticConstantModel(n::Int; kwargs...)
    QuadraticConstantModel(;w = zeros(n2p(n, QuadraticConstantModel)), kwargs...)
end


function feature!(m::QuadraticConstantModel, x)
    ϕ = m.ϕ
    k = 1
    for i in eachindex(x)
        for j in i:length(x)
            ϕ[k] = x[i]*x[j]
            k += 1
        end
        # ϕ[k] = x[i]
        # k += 1
    end
    ϕ[k] = 1
    @assert k == length(ϕ)
    ϕ
end

function Qmats(m::QuadraticConstantModel,x)
    nx = length(x)
    np = nx
    Q  = m.Q
    k  = 0
    for i = 1:np
        Q[i,i] = Q[i,i] = m.w[k+=1]
        for j = i+1:np
            Q[i,j] = Q[j,i] = 0.5m.w[k+=1]
        end
    end
    Symmetric(Q), m.w[k+=1]
end










# using OnlineStats
# import OnlineStats: VectorOb, _merge!, value, _fit!, smooth!, nvars, smooth_syr!
#
# #-----------------------------------------------------------------------# QuadModel
# """
#     QuadModel(p=0; weight=EqualWeight())
#     QuadModel(::Type{T}, p=0; weight=EqualWeight())
# Calculate a quadratic model x'Qx of `p` variables.  If the number of variables is
# unknown, leave the default `p=0`.
# # Example
#     o = fit!(QuadModel(), randn(100, 4))
#     cor(o)
#     cov(o)
#     mean(o)
#     var(o)
# """
# mutable struct QuadModel{T,W} <: OnlineStat{VectorOb} where T<:Number
#     Q::Matrix{T}
#     y::Lag{Vector{T}}
#     Δx::Lag{Vector{T}}
#     weight::W
#     n::Int
# end
#
# nvars(o::QuadModel) = size(o.Q, 1)
# function QuadModel(::Type{T}, p::Int=0; weight = EqualWeight()) where T<:Number
#     QuadModel(Matrix{T}(I,p,p), Lag(Vector{T},2), Lag(Vector{T},2), weight, 0)
# end
#
# QuadModel(p::Int=0; weight = EqualWeight()) = QuadModel(Matrix{Float64}(I,p,p), Lag(Vector{Float64},2), Lag(Vector{Float64},2), weight, 0)
#
# function _fit!(o::QuadModel{T}, x, y) where {T}
#     γ = o.weight(o.n += 1)
#     if isempty(o.Q)
#         p = length(x)
#         o.y = Lag(Vector{T},2)
#         o.Δx = Lag(Vector{T},2)
#         o.Q = zeros(T, p, p)
#     end
#
#     fit!(o.y, o.Q*x)
#     fit!(o.Δx, x)
#     # smooth_syr!(o.Q, x, γ)
#     SR1!(o.Q, diff(value(o.y))[], diff(value(o.Δx))[], γ)
# end
#
#
# function value(o::QuadModel, x)
#     x'o.Q*x
# end
#
# function _merge!(o::QuadModel, o2::QuadModel)
#     γ = o2.n / (o.n += o2.n)
#     smooth!(o.Q, o2.Q, γ)
#     smooth!(o.b, o2.b, γ)
#     o
# end
#
# function SR1!(Q, y, Δx, γ)
#     @show y, Δx
#     Δ = y-Q*Δx
#     Q .= Q .+ γ.*((Δ*Δ')/(Δ'Δx) .- Q)
#
#     # for j in 1:size(Q, 2), i in 1:j
#     #     Q[i, j] = smooth(Q[i,j], Δ[i] * conj(Δ[j]), γ)
#     # end
#
# end
#
# using LinearOperators
# Q = randn(2,2)
# Q = Q'Q
# f(x) = x'Q*x
# o = QuadModel(2)
# o = LSR1Operator(2)
# Base.push!(o::LSR1Operator, s) = push!(o,s,o*s)
# ##
# x = randn(2)
# # _fit!(o,x,y
