
function argmax_u(m::QuadraticModel, x, domain)
    actiondomain = domain[m.actiondims]
    Quu, Qux, qu = Qmats(m,x)
    RHS = (Qux*x + qu)
    as = -(Quu-1e-8I)\RHS # Negated both Q and RHS to be able to check posdef
    posdef = isposdef(Quu)
    if as ∈ actiondomain && posdef
        return as
    elseif posdef # but argmax not in domain
        lb = float.(getindex.(actiondomain, 1))
        ub = float.(getindex.(actiondomain, 2))
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
argmax_u(g::LeafNode, x) = argmax_u(g.model, x, g.domain)
argmax_u(g::AbstractNode, x) = argmax_u(walk_down(g,x,(0,)), x) # TODO: fix this [0]


function newton_ascent(Quu, RHS, domain)
    u = centroid(domain) # start point
    α = 1maximum(volume(d) for d in domain)
    Quuf = cholesky(Positive, Quu)
    grad(u) = -α*(Quuf\(2Quu*u + RHS))
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

function newton(m::QuadraticModel, x, u, y)
    ϕ = feature!(m,x,u)
    newton(m.updater, ϕ, y)
end
