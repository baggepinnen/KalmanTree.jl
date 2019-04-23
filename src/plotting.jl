@recipe function plot_tree(g::AbstractNode, indicate = :cov; dims=1:2)
    rect(d) = Shape([d[dims[1]][1],d[dims[1]][1],d[dims[1]][2],d[dims[1]][2],d[dims[1]][1]],  [d[dims[2]][1],d[dims[2]][2],d[dims[2]][2],d[dims[2]][1],d[dims[2]][1]])
    if indicate == :cov
        colors = [tr(l.model.updater.P) for l in Leaves(g)]
    else
        colors = [predict(l, centroid(l)) for l in Leaves(g)]
    end
    colors .-= minimum(colors)
    colors ./= maximum(colors)
    colorbar := true
    label := ""
    legend := false
    cg = cgrad(:inferno)
    for (i,l) in enumerate(Leaves(g))
        c = colors[i]
        @series begin
            color := cg[c]
            rect(l.domain)
        end
    end
    nothing
end

@userplot Gridmat
@recipe function gridmat(gm::Gridmat)
    g = gm.args[1]
    d = g.domain
    dims = length(d)
    colorbar := false
    label := ""
    legend := false
    cg = cgrad(:inferno)
    layout := (dims,dims)
    seriestype := :surface
    indmat = LinearIndices((dims,dims))'
    for d1 = 1:dims
        for d2 = 1:d1
            subplot := indmat[d1,d2]
            ylabel := string(d1)
            xlabel := string(d2)
            for l in Leaves(g)
                c = centroid(l)
                @series begin
                    y = LinRange(d[d1]...,20)
                    x = LinRange(d[d2]...,20)
                    pf = (x,y) -> begin
                    c[d2] = x
                    c[d1] = y
                    predict(g, c)
                    end
                    x,y,pf
                end
            end
        end
    end
    nothing
end
