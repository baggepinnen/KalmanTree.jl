@recipe function plot_tree(g::AbstractNode, indicate = :cov; dims=1:2)
    rect(d) = Shape([d[dims[1]][1],d[dims[1]][1],d[dims[1]][2],d[dims[1]][2],d[dims[1]][1]],  [d[dims[2]][1],d[dims[2]][2],d[dims[2]][2],d[dims[2]][1],d[dims[2]][1]])
    covs = [tr(l.model.updater.P) for l in Leaves(g)]
    mc = maximum(covs)
    colorbar := true
    label := ""
    legend := false
    cg = cgrad(:inferno)
    for l in Leaves(g)
        if indicate == :cov
            c = tr(l.model.updater.P)/mc
        else
            c = (predict(l, centroid(l))+2)/4
        end
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
            for l in Leaves(g)
                c = centroid(l)
                @series begin
                    xlabel --> string(d1)
                    ylabel --> string(d2)
                    x = LinRange(d[d1]...,20)
                    y = LinRange(d[d2]...,20)
                    pf = (x,y) -> begin
                    c[d1] = x
                    c[d2] = y
                    predict(g, c)
                    end
                    x,y,pf
                end
            end
        end
    end
    nothing
end
