
volume(g::AbstractNode) = volume(g.domain)
volume(d::Tuple) = d[2]-d[1]

function volume(d::AbstractVector)
    prod(volume, d)
end

centroid(g::AbstractNode) = centroid(g.domain)
centroid(d::AbstractVector) = centroid.(d)
centroid(d::Tuple) = (d[2]+d[1])/2
