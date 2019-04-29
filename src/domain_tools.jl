
volume(g::AbstractNode) = volume(g.domain)
volume(d::Tuple) = d[2]-d[1]
volume(d::AbstractVector) = prod(volume, d)

centroid(g::AbstractNode) = centroid(g.domain)
centroid(d::AbstractVector) = centroid.(d)
centroid(d::Tuple) = (d[2]+d[1])/2

Base.:∈(x::AbstractArray, dom::Vector{<:Tuple}) = all(x ∈ d for (x,d) in zip(x,dom))
Base.:∈(x::Number, dom::Tuple{<:Number,<:Number}) = dom[1] ≤ x ≤ dom[2]
