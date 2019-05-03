module KalmanTree
using Statistics, LinearAlgebra
using Parameters, AbstractTrees, RecipesBase, DifferentialDynamicProgramming, PositiveFactorizations, OnlineStats, FillArrays

using Plots # for cgrad

export Grid, LeafNode, countnodes, walk_down, walk_up, argmax_u, volume, visited
export QuadraticModel, QuadraticOnlyModel, QuadraticConstantModel, update!, predict, NewtonUpdater, GradientUpdater, RLSUpdater, KalmanUpdater, feature!, innovation_var, parameter_cov
export RandomSplitter, TraceSplitter, QuadformSplitter, InnovationSplitter, VolumeWrapper, VisitedWrapper


export print_tree, Leaves



include("tree_tools.jl")
include("domain_tools.jl")
include("splitters.jl")
include("quadmodel.jl")
include("updaters.jl")
include("argmax.jl")
include("plotting.jl")

end


# TODO: figure out how to split over a and maintain easy argmax_a
#= Notes:
Should splitting along action dimensions be allowed? Then how does one find argmaxₐ(Q)? since this max can be outside the domain of the model.
If splts along action dimensions are kept at the bottom of the tree, one could operate on a subtree when finding argmaxₐ(Q). Each argmax is a box-constrained QP where the domain in a-dimensions determine the bounds. With this strategy, argmaxₐ must be carried out as many times as there are leaves in the subtree corresponding to the s-coordinate.
Unfortunately, each split of the state-space then doubles the "action subtree" instead of increasing the node count by 1.

A KalmanUpdater is not suitable to indicate which direction in parameter space seem to be nonlinear, or fit the current model poorly. The parameter covariance matrix is completely determined by input data, no influence of fit/ output data. Estimate innovation variance as (s-c)'Q(s-c) + q where c is the cell centroid, s is the state and q is the variance at the centroid. How to keep tihs Q posdef? BFGS updates? Kalmanupdater with cholesky Q = LL'?
=#
