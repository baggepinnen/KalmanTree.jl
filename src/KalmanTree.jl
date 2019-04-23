module KalmanTree
using Statistics, LinearAlgebra
using Parameters, AbstractTrees, RecipesBase, DifferentialDynamicProgramming, PositiveFactorizations, OnlineStats

export Grid, LeafNode, countnodes, walk_down, walk_up, argmax_u
export QuadraticModel, update!, predict, NewtonUpdater, GradientUpdater, RLSUpdater, KalmanUpdater, feature!
export RandomSplitter, TraceSplitter, QuadformSplitter, InnovationSplitter


export print_tree



include("tree_tools.jl")
include("domain_tools.jl")
include("splitters.jl")
include("models.jl")
include("plotting.jl")

end
