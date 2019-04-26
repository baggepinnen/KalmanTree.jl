# using Pkg
# Pkg.add("Plots")
# Pkg.add("BasisFunctionExpansions")
# Pkg.add("ValueHistories")
# Pkg.add("https://github.com/JuliaML/OpenAIGym.jl")
using Distributed
addprocs(4)
@everywhere begin
    using KalmanTree, OpenAIGym, ValueHistories, Plots, Random, LinearAlgebra, Hyperopt
    default(size=(1800,1000))


    nx,nu = 4,1
    domain = [(0,1), (-1.1,1.1) ,(-3.5,3.5) ,(-0.3,0.3) ,(-3.5,3.5)]

    splitter = InnovationSplitter(allowed_dims=2:5) |> VolumeWrapper |> VisitedWrapper

    struct Qfun{T1,T2}
        grid::T1
        splitter::T2
    end

    (Q::Qfun)(s,a) = KalmanTree.predict(Q.grid, s, a)

    """This function makes for a nice syntax of updating the Q-function"""
    function Base.setindex!(Q::Qfun, q, s, a)
        KalmanTree.update!(Q.grid, s, a, q)
    end


    mutable struct ϵGreedyPolicy{T} <: AbstractPolicy
        ϵ::Float64
        decay_rate::Float64
        Q::T
    end

    """Calling this function decays the ϵ"""
    function decay!(p::ϵGreedyPolicy)
        p.ϵ *= p.decay_rate
    end

    """This is our ϵ-greedy action function"""
    function Reinforce.action(p::ϵGreedyPolicy, r, s, A)
        rand() < p.ϵ ? rand(0:1) : p.Q(s,1) > p.Q(s,0) ? 1 : 0
    end

    """max(Q(s,a)) over a"""
    function max_a(Q, s)
        max(Q(s,1), Q(s,0))
    end

    argmax_a(Q,s) = Q(s,1) > Q(s,0) ? 1 : 0

    env = GymEnv("CartPole-v0");

    function Qlearning(Q, policy, num_episodes, α; plotting=true, target_update_interval=100)
        γ            = 0.99; # Discounting factor
        Qt = deepcopy(Q)
        plotting && (fig = plot(layout=2, show=true))
        reward_history = ValueHistories.History(Float64)
        m = zeros(4)
        for i = 1:num_episodes
            ep = Episode(env, policy)
            # α *= decay_rate # Decay the learning rate
            decay!(policy) # Decay greedyness
            if i % target_update_interval == 0
                Qt = deepcopy(Q)
                KalmanTree.find_and_apply_split(Q.grid, Q.splitter)
                # @show countnodes(Q.grid)
            end
            for (s,a,r,s1) in ep # An episode object is iterable
                # Q[s,a] = α*(r + γ*max_a(Q, s1) - Q(s,a)) # Update Q using Q-learning
                Q[s,a] = α*(r + γ*Qt(s1,argmax_a(Q, s1)) - Qt(s,a)) # Update Q using double Q-learning
                m = max.(m,abs.(s))
            end
            push!(reward_history, i, ep.total_reward)
            # i % 20 == 0 && println("Episode: $i, reward: $(ep.total_reward)")
            if plotting && i % 50 == 0
                p1 = plot(reward_history, show=false)
                p2 = gridmat(Q.grid, show=false, axis=false)
                plot(p1,p2); gui()
            end
        end
        # plot(reward_history, title="Rewards", xlabel="Episode", show=true)
        reward_history
    end
end
#

# ho = Hyperopt.@phyperopt for i=50, sampler = RandomSampler(),
#     α   = LinRange(0.5,3,200),
#     λ   = exp10.(LinRange(-3,1,200)),
#     tui = round.(Int, exp10.(LinRange(1, 2, 200)))
    # P0  = exp10.(LinRange(0, 15, 200))

    @show i
    α, λ, tui, P0 = 3, 1e-2, 20, 1.5e10
    updater = KalmanUpdater(nx+nu, λ=λ)
    # updater  = NewtonUpdater(α=λ)
    m        = QuadraticModel(nx+nu; updater=updater, actiondims=1:1)
    gridm        = Grid(domain, m, splitter, initial_split=2:5)
    Q            = Qfun(gridm, splitter); # Q is now our Q-function approximator
    num_episodes = 300
    # α          = .5 # Initial learning rate
    ϵ            = 0.5 # Initial chance of choosing random action
    decay_rate   = 0.992 # decay rate for learning rate and ϵ
    # @info "Final ϵ: ", decay_rate^num_episodes
    policy = ϵGreedyPolicy(ϵ, decay_rate, Q);
    rh = Qlearning(Q, policy, num_episodes, α, plotting = false, target_update_interval=tui)
    sum(rh.values)
# end

# plot(ho, smooth=true)

# xp = map(volume, Leaves(gridm))
# yp = map(innovation_var, Leaves(gridm))
# scatter(xp,yp)
# julia> maximum(ho)
# (Real[0.67, 0.999854, 28, 6.25055], 200.0)

#
# struct BoltzmannPolicy <: AbstractPolicy end
#
# decay!(policy::BoltzmannPolicy) = nothing
#
# """This is our Boltzmann exploration action function"""
# function Reinforce.action(policy::BoltzmannPolicy, r, s, A)
#     Q1,Q0 = Q(s,1), Q(s,0)
#     prob1 = exp(Q1)/(exp(Q1)+exp(Q0))
#     rand() < prob1 ? 1 : 0
# end
#
# policy = BoltzmannPolicy()
# m        = QuadraticModel(nx+nu; actiondims=1:1, λ=λ, P0=P0)
# gridm        = Grid(domain, m, splitter, initial_split=0)
# Q            = Qfun(gridm, splitter);
#
# @time Qlearning(Q,policy, num_episodes, α, plotting = false)
