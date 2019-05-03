
using Test, LinearAlgebra, Random, Statistics, Plots
using KalmanTree
using KalmanTree: depthfirst, breadthfirst, allowed_dims

@testset "Nodes" begin
    root = LeafNode()
    @test root.parent === nothing
    root = split(root, 1, 0.0)
    @test root.dim == 1
    @test root.left isa LeafNode
    @test root.right isa LeafNode
    @test root.left.parent === root
    @test root.right.parent === root
    @test root.left.domain[1] == (-1,0)
    @test root.right.domain[1] == (0,1)
    @test all(!visited, Leaves(root))
    @test !visited(root)

    split(root.left, 1, -0.5)
    split(root.right, 1)
    @test root.left.dim == 1
    @test root.right.dim == 1
    @test root.left.left.domain[1] == (-1,-0.5)
    @test root.left.right.domain[1] == (-0.5,0)
    @test root.right.domain[1] == (0,1)
    @test all(!visited, Leaves(root))
    @test !visited(root)


    @test root.left.left isa LeafNode
    @test walk_up(root.left.left,0) === (root,2)
    @test walk_down(root, -2) === root.left.left
    @test walk_down(root, 2) === root.right.right
    @test walk_down(root, 0.4) === root.right.left
    @test walk_down(root, -0.4) === root.left.right
    @test all(!visited, Leaves(root))
    @test !visited(root)

    @test countnodes(root) == 4

    counter = 0
    depthfirst(root) do node
        counter += 1
    end
    @test counter == 4

    domain = [(-1,1),(-1,1),(-1,1)]
    splitter = RandomSplitter(1:3)
    g = Grid(domain, nothing, splitter)
    @test countnodes(g) == 2^3
    @test g.domain == domain
    @test g.left.domain == [(-1,0),(-1,1),(-1,1)]
    @test g.right.domain == [(0,1),(-1,1),(-1,1)]
    @test g.left.left.domain == [(-1,0),(-1,0),(-1,1)]


    domain = [(-1,1),(-1,1),(-1,1),(-2,2)]
    splitter = RandomSplitter(1:4)
    g = Grid(domain, nothing, splitter)
    @test countnodes(g) == 2^4

    @test g.dim == 1
    @test g.left.dim == 2
    @test g.left.left.dim == 3
    @test g.left.left.right.dim == 4
end

@testset "QuadraticModel" begin
    # Models and updaters

    N = 200
    x = [randn(2) for i = 1:N]
    u = [randn(3) for i = 1:N]
    Q = randn(5,5)
    Q = (Q + Q')/2 + 5I
    @test isposdef(Q)
    q = randn(5)
    c = randn()
    fq(x,u) = fq([u;x])
    fq(x) = x'Q*x + q'x + c
    y = map(fq,x,u)

    m = QuadraticModel(5, updater=KalmanUpdater(5), actiondims=1:3)
    P0 = det(cov(m))
    update!(m,x[1],u[1],y[1])
    foreach(x,u,y) do x,u,y
        update!(m,x,u,y)
    end
    foreach(x,u,y) do x,u,y
        update!(m,x,u,y)
    end

    @test all(zip(x,u,y)) do (x,u,y)
        abs(y - predict(m,x,u)) < 0.0001
    end

    @test det(cov(m)) < P0
    @test cond(cov(m)) < 100
    Quu,Qux, qu = KalmanTree.Qmats(m,x[1])

    @test isposdef(-Quu)
    @test sum(abs, -Quu \ Q[1:3,1:3] - I) < 3e-6
    @test sum(abs, -2qu - q[1:3]) < 3e-6
    @test sum(abs, -Qux - Q[1:3,4:5]) < 3e-6
    @test abs(m.w[end] - c) < 1e-5


    @testset "QuadraticOnlyModel" begin
        Q = randn(2,2)
        Q = Q'Q
        f(x) = x'Q*x
        y = map(f,x)
        m = QuadraticOnlyModel(2, updater=KalmanUpdater(2, QuadraticOnlyModel))
        foreach(x,y) do x,y
            update!(m,x,y)
        end
        Qxx = KalmanTree.Qmats(m,x[1])
        @test isposdef(Qxx)
        @test sum(abs, Qxx - Q) < 3e-6
    end

    @testset "QuadraticConstantModel" begin
        Q = randn(2,2)
        Q = Q'Q
        f(x) = x'Q*x + 0.1
        y = map(f,x)
        m = QuadraticConstantModel(2, updater=KalmanUpdater(2, QuadraticConstantModel))
        foreach(x,y) do x,y
            update!(m,x,y)
        end
        Qxx, q = KalmanTree.Qmats(m,x[1])
        @test isposdef(Qxx)
        @test sum(abs, Qxx - Q) < 3e-6
        @test q ≈ 0.1 atol=0.001
    end


    @testset "QuadraticConstantModel as Variance model" begin

        N = 200
        x = [randn(2) for i = 1:N]
        u = [randn(3) for i = 1:N]
        Q = randn(5,5)
        Q = (Q + Q')/2 + 5I
        @test isposdef(Q)
        q = randn(5)
        c = randn()
        fq(x,u) = fq([u;x]) + dot(x,x)*randn()
        fq(x) = x'Q*x + q⋅x + c
        y = map(fq,x,u)
        σ2 = QuadraticConstantModel(2, updater=KalmanUpdater(2, QuadraticConstantModel))

        m = QuadraticModel(5, updater=KalmanUpdater(5, σ2=σ2), actiondims=1:3)
        update!(m,x[1],u[1],y[1])
        foreach(x,u,y) do x,u,y
            update!(m,x,u,y)
        end

        foreach(x,y) do x,y
            update!(m,x,y)
        end
        Qxx, q = KalmanTree.Qmats(σ2,x[1])
        @test_skip isposdef(Qxx)

    end


    @testset "Test other updaters" begin
        updater = NewtonUpdater(0.5)
        m = QuadraticModel(5, actiondims=1:3, updater = updater)
        for i = 1:5
            foreach(x,u,y) do x,u,y
                update!(m,x,u,y)
            end
        end

        @test mean(zip(x,u,y)) do (x,u,y)
            abs2(y - predict(m,x,u))
        end  < 1e-6

        updater = GradientUpdater(0.01)
        m = QuadraticModel(5, actiondims=1:3, updater = updater)
        for i = 1:50
            foreach(x,u,y) do x,u,y
                update!(m,x,u,y)
            end
        end

        @test mean(zip(x,u,y)) do (x,u,y)
            abs2(y - predict(m,x,u))
        end  < 1e-6
    end

    @testset "innovation variance KalmanUpdater" begin
        import Base.Iterators: cycle, take
        σ = 0.1
        N = 2000
        x = [randn(2) for i = 1:N]
        u = [randn(3) for i = 1:N]
        y = map(fq,x,u) .+ σ .* randn.()

        m = QuadraticModel(5, updater=KalmanUpdater(5, λ=0.001), actiondims=1:3)
        m.updater.σ2.weight.α = 0.01
        vars = []
        foreach(x,u,y) do x,u,y
            update!(m,x,u,y)
            push!(vars, innovation_var(m))
        end
        # plot(sqrt.(vars))
        @test sqrt(innovation_var(m)) > 0.9σ

        @test @show(mean(zip(x,u,y)) do (x,u,y)
            abs2(y - predict(m,x,u))
        end |> sqrt) < 1.1σ

        foreach(take(cycle(zip(x,u,y)),10N)) do (x,u,y)
            update!(m,x,u,y)
        end
        # @test sqrt(innovation_var(m)) < σ # Expect some overfitting
        @test  @show(mean(zip(x,u,y)) do (x,u,y)
            abs2(y - predict(m,x,u))
        end |> sqrt) < 1.1σ
    end
end

@testset "innovation variance RLSUpdater" begin
    import Base.Iterators: cycle, take
    σ = 0.1
    N = 2000
    x = [randn(2) for i = 1:N]
    u = [randn(3) for i = 1:N]
    y = map(fq,x,u) .+ σ .* randn.()

    m = QuadraticModel(5, updater=RLSUpdater(5, λ=0.999), actiondims=1:3)

    vars = []
    foreach(x,u,y) do x,u,y
        update!(m,x,u,y)
        push!(vars, innovation_var(m))
    end
    # plot(sqrt.(vars))

    @test sqrt(innovation_var(m)) > 0.9σ

    @test @show(mean(zip(x,u,y)) do (x,u,y)
        abs2(y - predict(m,x,u))
    end |> sqrt) < 1.1σ

    foreach(take(cycle(zip(x,u,y)),10N)) do (x,u,y)
        update!(m,x,u,y)
    end
    # @test sqrt(innovation_var(m)) < σ # Expect some overfitting
    @test  @show(mean(zip(x,u,y)) do (x,u,y)
        abs2(y - predict(m,x,u))
    end |> sqrt) < 1.1σ
end


@testset "innovation variance NewtonUpdater" begin
    import Base.Iterators: cycle, take
    σ = 0.1
    N = 2000
    x = [randn(2) for i = 1:N]
    u = [randn(3) for i = 1:N]
    y = map(fq,x,u) .+ σ .* randn.()

    m = QuadraticModel(5, updater=NewtonUpdater(0.1), actiondims=1:3)

    vars = []
    foreach(x,u,y) do x,u,y
        update!(m,x,u,y)
        push!(vars, innovation_var(m))
    end
    # plot(sqrt.(vars))

    @test sqrt(innovation_var(m)) > 0.9σ

    @test @show(mean(zip(x,u,y)) do (x,u,y)
        abs2(y - predict(m,x,u))
    end |> sqrt) < 1.4σ

    foreach(take(cycle(zip(x,u,y)),10N)) do (x,u,y)
        update!(m,x,u,y)
    end
    # @test sqrt(innovation_var(m)) < σ # Expect some overfitting
    @test  @show(mean(zip(x,u,y)) do (x,u,y)
        abs2(y - predict(m,x,u))
    end |> sqrt) < 1.15σ
end




# Integration tests
@testset "Integration" begin
    f = (x,u) -> sin(3sum(x)) + sum(-(u.-x).^2)
    function integrated_test(nx,nu)
        domain = fill((-1.,1.), nx+nu)
        model = QuadraticModel(nx+nu; updater=KalmanUpdater(nx+nu), actiondims=1:nu)
        splitter = InnovationSplitter(nu+1:nu+nx) |> VolumeWrapper |> VisitedWrapper
        g = Grid(domain, model, splitter)
        for i = 1:10000
            if i % 100 == 0
                KalmanTree.find_and_apply_split(g, splitter)
                # @show countnodes(g)
            end
            x = 2 .*rand(nx) .-1
            u = 2 .*rand(nu) .-1
            y = f(x,u) + 0.1*(sum(x)+sum(u))*randn()
            # yh = predict(g, x, u)
            # @show i
            # @show y-yh
            update!(g,x,u,y)
            # yh = predict(g, x, u)
            # @show y-yh
        end
        g
        predfun = (x,u)->predict(g,x,u)
        errorfun = (x,u)->predfun(x,u)-f(x,u)
        g,predfun,errorfun
    end
    xu = LinRange(-1,1,30),LinRange(-1,1,30)
    g,predfun,errorfun = integrated_test(1,1)
    @test visited(g)
    @test all(x->abs(errorfun(x...))<1, Iterators.product(xu...))

    # plot(g, :value, dims=allowed_dims(splitter))
    # plot(g, :cov, dims=allowed_dims(splitter))
    plot(g, :value)
    plot(g, :cov)


    # po = (zlims=(-1.5,1.5), clims=(-2,2))
    # surface(xu..., f; title="True fun", layout=5, po...)
    # surface!(xu..., predfun; title="Approximation", subplot=2, po...)
    # surface!(xu..., errorfun; title="Error", subplot=3, po...)
    # plot!(g, :value, title="Grid cells", subplot=4)
    # plot!(g, :cov, title="Grid cells", subplot=5)

    @test mean(1:20) do i
        g,predfun,errorfun = integrated_test(1,1)
        mean(x->abs2(errorfun(x...)), Iterators.product(xu...))
    end < 0.010
    @test mean(1:20) do i
        g,predfun,errorfun = integrated_test(2,2)
        x,u = [2rand(2).-1 for _ in 1:10],[2rand(2).-1 for _ in 1:10]
        mean(x->abs2(errorfun(x...)), zip(x,u))
    end < 0.03
end

##


@testset "argmax_u" begin


    f = (x,u) -> sin(3sum(x)) + sum(-(u.-x).^2)
    function argmaxtest(nx,nu)
        domain = fill((-1.,1.), nx+nu)
        model = QuadraticModel(nx+nu; updater=KalmanUpdater(nx+nu), actiondims=1:nu)
        splitter = InnovationSplitter(nu+1:nu+nx) |> VolumeWrapper |> VisitedWrapper
        g = Grid(domain, model, splitter)
        X,U,Y = [],[],[]
        for i = 1:10000
            if i % 100 == 0
                KalmanTree.find_and_apply_split(g, splitter)
            end
            x = 2 .*rand(nx) .-1
            u = 2 .*rand(nu) .-1
            y = f(x,u) + 0.1*(sum(x)+sum(u))*randn()
            push!(X,x)
            push!(U,u)
            push!(Y,y)
            KalmanTree.update!(g,x,u,y)

        end
        X,U,Y,g
    end
    X,U,Y,g = argmaxtest(1,1)
    for i = 1:100
        # E = @benchmark begin
        x,u = (X[rand(1:length(X))],U[rand(1:length(X))])
        n = walk_down(g,x,u)
        um = KalmanTree.argmax_u(n, x)
        # end
        # display(E)

        # plot(u->KalmanTree.predict(n.model, x, u), -2,2, title="Q(a)", legend=false)
        # vline!([um])
        # vline!([n.domain[n.model.actiondims][]...], l=(:dash,:black))

        @test um[] ∈ (n.domain[n.model.actiondims][]...,)
        # test that a small ϵ makes the value worse, unless we're at the boundary of the domain
        @test KalmanTree.predict(n.model, x, um[]) > KalmanTree.predict(n.model, x, um[]+1e-4) || um[] == n.domain[n.model.actiondims][][2]
        @test KalmanTree.predict(n.model, x, um[]) > KalmanTree.predict(n.model, x, um[]-1e-4) || um[] == n.domain[n.model.actiondims][][1]
    end

    # pf = (u1,u2)->predict(n.model, x, [u1;u2])
    # surface(xu...,pf, title="Q(a)", legend=false)
    # scatter3d!(um[1:1],um[2:2], [pf(um...)], m=(10,:cyan))
end
