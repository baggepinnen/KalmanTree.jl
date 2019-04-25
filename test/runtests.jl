
using Test, LinearAlgebra, Random, Statistics, Plots
using KalmanTree
using KalmanTree: depthfirst, breadthfirst, allowed_dims
@testset "KalmanTree" begin
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

    @testset "Models and updaters" begin
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

        m = QuadraticModel(5, λ=1, P0=100000, actiondims=1:3)
        P0 = det(cov(m))
        update!(m,x[1],u[1],y[1])
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
        @test sum(abs, -Quu \ Q[1:3,1:3] - I) < 5
        @test sum(abs, -2qu - q[1:3]) < 1e-5
        @test sum(abs, -Qux - Q[1:3,4:5]) < 1.3e-5
        @test abs(m.w[end] - c) < 1e-5




        updater = NewtonUpdater(0.5, 0.999)
        m = QuadraticModel(5, actiondims=1:3, updater = updater)
        for i = 1:5
            foreach(x,u,y) do x,u,y
                update!(m,x,u,y)
            end
        end

        @test mean(zip(x,u,y)) do (x,u,y)
            abs2(y - predict(m,x,u))
        end  < 1e-6

        updater = GradientUpdater(0.01, 0.999)
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


    # Integration tests
    @testset "Integration" begin
        f = (x,u) -> sin(3sum(x)) + sum(-(u.-x).^2)
        function integrated_test(nx,nu)
            domain = fill((-1.,1.), nx+nu)
            model = QuadraticModel(nx+nu; actiondims=1:nu)
            splitter = InnovationSplitter(nu+1:nu+nx) |> VolumeWrapper |> VisitedWrapper
            g = Grid(domain, model, splitter)
            # X,U,Y = [],[],[]
            for i = 1:10000
                if i % 100 == 0
                    KalmanTree.find_and_apply_split(g, splitter)
                    # @show countnodes(g)
                end
                x = 2 .*rand(nx) .-1
                u = 2 .*rand(nu) .-1
                y = f(x,u) + 0.1*(sum(x)+sum(u))*randn()
                # push!(X,x)
                # push!(U,u)
                # push!(Y,y)
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
end
##

# E = @benchmark begin
# x,u = (X[rand(1:length(X))],U[rand(1:length(X))])
# n = walk_down(g,x,u)
# um = argmax_u(n, x)
# argmax_u(n, x)
# end
# display(E)
# plot(u->predict(n.model, x, u), -2,2, title="Q(a)", legend=false)
# vline!([um])
# vline!([n.domain[n.model.actiondims][]...], l=(:dash,:black))
# pf = (u1,u2)->predict(n.model, x, [u1;u2])
# surface(xu...,pf, title="Q(a)", legend=false)
# scatter3d!(um[1:1],um[2:2], [pf(um...)], m=(10,:cyan))
