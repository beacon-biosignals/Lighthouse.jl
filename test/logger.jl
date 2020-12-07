@testset "`Log forwarding`" begin
    mktempdir() do logdir
        vector = rand(10)
        data = Dict(
            "string" => "asdf",
            "int" => 42,
            "float64" => 42.0,
            "float32" => 42.0f0,
            "vector" => vector,
            "dict" => Dict("a" => 0, 42 => identity))
        logger = LearnLogger(logdir, "test_run")
        channel = Channel()
        forwarding_task = Lighthouse.forward_logs(channel, logger)
        N = 5
        for _ in 1:N
            for (field, value) in data
                put!(channel, field => value)
            end
            put!(channel, "events" => "happens")
        end
        put!(channel, "plotted" => vector)
        close(channel)
        wait(forwarding_task)
        loaded = logger.logged
        for (k,v) in data
            if typeof(v) <: Union{Real,Complex}
                @test length(loaded[k]) == N
                @test first(loaded[k]) == v
            end
        end
    end
end

