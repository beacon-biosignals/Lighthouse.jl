@testset "`Log forwarding`" begin
    mktempdir() do logdir
        vector = rand(10)
        data = Dict("string" => "asdf", "int" => 42, "float64" => 42.0, "float32" => 42.0f0,
                    "vector" => vector, "dict" => Dict("a" => 0, 42 => identity))
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
        for (k, v) in data
            if typeof(v) <: Union{Real,Complex}
                @test length(loaded[k]) == N
                @test first(loaded[k]) == v
            end
        end
    end
end

@testset "`Generic datastructure logging`" begin
    mktempdir() do logdir
        logger = LearnLogger(logdir, "test_run")
        @test isnothing(Lighthouse.log_line_series!(logger, "foo", 3, 2))

        @test isnothing(step_logger!(logger))
    end
end

@testset "log_values!" begin
    mktempdir() do logdir
        logger = LearnLogger(logdir, "test_run")
        log_values!(logger, Dict("a" => 1, "b" => 2))
        @test logger.logged["a"] == [1]
        @test logger.logged["b"] == [2]
    end
end

@testset "summarize_array" begin
    mktempdir() do logdir
        logger = LearnLogger(logdir, "test_run")
        x = summarize_array(logger, [1.0, 2.0, 3.0])
        @test x == 2
        log_value!(logger, "summary", x)
        @test logger.logged["summary"] == [x]
    end
end
