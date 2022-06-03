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

@testset "`log_array` and `log_arrays!`" begin
    mktempdir() do logdir
        logger = LearnLogger(logdir, "test_run")
        log_array!(logger, "arr", [1.0, 2.0, 3.0])
        @test logger.logged["arr"] == [2.0] # defaults to the mean
        log_arrays!(logger, Dict("arr2" => [1.0, 2.0, 3.0], "arr3" => [1.0]))
        @test logger.logged["arr2"] == [2.0]
        @test logger.logged["arr3"] == [1.0]
    end
end


@testset "`log_foo_row!`" begin
    predicted_soft_labels = [0.51 0.49
                             0.49 0.51
                             0.1 0.9
                             0.9 0.1
                             0.0 1.0]
    elected_hard_labels = [1, 2, 2, 2, 1]
    predicted_hard_labels = [1,2,2,1,2]
    thresholds = [0.25, 0.5, 0.75]
    class_index = 2
    class_labels = ["a", "b"]
    tradeoff_metrics = get_tradeoff_metrics(predicted_soft_labels,
                                           elected_hard_labels,
                                           class_index; thresholds, class_labels)
    hardened_metrics = get_hardened_metrics(predicted_hard_labels, elected_hard_labels, class_index;
                              class_labels)
    k, n = 2, 3
    rng = StableRNG(22)
    votes = [rand(rng, possible_vote_labels) for sample in 1:(n * 10), voter in 1:7]
    votes[:, [1, 2, 3]] .= votes[:, 4] # Voter 1-3 voted identically to voter 4 (force non-zero agreement)
    label_metrics =get_label_metrics_multirater(votes, class_index; class_labels)

    mktempdir() do logdir
        logger = LearnLogger(logdir, "test_run")
        log_tradeoff_metrics(logger, "foo-", tradeoff_metrics)
        log_hardened_metrics(logger, "bar-", hardened_metrics)
        log_label_metrics(logger, "baz-", label_metrics)
        @test haskey(logger.logged,"baz-ira_kappa")
        @test haskey(logger.logged,"bar-ea_kappa")
        @test haskey(logger.logged,"foo-roc_auc")
    end


end
