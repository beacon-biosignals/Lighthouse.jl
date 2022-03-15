@testset "extract_common_entries" begin
    common_keys = ("class_labels",)
    consistent_results = [
        (; name="model1", data=Dict("class_labels" => ["hi", "bye"], "unrelated"=>1)),
        (; name="model2", data=Dict("class_labels" => ["hi", "bye"], "unrelated"=>2))
    ]
    @test Lighthouse.extract_common_entries(consistent_results, common_keys) == Dict("class_labels" => ["hi", "bye"])

    inconsistent_results = [
        (; name="model1", data=Dict("class_labels" => ["hi", "bye"], "unrelated"=>1)),
        (; name="model2", data=Dict("class_labels" => ["hi", "goodbye"], "unrelated"=>2))
    ]
    @test_throws ArgumentError Lighthouse.extract_common_entries(inconsistent_results, common_keys)
end

@testset "binary_comparison_metrics_plot" begin
    results_1 = train_binary_model(mktempdir(); rng = StableRNG(1))
    plot_data_1 = last(results_1.logger.logged["test_set_evaluation/metrics_per_epoch"])
    plot_data_1["optimal_threshold_class"] = 1
    m1 = (; name = "Model1", data = plot_data_1)

    results_2 = train_binary_model(mktempdir(); rng = StableRNG(2))
    plot_data_2 = last(results_2.logger.logged["test_set_evaluation/metrics_per_epoch"])
    plot_data_2["optimal_threshold_class"] = 1
    m2 = (; name = "Model2", data = plot_data_2)

    binary_comparison_metrics = Lighthouse.binary_comparison_metrics_plot((m1, m2))
    @testplot binary_comparison_metrics
end
