mutable struct TestClassifier <: AbstractClassifier
    dummy_loss::Float64
    classes::Vector
end

Lighthouse.classes(c::TestClassifier) = c.classes

function Lighthouse.train!(c::TestClassifier, dummy_batches, logger)
    for (dummy_input_batch, loss_delta) in dummy_batches
        c.dummy_loss += loss_delta
        Lighthouse.log_value!(logger, "train/loss_per_batch", c.dummy_loss)
    end
    return c.dummy_loss
end

const RNG_LOSS = StableRNG(22)

function Lighthouse.loss_and_prediction(c::TestClassifier, dummy_input_batch)
    dummy_soft_label_batch = rand(RNG_LOSS, length(c.classes), size(dummy_input_batch)[end])
    # Fake a softmax
    dummy_soft_label_batch .= dummy_soft_label_batch ./ sum(dummy_soft_label_batch; dims=1)
    return c.dummy_loss, dummy_soft_label_batch
end

@testset "Multi-class learn!(::TestModel, ...)" begin
    mktempdir() do tmpdir
        model = TestClassifier(1000000.0, ["class_$i" for i in 1:5])
        k, n = length(model.classes), 3
        rng = StableRNG(22)
        train_batches = [(rand(rng, 4 * k, n), -rand(rng)) for _ in 1:100]
        test_batches = [((rand(rng, 4 * k, n),), (n * i - n + 1):(n * i)) for i in 1:10]
        possible_vote_labels = collect(0:k)
        votes = [rand(rng, possible_vote_labels) for sample in 1:(n * 10), voter in 1:7]
        votes[:, [1, 2, 3]] .= votes[:, 4] # Voter 1-3 voted identically to voter 4 (force non-zero agreement)
        logger = LearnLogger(joinpath(tmpdir, "logs"), "test_run")
        limit = 5
        let counted = 0
            upon_loss_decrease = Lighthouse.upon(logger,
                                                 "test_set_prediction/mean_loss_per_epoch";
                                                 condition=<, initial=Inf)
            callback = n -> begin
                upon_loss_decrease() do _
                    counted += n
                    @debug counted n
                end
            end
            elected = majority.((rng,), eachrow(votes),
                                (1:length(Lighthouse.classes(model)),))
            Lighthouse.learn!(model, logger, () -> train_batches, () -> test_batches, votes,
                              elected; epoch_limit=limit, post_epoch_callback=callback)
            @test counted == sum(1:limit)
        end
        @test length(logger.logged["train/loss_per_batch"]) == length(train_batches) * limit
        for key in ["test_set_prediction/loss_per_batch",
                    "test_set_prediction/time_in_seconds_per_batch",
                    "test_set_prediction/gc_time_in_seconds_per_batch",
                    "test_set_prediction/allocations_per_batch",
                    "test_set_prediction/memory_in_mb_per_batch"]
            @test length(logger.logged[key]) == length(test_batches) * limit
        end
        for key in ["test_set_prediction/mean_loss_per_epoch",
                    "test_set_evaluation/time_in_seconds_per_epoch",
                    "test_set_evaluation/gc_time_in_seconds_per_epoch",
                    "test_set_evaluation/allocations_per_epoch",
                    "test_set_evaluation/memory_in_mb_per_epoch"]
            @test length(logger.logged[key]) == limit
        end
        @test length(logger.logged["test_set_evaluation/metrics_per_epoch"]) == limit

        # Test multiclass optimal_threshold param invalid
        @test_throws ArgumentError Lighthouse.learn!(model, logger, () -> train_batches,
                                                     () -> test_batches, votes;
                                                     epoch_limit=limit,
                                                     optimal_threshold_class=1)

        # Test `predict!`
        num_samples = sum(b -> size(b[1][1], 2), test_batches)
        predicted_soft = zeros(num_samples, length(model.classes))
        predict!(model, predicted_soft, test_batches, logger; logger_prefix="halloooo")
        @test !all(predicted_soft .== 0)
        @test length(logger.logged["halloooo/mean_loss_per_epoch"]) == 1
        @test length(logger.logged["halloooo/loss_per_batch"]) == length(test_batches)
        @test length(logger.logged["halloooo/time_in_seconds_per_batch"]) ==
              length(test_batches)

        # Test `evaluate!`
        n_examples = 40
        num_voters = 20
        predicted_soft = rand(rng, Float32, n_examples, length(model.classes))
        predicted_hard = map(label -> Lighthouse.onecold(model, label),
                             eachrow(predicted_soft))
        votes = [rand(rng, possible_vote_labels)
                 for sample in 1:n_examples, voter in 1:num_voters]
        votes[:, 3] .= votes[:, 4] # Voter 4 voted identically to voter 3 (force non-zero agreement)
        elected_hard = map(row -> majority(rng, row, 1:length(model.classes)),
                           eachrow(votes))
        evaluate!(predicted_hard, predicted_soft, elected_hard, model.classes, logger;
                  logger_prefix="wheeeeeee", logger_suffix="_for_all_time", votes)
        @test length(logger.logged["wheeeeeee/time_in_seconds_for_all_time"]) == 1
        @test length(logger.logged["wheeeeeee/metrics_for_all_time"]) == 1

        # Test plotting with no votes directly with eval row
        eval_row = Lighthouse.evaluation_metrics_record(predicted_hard, predicted_soft,
                                                        elected_hard, model.classes;
                                                        votes=nothing)
        all_together_no_ira = evaluation_metrics_plot(eval_row)
        @testplot all_together_no_ira

        # Round-trip `onehot` for codecov
        onehot_hard = map(h -> vec(Lighthouse.onehot(model, h)), predicted_hard)
        @test map(h -> findfirst(h), onehot_hard) == predicted_hard

        # Test startified eval
        strata = [Set("group $(j % Int(ceil(sqrt(j))))" for j in 1:(i - 1))
                  for i in 1:size(votes, 1)]
        plot_data = evaluation_metrics(predicted_hard, predicted_soft, elected_hard,
                                       model.classes, 0.0:0.01:1.0; votes, strata)
        @test haskey(plot_data, "stratified_kappas")
        plot = evaluation_metrics_plot(plot_data)

        test_evaluation_metrics_roundtrip(plot_data)

        plot2, plot_data2 = @test_deprecated evaluation_metrics_plot(predicted_hard,
                                                                     predicted_soft,
                                                                     elected_hard,
                                                                     model.classes,
                                                                     0.0:0.01:1.0;
                                                                     votes=votes,
                                                                     strata=strata)
        @test isequal(plot_data, plot_data2) # check these are the same
        test_evaluation_metrics_roundtrip(plot_data2)

        # Test plotting
        plot_data = last(logger.logged["test_set_evaluation/metrics_per_epoch"])
        @test isa(plot_data["thresholds"], AbstractVector)

        @test isa(last(plot_data["per_class_pr_curves"]),
                  Tuple{Vector{Float64},Vector{Float64}})
        pr = plot_pr_curves(plot_data["per_class_pr_curves"], plot_data["class_labels"])
        @testplot pr

        roc = plot_roc_curves(plot_data["per_class_roc_curves"],
                              plot_data["per_class_roc_aucs"], plot_data["class_labels"])
        @testplot roc

        # Kappa no IRA
        kappas_no_ira = plot_kappas(vcat(plot_data["multiclass_kappa"],
                                         plot_data["per_class_kappas"]),
                                    vcat("Multiclass", plot_data["class_labels"]))
        @testplot kappas_no_ira

        # Kappa with IRA
        kappas_ira = plot_kappas(vcat(plot_data["multiclass_kappa"],
                                      plot_data["per_class_kappas"]),
                                 vcat("Multiclass", plot_data["class_labels"]),
                                 vcat(plot_data["multiclass_IRA_kappas"],
                                      plot_data["per_class_IRA_kappas"]))
        @testplot kappas_ira

        reliability_calibration = plot_reliability_calibration_curves(plot_data["per_class_reliability_calibration_curves"],
                                                                      plot_data["per_class_reliability_calibration_scores"],
                                                                      plot_data["class_labels"])
        @testplot reliability_calibration

        confusion_row = plot_confusion_matrix(plot_data["confusion_matrix"],
                                              plot_data["class_labels"], :Row)
        @testplot confusion_row

        confusion_col = plot_confusion_matrix(plot_data["confusion_matrix"],
                                              plot_data["class_labels"], :Column)
        @testplot confusion_col

        confusion_basic = plot_confusion_matrix(plot_data["confusion_matrix"],
                                                plot_data["class_labels"])
        @testplot confusion_basic

        @test_throws ArgumentError plot_confusion_matrix(plot_data["confusion_matrix"],
                                                         plot_data["class_labels"], :norm)

        all_together_2 = evaluation_metrics_plot(plot_data)
        @testplot all_together_2

        all_together_3 = evaluation_metrics_plot(EvaluationV1(plot_data))
        @testplot all_together_3

        #savefig(all_together_2, "/tmp/multiclass.png")
    end
end

@testset "2-class `learn!(::TestModel, ...)`" begin
    mktempdir() do tmpdir
        model = TestClassifier(1000000.0, ["class_$i" for i in 1:2])
        k, n = length(model.classes), 3
        rng = StableRNG(23)
        train_batches = [(rand(rng, 4 * k, n), -rand(rng)) for _ in 1:100]
        test_batches = [((rand(rng, 4 * k, n),), (n * i - n + 1):(n * i)) for i in 1:10]
        possible_vote_labels = collect(0:k)
        votes = [rand(rng, possible_vote_labels) for sample in 1:(n * 10), voter in 1:7]
        votes[:, [1, 2, 3]] .= votes[:, 4] # Voter 1-3 voted identically to voter 4 (force non-zero agreement)
        logger = LearnLogger(joinpath(tmpdir, "logs"), "test_run")
        limit = 5
        let counted = 0
            upon_loss_decrease = Lighthouse.upon(logger,
                                                 "test_set_prediction/mean_loss_per_epoch";
                                                 condition=<, initial=Inf)
            callback = n -> begin
                upon_loss_decrease() do _
                    counted += n
                    @debug counted n
                end
            end
            elected = majority.((rng,), eachrow(votes),
                                (1:length(Lighthouse.classes(model)),))
            Lighthouse.learn!(model, logger, () -> train_batches, () -> test_batches, votes,
                              elected; epoch_limit=limit, post_epoch_callback=callback)
            @test counted == sum(1:limit)
        end
        # Binary classification logs some additional metrics
        @test length(logger.logged["test_set_evaluation/spearman_correlation_per_epoch"]) ==
              limit
        plot_data = last(logger.logged["test_set_evaluation/metrics_per_epoch"])
        @test haskey(plot_data, "spearman_correlation")
        test_evaluation_metrics_roundtrip(plot_data)

        # No `optimal_threshold_class` during learning...
        @test !haskey(plot_data, "optimal_threshold")
        @test !haskey(plot_data, "optimal_threshold_class")

        # And now, `optimal_threshold_class` during learning
        elected = majority.((rng,), eachrow(votes), (1:length(Lighthouse.classes(model)),))
        Lighthouse.learn!(model, logger, () -> train_batches, () -> test_batches, votes,
                          elected; epoch_limit=limit, optimal_threshold_class=2,
                          test_set_logger_prefix="validation_set")
        plot_data = last(logger.logged["validation_set_evaluation/metrics_per_epoch"])
        @test haskey(plot_data, "optimal_threshold")
        @test haskey(plot_data, "optimal_threshold_class")
        @test plot_data["optimal_threshold_class"] == 2
        test_evaluation_metrics_roundtrip(plot_data)

        # `optimal_threshold_class` param invalid
        @test_throws ArgumentError Lighthouse.learn!(model, logger, () -> train_batches,
                                                     () -> test_batches, votes;
                                                     epoch_limit=limit,
                                                     optimal_threshold_class=3)

        # Test `evaluate!` for votes, no votes
        n_examples = 40
        num_voters = 20
        predicted_soft = rand(rng, Float32, n_examples, length(model.classes))
        predicted_soft .= predicted_soft ./ sum(predicted_soft; dims=2)
        predicted_hard = map(label -> Lighthouse.onecold(model, label),
                             eachrow(predicted_soft))
        votes = [rand(rng, possible_vote_labels)
                 for sample in 1:n_examples, voter in 1:num_voters]
        votes[:, 3] .= votes[:, 4] # Voter 4 voted identically to voter 3 (force non-zero agreement)
        elected_hard = map(row -> majority(rng, row, 1:length(model.classes)),
                           eachrow(votes))

        evaluate!(predicted_hard, predicted_soft, elected_hard, model.classes, logger;
                  logger_prefix="wheeeeeee", logger_suffix="_for_all_time", votes=nothing)
        plot_data = last(logger.logged["wheeeeeee/metrics_for_all_time"])
        @test !haskey(plot_data, "per_class_IRA_kappas")
        @test !haskey(plot_data, "multiclass_IRA_kappas")
        test_evaluation_metrics_roundtrip(plot_data)

        evaluate!(predicted_hard, predicted_soft, elected_hard, model.classes, logger;
                  logger_prefix="wheeeeeee", logger_suffix="_for_all_time", votes=votes)
        plot_data = last(logger.logged["wheeeeeee/metrics_for_all_time"])
        @test haskey(plot_data, "per_class_IRA_kappas")
        @test haskey(plot_data, "multiclass_IRA_kappas")
        test_evaluation_metrics_roundtrip(plot_data)

        # Test `evaluate` for different optimal_threshold classes
        evaluate!(predicted_hard, predicted_soft, elected_hard, model.classes, logger;
                  logger_prefix="wheeeeeee", logger_suffix="_for_all_time", votes=votes,
                  optimal_threshold_class=1)
        plot_data_1 = last(logger.logged["wheeeeeee/metrics_for_all_time"])
        evaluate!(predicted_hard, predicted_soft, elected_hard, model.classes, logger;
                  logger_prefix="wheeeeeee", logger_suffix="_for_all_time", votes=votes,
                  optimal_threshold_class=2)
        plot_data_2 = last(logger.logged["wheeeeeee/metrics_for_all_time"])
        test_evaluation_metrics_roundtrip(plot_data_2)

        # The thresholds should not be identical (since they are *inclusive* when applied:
        # values greater than _or equal to_ the threshold are given the class value)
        @test plot_data_1["optimal_threshold"] != plot_data_2["optimal_threshold"]

        # The two threshold options yield different results
        thresh_from_roc = Lighthouse._get_optimal_threshold_from_ROC(plot_data_2["per_class_roc_curves"];
                                                                     thresholds=plot_data_2["thresholds"],
                                                                     class_of_interest_index=plot_data_2["optimal_threshold_class"])
        thresh_from_calibration = Lighthouse._calculate_optimal_threshold_from_discrimination_calibration(predicted_soft,
                                                                                                          votes;
                                                                                                          thresholds=plot_data_2["thresholds"],
                                                                                                          class_of_interest_index=plot_data_2["optimal_threshold_class"]).threshold
        @test !isequal(thresh_from_roc, thresh_from_calibration)
        @test isequal(thresh_from_calibration, plot_data_2["optimal_threshold"])

        # Also, let's make sure we get an isolated discrimination plots
        discrimination_cal = Lighthouse.plot_binary_discrimination_calibration_curves(plot_data_1["discrimination_calibration_curve"],
                                                                                      plot_data_1["discrimination_calibration_score"],
                                                                                      plot_data_1["per_expert_discrimination_calibration_curves"],
                                                                                      plot_data_1["per_expert_discrimination_calibration_scores"],
                                                                                      plot_data_1["optimal_threshold"],
                                                                                      plot_data_1["class_labels"][plot_data_1["optimal_threshold_class"]])
        @testplot discrimination_cal

        discrimination_cal_no_experts = Lighthouse.plot_binary_discrimination_calibration_curves(plot_data_1["discrimination_calibration_curve"],
                                                                                                 plot_data_1["discrimination_calibration_score"],
                                                                                                 missing,
                                                                                                 missing,
                                                                                                 plot_data_1["optimal_threshold"],
                                                                                                 plot_data_1["class_labels"][plot_data_1["optimal_threshold_class"]])

        # Test binary discrimination with no multiclass votes
        plot_data_1["per_expert_discrimination_calibration_curves"] = missing
        no_expert_calibration = evaluation_metrics_plot(EvaluationV1(plot_data_1))
        @testplot no_expert_calibration

        # Test that plotting succeeds (no specialization relative to the multi-class tests)
        plot_data = last(logger.logged["validation_set_evaluation/metrics_per_epoch"])
        all_together = evaluation_metrics_plot(plot_data)
        #savefig(all_together, "/tmp/binary.png")
        @testplot all_together

        @testplot discrimination_cal_no_experts
    end
end

@testset "Invalid `_calculate_ira_kappas`" begin
    classes = ["roy", "gee", "biv"]
    @test isequal(Lighthouse._calculate_ira_kappas([1; 1; 1; 1], classes),
                  (; per_class_IRA_kappas=missing, multiclass_IRA_kappas=missing))  # Only one voter...
    @test isequal(Lighthouse._calculate_ira_kappas([1 0; 1 0; 0 1], classes),
                  (; per_class_IRA_kappas=missing, multiclass_IRA_kappas=missing))  # No observations in common...
end

@testset "Calculate `_spearman_corr`" begin
    # Test failures for...
    # ...nonbinary input
    @test_throws ArgumentError Lighthouse._calculate_spearman_correlation([0.9 0.1], [1],
                                                                          ["oh" "em" "gee"])
    # ...non-softmax input
    @test_throws ArgumentError Lighthouse._calculate_spearman_correlation([0.9 0.9], [1],
                                                                          ["oh" "em"])

    # Test class order doesn't matter
    predicted = [0.1 0.9; 0.3 0.7; 1 0]
    votes = [0 1 1 2; 2 2 0 0; 0 2 2 1]
    predicted_flipped = [0.9 0.1; 0.7 0.3; 0 1]
    votes_flipped = [0 2 2 1; 1 1 0 0; 0 1 1 2]
    @test Lighthouse._calculate_spearman_correlation(predicted, votes, ["oh" "em"]) ==
          Lighthouse._calculate_spearman_correlation(predicted_flipped, votes_flipped,
                                                     ["em" "oh"])

    # Test complete agreement
    predicted_soft = [0.0 1.0; 1.0 0.0; 1.0 0.0]
    votes = [2; 1; 1]
    sp = Lighthouse._calculate_spearman_correlation(predicted_soft, votes, ["oh" "em"])
    @test sp.ρ == 1

    # Test complete disagreement
    votes = [1; 2; 2]
    sp = Lighthouse._calculate_spearman_correlation(predicted_soft, votes, ["oh" "em"])
    @test sp.ρ == -1

    # Test NaN spearman due to unranked input
    votes = [1; 2; 2]
    predicted_soft = [0.3 0.7
                      0.3 0.7
                      0.3 0.7]
    sp = Lighthouse._calculate_spearman_correlation(predicted_soft, votes, ["oh" "em"])
    @test isnan(sp.ρ)

    # Test CI
    sp = Lighthouse._spearman_corr([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
                                   [0.1, 0.3, 0.2, 0.1, 0.3, 0.1, 0.6])
    @test -1.0 < sp.ci_lower < sp.ρ < sp.ci_upper < 1.0
end

@testset "Discrimination calibration" begin
    # Test single voter has same calibration as "whole" by definition
    votes = [1; 1; 1; 2; 2; 2]
    @test length(Lighthouse._elected_probabilities(votes, 1)) == 6
    @test Lighthouse._elected_probabilities(votes, 1) == [1; 1; 1; 0; 0; 0]

    # Test single voter discrimination calibration
    single_voter_calibration = Lighthouse._calculate_voter_discrimination_calibration(votes;
                                                                                      class_of_interest_index=1)
    @test length(single_voter_calibration.mse) == 1

    # Test multi-voter voter discrimination calibration
    votes = [0 1 1 1
             1 2 0 0
             2 1 2 2] # Note: voters 3 and 4 have voted identically
    voter_calibration = Lighthouse._calculate_voter_discrimination_calibration(votes;
                                                                               class_of_interest_index=1)
    @test length(voter_calibration.mse) == size(votes, 2)
    @test length(voter_calibration.plot_curve_data) == size(votes, 2)
    @test voter_calibration.mse[1] > voter_calibration.mse[3]
    @test !isequal(voter_calibration.plot_curve_data[1],
                   voter_calibration.plot_curve_data[2])
    @test isequal(voter_calibration.plot_curve_data[3],
                  voter_calibration.plot_curve_data[4])

    # Test predicted voter discrimination calibration
    predicted_soft_labels = [1.0 0.0; 0.0 1.0; 0.0 1.0]
    cal = Lighthouse._calculate_optimal_threshold_from_discrimination_calibration(predicted_soft_labels,
                                                                                  votes;
                                                                                  thresholds=0.0:0.01:1.0,
                                                                                  class_of_interest_index=1)
    @test all(cal.mse .<= voter_calibration.mse)
    cal2 = Lighthouse._calculate_optimal_threshold_from_discrimination_calibration(predicted_soft_labels,
                                                                                   votes;
                                                                                   thresholds=0.0:0.01:1.0,
                                                                                   class_of_interest_index=2)
    @test cal.mse == cal2.mse
    @test cal.plot_curve_data[2] != cal2.plot_curve_data[2]
end

@testset "2-class per_class_confusion_statistics" begin
    predicted_soft_labels = [0.51 0.49
                             0.49 0.51
                             0.1 0.9
                             0.9 0.1
                             0.0 1.0]
    elected_hard_labels = [1, 2, 2, 2, 1]
    thresholds = [0.25, 0.5, 0.75]
    class_1, class_2 = Lighthouse.per_class_confusion_statistics(predicted_soft_labels,
                                                                 elected_hard_labels,
                                                                 thresholds)
    stats_1, stats_2 = class_1[1], class_2[1] # threshold == 0.25
    @test stats_1.actual_negatives == stats_2.actual_positives == 3
    @test stats_1.actual_positives == stats_2.actual_negatives == 2
    @test stats_1.predicted_positives == 3
    @test stats_2.predicted_positives == 4
    @test stats_1.predicted_negatives == 2
    @test stats_2.predicted_negatives == 1
    @test stats_1.true_positives == 1
    @test stats_2.true_positives == 2
    @test stats_1.true_negatives == 1
    @test stats_2.true_negatives == 0
    @test stats_1.false_positives == 2
    @test stats_2.false_positives == 2
    @test stats_1.false_negatives == 1
    @test stats_2.false_negatives == 1
    @test stats_1.true_positive_rate == 0.5
    @test stats_2.true_positive_rate == 2 / 3
    @test stats_1.true_negative_rate == 1 / 3
    @test stats_2.true_negative_rate == 0.0
    @test stats_1.false_positive_rate == 2 / 3
    @test stats_2.false_positive_rate == 1.0
    @test stats_1.false_negative_rate == 0.5
    @test stats_2.false_negative_rate == 1 / 3
    @test stats_1.precision == 1 / 3
    @test stats_2.precision == 0.5

    stats_1, stats_2 = class_1[2], class_2[2] # threshold == 0.5
    @test stats_1.actual_negatives == stats_2.actual_positives == 3
    @test stats_1.actual_positives == stats_2.actual_negatives == 2
    @test stats_1.predicted_negatives == stats_2.predicted_positives == 3
    @test stats_1.predicted_positives == stats_2.predicted_negatives == 2
    @test stats_1.true_negatives == stats_2.true_positives == 2
    @test stats_1.true_positives == stats_2.true_negatives == 1
    @test stats_1.false_positives == stats_2.false_negatives == 1
    @test stats_1.false_negatives == stats_2.false_positives == 1
    @test stats_1.true_positive_rate == stats_2.true_negative_rate == 0.5
    @test stats_1.true_negative_rate == stats_2.true_positive_rate == 2 / 3
    @test stats_1.false_positive_rate == stats_2.false_negative_rate == 1 / 3
    @test stats_1.false_negative_rate == stats_2.false_positive_rate == 0.5
    @test stats_1.precision == 0.5 && stats_2.precision == 2 / 3

    stats_1, stats_2 = class_1[3], class_2[3] # threshold == 0.75
    @test stats_1.actual_negatives == stats_2.actual_positives == 3
    @test stats_1.actual_positives == stats_2.actual_negatives == 2
    @test stats_1.predicted_positives == 1
    @test stats_2.predicted_positives == 2
    @test stats_1.predicted_negatives == 4
    @test stats_2.predicted_negatives == 3
    @test stats_1.true_positives == 0
    @test stats_2.true_positives == 1
    @test stats_1.true_negatives == 2
    @test stats_2.true_negatives == 1
    @test stats_1.false_positives == 1
    @test stats_2.false_positives == 1
    @test stats_1.false_negatives == 2
    @test stats_2.false_negatives == 2
    @test stats_1.true_positive_rate == 0.0
    @test stats_2.true_positive_rate == 1 / 3
    @test stats_1.true_negative_rate == 2 / 3
    @test stats_2.true_negative_rate == 0.5
    @test stats_1.false_positive_rate == 1 / 3
    @test stats_2.false_positive_rate == 0.5
    @test stats_1.false_negative_rate == 1.0
    @test stats_2.false_negative_rate == 2 / 3
    @test stats_1.precision == 0.0
    @test stats_2.precision == 0.5
end

@testset "3-class per_class_confusion_statistics" begin
    predicted_soft_labels = [1/3 1/3 1/3
                             0.1 0.7 0.2
                             0.25 0.25 0.5
                             0.4 0.5 0.1
                             0.0 0.0 1.0
                             0.2 0.5 0.3
                             0.5 0.4 0.1]
    elected_hard_labels = [1, 2, 2, 1, 3, 3, 1]
    # TODO would be more robust to have multiple thresholds, but our naive tests
    # here will have to be refactored to avoid becoming a nightmare if we do that
    thresholds = [0.5]
    class_1, class_2, class_3 = Lighthouse.per_class_confusion_statistics(predicted_soft_labels,
                                                                          elected_hard_labels,
                                                                          thresholds)
    stats_1, stats_2, stats_3 = class_1[], class_2[], class_3[] # threshold == 0.5
    @test stats_1.predicted_positives == 1
    @test stats_2.predicted_positives == 3
    @test stats_3.predicted_positives == 2
    @test stats_1.predicted_negatives == 6
    @test stats_2.predicted_negatives == 4
    @test stats_3.predicted_negatives == 5
    @test stats_1.actual_positives == 3
    @test stats_2.actual_positives == 2
    @test stats_3.actual_positives == 2
    @test stats_1.actual_negatives == 4
    @test stats_2.actual_negatives == 5
    @test stats_3.actual_negatives == 5
    @test stats_1.true_positives == 1
    @test stats_2.true_positives == 1
    @test stats_3.true_positives == 1
    @test stats_1.true_negatives == 4
    @test stats_2.true_negatives == 3
    @test stats_3.true_negatives == 4
    @test stats_1.false_positives == 0
    @test stats_2.false_positives == 2
    @test stats_3.false_positives == 1
    @test stats_1.false_negatives == 2
    @test stats_2.false_negatives == 1
    @test stats_3.false_negatives == 1
    @test stats_1.true_positive_rate == 1 / 3
    @test stats_2.true_positive_rate == 0.5
    @test stats_3.true_positive_rate == 0.5
    @test stats_1.true_negative_rate == 1.0
    @test stats_2.true_negative_rate == 0.6
    @test stats_3.true_negative_rate == 0.8
    @test stats_1.false_positive_rate == 0.0
    @test stats_2.false_positive_rate == 0.4
    @test stats_3.false_positive_rate == 0.2
    @test stats_1.false_negative_rate == 2 / 3
    @test stats_2.false_negative_rate == 0.5
    @test stats_3.false_negative_rate == 0.5
    @test stats_1.precision == 1.0
    @test stats_2.precision == 1 / 3
    @test stats_3.precision == 0.5
end
