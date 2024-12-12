@testset "agreement/confusion matrix tests" begin
    hard_label_pairs = zip([1, 1, 3, 1, 3, 1, 2, 1], [2, 2, 1, 1, 3, 2, 3, 1])
    c = confusion_matrix(3, hard_label_pairs)
    @test c == [2 3 0
                0 0 1
                1 0 1]
    kappa, percent_agreement = cohens_kappa(3, hard_label_pairs)
    chance = Lighthouse._probability_of_chance_agreement(3, hard_label_pairs)
    @test chance == (5 * 3 + 1 * 3 + 2 * 2) / 8^2
    @test accuracy(c) == percent_agreement == 3 / 8
    @test kappa == (3 / 8 - chance) / (1 - chance)
    stats = binary_statistics(c, 3)
    total = sum(c)
    @test total == 8
    @test stats.predicted_positives == 2
    @test stats.predicted_negatives == 6
    @test stats.actual_positives == 2
    @test stats.actual_negatives == 6
    @test stats.true_positives == 1
    @test stats.true_negatives == 5
    @test stats.false_positives == 1
    @test stats.false_negatives == 1
    @test stats.true_positive_rate == 0.5
    @test stats.true_negative_rate == 5 / 6
    @test stats.false_positive_rate == 1 / 6
    @test stats.false_negative_rate == 0.5
    @test stats.precision == 0.5
    @test stats.f1 == 0.5
    @test stats.true_positives + stats.true_negatives + stats.false_positives +
          stats.false_negatives == total
    @test stats.actual_positives + stats.actual_negatives == total
    @test stats.predicted_positives + stats.predicted_negatives == total

    labels = rand(StableRNG(42), 1:3, 100)
    hard_label_pairs = zip(labels, labels)
    c = confusion_matrix(3, hard_label_pairs)
    @test tr(c) == length(labels) == sum(c)
    kappa, percent_agreement = cohens_kappa(3, hard_label_pairs)
    @test accuracy(c) == percent_agreement == 1
    @test kappa == 1
    for i in 1:3
        stats = binary_statistics(c, i)
        @test stats.predicted_positives == stats.true_positives == stats.actual_positives
        @test stats.predicted_negatives == stats.true_negatives == stats.actual_negatives
        @test stats.false_positives == stats.false_negatives == 0
        @test stats.true_positive_rate == stats.true_negative_rate == 1
        @test stats.false_positive_rate == stats.false_negative_rate == 0
        @test stats.precision == 1
        @test stats.f1 == 1
    end

    n, k = 1_000_000, 2
    rng = StableRNG(42)
    hard_label_pairs = zip(rand(rng, 1:k, n), rand(rng, 1:k, n))
    c = confusion_matrix(k, hard_label_pairs)
    kappa, percent_agreement = cohens_kappa(3, hard_label_pairs)
    @test percent_agreement == accuracy(c)
    @test isapprox(percent_agreement, 0.5; atol=0.02)
    @test isapprox(kappa, 0.0; atol=0.02)
    stats = binary_statistics(c, 1)
    @test isapprox(stats.predicted_positives, 500_000; atol=2000)
    @test isapprox(stats.predicted_negatives, 500_000; atol=2000)
    @test isapprox(stats.actual_positives, 500_000; atol=2000)
    @test isapprox(stats.actual_negatives, 500_000; atol=2000)
    @test isapprox(stats.true_positives, 250_000; atol=2000)
    @test isapprox(stats.true_negatives, 250_000; atol=2000)
    @test isapprox(stats.false_positives, 250_000; atol=2000)
    @test isapprox(stats.false_negatives, 250_000; atol=2000)
    @test isapprox(stats.true_positive_rate, 0.5; atol=0.02)
    @test isapprox(stats.true_negative_rate, 0.5; atol=0.02)
    @test isapprox(stats.false_positive_rate, 0.5; atol=0.02)
    @test isapprox(stats.false_negative_rate, 0.5; atol=0.02)
    @test isapprox(stats.precision, 0.5; atol=0.02)
    @test isapprox(stats.f1, 0.5; atol=0.02)

    @test confusion_matrix(10, ()) == zeros(10, 10)
    @test all(isnan, cohens_kappa(10, ()))
    @test isnan(accuracy(zeros(10, 10)))
    stats = binary_statistics(zeros(10, 10), 1)
    @test stats.predicted_positives == 0
    @test stats.predicted_negatives == 0
    @test stats.actual_positives == 0
    @test stats.actual_negatives == 0
    @test stats.true_positives == 0
    @test stats.true_negatives == 0
    @test stats.false_positives == 0
    @test stats.false_negatives == 0
    @test isnan(stats.true_positive_rate)
    @test isnan(stats.true_negative_rate)
    @test isnan(stats.false_positive_rate)
    @test isnan(stats.false_negative_rate)
    @test isnan(stats.precision)
    @test isnan(stats.f1)

    c = [0 0
         0 8]
    stats = binary_statistics(c, 1)
    @test stats.true_positives == 0
    @test stats.true_negatives == 8
    @test stats.false_positives == 0
    @test stats.false_negatives == 0
    @test isnan(stats.f1)
    @test isnan(stats.true_positive_rate)
    @test isnan(stats.false_negative_rate)

    c = [0 2
         0 6]
    stats = binary_statistics(c, 1)
    @test stats.true_positives == 0
    @test stats.true_negatives == 6
    @test stats.false_positives == 2
    @test stats.false_negatives == 0
    @test stats.f1 == 0
    @test isnan(stats.true_positive_rate)
    @test isnan(stats.false_negative_rate)

    c = [0 0
         2 6]
    stats = binary_statistics(c, 1)
    @test stats.true_positives == 0
    @test stats.true_negatives == 6
    @test stats.false_positives == 0
    @test stats.false_negatives == 2
    @test stats.f1 == 0
    for p in 0:0.1:1
        @test Lighthouse._cohens_kappa(p, p) == 0
        if p > 0
            @test Lighthouse._cohens_kappa(p / 2, p) < 0
            @test Lighthouse._cohens_kappa(p, p / 2) > 0
        end
    end

    @test_throws ArgumentError cohens_kappa(3, [(4, 5), (8, 2)])
end

@testset "`calibration_curve`" begin
    rng = StableRNG(42)
    probs = rand(rng, 1_000_000)
    bitmask = rand(rng, Bool, 1_000_000)
    bin_count = 12
    bins, fractions, totals, mean_squared_error = calibration_curve(probs, bitmask;
                                                                    bin_count=bin_count)
    @test bin_count == length(bins)
    @test first(first(bins)) == 0.0 && last(last(bins)) == 1.0
    @test all(!ismissing, fractions)
    @test all(!isnan, fractions)
    @test all(!iszero, totals)
    @test all(isapprox.(fractions, 0.5; atol=0.02))
    @test all(isapprox.(totals, length(probs) / bin_count; atol=1000))
    @test sum(totals) == length(probs)
    @test isapprox(ceil(mean(fractions) * length(bitmask)), count(bitmask); atol=1)
    @test isapprox(mean_squared_error, inv(bin_count); atol=0.002)

    rng = StableRNG(42)
    probs = range(0.0, 1.0; length=1_000_000)
    bitmask = [rand(rng) <= p for p in probs]
    bin_count = 10
    bins, fractions, totals, mean_squared_error = calibration_curve(probs, bitmask;
                                                                    bin_count=bin_count)
    ideal = range(mean(first(bins)), mean(last(bins)); length=bin_count)
    @test bin_count == length(bins)
    @test first(first(bins)) == 0.0 && last(last(bins)) == 1.0
    @test all(!ismissing, fractions)
    @test all(!isnan, fractions)
    @test all(!iszero, totals)
    @test all(isapprox.(fractions, ideal; atol=0.01))
    @test all(totals .== 1_000_000 / bin_count)
    @test isapprox(ceil(mean(fractions) * length(bitmask)), count(bitmask); atol=1)
    @test isapprox(mean_squared_error, 0.0; atol=0.00001)

    bitmask = reverse(bitmask)
    bins, fractions, totals, mean_squared_error = calibration_curve(probs, bitmask;
                                                                    bin_count=bin_count)
    @test bin_count == length(bins)
    @test first(first(bins)) == 0.0 && last(last(bins)) == 1.0
    @test all(!ismissing, fractions)
    @test all(!isnan, fractions)
    @test all(!iszero, totals)
    @test all(isapprox.(fractions, reverse(ideal); atol=0.01))
    @test all(totals .== 1_000_000 / bin_count)
    @test isapprox(ceil(mean(fractions) * length(bitmask)), count(bitmask); atol=1)
    @test isapprox(mean_squared_error, 1 / 3; atol=0.01)

    # Handle garbage input---ensure non-existant results are NaN
    probs = fill(-1, 40)
    bitmask = zeros(Bool, 40)
    bins, fractions, totals, mean_squared_error = calibration_curve(probs, bitmask;
                                                                    bin_count)
    @test bin_count == length(bins)
    @test first(first(bins)) == 0.0 && last(last(bins)) == 1.0
    @test all(isnan, fractions)
    @test all(iszero, totals)
    @test isnan(mean_squared_error)
end

@testset "`calibration_curve`" begin
    @test binarize_by_threshold(0.2, 0.8) == false
    @test binarize_by_threshold(0.2, 0.2) == true
    @test binarize_by_threshold(0.3, 0.2) == true
    @test binarize_by_threshold.([0, 0, 0], 0.2) == [0, 0, 0]
end

@testset "Metrics hardening/binarization" begin
    predicted_soft_labels = [0.51 0.49
                             0.49 0.51
                             0.1 0.9
                             0.9 0.1
                             0.0 1.0]
    elected_hard_labels = [1, 2, 2, 2, 1]
    thresholds = [0.25, 0.5, 0.75]
    i_class = 2
    class_labels = ["a", "b"]
    default_metrics = get_tradeoff_metrics(predicted_soft_labels,
                                           elected_hard_labels,
                                           i_class; thresholds, class_labels)

    # Use bogus threshold/hardening function to prove that hardening function is
    # used internally
    scaled_binarize_by_threshold = (soft, threshold) -> soft >= threshold / 10
    scaled_thresholds = 10 .* thresholds
    scaled_metrics = get_tradeoff_metrics(predicted_soft_labels,
                                          elected_hard_labels,
                                          i_class; thresholds=scaled_thresholds,
                                          binarize=scaled_binarize_by_threshold,
                                          class_labels)
    @test isequal(default_metrics, scaled_metrics)

    # Discrim calibration
    votes = [1 1 1
             0 2 2
             1 2 2
             1 1 2
             0 1 1]
    cal = Lighthouse._calculate_optimal_threshold_from_discrimination_calibration(predicted_soft_labels,
                                                                                  votes;
                                                                                  thresholds,
                                                                                  class_of_interest_index=i_class)

    scaled_cal = Lighthouse._calculate_optimal_threshold_from_discrimination_calibration(predicted_soft_labels,
                                                                                         votes;
                                                                                         class_of_interest_index=i_class,
                                                                                         thresholds=scaled_thresholds,
                                                                                         binarize=scaled_binarize_by_threshold)
    for k in keys(cal)
        if k == :threshold
            @test cal[k] * 10 == scaled_cal[k] # Should be the same _relative_ threshold
        else
            @test isequal(cal[k], scaled_cal[k])
        end
    end

    conf = Lighthouse.per_class_confusion_statistics(predicted_soft_labels,
                                                     elected_hard_labels, thresholds)
    scaled_conf = Lighthouse.per_class_confusion_statistics(predicted_soft_labels,
                                                            elected_hard_labels,
                                                            scaled_thresholds;
                                                            binarize=scaled_binarize_by_threshold)
    @test isequal(conf, scaled_conf)
end
