# Make sure we cover all edge cases in the plotting code
using Makie.Colors: Gray
@testset "plotting" begin
    @testset "NaN color" begin
        confusion = [
            NaN 0;
            1.0 0.5
        ]
        nan_confusion = plot_confusion_matrix(confusion, ["test1", "test2"], :Row)
        @testplot nan_confusion
        nan_custom_confusion = with_theme(ConfusionMatrix = (Heatmap=(nan_color=:red,), Text=(color=:red,))) do
            plot_confusion_matrix(confusion, ["test1", "test2"], :Row)
        end
        @testplot nan_custom_confusion
    end

    @testset "Kappa placement" begin
        classes = ["class $i" for i in 1:5]
        kappa_text_placement = with_theme(Kappas = (Text=(color=Gray(0.5),),)) do
            plot_kappas((1:5) ./ 5 .- 0.1, classes, (1:5) ./ 5, color = [Gray(0.4), Gray(0.2)])
        end
        @testplot kappa_text_placement

        kappa_text_placement_single = with_theme(Kappas = (Text=(color=:red,),)) do
            plot_kappas((1:5) ./ 5, classes, color = [Gray(0.4), Gray(0.2)])
        end
        @testplot kappa_text_placement_single
    end

    @testset "binary discriminiation calibration curves" begin
        rng = StableRNG(22)
        curves = [(LinRange(0, 1, 10), range(0, stop=i/2, length=10) .+ (randn(rng, 10) .* 0.1)) for i in -1:3]
        binary_discrimination_calibration_curves_plot = with_theme(
                BinaryDiscriminationCalibrationCurves = (
                    Ideal = (
                        linewidth = 3,
                        color = (:green, 0.5)
                    ),
                    CalibrationCurve = (
                        solid_color = :green,
                        markersize = 50, # should be overwritten by user kw
                        linewidth = 5,
                    ),
                    PerExpert = (
                        solid_color = :red,
                        linewidth=1
                    ),
                )
            ) do
            Lighthouse.plot_binary_discrimination_calibration_curves(
                curves[3],
                rand(rng, 5),
                curves[[1, 2, 4, 5]],
                nothing, nothing,
                "",
                markersize=10
            )
        end
        @testplot binary_discrimination_calibration_curves_plot
    end
end
