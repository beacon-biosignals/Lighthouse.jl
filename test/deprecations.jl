@testest "deprecations" begin
    @test_throws ErrorException Lighthouse.evaluation_metrics_row()
end
