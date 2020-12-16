@testset "`majority`" begin
    @test majority([1, 2, 1, 3, 2, 2, 3], 1:3) == 2
    @test majority([1, 2, 1, 3, 2, 2, 3, 4], 3:4) == 3
    rng = MersenneTwister(42)
    picked = [majority(rng, 1:2, 1:2) for _ in 1:1_000_000]
    @test isapprox(count(==(1), picked), 500_000; atol=1000)
end

@testset "`Lighthouse.area_under_curve`" begin
    @test_throws ArgumentError Lighthouse.area_under_curve([0, 1, 2], [0, 1])
    @test_throws ArgumentError Lighthouse.area_under_curve([], [])
    @test isapprox(Lighthouse.area_under_curve(collect(0:0.01:1), collect(0:0.01:1)), 0.5;
                   atol=0.01)
    @test isapprox(Lighthouse.area_under_curve(collect(0:0.01:(2π)), sin.(0:0.01:(2π))),
                   0.0; atol=0.01)
end

@testset "`Lighthouse.area_under_curve_unit_square`" begin
    @test_throws ArgumentError Lighthouse.area_under_curve_unit_square([0, 1, 2], [0, 1])
    @test_throws ArgumentError Lighthouse.area_under_curve_unit_square([], [])
    @test isapprox(Lighthouse.area_under_curve_unit_square(collect(0:0.01:1),
                                                           collect(0:0.01:1)), 0.5;
                   atol=0.01)
    @test isapprox(Lighthouse.area_under_curve_unit_square(collect(0:0.01:(2π)),
                                                           sin.(0:0.01:(2π))), 0.459;
                   atol=0.01)
end
