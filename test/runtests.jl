include("set_up_tests.jl")

@testset "Aqua" begin
    Aqua.test_all(Lighthouse; ambiguities=false)
end

include("plotting.jl")
include("metrics.jl")
include("learn.jl")
include("utilities.jl")
include("logger.jl")
include("row.jl")
