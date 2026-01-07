include("set_up_tests.jl")

@testset "Aqua" begin
    # we ignore Makie because we do a manual install of a specific
    # version to test specific versions
    Aqua.test_all(Lighthouse;
                  ambiguities=false, state_deps=(;ignore=:Makie))
end

include("plotting.jl")
include("metrics.jl")
include("learn.jl")
include("utilities.jl")
include("logger.jl")
include("row.jl")
