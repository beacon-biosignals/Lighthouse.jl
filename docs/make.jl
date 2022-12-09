using Lighthouse
using Documenter

makedocs(; modules=[Lighthouse], sitename="Lighthouse",
         authors="Beacon Biosignals and other contributors",
         pages=["API Documentation" => "index.md",
                "Plotting" => "plotting.md"],
         # makes docs fail hard if there is any error building the examples,
         # so we don't just miss a build failure!
         strict=true)

deploydocs(; repo="github.com/beacon-biosignals/Lighthouse.jl.git",
           devbranch="main", push_preview=true)
