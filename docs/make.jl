using Lighthouse
using Documenter

makedocs(; modules=[Lighthouse], sitename="Lighthouse",
         authors="Beacon Biosignals and other contributors",
         pages=["API Documentation" => "index.md",
                "Plotting" => "plotting.md"])

deploydocs(repo="github.com/beacon-biosignals/Lighthouse.jl.git",
           devbranch="main", push_preview=true)
