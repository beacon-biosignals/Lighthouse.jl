using Lighthouse
using Documenter

makedocs(; modules=[Lighthouse], sitename="Lighthouse.jl",
         authors="Beacon Biosignals and other contributors",
         repo="https://github.com/beacon-biosignals/Lighthouse.jl/blob/{commit}{path}#{line}",
         pages=["API Documentation" => "index.md",
                "Plotting" => "plotting.md"],
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                repolink="https://github.com/beacon-biosignals/Lighthouse.jl",
                                canonical="https://beacon-biosignals.github.io/Lighthouse.jl/stable",
                                assets=String[]))

deploydocs(; repo="github.com/beacon-biosignals/Lighthouse.jl",
           devbranch="main",
           push_preview=true)
