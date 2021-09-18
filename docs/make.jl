using TensorContractionsXD
using Documenter

DocMeta.setdocmeta!(TensorContractionsXD, :DocTestSetup, :(using TensorContractionsXD); recursive=true)

makedocs(;
    modules=[TensorContractionsXD],
    authors="PhysicsCodesLab",
    repo="https://github.com/PhysicsCodesLab/TensorContractionsXD.jl/blob/{commit}{path}#{line}",
    sitename="TensorContractionsXD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PhysicsCodesLab.github.io/TensorContractionsXD.jl",
        assets=String[],
    ),
    pages = [
        "Home" => ["index.md",
                    "indexnotation.md",
                    "functions.md",
                    "cache.md",
                    "implementation.md"]
    ],
)

deploydocs(;
    repo="github.com/PhysicsCodesLab/TensorContractionsXD.jl",
    devbranch="master",
)
