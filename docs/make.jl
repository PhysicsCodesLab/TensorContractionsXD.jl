using TensorOperationsXD
using Documenter

DocMeta.setdocmeta!(TensorOperationsXD, :DocTestSetup, :(using TensorOperationsXD); recursive=true)

makedocs(;
    modules=[TensorOperationsXD],
    authors="PhysicsCodesLab",
    repo="https://github.com/PhysicsCodesLab/TensorOperationsXD.jl/blob/{commit}{path}#{line}",
    sitename="TensorOperationsXD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PhysicsCodesLab.github.io/TensorOperationsXD.jl",
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
    repo="github.com/PhysicsCodesLab/TensorOperationsXD.jl",
    devbranch="master",
)
