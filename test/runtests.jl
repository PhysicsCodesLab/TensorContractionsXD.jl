using Test
using TensorContractionsXD
using Random
using LinearAlgebra
using CUDA

Random.seed!(1234567)

TensorContractionsXD.enable_blas()
TensorContractionsXD.enable_cache()
include("methods.jl")
include("tensor.jl")
TensorContractionsXD.disable_cache()
include("methods.jl")
include("tensor.jl")
TensorContractionsXD.disable_blas()
include("methods.jl")
include("tensor.jl")
TensorContractionsXD.enable_blas()
TensorContractionsXD.enable_cache()

if CUDA.functional()
    include("cutensor.jl")
end

include("tensoropt.jl")
include("auxiliary.jl")
