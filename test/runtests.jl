using Test
using TensorOperationsXD
using Random
using LinearAlgebra
using CUDA

Random.seed!(1234567)

TensorOperationsXD.enable_blas()
TensorOperationsXD.enable_cache()
include("methods.jl")
include("tensor.jl")
TensorOperationsXD.disable_cache()
include("methods.jl")
include("tensor.jl")
TensorOperationsXD.disable_blas()
include("methods.jl")
include("tensor.jl")
TensorOperationsXD.enable_blas()
TensorOperationsXD.enable_cache()

if CUDA.functional()
    include("cutensor.jl")
end

include("tensoropt.jl")
include("auxiliary.jl")
