
using LinearAlgebra, GPUArrays, Metal

gpu = MtlArray

N = 1000

A = randn(Float32, N,N)
B = randn(Float32, N,N)
C = similar(A, (N,N))

mul!(C,A,B)


Agpu = A |> gpu
Bgpu = B|> gpu
Cgpu = similar(Agpu(N,N))

mul!(Cgpu, Agpu, Bgpu)

##

b = randn(Float32,N)
b_gpu |> gpu
Agpu \ b_gpu