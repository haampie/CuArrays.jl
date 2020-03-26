# integration with LinearAlgebra.jl

CuMatOrAdj{T} = Union{StridedGPUMatrix, LinearAlgebra.Adjoint{T, <:StridedGPUMatrix{T}}, LinearAlgebra.Transpose{T, <:StridedGPUMatrix{T}}}
CuOrAdj{T} = Union{StridedGPUVecOrMat, LinearAlgebra.Adjoint{T, <:StridedGPUVecOrMat{T}}, LinearAlgebra.Transpose{T, <:StridedGPUVecOrMat{T}}}


# matrix division

function Base.:\(_A::CuMatOrAdj, _B::CuOrAdj)
    A, B = copy(_A), copy(_B)
    A, ipiv = CuArrays.CUSOLVER.getrf!(A)
    return CuArrays.CUSOLVER.getrs!('N', A, ipiv, B)
end
