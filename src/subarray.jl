import Base: view

using Base: ScalarIndex, ViewIndex, Slice, @boundscheck,
            to_indices, compute_offset1, unsafe_length, _maybe_reshape_parent, index_ndims


## operations

# copyto! doesn't know how to deal with SubArrays, but broadcast does
# FIXME: use the rules from Adapt.jl to define copyto! methods in GPUArrays.jl
function Base.copyto!(dest::AbstractGPUArray{T,N}, src::SubArray{T,N,<:AbstractGPUArray{T}}) where {T,N}
    view(dest, axes(src)...) .= src
    dest
end

# copying to a CPU array requires an intermediate copy
# TODO: support other copyto! invocations (GPUArrays.jl copyto! defs + Adapt.jl rules)
function Base.copyto!(dest::AbstractArray{T,N}, src::SubArray{T,N,AT}) where {T,N,AT<:AbstractGPUArray{T}}
    temp = similar(AT, axes(src))
    copyto!(temp, src)
    copyto!(dest, temp)
end

# upload the SubArray indices when adapting to the GPU
# (can't do this eagerly or the view constructor wouldn't be able to boundscheck)
# FIXME: alternatively, have users do `cu(view(cu(A), inds))`, but that seems redundant
Adapt.adapt_structure(to::CUDAnative.Adaptor, A::SubArray) =
    SubArray(adapt(to, parent(A)), adapt(to, adapt(CuArray, parentindices(A))))

using Base: FastContiguousSubArray, ReinterpretArray, ReshapedArray, SubArray, RangeIndex, AbstractCartesianIndex

StridedFastContiguousGPUSubArray{T,N,A<:CuArray} = FastContiguousSubArray{T,N,A}
StridedReinterpretGPUArray{T,N,A<:Union{CuArray,StridedFastContiguousGPUSubArray}} = ReinterpretArray{T,N,S,A} where S
StridedReshapedGPUArray{T,N,A<:Union{CuArray,StridedFastContiguousGPUSubArray,StridedReinterpretGPUArray}} = ReshapedArray{T,N,A}
StridedGPUSubArray{T,N,A<:Union{CuArray,StridedReshapedGPUArray,StridedReinterpretGPUArray}, I<:Tuple{Vararg{Union{RangeIndex, AbstractCartesianIndex}}}} = SubArray{T,N,A,I}
StridedGPUArray{T,N} = Union{CuArray{T,N}, StridedGPUSubArray{T,N}, StridedReshapedGPUArray{T,N}, StridedReinterpretGPUArray{T,N}}
StridedGPUVector{T} = Union{CuArray{T,1}, StridedGPUSubArray{T,1}, StridedReshapedGPUArray{T,1}, StridedReinterpretGPUArray{T,1}}
StridedGPUMatrix{T} = Union{CuArray{T,2}, StridedGPUSubArray{T,2}, StridedReshapedGPUArray{T,2}, StridedReinterpretGPUArray{T,2}}
StridedGPUVecOrMat{T} = Union{StridedGPUVector{T}, StridedGPUMatrix{T}}


import Base: unsafe_convert

unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{RangeIndex}}}) where {T,N,P} =
    unsafe_convert(CuPtr{T}, V.parent) + (Base.first_index(V)-1)*sizeof(T)

