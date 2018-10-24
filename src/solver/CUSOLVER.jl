module CUSOLVER

import CUDAdrv
using CUDAdrv: CuContext, CuDevice
using CUDAnative

using ..CuArrays
const cudaStream_t = Ptr{Nothing}

using ..CuArrays: libcusolver, configured, _getindex

using LinearAlgebra
using SparseArrays 

import Base.one
import Base.zero
import CuArrays.CUSPARSE.CuSparseMatrixCSR
import CuArrays.CUSPARSE.CuSparseMatrixCSC
import CuArrays.CUSPARSE.cusparseMatDescr_t

include("libcusolver_types.jl")
include("error.jl")
include("libcusolver.jl")

const libcusolver_handles_dense = Dict{CuContext,cusolverDnHandle_t}()
const libcusolver_handle_dense = Ref{cusolverDnHandle_t}()
const libcusolver_handles_sparse = Dict{CuContext,cusolverSpHandle_t}()
const libcusolver_handle_sparse = Ref{cusolverSpHandle_t}()

function __init__()
    configured || return

    # initialize the library when we switch devices
    callback = (dev::CuDevice, ctx::CuContext) -> begin
        libcusolver_handle_dense[] = get!(libcusolver_handles_dense, ctx) do
            @debug "Initializing dense CUSOLVER for $dev"
            handle = Ref{cusolverDnHandle_t}()
            cusolverDnCreate(handle)
            handle[]
        end
        libcusolver_handle_sparse[] = get!(libcusolver_handles_sparse, ctx) do
            @debug "Initializing sparse CUSOLVER for $dev"
            handle = Ref{cusolverSpHandle_t}()
            cusolverSpCreate(handle)
            handle[]
        end
    end
    push!(CUDAnative.device!_listeners, callback)

    # deinitialize when exiting
    atexit() do
        libcusolver_handle_dense[] = C_NULL
        libcusolver_handle_sparse[] = C_NULL

        for (ctx, handle) in libcusolver_handles_dense
            if CUDAdrv.isvalid(ctx)
                cusolverDnDestroy(handle)
            end
        end
        for (ctx, handle) in libcusolver_handles_sparse
            if CUDAdrv.isvalid(ctx)
                cusolverSpDestroy(handle)
            end
        end
    end
end

include("sparse.jl")
include("dense.jl")
include("highlevel.jl")

end
