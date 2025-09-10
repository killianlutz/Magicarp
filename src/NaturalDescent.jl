module NaturalDescent

export hyperparameters, preallocate, setup_leastsquares
export optimize!
export postprocess
export final_state!
export refine_mesh!

using Reexport # exports of QuantumTools -> into those of this module
include("../src/QuantumTools.jl")
@reexport using .QuantumTools 

using GLMakie
using Parameters: @with_kw
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Base.Threads

import LinearSolve as LinS
import SciMLOperators as Op

include("./forward_backward.jl")
include("./vjp_jvp.jl")
include("./natgrad.jl")
include("./optimize.jl")

end # module NaturalDescent