using Pkg
Pkg.activate("../magicarp")

include("../src/NaturalDescent.jl")
using .NaturalDescent # local module
using JLD2: @save, @load
using Random: default_rng, seed!

rng = default_rng()
seed!(rng, 23902)
isdir("sims") ? nothing : mkdir("sims")

begin
    include("./parameters.jl")

    graph = TDgraph() # linear_graph(dim)
    # display(showgraph(dim, graph)) 
    H::Vector{T} = hamiltonians(dim, graph; σz=false); # control hamiltonians
    
    gate::T = sample_spunitary(dim; rng);              # target  gate
    hp = hyperparameters(gate, H; η=0.0, nt);

    z::T = zeros(ComplexF64, dim, dim)                 # g matrix
    ξ::Vector{Float64} = randn(rng, n)                 # coefficients of g in su(d) basis
    coeffs_to_basis!(z, ξ, hp)                         # construct g based on its coefficients

    PA = preallocate(dim, T)
    LS = setup_leastsquares(z, hp, PA; regularization)
    p = (; ξ, z, hp, LS, PA...)
end;

# iterative mesh refinement via "mesh_schedule"
history, retcode = optimize!(p, lsearch_p, mesh_schedule; descent_p..., rng);
IF, GT = history;

### post-process
t, x, u = state_control(z, hp; nt=2_000); # x, u = state, control
gate_time = gatetime(z, hp)
IFval = validate(z, hp; nt=2_000)

fig = postprocess(t, u, IFval, IF, GT)

### save 
# @save "./sims/TD_0.jld2" gate z ξ hp IF GT retcode
# @load "./sims/TD_0.jld2" gate z ξ hp IF GT retcode
