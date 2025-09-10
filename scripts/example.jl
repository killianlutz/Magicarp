using Pkg
Pkg.activate("../magicarp")

include("../src/NaturalDescent.jl")
using .NaturalDescent # local module
using JLD2: @save, @load
using Random: default_rng, seed!

rng = default_rng()
seed!(rng, 23902)

### preallocate stuff
begin
    include("./parameters.jl")

    graph = linear_graph(dim) #TDgraph()
    H::Vector{T} = hamiltonians(dim, graph; σz=false); # control hamiltonians
    
    gate::T = one(first(H))                          # target  gate
    hp = hyperparameters(gate, H; η=0.0, nt);
    
    z::T = zeros(ComplexF64, dim, dim)                 # g matrix
    ξ::Vector{Float64} = randn(rng, n)                 # coefficients of g in su(d) basis

    PA = preallocate(dim, T)
    LS = setup_leastsquares(z, hp, PA; regularization)
    p = (; ξ, z, hp, LS, PA...)
end;

### loop over samples and save results
n_samples = 50 # number of random gates 
retcodes = zeros(Int, n_samples)
for i in 1:n_samples
    gate .= sample_spunitary(dim; rng)
    hp.q .= gate
    ξ .= ones(n)/sqrt(n)
    coeffs_to_basis!(z, ξ, hp)

    history, retcode = optimize!(p, lsearch_p, mesh_schedule; descent_p..., rng);
    IF, GT = history
    retcodes[i] = retcode

    @save "./sims/$(dim)_xy_$(i).jld2" gate z ξ hp IF GT retcode
end

### check results
@load "./sims/4_xy_1.jld2"

t, x, u = state_control(z, hp; nt=2_000);
gate_time = gatetime(z, hp)
IFval = validate(z, hp; nt=2_000)

fig = postprocess(t, u, IFval, IF, GT)

### retrieve gate times 
GTs = map(1:n_samples) do i
    @load "./sims/4_xy_$(i).jld2" GT
    GT[end]
end
