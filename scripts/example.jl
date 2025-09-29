using Pkg
Pkg.activate("../magicarp")

include("../src/NaturalDescent.jl")
using .NaturalDescent # local module
using JLD2: @save, @load
using Random
using Base.Threads

begin
    include("./parameters.jl")

    n_samples = 20 # number of random gates 
    rngs = map(i -> MersenneTwister(2678923 + i), 1:n_samples)
    retcodes = zeros(Int, n_samples) # success ?

    graph = dim == 16 ? TDgraph() : linear_graph(dim)
    H::Vector{T} = hamiltonians(dim, graph; σz=false); # control hamiltonians
    gates = map(rng -> sample_spunitary(dim; rng), rngs)
end;

### preallocate memory for each available threads
n_threads = Threads.nthreads() 
problems = map(1:n_threads) do i
    gate::T = one(first(H))                            # target  gate
    hp = hyperparameters(gate, H; η=0.0, nt);

    z::T = zeros(ComplexF64, dim, dim)                 # g matrix
    ξ::Vector{Float64} = randn(rngs[i], n)             # coefficients of g in su(d) basis

    PA = preallocate(dim, T)
    LS = setup_leastsquares(z, hp, PA; regularization)
    p = (; ξ, z, hp, LS, PA...)
end;

### initialize new problem, solve and save
initprob! = (p, rng, i) -> begin
    p.hp.q .= gates[i]
    randn!(rng, p.ξ)
    p.ξ ./= sqrt(length(p.ξ))
    coeffs_to_basis!(p.z, p.ξ, p.hp)

    return p
end

solve_save! = (p, rng, i) -> begin
    history, retcode = optimize!(p, lsearch_p, mesh_schedule; descent_p..., rng);
    IF, GT = history
    @save "./sims/$(dim)_xy_$(i).jld2" gate=p.hp.q z=p.z ξ=p.ξ hp=p.hp IF GT retcode
    
    return retcode
end

### warm-up (compilation)
let
    @Threads.threads :dynamic for i in 1:n_threads
        nothing
    end

    p = first(problems)
    rng = first(rngs)
    initprob!(p, rng, 1)
    solve_save!(p, rng, 1)
end

### loop over samples in parallel and save results
Threads.@threads :dynamic for i in 1:n_samples
    p = problems[Threads.threadid()] # use thread specific memory
    rng = rngs[i]
    
    initprob!(p, rng, i)
    retcodes[i] = solve_save!(p, rng, i)
end

### check results
@load "./sims/4_xy_20.jld2"

t, x, u = state_control(z, hp; nt=2_000);
gate_time = gatetime(z, hp)
IFval = validate(z, hp; nt=2_000)

fig = postprocess(t, u, IFval, IF, GT)

### retrieve gate times 
GTs = map(1:n_samples) do i
    @load "./sims/4_xy_$(i).jld2" GT
    GT[end]
end
