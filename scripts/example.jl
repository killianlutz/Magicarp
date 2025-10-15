using Pkg
Pkg.activate("../magicarp")

include("../src/NaturalDescent.jl")
using .NaturalDescent # local module
using JLD2: @save, @load
using Random

using Base.Threads # easy parallelization

begin
    include("./parameters.jl")

    n_samples = 10 # number of gates to be compiled

    rngs = map(i -> MersenneTwister(2678923 + i), 1:n_samples)
    retcodes = zeros(Int, n_samples) # success ?
    gates = map(rng -> sample_spunitary(dim; rng), rngs)

    graph = dim == 16 ? TDgraph() : linear_graph(dim)
    H::Vector{T} = hamiltonians(dim, graph; σz=false); # control hamiltonians
end;

### preallocate memory for each available threads
n_threads = Threads.nthreads() 
problems = map(1:n_threads) do i
    gate::T = one(first(H))                            # target  gate
    hp = hyperparameters(gate, H; η=0.0, nt);

    z::T = zeros(ComplexF64, dim, dim)                 # g matrix
    ξ::Vector{Float64} = randn(n)             # coefficients of g in su(d) basis

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
    refine_mesh!(p, nt)
    history, retcode = optimize!(p, lsearch_p, mesh_schedule; descent_p..., rng);
    IF, GT = history
    @save "./sims/$(dim)_xy_$(i).jld2" gate=p.hp.q z=p.z ξ=p.ξ hp=p.hp IF GT retcode
    
    return retcode
end

### warm-up (compilation)
let
    @Threads.threads for i in 1:n_threads
        nothing
    end

    p = first(problems)
    rng = Random.default_rng()

    initprob!(p, rng, 1)

    # problem initialized with a local minimizer
    randn!(rng, p.ξ)
    coeffs_to_basis!(p.z, p.ξ, p.hp)
    p.hp.q .= final_state!(p.z, p.hp)

    solve_save!(p, rng, 1)
end

### loop over samples in parallel and save results
Threads.@threads for i in 1:n_samples
    p = problems[Threads.threadid()] # use thread specific memory
    rng = rngs[i]
    
    initprob!(p, rng, i)
    retcodes[i] = solve_save!(p, rng, i)
end

for i in 1:n_samples
    p = problems[Threads.threadid()] # use thread specific memory
    rng = rngs[i]
    
    initprob!(p, rng, i)
    retcodes[i] = solve_save!(p, rng, i)
end

### check results
dim = 16
@load "./sims/$(dim)_xy_3.jld2" gate z ξ hp IF GT retcode

t, x, u = state_control(z, hp; nt=2_000);
gate_time = gatetime(z, hp)
IFval = validate(z, hp; nt=2_000)

fig = postprocess(t, u, IFval, IF, GT)

### retrieve gate times 
GTs = map(1:n_samples) do i
    if isfile("./sims/$(dim)_xy_$(i).jld2")
        @load "./sims/$(dim)_xy_$(i).jld2" GT
        GT[end]
    else
        missing
    end
end
