include("../src/oc_grad.jl")
using GLMakie
using Optimisers
using JLD2
cd("./sims/")

##### PERFORMANCES when dim = 10
# @btime forward_evolution!($z, $hp) # 759.167 μs (0 allocations: 0 bytes)
# @btime backward_evolution!($z, $hp) #  3.675 ms (0 allocations: 0 bytes)
# @btime projection!($z, $hp) #  2.755 μs (0 allocations: 0 bytes)
# @btime cost!($hp) # 116.858 ns (0 allocations: 0 bytes)
# @btime grad_cost!($hp) # 170.125 ns (0 allocations: 0 bytes)
# @btime evalcost!($z, $hp) # 805.417 μs (0 allocations: 0 bytes)
# @btime update!($z, 1, $hp) # 53.023 ns (0 allocations: 0 bytes)

# @btime gradstep!($z, $hp) # 7.718 ms (0 allocations: 0 bytes)
# # before : 47.857 ms (51204 allocations: 1.04 MiB)
#####

dim = 16
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64} 
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}

gate::T = sample_spunitary(dim)
H::Vector{S} = TDhamiltonians(σz=false);

rule = Adam(2e-2)
z, hp, hist = homotopy(gate, H, rule; nη=20, ngrad=200);
# z, hp, hist = homotopy(gate, H; nη=20, ngrad=100);

# @load "triple_decker_qft.jld2"

z, hp, hist = descent!(z, hp, rule; ngrad=5_000, IFabstol=1e-5, hist);
# z, hp, hist = descent!(z, hp; ngrad=5_000, IFabstol=1e-5, hist);
t, x, u = state_control(z, hp);

# @save "triple_decker_qft.jld2" z hp hist

begin
    IFhist, CLhist = hist

    idx = 1:length(IFhist)
    fig = Figure(size=(1_000, 500))
    axs = Axis(
        fig[1, 1], 
        xscale=log10, 
        yscale=log10,
        xlabel="iterations",
        ylabel="cost",
        title=L"T = %$(last(CLhist))"
        )
    scatter!(axs, idx, IFhist, color=:purple, label="infidelity")
    scatter!(axs, idx, CLhist, color=:teal, label="gate time")
    axislegend(axs, position=:lb)

    axs = Axis(fig[1, 2],
        xlabel=L"s",
        ylabel=L"u_j"
        )

    U = reduce(hcat, u)
    rows = eachrow(U)
    colgrad = cgrad(:RdBu, length(rows); categorical=true)
    colors = to_color.(colgrad)
    foreach(zip(rows, colors)) do (u, c)
        lines!(axs, t, u, color=c)
    end
    fig
end
