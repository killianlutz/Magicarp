include("../src/oc_grad.jl")
using GLMakie
using JLD2

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
V = Matrix{ComplexF64} # Hermitian????
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}

gate::T = toSU(QFT(dim));
H::Vector{S} = TDhamiltonians(σz=false);
z, hp, hist = homotopy(gate, H; nη=20, ngrad=100);
z, hp, hist = descent!(z, hp; ngrad=5_000, IFatol=1e-6, hist);
# t, x, u = state_control(z, hp)

begin
    IFhist, CLhist = hist

    idx = 1:length(IFhist)
    fig = Figure(size=(1_000, 500))
    axs = Axis(
        fig[1, 1], 
        xscale=log10, 
        yscale=log10,
        xlabel="iterations",
        ylabel="cost"
        )
    scatter!(axs, idx, IFhist, color=:purple, label="infidelity")
    scatter!(axs, idx, CLhist, color=:teal, label="gate time")
    axislegend(axs, position=:lb)
    fig
end
