include("../src/oc_grad.jl")
using GLMakie

dim = 4
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}

gate::T = toSU(QFT(dim));
H::Vector{S} = hamiltonians(dim, σz=false);
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