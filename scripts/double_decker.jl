include("../src/oc_grad.jl")
using GLMakie

dim = 4
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}

gate::T = toSU(QFT(dim));
H::Vector{S} = hamiltonians(dim, σz=false);

rule = NAdam()
z, hp, hist = homotopy(gate, H, rule; nη=20, ngrad=100);
# z, hp, hist = homotopy(gate, H; nη=20, ngrad=100);

# @load "double_decker_qft.jld2"

z, hp, hist = descent!(z, hp, rule; ngrad=5_000, IFabstol=1e-5, hist);
# z, hp, hist = descent!(z, hp; ngrad=5_000, IFabstol=1e-5, hist);
t, x, u = state_control(z, hp);

# @save "double_decker_qft.jld2" z hp hist

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
    colgrad = cgrad([:purple, :teal], length(rows); categorical=true)
    colors = to_color.(colgrad)
    foreach(zip(rows, colors)) do (u, c)
        lines!(axs, t, u, color=c)
    end
    fig
end