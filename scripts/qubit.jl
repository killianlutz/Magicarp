include("../src/oc_grad.jl")
using GLMakie

dim = 2
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}
pauli = hamiltonians(dim; σz=true)

gate::T = toSU(QFT(dim));
H::Vector{S} = hamiltonians(dim, σz=false);
gate = exp(-im*pauli[3])
rule = NAdam()
z, hp, hist = homotopy(gate, H, rule; nη=20, ngrad=300);
# z, hp, hist = homotopy(gate, H; nη=20, ngrad=200);

# @load "qubit_qft.jld2"

z, hp, hist = descent!(z, hp, rule; ngrad=5_000, IFabstol=1e-5, hist);
# z, hp, hist = descent!(z, hp; ngrad=5_000, IFabstol=1e-5, hist);
t, x, u = state_control(z, hp);

# @save "qubit_qft.jld2" z hp hist

begin ## HISTORY
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

    U = reduce(hcat, u)
    axs = Axis(fig[1, 2],
        xlabel=L"s",
        ylabel=L"u_j"
        )
    rows = enumerate(eachrow(U))
    colors = [:purple, :teal, :orange]
    foreach(rows) do (j, u)
        scatter!(axs, t, u, color=colors[j], label=L"j=%$(j)")
    end
    axislegend(axs, position=:lb)
    fig
end

begin ## BLOCH
    pauli = hamiltonians(dim; σz=true)
    conjpauli = map(h -> gate*h*gate', pauli)

    cpgen = map(h -> pauli_vector(h, pauli), conjpauli)
    qgen = pauli_vector(im*log(gate), pauli)
    zgen = pauli_vector(z, pauli)

    L = norm(first(u))
    M = max(1.0, max(L, norm(qgen), norm(zgen)))

    cpnormal = [Point3(g/M) for g in cpgen]
    qnormal = [Point3(qgen/M)]
    znormal = [Point3(zgen/M)]
    unormal = map(u) do v
        Point3(first(v), last(v), 0.0)/M
    end

    Pqnormal = [Point3(qgen[1:2]..., 0.0)/M]
    Pznormal = [Point3(zgen[1:2]..., 0.0)/M]
    origin = [zero(Point3)]

    S1 = Sphere(zero(Point3), 1.0)

    fig = Figure()
    axs = Axis3(
        fig[1, 1],
        aspect=:equal,
        xlabel=L"X",
        ylabel=L"Y",
        zlabel=L"Z"
        )
    
    arrows!(axs, origin, cpnormal, color=:black, arrowsize=1/10, alpha=1/10, linewidth=0.05)
    arrows!(axs, origin, qnormal, color=:orange, arrowsize=1/10, alpha=2/10, linewidth=0.05)
    arrows!(axs, origin, znormal, color=:red, arrowsize=1/10, alpha=2/10, linewidth=0.05)
    arrows!(axs, origin, Pqnormal, color=:orange, arrowsize=1/10, alpha=1/10, linewidth=0.02)
    arrows!(axs, origin, Pznormal, color=:red, arrowsize=1/10, alpha=1/10, linewidth=0.02)
    
    usc  = scatter!(axs, unormal, color=t, colormap=:deep)
    cpsc = scatter!(axs, cpnormal, color=:black)
    qsc  = scatter!(axs, first(qnormal), color=:orange)
    zsc  = scatter!(axs, first(znormal), color=:red)

    wireframe!(axs, S1, color=:gray, alpha=0.1)

    Colorbar(fig[1, 2][1:2, 1], usc, label=L"t")
    Legend(fig[1, 2][3:4, 1], [cpsc, qsc, zsc], [L"qh_{j}q^\dagger", L"i \log q", L"z"])
    fig
end