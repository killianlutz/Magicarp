include("../src/oc_grad.jl")
include("../src/qubitsgen.jl")
using GLMakie
using JLD2
cd("./sims/")

dim = 4
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}

gate::T = toSU(QFT(dim))
H::Vector{S} = hamiltonians(dim, σz=false);

rule = NAdam(1e-2)
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


dim = 4
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}
H::Vector{S} = hamiltonians(dim, σz=false);

seed!(3784334033809)
ngen = 2_500
qs::Vector{T} = map(_ -> sample_spunitary(dim), 1:ngen);
zs = Vector{T}(undef, ngen)

# verbose = false
# IFabstol = 1e-5
# rule = NAdam(1e-2)
# @threads for i in eachindex(zs)
#     z, hp, hist = homotopy(qs[i], H, rule; nη=20, ngrad=100, verbose);
#     IFhist, _ = hist
#     if last(IFhist) > IFabstol
#         z, hp, hist = descent!(z, hp, rule; ngrad=5_000, IFabstol=1e-5, hist, verbose);
#     end
#     zs[i] = z
# end
# Ts = map(z -> curve_length(z, H), zs)

# @load "mcstats_xy_ddecker.jld2" qs zs Ts

begin
    meanT = sum(Ts)/length(Ts)
    medianT = median(Ts)
    nbins = 15
    colgrad = cgrad(:viridis, nbins; categorical=true);
    histcolor = to_color.(colgrad)
    strokewidth = 2/10

    fig = Figure()
    axh = Axis(
        fig[1, 1], 
        aspect=AxisAspect(1),
        xlabel=L"f",
        ylabel=L"T(q)",
        title=L"\mathrm{Frequency~ (3.5⋅10^3~ samples)}"
        )
    hist!(axh, Ts; bins=nbins, direction=:y, normalization=:probability, color=histcolor, strokewidth)
    vlines!(axh, meanT, color=:black, linestyle=:dot, linewidth=1.0)
    vlines!(axh, medianT, color=:black, linestyle=:dash, linewidth=1.0)
    hidespines!(axh)

    fig
end

# @save "mcstats_xy_ddecker.jld2" qs zs Ts
