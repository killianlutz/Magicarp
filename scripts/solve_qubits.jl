include("../src/oc_grad.jl")
include("../src/qubitsgen.jl")
using GLMakie
using JLD2
cd("./sims/")

dim = 2
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}
pauli = hamiltonians(dim; σz=true)

#### load calculation results
# @load "solve_qubits.jld2" vs zs

#### sample generators in R^3
seed!(378433403380)
ngen = 10_000
# vs = [sample_sphere() for _ in 1:ngen]

#### stereo proj
p = stereographic.(vs)
polar = cartesian_to_polar(p)
radii = polar[1, :]
phase = polar[2, :]

#### embed generators in su(2) to generate SU(2) gate
hs = map(s -> sphere_to_su(s, pauli), vs)
qs = map(h -> cis(Hermitian(-h)), hs)

#### solve control problem for z
# zs = Vector{T}(undef, ngen)

verbose = false
H::Vector{S} = hamiltonians(dim, σz=false);
# @threads for i in eachindex(zs)
#     z, hp, hist = homotopy(qs[i], H; nη=20, ngrad=200, verbose);
#     z, hp, hist = descent!(z, hp; ngrad=5_000, IFatol=1e-5, hist, verbose);
#     zs[i] = z
# end

#### embed back in the sphere
vzs = map(z -> su_to_sphere(z, pauli), zs)
Ts = map(norm_equator, vzs)

#### delete equator (x_3 ≃ 0) due to numerical error prone
begin
    abstol = 5e-2
    isoutside_equator = map(vzs) do vz
        x3 = last(vz)
        abs(x3) > abstol ? true : false
    end

    vs_out = vs[isoutside_equator]
    vzs_out = vzs[isoutside_equator]
    Ts_out = Ts[isoutside_equator]
    color = radii[isoutside_equator]
end

#### plot results
begin
    S1 = Circle(zero(Point2), 1.0)
    S2 = Sphere(zero(Point3), 1.0)
    cmap = :thermal
    # cmap = cgrad([:purple, :orange], ngen)
    marker = :cross
    colorscale = log10

    nbins = 15
    colgrad = cgrad(:viridis, nbins; categorical=true);
    histcolor = to_color.(colgrad)
    strokewidth = 2/10
end

using CairoMakie
GLMakie.activate!()
begin
    fig = Figure(size=(1_000, 1_000))
    
    begin
        axq = Axis3(
            fig[1, 1], 
            aspect=:equal,
            xlabel=L"x_1",
            ylabel=L"x_2",
            zlabel=L"x_3",
            title=L"\mathrm{Gate~} q"
            )
        scatter!(axq, vs_out; color, colormap=cmap, colorscale, marker)
        wireframe!(axq, S2, color=:black, alpha=0.1)
    end

    begin
        axz = Axis3(
            fig[1, 2], 
            aspect=:equal,
            xlabel=L"x_1",
            ylabel=L"x_2",
            zlabel=L"x_3",
            title=L"\mathrm{Control~} z = z(q)"
            )
        scatter!(axz, vzs_out; color, colormap=cmap, colorscale, marker)
        wireframe!(axz, S2, color=:black, alpha=0.1)
    end

    begin
        axt = Axis3(
            fig[2, 1], 
            aspect=:equal,
            xlabel=L"x_1",
            ylabel=L"x_2",
            zlabel=L"x_3",
            title=L"\mathrm{Time~} T = T(q)"
            )
        scatter!(axt, vs_out; color=Ts_out, colormap=:viridis, marker)
        wireframe!(axt, S2, color=:black, alpha=0.1)
    end

    begin
        axh = Axis(
            fig[2, 2], 
            aspect=AxisAspect(1),
            xlabel=L"f",
            ylabel=L"T(q)",
            title=L"\mathrm{Frequency~ (10^4~ samples)}"
            )
        hist!(axh, Ts_out; bins=nbins, direction=:x, normalization=:probability, color=histcolor, strokewidth)
    end
    foreach(hidespines!, [axq, axz, axt, axh])
    fig
end

#### save results
# GLMakie.save("magicarpe_stats.png", fig)
# @save "solve_qubits.jld2" vs zs