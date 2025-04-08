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

#### load past calculation results
# @load "mcstats_xy.jld2" vs zs Ts
# @load "mcstats_xz.jld2" vs zs Ts
# @load "mcstats_yz.jld2" vs zs Ts
ctrl = [1, 2]

#### sample generators in R^3
seed!(378433403380)
ngen = 10_000
vs = [sample_sphere() for _ in 1:ngen]

#### stereo proj
p = stereographic.(vs)
polar = cartesian_to_polar(p)
radii = polar[1, :]
phase = polar[2, :]

#### radius for the generator
# R = 1.0
# vs .*= R

#### embed generators in su(2) to generate SU(2) gate
hs = map(s -> sphere_to_su(s, pauli), vs)
qs = map(h -> cis(Hermitian(-h)), hs)

#### solve control problem for z
# zs = Vector{T}(undef, ngen)

# verbose = false
# IFabstol = 1e-5
# rule = NAdam()
H::Vector{S} = pauli[ctrl];
# @threads for i in eachindex(zs)
#     z, hp, hist = homotopy(qs[i], H, rule; nη=20, ngrad=100, verbose);
#     IFhist, _ = hist
#     if last(IFhist) > IFabstol
#         z, hp, hist = descent!(z, hp, rule; ngrad=5_000, IFabstol=1e-5, hist, verbose);
#     end
#     zs[i] = z
# end

#### embed back in the sphere
vzs = map(z -> su_to_sphere(z, pauli), zs)
Ts = map(z -> curve_length(z, H), zs)

#### plot results
begin
    S1 = Circle(zero(Point2), 1.0)
    S2 = Sphere(zero(Point3), 1.0)
    cmap = :thermal
    marker = :cross
    colorscale = log10

    nbins = 15
    colgrad = cgrad(:viridis, nbins; categorical=true);
    histcolor = to_color.(colgrad)
    strokewidth = 2/10

    showphases = [-π/3, -π/2, -2π/3]
    phase_colors = [:red, :green, :blue]
    phase_abstol = 7e-2
end

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
        scatter!(axq, vs; color=radii, colormap=cmap, colorscale, marker)
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
        scatter!(axz, vzs; color=radii, colormap=cmap, colorscale, marker)
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
        scatter!(axt, vs; color=Ts, colormap=:viridis, marker)
        wireframe!(axt, S2, color=:black, alpha=0.1)
    end

    begin
        meanT = sum(Ts)/length(Ts)
        medianT = median(Ts)
        axh = Axis(
            fig[2, 2], 
            aspect=AxisAspect(1),
            xlabel=L"T(q)",
            ylabel=L"f",
            title=L"\mathrm{Frequency~ (10^4~ samples)}"
            )
        hist!(axh, Ts; bins=nbins, direction=:y, normalization=:probability, color=histcolor, strokewidth)
        vlines!(axh, meanT, color=:black, linestyle=:dot, linewidth=1.0)
        vlines!(axh, medianT, color=:black, linestyle=:dash, linewidth=1.0)
    end

    foreach(showphases, phase_colors) do φ, c
        δφ = phase .- φ
        keep = abs.(δφ) .<= phase_abstol
        vsφ = view(vs, keep)
        vzsφ = view(vzs, keep)
        scatter!(axq, vsφ, color=c)
        scatter!(axz, vzsφ, color=c)
    end

    foreach(hidespines!, [axq, axz, axt, axh])
    fig
end


#### save results
# GLMakie.save("mcstats_oo.png", fig)
# @save "mcstats_oo.jld2" vs zs Ts