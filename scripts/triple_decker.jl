include("../src/oc_grad.jl")
using GLMakie
using JLD2

dim = 16
nt = 100
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = T # SparseMatrixCSC{ComplexF64}


begin
    gate::T = QFT(dim)
    phase = angle(det(gate)) 
    gate = gate*exp(-im*phase/dim) # unit determinant
    gate_log = im*log(gate)
    

    H::Vector{S} = hamiltonians(dim; σz=false)
    z::V = convert(V, -gate_log)
    projhermitian!(z)
    projtraceless!(z)
    preallocs = preallocate(dim, T, V)
end;


begin
    η = 1.0
    nη = 20
    δη = η/nη
    exp_step = exp(-im*(δη/η)*gate_log)
    q::T = one(gate)
    hp = (; H, q, η, nt, PA=preallocs);
end;

infidelity_hist = zeros(10_000)
length_hist = zeros(10_000)
count = 0
IF = Inf64
CL = Inf64

### homotopy
for k in 1:nη
    @show k
    q::T = q*exp_step
    η -= δη
    hp = (; H, q, η, nt, PA=preallocs);

    for _ in 1:100
        ρ, J = gradstep!(z, hp)
        if ρ < 1e-10
            println("stuck after $(count) iterations; aborting...")
            break
        else
            CL, IF = metrics(z, gate, hp)
            count += 1
            infidelity_hist[count] = IF
            length_hist[count] = CL
        end
    end
    println(IF)
end

for i in 1:500
    ρ, J = gradstep!(z, hp)
    if ρ < 1e-10
        println("stuck after $(count) iterations; aborting...")
        break
    else
        CL, IF = metrics(z, gate, hp)
        count += 1
        infidelity_hist[count] = IF
        length_hist[count] = CL
    end
    mod(i, 50) == 0 ? println(IF) : nothing
end

begin
    fig = Figure()
    axs = Axis(
        fig[1, 1], 
        xscale=log10, 
        yscale=log10,
        )
    y1 = view(infidelity_hist, 1:count)
    y2 = view(length_hist, 1:count)
    x = 1:count
    scatter!(axs, x, y1, color=:red, label="IF")
    scatter!(axs, x, y2, color=:blue, label="CL")
    axislegend(axs, position=:lb)
    fig
end

# cd("./sims")
# @save "triple_decker_qft.jld2" z y1 y2
