include("../src/oc_grad.jl")
using GLMakie

dim = 4
nt = 100
<<<<<<< HEAD
S = SparseMatrixCSC{ComplexF64}
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}

begin
    gate = QFT(dim)
=======
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = T # SparseMatrixCSC{ComplexF64}


begin
    gate::T = QFT(dim)
>>>>>>> dev
    phase = angle(det(gate)) 
    gate = gate*exp(-im*phase/dim) # unit determinant
    gate_log = im*log(gate)

<<<<<<< HEAD
    H = hamiltonians(dim; σz=false)
    z = convert(T, -gate_log)
end

begin
    x0 = convert(T, I(dim))
    v = OdePoint3{T, V}(similar(x0), similar(x0), similar(z))
    xs = [similar(x0) for _ in 1:3]
    vs = [similar(v) for _ in 1:3]
    Pz = similar(z)
    dJ = similar(z)
    zρ = similar(z)
    xtemp = similar(x0)
    vtemp1 = similar(v.x)
    vtemp2 = similar(v.g)
    preallocs = (; x0, v, xs, vs, Pz, dJ, zρ, xtemp, vtemp1, vtemp2)
=======
    H::Vector{S} = hamiltonians(dim; σz=false)
    z::V = convert(V, -gate_log)
    preallocs = preallocate(dim, T, V)
>>>>>>> dev
end;

begin
    η = 1.0
    nη = 20
    δη = η/nη
    exp_step = exp(-im*(δη/η)*gate_log)
<<<<<<< HEAD
    q = one(gate)
    hp = (; H, q, η, nt, PA=preallocs);
end

infidelity_hist = zeros(20_000)
length_hist = zeros(20_000)
=======
    q::T = one(gate)
    hp = (; H, q, η, nt, PA=preallocs);
end;

infidelity_hist = zeros(10_000)
length_hist = zeros(10_000)
>>>>>>> dev
count = 0
IF = Inf64
CL = Inf64

### homotopy
for k in 1:nη
    @show k
    η -= δη
    q = q*exp_step
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