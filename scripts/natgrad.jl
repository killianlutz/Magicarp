include("../src/natgradv0.jl")
include("../src/qubitsgen.jl")
using GLMakie
using Random
import LinearSolve as ls
import SciMLOperators as op

# "coarse to fine mesh approach"
dim = 9
nt = 200
T = Matrix{ComplexF64}
V = Matrix{ComplexF64}
S = Matrix{ComplexF64} # SparseMatrixCSC{ComplexF64}
pauli = hamiltonians(dim; σz=false)

function golden_section_search(f::Function, interval::NTuple{2,<:Real}, abstol::Real, max_iterations::Integer; reference_value::Real=Inf)
    a, b = interval
    γ = oftype(a, (1 + sqrt(5)) / 2) # golden number

    c = b - (γ - 1) * (b - a)
    d = a + (γ - 1) * (b - a)
    left_value = f(c)
    right_value = f(d)

    count = 1
    while (count <= max_iterations) && (b - a > abstol)
        if left_value <= right_value || isnan(right_value)
            b = d; d = c;
            c = b - (γ - 1) * (b - a)
            # only 1 evaluation of f instead of 2
            right_value = left_value
            left_value = f(c)
        else left_value > right_value
            a = c; c = d;
            d = a + (γ - 1) * (b - a)

            left_value = right_value
            right_value = f(d)
        end
        count += 1
    end

    # estimated minimizer 
    midpoint = (a + b) / 2
    midpoint_value = f(midpoint)
    if reference_value > midpoint_value
        # println("GS ----- steps: $(count) // minimizer estimate: $(midpoint) // Δf = $(reference_value-midpoint_value)")
        return (midpoint, midpoint_value)
    else
        # The line search failed to reduce the reference_value of f
        # println("GS FAILED ----- steps: $(count) // last estimate: $(midpoint)")
        return (eps(a), f(eps(a)))
    end
end

function matrix_basis(dim)
    T = SparseMatrixCSC{ComplexF64, Int64}
    B = Vector{T}()

    for i in 1:dim, j in 1:dim
        for x in [1, im]
            Eij = sparse([i], [j], [x], dim, dim)
            push!(B, Eij)
        end
    end

    return B
end

function vjp!(δz, z, v, hp)
    forward_evolution!(z, hp) # compute x(1)
    δz .= backward_evolution!(z, v, hp) # adjoint applied to v
end

function vjp(z, hp, v)
    #### CAREFUL WITH VARIABLE POINTERS... In doubt, create array in jvp and vjp
    forward_evolution!(z, hp) # compute x(1)
    backward_evolution!(z, v, hp) # adjoint applied to v
end

function jvp!(δx, z, δz, hp)
    δx .= linearized_evolution!(z, δz, hp) # directional derivative
end

function jvp(z, δz, hp)
    #### CAREFUL WITH VARIABLE POINTERS... In doubt, create array in jvp and vjp
    linearized_evolution!(z, δz, hp) # directional derivative
end

function final_state!(z, hp)
    forward_evolution!(z, hp)
end

function loss_grad!(grad, x, hp)
    Q = hp.q
    # dotp = dot(Q, x)
    # grad .= -2 .* dotp .* Q

    grad .= x .- Q

    return grad
end


# function loss_hessian(hp, basis)
#     Q = hp.q
#     n = length(basis)
#     H = Matrix{Float64}(undef, n, n)
    
#     for i in 1:n, j in 1:n
#         Ei = basis[i]
#         Ej = basis[j]

#         xi = dot(Q, Ei)
#         xj = dot(Q, Ej)
#         H[i, j] = -2*real(conj(xi)*xj)
#     end

#     return H
# end

function basis_to_coeffs!(ξ, z, hp)
    su_basis = hp.basis

    for (i, h) in enumerate(su_basis)
        ξ[i] = real(dot(h, z))
    end

    return ξ
end

function subspace_projection!(ξ, orthog_space)
    for i in orthog_space
        ξ[i] = 0.0
    end

    return ξ
end

function rand_orthog_space!(orthog_space, full_space)
    randperm!(full_space)

    for i in eachindex(orthog_space)
        orthog_space[i] = full_space[i]
    end

    return orthog_space
end

function jvp_ode!(dv, v, p, t)
    # preallocations
    yz = p.temp1
    ya = p.temp2

    mul!(dv.g, v.x, p.z)    # x*z
    mul!(v.g, dv.g, v.x')  # x*z*x'

    mul!(dv.g, v.x, p.δz)   # x*δz
    mul!(yz, dv.g, v.x')  # x*δz*x'

    mul!(ya, v.x, v.a)    # x*a
    mul!(dv.a, ya, p.z)   # x*a*z
    mul!(ya, dv.a, v.x')  # x*a*z*x'


    fill!(dv.x, 0)
    fill!(dv.a, 0)
    # dv.g and v.g are not used here besides preallocated arrays

    for h in p.H 
        dot_z = real(dot(h, v.g))
        dot_δz = real(dot(h, yz))
        dot_a = real(dot(h, ya))

        α = -im*dot_z
        β = -im*(2*dot_a + dot_δz)

        dv.x .+= α.*h
        dv.a .+= β.*h
    end

    # dx
    mul!(yz, dv.x, v.x)
    dv.x .= yz
    # da
    mul!(ya, dv.a, v.x) # (x da x') * x
    mul!(dv.a, v.x', ya) # [x' * (x da x') * x] = da
    
    return dv
end

function linearized_evolution!(z, δz, hp)
    v = hp.PA.v
    x0 = hp.PA.x0
    temp1 = hp.PA.vtemp1
    temp2 = hp.PA.vtemp2
    δx = hp.PA.Pz
    p = (; H=hp.H, z, δz, temp1, temp2)
    
    v.x .= x0
    fill!(v.a, 0) # linearization starts at zero since x(0) fixed
    v1 = heun!(jvp_ode!, v, p, hp.nt, hp.PA.vs)
    mul!(δx, v1.x, v1.a)

    return δx
end

########## MATRIX FREE
function gram_operator!(w, δξ, z, p, t)
    hp, orthog_space, α = p
    δz = hp.PA.zρ
    forward = hp.PA.Pz

    subspace_projection!(δξ, orthog_space)

    coeffs_to_basis!(δz, δξ, hp)
    jvp!(forward, z, δz, hp) # A*δz
    vjp!(δz, z, forward, hp) # A'*(A*δz)
    basis_to_coeffs!(w, δz, hp)
    w .+= α .* δξ # regularization

    subspace_projection!(w, orthog_space)
    nothing
end

function gram_operator!(δξ, z, p, t)
    w = similar(ξ)
    gram_operator!(w, δξ, z, p, t)
    return w
end

function standard_gradient!(p)
    _, z, hp, _, ∇ℓ, ∇J, _, δz, _ = p

    x = final_state!(z, hp)
    loss_grad!(∇ℓ, x, hp)
    vjp!(δz, z, ∇ℓ, hp)
    basis_to_coeffs!(∇J, δz, hp)

    return ∇J
end

function descent_step!(p, line_search!, val; natgrad=false, orthog_space=[])
    ξ, z, hp, LS, _, ∇J, δξ, _, μ = p

    # loss gradient into coordinates M_d(C)
    ∇J = standard_gradient!(p)
    subspace_projection!(∇J, orthog_space)

    # gradient or nat. gradient step calculation
    # ∇J ./= (1 .+ norm(∇J))

    if natgrad
        LS.A.p = (hp, orthog_space, last(LS.A.p))
        LS.A.u .= z
        LS.b .= ∇J
        δξ .= ls.solve!(LS) # solve HJ*x = ∇J -> ascent direction
    else
        δξ .= ∇J
    end

    δξ ./= (1 .+ norm(δξ))

    # linesearch
    e, next_val = line_search!(val)
    ρ = 10.0^e

    # update if decrease
    if ρ >= 1e-15 && next_val < val
        ξ .-= ρ .* δξ
        coeffs_to_basis!(z, ξ, hp)
        val = next_val
    end

    return val
end

begin
    interval = (-6.0, 1.0) # max step ∝ d ?
    abstol = 1e-1
    max_iterations = 200
    golden_section = (f, refval) -> golden_section_search(f, interval, abstol, max_iterations; reference_value=refval)
end 

begin
    mat_basis = matrix_basis(dim)
    su_basis = subasis(dim)
    n = length(su_basis)
    m = length(mat_basis)
    bases = (su_basis, mat_basis)
end

begin
    ∇ℓ::T = Matrix{ComplexF64}(undef, dim, dim)
    ∇J::Vector{Float64} = Vector{Float64}(undef, n)
    δξ::Vector{Float64} = Vector{Float64}(undef, n)
    δz::T = Matrix{ComplexF64}(undef, dim, dim)
    μ::Vector{Float64} = similar(δξ)
    IFhist = zeros(10_000)
    CLhist = zeros(10_000)

    full_space = collect(1:n)
    orthog_space = full_space[1:5]
    # full_space::Vector{Int64} = collect(1:n)
    # orthog_space::Vector{Int64} = full_space[1:5]
end;

begin
    gate::T = sample_spunitary(dim);
    H::Vector{S} = hamiltonians(dim, σz=false);
    hp = hyperparameters(gate, H; η=0.0, nt=200);
end;

# check time minimality by taking random z, calculating V(1)
# and using it a starting gate?
begin
    z::T = initialguess(dim)
    z::T = initialguess(dim; z0=projection!(-im*log(gate), hp))
    ξ::Vector{Float64} = basis_to_coeffs(z, hp)

    ξ::Vector{Float64} = 5*randn(n)
    coeffs_to_basis!(z, ξ, hp)

    val = evalcost!(z, hp)
end

begin
    algs = [
        ls.KrylovJL_GMRES(), 
        ls.KrylovJL_MINARES(),
        ls.KrylovJL_CG()
        ]
    α = 1e-4 # regularization; depends on d?
    gram = op.FunctionOperator(
        (w, v, z, p, t) -> gram_operator!(w, v, z, p, t), μ, μ; 
        u=z, 
        p=(hp, orthog_space, α), 
        islinear=true, 
        isconstant=true, 
        issymmetric=true, # true but numerical errors?
        isinplace=true
        )
    LP = ls.LinearProblem(gram, ∇J)
    LS = ls.init(LP, algs[1])
end;

begin
    p = (; ξ, z, hp, LS, ∇ℓ, ∇J, δξ, δz, μ);
    line_eval! = e -> begin
        ρ = 10.0^e
        μ .= ξ .- ρ .* δξ
        coeffs_to_basis!(z, μ, hp)
        evalcost!(z, hp)
    end
    line_search! = refval -> golden_section(line_eval!, refval)
end

i = 1
IFhist[1] = val
CLhist[1] = curve_length(z, H)
println("iter. 0 || cost: $(val)")


orthog_space = full_space[1:10]
for _ in 1:1_000
    i += 1

    rand_orthog_space!(orthog_space, full_space)
    # orthog_space = []

    natgrad = true
    val = descent_step!(p, line_search!, val; natgrad, orthog_space)

    if i <= length(IFhist)
        IFhist[i] = val
        CLhist[i] = curve_length(z, H)
    end
    if mod(i, 5) == 0
        println("iter. $(i) || cost: $(val)")
    end    
    if val <= 1e-6
        break
    end
end

# post-process
t, x, u = state_control(z, hp);
gate_time = curve_length(z, H)
IF = validate(gate, z, H; nt=1_500)
last_idx = min(length(IFhist), i)

begin
    idx = 1:last_idx
    fig = Figure(size=(1_000, 500))
    axs = Axis(
        fig[1, 1], 
        xscale=log10, 
        yscale=log10,
        xlabel="iterations",
        ylabel="cost",
        title=L"T = %$(CLhist[last_idx])"
        )
    scatter!(axs, idx, IFhist[1:last_idx], color=:purple, label="infidelity")
    scatter!(axs, idx, CLhist[1:last_idx], color=:teal, label="gate time")
    axislegend(axs, position=:lb)

    axs = Axis(fig[1, 2],
        xlabel=L"s",
        ylabel=L"u_j"
        )

    U = reduce(hcat, u)
    rows = eachrow(U)
    colgrad = cgrad([:green, :orange], length(rows); categorical=true)
    colors = to_color.(colgrad)
    foreach(enumerate(zip(rows, colors))) do (i, v)
        u, c = v
        lines!(axs, t, u, color=c, alpha=1/log2(i))
    end
    fig
end


### quadratic approx

# f = (p, v, t) -> begin
#     ξ, z, hp, _, _, ∇J, _, _, _ = p

#     μ .= ξ .+ t.*v
#     coeffs_to_basis!(z, μ, hp)
#     standard_gradient!(p)
#     dot(∇J, v)
# end

# d2J! = (p, v, t) -> begin
#     (f(p, v, t) - f(p, v, 0.0))/t
# end
# quadratic! = (p, t) -> begin 
#     ξ, z, hp, LS, _, ∇J, _, _, _ = p
#     coeffs_to_basis!(z, ξ, hp)
#     standard_gradient!(p)
#     LS.A.u .= z # update operator at the new z point
#     LS.b .= ∇J
#     δξ .= ls.solve!(LS)

#     s -> -f(p, δξ, 0.0)*s + 0.5*s^2 * d2J!(p, δξ, t)
# end


# fig = Figure()
# axs = Axis(fig[1, 1])
# t = 10.0 .^range(-5, 1, 100)
# for ϵ ∈ 10.0 .^ (-8:-2)
#     Q = quadratic!(p, ϵ)
#     Qt = Q.(t)
#     lines!(axs, t, Qt)
# end
# fig