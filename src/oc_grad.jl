using LinearAlgebra
using Parameters: @with_kw
using SparseArrays: sparse, SparseMatrixCSC
using FFTW: ifft
using Base.Threads

@with_kw mutable struct OdePoint3{T<:AbstractMatrix,V<:AbstractMatrix}
    x::T
    a::T
    g::V
end

function Base.similar(v::OdePoint3)
    x = similar(v.x)
    a = similar(v.a)
    g = similar(v.g)

    T = typeof(x)
    V = typeof(g)
    OdePoint3{T, V}(x, a, g)
end


function preallocate(dim, T, V)
    U = OdePoint3{T, V}

    x0::T = convert(T, I(dim))
    v::U = OdePoint3{T, V}(similar(x0), similar(x0), convert(V, similar(x0)))
    xs::Vector{T} = [similar(x0) for _ in 1:4]
    vs::Vector{U} = [similar(v) for _ in 1:4]
    Pz::V = similar(v.g)
    dJ::V = similar(v.g)
    zρ::V = similar(v.g)
    xtemp::T = similar(x0)
    vtemp1::T = similar(v.x)
    vtemp2::V = similar(v.g)
    preallocs = (; x0, v, xs, vs, Pz, dJ, zρ, xtemp, vtemp1, vtemp2)

    return preallocs
end

function projection!(z, hp)
    fill!(hp.PA.Pz, 0)
    for h in hp.H
        c = real(dot(h, z))
        hp.PA.Pz .+= c .* h
    end
    return hp.PA.Pz
end

function projhermitian!(x)
    x .= (x .+ x')./2 
end

function projtraceless!(x)
    dim = size(x, 1)
    α = tr(x)/dim
    x .-= α .* I(dim)
end


function state_ode!(dx, x, p, t)
    y = p.temp
    mul!(y, x, p.z) # = x*z
    mul!(dx, y, x') # = x*z*x'
    fill!(y, 0)

    for h in p.H
        c = -im*real(dot(h, dx))
        y .+= c .* h
    end
    mul!(dx, y, x)

    return dx
end

function grad_ode!(dv, v, p, t)
    # preallocations
    yz = p.temp1
    ya = p.temp2
    mul!(dv.g, v.x, p.z) # x*z
    mul!(yz, dv.g, v.x') # x*z*x'
    mul!(dv.a, v.x, v.a) # x*a
    mul!(ya, dv.a, v.x') # x*a*x'

    fill!(dv.x, 0)
    fill!(dv.a, 0)
    fill!(dv.g, 0)

    for h in p.H 
        dot_z = real(dot(h, yz)) # Re tr(h_j(s)*z)
        dot_a = imag(dot(h, ya)) # Im tr(h_j(s)*a)

        dv.x .+= im .* dot_z.*h
        dv.a .-= 2 .* dot_a.*h
        dv.g .-= dot_a.*h
    end

    # dx
    mul!(yz, dv.x, v.x)
    dv.x .= yz
    # da
    mul!(ya, dv.a, v.x)
    mul!(yz, ya, p.z)
    mul!(dv.a, v.x', yz)
    # dg
    mul!(yz, dv.g, v.x)
    mul!(dv.g, v.x', yz)

    return dv
end

function heun!(f, y0, p, nt, ys)
    # f(dy, y, p, t)
    dt = 1.0/(nt - 1)
    y1, y2, y3, y = ys # preallocations of typeof(y)

    t = 0.0
    y .= y0
    for _ in 1:nt-1
        f(y1, y, p, t)
        y2 .= y .+ dt .* y1

        t += dt
        f(y3, y2, p, t)

        y .+= (y1 .+ y3).*dt./2
    end

    return y
end

function heun!(velocity, y0::OdePoint3, p, nt, ys)
    # velocity(dy, y, p, t)
    dt = 1.0/(nt - 1)    
    y1, y2, y3, y = ys # preallocations of typeof(y)

    t = 0.0
    y.x .= y0.x
    y.a .= y0.a
    y.g .= y0.g
    for _ in 1:nt-1
        velocity(y1, y, p, t) # at t_{i}
        y2.x .= y.x .+ dt .* y1.x
        y2.a .= y.a .+ dt .* y1.a
        y2.g .= y.g .+ dt .* y1.g

        t += dt
        velocity(y3, y2, p, t) # at t_{i+1}
        y.x .+= (y1.x .+ y3.x).*dt./2
        y.a .+= (y1.a .+ y3.a).*dt./2
        y.g .+= (y1.g .+ y3.g).*dt./2
    end

    return y
end

function infidelity(x1, q)
    dim = size(x1, 1)
    fidelity = abs2(dot(q, x1))/dim^2
    abs(1.0 - fidelity)
end

function cost!(hp)
    Pz = hp.PA.Pz
    x1 = hp.PA.v.x
    0.5*sum(η*abs2(x) + (1 - η)*abs2(y - z) for (x, y, z) in zip(Pz, x1, hp.q))
end

function grad_cost!(hp)
    dJ = hp.PA.dJ
    Pz = hp.PA.Pz
    g = hp.PA.v.g
    dJ .= η .* Pz .+ (1 .- η) .* g

    α = norm(dJ)
    if !isapprox(α, 0.0)
        dJ ./= α
    end

    return dJ
end

function forward_evolution!(z, hp)
    v = hp.PA.v
    x0 = hp.PA.x0
    temp = hp.PA.xtemp
    p = (; H=hp.H, z, temp)

    v.x .= heun!(state_ode!, x0, p, hp.nt, hp.PA.xs)
end

function backward_evolution!(z, hp)
    v = hp.PA.v
    temp1 = hp.PA.vtemp1
    temp2 = hp.PA.vtemp2
    p = (; H=hp.H, z, temp1, temp2)
    
    terminal_adjoint!(hp) 
    v1 = heun!(grad_ode!, v, p, hp.nt, hp.PA.vs)
    v.g .= projhermitian!(v1.g)
end

function update!(z, ρ, hp)
    dJ = hp.PA.dJ
    z .-= ρ .* dJ
end

function terminal_adjoint!(hp)
    q = hp.q
    v = hp.PA.v
    x0 = hp.PA.x0

    # state
    # v.x is set appropriately in forward_evolution!
    # adjoint
    mul!(v.a, v.x', q, -1, 0)
    v.a .+= x0 # v.a .= x0 .- v.x'*q
    # gradient
    fill!(v.g, 0)

    return v
end

function gradstep!(z, hp)
    forward_evolution!(z, hp) # returns v.x(1)
    backward_evolution!(z, hp) # returns v.g(1)
    projection!(z, hp)
    J = cost!(hp)
    grad_cost!(hp)
    ρ = linesearch!(z, hp; reference_value=J)
    update!(z, ρ, hp)
    return (ρ, J)
end

function golden_section_search(f::Function, interval::NTuple{2,<:Real}, abstol::Real, max_iterations::Integer; reference_value::Real=Inf)
    a, b = interval
    γ = oftype(a, (1 + sqrt(5)) / 2) # golden number

    c = b - (γ - 1) * (b - a)
    d = a + (γ - 1) * (b - a)
    left_value = f(c)
    right_value = f(d)

    count = 1
    while (count <= max_iterations) && (b - a > abstol)
        if left_value <= right_value
            b = d; d = c;
            c = b - (γ - 1) * (b - a)
            # only 1 evaluation of f instead of 2
            right_value = left_value
            left_value = f(c)
        elseif left_value > right_value
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

function linesearch!(z, hp; reference_value=Inf)
    zρ = hp.PA.zρ
    dJ = hp.PA.dJ
    ρ_max = 1e1
    interval = (0.0, ρ_max)
    abstol = 1e-10
    max_iter = 200

    ls = ρ -> begin
        zρ .= z .- ρ .* dJ
        evalcost!(zρ, hp)
    end

    ρ, val = golden_section_search(ls, interval, abstol, max_iter; reference_value)
    if val - reference_value ≥ -1e-12
        ρ_max = ρ_max/100
        interval = (0.0, ρ_max)
        ρ, val = golden_section_search(ls, interval, abstol, max_iter; reference_value)
    end

    return ρ
end

function evalcost!(z, hp)
    forward_evolution!(z, hp) # updates v.x
    projection!(z, hp)
    cost!(hp)
end

function metrics(z, gate, hp)
    v = hp.PA.v
    Pz = hp.PA.Pz
    forward_evolution!(z, hp) # updates v.x
    IF = infidelity(v.x, gate)
    projection!(z, hp)
    CL = norm(Pz)
    return (; CL, IF)
end

function heunstep(f, y0, dt)
    # \dot{x} = f(x)
    y1 = f(y0)
    y2 = y0 .+ dt .* y1
    y3 = f(y2)
    y4 = y0 .+ (y1 .+ y3).*dt./2

    return y4
end

function feedback(z, x, H)
    map(H) do h
        real(dot(x, h, x*z))
    end
end

function state_control(z, hp)
    x0 = hp.PA.x0
    H = hp.H
    nt = hp.nt

    f = x -> begin
        dx = similar(x)
        fill!(dx, 0)
        y = x*z
        for h in H
            c = -im*real(dot(x, h, y))
            mul!(dx, h, x, c, 1)
        end
        return dx
    end


    t = range(0.0, 1.0, nt)
    x = [similar(x0) for _ in 1:nt]
    u = Vector{Vector{Float64}}(undef, nt)

    dt = t[2] - t[1]
    x[1] .= x0    
    u[1] = feedback(z, x[1], H)
    for t in 1:nt-1
        x[t+1] = heunstep(f, x[t], dt)
        u[t+1] = feedback(z, x[t+1], H)
    end

    return (; t, x, u)
end

function pauli_vector(z, pauli)
    map(σ -> real(dot(z, σ)), pauli)
end

function hamiltonians(dim; σz=false)
    m = σz ? 3(dim - 1) : 2(dim - 1)
    H = Vector{S}(undef, m)

    vxy = [ComplexF64.([1, 1]/sqrt(2)), ComplexF64.([-im, im]/sqrt(2))]
    vz = ComplexF64.([1, -1]/sqrt(2))

    count = 1
    for k in 1:dim-1
        for v in vxy
            i = [k, k+1]
            j = [k+1, k]

            H[count] = sparse(i, j, v, dim, dim)
            count += 1
        end

        if σz
            i = [k, k+1]
            j = [k, k+1]

            H[count] = sparse(i, j, vz, dim, dim)
            count += 1
        end
    end
    return H
end

function TDhamiltonians(; σz=false)
    dim = 16
    H = Vector{S}(undef, dim^2)

    vxy = [ComplexF64.([1, 1]/sqrt(2)), ComplexF64.([-im, im]/sqrt(2))]
    vz = ComplexF64.([1, -1]/sqrt(2))

    count = 1
    for n in 1:3
        k_min = 4*(n - 1) + 1
        k_max = k_min + 4 - 1

        for k in k_min:k_max
            for v in vxy
                is = [[k, k+1], [k, k+4]]
                js = [[k+1, k], [k+4, k]]

                for (i, j) in zip(is, js)
                    H[count] = sparse(i, j, v, dim, dim)
                    count += 1
                end
            end

            if σz
                is = [[k, k+1], [k, k+4]]
                js = [[k, k+1], [k, k+4]]

                for (i, j) in zip(is, js)
                    H[count] = sparse(i, j, vz, dim, dim)
                    count += 1
                end
            end
        end
    end
    return H[1:count-1]
end

function QFT(dim)
    α = ComplexF64(sqrt(dim))
    x = Matrix(α*I(dim))
    y = map(ifft, eachcol(x))
    reduce(hcat, y)
end