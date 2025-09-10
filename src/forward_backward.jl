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

    # x0::T = gate
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

function hyperparameters(gate::T, H; η=0.0, nt=100) where T<:AbstractMatrix
    dim = size(gate, 1)
    PA = preallocate(dim, T, T) # ASSUME: V type = T type
    basis = subasis(dim)
    hp = (; H, q=gate, basis, η, nt, PA)
    return hp
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

function forward_evolution!(z, hp)
    v = hp.PA.v
    x0 = hp.PA.x0
    temp = hp.PA.xtemp
    p = (; H=hp.H, z, temp)

    v.x .= heun!(state_ode!, x0, p, hp.nt, hp.PA.xs)
end

function backward_evolution!(z, dir, hp)
    v = hp.PA.v
    temp1 = hp.PA.vtemp1
    temp2 = hp.PA.vtemp2
    p = (; H=hp.H, z, temp1, temp2)
    
    terminal_adjoint!(dir, hp) 
    v1 = heun!(grad_ode!, v, p, hp.nt, hp.PA.vs)
    projhermitian!(v.g, v1.g)
end

function terminal_adjoint!(dir, hp)
    v = hp.PA.v

    # state
    # v.x is set appropriately in forward_evolution!
    # adjoint
    mul!(v.a, v.x', dir, 1, 0) # = v.x'*dir
    # gradient
    fill!(v.g, 0)

    return v
end

function cost!(hp)
    Pz = hp.PA.Pz
    x1 = hp.PA.v.x
    q = hp.q
    η  = hp.η
    dim2 = length(x1)

    η*sum(abs2, Pz) + (1 - η)*abs(1 - abs2(dot(x1, q))/dim2)
end

function evalcost!(z, hp)
    forward_evolution!(z, hp) # updates v.x
    projection!(z, hp)
    cost!(hp)
end