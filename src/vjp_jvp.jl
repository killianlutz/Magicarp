
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
