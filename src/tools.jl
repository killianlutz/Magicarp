
function gatetime(z, hp)
    H = hp.H
    sqrt(sum(abs2, real(dot(h, z)) for h in H))
end

function infidelity(x1, q)
    dim = size(x1, 1)
    fidelity = abs(dot(q, x1))/dim
    abs(1.0 - fidelity)
end

function grad_infidelity(x1, q)
    # gradient of 1 - abs2(infidelity)/d^2
    (-2/length(q))*dot(q, x1)*q
end

function initialguess(dim; z0=missing, rng=default_rng())
    z = zeros(ComplexF64, dim, dim)
    if ismissing(z0)
        z .= randn(rng, ComplexF64, dim, dim)
    else
        z .= z0
    end
    projhermitian!(z)
    projtraceless!(z)
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

function projhermitian!(dx, x)
    dx .= (x .+ x')./2 
end

function projtraceless!(x)
    dim = size(x, 1)
    α = tr(x)/dim
    x .-= α .* I(dim)
end

function basis_to_coeffs(x, hp)
    return [real(dot(E, x)) for E in hp.basis]
end

function basis_to_coeffs!(ξ, z, hp)
    su_basis = hp.basis
    for (i, h) in enumerate(su_basis)
        ξ[i] = real(dot(h, z))
    end
    return ξ
end

function coeffs_to_basis!(dz, ξ, hp)
    fill!(dz, 0)
    for (c, E) in zip(ξ, hp.basis)
        dz .+= c .* E
    end
    return dz
end

function subspace_projection!(ξ, orthog_space)
    for i in orthog_space
        ξ[i] = 0.0
    end

    return ξ
end

function rand_orthog_space!(orthog_space, full_space; rng=default_rng())
    if !isempty(orthog_space)
        randperm!(full_space)

        for i in eachindex(orthog_space)
            orthog_space[i] = full_space[i]
        end
    end

    return nothing
end
