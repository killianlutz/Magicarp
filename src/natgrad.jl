
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

function loss_grad!(grad, x, hp)
    q = hp.q
    # squared infidelity
    # dotp = dot(q, x)
    # grad .= -2 .* dotp .* q

    # ms = -sign(dot(q, x))
    # grad .= ms .* q

    # absolute infidelity
    # d = size(q, 1)
    # dotp = dot(q, x)
    # grad .= (dotp ./ abs(dotp)) .* q./sqrt(d)
    # dotp = real(dot(grad, x))
    # grad .= -1 .* (grad .- dotp .* x./d)

    # norm of error
    grad .= x .- q

    return grad
end

function standard_gradient!(p)
    _, z, hp, _, ∇ℓ, ∇J, _, δz, _ = p

    x = final_state!(z, hp)
    loss_grad!(∇ℓ, x, hp)
    vjp!(δz, z, ∇ℓ, hp)
    basis_to_coeffs!(∇J, δz, hp)

    return ∇J
end