function sample_sphere()
    v = randn(Point3)
    α = norm(v)
    if isapprox(α, 0.0)
        v = randn(Point3)
        return v/norm(v)
    else
        return v/α
    end
end

function stereographic(v)
    x, y, z = v
    α = isapprox(z, 1.0) ? Inf64 : 1/(1 - z)
    X = α*x
    Y = α*y
    Point2(X, Y)
end

function cartesian_to_polar(v)
    polar = map(v) do (x, y)
        z = x + im*y
        r = abs(z)
        φ = angle(z)
        return [r, φ]
    end
    reduce(hcat, polar)
end

function su_to_sphere(h, pauli)
    v = map(σ -> real(dot(h, σ)), pauli)
    Point3(v)
end

function sphere_to_su(v, pauli)
    h = zero(first(pauli))
    for (vi, σi) in zip(v, pauli)
        h .+= vi .* σi
    end
    return h
end

function sample_spunitary(dim; rng=default_rng())
    A = randn(rng, ComplexF64, dim, dim)
    Q, _ = qr!(A)
    φ = angle(det(Q)) 

    return Q*exp(-im*φ/dim) 
end
