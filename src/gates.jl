using LinearAlgebra
using FFTW

# H, SUMX, T gates

function Xd(dim; su=true)
    v = ones(dim - 1)
    X = diagm(-1 => v, dim-1 => [1])
    X = Matrix{ComplexF64}(X)

    su ? X = toSU(X) : nothing
    return X
end

function SUMX(dim; su=true)
    sdim = sqrt(dim)
    if !isinteger(sdim)
        return "dim is not a square"
    else
        sdim = convert(Int, sdim)
    end

    X = Xd(sdim)
    Z = zero(X)
    Id = one(X)
    S = [
        Id Z Z Z; 
         Z X Z Z;
         Z Z X Z; 
         Z Z Z X
         ]

    su ? S = toSU(S) : nothing
    return S
end

function Hd(dim; su=true)
    Id = convert(Matrix{ComplexF64}, I(dim))
    H = sqrt(dim)*ifft(Id, 1)

    su ? H = toSU(H) : nothing
    return H
end

function Td(dim; su=true)
    t = [cis(2π/dim * (i - 1)/4) for i in 1:dim]
    T = diagm(t)

    su ? T = toSU(T) : nothing
    return T
end

function sample_unitary(dim; rng=default_rng())
    A = randn(rng, ComplexF64, dim, dim)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R

    foreach(eachcol(Q), diag(R)) do col, r
        col .*= r/abs(r)
    end
    return Q
end

function toSU(Q)
    dim = size(Q, 1)
    φ = angle(det(Q)) 
    return Q*cis(-φ/dim) 
end

function sample_spunitary(dim; rng=default_rng())
    toSU(sample_unitary(dim; rng))
end