function optimize!(p, lsearch_p;
        IFabstol=1e-4, 
        nsteps=100, 
        natgrad=true, 
        dropout=0, 
        verbose_every=-2,
        past_history=missing,
        with_reset=false,
        rng=default_rng()
        )

    z, hp = p.z, p.hp 
    n = length(p.ξ)

    # optimization history
    cost = evalcost!(z, hp)
    gtime = gatetime(z, hp)

    IF = zeros(nsteps + 1) # infidelity
    GT = zeros(nsteps + 1) # gate time
    IF[1] = cost
    GT[1] = gtime
    
    # random dropout (not effective if dropout = 0)
    codimension = max(0, min(dropout, n - 1))
    full_space = collect(1:n)
    orthog_space = collect(1:codimension)

    # golden section algorithm
    line_search! = setup_linesearch(p, lsearch_p)


    if verbose_every > 0
        @printf "iter: %.2f || IF = %.4e || GT = %.4e \n" 0 cost gtime
    end

    if cost > 0.9*IFabstol
        for i in 1:nsteps
            rand_orthog_space!(orthog_space, full_space; rng)
            cost = descent_step!(p, line_search!, cost; natgrad, orthog_space, rng)
            gtime = gatetime(z, hp)

            IF[i+1] = cost
            GT[i+1] = gtime
            if isdone(i, nsteps, verbose_every, cost, gtime, IFabstol)
                break
            end

            if with_reset & (i % 50 == 0)
                if is_stuck(IF, i+1; window=50, reltol=1e-2) 
                    verbose_every > 0 ? println("///// RESET /////") : nothing
                    cost, gtime = restart(p; rng)
                end
            end
        end
    end

    keep = filter(i -> !iszero(IF[i]), 1:nsteps)
    IF = IF[keep]
    GT = GT[keep]
    if !ismissing(past_history)
        a, b = past_history
        IF = [a; IF]
        GT = [b; GT]
    end

    return (; IF, GT)
end

function optimize!(p, lsearch_p, mesh_schedule;
        IFabstol=1e-4, 
        nsteps=100, 
        natgrad=true, 
        dropout=0, 
        verbose_every=-2,
        past_history=missing,
        with_reset=false,
        rng=default_rng()
        )
    descent_p = (; IFabstol, nsteps, natgrad, dropout, verbose_every, with_reset)
    nt_list, nt_check = mesh_schedule
    retcode = 0

    # run optimizer once
    history = optimize!(p, lsearch_p; descent_p..., past_history, rng)
    
    # check validation with independant solver
    if isvalidate(p, nt_check, IFabstol, verbose_every)
        retcode = 1
    else
        # refine mesh until validation check holds
        for nt in nt_list
            verbose_every > 0 ? println("***** refining mesh || nt = $(nt)") : nothing
            p = refine_mesh!(p, nt)
            history = optimize!(p, lsearch_p; descent_p..., past_history=history, rng)

            if isvalidate(p, nt_check, IFabstol, verbose_every)
                retcode = 1
                break
            end
        end
    end

    if retcode == 0
        println("###### failure || IFval > IFabstol")
    else
        println("###### success || IFval <= IFabstol")
    end

    return (; history, retcode)
end

function isvalidate(p, nt_check, IFabstol, verbose)
    IFval = validate(p.z, p.hp; nt=nt_check)
    if verbose > 0
        @printf ">>>>> validation || IF = %.4e \n" IFval
    end
    return IFval <= IFabstol
end


function descent_step!(p, line_search!, cost; natgrad=false, orthog_space=[], rng=default_rng())
    _, z, hp, LS, _, ∇J, δξ, _, _ = p

    # loss gradient into coordinates M_d(C)
    ∇J = standard_gradient!(p)
    subspace_projection!(∇J, orthog_space)

    # gradient or nat. gradient step calculation
    if natgrad
        LS.A.p = (hp, orthog_space, last(LS.A.p))
        LS.A.u .= z
        LS.b .= ∇J
        δξ .= LinS.solve!(LS) # solve HJ*x = ∇J -> ascent direction
    else
        δξ .= ∇J
    end

    δξ ./= norm(δξ)

    cost = next_point!(p, line_search!, cost; rng)

    return cost
end

function next_point!(p, line_search!, cost; rng=default_rng())
    ξ = p.ξ
    δξ = p.δξ
    z = p.z
    hp = p.hp

    # linesearch
    e, next_val = line_search!(cost)
    ρ = 10.0^e

    # update if decrease
    decrease = cost - next_val
    explore = decrease <= 0 && rand(rng) <= cost^2

    if decrease > 0 || explore
        ξ .-= ρ .* δξ
        coeffs_to_basis!(z, ξ, hp)
        cost = next_val
    end

    return cost
end

function refine_mesh!(p, nt)
    # create new namedTuple hyperparameters hp
    # refining mesh without allocating new arrays
    ξ, z, hp = p.ξ, p.z, p.hp
    
    H, q, basis, η, PA = hp.H, hp.q, hp.basis, hp.η, hp.PA
    hp = (; H, q, basis, η, nt, PA) # ONLY CHANGE: finer mesh dt=1/nt

    ∇ℓ, ∇J, δξ, δz, μ = p.∇ℓ, p.∇J, p.δξ, p.δz, p.μ
    PA = (; ∇ℓ, ∇J, δξ, δz, μ) # kept identical
    LS = p.LS # kept identical

    p = (; ξ, z, hp, LS, PA...)
    return p
end

function is_stuck(IF, last_index; window=20, reltol=1.0)
    i = last_index
    if i >= window
        mean = sum(IF[k] for k in i-window+1:i)/window
        lastIF = IF[i]
        # not stuck if sufficient decrease: y - x >= τ*x
        return mean - lastIF < reltol*mean
    else
        return false
    end
end

function restart(p; rng=default_rng())
    ξ, z, hp = p.ξ, p.z, p.hp

    randn!(rng, ξ)
    ξ ./= sqrt(length(ξ))
    coeffs_to_basis!(z, ξ, hp)
    
    cost = evalcost!(z, hp)
    gtime = gatetime(z, hp)
    return (cost, gtime)
end

function preallocate(dim, T)
    n = dim^2 - 1

    ∇ℓ::T = Matrix{ComplexF64}(undef, dim, dim)
    ∇J::Vector{Float64} = Vector{Float64}(undef, n)
    δξ::Vector{Float64} = Vector{Float64}(undef, n)
    δz::T = Matrix{ComplexF64}(undef, dim, dim)
    μ::Vector{Float64} = similar(δξ)

    return (; ∇ℓ, ∇J, δξ, δz, μ)
end

function setup_leastsquares(z, hp, PA; regularization=1e-3, alg=LinS.KrylovJL_GMRES())
    # matrix-free algorithms:
    # LinS.KrylovJL_GMRES(), 
    # LinS.KrylovJL_MINARES(),
    # LinS.KrylovJL_CG()

    x = PA.μ
    b = PA.∇J
    α = regularization 
    orthog_space = Vector{Int64}()

    # abstract operator representing gram matrix of linearization about z
    A = Op.FunctionOperator(
        (w, v, z, p, t) -> gram_operator!(w, v, z, p, t), x, x; 
        u=z, 
        p=(hp, orthog_space, α), 
        islinear=true, 
        isconstant=true, 
        issymmetric=true,
        isinplace=true
        )
    LP = LinS.LinearProblem(A, b)
    LS = LinS.init(LP, alg) # preallocates arrays

    return LS
end

function setup_linesearch(p, lsearch_p)
    ξ, z, hp, μ, δξ = p.ξ, p.z, p.hp, p.μ, p.δξ

    # evaluate infidelity along the negative span of δξ passing by ξ
    line_eval! = x -> begin
        ρ = 10.0^x
        μ .= ξ .- ρ .* δξ
        coeffs_to_basis!(z, μ, hp)
        evalcost!(z, hp)
    end
    
    line_search! = refval -> golden_section(line_eval!, lsearch_p; reference_value=refval)

    return line_search!
end

function golden_section(f::Function, p; reference_value::Real=Inf)
    interval, abstol, max_iterations = p
    
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

function isdone(i, nsteps, every, cost, gtime, IFabstol)
    if every > 0 && mod(i, every) == 0
        fraction = i/nsteps
        @printf "iter: %.2f || IF = %.4e || GT = %.4e \n" fraction cost gtime
    end    
    if cost <= IFabstol
        return true
    else
        return false
    end
end

function postprocess(t, u, IFval, IF, GT; fig=Figure(size=(1_000, 500)))
    # history
    gtime = round(last(GT), digits=3)
    axs = Axis(
        fig[1, 1], 
        xscale=log10, 
        yscale=log10,
        xlabel="iterations",
        ylabel="cost",
        title=L"T = %$(gtime)"
        )

    scatterlines!(axs, IF, color=:purple, label="infidelity")
    scatterlines!(axs, GT, color=:teal, label="gate time")
    hlines!(axs, IFval, color=:purple, label="validation", linestyle=:dashdot)
    axislegend(axs, position=:lt)

    # controls
    m = length(first(u))
    axs = Axis(fig[1, 2],
        xlabel=L"t/T",
        ylabel=L"u_j",
        title=L"1 \leq j \leq %$(m)"
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