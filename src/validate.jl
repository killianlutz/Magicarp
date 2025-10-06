function feedback(z, x, H)
    map(H) do h
        real(dot(x, h, x*z))
    end
end

function schrodinger!(dx, x, p, _)
    fill!(dx, 0)
    z, H = p

    y = x*z
    for h in H
        c = -im*real(dot(x, h, y))
        mul!(dx, h, x, c, 1)
    end

    return dx
end

function solve_schrodinger(z, H, nt, saveat)
    dt = 1/nt
    dtmax = dt
    p = (z, H)
    x0 = one(z)
    tspan = (0.0, 1.0)

    ode = ODEProblem(schrodinger!, x0, tspan, p)
    return solve(ode; saveat, dt, dtmax)
end

function state_control(z, hp; nt=hp.nt)
    state_control(z, hp.H, nt)
end

function state_control(z, H, nt)
    saveat = range(0.0, 1.0, 200)
    sol = solve_schrodinger(z, H, nt, saveat)

    t = sol.t
    x = sol.u
    u = map(x) do xt
        feedback(z, xt, H)
    end

    return (; t, x, u)
end

function validate(z, hp; nt=hp.nt)
    validate(hp.q, z, hp.H; nt)
end

function validate(q, z, H; nt=300)
    saveat = [1.0]
    sol = solve_schrodinger(z, H, nt, saveat)
    x1 = last(sol.u)
    infidelity(x1, q)
end