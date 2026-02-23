begin
    dim = 4         # qudit dimension
    n = dim^2 - 1   # su(d) dimension
    nt = 100        # uniform mesh
    scheme = :RK4  # epxlicit ode solver ∈ [:heun, :RK4]
    T = Matrix{ComplexF64} # matrix type

    # mesh refining if needed
    nt_list = [25, 50, 50, 75, 100, 125, 150, 175, 200] # = Vector{Int64}() <-> no refining
    nt_list = max.(nt, nt_list)
    nt_check = 2_000                 # for validation, dt = 1/nt_check
    mesh_schedule = (nt_list, nt_check)

    # natural gradient 
    natgrad = true        # standard versus natural gradient step
    regularization = 1e-3 # least squares |Ax - b|^2 + reg*|x|^2

    # line search
    lsearch_p = (; 
        interval = (-4.0, 0.0), # stepsize ∈ 10.^interval
        abstol = 1e-1,
        max_iterations = 200
        )

    # descent
    nsteps = 100    # iterations per mesh size
    IFabstol = 1e-4
    dropout = 0       # = 0  <-> no dropout
    verbose_every = 5 # = -2 <-> no verbose
    with_reset = true # start over gain if IF threshold over 100 iterations
    descent_p = (; IFabstol, nsteps, natgrad, dropout, verbose_every, with_reset)
end
