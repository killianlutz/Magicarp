begin
    dim = 4        # qudit dimension
    n = dim^2 - 1   # su(d) dimension
    nt = 200        # uniform mesh
    T = Matrix{ComplexF64}

    # natural gradient 
    natgrad = true        # standard versus natural gradient step
    regularization = 1e-3 # least squares |Ax - b|^2 + reg*|x|^2

    # line search
    lsearch_p = (; 
        interval = (-6.0, 1.0), # stepsize ∈ 10.^interval
        abstol = 5e-1,
        max_iterations = 200
        )

    # descent
    nsteps = 1_000       # iteration per mesh size
    IFabstol = 1e-4
    dropout = 0       # = 0  <-> no dropout
    verbose_every = 5 # = -2 <-> no verbose
    with_reset = true # start over gain if IF threshold over 100 iterations
    descent_p = (; IFabstol, nsteps, natgrad, dropout, verbose_every, with_reset)

    # mesh refining if needed
    nt_list = [400, 600, 800, 1_000] # = Vector{Int64}() <-> no refining
    nt_check = 2_000                 # for validation, dt = 1/nt_check
    mesh_schedule = (nt_list, nt_check)
end
