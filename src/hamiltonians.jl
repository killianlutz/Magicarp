using SparseArrays

function linear_graph(dim)
    return [[i, i+1] for i in 1:dim-1]
end

function TDgraph()
    graph_adjacent = [
        [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8],
        [9, 10], [10, 11], [11, 12],
        [13, 14], [14, 15], [15, 16]
    ]
    graph_jump = [
        [1, 5], [2, 6], [3, 7], [4, 8],
        [5, 9], [6, 10], [7, 11], [8, 12],
        [9, 13], [10, 14], [11, 15], [12, 16]
    ]

    graph = (graph_adjacent, graph_jump)
    return reduce(vcat, graph)
end

function showgraph(dim, graph)
    if dim == 16
        nodes = vec([Point2(i, j) for i in 1:4, j in 1:4])
    else
        nodes = [Point2(0, i) for i in 1:dim]
    end
    edges = [Tuple(nodes[idx_pair]) for idx_pair in graph]
    
    e = length(edges)
    colors = map(edge -> first(intersect(edge...)), edges)
    
    fig = Figure()
    axs = Axis(fig[1, 1], title="$(e) edges", xlabel="i", ylabel="j")
    linesegments!(axs, edges, color=colors)
    scatter!(axs, nodes, color=:black)
    return fig
end

function hamiltonians(dim, graph::Vector; σz=false)
    n = σz ? 3*length(graph) : 2*length(graph)
    H = Vector{Matrix{ComplexF64}}(undef, n)

    x = Vector{ComplexF64}([1, 1]/sqrt(2))
    y = Vector{ComplexF64}([-im, im]/sqrt(2))
    z = Vector{ComplexF64}([1, -1]/sqrt(2))

    count = 1
    for idx_pair in graph
        i, j = idx_pair # sorted

        # σx and y
        is = [i, j]
        js = [j, i]
        for v in (x, y)
            H[count] = sparse(is, js, v, dim, dim)
            count += 1
        end

        # σz
        if σz
            is = [i, j]
            js = [i, j]

            H[count] = sparse(is, js, z, dim, dim)
            count += 1
        end
    end

    return H[1:count-1]
end

function subasis(dim)
    n = convert(Int, dim*(dim - 1)/2)
    H = zeros(ComplexF64, dim, dim)
    D = [similar(H) for _ in 1:dim-1]
    S = [similar(H) for _ in 1:n]
    A = [similar(H) for _ in 1:n]

    for k in 1:dim-1
        fill!(D[k], 0)
        for i in 1:k
            D[k][i, i] = 1
        end
        D[k][k+1, k+1] = -k
        D[k] ./= norm(D[k])
    end

    k = 1
    for i in 1:dim, j in 1:i-1
        fill!(S[k], 0)
        S[k][i, j] = 1
        S[k][j, i] = 1
        S[k] ./= norm(S[k])

        fill!(A[k], 0)
        A[k][i, j] = im 
        A[k][j, i] = -im
        A[k] ./= norm(A[k])

        k += 1
    end

    return vcat(D, S, A)
end