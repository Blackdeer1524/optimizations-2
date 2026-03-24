using Optim
using PyPlot


ENV["PYTHON"] = "/home/blackdeer/anaconda3/bin/python"
import Pkg
Pkg.build("PyCall")

# Barrier functions
logarithmic_barrier(g) = x -> -log(-g(x))
reverse_barrier(g) = x -> -1 / g(x)

function barrier_functions_method_multi(f, gs, x0;
    G=reverse_barrier,
    r_sequence_list=nothing,
    method=NelderMead(),
    iterations=1000,
    verbose=false)

    for (i, g) in enumerate(gs)
        if g(x0) > 0
            error("g$i(x0) <= 0 does not hold.")
        end
    end

    if r_sequence_list === nothing
        m = length(gs)
        base_values = 10.0 .^ (0:-1:-7)
        r_sequence_list = [[fill(r, m)...] for r in base_values]
    end

    history = []
    x = copy(x0)

    for (idx, r_vec) in enumerate(r_sequence_list)
        function Q(x)
            val = f(x)
            for (i, g) in enumerate(gs)
                if g(x) > 0
                    return Inf
                end
                val += r_vec[i] * G(g)(x)
            end
            return val
        end

        result = optimize(Q, x, method, Optim.Options(iterations=iterations))
        x_new = result.minimizer
        Q_min = result.minimum
        f_min = f(x_new)

        push!(history, Dict(
            :r_vec => copy(r_vec),
            :x => x_new,
            :f_x => f_min,
            :Q_x => Q_min
        ))

        if verbose
            r_str = join(round.(r_vec, sigdigits=2), ", ")
            println("r=[$r_str]: x=$x_new, f(x)=$f_min")
        end

        x = x_new
    end

    return history
end

# -----------------------------------------------------------------------------
# 1D Visualization
# -----------------------------------------------------------------------------
function plot_barrier_1d_multi(f, gs, history;
    x_range=(1.5, 5.0),
    title="Barrier method with multiple parameters")

    r_vecs = [entry[:r_vec] for entry in history]
    n = length(r_vecs)
    x_grid = collect(range(x_range[1], x_range[2], length=1000))
    f_vals = [f([x]) for x in x_grid]

    cmap = get_cmap("plasma")
    colors = [cmap((i - 1) / max(n - 1, 1)) for i in 1:n]

    fig, ax = subplots(figsize=(16, 12))
    ax.set_xlabel("x")
    ax.set_ylabel("f(x), Q_r(x)")
    ax.set_title(title)
    ax.set_ylim(0, 10)

    ax.plot(x_grid, f_vals, label="f(x)", lw=3, color="black")

    x_feasible = filter(x -> x >= 2, x_grid)
    f_feasible = [f([x]) for x in x_feasible]
    ax.fill_between(x_feasible, 0, f_feasible,
        alpha=0.2, color="green", label="Feasible region")

    ax.axvline(x=2.0, linestyle="--", color="red", label="Boundary", lw=2)

    for (i, entry) in enumerate(history)
        r_vec = entry[:r_vec]
        x_min = entry[:x][1]
        r_str = join(round.(r_vec, sigdigits=2), ", ")
        f_min_val = entry[:f_x]
        c = colors[i]

        function Q_r(x)
            val = f([x])
            for (j, g) in enumerate(gs)
                if g([x]) > 0
                    return Inf
                end
                val += r_vec[j] * reverse_barrier(g)([x])
            end
            return val
        end

        Q_vals = [x >= 2 ? Q_r(x) : NaN for x in x_grid]
        Q_min_val = Q_r(x_min)

        ax.plot(x_grid, Q_vals, label="r = [$r_str]", lw=2, color=c, alpha=0.7)
        ax.scatter([x_min], [Q_min_val], color=[c], s=64, zorder=5)

        if i == 1
            ax.scatter([x_min], [f_min_val], color=[c], s=100, marker="*", zorder=5,
                label="Projected minima on f(x)")
        else
            ax.scatter([x_min], [f_min_val], color=[c], s=100, marker="*", zorder=5)
        end
    end

    ax.scatter([2.0], [f([2.0])], color="red", s=150, marker="*",
        label="Theoretical optimum", zorder=6)
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig
end

# -----------------------------------------------------------------------------
# 2D Visualization for box constraints
# -----------------------------------------------------------------------------
function plot_barrier_2d_multi(f, gs, history;
    x_range=(1.5, 5.0),
    y_range=(1.5, 5.0),
    title="Barrier method: f(x,y)=x²+y²",
    elevation=30,
    azimuth=45,
    show_q_functions=true)

    n_points = 50
    x_vals = collect(range(x_range[1], x_range[2], length=n_points))
    y_vals = collect(range(y_range[1], y_range[2], length=n_points))

    # X[i,j] = x_vals[i], Y[i,j] = y_vals[j]  →  shape (n_x, n_y)
    X = [x for x in x_vals, y in y_vals]
    Y = [y for x in x_vals, y in y_vals]

    f_grid = [f([x, y]) for x in x_vals, y in y_vals]
    for i in 1:n_points, j in 1:n_points
        if !(x_vals[i] >= 2 && y_vals[j] >= 2)
            f_grid[i, j] = NaN
        end
    end

    x_path = [entry[:x][1] for entry in history]
    y_path = [entry[:x][2] for entry in history]
    z_path_f = [entry[:f_x] for entry in history]
    z_path_q = [entry[:Q_x] for entry in history]
    r_vecs = [entry[:r_vec] for entry in history]

    n = length(r_vecs)
    cmap_thermal = get_cmap("plasma")
    colors = [cmap_thermal((i - 1) / max(n - 1, 1)) for i in 1:n]

    fig = figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y), Q_r(x,y)")
    ax.view_init(elev=elevation, azim=azimuth)

    ax.plot_surface(X, Y, f_grid, alpha=0.5, cmap="viridis")

    if show_q_functions
        for (idx, entry) in enumerate(history)
            r_vec = entry[:r_vec]
            q_grid = fill(NaN, n_points, n_points)

            for i in 1:n_points, j in 1:n_points
                x = x_vals[i]
                y = y_vals[j]
                if x >= 2 && y >= 2
                    val = f([x, y])
                    for (k, g) in enumerate(gs)
                        val += r_vec[k] * reverse_barrier(g)([x, y])
                    end
                    q_grid[i, j] = val
                end
            end

            r_str = join(round.(r_vec, sigdigits=2), ", ")
            ax.plot_surface(X, Y, q_grid, alpha=0.3, color=colors[idx],
                label="r = [$r_str]")
        end
    end

    x_boundary_y = y_vals
    x_boundary_z = [f([2.0, y]) for y in y_vals]
    ax.plot(fill(2.0, n_points), x_boundary_y, x_boundary_z,
        linestyle="--", color="red", lw=2, label="x = 2 boundary")

    y_boundary_x = x_vals
    y_boundary_z = [f([x, 2.0]) for x in x_vals]
    ax.plot(y_boundary_x, fill(2.0, n_points), y_boundary_z,
        linestyle="--", color="red", lw=2, label="y = 2 boundary")

    for i in 1:length(x_path)
        c = colors[i]
        if i == 1
            ax.scatter([x_path[i]], [y_path[i]], [z_path_q[i]],
                color=[c], s=64, zorder=5, label="Minima on Q functions")
            ax.scatter([x_path[i]], [y_path[i]], [z_path_f[i]],
                color=[c], s=100, marker="*", zorder=5, label="Projected minima on f(x,y)")
        else
            ax.scatter([x_path[i]], [y_path[i]], [z_path_q[i]], color=[c], s=64, zorder=5)
            ax.scatter([x_path[i]], [y_path[i]], [z_path_f[i]], color=[c], s=100, marker="*", zorder=5)
        end
    end

    if length(x_path) > 1
        ax.plot(x_path, y_path, z_path_q,
            color="gray", lw=2, linestyle=":", label="Path of Q minima")
    end

    ax.scatter([2.0], [2.0], [f([2.0, 2.0])], color="red", s=150, marker="*",
        label="Theoretical optimum (2,2)", zorder=6)

    ax.legend()
    fig.tight_layout()

    return fig
end

# -----------------------------------------------------------------------------
# 2D Visualization for parabolic constraint
# -----------------------------------------------------------------------------
function plot_barrier_2d_parabolic(f, gs, history;
    x_range=(-2.0, 2.0),
    y_range=(1.0, 5.0),
    title="Barrier method: f(x,y)=x²+y², y ≥ x² + 1",
    elevation=30,
    azimuth=45,
    show_q_functions=true)

    n_points = 400
    x_vals = collect(range(x_range[1], x_range[2], length=n_points))
    y_vals = collect(range(y_range[1], y_range[2], length=n_points))

    # f_grid[j,i] = f at (x_vals[i], y_vals[j])  →  shape (n_y, n_x)
    f_grid = fill(NaN, n_points, n_points)
    for i in 1:n_points, j in 1:n_points
        x = x_vals[i]
        y = y_vals[j]
        if y >= x^2 + 1 - 1e-10
            f_grid[j, i] = f([x, y])
        end
    end

    x_path = [entry[:x][1] for entry in history]
    y_path = [entry[:x][2] for entry in history]
    z_path_f = [entry[:f_x] for entry in history]
    z_path_q = [entry[:Q_x] for entry in history]
    r_vecs = [entry[:r_vec] for entry in history]

    n = length(r_vecs)
    cmap_thermal = get_cmap("plasma")
    colors = [cmap_thermal((i - 1) / max(n - 1, 1)) for i in 1:n]

    # X[j,i] = x_vals[i], Y[j,i] = y_vals[j]  →  shape (n_y, n_x), matches f_grid
    X = [x for y in y_vals, x in x_vals]
    Y = [y for y in y_vals, x in x_vals]

    fig = figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y), Q_r(x,y)")
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_zlim(0, 10)

    ax.plot_surface(X, Y, f_grid, alpha=0.5, cmap="viridis")

    if show_q_functions
        for (idx, entry) in enumerate(history)
            r_vec = entry[:r_vec]
            q_grid = fill(NaN, n_points, n_points)

            for i in 1:n_points, j in 1:n_points
                x = x_vals[i]
                y = y_vals[j]
                if y >= x^2 + 1 - 1e-10
                    val = f([x, y])
                    for (k, g) in enumerate(gs)
                        val += r_vec[k] * logarithmic_barrier(g)([x, y])
                    end
                    q_grid[j, i] = val
                end
            end

            r_str = join(round.(r_vec, sigdigits=2), ", ")
            ax.plot_surface(X, Y, q_grid, alpha=0.3, color=colors[idx],
                label="Q_r, r = [$r_str]")
        end
    end

    boundary_x = x_vals
    boundary_y = [x^2 + 1 for x in boundary_x]
    boundary_z = [f([x, x^2 + 1]) for x in boundary_x]
    ax.plot(boundary_x, boundary_y, boundary_z,
        linestyle="-", color="red", lw=3, label="Boundary: y = x² + 1")

    for i in 1:length(x_path)
        c = colors[i]
        if i == 1
            ax.scatter([x_path[i]], [y_path[i]], [z_path_q[i]],
                color=[c], s=64, zorder=5, label="Minima on Q functions")
            ax.scatter([x_path[i]], [y_path[i]], [z_path_f[i]],
                color=[c], s=100, marker="*", zorder=5, label="Projected minima on f(x,y)")
        else
            ax.scatter([x_path[i]], [y_path[i]], [z_path_q[i]], color=[c], s=64, zorder=5)
            ax.scatter([x_path[i]], [y_path[i]], [z_path_f[i]], color=[c], s=100, marker="*", zorder=5)
        end
    end

    if length(x_path) > 1
        ax.plot(x_path, y_path, z_path_q,
            color="black", lw=2, linestyle=":", label="Path of Q minima")
    end

    ax.scatter([0.0], [1.0], [1.0], color="red", s=150, marker="*",
        label="Theoretical optimum (0,1)", zorder=6)

    ax.legend()
    fig.tight_layout()

    return fig
end

# =============================================================================
# EXAMPLES
# =============================================================================

println("f(x) = x^2, x >= 2")

f1(x) = x[1]^2
g1(x) = 2.0 - x[1]
gs1 = [g1]
x0_1d = [5.0]

base_values = 10.0 .^ (0:-1:-2)
r_1d = [[r] for r in base_values]

hist_1d = barrier_functions_method_multi(f1, gs1, x0_1d,
    r_sequence_list=r_1d,
    verbose=true)

p1d = plot_barrier_1d_multi(f1, gs1, hist_1d,
    title="1D case: f(x)=x², x≥2")
display(p1d)


println("\nf(x,y) = x^2 + y^2, x >= 2, y >= 2, synchronous reduction")

f2(x) = x[1]^2 + x[2]^2
g2_1(x) = 2.0 - x[1]
g2_2(x) = 2.0 - x[2]
gs2 = [g2_1, g2_2]
x0 = [5.0, 5.0]

r_sync = [[r, r] for r in base_values]

hist_sync = barrier_functions_method_multi(f2, gs2, x0,
    r_sequence_list=r_sync,
    verbose=true)

p3d_sync = plot_barrier_2d_multi(f2, gs2, hist_sync, show_q_functions=true)
display(p3d_sync)


println("\nf(x,y) = x^2 + y^2, x >= 2, y >= 2, asynchronous reduction")

r_async = [
    [1.0, 1.0],
    [0.01, 1.0],
    [0.01, 0.01],
]

hist_async = barrier_functions_method_multi(f2, gs2, x0,
    r_sequence_list=r_async,
    verbose=true)

p3d_async = plot_barrier_2d_multi(f2, gs2, hist_async, show_q_functions=true)
display(p3d_async)


println("\nf(x,y) = x^2 + y^2, y ≥ x² + 1")

f_parabolic(x) = x[1]^2 + x[2]^2
g_parabolic(x) = x[1]^2 + 1 - x[2]
gs_parabolic = [g_parabolic]
x0_parabolic = [0.0, 3.0]

base_values_parabolic = 10.0 .^ (0:-1:-2)
r_parabolic_sync = [[r] for r in base_values_parabolic]

hist_parabolic = barrier_functions_method_multi(f_parabolic, gs_parabolic, x0_parabolic,
    r_sequence_list=r_parabolic_sync,
    verbose=true, G=logarithmic_barrier)

p_parabolic = plot_barrier_2d_parabolic(f_parabolic, gs_parabolic, hist_parabolic,
    show_q_functions=true)
display(p_parabolic)
