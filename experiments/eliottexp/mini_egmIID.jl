using Interpolations
using Dolo: UNormal, discretize  # dependence on Dolo is actually quite minimal, we just use the exogenous shock
using StaticArrays
using LabelledArrays
using Plots

# We define the model here
m = let
    # parameter calibration

    γ = 4.0
    σ_y = 0.1
    β = 0.96
    r = 1.038
    p = (;β, γ, r, σ_y)

    # The following computes a set of nodes / weights
    exogenous = UNormal(;σ=σ_y)
    dp = discretize(exogenous)
    x = SVector(dp.integration_nodes[:]...) # nodes 
    w = SVector(dp.integration_weights[:]...) # weights

    # Initial decision rule (consumption as a function of available wealth)
    #φ = w -> min(w, 1.0 + 0.01*(w-1.0))
    φ = w -> w*0.9 #  min(w, 1.0 + 0.01*(w-1.0))

    # This is the discretization of...
    N = 100
    w_grid = range(0.4, 10; length=N) # the state-space
    a_grid = range(0.0, 10; length=N) # the post-states
    
    (;p, φ, w_grid, a_grid, integration=(w, x))

end

function consumption_a(φ1, m)
    """Computes consumption at each point of the post-state grid given:
    - φ1: decision rule for consumption tomorrow
    - m:  model
    """
    
    c_a = zeros(length(m.a_grid))
    
    (weights, nodes) = m.integration
    for (n,a) in enumerate(m.a_grid)
        e = 0.0
        for i in 1:length(weights)
            w = weights[i]
            ε = nodes[i]
            W = exp(ε) + a*m.p.r
            C = φ1(W)
            e += m.p.β*w*C^(-m.p.γ)*(m.p.r)
        end
        c_a[n] = (e)^(-1.0/m.p.γ)
    end

    return c_a

end


function egm(m; φ=m.φ, T=500, trace=false, resample=false, 
    
    τ_η=1e-8)
    """Computes the time iteration solution given:
    - φ: initial decision rule for consumption
    - m: model
    """

    logs = [] # to keep all successive decision rules

    local w_grid, c_a, φ0
    a_grid = m.a_grid

    φ0 = φ
    
    for t in 1:T
        trace ? push!(logs, deepcopy(φ0)) : nothing
        c_a = consumption_a(φ0, m)
        w_grid = a_grid + c_a   # a = w-c
        c_a = min(w_grid, c_a)
        φ0 = LinearInterpolation(w_grid, c_a; extrapolation_bc=Line()) 
    end

    if resample
        # we resample the solution so that it interpolates exactly
        # on the grid specified in m.w_grid (not the endogenous one.)
        nx = min.(m.w_grid, φ0(m.w_grid))
        φ0 = LinearInterpolation(m.w_grid, nx; extrapolation_bc=Line())
    end

    if trace
        return φ0, logs, w_grid, c_a
    else
        return φ0
    end

end

## Plots

# fixed vs endogeneous GOOD
xvec = range(0,10;length=100)
φ = egm(m; resample=false)
φs = egm(m; resample=true)
plot(xvec, xvec; label="w")
plot!(φ.itp.knots[1], φ.itp.coefs; label="c(W) endogenous")
plot!(φs.itp.ranges[1], φs.itp.itp.coefs; label="c(w) on the initial W-grid", xlabel="State w", ylabel="Control c(w)")
# above similar to plot!(w, min.(w,trace[500](w))) but need:
#trace = soltrace[2]
#w = soltrace[1].itp.ranges[1]



# iterations GOOD
@time soltrace = egm(m; resample=true, trace = true) 
@time sol = egm(m; resample=true)
xvec = range(0,10;length=100)
function convergenceEGM(soltrace)
        soltrace = soltrace
        trace = soltrace[2]
        w = soltrace[1].itp.ranges[1]
        plt = plot()
        plot!(plt, xvec, xvec; label="w", ylims=(0,10))
    for i=1:length(trace)
        plot!(plt, w, min.(w,trace[i](w)))
    end
    plot!(plt, xlabel = "Wealth", ylabel = "Consumption")
    plot!(plt, legend = false)
end
convergenceEGM(soltrace)