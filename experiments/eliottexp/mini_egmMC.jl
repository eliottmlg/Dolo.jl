using Pkg
using Interpolations
using Plots
using Dolo: UNormal, discretize  # dependence on Dolo is actually quite minimal, we just use the exogenous shock
using Dolo
using StaticArrays
using QuantEcon
using LinearAlgebra

## constructing markov chain from AR1

# from QuantEcon
dpQE = QuantEcon.rouwenhorst(3, 0.9, 0.1, 0.0)

# generating AR(1)
VAR1(;ρ::Float64=0.0, Σ=ones(1,1)) = VAR1(ρ, Σ) 
sigma = Array{Float64}(undef, 1, 1)
sigma[1,1] = 0.1 
ar1 = Dolo.VAR1(ρ = 0.9, Σ = sigma)

# converting AR(1) into Markov chain
mc = Dolo.discretize(ar1)
typeof(mc)
nodes = mc.values
transitions = mc.transitions

# We define the model here
m = let
    # parameter calibration

    γ = 4.0   
    σ_y = 0.1
    β = 0.96
    r = 1.038
    rho = 0.9 #
    p = (;β, γ, r, σ_y, rho)

    # The following computes a set of nodes / weights
 #   exogenous = UNormal(;σ=σ_y) # this shock is iid 
 #   dp = discretize(exogenous)

    # Markov process
    sigma = Array{Float64}(undef, 1, 1)
    sigma[1,1] = 0.01
    ar1 = Dolo.VAR1(ρ = 0.95, Σ = sigma)
    mc = Dolo.discretize(ar1; n=9)
    states = mc.values
    transitions = mc.transitions

    #=
    x = SVector(dp.integration_nodes[:]...) # nodes  
    w = SVector(dp.integration_weights[:]...) # weights
    =#

    # Initial decision rule (consumption as a function of available wealth)
    #φ0 = w -> min(w, 1.0 + 0.01*(w-1.0))
    φ0 = Vector{Any}(undef, 9)
    for i in 1:size(states,1)
    φ0[i] = w -> min(w, 1.0 + 0.01*(w-1.0))
    end
    φ0  # vector of length 3 of anonymous functions
    #φ0 = reduce(hcat, φ0) # into matrix

    # This is the discretization of...
    N = 100
    w_grid = range(0.8, 10; length=N) # the state-space
    a_grid = range(0.0, 10; length=N) # the post-states
    
    (;p, φ=φ0, a_grid, w_grid, N, markovprocess=(states, transitions))

end


function consumption_a(φ1, m)
    """Computes consumption at each point of the post-state grid given:
    - φ1: decision rule for consumption tomorrow
    - m:  model
    """
    #(weights, nodes) = m.integration
    (states, transitions) = m.markovprocess

    c_a = Matrix{Float64}(undef, length(m.a_grid), size(states,1))
    #c_a = zeros(length(m.a_grid))

    for i in 1:size(states,1)

        for (n,a) in enumerate(m.a_grid)

            rhs = zeros(size(states,1))
            Φ = zeros(size(states,1))

            for j in 1:size(states,1)

                #e = 0.0
                #for i in 1:length(weights) # # βREyu'(c'(M'))
                #w = weights[i] # probabilities of each nodes
                #ε = nodes[i] # nodes of the stochastic process

                inc = states[j,1]
                prob = transitions[i,j]

                #W = exp(ε) + a*m.p.r # M' = AR + y

                W = exp(inc) + a*m.p.r # exp
                C = φ1[j](W) # c'(M') using c_(i-1)(.)    j au lieu de i

                #e += m.p.β * w * C^(-m.p.γ)*(m.p.r) 

                Φ[j] = m.p.β * prob * C^(-m.p.γ)*(m.p.r) 

            end

            #rhs = LinearAlgebra.dot(Φ, transitions[i,:])
            rhs = sum(transitions[i,j]*Φ[j] for j=1:size(states,1))

            #c_a[n] = (e)^(-1.0/m.p.γ)
            c_a[n, i] = (rhs)^(-1.0/m.p.γ)
        end 
    end
    return c_a # matrix length(a_grid) x length(states) containing updated consumption levels (Float64)
end


function egm(m; φ=m.φ, T=500, trace=false, resample=false, 
    
    τ_η=1e-8)
    """Computes the time iteration solution given:
    - φ: initial decision rule for consumption
    - m: model
    """

    states = m.markovprocess[1]
    logs = [] # to keep all successive decision rules
    logs_grids = []

    local w_grid, c_a, φ0
    a_grid = m.a_grid

    φ0 = φ
    w_grid = m.w_grid
    
    for t in 1:T
        trace ? push!(logs, deepcopy(φ0)) : nothing
        trace ? push!(logs_grids, deepcopy(w_grid)) : nothing
        c_a = consumption_a(φ0, m) # c_new = (u')^(-1)(A)
        
        for i in 1:size(states,1)
                
            w_grid = a_grid + c_a[:,i] # M_new = A + c_new
            c_a[:,i] = min(w_grid, c_a[:,i]) # c_new cannot exceed M
            φ0[i] = LinearInterpolation(w_grid, c_a[:,i]; extrapolation_bc=Line()) # reconstructing policy function c_new(M)
        
        end
    end

    if resample
        for i in 1:size(states,1)
            # we resample the solution so that it interpolates exactly
            # on the grid specified in m.w_grid (not the endogenous one.)
            nx = min.(φ0[i](m.w_grid),m.w_grid) # added min
            φ0[i] = LinearInterpolation(m.w_grid, nx; extrapolation_bc=Line())
        end
    end

    if trace
        return φ0, logs, w_grid, c_a, logs_grids
    else
        return φ0 # 3-element vector, each element is an object of linear interpolation containing grid and values on the grid (10-element vector containing optimal dr(i,a))
    end

end


## Plots
# results for all markov states, constraints VS No constraints: GOOD
@time φs = egm(m; resample=true)
function result(φs)
    xvec = range(0,m.w_grid[m.N];length=100)
    plt = plot(xvec, xvec; xlims=(0,10), ylims=(0,10), xlabel="State w", ylabel="Control c(w)", label = "w")
    for i in 1:length(m.markovprocess[1])
        x = φs[i].itp.ranges[1]
        plt = plot!(x, φs[i](x))
        plt = plot!(x, min.(x,φs[i](x)); marker="o")
    end
    plt
end 
result(φs) 

# iterations GOOD, the policy function seems to converge at the 2nd iteration only
@time soltrace = egm(m; resample=true, trace = true) 
function convergenceEGM(soltrace)
        soltrace = soltrace
        logs = soltrace[2]
        xvec = range(0,10;length=100)
        x = soltrace[1][4].itp.ranges[1] # chosen the 4th markov state
        plt = plot(xvec, xvec; xlims=(0,10), ylims=(0,10), xlabel="State w", ylabel="Control c(w)", legend = :bottomright)
        plt = plot!(xlabel = "Wealth", ylabel = "Consumption")
        plt = plot!(legend = false)
        for i in 1:length(logs)
            plt = plot!(x, min.(x, logs[i][4](x))) # soltrace[log][ith iteration][markov state][value of consumption on the nth point of the w-grid]
        end
        plt
end
convergenceEGM(soltrace)
# for instance 
logs = soltrace[2]
logs[2][4]-logs[500][4] # comparing the 2nd and the 500th iterations for markov state 4, returns zeros 
