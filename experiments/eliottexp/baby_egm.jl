## Implementing EGM generalised to all types of shocks + finding why decision rules are flat

using Pkg
using Interpolations
using Plots
using Dolo
using StaticArrays
using QuantEcon
using LinearAlgebra

## first baby_egm
## second previously egmDolo


# We define the model here
m = let
    # parameter calibration

    γ = 4.0   
    σ_y = 0.1
    β = 0.96
    r = 1.02
    rho = 0.9
    N_mc = 3
    cbar = 0.9
    sigma = Array{Float64}(undef, 1, 1)
    sigma[1,1] = σ_y^2
    p = (;β, γ, r, σ_y, rho, N_mc, cbar, sigma)

    # Markov process
    ar1 = Dolo.VAR1(ρ = rho, Σ = sigma)
    mc = Dolo.discretize(ar1; n=N_mc)
    states = mc.values
    transitions = mc.transitions

    # Initial decision rule (consumption as a function of available wealth)
    φ0 = Vector{Any}(undef, N_mc)
    for i in 1:size(states,1)
    φ0[i] = w -> min(w, 1.0 + 0.01*(w-1.0))
    end
    φ0  # vector of length nb_of_markov_states of anonymous functions

    # This is the discretization of...
    N = 50
    w_grid = range(0.01, 4; length=N) # the state-space
    a_grid = range(0.01, 4; length=N) # the post-states
    
    (;p, φ=φ0, a_grid, w_grid, N, markovprocess=(states, transitions))

end


function consumption_a(φ1, m)
    """Computes consumption at each point of the post-state grid given:
    - φ1: decision rule for consumption tomorrow
    - m:  model
    """

    (states, transitions) = m.markovprocess
    c_a = Matrix{Float64}(undef, length(m.a_grid), size(states,1))

    for i in 1:size(states,1)  # βREyu'(c'(M'))

        for (n,a) in enumerate(m.a_grid)

            rhs = zeros(size(states,1))
            Φ = zeros(size(states,1))

            for j in 1:size(states,1)

                inc = states[j,1]
                prob = transitions[i,j]
                W = exp(inc) + a*m.p.r # M' = AR + y
                C = φ1[j](W) # c'(M') using c_(i-1)(.)  
                Φ[j] = m.p.β * prob * (C/m.p.cbar)^(-m.p.γ) * (m.p.r) 

            end

            rhs = LinearAlgebra.dot(Φ, transitions[i,:])
            c_a[n, i] = m.p.cbar * (rhs)^(-1.0/m.p.γ)
        end 
    end
    return c_a # matrix length(a_grid) x length(states) containing updated consumption levels (Float64)
end


function egm(m; φ=m.φ, T=500, trace=false, resample=false, τ_η=1e-8)
    """Computes the time iteration solution given:
    - φ: initial decision rule for consumption
    - m: model
    """

    states = m.markovprocess[1]
    logs = [] # to keep all successive decision rules

    local w_grid, c_a, φ0

    a_grid = m.a_grid
     φ0 = φ
    
    for t in 1:T
        
        trace ? push!(logs, deepcopy(φ0)) : nothing
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
            nx = φ0[i](m.w_grid)
            φ0[i] = LinearInterpolation(m.w_grid, nx; extrapolation_bc=Line())
        end
    end

    if trace
        return φ0, logs, w_grid, c_a
    else
        return φ0 # 3-element vector, each element is an object of linear interpolation containing grid and values on the grid (10-element vector containing optimal dr(i,a))
    end

end



## Plots
# results for all markov states, constraints VS No constraints: GOOD
@time φs = egm(m; resample=true)
function result(φs)
    xvec = range(0,m.w_grid[m.N];length=100)
    plt = plot(xvec, xvec; xlims=(0,5), ylims=(0,6), xlabel="State w", ylabel="Control c(w)", legend = :bottomright)
    for i in 1:length(m.markovprocess[1])
        x = φs[i].itp.ranges[1]
        plt = plot!(x, φs[i](x))
        plt = plot!(x, min.(x,φs[i](x)); marker="o")
    end
    plt
end 
result(φs) 


# iterations NOT GOOD, the policy function seems to converge at the 2nd iteration only
@time soltrace = egm(m; resample=true, trace = true) 
function convergenceEGM(soltrace)
        soltrace = soltrace
        logs = soltrace[2]
        xvec = range(0,10;length=100)
        x = soltrace[1][2].itp.ranges[1] # chosen the 4th markov state
        plt = plot(xvec, xvec; xlims=(0,5), ylims=(0,6), xlabel="State w", ylabel="Control c(w)", legend = :bottomright)
        plt = plot!(xlabel = "Wealth", ylabel = "Consumption")
        plt = plot!(legend = false)
        for i in 1:length(logs)
            plt = plot!(x, min.(x, logs[i][2](x))) # soltrace[log][ith iteration][markov state][value of consumption on the nth point of the w-grid]
        end
        plt
end
convergenceEGM(soltrace)
# for instance 
logs = soltrace[2]
logs[2][2]-logs[500][2] # comparing the 2nd and the 500th iterations for markov state 2, returns zeros 


###############################################################################################


# We define the model here
m = let
    # parameter calibration

    γ = 4.0   
    σ_y = 0.1
    β = 0.96
    r = 1.02
    rho = 0.9
    N_mc = 3
    cbar = 0.9
    sigma = Array{Float64}(undef, 1, 1)
    sigma[1,1] = σ_y^2
    p = (;β, γ, r, σ_y, rho, N_mc, cbar, sigma)

    # Markov process
    ar1 = Dolo.VAR1(ρ = rho, Σ = sigma)
    mc = Dolo.discretize(ar1; n=N_mc)
    states = mc.values
    transitions = mc.transitions
    size_states = Dolo.n_nodes(mc)
    #j_after_i = Dolo.i_n_inodes(mc,i)

    # Initial decision rule (consumption as a function of available wealth)
    φ0 = Vector{Any}(undef, N_mc)
    for i in 1:size_states
    φ0[i] = w -> min(w, 1.0 + 0.01*(w-1.0))
    end
    φ0  # vector of length nb_of_markov_states of anonymous functions

    # This is the discretization of...
    N = 50
    w_grid = range(0.01, 4; length=N) # the state-space
    a_grid = range(0.01, 4; length=N) # the post-states
    
    (;p, φ=φ0, a_grid, w_grid, N, markovprocess=(states, transitions, size_states))

end





function consumption_a(model; 
    dr0=Dolo.ConstantDecisionRule(model),
    φ1)
    """Computes consumption at each point of the post-state grid given:
    - φ1: decision rule for consumption tomorrow
    - m:  model
    """
    (states, transitions) = m.markovprocess
    c_a = Matrix{Float64}(undef, length(m.a_grid), size_states)

    for i in 1:size_states  # βREyu'(c'(M'))

        for (n,a) in enumerate(m.a_grid)

            rhs = zeros(size_states)
            Φ = zeros(size_states)

            for j in 1:size_states

                inc = states[j,1]
                prob = transitions[i,j]
                W = exp(inc) + a*m.p.r # M' = AR + y
                C = φ1[j](W) # c'(M') using c_(i-1)(.)  
                Φ[j] = m.p.β * prob * (C/m.p.cbar)^(-m.p.γ) * (m.p.r) 

            end

            rhs = LinearAlgebra.dot(Φ, transitions[i,:])
            c_a[n, i] = m.p.cbar * (rhs)^(-1.0/m.p.γ)
        end 
    end
    return c_a # matrix length(a_grid) x length(states) containing updated consumption levels (Float64)
end


function egm(m; φ=m.φ, T=500, trace=false, resample=false, τ_η=1e-8)
    """Computes the time iteration solution given:
    - φ: initial decision rule for consumption
    - m: model
    """

    size_states = m.markovprocess[3]
    logs = [] # to keep all successive decision rules

    local w_grid, c_a, φ0

    a_grid = m.a_grid
     φ0 = φ
    
    for t in 1:T
        
        trace ? push!(logs, deepcopy(φ0)) : nothing
        c_a = consumption_a(φ0, m) # c_new = (u')^(-1)(A)
        
        for i in 1:size_states 
            w_grid = a_grid + c_a[:,i] # M_new = A + c_new
            c_a[:,i] = min(w_grid, c_a[:,i]) # c_new cannot exceed M
            φ0[i] = LinearInterpolation(w_grid, c_a[:,i]; extrapolation_bc=Line()) # reconstructing policy function c_new(M)
        end
    end

    if resample
        for i in 1:size_states
            # we resample the solution so that it interpolates exactly
            # on the grid specified in m.w_grid (not the endogenous one.)
            nx = φ0[i](m.w_grid)
            φ0[i] = LinearInterpolation(m.w_grid, nx; extrapolation_bc=Line())
        end
    end

    if trace
        return φ0, logs, w_grid, c_a
    else
        return φ0 # 3-element vector, each element is an object of linear interpolation containing grid and values on the grid (10-element vector containing optimal dr(i,a))
    end

end

#Results along markov states NOT GOOD
@time φs = egm(m; resample=true)
function result(φs)
    xvec = range(0,m.w_grid[m.N];length=50)
    x = φs[1].itp.ranges[1] # grid for resample (same along markov states), for not resample φs[i].itp.knots[1]
    plt = plot(xvec, xvec; xlims=(0,4), ylims=(0,10), xlabel="State w", ylabel="Control c(w)", legend = :bottomright)
    for i in 1:length(m.markovprocess[1])
        plt = plot!(x, min.(x,φs[i](x)); marker= "o")
        plt = plot!(x, φs[i]; marker= "o")
    end
    plt
end 
result(φs)

# iterations for fixed markov state NOT GOOD
@time soltrace = egm(m; resample=true, trace = true) 
xvec = range(0,4;length=100)
function convergenceEGM(soltrace)
        soltrace = soltrace
        log = soltrace[2]
        x = soltrace[1][1].itp.ranges[1]
        plt = plot()
        plot!(plt, xvec, xvec; label="w", ylims=(0,4))
    for i=1:20:length(log)
        plot!(plt, x, min.(x,log[i][2]); marker= "o")
        plot!(plt, x, min.(x, log[i][2](x))) # soltrace[log][ith iteration][markov state][value of consumption on the nth point of the w-grid]
    end
    plot!(plt, xlabel = "Wealth", ylabel = "Consumption")
    plot!(plt, legend = true)
end
convergenceEGM(soltrace)

