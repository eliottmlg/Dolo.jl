## Implementing EGM generalised to all types of shocks + finding why decision rules are flat

using Pkg
using Interpolations
using Plots
using Dolo
using Dolo: UNormal, discretize 
using StaticArrays
using QuantEcon
using LinearAlgebra
import Dolang
using LaTeXStrings

# Changes to Dolo.jl, so need to run the following:
# model.jl: 
# discretizemodel()
# get_factory_noloopEGM()

# declaring model
filename = "C:/Users/t480/GitHub/Dolo.jl/examples/models/consumption_savings_iid.yaml"
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/julia_training/EGM/consumption_savings_ar1.yaml"
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/julia_training/EGM/consumption_savings_mc.yaml"
filename = "C:/Users/t480/GitHub/Dolo.jl/examples/models/rbc_catastrophe_newstyle.yaml"

readlines(filename)
model = yaml_import(filename)  


#equations
    #=
    #equations
    F_tran = Dolo.get_factory(model, "transition")
    F_arb = Dolo.get_factory(model, "arbitrage")[1]
    F_g = Dolo.get_factory(model, "half_transition")
    F_τ = Dolo.get_factory(model, "direct_response_egm")
    F_h = Dolo.get_factory(model, "expectation")
    F_aτ =  Dolo.get_factory(model, "reverse_state")

    code_tran = Dolang.gen_generated_gufun(F_tran)
    tran = eval(code_tran)
    code_arb = Dolang.gen_generated_gufun(F_arb)
    arbi = eval(code_arb)
    code_g = Dolang.gen_generated_gufun(F_g)
    g = eval(code_g)
    code_τ = Dolang.gen_generated_gufun(F_τ)
    τ = eval(code_τ)
    code_aτ = Dolang.gen_generated_gufun(F_aτ)
    aτ = eval(code_aτ)
    code_h = Dolang.gen_generated_gufun(F_h)
    h = eval(code_h)
    =#

    
    # exogenous shock
    shock = Dolo.get_exogenous(model)
    dp = Dolo.discretize(shock)
    size_states = Dolo.n_inodes(dp,1)

        
    # calibration 
    m_, s_, mr_, a_, x_, p_ = model.calibration[:exogenous, :states, :expectations, :poststates, :controls, :parameters]
    m,s,mr,a,x,p = [SVector(e...) for e in model.calibration[:exogenous, :states, :expectations, :poststates, :controls, :parameters]]

    # grids
    grid, dprocess = Dolo.discretize(model)
    grid_endo = grid.endo
    grid_exo = grid.exo
    grid_fixed = grid_endo
    s0 = deepcopy(Dolo.nodes(grid_endo))
    a0 = deepcopy(Dolo.nodes(grid_fixed))
    s0Float = deepcopy(reinterpret(Float64, s0)) # convert from Vector(Svector(Float64)) to Vector(Float64)


# initial policy function
function φfunction(i, ss::SVector{1})::SVector # i is the current exogenous state, takes state at i
    #xx = min(ss[1], 1.0 + 0.01*(ss[1]-1.0))
    xx = 0.9*ss[1]
    xx = SVector(xx...)
    return ss, xx
end # to call: φfunction(1,s0[1])

φ0 = Vector{Any}(undef,size_states)
for i in 1:size_states
    φ0[i] = LinearInterpolation(s0Float, s0Float, extrapolation_bc = Line())
end 


function consumption_a(model,φ)
    φ1 = φ
    c_a = Matrix{typeof(x)}(undef, length(a0), size_states)
    
    for i in 1:size_states
        for (n,a) in enumerate(a0) 
            
            m = SVector(Dolo.inode(dp,i,i)...)
            zz = mr*0.0
            zzj = zeros(SVector{1,Float64}, size_states)
                        
            for j in 1:size_states
                
                M = SVector(Dolo.inode(dp,i,j)...)
                w = Dolo.iweight(dp,i,j)

                ss = Dolo.half_transition(model,m,a,M,p) # S = g(m,a,M), half_transition

                #ss, xx = φ1(i,ss) # c'(M') using c_(i-1)(.)  
                xx = SVector(φ1[j](ss)...)
                
                zzj[j] = w * Dolo.expectation(model,M,ss,xx,p) # z = E(h(M,S,X)), expectation

            end
            zz += sum(zzj[j] for j=1:size_states)
            c_a[n,i] = Dolo.direct_response_egm(model,m,a,zz,p) # c = \tau(m,a,z), direct_response_egm
        end    
    end
    return c_a
end

# computing updated deicisions after from initial decision rule to iteration one  
cprime = consumption_a(model, φ0)# same consumption for all state of m
# expected since shock is iid

# interpolation 
itp = Vector{Any}(undef,size_states) # empty vector to host interpolation objects 
s0Float = deepcopy(reinterpret(Float64, s0)) # convert from Vector(Svector(Float64)) to Vector(Float64)
cprimeFloat = deepcopy(reinterpret(Float64, cprime)) # for function LinearInterpolation

struct MyDR # creating a structure for neater incorporation
    itp::Vector{Any}
end
# (mydr::MyDR)(i,s) = mydr.itp[i](s)
# interpolated policy functions
mydr(i,s) = φs[2].itp[i](s)
mydrlogs(i,s,t) = φs[3][t].itp[i](s)

#=
function toyegm(model; φfunction=nothing, T=500, trace=false, resample=false, τ_η=1e-8)
    logs = []
    
    local cprime, φ0

    φ0 = φfunction
    w_grid = s0

    # here 

    if resample
        for t in 1:T
            cprime = consumption_a(model,φ0) # c_new = (u')^(-1)(A)
            nx = cprime
            for i in 1:size_states
                for n in 1:length(w_grid)
                    w_grid[n] = Dolo.reverse_state(model,m,a,cprime[n,i],p) # s = a\tau(m,a,x), reverse_state
                    nx[n,i] = min(s0[n], itp[i](s0[n])) # c_new cannot exceed M
                end
                nxFloat = reinterpret(Float64,nx) # needs one poststate, many controls allowed 
                s0Float = reinterpret(Float64,s0) # needs one poststate, many controls allowed 
                itp[i] = LinearInterpolation(s0Float,nxFloat[:,i];extrapolation_bc=Line())
            end # itp linear combs
            trace ? res = MyDR(itp) : nothing
            trace ? push!(logs,deepcopy(res)) : nothing
        end
    else
        for t in 1:T
            cprime = consumption_a(model,φ0) # c_new = (u')^(-1)(A)
            for i in 1:size_states
                for n in 1:length(w_grid)
                    w_grid[n] = Dolo.reverse_state(model,m,a,cprime[n,i],p) # s = a\tau(m,a,x), reverse_state
                    cprime[n,i] = min(w_grid[n], cprime[n,i]) # c_new cannot exceed M
                end
                cprimeFloat = reinterpret(Float64, cprime) # only works in one dimension 
                w_gridFloat = reinterpret(Float64,w_grid) # needs one poststate, many controls allowed 
                itp[i] = LinearInterpolation(w_gridFloat, cprimeFloat[:,i], extrapolation_bc = Line()) # cprime good?
            end # itp linear combs
            trace ? res = MyDR(itp) : nothing
            trace ? push!(logs,deepcopy(res)) : nothing
        end
    end

    dr = cprime
    res = MyDR(itp)
    
    if trace
        return dr, res, logs
    else
        return dr, res
    end     
end
=#

function toyegm(model, φ=nothing; T=500, trace=false, resample=false, τ_η=1e-8)
    logs = []
    
    local cprime, φ0

    φ0 = φ
    w_grid = s0

        for t in 1:T
            
            trace ? push!(logs, deepcopy(φ0)) : nothing
            #trace ? res = MyDR(φ0) : nothing
            #trace ? push!(logs,deepcopy(res)) : nothing
            cprime = consumption_a(model,φ0) # c_new = (u')^(-1)(A)

            for i in 1:size_states

                for n in 1:length(w_grid)

                    w_grid[n] = Dolo.reverse_state(model,m,a,cprime[n,i],p) # s = a\tau(m,a,x), reverse_state
                    cprime[n,i] = min(w_grid[n], cprime[n,i]) # c_new cannot exceed M

                end

                cprimeFloat = reinterpret(Float64, cprime) # only works in one dimension 
                w_gridFloat = reinterpret(Float64,w_grid) # needs one poststate, many controls allowed 
                φ0[i] = LinearInterpolation(w_gridFloat, cprimeFloat[:,i], extrapolation_bc = Line()) # cprime good?
            end # itp linear combs
            #φ0 = itp
            #trace ? res = MyDR(itp) : nothing
            #trace ? push!(logs,deepcopy(res)) : nothing
        end

    dr = cprime
    #res = MyDR(itp)
    #φ0 = itp
    
    if trace
        return φ0, logs
    else
        return φ0 
    end     
end

function egm(model, φ=nothing; T=500, trace=false, resample=false, τ_η=1e-8)
    logs = []
    
    local cprime, φ0

    φ0 = φ
    w_grid = s0

        for t in 1:T
            
            trace ? push!(logs, deepcopy(φ0)) : nothing
            #trace ? res = MyDR(φ0) : nothing
            #trace ? push!(logs,deepcopy(res)) : nothing
            cprime = consumption_a(model,φ0) # c_new = (u')^(-1)(A)

            for i in 1:size_states

                for n in 1:length(w_grid)

                    w_grid[n] = Dolo.reverse_state(model,m,a,cprime[n,i],p) # s = a\tau(m,a,x), reverse_state
                    cprime[n,i] = min(w_grid[n], cprime[n,i]) # c_new cannot exceed M

                end

                cprimeFloat = reinterpret(Float64, cprime) # only works in one dimension 
                w_gridFloat = reinterpret(Float64,w_grid) # needs one poststate, many controls allowed 
                φ0[i] = LinearInterpolation(w_gridFloat, cprimeFloat[:,i], extrapolation_bc = Line()) # cprime good?
            end # itp linear combs
            #φ0 = itp
            #trace ? res = MyDR(itp) : nothing
            #trace ? push!(logs,deepcopy(res)) : nothing
        end
#=
    if resample
            for i in 1:size_states
                for n in 1:length(w_grid)
                    # we resample the solution so that it interpolates exactly
                    # on the grid specified in m.w_grid (not the endogenous one.)
                    #nx = min.(w_grid, mydr(i,w_grid))
                    nx = cprime
                    nx[n,i] = min(s0[n], φ0[i](s0[n]))
                    nxFloat = reinterpret(Float64,nx) # needs one poststate, many controls allowed 
                    s1Float = reinterpret(Float64,s0) # needs one poststate, many controls allowed 
                    #itp[i] = LinearInterpolation(w_grid, nx, extrapolation_bc=Line())
                    φ0[i] = LinearInterpolation(s1Float,nxFloat[:,i];extrapolation_bc=Line())
                end
            end
    end
=#
    if resample
        for i in 1:size_states
                nx = min.(s0Float, φ0[i](s0Float))
                nxFloat = reinterpret(Float64,nx) # needs one poststate, many controls allowed 
                s1Float = reinterpret(Float64,s0) # needs one poststate, many controls allowed 
                #itp[i] = LinearInterpolation(w_grid, nx, extrapolation_bc=Line())
                φ0[i] = LinearInterpolation(s1Float,nxFloat;extrapolation_bc=Line())
            end
        end
        #res = MyDR(itp)
        #mydr(i,w_grid) = res.itp[i](w_grid)
        # mydr(1,x)


    #dr = cprime
    #res = MyDR(itp)
    
    if trace
        return φ0, logs
    else
        return φ0
    end     
end

# 04082022
# same as 0208 below
φs = toyegm(model, φ0; T=10)
φs[3] # 02/08/2022 trying to reproduce mini_egmIID, adding φ0 to toyegm()
φs[3] # two element vector for two markov states
φs[3][1] # each markov state has its decision rule
# the difference maybe that here it is a vector of int objects while in mini_egmIID it is an int objects
φs[3][1].itp # now we are getting the int objects, so before it was just values ?
φs[3][1].itp.knots # yes
φs[3][1].itp.knots[1] 
φs[3][1].itp.coefs # yes
# now lets try to plot  
function plot1(size_states)
    plt = plot()
    plot!(plt, s0Float, s0Float; label="w")
    for i in 1:size_states
        plt = plot!(plt, φs[i].itp.knots[1], φs[i].itp.coefs; legend = true)
    end
    plt
end
plot1(size_states)
# we obtain that the optimal decision rule is simply to consume evrything 
# so utility function is not concave enough? people are unsatiated even at high consumption levels 

s0 # s0 is changed, why? s0 is changed with resample and trace = false

# 15082022
φs = toyegm(model, φ0; T=50,resample=false,trace=false)
function plot1(size_states)
    plt = plot(s0Float, s0Float; label="w")
    for i in 1:size_states
        x = φs[i].itp.knots[1]
        plt = plot!(x, min.(x,φs[i](x)); legend = true)
    end
    plt
end
plot1(size_states)

φs = egm(model, φ0; T=25,resample=true)
φs[1].itp.knots
φs[1].itp.coefs
φs[1](φs[1].itp.knots[1])
φs[1](s0Float)
vec(s0Float)
min.(s0Float, φs[1](s0Float))
vec = Vector{Float64}(undef,length(s0Float))
s0Float[1]










φs = egm(model, φ0; T=25,resample=true)
function plot1(size_states)
    plt = plot(s0Float, s0Float; label="w")
    for i in 1:size_states
        x = φs[i].itp.knots[1]
        plt = plot!(x, min.(x,φs[i](x)); legend = true)
    end
    plt
end
plot1(size_states)

φs = egm(model, φ0; T=25,resample=true, trace = true)
function iterationDR(T) 
    plt = plot(s0Float,s0Float; xlabel="State w", ylabel="Control c(w)", label = "w")
    x = φs[1][1].itp.knots[1]
    logs = φs[2]
    for t in 1:T
        plot!(plt, x, min.(x,logs[t][1](x)))
    end
    plt
end
iterationDR(25)










# plotting functions 
function statesDR(size_states)
    plt = plot()
    plot!(plt, s0Float, s0Float)
    for i in 1:size_states
        plt = plot!(plt, s0Float, min.(s0Float,mydr(i,s0Float)); legend = true)
    end
    plt
end

statesDR(size_states)

φs = egm(model; φfunction,T=10,resample=false,trace=true) # trace = true?
function iterationDR(T) 
    plt = plot()
    plot!(plt, s0Float,mydrlogs(1,s0Float,1); legend=true)
    for t in 1:T
        plot!(plt, s0Float, mydrlogs(1,s0Float,t))
    end
    plot!(plt, s0Float,mydrlogs(1,s0Float,T))
    plt
end

mydrlogs(1,s0Float,10)
iterationDR(10)


# results output function
function results(model=model,trace=false,resample=false,graphstate=false,graphiteration=false)
    @time φs = toyegm(model; φfunction,T=500,resample=resample,trace=trace)
    mydr(i,s) = φs[2].itp[i](s)
    mydrlogs(i,s,t) = φs[3][t].itp[i](s)
    size_states = size(φs[1],2)
    graphstate ? statesDR(size_states) : nothing
    graphiteration ? iterationDR(T) : nothing
end

results(model;trace=false, resample=true, graphstate=true, graphiteration=false)


















# policy functions for each state
function policyfunction(size_states)
    plt = plot()
    plot!(plt, s0Float, s0Float)
    for i in 1:size_states
        plt = plot!(plt, s0Float, mydr(i,s0Float); legend = true)
    end
    plt
end
policyfunction(size_states)


# policy functions at each iteration
@time φs = toyegm(model; φfunction, resample=true,trace = true)
mydrlogs(1,s0Float,1)
function iterationEGM(T) 
    plt = plot()
    plot!(plt, s0Float,mydrlogs(1,s0Float,1); marker="o", legend=false)
    for t in 1:T
        plot!(plt, s0Float, mydrlogs(1,s0Float,t))
    end
    plot!(plt, s0Float,mydrlogs(1,s0Float,T); marker="o")
    plt
end
iterationEGM(500)







# iteration by iteration

# t = 1
cprime = consumption_a(model,φfunction) # c_new = (u')^(-1)(A)
w_grid = s0
        for i in 1:size_states
            for n in 1:length(w_grid)
                w_grid[n] = Dolo.reverse_state(model,m,a,cprime[n,i],p) # s = a\tau(m,a,x), reverse_state
                cprime[n,i] = min(w_grid[n], cprime[n,i]) # c_new cannot exceed M
            end
            cprimeFloat = reinterpret(Float64, cprime)
            s0Float = reinterpret(Float64,w_grid)
            itp[i] = LinearInterpolation(s0Float, cprimeFloat[:,i], extrapolation_bc = Line())
        end
cprime # exactly the same because
itp[1]
s0Float
function plotcprime1(itp)
    plt = plot()
    #plot!(plt, s0Float,s0Float; legend = true)
    plot!(plt, s0Float, itp[1]; marker = "o")
    plt
end
plotcprime1(itp)

# t = 2

cprime = consumption_a(model,itp) # c_new = (u')^(-1)(A)
w_grid = s0
        for i in 1:size_states
            for n in 1:length(w_grid)
                w_grid[n] = aτ(m,a,cprime[n,i],p) # s = a\tau(m,a,x), reverse_state
                cprime[n,i] = min(w_grid[n], cprime[n,i]) # c_new cannot exceed M
            end
            cprimeFloat = reinterpret(Float64, cprime)
            s0Float = reinterpret(Float64,w_grid)
            itp[i] = LinearInterpolation(s0Float, cprimeFloat[:,i], extrapolation_bc = Line())
        end
cprime # exactly the same because
itp[1]
s0Float
function plotcprime1(itp)
    plt = plot(ylims = 1.2)
    plot!(plt, s0Float,s0Float; legend = true)
    plot!(plt, s0Float, itp[1]; marker = "o")
    plt
end
plotcprime1(itp)



# plotting for exogenous state 1 updated consumption along fixed grid A 
function plotcprime1(cprime)
    plt = plot()
    for i in 1:size_states
        plot!(plt, a0, cprime[:,i]; marker = "o", legend = false)
    end
    plt
end
plotcprime1(cprime)
