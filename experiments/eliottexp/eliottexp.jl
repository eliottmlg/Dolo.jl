using Dolo
using StaticArrays

model = Model("examples/models/consumption_savings_iid.yaml")

m,s,x,a,z,p = model.calibration[:exogenous,:states,:controls,:poststates,:expectations,:parameters]

z

Dolo.half_transition(model, m,a,m,p)


Dolo.direct_response_egm(model, m,a,z,p)
Dolo.reverse_state(model, m,a,x,p)

function carre(x)
    local y
    y=2
    x^y
end
carre(3)

itpexp = Vector{Any}(undef,2)
itpexp[1] = LinearInterpolation(1:10, 2:11, extrapolation_bc = Line()) 
itpexp[2] = LinearInterpolation(1:10, 2:11, extrapolation_bc = Line()) 
typeof(itpexp[1])
itpexp[1](1)

# need to turn initial decision rule into object that can be called like itp object
# then convert float into svector to be passed on to next function in consumption_a
# itp object returns Float64 and not svectors 

function φfunction(i, ss::SVector{1})::SVector # i is the current exogenous state, takes state at i
    #xx = min(ss[1], 1.0 + 0.01*(ss[1]-1.0))
    xx = 0.9*ss[1]
    xx = SVector(xx...)
    return ss, xx
end
φfunction(1,2)

φ0 = Vector{Any}(undef,5)
for i in 1:5
    φ0[i] = LinearInterpolation(1:1000, 2:1001, extrapolation_bc = Line())
end 
φ0[1](1)
φ0[i](s) = φfunction(i,s)[2]

SVector(φ0[1](1)...)
