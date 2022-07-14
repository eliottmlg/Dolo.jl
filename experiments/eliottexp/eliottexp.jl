using Dolo
using StaticArrays

model = Model("examples/models/consumption_savings_iid.yaml")

m,s,x,a,z,p = model.calibration[:exogenous,:states,:controls,:poststates,:expectations,:parameters]

z

Dolo.half_transition(model, m,a,m,p)


Dolo.direct_response_egm(model, m,a,z,p)
Dolo.reverse_state(model, m,a,x,p)
