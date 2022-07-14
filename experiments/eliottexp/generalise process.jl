
# Finding which functions to use to extract nodes, weights, and integration nodes of processes

# declaring model
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/Dolo.jl/examples/models/consumption_savings_iid.yaml"
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/julia_training/EGM/consumption_savings_mc.yaml"
readlines(filename)
model = yaml_import(filename)

# also understanding the declaration of exogenous processes 
# rbc_mc
exogenous: 
    z, z2: !MarkovChain
        values: [[-0.01, 0.1],[0.01, 0.5]]
        transitions: [[0.4, 0.6], [0.6, 0.4]]

# so if only one exogenous variable:
exogenous: 
    z: !MarkovChain
        values: [[-0.01, 0.1]]
        transitions: [[0.4, 0.6], [0.6, 0.4]]
# actually above wrong, if 1 exo then 
exogenous: 
    z: !MarkovChain
        values: [[-0.01], [0.1]]
        transitions: [[0.4, 0.6], [0.6, 0.4]] # with [[w11,w12],[w21,w22]]
# sudden stop AR1
exogenous: 
    ly: !VAR1
        ρ: 0.01
        Σ: 0.065^2
        N: 2
# sudden stop MC
exogenous: 
    y: !MarkovChain
        values: [[ 0.9 - delta_y  ],  # bad state
                [ 1.0 ]]          # good state
        transitions: [[ 0.5, 0.5 ],   # probabilities   [p(L|L), p(H|L)]
                    [ 0.5, 0.5 ]]     # probabilities   [p(L|H), p(H|H)]

#so we know how to interpret values and transition matrixes of MC processes now 
# but another problem is how to deal with 
options:
    discretization:
        endo:
            n: [1000]
        exo:
            n: 7 # here the exogenous grid when the process is markov or other 
            
# try IID first 
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/Dolo.jl/examples/models/consumption_savings_iid.yaml"
readlines(filename)
model = yaml_import(filename)
dp = Dolo.discretize(model.exogenous)
Dolo.n_inodes(dp,1) # so 5 integration nodes 
Dolo.n_nodes(dp) # but 0 proper nodes 
Dolo.inode(dp,1,2) # access integration nodes 
Dolo.iweight(dp,1,3) # access integration weights, from state 1(irrelvant) to state 3   

# MC
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/julia_training/EGM/consumption_savings_mc.yaml"
readlines(filename)
model = yaml_import(filename)
dp = Dolo.discretize(model.exogenous)
Dolo.n_inodes(dp,1) # so 2 integration nodes 
Dolo.n_nodes(dp) # but 2 proper nodes 
Dolo.inode(dp,2,1) # access integration nodes 
SVector(Dolo.inode(dp,2,1)...)
Dolo.inode(dp,1,2)
SVector(Dolo.iweight(dp,1,2)...) # access integration weights, from state 1(irrelvant) to state 3   

# VAR1
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/julia_training/EGM/consumption_savings_ar1.yaml"
readlines(filename)
model = yaml_import(filename)
dp = Dolo.discretize(model.exogenous)
Dolo.n_inodes(dp,1) # so 5 integration nodes 
Dolo.n_nodes(dp) # but 5 proper nodes 
Dolo.inode(dp,1,1) # access integration nodes 
Dolo.iweight(dp,1,2) # access integration weights, from state 1(irrelvant) to state 3   

#########

# Adressing discretize() to obtain object grid, to get s0,a0

# MC
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/julia_training/EGM/consumption_savings_mc.yaml"
readlines(filename)
model = yaml_import(filename)
typeof(model)
Dolo.discretize(model) # does not work with MC process
Dolo.get_discretization_options(model) # works fine 
# matching error, n is inputed implicitly 
# in fact the wrong function is called (from processes.jl)
# need to find a way to use the right method for the function
# WAY FOUND: change name discretize() to discretizemodel()
# so will have to run the function running egm 
Dolo.discretizemodel(model) # type CartesianDomain has no field endo
# conducting tests 
model.data
calibration = Dolo.get_calibration(model)
symbols = Dolo.get_symbols(model)
exogenous = Dolo.get_exogenous(model, symbols[:exogenous], calibration.flat)
Dolo.get_exogenous(model)
Dolo.get_domain(model)
model.domain
# come back to discretize(), maybe it uses another method I thought
Dolo.discretize(model;Dict()...)


# IID 
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/Dolo.jl/examples/models/consumption_savings_iid.yaml"
readlines(filename)
model = yaml_import(filename)
typeof(model)
grid, dp = Dolo.discretize(model) # but works with IID process
model.domain
Dolo.discretizemodel(model)# does not work either on IID

# VAR1
filename = "C:/Users/t480/GitHub/Pablo-Winant-internship/julia_training/EGM/consumption_savings_ar1.yaml"
readlines(filename)
model = yaml_import(filename)
typeof(model)
Dolo.discretize(model)
