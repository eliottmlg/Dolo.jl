(to open preview window: ctrl+shift+v)

# Implementing EGM in Dolo

The Dolo program reads YAML files written with the following structure:

```r
symbols:
exogenous: []
states: []
controls: []
parameters: []

equations:
transitions: 
-...
arbitrage: 
-...
half_transition: |
...
reverse_state: |
...
expectation: |
...
direct_response_egm: |
...

calibration:

domain:

exogenous: 

options:
```

## How is Dolo used to solve a model in Julia?
The modeller enters the equations describing the model and the values of parameters in the dedicated section, after which she uses an algorithm to solve the model. Algorithms include Value iteration (VI), Time iteration (TI), or Endogenous gridpoint method (EGM). To start the resolution, Dolo first needs to reads the model. It does so using a compiler-file, which extracts the equations and the parameter values from the YAML file. This information will then be called by the algorithm-file, which is the file describing the method of resolution chosen by the modeller (EGM, TI, VI).
The algorithm-file will run resolution routines and will print the results in the Julia REPL.

## EGM

The EGM is a solution method for a certain class of models. In particular, it handles models with one continuous state variable and one continous control variable.

Consider a model of the following form:

$\mathbf{z_t} = E_t[h(\mathbf{m_{t+1}},\mathbf{s_{t+1}},\mathbf{x_{t+1}})]$, the expectation equation

$\mathbf{s_t} = g(\mathbf{m_{t-1}},\mathbf{a_{t-1}},\mathbf{m_t})$, the state half-transition equation

$\mathbf{x_t} = \tau(\mathbf{m_t},\mathbf{a_t},\mathbf{z_t})$, the direct-response equation

$\mathbf{s_t} = a\tau(\mathbf{m_t},\mathbf{a_t},\mathbf{x_t})$, the reverse-state equation

where $\mathbf{m_t}$ is a vector of exogenous shocks, $\mathbf{x_t}$ is a vector of decision values, $\mathbf{s_t}$ is the discretized state-space, $\mathbf{a_t}$ is the discretized post-state space.


The EGM consists in approximating the policy function as a function of shocks and the state, $\phi(\mathbf{m_t},\mathbf{s_t})$, alike the Time iteration algorithm. The difference lies in the fact that optimal decisions are computed only at gridpoints that satisfy the Euler equation and these computations do not involve any root-finding problem. Decisions can be written in two equivalent ways, in terms of the state or the post-state $x_t^{i,j}=\phi(m_{t}^j,s_t^i) = \psi(m_{t}^j,a_t^i)$ where $j=1,2,...,J$ indexes the current state of the shock and $i=1,2,...,I$ indexes the point on the fixed grid $\mathbf{a}$. At step $n$, the current guess for the control, $\phi_n(m_{t}^j,s_t^i)$, serves as the control being used in the next period.

$x_{t,n}^{i,j} = x_{t,n}(m_{t}^j, a^i_t)$ is the value of the control at $t$, given we are in point $i$ of the fixed grid $\mathbf{a}$ and the current state of the shock is $j$, at the iteration $n$. $\mathbf{x_{t,n}} = \mathbf{x_{t,n}(\mathbf{m_t},\mathbf{a_t})}$ is the vector of decisions at $t$ in iteration $n$. Finally,
 $\phi_n(\mathbf{m_t},\mathbf{s_t})$ is the policy function (obtained after interpolating $\mathbf{x_{t,n}}$ on $\mathbf{s_{t,n}}$) at iteration $n$.

Here is an outline of the algorithm:

1. Discretize the continuous post-state variable $a$ on a fixed grid $\mathbf{a}$, the continuous state $s$ on an endogenous grid $\mathbf{s_0}$, and the continuous shock $m$ on an exogenous grid $M$.
2. Start with an initial guess of the policy function $\phi_0(\mathbf{m_t},\mathbf{s_t}) = \mathbf{m_t}$, a mapping from the state-grid and the exogenous-grid to the set of decisions $x_t$.
3. At the current guess $\phi_n(\mathbf{m_t},\mathbf{s_t})$, execute the following for $j=1$ through $j=J$: $\forall a_t^i \in \mathbf{a}$, compute $z(m_{t}^j,a_t^i) = E_t[h(m_{t+1},s_{t+1},x_{t+1})] = E_{m_{t+1}^k|m_{t}^j}[h(m_{t+1}^k,g(m_{t}^j,a^i_t,m_{t+1}^k),\phi_{n}(m_{t+1}^k,g(m_{t}^j,a_t^i,m_{t+1}^k)))]$ for $k=1,...,K$ the index for the state of the shock in the next period. This corresponds to the RHS of the Euler equation.
4. Compute for all pairs $(a_t^i, m_{t}^j)$ the updated decisions $\mathbf{x_{t,n+1}}$ using the direct-response equation: $x_{t,n+1}^{i,j}(m_{t}^j, a^i_t) = \tau(m_{t}^j,a^i_t,z(m_{t}^j,a^i_t))$
5. Construct the new endogenous grid $\mathbf{s_{t,n+1}}$ by computing for all pairs $(m_{t}^j,a^i_t)$ the state $\mathbf{s_{t,n+1}}$ using the reverse-state equation: $s_{t,n+1}^i = a\tau(m_{t}^j,a^i_t,x_{t,n+1}^{i,j})$
6. Interpolate each element of $\mathbf{x_{t,n+1}}$ on endogenous gridpoints of $\mathbf{s_{t,n+1}}$ to get the policy function $\phi_{n+1}(\mathbf{m_{t}},\mathbf{s_t})$. 
7. Check for convergence, i.e. if the norm $\eta_n = \lVert \phi_{n+1}(\mathbf{m_{t}},\mathbf{s_t}) - \phi_n(\mathbf{m_{t}},\mathbf{s_t}) \rVert$ is below tolerance level. If so, stop the loop. If not, start over from 3. 
