# KOrderPerturbations.jl

## Derivatives

We work with vectors of multivariate functions. Their derivatives are
stored in matrices. The row of a derivative matrix corresponds to the
function in the vector of function. The columns of the derivative
matrix unfold all the partial derivatives

```math
\frac{\partial^2 F}{\partial x \partial x} = \left[\begin{array}{cccccc}
\frac{\partial^2 F_1}{\partial x_1\partial x_1} & \frac{\partial^2 F_1}{\partial x_1\partial x_2} & \ldots & \frac{\partial^2 F_1}{\partial x_2\partial x_1} & \ldots & \frac{\partial^2 F_1}{\partial x_n\partial x_n} \\
\rule{0pt}{15pt}\frac{\partial^2 F_2}{\partial x_1\partial x_1} & \frac{\partial^2 F_2}{\partial x_1\partial x_2} & \ldots & \frac{\partial^2 F_2}{\partial x_2\partial x_1} & \ldots & \frac{\partial^2 F_2}{\partial x_n\partial x_n} \\
\vdots & \vdots \ddots & \vdots & \ddots & \vdots\\
\frac{\partial^2 F_m}{\partial x_1\partial x_1} & \frac{\partial^2 F_m}{\partial x_1\partial x_2} & \ldots & \frac{\partial^2 F_m}{\partial x_2\partial x_1} & \ldots & \frac{\partial^2 F_m}{\partial x_n\partial x_n}
\end{array}\right]
```

## Faa di Bruno formula

If
$y = h(x)$ and $G(x) = g(h(x))$, then
```math
  \left[G_{x^k}\right]^\gamma_{\alpha_1,\ldots,\alpha_k} = 
\sum_{i=1}^k\left[g_{y^i}\right]^\gamma_{\beta_1\ldots\beta_i}\sum_{c \in{\mathcal
M}_{k,i}}\prod_{m=1}^i\left[h_{x^{|c_m|}}\right]^{\beta_m}_{\alpha(c_m)}
```

where ${\mathcal M}_{k,i}$ is the set of all partitions of the set of $k$
indices with $i$ classes, $|.|$ is the cardinality of a set, $c_m$ is $m$-th class of partition $c$, and ${\mathbb \alpha}(c_m)$ is a sequence of $\alpha$'s indexed by $c_m$. 

Note that ${\mathcal M}_{k,1}$ contains a single partition with all the
elements of the set: $\{\{1,\ldots,k\}\}$. 

${\mathcal M}_{k,k}$ contains $k$ partitions with one element each:  $\{\{1\},
\{2\}, \ldots, \{k\}\}$.
