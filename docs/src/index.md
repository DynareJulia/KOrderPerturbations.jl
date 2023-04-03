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

${\mathcal M}_{k,k}$ contains $k$ partitions with one element each:  

$\{\{1\},\{2\}, \ldots, \{k\}\}$.

### Implementation

####

1. ${\mathcal M}_{k,i}$ is obtained by ``Combinatorics.partitions(k,
   i)``
   
2. 
```math
 \prod_{m=1}^i\left[h_{x^{|c_m|}}\right]^{\beta_m}_{\alpha(c_m)}
 ```
   compute Kronecker product in a loop for

```math
   K = \left[h_{x^{|c_1|}}\right]\otimes \left[h_{x^{|c_2|}}\right]\otimes
  \ldots\otimes \left[h_{x^{|c_i|}}\right]$
  ```
3. ``reshape()`` the resulting matrix in a multidimensional array
4. 
```math
\sum_{c \in{\mathcal
M}_{k,i}}\prod_{m=1}^i\left[h_{x^{|c_m|}}\right]^{\beta_m}_{\alpha(c_m)}

```
sum all the permutations of the matrix with ``PermutedDimsArray``

5. 
```math
\sum_{i=1}^k\left[g_{y^i}\right]^\gamma_{\beta_1\ldots\beta_i}\sum_{c \in{\mathcal
M}_{k,i}}\prod_{m=1}^i\left[h_{x^{|c_m|}}\right]^{\beta_m}_{\alpha(c_m)}

```
redo steps 2-4 for every $i=1,\ldots, k$ and multiply by
$\left[g_{y^i}\right]^\gamma_{\beta_1\ldots\beta_i}$.

### Tests

Compute the third order partial derivatives for 
```math
y = h(x) = \left[\begin{array}{c} x_1^3 +3x_2 \\ -x_2^3 + 2 x_3^2 \\ x_3^3
\end{array}\right]\\
g(y) = \left[\begin{array}{c} y_1^3 + y_2 \\ y_2^3 + 2y_2^2 + 2 y_3 \\ y_3^3 + 3y_2
\end{array}\right]
```

## Taylor expansion 

Taylor expansion:
```math
\sum_{i=1}^p\left[f_{y^i}\right]_{\beta_1\ldots\beta_i}\sum_{c \in{\mathcal
M}_{p,i}}\prod_{m=1}^i\left[g_{x^{|c_m|}}\right]^{\beta_m}_{\alpha(c_m)}
= 0
```
Note that $g_{y^p}$ appears only in the first term of the sum above
and

```math
\left[f_y\right]\left[g_{x^p}\right] = -\sum_{i=2}^p\left[f_{y^i}\right]_{\beta_1\ldots\beta_i}\sum_{c \in{\mathcal
M}_{p,i}}\prod_{m=1}^i\left[g_{x^{|c_m|}}\right]^{\beta_m}_{\alpha(c_m)}
```

Original DSGE model:
```math
   \mathbb{E}_tF(y_{t-1}, u_t, \sigma) =  \mathbb{E}_tf\left(g\left(g\left(y_{t-1},u_t,\sigma\right),\sigma\epsilon_{t+1},\sigma\right),g\left(y_{t-1},u_t,\sigma\right),y_{t-1},u_t\right)=0
```
that involves two compositions of functions.

The Taylor expansion is then
```math
    (f_{y^+}g_y + f_0)g_{y^p} + f_{y^+}g_{y^p}g_y^{\otimes^p} = K_1
```
where only the terms containing $g_{y^p}$ are explicit and all other
terms are summarized in $K_1$.

### Solving for $g_{y^p}$

The above equation can be rewritten

```math
    g_{y^p} + (f_{y^+}g_y + f_0)^{-1}f_{y^+}g_{y^p}g_y^{\otimes^p} = (f_{y^+}g_y + f_0)^{-1}K_1
```

and solved with a suitable algorithm

### Solving for $g_{y^iu^k}$

Solving for cross-derivatives $g_{y^iu^k} such that $i+k=p$ requires
only solving the linear system

```math
   (f_{y^+}g_y + f_0)g_{y^iu^k} = K_2
```

