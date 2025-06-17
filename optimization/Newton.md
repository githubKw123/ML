### 牛顿法

牛顿法其实跟梯度法类似，只是其利用二阶信息作为下降方向

因为之研究方向， 我们令$\alpha_k=1$,因此有$x_{k+1}=x_k+p_k$,所以有$f(x_{k+1})=f(x_k+p_k)$,那么一阶泰勒展开就有

$$
f(x_{k})+\nabla f_k^Tp_k+ \frac12 p_k^T  \nabla^2 f_k^T p_k+o(...)
$$

那么求解 $\phi(p) = f(x_{k})+\nabla f_k^Tp_k+ \frac12 p_k^T  \nabla^2 f_k^T p_k$的最小值就行。

$$
\phi'(p)=\nabla f_k^T+\nabla^2 f_k^T p =0
$$

$$
\nabla^2 f_k^T p = -\nabla f_k^T
$$

这其实是个方程组，叫做牛顿方程，求解这个牛顿方差即可

