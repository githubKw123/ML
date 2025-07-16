# 凸函数|严格凸|强凸

### 凸函数

**定义**

$$
\theta f(x)+(1-\theta)f(y)\geq f(\theta x+(1-\theta)y),for\forall x,y\in\mathbb{R}^n,\theta\in[0,1]
$$

![image.png](assets/4)

**判定条件**

$$
g(t)=f(x+vt)
$$

$f$为高维函数，$g$为切出来任意的一个函数，$f$为凸与$g$为凸是等价的

**一阶条件**

如果$f$可微，$f$为凸等价于$f(y)>=f(x)+f'(x)(y-x)$，

如果$f$二阶可微，那么二阶导大于零即可

### 强凸

**定义**

两端距离不仅要大于等于0，还要大于等于一个正数，m强凸

$$
\theta f(x)+(1-\theta)f(y)\geq f(\theta x+(1-\theta)y)+\frac{m}{2}\theta(1-\theta)||x-y||^2,for\forall x,y\in\mathbb{R}^n,\theta\in[0,1]
$$

## 约束优化问题

支持向量机本质上是一个优化问题需要用到拉格朗日对偶性进行求解，这里对其进行解释，一般地，约束优化问题（原问题）可以写成：

$$
min_{x\in\mathbb{R^p}}f(x)\\s.t.\ m_i(x)\le0,i=1,2,\cdots,M\\ n_j(x)=0,j=1,2,\cdots,N
$$

定义 Lagrange 函数：

$$
L(x,\lambda,\eta)=f(x)+\sum\limits_{i=1}^M\lambda_im_i(x)+\sum\limits_{i=1}^N\eta_in_i(x)
$$

那么原问题可以等价于无约束形式：

$$
\min_{x\in\mathbb{R}^p}\max_{\lambda,\eta}L(x,\lambda,\eta)\ s.t.\ \lambda_i\ge0
$$

这是由于，当满足原问题的不等式约束的时候，$\lambda_i=0$ 才能取得最大值，直接等价于原问题，如果不满足原问题的不等式约束，那么最大值就为 $+\infin$，由于需要取最小值，于是不会取到这个情况。

这个问题的对偶形式：

$$
\max_{\lambda,\eta}\min_{x\in\mathbb{R}^p}L(x,\lambda,\eta)\ s.t.\ \lambda_i\ge0
$$

对偶问题是关于 $ \lambda, \eta$ 的最大化问题。

由于：

$$
\max_{\lambda_i,\eta_j}\min_{x}L(x,\lambda_i,\eta_j)\le\min_{x}\max_{\lambda_i,\eta_j}L(x,\lambda_i,\eta_j)
$$

> 证明：显然有 $\min\limits_{x}L\le L\le\max\limits_{\lambda,\eta}L$，于是显然有 $\max\limits_{\lambda,\eta}\min\limits_{x}L\le L$，且 $\min\limits_{x}\max\limits_{\lambda,\eta}L\ge L$。

对偶问题的解小于原问题，有两种情况：

1. 强对偶：可以取等于号
2. 弱对偶：不可以取等于号

其实这一点也可以通过一张图来说明：

对于一个凸优化问题，有如下定理：

> 如果凸优化问题满足某些条件如 Slater 条件，那么它和其对偶问题满足强对偶关系。记问题的定义域为：$\mathcal{D}=domf(x)\cap dom m_i(x)\cap domn_j(x)$。于是 Slater 条件为：
> 
> $$
> exist\hat{x}\in Relint\mathcal{D}\ s.t.\ \forall i=1,2,\cdots,M,m_i(x)\lt0
> $$
> 
> 其中 Relint 表示相对内部（不包含边界的内部）。

1. 对于大多数凸优化问题，Slater 条件成立。
2. 松弛 Slater 条件，如果 M 个不等式约束中，有 K 个函数为仿射函数，那么只要其余的函数满足 Slater 条件即可。

上面介绍了原问题和对偶问题的对偶关系，但是实际还需要对参数进行求解，求解方法使用 KKT 条件进行：

> KKT 条件和强对偶关系是等价关系。KKT 条件对最优解的条件为：
> 
> 1. 可行域：
>    
>    $$
>    m_i(x^*)\le0\\
>    n_j(x^*)=0\\
>    \lambda^*\ge0
>    $$
> 2. 互补松弛 $\lambda^*m_i(x^*)=0,\forall m_i$，对偶问题的最佳值为 $d^*$，原问题为 $p^*$
>    
>    $$
>    d^*=\max_{\lambda,\eta}g(\lambda,\eta)=g(\lambda^*,\eta^*)=\min_{x}L(x,\lambda^*,\eta^*)\\
>    \le L(x^*,\lambda^*,\eta^*)=f(x^*)+\sum\limits_{i=1}^M\lambda^*m_i(x^*)\\
>    \le f(x^*)=p^*
>    $$
>    
>    为了满足相等，两个不等式必须成立，于是，对于第一个不等于号，需要有梯度为0条件，对于第二个不等于号需要满足互补松弛条件。
> 3. 梯度为0：$\frac{\partial L(x,\lambda^*,\eta^*)}{\partial x}|_{x=x^*}=0$

