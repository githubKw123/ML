# 变分推断

#### 背景

如果推断要求的后验的参数空间十分大，无法精确求解，只能通过近似方式求解，包括：
确定性近似-如变分推断
随机近似-如 MCMC，MH，Gibbs

#### 基于平均场假设的变分推断

我们记 $X$ 为观测数据
$Z$ 为隐变量和参数的集合（这里把参数$\theta$也包含进来是因为贝叶斯派认为$\theta$不是确定的值而是一个随机变量，所以这里把两个随机变量写在了一起）

EM 中的推导：

$$
\log p(X)=\log p(X,Z)-\log p(Z|X)=\log\frac{p(
X,Z)}{q(Z)}-\log\frac{p(Z|X)}{q(Z)}
$$

左右两边分别积分：

$$
Left:\int_Zq(Z)\log p(X)dZ=\log p(X)\\
Right:\int_Z[\log \frac{p(X,Z)}{q(Z)}-\log \frac{p(Z|X)}{q(Z)}]q(Z)dZ\\=\int_Z\log \frac{p(X,Z)}{q(Z)}{q(Z)}dZ-\int_Z\log \frac{p(Z|X)}{q(Z)}q(Z)dZ\\
=ELBO+KL(q,p)\\
=L(q)+KL(q,p)
$$

对于右边的式子，我们用一个以$q$为输入函数代替ELBO,为什么要这样做：
因为我们的目的是求后验$p(Z|X)$，这是很难求的，但是我们发现KL是关于$p$和$q$的，如果我们求一个与$p$很接近的$q$不就可以了
又考虑到等式左边为常数，让$q$接近的$p$就等价于KL散度等于零等价于让$L(q)$最大,于是问题就等价于：

$$
\hat{q}(Z)=\mathop{argmax}\limits_{q(Z)}L(q)
$$

$q(Z)$是好多隐变量和参数组成的联合概率分布， 为了计算方便，我们假设 $q(Z)$ 可以划分为 $M$ 个独立的组（平均场近似）：

$$
q(Z)=\prod\limits_{i=1}^Mq_i(z_i)
$$

因此，在 $L(q)=\int_Zq(Z)\log p(X,Z)dZ-\int_Zq(Z)\log{q(Z)}$ 中，看 $p(Z_j)$

第一项,这里$Z=(z_1,z_2,...,z_m)$：

$$
\int_Zq(Z)\log p(X,Z)dZ=\int_Z\prod\limits_{i=1}^Mq_i(z_i)\log p(X,Z)dZ\\
=\int_{z_j}q_j(z_j)[\int_{Z-z_{j}}\prod\limits_{i\ne j}q_i(z_i)\log p(X,Z)d(Z-z_j)]dz_j\\
=\int_{z_j}q_j(z_j)\mathbb{E}_{\prod\limits_{i\ne j}q_i(z_i)}[\log p(X,Z)]dz_j
$$

第二项：

$$
\int_Zq(Z)\log q(Z)dZ=\int_Z\prod\limits_{i=1}^Mq_i(z_i)\sum\limits_{i=1}^M\log q_i(z_i)dZ
$$

展开求和项第一项为：

$$
\int_Z\prod\limits_{i=1}^Mq_i(z_i)\log q_1(z_1)dZ=\\
\int_{z_1}q_1(z_1)\log q_1(z_1)dz_1\int_{z_2}q_1(z_2)dz_2\int_{z_3}q_1(z_3)dz_3......\\
=\int_{z_1}q_1(z_1)\log q_1(z_1)dz_1
$$

所以第二项就可以写成：

$$
\int_Zq(Z)\log q(Z)dZ=\sum\limits_{i=1}^M\int_{z_i}q_i(z_i)\log q_i(z_i)dz_i=\\
\int_{z_j}q_j(z_j)\log q_j(z_j)dz_j+Const
$$

这样的话两项就有一个很相似的形式，令 $\mathbb{E}_{\prod\limits_{i\ne j}q_i(z_i)}[\log p(X,Z)]=\log \hat{p}(X,z_j)$ ，让两项相减，可以得到：

$$
\int_{z_j}q_j(z_j)\log\frac{q_j(z_j)}{\hat{p}(X,z_j)}dz_j\\
=-KL(q_j||\hat{p}(X,z_j))
$$

于是我们令 $q_j(z_j)=\hat{p}(X,z_j)=\mathbb{E}_{\prod\limits_{i\ne j}q_i(z_i)}[\log p(X,Z)]$ 才能得到最大值。

对于上面的等式，我们可以用坐标上升法进行求解，我们看到，对每一个 $q_j$，都是固定其余的 $q_i$，求这个值，于是可以使用坐标上升的方法进行迭代求解，上面的推导针对单个样本，但是对数据集也是适用的。

基于平均场假设的变分推断存在一些问题：

1. 假设太强，$Z$ 非常复杂的情况下，假设不适用
2. 期望中的积分，可能无法计算

## SGVI

基于平均场的变分推断可以导出坐标上升的算法，但是这个假设在一些情况下假设太强，同时积分也不一定能算。
我们知道，优化方法除了坐标上升，还有梯度上升的方式，我们希望通过梯度上升来得到变分推断的另一种算法。

我们的目标函数：

$$
\hat{q}(Z)=\mathop{argmax}_{q(Z)}L(q)
$$

这里要进行梯度上升，就要给q一个形式，所以我们假定 $q(Z)=q_\phi(Z)$，是和 $\phi$ 这个参数相连的概率分布，于是：

$$
\mathop{argmax}\limits_{q(Z)}L(q)=\mathop{argmax}\limits_{\phi}L(\phi)
$$


$$
L(\phi)=\mathbb{E}_{q_\phi}[\log p_\theta(x^i,z)-\log q_\phi(z)]
$$

这里 $x^i$ 表示第 $i$ 个样本。

下面对$L$求梯度

$$
\nabla_\phi L(\phi)=\nabla_\phi\mathbb{E}_{q_\phi}[\log p_\theta(x^i,z)-\log q_\phi(z)]\\
=\nabla_\phi\int q_\phi(z)[\log p_\theta(x^i,z)-\log q_\phi(z)]dz\\
=\int\nabla_\phi q_\phi(z)[\log p_\theta(x^i,z)-\log q_\phi(z)]dz+\int q_\phi(z)\nabla_\phi [\log p_\theta(x^i,z)-\log q_\phi(z)]dz\\
=\int\nabla_\phi q_\phi(z)[\log p_\theta(x^i,z)-\log q_\phi(z)]dz-\int q_\phi(z)\nabla_\phi \log q_\phi(z)dz\\
=\int\nabla_\phi q_\phi(z)[\log p_\theta(x^i,z)-\log q_\phi(z)]dz-\int \nabla_\phi q_\phi(z)dz\\
=\int\nabla_\phi q_\phi(z)[\log p_\theta(x^i,z)-\log q_\phi(z)]dz- \nabla_\phi\int q_\phi(z)dz\\
=\int\nabla_\phi q_\phi(z)[\log p_\theta(x^i,z)-\log q_\phi(z)]dz- \nabla_\phi 1\\
=\int\nabla_\phi q_\phi(z)[\log p_\theta(x^i,z)-\log q_\phi(z)]dz\\=\int q_\phi(\nabla_\phi\log q_\phi)(\log p_\theta(x^i,z)-\log q_\phi(z))dz\\=\mathbb{E}_{q_\phi}[(\nabla_\phi\log q_\phi)(\log p_\theta(x^i,z)-\log q_\phi(z))]
$$

这个期望可以通过蒙特卡洛采样来近似，从而得到梯度，然后利用梯度上升的方法来得到参数：





$$
z^l\sim q_\phi(z)\\
\mathbb{E}_{q_\phi}[(\nabla_\phi\log q_\phi)(\log p_\theta(x^i,z)-\log q_\phi(z))]\sim \frac{1}{L}\sum\limits_{l=1}^L(\nabla_\phi\log q_\phi)(\log p_\theta(x^i,z)-\log q_\phi(z))
$$

但是由于求和符号中存在一个对数项，导致我们采样$q_\phi(z)$如果解近0，这个函数值是变化是非常大的，导致了方差很大，需要采样的样本非常多。
为了解决方差太大的问题，我们采用 Reparameterization 的技巧。

考虑对于下面的式子，我们能不能让$q_\phi$跟$\phi$没有关系呢，也就是让$\phi$确定，让分布的随机性转移到另一个参数$\varepsilon$上：

$$
\nabla_\phi L(\phi)=\nabla_\phi\mathbb{E}_{q_\phi}[\log p_\theta(x^i,z)-\log q_\phi(z)]
$$

我们取：$z=g_\phi(\varepsilon,x^i),\varepsilon\sim p(\varepsilon)$，于是对后验：$z\sim q_\phi(z|x^i)$，有定理$|q_\phi(z|x^i)dz|=|p(\varepsilon)d\varepsilon|$。代入上面的梯度中：

$$
\nabla_\phi L(\phi)=\nabla_\phi\mathbb{E}_{q_\phi}[\log p_\theta(x^i,z)-\log q_\phi(z)]\\
=\nabla_\phi L(\phi)=\nabla_\phi\int[\log p_\theta(x^i,z)-\log q_\phi(z)]q_\phi dz\\
=\nabla_\phi\int[\log p_\theta(x^i,z)-\log q_\phi(z)]p_\varepsilon d\varepsilon\\
=\mathbb{E}_{p(\varepsilon)}[\nabla_\phi[\log p_\theta(x^i,z)-\log q_\phi(z)]]\\
=\mathbb{E}_{p(\varepsilon)}[\nabla_z[\log p_\theta(x^i,z)-\log q_\phi(z)]\nabla_\phi z]\\
=\mathbb{E}_{p(\varepsilon)}[\nabla_z[\log p_\theta(x^i,z)-\log q_\phi(z)]\nabla_\phi g_\phi(\varepsilon,x^i)]
$$

对这个式子进行蒙特卡洛采样，然后计算期望，得到梯度。

之后梯度上升：

$$
\phi^{t+1}\leftarrow\phi^{t}+\lambda^t\nabla_\phi L(\phi)
$$

