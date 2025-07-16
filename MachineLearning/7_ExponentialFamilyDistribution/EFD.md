 # 指数族分布

**定义**：

指数族是一类分布，包括**高斯分布、伯努利分布、二项分布、泊松分布、Beta 分布、Dirichlet 分布、Gamma 分布**等一系列分布。

指数族分布可以写为统一的形式：

$$
p(x|\eta)=h(x)\exp(\eta^T\phi(x)-A(\eta))=\frac{1}{\exp(A(\eta))}h(x)\exp(\eta^T\phi(x))
$$

其中
$\eta$ : 参数向量
$A(\eta)$ : 对数配分函数（类似归一化因子）。

**特点：**
**(1)充分统计量**：$ \phi(x)$ 叫做充分统计量，包含样本集合所有的信息，有了这个量，样本就可以扔掉了，例如高斯分布中的均值和方差。充分统计量在在线学习中有应用，对于一个数据集，只需要记录样本的充分统计量即可。
**(2)共轭**：$p(z|x) \sim p(x|z)p(z)$，给定一个似然$p(x|z)$，如果它具有一个与其共轭的先验$p(z)$，那么后验跟先验有相同的分布形式，以此简化贝叶斯公式。
**(3)最大熵**：最大熵原理是在满足所有已知约束条件（如均值、方差或其他统计量）的情况下，选择熵最大（最随机）的概率分布。给定一组随机的数据，通过最大熵原则推出来的无信息先验$p(z)$就是符合指数族分布的。

**引申方法：**

**(1)广义线性模型**：

$$
线性组合：y=f(w^Tx)\\
link \ fuction \\ 指数族分布y|x\sim Exp Family
$$

**(2)概率图模型**
**(3)变分推断**

## 一维高斯分布

一维高斯分布可以写成：

$$
p(x|\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

将这个式子改写：

$$
\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x^2-2\mu x+\mu^2))\\
=\exp(\log(2\pi\sigma^2)^{-1/2})\exp(-\frac{1}{2\sigma^2}\begin{pmatrix}-2\mu&1\end{pmatrix}\begin{pmatrix}x\\x^2\end{pmatrix}-\frac{\mu^2}{2\sigma^2})\\
=\exp{(\frac{\mu}{\sigma^2}-\frac{1}{2\sigma^2})\begin{pmatrix}x\\x^2\end{pmatrix}-(-\frac{\mu^2}{2\sigma^2}+\frac12\log(2\pi\sigma^2))}
$$

所以：

$$
\eta=\begin{pmatrix}\frac{\mu}{\sigma^2}\\-\frac{1}{2\sigma^2}\end{pmatrix}=\begin{pmatrix}\eta_1\\\eta_2\end{pmatrix}
$$

于是 $A(\eta)$：

$$
A(\eta)=-\frac{\eta_1^2}{4\eta_2}+\frac{1}{2}\log(-\frac{\pi}{\eta_2})
$$

## 充分统计量和对数配分函数的关系

对于

$$
p(x|\eta)=h(x)\exp(\eta^T\phi(x)-A(\eta))
$$

$\eta$为参数向量，$A(\eta)$为对数配分函数，$\phi(x)$为充分统计量，我们想知道$A(\eta)$与$\phi(x)$具体又怎样的联系。

$$
p(x|\eta)=h(x)\exp(\eta^T\phi(x)-A(\eta))\\=\frac{1}{\exp(A(\eta))}h(x)\exp(\eta^T\phi(x))
$$

这样能得到（类似于贝叶斯下面积分部分）：

$$
\exp(A(\eta))=\int h(x)\exp(\eta^T\phi(x))dx
$$

两边对参数求导：

$$
\exp(A(\eta))A'(\eta)=\int h(x)\exp(\eta^T\phi(x))\phi(x)dx =\frac{\int h(x)\exp(\eta^T\phi(x))\phi(x)dx}{\exp(A(\eta))} \\
=\int h(x)\exp(\eta^T\phi(x)-A(\eta))\phi(x)dx =\int p(x|\eta) \phi(x)dx \\
\Longrightarrow A'(\eta)=\mathbb{E}_{p(x|\eta)}[\phi(x)]
$$

$$

$$

类似的：

$$
A''(\eta)=Var_{p(x|\eta)}[\phi(x)]
$$

由于方差为正，于是 $A(\eta)$ 一定是凸函数。

综上对数配分函数的一阶导是充分统计量的方差，二阶导为充分统计量的均值

## 充分统计量和极大似然估计

从极大似然估计的角度来看参数$\eta$怎么求

对于独立全同采样得到的数据集 $\mathcal{D}=\{x_1,x_2,\cdots,x_N\}$。

$$
\eta_{MLE}=\mathop{argmax}_\eta\sum\limits_{i=1}^N\log p(x_i|\eta)\\=\mathop{argmax}_\eta\sum\limits_{i=1}^N log(h(x_i)exp(\eta^T\phi(x_i)-A(\eta)))\\
=\mathop{argmax}_\eta\sum\limits_{i=1}^N(\eta^T\phi(x_i)-A(\eta))\\


\Longrightarrow A'(\eta_{MLE})=\frac{1}{N}\sum\limits_{i=1}^N\phi(x_i)
$$

由此可以看到，$A(\eta)$是知道的，$A'(\eta)$是知道的，那么$A'(\eta)$的具体函数值知道后（上式），肯定能得到对应的变量$\eta_{MLE}$。

## 最大熵

信息量：$-logp$

> $p$为随机事件或随机变量发生的概率，随机事件越容易发生，随机变量越准确，信息量越少

信息熵：$Entropy=E[-logp]=\int-p(x)\log(p(x))dx$

> 熵是可能性的衡量，一般地，对于完全随机的变量（等可能），信息熵最大。

最大熵原则：主张在既定事实，或者已知约束条件下，选择熵最大的概率分布作为最合理的分布。

> 我们的假设为最大熵原则，假设数据是离散分布的，$k$ 个特征的概率分别为 $p_k$，最大熵原理可以表述为：
> 
> $$
> max\{H(p)\}=\min\{\sum\limits_{k=1}^Kp_k\log p_k\}\ s.t.\ \sum\limits_{k=1}^Kp_k=1
> $$
> 
> 利用 Lagrange 乘子法：
> 
> $$
> (p,\lambda)=\sum\limits_{k=1}^Kp_k\log p_k+\lambda(1-\sum\limits_{k=1}^Kp_k)
> $$
> 
> 于是可得：
> 
> $$
> _1=p_2=\cdots=p_K=\frac{1}{K}
> $$
> 
> 因此等可能的情况熵最大。

那么机器学习中，我们有的$N$个样本就是既定事实，我们对这个既定事实求概率分布

那么怎么把这个转换成约束呢，一个数据集 $\mathcal{D}$可以简单地转换为经验分布为 $\hat{p}(x=X)=\frac{Count(X)}{N}$（类似古典概率，样本中X的数除以样本总数）

那么有了这个经验分布，所有的数字特征就可以求出来，包括期望、均值，也包括对应函数的期望均值总能求出来（因为是古典概率相当于直接数数吗），因此约束可以表示为：

对任意一个函数$f(x)$其均值是可知的，$\mathbb{E}_{\hat{p}}[f(x)]=\Delta$

于是有**模型**：

$$
\max\{H(p)\}=\min\{\sum\limits_{k=1}^Np_k\log p_k\}\\ s.t.\ \sum\limits_{k=1}^Np_k=1,\\ \mathbb{E}_p[f(x)]=\Delta
$$

Lagrange 函数为：

$$
L(p,\lambda_0,\lambda)=\sum\limits_{k=1}^Np_k\log p_k+\lambda_0(1-\sum\limits_{k=1}^Np_k)+\lambda^T(\Delta-\mathbb{E}_p[f(x)])
$$

求导得到($\mathbb{E}_p[f(x)]=\sum\limits_{k=1}^Np(x)f(x)$)：

$$
\frac{\partial}{\partial p(x)}L=\sum\limits_{k=1}^N(\log p(x)+1)-\sum\limits_{k=1}^N\lambda_0-\sum\limits_{k=1}^N\lambda^Tf(x)\\
\Longrightarrow\sum\limits_{k=1}^N[\log p(x)+1-\lambda_0-\lambda^Tf(x)]=0
$$

因为p(x)是向量，求导后结果也是向量，向量的每个元素都是0，所以对每个pi(x)求导后都是0，联想到矩阵论中的矩阵求导

由于数据集是任意的，对数据集求和也意味着求和项里面的每一项都是0：

$$
p(x)=\exp(\lambda^Tf(x)+\lambda_0-1)
$$

这里推出来的就是一个指数族分布。

