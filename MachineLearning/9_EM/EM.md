# EM算法

**动机：** 对于一般的概率模型$p(x|\theta)$，MLE 对参数的估计记为：$\theta_{MLE}=\mathop{argmax}\limits_\theta\log p(x|\theta)$，但是对于包含隐变量的模型来说，求出它的解析解是十分困难的。

EM算法的目的是解决具有隐变量的概率的参数估计（极大似然估计），其中E步表示期望，M步表示最大。

**（注意：EM是一种算法，类似于梯度下降而不是模型）**

**算法步骤：**

EM 算法对这个问题的解决方法是采用迭代的方法：

$$
\theta^{t+1}=\mathop{argmax}\limits_{\theta}\int_z\log [p(x,z|\theta)]p(z|x,\theta^t)dz
$$

其中$p(z|x,\theta^t)$表示给定$x$和上一时刻参数$\theta$的后验

$p(x,z|\theta)$叫完整数据，表示联合概率分布

上式其实可以看成一个期望：

$$
\theta^{t+1}=\mathop{argmax}\limits_{\theta}\int_z\log [p(x,z|\theta)]p(z|x,\theta^t)dz=\mathbb{E}_{z|x,\theta^t}[\log p(x,z|\theta)]
$$

这个公式包含了迭代的两步：

1. E step：计算 $\log p(x,z|\theta)$ 在概率分布 $p(z|x,\theta^t)$ 下的期望
2. M step：计算使这个期望最大化的参数得到下一个 EM 步骤的输入

**收敛性证明**

> 求证：$\log p(x|\theta^t)\le\log p(x|\theta^{t+1})$
>
> 证明：$\log p(x|\theta)=\log p(z,x|\theta)-\log p(z|x,\theta)$，左右两边对$p(z|x,\theta)$求期望：
>
> $$
> left:\int_zp(z|x,\theta^t)\log p(x|\theta)dz=\log p(x|\theta) \int_zp(z|x,\theta^t)dz=\log p(x|\theta)
> $$

> $$
> right:\int_zp(z|x,\theta^t)\log p(x,z|\theta)dz-\int_zp(z|x,\theta^t)\log p(z|x,\theta)dz=Q(\theta,\theta^t)-H(\theta,\theta^t)
> $$
>
> 所以：
>
> $$
> log p(x|\theta)=Q(\theta,\theta^t)-H(\theta,\theta^t)
> $$
>
> 由于 $Q(\theta,\theta^t)=\int_zp(z|x,\theta^t)\log p(x,z|\theta)dz$，而 $\theta^{t+1}=\mathop{argmax}\limits_{\theta}\int_z\log [p(x,z|\theta)]p(z|x,\theta^t)dz$，所以 $Q(\theta^{t+1},\theta^t)\ge Q(\theta^t,\theta^t)$。
>
> 要证 $\log p(x|\theta^t)\le\log p(x|\theta^{t+1})$，需证：$H(\theta^t,\theta^t)\ge H(\theta^{t+1},\theta^t)$：
>
> $$
> H(\theta^{t+1},\theta^t)-H(\theta^{t},\theta^t)=\int_zp(z|x,\theta^{t})\log p(z|x,\theta^{t+1})dz-\int_zp(z|x,\theta^t)\log p(z|x,\theta^{t})dz\\
> =\int_zp(z|x,\theta^t)\log\frac{p(z|x,\theta^{t+1})}{p(z|x,\theta^t)}=-KL(p(z|x,\theta^t),p(z|x,\theta^{t+1}))\le0
> $$
>
> 综合上面的结果：
>
> $$
> log p(x|\theta^t)\le\log p(x|\theta^{t+1})
> $$

**推导**

1.ELBO推导

对于要求的$p(x|\theta)$：

$$
\log p(x|\theta)=\log p(z,x|\theta)-\log p(z|x,\theta)=\log \frac{p(z,x|\theta)}{q(z)}-\log \frac{p(z|x,\theta)}{q(z)}
$$

分别对两边求期望 $\mathbb{E}_{q(z)}$：

$$
Left:\int_zq(z)\log p(x|\theta)dz=\log p(x|\theta)\\
Right:\int_zq(z)\log \frac{p(z,x|\theta)}{q(z)}dz-\int_zq(z)\log \frac{p(z|x,\theta)}{q(z)}dz=ELBO+KL(q(z),p(z|x,\theta))
$$

对于左式，把$\log p(x|\theta)$提出来，里面是一个对概率密度函数的积分，值为1，所以不变

对于右式，前一部分叫做Evidence Lower Bound(ELBO)，后一部分是KL散度，因为KL散度是大于等于0的，所以$\log p(x|\theta)\ge ELBO$，等于号取在 KL 散度为0是，即：$q(z)=p(z|x,\theta)$

EM 算法的目的是将 ELBO 最大化，根据上面的证明过程，在每一步 EM 后，求得了最大的ELBO，并根据这个使 ELBO 最大的参数代入下一步中：

$$
\hat{\theta}=\mathop{argmax}_{\theta}ELBO=\mathop{argmax}_\theta\int_zq(z)\log\frac{p(x,z|\theta)}{q(z)}dz
$$

由于 $ q(z)=p(z|x,\theta^t)$ 的时候，这一步的最大值才能取等号，所以：

$$
\hat{\theta}=\mathop{argmax}_{\theta}ELBO\\
=\mathop{argmax}_\theta\int_zq(z)\log\frac{p(x,z|\theta)}{q(z)}dz\\
=\mathop{argmax}_\theta\int_zp(z|x,\theta^t)\log\frac{p(x,z|\theta)}{p(z|x,\theta^t)}d z\\
=\mathop{argmax}_\theta\int_z p(z|x,\theta^t)\log p(x,z|\theta)
$$

这个式子就是上面 EM 迭代过程中的式子。

2.Jensen 不等式角度

> Jensen 不等式就是凸函数的性质，这里函数为log，所以考虑为凹函数性质
> $ f(\theta x+(1-\theta)y) \geq \theta f(x)+(1-\theta)f(y)$
> 取$\theta=\frac12$,有$f(\frac12 x+\frac12 y) \geq \frac12 f(x)+\frac12 f(y)$
> 也就是期望的函数值大于函数值的期望

$$
\log p(x|\theta)=\log\int_zp(x,z|\theta)dz=\log\int_z\frac{p(x,z|\theta)q(z)}{q(z)}dz\\
=\log \mathbb{E}_{q(z)}[\frac{p(x,z|\theta)}{q(z)}]\ge \mathbb{E}_{q(z)}[\log\frac{p(x,z|\theta)}{q(z)}]
$$

其中，右边的式子就是 ELBO，等号在 $ p(x,z|\theta)=Cq(z)$ 时成立。于是：

$$
\int_zq(z)dz=\frac{1}{C}\int_zp(x,z|\theta)dz=\frac{1}{C}p(x|\theta)=1\\
\Rightarrow q(z)=\frac{1}{p(x|\theta)}p(x,z|\theta)=p(z|x,\theta)
$$

我们发现，这个过程就是上面的最大值取等号的条件。

## 广义 EM

**隐变量生成模型：** 对于一组可观测的样本$X=\{x_i\}_{i=1}^N$,它的形式可能非常复杂，也就是说求解$p(x|\theta)$时十分困难的，于是我们人为地引入一组隐变量$Z=\{z_i\}_{i=1}^N$,$Z$用于支撑生成$X$.（注意隐变量是取决于观测数据的，观测数据不同隐变量也会有所不同，从这个角度看，隐变量可以看作一种“参数”）

**EM算法：** EM算法就是对于这类问题进行参数估计的，对学习任务 $p(x|\theta)$，就是学习任务 $\frac{p(x,z|\theta)}{p(z|x,\theta)}$。

对于正常的MLE，求解$logp(x|\theta)$,依照上一部分的推导，可以写作：

$$
logp(x|\theta)=ELBO+KL(q(z),p(z|x,\theta))
$$

在前面的推导中，我们假定了在 E 步骤中，$q(z)=p(z|x,\theta)$，也就是让KL=0，但这其实是有失偏颇的，$p(z|x,\theta)$ 如果无法求解，那么必须使用采样（MCMC）或者变分推断等方法来近似推断这个后验。

我们观察 KL 散度的表达式，为了最大化 ELBO，在固定的 $\theta$ 时，我们需要最小化 KL 散度，于是：

$$
\hat{q}(z)=\mathop{argmin}_qKL(p,q)=\mathop{argmax}_qELBO
$$

这就是广义 EM 的基本思路,先固定$\theta$求最好的$q$,再固定$q$，求最好的$\theta$：

1. E step：

   $$
   \hat{q}^{t+1}(z)=\mathop{argmax}_q\int_zq(z)\log\frac{p(x,z|\theta^t)}{q(z)}dz,fixed\ \theta
   $$
2. M step：

   $$
   \hat{\theta}=\mathop{argmax}_\theta \int_zq^{t+1}(z)\log\frac{p(x,z|\theta)}{q^{t+1}(z)}dz,fixed\ \hat{q}
   $$

对于上面的积分：

$$
ELBO=\int_zq(z)\log\frac{p(x,z|\theta)}{q(z)}dz=\mathbb{E}_{q(z)}[p(x,z|\theta)]+Entropy(q(z))
$$

**EM 的推广**

EM 算法类似于坐标上升法，固定部分参数，优化其他参数，再一遍一遍的迭代。这样的话，其实先E步还是先M步是无所谓的

那么对于E步存在的问题：无法求解 $z$ 后验概率，可以用变分（VBEM）或者蒙特卡洛（MCEM）的方式求解
