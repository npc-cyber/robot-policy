# ppo算法推荐链接

[写得非常好的ppo 适合强化学习入门](https://zhuanlan.zhihu.com/p/3333839684)

定义 policy 

## Advantage Function 

在策略梯度方法中，优势函数（Advantage Function）一般定义为：

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$

它衡量的是：**在状态 $ s_t $ 下执行动作 $ a_t $ 相比于“平均水平”有多好**。

ppo用了一个gae的优势函数，可以根据时间反向递推 假设采样了 2 1 0 三个时间步

$$
A_{t2} = \delta_{t2} = r_{t2} + \gamma V(s_{t3}) - V(s_{t2})
$$

$$
A_{t1} = \delta_{t1} + \gamma \lambda A_{t2}
$$

$$
A_{t0} = \delta_{t0} + \gamma \lambda A_{t1} = \delta_{t0} + \gamma \lambda (\delta_{t1} + \gamma \lambda A_{t2}) = \delta_{t0} + \gamma \lambda \delta_{t1} + \gamma \lambda \gamma \lambda \delta_{t2}
$$

“偏差（Bias）”和“方差（Variance）”是机器学习和统计学中的核心概念，用来描述一个估计方法的好坏。

偏差（Bias）：平均命中点离靶心有多远？

方差（Variance）：子弹之间的分散程度？

$\gamma \lambda$ 可以用来平衡偏差与方差

$\gamma \lambda = 0$ 相当于只看当前步

$\gamma \lambda = 1$ 相当于看所有步

GAE 本质上是 对不同步长的 n-step 回报进行加权平均
 
## Clipped Surrogate Objective 优势函数

Clipped Surrogate Objective 可以直接说说成优势函数的损失函数

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ 
\min\left( 
r_t(\theta) \hat{A}_t,\  
\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t 
\right) 
\right]
$$

其中：
- $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$：**概率比**，表示新旧策略(之所以说新旧是应为policy状态更新了) 在状态 $s_t$ 下选择动作 $a_t$ 的概率之比。
- $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$: 实际上代码上取的是对数概率相减 再exp
- $\hat{A}_t$：优势函数估计（通常使用 GAE）。
- $\epsilon$：**裁剪范围**（如 0.1 或 0.2），控制允许偏离 1 的程度。
- $\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)$：将 $r_t(\theta)$ 限制在 $[1 - \epsilon, 1 + \epsilon]$ 区间内。
- $\mathbb{E}_t$：对所有时间步 $t$ 求期望（实践中是对一个 batch 的样本求平均）。也就是代码里面的 torch.max(pg1,pg2).mean()

## value func 价值函数

一般价值函数可以取未来的奖励，也可以取网络预测的未来价值



## 2. 价值函数的学习目标

为了训练价值函数，我们通常使用 **均方误差（Mean Squared Error, MSE）** 作为损失函数：

$$
\mathcal{L}^V(\theta_v) = \mathbb{E}_t \left[ \frac{1}{2} \left( V(s_t; \theta_v) - \hat{V}_t^{\text{target}} \right)^2 \right]
$$

其中：
- $V(s_t; \theta_v)$：Critic 网络输出的价值估计
- $\hat{V}_t^{\text{target}}$：价值目标（target），通常使用 **n-step 回报** 或 **GAE 结合价值估计** 构造。
- $\hat{V}_t^{\text{target}}$：代码里面用的是policy计算旧action的新value - (优势函数 + 旧value)
