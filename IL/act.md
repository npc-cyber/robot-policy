# ACT算法 

[ACT论文地址](https://arxiv.org/abs/2304.13705)

[ACT代码地址](https://github.com/tonyzhaozh/act)

## ACT算法值得关注的点
1. action时间加权处理 Figure5
   每个time都预测k步，每次执行的action都是policy不同time预测的action的加权和，时间越久权重越高。
   成功率基本上会高3%左右。
   
   ```
    import numpy as np
    k = 0.01
    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    exp_weights = exp_weights / exp_weights.sum()
   ```
   
2. 模型本身 Figure4 Figure11
   left 通过transformer encoder编码 cls(nn.Embedding 没有用到图像做条件是想训练的快点 而是用了可以学习的emb) state action+posemb 预测隐变量z
   right1 通过resnet18(注意是cnn不是transfomer，面试时尴尬了)提取图像特征，随后图像特征 state 隐变量z 通过 transformer encoder编码
   right2 通过transformer decoder解码出action
   **注意Figure11 测试时不推理隐变量z 直接初始化全0**
   隐变量z 应该就是机械臂的数据分布
   
3. 结果 Figure8
   | 因素 | 影响 | 
    | --- | --- |
    | k的长度   | k=1时是纯开环了 随着k的增加 成功率上升到峰值后下降   | 
    | action时间加权   | 成功率增加3%左右   |
   | 带不带CAVE 训练时是否预测隐变量z 不预测的话其实就是BC算法| scripts采集数据成功率高1% 人类采集数据成功率高很多 |
   | 采集帧率| 高一点好 很多任务需要实时反馈的 |

4. 难点CVAE的处理 [CVAE](https://zhuanlan.zhihu.com/p/611498730)
   AE->VAE->CVAE
   | 模型 | 原理 |
    | --- | --- |
   | AE | x->通过模型预测均值方差->y |
   | VAE | x->通过模型预测均值方差z->y 增加了通过KL散度比较预测值z与标准高斯分布的差异，后续可以通过高斯分布直接求y |
   | CVAE | x y ->通过模型预测均值方差z -> 通过x z->y |
   
6. ACT提到了action最好是主从臂 比如夹爪很多时候需要更大力
   It is important to use the leader joint positions instead of the follower’s, because the amount of force applied is implicitly defined by the difference between them, through the low-level PID controller. 
