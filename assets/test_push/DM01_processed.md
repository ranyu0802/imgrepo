如何实现
1. Noise Scheduler (Sequentially Add Noise)
2. Neural Networks Predicts Noise and Images (UNet) 
3. Timestep Encoding
<!--more--> 
## Introduction

假设给定一类2D Points，如何生成新的点，也是属于这个Set ?
How can we sample a new plausible 2D point, given a set of points?
![](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/DDPM.png)




从统计的角度来看(Statistical Perspective)
+ 假设这些数据服从某种分布(underlying distribution)
+ 并且给的数据是来这个分布的样本(Sample)

> 因此如何生成新的点这个问题
> 转变成了**如果有了分布，怎么得到新的数据？**


## 图片数据

假设RGB Image的resolution是 `[256,256, 3]`
将一张图片就是这个RGB Space中的一个点   
![](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/DDPM-1.png)
将图片集$\{ x_1,x_2,...,x_n \}$ 看作来自某个 分布 $p(x)$ 的样本

**怎么得到这个分布？**


## 重参数化 Reparameterization Trick

- A sample from a normal distribution $( z \sim \mathcal{N}(\mu, \Sigma) )$ can be rewritten as follows:  

$$  
z = \mu + \Sigma^{\frac{1}{2}} \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)
$$
  
有标准正态分布，就能产生任意normal distribution的Sample
 We just need a standard normal sampler to sample from an arbitrary normal distribution.


若希望从高斯分布 $N(\mu, \sigma)$ 中采样，可以先从标准分布  $N(0, 1)$ 采样出  $z$ ，再得到 $\sigma * z + \mu$。  
这样做的好处是将随机性转移到了  $z$  这个常量上，而 $\sigma$  和  $\mu$ 则当做仿射变换网络的一部分。



### 基本思想（The Basic Idea）

- **将一个来自简单分布 $p(z)$ 的Sample**  例如标准正态分布  $\mathcal{N}(z; \mathbf{0}, \mathbf{I})$  **映射到数据分布** $p(x)$ 。
    
    - ( z )：潜变量（Latent variable）
        
    - ( p(z) )：潜在分布（Latent distribution）
        
- **从  p(z) 中采样**，并将其映射为一个数据点。
- 箭头 $D(z)$：表示将潜变量映射到数据点的解码过程（Decoder）

![|435x171](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/DDPM-2.png)

## Autoencoder

Autoencoder 是一种**神经网络**，其目的是在**将输入数据编码为低维潜在向量（latent vector）**的同时，**重构原始输入数据**。

- **编码器（Encoder，E）**：将输入数据压缩到潜在空间。
    
- **潜在表示（Latent）**：低维的特征向量，用于表示输入数据的核心信息。
    
- **解码器（Decoder，D）**：从潜在向量中重建出原始数据
![](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/DDPM-3.png)



- 我们真正**需要**的是**解码器（decoder）**，它负责将潜在向量（latent）映射回输入数据（input data）。  
    👉 即：latent → input data。
- 但问题是：**如何保证**某个潜在向量（latent）一定能够被**映射到数据分布中的一个有效数据点**？


##  2. Variational Autoencoders &  Hierarchical Variational Autoencoders

### 2.0 Basic Knowledge

#### 边缘分布（Marginal Distribution）

随机变量集合的一个 **子集（subset）** 的 **边缘分布**，是该子集内变量的概率分布。

边缘分布可表示为：

$$
p(x) = \int p(x, z) \, dz
$$

即：对 $z$ 积分，从而将 $z$ 消去，得到关于 $x$ 的边缘分布。

#### 期望值（Expected Value）

期望值是随机变量可能取值的 **加权平均数**，权重由这些取值出现的 **概率** 决定。

其数学表达式为：

$$
\mathbb{E}_{p(x)}[x] = \int x \cdot p(x) \, dx
$$
#### 贝叶斯法则（Bayes’ Rule）

贝叶斯法则是一种用于确定 **事件的条件概率（conditional probability）** 的数学公式。

其表达式为：

$$
p(z|x) = \frac{p(x|z) \, p(z)}{p(x)}
$$

其中：

- $p(z|x)$：后验概率（Posterior）  
- $p(x|z)$：似然（Likelihood）  
- $p(z)$：先验概率（Prior）  
- $p(x)$：边缘概率（Marginal）

根据联合分布关系，也可写为：

$$
p(z|x)p(x) = p(x|z)p(z) = p(x, z)
$$

#### Kullback–Leibler（KL）散度

Kullback–Leibler（KL）散度是一种衡量两个概率分布之间差异的指标。  
它用于度量一个概率分布 $p$ 与一个参考分布 $q$ 之间的不同程度。

其定义为：

$$
D_{KL}(p \parallel q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx 
= \mathbb{E}_{p(x)} \left[ \log \frac{p(x)}{q(x)} \right]
$$

https://www.cnblogs.com/qizhou/p/13804283.html

两个多元正态分布的KL散度、巴氏距离和W距离 - 苏剑林的文章 - 知乎
https://zhuanlan.zhihu.com/p/387938179


####  詹森不等式（Jensen’s Inequality）

一、凸函数的定义（Convex Function）

若函数 $f$ 是 **凸函数（convex function）**，则对于任意的 $x_1, x_2$ 和 $t \in [0,1]$，有：

$$
f(tx_1 + (1 - t)x_2) \leq t f(x_1) + (1 - t) f(x_2)
$$

几何意义：  
在凸函数中，**连接函数上两点的直线总位于函数曲线之上**。
![](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/DDPM-4.png)



 二、多点形式（仿射组合形式）

若 $f$ 是 **凸函数**，则对于任意的一组点 $x_i$ 和权重 $t_i \in [0,1]$，有：

$$
f\left( \sum_i t_i x_i \right) \leq \sum_i t_i f(x_i)
$$

其中，$\sum_i t_i x_i$ 称为 **仿射组合（Affine combination）**，  
表示加权平均形式，并满足 $\sum_i t_i = 1$。

![](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/DDPM-5.png)


三、随机变量形式（期望形式）

如果 $x$ 是一个 **随机变量**，且 $f$ 是一个 **凸函数（convex function）**，则有：

$$
f\left( \mathbb{E}_{p(x)}[x] \right) \leq \mathbb{E}_{p(x)}[f(x)]
$$

由于 **期望值（expected value）** 是加权平均（仿射组合）的一种形式，  
因此该不等式是詹森不等式的概率形式表达。





### 2.1 VAE

+ 将从潜在分布 $p(z)$ 到数据分布 $p(x)$ 的映射，表示为一个条件分布 $p(x|z)$
+ 假设Variance Fixed 的情况下
+ **解码器（decoder）** 或 **生成器（generator）**视为在预测 **条件分布** $p(x|z)$ 的 **均值（mean）**

$$
p(z) = \mathcal{N}(z; 0, \sigma^2 I)
$$
$$
p(x|z) = \mathcal{N}(x; D(z), \sigma^2 I)
$$

其中：

- $D(z)$：由潜变量 $z$ 经过解码器生成的数据分布均值  
- $\sigma^2 I$：固定方差（Fixed variance）
![](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/DDPM-6.png)


对于所有给定的真实图像 $x$，  
我们的目标是 **最大化边缘概率（marginal probability）**：

$$
p(x) = \int p(x, z) \, dz = \int p(x|z) \, p(z) \, dz
$$

其中：

- $p(x|z)$：由潜变量 $z$ 生成数据 $x$ 的条件概率  
- $p(z)$：潜在分布（通常为标准正态分布）  
- 目标：通过优化模型参数，使生成数据的边缘概率最大化。

如何计算这个 **积分（integral）**？
使用 **蒙特卡洛方法（Monte-Carlo method）** 对 $x$ 和 $z$ 进行采样时，  
计算量非常大，耗时过长。
因此：
$$
\text{Intractable（难以求解）.}
$$

利用Bayes's Rule
$$
p(x, z) = p(x|z)p(z) = p(z|x)p(x)
$$
可以这样计算：

$$
p(x) = \frac{p(x, z)}{p(z|x)} = \frac{p(x|z)p(z)}{p(z|x)}
$$
但问题在于：  
**这个条件分布 $p(z|x)$ 是未知的（unknown）**，  
因此我们无法直接计算它，需要引入一个近似分布$q_{\phi}(z|x)$来替代。

利用KL散度
https://yonigottesman.github.io/2023/03/11/vae.html
https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html
https://faculty.washington.edu/yenchic/short_note/note_vae.pdf


证据下界（Evidence Lower Bound, ELBO）

我们希望最大化边缘似然：

$$
\log p(x) = \log \int p(x, z) \, dz
$$

引入一个近似分布 $q_\phi(z|x)$（编码器），可写为：

$$
\log p(x) = \log \int p(x, z) \frac{q_\phi(z|x)}{q_\phi(z|x)} \, dz
$$

将积分形式改写为对 $q_\phi(z|x)$ 的期望：

$$
\log p(x) = \log \, \mathbb{E}_{q_\phi(z|x)} 
\left[ \frac{p(x, z)}{q_\phi(z|x)} \right]
$$

由于 **$\log$ 是凹函数（concave function）**，  
根据 **詹森不等式（Jensen’s Inequality）**，有：

$$
\log \, \mathbb{E}_{q_\phi(z|x)} 
\left[ \frac{p(x, z)}{q_\phi(z|x)} \right]
\geq 
\mathbb{E}_{q_\phi(z|x)} 
\left[ \log \frac{p(x, z)}{q_\phi(z|x)} \right]
$$

右侧即为 **ELBO（证据下界）**，是对 $\log p(x)$ 的下界。

变分分布（Variational Distribution）

- $q_\phi(z|x)$ 是一个 **变分分布（variational distribution）**，带有参数 $\phi$。  
- 例如：可以设为一个 **高斯分布（Gaussian distribution）**，其均值和方差由参数决定。  
- 我们将 $q_\phi(z|x)$ 视为一个 **任意的条件分布（arbitrary conditional distribution）**，它 **不一定等同于** 真实的后验分布 $p(z|x)$。  
- 由于真实的 $p(z|x)$ **未知（unknown）**，我们使用这个 **代理分布（proxy distribution）** $q_\phi(z|x)$ 来进行近似。

期望项解释（Expectation Term）

$$
\mathbb{E}_{q_\phi(z|x)} 
\left[ \log \frac{p(x, z)}{q_\phi(z|x)} \right]
$$

表示在从 **变分分布（variational distribution）** $q_\phi(z|x)$ 中采样得到的 $z$ 上，  
计算该表达式的 **期望值（expected value）**。


![[DDPM-6.png]]

### 2.2 Hierarchical Variational Autoencoders
https://www.zhihu.com/tardis/zm/art/600047951?source_id=1003
https://www.zhangzhenhu.com/aigc/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.html




![](https://raw.githubusercontent.com/ranyu0802/imgrepo/main/assets/test_picgo5/202501.png)