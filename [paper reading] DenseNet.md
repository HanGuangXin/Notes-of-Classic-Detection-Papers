# [paper reading] DenseNet

>   https://arxiv.org/abs/1608.06993

|  topic   |                          motivation                          |                          technique                           |                         key element                          |                         use yourself                         |     relativity      |
| :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------: |
| DenseNet | [Problem to Solve](#Problem to Solve)<br />[Modifications](#Modifications) | [DenseNet Architecture](DenseNet Architecture)<br />[Advantages](#Advantages) | [Dense Block](#Dense Block)<br />[Transition Layers](#Transition Layers)<br />[Growth Rate](#Growth Rate)<br />[Bottleneck Structure](#Bottleneck Structure) | [Bottleneck Structure](#Bottleneck Structure)<br />[Feature Reuse](#Feature Reuse)<br />[Transition Layers](#Transition Layers) | blogs<br />articles |

## Motivation

### Problem to Solve

DenseNet 是在 ResNet 的基础上进行改进。

ResNet 中 **identity function** 和 **weight layers output** 以**求和的方式结合**，会**阻碍信息的传递**。

个人理解：在**channel维度上的拼接**更能保持**不同path信息的独立性**（而 ResNet 会因为相加造成**特征混叠**的情况）

### Modifications

#### Feature Concatenate

-   特征拼接的方式：**element-level 的求和** ==> **channel-level 的拼接**

#### Skip Connection

-   DenseNet **大幅度扩展了 skip connection**，取得了一系列的优点

    **训练一个完全dense的网络，然后在上面剪枝才是最好的方法，unet++如是说。**

## Technique

### DenseNet Architecture

<img src="[paper reading] DenseNet.assets/image-20201012213155211.png" alt="image-20201012213155211" style="zoom: 67%;" />

<img src="[paper reading] DenseNet.assets/image-20201012213306012.png" alt="image-20201012213306012" style="zoom: 67%;" />

DenseNet的 forward propagation 的公式：

>   注意：该公式并**不局限于一个 dense block**，而**在整个 DenseNet 满足**。

$$
\mathbf{x}_{\ell}=H_{\ell}\left(\left[\mathbf{x}_{0}, \mathbf{x}_{1}, \ldots, \mathbf{x}_{\ell-1}\right]\right)
$$

-   $\left[\mathbf{x}_{0}, \mathbf{x}_{1}, \ldots, \mathbf{x}_{\ell-1}\right]$ 

    第 0，……，$\ell-1$ 层 feature map 的拼接（concatenation）

-   $H_{\ell}$

    是**复合的函数**，依次包括三个部分：

    -   **Batch Normalization**
    -   **ReLU**
    -   **3x3 Conv**（1个）

    >   $H_{\ell}(·)$ 还包括了**维度压缩**的过程，以**提高计算效率、学习紧凑的feature representation**，详见 [bottleneck layers](#bottleneck layers) 

### Advantages

#### Parameter Efficiency & Model Compactness

-   **parameter efficiency** ==> **less overfitting**（**参数的高效利用**会一定程度上**避免过拟合**）

    >   One positive side-effect of the **more efficient use of parameters** is a tendency of DenseNets to be **less prone to overfitting**.

-   **feature reuse** ==> **model compactness**

实现的方式有两个：

-   **bottleneck structure**
-   **compression of transition layers**

>   The DenseNet-BC with **bottleneck structure** and **dimension reduction** at transition layers is particularly **parameter-efficient**.

#### Feature Reuse & Collective Knowledge

-   **Collective Knowledge**

    每一层均**可获得其较前层的 feature map**。这些**不同层的 feature map 共同构成了 collective knowledge**。

    >   One explanation for this is that **each layer has accessto all the preceding feature-maps in its block** and, therefore,to the **network’s “collective knowledge”**.

-   **Feature Reuse**

    $L$ 层的 DenseNet 有 $\frac{L(L+1)}{2}$ 条 connection，这些 connection 实现了 feature reuse。

    -   **同block的layers**通过 **shortcut connection** 直接利用前层的 feature map
    -   **不同block的layers**通过 **transition layers** 利用被降维的前层的 feature map

    <img src="[paper reading] DenseNet.assets/image-20201013163948443.png" alt="image-20201013163948443" style="zoom:80%;" />

    -   **同block深层可以直接利用浅层特征**

        **Dense Block 中每层**都会把**权重分散**到**同block的许多input**上

        >   All layers **spread their weights** over **many inputs** within **the same block**. This indicates that **features extracted by very early layers** are, indeed, **directly** used by **deep layers** throughout **the same dense block**.

    -   **transition layers 实现了间接的特征复用**

        **transition layers** 也将权重分散到了**之前 Dense Block 的层中**

        >   indicating **information** flow from the **first to the last layers** of the DenseNet through **few indirections**.

    -   **transition layers 输出冗余**

        在**第2和第3个 Dense Block** 中对 **transition layers 的输出**都分配了**最低的权重**，说明 **transition layers 的输出特征冗余**（即便在 transition layers 进行了 Compression 也是如此）

        >   The layers within the s**econd and third dense block** consistently assign **the least weight** to the **outputs of the transition layer** (the top row of the triangles), indicating that the **transition layer outputs many redundant features** (with low weight on average). This is in keeping with the strong results of **DenseNet-BC** where **exactly these outputs are compressed**.

    -   **深层中依然会产生 high-level 的信息**

        最后一层**分类器**将**权重分散到其所有的输出**，但**明显地偏向最终的 feature map**，说明**网络的深层依旧产生 high-level 特征**

        >   Although the **final classification layer**, shown on the very right, also **uses weights across the entire dense block**, there seems to be a **concentration towards final feature-maps**, suggesting that there may be some **more high-level features produced late in the network**.

#### Implicit Deep Supervision

**分类器**可以通过**更短的路径（至多2~3个 transition layers）**，去**直接监督所有的层**，从而实现**隐性的 deep supervision**。

>   One explanation for the improved **accuracy** of dense convolutional networks may be that **individual layers** receive **additional supervision** from the **loss function** through the **shorter connections**.
>
>   **DenseNets** perform a similar **deep supervision** in an **implicit fashion**: a **single classifier** on top of the network provides **direct supervision** to **all layers** through **at most two or three transition layers**.

>   其实 **ResNet 也具有 Deep Supervision 的思想**，即**深层的分类器直接监督浅层**，详见论文：[Residual Networks Behave Like Ensembles of Relatively Shallow Networks](http://papers.nips.cc/paper/6556-residual-networks-behave-like-ensembles-of-relatively-shallow-networks) ，该论文在 [ResNet](./[paper reading] ResNet.md) 中有详细的解读。

#### Diversified Depth

-   DenseNet 是 **statistic depth** 的一个**特例**

    >   there is a **small probability** for any two layers, between the same pooling layers, to be directly connected—if all intermediate layers are randomly dropped.

-   针对 ResNet 的 **ensemble-like behavior** 同样适用于 DenseNet

    因为 **ensemble-like behavior** 的基础 “**collection of path of different length**” 在 DenseNet 依旧成立

## Key Element

### Dense Block

<img src="[paper reading] DenseNet.assets/image-20201012213225227.png" alt="image-20201012213225227" style="zoom:80%;" />

### Transition Layers

==> 实现 **dense block 之间的 Down Sampling**

#### **Components（组成部分）**

**依次**包括以下三个部分：

-   **Batch Normalization**
-   **1x1 Conv**
-   **2x2 average pooling**

#### **Compression（压缩）**

对于一个 dense block 产生的 $m$ 个feature map，Transition Layers 会生成 $\lfloor \theta_m \rfloor$ 个 feature map，其中 **compression factor** $\theta$ 满足 $0<\theta \leqslant 1 $

>   If a dense block contains $m$ feature-maps,  we let the following transition layer generateb $\lfloor \theta_m \rfloor$ output feature-maps, where $0<θ≤1$ is referred to as the compression factor.

在实验中，$\theta$ 选择为 0.5 （同时使用 $\theta<1$ 和 bottleneck 的模型称为 **DenseNet-BC**）

### Growth Rate

-   **实际意义**

    每层对 **global state** 贡献多少的 **new information** (因为每层会**自己产生** $k$ 个 feature map)

    >   The **growthrate** regulates how much **new information** each layer **contributes** to the **global state**.

    -   第 $\ell^{th}$ 层的 **input** feature map 的channel数 ==> **前层的 feature map 在深度上叠加**
        $$
        k_0 + k ×(\ell - 1)
        $$

    -   第 $\ell^{th}$ 层的 **output** feature map 的channel数 ==> **固定值 Growth Rate** $k$
        $$
        k
        $$

        >   至于为什么能做到每层的 output feature map 的 channel **固定**为 **Growth Rate** $k$，参见 [bottleneck layers](#bottleneck layers) 

### Bottleneck Structure

-   **原因和优势**

    -   **原因**

        如果不增加 bottleneck layers，每个 layer 的输出 **feature map 的通道**会**指数增长**。

        >   举一个例子，假设每层都依照 Growth Rate 产生 $k_0$ 个 channel 的 feature map。则：
        >
        >   -   第1层 feature map 的 channel：
        >       $$
        >       c_1 = k_0
        >       $$
        >
        >   -   第2层 feature map 的 channel：
        >       $$
        >       c_2 = k_0 + k_0 = 2k_0
        >       $$
        >
        >   -   第3层 feature map 的 channel：
        >       $$
        >       c_3 = k_0 + 2k_0 + k_0 = 4k_0
        >       $$
        >
        >   -   ……
        >
        >   -   第 $\ell$ 层 feature map 的 channel：
        >       $$
        >       c_{\ell} = 2^{\ell-1}·k_0
        >       $$

        这种指数级别的通道数是不允许存在的，过多的通道数会极大的增加参数量，从而极大降低运行速度。

    -   **优势**

        1.  提高了计算效率
        2.  学习紧凑的 feature representation

-   **原理**

    >   注意：**1x1 Conv** 的位置是在 3x3 Conv**（正常操作）之前**，先对 input feature map 进行降维。
    >
    >   **否则起不到 computational efficiency 的效果**！

    在**每个 3x3 Conv** 前加上 **1x1 Conv** ，对 **channel 维度进行降维压缩** 

    >   a **1×1 convolution** can be introduced as **bottleneck** layer **before each 3×3 convolution** to reduce the number of input feature-maps, and thus  to **improve computational efficiency**.

    $$
    BN-ReLU -Conv(1×1)  ==> BN-ReLU -Conv(3×3)
    $$

-   **参数设置**

    在论文中，作者令每个 **1x1 Conv 产生 4$k$ feature maps**（将对应网络结构称为 **DenseNet-B**）

## Math

本文没有大量的数学公式，故将math分散在了各章节。

## Use Yourself

### [Bottleneck Structure](#Bottleneck Structure)

bottleneck structure 是在 block-level 起作用，在以下方面具有良好的作用：

-   控制channel维度
-   提高参数效率
-   提高计算效率

### [Transition Layers](#Transition Layers)

bottleneck structure 是在 layer-level 起作用，优势与 Bottleneck Structure 类似：

-   控制channel维度
-   提高参数效率
-   提高计算效率

### [Feature Reuse](#Feature Reuse)

具有以下的优点：

-   multi-level：可以同时利用 low-level 和 high-level 的优势
-   multi-scale：low-level 一般具有较高的空间分辨率，而 high-level 一般具有较低的空间分辨率
-   model compactness：避免了特征的重复学习

## Articles

无

## Blogs

-   **指数增长的通道数**：[深入解读DenseNet（附源码）](https://zhuanlan.zhihu.com/p/54767597)





