# [paper reading] CenterNet (Object as Points)

|               topic               |                        motivation                        |                     technique                     |                         key element                          |                             math                             |                 use yourself                  |
| :-------------------------------: | :------------------------------------------------------: | :-----------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-------------------------------------------: |
| CenterNet<br />(Object as Points) | [Problem to Solve](#Problem to Solve)<br />[Idea](#Idea) | [CenterNet Architecture](#CenterNet Architecture) | [Center Point & Anchor](#Center Point & Anchor)<br />[Getting Ground-Truth](#Getting Ground-Truth)<br />[Model Output](#Model Output)<br />[Data Augmentation](#Data Augmentation)<br />[Inference](#Inference)<br />[TTA](#TTA)<br />[Compared with SOTA](#Compared with SOTA)<br />[Additional Experiments](#Additional Experiments) | [Loss Function](#Loss Function)<br />[KeyPoint Loss $\text{L}_k$](#KeyPoint Loss $\text{L}_k$)<br />[Offset Loss $\text{L}_{off}$](#Offset Loss $\text{L}_{off}$)<br />[Size Loss $\text{L}_{size}$](#Size Loss $\text{L}_{size}$) | [Getting Ground-Truth](#Getting Ground-Truth) |

## Motivation

### Problem to Solve

**anchor-based method**有以下的缺点：

- **wasteful & inefficient**：

  需要对**object**进行**饱和式检测**（饱和式地列出object的潜在位置）

- **need post-processing**（e.g. **NMS**）

### Idea

- 从**本质**上讲：

  将**Object Detection**转化为**Standard Keypoint Estimation**

- 从**思路**上讲：

  使用**bounding box的center point**表示一个**object**

- 从**具体流程**上讲：

  使用**keypoint estimation**寻找**center point**，并**根据center point回归其他的属性**（因为**其他的属性**都和**center point**存在**确定的数学关系**）

## Technique

### CenterNet Architecture

#### Components

- **Backbone**
  - **Stacked Hourglass Network**
  
      >   详见 [CornerNet](./[paper reading] RetinaNet.md)
  
  - **Upconvolutional Residual Netwotk**
  
  - **Deep Layer Aggregation**（DLA）
  
- **Task-Specific Modality**

    - **1 个 3×3 Convolution**
    - **ReLU**
    - **1 个 1×1 Convolution**

#### Advantage

- **simpler & faster & accurate**

- **end-to-end differential**

  **所有的输出**都是**直接**从**keypoint estimation network**输出，**不需要NMS**（以及其他post-processing）

  Peak Keypoint Extraction由 $3×3 \  \text{Max Pooling}$ 实现，足够用来替换NMS

- **estimate additional object properties in one single forward pass**

  在**单次前向传播**中，可以估计出**多种object properties**

## Key Element

### Center Point & Anchor

#### Connection

**center point**可以看作是**shape-agnostic anchor**（**形状不可知**的anchor）

即：相当于是在**image的每个location**对应**1个shape-agnostic ancho**r（e.g. **center keypoint**）

#### Difference

- **center point**仅仅与**location**有关（**与box overlap无关**）

  即：**不需要手动设置foreground和background的threshold**

- **每个object**仅对应**1个center point**

  **直接**在**keypoint heatmap**上提取**local peak**，**不存在重复检测**的问题

- **CenterNet**有**更大的输出分辨率**

  **降采样步长为4**（常见为16）

### Getting Ground-Truth

>   详见 [Symbol Definition](#Symbol Definition)

#### Keypoint Ground-Truth

##### Ground-Truth：Input Image ==> Output Feature Map

-   $p \in \mathcal{R}^2$ ：**ground-truth keypoint**
-   $\widetilde{p} = \lfloor\frac pR \rfloor$ ：**low-resolution equivalent**

将**image**的**ground-truth keypoint** $p$ 映射为**output feature map**上**ground-truth keypoint** $\widetilde p$
$$
\widetilde{p} = \lfloor\frac pR \rfloor
$$

##### Gaussian Penalty Reduction

$$
Y_{x y c}=\exp \left(-\frac{\left(x-\tilde{p}_{x}\right)^{2}+\left(y-\tilde{p}_{y}\right)^{2}}{2 \sigma_{p}^{2}}\right)
$$

-   $\sigma_{p}$ ：**object size-adaptive**的**标准差**

>   如果**同一个类别**的**2个Gaussian**发生**重叠**，则取**element-wise maximum**

**keypoint heatmap**：
$$
\hat{Y}\in[0,1]^{\frac{W}{R}×\frac HR×C}
$$

-   $\hat Y _{x,y,c} =1$ ==> **keypoint**
-   $\hat Y _{x,y,c} =0$ ==> **background**

>   **注意**：这里的**center**是**bounding box**的**几何中心**，即**center**到**左右边和上下边**的**距离是相等的**
>
>   这样**需要回归的量**就从**4个distance**变成了**2个**

#### Size Ground-Truth

bounding box 用4个点表示（第 $k$ 个object，类别为 $c_k$）：
$$
(x_1^{(k)}, y_1^{(k)}, x_2^{(k)}, y_2^{(k)})
$$
**Center** 表示为：
$$
p_k = \big(  \frac{x_1^{(k)} +  x_2^{(k)}  }{2}   , \frac{y_1^{(k)} +  y_2^{(k)}  }{2}   \big)
$$
**Size Ground-Truth** 表示为：
$$
s_k = \big(x_2^{(k)} - x_1^{(k)}, y_2^{(k)}-y_1^{(k)} \big) = (W,H)
$$

>   注意：**不对scale进行归一化**，而是**直接使用raw pixel coordinate**

### Model Output

>   **Input & Output Resolution**：
>
>   -   512×512
>   -   128×128

**所有的输出共享一个共用的全卷积网络**

-   **keypoint** $\hat Y$ ==> $C$
-   **offset** $\hat O$ ==> 2
-   **size** $\hat S$ ==> 2

即：每个**location**都有**C+4个output**

对于each modality，在将feature经过：

-   **1 个 3×3 Convolution**
-   **ReLU**
-   **1 个 1×1 Convolution**

### Data Augmentation

-   **random flip**

-   **random scaling**（0.6~1.3）

-   **cropping**
-   **color jittering**

### Inference

**CenterNet**的**Inference**是**single network forward pass**

1. 将**image**输入**backbone（e.g. FCN）**，得到**3个输出**：

   -   **keypoint** $\hat Y$ ==> $C$

       **heatmap的peak**对应**object的center**（取**top-100**）

       **peak的判定**：值 $\ge$ 其8个邻居

   -   **offset** $\hat O$ ==> 2

   -   **size** $\hat S$ ==> 2

2. 根据**keypoint** $\hat Y$、 **offset** $\hat O$、**size** $\hat S$ 计算bounding box

    <img src="[paper reading] CenterNet (Object as Points).assets/image-20201105202743552.png" alt="image-20201105202743552" style="zoom:80%;" />

    -   $(\delta \hat x_i, \delta \hat x_i) = \hat O_{\hat x_i,  \hat y_i}$ ：**offset prediction**
    -   $( \hat w_i, \hat h_i) = \hat S _{\hat x_i,  \hat y_i}$ ：**size prediction**

3. 计算**keypoint**的**confidence**：keypoint对应位置的value
    $$
    \hat Y_{x_i,y_ic}
    $$

### TTA 

有3种TTA方式：

1.  **no augmentation**

2.  **flip augmentation**

    **flip**：在**decoding**之前，进行**output average**

3.  **flip & multi-scale（0.5，0.75，1，1.25，1.5）**

    **multi-scale**：使用**NMS**对结果进行聚合

### Compared with SOTA

<img src="[paper reading] CenterNet (Object as Points).assets/image-20201105210709222.png" alt="image-20201105210709222" style="zoom: 67%;" />

### Additional Experiments

#### Center Point Collision

多个object经过下采样，其center keypoint有可能重叠

CenterNet可以减少Center Keypoint的冲突

#### NMS

CenterNet使用了NMS提升很小，说明CenterNet不需要NMS

#### Training & Testing Resolution

1.  低分辨率速度最快但是精度最差
2.  高分辨率精度提高，但速度降低
3.  原尺寸速度略高于高分辨率，但速度略慢

<img src="[paper reading] CenterNet (Object as Points).assets/image-20201105211137786.png" alt="image-20201105211137786" style="zoom: 67%;" />

#### Regression Loss

smooth L1 Loss的效果略差于L1 Loss

<img src="[paper reading] CenterNet (Object as Points).assets/image-20201105211241160.png" alt="image-20201105211241160" style="zoom: 67%;" />

#### Bounding Box Size Weight

$\lambda_{size}$ 为0.1时最佳，增大时AP快速衰减，减小时鲁棒

<img src="[paper reading] CenterNet (Object as Points).assets/image-20201105211434313.png" alt="image-20201105211434313" style="zoom:67%;" />

#### Training Schedule

训练时间更长，效果更好

<img src="[paper reading] CenterNet (Object as Points).assets/image-20201105211459264.png" alt="image-20201105211459264" style="zoom:67%;" />



## Math

>   ### Symbol Definition
>
>   -   $I \in R^{W×H×3}$ ：**image**
>   -   $R$ ：**output stride**，实验中为4
>   -   $C$ ：**keypoint**的**类别数**

### Loss Function

$$
\text{L}_{det} = \text{L}_k + \lambda_{size} \text{L}_{size} + \lambda_{off} \text{L}_{off}
$$

- $\lambda_{size} = 0.1$
- $\lambda_{off} = 1$

### KeyPoint Loss $\text{L}_k$

**penalty-reduced** pixel-wise **logistic regression** with **focal loss**

<img src="[paper reading] CenterNet (Object as Points).assets/3A35DCC9DDB35ACDE954365EE4354074.png" alt="3A35DCC9DDB35ACDE954365EE4354074" style="zoom: 33%;" />

-   $\hat{Y}_{xyc}$ ：**predicted keypoint confidence**
-   $\alpha =2,\beta=4$

### Offset Loss $\text{L}_{off}$

目的：恢复由**下采样**带来的**离散化错误**（discretization error）

<img src="[paper reading] CenterNet (Object as Points).assets/image-20201105192719073.png" alt="image-20201105192719073" style="zoom:80%;" />

-   $\hat O \in \mathcal R^{\frac{W}{R}×\frac HR×2}$ ：**predicted local offset**

**注意**：

-   仅仅对**keypoint location**（**positive**）计算
-   **所有的类别**共享**相同的offset prediction**

### Size Loss $\text{L}_{size}$

<img src="[paper reading] CenterNet (Object as Points).assets/image-20201105194716898.png" alt="image-20201105194716898" style="zoom: 80%;" />

-   $\hat{S}_{p_{k}} \in \mathcal R^{\frac{W}{R}×\frac HR×2}$ 
-   $s_k = \big(x_2^{(k)} - x_1^{(k)}, y_2^{(k)}-y_1^{(k)} \big)$

## Use Yourself

### Getting Ground-Truth

**以何种方式表示ground-truth，决定了所采用的方法**

从本质来说，使用**corner coordinate表示ground-truth**，都是在**某个“基准”上**，**直接**对**ground-truth**进行**regression**

-   **anchor-based**是sliding window的思想，先**通过anchor获得bounding box**（避免了对于bbox的detection），再对**4个corner的坐标进行regression**
-   **keypoint-based** 的**CornerNet**（包括使用center来矫正的CenterNet (Triplet) ）需要**先检测corner**，再对其**坐标进行regression**

若使用**center+distance**表示**ground-truth**（e.g. CenterNet (object as points)），本质上则是**检测center**，并在**center上对2个distance进行regression**，之后进行**decode得到bounding box**

这给我一个启发：对于**复杂的ground-truth**，应用一些方法将其转化为**具有一些实际物理意义的低维数据**，最后通过**数学或其他关系**反向decode**还原原始的ground-truth**，是不是会更好呢？

## Related work

### Anchor-Based Method

#### Essence

将**detection**降级为**classification**

#### Two-Stage Method

1. 在**image**上放置**anchor**（同 [One-Stage Method](#One-Stage Method)）

   即：在**low-resolution**上**dense & grid采样anchor**，分类为**foreground/background** ==> **proposal**

   > 具体的label：
   >
   > - **foreground**：
   >
   >   与**任意ground-truth box**有 **> 0.7 的IoU**
   >
   > - **background**：
   >
   >   与**任意ground-truth box**有 **< 0.3 的IoU**
   >
   > - **ignored**：
   >
   >   与**任意ground-truth box** 的**IoU** $\in [0.3, 0.7]$

2. 对**anchor**进行**feature resample**

比如：

- **RCNN**：在**image**上取**crop**
- **Fast-RCNN**：在**feature map**上取**crop**

#### One-Stage Method

1. 在**image**上放置**anchor**
2. **直接**对**anchor**位置进行**分类**

one-stage method的一些**改进**：

- **anchor shape prior**
- **different feature resolution**（e.g. **Feature Pyramid Network**）
- **loss re-weighting**（e.g. **Focal Loss**）

#### Post-Processing（NMS）

- **Purpose**：根据**IoU**，抑制**相同instance的detections**

- **Drawback**：难以**differentiate**和**train**，导致绝大部分的detector**无法做到end-to-end trainable**

### KeyPoint-Based Method

#### Essence

将**detection**转化为**keypoint estimation**

其**Backbone**均为**KeyPoint Estimation Network**

#### CornerNet

检测**2个corner**作为**keypoints**，表示**1个bounding box** 

#### ExtremeNet

检测 **top-most**, **left-most**, **bottom-most**, **right-most** ,**center** 作为keypints

#### Drawback

对**1个object检测多个keypoint**，其需要**额外的grouping stage**（导致算法速度的降低）