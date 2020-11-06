# [paper reading] CenterNet (Triplets)

|          topic          |                          motivation                          |                          technique                           |                         key element                          |                             math                             | use yourself |          relativity           |
| :---------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------: | :---------------------------: |
| CenterNet<br />(triple) | [Problem to Solve](#Problem to Solve)<br />[Idea](#Idea)<br />[Intuition](#Intuition) | [CenterNet Architecture](#CenterNet Architecture)<br />[Center Pooling](#Center Pooling)<br />[Cascade Corner Pooling](#Cascade Corner Pooling)<br />[Central Region Exploration](#Central Region Exploration) | [Baseline：CornerNet](#Baseline：CornerNet)<br />[Generating BBox](#Generating BBox)<br />[Training](#Training)<br />[Inferencing](#Inferencing)<br />[Ablation Experiment](#Ablation Experiment)<br />[Error Analysis](#Error Analysis)<br />[Metric AP & AR & FD](#Metric AP & AR & FD)<br />[Small & Medium & Large](#Small & Medium & Large) | [Central Region](#Central Region)<br />[Loss Function](#Loss Function) |      ……      | [Related Work](#Related Work) |

## Motivation

### Problem to Solve

**keypoint-based**方法的弊端（这里主要指的是**CornerNet**）：

由于**缺少**对于**cropped region**的**additional look**，无法获得**bounding box region**的**visual pattern**，会导致产生大量的**incorrect bounding box**

<img src="[paper reading] CenterNet (Triplets).assets/image-20201104202625913(复件).png" alt="image-20201104202625913(复件)" style="zoom: 100%;" />

<center>
    ① CornerNet 会产生很多的错误的bounding box
</center>

### Idea

用一个**keypoint triplet**（**top-left corner** & **bottom-right corner** & **center**）表示一个**object**。

即在由**top-left corner & bottom-right corner**去**encode边界信息**的同时，通过引入**center**，使得模型可以**explore**每个**predicted bounding box的visual patter**（获得**object的internal信息**）

在**具体的做法**上，是将 **visual patterns within object** 转化成 **keypoint detection**

<img src="[paper reading] CenterNet (Triplets).assets/image-20201104202625913.png" alt="image-20201104202625913" style="zoom: 100%;" />




<center>
    ② 检查Central Region可以找出正确的prediction
</center>

### Intuition

该思路部分沿袭**RoI Pooling**的思想，通过**efficient discrimination（Central Region）**，使得**one-stage**方法**一定程度上**具有了**two-stage**方法的**resample能力**

具体来说：如果**predicted bounding box**和**ground-truth box**有**高IoU**，则**Center-Region**中的**Center KeyPoint**也会被预测为**相同的类别**

## Technique

### CenterNet Architecture

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105095400153.png" alt="image-20201105095400153" style="zoom: 67%;" />

#### Components

-   **[Center Pooling](#Center Pooling)**
-   **[Cascade Corner Pooling](#Cascade Corner Pooling)**
-   **[Central Region Exploration](#Central Region Exploration)**

#### Improvement

-   **AP Improvement**

    small、medium、large object的**AP均有提升**，**绝大部分的提升**来自**small object**

    原因：**Center Information**。**incorrect bounding box越小**，能在其**Central Region检测到center keypoint的可能性越小**

    <img src="[paper reading] CenterNet (Triplets).assets/image-20201105135615588.png" alt="image-20201105135615588" style="zoom: 80%;" />

    <center>
        small object
    </center>
<img src="[paper reading] CenterNet (Triplets).assets/image-20201105135713341.png" alt="image-20201105135713341" style="zoom:80%;" />
    
<center>
        medium & large object
    
-   **AR Improvement**

    原因：**滤除**了**incorrect bounding box**，相当于**提升**了**accurate location but lower scores**的**bounding box**的**confidence**

### Center Pooling

>   Cascade Corner Pooling 和 Center Pooling 都可以通过结合不同方向的 Corner Pooling 实现

#### Why

**geometric center**并**不一定**带有**recognizable visual pattern**

#### Purpose

**better detection of center keypoint！！！**

具体来说，是为**Central Region**提供**recognizable visual pattern**，以感知proposal中心位置的信息，从而**检测bounding box的正确性**

#### Steps

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105102937735.png" alt="image-20201105102937735" style="zoom:80%;" />

对于Center Pooling的输入feature map，在**水平和垂直方向**取**max summed response**

1.  backbone输出feature map
2.  在水平和垂直方向分别找到最大值
3.  将其加到一起

![image-20201105123057169]([paper reading] CenterNet (Triplets).assets/image-20201105123057169.png)

### Cascade Corner Pooling

>   Cascade Corner Pooling 和 Center Pooling 都可以通过结合不同方向的 Corner Pooling 实现

#### Why

**corner**在**object之外**，缺少**local appearance feature**

#### Purpose

**better detection of corners！！！**

具体来说，是**丰富**top-left **corner**和bottom-right **corner**收集的**信息**，以**同时感知boundary和internal信息**

#### Steps

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105122244431.png" alt="image-20201105122244431" style="zoom:80%;" />

在输入feature map的boundary和internal方向，去max summed response（双方向的pooling更稳定更鲁棒，能提高准确率和召回率）

1.  在**boundary**方向上找**boundary max**
2.  在**boundary max**的位置，向**internal**方向上找**internal max**
3.  把**2个max加起来**（加到**corner的位置**）

![image-20201105123054029]([paper reading] CenterNet (Triplets).assets/image-20201105123054029.png)



### Central Region Exploration

#### Scale-Aware Central Region

-   **原因**：

    $\text{recall} \ vs. \text{precision}$

-   **Central Region的选择**：

    对不同**size的bounding box**生成**不同大小Central Region** 

    -   **small bounding box** ==> **large central region**

        原因：**small center region**会导致**small bounding box**的**low recall**

    -   **large bounding box** ==> **small central region**

        原因：**small center region**会导致**small bounding box**的**low recall**

    在实验中，使用2中Central Region：

    <img src="[paper reading] CenterNet (Triplets).assets/image-20201105101810575.png" alt="image-20201105101810575" style="zoom:67%;" />

    具体使用哪种，由**bounding box的scale**决定：

    -   $< 150$：n = 3 (left)
    -   $> 150$：n = 5 (right)

#### Exploration

-   **center keypoint**落到**Central Region中**
-   **center keypoint**和**bounding box**的**类别相同**

## Key Element

### Baseline：CornerNet

#### Three outputs

-   **heatmap**：

    -   top-left corner
    -   bottom-right corner

    每个heatmap都包括2个部分：

    1.  不同**category**的**keypoint的位置**
    2.  每个**keypoint**的**confidence score**

-   **embedding**：

    对**corner**进行**分组**

-   **offset**：

    把**corner**从**heatmap**去**remap**到**input image**

#### Generate BBox

1.  对**top-left corner和bottom-right corner**分别**取top-100**
2.  根据**embedding distance**对**corner**进行**分组**（embedding distance < $Threshold$）
3.  计算**bounding box**的**confidence score**（2个corner score的**平均**）

#### Drawbacks

**CornerNet**的**False Discovery Rate（FD）**很高（即：有**大量的incorrect bounding box**）

>   AP & FD的含义，见 [Metric AP & AR & FD](#Metric AP & AR & FD)

### Generating BBox

1.  选取 **top-k** 个**center keypoints**

2.  **center keypoint**去**remap**到**input image**（使用**offset**）

3.  在**bounding box**中定义**Central Region**

4.  保留**符合要求**的**bounding box**

    -   **center keypoint**落到**Central Region中**
    -   **center keypoint**和**bounding box**的**类别相同**

5.  计算**bounding box**的**score**

    为**top-left corner**、**bottom-right corner**、**center**的**average score**

### Training

#### Input & Output Size

-   input size：511×511
-   output size：128×128

#### Data Augmentation

同 CornerNet

### Inferencing

#### Single-Scale Testing

以**原分辨率**，将**original**和**flipped**输入网络

#### Multi-Scale Testing

以分辨率 $[0.6, 1.0, 1.2,1.5,1.8]$，将**original**和**flipped**输入网络

#### Steps 

1.  根据**70**对**Triplet**确定**70**对**bounding box**

    详见 [Generating BBox](#Generating BBox)

2.  将**flipped image**再次flip，合并到**原image**上

3.  **Post-Processing**：**Soft-NMS**

4.  取**top-100**的**bounding box**

### Ablation Experiment

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105140219262.png" alt="image-20201105140219262" style="zoom:80%;" />

#### Incorrect Bounding Box Reduction

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105140402990.png" alt="image-20201105140402990" style="zoom: 80%;" />

#### Inference Speed

visual patterns exploration的cost很小

CenterNet某版本可以在精度和速度上同时超过CornerNet某版本

#### Center Pooling Ablation

-   **结论**：

    **Center Pooling**可以大幅度提高**large object**的**AP**

-   **原因**：

    -   **Center Pooling**可以提取**更丰富的internal visual patterns**
    -   **larger object**包含**更多的internal visual pattern**

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105141036192.png" alt="image-20201105141036192" style="zoom:67%;" />

#### Cascade Corner Pooling Ablation

-   **结论**：

    -   由于**large object**有**丰富的internal visual patterns**，**Cascade Corner Pooling**可以看到**更多的object**

    -   **过于丰富的internal visual patterns**会**影响其对boundary的敏感**，导致**inaccurate bounding box**
        -   可以通过Center Pooling抑制错误的Bounding box

#### Central Region Exploration Ablation

-   **结论**：

    提升了**整体的AP**，其中**小目标AP**提升最大

-   **原因**：

    **小目标**的**center keypoint**更**容易被located**

### Error Analysis

1.  Exploration of visual patterns依赖于center keypoint实现 ==> Center keypoint的丢失会导致CenterNet丢失bounding box的visual pattern

2.  Center keypoint还有很大的提升空间

    

### Metric AP & AR & FD

#### AP：Average Precision Rate

是在**所有category**上，以**10个Threshold**（e.g. $0.5:0.05:0.95$）上计算

可以反映网络可以预测多少**高质量的bounding box**（一般**IoU**$\ge0.5$）

>   是MS-COCO数据集最重要的metric

#### AR：Maximum Recall Rate

在**每张图片**上取**固定数量的detection**，在**所有类别**和**10个IoU Threshold**上取**平均**

#### FD：False Discovery Rate

反映**incorrect bounding box的比例**
$$
\text{FD} = 1-\text{AP}
$$


### Small & Medium & Large

-   **small object**：$\text{area}<32^2$
-   **medium object**：$32^2<\text{area}<96^2$

-   **large object**：$\text{area}>96^2$

## Math

### Central Region

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105102049760.png" alt="image-20201105102049760" style="zoom: 80%;" />

### Loss Function

主要分为：

-   **Detection Loss**

    -   **Corner** Detection Loss $\text{L}_{\text{det}}^{\text{co}}$
    -   **Center** Detection Loss $\text{L}_{\text{det}}^{\text{ce}}$

-   **Pull & Push Loss**

    仅对**Corner**进行

    -   **Pull** Loss $\text{L}_{\text{pull}}^{\text{co}}$
    -   **Push** Loss $\text{L}_{\text{push}}^{\text{co}}$

-   **Offset Loss**

    -   **Corner** offset Loss $\text{L}_{\text{off}}^{\text{co}}$
    -   **Center** offset Loss $\text{L}_{\text{off}}^{\text{ce}}$

<img src="[paper reading] CenterNet (Triplets).assets/image-20201105130407319.png" alt="image-20201105130407319" style="zoom:50%;" />

-   $\alpha=\beta = 0.1$
-   $\gamma=1$

## Use Yourself

……

## Related Work

### Anchor-Based Method

#### Introduction

Anchor-Based Method有2个关键点：

-   放置**预定义size和ratio**的**anchor**
-   根据**ground-truth**对**positive bounding box**进行**regression**

#### drawbacks

-   需要**大量的anchor**（以保持和**ground-truth box**的**足够高的IoU**）
-   **anchor**的**size和ratio**需要**手工设计**（带来大量的超参数需要调试）

-   **anchor和ground-truth没有对齐**

### KeyPoint-Based Method

>   这里主要指的是**CornerNet**

#### Introduction

即：使用**一对corner**表示**一个object**

#### drawbacks

-   referring到**global信息**的**能力相对较弱**

    换句话说，即：**对object的boundary信息敏感**

-   **无法确知哪对KeyPoints应该表示object**

详见 [Problem to Solve](#Problem to Solve)

### Two-Stage Method

#### Steps

-   **Extract RoIs** ==> **stage-1** 
-   **classify & regress RoIs** ==> **stage-2**

#### Models

**RCNN**：

-   selective search获得RoI
-   CNN作为classifier

**SPP-Net & Faster-RCNN**：

-   在**feature map**中提取RoIs

**Faster-RCNN**：

-   使用**RPN**对**anchor**进行**regression**，实现了**end-to-end**训练

**Mask-RCNN**：

-   Faster-RCNN + mask-prediction branch
-   同时实现detection和segmentation

**R-FCN**：

-   将**FC层**替换成了**position-sensitive score maps**

**Cascade RCNN**：

通过训练一系列**IoU阈值逐渐升高的detector**，解决了2个问题：

-   **训练**时的**overfitting**
-   **推断**时的**quality mismatch**

### One-stage Method

>   one-stage方法的通病：**缺少**对**cropped region**的**additional look**

#### Steps

**直接**对**anchor box**进行**classify**和**regress**

#### Models

**YOLOv1**：

-   **image** ==> **S×S grid**
-   **不使用anchor**，直接去学习bounding box的size

**YOLOv2**：

-   重新使用了**较多的anchor**
-   使用了**新的bounding box regression**方法

**SSD**：

-   使用**不同convolutional stage**的**feature map**进行**classify**和**regress**

**DSSD**：

-   **SSD** + **deconvolution** ==> 结合**low-level和high-level的feature**

**R-SSD**：

-   对不同feature layer，进行pooling和deconvolution ==> 结合**low-level和high-level的feature**

**RON**：

-   reverse connection
-   objectness prior 

**RefineDet**：

-   对location和size进行2次refine，继承了one-stage和two-stage的优点

**CornerNet**：

-   keypoint-based method
-   用一对corner表示一个object

## Problems

-   [ ] Cascade Corner Pooling的internal方向，怎么找boundary方向的最大值呢？
-   [x] AP和AR的含义到底是什么？
-   [ ] 为什么CornerNet去referring目标的global information的能力很弱？

