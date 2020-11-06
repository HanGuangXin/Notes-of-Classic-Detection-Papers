## To Myself

本文列出的论文是Detection方向入门必读的论文。

写作这些Notes的初衷，是希望能找到一个相对高效和稳定的方法，来对一个方向建立起基本的知识体系。也希望这个方法能推广应用到我对于其他的方向的学习上。

从零基础入门Detection，到现在完成这些工作，大概花了3个月的时间。

一方面，我觉得达到这样的程度，耗费的时间还是有点长的，不过这算是人生的第一次，也情有可原。吸取经验和教训就好了！

另一方面，经过对这些论文的精读，我能明显感觉到自己知识的深度和广度有了很大的提升，成长还是很明显的（当然还是很菜）。

总结一下，我认为自己在这个过程中有这些收货：

1.  从结果上看，我基本上算是建立了对Detection方向的基本理解和感知，在Computer Vision上，算是扣好了第一颗扣子。

2.  从方法上看，今后我要面对一个全新的方向，我也有了方法论，知道该如何下手。

3.  对于问题的细致思考和辩证思考、钻研，对于思维的提升有很大的帮助。要多问自己为什么。

    现在如果让我去讲这些论文，我相信自己能够讲得明白！

当然，也存在一些不足：

1.  在之后读论文的时候，应该对照着代码一起来。

    这样，对于论文中一些难以理解的点，可以直接去看代码实现，很多疑难问题就可以迎刃而解。

    而且，这对于代码能力的提高也有好处。

2.  从具体的流程上，也有必须要优化的地方。

    -   首先，入门一开始的时候，没有必要去一篇一篇地去做Overview。

        要了解整个方向，直接上综述就好了。

        所谓的第一轮可以大幅度地简化。

    -   其次，读论文的时候一定尽可能带上代码。

        所谓的后两轮也可以合并到一起。

    这样，原本的 “Overview + Paper + Code” 就可以变为 “1~2 Survey + Paper & Code”

## How To Use the Notes

我推荐您：

1.  将博客作为参考，以博客的行文思路去理解论文（博客的行文思路来自于Andrew Ng关于 “Reading Research Paper” 的课程）。
2.  对于一些不明白的地方，去找博客的相应位置。

我个人认为，**需要到能给其他人讲论文的程度，这里的Notes可以满足您70~90%的需求**

## Idea of Writing

- **Topic**

  论文的题目

- **Motivation**

  包含论文要解决的问题、论文的Idea等

- **Technique**

  论文的模型、顶层的技术等

- **Key Element**

  一些底层的技术、关键概念、细节等

- **Math**

  所有的数学内容，包括数学表示、损失函数等

- **Use Yourself**

  对于论文一些要点的个人理解、一些在今后可能有用的idea等

- **Relativity**

  相关的论文、博客等

## Sections of Each Notes

### Overview

|                            Topic                             |                          Motivation                          |                          Technique                           |                         Key Element                          |                     Use Yourself / Math                      |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|         [GoogLeNet](./[paper reading] GoogLeNet.md)          |             sparse structure<br />& dense matrix             |                          Inception                           |                 Inception path<br />1x1 conv                 | Architecture<br />data augmentation<br />model ensemble<br />Test Time Augmentation<br />Inception |
|            [ResNet](./[paper reading] ResNet.md)             |           problems to slove<br />identity mapping            | residual learning<br />residual block<br />ResNet Architecture |         shortcut connection<br />feature concatenate         | ResNet Architecture<br />residual block<br />shortcut connection<br />ResNet V2 |
|          [DenseNet](./[paper reading] DenseNet.md)           | [Problem to Solve](#Problem to Solve)<br />[Modifications](#Modifications) | [DenseNet Architecture](DenseNet Architecture)<br />[Advantages](#Advantages) | [Dense Block](#Dense Block)<br />[Transition Layers](#Transition Layers)<br />[Growth Rate](#Growth Rate)<br />[Bottleneck Structure](#Bottleneck Structure) | [Bottleneck Structure](#Bottleneck Structure)<br />[Feature Reuse](#Feature Reuse)<br />[Transition Layers](#Transition Layers) |
|       [Faster RCNN](./[paper reading] Faster RCNN.md)        | [Problem to Solve](#Problem to Solve)<br />[Proposal with CNN](#Proposal with CNN) | [Architecture](#Architecture)<br />[Region Proposal Network](#Region Proposal Network)<br />[RoI Pooling](#RoI Pooling)<br />[Anchor](#Anchor) | [anchor/proposals/bbox](#anchor/proposals/bbox)<br />[Pyramids](#Pyramids)<br />[Positive & Negative Label](#Positive & Negative Label)<br />[Sampling Strategy](#Sampling Strategy)<br />[4-Step Alternating Training](#4-Step Alternating Training) | [Architecture](#Architecture)<br />[Region Proposal Network](#Region Proposal Network)<br />[RoI Pooling](#RoI Pooling)<br />[Anchor](#Anchor) |
|           [YOLO v1](./[paper reading] YOLO v1.md)            | [Problem to Solve](#Problem to Solve)<br />[Detection as Regression](#Detection as Regression) |        [YOLO v1 Architecture](#YOLO v1 Architecture)         | [Grid Partition](#Grid Partition)<br />[Model Output](#Model Output)<br />[“Responsible”](#"Responsible")<br />[NMS](#NMS)<br />[Getting Detections](#Getting Detections)<br />[Error Type Analysis](#Error Type Analysis)<br />[Data Augmentation](#Data Augmentation) | [Classification & Detection](#Classification & Detection)<br />[Explicit & Implicit Grid](#Explicit & Implicit Grid)<br />[Handle Small Object](#Handle Small Object)<br />[Multi-Task Output](#Multi-Task Output)<br />[Weighted Loss](#Weighted Loss)<br />[NMS](#NMS)<br />[Data Augmentation](#Data Augmentation) |
|               [SSD](./[paper reading] SSD.md)                | [Problem to Solve](#Problem to Solve)<br />[Contributions](#Contributions) | [SSD Architecture](#SSD Architecture)<br />[Pros & Cons](#Pros & Cons) | [Higher Speed](#Higher Speed)<br />[What Is Resample](#What Is Resample)<br />[Low & High Level Feature](#Low & High Level Feature)<br />[Small Object Difficulties](#Small Object Difficulties)<br />[Data Flow](#Data Flow)<br />[Anchor & GtBox Matching](#Anchor & GtBox Matching)<br />[Foreground & Background](#Foreground & Background)<br />[Hard Negative Mining](#Hard Negative Mining)<br />[NMS](#NMS)<br />[Data Augmentation](#Data Augmentation)<br />[Testing (Inferencing) Step](#Testing (Inferencing) Step)<br />[Performance Analysis](#Performance Analysis)<br />[Model Analysis](#Model Analysis) | [Convolution For Speed](#Convolution For Speed)<br />[Feature Pyramids Fashion](#Feature Pyramids Fashion)<br />[Positive & Negative Imbalance](#Positive & Negative Imbalance)<br />[Zoom In & Zoom Out](#Zoom In & Zoom Out) |
|         [RetinaNet](./[paper reading] RetinaNet.md)          | [Problem to Solve](#Problem to Solve)<br />[Negative ==> Easy](#Negative ==> Easy) |    [Focal Loss](#Focal Loss)<br />[RetinaNet](#RetinaNet)    | [Class Imbalance](#Class Imbalance)<br />[Feature Pyramid Network](#Feature Pyramid Network)<br />[ResNet-FPN Backbone](#ResNet-FPN Backbone)<br />[Classify & Regress FCN](#Classify & Regress FCN)<br />[Post Processing](#Post Processing)<br />[Anchor Design](#Anchor Design)<br />[Anchor & GT Matching](#Anchor & GT Matching)<br />[prior $\pi$ Initialization](#prior $\pi$ Initialization)<br />[Ablation Experiments](#Ablation Experiments) | [Data Analysis](#Data Analysis)<br />[Feature Pyramid Network](#Feature Pyramid Network)<br />[More Scale $\not= $ Better](#More Scale $\not= $ Better) |
|         [CornerNet](./[paper reading] CornerNet.md)          | [Problem to Solve](#Problem to Solve)<br />[KeyPoints (Anchor-Free)](#KeyPoints (Anchor-Free)) | [CornerNet](#CornerNet)<br />[Stacked Hourglass Network](#Stacked Hourglass Network)<br />[Prediction Module](#Prediction Module)<br />[Corner Pooling](#Corner Pooling) | [Why CornerNet Better?](#Why CornerNet Better?)<br />[Why Corner Pooling Works](#Why Corner Pooling Works)<br />[Grouping Corners](#Grouping Corners)<br />[Getting Bounding Box](#Getting Bounding Box)<br />[Data Augmentation](#Data Augmentation)<br />[Ablation Experiments](#Ablation Experiments) | [Loss Function](#Loss Function)<br />[Corner Pooling Math](#Corner Pooling Math) |
| [CenterNet<br />(triple)](./[paper reading] CenterNet (Triplets).md) | [Problem to Solve](#Problem to Solve)<br />[Idea](#Idea)<br />[Intuition](#Intuition) | [CenterNet Architecture](#CenterNet Architecture)<br />[Center Pooling](#Center Pooling)<br />[Cascade Corner Pooling](#Cascade Corner Pooling)<br />[Central Region Exploration](#Central Region Exploration) | [Baseline：CornerNet](#Baseline：CornerNet)<br />[Generating BBox](#Generating BBox)<br />[Training](#Training)<br />[Inferencing](#Inferencing)<br />[Ablation Experiment](#Ablation Experiment)<br />[Error Analysis](#Error Analysis)<br />[Metric AP & AR & FD](#Metric AP & AR & FD)<br />[Small & Medium & Large](#Small & Medium & Large) | [Central Region](#Central Region)<br />[Loss Function](#Loss Function) |
| [CenterNet<br />(Object as Points)](./[paper reading] CenterNet (Object as Points).md) |   [Problem to Solve](#Problem to Solve)<br />[Idea](#Idea)   |      [CenterNet Architecture](#CenterNet Architecture)       | [Center Point & Anchor](#Center Point & Anchor)<br />[Getting Ground-Truth](#Getting Ground-Truth)<br />[Model Output](#Model Output)<br />[Data Augmentation](#Data Augmentation)<br />[Inference](#Inference)<br />[TTA](#TTA)<br />[Compared with SOTA](#Compared with SOTA)<br />[Additional Experiments](#Additional Experiments) | [Loss Function](#Loss Function)<br />[KeyPoint Loss $\text{L}_k$](#KeyPoint Loss $\text{L}_k$)<br />[Offset Loss $\text{L}_{off}$](#Offset Loss $\text{L}_{off}$)<br />[Size Loss $\text{L}_{size}$](#Size Loss $\text{L}_{size}$) |
|              [FCOS](./[paper reading] FCOS.md)               |       [Idea](#Idea)<br />[Contribution](#Contribution)       | [FCOS Architecture](#FCOS Architecture)<br />[Center-ness](#Center-ness)<br />[Multi-Level FPN Prediction](#Multi-Level FPN Prediction) | [Prediction Head](#Prediction Head)<br />[Training Sample & Label](#Training Sample & Label)<br />[Model Output](#Model Output)<br />[Feature Pyramid](#Feature Pyramid)<br />[Inference](#Inference)<br />[Ablation Study](#Ablation Study)<br />[FCN & Detection](#FCN & Detection)<br />[FCOS $vs.$ YOLO v1](#FCOS $vs.$ YOLO v1) | [Symbol Definition](#Symbol Definition)<br />[Loss Function](#Loss Function)<br />[Center-ness](#Center-ness)<br />[Remap of Feature & Image](#Remap of Feature & Image) |

### Details

#### [GoogLeNet](./[paper reading] GoogLeNet.md)

- Basic Concept
  - 结构的稀疏连接
    - 层间连接的稀疏结构
    - 特征连接的稀疏结构
  - 稀疏/密集分布的特征集
- motivation
  - problem to solve
    - sparse structure (of network)
    - dense matrix (of features)
  - sparse structure & dense matrix
- technique
  - Inception
- key elements
  - Inception path
  - 1x1 conv
- use yourself
  - Architecture
  - data augmentation
  - model ensemble
  - Test Time Augmentation
- blogs
- modifications
  - Inception V2

#### [ResNet](./[paper reading] ResNet.md)

- summary
- motivation
  - problems to slove
    - network degradation
    - vanishing/exploding gradient
  - Relationship of degradation & gradient vanishing 
  - identity mapping
- technique
  - residual learning
  - residual block
  - ResNet Architecture
- key element
  - shortcut connection
  - feature concatenate
- math
  - forward/backward propagation
  - residual learning
    - address degradation
    - address gradient vanishing
- use yourself
- articles
  - ResNet V2
  - Shattered Gradients (Gradient correlation)
  - Ensemble-like behavior
  - Hard For Identity Mapping
- blogs

#### [DenseNet](./[paper reading] DenseNet.md)

- Motivation
  - Problem to Solve
  - Modifications
    - Skip Connection
    - Feature Concatenate
- Technique
  - DenseNet Architecture
  - Advantages
    - Parameter Efficiency & Model Compactness
    - Feature Reuse & Collective Knowledge
    - Implicit Deep Supervision
    - Diversified Depth
- Key Element
  - Dense Block
  - Transition Layers
    - Components
    - Compression
  - Growth Rate
  - Bottleneck Structure
- Math
- Use Yourself
  - Bottleneck Structure
  - Transition Layers
  - Feature Reuse
- Blogs

#### [Faster RCNN](./[paper reading] Faster RCNN.md)

- Motivation
  - Problem to Solve
  - Proposal with CNN
- Technique
  - Architecture
  - Region Proposal Network
  - RoI Pooling
  - Anchor
- Key Element
  - anchor/proposals/bbox
  - Pyramids
  - Positive & Negative Sample
  - Sampling Strategy
  - 4-Step Alternating Training
- Math
  - Loss Function
  - Coordinates Parametrization
- Use Yourself
  - Architecture
  - Region Proposal Network
  - RoI Pooling
  - Anchor
- Articles
  - R-CNN
  - YOLO V1
- Blogs

#### [YOLO v1](./[paper reading] YOLO v1.md)

- Motivation
  - Problem to Solve
  - Detection as Regression
- Technique
  - YOLO v1 Architecture
- Key Element
  - Grid Partition
  - Model Output
  - “Responsible”
  - NMS
  - Getting Detections
  - Error Type Analysis
  - Handle Small Object
  - Data Augmentation
- Use Yourself
  - Classification & Detection
  - Explicit & Implicit Grid
  - Multi-Task Output
  - Loss & Sample & box
  - NMS
  - Data Augmentation
- Math
  - Activation Function
  - Loss Function
- Articles
  - R-CNN Based
  - Deep MultiBox
  - OverFeat
  - MultiGrasp
- Blogs

#### [SSD](./[paper reading] SSD.md)

- Motivation
  - Problem to Solve
  - Contributions
- Technique
  - SSD Architecture
  - Pros & Cons
- Key Element
  - Higher Speed
  - What Is Resample
  - Low & High Level Feature
  - Small Object Difficulties
  - Data Flow
  - Anchor & GtBox Matching
  - Foreground & Background
  - Hard Negative Mining
  - NMS
  - Data Augmentation
  - Testing (Inferencing) 
  - Performance Analysis
  - Model Analysis
- Use Yourself
  - Convolution For Speed
  - Feature Pyramids Fashion
  - Positive & Negative Imbalance
  - Zoom In & Zoom Out
- Math
  - Loss Function
  - Multi-Level & Anchor
  - Layer Output & Filters
  - Model Output
- Blogs

#### [RetinaNet](./[paper reading] RetinaNet.md)

- Motivation
  - Problem to Solve
  - Negative ==> Easy
- Technique
  - Focal Loss
  - RetinaNet
- Key Element
  - Class Imbalance
  - Feature Pyramid Network
  - ResNet-FPN Backbone
  - Classify & Regress FCN
  - Post Processing
  - Anchor Design
  - Anchor & GT Matching
  - prior $\pi$ Initialization
  - Ablation Experiments
- Math
  - Cross Entropy Math
    - Standard Cross Entropy
    - Balanced Cross Entropy
  - Focal Loss Math
- Use Yourself
  - Data Analysis
  - Feature Pyramid Network
  - More Scale $\not= $ Better
- Related Work
  - Two-Stage Method
  - One-Stage Method
- Related Articles
- Blogs

#### [CornerNet](./[paper reading] CornerNet.md)

- Motivation
  - Problem to Solve
  - KeyPoints (Anchor-Free)
- Technique
  - CornerNet
  - Stacked Hourglass Network
  - Prediction Module
  - Corner Pooling
- Key Element
  - Why CornerNet Better?
  - Why Corner Pooling Works
  - Grouping Corners
  - Getting Bounding Box
  - Data Augmentation
  - Ablation Experiments
- Math
  - Loss Function
  - Corner Pooling Math
- Use Yourself
  - Network Design
  - Intuition & Interpretability
  - Divide Task
  - TTA
- Related Work
  - Two-Stage
  - One-Stage
- Blogs

#### [CenterNet (triple)](./[paper reading] CenterNet (Triplets).md)

- Motivation
  - Problem to Solve
  - Idea
  - Intuition
- Technique
  - CenterNet Architecture
  - Center Pooling
  - Cascade Corner Pooling
  - Central Region Exploration
- Key Element
  - Baseline：CornerNet
  - Generating BBox
  - Training
  - Inferencing
  - Ablation Experiment
  - Error Analysis
  - Metric AP & AR & FD
  - Small & Medium & Large
- Math
  - Central Region
  - Loss Function
- Use Yourself
- Related Work
  - Anchor-Based Method
  - KeyPoint-Based Method
  - Two-Stage Method
  - One-stage Method
- Problems

#### [CenterNet (Object as Points)](./[paper reading] CenterNet (Object as Points).md)

- Motivation
  - Problem to Solve
  - Idea
- Technique
  - CenterNet Architecture
- Key Element
  - Center Point & Anchor
  - Getting Ground-Truth
  - Model Output
  - Data Augmentation
  - Inference
  - TTA 
  - Compared with SOTA
  - Additional Experiments
- Math
  - Loss Function
  - KeyPoint Loss $\text{L}_k$
  - Offset Loss $\text{L}_{off}$
  - Size Loss $\text{L}_{size}$
- Use Yourself
- Related work
  - Anchor-Based Method
  - KeyPoint-Based Method

#### [FCOS](./[paper reading] FCOS.md)

- Motivation
  - Idea
  - Contribution
- Techniques
  - FCOS Architecture
  - Center-ness
  - Multi-Level FPN Prediction
- Key Elements
  - Prediction Head
  - Training Sample & Label
  - Model Output
  - Feature Pyramid
  - Inference
  - Ablation Study
  - FCN & Detection
  - FCOS $vs.$ YOLO v1
- Math
  - Symbol Definition
  - Loss Function
  - Center-ness
  - Remap of Feature & Image
- Use Yourself
- Related Work
  - Drawbacks of Anchor
  - DenseBox-Based
  - Anchor-Based Detector
  - YOLO v1
  - CornerNet