#### 《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》论文阅读笔记

# 图像序列化

## 1、像素
用 ![](https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{H\times{}W\times{}C}) 来表征一个图像，这个图像被分成 ![](https://latex.codecogs.com/svg.latex?H\times{}W) 个像素，![](https://latex.codecogs.com/svg.latex?H) 行 ![](https://latex.codecogs.com/svg.latex?W) 列，高（Height）即为 ![](https://latex.codecogs.com/svg.latex?H) ，宽（Width）即为 ![](https://latex.codecogs.com/svg.latex?W) 。<br>
每个像素由 ![](https://latex.codecogs.com/svg.latex?C) （例如3）个通道（Channel）表征，每个通道表征一种原色（例如RGB，即Red、Green、Blue中的某一个）。<br>
通道值为 ![](https://latex.codecogs.com/svg.latex?0) 时，该通道不发光；<br>
通道值最大（在8-bit图中对应 ![](https://latex.codecogs.com/svg.latex?2^8-1=255) ）时，该通道亮度最大。<br>
例如R通道（Red），通道值最大时该通道表征为最亮的红色。<br>
像素的颜色为混合色，由 ![](https://latex.codecogs.com/svg.latex?C) 个通道进行综合表征。在特殊情况下，![](https://latex.codecogs.com/svg.latex?C) 个通道的值均为 ![](https://latex.codecogs.com/svg.latex?0) ，此时像素表征为纯黑色，![](https://latex.codecogs.com/svg.latex?C) 个通道的值均达到最大时像素表征为纯白色。<br>
## 2、小块
图像的分辨率与分割出来的像素的数量有关，用行数、列数来表示，即 ![](https://latex.codecogs.com/svg.latex?(H,W)) 。在这张分割像素图上的左上角取一个小块（Patch），这个小块由 ![](https://latex.codecogs.com/svg.latex?P\times{}P) 个像素组成（ ![](https://latex.codecogs.com/svg.latex?P) 行 ![](https://latex.codecogs.com/svg.latex?P) 列），分辨率即为 ![](https://latex.codecogs.com/svg.latex?(P,P)) 。![](https://latex.codecogs.com/svg.latex?P) 的值一般取为 ![](https://latex.codecogs.com/svg.latex?H) 与 ![](https://latex.codecogs.com/svg.latex?W) 的公因数，这样的话就可以从左上角开始，从左到右、从上到下分割出 ![](https://latex.codecogs.com/svg.latex?N=HW/P^2) 个完整的小块。<br>
每个小块包含 ![](https://latex.codecogs.com/svg.latex?P^2) 个像素，每个像素又由 ![](https://latex.codecogs.com/svg.latex?C) 个通道值（实数）来表征，那么就可以用一个含有 ![](https://latex.codecogs.com/svg.latex?P^2C) 个元素的实数向量 ![](https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{P^2C}) 来直接表征这个小块。<br>
输入到模型的是整张图而非单个小块，因此，实际输入是一个含有 ![](https://latex.codecogs.com/svg.latex?N\times{}P^2C) 个元素的实数张量 ![](https://latex.codecogs.com/svg.latex?x_p\in\mathbb{R}^{N\times{}P^2C}) ，即一个 ![](https://latex.codecogs.com/svg.latex?N) 行 ![](https://latex.codecogs.com/svg.latex?P^2C) 列的矩阵，直接表征 ![](https://latex.codecogs.com/svg.latex?N) 个小块，也就是整张图。<br>
## 3、像素通道值的标准化
应注意，![](https://latex.codecogs.com/svg.latex?N\times{}P^2C) 个通道值会被转换为浮点数，并不会保持为原先的范围为 ![](https://latex.codecogs.com/svg.latex?\left[0,255\right]) 的整数。具体转换过程为：<br>
1、每个通道值（包括R通道值、G通道值、B通道值）从原先的整型切换为浮点型，随后除以 `255.0` 。该步骤可表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?c_{rescaled}=\frac{c}{255.0};">
</p>

2、如果采用ImageNet的统计值，那么<br>
对于R通道值，先减去 `0.485` ，再除以 `0.229` ；<br>
对于G通道值，先减去 `0.456` ，再除以 `0.224` ；<br>
对于B通道值，先减去 `0.406` ，再除以 `0.225` ；<br>
`mean = [0.485, 0.456, 0.406]` ，`std = [0.229, 0.224, 0.225]` ，该步骤可统一表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?c_{standardized}=\frac{c_{rescaled}-mean}{std}.">
</p>

## 4、图像序列化
![](https://latex.codecogs.com/svg.latex?x_p\in\mathbb{R}^{N\times{}P^2C}) 张量可以理解为表征了一个含有 ![](https://latex.codecogs.com/svg.latex?N) 个token的序列，每个token的表征维度大小暂时为 ![](https://latex.codecogs.com/svg.latex?P^2C) 。这与一个普通的文本输入序列没有区别。
<br><br><br><br><br>

# 嵌入层

嵌入维度（Dimension）大小为 ![](https://latex.codecogs.com/svg.latex?D) ，注意与在自注意力层之后的前馈层内部才短暂出现的隐藏（Hidden）维度的大小 ![](https://latex.codecogs.com/svg.latex?H) 作区分，前者小于后者。<br>
在序列刚刚进入模型时的词（图像小块）嵌入阶段，需要让 ![](https://latex.codecogs.com/svg.latex?x_p^i\in\mathbb{R}^{1\times{}P^2C}) 乘一个可训练的投影矩阵 ![](https://latex.codecogs.com/svg.latex?E\in\mathbb{R}^{P^2C\times{}D}) ，投影为 ![](https://latex.codecogs.com/svg.latex?x_p^iE\in\mathbb{R}^{1\times{}D}) 。整个图像小块序列经过该阶段得到的输出称为图像小块嵌入，具体为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?z_0=\begin{bmatrix}x_{\text{class}}\\x_p^1E\\x_p^2E\\{}\vdots\\x_p^NE\end{bmatrix}+E_{\text{pos}},\quad{}x_{\text{class}}\in\mathbb{R}^{1\times{}D},\quad{}E_{\text{pos}}\in\mathbb{R}^{(N+1)\times{}D},">
</p>

![](https://latex.codecogs.com/svg.latex?z_0) 的下标 ![](https://latex.codecogs.com/svg.latex?0) 表示第 ![](https://latex.codecogs.com/svg.latex?0) 层，即“嵌入层”，区别于第 ![](https://latex.codecogs.com/svg.latex?1) 层、第 ![](https://latex.codecogs.com/svg.latex?2) 层、... ![](https://latex.codecogs.com/svg.latex?\space) 、第 ![](https://latex.codecogs.com/svg.latex?L) 层等编码器层。<br>
![](https://latex.codecogs.com/svg.latex?z_0\in\mathbb{R}^{(N+1)\times{}D}) 是一个 ![](https://latex.codecogs.com/svg.latex?N+1) 行 ![](https://latex.codecogs.com/svg.latex?D) 列的矩阵，由 ![](https://latex.codecogs.com/svg.latex?N+1) 个形状为 `[1,D]` 的张量沿行方向拼接而成，其第0个子张量 ![](https://latex.codecogs.com/svg.latex?z_0^0=x_{\text{class}}) 经过 ![](https://latex.codecogs.com/svg.latex?L) 个编码器层的变换，在第 ![](https://latex.codecogs.com/svg.latex?L) 个编码器层的输出（编码器的最终输出）当中的状态为 ![](https://latex.codecogs.com/svg.latex?z_L^0=y_{\text{class}}) ，用于表征图像内容的类别，并且将直接作为一个分类头（由一个前馈层或一个单一的线性投影层实现）的输入，模型根据分类头的输出判断图像内容类别的名称（如“树”、“车”、“房子”等）。<br>
![](https://latex.codecogs.com/svg.latex?z_0^0=x_{\text{class}}) 与 ![](https://latex.codecogs.com/svg.latex?z_L^0=y_{\text{class}}) 的上标 ![](https://latex.codecogs.com/svg.latex?0) 表示序列的位置 ![](https://latex.codecogs.com/svg.latex?0) 处，序列的实际内容元素占据位置 ![](https://latex.codecogs.com/svg.latex?1) 、位置 ![](https://latex.codecogs.com/svg.latex?2) 、![](https://latex.codecogs.com/svg.latex?\cdots) ![](https://latex.codecogs.com/svg.latex?\space) 、位置 ![](https://latex.codecogs.com/svg.latex?L)。<br>
通过引入这样一个可训练的类别张量，就可以让模型在训练过程中将图像内容的类别信息注入这个类别张量，从而在最后只需将这个经过层层变换的类别张量作为分类头的唯一输入，即可得到输出的类别结果。<br>
可训练的 ![](https://latex.codecogs.com/svg.latex?E_{\text{pos}}) 用于在训练过程中注入各个图像小块的二维位置信息。
